"""Correct & Smooth node classification using only ordinary PyTorch operations.

The base MLP ignores the graph. The graph is used afterward to propagate
prediction errors (Correct) and class predictions (Smooth).
"""

import argparse
import copy
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    split: dict[str, torch.Tensor]
    num_classes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_graph() -> GraphData:
    """Small homophilic graph: connected nodes usually share a class."""
    x = torch.tensor([
        [1.00, 0.10], [0.90, 0.20], [0.70, 0.30], [0.55, 0.45],
        [0.45, 0.55], [0.30, 0.70], [0.20, 0.90], [0.10, 1.00],
        [0.80, 0.35], [0.35, 0.80], [0.60, 0.40], [0.40, 0.60],
    ], dtype=torch.float)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 8), (8, 10), (10, 0),
        (4, 5), (5, 6), (6, 7), (7, 9), (9, 11), (11, 4),
        (3, 4),
    ]
    edges = edges + [(dst, src) for src, dst in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    split = {
        "train": torch.tensor([0, 2, 5, 7]),
        "valid": torch.tensor([1, 4, 8, 9]),
        "test": torch.tensor([3, 6, 10, 11]),
    }
    return GraphData(x, edge_index, labels, split, num_classes=2)


def load_ogb(name: str, root: str, max_nodes: int | None, seed: int) -> GraphData:
    """Load an OGB node dataset. PyG is used only as the dataset container."""
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=name, root=root)
    graph = dataset[0]
    x = graph.x.float()
    y = graph.y.view(-1).long()
    edge_index = graph.edge_index.long()

    # C&S assumes information can move both ways across each graph connection.
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    split = {key: value.view(-1) for key, value in dataset.get_idx_split().items()}

    if max_nodes is not None and max_nodes < x.size(0):
        n = max_nodes
        keep = (edge_index[0] < n) & (edge_index[1] < n)
        x, y, edge_index = x[:n], y[:n], edge_index[:, keep]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        train_end, valid_end = int(0.6 * n), int(0.8 * n)
        split = {
            "train": order[:train_end],
            "valid": order[train_end:valid_end],
            "test": order[valid_end:],
        }
        print("Using a random 60/20/20 split for the node subset.")

    return GraphData(x, edge_index, y, split, dataset.num_classes)


class MLP(nn.Module):
    """Feature-only base classifier; it never receives edge_index."""

    def __init__(self, in_channels: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden)
        self.linear2 = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear2(x)


def normalized_neighbor_sum(
    values: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Compute D^(-1/2) A D^(-1/2) values without a sparse-graph library."""
    src, dst = edge_index
    num_nodes = values.size(0)
    degree = torch.bincount(dst, minlength=num_nodes).to(values.dtype).clamp_min(1)
    weights = degree[src].rsqrt() * degree[dst].rsqrt()
    messages = values[src] * weights.unsqueeze(1)
    result = torch.zeros_like(values)
    result.index_add_(0, dst, messages)
    return result


@torch.no_grad()
def propagate(
    initial: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    steps: int,
    clamp: tuple[float, float],
    fixed_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Repeatedly apply X <- alpha P X + (1-alpha) X_initial."""
    result = initial.clone()
    for _ in range(steps):
        result = alpha * normalized_neighbor_sum(result, edge_index)
        result.add_(initial, alpha=1.0 - alpha)
        result.clamp_(clamp[0], clamp[1])
        if fixed_indices is not None:
            result[fixed_indices] = initial[fixed_indices]
    return result


class CorrectAndSmooth:
    """From-scratch Correct & Smooth post-processor."""

    def __init__(
        self,
        correction_steps: int = 50,
        correction_alpha: float = 0.8,
        smoothing_steps: int = 50,
        smoothing_alpha: float = 0.8,
        autoscale: bool = True,
        fixed_scale: float = 1.0,
    ) -> None:
        self.correction_steps = correction_steps
        self.correction_alpha = correction_alpha
        self.smoothing_steps = smoothing_steps
        self.smoothing_alpha = smoothing_alpha
        self.autoscale = autoscale
        self.fixed_scale = fixed_scale

    @torch.no_grad()
    def correct(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        train_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate known training errors to correct all node predictions."""
        classes = probabilities.size(1)
        residual_seed = torch.zeros_like(probabilities)
        true_train = F.one_hot(labels[train_index], classes).to(probabilities.dtype)
        residual_seed[train_index] = true_train - probabilities[train_index]

        fixed = None if self.autoscale else train_index
        propagated_error = propagate(
            residual_seed,
            edge_index,
            self.correction_alpha,
            self.correction_steps,
            clamp=(-1.0, 1.0),
            fixed_indices=fixed,
        )

        if self.autoscale:
            # Match the average size of propagated errors to observed train errors.
            average_error = residual_seed[train_index].abs().sum() / train_index.numel()
            node_error = propagated_error.abs().sum(dim=1, keepdim=True)
            scale = average_error / node_error.clamp_min(1e-12)
            scale[(node_error == 0) | (scale > 1000)] = 1.0
            corrected = probabilities + scale * propagated_error
        else:
            corrected = probabilities + self.fixed_scale * propagated_error

        corrected = torch.where(torch.isnan(corrected), probabilities, corrected)
        return corrected

    @torch.no_grad()
    def smooth(
        self,
        corrected: torch.Tensor,
        labels: torch.Tensor,
        train_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Inject true training labels and smooth predictions across the graph."""
        classes = corrected.size(1)
        smooth_seed = corrected.clone()
        smooth_seed[train_index] = F.one_hot(
            labels[train_index], classes
        ).to(corrected.dtype)
        return propagate(
            smooth_seed,
            edge_index,
            self.smoothing_alpha,
            self.smoothing_steps,
            clamp=(0.0, 1.0),
        )

    @torch.no_grad()
    def __call__(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        train_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        corrected = self.correct(probabilities, labels, train_index, edge_index)
        smoothed = self.smooth(corrected, labels, train_index, edge_index)
        return corrected, smoothed


@torch.no_grad()
def accuracy(scores: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    return (scores[index].argmax(dim=1) == labels[index]).float().mean().item()


def report(name: str, scores: torch.Tensor, data: GraphData) -> None:
    values = {
        key: accuracy(scores, data.y, index)
        for key, index in data.split.items()
    }
    print(
        f"{name:>10} | train={values['train']:.4f} "
        f"valid={values['valid']:.4f} test={values['test']:.4f}"
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data = make_synthetic_graph() if args.synthetic else load_ogb(
        args.dataset, args.data_root, args.max_nodes, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.split = {key: value.to(device) for key, value in data.split.items()}

    model = MLP(data.x.size(1), args.hidden, data.num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_valid = -1.0
    best_state = None

    print(
        f"device={device} nodes={data.x.size(0):,} "
        f"edges={data.edge_index.size(1):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x)
        loss = F.cross_entropy(
            logits[data.split["train"]], data.y[data.split["train"]]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x)
            valid = accuracy(logits, data.y, data.split["valid"])
        if valid > best_valid:
            best_valid = valid
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} valid={valid:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    torch.save(best_state, args.output)
    model.eval()
    with torch.no_grad():
        probabilities = model(data.x).softmax(dim=1)

    c_and_s = CorrectAndSmooth(
        correction_steps=args.correction_steps,
        correction_alpha=args.correction_alpha,
        smoothing_steps=args.smoothing_steps,
        smoothing_alpha=args.smoothing_alpha,
        autoscale=not args.fixed_scale,
        fixed_scale=args.scale,
    )
    corrected, smoothed = c_and_s(
        probabilities,
        data.y,
        data.split["train"],
        data.edge_index,
    )

    print("\nAccuracy before and after graph post-processing")
    report("MLP", probabilities, data)
    report("Correct", corrected, data)
    report("C & S", smoothed, data)
    print(f"Saved best base MLP checkpoint to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct & Smooth implemented with plain PyTorch"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--correction-steps", type=int, default=50)
    parser.add_argument("--correction-alpha", type=float, default=0.8)
    parser.add_argument("--smoothing-steps", type=int, default=50)
    parser.add_argument("--smoothing-alpha", type=float, default=0.8)
    parser.add_argument("--fixed-scale", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output", default="best_mlp.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
