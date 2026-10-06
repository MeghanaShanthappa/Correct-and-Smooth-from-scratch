# Correct & Smooth from scratch

An educational PyTorch implementation of **Correct & Smooth (C&S)** for
transductive node classification. It does not use PyG's `CorrectAndSmooth` or a
built-in message-passing layer.

C&S is a post-processing method rather than a new GNN layer:

1. An MLP predicts each node from its features only.
2. **Correct** propagates the MLP's known training errors through the graph.
3. **Smooth** injects true training labels and propagates the corrected predictions.

The graph operation `D^(-1/2) A D^(-1/2) X` is implemented directly with
`torch.bincount` and `index_add_`.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Small test

```bash
python3 correct_and_smooth.py --synthetic --epochs 100
```

## OGB experiment

```bash
python3 correct_and_smooth.py \
  --dataset ogbn-arxiv \
  --epochs 200 \
  --correction-alpha 0.8 \
  --smoothing-alpha 0.8
```

For a smaller learning run:

```bash
python3 correct_and_smooth.py --dataset ogbn-arxiv --max-nodes 20000
```

The subset command creates a random 60/20/20 split and therefore does not produce
an official OGB benchmark score.

## Core update

Both stages repeatedly perform:

```text
result = alpha * normalized_neighbor_sum(result)
       + (1 - alpha) * initial_signal
```

The correction signal is the training residual `true_label - prediction`. The
smoothing signal uses true one-hot labels on training nodes and corrected
predictions everywhere else.

Paper: Qian Huang et al., *Combining Label Propagation and Simple Models
Out-performs Graph Neural Networks*, ICLR 2021.
