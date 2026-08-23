"""Output heads for all four models.

Three interchangeable ways to turn a feature vector into a DR grade. Every
backbone is trained with all three, so the value of modelling the label
ordering is measured independently of which network produced the features.
This is Objective 2 of the synopsis.

All three take (B, in_dim) and return logits. Their shapes differ on purpose:

  SoftmaxHead     -> (B, 5)   nominal, ignores ordering
  RegressionHead  -> (B, 1)   one continuous score, thresholded at decode
  CoralHead       -> (B, 4)   K-1 cumulative logits, P(y > k) for k = 0..3

CORAL (Cao, Mirjalili & Raschka, 2020) is the interesting one. It uses a single
shared weight vector with K-1 independent bias terms. Sharing the weights is
exactly what guarantees the cumulative probabilities stay monotonic, so the
model can never claim P(y > 2) > P(y > 1). Be ready to explain in the viva why
that property does not hold if you give each threshold its own weight vector.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 5


class SoftmaxHead(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        raise NotImplementedError("heads: not implemented yet")

    def forward(self, x):
        raise NotImplementedError


class RegressionHead(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CoralHead(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


HEADS = {"softmax": SoftmaxHead, "regress": RegressionHead, "coral": CoralHead}
