"""Output heads for all four models.

Three interchangeable ways to turn a feature vector into a DR grade, so the
value of modelling the label ordering is measured independently of which
backbone produced the features. All take (B, in_dim):

  SoftmaxHead     -> (B, 5)   nominal, ignores ordering
  RegressionHead  -> (B, 1)   one continuous score, thresholded at decode
  CoralHead       -> (B, 4)   K-1 cumulative logits, P(y > k) for k = 0..3
"""

import torch
import torch.nn as nn

NUM_CLASSES = 5


class SoftmaxHead(nn.Module):
    """Nominal baseline: treats the five grades as unordered categories."""

    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    """Predicts one continuous score, thresholded at decode time."""

    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)


class CoralHead(nn.Module):
    """CORAL (Cao, Mirjalili and Raschka, 2020): K-1 cumulative logits."""

    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        # one shared weight vector with K-1 free biases. Sharing the weights is
        # what keeps the cumulative probabilities monotonic; give each threshold
        # its own vector and the model can claim P(y>2) > P(y>1).
        self.shared = nn.Linear(in_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.shared(x) + self.biases


HEADS = {"softmax": SoftmaxHead, "regress": RegressionHead, "coral": CoralHead}
