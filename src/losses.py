"""Losses and decoders, one pair per head type.

Pairs with src/models/heads.py: each head needs a loss matching its output
shape and a decoder turning those outputs back into a grade in [0, 4].
"""

import torch
import torch.nn.functional as F

NUM_CLASSES = 5


def loss_fn(logits, target, head_type, class_weights=None):
    if head_type == "softmax":
        return F.cross_entropy(logits, target, weight=class_weights)

    if head_type == "regress":
        return F.mse_loss(logits.squeeze(1), target.float())

    if head_type == "coral":
        # levels[i, k] = 1 if grade i exceeds k, turning one 5-way problem into
        # four binary ones that are consistent by construction
        k = torch.arange(NUM_CLASSES - 1, device=target.device)
        levels = (target.unsqueeze(1) > k.unsqueeze(0)).float()
        per_task = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
        if class_weights is not None:
            per_task = per_task * class_weights[target].unsqueeze(1)
        return per_task.sum(dim=1).mean()

    raise ValueError(f"unknown head type {head_type!r}")


@torch.no_grad()
def decode(logits, head_type):
    """Model output -> integer grade in [0, 4]."""
    if head_type == "softmax":
        return logits.argmax(dim=1)

    if head_type == "regress":
        return logits.squeeze(1).round().clamp(0, NUM_CLASSES - 1).long()

    if head_type == "coral":
        return (torch.sigmoid(logits) > 0.5).sum(dim=1).clamp(0, NUM_CLASSES - 1).long()

    raise ValueError(f"unknown head type {head_type!r}")
