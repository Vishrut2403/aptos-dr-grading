"""Losses and decoders, one pair per head type.

Pairs with src/models/heads.py. Each head needs a loss matching its output
shape and a decoder turning those outputs back into an integer grade in [0, 4].

loss_fn(logits, target, head_type, class_weights=None) -> scalar tensor
  softmax : cross entropy over 5 classes
  regress : MSE against the grade as a float
  coral   : binary cross entropy over the K-1 cumulative tasks, where
            levels[i, k] = 1 if the true grade of sample i exceeds k. Sum over
            the K-1 tasks, then average over the batch.

decode(logits, head_type) -> LongTensor (B,) in [0, 4]
  softmax : argmax
  regress : round and clamp
  coral   : count how many cumulative thresholds the sample clears, i.e.
            (sigmoid(logits) > 0.5).sum(dim=1)

class_weights is an optional (5,) tensor of inverse-frequency weights, used
when --balanced is passed to src/train.py.
"""

import torch
import torch.nn.functional as F

NUM_CLASSES = 5


def loss_fn(logits, target, head_type, class_weights=None):
    raise NotImplementedError("losses: not implemented yet")


@torch.no_grad()
def decode(logits, head_type):
    raise NotImplementedError("decoders: not implemented yet")
