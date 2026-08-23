"""Model 1: fully connected baseline.

The point of this model is to be the floor. It flattens the image, so it has no
notion that neighbouring pixels are related and every spatial relationship has
to be learned by brute force in the first linear layer. Whatever the CNN gains
over this number is the value of convolutional inductive bias, which is
Objective 1 of the synopsis.

Run at 64x64, not 224x224. Flattening a 224x224x3 image gives a 150,528-element
vector, and the first layer alone would then hold more parameters than the other
three models in the study combined.

Be ready to explain in the viva:
  - why this model is expected to collapse onto the majority No-DR class
  - why it can carry more parameters than EfficientNet-B0 and still lose
  - what BatchNorm and Dropout are each doing here

Suggested shape: three hidden layers 1024-512-256, each Linear -> BatchNorm1d
-> ReLU -> Dropout. Set self.out_dim to the width of the last hidden layer;
src/models/__init__.py reads it to size the output head.
"""

import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, img_size=64, widths=(1024, 512, 256), dropout=0.3):
        super().__init__()
        self.out_dim = widths[-1]
        raise NotImplementedError("Model 1: not implemented yet")

    def forward(self, x):
        """x: (B, 3, 64, 64) -> (B, self.out_dim)"""
        raise NotImplementedError
