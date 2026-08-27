"""Model 1: fully connected baseline, the floor of the comparison.

Flattens the image, so it has no notion that neighbouring pixels are related.
Whatever the CNN gains over this is the value of convolutional inductive bias.

Runs at 64x64, not 224: flattening 224x224x3 gives a 150,528-element vector and
the first layer alone would outweigh the other three models combined.
"""

import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, img_size=64, widths=(1024, 512, 256), dropout=0.3):
        super().__init__()
        in_dim = 3 * img_size * img_size
        layers = [nn.Flatten()]
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.BatchNorm1d(w),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = w
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        """x: (B, 3, 64, 64) -> (B, self.out_dim)"""
        return self.net(x)
