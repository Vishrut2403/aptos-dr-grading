"""Model 2: VGG-style CNN trained from scratch, no pretrained weights.

This and the MLP are the two models that never see outside data, so the gap
between this and EfficientNet-B0 is what isolates the value of pretraining.

Five stages so the receptive field covers a whole fundus image by the last
layer. Width stays modest because 3,662 images is not much data.
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256, 512), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 3
        for c_out in widths:
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = c_in

    def forward(self, x):
        """x: (B, 3, 224, 224) -> (B, self.out_dim)"""
        x = self.features(x)
        return self.drop(self.pool(x).flatten(1))
