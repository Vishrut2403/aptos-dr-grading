"""Model 2: CNN trained from scratch.

No pretrained weights anywhere. This model and the MLP are the two that never
see outside data, so the gap between this and EfficientNet-B0 is what isolates
the value of ImageNet pretraining.

Be ready to explain in the viva:
  - what a convolution gives you that a dense layer does not
  - how padding, stride and pooling change the receptive field
  - why width is kept modest on a 3,662-image dataset

Suggested shape: VGG-style, five stages with widths 32-64-128-256-512. Each
stage is two 3x3 convolutions with BatchNorm and ReLU, then MaxPool2d(2). Five
stages so the receptive field covers a whole fundus image by the last layer.
Finish with AdaptiveAvgPool2d(1) and flatten rather than a large dense layer.
Set self.out_dim to the final channel count.
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256, 512), dropout=0.3):
        super().__init__()
        self.out_dim = widths[-1]
        raise NotImplementedError("Model 2: not implemented yet")

    def forward(self, x):
        """x: (B, 3, 224, 224) -> (B, self.out_dim)"""
        raise NotImplementedError
