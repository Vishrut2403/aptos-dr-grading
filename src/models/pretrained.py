"""Models 3 and 4: the two ImageNet-pretrained backbones.

Model 3, EfficientNet-B0.
Model 4, ViT-S/16.

Both load through timm with the classifier removed (num_classes=0), so the only
thing differing between them is the architecture: same pretraining corpus, same
input resolution, same heads, same schedule. That is what makes the CNN versus
transformer comparison in Objective 1 mean anything.

Whoever implements TimmBackbone should set self.out_dim from the created
model's num_features.

For the viva on Model 3: compound scaling (why depth, width and resolution are
scaled together), MBConv inverted residuals, squeeze-and-excitation, and the
mechanics of fine-tuning versus freezing.

For the viva on Model 4: patch embedding, self-attention, positional encoding,
and why a ViT usually needs more data than a CNN to reach the same accuracy.
"""

import torch.nn as nn


class TimmBackbone(nn.Module):
    """Shared wrapper: create a timm model with no classifier, expose out_dim."""

    def __init__(self, name, pretrained=True, drop_rate=0.0):
        super().__init__()
        raise NotImplementedError("TimmBackbone: not implemented yet")

    def forward(self, x):
        """x: (B, 3, 224, 224) -> (B, self.out_dim)"""
        raise NotImplementedError


def efficientnet_b0(pretrained=True, **kw):
    """Model 3. timm name: efficientnet_b0"""
    raise NotImplementedError("Model 3: not implemented yet")


def resnet50(pretrained=True, **kw):
    """Fallback for Model 3 if EfficientNet proves unstable. timm: resnet50"""
    raise NotImplementedError


def vit_small(pretrained=True, **kw):
    """Model 4. timm name: vit_small_patch16_224"""
    raise NotImplementedError("Model 4: not implemented yet")


def vit_base(pretrained=True, **kw):
    """Stretch goal for Model 4. Measured to fit 6GB at batch 32 with AMP.
    timm name: vit_base_patch16_224"""
    raise NotImplementedError


def deit_small(pretrained=True, **kw):
    """Fallback for Model 4: same size as ViT-S, trained with heavier
    augmentation, which usually helps on small datasets.
    timm name: deit_small_patch16_224"""
    raise NotImplementedError
