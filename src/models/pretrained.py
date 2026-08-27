"""Models 3 (EfficientNet-B0) and 4 (ViT-S/16), both ImageNet-pretrained.

Loaded through timm with the classifier removed, so architecture is the only
thing differing between them: same pretraining corpus, resolution, heads and
schedule. That is what makes the comparison in Objective 1 mean anything.

Viva, Model 3: compound scaling, MBConv inverted residuals, squeeze-and-
excitation, fine-tuning versus freezing.
Viva, Model 4: patch embedding, self-attention, positional encoding, and why a
ViT needs more data than a CNN.
"""

import torch.nn as nn


class TimmBackbone(nn.Module):
    """Shared wrapper: create a timm model with no classifier, expose out_dim."""

    def __init__(self, name, pretrained=True, drop_rate=0.0):
        super().__init__()
        import timm

        self.net = timm.create_model(name, pretrained=pretrained,
                                     num_classes=0, drop_rate=drop_rate)
        self.out_dim = self.net.num_features

    def forward(self, x):
        """x: (B, 3, 224, 224) -> (B, self.out_dim)"""
        return self.net(x)


def efficientnet_b0(pretrained=True, **kw):
    """Model 3. Compound-scaled CNN of MBConv blocks with squeeze-and-excitation."""
    return TimmBackbone("efficientnet_b0", pretrained=pretrained, **kw)


def resnet50(pretrained=True, **kw):
    """Fallback for Model 3 if EfficientNet proves unstable. timm: resnet50"""
    return TimmBackbone("resnet50", pretrained=pretrained, **kw)


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
