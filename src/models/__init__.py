"""Model registry.

Every model in the study is (backbone + head). Only the backbone differs; the
head comes from src/models/heads.py and is identical across all four, which is
what makes the comparison controlled.

Shared infrastructure: changing this changes every model's numbers, so agree it
with the team first.
"""

import torch.nn as nn

from .heads import HEADS, NUM_CLASSES
from .mlp import MLPBackbone
from .cnn_scratch import SmallCNN
from .pretrained import efficientnet_b0, resnet50, vit_small, vit_base, deit_small


class DRModel(nn.Module):
    def __init__(self, backbone, head_type="coral", num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.head = HEADS[head_type](backbone.out_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# name -> (builder, native input resolution)
REGISTRY = {
    "mlp":         (lambda **kw: MLPBackbone(img_size=64, **kw),  64),
    "cnn_scratch": (lambda **kw: SmallCNN(**kw),                 224),
    "effnet_b0":   (efficientnet_b0,                             224),
    "resnet50":    (resnet50,                                    224),
    "vit_small":   (vit_small,                                   224),
    "vit_base":    (vit_base,                                    224),
    "deit_small":  (deit_small,                                  224),
}

UNPRETRAINED = {"mlp", "cnn_scratch"}


def build(name, head_type="coral", pretrained=True, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(REGISTRY)}")
    builder, img_size = REGISTRY[name]
    if name in UNPRETRAINED:
        backbone = builder(**kw)
    else:
        backbone = builder(pretrained=pretrained, **kw)
    return DRModel(backbone, head_type=head_type), img_size
