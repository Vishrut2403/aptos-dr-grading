"""Dataset, augmentation, and the one split every model shares.

The split is written to disk once and reloaded by every training run, so "same
train/val/test split across all four models" is enforced by code rather than by
four people remembering to pass the same seed. Do not add a second way to split.

make_split(labels_csv, out_json, seed=42, val_frac=0.15, test_frac=0.15)
    Stratified on the diagnosis column, because grades 3 and 4 are rare enough
    that an unstratified split can leave a test set with almost none of them.
    Writes {"seed": int, "train": [...], "val": [...], "test": [...]} where the
    values are row indices into labels.csv.

APTOSDataset(npy_path, labels_csv, indices, img_size=224, train=False)
    Loads the cached array with mmap_mode="r" so four concurrent training runs
    do not each hold a copy in RAM. Returns (float tensor CHW, int label),
    normalised with the ImageNet mean and std since three of the four models
    expect that.

    Augmentation, train split only. Fundus images have no canonical orientation,
    so free rotation and both flips are safe here in a way they would not be for
    natural images. Keep the photometric jitter mild: Ben Graham normalisation
    has already removed most of the inter-camera colour variation, and piling
    more on top mostly destroys signal.

class_weights(labels_csv, indices, num_classes=5, device="cuda")
    Inverse-frequency weights normalised to mean 1, so the loss stays on a
    comparable scale to the unweighted run. Used when --balanced is passed.

For the viva: why the split is stratified, why augmentation is applied to the
training split only, and whether class weighting or oversampling is the better
answer to a 9:1 imbalance.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_split(labels_csv, out_json, seed=42, val_frac=0.15, test_frac=0.15):
    raise NotImplementedError("make_split: not implemented yet")


class APTOSDataset(Dataset):
    def __init__(self, npy_path, labels_csv, indices, img_size=224, train=False):
        raise NotImplementedError("APTOSDataset: not implemented yet")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def class_weights(labels_csv, indices, num_classes=5, device="cuda"):
    raise NotImplementedError("class_weights: not implemented yet")
