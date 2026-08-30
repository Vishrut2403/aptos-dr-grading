"""Dataset, augmentation, and the one split every model shares.

The split is written to disk once and reloaded by every training run, so "same
train/val/test split across all four models" is enforced by code rather than by
four people remembering to pass the same seed. Do not add a second way to split.
"""

import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_split(labels_csv, out_json, seed=42, val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test row indices, written once and reused by all runs."""
    df = pd.read_csv(labels_csv)
    idx = np.arange(len(df))
    y = df["diagnosis"].values

    # stratified because grades 3 and 4 are rare enough that an unstratified
    # draw can leave the test set with almost none of them
    tr, tmp = train_test_split(idx, test_size=val_frac + test_frac,
                               stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=test_frac / (val_frac + test_frac),
                              stratify=y[tmp], random_state=seed)

    split = {"seed": seed, "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
    with open(out_json, "w") as f:
        json.dump(split, f)
    return split


class APTOSDataset(Dataset):
    """Cached uint8 array -> (normalised float CHW tensor, int grade)."""

    def __init__(self, npy_path, labels_csv, indices, img_size=224, train=False):
        # mmap so four concurrent runs share one copy of the 1.1 GB cache
        self.images = np.load(npy_path, mmap_mode="r")
        self.labels = pd.read_csv(labels_csv)["diagnosis"].values
        self.indices = np.asarray(indices)
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.indices)

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        if np.random.rand() < 0.5:
            img = img[::-1, :]
        # a fundus image has no canonical orientation, so free rotation is safe
        # here in a way it would not be for natural images
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-180, 180),
                                    np.random.uniform(0.9, 1.1))
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderValue=(0, 0, 0))
        # jitter stays mild: Ben Graham already removed most of the inter-camera
        # colour variation, and piling more on top destroys signal
        return np.clip(img.astype(np.float32) * np.random.uniform(0.9, 1.1)
                       + np.random.uniform(-10, 10), 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        j = self.indices[i]
        img = np.array(self.images[j])

        if self.train:
            img = self._augment(img)
        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), int(self.labels[j])


def class_weights(labels_csv, indices, num_classes=5, device="cpu"):
    """Inverse-frequency weights, mean 1 so the loss scale matches an unweighted run."""
    y = pd.read_csv(labels_csv)["diagnosis"].values[np.asarray(indices)]
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), device=device)
