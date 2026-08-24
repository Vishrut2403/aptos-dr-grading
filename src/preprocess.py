"""Crop the fundus circle, normalise illumination, cache to a uint8 array.

    python -m src.preprocess --raw data/raw/aptos2019 --out data/processed --size 320

Writes images_{size}.npy of shape (N, size, size, 3) and labels.csv, in the
same row order.
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def crop_fundus(img, tol=7):
    """Trim the black border. Thresholds on grey, not per channel."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = grey > tol
    if mask.sum() == 0:
        return img
    rows, cols = np.where(mask)
    return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def ben_graham(img, size, sigma_frac=30.0, alpha=4.0, beta=-4.0, gamma=128.0):
    """Subtract a heavy local average, leaving the lesions."""
    blur = cv2.GaussianBlur(img, (0, 0), size / sigma_frac)
    return cv2.addWeighted(img, alpha, blur, beta, gamma)


def circle_mask(img):
    """Zero the corners outside the retina."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * 0.95), 1, -1)
    return img * mask[..., None]


def process_one(path, size, apply_ben=True):
    """One path -> (size, size, 3) uint8, or None if unreadable."""
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is None:
        return None
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = crop_fundus(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if apply_ben:
        img = ben_graham(img, size)
    img = circle_mask(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def _worker(job):
    idx, path, size, apply_ben = job
    try:
        return idx, process_one(path, size, apply_ben=apply_ben)
    except Exception:
        return idx, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--no-ben", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    a = ap.parse_args()

    csv_path = os.path.join(a.raw, "train.csv")
    img_dir = os.path.join(a.raw, "train_images")
    for p in (csv_path, img_dir):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}. Run ./scripts/download_data.sh first.")

    df = pd.read_csv(csv_path)
    os.makedirs(a.out, exist_ok=True)
    cv2.setNumThreads(1)  # otherwise it fights the process pool for cores

    jobs = [(i, os.path.join(img_dir, f"{iid}.png"), a.size, not a.no_ben)
            for i, iid in enumerate(df["id_code"])]
    arr = np.zeros((len(df), a.size, a.size, 3), dtype=np.uint8)
    keep, failed = [], []

    # Write to arr[idx], never append: imap_unordered comes back out of order
    # and appending would misalign every image against its label.
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = pool.imap_unordered(_worker, jobs, chunksize=16)
            for idx, out in tqdm(results, total=len(jobs), desc=f"preprocess x{a.workers}"):
                if out is None:
                    failed.append(df["id_code"].iloc[idx])
                else:
                    arr[idx] = out
                    keep.append(idx)
    else:
        for job in tqdm(jobs, desc="preprocess"):
            idx, out = _worker(job)
            if out is None:
                failed.append(df["id_code"].iloc[idx])
            else:
                arr[idx] = out
                keep.append(idx)

    if failed:
        print(f"\n{len(failed)} unreadable, dropped: {failed[:10]}")

    keep.sort()
    arr = arr[keep]
    df = df.iloc[keep].reset_index(drop=True)

    np.save(os.path.join(a.out, f"images_{a.size}.npy"), arr)
    df.to_csv(os.path.join(a.out, "labels.csv"), index=False)

    print(f"\ncached {len(df)} images at {a.size}px -> {arr.nbytes / 2**20:.0f} MiB")
    print(df["diagnosis"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
