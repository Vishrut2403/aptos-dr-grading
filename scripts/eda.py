"""Exploratory analysis: class balance and image quality across the 3,662 images.

Quality is measured on the raw files, not the cache, because Ben Graham
normalisation removes exactly the illumination variation we want to quantify.

    python scripts/eda.py --raw data/raw/aptos2019 --out reports
"""

import argparse, json, os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
PROBE = 512  # every image is measured at one size; APTOS resolutions vary 12-fold


def _retina_mask(gray):
    """Largest bright blob: the illuminated circle, everything else is letterbox."""
    _, th = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return None, None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i, stats[i], cent[i]


def measure(args):
    path, idx = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return idx, None
    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (PROBE, PROBE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask, stats, cent = _retina_mask(gray)
    if mask is None:
        return idx, None
    pix = gray[mask]
    radius = 0.5 * max(stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT])

    return idx, {
        "w": w0, "h": h0,
        # focus: Laplacian variance inside the retina only, so the black
        # surround cannot inflate or deflate it
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var()),
        "mean": float(pix.mean()),
        "std": float(pix.std()),
        "clip_hi": float((pix > 250).mean()),
        "clip_lo": float((pix < 10).mean()),
        # how far the retinal circle sits from the frame centre, in radii
        "offset": float(np.hypot(cent[0] - PROBE / 2, cent[1] - PROBE / 2) / max(radius, 1)),
        "fill": float(mask.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/aptos2019")
    p.add_argument("--out", default="reports")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    df = pd.read_csv(os.path.join(a.raw, "train.csv"))
    jobs = [(os.path.join(a.raw, "train_images", f"{r}.png"), i)
            for i, r in enumerate(df["id_code"])]

    rows = [None] * len(jobs)
    with Pool(a.workers) as pool:
        for n, (idx, m) in enumerate(pool.imap_unordered(measure, jobs, chunksize=16)):
            rows[idx] = m  # by index, never append: imap_unordered returns out of order
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    ok = [i for i, r in enumerate(rows) if r is not None]
    q = pd.DataFrame([rows[i] for i in ok])
    q["grade"] = df["diagnosis"].values[ok]
    os.makedirs(a.out, exist_ok=True)
    q.to_csv(os.path.join(a.out, "eda_quality.csv"), index=False)

    # thresholds are percentile-based, so they describe this dataset rather than
    # importing cutoffs from a different camera population
    blur_t = q["blur"].quantile(0.05)
    dark_t, bright_t = q["mean"].quantile(0.05), q["mean"].quantile(0.95)
    summary = {
        "n": int(len(q)), "unreadable": int(len(rows) - len(ok)),
        "grades": {str(k): int(v) for k, v in q["grade"].value_counts().sort_index().items()},
        "resolutions": int(q.groupby(["w", "h"]).ngroups),
        "res_min": [int(q["w"].min()), int(q["h"].min())],
        "res_max": [int(q["w"].max()), int(q["h"].max())],
        "blur": {"p5": float(blur_t), "median": float(q["blur"].median()),
                 "n_below_p5": int((q["blur"] < blur_t).sum())},
        "exposure": {"dark_p5": float(dark_t), "bright_p95": float(bright_t),
                     "median": float(q["mean"].median()),
                     "n_dark": int((q["mean"] < dark_t).sum()),
                     "n_bright": int((q["mean"] > bright_t).sum())},
        "offcentre": {"median": float(q["offset"].median()),
                      "n_gt_0.10r": int((q["offset"] > 0.10).sum()),
                      "n_gt_0.25r": int((q["offset"] > 0.25).sum())},
        "fill": {"median": float(q["fill"].median()), "min": float(q["fill"].min())},
        "blur_median_by_grade": {str(g): float(v) for g, v in
                                 q.groupby("grade")["blur"].median().items()},
    }
    with open(os.path.join(a.out, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
