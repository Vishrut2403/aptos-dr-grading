"""Report figures for the preprocessing section.

    python -m scripts.make_figures --raw data/raw/aptos2019 --out reports/figures
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocess import ben_graham, circle_mask, crop_fundus

SIZE = 320


def stages(path, size=SIZE):
    """One image at each pipeline stage, for the walkthrough figure."""
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    cropped = crop_fundus(img)
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    normed = np.clip(ben_graham(resized, size), 0, 255).astype(np.uint8)
    return [("raw", img), ("cropped", cropped), ("resized", resized),
            ("Ben Graham", normed), ("masked", circle_mask(normed))]


def final(path, size=SIZE):
    return stages(path, size)[-1][1]


def panel(axes, images, titles, ylabel=None):
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect(1); ax.set_frame_on(False)
        if t:
            ax.set_title(t, fontsize=8)
    if ylabel:
        axes[0].set_ylabel(ylabel, fontsize=8)


def fig_pipeline(raw_dir, out_dir, image_id):
    st = stages(os.path.join(raw_dir, "train_images", f"{image_id}.png"))
    fig, axes = plt.subplots(1, len(st), figsize=(2.0 * len(st), 2.3))
    panel(axes, [i for _, i in st], [f"{n}\n{i.shape[1]}x{i.shape[0]}" for n, i in st])
    fig.suptitle("Preprocessing stages", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_pipeline_stages.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_cameras(raw_dir, out_dir, ids, labels_csv):
    grades = pd.read_csv(labels_csv).set_index("id_code")["diagnosis"]
    fig, axes = plt.subplots(2, len(ids), figsize=(2.0 * len(ids), 4.6))
    for col, iid in enumerate(ids):
        path = os.path.join(raw_dir, "train_images", f"{iid}.png")
        raw = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(raw)
        axes[1, col].imshow(final(path))
        axes[0, col].set_title(f"{raw.shape[1]}x{raw.shape[0]}\ngrade {grades[iid]}", fontsize=8)
        for row in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            axes[row, col].set_box_aspect(1); axes[row, col].set_frame_on(False)
    axes[0, 0].set_ylabel("raw", fontsize=9)
    axes[1, 0].set_ylabel("processed", fontsize=9)
    fig.suptitle("Five cameras before and after normalisation", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_camera_normalisation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_distribution(labels_csv, out_dir):
    df = pd.read_csv(labels_csv)
    counts = df["diagnosis"].value_counts().sort_index()
    names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bars = ax.bar(range(len(counts)), counts.values, color="#4878a8")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{i}\n{n}" for i, n in enumerate(names)], fontsize=8)
    ax.set_ylabel("images", fontsize=9)
    ax.set_title(f"Grade distribution, n = {len(df)}", fontsize=9)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, str(v), ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig3_grade_distribution.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/aptos2019")
    ap.add_argument("--labels", default="data/processed/labels.csv")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # one per distinct camera resolution, together spanning all five grades
    ids = ["014508ccb9cb", "259d30f693b6", "00e4ddff966a", "0cb14014117d", "0981195eb9fb"]

    for p in (fig_pipeline(a.raw, a.out, "0cb14014117d"),
              fig_cameras(a.raw, a.out, ids, a.raw + "/train.csv"),
              fig_distribution(a.labels, a.out)):
        print("wrote", p)


if __name__ == "__main__":
    main()
