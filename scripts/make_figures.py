"""Report figures for the preprocessing section.

    python -m scripts.make_figures --raw data/raw/aptos2019 --out reports/figures
"""

import argparse
import json
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


# categorical slots 1-4, validated for CVD separation; marker shape is the
# secondary encoding the two low-contrast hues need
HEADS = [("softmax", "#2a78d6", "o"), ("regress", "#eb6834", "s"),
         ("coral", "#1baf7a", "^"), ("coral_refit", "#eda100", "D")]


def fig_results(ci_json, out_dir):
    runs = {(r["model"], r["head"]): r for r in json.load(open(ci_json))["runs"]}
    models = [("mlp", "MLP"), ("cnn_scratch", "CNN (scratch)"),
              ("effnet_b0", "EfficientNet-B0"), ("vit_small", "ViT-S/16")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y, ticks = 0, []
    for key, label in models:
        ticks.append((y + 1.5, label))
        for j, (head, color, marker) in enumerate(HEADS):
            r = runs[(key, head)]
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [y + j] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(r["qwk"], y + j, marker, color=color, ms=6,
                    mec="#fcfcfb", mew=1, label=head if key == "mlp" else None)
        y += len(HEADS) + 1.5
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=8, color="#0b0b0b")
    ax.invert_yaxis()
    ax.set_xlabel("test quadratic weighted kappa, 95% bootstrap CI", fontsize=8, color="#52514e")
    ax.tick_params(axis="x", labelsize=8, colors="#52514e")
    ax.grid(axis="x", color="#e4e3dd", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_results_ci.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/aptos2019")
    ap.add_argument("--labels", default="data/processed/labels.csv")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--results-only", action="store_true", help="skip figures needing raw images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # one per distinct camera resolution, together spanning all five grades
    ids = ["014508ccb9cb", "259d30f693b6", "00e4ddff966a", "0cb14014117d", "0981195eb9fb"]

    figs = [fig_results("results/bootstrap_ci.json", a.out)]
    if not a.results_only:
        figs += [fig_pipeline(a.raw, a.out, "0cb14014117d"),
                 fig_cameras(a.raw, a.out, ids, a.raw + "/train.csv"),
                 fig_distribution(a.labels, a.out)]
    for p in figs:
        print("wrote", p)


if __name__ == "__main__":
    main()
