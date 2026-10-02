"""Per-grade errors, referral misses and the sharpness confound, for one trained model.

The Final criterion needs an error analysis, not just a comparison table. Three
questions: which grades does the model confuse, how many referable cases does it
send home, and does it do better on the sharper images that the EDA found are
mostly grade 0.

    python -m scripts.error_analysis --model effnet_b0 --head softmax
"""

import argparse, json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import APTOSDataset
from src.losses import decode
from src.metrics import evaluate
from src.models import build

NPY, CSV = "data/processed/images_224.npy", "data/processed/labels.csv"
NAMES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]


def predict(model, head, idx, img_size, device):
    ds = APTOSDataset(NPY, CSV, idx, img_size=img_size, train=False)
    P, Y = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, num_workers=2):
            P.append(decode(model(x.to(device)).float(), head).cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="effnet_b0")
    ap.add_argument("--head", default="softmax")
    ap.add_argument("--out", default="results/error_analysis.json")
    a = ap.parse_args()

    split = json.load(open("data/processed/split_seed42.json"))
    test = np.array(split["test"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, img_size = build(a.model, head_type=a.head, pretrained=False)
    model.load_state_dict(torch.load(f"results/{a.model}_{a.head}.pt", map_location="cpu")["model"])
    model.to(device).eval()
    y, p = predict(model, a.head, test, img_size, device)

    # per grade: how often it is found, and which side the misses land on
    per_grade = []
    for g in range(5):
        m = y == g
        per_grade.append({"grade": NAMES[g], "n": int(m.sum()),
                          "recall": float((p[m] == g).mean()),
                          "mean_pred": float(p[m].mean()),
                          "under": int((p[m] < g).sum()), "over": int((p[m] > g).sum())})

    # referral: grades 2-4 need an ophthalmologist, so a miss here is the costly error
    ref_true, ref_pred = y >= 2, p >= 2
    referral = {"n_referable": int(ref_true.sum()),
                "missed": int((ref_true & ~ref_pred).sum()),
                "missed_rate": float((~ref_pred[ref_true]).mean()),
                "false_alarm": int((~ref_true & ref_pred).sum()),
                "sensitivity": float(ref_pred[ref_true].mean()),
                "specificity": float((~ref_pred[~ref_true]).mean())}

    # sharpness confound: the EDA found grade 0 images are sharper (median Laplacian
    # variance 68.8 against 38-40), so accuracy by blur quartile says whether the
    # model is partly reading image quality rather than pathology
    blur = pd.read_csv("reports/eda_quality.csv")["blur"].values[test]
    q = np.quantile(blur, [0.25, 0.5, 0.75])
    bins = np.digitize(blur, q)
    sharpness = [{"quartile": int(b + 1), "n": int((bins == b).sum()),
                  "accuracy": float((y[bins == b] == p[bins == b]).mean()),
                  "median_blur": float(np.median(blur[bins == b])),
                  "frac_grade0": float((y[bins == b] == 0).mean())} for b in range(4)]
    correct = y == p
    sharpness_gap = {"median_blur_correct": float(np.median(blur[correct])),
                     "median_blur_wrong": float(np.median(blur[~correct]))}

    out = {"model": a.model, "head": a.head, "overall": evaluate(y, p),
           "per_grade": per_grade, "referral": referral,
           "sharpness": sharpness, "sharpness_gap": sharpness_gap}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"{a.model} / {a.head}   test QWK {out['overall']['qwk']:.4f}\n")
    print(f"{'grade':<17}{'n':>5}{'recall':>9}{'mean pred':>11}{'under':>7}{'over':>6}")
    for r in per_grade:
        print(f"{r['grade']:<17}{r['n']:>5}{r['recall']:>9.3f}{r['mean_pred']:>11.2f}"
              f"{r['under']:>7}{r['over']:>6}")
    print(f"\nreferable (grade >= 2): {referral['n_referable']} images, "
          f"{referral['missed']} missed ({referral['missed_rate']:.1%}), "
          f"sensitivity {referral['sensitivity']:.3f}, specificity {referral['specificity']:.3f}")
    print(f"\n{'blur quartile':<15}{'n':>5}{'accuracy':>10}{'median blur':>13}{'frac grade 0':>14}")
    for r in sharpness:
        print(f"{r['quartile']:<15}{r['n']:>5}{r['accuracy']:>10.3f}"
              f"{r['median_blur']:>13.1f}{r['frac_grade0']:>14.3f}")
    print(f"\nmedian blur, correct {sharpness_gap['median_blur_correct']:.1f} "
          f"vs wrong {sharpness_gap['median_blur_wrong']:.1f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
