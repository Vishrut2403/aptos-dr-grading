"""Refit the CORAL thresholds on the validation split, no retraining.

The trained heads leave their four biases pinned near the zero init while the
shared weight scales the logit to a range of about 36, so all four thresholds
land in a pinhole and grades 1 and 3 become unreachable. The 1-D projection
itself is fine (Spearman 0.89 against the true grade), so we keep the network
and only recut it, placing each threshold at the validation quantile that
matches the training base rate for that task.

    python -m scripts.refit_coral
"""

import glob, json, os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import APTOSDataset
from src.metrics import evaluate
from src.models import build

NPY, CSV = "data/processed/images_224.npy", "data/processed/labels.csv"


def shared_logits(model, idx, img_size, device):
    """The scalar w.x before the biases are added, plus the true grades."""
    ds = APTOSDataset(NPY, CSV, idx, img_size=img_size, train=False)
    L, Y = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, num_workers=2):
            L.append(model.head.shared(model.backbone(x.to(device))).squeeze(1).cpu())
            Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def main():
    split = json.load(open("data/processed/split_seed42.json"))
    labels = pd.read_csv(CSV)["diagnosis"].values
    train_y = labels[np.array(split["train"])]
    rates = [(train_y > k).mean() for k in range(4)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt in sorted(glob.glob("results/*_coral.pt")):
        name = os.path.basename(ckpt)[:-len("_coral.pt")]
        blob = torch.load(ckpt, map_location="cpu")
        model, img_size = build(name, head_type="coral", pretrained=False)
        model.load_state_dict(blob["model"])
        model.to(device).eval()

        lv, _ = shared_logits(model, split["val"], img_size, device)
        lt, yt = shared_logits(model, split["test"], img_size, device)

        cuts = np.array([np.quantile(lv, 1 - r) for r in rates])
        assert (cuts[:-1] <= cuts[1:]).all(), f"{name}: cuts not ascending"

        old_b = model.head.biases.detach().cpu().numpy()
        old = evaluate(yt, (lt[:, None] > -old_b[None, :]).sum(1))
        new = evaluate(yt, (lt[:, None] > cuts[None, :]).sum(1))

        out = dict(blob["args"])
        out["head"] = "coral_refit"
        json.dump({"tag": f"{name}_coral_refit", "args": out,
                   "best_epoch": blob["args"].get("epochs"), "test": new,
                   "thresholds": cuts.tolist(), "learned_biases": old_b.tolist(),
                   "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                   "history": []},
                  open(f"results/{name}_coral_refit.json", "w"), indent=2)
        print(f"{name:14} qwk {old['qwk']:.4f} -> {new['qwk']:.4f}   "
              f"mae {old['mae']:.3f} -> {new['mae']:.3f}")


if __name__ == "__main__":
    main()
