"""The common evaluation protocol: every model is scored by this one function.

    python -m src.metrics results
"""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

NUM_CLASSES = 5
LABELS = list(range(NUM_CLASSES))


def evaluate(y_true, y_pred):
    """True and predicted grades, each (N,) -> the metric dict for one run."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        # without labels= sklearn infers the set from the values present, so a
        # split missing a grade renumbers them and changes every distance
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "referable_f1": float(f1_score(y_true >= 2, y_pred >= 2, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def comparison_table(results):
    """Render the single cross-model table, from the dicts src/train.py writes."""
    cols = ["qwk", "accuracy", "macro_f1", "mae", "referable_f1"]
    head = f"{'model':<16}{'head':<9}{'params(M)':>10}" + "".join(f"{c:>13}" for c in cols)
    lines = [head, "-" * len(head)]
    for r in sorted(results, key=lambda r: -r["test"]["qwk"]):
        t = r["test"]
        lines.append(f"{r['args']['model']:<16}{r['args']['head']:<9}{r['params_M']:>10}"
                     + "".join(f"{t[c]:>13.4f}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    import glob, json, sys

    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(f"{d}/*.json"))
    if not files:
        sys.exit(f"no result files in {d}/")
    print(comparison_table([json.load(open(p)) for p in files]))
