"""Bootstrap confidence intervals on test QWK, from the saved confusion matrices.

Objective 1 promises kappa with confidence intervals. A confusion matrix fully
determines the multiset of (true, predicted) pairs on the test split, so the
interval can be resampled from it without retraining or re-running inference.

    python -m scripts.bootstrap_ci [n_boot]
"""

import glob, json, sys

import numpy as np

from src.metrics import evaluate


def pairs(confusion):
    """Confusion matrix -> the (true, pred) pairs it implies, one row per image."""
    c = np.asarray(confusion)
    t, p = np.nonzero(c)
    return (np.repeat(t, c[t, p]), np.repeat(p, c[t, p]))


def ci(confusion, n_boot, rng):
    y, yhat = pairs(confusion)
    n = len(y)
    assert n > 0, "empty confusion matrix"
    qwk = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        qwk[b] = evaluate(y[i], yhat[i])["qwk"]
    return evaluate(y, yhat)["qwk"], np.percentile(qwk, [2.5, 97.5]), qwk.std()


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(42)
    rows = []
    for f in sorted(glob.glob("results/*.json")):
        d = json.load(open(f))
        if "test" not in d:
            continue
        point, (lo, hi), sd = ci(d["test"]["confusion"], n_boot, rng)
        rows.append((d["args"]["model"], d["args"]["head"], point, lo, hi, sd))

    rows.sort(key=lambda r: -r[2])
    print(f"{'model':<14}{'head':<13}{'qwk':>8}{'95% CI':>20}{'sd':>8}")
    print("-" * 63)
    for m, h, q, lo, hi, sd in rows:
        print(f"{m:<14}{h:<13}{q:>8.4f}   [{lo:.4f}, {hi:.4f}]{sd:>8.4f}")

    out = [{"model": m, "head": h, "qwk": q, "ci95": [lo, hi], "sd": sd}
           for m, h, q, lo, hi, sd in rows]
    json.dump({"n_boot": n_boot, "seed": 42, "runs": out},
              open("results/bootstrap_ci.json", "w"), indent=2)
    print(f"\nwrote results/bootstrap_ci.json  ({n_boot} resamples)")


if __name__ == "__main__":
    main()
