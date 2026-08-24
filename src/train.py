"""Common training harness. Every member runs this same script with a different
--model flag, which is what makes the comparison table in the report honest.

    python -m src.train --model vit_small --head coral --epochs 25
"""

import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import build
from .losses import loss_fn, decode
from .metrics import evaluate
from .data import APTOSDataset, make_split, class_weights


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--head", default="coral", choices=["softmax", "regress", "coral"])
    p.add_argument("--data", default="data/processed")
    p.add_argument("--cache-size", type=int, default=320)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--balanced", action="store_true", help="inverse-frequency class weights")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", default="results")
    return p.parse_args()


def run_epoch(model, loader, head, device, opt=None, scaler=None, cw=None, accum=1):
    train = opt is not None
    model.train(train)
    total, n, preds, gts = 0.0, 0, [], []

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler is not None and scaler.is_enabled()):
                out = model(x)
                loss = loss_fn(out.float(), y, head, class_weights=cw)

        if train:
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        total += loss.item() * y.size(0)
        n += y.size(0)
        preds.append(decode(out.float(), head).cpu().numpy())
        gts.append(y.cpu().numpy())

    return total / n, np.concatenate(gts), np.concatenate(preds)


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    npy = os.path.join(a.data, f"images_{a.cache_size}.npy")
    csv = os.path.join(a.data, "labels.csv")
    split_path = os.path.join(a.data, f"split_seed{a.seed}.json")
    if not os.path.exists(split_path):
        make_split(csv, split_path, seed=a.seed)
    split = json.load(open(split_path))

    model, img_size = build(a.model, head_type=a.head, pretrained=not a.no_pretrained)
    model = model.to(device)
    if a.grad_ckpt and hasattr(model.backbone, "net") and hasattr(model.backbone.net, "set_grad_checkpointing"):
        model.backbone.net.set_grad_checkpointing(True)

    sets = {k: APTOSDataset(npy, csv, split[k], img_size=img_size, train=(k == "train"))
            for k in ("train", "val", "test")}
    loaders = {k: DataLoader(v, batch_size=a.batch, shuffle=(k == "train"),
                             num_workers=a.workers, pin_memory=cuda, drop_last=(k == "train"))
               for k, v in sets.items()}

    cw = class_weights(csv, split["train"], device=device) if a.balanced else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # fp16 is a CUDA-only win here; on CPU it just warns and disables itself
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not a.no_amp)

    tag = f"{a.model}_{a.head}" + ("_bal" if a.balanced else "")
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"{tag}.pt")

    best_qwk, best_epoch, history = -1.0, -1, []
    for ep in range(a.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_y, tr_p = run_epoch(model, loaders["train"], a.head, device, opt, scaler, cw, a.accum)
        va_loss, va_y, va_p = run_epoch(model, loaders["val"], a.head, device, scaler=scaler)
        sched.step()

        tr_m, va_m = evaluate(tr_y, tr_p), evaluate(va_y, va_p)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_qwk": tr_m["qwk"], **{f"val_{k}": v for k, v in va_m.items() if k != "confusion"}})
        peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
        print(f"ep {ep:3d} | train {tr_loss:.4f} qwk {tr_m['qwk']:.4f} | "
              f"val {va_loss:.4f} qwk {va_m['qwk']:.4f} acc {va_m['accuracy']:.4f} "
              f"f1 {va_m['macro_f1']:.4f} | {time.perf_counter()-t0:.1f}s | {peak:.0f}MiB")

        if va_m["qwk"] > best_qwk:
            best_qwk, best_epoch = va_m["qwk"], ep
            torch.save({"model": model.state_dict(), "args": vars(a), "val": va_m}, ckpt)
        elif ep - best_epoch >= a.patience:
            print(f"early stop: no val QWK improvement for {a.patience} epochs")
            break

    # test with the best-validation checkpoint, never the last one
    model.load_state_dict(torch.load(ckpt)["model"])
    _, te_y, te_p = run_epoch(model, loaders["test"], a.head, device, scaler=scaler)
    te_m = evaluate(te_y, te_p)

    result = {"tag": tag, "args": vars(a), "best_epoch": best_epoch,
              "best_val_qwk": best_qwk, "test": te_m, "history": history,
              "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
              "peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None}
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTEST  qwk {te_m['qwk']:.4f}  acc {te_m['accuracy']:.4f}  "
          f"macroF1 {te_m['macro_f1']:.4f}  MAE {te_m['mae']:.3f}  refF1 {te_m['referable_f1']:.4f}")
    print("confusion:")
    for row in te_m["confusion"]:
        print("  " + " ".join(f"{v:4d}" for v in row))


if __name__ == "__main__":
    main()
