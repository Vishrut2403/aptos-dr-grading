"""Feasibility probe: can each of the four models train inside 6 GB of VRAM?

Runs real forward+backward+step on synthetic batches of the right shape and
reports peak allocated memory and throughput. No dataset needed, so this can
be run before the APTOS download finishes.

    python -m src.vram_bench
"""

import argparse, json, time
import torch

from .models import build
from .losses import loss_fn, NUM_CLASSES


def probe(name, head_type, batch_size, amp=True, steps=6, grad_ckpt=False, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, img_size = build(name, head_type=head_type, pretrained=False)
    model = model.to(device).train()

    if grad_ckpt and hasattr(model.backbone, "net") and hasattr(model.backbone.net, "set_grad_checkpointing"):
        model.backbone.net.set_grad_checkpointing(True)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    t0 = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = loss_fn(out.float(), y, head_type)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if i == 1:  # skip warmup steps
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    timed_steps = steps - 2
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20

    del model, opt, x, y, out, loss
    torch.cuda.empty_cache()

    return {
        "model": name, "head": head_type, "img": img_size, "batch": batch_size,
        "amp": amp, "grad_ckpt": grad_ckpt,
        "params_M": round(n_params / 1e6, 2),
        "peak_MiB": round(peak, 1), "reserved_MiB": round(reserved, 1),
        "img_per_s": round(batch_size * timed_steps / elapsed, 1),
        "s_per_epoch_3662": round(3662 / (batch_size * timed_steps / elapsed), 1),
    }


PLAN = [
    # (model, head, batch, amp, grad_ckpt)
    ("mlp",         "coral",  256, True,  False),
    ("cnn_scratch", "coral",   32, True,  False),
    ("cnn_scratch", "coral",   64, True,  False),
    ("effnet_b0",   "coral",   32, True,  False),
    ("resnet50",    "coral",   32, True,  False),
    ("vit_small",   "coral",   32, True,  False),
    ("vit_small",   "coral",   64, True,  False),
    ("vit_base",    "coral",   16, True,  False),
    ("vit_base",    "coral",   32, True,  False),
    ("vit_base",    "coral",   32, True,  True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/vram_bench.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"{torch.cuda.get_device_name(0)}  total VRAM {total:.0f} MiB\n")

    hdr = f"{'model':<13}{'head':<8}{'bs':>5}{'ckpt':>6}{'params(M)':>11}{'peak(MiB)':>11}{'img/s':>9}{'s/epoch':>9}  fits"
    print(hdr); print("-" * len(hdr))

    rows = []
    for name, head, bs, amp, ck in PLAN:
        try:
            r = probe(name, head, bs, amp=amp, grad_ckpt=ck)
            r["fits"] = r["reserved_MiB"] < total * 0.92
            rows.append(r)
            print(f"{r['model']:<13}{r['head']:<8}{bs:>5}{str(ck):>6}{r['params_M']:>11}"
                  f"{r['peak_MiB']:>11}{r['img_per_s']:>9}{r['s_per_epoch_3662']:>9}  "
                  f"{'YES' if r['fits'] else 'no'}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"model": name, "head": head, "batch": bs, "grad_ckpt": ck, "fits": False, "error": "OOM"})
            print(f"{name:<13}{head:<8}{bs:>5}{str(ck):>6}{'':>11}{'OOM':>11}{'':>9}{'':>9}  no")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "total_MiB": total, "runs": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
