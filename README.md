# Diabetic Retinopathy Severity Grading — ICT 4442 Deep Learning Mini Project

A controlled comparison of four architecture families on APTOS 2019, all sharing one
preprocessing pipeline, one train/val/test split, one optimiser and one metric set.

| Model | Family | Pretrained |
|---|---|---|
| `mlp` | Fully connected baseline | no |
| `cnn_scratch` | Convolutional | no |
| `effnet_b0` | Convolutional | ImageNet |
| `vit_small` | Attention / Transformer | ImageNet |

## Setup

```bash
./scripts/setup_env.sh
```

Creates `.venv` and installs the dependencies. It reuses an existing PyTorch
install if there is one, rather than pulling a second multi-gigabyte CUDA wheel;
otherwise it installs everything from `requirements.txt`. It reports whether
CUDA was found.

Everything runs on CPU too, just slowly. The measured GPU figures in
`results/vram_bench.json` are from an RTX 3060 laptop GPU with 6 GB.

## Data — where it comes from

**APTOS 2019 Blindness Detection**, run by the Asia Pacific Tele-Ophthalmology Society with
images from Aravind Eye Hospital, India. It is hosted as a Kaggle *competition* dataset:
<https://www.kaggle.com/competitions/aptos2019-blindness-detection>

Free for academic use, but gated. Two steps, both needing a browser:

1. **Accept the data-use agreement.** Open the competition page and click
   **Late Submission** (Kaggle shows "Join Competition" only while a competition
   is still running, and renames the button once it closes). Accept the rules in
   the dialog. This enters you into nothing, the competition ended in 2019; it is
   how Kaggle records that you accepted the licence on patient medical imagery.
   The API returns HTTP 403 until you do, and this is the usual reason the
   download fails.
2. **Authenticate the CLI.** Kaggle CLI 2.2+ uses a browser flow, so there is no
   token file to manage:
   ```bash
   .venv/bin/kaggle auth login
   ```
   If you would rather not use OAuth, generate a token at
   <https://www.kaggle.com/settings/api> and either `export KAGGLE_API_TOKEN=...`
   or save it to `~/.kaggle/access_token`.

Then:

```bash
./scripts/download_data.sh          # ~9 GB, needs ~10 GB free while unzipping
```

You get `train.csv` (3,662 rows: `id_code`, `diagnosis` 0–4), `train_images/` (3,662 PNGs),
plus `test.csv` / `test_images/` whose **labels are withheld** — so every split in this project
is drawn from the 3,662 labelled training images. The script prints the actual class
distribution at the end; check it against the figures quoted in the synopsis.

Then build the cache:

```bash
.venv/bin/python -m src.preprocess --raw data/raw/aptos2019 --out data/processed --size 320
```

This crops the retinal circle, applies Ben Graham illumination normalisation, and caches one
uint8 array (~1.1 GB at 320 px) so the dataloader is not the bottleneck.

### Other things the project needs

| Requirement | Status | Notes |
|---|---|---|
| Python 3.14 + PyTorch 2.10 (CUDA 12.8) | already installed | reused via `scripts/setup_env.sh`, not re-downloaded |
| timm, scikit-learn, pandas, OpenCV | installed | `scripts/setup_env.sh` |
| ImageNet pretrained weights (EfficientNet-B0, ViT-S/16) | verified working | pulled automatically by `timm` from Hugging Face on first use; no account or token needed, ~110 MB total |
| GPU | RTX 3060 Laptop, 5,807 MiB usable | every model measured to fit; see `results/vram_bench.json` |
| Disk | ~10 GB for raw data + ~1.1 GB cache | 125 GB free at setup |
| Kaggle account | **you must create this** | required for the dataset, see above |
| Shared GitHub repo with per-member commits | **not yet created** | the guidelines require commit history as contribution evidence from the start, not one push at the end |
| Turnitin report | at final submission | ≤15% plagiarism, ≤20% AI content; LLM use must be declared with where exactly it was used |

Messidor-2 (1,748 independently graded fundus images) is noted in the synopsis as an optional
external-validation set. It needs a separate request form at
<https://www.adcis.net/en/third-party/messidor2/> and is a stretch goal, not a dependency.

## Feasibility check (no data required)

```bash
.venv/bin/python -m src.vram_bench
```

Measures peak VRAM and throughput for every model/batch-size combination on synthetic
batches, to confirm the study fits in 6 GB before committing to it.

## Training

Every member runs the same harness with a different `--model`:

```bash
.venv/bin/python -m src.train --model mlp         --head coral --epochs 40 --batch 256
.venv/bin/python -m src.train --model cnn_scratch --head coral --epochs 40 --batch 32
.venv/bin/python -m src.train --model effnet_b0   --head coral --epochs 25 --batch 32 --lr 3e-4
.venv/bin/python -m src.train --model vit_small   --head coral --epochs 25 --batch 32 --lr 1e-4
```

Results land in `results/<model>_<head>.json` with the full metric set and epoch history.

## Layout

```
src/
  preprocess.py      circle crop, Ben Graham normalisation, caching
  data.py            dataset, augmentation, the one shared split
  models/
    __init__.py      registry and DRModel assembly
    heads.py         softmax / regression / CORAL heads
    mlp.py           Model 1, fully connected baseline
    cnn_scratch.py   Model 2, VGG-style CNN, no pretraining
    pretrained.py    Model 3 EfficientNet-B0, Model 4 ViT-S/16
  losses.py          loss and decoder per head type
  metrics.py         the common evaluation protocol
  train.py           the harness every model is trained with
  vram_bench.py      memory and throughput feasibility probe

scripts/             environment setup and dataset download
results/             metrics JSON (committed); model weights (ignored)
data/                downloaded and cached data (ignored)
local/               not for publication (ignored, see above)
```

## Implementation status

The four backbones, the heads, the losses, the metrics, the preprocessing and
the dataset ship as documented skeletons that raise `NotImplementedError`. Each
carries the architecture, the contract it must satisfy, and what to be ready to
defend. The shared harness (`src/train.py`), the registry
(`src/models/__init__.py`) and everything in `scripts/` are complete.

Authorship is recorded by git commit history, so commit under your own name.
