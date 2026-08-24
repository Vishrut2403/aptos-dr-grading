# Diabetic Retinopathy Severity Grading

We compare four architecture families on
APTOS 2019. Everything goes through the same preprocessing, the same
train/val/test split, the same optimiser and the same metrics, so the model is
the only thing that changes between runs.

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

Makes `.venv` and installs the dependencies. If you already have PyTorch it
reuses it rather than pulling down another multi-gigabyte CUDA wheel, otherwise
it installs everything from `requirements.txt`. It prints whether it found CUDA.

A GPU is optional. The reference figures in `results/vram_bench.json` come from
a 6 GB RTX 3060 laptop GPU, where every model fits with room to spare. On CPU
the code runs unchanged, it just turns mixed precision off by itself. Measured
on 8 threads: the MLP trains in a couple of minutes, and the other three sit
around 5 to 6 minutes per epoch, so a full run is roughly 3 to 5 hours. Slow
enough that you won't want to iterate that way, fast enough to leave overnight.

## Getting the data

**APTOS 2019 Blindness Detection**, run by the Asia Pacific Tele-Ophthalmology
Society with images from Aravind Eye Hospital, India. It lives on Kaggle as a
competition dataset:
<https://www.kaggle.com/competitions/aptos2019-blindness-detection>

Free for academic use but gated, so there are two things to do in a browser
before anything downloads.

**1. Accept the data-use agreement.** Open the competition page and click
**Late Submission**. Kaggle only shows a "Join Competition" button while a
competition is still open and renames it afterwards, which trips people up. This
one closed in 2019, so you're entering nothing; the click is just how Kaggle
records that you accepted the licence on patient medical images. The API returns
403 until you do it, and that's almost always why a download fails.

**2. Log the CLI in.** Kaggle CLI 2.2+ uses a browser flow, so there's no token
file to manage:

```bash
.venv/bin/kaggle auth login
```

If you'd rather avoid OAuth, generate a token at
<https://www.kaggle.com/settings/api> and either `export KAGGLE_API_TOKEN=...`
or save it to `~/.kaggle/access_token`.

Then:

```bash
./scripts/download_data.sh          # ~9 GB, needs ~10 GB free while it unzips
```

You get `train.csv` (3,662 rows of `id_code`, `diagnosis` 0-4) and
`train_images/`. There's also `test.csv` and `test_images/`, but Kaggle withheld
those labels, so every split in this project comes out of the 3,662 labelled
training images. The script prints the class distribution when it's done; it
should be 1805 / 370 / 999 / 193 / 295.

Then build the cache:

```bash
.venv/bin/python -m src.preprocess --raw data/raw/aptos2019 --out data/processed --size 320
```

This crops the retinal circle out of the black border, applies Ben Graham
illumination normalisation, and writes a single uint8 array (about 1.1 GB at
320 px) so the dataloader doesn't end up being the bottleneck.

### What else you need

| Thing | Notes |
|---|---|
| Python 3.11+, PyTorch 2.6+ | `scripts/setup_env.sh` handles it |
| timm, scikit-learn, pandas, OpenCV | same script |
| ImageNet weights for EfficientNet-B0 and ViT-S/16 | timm pulls them from Hugging Face on first use, ~110 MB, no account or token needed |
| GPU | optional. Every model was measured to fit in 6 GB, see `results/vram_bench.json` |
| Disk | ~10 GB raw plus ~1.1 GB cache |
| Kaggle account | you have to make this yourself, see above |

Messidor-2 (1,748 independently graded fundus images) is listed in the synopsis
as an optional external validation set. It needs a separate request form at
<https://www.adcis.net/en/third-party/messidor2/>. Stretch goal, nothing depends
on it.

## Checking it fits, before downloading anything

```bash
.venv/bin/python -m src.vram_bench
```

Runs a forward and backward pass on synthetic batches and reports peak VRAM and
throughput for every model and batch size. No data required.

## Training

Same harness for every model, only `--model` changes:

```bash
.venv/bin/python -m src.train --model mlp         --head coral --epochs 40 --batch 256
.venv/bin/python -m src.train --model cnn_scratch --head coral --epochs 40 --batch 32
.venv/bin/python -m src.train --model effnet_b0   --head coral --epochs 25 --batch 32 --lr 3e-4
.venv/bin/python -m src.train --model vit_small   --head coral --epochs 25 --batch 32 --lr 1e-4
```

Results go to `results/<model>_<head>.json` with the full metric set and the
per-epoch history.

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
  losses.py          loss and decoder for each head type
  metrics.py         the shared evaluation protocol
  train.py           the harness every model is trained with
  vram_bench.py      memory and throughput probe

scripts/             environment setup and dataset download
results/             metrics JSON (committed); model weights (ignored)
data/                downloaded and cached data (ignored)
local/               not for publication (ignored)
```

## What's implemented so far

`src/train.py`, `src/models/__init__.py`, `src/vram_bench.py` and everything in
`scripts/` are done and working. The four backbones, the heads, the losses, the
metrics, the preprocessing and the dataset are skeletons that raise
`NotImplementedError`. Each one has a docstring with the architecture, the
contract it has to satisfy, and what to be ready to explain about it.

Note that the model files never touch an image, they take a tensor and return a
tensor, so you can write and test one against `torch.randn(4, 3, 224, 224)`
before the preprocessing cache exists, and without a GPU.

Accuracy metrics are hardware independent, but the `s_per_epoch` and `peak_MiB`
fields in the results JSON are not. Prototype anywhere; the four runs that go in
the report should all come off the same machine or the timing and memory columns
aren't comparable.

Authorship is tracked through git history, so commit under your own name. If you
used an LLM anywhere, say so in the commit message and say where.
