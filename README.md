# Separating Architecture from Pretraining in Diabetic Retinopathy Grading

ICT 4442 Deep Learning mini project, Team G3. A controlled comparison of four
backbones for five-grade diabetic retinopathy (DR) grading on APTOS 2019, each
trained with three output heads. Every run shares the same preprocessing, the
same stratified split, the same schedule and the same metrics, and the backbones
are ordered so that each step changes one thing:

| Model | Family | Pretrained | What the step from the previous model adds |
|---|---|---|---|
| `mlp` | Fully connected | no | (baseline) |
| `cnn_scratch` | VGG-style CNN | no | convolution |
| `effnet_b0` | EfficientNet-B0 | ImageNet-1k | pretraining |
| `vit_small` | ViT-S/16 | ImageNet-21k | attention instead of convolution |

Heads: `softmax` (5 classes, cross-entropy), `regress` (one score, MSE) and
`coral` (CORAL ordinal head, four rank-consistent binary tasks).

## Results

Test set of 550 images; quadratic weighted kappa (QWK) with 95% bootstrap
intervals, and the regression head retrained with three seeds on the same split.

| Backbone | Best head | Test QWK [95% CI] | Regression, 3 seeds |
|---|---|---|---|
| EfficientNet-B0 | CORAL, refitted | **0.8956** [0.865, 0.921] | 0.8737 ± 0.0117 |
| ViT-S/16 | Regression | 0.8922 [0.858, 0.919] | **0.8896 ± 0.0047** |
| CNN (scratch) | Regression | 0.8785 [0.855, 0.900] | 0.8611 ± 0.0180 |
| MLP | Regression | 0.6174 [0.571, 0.663] | 0.6065 ± 0.0120 |

- **Spatial structure is the decisive step:** every other backbone beats the MLP
  by about 0.26 QWK.
- **The three deep models cannot be ranked:** the intervals of the top seven of
  sixteen configurations overlap, and seed-to-seed swings (up to 0.036) are as
  large as the gaps between models.
- **Regression is the best trained head on all four backbones.** The CORAL head
  lost everywhere because its learned thresholds collapsed; refitting only the
  thresholds on validation data lifted EfficientNet-B0 from 0.7596 to 0.8956.
- **Referral misses** (referable eyes, grade 2 or above, predicted 0 or 1, out of
  223): ViT 18, CNN 19, EfficientNet-B0 27, MLP 39.
- **Ben Graham normalisation had no measurable effect** in a controlled ablation.
- **Image sharpness is confounded with the grade:** grade 0 photographs are much
  sharper, and wrongly graded images are blurrier for every model.

The full table of sixteen configurations is in `results/` and is printed by
`python -m src.metrics results`.

## Team

| Member | Model | Also | GitHub |
|---|---|---|---|
| Aditya Verma | Model 1, MLP | data split, exploratory analysis | [@VermaAditya18](https://github.com/VermaAditya18) |
| Harshit Anand | Model 2, CNN from scratch | augmentation, class weights | [@Harshitdot1106](https://github.com/Harshitdot1106) |
| Vishrut Sachan | Model 3, EfficientNet-B0 | preprocessing, evaluation, experiments | [@Vishrut2403](https://github.com/Vishrut2403) |
| Saurabh Sharma | Model 4, ViT-S/16 | output heads, losses, CORAL refit | [@Saurabhkiwi](https://github.com/Saurabhkiwi) |

## Reproducing the results

All commands run from the repository root. Results go to `results/` as JSON;
model weights (`*.pt`) are written alongside but not committed.

**1. Environment.**

```bash
./scripts/setup_env.sh
```

Creates `.venv` and installs PyTorch, timm, scikit-learn, pandas and OpenCV,
reusing an existing PyTorch install if there is one. A GPU is optional: all
results come from one 6 GB RTX 3060 laptop GPU, and on CPU the code runs
unchanged with mixed precision switched off (measured on 8 threads: the MLP
trains in minutes, the other models take about 5 to 6 minutes per epoch).

**2. Data.** APTOS 2019 Blindness Detection is a Kaggle competition dataset,
free for academic use but gated. Open
<https://www.kaggle.com/competitions/aptos2019-blindness-detection>, click
**Late Submission** and accept the rules (downloads fail with 403 until you do),
then:

```bash
.venv/bin/kaggle auth login
./scripts/download_data.sh          # about 9 GB; prints the class counts 1805/370/999/193/295
```

Only the 3,662 labelled training images are used; Kaggle withholds the test labels.

**3. Preprocessing cache** (crop, resize to 224 px, Ben Graham normalisation,
circular mask, stored as one uint8 array):

```bash
.venv/bin/python -m src.preprocess --raw data/raw/aptos2019 --out data/processed --size 224
```

The split is already committed as `data/processed/split_seed42.json`
(2,563 / 549 / 550), so every run uses the same images.

**4. The 4 × 3 grid, CORAL refit, intervals and figure.**

```bash
./scripts/run_ablation.sh                         # 12 training runs, skips finished ones
.venv/bin/python -m scripts.refit_coral           # recut the CORAL thresholds, no retraining
.venv/bin/python -m scripts.bootstrap_ci          # 95% intervals -> results/bootstrap_ci.json
.venv/bin/python -m scripts.make_figures --results-only
.venv/bin/python -m src.metrics results           # the 16-configuration comparison table
```

**5. Three seeds** (seeds 1 and 2 for every backbone's regression head; the
split stays fixed at seed 42):

```bash
./scripts/run_seeds.sh
```

**6. Ben Graham ablation** (a second cache without the normalisation, same split):

```bash
.venv/bin/python -m src.preprocess --raw data/raw/aptos2019 --out data/processed_noben --size 224 --no-ben
cp data/processed/split_seed42.json data/processed_noben/
.venv/bin/python -m src.train --data data/processed_noben --cache-size 224 --out results_noben --model mlp --head softmax --epochs 40 --batch 256 --lr 3e-4
.venv/bin/python -m src.train --data data/processed_noben --cache-size 224 --out results_noben --model mlp --head regress --epochs 40 --batch 256 --lr 3e-4
.venv/bin/python -m src.train --data data/processed_noben --cache-size 224 --out results_noben --model cnn_scratch --head softmax --epochs 40 --batch 32 --lr 3e-4
.venv/bin/python -m src.train --data data/processed_noben --cache-size 224 --out results_noben --model effnet_b0 --head softmax --epochs 25 --batch 32 --lr 3e-4
```

**7. Error analysis** (per-grade recall, referral misses, sharpness confound):

```bash
for m in mlp cnn_scratch effnet_b0 vit_small; do
  .venv/bin/python -m scripts.error_analysis --model $m --head regress \
      --out results/error_analysis_${m}_regress.json
done
```

Training is resumable: rerunning an interrupted `src.train` command continues
from its last completed epoch, and refuses to resume if the settings differ.

## Layout

```
src/
  preprocess.py      retina crop, Ben Graham normalisation, cached uint8 array
  data.py            dataset, augmentation, the one shared split, class weights
  models/
    __init__.py      model registry, backbone + head assembly
    heads.py         softmax, regression and CORAL heads
    mlp.py           Model 1, fully connected baseline
    cnn_scratch.py   Model 2, VGG-style CNN, no pretraining
    pretrained.py    Model 3 EfficientNet-B0, Model 4 ViT-S/16 (timm)
  losses.py          loss and decoder for each head
  metrics.py         the shared evaluation protocol and comparison table
  train.py           the training harness every model uses
  vram_bench.py      GPU memory and throughput probe
scripts/
  setup_env.sh, download_data.sh       environment and data
  run_ablation.sh, run_seeds.sh        the 4 x 3 grid and the extra seeds
  refit_coral.py                       CORAL threshold refit
  bootstrap_ci.py                      bootstrap confidence intervals
  error_analysis.py                    per-grade, referral and sharpness analysis
  eda.py, make_figures.py              exploratory analysis and report figures
results/             metrics for every run (weights are not committed)
results_noben/       the Ben Graham ablation runs
reports/             exploratory analysis output and figures
data/                raw images and caches (not committed)
```

## Notes

- The APTOS images are patient data distributed under Kaggle's competition
  rules. They are not redistributed here; never commit anything under `data/`.
- Accuracy metrics are hardware independent; the timing and peak-memory fields
  in the results JSON describe the RTX 3060 used for every reported run.
- Claude (Anthropic) was used as a coding and drafting assistant during
  development; the final report's AI-use declaration lists where.
