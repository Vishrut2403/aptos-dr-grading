#!/usr/bin/env bash
# Two extra training seeds for each backbone's best trained head (regression for
# all four), on the same 224 cache and the same seed-42 split. Seed 42 is the
# existing run in results/. Skips any run already done.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {  # model epochs batch lr seed
  [ -f "results/$1_regress_s$5.json" ] && { echo "skip  $1 seed $5"; return; }
  echo "run   $1 seed $5"
  $PY -m src.train --data data/processed --cache-size 224 --model "$1" --head regress \
      --epochs "$2" --batch "$3" --lr "$4" --seed "$5" || echo "FAILED $1 seed $5"
}

for seed in 1 2; do
  run mlp         40 256 3e-4 "$seed"
  run cnn_scratch 40 32  3e-4 "$seed"
  run effnet_b0   25 32  3e-4 "$seed"
  run vit_small   25 32  1e-4 "$seed"
done
echo "=== DONE ==="
