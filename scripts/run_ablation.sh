#!/usr/bin/env bash
# The 4x3 backbone-by-head ablation, resumable.
#
# Skips any run whose results JSON already exists AND was produced from the
# same cache, so it can be stopped with Ctrl-C and restarted without losing
# finished work. Every run must share one cache: preprocessing changes the
# numbers (cnn_scratch/softmax scored 0.8753 at 320 and 0.8449 at 224).
#
#     ./scripts/run_ablation.sh          # 224 cache, the default
#     CACHE=320 ./scripts/run_ablation.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CACHE="${CACHE:-224}"

run() {  # model head epochs batch lr
  local tag="$1_$2" f="results/$1_$2.json"
  if [ -f "$f" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$f'))['args']['cache_size']==$CACHE else 1)" 2>/dev/null; then
    echo "skip  $tag (already done at ${CACHE}px)"; return
  fi
  echo "run   $tag"
  $PY -m src.train --data data/processed --cache-size "$CACHE" \
      --model "$1" --head "$2" --epochs "$3" --batch "$4" --lr "$5" || echo "FAILED $tag"
}

for head in softmax regress coral; do
  run mlp         "$head" 40 256 3e-4
  run cnn_scratch "$head" 40 32  3e-4
  run effnet_b0   "$head" 25 32  3e-4
  run vit_small   "$head" 25 32  1e-4
done

echo "=== DONE ==="
$PY -m src.metrics results
