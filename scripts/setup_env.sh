#!/usr/bin/env bash
# Create .venv and install everything needed to run the project.
#
#     ./scripts/setup_env.sh
#
# If a working PyTorch is already installed for the system Python, this reuses
# it instead of downloading a second multi-gigabyte CUDA wheel. Otherwise it
# installs everything from requirements.txt normally.
set -euo pipefail
cd "$(dirname "$0")/.."

PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SP=".venv/lib/python${PYV}/site-packages"

# Does the system Python already have torch, and where does it live?
TORCH_DIR=$(python3 - <<'PY' 2>/dev/null || true
import importlib.util, pathlib
spec = importlib.util.find_spec("torch")
if spec and spec.origin:
    print(pathlib.Path(spec.origin).parent.parent)
PY
)

if [ -n "$TORCH_DIR" ] && [ -d "$TORCH_DIR" ]; then
  echo "Reusing the PyTorch already installed at $TORCH_DIR"
  python3 -m venv --system-site-packages .venv
  # A venv inherits the system tree but not ~/.local, so link the torch stack
  # in by hand when that is where it lives.
  if [[ "$TORCH_DIR" == "$HOME/.local"* ]]; then
    shopt -s nullglob
    for p in torch torchgen functorch nvidia triton; do
      [ -e "$TORCH_DIR/$p" ] && ln -sfn "$TORCH_DIR/$p" "$SP/$p"
    done
    for d in "$TORCH_DIR"/torch-*.dist-info "$TORCH_DIR"/triton-*.dist-info; do
      ln -sfn "$d" "$SP/$(basename "$d")"
    done
  fi
  # Matching torchvision for the installed torch, then the rest without
  # letting pip drag in a second torch.
  TV=$(.venv/bin/python - <<'PY'
import torch
major, minor = torch.__version__.split(".")[:2]
print(f"0.{int(minor) + 15}.*")   # torch 2.10 -> torchvision 0.25
PY
)
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q --no-deps "torchvision==$TV" || \
    .venv/bin/python -m pip install -q --no-deps torchvision
  .venv/bin/python -m pip install -q \
    timm safetensors huggingface-hub \
    scikit-learn scipy joblib threadpoolctl narwhals \
    pandas python-dateutil pytz tzdata \
    opencv-python-headless tqdm kaggle
else
  echo "No PyTorch found, installing everything from requirements.txt"
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

echo
.venv/bin/python - <<'PY'
import torch, torchvision, timm, sklearn, pandas, cv2
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
print("timm       ", timm.__version__)
print("CUDA       ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU only, training will be slow)")
PY
echo
echo "Done. Next: ./scripts/download_data.sh"
