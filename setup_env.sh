#!/usr/bin/env bash
# One-time setup (Git Bash on Windows). Run from the project folder:
#   bash setup_env.sh            # CUDA 12.4 build of PyTorch (default)
#   bash setup_env.sh cu121      # older driver
set -e
CUDA_TAG="${1:-cu124}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first:  pip install uv"
  exit 1
fi

uv venv .venv --python 3.11
source .venv/Scripts/activate

uv pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install -r requirements.txt

python check_gpu.py
echo
echo "Done. Activate later with:  source .venv/Scripts/activate"
