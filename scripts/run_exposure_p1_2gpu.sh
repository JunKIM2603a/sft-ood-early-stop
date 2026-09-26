#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-scientific}"
CONFIG="configs/exposure_sentinel_3b_fulltrain.yaml"
OUTPUT="checkpoints/exposure_p1/qwen2.5-3b_fulltrain_lr-1e-06_seed-42"

if ! python -c "import deepspeed" >/dev/null 2>&1; then
  echo "ERROR: DeepSpeed is not installed."
  echo "Run: bash scripts/setup_deepspeed_fullft.sh"
  exit 1
fi

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f data/local/generalpoints_exposure_p1/manifest.json ]]; then
  echo "ERROR: P1 exposure manifest is missing."
  echo "Run: python scripts/data/freeze_exposure_p1.py"
  exit 1
fi

COMMON=(
  --config "${CONFIG}"
  --output-dir "${OUTPUT}"
)

if [[ "${MODE}" == "smoke" ]]; then
  echo "Launching P1 3B/full-train ZeRO-3 engineering smoke"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py     "${COMMON[@]}"     --smoke     --overwrite-output
elif [[ "${MODE}" == "scientific" ]]; then
  echo "Launching P1 3B/full-train exposure sentinel"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py     "${COMMON[@]}"
else
  echo "Usage: $0 [smoke|scientific]"
  exit 2
fi
