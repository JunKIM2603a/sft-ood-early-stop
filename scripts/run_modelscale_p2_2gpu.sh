#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-scientific}"
CONFIG="configs/modelscale_sentinel_7b_fulltrain.yaml"
DS_CONFIG="configs/deepspeed_zero3_nvme.json"
OUTPUT="checkpoints/modelscale_p2/qwen2.5-7b_fulltrain_lr-1e-06_seed-42"

if ! python -c "import deepspeed" >/dev/null 2>&1; then
  echo "ERROR: DeepSpeed is not installed."
  exit 1
fi

if [[ ! -f artifacts/audits/p2_7b_preflight.json ]]; then
  echo "ERROR: P2 preflight is missing."
  echo "Run: python scripts/data/preflight_modelscale_p2.py"
  exit 1
fi

mkdir -p artifacts/deepspeed_nvme/p2_7b

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

COMMON=(
  --config "${CONFIG}"
  --deepspeed-config "${DS_CONFIG}"
  --output-dir "${OUTPUT}"
)

if [[ "${MODE}" == "smoke" ]]; then
  echo "Launching P2 Qwen2.5-7B/full-train ZeRO-Infinity smoke"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py     "${COMMON[@]}"     --smoke     --overwrite-output     --min-free-disk-gib 400
elif [[ "${MODE}" == "scientific" ]]; then
  echo "Launching P2 Qwen2.5-7B/full-train model-scale sentinel"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py     "${COMMON[@]}"     --min-free-disk-gib 400
else
  echo "Usage: $0 [smoke|scientific]"
  exit 2
fi
