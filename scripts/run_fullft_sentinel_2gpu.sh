#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-scientific}"

if ! python -c "import deepspeed" >/dev/null 2>&1; then
  echo "ERROR: DeepSpeed is not installed."
  echo "Run: bash scripts/setup_deepspeed_fullft.sh"
  exit 1
fi

# Accelerate explicitly requires these on RTX 4000-series multi-GPU systems
# when the process is launched via torchrun rather than accelerate launch.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

if [[ "${MODE}" == "smoke" ]]; then
  echo "Launching 2-GPU DeepSpeed ZeRO-3 CPU-offload full-FT smoke test"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py     --smoke     --overwrite-output
elif [[ "${MODE}" == "scientific" ]]; then
  echo "Launching 2-GPU DeepSpeed ZeRO-3 CPU-offload capacity sentinel"
  torchrun     --standalone     --nproc_per_node=2     scripts/train_fullft_sentinel_zero3.py
else
  echo "Usage: $0 [smoke|scientific]"
  exit 2
fi
