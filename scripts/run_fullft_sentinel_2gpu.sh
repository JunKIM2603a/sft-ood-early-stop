#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-scientific}"
ACCEL_CONFIG="configs/accelerate_fullft_fsdp2.yaml"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "ERROR: missing ${ACCEL_CONFIG}"
  exit 1
fi

if [[ "${MODE}" == "smoke" ]]; then
  echo "Launching 2-GPU explicit FSDP2 + CPU-offload full-FT smoke test"
  accelerate launch     --config_file "${ACCEL_CONFIG}"     scripts/train_fullft_sentinel.py     --smoke     --overwrite-output
elif [[ "${MODE}" == "scientific" ]]; then
  echo "Launching 2-GPU explicit FSDP2 + CPU-offload full-FT capacity sentinel"
  accelerate launch     --config_file "${ACCEL_CONFIG}"     scripts/train_fullft_sentinel.py
else
  echo "Usage: $0 [smoke|scientific]"
  exit 2
fi
