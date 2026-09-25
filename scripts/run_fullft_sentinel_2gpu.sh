#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-scientific}"

COMMON=(
  --multi_gpu
  --num_processes 2
  --mixed_precision bf16
)

if [[ "${MODE}" == "smoke" ]]; then
  echo "Launching 2-GPU FSDP2 full-FT smoke test"
  accelerate launch "${COMMON[@]}"     scripts/train_fullft_sentinel.py     --smoke     --overwrite-output
elif [[ "${MODE}" == "scientific" ]]; then
  echo "Launching 2-GPU FSDP2 full-FT capacity sentinel"
  accelerate launch "${COMMON[@]}"     scripts/train_fullft_sentinel.py
else
  echo "Usage: $0 [smoke|scientific]"
  exit 2
fi
