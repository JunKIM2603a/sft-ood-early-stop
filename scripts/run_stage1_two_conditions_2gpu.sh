#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <lr_gpu0> <seed_gpu0> <lr_gpu1> <seed_gpu1>"
  exit 2
fi

LR0="$1"
SEED0="$2"
LR1="$3"
SEED1="$4"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="artifacts/logs/stage1"
mkdir -p "${LOG_DIR}"

slug() {
  python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.0e}".replace("+", ""))
PY
}

RUN0="lr-$(slug "${LR0}")_seed-${SEED0}"
RUN1="lr-$(slug "${LR1}")_seed-${SEED1}"

echo "Launching two independent Stage-1 conditions"
echo "GPU 0 -> LR=${LR0}, seed=${SEED0} (${RUN0})"
echo "GPU 1 -> LR=${LR1}, seed=${SEED1} (${RUN1})"

CUDA_VISIBLE_DEVICES=0 python scripts/train_stage1.py   --learning-rate "${LR0}"   --seed "${SEED0}"   > "${LOG_DIR}/${RUN0}.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/train_stage1.py   --learning-rate "${LR1}"   --seed "${SEED1}"   > "${LOG_DIR}/${RUN1}.log" 2>&1 &
PID1=$!

echo "PID GPU0: ${PID0}"
echo "PID GPU1: ${PID1}"

set +e
wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
set -e

echo "GPU0 exit status: ${STATUS0}"
echo "GPU1 exit status: ${STATUS1}"

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "ERROR: at least one training condition failed."
  echo "Inspect:"
  echo "  ${LOG_DIR}/${RUN0}.log"
  echo "  ${LOG_DIR}/${RUN1}.log"
  exit 1
fi

echo "PASS: both Stage-1 training conditions completed."
