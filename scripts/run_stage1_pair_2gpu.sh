#!/usr/bin/env bash
set -uo pipefail

LR="${1:-5e-5}"
SEED0="${2:-42}"
SEED1="${3:-43}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="artifacts/logs/stage1"
mkdir -p "${LOG_DIR}"

RUN0="lr-$(python - <<PY
print(f"{float('${LR}'):.0e}".replace("+",""))
PY
)_seed-${SEED0}"
RUN1="lr-$(python - <<PY
print(f"{float('${LR}'):.0e}".replace("+",""))
PY
)_seed-${SEED1}"

echo "Launching two independent Stage-1 runs"
echo "GPU 0 -> LR=${LR}, seed=${SEED0}"
echo "GPU 1 -> LR=${LR}, seed=${SEED1}"
echo "Logs: ${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python scripts/train_stage1.py   --learning-rate "${LR}"   --seed "${SEED0}"   > "${LOG_DIR}/${RUN0}.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/train_stage1.py   --learning-rate "${LR}"   --seed "${SEED1}"   > "${LOG_DIR}/${RUN1}.log" 2>&1 &
PID1=$!

echo "PID GPU0: ${PID0}"
echo "PID GPU1: ${PID1}"
echo "Use another terminal for: tail -f ${LOG_DIR}/${RUN0}.log"
echo "Use another terminal for: tail -f ${LOG_DIR}/${RUN1}.log"

set +e
wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
set -e

echo "GPU0 run exit status: ${STATUS0}"
echo "GPU1 run exit status: ${STATUS1}"

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "ERROR: at least one Stage-1 run failed. Inspect logs before evaluation."
  exit 1
fi

echo "PASS: both Stage-1 training runs completed."
