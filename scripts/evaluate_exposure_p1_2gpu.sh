#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="qwen2.5-3b_fulltrain_lr-1e-06_seed-42"
RUN_DIR="checkpoints/exposure_p1/${RUN_NAME}"
MANIFEST="data/local/generalpoints_exposure_p1/manifest.json"

if [[ ! -f "${RUN_DIR}/checkpoints.jsonl" ]]; then
  echo "ERROR: missing ${RUN_DIR}/checkpoints.jsonl"
  exit 1
fi

readarray -t SHARDS < <(
python - "${RUN_DIR}/checkpoints.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
steps = [int(row["optimizer_step"]) for row in rows]
print(",".join(str(step) for step in steps[0::2]))
print(",".join(str(step) for step in steps[1::2]))
PY
)

STEPS0="${SHARDS[0]}"
STEPS1="${SHARDS[1]}"

echo "GPU0 steps: ${STEPS0}"
echo "GPU1 steps: ${STEPS1}"

mkdir -p artifacts/logs/exposure_p1_eval

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage1.py   --manifest "${MANIFEST}"   --run-dir "${RUN_DIR}"   --steps "${STEPS0}"   --skip-id-validation-loss   --eval-batch-size 32   --loss-batch-size 16   --output-root results/exposure_p1_gpu0   --raw-root results/raw/exposure_p1_gpu0   > artifacts/logs/exposure_p1_eval/gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_stage1.py   --manifest "${MANIFEST}"   --run-dir "${RUN_DIR}"   --steps "${STEPS1}"   --skip-id-validation-loss   --eval-batch-size 32   --loss-batch-size 16   --output-root results/exposure_p1_gpu1   --raw-root results/raw/exposure_p1_gpu1   > artifacts/logs/exposure_p1_eval/gpu1.log 2>&1 &
PID1=$!

echo "GPU0 eval PID: ${PID0}"
echo "GPU1 eval PID: ${PID1}"

set +e
wait "${PID0}"
S0=$?
wait "${PID1}"
S1=$?
set -e

echo "GPU0 exit: ${S0}"
echo "GPU1 exit: ${S1}"

if [[ "${S0}" -ne 0 || "${S1}" -ne 0 ]]; then
  echo "ERROR: P1 evaluation failed."
  echo "Inspect artifacts/logs/exposure_p1_eval/gpu0.log and gpu1.log"
  exit 1
fi

python scripts/merge_exposure_p1_eval.py

echo "PASS: P1 exposure evaluation and merge completed."
