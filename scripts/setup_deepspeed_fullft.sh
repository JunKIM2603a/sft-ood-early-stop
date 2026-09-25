#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.fullft.txt

python - <<'PY'
import deepspeed
import torch

print("DeepSpeed:", deepspeed.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

if command -v ds_report >/dev/null 2>&1; then
  ds_report
else
  echo "WARN: ds_report command not found, but Python import succeeded."
fi

echo
echo "DeepSpeed full-FT dependency setup completed."
