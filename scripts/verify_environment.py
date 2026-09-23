#!/usr/bin/env python3
"""Fast environment smoke test for the Stage-1 machine."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=== Python ===")
    print("version:", platform.python_version())
    print("executable:", sys.executable)

    if sys.version_info[:2] != (3, 11):
        print("FAIL: expected Python 3.11.x")
        return 1

    try:
        import torch
        import transformers
        import datasets
        import peft
        import accelerate
        import huggingface_hub
    except Exception as exc:
        print(f"FAIL: core import failed: {exc}")
        return 1

    print("\n=== Packages ===")
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("datasets:", datasets.__version__)
    print("peft:", peft.__version__)
    print("accelerate:", accelerate.__version__)
    print("huggingface_hub:", huggingface_hub.__version__)

    print("\n=== CUDA ===")
    print("torch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available to PyTorch")
        return 1

    if torch.cuda.device_count() < 2:
        print("WARN: fewer than 2 GPUs are visible; Stage 1 can still run serially.")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {torch.cuda.get_device_name(index)} | "
            f"{props.total_memory / 1024**3:.2f} GiB"
        )

    print("BF16 supported:", torch.cuda.is_bf16_supported())
    if not torch.cuda.is_bf16_supported():
        print("FAIL: Stage 1 is configured for bf16")
        return 1

    print("\nPASS: core Stage-1 environment is usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
