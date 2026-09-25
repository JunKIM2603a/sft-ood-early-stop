# Full-FT capacity sentinel backend

## Why the backend changed

The frozen six-run LoRA grid produced zero clear peak-to-decline runs, so the
predeclared full-FT capacity sentinel is required.

Two FSDP2 smoke attempts were made. The second attempt verified that FSDP2,
CPU offload, and activation checkpointing were active, but both ranks still
failed during the first Adam optimizer step while allocating Adam moment
tensors on GPU.

Therefore the next backend is **DeepSpeed ZeRO-3 with explicit CPU optimizer
and parameter offload**.

This is a systems/memory change only. Scientific variables stay fixed:

- Qwen2.5-3B-Instruct
- frozen 4,096 SFT examples
- LR 1e-6
- seed 42
- 1 epoch
- effective batch 64
- 64 optimizer steps
- BF16
- cosine scheduler

## Install the optional dependency

DeepSpeed is intentionally kept out of the primary LoRA environment lock.

```bash
git pull
bash scripts/setup_deepspeed_fullft.sh
```

Then verify:

```bash
python -c "import deepspeed; print(deepspeed.__version__)"
```

## Validate the plan

```bash
pytest -q tests/test_deepspeed_sentinel_plan.py
```

## Retry the engineering smoke

```bash
bash scripts/run_fullft_sentinel_2gpu.sh smoke
```

A valid run should report a distributed type containing `DEEPSPEED` and
print:

```text
ZeRO stage: 3
Optimizer offload: CPU
Parameter offload: CPU
```

Only after this smoke passes:

```bash
bash scripts/run_fullft_sentinel_2gpu.sh scientific
```

The existing two-GPU full-FT evaluation pipeline remains the next step after
the scientific run.

## Notes

The ZeRO-3 configuration uses conservative memory settings:

- per-device micro-batch 1;
- gradient accumulation 32;
- optimizer state/computation on CPU;
- model parameters offloaded to CPU when not resident;
- communication overlap disabled to avoid extra GPU buffers;
- smaller gather/prefetch buckets;
- full 16-bit model gathered only when checkpoints are saved.

This will be slower than LoRA or pure-GPU full FT, but the capacity sentinel is
only 64 optimizer steps.
