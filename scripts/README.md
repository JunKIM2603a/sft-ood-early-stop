# Scripts

Planned command-line entry points:

- `train_stage1.py` — run one frozen Stage-1 LR/seed LoRA condition and save 10-step adapters
- `evaluate_checkpoints.py` — evaluate ID/OOD metrics for a checkpoint series
- `measure_functional_drift.py` — compute fixed-anchor KL drift
- `measure_spectral_drift.py` — compute singular-vector / principal-angle drift
- `select_checkpoint.py` — apply label-free and baseline selectors
- `aggregate_results.py` — compute regret tables and main-figure inputs

Keep orchestration thin; reusable logic belongs in `src/sft_ood_early_stop/`.


## Current smoke-test entry points

Before implementing the full training grid:

```bash
python scripts/verify_environment.py
python scripts/data/smoke_generalpoints.py
CUDA_VISIBLE_DEVICES=<gpu> python scripts/smoke/train_lora_smoke.py
```

See `docs/gpu_smoke_test.md` for the PASS criteria and expected artifacts.


## Stage-1 preflight and first pilot

After the GPU smoke test passes:

```bash
python scripts/data/audit_generalpoints_tokens.py
pytest -q tests/test_stage1_plan.py

CUDA_VISIBLE_DEVICES=<gpu> \
python scripts/train_stage1.py \
  --learning-rate 1e-5 \
  --seed 42
```

Do not launch the remaining five LR×seed runs until this first run finishes at
exactly optimizer step 192 and all expected checkpoints exist. See
`docs/stage1_training.md`.


## Checkpoint evaluation

After the first Stage-1 run completes:

```bash
python scripts/data/smoke_generalpoints.py
python scripts/data/audit_generalpoints_tokens.py
pytest -q tests/test_generalpoints_verifier.py

CUDA_VISIBLE_DEVICES=<gpu> \
python scripts/evaluate_stage1.py \
  --steps 0,10 \
  --max-examples 32 \
  --eval-batch-size 16
```

See `docs/stage1_evaluation.md` before launching the full 21-checkpoint
trajectory.
