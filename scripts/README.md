# Scripts

Planned command-line entry points:

- `train_sft.py` — run one LR/seed SFT condition and save frequent checkpoints
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
