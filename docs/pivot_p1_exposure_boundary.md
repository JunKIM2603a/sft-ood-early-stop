# Pivot P1 — Exposure Boundary at Fixed 3B Scale

Status: **FROZEN BEFORE EXECUTION**  
Date: 2026-09-26

## Why the original selector study is paused

The original reproduction gate produced:

- six Qwen2.5-3B LoRA conditions: 0 clear OOD-forgetting runs;
- one Qwen2.5-3B full-FT capacity sentinel: no clear OOD forgetting;
- the full-FT sentinel acquired the task only weakly (ID max 5%).

Therefore functional/spectral checkpoint-selector development remains paused.

## What P1 changes

P1 changes exactly one scientific factor relative to the 3B full-FT sentinel:
**SFT exposure**.

Kept fixed:

- Qwen/Qwen2.5-3B-Instruct;
- full-parameter SFT;
- LR 1e-6;
- seed 42;
- 1 epoch;
- global effective batch 64;
- BF16;
- cosine schedule;
- same ID and OOD evaluation splits;
- same exact-success verifier.

Changed:

- 4,096 SFT examples -> complete current
  `Xiaofeng77/answer-only-gp-l-only-10k::train` split.

The exact full-train row count is frozen at preflight rather than inferred from
the dataset name.

## Important contamination consequence

The complete train split includes the examples previously reserved as the
512-example ID-loss validation set and the functional-KL anchor.

Therefore P1 explicitly disables:

- ID validation loss as a selector/baseline;
- functional KL / spectral selector development.

P1 uses only separate official ID task accuracy and OOD task accuracy for the
reproduction gate.

## Preflight

```bash
python scripts/data/freeze_exposure_p1.py
pytest -q tests/test_exposure_p1_plan.py
```

The preflight:

- freezes exact row count and dataset fingerprint;
- hashes the full training content;
- verifies four display cards per sample;
- rejects training values above 10;
- audits every training example against the 2,048-token limit.

## Engineering smoke

```bash
bash scripts/run_exposure_p1_2gpu.sh smoke
```

## Scientific run

Only after smoke passes:

```bash
bash scripts/run_exposure_p1_2gpu.sh scientific
```

DeepSpeed ZeRO-3 CPU offload is retained because it already completed the 3B
full-FT capacity sentinel on the available 2×RTX 4090 hardware.

## Evaluation

```bash
bash scripts/evaluate_exposure_p1_2gpu.sh
```

The saved checkpoints are automatically split across GPU 0 and GPU 1.

Output:

```text
results/exposure_p1/qwen2.5-3b_fulltrain_lr-1e-06_seed-42/
├── checkpoint_metrics.csv
├── checkpoint_metrics.jsonl
└── exposure_boundary_summary.json
```

## Decision

If clear forgetting appears **and** ID acquisition passes:

```text
EXPOSURE_BOUNDARY_CANDIDATE__REPLICATE_SEED43
```

Do not reopen drift-selector development from one seed alone. Replicate the
same full-train condition with seed 43 first.

If clear forgetting is still absent, or the 3B model remains under-learned:

```text
P2_MODEL_SCALE_SENTINEL_7B
```

The next experiment then changes model scale while keeping the full-train
exposure recipe as close as hardware permits.

## External protocol reference

The public Qwen GeneralPoints SFT script in
`jinhangzhan/RL_Heals_SFT/sft/sft_scripts/gp_l-qwen.sh` uses
Qwen2.5-7B-Instruct, full fine-tuning, LR 1e-6, one epoch, and global batch 64.
P1 intentionally keeps 3B first so data exposure and model scale are not
changed simultaneously.
