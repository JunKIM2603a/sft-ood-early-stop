# Complete the Frozen Six-Run LoRA Grid

Observed so far:

| LR | seed | acquisition | clear peak→decline |
|---:|---:|:---:|:---:|
| 1e-5 | 42 | weak / post-pilot guard fail | no |
| 5e-5 | 42 | pass | no |
| 5e-5 | 43 | pass | no |

The remaining frozen conditions are:

- 1e-5 / seed 43
- 5e-6 / seed 42
- 5e-6 / seed 43

Because the grid was predeclared, complete these conditions before concluding
that LoRA does not reproduce the target phenomenon.

## Wave 1 — two GPUs

Train two independent conditions concurrently:

```bash
bash scripts/run_stage1_two_conditions_2gpu.sh \
  1e-5 43 \
  5e-6 42
```

Then evaluate both concurrently:

```bash
bash scripts/evaluate_stage1_two_conditions_2gpu.sh \
  1e-5 43 \
  5e-6 42
```

## Wave 2 — final remaining condition

Train:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_stage1.py \
  --learning-rate 5e-6 \
  --seed 43
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_stage1.py \
  --run-dir checkpoints/stage1/lr-5e-06_seed-43 \
  --steps all \
  --eval-batch-size 32 \
  --loss-batch-size 16

python scripts/summarize_stage1_run.py \
  --metrics results/stage1/lr-5e-06_seed-43/checkpoint_metrics.jsonl
```

GPU 1 is intentionally left free during this final single condition; do not
start an unplanned extra scientific condition just to keep the device busy.

## Aggregate all six

```bash
python scripts/aggregate_stage1_grid.py
```

Outputs:

```text
results/stage1_grid/
├── stage1_grid_summary.csv
└── stage1_grid_summary.json
```

Decision rule:

- at least 2/6 original clear peak→decline -> GO
- exactly 1/6 -> CONDITIONAL GO
- 0/6 -> do **not** declare KILL yet; run the predeclared full-FT capacity
  sentinel first.

The post-pilot acquisition guard is reported separately and never replaces the
original criterion.
