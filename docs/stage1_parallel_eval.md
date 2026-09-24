# Merge the 2-GPU Stage-1 Evaluation

After both GPU evaluation processes finish, merge the two disjoint checkpoint
shards.

Expected split:

- GPU 0: 0,20,40,60,80,100,120,140,160,180,192
- GPU 1: 10,30,50,70,90,110,130,150,170,190

Expected input roots:

- `results/stage1_gpu0/lr-1e-05_seed-42/`
- `results/stage1_gpu1/lr-1e-05_seed-42/`

## Merge and validate

```bash
python scripts/merge_stage1_eval.py \
  --input-roots results/stage1_gpu0 results/stage1_gpu1 \
  --run-name lr-1e-05_seed-42 \
  --output-dir results/stage1/lr-1e-05_seed-42 \
  --overwrite
```

The merger rejects:

- smoke rows instead of FULL evaluation;
- missing checkpoint steps;
- unexpected checkpoint steps;
- conflicting duplicate checkpoints;
- changing ID/OOD population sizes across checkpoints.

It also writes `phenomenon_summary.json`, applying the predeclared
single-run peak-to-decline rule.

## Plot the first curve

```bash
python scripts/plot_stage1_curve.py
```

Output:

```text
results/stage1/lr-1e-05_seed-42/id_ood_curve.png
```

This first curve classifies only the `LR=1e-5, seed=42` condition. The Stage-1
GO criterion still requires at least two of the six LR×seed conditions.
