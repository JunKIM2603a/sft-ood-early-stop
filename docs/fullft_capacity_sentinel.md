# Full-FT Capacity Sentinel

The frozen six-run LoRA grid completed with:

- 6/6 runs evaluated;
- 0/6 clear peak-to-decline runs;
- only 2/6 runs passing the post-pilot ID-acquisition sanity guard.

Per the predeclared Stage-1 protocol, this is **not yet a KILL/PIVOT result**.
The next required experiment is a full-parameter SFT capacity sentinel.

## Why two GPUs are used together

Full fine-tuning stores trainable model parameters, gradients, and Adam
optimizer state. The 3B model fits for inference on one 4090, but full-FT
optimizer state is much larger.

The sentinel therefore uses Transformers FSDP2 across both RTX 4090s. FSDP2
shards parameters, gradients, and optimizer state across devices. The official
Transformers 5.17 documentation recommends FSDP for this memory regime and
supports activation checkpointing and full-state checkpoints.

## Frozen scientific condition

```text
Model: Qwen/Qwen2.5-3B-Instruct
Method: full parameter SFT
Training examples: frozen 4,096 GeneralPoints examples
Epochs: 1
LR: 1e-6
Global effective batch: 64
Expected optimizer steps: 64
BF16
Cosine scheduler
Warmup steps: 2
FSDP2 across GPU 0 + GPU 1
```

Per-rank batch layout:

```text
per-device batch 2
× 2 GPUs
× gradient accumulation 16
= global effective batch 64
```

Checkpoint states:

```text
0, 10, 20, 30, 40, 50, 60, 64
```

Intermediate checkpoints save **model only**; optimizer-state resume is not
needed for the planned trajectory evaluation.

## Disk requirement

A 3B BF16 full checkpoint is several GiB. Seven non-base states can consume
tens of GiB.

The scientific script therefore requires at least 50 GiB free under the
checkpoint filesystem before it starts. Do not bypass this check unless disk
capacity has been manually verified.

## Step 1 — engineering smoke

Run first:

```bash
bash scripts/run_fullft_sentinel_2gpu.sh smoke
```

This uses only 128 frozen examples and 2 optimizer steps, with no scientific
checkpoint trajectory. It verifies that FSDP2 full parameter forward/backward
fits on the two 4090s.

Only continue if the script prints PASS.

## Step 2 — scientific sentinel

```bash
bash scripts/run_fullft_sentinel_2gpu.sh scientific
```

Output:

```text
checkpoints/fullft_sentinel/qwen2.5-3b_lr-1e-06_seed-42/
├── checkpoint-10/
├── checkpoint-20/
├── ...
├── checkpoint-60/
├── checkpoint-64/
├── checkpoints.jsonl
└── run_summary.json
```

## Step 3 — evaluate the trajectory

The Stage-1 evaluator will be extended to auto-detect full-model checkpoints,
so the same ID/OOD metrics and the same independent GeneralPoints verifier are
used for LoRA and full-FT trajectories.

Interpretation:

- full FT clear forgetting, LoRA none -> adaptation-capacity boundary finding;
- full FT no clear forgetting -> controlled small-model reproduction has
  failed under both adaptation regimes, triggering KILL/PIVOT analysis;
- any acquisition failure must still be reported separately from forgetting.

## Two-GPU evaluation

After the scientific sentinel completes, evaluate the 8 states in parallel:

```bash
bash scripts/evaluate_fullft_sentinel_2gpu.sh
```

Allocation:

```text
GPU 0 -> steps 0,20,40,60
GPU 1 -> steps 10,30,50,64
```

The launcher then merges both shards and writes:

```text
results/fullft_sentinel/qwen2.5-3b_lr-1e-06_seed-42/
├── checkpoint_metrics.csv
├── checkpoint_metrics.jsonl
└── capacity_sentinel_summary.json
```

Possible decisions:

- `ADAPTATION_CAPACITY_BOUNDARY__CONDITIONAL_GO`
- `KILL_OR_PIVOT__PHENOMENON_NOT_REPRODUCED`
- `FULLFT_SENTINEL_UNDER_LEARNED__INCONCLUSIVE`
