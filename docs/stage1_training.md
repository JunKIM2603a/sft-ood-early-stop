# Stage-1 Training: Preflight and Single Pilot

This document covers the next step after the GPU LoRA smoke tests passed.

Observed engineering result on RTX 4090:

- Qwen2.5-3B-Instruct + LoRA r=32 fits comfortably.
- micro-batch 8 completed forward/backward/update/save/reload.
- observed peak allocated memory was about 10.6 GiB on the smoke batch.
- therefore the frozen default is:
  - micro-batch = 8
  - gradient accumulation = 8
  - effective batch = 64

The scientific run is still gated by an audit of **all 4,096 frozen SFT
examples**.

## 1. Audit all token lengths

Run:

```bash
conda activate sft-ood-es
git pull
python scripts/data/audit_generalpoints_tokens.py
```

This scans exactly the 4,096 SFT indices in the manifest and writes:

```text
artifacts/audits/generalpoints_stage1_tokens.json
```

PASS means every example fits within 2,048 tokens **without truncation**.

Why this is required:

The smoke examples were ~260 tokens, but the decisive run must not assume the
whole dataset has the same length. Truncating only rare long examples would
silently change the frozen training population.

## 2. Check the frozen schedule invariants

```bash
pytest -q tests/test_stage1_plan.py
```

Expected:

```text
2 passed
```

The test verifies:

```text
4096 examples
÷ (micro-batch 8 × grad-accum 8)
= 64 optimizer steps / epoch

64 × 3 epochs
= 192 optimizer steps
```

Expected saved states:

```text
0, 10, 20, ..., 190, 192
```

There are 21 states including the base model.

## 3. Launch only the first pilot condition

Do **not** launch all six runs yet.

Pick one free GPU:

```bash
nvidia-smi
```

Then run LR=1e-5, seed=42:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_stage1.py \
  --learning-rate 1e-5 \
  --seed 42
```

The physical GPU exposed by `CUDA_VISIBLE_DEVICES` appears as `cuda:0`
inside the process.

## 4. What the trainer guarantees

Before training it checks:

- LR and seed are members of the frozen grid.
- micro-batch × grad accumulation = effective batch.
- derived step count equals the predeclared 64 steps/epoch and 192 total.
- token audit is a PASS for the same frozen SFT subset.
- no selected example is silently truncated.

Training uses:

- Qwen/Qwen2.5-3B-Instruct
- BF16
- PyTorch SDPA
- LoRA r=32, alpha=16, dropout=0, all-linear
- answer-only autoregressive CE
- AdamW
- weight decay 0
- cosine schedule
- warmup ratio 0.03 → 6 warmup optimizer steps
- deterministic epoch shuffle using seed + epoch index
- 3 epochs

## 5. Output

The first run goes to:

```text
checkpoints/stage1/lr-1e-05_seed-42/
├── run_config.json
├── train_metrics.jsonl
├── checkpoints.jsonl
├── run_summary.json
├── tokenizer/
├── checkpoint-000000/
│   └── checkpoint_meta.json
├── checkpoint-000010/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoint_meta.json
├── ...
├── checkpoint-000190/
└── checkpoint-000192/
```

Checkpoint 0 is a metadata pointer to the untouched base model; the repository
does not duplicate the 3B base weights.

The LoRA checkpoints are intended for evaluation, not exact optimizer-state
resume. Optimizer moments are deliberately not copied into every checkpoint,
which would waste many gigabytes and is unnecessary for the planned
checkpoint-trajectory evaluation.

## 6. What to inspect after the first run

Do not launch the remaining five conditions until checking:

1. final optimizer step is exactly 192;
2. `train_metrics.jsonl` contains 192 rows;
3. loss remains finite;
4. LR warms up then follows a cosine decay;
5. peak GPU memory remains safe;
6. checkpoints 0/10/.../190/192 all exist.

Only then implement/run checkpoint evaluation and produce the first ID/OOD
curve. Drift metrics remain deferred until the phenomenon gate.
