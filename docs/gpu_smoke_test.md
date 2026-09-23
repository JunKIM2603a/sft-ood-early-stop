# GPU Training Smoke Test

Purpose: verify that the **frozen Stage-1 model and LoRA configuration** can
actually train on one RTX 4090 before implementing or launching the 6-run
experiment grid.

This is an engineering check, not a research result.

## What it tests

```text
GeneralPoints answer-only sample
          │
          ▼
Qwen2.5-3B-Instruct (BF16, SDPA)
          │
          ▼
LoRA r=32 / alpha=16 / all-linear
          │
          ▼
forward → finite CE loss
          │
          ▼
backward → finite nonzero LoRA gradient
          │
          ▼
AdamW optimizer step
          │
          ▼
LoRA-B weight actually changes
          │
          ▼
save adapter
          │
          ▼
reload adapter → finite forward loss
```

PEFT officially supports `target_modules="all-linear"`; this applies LoRA to
the transformer's linear layers while excluding the output layer for a
`PreTrainedModel`. Transformers' `sdpa` attention backend uses PyTorch
scaled-dot-product attention and requires no separate flash-attn installation.

## Prerequisites

From the repository root:

```bash
conda activate sft-ood-es
git pull
python scripts/verify_environment.py
python scripts/data/smoke_generalpoints.py
```

The data smoke test must have created:

```text
data/local/generalpoints_stage1/manifest.json
```

The training smoke test reads its sample indices from this manifest, so it
cannot silently use a different training population.

## Choose a free GPU

Check current GPU use:

```bash
nvidia-smi
```

Then expose **one** GPU to the process. For example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/smoke/train_lora_smoke.py
```

Inside the process that physical GPU becomes `cuda:0`. This is intentional.

Defaults:

- model: `Qwen/Qwen2.5-3B-Instruct`
- bf16
- PyTorch SDPA
- LoRA r=32, alpha=16, dropout=0, all-linear
- GeneralPoints answer-only training data
- 4 samples
- 2 optimizer steps
- micro-batch size 1
- max length 2048
- learning rate 1e-5
- gradient checkpointing enabled

The small micro-batch/step count is only for the engineering smoke test. It
does **not** change the frozen scientific Stage-1 effective batch size of 64.

## Expected success output

The exact values vary, but the end should look like:

```text
Trainable parameters: ... (...%)
step=1/2 loss=... lora_grad_l2=...
step=2/2 loss=... lora_grad_l2=...
Tracked LoRA-B max |delta|: ...e-...
Saving adapter: artifacts/smoke/qwen2.5-3b-lora-r32
Reload forward loss: ...

PASS: Qwen2.5-3B + LoRA GPU training smoke test succeeded.
```

The script also prints GPU allocated/reserved/peak/free memory throughout the
run. Record the peak memory before deciding whether the full Stage-1 batch
implementation needs gradient accumulation.

## Artifacts

The output directory is gitignored:

```text
artifacts/smoke/qwen2.5-3b-lora-r32/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer files...
└── smoke_result.json
```

`smoke_result.json` records:

- GPU/PyTorch/CUDA identity
- data indices used
- LoRA configuration
- trainable parameter count
- per-step losses
- per-step LoRA gradient norms
- observed LoRA parameter change
- GPU memory
- reload loss

## If GPU memory is insufficient

Do **not** immediately install bitsandbytes or change the scientific protocol.

First retry only the engineering smoke with:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/smoke/train_lora_smoke.py \
  --samples 2 \
  --steps 1
```

Micro-batch already defaults to 1, so reducing `samples` changes only how
many examples are pre-encoded, not the scientific experiment.

If the **model itself** cannot fit with LoRA r=32/BF16 on a 24GB 4090, stop and
revisit the Stage-1 implementation before launching any runs.

## Rerunning

The script refuses to silently overwrite a previous smoke adapter. To rerun:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/smoke/train_lora_smoke.py \
  --overwrite-output
```

To skip only the final adapter reload test:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/smoke/train_lora_smoke.py \
  --skip-reload-check
```

## PASS criterion

A PASS requires all of the following:

1. CUDA + BF16 available.
2. Model loads in BF16 on one GPU.
3. LoRA is the only trainable adaptation (base weights frozen).
4. Loss is finite.
5. LoRA gradient norm is finite and > 0.
6. A LoRA-B parameter changes after optimizer steps.
7. Adapter files are saved.
8. Unless explicitly skipped, the saved adapter reloads and produces a finite
   forward loss.

Only after this PASS should the real Stage-1 training/evaluation pipeline be
implemented.
