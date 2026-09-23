# Stage-1 Environment Setup

Target: Linux x86_64, Python 3.11, NVIDIA RTX 4090 ×2.

## Create / update the environment

```bash
conda env create -f environment.yml
conda activate sft-ood-es
git lfs install
```

If it already exists:

```bash
conda activate sft-ood-es
python -m pip install -r requirements.lock.txt
```

## Verify CUDA and packages

```bash
python scripts/verify_environment.py
```

## GeneralPoints data layout

Stage 1 uses two official repositories for different roles:

| Role | Repository |
|---|---|
| Answer-only SFT training | `Xiaofeng77/answer-only-gp-l-only-10k` |
| ID/OOD evaluation | `Xiaofeng77/gp-l-only-10k` |
| Prompt-diverse control | `Xiaofeng77/diverse-answer-only-gp-l-only-10k` |

The RL/evaluation repository currently exposes:

```text
train
test_5cards
test_face_cards_as_regular
test_fake
test
test_large
```

It does **not** currently expose a `test_id` split. The smoke test therefore
prefers `test_id` if a future release adds it, otherwise uses `test` only
after verifying from the actual metadata/prompt that it is J=Q=K=10.

Run:

```bash
python scripts/data/smoke_generalpoints.py
```

The test verifies that:

- SFT training data has an `answer` column;
- 4,096 SFT + 512 anchor examples fit in the train split;
- ID evaluation really uses J=Q=K=10;
- OOD evaluation really uses J=11,Q=12,K=13;
- SFT/anchor indices are deterministic and disjoint.

It writes only a local ignored manifest:

```text
data/local/generalpoints_stage1/manifest.json
```

## Exact environment snapshot

After environment + data smoke tests pass:

```bash
bash scripts/refresh_lock.sh
nvidia-smi | tee environment.nvidia.txt
```

Do not install `flash-attn`, DeepSpeed, or bitsandbytes yet; the primary
LoRA pilot does not require them.
