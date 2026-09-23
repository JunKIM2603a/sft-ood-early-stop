# Stage-1 Environment Setup

Target: Linux x86_64, Python 3.11, NVIDIA RTX 4090 ×2.

## 1. Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate sft-ood-es
git lfs install
```

If the environment already exists:

```bash
conda activate sft-ood-es
python -m pip install -r requirements.lock.txt
```

The bootstrap lock pins the research-critical stack, including PyTorch
2.7.1 with the CUDA 12.6 wheel, Transformers 5.17.0, Datasets 5.0.1,
PEFT 0.21.0, Accelerate 1.15.0, and huggingface-hub 1.32.0.

## 2. Verify CUDA and packages

```bash
python scripts/verify_environment.py
```

Expected minimum result:

```text
CUDA available: True
GPU count: 2
BF16 supported: True
PASS: core Stage-1 environment is usable.
```

A single visible GPU is not fatal; the six Stage-1 runs can be executed
serially. BF16 failure is treated as an environment failure because the frozen
pilot uses bf16.

## 3. Hugging Face authentication

Public Stage-1 model/data downloads do not require authentication in normal
circumstances, but logging in is useful for rate limits and future gated repos.

```bash
hf auth login
hf auth whoami
```

If `hf` is missing:

```bash
python -m pip install -U huggingface-hub
```

Always run this inside `(sft-ood-es)`, not the Conda `(base)` environment.

## 4. Cache location

For a large local disk, set a persistent cache path, for example:

```bash
mkdir -p /home/junkim2603a/hf_cache
export HF_HOME=/home/junkim2603a/hf_cache
```

Add the export to `~/.bashrc` only if that filesystem has enough space.

## 5. Download + verify GeneralPoints

Run:

```bash
python scripts/data/smoke_generalpoints.py
```

The script verifies:

- dataset repo: `Xiaofeng77/gp-l-only-10k`;
- required splits exist: `train`, `test_id`,
  `test_face_cards_as_regular`;
- key columns exist: `data_source`, `extra_info`, `question`;
- examples have four-card metadata and target 24;
- sampled train/ID metadata uses face-cards-as-10;
- sampled OOD metadata uses face cards as regular values;
- train split has enough rows for 4,096 SFT + 512 anchor prompts;
- deterministic SFT and anchor subsets are disjoint.

The full dataset remains in the Hugging Face cache. Only a local ignored
manifest is written to:

```text
data/local/generalpoints_stage1/manifest.json
```

The manifest records the deterministic indices and SHA-256 digests so the
exact Stage-1 subset can be reconstructed.

## 6. Record the exact machine environment

After both smoke tests pass:

```bash
bash scripts/refresh_lock.sh
```

This writes:

```text
requirements.freeze.txt
```

Commit that file before the first decisive GPU run. It is the exact transitive
package snapshot of the experiment machine. `requirements.lock.txt` remains
the curated bootstrap lock.

Also record the NVIDIA driver/runtime for the experiment report:

```bash
nvidia-smi | tee environment.nvidia.txt
```

## 7. Intentionally not installed yet

Do not add these until required:

- `flash-attn`: PyTorch SDPA is sufficient for the first pilot.
- `deepspeed`: needed only if the full-FT capacity sentinel cannot fit.
- `bitsandbytes`: the frozen primary pilot uses no quantization.

This keeps the primary LoRA environment easier to reproduce and debug.
