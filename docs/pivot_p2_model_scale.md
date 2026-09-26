# Pivot P2 — Model-Scale Boundary at Fixed Full-Train Exposure

Status: **FROZEN BEFORE EXECUTION**  
Date: 2026-09-27

## Evidence entering P2

P1 changed only SFT exposure from 4,096 examples to the complete frozen 10K
train split at Qwen2.5-3B scale.

Observed P1 seed 42:

- ID accuracy: 0.0 -> 10.0%;
- OOD peak: 11.6% at step 80;
- final OOD: 9.8%;
- decline: 1.8 percentage points;
- required decline: 5 percentage points;
- clear forgetting: false.

Thus larger exposure improved acquisition and OOD performance, but did not
recover the target peak-to-decline trajectory.

## P2 scientific change

P2 changes **model scale only**:

```text
P1: Qwen2.5-3B-Instruct + frozen full train
P2: Qwen2.5-7B-Instruct + the same frozen full train
```

Held fixed:

- full-parameter SFT;
- LR 1e-6;
- seed 42;
- one epoch;
- effective batch 64;
- BF16;
- cosine scheduler;
- warmup ratio 0.03;
- same ID/OOD task evaluation;
- same exact-success verifier.

The public Jin et al. Qwen script also uses Qwen2.5-7B-Instruct, full FT,
LR 1e-6, one epoch, and global batch 64.

## Protocol-alignment caveat

The Jin et al. script points to `SFTvsRL_Data/SFT_Data/gp-l/data.json`.
Our frozen P1/P2 train source is
`Xiaofeng77/answer-only-gp-l-only-10k`, released with a later study.

Before P2, run a public-data alignment audit:

```bash
python scripts/data/audit_original_gp_protocol.py
```

If the original public GP-L file is unavailable or overlap is incomplete, P2
is interpreted as a **model-scale boundary test within the current 10K
protocol**, not an exact data replication of Jin et al.

## Hardware strategy

The public script requests four GPUs and large host RAM. Our environment uses
2×RTX 4090 and 128GB RAM, so P2 uses DeepSpeed ZeRO-3 / ZeRO-Infinity with
both optimizer and parameters offloaded to NVMe.

DeepSpeed documents NVMe offload for ZeRO-3 parameter and optimizer state.

Preflight:

```bash
python scripts/data/preflight_modelscale_p2.py
pytest -q tests/test_modelscale_p2_plan.py
```

The preflight verifies:

- exact reuse of the frozen P1 dataset;
- all examples fit the 7B tokenizer length limit;
- >=400 GiB free on the offload/checkpoint filesystem;
- adequate host RAM;
- DeepSpeed async-I/O compatibility;
- writable NVMe offload directory.

## Execution

Engineering smoke:

```bash
bash scripts/run_modelscale_p2_2gpu.sh smoke
```

Scientific run after PASS:

```bash
bash scripts/run_modelscale_p2_2gpu.sh scientific
```

Evaluation:

```bash
bash scripts/evaluate_modelscale_p2_2gpu.sh
```

## Decision

If clear forgetting appears and acquisition passes:

```text
MODEL_SCALE_BOUNDARY_CANDIDATE__REPLICATE_SEED43
```

Replicate the 7B full-train condition at seed 43 before reopening functional
or spectral drift development.

If 7B acquires the task but still shows no clear forgetting:

```text
NEAR_OFFICIAL_RECIPE_NEGATIVE__PROTOCOL_ALIGNMENT_CONTROL
```

Then the next step is not more proxy engineering. It is exact original-data /
prompt-protocol alignment, because the current 10K data source is not proven
identical to the Jin et al. training file.
