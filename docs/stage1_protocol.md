# Stage 1 — Literature-Grounded Reproduction Protocol

Status: **FROZEN FOR IMPLEMENTATION**
Date: 2026-09-23

The purpose of Stage 1 is **not** to prove the proposed selector. It is to determine whether a usable non-monotonic OOD trajectory exists in the available 2×RTX 4090 regime.

## Decision

### Primary model

`Qwen/Qwen2.5-3B-Instruct`

Why:

- 3.086B parameters: inside the planned 1.5B–3B regime.
- Same Qwen2.5 family as Jin et al.'s official Qwen2.5-7B-Instruct GeneralPoints experiment.
- Small enough for one LoRA run per RTX 4090 24GB.
- Using the Instruct checkpoint minimizes output-format failure on the formula task.

This is a **resource-adapted protocol reproduction**, not an exact replication of Jin et al.

## Primary controlled task

### Dataset

`Xiaofeng77/gp-l-only-10k`

Use the official fixed-prompt GeneralPoints release associated with *Debunk the Myth of SFT Generalization*.

### Training subset

- source split: `train` (10k rows in the official release)
- take **4,096 examples**
- deterministic shuffle seed: **20260923**
- use answer-only supervision for the initial reproduction
- keep the prompt's training rule unchanged: **J=Q=K=10**

Why 4,096:

- inside the project's 2K–5K pilot budget;
- enough for repeated exposure while allowing dense checkpoints;
- power-of-two size makes step accounting simple.

### Fixed anchor set

From the same shuffled official train pool, after removing the 4,096 SFT examples:

- reserve the next **512 prompts**
- discard / hide target responses from all selector code
- use these prompts only for base-relative functional KL
- never use OOD prompts or OOD labels in the anchor set

### ID validation

Use the official GeneralPoints **ID test split** from the dataset/repository unchanged.

Scoring: programmatic task success (all supplied cards used exactly once; formula evaluates to 24 under J/Q/K=10).

Also compute teacher-forced ID validation loss where targets are available.

### OOD oracle evaluation

Use the official **face-cards-as-regular** rule-shift split unchanged:

- J=11
- Q=12
- K=13

Scoring: the same programmatic task-success verifier, but under the OOD rule.

The OOD split is **analysis/oracle only**. Its targets/scores must not enter selector fitting, thresholds, layer choice, smoothing, or hyperparameter choice.

## LoRA configuration

Derived from the official GeneralPoints PEFT script of Lin et al., then downscaled to Qwen2.5-3B:

- method: LoRA
- rank: **32**
- alpha: **16**
- target modules: **all linear layers**
- dropout: **0.0** (project choice; record as such)
- dtype: **bf16**
- quantization: **none in the main pilot**
- gradient checkpointing: enabled
- max sequence length: **2048**
- effective batch size: **64**

The main reason for matching the public rank/alpha/all-linear recipe is to avoid making LoRA capacity an arbitrary hidden variable.

## Learning-rate sweep

[
oxed{{5\times10^{-6}, 1\times10^{-5}, 5\times10^{-5}}}
]

Rationale:

- `1e-5` is the task-specific LoRA LR used in the released GeneralPoints PEFT script.
- `5e-6` checks a conservative regime.
- `5e-5` checks a stronger-update regime where forgetting may be easier to reveal.
- Jin's `1e-6` is **not** used as the LoRA center because it comes from full-parameter SFT.

Seeds:

- **42**
- **43**

Total primary runs: **3 LR × 2 seeds = 6**

## Training length and checkpoint density

- epochs: **3**
- scheduler: cosine
- warmup ratio: **0.03**
- weight decay: **0.0**
- effective batch size: **64**
- expected optimizer steps per epoch with 4,096 samples: **64**
- expected total optimizer steps: **192**
- save checkpoint every **10 optimizer steps**
- evaluate ID/OOD after every saved checkpoint

A 50–100 step interval would be too sparse for this short controlled run. Ten optimizer steps yields roughly 20 observations over the training trajectory.

Always evaluate the **base model at step 0**.

## What counts as OOD forgetting

Do not label any noisy local maximum as forgetting.

For a run to count as a clear peak→decline case:

1. the OOD maximum occurs at a non-final checkpoint;
2. at least **two later checkpoints** remain below that peak;
3. peak-to-late decline is at least

[
max(5 	ext{percentage points}, 2	imes SE_{	ext{binomial}})
]

using the actual OOD evaluation-set size;
4. the ID trajectory does not show an equally large synchronized collapse.

Store the raw per-example success vector so confidence intervals / paired bootstrap checks can be added without rerunning inference.

## Stage-1 gate

### GO

At least **2 of the 6** LR×seed primary runs satisfy the clear peak→decline rule.

Then proceed to:

1. prompt-diversity boundary control;
2. functional/spectral drift measurement;
3. OOD-free selector comparison.

### CONDITIONAL GO

Any of these holds:

- only one primary run shows clear forgetting;
- forgetting appears only at the highest LR;
- the curve is strongly seed-sensitive;
- fixed-prompt forgetting is reproduced but disappears under prompt diversity;
- OOD dynamics are non-monotonic but look like dip→recovery rather than peak→decline.

The thesis question should then pivot toward **boundary conditions of SFT OOD dynamics** rather than claiming a universal forgetting curve.

### KILL / PIVOT

No primary LoRA run shows a meaningful non-monotonic OOD deterioration **and** the capacity sentinel below also fails to reproduce it.

Do not develop a selector for a phenomenon that was not reproduced.

## Mandatory boundary control after a primary GO

Lin et al. show that fixed-prompt GeneralPoints failures can be caused by instruction shortcutting.

Therefore after GO, run a **prompt-diverse sentinel**:

- dataset: official prompt-diverse answer-only GeneralPoints release
- same Qwen2.5-3B-Instruct model
- same LoRA rank/alpha/targets
- LR: **1e-5**
- seeds: **42, 43**
- epochs: **3**

Interpretation:

- forgetting persists: stronger evidence that the trajectory is not only a fixed-prompt artifact;
- forgetting disappears: valuable boundary-condition result; do not generalize the fixed-prompt finding beyond that regime.

## Capacity sentinel before declaring KILL

Biderman et al. show that LoRA often forgets less than full fine-tuning. A LoRA-only null can therefore be caused by the adaptation method.

If all six LoRA runs are null, run one closest-mechanism sentinel before KILL:

- same `Qwen/Qwen2.5-3B-Instruct`
- same fixed-prompt GeneralPoints data
- **full-parameter SFT**
- LR: **1e-6** (Jin official Qwen GeneralPoints full-FT LR)
- 1 epoch
- effective batch 64
- bf16 + gradient checkpointing
- DeepSpeed ZeRO-3 / CPU offload only if required by 2×24GB VRAM

If full FT shows forgetting but LoRA does not, the result supports an **adaptation-capacity boundary condition**, not a global null.

## Drift collection in Stage 1

The following may be computed once checkpoints exist, but they must not affect the Stage-1 phenomenon gate:

### Functional KL

On the fixed 512-prompt anchor set:

[
D_{func}(t)=
\mathbb{E}_{x\in A}
\mathrm{KL}
\left[
p_{\theta_t}(\cdot|x)
\Vert
p_{\theta_0}(\cdot|x)
\right].
]

Freeze in advance:

- prompt set
- token positions included in KL
- vocabulary treatment
- aggregation rule

### Parameter drift

- LoRA update Frobenius / L2 norm
- merged-weight L2 distance from base

### Spectral drift

For selected attention/MLP weight matrices after merging the LoRA update:

- top-k singular-vector principal angles against base
- rotation of left/right dominant subspaces
- singular-value change as a comparison signal
- optional OPLoRA-style dominant-subspace interference `rho_k`

Layer and k choices must be frozen without OOD labels.

### CRC baseline

Because UAI 2026 already introduced an OOD-free representation-collapse selector, CRC is a **mandatory Stage-2 baseline**, subject to faithful reimplementation from the paper.

## Public-benchmark confirmation after controlled reproduction

The controlled task is deliberately diagnostic. It is not sufficient for a broad reasoning claim.

After Stage-1 GO/CONDITIONAL GO, the next confirmation experiment should use:

- model family: Qwen2.5-3B
- SFT source: a fixed 4K subset of verified traces from `open-r1/OpenR1-Math-220k`
- ID benchmark: **MATH-500**
- OOD reasoning benchmarks: **GPQA-Diamond** and **MMLU-Pro**
- optional general-capability OOD: **IFEval**

These benchmark roles follow the public reasoning protocols used by recent SFT-generalization work rather than an arbitrary GSM8K→MATH split.

The exact public-benchmark training schedule is intentionally **not** allowed to influence the controlled Stage-1 gate.
