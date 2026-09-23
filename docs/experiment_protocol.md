# Experiment Protocol

## Core rule

The project is gated by **phenomenon reproduction**. Proxy development starts only after a predeclared Stage-1 GO or CONDITIONAL GO.

The exact frozen Stage-1 settings live in `docs/stage1_protocol.md` and `configs/pilot.yaml`. The public Hub currently names the canonical ID evaluation split `test` rather than `test_id`; code must verify its J=Q=K=10 semantics rather than trusting the split name.

## Stage 1: controlled reproduction

Primary setup:

- `Qwen/Qwen2.5-3B-Instruct`
- LoRA r=32, alpha=16, all-linear
- SFT train: `Xiaofeng77/answer-only-gp-l-only-10k`
- ID/OOD evaluation: `Xiaofeng77/gp-l-only-10k`
- fixed-prompt ID rule: J=Q=K=10
- OOD oracle rule: J=11,Q=12,K=13
- 4,096 SFT examples
- 512 fixed unlabeled anchor prompts
- LR: 5e-6 / 1e-5 / 5e-5
- seeds: 42 / 43
- 3 epochs
- effective batch 64
- save every 10 optimizer steps

This is a resource-adapted reproduction of the controlled GeneralPoints phenomenon used in prior SFT-vs-RL / OOD-forgetting work. It is not an exact replication of the 7B/11B full-FT studies.

## Why a prompt-diversity control is mandatory

Later GeneralPoints work shows that fixed prompts can create shortcut behavior: the model may ignore changed instructions and keep applying the training semantics.

Therefore, if fixed-prompt forgetting is reproduced, run the predeclared prompt-diverse control before generalizing the result.

A fixed-prompt-only effect is a valid **boundary-condition finding**, not a failed experiment.

## LoRA null-result safeguard

Prior TMLR evidence shows LoRA tends to preserve outside-domain behavior better than full FT.

Therefore a complete LoRA null is followed by one predeclared full-FT capacity sentinel on the same 3B model/data before declaring KILL/PIVOT.

## Checkpoint measurements

At checkpoint t, record:

1. optimizer step and consumed examples
2. train loss
3. ID validation loss
4. ID task success
5. OOD oracle task success
6. parameter / merged-weight L2 drift
7. LoRA update norm
8. functional KL from base
9. singular-value changes
10. singular-vector / principal-angle drift

The OOD metric is logged for research analysis only; it must not be exposed to selector logic.

## Fixed anchor set

The anchor set contains 512 prompts disjoint from the 4,096 SFT examples.

[
D_{func}(t)
=
\mathbb{E}_{x\in A}
\mathrm{KL}
\left[
p_{\theta_t}(\cdot|x)
\Vert
p_{\theta_0}(\cdot|x)
\right].
]

Before the first run, freeze:

- which token positions contribute to KL;
- whether logits are compared over all vocabulary tokens or a documented subset;
- per-token and per-example aggregation;
- any clipping / numerical stabilization.

No OOD prompt or OOD label may be used to choose these settings.

## Preventing leakage

OOD labels may be used only for:

- post-hoc oracle checkpoint identification;
- final research evaluation of selector regret;
- statistical confirmation that Stage 1 did or did not reproduce the phenomenon.

OOD labels must not tune:

- proxy thresholds
- smoothing parameters
- change-point hyperparameters
- monitored layers
- SVD rank k
- proxy weights/combinations
- anchor-set composition

Hyperparameters for a proposed selector must be fixed from theory, ID-only information, or a separate OOD-label-free development protocol.

## Stage-1 phenomenon criterion

A run counts as clear peak→decline only if:

- the OOD peak is not the final checkpoint;
- at least two later checkpoints remain below the peak;
- decline is at least max(5 percentage points, 2×binomial SE);
- there is no equally large synchronized ID collapse.

### GO

At least 2/6 primary LR×seed runs meet the criterion.

### CONDITIONAL GO

The effect is isolated to one seed/LR, changes into dip→recovery, or vanishes under prompt diversity.

### KILL / PIVOT

No meaningful non-monotonic OOD deterioration after the six LoRA runs plus the capacity sentinel.

## Stage 2: validation-free checkpoint selection

Only after GO / CONDITIONAL GO.

### Selectors

Baselines:

- train loss
- ID validation loss
- ID validation task score
- fixed checkpoint
- parameter L2 drift
- **CRC (Collapse Risk Criterion)**

Proposed / mechanism-derived:

- functional KL
- singular-vector / principal-angle rotation
- combined functional + spectral proxy
- optional OPLoRA-style dominant-subspace interference diagnostic

### Primary endpoint

For each training run:

[
\mathrm{Regret}
=
\mathrm{OOD}_{oracle}
-
\mathrm{OOD}_{selected}.
]

Correlation is secondary. A useful diagnostic must translate into low checkpoint-selection regret.

## Public-benchmark confirmation

Controlled GeneralPoints is diagnostic and cheap, but a broad thesis claim requires a public reasoning confirmation.

After the controlled gate:

- 4K verified math traces from OpenR1-Math-220k
- ID: MATH-500
- OOD reasoning: GPQA-Diamond, MMLU-Pro
- optional capability shift: IFEval

Do not substitute an arbitrary GSM8K→MATH split and call it canonical OOD.
