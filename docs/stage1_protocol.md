# Stage 1 — Literature-Grounded Reproduction Protocol

Status: **FROZEN FOR IMPLEMENTATION**
Date: 2026-09-23

## Primary model

`Qwen/Qwen2.5-3B-Instruct`

## GeneralPoints data roles

The official release separates SFT examples and RL/evaluation examples across
different Hugging Face repositories.

### SFT training

`Xiaofeng77/answer-only-gp-l-only-10k`

- official non-diverse, answer-only SFT data
- use 4,096 deterministic train examples
- seed 20260923
- reserve the next 512 train examples as the unlabeled functional-KL anchor
- reserve the following 512 examples as labeled ID-loss validation
- all three subsets are pairwise disjoint
- training rule: J=Q=K=10

### Evaluation

`Xiaofeng77/gp-l-only-10k`

The currently published evaluation repository exposes six splits:
`train`, `test_5cards`, `test_face_cards_as_regular`, `test_fake`,
`test`, and `test_large`.

For ID task evaluation:

1. use `test_id` if a future official release provides it;
2. otherwise use `test` **only after the smoke test verifies J=Q=K=10**.

ID validation loss is measured separately on the frozen 512-example labeled
holdout from the answer-only SFT source. This prevents the OOD oracle split from
being involved in the loss-based selector baseline.

For the primary OOD oracle:

- `test_face_cards_as_regular`
- verify J=11,Q=12,K=13 from actual metadata/prompt

The split name is never trusted without semantic verification.

## LoRA configuration

- r=32
- alpha=16
- all-linear
- dropout=0
- bf16
- no quantization
- gradient checkpointing
- max length 2048
- micro-batch 8
- gradient accumulation 8
- effective batch 64
- all 4,096 frozen SFT examples must pass the 2,048-token audit without truncation

## LR × seed grid

- LR: 5e-6, 1e-5, 5e-5
- seeds: 42, 43
- 6 primary runs
- 3 epochs
- cosine schedule
- warmup 0.03
- save/evaluate every 10 optimizer steps
- expected 64 optimizer steps/epoch, 192 total
- always evaluate step-0 base model

The micro-batch 8 choice was accepted only after an RTX 4090 forward/backward/update/save/reload smoke test peaked at about 10.6 GiB allocated VRAM.

## Clear peak→decline criterion

A run counts only if:

1. OOD maximum is non-final;
2. at least two later checkpoints remain lower;
3. decline ≥ max(5 percentage points, 2×binomial SE);
4. ID does not show an equally large synchronized collapse.

## Gate

- **GO:** at least 2/6 primary runs pass.
- **CONDITIONAL GO:** one run only, high-LR only, strong seed sensitivity,
  prompt-diversity sensitivity, or dip→recovery.
- **KILL/PIVOT:** null after all LoRA runs plus the capacity sentinel.

## Mandatory prompt-diversity control after GO

Use `Xiaofeng77/diverse-answer-only-gp-l-only-10k` with LR 1e-5,
seeds 42/43 and the same model/LoRA recipe.

## Capacity sentinel before KILL

If all LoRA runs are null:

- same Qwen2.5-3B-Instruct
- same fixed-prompt task
- full-parameter SFT
- LR 1e-6
- 1 epoch

## Stage 2

Only after GO / CONDITIONAL GO measure and compare:

- train loss
- ID validation loss / score
- fixed step
- parameter L2
- CRC
- functional KL
- singular-vector / principal-angle drift
- combined proxy

Primary endpoint:

```text
Regret = OOD_oracle - OOD_selected
```


## Post-pilot interpretability amendment — 2026-09-24

This section was added **after** the first completed condition
(`LR=1e-5, seed=42`). It is not part of the original predeclared
peak-to-decline criterion and must not be presented as if it were.

The first run showed:

- large ID validation-loss reduction;
- near-perfect output parsing / high valid-formula rate;
- only +3.2 percentage points maximum ID task-accuracy gain;
- no clear OOD peak-to-decline under the original criterion.

Therefore all remaining runs will report two distinct judgments:

1. **Original predeclared forgetting criterion** — unchanged.
2. **Acquisition-qualified interpretation** — requires maximum ID task
   accuracy to improve by at least 5 percentage points over step 0 before a
   forgetting interpretation is treated as informative.

The 5-point threshold reuses the existing minimum practical effect-size floor.
It is a post-pilot sanity guard, not a replacement primary endpoint.

A run failing this guard is labeled:

`UNDER_LEARNED_INCONCLUSIVE_FOR_FORGETTING`

rather than being counted as evidence that OOD forgetting is absent.
