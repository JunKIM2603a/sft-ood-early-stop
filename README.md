# SFT OOD Early Stop

> **Research question:** Can we select a useful SFT checkpoint around the onset of OOD degradation **without using OOD validation labels**?

This project studies whether **base-relative functional drift** and **parameter spectral drift** can provide practical checkpoint-selection signals during LLM supervised fine-tuning (SFT).

## Current evidence status

The literature check is now complete enough to freeze Stage 1.

Important corrections from the initial idea:

- OOD peak→decline during SFT has been reported, but it is **not established as universal**.
- recent work reports other trajectories such as **dip→recovery**, making optimization depth a competing explanation;
- on GeneralPoints, later work shows fixed prompts can create instruction shortcuts, so a **prompt-diversity control is mandatory**;
- LoRA can preserve outside-domain behavior better than full fine-tuning, so a LoRA-only null is not sufficient for a strong KILL claim;
- UAI 2026 already proposes **CRC**, an ID-computable OOD-free checkpoint selector based on representation collapse.

Therefore the novelty is **not** “the first OOD-free checkpoint selector.” The defensible question is whether functional KL and parameter singular-subspace drift are useful prospective selectors specifically along SFT-induced OOD-degradation trajectories, and whether they outperform conventional selectors and CRC.

## Core hypotheses

- **H1 — Non-monotonic OOD dynamics:** under at least some SFT conditions, OOD performance reaches a useful early checkpoint and later degrades differently from ID performance.
- **H2 — Drift change point:** functional KL or singular-vector/principal-angle drift changes before or around the OOD degradation.
- **H3 — Practical selection:** an OOD-label-free drift selector achieves lower checkpoint regret than train loss, ID validation, fixed-step selectors, and the CRC baseline.

### Competing hypotheses

- **C1:** OOD forgetting is conditional on optimization, data/prompt design, model capability, or adaptation method.
- **C2:** spectral drift is correlated with OOD behavior but is not operationally useful for checkpoint selection.
- **C3:** simple functional KL is a stronger practical signal than parameter spectral quantities.
- **C4:** representation-collapse CRC is already as good as or better than the proposed drift signals.

## Frozen Stage 1

Detailed protocol: `docs/stage1_protocol.md`

### Primary model

`Qwen/Qwen2.5-3B-Instruct`

### Controlled task

GeneralPoints uses separate official repositories for training and evaluation:

- SFT train: `Xiaofeng77/answer-only-gp-l-only-10k`, deterministic **4,096-example** subset of its train split
- ID/OOD evaluation: `Xiaofeng77/gp-l-only-10k`; use `test` as ID only after verifying J=Q=K=10 semantics
- ID rule: **J=Q=K=10**
- OOD oracle rule: **J=11, Q=12, K=13**
- fixed unlabeled functional-KL anchor: **512 prompts** disjoint from SFT train

### LoRA

- r=32
- alpha=16
- all-linear target modules
- bf16
- no quantization in the main pilot
- effective batch size 64
- 3 epochs

### LR × seed grid

```text
LR:   5e-6, 1e-5, 5e-5
seed: 42, 43
------------------------
6 primary runs
```

`1e-5` is the task-specific LoRA anchor from the released GeneralPoints PEFT recipe. The other two LRs probe a conservative and stronger-update regime.

### Checkpoints

- base model = step 0
- save every **10 optimizer steps**
- expected ~192 optimizer steps total
- evaluate every saved checkpoint

The short run requires denser checkpoints than the earlier generic 50–100-step placeholder.

## Stage-1 gate

A run counts as clear peak→decline only if:

1. OOD peak is not the final checkpoint;
2. at least two later checkpoints remain below the peak;
3. decline is at least max(5 percentage points, 2×binomial SE);
4. there is no equally large synchronized ID collapse.

- **GO:** at least 2/6 primary runs pass.
- **CONDITIONAL GO:** effect is isolated to one LR/seed, turns into dip→recovery, or disappears with prompt diversity.
- **KILL / PIVOT:** no meaningful non-monotonic OOD degradation after the six LoRA runs **and** the predeclared full-FT capacity sentinel.

## Boundary controls

### Prompt-diversity control

After a primary GO:

- official diverse-answer-only GeneralPoints data
- LR 1e-5
- seeds 42/43
- same model / LoRA / epochs

This tests whether the observed rule-shift failure is primarily a fixed-prompt shortcut.

### Capacity sentinel

If all six LoRA runs are null:

- same Qwen2.5-3B-Instruct
- same fixed-prompt task
- full-parameter SFT
- LR 1e-6
- 1 epoch

This checks the known possibility that LoRA itself suppresses forgetting.

## Primary endpoint

```text
Regret = OOD_oracle - OOD_selected_by_proxy
```

Selectors to compare after the Stage-1 gate:

1. training loss
2. ID validation loss / ID metric
3. fixed checkpoint
4. parameter L2 drift
5. **CRC representation-collapse baseline**
6. functional KL drift
7. singular-vector / principal-angle drift
8. combined proxy
9. optional OPLoRA-style dominant-subspace interference diagnostic

Correlation is secondary; the main question is whether the signal actually selects a low-regret checkpoint.

## Public reasoning confirmation

Controlled GeneralPoints is a diagnostic gate, not sufficient evidence for broad reasoning claims.

After GO / CONDITIONAL GO:

- SFT: fixed 4K subset from `open-r1/OpenR1-Math-220k`
- ID: MATH-500
- OOD reasoning: GPQA-Diamond, MMLU-Pro
- optional broader shift: IFEval

GSM8K is not the primary OOD protocol because the original GSM8K paper does not define a canonical cross-benchmark OOD split.

## Repository guide

```text
.
├── README.md
├── configs/
│   └── pilot.yaml                    # frozen Stage-1 machine-readable config
├── docs/
│   ├── research_plan.md
│   ├── literature_matrix.md          # verified evidence / competing results
│   ├── stage1_protocol.md            # frozen Stage-1 decision
│   ├── novelty_positioning.md        # revised novelty after CRC
│   └── experiment_protocol.md
├── scripts/
├── src/
├── tests/
├── data/
└── results/
```

## Research principle

Do **not** force a universal OOD-forgetting story. A negative or conditional result is useful if it identifies the optimization, prompt/data, model-capacity, or adaptation regime in which the phenomenon appears.
