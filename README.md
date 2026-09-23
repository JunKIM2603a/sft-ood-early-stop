# SFT OOD Early Stop

> **Research question:** Can we select a checkpoint just before OOD forgetting begins during supervised fine-tuning **without using OOD validation labels**?

This repository studies whether **functional drift** and **spectral drift** can provide a practical checkpoint-selection signal during LLM supervised fine-tuning (SFT).

## Motivation

Recent work suggests that OOD reasoning performance may peak early during SFT and then decline even while in-distribution (ID) performance or training loss continues to improve. This project therefore treats **phenomenon reproduction as a gate**: first establish whether the effect appears in a small open model, then evaluate label-free checkpoint selectors.

## Core hypotheses

- **H1 — Early OOD peak:** OOD performance peaks earlier than ID performance during SFT.
- **H2 — Drift change point:** functional drift or singular-vector/principal-angle rotation shows a change point that precedes or coincides with OOD decline.
- **H3 — Practical selection:** an OOD-label-free drift-based selector achieves lower checkpoint regret than train-loss, ID-validation, or fixed-step selectors.

### Competing hypotheses

- **C1:** OOD forgetting appears only in some model/data/LR regimes.
- **C2:** spectral drift correlates with OOD performance but is not useful for practical early stopping.
- **C3:** simple functional KL drift is a stronger signal than spectral quantities.

## Primary endpoint

The main endpoint is **checkpoint-selection regret**, not correlation:

```text
Regret = OOD_oracle - OOD_selected_by_proxy
```

Selectors to compare:

1. training loss
2. ID validation loss / ID metric
3. fixed epoch / fixed step
4. parameter-norm drift
5. functional KL drift
6. spectral rotation
7. combined proxy

## Minimum decisive experiment

### Stage 1 — Verify the phenomenon

- one open LLM in the **1.5B–3B** range
- **LoRA**
- ID SFT data: roughly **2K–5K** examples
- about **3 learning rates**
- **2 seeds**
- **1–3 epochs**
- frequent checkpoint evaluation, targeting roughly every **50–100 optimizer steps** when practical

At each checkpoint record:

- train loss
- ID validation loss
- ID accuracy / exact match
- OOD accuracy / exact match
- parameter L2 drift
- functional KL drift
- singular-vector / principal-angle drift

### Stage 1 gate

- **GO:** clear OOD peak-and-decline behavior appears in at least two training conditions and differs from ID behavior.
- **CONDITIONAL GO:** forgetting appears only under some LR/model conditions; pivot toward studying its boundary conditions.
- **KILL / PIVOT:** no meaningful forgetting appears across reasonable LR/seed conditions.

Stage 2 begins only after Stage 1 passes or conditionally passes.

## Functional drift

For a fixed anchor set (A):

[
D_{func}(t)
=
\mathbb{E}_{x \in A}
\mathrm{KL}
\left[
p_{\theta_t}(\cdot|x)
\;\|\;
p_{\theta_0}(\cdot|x)
\right].
]

The anchor set must remain fixed across checkpoints.

## Planned main figure

x-axis: SFT step

Plot or align:

- ID performance
- OOD performance
- functional drift
- spectral drift

Report a separate selector table:

| Selector | Selected checkpoint | OOD score | Regret |
|---|---:|---:|---:|
| Oracle OOD | TBD | TBD | 0 |
| Train loss | TBD | TBD | TBD |
| ID selector | TBD | TBD | TBD |
| Functional KL | TBD | TBD | TBD |
| Spectral rotation | TBD | TBD | TBD |
| Combined proxy | TBD | TBD | TBD |

## Hardware

- NVIDIA GeForce RTX 4090 24GB × 2
- assign different LR/seed conditions to the two GPUs for parallel pilot runs

## Milestones

- **~2026-10-02:** literature verification + professor proposal
- **next 1–2 days after PASS:** base/SFT evaluation pipeline
- **following 1–2 days:** forgetting-curve reproduction
- **2026-10-12:** main finding + OOD/drift figure
- **November:** expand model/data/LR/seed robustness
- **2026-12-14:** experiments and thesis complete

## Repository layout

```text
.
├── README.md
├── configs/              # experiment configs
├── docs/                 # protocol, metrics, literature notes
├── scripts/              # training/evaluation entry points
├── src/                  # reusable implementation
├── tests/                # unit/smoke tests
├── data/                 # local datasets/caches (gitignored)
├── checkpoints/          # local model checkpoints (gitignored)
└── results/              # compact tables/figures/metadata
```

## Research principle

Do **not** assume OOD forgetting is universal. The first experiment is a reproduction gate. A negative result is useful if it identifies the model/data/LR regimes in which the phenomenon does or does not occur.
