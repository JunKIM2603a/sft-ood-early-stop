# Experiment Protocol

## Stage 1: Phenomenon reproduction

### Goal

Establish whether OOD performance actually peaks and declines during LoRA SFT in the available 1.5B–3B model regime.

### Pilot design

- one open model in the 1.5B–3B range
- LoRA
- 2K–5K ID training examples
- approximately 3 learning-rate conditions
- 2 seeds
- 1–3 epochs
- frequent checkpoint saves and evaluations

Do not begin with a full Cartesian hyperparameter sweep.

### Checkpoint measurements

At checkpoint (t), record:

1. optimizer step / consumed training examples
2. train loss
3. ID validation loss
4. ID task performance
5. OOD task performance
6. parameter L2 drift from (	heta_0)
7. functional KL drift from the base model
8. singular-vector / principal-angle drift

All metrics must be tied to the exact checkpoint identifier.

## Fixed anchor set

The anchor set (A) is fixed before proxy comparison.

[
D_{func}(t)
=
\mathbb{E}_{x\in A}
\mathrm{KL}
\left[
p_{\theta_t}(\cdot|x)
\|p_{\theta_0}(\cdot|x)
\right].
]

Record token-selection and aggregation details because KL values depend on where logits are measured.

## Preventing leakage

OOD labels may be used only for:

- post-hoc oracle checkpoint identification;
- research evaluation of proxy regret.

OOD labels must not be used to tune:

- proxy thresholds;
- smoothing parameters;
- change-point hyperparameters;
- layer choices;
- proxy combinations.

Any such tuning requires a separate development protocol that remains OOD-label-free.

## Selector evaluation

For each run, produce:

| Selector | Selected step | OOD score | Oracle OOD score | Regret |
|---|---:|---:|---:|---:|
| Oracle | | | | 0 |
| Train loss | | | | |
| ID validation | | | | |
| Fixed step | | | | |
| Parameter L2 | | | | |
| Functional KL | | | | |
| Spectral rotation | | | | |
| Combined proxy | | | | |

## Stage-1 decision rule

**GO**

A clear peak-then-decline OOD curve appears in at least two training conditions and is meaningfully different from the ID trajectory.

**CONDITIONAL GO**

Forgetting appears only under some LR/model conditions. Reframe the next stage around boundary conditions and test selectors only where the phenomenon exists.

**KILL / PIVOT**

Reasonable LR/seed conditions fail to reproduce meaningful OOD forgetting. Do not force the proxy-selection story; document the reproduction failure and reassess the research question.
