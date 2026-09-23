# Novelty Positioning After 2026 Literature Check

## Important correction

The project must **not** claim:

> “We are the first to select an OOD-reliable SFT checkpoint without OOD labels.”

That claim is already occupied by closely related work.

Vo & Nguyen (UAI 2026) propose the **Collapse Risk Criterion (CRC)**, computed from ID-validation representations, specifically as a practical surrogate for OOD-free checkpoint selection.

## Defensible research gap

A narrower and stronger question remains:

> In trajectories where SFT itself induces OOD degradation, can **base-relative functional drift** and **parameter singular-subspace drift** identify the useful checkpoint prospectively, and how do they compare with CRC and conventional selectors?

This differs from CRC in the signal family and causal hypothesis:

| Axis | CRC | This project |
|---|---|---|
| Main internal object | Hidden-state representation collapse | Base-relative output behavior + parameter geometry |
| Spectral object | Effective rank of representation covariance | Singular-vector / principal-angle rotation of weight matrices |
| Functional signal | Not the central signal | KL from base model on a fixed unlabeled anchor set |
| Target phenomenon | OOD reliability under distribution shift | Explicit SFT checkpoint trajectory around OOD forgetting / non-monotonic dynamics |
| Primary evaluation | OOD selection performance | Checkpoint regret + temporal relation to OOD peak/decline |
| Key comparison | ID loss/confidence | ID loss, fixed step, norm drift, **CRC**, functional KL, spectral rotation |

## Contribution ladder

A strong thesis does not require every hypothesis to succeed.

### Outcome A — strongest

- OOD peak→decline reproduced in multiple conditions.
- Functional/spectral change point is stable across seeds/LRs.
- Proposed selector has lower regret than train loss, ID validation, fixed step, and CRC.
- Result survives the prompt-diversity control and public reasoning confirmation.

### Outcome B — still strong

- Forgetting is real only in some optimization/data regimes.
- Drift predicts the boundary or identifies the regime, even if one universal stopping rule fails.

Contribution: **boundary conditions + diagnostic checkpoint signal**.

### Outcome C — useful negative result

- CRC or functional KL works, but singular-vector rotation does not translate into prospective selection.

Contribution: **mechanistic correlation is not sufficient for early stopping**.

### Outcome D — useful negative result

- LoRA suppresses the forgetting observed under full FT.

Contribution: **adaptation method / update rank is itself a boundary condition for OOD forgetting**, connected to prior LoRA-retention and OPLoRA results.

## Claims to avoid until evidence exists

Do not claim before the experiments:

- OOD forgetting is universal.
- singular-vector rotation causes forgetting in our setting.
- functional KL is superior to CRC.
- a change point precedes the OOD decline.
- the GeneralPoints result automatically transfers to broad mathematical reasoning.
- a fixed-prompt GeneralPoints decline is necessarily genuine reasoning loss rather than instruction shortcutting.

The experiment is designed precisely to distinguish these possibilities.
