# Research Plan

## Core question

Can an SFT checkpoint immediately before OOD forgetting be selected without access to OOD validation labels?

## Why this is not yet assumed to work

The project treats recent reports of SFT OOD forgetting as a phenomenon that must first be reproduced in the target compute/model regime. Some core evidence is recent or preprint-level, so reproduction is a prerequisite for method claims.

## Hypotheses

### H1 — Early OOD peak

OOD performance reaches its maximum earlier than ID performance during SFT.

### H2 — Drift change point

Relative to the base model, functional drift or singular-vector / principal-angle rotation exhibits a change point that precedes or coincides with OOD decline.

### H3 — Validation-free selection

A selector that does not use OOD labels selects checkpoints with lower OOD regret than conventional train-loss, ID-validation, or fixed-step selectors.

## Competing hypotheses

- C1: forgetting is conditional on model, data, or learning rate.
- C2: spectral drift is descriptive but not operationally useful.
- C3: functional KL is a stronger checkpoint signal than spectral geometry.

## Decision order

1. Verify the literature and reproduce the phenomenon.
2. Freeze the definition of ID, OOD, anchor data, metrics, and Stage-1 gate.
3. Run the minimum decisive experiment.
4. Only after GO / CONDITIONAL GO, develop proxy selectors.
5. Compare selectors by checkpoint regret, not by correlation alone.
6. Expand robustness only after the main finding is secured.

## Primary endpoint

For each run:

[
\mathrm{Regret}
=
\mathrm{OOD}_{oracle}
-
\mathrm{OOD}_{selected\ by\ proxy}.
]

Aggregate regret across LR/seed conditions. Report the full per-run distribution as well as summary statistics.

## Main contribution target

A practical OOD-validation-free checkpoint-selection method for SFT, together with evidence about when OOD forgetting and drift signals do or do not appear.

## Valuable negative outcomes

The project remains useful if it shows that:

- forgetting is restricted to particular regimes;
- spectral quantities correlate with OOD but fail as selectors;
- functional KL dominates the more complex spectral signals;
- a claimed forgetting pattern cannot be reproduced under the defined small-model regime.
