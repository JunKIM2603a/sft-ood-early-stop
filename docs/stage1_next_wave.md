# Stage-1 Next Wave After the First 1e-5 / Seed-42 Run

The first completed condition was technically healthy but under-learned:

- ID validation loss fell strongly.
- parse/valid-formula rates approached 1.
- ID task accuracy improved only from 0% to 3.2%.
- OOD peak-to-final decline was only 1.6 percentage points.
- therefore the original clear peak-to-decline criterion was false.

This does **not** establish that forgetting is absent. The run did not acquire
enough ID task success to make a forgetting interpretation strong.

## 1. Ground-truth verifier sanity

Before spending more GPU time, verify that the official answers pass our
independent scorer:

```bash
python scripts/sanity_check_generalpoints_verifier.py
```

Default target is 100% on the frozen 512-example ID-loss validation subset.
If it fails, inspect the generated JSON report and fix the verifier before
running more scientific conditions.

## 2. Post-pilot acquisition guard

Added on 2026-09-24 **after observing the first run**. It is not retroactively
claimed as part of the original predeclared forgetting criterion.

For interpretation, report both:

1. the original peak-to-decline criterion;
2. an acquisition-qualified result requiring max ID task accuracy to improve
   by at least 5 percentage points over the step-0 base.

The 5-point guard reuses the original minimum practical effect-size floor. It
is a sanity/interpretability guard, not a new primary endpoint.

Run on the completed first condition:

```bash
python scripts/summarize_stage1_run.py \
  --metrics results/stage1/lr-1e-05_seed-42/checkpoint_metrics.jsonl
```

Expected classification from the observed 3.2-point ID gain:

```text
UNDER_LEARNED_INCONCLUSIVE_FOR_FORGETTING
```

## 3. Stronger-LR two-GPU wave

The next most informative conditions are LR=5e-5 with both seeds.

One-command launcher:

```bash
bash scripts/run_stage1_pair_2gpu.sh 5e-5 42 43
```

Equivalent conceptual allocation:

```text
GPU 0 -> LR 5e-5, seed 42
GPU 1 -> LR 5e-5, seed 43
```

Each GPU runs one independent model; do not place two training processes on
the same GPU.

## 4. Evaluate both completed runs in parallel

After both trainings pass:

```bash
bash scripts/evaluate_stage1_pair_2gpu.sh 5e-5 42 43
```

This evaluates all 21 checkpoints of seed 42 on GPU 0 and all 21 checkpoints
of seed 43 on GPU 1, then writes an interpretability summary for each run.

## 5. Decision after the 5e-5 pair

If one or both 5e-5 runs show meaningful ID acquisition:

- interpret their OOD trajectories;
- count clear peak-to-decline runs under the original criterion;
- separately report acquisition-qualified clear forgetting.

If both remain under-learned, do **not** spend the next wave on 5e-6 first.
Instead use an acquisition sentinel closer to the released PEFT recipe
(e.g. more data/epochs) before concluding the phenomenon is absent.
