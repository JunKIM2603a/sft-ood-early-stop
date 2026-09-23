# Stage-1 Checkpoint Evaluation

The first scientific training condition (`LR=1e-5, seed=42`) completed all 192 optimizer steps with stable finite loss/gradients and about 11.36 GiB peak allocated VRAM.

The next goal is the first checkpoint trajectory:

```text
step   ID validation loss   ID task accuracy   OOD task accuracy
0              ...                ...                ...
10             ...                ...                ...
...
192            ...                ...                ...
```

No functional/spectral drift is used yet. This remains the Stage-1 phenomenon gate.

## 1. Freeze ID-loss validation before viewing OOD results

The existing manifest already fixed 4,096 SFT examples and 512 unlabeled functional-KL anchor prompts. For the `ID validation loss` baseline, the protocol now fixes the next 512 examples from the same deterministic shuffled answer-only pool:

```text
seed 20260923
first 4096  -> SFT train
next 512    -> unlabeled anchor
next 512    -> labeled ID-loss validation
```

This does not alter the completed SFT run.

Run:

```bash
git pull
python scripts/data/smoke_generalpoints.py
python scripts/data/audit_generalpoints_tokens.py
```

If an older manifest exists, the data smoke refuses to rewrite it if the previous SFT or anchor indices differ.

## 2. Verify the independent GeneralPoints scorer

```bash
pytest -q tests/test_generalpoints_verifier.py
```

The scorer accepts a response only when its formula can be parsed, uses only integer constants with parentheses and `+ - * /`, uses exactly the multiset of the four `display_cards`, and evaluates exactly to 24. It uses Python AST plus exact Fraction arithmetic, not `eval`.

## 3. Evaluation smoke first

Do not evaluate all 21 checkpoints immediately. First test base plus one LoRA checkpoint on 32 examples per task split:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_stage1.py \
  --steps 0,10 \
  --max-examples 32 \
  --eval-batch-size 16
```

This checks base evaluation, adapter reload, batched greedy generation, ID validation-loss computation, verifier parsing, and result writing. These rows are marked `SMOKE_MAX_32` and are not research results.

## 4. Full trajectory

After the evaluation smoke passes, use a clean summary directory:

```bash
rm -rf results/stage1/lr-1e-05_seed-42
rm -rf results/raw/stage1/lr-1e-05_seed-42

CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_stage1.py \
  --steps all \
  --eval-batch-size 32 \
  --loss-batch-size 16
```

The evaluator uses deterministic greedy generation (`do_sample=False`).

## 5. Outputs

Small summary files:

```text
results/stage1/lr-1e-05_seed-42/
├── checkpoint_metrics.jsonl
└── checkpoint_metrics.csv
```

Raw generations are written under the gitignored path `results/raw/stage1/lr-1e-05_seed-42/`.

Each checkpoint summary includes ID validation loss, ID task accuracy, OOD task accuracy, ID/OOD parse rate, ID/OOD valid-formula rate, and evaluated example counts.

## 6. Why parse/valid rates matter

If OOD accuracy falls, we need to distinguish reasoning/rule failure from output-format collapse. A drop in accuracy with stable parse/valid rates is different from a formatting failure.

## 7. Decision after the full first curve

Only after all 21 states are evaluated do we classify the first condition as clear peak-to-decline, monotonic/improving, dip-to-recovery, or noisy/ambiguous. Do not tune drift-based proxy thresholds before this classification.
