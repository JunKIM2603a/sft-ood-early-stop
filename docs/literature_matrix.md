# Literature Evidence Matrix

Last verified: 2026-09-23

This document separates **claims supported by primary sources** from project interpretation. Recent preprints are treated as provisional evidence, not established facts.

## Summary matrix

| Work | Evidence status | Problem | Model(s) | Data / objective | ID / OOD definition | Main metric / curve | Parameter / representation analysis | Checkpoint-selection gap | Implication for this project |
|---|---|---|---|---|---|---|---|---|---|
| Jin et al., *RL Fine-Tuning Heals OOD Forgetting in SFT* (arXiv:2509.12235, v3 2026) | **Primary paper + official code verified; preprint** | Explain why SFT→RL can outperform SFT alone and whether SFT loses OOD ability over training | LLaMA-3.2-11B-Vision, Qwen2.5-7B; v3 also reports a public Qwen2.5-3B setting | Full-parameter SFT; controlled GeneralPoints / navigation / matrix tasks; public Open-R1 math experiment added in v3 | GeneralPoints ID: J/Q/K=10. OOD: standard J=11,Q=12,K=13. Public experiment uses math benchmarks as ID and broader reasoning/general benchmarks as OOD | OOD can peak early and then decline while ID continues improving; train/test loss does not identify the OOD-best checkpoint | SVD of parameter matrices: singular values comparatively stable; **singular-vector rotation** tracks forgetting/restoration | Identifies the failure of loss-based selection and a mechanism, but does **not** establish a practical OOD-label-free selector from vector rotation | Direct motivation for H1/H2. Exact published controlled recipe is much larger and uses full FT, so our 3B+LoRA run is a resource-adapted reproduction, not an exact replication |
| Chu et al., *SFT Memorizes, RL Generalizes* (ICML 2025; arXiv:2501.17161) | **Peer-reviewed + official data/code verified** | Compare SFT vs RL generalization under controlled rule and visual shifts | Llama-3.2-Vision family in original release | SFT vs RL on GeneralPoints and V-IRL | GeneralPoints provides explicit rule shifts, including face-card remapping | Task success / generalization performance | Behavioral analysis; not a checkpoint-selection method | No validation-free SFT checkpoint selector | Provides the clean, non-arbitrary controlled ID/OOD protocol used by later Jin work |
| Lin et al., *Debunk the Myth of SFT Generalization* (arXiv:2510.00237) | **Primary preprint + official code/data verified** | Test whether reported SFT OOD failures are partly caused by fixed-prompt shortcuts | Qwen2.5-7B and Llama-3.1-8B-Instruct; official PEFT script also includes Qwen3-8B LoRA | GeneralPoints + Sokoban; answer-only / CoT; fixed vs diverse prompts | Explicit instruction/rule variants and difficulty variants | Shows prompt diversity can remove much of the instruction-variant failure; CoT helps difficulty transfer | No functional/spectral checkpoint selector | No OOD-free checkpoint selection focus | Critical competing explanation: a GeneralPoints decline can be a **fixed-prompt artifact**. Therefore a prompt-diverse control is required before claiming broad OOD forgetting |
| Ren et al., *Rethinking Generalization in Reasoning SFT* (arXiv:2604.06628, updated 2026) | **Primary paper/code verified; recent preprint/conference-version metadata should be treated cautiously** | Determine when long-CoT SFT generalizes across domains | Qwen3 1.7B/4B/8B/14B, Qwen2.5 incl. 3B, InternLM variants | Math-CoT-20k from OpenR1-Math-220k; standard autoregressive SFT | ID: MATH500/AIME24. OOD reasoning includes LiveCodeBench, GPQA-Diamond, MMLU-Pro; also general capability/safety | Reports **dip-and-recovery** in some OOD metrics under longer optimization; behavior depends on model/data/training depth | Optimization/data/capability analysis rather than singular-vector selector | Does not solve our specific base-relative drift selector problem | Strong evidence for C1: non-monotonic OOD dynamics need not be peak→decline; insufficient training can produce a misleading conclusion |
| Vo & Nguyen, *Collapse-Aware Regularization for Reliable Reasoning Under Distribution Shift* (UAI 2026, PMLR 337) | **Peer-reviewed; core claim verified from official PMLR page; detailed experiment settings only partially extracted** | Select checkpoints that remain reliable OOD when ID validation loss is misleading | Three Transformer backbones; four reasoning benchmarks | Reasoning fine-tuning / evaluation under distribution shifts | Depth/difficulty and template/rule shifts | Defines **CRC** from ID-validation representations; reports stronger relation to OOD error than ID loss/confidence and improved OOD checkpoint selection | Effective rank of layerwise representation covariance + average cosine alignment | **Already proposes an OOD-free checkpoint selector** | Changes novelty: we must compare against CRC. Our contribution cannot be “first OOD-free selector”; it must concern SFT-induced forgetting and base-relative functional/parameter drift |
| Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024; arXiv:2405.09673) | **Peer-reviewed / accepted TMLR; primary abstract verified** | Compare LoRA and full FT on learning vs retention | Llama-2 7B/13B; math and code settings | Instruction FT and continued pretraining | Target-domain learning vs outside-domain retention | Standard low-rank LoRA learns less but tends to preserve outside-domain performance better than full FT | Shows full-FT perturbations can have far higher effective rank than typical LoRA updates | Not a checkpoint selector | Major Stage-1 risk: **LoRA can suppress forgetting**. A LoRA-only null result must not be overinterpreted |
| Xiong & Xie, *OPLoRA* (AAAI 2026) | **Peer-reviewed official AAAI source verified** | Reduce PEFT catastrophic forgetting by preventing LoRA interference with dominant pretrained singular subspaces | LLaMA-2 7B, Qwen2.5 7B | LoRA / OPLoRA across commonsense, math, code | Target task vs retained pretrained capabilities | Task performance + forgetting | Projects LoRA updates away from top-k singular subspaces; introduces update/subspace alignment metric rho_k | Mitigation, not our checkpoint-selection question | Strong mechanistic neighbor for H2. Add rho_k / dominant-subspace interference as a later spectral diagnostic if time permits |
| Murtaza et al., *When Synthetic Data Hurts* (arXiv:2609.10750; EMNLP Industry-track acceptance reported by authors) | **Recent preprint + official repository verified** | Catastrophic forgetting in LLM-agent skill retrieval after synthetic-data fine-tuning | Qwen3-Embedding-0.6B retriever/reranker | Real + synthetic skill retrieval; LoRA-based recipes and continual-learning regularizers | Real ID / synthetic ID / Terminal-Bench-derived OOD rings | Recall@10; synthetic ID can improve while real/OOD retrieval degrades | Embedding-anchor, LwF, EWC, L2-init; no singular-vector checkpoint rule | No generative-reasoning checkpoint selector | Confirms forgetting under small-model LoRA in a different retrieval domain, but is not direct evidence for generative math reasoning |
| Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K; arXiv:2110.14168) | **Primary dataset paper verified** | Grade-school mathematical reasoning benchmark | Various LM/verifier setups | 7,473 train / 1,319 test in the canonical HF release | **No canonical OOD split is defined by GSM8K itself** | Exact-answer accuracy | None relevant | None | Do **not** make “GSM8K→some other benchmark” the primary Stage-1 OOD protocol unless grounded by a prior paper; that would be an investigator-defined shift |

## Primary-source details that materially affect the pilot

### 1. Jin et al.: closest phenomenon paper

Official Qwen GeneralPoints SFT script in `jinhangzhan/RL_Heals_SFT` uses:

- Qwen2.5-7B-Instruct
- full-parameter SFT (`lora_enable=False`)
- learning rate `1e-6`
- 1 epoch
- effective minibatch 64 (16 per process × 4)
- bf16
- cosine schedule with warmup ratio 0.03
- checkpoint save every 100 steps
- ID GeneralPoints evaluation with face cards treated as 10
- OOD evaluation with face cards treated as regular values (J=11, Q=12, K=13)
- 234 evaluation trajectories in the released evaluation scripts

This exact setup does **not** fit our 3B+LoRA resource-adapted pilot directly. The `1e-6` LR is a full-FT value and should not simply be copied to LoRA.

### 2. Lin et al.: directly relevant GeneralPoints LoRA recipe

The official GeneralPoints PEFT script in `XiaofengLin7/debunking-sft-generalization` uses:

- LoRA rank 32
- LoRA alpha 16
- `target_modules=all-linear`
- learning rate `1e-5`
- effective train batch size 64
- 5 epochs
- max length 2048
- a 10k fixed-prompt GeneralPoints training split

The official collection separates roles: `Xiaofeng77/answer-only-gp-l-only-10k` is the non-diverse answer-only **SFT** source, while `Xiaofeng77/gp-l-only-10k` is the RL/evaluation release with the explicit GeneralPoints test variants. The public evaluation repo currently exposes `test` rather than `test_id`; the repository code still contains local `test_id.parquet` references, so our loader verifies the actual rule semantics instead of assuming the split name.

This is the strongest task-specific LR anchor for our **LoRA** pilot. We therefore center the LR sweep on `1e-5` rather than importing Jin's full-FT `1e-6`.

### 3. Why GeneralPoints is only the first gate

GeneralPoints has three advantages for Stage 1:

1. the ID/OOD shift is explicit and inherited from prior work rather than invented post hoc;
2. exact answers can be programmatically verified, making frequent checkpoint evaluation cheap;
3. Jin et al. explicitly use controlled tasks for dense checkpoint trajectories.

But Lin et al. show that the same rule-shift failure can be strongly affected by prompt diversity. Therefore:

> A fixed-prompt GeneralPoints peak→decline is evidence that we reproduced the published controlled phenomenon, **not by itself proof of broad reasoning OOD forgetting**.

A prompt-diversity control and later public-benchmark confirmation are required before making a broader claim.

## What is actually novel enough to test

After the 2026 CRC paper, the defensible research gap is:

> During **SFT-induced OOD forgetting trajectories**, can cheap **base-relative functional drift and parameter singular-subspace drift** prospectively identify the useful checkpoint, and do they outperform conventional selectors **and CRC** under the same OOD-label-free protocol?

The project should therefore compare:

- train loss
- ID validation loss / ID score
- fixed step
- parameter L2 drift
- CRC (representation-collapse baseline)
- functional KL drift
- singular-vector / principal-angle drift
- optional OPLoRA-style dominant-subspace interference
- combined proxy

The primary endpoint remains **OOD checkpoint regret**, not correlation.

## Reference identifiers / official sources

- Jin et al.: arXiv:2509.12235; official code `jinhangzhan/RL_Heals_SFT`
- Chu et al.: arXiv:2501.17161 / ICML 2025; official data `tianzhechu/SFTvsRL_Data`
- Lin et al.: arXiv:2510.00237; official code `XiaofengLin7/debunking-sft-generalization`; SFT data `Xiaofeng77/answer-only-gp-l-only-10k`; evaluation data `Xiaofeng77/gp-l-only-10k`
- Ren et al.: arXiv:2604.06628; official code `Nebularaid2000/rethink_sft_generalization`
- Vo & Nguyen: PMLR 337, UAI 2026, paper id `vo26a`
- Biderman et al.: arXiv:2405.09673 / TMLR 2024
- Xiong & Xie: AAAI 2026, DOI 10.1609/aaai.v40i40.40703
- Murtaza et al.: arXiv:2609.10750; official code `manulife-ai/emnlp2026`
- GSM8K: arXiv:2110.14168
