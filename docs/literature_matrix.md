# Literature Evidence Matrix

Use one row per paper / experimental protocol. Separate claims directly supported by the paper from project interpretation.

| Paper | Status | Problem | Model(s) | Data | Training objective | ID definition | OOD definition | Metric | Forgetting curve evidence | Parameter / representation analysis | Checkpoint-selection gap | Replication notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jin et al., *RL Fine-Tuning Heals OOD Forgetting in SFT* | To verify | | | | | | | | | | | |
| *When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents* | To verify | | | | | | | | | | | |
| GSM8K original / protocol references | To verify | | | | | | | | | | | |
| Recent catastrophic / OOD forgetting work | To collect | | | | | | | | | | | |

## Evidence-status vocabulary

- **Verified:** primary source inspected; experimental detail and claim match.
- **Partially verified:** primary source inspected but a key detail remains ambiguous.
- **Preprint / provisional:** recent evidence that should not be presented as established fact.
- **Secondary only:** claim currently supported only by another paper, blog, or summary.
- **Unverified:** not yet checked.

## Mandatory questions for each paper

1. What exact phenomenon was claimed?
2. What models and parameter scales were used?
3. What training data and quantity were used?
4. What objective and optimizer regime were used?
5. How were ID and OOD operationally defined?
6. What metric demonstrates forgetting?
7. Is the peak-and-decline curve actually visible and statistically credible?
8. What parameter/representation quantities were measured?
9. Were those quantities merely correlated with performance, or used prospectively?
10. Did the paper solve checkpoint selection without OOD labels?
