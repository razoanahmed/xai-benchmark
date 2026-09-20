# DATASET_SPEECH.md — Supervisor-Facing Summary

**Cross-Domain Benchmarking of Explainable AI Techniques for Temporal Tabular Data**

Status as of 2026-09-14: all 13 stages (0 through 12, plus 6.5) complete. Full write-up in `docs/WRITEUP.md`.

---

## The question

> How do SHAP, LIME, and Permutation Importance compare in explanation agreement and computational cost across XGBoost, LSTM, and FT-Transformer models, and do those results hold across healthcare, finance, and cybersecurity domains?

Three datasets (Sepsis / healthcare, Sparkov / finance, CIC-IDS2017 / cybersecurity) × three model types × three XAI methods = 27 experiment cells, all run on identical infrastructure (AWS SageMaker Processing, `ml.m5.xlarge`) so cost and runtime are comparable across cells.

## What was built

A config-driven pipeline (`src/pipeline/`) taking each dataset from raw files through cleaning, a shared 25-feature reduction method (mutual information, train-only), model training, explanation generation, and metric computation. All 27 explanation-generation jobs and their cost/runtime figures come from real AWS Processing Jobs, not local estimates. All quality/agreement metrics are defined normatively in `docs/METRICS.md`, written before Stage 8 started and implemented exactly as specified — no metric was added ad hoc.

## Headline findings

**1. Do SHAP, LIME, and Permutation Importance agree with each other on the same model?**
Inconsistently, and not in the way a naive intuition would predict. SHAP agrees most *reliably* with Permutation Importance (mean Spearman ρ = 0.57, low variance) — not with LIME, despite SHAP and LIME both being local/instance-based methods. SHAP↔LIME agreement is highly unstable (ρ ranges from −0.40 to 0.87 across the 9 model/dataset cells) — the two most commonly paired methods in practice sometimes agree strongly and sometimes not at all.

**2. Does a single method agree with itself across XGBoost, LSTM, and FT-Transformer?**
SHAP is the most self-consistent across architectures (ρ = 0.57 overall, 0.69 excluding cells where the two models being compared differ by more than 0.10 ROC-AUC). Permutation Importance — despite being the only method that never touches model internals — is the *least* self-consistent (0.42 / 0.56). Model-quality confounds are real and unevenly distributed: Sparkov has none; Sepsis and CIC-IDS2017 each have several, concentrated wherever LSTM is being compared.

**3. Do these patterns hold across domains?**
No, not exactly — at the strictest bar (identical pairwise ordering across all three domains), 0 of 8 metrics tested preserve it. But it is not random either: in 6 of those 8, Sparkov and CIC-IDS2017 consistently agree with each other on the ordering, and Sepsis is the domain that breaks it. We investigated why (see below) and the first explanation we tried was wrong.

**A correction made along the way, worth reporting because of how it was found, not despite it:** the initial hypothesis — that Sepsis breaks the pattern because of weak neural models and severe class imbalance — was checked against a concrete worked example (SHAP vs. LIME on Sepsis/XGBoost, the single worst disagreement in the whole grid: zero overlap in the top 5 features) and did not hold up. XGBoost is Sepsis's *best*-performing model, not a weak one, and Sparkov has nearly identical class imbalance (2.0% vs. 1.8% positive) without showing the same effect. Feature collinearity was also checked and ruled out — Sepsis has the *lowest* average feature correlation of the three datasets, not the highest. The honest finding: something about Sepsis specifically causes this, and it isn't any of the three most obvious candidates. That mechanism is an open question for future work, not a solved one.

**Cost.** Permutation Importance is cheapest for every model. FT-Transformer + LIME is a dramatic outlier — roughly 9× the cost of the cheapest method on the same model, and about 25× slower than a local Apple Silicon (MPS) timing taken earlier in the project would have predicted, since AWS's CPU-only instance has no equivalent acceleration. Total cost for the full 27-cell sweep: **$0.65** (at an unverified US-region rate — see limitations).

## Limitations, stated plainly

- **Correctness, not causal certainty**, for explanation agreement — the disagreement patterns are measured precisely; *why* Sepsis behaves differently is not yet known.
- **Faithfulness, robustness, and complexity** are explicitly out of scope (documented in `METRICS.md` §4) — this study establishes *that* the methods disagree and *what it costs*, not *which one is right*.
- **The hourly SageMaker rate used for cost figures ($0.230/hr) is a US-region estimate**, not verified for the `ca-central-1` region actually used — total cost is small enough ($0.65) that this doesn't change any conclusion, but the absolute dollar figures shouldn't be over-trusted.
- **A memory metric was investigated and deliberately excluded** — CloudWatch's sampling was too uneven across cells (1–2 datapoints for most, 25–29 for the three FT-Transformer+LIME cells) to support a fair comparison.

## What's left

Nothing in the planned 13-stage scope — this document and `docs/WRITEUP.md` were the last deliverable. Three things remain genuinely open, none of them blocking:

1. **Supervisor confirmation of the stage breakdown itself** — `docs/PROJECT_PLAN.md`'s 13 stages were inferred from `CLAUDE.md`, never checked against an actual syllabus or supervisor assignment. Worth a quick confirmation.
2. **The `ca-central-1` SageMaker hourly rate used for cost figures is an unverified US-region estimate** — total cost is small enough ($0.65) that this doesn't change any conclusion, but the exact dollar figures in `docs/WRITEUP.md` §6 could be tightened if the real rate is checked.
3. **The Sepsis SHAP-LIME divergence mechanism is unidentified** — `docs/WRITEUP.md` §4 rules out three candidate explanations but doesn't resolve what actually causes it. Stated there as future work, not a gap in this study's own scope.
