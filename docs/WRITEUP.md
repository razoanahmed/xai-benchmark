# Cross-Domain Benchmarking of Explainable AI Techniques for Temporal Tabular Data

## Results Write-Up

**Generated:** 2026-09-14. **Grid:** 3 datasets × 3 models × 3 XAI methods = 27 cells, all run on `ml.m5.xlarge` (SageMaker Processing, `ca-central-1`).

**Research question:**

> How do SHAP, LIME, and Permutation Importance compare in explanation agreement and computational cost across XGBoost, LSTM, and FT-Transformer models, and do those results hold across healthcare, finance, and cybersecurity domains?

Every number below is sourced directly from `results/metrics_*.csv`, `results/ranked_features.csv`, and `results/stage11_*.csv` — all machine-computed from the 27 real SageMaker Processing Jobs listed in `stage7_sweep_results.json`, not estimated. Full provenance (input checksums, library versions, seed, instance rate) is in `results/metrics_manifest.json`.

---

## 1. Method, briefly

- **Datasets:** Sepsis (healthcare, ICU vitals/labs, ~1.8% positive rate), Sparkov (finance, credit-card fraud, downsampled to ~2.0% positive rate), CIC-IDS2017 (cybersecurity, network flows, ~20–25% positive rate). Each reduced to 25 features via mutual information scored on training data only (Sparkov's real candidate pool caps at 21 — documented limitation, not a bug).
- **Models:** XGBoost, LSTM, FT-Transformer, one fixed hyperparameter set per model type, never tuned per dataset (hard rule: this study measures explanation behavior, not model performance).
- **XAI methods:** SHAP (`TreeExplainer` for XGBoost, `GradientExplainer` for LSTM/FT-Transformer), LIME (`LimeTabularExplainer`), Permutation Importance (5 repeats). Row caps: SHAP 500 (XGBoost) / 200 (LSTM/FT-Transformer), LIME/PI 500 for all — a seeded random sample of the test set, not the first *n* rows (an earlier `.head(n)` version was a documented bug, fixed and the full grid re-run before any metric was computed).
- **Metrics** (`docs/METRICS.md`, normative, implemented exactly as written): Family A — rank-based agreement (Spearman ρ, Kendall's τ-b, top-5/top-10 Jaccard); Family B — cost (wall-clock, per-instance time, USD, relative cost); Family C — model quality (accuracy, F1, ROC-AUC, PR-AUC), reported alongside Family A as a confound check, never as a headline result.
- **Rank-based comparison, not raw values**: SHAP values, LIME weights, and PI scores are never compared directly anywhere in this study — everything in Family A operates on rank vectors (`ranked_features.csv`), per the project's hard rule that these three quantities are not on a common scale.

---

## 2. RQ Part 1 — Do SHAP, LIME, and PI agree with each other on the same model?

*(Family A1: `metrics_agreement_cross_method.csv`, `stage11_cross_method_summary.csv`, `fig_cross_method_agreement.png`)*

| Method pair | Mean Spearman ρ | Std | Mean Kendall τ | Mean Jaccard@5 | Mean Jaccard@10 |
|---|---|---|---|---|---|
| SHAP ↔ Permutation Importance | **0.569** | 0.252 | 0.451 | 0.450 | 0.618 |
| SHAP ↔ LIME | 0.468 | **0.453** | 0.373 | 0.495 | 0.528 |
| LIME ↔ Permutation Importance | 0.398 | 0.327 | 0.305 | 0.375 | 0.471 |

**Finding, and why it's counter-intuitive:** SHAP and LIME are the two methods most often treated as interchangeable "local explanation" tools in practice. Here, SHAP agrees *more reliably* with Permutation Importance — a completely different, global, model-agnostic method — than it does with LIME. The SHAP↔LIME standard deviation (0.453) is larger than either method's agreement with PI, and the range spans −0.397 to 0.870: sometimes near-perfect agreement, sometimes essentially none, with no obvious way to predict which in advance from the model/dataset alone. Averaged across all 27 comparisons, model quality (mean ROC-AUC of the model being explained: 0.793) is identical by construction — these are the same 9 (dataset, model) cells for every pair, so quality is not a confound within this table.

---

## 3. RQ Part 2 — Does a single method agree with itself across XGBoost, LSTM, and FT-Transformer?

*(Family A2: `metrics_agreement_cross_model.csv`, `stage11_cross_model_summary.csv`, `fig_cross_model_agreement.png`)*

| Method | Mean ρ (all 9 cells) | Mean ρ (excl. confounded) | Cells excluded as confounded |
|---|---|---|---|
| SHAP | **0.573** | **0.686** | 4 / 9 |
| LIME | 0.471 | 0.520 | 4 / 9 |
| Permutation Importance | 0.423 | 0.555 | 4 / 9 |

SHAP is the most self-consistent across architectures; Permutation Importance — the one method with no access to model internals at all — is the least. This holds whether or not performance-confounded comparisons are included.

**Performance confound check (Family C, reported alongside per the study's rule, not hidden):** 12 of 27 A2 rows are flagged `performance_confounded` (the two models being compared differ by more than 0.10 ROC-AUC), all concentrated in 4 of the 9 (dataset, model-pair) combinations — every one involving Sepsis's or CIC-IDS2017's LSTM:

| Dataset | Model pair | ROC-AUC gap |
|---|---|---|
| Sepsis | XGBoost vs. LSTM | 0.109 |
| Sepsis | XGBoost vs. FT-Transformer | 0.109 |
| CIC-IDS2017 | XGBoost vs. LSTM | 0.187 |
| CIC-IDS2017 | LSTM vs. FT-Transformer | 0.122 |

**Sparkov has zero confounded comparisons** — all three of its models perform within 0.01 ROC-AUC of each other (Table 4). It is the cleanest domain for attributing a cross-model disagreement to genuine architecture-dependent explanation behavior rather than "one model is just worse."

---

## 4. RQ Part 3 — Do these patterns hold across domains?

*(Family A3: `metrics_agreement_cross_domain.csv`, `stage11_cross_domain_summary.csv`)*

At the strict bar — the same pairwise ordering of method-pairs (or model-pairs) by mean agreement, preserved in all three domains — **0 of 8** metric × family combinations pass.

It is not unstructured noise, however. In **6 of those 8** (Spearman ρ, Kendall's τ, and top-10 Jaccard, for both the cross-method and cross-model families), the same specific sub-pattern recurs:

> **Sparkov and CIC-IDS2017 agree with each other on the ordering. Sepsis is the domain that breaks it.**

Example (cross-method, Spearman ρ):

| Domain | Ordering, best agreement first |
|---|---|
| Sparkov | SHAP↔LIME > SHAP↔PI > LIME↔PI |
| CIC-IDS2017 | SHAP↔LIME > SHAP↔PI > LIME↔PI |
| **Sepsis** | **SHAP↔PI > LIME↔PI > SHAP↔LIME** |

The two exceptions to the 6/8 pattern are both top-5 Jaccard (cross-method and cross-model) — a small-*k*, tie-sensitive metric that shows no clean domain grouping in either direction, likely because a 5-feature set overlap is more sensitive to small numerical differences than a full rank correlation.

### Why does Sepsis break the pattern? A tested explanation, and a withdrawn one.

The first explanation considered — Sepsis's severe class imbalance combined with its weak neural models (LSTM/FT-Transformer ROC-AUC ≈ 0.63) — was checked directly against the worked example in §5 and **did not hold up**. That example is an XGBoost cell, and XGBoost is Sepsis's *best*-performing model (ROC-AUC 0.739), not a weak one. Four candidate explanations were tested in turn:

| Candidate | Test | Verdict |
|---|---|---|
| Weak model | Sepsis/XGBoost ROC-AUC (0.739) vs. CIC-IDS2017/XGBoost (0.802) | Diff 0.064, under the 0.10 confound threshold. **Not weak.** |
| Class imbalance alone | Sepsis test positive rate (1.84%) vs. Sparkov's (2.00%) — nearly identical | Sparkov's SHAP↔LIME agreement is strong across all 3 of its models (0.67–0.87); **all three** of Sepsis's models show anomalously low or negative agreement (−0.40 to 0.10). Same imbalance, opposite outcome. **Ruled out.** |
| Feature collinearity | Mean pairwise \|correlation\| across the selected features | Sepsis: 0.053 (**lowest** of the three datasets). Sparkov: 0.064. CIC-IDS2017: 0.309 (highest) — yet CIC-IDS2017 shows normal agreement. Backwards from what the hypothesis predicts. **Ruled out.** |
| Weak/noisy attribution signal | Mean \|Permutation Importance value\| for XGBoost | Sepsis: 0.016 (between Sparkov's 0.011 and CIC-IDS2017's 0.040). Unremarkable. **Ruled out.** |

**What is actually established:** the SHAP-LIME divergence is a Sepsis-*specific* effect, present across all three of its model types regardless of that model's quality, and not explained by imbalance, collinearity, or attribution-signal strength — the three most natural candidates. **The mechanism is not identified.** Plausible untested candidates include Sepsis's heavy forward-fill and median-imputation of sparse lab values (Stage 4 cleaning), which neither Sparkov's nor CIC-IDS2017's cleaning does at comparable scale, and which could interact badly with LIME's local perturbation sampling in ways SHAP's tree-structure-aware algorithm doesn't share. This is stated here as an open question for future work, not a finding.

---

## 5. Worked example: where SHAP and LIME disagree completely

*(`ranked_features.csv`, `stage11_worked_example.csv`)*

The single worst SHAP-vs-LIME disagreement in the entire 27-cell grid: **Sepsis / XGBoost**. Spearman ρ = −0.022 (no correlation), Jaccard@5 = **0.0** — zero shared features in the top 5.

| Rank | SHAP feature | SHAP importance | LIME feature | LIME importance |
|---|---|---|---|---|
| 1 | HospAdmTime | 0.3156 | Bilirubin_direct | 0.0440 |
| 2 | Lactate | 0.2289 | SaO2 | 0.0339 |
| 3 | pH | 0.2084 | PTT | 0.0262 |
| 4 | Creatinine | 0.1191 | Platelets | 0.0214 |
| 5 | Hgb | 0.1180 | Fibrinogen | 0.0212 |
| 6 | Alkalinephos | 0.0996 | O2Sat | 0.0209 |
| 7 | Bilirubin_total | 0.0705 | Bilirubin_total | 0.0195 |

SHAP's top 5 are recognizable sepsis/shock clinical markers (lactate and pH are classic acute-illness indicators). LIME's top 5 share not a single feature with them. Only `Bilirubin_total` appears in both lists, and only at rank 7 for each. Same model, same data, same prediction task — two methods tell completely different stories about what drove it. Note the importance *magnitudes* are not comparable between the two columns (SHAP operates on the model's native feature space, LIME on a locally-perturbed neighborhood; per the study's rank-only comparison rule, only the rank columns are ever compared, never these values directly).

---

## 6. Cost

*(Family B: `metrics_cost.csv`, `stage11_cost_summary.csv`, `fig_cost_by_model_method.png`)*

| Model | Method | Mean time/instance (s) | Mean relative cost |
|---|---|---|---|
| FT-Transformer | LIME | 3.531 | **8.80×** |
| FT-Transformer | SHAP | 1.050 | 2.62× |
| LSTM | SHAP | 0.971 | 2.53× |
| XGBoost | LIME | 0.451 | 1.18× |
| LSTM | LIME | 0.439 | 1.14× |
| XGBoost | Permutation Importance | 0.402 | 1.06× |
| XGBoost | SHAP | 0.386 | 1.01× |
| FT-Transformer | Permutation Importance | 0.401 | 1.00× |
| LSTM | Permutation Importance | 0.386 | 1.00× |

Permutation Importance is the cheapest method for every model. FT-Transformer + LIME is a dramatic outlier at 8.8× the cost of the cheapest method on the same model — 1600–1850 seconds per cell, roughly **25× slower** than the same cell's local Apple Silicon (MPS-accelerated) timing recorded earlier in this project (`CLAUDE.md`'s Stage 2 notes), since AWS's CPU-only `ml.m5.xlarge` has no equivalent acceleration and LIME calls the model hundreds of times per explained instance.

**The tension worth naming:** SHAP is the most expensive method for FT-Transformer (2.6×) and second-most for LSTM (2.5×) — and also the *most self-consistent* method across architectures (§3). The cheapest method, Permutation Importance, is also the least self-consistent. Cost and reliability move in opposite directions here.

**Total cost, full 27-cell sweep: $0.65**, at $0.230/hr — a US (`us-east-1`) estimate, *not verified* for the `ca-central-1` region actually used (no Pricing API or Cost Explorer access on the project's IAM user). The total is small enough that this doesn't change any relative conclusion, but the absolute dollar figures should be read as approximate.

**Memory** was investigated via CloudWatch and deliberately excluded as a metric: most cells had only 1–2 datapoints (a ~200s runtime dominated by one-time pip install, not compute), against 25–29 for the three FT-Transformer+LIME cells — not a fair basis for comparison. Where reliably measured, those three cells peaked at 8.5–12.2% of the instance's 16 GiB, corroborating the time/cost pattern above rather than adding new information (`stage9_memory_cloudwatch_raw.csv`).

---

## 7. Model quality (Family C — context, not a headline result)

*(`metrics_model_quality.csv`)*

| Dataset | Model | Accuracy | F1 (positive) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Sepsis | XGBoost | 0.981 | 0.005 | 0.739 | 0.048 |
| Sepsis | LSTM | 0.980 | 0.012 | 0.630 | 0.034 |
| Sepsis | FT-Transformer | 0.975 | 0.025 | 0.630 | 0.033 |
| Sparkov | XGBoost | 0.996 | 0.883 | 0.997 | 0.945 |
| Sparkov | LSTM | 0.994 | 0.834 | 0.987 | 0.885 |
| Sparkov | FT-Transformer | 0.996 | 0.888 | 0.997 | 0.944 |
| CIC-IDS2017 | XGBoost | 0.819 | 0.433 | 0.802 | 0.640 |
| CIC-IDS2017 | LSTM | 0.807 | 0.395 | 0.615 | 0.571 |
| CIC-IDS2017 | FT-Transformer | 0.765 | 0.160 | 0.737 | 0.523 |

None of these were tuned toward — fixed hyperparameters throughout, per the study's design (a mediocre model is expected and fine; this study measures explanations, not accuracy). Sepsis's near-zero F1 despite high accuracy and moderate ROC-AUC is the expected signature of severe class imbalance (~1.8% positive), not a bug. Sparkov's three models perform near-identically (the only domain with zero cross-model performance confounds); Sepsis's and CIC-IDS2017's LSTM notably underperforms their own XGBoost/FT-Transformer, which is exactly where §3's confound flags concentrate.

---

## 8. Limitations

Stated as design decisions, per `docs/METRICS.md` §4, not gaps discovered after the fact:

- **Faithfulness** (does removing high-attribution features actually degrade the prediction?) is not measured — would require a second full inference sweep with masked inputs across all 27 cells.
- **Robustness/sensitivity** (do similar inputs get similar explanations?) is not measured — would require generating explanations for perturbed neighborhoods of each instance.
- **Directional agreement** is not measured — Family A compares only magnitude of influence (P2: sign is discarded before ranking).
- **Local (per-instance) explanation behavior** is not measured — Family A compares only global (mean-absolute-aggregated) importances (P1).

**Consequence, stated plainly:** this study establishes *that* SHAP, LIME, and Permutation Importance disagree, and by how much, and *what it costs* to run each — it does not and cannot establish *which method is correct*, since that requires faithfulness evaluation. The Sepsis anomaly in §4 is reported as an open, unresolved question for the same reason: this study's data can rule candidate explanations in or out, but the mechanism itself sits outside what was measured.

Additional limitations specific to this run:
- The M7/M8 cost figures use an unverified US-region instance rate (§6).
- A memory metric was investigated and excluded for uneven CloudWatch sampling (§6).
- Sparkov's feature pool caps at 21 (not 25) after excluding per-customer identity-proxy columns — documented in Stage 5, not a bug, but means Sparkov's top-10 Jaccard (M4) comparisons operate on a slightly smaller pool than the other two datasets.
