# METRICS.md — Measurement Rules (Normative)

This document defines every metric computed in Stage 8 and later. It is
normative: Stage 8 code implements exactly what is specified here and
nothing else. If a metric is needed that is not in this document, this
document is amended first, then the code is written.

Research question this serves:

> How do SHAP, LIME, and Permutation Importance compare in **explanation
> agreement** and **computational cost** across XGBoost, LSTM, and
> FT-Transformer models, and do those results hold across healthcare,
> finance, and cybersecurity domains?

Two metric families answer that question directly (A and B). A third
family (C) exists as a control, not as a headline result.

---

## 0. Preprocessing — making three methods comparable

SHAP, LIME, and Permutation Importance produce values that are **not on a
common scale** and **not at a common granularity**. They must be
normalised before any comparison. These five rules apply to every cell
before any metric in this document is computed.

### P1 — Aggregate local explanations to global

SHAP and LIME produce one attribution per feature *per explained
instance*. Permutation Importance produces one score per feature for the
whole dataset.

To compare them, aggregate SHAP and LIME to global importance by taking
the **mean absolute attribution across all explained instances**:

```
global_importance(feature_j) = mean_i( | attribution(i, j) | )
```

Permutation Importance is already global and is used as-is.

**Recorded consequence:** all comparisons in this document are between
*global* feature importances. Local (per-instance) explanation behaviour
is not measured. This is a deliberate scope boundary, not an oversight.

### P2 — Discard sign

SHAP and LIME attributions are signed (a feature can push a prediction up
or down). Permutation Importance is unsigned by construction.

Absolute values are taken in P1. All three methods are therefore compared
on **magnitude of influence**, not direction.

**Recorded consequence:** directional disagreement between SHAP and LIME
is invisible to these metrics.

### P3 — Compare ranks, never raw values

The three methods measure different quantities in different units:

| Method | Unit of its scores |
|---|---|
| SHAP | Units of model output (log-odds / probability) |
| LIME | Coefficients of a local linear surrogate |
| Permutation Importance | Units of the loss metric that degraded |

Raw magnitudes are therefore **meaningless to compare across methods**.
Every cell's global importances are converted to a **rank vector** over
features, where rank 1 = most important.

All Family A metrics operate on rank vectors only.

### P4 — Identical feature set and order

For a given `(dataset, model)`, the three method rankings must cover the
**same feature list in the same order** before ranking. Any feature
present in one method's output but absent from another's is a bug, not a
result — fail loudly rather than dropping it.

### P5 — Ties and negative values

- **Negative Permutation Importance** (shuffling a feature improved the
  model, by chance) is clipped to `0` before ranking. The count of clipped
  features is recorded per cell.
- **Ties** are assigned average ranks (`scipy.stats.rankdata` method
  `"average"`).
- Because Permutation Importance often produces many zero-importance
  features, tie counts per cell are recorded and reported alongside every
  correlation, since heavy ties weaken Spearman.

---

## 1. Family A — Explanation Agreement (primary)

The core of the study. Three sub-families, each answering a different part
of the research question.

### Metrics used throughout Family A

Given two rank vectors `r1`, `r2` over the same `p` features:

| ID | Metric | Definition | Range | Reads as |
|---|---|---|---|---|
| **M1** | Spearman rank correlation `ρ` | Pearson correlation of the two rank vectors | −1 to 1 | 1 = identical ordering |
| **M2** | Kendall's tau-b `τ` | Concordant minus discordant pairs, tie-corrected | −1 to 1 | 1 = identical ordering |
| **M3** | Top-5 Jaccard | \|top5(r1) ∩ top5(r2)\| / \|top5(r1) ∪ top5(r2)\| | 0 to 1 | 1 = same top 5 features |
| **M4** | Top-10 Jaccard | Same, at k=10 | 0 to 1 | 1 = same top 10 features |

**Why all four:** M1 and M2 measure agreement over the *whole* ranking.
M3 and M4 measure agreement where it matters in practice — a practitioner
reads the top of the list, not the bottom. These can disagree sharply, and
that disagreement is itself a finding. M2 is reported alongside M1
specifically because tie-heavy Permutation Importance rankings distort M1.

`k=10` is skipped for any dataset with fewer than 20 features after
reduction; record `null`, do not silently shrink `k`.

### A1 — Cross-method agreement

*Do the three XAI methods agree with each other on the same model?*

For each of the 9 `(dataset, model)` pairs, compute M1–M4 for each of the
three method pairs:

- SHAP ↔ LIME
- SHAP ↔ Permutation Importance
- LIME ↔ Permutation Importance

**Output: 27 comparisons × 4 metrics.**

### A2 — Cross-model agreement

*Does the same XAI method give the same answer on different
architectures?*

For each of the 9 `(dataset, method)` pairs, compute M1–M4 for each of the
three model pairs:

- XGBoost ↔ LSTM
- XGBoost ↔ FT-Transformer
- LSTM ↔ FT-Transformer

**Output: 27 comparisons × 4 metrics.**

This sub-family tests the model-dependency finding reported by Salih et
al. (2024) for shallow models, extended here to deep architectures.

### A3 — Cross-domain stability

*Do the A1 and A2 patterns hold across healthcare, finance, and
cybersecurity?*

Not a new metric. A3 is the comparison of A1 and A2 results **across the
three datasets**. Report, for each metric pair:

- Mean agreement per domain
- Spread across domains (max − min)
- Whether the *rank ordering of method pairs by agreement* is preserved
  across all three domains

The last item is the key claim: if SHAP↔LIME agreement exceeds
SHAP↔PFI agreement in all three domains, that pattern is
domain-independent. If it inverts in any domain, it is not.

---

## 2. Family B — Computational Cost (primary)

Sourced from SageMaker Processing Job metadata already captured in
`stage7_sweep_results.json`. No re-runs required.

| ID | Metric | Definition | Unit |
|---|---|---|---|
| **M5** | Wall-clock time | `elapsed_seconds` reported per job | seconds |
| **M6** | Time per instance | `elapsed_seconds / n_rows_explained` | seconds/instance |
| **M7** | Estimated cost | `(elapsed_seconds / 3600) × instance_hourly_rate` | USD |
| **M8** | Relative cost | M6 for this cell ÷ M6 of the cheapest method on the same `(dataset, model)` | ratio |

**M6 is the headline cost metric, not M5.** Wall-clock alone is not
comparable across cells because `n_rows_explained` and feature count
differ between datasets. Normalising per instance is what makes the
numbers mean something.

**M8** answers the practitioner's question directly: *how many times more
expensive is SHAP than Permutation Importance on this model?*

Required recorded fields for reproducibility: instance type
(`ml.m5.xlarge`), region (`ca-central-1`), and the hourly rate used for
M7, with the date the rate was checked.

**Scope note:** M5 measures the full Processing Job duration, which
includes container startup and data download, not explainer compute
alone. This inflates fast cells more than slow ones. Record this as a
stated limitation rather than attempting to subtract it.

---

## 3. Family C — Model Quality (control, not a result)

These are the "traditional" metrics. They exist to **rule out a
confound**, not to answer the research question.

If XGBoost achieves 0.95 AUC and the LSTM achieves 0.60 on the same
dataset, then any disagreement between their explanations is partly
explained by one model simply being worse — not by architecture-dependent
explanation behaviour. BEExAI (2024) handles this by restricting analysis
to models of comparable performance, and Molnar et al. (2022) name poor
model generalisation as a distinct interpretation pitfall.

All three datasets are binary classification.

| ID | Metric | Why |
|---|---|---|
| **M9** | Accuracy | Baseline readability |
| **M10** | F1 (positive class) | All three datasets are class-imbalanced |
| **M11** | ROC-AUC | Threshold-independent |
| **M12** | PR-AUC | More informative than ROC-AUC under heavy imbalance |

Computed on the held-out test split for all 9 `(dataset, model)` pairs.

**Reporting rule:** every Family A result is reported **alongside** the
M11 values of the models being compared. Any comparison where the two
models differ by more than **0.10 ROC-AUC** is flagged as
performance-confounded in the results table and excluded from headline
claims about architecture-dependent disagreement.

---

## 4. Explicitly out of scope

The following are standard XAI evaluation properties that this study does
**not** measure. Each is listed with the reason, so the omission is a
stated design decision rather than a gap.

| Property | Why excluded |
|---|---|
| **Faithfulness** (does removing high-attribution features degrade the prediction?) | Requires a new inference sweep with masked inputs across all 27 cells — a second full SageMaker grid. Outside the timeline. |
| **Robustness / Sensitivity** (do similar inputs get similar explanations?) | Requires generating explanations for perturbed neighbourhoods of each instance; cost scales with the perturbation count. |
| **Complexity / Sparseness** | Computable from existing outputs but does not bear on agreement or cost. |
| **Directional agreement** | Excluded by rule P2 above. |
| **Local (per-instance) agreement** | Excluded by rule P1 above. |

**Consequence to state plainly in the paper:** this study can establish
*that* the three methods disagree and *what it costs* to run each. It
cannot establish *which method is correct*, because correctness requires
faithfulness evaluation. This is a limitation, not a finding, and belongs
in the Limitations section and in Future Work.

---

## 5. Output contract

Stage 8 writes the following, and nothing else:

```
results/
├── metrics_agreement_cross_method.csv    # A1: 27 rows × M1–M4
├── metrics_agreement_cross_model.csv     # A2: 27 rows × M1–M4
├── metrics_agreement_cross_domain.csv    # A3: summary comparison
├── metrics_cost.csv                      # B:  27 rows × M5–M8
├── metrics_model_quality.csv             # C:  9 rows × M9–M12
├── metrics_manifest.json                 # provenance (see below)
└── ranked_features.csv                   # intermediate, not a metric -- see note below
```

`ranked_features.csv` is the P3/P5-ranked form of every cell's already
P1/P2-aggregated importance (one row per `dataset` × `model` × `method` ×
`feature`, carrying that feature's rank and its global importance value
after P1/P2 aggregation, before P5 clipping) — it is what Family A's
comparisons are computed *from*, not a new metric itself. Persisted
because Stage 11's aggregation work needs the actual rankings, not just
the summary agreement scores Family A reduces them to.

`metrics_manifest.json` records, for every metric file: the Stage 7 input
files consumed with their checksums, the library versions used
(`scipy`, `numpy`, `pandas`), the random seed where applicable, the
hourly instance rate and the date it was checked, and the UTC timestamp of
the run.

Every CSV row carries its full cell identity — `dataset`, `model`,
`method` (or the pair being compared) — so that no row's meaning depends
on its position in the file.

---

## 6. Amendment log

| Date | Change | Reason |
|---|---|---|
| _(initial)_ | Document created | Unblock Stage 8 |
| 2026-09-14 | Section 5: documented `ranked_features.csv` alongside the 6 metric files | Not a new metric (no new M-ID) — the already-computed P3/P5 rank vectors, persisted because Stage 11 needs the actual rankings, not just Family A's summary agreement scores. Added while closing out Stage 10, whose two requirements (rank before comparing; never compare raw values across methods) were confirmed already satisfied by Stage 8's Family A. |
