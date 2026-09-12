# PROJECT_PLAN.md — Stage Schedule

**Status: DRAFT.** This is a proposed 12-stage breakdown based on the rules and grid in `CLAUDE.md`. It has not been checked against your supervisor's actual plan — review and correct before treating it as normative.

---

## Stage 0 — Environment

Set up the repo structure, Python environment, and `requirements.txt`. Pin dependency versions. Set a global random seed policy so runs are reproducible.

## Stage 1 — Data acquisition ✅ DONE

All three datasets (Sepsis, Sparkov, CIC-IDS2017) downloaded and verified into `data/raw/`.

## Stage 2 — Throwaway timing test

Run a tiny end-to-end slice (n=5) of loading → model fit → one explanation method, per dataset, just to measure wall-clock time. Multiply up to estimate full-run cost before committing to Stage 3+. This is a scratch script, not part of the real pipeline — do not build on top of it.

## Stage 3 — Config-driven pipeline skeleton ✅ DONE

Built the pipeline so a dataset is a settings file (paths, time column, group column, target, split rule), not new code. Loading, cleaning, and splitting logic are shared across all three datasets via `src/pipeline/`, driven by `configs/*.yaml`. See "Stage 3 pipeline skeleton" in `CLAUDE.md` for details. Cleaning is currently a deliberate no-op stub — the real per-dataset quirks are Stage 4.

## Stage 4 — Per-dataset preprocessing

Apply the documented quirks per dataset:
- Sepsis: `sep='|'`, extract patient ID from filename.
- Sparkov: drop unnamed index column, downsample negatives to raise positive rate to ~2%.
- CIC-IDS2017: strip leading spaces from column names, drop duplicate rows, handle infinite values in flow columns.

Split by the rule in `CLAUDE.md` (patient / date / date) — never a random split.

## Stage 5 — Feature reduction

Reduce every dataset to exactly 25 features, using one consistent method across all three.

## Stage 6 — Model training

Train XGBoost, LSTM, and FT-Transformer on each dataset (9 models total). No accuracy tuning — a mediocre model is expected and fine.

## Stage 7 — Explanation generation

Run SHAP, LIME, and Permutation Importance against each trained model. Cap SHAP/LIME at 200–500 explained rows. This produces the 27 experiment cells (3 datasets × 3 models × 3 XAI methods).

## Stage 8 — Explanation quality metrics

Compute the quality metrics defined in `docs/METRICS.md`. **`METRICS.md` does not exist yet** — it must be written (or provided) before this stage starts. Do not invent metrics ad hoc.

## Stage 9 — Computational cost metrics

Record runtime and memory cost for each XAI method run in Stage 7.

## Stage 10 — Rank-based comparison

Convert each method's per-feature attribution into ranks before comparing across methods. Never compare SHAP values, LIME weights, and PI scores directly — they are not on the same scale.

## Stage 11 — Aggregation and analysis

Aggregate results across all 27 experiments into summary tables/figures. Identify patterns by domain, model type, and method.

## Stage 12 — Write-up

Produce the supervisor-facing summary (`docs/DATASET_SPEECH.md`) and final results writeup in `results/`.

---

## Open items before this plan is trustworthy

1. `docs/METRICS.md` doesn't exist — needed before Stage 8.
2. `docs/DATASET_SPEECH.md` doesn't exist — needed for Stage 12, may be useful earlier too.
3. Confirm this 12-stage breakdown matches whatever your supervisor actually assigned — this draft was inferred from `CLAUDE.md`, not sourced from a syllabus or supervisor doc.
