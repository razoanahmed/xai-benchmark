# PROJECT_PLAN.md — Stage Schedule

**Status: DRAFT.** This is a proposed 13-stage breakdown (0 through 12, plus 6.5) based on the rules and grid in `CLAUDE.md`. It has not been checked against your supervisor's actual plan — review and correct before treating it as normative.

---

## Stage 0 — Environment

Set up the repo structure, Python environment, and `requirements.txt`. Pin dependency versions. Set a global random seed policy so runs are reproducible.

## Stage 1 — Data acquisition ✅ DONE

All three datasets (Sepsis, Sparkov, CIC-IDS2017) downloaded and verified into `data/raw/`.

## Stage 2 — Throwaway timing test

Run a tiny end-to-end slice (n=5) of loading → model fit → one explanation method, per dataset, just to measure wall-clock time. Multiply up to estimate full-run cost before committing to Stage 3+. This is a scratch script, not part of the real pipeline — do not build on top of it.

## Stage 3 — Config-driven pipeline skeleton ✅ DONE

Built the pipeline so a dataset is a settings file (paths, time column, group column, target, split rule), not new code. Loading and splitting logic are shared across all three datasets via `src/pipeline/`, driven by `configs/*.yaml`. See "Stage 3 pipeline skeleton" in `CLAUDE.md` for details. Cleaning was a deliberate no-op stub at this point — the real per-dataset quirks were Stage 4.

## Stage 4 — Per-dataset preprocessing ✅ DONE

Applied the documented quirks per dataset in `src/pipeline/cleaning.py`:
- Sepsis: forward-fill lab columns within each patient's timeline, then fill remaining gaps with training-set medians.
- Sparkov: drop unnamed index column, downsample negatives within train and test independently to raise positive rate to ~2% on both sides.
- CIC-IDS2017: drop null-`Label` rows, drop duplicate rows, drop rows with an infinite value in any numeric column.

Split by the rule in `CLAUDE.md` (patient / date / date) — never a random split — already handled in Stage 3; cleaning now runs after splitting so Sepsis's median fill only ever sees training data. See "Stage 4 per-dataset cleaning" in `CLAUDE.md` for details, including the two methodology calls made with the user (downsample both splits for Sparkov; train-median fill for Sepsis) and the degenerate all-missing-column edge case found and fixed.

## Stage 5 — Feature reduction ✅ DONE

Reduced every dataset using one consistent method: mutual information with the target, scored on training data only, top-k kept. Sepsis and CIC-IDS2017 land at exactly 25 features. Sparkov caps at 21 — most of its raw columns turned out to be per-customer identifiers in disguise (see CLAUDE.md's Stage 5 notes), and standard feature engineering closed most but not all of the gap to 25. Documented as a known limitation rather than padded further. See "Stage 5 feature reduction" in `CLAUDE.md` for the full reasoning, including three rounds of methodology decisions made with the user and two bugs found and fixed during verification.

## Stage 6 — Model training ✅ DONE

Trained XGBoost, LSTM, and FT-Transformer on each dataset (9 models total, all at full scale) with fixed, untuned hyperparameters shared across all three datasets per model type (`configs/models.yaml`). LSTM and FT-Transformer both treat each row as one independent sample, not a sequence, so every model explains the same instance unit for the Stage 7+ comparison. Hit and fixed two real environment bugs along the way (a torch/XGBoost import-order segfault, an FT-Transformer out-of-memory on a large unbatched prediction pass) — see "Stage 6 model training" in `CLAUDE.md` for the full account, including the fixed hyperparameters and full-scale results table. No accuracy tuning was done — the resulting models are mediocre in places (e.g. CIC-IDS2017's neural models overfit, test AUC 0.6-0.74), which is expected and fine.

## Stage 6.5 — SageMaker setup ✅ DONE

Set up the AWS side before any explanation work runs on it: S3 bucket, IAM role, upload the processed data and the 9 trained models from Stage 6 to S3, and verify a Processing Job runs end to end on a trivial smoke test before trusting it with the real Stage 7 sweep. Training (Stage 6) stays local — training time is not one of this project's metrics, so there's no reason to pay for it on AWS.

Bucket `directed-study-sagemaker-project` and role `directed-study-sagemaker-iamrole` created in `ca-central-1`; identifiers recorded in `configs/aws.yaml`. Along the way, found and fixed a real bug where `models.py` let a dev-scale test run silently overwrite the full-scale Stage 6 models (see CLAUDE.md's Stage 6.5 notes) — all 9 models were retrained at full scale and re-verified against the Stage 6 results table before uploading. Smoke-tested a real Processing Job on `ml.m5.xlarge` end to end (S3 read → container → S3 write), confirmed by reading the output back from S3 directly. `sagemaker` pip package needed pinning to `2.257.6` (the classic v2 API) since a v3 rewrite has since become the default install. Instance type for Stage 7: **`ml.m5.xlarge`**.

## Stage 7 — Explanation generation ✅ DONE

Run SHAP, LIME, and Permutation Importance against each trained model, entirely on AWS SageMaker as Processing Jobs. Cap SHAP/LIME at 200–500 explained rows. This produces the 27 experiment cells (3 datasets × 3 models × 3 XAI methods).

**Hard rule: all 27 cells run on the same SageMaker instance type, with none run locally.** Mixing local and SageMaker runs (or mixing instance types across cells) invalidates the Stage 9 cost comparison — cost and runtime are only comparable across cells if the hardware is held constant. Pick the instance type once, for the whole sweep, before starting.

**Reproducibility:** record the exact instance type used (e.g. `ml.m5.xlarge`) in the results. Anyone re-running this benchmark needs to know what hardware the cost/runtime numbers came from.

All 27 cells ran on `ml.m5.xlarge`, one SageMaker Processing Job per cell (never batched, so each job's own reported duration is directly Stage 9's per-cell runtime with no extra accounting). Four real bugs found and fixed along the way — a missing `configs/models.yaml` mount, a `numpy>=2`/torch ABI conflict forced by `shap`'s own dependency requirement, a `shap==0.49.1` regression parsing some XGBoost 3.2.0 tree JSON, and a `test_df.head(n_cap)` sampling bias that affected every cell (not just the ones that crashed from it) — see CLAUDE.md's "Stage 7 explanation generation" notes for the full account. Because the sampling fix changed what "explained rows" means for every cell, the entire 27-cell grid was re-run from scratch rather than only patching the failures, so the whole study uses one consistent sampling methodology. Final state independently verified against AWS (`describe-processing-job`) and S3 directly for all 27 cells, not inferred from the sweep script's own exit code.

## Stage 8 — Explanation quality metrics ✅ DONE

Compute the quality metrics defined in `docs/METRICS.md`. **`METRICS.md` does not exist yet** — it must be written (or provided) before this stage starts. Do not invent metrics ad hoc.

Implemented exactly what `docs/METRICS.md` specifies, nothing more: `src/pipeline/metrics.py` (P1–P5 ranking rules, M1–M4 agreement), `src/pipeline/quality_metrics.py` (Family C, M9–M12, one subprocess per `(dataset, model)` for the same torch/XGBoost isolation reason as Stage 7), `src/pipeline/run_stage8.py` (orchestrates everything, writes exactly the 6 files in METRICS.md section 5's output contract to `results/`). M5's wall-clock time is sourced from AWS's own `describe_processing_job` API per cell, not the local `wall_seconds` proxy in `stage7_sweep_results.json`. The M7/M8 hourly rate ($0.230/hr) is a US (`us-east-1`) estimate, not verified for `ca-central-1` — the CLI IAM user has no Pricing API or Cost Explorer access; decided with the user to proceed with the flagged estimate rather than block on an IAM change. See CLAUDE.md's "Stage 8 explanation quality metrics" notes for full detail, including a real cost finding (FT-Transformer+LIME is ~25× slower on AWS's CPU-only instance than the local MPS-accelerated timing suggested — worth highlighting in the write-up).

This Stage's original one-line description mentioned a **memory** metric that `docs/METRICS.md`'s Family B never defined. **Resolved** — see Stage 9 below: evaluated and deliberately excluded, not added to METRICS.md.

## Stage 9 — Computational cost metrics ✅ DONE

Record runtime, memory, and **cost in dollars** for each XAI method run in Stage 7. Runtime comes for free once Stage 7 runs on SageMaker — each Processing Job reports its own duration, so there's no separate timing step to build. Dollar cost = job duration × the recorded instance type's per-second SageMaker price.

**Runtime and dollar cost:** covered entirely by Stage 8's Family B (`results/metrics_cost.csv`: M5 wall-clock — sourced from AWS's own `describe_processing_job` API, not a local proxy — M6 time-per-instance, M7 estimated cost, M8 relative cost). Nothing further needed for these two.

**Memory: evaluated, then deliberately excluded as a metric** (decided with the user; `docs/METRICS.md` was not amended). CloudWatch's `MemoryUtilization` for the 27 Processing Jobs was checked and confirmed retrievable — no re-running required — but the sampling turned out too uneven across cells to support a fair comparison: CloudWatch reports this metric on a 1-minute period, and most of the 27 cells ran for only ~200 seconds total, dominated by the one-time pip install rather than the sub-second explanation compute itself, leaving most cells with just 1–2 datapoints (effectively one arbitrarily-timed snapshot, most likely taken during the pip install, not the actual SHAP/LIME/PI call). The three FT-Transformer+LIME cells, the slowest in the grid, were the exception — 25–29 datapoints each, a genuinely sampled maximum. Comparing "max of 1 sample" against "max of 29 samples" across the grid isn't a fair basis for a metric, so it was left out.

**Qualitative observation, not a metric:** the only reliably-measured cells (FT-Transformer+LIME, 16 GiB `ml.m5.xlarge`) peaked at 8.5–12.2% memory utilization (Sepsis 12.17%, CIC-IDS2017 11.97%, Sparkov 8.49%) — the same three cells Stage 8 already flagged as the time/cost outliers. This corroborates those findings rather than independently confirming anything new: the memory data doesn't reveal a different pattern than time and cost already showed.

The raw CloudWatch data (all 27 cells, including the unreliable low-sample-count ones, with sample sizes and window timestamps) is saved to `results/stage9_memory_cloudwatch_raw.csv` via `src/pipeline/run_stage9.py`, on record even though it isn't used as a metric.

## Stage 10 — Rank-based comparison ✅ DONE

Convert each method's per-feature attribution into ranks before comparing across methods. Never compare SHAP values, LIME weights, and PI scores directly — they are not on the same scale.

Both requirements were already satisfied by Stage 8's Family A before this stage was explicitly checked: `metrics.py`'s `rank_cell()` converts every cell's importance to ranks (P3) before any comparison, and `metrics_agreement_cross_method.csv`/`metrics_agreement_cross_model.csv` compare only rank-derived statistics (Spearman, Kendall's tau-b, top-k Jaccard) — no raw SHAP value ever sits next to a raw LIME weight or PI score anywhere in Stage 8's output. Confirmed by an item-by-item comparison against this stage's original two-sentence description before declaring it done, rather than assuming the overlap with Stage 8.

The one real gap found in that comparison: the per-feature rank vectors themselves were never persisted, only the summary agreement scores computed from them — not required by this stage's literal text, but a genuine prerequisite for Stage 11 ("which feature ranks #1 most often", "show XGBoost's SHAP ranking next to its LIME ranking"), which needs the actual rankings, not just how much two methods agree. Closed by adding `results/ranked_features.csv` (one row per `dataset`×`model`×`method`×`feature`, its rank, and its importance after P1/P2 aggregation) to Stage 8's output — documented in `docs/METRICS.md` section 5 as an intermediate, not a new metric, since it introduces no new M-ID.

## Stage 11 — Aggregation and analysis ✅ DONE

Aggregate results across all 27 experiments into summary tables/figures. Identify patterns by domain, model type, and method.

`src/pipeline/run_stage11.py` re-aggregates Stage 8's output (no new per-cell metrics) into 5 CSVs + 3 figures in `results/`, structured around the research question's three parts. See CLAUDE.md's "Stage 11 aggregation and analysis" notes for the full set of findings, including the headline one: cross-domain rank-order preservation (A3) holds in 0/8 metric×family combinations at the strict "identical ordering" bar, but 6/8 share a specific sub-pattern — Sepsis is consistently the domain that breaks the ordering that otherwise holds between Sparkov and CIC-IDS2017.

**Correction (2026-09-14):** the first-pass explanation for that ("Sepsis's severe imbalance and weak neural models") does not hold up against the worked example, which is an XGBoost cell — Sepsis's *best* model, not a weak one. Checked and ruled out in turn: weak model (Sepsis/XGBoost's AUC is within the confound threshold of CIC-IDS2017's), class imbalance alone (Sparkov has nearly identical imbalance but normal agreement), feature collinearity (Sepsis has the *lowest* mean feature correlation of the three datasets, not the highest), and weak attribution signal. The real mechanism is not identified — see CLAUDE.md's Stage 11 notes for the full test-by-test account. State as an open question in the write-up, not a resolved finding.

## Stage 12 — Write-up ✅ DONE

Produce the supervisor-facing summary (`docs/DATASET_SPEECH.md`) and final results writeup in `results/`.

Both written 2026-09-14. `docs/DATASET_SPEECH.md`: concise, supervisor-facing — research question, headline findings for all 3 RQ parts, the Sepsis-explanation correction (reported as a finding, not hidden), cost, and limitations. `docs/WRITEUP.md`: the full write-up — methodology, all 3 RQ parts with complete tables, the §5 worked example, cost, Family C quality context, and limitations, every number sourced directly from `results/metrics_*.csv` / `ranked_features.csv` / `stage11_*.csv`, cross-checked against `metrics_manifest.json` for provenance. Both explicitly carry the corrected Stage 11 finding (see Stage 11 above) rather than the original, unverified explanation — the point of Stage 8's rigor (checking things against data before asserting them) was to make write-ups like this trustworthy, not just to produce numbers.

Moved from `results/WRITEUP.md` to `docs/WRITEUP.md` on 2026-09-14 so it's version-controlled alongside `DATASET_SPEECH.md`, `PROJECT_PLAN.md`, and `METRICS.md` — `results/` stays gitignored for the generated CSVs/figures those documents reference, but the write-up prose itself isn't a generated artifact in the same sense and belongs in source control.

With this, all 13 stages (0 through 12, plus 6.5) are complete.

---

## Open items before this plan is trustworthy

1. ~~`docs/METRICS.md` doesn't exist — needed before Stage 8.~~ **Resolved 2026-09-14** — written, see Stage 8 above.
2. ~~`docs/DATASET_SPEECH.md` doesn't exist — needed for Stage 12, may be useful earlier too.~~ **Resolved 2026-09-14** — written, see Stage 12 above.
3. Confirm this 13-stage breakdown matches whatever your supervisor actually assigned — this draft was inferred from `CLAUDE.md`, not sourced from a syllabus or supervisor doc.
4. ~~Stage 6.5's AWS resources (S3 bucket, IAM role) and Stage 7's instance type aren't chosen yet.~~ **Resolved 2026-09-13**: bucket, role, and instance type (`ml.m5.xlarge`) all set up and verified — see Stage 6.5 above. Estimated full Stage 7 cost (see 2026-09-12 discussion, scaled from Stage 2's timing numbers): roughly $1-3, well inside the <$100 budget — dominated by per-job startup overhead across 27 Processing Jobs, not actual compute.
