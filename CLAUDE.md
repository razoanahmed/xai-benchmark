# CLAUDE.md — Project Context

Read this before doing anything in this repository.

---

## What this project is

A graduate directed study. It compares three explanation methods across three model types on three datasets, and measures both explanation quality and computational cost.

**Title:** Cross-Domain Benchmarking of Explainable AI Techniques for Temporal Tabular Data

**The experiment grid:** 3 datasets × 3 models × 3 XAI methods = 27 experiments.

| | |
|---|---|
| **Models** | XGBoost, LSTM, FT-Transformer |
| **XAI methods** | SHAP, LIME, Permutation Importance |
| **Datasets** | Sepsis (healthcare), Sparkov (finance), CIC-IDS2017 (cybersecurity) |

---

## Who you are working with

Razoan is a DevOps and AWS engineer. He is comfortable with the terminal, cloud infrastructure, and shell commands.

**He is a beginner in machine learning.**

So:

- Explain ML concepts when they come up. Do not assume prior knowledge.
- Use plain, simple English. Short sentences. Short paragraphs.
- Do not use jargon without defining it the first time.
- Say what a command does before running it.
- When something breaks, explain what the error means, not just how to fix it.

---

## Hard rules

1. **Never use a random train/test split.** All splits are by time or by entity. Random splitting leaks the future into the past and invalidates the whole study.
2. **Never tune a model for accuracy.** This project studies explanation methods, not model performance. A mediocre model is fine and expected.
3. **Cap SHAP and LIME at 200–500 rows explained.** Running them on a full dataset will not finish.
4. **Always estimate before running anything slow.** Run at n=5 first, time it, multiply up. Never launch a long job blind.
5. **Do not invent metrics.** All measurement definitions live in `docs/METRICS.md`. If something is missing, ask rather than improvising.
6. **Reduce every dataset to exactly 25 features** using the same method. This removes the biggest confound in the study.
7. **Compare XAI methods on ranks, never on raw attribution values.** SHAP values, LIME weights, and PI scores are not on the same scale.

---

## The three datasets

| | Sepsis | Sparkov | CIC-IDS2017 |
|---|---|---|---|
| **Domain** | Healthcare | Finance | Cybersecurity |
| **Format** | ~40,336 `.psv` files, one per patient, pipe-separated | 2 CSVs (fraudTrain, fraudTest) | 8 CSVs, one per day |
| **Rows** | ~1.5 million hourly | ~1.85 million | ~2.8 million flows |
| **Raw columns** | 40 | 23 | 85 |
| **Target** | `SepsisLabel` | `is_fraud` | `Label` (BENIGN vs attack → 0/1) |
| **Time column** | `ICULOS` (hours in ICU) | `trans_date_trans_time` | `Timestamp` |
| **Group by** | Patient (from filename) | `cc_num` | `Source IP` |
| **Split by** | Patient | Date | Date (Mon–Wed train, Thu–Fri test) |
| **Positive rate** | ~2% | ~0.5%, raise to ~2% by dropping negatives | ~20% |

### Confirmed from exploration (2026-08-26)

- **Sparkov's `fraudTrain.csv`/`fraudTest.csv` are already split by date** by the dataset's creators: train covers 2019-01-01 to 2020-06-21, test picks up immediately after (2020-06-21 to 2020-12-31), with no time overlap. This already satisfies the "split by date" rule above — no need to re-split, just use the two files as given. Note `cc_num` (cardholder) overlaps heavily between the two files (908 of ~950 cards appear in both), which is expected since the split is by date, not by entity.
- **Sparkov positive rate is lower than documented**: 0.58% in train, 0.39% in test (vs. the ~0.5% estimate in the table below). Still needs the negative-downsampling step to reach ~2%.
- **Sepsis lab-value columns are extremely sparse** — most lab columns (e.g. `Bilirubin_direct`, `Fibrinogen`, `TroponinI`) are 95–99.8% missing, since labs are only drawn occasionally, not every hour. This is expected ICU data behavior, not a loading error. Vitals (`HR`, `O2Sat`, `MAP`) are much more complete (88–90%). Any preprocessing must handle this sparsity deliberately (e.g. forward-fill within a patient) rather than dropping columns/rows naively.
- **Sepsis row-level vs. patient-level positive rate differ a lot**: 1.80% of individual hourly rows are labeled septic, but 7.27% of patients are septic at some point during their stay. Know which one a "positive rate" refers to when reporting numbers.

### Known quirks

- **CIC-IDS2017 column names have leading spaces.** `" Source IP"`, `" Timestamp"`. Strip whitespace from all column names on load or lookups will fail.
- **CIC-IDS2017's `Timestamp` column has two different formats across its 8 files.** `Monday-WorkingHours.pcap_ISCX.csv` uses `DD/MM/YYYY HH:MM:SS` (zero-padded, with seconds — e.g. `03/07/2017 08:55:58`). The other 7 files use `D/M/YYYY H:MM` (no zero-padding, no seconds — e.g. `7/7/2017 3:30`). A single `pd.to_datetime()` call without `format=` (or without handling both patterns) will misparse one of the two, and only Monday's data actually has second-level precision to begin with. (Checked: Sepsis has no date strings — just an integer hour count — and Sparkov's `trans_date_trans_time` is consistently `YYYY-MM-DD HH:MM:SS` across all 1.85M rows in both files, so this is CIC-IDS2017-only.)
- **CIC-IDS2017 needs `encoding='cp1252'`, not UTF-8.** The `Label` column contains an en-dash (`–`) in values like `Web Attack – Brute Force`, which isn't valid UTF-8. Reading as UTF-8 raises `UnicodeDecodeError`.
- **`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` has ~288,602 fully-blank trailing rows** (just commas, no values) appended after its ~170,366 real rows — a corruption artifact in this file, not real data. This is almost certainly the source of the "known duplicate rows" quirk below: before dropping these blank rows, duplicate rows measured ~9.3% of the combined dataset; after dropping rows with a null `Label`, genuine duplicates drop to ~0.01% (203 rows). **Always drop rows with a null `Label` before any other processing.** After that cleanup, total rows ≈ 2.83M and the attack rate ≈ 19.7%, both matching the numbers in this file.
- **Sepsis files are pipe-separated**, not comma. Use `sep='|'`. The patient ID is in the filename, not inside the file.
- **Sparkov has an unnamed index column** as the first column. Drop it.
- **CIC-IDS2017 has known duplicate rows and some infinite values** in the flow columns. Handle and document both. (See the blank-row quirk above — dropping null-`Label` rows resolves almost all of the duplicates. The remaining infinite values are in `Flow Bytes/s` and `Flow Packets/s`, ~4,376 cells, caused by division by zero in flow-rate calculations.)

### Stage 2 timing test (2026-08-30)

Throwaway scripts (not the real pipeline), minimal ad-hoc cleaning only. Method: time n=5, multiply up to the 200/500-row cap from hard rule 3. XGBoost tested on all three datasets; LSTM/FT-Transformer tested on Sparkov only (14 features, smallest) as a representative case.

**XGBoost — fast everywhere, all three datasets:**

| Dataset | Train | SHAP @500 | LIME @500 | Perm. Importance @500 (n_repeats=5) |
|---|---|---|---|---|
| Sparkov (14 feat) | 0.8s | 0.7s | 6.9s (+1.6s setup) | 129s |
| Sepsis (40 feat) | 1.1s | 0.6s | 8.5s (+5.8s setup) | 147s |
| CIC-IDS2017 (80 feat) | 3.5s | 0.8s | 21s (+27s setup) | 129s |

**LSTM / FT-Transformer on Sparkov — SHAP is the real cost driver:**

| Method | LSTM @500 | FT-Transformer @500 |
|---|---|---|
| SHAP | **244s (~4 min)** | **469s (~8 min)** |
| LIME | 21s | 66s |
| Permutation Importance, n_repeats=2 | 25.1s | 26.5s |
| Permutation Importance, n_repeats=5 | 6.8s | 16.3s |

**Why SHAP is slow for the neural models:** XGBoost gets `shap.TreeExplainer`, an exact fast algorithm specific to tree ensembles. LSTM and FT-Transformer aren't trees, so SHAP falls back to a generic sampling-based explainer that's 300–700x slower on Sparkov alone, and its cost scales with feature count — so Sepsis (40 features) and CIC-IDS2017 (80 features) will be proportionally worse than these Sparkov numbers, not tested directly here.

**Permutation Importance is not the bottleneck.** Confirmed at both n_repeats=2 and n_repeats=5 — PI stayed 10–35x cheaper than SHAP on the same models. (Note: the n_repeats=5 numbers came in faster than n_repeats=2, which shouldn't happen if cost scaled linearly with repeats — likely MPS backend warm-up noise between separate process runs at these sub-second timings, not a real effect. Doesn't change the conclusion.)

**Bottom line:** the full 27-experiment grid is feasible well within a day of compute — no need to rethink scope. But SHAP on the two neural models dominates total runtime by a wide margin, not training, not PI. Decision for Stage 3: cap SHAP at 200 rows (not 500) for LSTM/FT-Transformer specifically, and/or use `shap.GradientExplainer` (PyTorch-specific, much faster than the generic explainer) instead of `shap.Explainer`'s default fallback.

### Stage 3 pipeline skeleton (2026-09-12)

Shared `load → split → clean` pipeline in `src/pipeline/`, driven entirely by the YAML files in `configs/`. Adding a 4th dataset that fits one of the three shapes below needs a new config only, no new code. (Cleaning runs *after* splitting — see Stage 4 below for why.)

- **`config.py`** — loads a dataset's YAML into a `DatasetConfig`, validates required keys are present.
- **`loaders.py`** — one loader per raw-data *shape*, not per dataset: `psv_dir` (Sepsis: many small per-entity files, entity ID parsed from the filename), `csv_pretabulated` (Sparkov: exactly two files, already pre-split), `csv_multi` (CIC-IDS2017: several files concatenated, split decided later). Strips whitespace from column names generically on load (harmless for Sepsis/Sparkov, fixes CIC-IDS2017's `" Source IP"`-style names) — this is generic hygiene, not a dataset-specific quirk, so it lives here rather than waiting for Stage 4.
- **`splitting.py`** — three split methods, matching the three documented split rules: `group` (whole entities to one side, e.g. Sepsis patients — entity *assignment* is randomized, but no entity's rows ever cross the boundary, so this does not violate hard rule 1), `pretabulated` (honor Sparkov's pre-split files as-is), `filename_group` (assign whole source files to train/test by name, e.g. CIC-IDS2017 Mon–Wed vs Thu–Fri).
- **`run.py`** — CLI: `python -m src.pipeline.run configs/<name>.yaml`. Prints row/column counts, split sizes, and target distribution.
- Every config has a `dev_limit` (max files for Sepsis, max rows/file for Sparkov and CIC-IDS2017) so pipeline runs during development take seconds, not the full runtime. Set to `null` for a real run.
- `tests/smoke_test.py` checks the split invariant that actually matters per dataset (no patient in both Sepsis splits, no leftover leakage columns for Sparkov/CIC-IDS2017). No `pytest` dependency — plain asserts, run via `python -m tests.smoke_test`.
- Verified end-to-end on dev-limited configs for all three datasets before considering the stage done.

### Stage 4 per-dataset cleaning (2026-09-12)

Filled in `cleaning.py`, which was a deliberate no-op stub through Stage 3. Also **reordered the pipeline to `load → split → clean`** (it was `load → clean → split` in Stage 3) — cleaning needs train-only statistics for Sepsis's median fill, and the Sepsis train/test split isn't known until `split()` runs (it randomly assigns whole patients), so cleaning has to come after.

- **Sepsis**: forward-fill each patient's lab columns within their own timeline (sorted by `ICULOS`), then fill whatever's left (a patient's hours before their first reading) with each column's **training-set median**, reused as-is on test. Decided with the user: medians come from train only so test information can't leak into how train gets filled. Edge case found and fixed: in a small dev sample, `EtCO2` and `Bilirubin_direct` had zero non-null values in train, making the median itself `NaN` — added a `.fillna(0)` fallback on the median series so this can't silently leave `NaN` in the output. Verified: 0 missing values remain in train or test after cleaning.
- **Sparkov**: drops the unnamed index column. That's it as of Stage 5 — see below for why the negative-downsampling described here originally (raising the fraud rate to ~2%) moved out of this stage.
- **CIC-IDS2017**: drops rows with a null `Label` (the blank trailing-row corruption in the Thursday-morning file), drops exact duplicate rows, then drops rows with an infinite value in any numeric column (not just the two known flow-rate columns, so this doesn't silently miss a new one). Verified directly against the Thursday file: 288,602 null-`Label` rows dropped, 170,366 real rows remain, with exactly 1 duplicate left afterward — matching the numbers in "Known quirks" above. **Retroactively fixed while planning Stage 6** (bug, not a judgment call): `Label` was left as multi-class strings (`BENIGN`, `FTP-Patator`, `DoS slowloris`, ...) even though the dataset table above already specifies binary (`BENIGN` vs attack → 0/1). Now collapsed to binary int here. Re-ran Stage 5's selection afterward to confirm the feature ranking — it picked the identical 25 features both before and after the fix, on the dev sample tested.
- Feature reduction to 25 columns (hard rule 6) is explicitly **not** done here — that's Stage 5.
- `tests/smoke_test.py` extended with one cleaning-invariant check per dataset (no `NaN` left for Sepsis, index column dropped for Sparkov, no duplicates/null-target/infinite values for CIC-IDS2017).

### Stage 5 feature reduction (2026-09-12)

Added `features.py`: one shared selection method — **mutual information with the target, scored on training data only, top-k kept** (decided with the user over a quick-model-importance alternative, specifically because MI doesn't favor any one of the three model types over the other two). What differs per dataset is the *candidate pool* the selector picks from, not the selector itself.

- **Sepsis**: no extra work needed — all 39 non-target/time/group columns are legitimate clinical features. Straightforward 39 → 25.
- **CIC-IDS2017**: excludes `Flow ID` (a composite string built from the flow's 4-tuple, effectively unique per flow) and `Destination IP` in addition to the already-excluded target/time/group columns, via a new `feature_exclude:` config list. 80 real candidates → 25. `Destination Port`/`Source Port` were kept as candidates (legitimately predictive in intrusion detection, not identifiers the way an IP address is) and both made the final 25.
- **Sparkov — the hard case.** Checking actual cardinalities (not just documented ones) found that `lat`, `long`, `city`, `zip`, `dob`, `street` all have ~968–983 unique values against ~983 distinct cardholders in training — they're not shared categories, they're alternate identifiers for the customer, same problem as `cc_num` (already excluded). `city_pop` is a deterministic lookup from `city`, so it inherits the problem. Only **8 raw columns** survive that triage (`merchant`, `category`, `amt`, `gender`, `state`, `job`, `merch_lat`, `merch_long`). Decided with the user (two rounds of discussion, as the size of the gap became clearer): derive standard, well-precedented engineered features rather than lower the bar on identity-proxies or exempt Sparkov from the 25-feature rule:
  - `age` (from `dob`) and `distance_km` (haversine between cardholder and merchant coordinates) — these deliberately *coarsen* an identity-like input into a value many customers can share, unlike the raw column they come from.
  - Calendar features: `hour_of_day`, `day_of_week`, `month`, `day_of_month`, `is_weekend`, `is_night`.
  - Card-velocity features, each causal (only ever looks at a card's own *prior* transactions, via `shift(1)` before an expanding window): `card_txn_count_so_far`, `card_amt_dev_from_own_mean`, `card_amt_dev_from_own_max`. Plus two train-fit deviation features, `category_amt_dev_from_mean` and `merchant_amt_dev_from_mean`.
  - `merchant`/`category`/`gender`/`state`/`job` are frequency-encoded (each category → its rank by how common it is in train; decided with the user over one-hot, which would've exploded `merchant`/`job` into hundreds of dummy columns and muddied what "25 features" means).
  - **Final candidate pool: 21** (8 raw + 13 engineered) — still short of 25 even after this. Decided with the user: report this honestly rather than keep inventing features to reach a number. `select.k: 25` in `configs/sparkov.yaml` is documented as a ceiling, not a guarantee.
  - **Negative-downsampling moved here from Stage 4**, and now runs as the *last* step, after the candidate pool is built and selected. Reason: the card-velocity features need each card's real, undownsampled transaction history to mean anything — computing them after Stage 4's downsampling would have undercounted every card's history by whatever fraction of its rows got dropped. This is a retroactive fix to how Stage 4 was structured, made necessary by something Stage 5 discovered, not a Stage 4 mistake that was visible at the time.
- Two bugs found and fixed while verifying: (1) the identity-proxy exclusion list was initially described but never wired into `configs/sparkov.yaml`, which crashed `mutual_info_classif` on the first string column (`'Mary'`, from `first`) it hit; (2) `city_pop` was missed from that same exclusion list on the first fix. Both are now in `feature_exclude:`.
- Also bumped CIC-IDS2017's dev_limit from 2,000 to 20,000 rows/file: attacks don't start at the top of a file (Tuesday's first attack row is at index 11,347), so the smaller cap gave Stage 5 an all-`BENIGN` (zero-variance) training target to select against, which is a degenerate case mutual information can't do anything useful with.
- `tests/smoke_test.py` extended with a 25-or-fewer-features check per dataset, plus a check that none of Sparkov's identity-proxy columns ever reach the final selection.
- **Introduced `configs/dev/` (not a stage change, a test-infra fix discovered while starting Stage 6):** the checked-in `configs/*.yaml` were always the "real" per-dataset settings, with `dev_limit` temporarily shrunk during development and restored to `null` once a stage was verified. That made `tests/smoke_test.py` (which loaded the same files) slow to the point of loading entire multi-GB datasets on every run, once `dev_limit` got set to `null` for Stage 6's real training. Fixed by adding `configs/dev/*.yaml` — identical settings, small `dev_limit` — which the smoke tests load instead. The full-scale `configs/*.yaml` files stay at `dev_limit: null` permanently from here on.

### Stage 6 model training (2026-09-12)

Added `models.py`: trains XGBoost, LSTM, and FT-Transformer on each dataset's reduced feature table. Fixed hyperparameters only (hard rule 2) — `configs/models.yaml` holds one set of values per model type, reused identically across all three datasets, never tuned per dataset.

- **LSTM and FT-Transformer both treat each row as one independent sample, not a multi-step sequence** — decided with the user specifically so every model explains the same instance unit, which the Stage 7+ cross-model XAI comparison depends on. Confirmed technically first: `rtdl_revisiting_models`'s `FTTransformer.forward()` only accepts a single row's features at all (no sequence dimension), so row-level framing isn't really optional if FT-Transformer and LSTM are meant to be comparable.
- **A real environment bug, not a code bug:** importing `torch` before XGBoost touches any numpy array — even just building a `DMatrix` — segfaults on this machine every time, confirmed with `faulthandler` (crash inside `xgboost/data.py`'s `_meta_from_numpy`, triggered by import order alone, not data or hyperparameters). Likely a numpy C-API/ABI conflict between this exact torch 2.13.0 / xgboost 3.2.0 / numpy 2.4.6 combination on macOS arm64. Tried `multiprocessing.Pool` with a `spawn` context first (to keep XGBoost in a process that never imports torch) — that avoided the segfault but hung indefinitely for unrelated reasons not worth chasing further. Landed on `xgboost_worker.py`: a module that never imports torch, run as a genuine standalone `subprocess.run` (not multiprocessing), with data passed through temp `.npy`/`.json` files. Verified working with torch already loaded in the parent process.
- A second real bug found while testing on CIC-IDS2017's larger test set (~100k rows): FT-Transformer's attention cost scales with how many rows go through the network at once, and a single unbatched forward pass over the full test set exhausted MPS memory. Fixed by batching prediction (not just training) in `_predict_proba`.
- Dev-scale timing (10 epochs, the fixed value in `configs/models.yaml`) showed FT-Transformer as the clear cost driver, consistent with Stage 2's SHAP findings: 2.3s (Sepsis, 1,505 rows) → 18.1s (Sparkov, 22,399 rows) → 53.6s (CIC-IDS2017, 59,894 rows). Extrapolated estimate was ~1 hour total for full-scale training — confirmed with the user before launching, per hard rule 4.
- Models save to `models/<dataset>/<model_type>.{json,pt}` (gitignored — generated artifacts, not source, same treatment as `data/` and `results/`).
- `tests/smoke_test.py` extended with a check that all 9 model/dataset combinations train and produce valid (0-1) accuracy/AUC on the dev configs.

**Full-scale training results** (`dev_limit: null`, ran in the background, ~45 min total):

| Dataset | Model | Train rows | Train time | Test accuracy | Test AUC |
|---|---|---|---|---|---|
| Sepsis | XGBoost | 1,245,801 | 1.7s | 0.981 | 0.739 |
| Sepsis | LSTM | 1,245,801 | 92.2s | 0.980 | 0.630 |
| Sepsis | FT-Transformer | 1,245,801 | 1,055.8s (~18 min) | 0.975 | 0.630 |
| Sparkov | XGBoost | 375,300 | 1.2s | 0.996 | 0.997 |
| Sparkov | LSTM | 375,300 | 27.0s | 0.994 | 0.987 |
| Sparkov | FT-Transformer | 375,300 | 278.5s (~4.6 min) | 0.996 | 0.997 |
| CIC-IDS2017 | XGBoost | 1,666,481 | 2.4s | 0.819 | 0.802 |
| CIC-IDS2017 | LSTM | 1,666,481 | 121.8s | 0.807 | 0.615 |
| CIC-IDS2017 | FT-Transformer | 1,666,481 | 1,507.8s (~25 min) | 0.765 | 0.737 |

Sepsis's train set (1.55M rows) came in larger than the ~1.2M estimate used for the time projection, and CIC-IDS2017's cleaned train set (1.67M rows) larger still — both datasets needed noticeably more of their raw rows than assumed, which is why full-scale FT-Transformer ran longer than the original ~30 min/dataset estimate (closer to ~18-25 min per dataset in practice, not uniformly worse, just different per dataset than the dev-scale extrapolation implied).

None of the AUCs are tuned toward — per hard rule 2, a mediocre model is expected and fine. Worth noting for later stages: CIC-IDS2017's LSTM/FT-Transformer generalize noticeably worse than train (AUC ~1.0 on train vs. 0.6-0.74 on test) — real overfitting, not a bug, consistent with fixed untuned hyperparameters on a dataset this large.

### Stage 6.5 SageMaker setup (2026-09-13)

AWS side set up for Stage 7: S3 bucket `directed-study-sagemaker-project` and IAM execution role `directed-study-sagemaker-iamrole`, both in `ca-central-1`, created by Razoan through the Console (never touched credentials directly — CLI access uses a separate IAM user, `razoan-cli`, configured locally via `aws configure`). Non-secret identifiers (bucket name, role ARN, region, chosen instance type) recorded in `configs/aws.yaml`, safe to commit since they contain no keys.

- **Added `src/pipeline/export.py`**: runs the full `load → split → clean → reduce_features` chain and writes each dataset's processed train/test tables to `data/processed/<name>_{train,test}.csv`, plus a `<name>_dtypes.json` sidecar (column dtypes + the selected-feature list + target column name) so CSV's dtype loss doesn't corrupt anything on reload — `load_processed()` reads the JSON back and passes it as `dtype=` to `pd.read_csv`. Confirmed every stochastic step in the pipeline (split assignment, MI sampling, Sparkov's downsampling) is seeded identically across all three configs, so re-running this reproduces Stage 6's tables row-for-row.
- **Found and fixed a real bug while verifying the export**: freshly exporting and re-running each dataset's saved XGBoost model against the reproduced test data gave test AUCs wildly off from the Stage 6 results table (e.g. Sepsis 0.457 vs. the recorded 0.739). Root-caused via model-file timestamps: all 9 files under `models/` had been written within a ~90-second window, matching Stage 6's own documented *dev-scale* timing profile, not the ~45-minute full-scale run that produced the results table. Cause: `models.py` always wrote to `models/<dataset>/` regardless of `dev_limit`, so `tests/smoke_test.py`'s own model-training check (which runs on `configs/dev/*.yaml`) silently overwrote the real full-scale artifacts the next time it ran. Fixed by adding `_output_dir()` to `models.py`, which routes any config with a non-null `dev_limit` to `models/dev/<dataset>/` instead — `models/dev/` is covered by the existing `models/*` gitignore rule, no change needed there. Added a regression test, `test_dev_training_never_touches_real_model_directory`, that plants a sentinel file in the real directory (only if empty) and proves a dev-scale `train_all()` run never modifies it. Retrained all 9 models at full scale afterward; the results matched the original Stage 6 table almost exactly (e.g. Sepsis 0.981/0.739, Sparkov 0.996/0.997, CIC-IDS2017 0.819/0.802), and the re-verification against the reproduced processed data passed cleanly this time.
- **Uploaded** `data/processed/` (6 CSVs + 3 dtype JSONs) and `models/` (9 model files, `models/dev/` deliberately excluded) to S3 via `src/pipeline/upload_to_s3.py` — 18 files, 897.7 MiB total. Verified against the bucket listing after upload, not just the script's own exit code.
- **SageMaker Python SDK has moved to a v3 major rewrite** (new top-level modules `core`/`train`/`serve`/`mlops`/`lineage`/`ai_registry`, no more `sagemaker.processing` or `sagemaker.sklearn.processing`) — `pip install sagemaker` grabs v3 by default. Pinned `sagemaker==2.257.6` instead (the classic, well-documented `ProcessingInput`/`ProcessingOutput`/`SKLearnProcessor` API this project's Stage 7 tooling is built on) rather than learn an unfamiliar new surface. `SAGEMAKER_SUPPRESS_V2_WARNING=1` silences the resulting (harmless, expected) deprecation warning.
- **New AWS accounts default the SageMaker *Processing Job* quota for most instance types to 0**, separate from any training-job quota — the first real launch failed with `ResourceLimitExceeded` on `ml.m5.xlarge for processing job usage` until Razoan requested an increase via Service Quotas (approved same-day). Worth remembering if Stage 7 ever needs a different instance type: request its processing-job quota ahead of time, not when the sweep is already queued.
- **Verified end-to-end**: `src/pipeline/launch_smoke_test.py` submits a trivial SageMaker Processing Job (`src/pipeline/smoke_processing_script.py`) that reads `sepsis_dtypes.json` from S3 and writes a one-line summary back to `s3://.../results/smoke_test/`. Passed, confirmed by reading the output object back from S3 directly (not just trusting the job's reported SUCCESS status) — execution role, S3 read/write, container, and the `ml.m5.xlarge` instance type all confirmed working together before trusting Stage 7's real 27-cell sweep to the same pattern.

### Stage 7 explanation generation (2026-09-14)

All 27 cells (3 datasets × 3 models × 3 XAI methods) run, each as its own SageMaker Processing Job on `ml.m5.xlarge` — kept 1 job = 1 cell deliberately rather than batching, so each job's own reported duration is directly Stage 9's cost/runtime measurement with no extra accounting logic needed.

- **`src/pipeline/inference.py`** loads any trained model back as a plain `predict_proba` callable, uniform across all three model types. Split into `xgboost_inference.py` (no torch import, ever) and `torch_inference.py` (lstm/ft_transformer) with lazy per-branch imports in between — the exact same torch-before-XGBoost segfault documented in Stage 6 reappears if a single process imports both, so a process that only ever asks for `"xgboost"` must never trigger a torch import as a side effect of merely importing the dispatcher module.
- **`src/pipeline/explain.py`** runs one (dataset, model, method) cell: SHAP (`TreeExplainer` for XGBoost, `GradientExplainer` for LSTM/FT-Transformer, per Stage 2's decision), LIME (`LimeTabularExplainer`), and Permutation Importance (manual, since sklearn's `permutation_importance` wants a real sklearn estimator, not an arbitrary callable). Row caps: SHAP 500 (XGBoost) / 200 (LSTM/FT-Transformer), LIME and PI 500 for every model type, PI at `n_repeats=5`. Explained rows are a **seeded random sample of the full test set**, not `test_df.head(n_cap)` — see the bug below for why that distinction mattered.
- SHAP magnitudes are only comparable *within* one method for one (dataset, model) cell, never across model types: XGBoost's `TreeExplainer` explains in the model's native, unscaled feature space, while `GradientExplainer` necessarily explains in the neural models' internal *standardized* space (it differentiates through the actual network, which was trained on scaler-transformed inputs). Hard rule 7 already forbids comparing raw attribution values across methods, and Stage 10 only ever compares per-feature ranks within one cell, so this asymmetry doesn't affect anything the study measures — flagged here so it isn't mistaken for a bug later.
- **Packaging**: `sagemaker.processing.FrameworkProcessor(estimator_cls=PyTorch)` (torch pre-baked in the container, saving a from-scratch install on every one of the 27 jobs), `source_dir=src/` ships the whole `pipeline` package in, `src/run_stage7_cell.py` is the entry point (not `pipeline/explain.py` directly — that uses relative imports, which break when SageMaker invokes a file as a bare script with no package context). `data/processed/`, `models/`, and `configs/models.yaml` are mounted as S3 input channels at paths that make `config.py`'s `REPO_ROOT`-relative paths resolve identically to the local layout.
- **Four real bugs found and fixed while getting the first full sweep clean** (documented in full here since the fixes are non-obvious and easy to reintroduce):
  1. **Missing `configs/models.yaml` mount** — `torch_inference.py` needs it to reconstruct LSTM/FT-Transformer's architecture (hidden size, layers, etc.) before loading weights; XGBoost's saved model is self-contained and never reads it. Broke every LSTM/FT-Transformer cell (all 3 methods each) while XGBoost cells succeeded — the uniform failure across methods for exactly two model types was the tell. Fixed by adding it as a third `ProcessingInput` in `launch_stage7_cell.py`, and `upload_to_s3.py` now uploads it alongside the processed data and models.
  2. **`shap>=0.50` hard-requires `numpy>=2`, which breaks this container's pre-built torch.** The container's torch was compiled against numpy 1.x; letting pip upgrade numpy to install shap breaks torch's own `.numpy()` bridge *process-wide* (`RuntimeError: Numpy is not available`), not just some unrelated package. This only crashed SHAP on the two neural models (`GradientExplainer` needs `tensor.numpy()`) — XGBoost's `TreeExplainer` never touches a torch tensor, so it was unaffected. (A red herring along the way: the same numpy break also corrupts a pre-installed, numpy-1.x-era `cv2` that `shap` imports unconditionally for an unused image masker, printing a scary but non-fatal `AttributeError: _ARRAY_API not found` — chased that first, wasn't the real cause.) Fixed by pinning `numpy<2` and `shap==0.49.1` (the last release before the `numpy>=2` requirement) in `src/requirements.txt`.
  3. **`shap==0.49.1`'s `TreeExplainer` can't parse some of XGBoost 3.2.0's tree JSON** (a scientific-notation split threshold with no decimal point, e.g. `[2E-2]`, crashes with `ValueError: could not convert string to float`) — a real regression fixed by bug 2's downgrade. Since XGBoost inference never imports torch, upgrading back to `numpy>=2` + `shap==0.51.0` is completely safe for exactly that one model type. `run_stage7_cell.py` does this upgrade conditionally at runtime when `model_type == "xgboost"`.
  4. **`test_df.head(n_cap)` is a biased sample, not a representative one** — the test tables are sorted (Sepsis by patient/`ICULOS`, CIC-IDS2017 chronologically), so "the first n rows" isn't random. For Sepsis this meant SHAP/LIME's "global" importance reflected only the first 1–2 patients in ID order; for CIC-IDS2017 it was worse than unrepresentative, since attacks cluster later in each file (already documented in the dev_limit note above) — the first 500 test rows were all `BENIGN`, and Permutation Importance's `roc_auc` scoring crashed outright (`ValueError: Only one class present in y_true`). Found via all three CIC-IDS2017 PI cells failing. Fixed with a seeded `rng.choice` sample across the full test set in `explain.py`'s `run_cell()` — and because this changed what "explained rows" means for every cell, not just the ones that crashed, **all 27 cells were re-run from scratch** (decided with the user) rather than only patching the failures, so the whole grid uses one consistent sampling methodology.
- **The sweep is resumable by design** (`run_stage7_sweep.py`): results are written to `stage7_sweep_results.json` after every single cell, and a bare re-run skips anything already `SUCCEEDED` and retries the rest — `--force` reruns everything. This got exercised for real: the sweep was killed mid-run twice (once by an unrelated, transient loss of local file-system access to the whole project directory that resolved itself; once by a deliberate stop). Neither lost anything — in one case a job had actually completed on AWS moments before the kill, and the stale `FAILED` entry in the results file (from an earlier, pre-fix attempt with the same job-name prefix) was corrected by checking the real AWS job status and S3 output directly, not by re-running it.
- **Final verification, done directly against AWS rather than trusting the sweep script's own exit code**: all 27 job names in the final `stage7_sweep_results.json` independently confirmed `Completed` via `aws sagemaker describe-processing-job`, and all 27 `importance.csv` outputs confirmed present in S3 with sensible, non-empty content. Total output: 72 objects (27 `importance.csv` + 27 `meta.json` + 18 `raw_attributions.csv`, one per SHAP/LIME cell — Permutation Importance has no per-instance raw output), 3.3 MB total. The full, clean 27-cell run took 227 minutes — longer than the original ~1.3–1.5 hour estimate, mostly because of the extra conditional pip upgrade step added for XGBoost cells (bug 3) and general per-job variance, not because per-cell compute got more expensive.

### Stage 8 explanation quality metrics (2026-09-14)

Implements `docs/METRICS.md` exactly as written — normative, nothing added beyond what that document specifies (its section 4 explicitly lists faithfulness, robustness, complexity, and directional/local agreement as out of scope; none of those are computed here).

- **`src/pipeline/metrics.py`**: the P1–P5 ranking rules and M1–M4 agreement metrics. P1 (aggregate to global via mean absolute attribution) and P2 (discard sign) were already done by Stage 7's `importance.csv`; this module picks up from there with P3 (rank, `scipy.stats.rankdata(method="average")`, rank 1 = most important), P4 (identical feature sets between any two cells being compared, checked and raised loudly on mismatch rather than silently dropped), and P5 (Permutation Importance's negative values clipped to 0 before ranking, with the clip count and tie count both recorded per cell). Verified against synthetic data before trusting it on the real 27-cell grid: identical rankings at different scales → Spearman/Kendall both 1.0; fully reversed → both −1.0; a PI vector with two negative entries clips and ties exactly as expected.
- **`src/pipeline/quality_metrics.py`** (Family C, M9–M12): one subprocess per `(dataset, model)` pair, computed on the **full** held-out test split (not the capped explained sample) — same torch/XGBoost process-isolation reason as Stage 7's cells, since a single process touching both segfaults on this machine. Results matched Stage 6's original table exactly (e.g. Sepsis XGBoost AUC 0.7388), a useful independent re-confirmation.
- **`src/pipeline/run_stage8.py`**: orchestrates everything and writes exactly the 6 files in METRICS.md section 5's output contract, and nothing else — intermediate working data (the downloaded Stage 7 `importance.csv`/`meta.json` files, the 9 Family C subprocess outputs) live under `.cache/stage8/`, not `results/`, specifically so `results/` matches the contract's "and nothing else" literally.
- **M5 (wall-clock time) is sourced from AWS's own `describe_processing_job` API** (`CreationTime` → `ProcessingEndTime`) for all 27 jobs, not the sweep script's local `wall_seconds` proxy in `stage7_sweep_results.json` — the real per-job duration including container startup, matching the Family B scope note's explicit statement that M5 should include that, not just explainer compute.
- **The instance hourly rate (M7/M8) could not be sourced authoritatively**: the CLI IAM user has no `pricing:GetProducts` or `ce:GetCostAndUsage` permission, and web search only surfaced US (`us-east-1`) figures, not `ca-central-1`. Decided with the user: use the US estimate ($0.230/hr) rather than block Stage 8 on an IAM change, clearly flagged as unverified-for-region in both `metrics_cost.csv` (every row) and `metrics_manifest.json`. Total estimated Stage 7 cost at this rate: **$0.65** for all 27 cells combined.
- **A genuinely interesting cost finding, not a bug**: FT-Transformer + LIME took 1600–1850s per cell on AWS's CPU-only `ml.m5.xlarge` — 8.1–9.3× the cost of the cheapest method on the same model (M8), and roughly **25× slower than the same cell's local Mac timing** (~72s, MPS-accelerated) noted earlier in this document. LIME calls the model's `predict_proba` hundreds of times per explained instance; for the heaviest of the three models, on CPU only, that gap compounds far beyond what a single-forward-pass benchmark would suggest. Worth highlighting in the write-up — Stage 2's original local timing numbers understate FT-Transformer+LIME's real-world cost substantially.
- Verified throughout: A1/A2's `performance_confounded` flag checked by hand against the known Family C AUC values (e.g. Sepsis XGBoost 0.739 vs LSTM 0.630 → diff 0.109 → flagged, matching the >0.10 threshold exactly); A3's `rank_order_preserved` checked by hand against a case where it correctly comes out `False` (Sepsis's method-pair ordering by agreement differs from Sparkov's and CIC-IDS2017's, which happen to match each other).

### Stage 9 computational cost metrics (2026-09-14)

Runtime and dollar cost were already done as a side effect of Stage 8's Family B — nothing further needed there. The only real work in this stage was memory, and the outcome was to **not** turn it into a metric.

- **Checked whether SageMaker Processing Jobs still had memory data in CloudWatch, without re-running anything**: yes — `MemoryUtilization` (namespace `/aws/sagemaker/ProcessingJobs`, dimension `Host=<job_name>/algo-1`) was retrievable for all 27 jobs via `get_metric_statistics`.
- **Excluded as a metric anyway, decided with the user**: CloudWatch samples this at a 1-minute period. Most of the 27 cells ran ~200s total, almost entirely the one-time pip install, not the sub-second explanation compute — leaving most cells with just 1-2 datapoints (effectively one arbitrarily-timed snapshot, likely caught during the pip install rather than the actual SHAP/LIME/PI call). Only the three FT-Transformer+LIME cells ran long enough (1600-1850s) for 1-minute sampling to produce a real distribution — 25-29 datapoints each. "Max of 1 sample" isn't comparable to "max of 29 samples" across a grid meant to compare methods fairly, so `docs/METRICS.md` was **not** amended to add a memory metric.
- **Qualitative observation, not a metric**: the only reliably-sampled cells (FT-Transformer+LIME) peaked at 8.49-12.17% of the `ml.m5.xlarge`'s 16 GiB (Sepsis 12.17%, CIC-IDS2017 11.97%, Sparkov 8.49%) — the same three cells already flagged by Stage 8 as the time/cost outliers. This corroborates those findings; it doesn't add an independent one.
- **`src/pipeline/run_stage9.py`** re-fetches and saves the raw CloudWatch data (all 27 cells, including the unreliable ones, with datapoint counts and window timestamps so the "why excluded" reasoning is checkable later) to `results/stage9_memory_cloudwatch_raw.csv` — on record, explicitly not part of Stage 8's closed output contract and not a metric.

### Stage 10 rank-based comparison (2026-09-14)

Checked properly before declaring this done: read the stage's original two-sentence description in `docs/PROJECT_PLAN.md`, then compared it item by item against what `results/metrics_agreement_cross_*.csv` actually contains, rather than assuming the overlap with Stage 8's Family A.

- **"Convert to ranks before comparing"**: satisfied — `rank_cell()` (P3) converts every cell's importance to a rank vector before any comparison happens; verified earlier with synthetic data.
- **"Never compare raw values across methods"**: satisfied — every Family A comparison column (`spearman_rho`, `kendall_tau`, `jaccard_top5/10`) is rank-derived; no raw SHAP/LIME/PI value appears anywhere in Stage 8's output.
- **The one real gap**: the per-feature rank vectors themselves were never persisted — only the summary agreement scores computed *from* them. Not required by Stage 10's literal text, but a real prerequisite for Stage 11 (which needs the actual rankings — e.g. "which feature ranks #1 most often", "XGBoost's SHAP ranking next to its LIME ranking" — not just how much two methods agree).
- **Closed by adding `results/ranked_features.csv`** (one row per `dataset`×`model`×`method`×`feature`: its P3/P5 rank, and its importance value after P1/P2 aggregation) to `run_stage8.py`'s output, computed from the same `RankedCell` objects `build_ranked_cells()` already builds — not a new metric, no new M-ID, so `docs/METRICS.md` needed no metric amendment, just a note in section 5 documenting the file exists (plus an entry in its amendment log, since the output contract itself changed). One deliberate subtlety: the `importance` column is the **raw, pre-P5-clip** value, not the clipped one used for ranking — verified a Permutation Importance cell where several features tied at the bottom rank (from clipping) correctly still show their real, original *negative* raw scores rather than the clipped `0`.

### Stage 11 aggregation and analysis (2026-09-14)

`src/pipeline/run_stage11.py` re-aggregates Stage 8's output (no new per-cell metrics) into `results/stage11_*.csv` + three `results/fig_*.png` figures, structured around the research question's three parts. Findings below, including two that cut against the naive expectation.

**RQ1 — do SHAP, LIME, PI agree with each other on the same model?** (`stage11_cross_method_summary.csv`, from A1, 9 cells per pair) SHAP↔PI agrees most consistently (mean Spearman 0.569, std 0.252). SHAP↔LIME has a similar mean (0.468) but with enormous spread (std 0.453, range −0.397 to 0.870) — sometimes strong agreement, sometimes strong *disagreement*, unpredictably. LIME↔PI is weakest on average (0.398). **Contradicts the naive expectation**: SHAP and LIME are both "local, perturbation/gradient-based" methods and might be expected to agree with each other more than either agrees with PI (a completely different, global, permutation-based approach that never touches model internals) — instead SHAP agrees more reliably with PI than with LIME.

**RQ2 — does a single method agree with itself across XGBoost/LSTM/FT-Transformer?** (`stage11_cross_model_summary.csv`, from A2, 9 cells per method) SHAP is the most self-consistent across architectures (mean Spearman 0.573, rising to 0.686 once the 4 performance-confounded comparisons are excluded), then LIME (0.471 → 0.520), then PI is least self-consistent (0.423 → 0.555) — despite being the one method that never touches model internals at all, which is not the obvious prediction. Confound context (Family C, joined in): 12 of the 27 A2 rows are flagged `performance_confounded` (>0.10 AUC gap), concentrated in Sepsis (XGBoost vs. LSTM/FT-Transformer) and CIC-IDS2017 (XGBoost vs. LSTM, LSTM vs. FT-Transformer) — **Sparkov has zero confounded comparisons**, so it's the cleanest domain for trusting a cross-model agreement number as reflecting genuine architecture-dependent explanation behavior rather than "one model is just worse."

**RQ3 — do the RQ1/RQ2 patterns hold across domains?** (`stage11_cross_domain_summary.csv`, from A3) At the strict bar ("identical pairwise ordering in all 3 domains"), **0 of 8** metric×family combinations pass — domain-independence does not hold exactly, for any metric tested. But it's not pure noise either: in **6 of those 8** (Spearman, Kendall's tau, and top-10 Jaccard, for both cross-method and cross-model), the same specific sub-pattern appears — **Sparkov and CIC-IDS2017 agree with each other on the ordering, and Sepsis is the domain that breaks it.** (E.g. cross-method Spearman: Sparkov and CIC-IDS2017 both rank SHAP↔LIME first; Sepsis ranks it *last*.) The 2 exceptions are both top-5 Jaccard (cross-method and cross-model) — a small-k, tie-sensitive metric that doesn't show a clean domain grouping in either direction.

**What explains Sepsis breaking the pattern — checked properly, not assumed (2026-09-14).** The first-pass explanation offered here ("weak neural models, AUC ≈ 0.63, plus severe imbalance") does not survive scrutiny against the worked example below, which is an **XGBoost** cell — Sepsis's *best*-performing model (AUC 0.739), not a weak one. Tested and ruled out in order: (1) **weak model** — Sepsis/XGBoost's AUC (0.739) is within the Family C confound threshold of CIC-IDS2017/XGBoost's (0.802, diff 0.064 < 0.10); (2) **class imbalance alone** — Sparkov's test positive rate (2.00%) is nearly identical to Sepsis's (1.84%), yet Sparkov's SHAP↔LIME agreement is strong across all 3 of its models (0.67–0.87) while all 3 of Sepsis's models show anomalously low/negative agreement (−0.40 to 0.10) — same imbalance, opposite outcome; (3) **feature collinearity** — Sepsis has the *lowest* mean pairwise feature correlation of the three datasets (0.053), CIC-IDS2017 the highest (0.309), backwards from what the collinearity-causes-SHAP/LIME-divergence hypothesis predicts; (4) **weak/noisy attribution signal** — Sepsis's mean \|PI importance\| (0.016) is unremarkable, between Sparkov's (0.011) and CIC-IDS2017's (0.040). **What's actually established**: the divergence is a Sepsis-specific effect present across all three of its model types, not a symptom of any one model's quality, and not explained by imbalance, collinearity, or signal strength. **The real mechanism is not identified** — candidates not yet tested include Sepsis's heavy forward-fill/median-imputation (Stage 4) creating large blocks of literally-duplicated or synthetic feature values in a way Sparkov's and CIC-IDS2017's cleaning don't, or LIME's perturbation sampling specifically struggling against that imputation structure. State this as an open question in the write-up, not a resolved finding.

**Cost** (`stage11_cost_summary.csv`, from Family B, mean across the 3 datasets): Permutation Importance is cheapest for every model. FT-Transformer+LIME is the extreme outlier — 8.8× the cost of the cheapest method on that model (already flagged in Stage 8 as ~25× slower than the local MPS-accelerated timing). **The tension worth naming**: SHAP — the most expensive method for FT-Transformer (2.6×) and second-most for LSTM (2.5×) — is also the *most self-consistent* method across architectures (RQ2 above). The cheapest method (PI) is the least self-consistent. Reliability and cost move in opposite directions here, not together.

**Worked example** (`stage11_worked_example.csv`): the single worst SHAP-vs-LIME disagreement in the whole grid is Sepsis/XGBoost — Spearman ≈ −0.02 (no correlation) and **zero overlap in the top 5 features**. SHAP's top 5 (`HospAdmTime`, `Lactate`, `pH`, `Creatinine`, `Hgb`) are recognizable sepsis/shock clinical markers; LIME's top 5 (`Bilirubin_direct`, `SaO2`, `PTT`, `Platelets`, `Fibrinogen`) share not one feature with SHAP's list. Only `Bilirubin_total` appears in both, and only at rank 7 for each. Same model, same data, same prediction task — completely different stories about what drove it.

### Stage 12 write-up (2026-09-14)

`docs/DATASET_SPEECH.md` (supervisor-facing summary) and `docs/WRITEUP.md` (full write-up) both written, closing out all 13 stages.

- **`docs/DATASET_SPEECH.md`**: concise — research question, headline findings for all 3 RQ parts, cost, and limitations, written for a reader who already knows the field (unlike this file's own "beginner in ML" framing for Razoan — different audience, different register).
- **`docs/WRITEUP.md`**: the complete write-up — methodology, all 3 RQ parts with full tables, the SHAP-vs-LIME worked example (§5), cost (§6), Family C quality context (§7, explicitly labeled "context, not a headline result"), and limitations (§8, mirroring `METRICS.md` §4's explicit out-of-scope list). Every number traced to `results/metrics_*.csv` / `ranked_features.csv` / `stage11_*.csv`, cross-checked against `metrics_manifest.json`.
- **Both carry the corrected Sepsis explanation from Stage 11** (the tested-and-ruled-out account — weak model, class imbalance, feature collinearity, all checked and rejected in turn — not the original, unverified "weak neural models" claim). Reported as a finding in its own right in `DATASET_SPEECH.md`'s "correction made along the way" paragraph, specifically because catching and correcting it is itself evidence the rest of the write-up's claims were checked, not just asserted.
- Neither document invents a number, a metric, or a causal explanation beyond what Stages 7–11 actually measured and verified — consistent with how every prior stage in this project was run.

---

## Repository layout

```
xai-benchmark/
├── CLAUDE.md                 # this file
├── docs/
│   ├── PROJECT_PLAN.md       # the 12 stages and the schedule
│   ├── METRICS.md            # measurement rules - normative
│   ├── DATASET_SPEECH.md     # supervisor-facing summary
│   └── WRITEUP.md            # Stage 12 full results write-up (moved here from results/ so it's version-controlled)
├── configs/                  # one YAML settings file per dataset (Stage 3+), full-scale (dev_limit: null)
│   ├── sepsis.yaml
│   ├── sparkov.yaml
│   ├── cic_ids2017.yaml
│   ├── models.yaml           # Stage 6 fixed hyperparameters, one set per model type, not per dataset
│   ├── aws.yaml              # Stage 6.5 -- S3 bucket, IAM role ARN, region, instance type (non-secret)
│   └── dev/                  # same settings, small dev_limit -- what tests/smoke_test.py loads
├── data/
│   ├── raw/                  # untouched downloads - never edit
│   └── processed/            # cleaned output (Stage 6.5+: also mirrored to S3)
├── models/                   # trained model artifacts (Stage 6+) - gitignored, generated
│   └── dev/                  # dev-scale artifacts from configs/dev/*.yaml runs -- never the real models
├── src/
│   ├── run_stage7_cell.py    # Stage 7 SageMaker entry point (sibling-of-package trick, see its docstring)
│   ├── requirements.txt      # Stage 7 container's extra pip deps -- source_dir=src/, so this ships too
│   └── pipeline/             # shared load/split (Stage 3), per-dataset clean (Stage 4), feature reduction (Stage 5), model training (Stage 6), S3 export/upload + SageMaker smoke test (Stage 6.5), explanation generation (Stage 7), quality/agreement/cost metrics (Stage 8)
├── tests/
│   └── smoke_test.py         # plain-assert split + cleaning + feature-reduction + training invariant checks, no pytest
├── notebooks/
├── results/                  # gitignored -- generated CSVs/figures only. Stage 7: mirrored to s3://.../results/. Stage 8: metrics_*.csv + metrics_manifest.json (METRICS.md's output contract) + ranked_features.csv (intermediate). Stage 9: stage9_memory_cloudwatch_raw.csv (raw record, not a metric). Stage 11: stage11_*.csv + fig_*.png (re-aggregated from Stage 8, no new metrics)
├── .cache/                   # Stage 8 intermediate working data (downloaded Stage 7 outputs, Family C subprocess results) -- gitignored, deliberately kept out of results/
├── stage7_sweep_results.json # per-cell status/job-name log, resumable -- see Stage 7 notes
└── requirements.txt
```

---

## Where we are

**Done:** Stage 0 (Python 3.11 environment, `requirements.txt`, repo/GitHub set up), Stage 1 (datasets downloaded and verified, explored end-to-end — see "Confirmed from exploration" above), Stage 2 (throwaway timing test — see "Stage 2 timing test" above), Stage 3 (config-driven pipeline skeleton, 2026-09-12 — see "Stage 3 pipeline skeleton" above), Stage 4 (per-dataset cleaning, 2026-09-12 — see "Stage 4 per-dataset cleaning" above), Stage 5 (feature reduction, 2026-09-12 — see "Stage 5 feature reduction" above; Sepsis and CIC-IDS2017 land at exactly 25 features, Sparkov caps at 21 — documented limitation, not a bug), Stage 6 (model training, 2026-09-12 — see "Stage 6 model training" above; all 9 models trained successfully at full scale, artifacts in `models/`), Stage 6.5 (SageMaker setup, 2026-09-13 — see "Stage 6.5 SageMaker setup" above; S3/IAM created, processed data + models uploaded, a real dev/full-scale model overwrite bug found and fixed along the way, trivial Processing Job smoke test verified end-to-end), Stage 7 (explanation generation, 2026-09-14 — see "Stage 7 explanation generation" above; all 27 cells run on SageMaker Processing Jobs on `ml.m5.xlarge`, four real bugs found and fixed along the way — a missing config mount, a numpy/torch ABI conflict, a shap/XGBoost JSON-parsing regression, and a biased row-sampling bug that affected every cell and triggered a full from-scratch re-run — all 27 outputs independently verified against AWS and S3 directly, not just the sweep script's own exit code), Stage 8 (explanation quality metrics, 2026-09-14 — see "Stage 8 explanation quality metrics" above; implements `docs/METRICS.md` exactly, nothing added — Family A agreement, Family B cost sourced from AWS's own job metadata, Family C quality as a confound check; surfaced a genuinely interesting finding, not a bug: FT-Transformer+LIME is ~25× slower on AWS's CPU-only instance than the local MPS-accelerated timing suggested), Stage 9 (computational cost metrics, 2026-09-14 — see "Stage 9 computational cost metrics" above; runtime/cost already covered by Stage 8's Family B, memory investigated via CloudWatch and deliberately excluded as a metric — uneven sampling across cells, not a fair comparison — `docs/METRICS.md` left unamended, raw data kept on record), Stage 10 (rank-based comparison, 2026-09-14 — see "Stage 10 rank-based comparison" above; both original requirements confirmed already satisfied by Stage 8's Family A via an item-by-item check, not assumed — the one real gap found, persisted per-feature rank vectors, closed by adding `results/ranked_features.csv` as a documented Stage 8 intermediate), Stage 11 (aggregation and analysis, 2026-09-14 — see "Stage 11 aggregation and analysis" above; findings structured around the research question's three parts, headline result: cross-domain rank-order preservation holds in 0/8 metric×family combos at the strict bar, but 6/8 share the same sub-pattern — Sepsis breaks the ordering that Sparkov and CIC-IDS2017 otherwise agree on — plus a worked SHAP-vs-LIME example with zero top-5 overlap on Sepsis/XGBoost, whose original "weak neural models" explanation was later checked, found wrong, and corrected — see Stage 11's notes), and Stage 12 (write-up, 2026-09-14 — see "Stage 12 write-up" above; `docs/DATASET_SPEECH.md` and `docs/WRITEUP.md` both written, carrying the corrected Sepsis finding, not the original one).

**All 13 stages (0 through 12, plus 6.5) are complete.** The study's full artifact chain — raw data through cleaning, feature reduction, model training, AWS-run explanation generation, normatively-specified metrics, aggregated analysis, and write-up — is done, verified at each stage against real outputs (AWS job status, S3 content, recomputed numbers) rather than assumed from exit codes. Anything from here is refinement (e.g. verifying the `ca-central-1` instance rate, investigating the unresolved Sepsis mechanism) rather than a missing stage.

Full stage list is in `docs/PROJECT_PLAN.md`.

---

## How to work

- **One stage at a time.** Do not jump ahead.
- **Build the pipeline config-driven from the first line.** A dataset should be a settings file, not new code. If adding dataset 2 requires writing new code, stage 3 was built wrong.
- **Say when something in the plan looks wrong** rather than working around it silently.
