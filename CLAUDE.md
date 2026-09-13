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

---

## Repository layout

```
xai-benchmark/
├── CLAUDE.md                 # this file
├── docs/
│   ├── PROJECT_PLAN.md       # the 12 stages and the schedule
│   ├── METRICS.md            # measurement rules - normative
│   └── DATASET_SPEECH.md     # supervisor-facing summary
├── configs/                  # one YAML settings file per dataset (Stage 3+), full-scale (dev_limit: null)
│   ├── sepsis.yaml
│   ├── sparkov.yaml
│   ├── cic_ids2017.yaml
│   ├── models.yaml           # Stage 6 fixed hyperparameters, one set per model type, not per dataset
│   └── dev/                  # same settings, small dev_limit -- what tests/smoke_test.py loads
├── data/
│   ├── raw/                  # untouched downloads - never edit
│   └── processed/            # cleaned output
├── models/                   # trained model artifacts (Stage 6+) - gitignored, generated
├── src/
│   └── pipeline/             # shared load/split (Stage 3), per-dataset clean (Stage 4), feature reduction (Stage 5), model training (Stage 6)
├── tests/
│   └── smoke_test.py         # plain-assert split + cleaning + feature-reduction + training invariant checks, no pytest
├── notebooks/
├── results/
└── requirements.txt
```

---

## Where we are

**Done:** Stage 0 (Python 3.11 environment, `requirements.txt`, repo/GitHub set up), Stage 1 (datasets downloaded and verified, explored end-to-end — see "Confirmed from exploration" above), Stage 2 (throwaway timing test — see "Stage 2 timing test" above), Stage 3 (config-driven pipeline skeleton, 2026-09-12 — see "Stage 3 pipeline skeleton" above), Stage 4 (per-dataset cleaning, 2026-09-12 — see "Stage 4 per-dataset cleaning" above), Stage 5 (feature reduction, 2026-09-12 — see "Stage 5 feature reduction" above; Sepsis and CIC-IDS2017 land at exactly 25 features, Sparkov caps at 21 — documented limitation, not a bug), and Stage 6 (model training, 2026-09-12 — see "Stage 6 model training" above; all 9 models trained successfully at full scale, artifacts in `models/`). The 27-experiment grid is confirmed feasible; SHAP on LSTM/FT-Transformer is the dominant cost.

**Next:** Stage 6.5 (AWS SageMaker setup — see `docs/PROJECT_PLAN.md`, added 2026-09-12), then Stage 7 (explanation generation, now run entirely on SageMaker Processing Jobs on one instance type, capped at 200–500 explained rows per hard rule 3).

Full stage list is in `docs/PROJECT_PLAN.md`.

---

## How to work

- **One stage at a time.** Do not jump ahead.
- **Build the pipeline config-driven from the first line.** A dataset should be a settings file, not new code. If adding dataset 2 requires writing new code, stage 3 was built wrong.
- **Say when something in the plan looks wrong** rather than working around it silently.
