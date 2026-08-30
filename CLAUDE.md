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

---

## Repository layout

```
xai-benchmark/
├── CLAUDE.md                 # this file
├── docs/
│   ├── PROJECT_PLAN.md       # the 12 stages and the schedule
│   ├── METRICS.md            # measurement rules - normative
│   └── DATASET_SPEECH.md     # supervisor-facing summary
├── data/
│   ├── raw/                  # untouched downloads - never edit
│   └── processed/            # cleaned output
├── src/
├── notebooks/
├── results/
└── requirements.txt
```

---

## Where we are

**Done:** Stage 0 (Python 3.11 environment, `requirements.txt`, repo/GitHub set up), Stage 1 (datasets downloaded and verified, explored end-to-end — see "Confirmed from exploration" above), and Stage 2 (throwaway timing test — see "Stage 2 timing test" above). The 27-experiment grid is confirmed feasible; SHAP on LSTM/FT-Transformer is the dominant cost.

**Next:** Stage 3 (the real config-driven pipeline).

Full stage list is in `docs/PROJECT_PLAN.md`.

---

## How to work

- **One stage at a time.** Do not jump ahead.
- **Build the pipeline config-driven from the first line.** A dataset should be a settings file, not new code. If adding dataset 2 requires writing new code, stage 3 was built wrong.
- **Say when something in the plan looks wrong** rather than working around it silently.
