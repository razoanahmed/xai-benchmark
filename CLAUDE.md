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

### Known quirks

- **CIC-IDS2017 column names have leading spaces.** `" Source IP"`, `" Timestamp"`. Strip whitespace from all column names on load or lookups will fail.
- **Sepsis files are pipe-separated**, not comma. Use `sep='|'`. The patient ID is in the filename, not inside the file.
- **Sparkov has an unnamed index column** as the first column. Drop it.
- **CIC-IDS2017 has known duplicate rows and some infinite values** in the flow columns. Handle and document both.

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

**Done:** Stage 1 — all three datasets downloaded and verified.

**Next:** Stage 0 (environment), then Stage 2 (a throwaway timing test), then Stage 3 (the real pipeline).

Full stage list is in `docs/PROJECT_PLAN.md`.

---

## How to work

- **One stage at a time.** Do not jump ahead.
- **Build the pipeline config-driven from the first line.** A dataset should be a settings file, not new code. If adding dataset 2 requires writing new code, stage 3 was built wrong.
- **Say when something in the plan looks wrong** rather than working around it silently.
