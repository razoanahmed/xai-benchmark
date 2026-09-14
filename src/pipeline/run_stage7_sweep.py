"""Stage 7: launch all 27 (dataset, model, XAI method) cells sequentially,
each as its own SageMaker Processing Job -- one job per cell, so each job's
own reported duration is directly the cost/runtime measurement Stage 9
needs (per docs/PROJECT_PLAN.md: "Runtime comes for free once Stage 7 runs
on SageMaker -- each Processing Job reports its own duration"). Batching
multiple cells into one job would break that clean 1:1 accounting, so this
stays sequential rather than parallel -- the account's SageMaker Processing
quota for ml.m5.xlarge is also only 1 concurrent instance (see CLAUDE.md's
Stage 6.5 notes), which would block real parallelism anyway.

Hard rule (PROJECT_PLAN.md Stage 7): every cell runs on the same instance
type (configs/aws.yaml's ml.m5.xlarge) -- launch_cell() reads that config
itself, so there's no per-cell override here to accidentally get wrong.

Resumable by design: results are written to stage7_sweep_results.json
after every single cell, keyed by "dataset/model_type/method", not just at
the end -- a crash partway through loses at most one in-flight cell.
Re-running this script skips any cell already marked SUCCEEDED and retries
everything else (FAILED or never attempted), so "re-run only the failed
cells" is just "run this again" -- no separate retry mode to keep in sync.
Pass --force to ignore existing results and rerun all 27 from scratch.

Usage:
    python -m src.pipeline.run_stage7_sweep
    python -m src.pipeline.run_stage7_sweep --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import REPO_ROOT
from .launch_stage7_cell import launch_cell

DATASETS = ["sepsis", "sparkov", "cic-ids2017"]
MODEL_TYPES = ["xgboost", "lstm", "ft_transformer"]
METHODS = ["shap", "lime", "permutation_importance"]

RESULTS_PATH = REPO_ROOT / "stage7_sweep_results.json"


def cell_key(dataset: str, model_type: str, method: str) -> str:
    return f"{dataset}/{model_type}/{method}"


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Ignore existing results and rerun all 27 cells from scratch")
    args = parser.parse_args()

    cells = [(d, m, x) for d in DATASETS for m in MODEL_TYPES for x in METHODS]
    results = {} if args.force else load_results()

    n_to_run = sum(1 for d, m, x in cells if results.get(cell_key(d, m, x), {}).get("status") != "SUCCEEDED")
    print(f"Stage 7 sweep: {len(cells)} cells total, {n_to_run} to run (skipping already-succeeded ones).\n")

    sweep_start = time.time()
    for i, (dataset, model_type, method) in enumerate(cells, start=1):
        key = cell_key(dataset, model_type, method)
        if results.get(key, {}).get("status") == "SUCCEEDED":
            print(f"--- Cell {i}/{len(cells)}: {key} -- already succeeded, skipping ---")
            continue

        print(f"--- Cell {i}/{len(cells)}: {key} ---")
        cell_start = time.time()
        try:
            job_name = launch_cell(dataset, model_type, method, wait=True)
            status = "SUCCEEDED"
        except Exception as e:
            job_name = None
            status = f"FAILED: {e}"
            print(f"  !! {status}")
        elapsed = time.time() - cell_start

        results[key] = {
            "dataset": dataset,
            "model_type": model_type,
            "method": method,
            "job_name": job_name,
            "status": status,
            "wall_seconds": elapsed,
        }
        save_results(results)  # after every cell, not just at the end
        print(f"  ({elapsed:.0f}s wall time for this cell)\n")

    total_elapsed = time.time() - sweep_start
    n_failed = sum(1 for r in results.values() if r["status"] != "SUCCEEDED")

    print(f"\nSweep run complete: {len(cells) - n_failed}/{len(cells)} total cells succeeded, {total_elapsed / 60:.1f} min this run.")
    if n_failed:
        print("Still failing -- re-run this script to retry just these:")
        for key, r in results.items():
            if r["status"] != "SUCCEEDED":
                print(f"  {key}: {r['status']}")
    print(f"Full results log: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
