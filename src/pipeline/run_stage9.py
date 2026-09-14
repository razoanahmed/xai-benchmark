"""Stage 9: computational cost metrics.

Runtime and dollar cost are already Stage 8's Family B (M5-M8, in
results/metrics_cost.csv) -- docs/METRICS.md is normative and defines no
separate Stage 9 metric, so there's nothing new to compute for those two.

Memory was evaluated and deliberately excluded as a metric (decided with
the user, not added to METRICS.md): CloudWatch's MemoryUtilization for
these Processing Jobs is sampled at a 1-minute period, but most of the 27
cells ran for only ~200s total, dominated by the one-time pip install, not
the sub-second explanation compute itself -- leaving most cells with just
1-2 datapoints (effectively a single, arbitrarily-timed snapshot, most
likely during the pip install rather than the actual SHAP/LIME/PI call)
against 25-29 datapoints for the three FT-Transformer+LIME cells (the only
ones slow enough for 1-minute sampling to mean anything). Comparing "max
of 1 sample" against "max of 29 samples" across the grid isn't a fair
comparison, so it doesn't become a metric.

This module exists only to pull that raw CloudWatch data one more time and
save it to results/ so it's on record -- explicitly NOT part of Stage 8's
output contract (which is closed, "and nothing else"), and NOT a metric,
just a documented, reproducible trace of what was checked and why it was
set aside.

Usage:
    python -m src.pipeline.run_stage9
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd

from .config import REPO_ROOT
from .upload_to_s3 import load_aws_config

SWEEP_RESULTS_PATH = REPO_ROOT / "stage7_sweep_results.json"
RESULTS_DIR = REPO_ROOT / "results"
INSTANCE_MEMORY_GIB = 16  # ml.m5.xlarge


def fetch_memory_utilization() -> pd.DataFrame:
    region = load_aws_config()["region"]
    sweep = json.loads(SWEEP_RESULTS_PATH.read_text())
    sm = boto3.client("sagemaker", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)

    rows = []
    for key, entry in sorted(sweep.items()):
        dataset, model_type, method = key.split("/")
        job_name = entry["job_name"]
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        start = desc["CreationTime"]
        end = desc["ProcessingEndTime"] + timedelta(minutes=2)

        resp = cw.get_metric_statistics(
            Namespace="/aws/sagemaker/ProcessingJobs",
            MetricName="MemoryUtilization",
            Dimensions=[{"Name": "Host", "Value": f"{job_name}/algo-1"}],
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=["Maximum"],
        )
        datapoints = resp["Datapoints"]
        max_pct = max((dp["Maximum"] for dp in datapoints), default=None)

        rows.append(
            {
                "dataset": dataset,
                "model": model_type,
                "method": method,
                "job_name": job_name,
                "max_memory_pct": max_pct,
                "approx_gib": (max_pct / 100) * INSTANCE_MEMORY_GIB if max_pct is not None else None,
                "n_datapoints": len(datapoints),
                "reliable_sample": len(datapoints) >= 10,  # rule of thumb: too few points to trust the max otherwise
                "instance_type": "ml.m5.xlarge",
                "instance_memory_gib": INSTANCE_MEMORY_GIB,
                "window_start_utc": start.astimezone(timezone.utc).isoformat(),
                "window_end_utc": end.astimezone(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("Stage 9: runtime + cost already covered by Stage 8's results/metrics_cost.csv (M5-M8).")
    print("Fetching raw CloudWatch MemoryUtilization for the record (not a metric -- see this module's docstring)...")

    df = fetch_memory_utilization()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stage9_memory_cloudwatch_raw.csv"
    df.to_csv(out_path, index=False)

    n_reliable = df["reliable_sample"].sum()
    print(f"Saved {len(df)} rows to {out_path.relative_to(REPO_ROOT)}")
    print(f"  {n_reliable}/{len(df)} cells have a sample size (>=10 datapoints) worth trusting the max of.")
    reliable = df[df["reliable_sample"]].sort_values("max_memory_pct", ascending=False)
    for _, row in reliable.iterrows():
        print(
            f"  {row['dataset']}/{row['model']}/{row['method']}: "
            f"{row['max_memory_pct']:.2f}% ({row['approx_gib']:.2f} GiB), n={row['n_datapoints']}"
        )
    print("\nStage 9 complete: cost/runtime in metrics_cost.csv, memory investigated and excluded (see PROJECT_PLAN.md).")


if __name__ == "__main__":
    main()
