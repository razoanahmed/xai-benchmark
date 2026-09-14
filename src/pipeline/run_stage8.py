"""Stage 8: compute exactly the metrics defined in docs/METRICS.md, and
nothing else. Writes the 6 files in that document's section 5 output
contract to results/.

Steps:
  1. Download Stage 7's importance.csv + meta.json for all 27 cells from
     S3 (checksummed, for the manifest).
  2. Fetch each cell's real SageMaker Processing Job duration from AWS
     directly (CreationTime -> ProcessingEndTime, i.e. including container
     startup -- METRICS.md's Family B scope note is explicit that this is
     what M5 should measure), not the sweep script's local wall_seconds
     proxy.
  3. Family C (M9-M12): one subprocess per (dataset, model) pair, for the
     same torch/XGBoost process-isolation reason as Stage 7's cells.
  4. Family A (M1-M4, sub-families A1/A2/A3) and Family B (M5-M8) from
     the downloaded/fetched data.

Usage:
    python -m src.pipeline.run_stage8
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import scipy

from .config import REPO_ROOT
from .metrics import TOP_K_LARGE_MIN_FEATURES, compute_agreement, rank_cell
from .upload_to_s3 import load_aws_config

DATASETS = ["sepsis", "sparkov", "cic-ids2017"]
MODEL_TYPES = ["xgboost", "lstm", "ft_transformer"]
METHODS = ["shap", "lime", "permutation_importance"]

RESULTS_DIR = REPO_ROOT / "results"
# Intermediate working data, deliberately NOT under results/: section 5's
# output contract says Stage 8 writes exactly 6 files there "and nothing
# else". .cache/ is gitignored the same way data/processed/ and models/ are.
CACHE_DIR = REPO_ROOT / ".cache" / "stage8"
STAGE7_RAW_DIR = CACHE_DIR / "stage7_raw"
QUALITY_TMP_DIR = CACHE_DIR / "quality"
SWEEP_RESULTS_PATH = REPO_ROOT / "stage7_sweep_results.json"

# Flagged per the user's explicit choice (no Pricing API / Cost Explorer
# access on the CLI IAM user, and web search only surfaced US-region
# pricing): a US (us-east-1) estimate, not a verified ca-central-1 rate.
INSTANCE_HOURLY_RATE_USD = 0.230
INSTANCE_RATE_SOURCE = "us-east-1 web-search estimate, NOT verified for ca-central-1 -- see CLAUDE.md Stage 8 notes"
INSTANCE_RATE_CHECKED_DATE = "2026-09-14"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_stage7_results(bucket: str) -> dict:
    """Pulls importance.csv + meta.json for all 27 cells into
    results/stage7_raw/<dataset>/<model>/<method>/, returns a manifest
    dict of relative_path -> sha256 for every file pulled.
    """
    s3 = boto3.client("s3")
    checksums = {}
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            for method in METHODS:
                local_dir = STAGE7_RAW_DIR / dataset / model_type / method
                local_dir.mkdir(parents=True, exist_ok=True)
                for filename in ["importance.csv", "meta.json"]:
                    key = f"results/stage7/{dataset}/{model_type}/{method}/{filename}"
                    local_path = local_dir / filename
                    s3.download_file(bucket, key, str(local_path))
                    checksums[str(local_path.relative_to(REPO_ROOT))] = _sha256(local_path)
    return checksums


def load_cell_importance(dataset: str, model_type: str, method: str) -> pd.DataFrame:
    path = STAGE7_RAW_DIR / dataset / model_type / method / "importance.csv"
    return pd.read_csv(path)


def load_cell_meta(dataset: str, model_type: str, method: str) -> dict:
    path = STAGE7_RAW_DIR / dataset / model_type / method / "meta.json"
    return json.loads(path.read_text())


def fetch_job_durations() -> dict[tuple[str, str, str], float]:
    """Real AWS Processing Job duration per cell: CreationTime ->
    ProcessingEndTime, in seconds. Includes container startup and data
    download, per METRICS.md's Family B scope note -- deliberately not
    just the ProcessingStart -> ProcessingEnd "script-running" window.
    """
    sweep_results = json.loads(SWEEP_RESULTS_PATH.read_text())
    sm = boto3.client("sagemaker", region_name=load_aws_config()["region"])

    durations = {}
    for key, entry in sweep_results.items():
        dataset, model_type, method = key.split("/")
        job_name = entry["job_name"]
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        elapsed = (desc["ProcessingEndTime"] - desc["CreationTime"]).total_seconds()
        durations[(dataset, model_type, method)] = elapsed
    return durations


def compute_family_c(output_dir: Path) -> dict[tuple[str, str], dict]:
    """One subprocess per (dataset, model) pair -- see quality_metrics.py's
    docstring for why this can't run in a single shared process.
    """
    results = {}
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            out_path = output_dir / f"quality_{dataset}_{model_type}.json"
            subprocess.run(
                [sys.executable, "-m", "src.pipeline.quality_metrics", dataset, model_type, str(out_path)],
                cwd=REPO_ROOT,
                check=True,
            )
            results[(dataset, model_type)] = json.loads(out_path.read_text())
    return results


def build_ranked_cells() -> dict[tuple[str, str, str], "RankedCell"]:  # noqa: F821
    ranked = {}
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            for method in METHODS:
                df = load_cell_importance(dataset, model_type, method)
                ranked[(dataset, model_type, method)] = rank_cell(df, method)
    return ranked


def build_ranked_features_table(ranked: dict) -> pd.DataFrame:
    """results/ranked_features.csv -- an intermediate, not a metric (see
    METRICS.md section 5): one row per (dataset, model, method, feature)
    with that feature's P3/P5 rank and its importance value after P1/P2
    aggregation. The importance column is deliberately the RAW Stage 7
    value (pre-P5-clip), not RankedCell.importance (which is clipped for
    permutation_importance) -- so a reader can see e.g. "rank 15, tied,
    but the actual raw PI score was -1.5" rather than having the negative
    value hidden by the clip that only affects ranking.
    """
    rows = []
    for (dataset, model_type, method), cell in ranked.items():
        raw_importance = load_cell_importance(dataset, model_type, method).set_index("feature")["importance"]
        for feature, rank in zip(cell.features, cell.ranks):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_type,
                    "method": method,
                    "feature": feature,
                    "rank": rank,
                    "importance": raw_importance.loc[feature],
                }
            )
    return pd.DataFrame(rows)


def build_family_a1(ranked: dict, quality: dict) -> pd.DataFrame:
    """A1: cross-method agreement. For each (dataset, model), compare each
    pair of the 3 methods. 9 x 3 = 27 rows.
    """
    rows = []
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            model_auc = quality[(dataset, model_type)]["roc_auc"]
            for method_a, method_b in combinations(METHODS, 2):
                cell_a = ranked[(dataset, model_type, method_a)]
                cell_b = ranked[(dataset, model_type, method_b)]
                agreement = compute_agreement(cell_a, cell_b, context=f"{dataset}/{model_type}: {method_a} vs {method_b}")
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_type,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_features": agreement.n_features,
                        "spearman_rho": agreement.spearman_rho,
                        "kendall_tau": agreement.kendall_tau,
                        "jaccard_top5": agreement.jaccard_top5,
                        "jaccard_top10": agreement.jaccard_top10,
                        "n_tied_a": cell_a.n_tied,
                        "n_tied_b": cell_b.n_tied,
                        "n_pi_clipped_a": cell_a.n_clipped,
                        "n_pi_clipped_b": cell_b.n_clipped,
                        "model_auc": model_auc,
                    }
                )
    return pd.DataFrame(rows)


def build_family_a2(ranked: dict, quality: dict) -> pd.DataFrame:
    """A2: cross-model agreement. For each (dataset, method), compare each
    pair of the 3 models. 9 x 3 = 27 rows.
    """
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            for model_a, model_b in combinations(MODEL_TYPES, 2):
                cell_a = ranked[(dataset, model_a, method)]
                cell_b = ranked[(dataset, model_b, method)]
                agreement = compute_agreement(cell_a, cell_b, context=f"{dataset}/{method}: {model_a} vs {model_b}")
                auc_a = quality[(dataset, model_a)]["roc_auc"]
                auc_b = quality[(dataset, model_b)]["roc_auc"]
                auc_diff = abs(auc_a - auc_b)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_features": agreement.n_features,
                        "spearman_rho": agreement.spearman_rho,
                        "kendall_tau": agreement.kendall_tau,
                        "jaccard_top5": agreement.jaccard_top5,
                        "jaccard_top10": agreement.jaccard_top10,
                        "n_tied_a": cell_a.n_tied,
                        "n_tied_b": cell_b.n_tied,
                        "n_pi_clipped_a": cell_a.n_clipped,
                        "n_pi_clipped_b": cell_b.n_clipped,
                        "model_a_auc": auc_a,
                        "model_b_auc": auc_b,
                        "auc_diff": auc_diff,
                        "performance_confounded": auc_diff > 0.10,
                    }
                )
    return pd.DataFrame(rows)


def build_family_a3(a1: pd.DataFrame, a2: pd.DataFrame) -> pd.DataFrame:
    """A3: cross-domain stability. Not a new metric -- summarizes A1/A2
    across the 3 datasets (domains). For each (family, pair, metric):
    mean agreement per domain, spread (max-min) across domains, and
    whether the ordering of pairs by mean agreement is preserved across
    all 3 domains.
    """
    rows = []
    metric_cols = ["spearman_rho", "kendall_tau", "jaccard_top5", "jaccard_top10"]

    def summarize(df: pd.DataFrame, pair_cols: tuple[str, str], family: str) -> None:
        df = df.copy()
        df["pair"] = df[pair_cols[0]] + "_vs_" + df[pair_cols[1]]
        for metric in metric_cols:
            # Mean agreement per (domain, pair): averaged across whichever
            # dimension isn't dataset or the pair itself (A1: the 3 models;
            # A2: the 3 methods).
            per_domain_pair = df.groupby(["dataset", "pair"])[metric].mean().reset_index()
            pivot = per_domain_pair.pivot(index="pair", columns="dataset", values=metric)

            # Rank order (by mean agreement, descending) preserved across
            # all 3 domains?
            orderings = [tuple(pivot[dataset].sort_values(ascending=False).index) for dataset in DATASETS]
            rank_order_preserved = len(set(orderings)) == 1 if pivot.notna().all().all() else None

            for pair in pivot.index:
                values = pivot.loc[pair]
                spread = float(values.max() - values.min()) if values.notna().all() else None
                for dataset in DATASETS:
                    rows.append(
                        {
                            "family": family,
                            "pair": pair,
                            "metric": metric,
                            "domain": dataset,
                            "mean_value": values[dataset] if pd.notna(values[dataset]) else None,
                            "spread_max_minus_min": spread,
                            "rank_order_preserved": rank_order_preserved,
                        }
                    )

    summarize(a1, ("method_a", "method_b"), "cross_method")
    summarize(a2, ("model_a", "model_b"), "cross_model")
    return pd.DataFrame(rows)


def build_family_b(durations: dict) -> pd.DataFrame:
    """B: M5-M8, cost. 27 rows."""
    rows = []
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            for method in METHODS:
                meta = load_cell_meta(dataset, model_type, method)
                elapsed = durations[(dataset, model_type, method)]  # M5
                n_rows = meta["n_rows_explained"]
                time_per_instance = elapsed / n_rows if n_rows else None  # M6
                cost_usd = (elapsed / 3600) * INSTANCE_HOURLY_RATE_USD  # M7
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_type,
                        "method": method,
                        "n_rows_explained": n_rows,
                        "elapsed_seconds": elapsed,
                        "time_per_instance_seconds": time_per_instance,
                        "estimated_cost_usd": cost_usd,
                        "instance_type": "ml.m5.xlarge",
                        "region": "ca-central-1",
                        "hourly_rate_usd": INSTANCE_HOURLY_RATE_USD,
                        "rate_checked_date": INSTANCE_RATE_CHECKED_DATE,
                    }
                )

    df = pd.DataFrame(rows)
    # M8: relative cost = this cell's M6 / cheapest M6 among the 3 methods
    # on the same (dataset, model).
    df["relative_cost"] = df.groupby(["dataset", "model"])["time_per_instance_seconds"].transform(
        lambda s: s / s.min()
    )
    return df


def build_family_c_table(quality: dict) -> pd.DataFrame:
    rows = [quality[(dataset, model_type)] for dataset in DATASETS for model_type in MODEL_TYPES]
    return pd.DataFrame(rows)[["dataset", "model", "n_test_rows", "accuracy", "f1_positive", "roc_auc", "pr_auc"]]


def main() -> None:
    aws_config = load_aws_config()
    bucket = aws_config["s3_bucket"]

    print("Downloading Stage 7 results from S3...")
    checksums = download_stage7_results(bucket)
    print(f"  {len(checksums)} files downloaded to {STAGE7_RAW_DIR.relative_to(REPO_ROOT)}")

    print("Fetching real Processing Job durations from AWS...")
    durations = fetch_job_durations()
    print(f"  {len(durations)} job durations fetched")

    print("Computing Family C (model quality) -- one subprocess per (dataset, model)...")
    QUALITY_TMP_DIR.mkdir(parents=True, exist_ok=True)
    quality = compute_family_c(QUALITY_TMP_DIR)

    print("Ranking all 27 cells (P3/P5)...")
    ranked = build_ranked_cells()

    print("Building Family A (agreement)...")
    a1 = build_family_a1(ranked, quality)
    a2 = build_family_a2(ranked, quality)
    a3 = build_family_a3(a1, a2)

    print("Building Family B (cost)...")
    b = build_family_b(durations)

    print("Building Family C (quality) table...")
    c = build_family_c_table(quality)

    print("Building ranked_features.csv (intermediate, not a metric -- see METRICS.md section 5)...")
    ranked_features = build_ranked_features_table(ranked)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    a1.to_csv(RESULTS_DIR / "metrics_agreement_cross_method.csv", index=False)
    a2.to_csv(RESULTS_DIR / "metrics_agreement_cross_model.csv", index=False)
    a3.to_csv(RESULTS_DIR / "metrics_agreement_cross_domain.csv", index=False)
    b.to_csv(RESULTS_DIR / "metrics_cost.csv", index=False)
    c.to_csv(RESULTS_DIR / "metrics_model_quality.csv", index=False)
    ranked_features.to_csv(RESULTS_DIR / "ranked_features.csv", index=False)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage7_input_files_sha256": checksums,
        "library_versions": {"scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "random_seed": 42,
        "random_seed_note": "Governs the upstream pipeline (split/sampling/training) Stage 8 reads from; Stage 8 itself introduces no new randomness.",
        "instance_type": "ml.m5.xlarge",
        "region": "ca-central-1",
        "hourly_rate_usd": INSTANCE_HOURLY_RATE_USD,
        "hourly_rate_source": INSTANCE_RATE_SOURCE,
        "hourly_rate_checked_date": INSTANCE_RATE_CHECKED_DATE,
    }
    (RESULTS_DIR / "metrics_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. Wrote 6 metric files + 1 intermediate to {RESULTS_DIR.relative_to(REPO_ROOT)}:")
    print(f"  metrics_agreement_cross_method.csv  ({len(a1)} rows)")
    print(f"  metrics_agreement_cross_model.csv   ({len(a2)} rows)")
    print(f"  metrics_agreement_cross_domain.csv  ({len(a3)} rows)")
    print(f"  metrics_cost.csv                    ({len(b)} rows)")
    print(f"  metrics_model_quality.csv           ({len(c)} rows)")
    print("  metrics_manifest.json")
    print(f"  ranked_features.csv                 ({len(ranked_features)} rows) -- intermediate, not a metric")


if __name__ == "__main__":
    main()
