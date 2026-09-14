"""Stage 11: aggregate the 27-experiment grid into summary tables and
figures, structured around the research question's three parts (see
CLAUDE.md):

  1. Cross-method agreement on the same model     -- from Family A1
  2. Cross-model self-consistency of one method    -- from Family A2
  3. Whether (1) and (2) hold across domains       -- A1/A2 grouped by dataset
  4. Cost picture                                  -- Family B
  5. Family C reported alongside so performance
     confounds stay visible, not hidden            -- already joined into A1/A2

Every number here is a re-aggregation of Stage 8's output (results/metrics_*.csv,
ranked_features.csv) -- Stage 11 computes no new per-cell metric, it only
summarizes ones that already exist.

Usage:
    python -m src.pipeline.run_stage11
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "a1": pd.read_csv(RESULTS_DIR / "metrics_agreement_cross_method.csv"),
        "a2": pd.read_csv(RESULTS_DIR / "metrics_agreement_cross_model.csv"),
        "a3": pd.read_csv(RESULTS_DIR / "metrics_agreement_cross_domain.csv"),
        "b": pd.read_csv(RESULTS_DIR / "metrics_cost.csv"),
        "c": pd.read_csv(RESULTS_DIR / "metrics_model_quality.csv"),
        "ranked": pd.read_csv(RESULTS_DIR / "ranked_features.csv"),
    }


def summarize_cross_method(a1: pd.DataFrame) -> pd.DataFrame:
    """RQ part 1: do SHAP, LIME, PI agree with each other on the same
    model? One row per method-pair, aggregated across all 9 (dataset,
    model) cells, with the average model_auc alongside (Family C, so
    quality context travels with the agreement number, not hidden).
    """
    df = a1.copy()
    df["pair"] = df["method_a"] + "_vs_" + df["method_b"]
    summary = df.groupby("pair").agg(
        n_cells=("spearman_rho", "count"),
        spearman_mean=("spearman_rho", "mean"),
        spearman_std=("spearman_rho", "std"),
        kendall_mean=("kendall_tau", "mean"),
        jaccard_top5_mean=("jaccard_top5", "mean"),
        jaccard_top10_mean=("jaccard_top10", "mean"),
        mean_model_auc=("model_auc", "mean"),
    ).reset_index().sort_values("spearman_mean", ascending=False)
    return summary


def summarize_cross_model(a2: pd.DataFrame) -> pd.DataFrame:
    """RQ part 2: does a single method agree with itself across XGBoost,
    LSTM, FT-Transformer? One row per method, aggregated across all 3
    model-pairs x 3 datasets = 9 cells per method -- both including and
    excluding performance-confounded comparisons (Family C's >0.10 AUC
    diff flag), so a reader can see how much the confound matters.
    """
    rows = []
    for method, group in a2.groupby("method"):
        clean = group[~group["performance_confounded"]]
        rows.append(
            {
                "method": method,
                "n_cells": len(group),
                "spearman_mean_all": group["spearman_rho"].mean(),
                "spearman_mean_excl_confounded": clean["spearman_rho"].mean() if len(clean) else None,
                "n_excluded_as_confounded": len(group) - len(clean),
                "kendall_mean_all": group["kendall_tau"].mean(),
                "jaccard_top5_mean_all": group["jaccard_top5"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_mean_all", ascending=False)


def summarize_cross_domain(a3: pd.DataFrame) -> pd.DataFrame:
    """RQ part 3: do the A1/A2 patterns hold across domains? For each
    (family, metric), the per-domain ranking of pairs by mean agreement,
    and whether that ordering is preserved across all 3 domains.
    """
    rows = []
    for (family, metric), group in a3.groupby(["family", "metric"]):
        pivot = group.pivot(index="pair", columns="domain", values="mean_value")
        pivot = pivot[["sepsis", "sparkov", "cic-ids2017"]]
        orderings = {domain: " > ".join(pivot[domain].sort_values(ascending=False).index) for domain in pivot.columns}
        preserved = group["rank_order_preserved"].iloc[0]
        rows.append({"family": family, "metric": metric, "preserved_across_all_3_domains": preserved, **orderings})
    return pd.DataFrame(rows)


def summarize_cost(b: pd.DataFrame) -> pd.DataFrame:
    """Family B, averaged across the 3 datasets per (model, method)."""
    summary = b.groupby(["model", "method"]).agg(
        mean_time_per_instance_s=("time_per_instance_seconds", "mean"),
        mean_relative_cost=("relative_cost", "mean"),
        mean_elapsed_s=("elapsed_seconds", "mean"),
    ).reset_index().sort_values("mean_relative_cost", ascending=False)
    return summary


def worked_example(ranked: pd.DataFrame, a1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The single worst SHAP-vs-LIME disagreement in the grid (by
    jaccard_top5), shown as an actual side-by-side feature table, not
    just the aggregate score that flags it.
    """
    shap_lime = a1[(a1["method_a"] == "shap") & (a1["method_b"] == "lime")].sort_values("jaccard_top5")
    worst = shap_lime.iloc[0]
    dataset, model = worst["dataset"], worst["model"]

    top_n = 7
    shap_top = ranked[(ranked.dataset == dataset) & (ranked.model == model) & (ranked.method == "shap")].sort_values("rank").head(top_n)
    lime_top = ranked[(ranked.dataset == dataset) & (ranked.model == model) & (ranked.method == "lime")].sort_values("rank").head(top_n)

    table = pd.DataFrame(
        {
            "rank": range(1, top_n + 1),
            "shap_feature": shap_top["feature"].to_numpy(),
            "shap_importance": shap_top["importance"].to_numpy(),
            "lime_feature": lime_top["feature"].to_numpy(),
            "lime_importance": lime_top["importance"].to_numpy(),
        }
    )
    meta = {
        "dataset": dataset,
        "model": model,
        "spearman_rho": float(worst["spearman_rho"]),
        "jaccard_top5": float(worst["jaccard_top5"]),
        "jaccard_top10": float(worst["jaccard_top10"]),
    }
    return table, meta


def _grouped_boxplot(ax, df: pd.DataFrame, group_col: str, value_col: str, title: str, wrap_labels: bool = False) -> None:
    """Plain matplotlib boxplot, groups ordered by descending mean -- avoids
    pandas' DataFrame.boxplot(by=...), whose automatic (alphabetical)
    category ordering silently conflicts with any manual positions/
    xticklabels override.
    """
    order = df.groupby(group_col)[value_col].mean().sort_values(ascending=False).index.tolist()
    data = [df.loc[df[group_col] == key, value_col].to_numpy() for key in order]
    labels = [key.replace("_vs_", "\nvs\n") if wrap_labels else key for key in order]

    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title, fontsize=10, wrap=True)
    ax.set_ylabel(value_col)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=8)


def make_figures(a1: pd.DataFrame, a2: pd.DataFrame, b: pd.DataFrame) -> None:
    # Figure 1: cross-method agreement by pair (RQ1)
    df = a1.copy()
    df["pair"] = df["method_a"] + "_vs_" + df["method_b"]
    fig, ax = plt.subplots(figsize=(6, 4))
    _grouped_boxplot(ax, df, "pair", "spearman_rho", "Cross-method agreement (Spearman rho), all 9 (dataset, model) cells", wrap_labels=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cross_method_agreement.png", dpi=150)
    plt.close(fig)

    # Figure 2: cross-model agreement by method (RQ2)
    fig, ax = plt.subplots(figsize=(6, 4))
    _grouped_boxplot(ax, a2, "method", "spearman_rho", "Cross-model self-consistency (Spearman rho), all 9 (dataset, method) cells")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cross_model_agreement.png", dpi=150)
    plt.close(fig)

    # Figure 3: cost by model x method
    cost = b.groupby(["model", "method"])["relative_cost"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cost.plot(kind="bar", ax=ax)
    ax.set_ylabel("Relative cost\n(x cheapest method on same model)")
    ax.set_title("Cost by model x method (mean across 3 datasets)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cost_by_model_method.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_inputs()

    cross_method = summarize_cross_method(data["a1"])
    cross_model = summarize_cross_model(data["a2"])
    cross_domain = summarize_cross_domain(data["a3"])
    cost = summarize_cost(data["b"])
    example_table, example_meta = worked_example(data["ranked"], data["a1"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cross_method.to_csv(RESULTS_DIR / "stage11_cross_method_summary.csv", index=False)
    cross_model.to_csv(RESULTS_DIR / "stage11_cross_model_summary.csv", index=False)
    cross_domain.to_csv(RESULTS_DIR / "stage11_cross_domain_summary.csv", index=False)
    cost.to_csv(RESULTS_DIR / "stage11_cost_summary.csv", index=False)
    example_table.to_csv(RESULTS_DIR / "stage11_worked_example.csv", index=False)

    import json

    (RESULTS_DIR / "stage11_worked_example_meta.json").write_text(json.dumps(example_meta, indent=2))

    make_figures(data["a1"], data["a2"], data["b"])

    print("Stage 11 wrote to results/:")
    for name in [
        "stage11_cross_method_summary.csv",
        "stage11_cross_model_summary.csv",
        "stage11_cross_domain_summary.csv",
        "stage11_cost_summary.csv",
        "stage11_worked_example.csv",
        "stage11_worked_example_meta.json",
        "fig_cross_method_agreement.png",
        "fig_cross_model_agreement.png",
        "fig_cost_by_model_method.png",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
