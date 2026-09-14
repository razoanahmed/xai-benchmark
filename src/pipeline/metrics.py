"""Stage 8: the ranking rules (P1-P5) and metrics (M1-M12) defined in
docs/METRICS.md. This module implements exactly what that document
specifies and nothing else -- METRICS.md is normative; if a metric is
needed that isn't in it, the document gets amended first, not this file.

P1 (aggregate to global) and P2 (discard sign) are already done by Stage 7
-- explain.py's SHAP/LIME paths already write importance.csv as
mean(|attribution|) per feature. This module starts from those already-
aggregated, already-unsigned per-feature importance vectors and handles
P3 (rank, never compare raw values), P4 (identical feature sets, checked
loudly), and P5 (clip negative PI before ranking; ties get average rank).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata

TOP_K_SMALL = 5
TOP_K_LARGE = 10
TOP_K_LARGE_MIN_FEATURES = 20  # M4 (top-10 Jaccard) is null below this feature count


@dataclass
class RankedCell:
    """One (dataset, model, method) cell's importance, ranked per P3/P5."""

    features: list[str]  # canonical order, alphabetical
    importance: np.ndarray  # after P5 clipping (permutation_importance only)
    ranks: np.ndarray  # rank 1 = most important (P3), ties = average rank (P5)
    n_clipped: int  # P5: count of negative PI values clipped to 0 (0 for shap/lime)
    n_tied: int  # count of features sharing their rank with >=1 other feature


def rank_cell(importance_df: pd.DataFrame, method: str) -> RankedCell:
    """importance_df: columns 'feature', 'importance' (Stage 7's importance.csv,
    already P1/P2-aggregated). Returns the P3/P5-ranked form, features sorted
    alphabetically so cells can be compared feature-by-feature positionally.
    """
    df = importance_df.sort_values("feature").reset_index(drop=True)
    features = df["feature"].tolist()
    importance = df["importance"].to_numpy(dtype=float)

    n_clipped = 0
    if method == "permutation_importance":
        n_clipped = int((importance < 0).sum())
        importance = np.clip(importance, a_min=0, a_max=None)

    # rank 1 = most important => rank the negated importance (rankdata gives
    # rank 1 to the smallest value by default).
    ranks = rankdata(-importance, method="average")
    n_tied = int((pd.Series(ranks).duplicated(keep=False)).sum())

    return RankedCell(features=features, importance=importance, ranks=ranks, n_clipped=n_clipped, n_tied=n_tied)


def check_same_feature_set(cell_a: RankedCell, cell_b: RankedCell, context: str) -> None:
    """P4: identical feature set (and, since both are built by rank_cell's
    alphabetical sort, identical order) is required before any comparison.
    Fail loudly -- a mismatch is a bug, not a result to route around.
    """
    if cell_a.features != cell_b.features:
        only_a = set(cell_a.features) - set(cell_b.features)
        only_b = set(cell_b.features) - set(cell_a.features)
        raise ValueError(
            f"P4 violation ({context}): feature sets differ. "
            f"Only in first: {sorted(only_a)}. Only in second: {sorted(only_b)}."
        )


def top_k_features(cell: RankedCell, k: int) -> set[str]:
    """The k feature NAMES with the largest importance (ties broken by
    feature name for a deterministic, reproducible set) -- not a cutoff on
    the rank number itself, which is ambiguous when ties straddle k.
    """
    order = sorted(zip(cell.features, cell.importance), key=lambda pair: (-pair[1], pair[0]))
    return {name for name, _ in order[:k]}


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


@dataclass
class AgreementResult:
    n_features: int
    spearman_rho: float
    kendall_tau: float
    jaccard_top5: float
    jaccard_top10: float | None  # null if n_features < TOP_K_LARGE_MIN_FEATURES


def compute_agreement(cell_a: RankedCell, cell_b: RankedCell, context: str) -> AgreementResult:
    """M1-M4 between two already-ranked cells over the same feature set."""
    check_same_feature_set(cell_a, cell_b, context)
    n_features = len(cell_a.features)

    # M1: Pearson correlation of the two rank vectors (== Spearman rho of
    # the underlying importances, using the same average-rank tie
    # convention P5 already applied).
    spearman_rho, _ = pearsonr(cell_a.ranks, cell_b.ranks)

    # M2: Kendall's tau-b, tie-corrected.
    kendall_tau_value, _ = kendalltau(cell_a.ranks, cell_b.ranks, variant="b")

    jaccard_top5 = jaccard(top_k_features(cell_a, TOP_K_SMALL), top_k_features(cell_b, TOP_K_SMALL))
    jaccard_top10 = None
    if n_features >= TOP_K_LARGE_MIN_FEATURES:
        jaccard_top10 = jaccard(top_k_features(cell_a, TOP_K_LARGE), top_k_features(cell_b, TOP_K_LARGE))

    return AgreementResult(
        n_features=n_features,
        spearman_rho=float(spearman_rho),
        kendall_tau=float(kendall_tau_value),
        jaccard_top5=jaccard_top5,
        jaccard_top10=jaccard_top10,
    )
