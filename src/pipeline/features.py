"""Stage 5: reduce every dataset to (at most) 25 features, using one shared
selection method -- mutual information with the target, computed on
training data only, top-k kept.

What differs per dataset is the CANDIDATE POOL the shared selector picks
from, not the selector itself:
  - Sepsis and CIC-IDS2017: every remaining column is already a legitimate
    feature (clinical vitals/labs; network flow statistics), so the pool
    is just "everything except target/time/group(/feature_exclude)".
  - Sparkov: most of its raw columns are per-customer identifiers in
    disguise (see CLAUDE.md's Stage 5 notes -- lat/long/city/zip/dob/street
    all have ~970 unique values out of ~983 customers). Only 8 raw columns
    survive that triage, so this module also derives the standard
    age/distance/calendar/velocity features needed to build a pool worth
    selecting from before the shared selector runs. Even after that,
    Sparkov's pool lands at 21 candidates -- short of 25 -- which is
    reported, not padded further with contrived columns.

Sparkov's negative-downsampling also lives here now, as the LAST step,
after the velocity/deviation features are computed (those need each
card's full, undownsampled history to mean anything).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .config import DatasetConfig


def reduce_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if config.name == "sepsis":
        return _reduce_generic(train_df, test_df, config)
    if config.name == "cic-ids2017":
        return _reduce_generic(train_df, test_df, config)
    if config.name == "sparkov":
        return _reduce_sparkov(train_df, test_df, config)
    raise ValueError(f"No Stage 5 feature reduction defined for {config.name!r}")


def _non_feature_columns(config: DatasetConfig) -> set[str]:
    return {config.group_column, config.time_column, config.target, *config.feature_exclude}


def _reduce_generic(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    candidates = [c for c in train_df.columns if c not in _non_feature_columns(config)]
    return _select_top_k(train_df, test_df, candidates, config)


def _select_top_k(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_cols: list[str],
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The one shared selection method. Ranks candidates by mutual
    information with the target, scored on training data only, and keeps
    the top k (or every candidate, if fewer than k exist).
    """
    k = config.select_params.get("k", 25)
    seed = config.select_params.get("seed")
    sample_size = config.select_params.get("mi_sample_size")

    mi_input = train_df
    if sample_size is not None and len(train_df) > sample_size:
        mi_input = train_df.sample(n=sample_size, random_state=seed)

    target = pd.factorize(mi_input[config.target])[0]
    scores = mutual_info_classif(mi_input[candidate_cols], target, random_state=seed)

    ranked = sorted(zip(candidate_cols, scores), key=lambda pair: pair[1], reverse=True)
    selected = [col for col, _ in ranked[:k]]

    keep_cols = [c for c in [*selected, config.target, config.time_column, config.group_column] if c in train_df.columns]
    return train_df[keep_cols].copy(), test_df[keep_cols].copy(), selected


def _reduce_sparkov(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = _add_sparkov_features(train_df, config)
    test_df = _add_sparkov_features(test_df, config)
    train_df, test_df = _encode_sparkov_categoricals(train_df, test_df)

    candidates = [c for c in train_df.columns if c not in _non_feature_columns(config)]
    train_df, test_df, selected = _select_top_k(train_df, test_df, candidates, config)

    train_df = _downsample_negatives(train_df, config)
    test_df = _downsample_negatives(test_df, config)
    return train_df, test_df, selected


def _haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def _add_sparkov_features(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    trans_dt = pd.to_datetime(df[config.time_column])
    dob_dt = pd.to_datetime(df["dob"])

    # Deliberately coarsened versions of identity-like raw columns (dob,
    # lat/long) -- age and distance are shared by many customers, unlike
    # the raw inputs they come from. See module docstring.
    df["age"] = (trans_dt - dob_dt).dt.days / 365.25
    df["distance_km"] = _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    df["hour_of_day"] = trans_dt.dt.hour
    df["day_of_week"] = trans_dt.dt.dayofweek
    df["month"] = trans_dt.dt.month
    df["day_of_month"] = trans_dt.dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)

    # Card-velocity features: each card's OWN history up to (not including)
    # the current transaction. Causal by construction (shift(1) before the
    # expanding window), computed independently within whichever of
    # train/test this df is -- a simplification documented in CLAUDE.md.
    df = df.sort_values([config.group_column, config.time_column])
    by_card_amt = df.groupby(config.group_column)["amt"]
    prior_mean = by_card_amt.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    prior_max = by_card_amt.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)

    df["card_txn_count_so_far"] = df.groupby(config.group_column).cumcount()
    df["card_amt_dev_from_own_mean"] = (df["amt"] - prior_mean).fillna(0)
    df["card_amt_dev_from_own_max"] = (df["amt"] - prior_max).fillna(0)

    # Train-fit lookups (category/merchant average spend) use the full
    # training set including self, which is a negligible amount of
    # smoothing given ~1,870 rows per merchant and far more per category --
    # not leave-one-out, but not a meaningful leak at this scale either.
    return df


def _encode_sparkov_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in ["merchant", "category", "gender", "state", "job"]:
        train_df[col], test_df[col] = _frequency_encode(train_df[col], test_df[col])

    for col in ["category", "merchant"]:
        train_mean = train_df.groupby(col)["amt"].transform("mean")
        lookup = train_df.groupby(col)["amt"].mean()
        train_df[f"{col}_amt_dev_from_mean"] = train_df["amt"] - train_mean
        test_df[f"{col}_amt_dev_from_mean"] = test_df["amt"] - test_df[col].map(lookup).fillna(lookup.mean())

    return train_df, test_df


def _frequency_encode(train_series: pd.Series, test_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Each category -> its frequency rank in train (1 = most common).
    Categories never seen in train fall back to the rarest rank + 1.
    """
    frequency_rank = train_series.value_counts().rank(method="dense", ascending=False)
    fallback_rank = frequency_rank.max() + 1
    return train_series.map(frequency_rank), test_series.map(frequency_rank).fillna(fallback_rank)


def _downsample_negatives(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    target_rate = config.clean_params.get("target_positive_rate", 0.02)
    seed = config.clean_params.get("seed")

    positives = df[df[config.target] == 1]
    negatives = df[df[config.target] == 0]

    n_negatives_to_keep = int(len(positives) * (1 - target_rate) / target_rate)
    n_negatives_to_keep = min(n_negatives_to_keep, len(negatives))

    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(negatives.index, size=n_negatives_to_keep, replace=False)

    return pd.concat([positives, negatives.loc[keep_idx]]).sort_index()
