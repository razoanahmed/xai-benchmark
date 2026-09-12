"""Stage 4: the real per-dataset cleaning quirks documented in CLAUDE.md.

clean() runs AFTER split(), not before -- so train-only statistics (the
Sepsis medians, in particular) are computed without looking at test rows.
Each dataset's quirks are handled independently; none of this logic is
shared, which is expected -- loaders.py and splitting.py are the shared
parts of Stage 3, this stage is deliberately dataset-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig


def clean(train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.name == "sepsis":
        return _clean_sepsis(train_df, test_df, config)
    if config.name == "sparkov":
        return _clean_sparkov(train_df, test_df, config)
    if config.name == "cic-ids2017":
        return _clean_cic_ids2017(train_df, test_df, config)
    return train_df, test_df


def _feature_columns(df: pd.DataFrame, config: DatasetConfig) -> list[str]:
    return [c for c in df.columns if c not in {config.group_column, config.time_column, config.target}]


def _clean_sepsis(train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill sparse lab values within each patient, then fill each
    patient's leading gap (before their first reading) with a training-set
    column median. Medians come from train only, and are reused on test
    as-is, so no test information leaks into how train is filled.
    """
    feature_cols = _feature_columns(train_df, config)

    def forward_fill_by_patient(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([config.group_column, config.time_column])
        df[feature_cols] = df.groupby(config.group_column)[feature_cols].ffill()
        return df

    train_df = forward_fill_by_patient(train_df)
    test_df = forward_fill_by_patient(test_df)

    # .fillna(0): guards the rare case where a column has zero non-null
    # values anywhere in train (seen with a small dev_limit sample), which
    # would otherwise leave the median itself NaN and do nothing below.
    medians = train_df[feature_cols].median().fillna(0)
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)

    return train_df, test_df


def _clean_sparkov(train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the unnamed index column Sparkov's CSVs ship with, then downsample
    negatives within train and within test independently so each side's
    fraud rate rises from its real ~0.4-0.6% to the target rate. Downsampling
    train and test separately (rather than only train) keeps enough fraud
    rows in the test side too -- at the raw rate, a 200-500 row SHAP/LIME
    sample would barely contain any fraud cases to explain.
    """
    target_rate = config.clean_params.get("target_positive_rate", 0.02)
    seed = config.clean_params.get("seed")

    def drop_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        return df.drop(columns=unnamed_cols)

    def downsample_negatives(df: pd.DataFrame) -> pd.DataFrame:
        positives = df[df[config.target] == 1]
        negatives = df[df[config.target] == 0]

        n_negatives_to_keep = int(len(positives) * (1 - target_rate) / target_rate)
        n_negatives_to_keep = min(n_negatives_to_keep, len(negatives))

        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(negatives.index, size=n_negatives_to_keep, replace=False)

        return pd.concat([positives, negatives.loc[keep_idx]]).sort_index()

    train_df = downsample_negatives(drop_unnamed_index(train_df))
    test_df = downsample_negatives(drop_unnamed_index(test_df))
    return train_df, test_df


def _clean_cic_ids2017(train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the blank trailing rows (no Label), drop exact duplicate rows,
    and drop the handful of rows with an infinite flow-rate value. Each
    step is small relative to the ~2.8M total rows, so dropping rather
    than imputing is the simpler, defensible choice here.
    """

    def clean_one(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[config.target])
        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(df[numeric_cols]).any(axis=1)
        df = df[~has_inf]

        return df

    return clean_one(train_df), clean_one(test_df)
