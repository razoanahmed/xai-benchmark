"""Train/test splitting, shared across datasets but never a random row split.

Three methods, matching the three datasets' documented split rules:
  - group:          assign whole entities (e.g. patients) to train or test,
                     so no entity's rows are split across both sides.
  - pretabulated:    the raw files already arrived pre-split (Sparkov);
                     just honor the _split tag the loader attached.
  - filename_group:  assign whole source files to train or test by name
                     (e.g. CIC-IDS2017's Mon-Wed train / Thu-Fri test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig


def split(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = config.split.method
    if method == "group":
        return _split_by_group(df, config)
    if method == "pretabulated":
        return _split_pretabulated(df, config)
    if method == "filename_group":
        return _split_by_filename_group(df, config)
    raise ValueError(f"Unknown split method: {method!r}")


def _split_by_group(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.split.test_size is None:
        raise ValueError(f"{config.name}: split method 'group' requires test_size")

    groups = df[config.group_column].unique()
    rng = np.random.default_rng(config.split.seed)
    shuffled = rng.permutation(groups)
    cutoff = int(len(shuffled) * (1 - config.split.test_size))
    train_groups = set(shuffled[:cutoff])

    is_train = df[config.group_column].isin(train_groups)
    return df[is_train].copy(), df[~is_train].copy()


def _split_pretabulated(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["_split"] == "train"].drop(columns=["_split"])
    test_df = df[df["_split"] == "test"].drop(columns=["_split"])
    return train_df.copy(), test_df.copy()


def _split_by_filename_group(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    def matches_any(filename: str, keywords: list[str]) -> bool:
        return any(keyword.lower() in filename.lower() for keyword in keywords)

    is_train = df["_source_file"].apply(lambda f: matches_any(f, config.split.train_files))
    is_test = df["_source_file"].apply(lambda f: matches_any(f, config.split.test_files))

    train_df = df[is_train].drop(columns=["_source_file"])
    test_df = df[is_test].drop(columns=["_source_file"])
    return train_df.copy(), test_df.copy()
