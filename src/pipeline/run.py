"""CLI entry point for the Stage 3 pipeline skeleton.

Usage:
    python -m src.pipeline.run configs/sepsis.yaml
    python -m src.pipeline.run configs/sparkov.yaml
    python -m src.pipeline.run configs/cic_ids2017.yaml

Runs load -> clean (currently a no-op, see cleaning.py) -> split, then
prints shapes and target distribution so you can sanity-check a dataset
without opening a notebook.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .cleaning import clean
from .config import load_config
from .loaders import load_raw
from .splitting import split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a dataset YAML config, e.g. configs/sepsis.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"[{config.name}] loading raw data...")
    df = load_raw(config)
    print(f"[{config.name}] loaded {len(df):,} rows, {df.shape[1]} columns")

    df = clean(df, config)

    train_df, test_df = split(df, config)
    print(f"[{config.name}] split: train={len(train_df):,} rows, test={len(test_df):,} rows")

    for label, part in [("train", train_df), ("test", test_df)]:
        target_col = part[config.target]
        if pd.api.types.is_numeric_dtype(target_col):
            print(f"[{config.name}]   {label} positive rate: {target_col.mean() * 100:.2f}%")
        else:
            print(f"[{config.name}]   {label} target distribution: {target_col.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
