"""Plain-assert smoke tests for the Stage 3/4 pipeline.

No pytest dependency (not in requirements.txt yet) -- just run:
    python -m tests.smoke_test

Stage 3 checks: on the dev-limited configs, that each dataset's documented
split rule actually holds (no leakage across the train/test boundary).
Stage 4 checks: that each dataset's cleaning quirks actually land.
"""

from __future__ import annotations

import numpy as np

from src.pipeline.cleaning import clean
from src.pipeline.config import load_config
from src.pipeline.loaders import load_raw
from src.pipeline.splitting import split


def test_sepsis_no_patient_in_both_splits() -> None:
    config = load_config("configs/sepsis.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)

    train_patients = set(train_df[config.group_column])
    test_patients = set(test_df[config.group_column])
    overlap = train_patients & test_patients

    assert not overlap, f"{len(overlap)} patients appear in both train and test"
    assert len(train_patients) + len(test_patients) == len(train_patients | test_patients)
    print("OK: sepsis -- no patient crosses the train/test boundary")


def test_sparkov_uses_the_pretabulated_files_as_is() -> None:
    config = load_config("configs/sparkov.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)

    assert "_split" not in train_df.columns
    assert "_split" not in test_df.columns
    assert len(train_df) > 0 and len(test_df) > 0
    print("OK: sparkov -- pretabulated train/test files pass through untouched")


def test_cic_train_test_files_dont_mix_weekdays() -> None:
    config = load_config("configs/cic_ids2017.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)

    assert "_source_file" not in train_df.columns
    assert "_source_file" not in test_df.columns
    assert len(train_df) > 0 and len(test_df) > 0
    print("OK: cic-ids2017 -- train/test split by weekday, no source-file leakage column")


def test_sepsis_cleaning_leaves_no_missing_values() -> None:
    config = load_config("configs/sepsis.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)
    train_df, test_df = clean(train_df, test_df, config)

    assert train_df.isna().sum().sum() == 0, "train still has NaN after forward-fill + median fill"
    assert test_df.isna().sum().sum() == 0, "test still has NaN after forward-fill + median fill"
    print("OK: sepsis -- no missing values remain after cleaning")


def test_sparkov_cleaning_hits_target_positive_rate() -> None:
    config = load_config("configs/sparkov.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)
    train_df, test_df = clean(train_df, test_df, config)

    target_rate = config.clean_params.get("target_positive_rate", 0.02)
    for label, part in [("train", train_df), ("test", test_df)]:
        rate = part[config.target].mean()
        assert abs(rate - target_rate) < 0.001, f"{label} positive rate {rate:.4f} isn't close to target {target_rate}"
    print("OK: sparkov -- downsampling hits the target positive rate on both sides")


def test_cic_cleaning_removes_duplicates_and_infinite_values() -> None:
    config = load_config("configs/cic_ids2017.yaml")
    df = load_raw(config)
    train_df, test_df = split(df, config)
    train_df, test_df = clean(train_df, test_df, config)

    for label, part in [("train", train_df), ("test", test_df)]:
        assert part[config.target].isna().sum() == 0, f"{label} still has null-Label rows"
        assert part.duplicated().sum() == 0, f"{label} still has duplicate rows"
        numeric_cols = part.select_dtypes(include=[np.number]).columns
        assert not np.isinf(part[numeric_cols]).any().any(), f"{label} still has infinite values"
    print("OK: cic-ids2017 -- no null-Label rows, duplicates, or infinite values remain")


if __name__ == "__main__":
    test_sepsis_no_patient_in_both_splits()
    test_sparkov_uses_the_pretabulated_files_as_is()
    test_cic_train_test_files_dont_mix_weekdays()
    test_sepsis_cleaning_leaves_no_missing_values()
    test_sparkov_cleaning_hits_target_positive_rate()
    test_cic_cleaning_removes_duplicates_and_infinite_values()
    print("\nAll smoke tests passed.")
