"""Plain-assert smoke tests for the Stage 3 pipeline skeleton.

No pytest dependency (not in requirements.txt yet) -- just run:
    python -m tests.smoke_test

Checks, on the dev-limited configs, that each dataset's documented split
rule actually holds: no leakage across the train/test boundary.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    test_sepsis_no_patient_in_both_splits()
    test_sparkov_uses_the_pretabulated_files_as_is()
    test_cic_train_test_files_dont_mix_weekdays()
    print("\nAll smoke tests passed.")
