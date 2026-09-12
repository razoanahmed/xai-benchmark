"""Raw-data loaders, one function per dataset *shape*, not per dataset.

Sepsis, Sparkov, and CIC-IDS2017 each fit one of three generic shapes:
  - psv_dir:          one small file per entity, in one or more folders.
  - csv_pretabulated: exactly two files, already pre-split into train/test.
  - csv_multi:        several files that get concatenated, split decided later.

Adding a fourth dataset that fits one of these shapes needs only a new
YAML config, no new code here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DatasetConfig


def load_raw(config: DatasetConfig) -> pd.DataFrame:
    if config.format == "psv_dir":
        df = _load_psv_dir(config)
    elif config.format == "csv_pretabulated":
        df = _load_csv_pretabulated(config)
    elif config.format == "csv_multi":
        df = _load_csv_multi(config)
    else:
        raise ValueError(f"Unknown dataset format: {config.format!r}")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_psv_dir(config: DatasetConfig) -> pd.DataFrame:
    if not config.id_from_filename_pattern:
        raise ValueError(f"{config.name}: psv_dir format requires id_from_filename_pattern")
    pattern = re.compile(config.id_from_filename_pattern)

    files: list[Path] = []
    for raw_path in config.paths:
        files.extend(sorted(config.resolve_path(raw_path).glob(config.file_glob)))

    if config.dev_limit.max_files is not None:
        files = files[: config.dev_limit.max_files]

    frames = []
    for file_path in files:
        match = pattern.match(file_path.name)
        if not match:
            continue  # e.g. a stray index.html sitting next to the real files
        entity_id = match.group(1)
        frame = pd.read_csv(file_path, sep=config.separator)
        frame[config.group_column] = entity_id
        frames.append(frame)

    if not frames:
        raise ValueError(f"{config.name}: no files matched {config.file_glob!r} under {config.paths}")

    return pd.concat(frames, ignore_index=True)


def _load_csv_pretabulated(config: DatasetConfig) -> pd.DataFrame:
    if not config.train_path or not config.test_path:
        raise ValueError(f"{config.name}: csv_pretabulated format requires train_path and test_path")
    nrows = config.dev_limit.max_rows_per_file

    train_df = pd.read_csv(config.resolve_path(config.train_path), nrows=nrows)
    test_df = pd.read_csv(config.resolve_path(config.test_path), nrows=nrows)
    train_df["_split"] = "train"
    test_df["_split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)


def _load_csv_multi(config: DatasetConfig) -> pd.DataFrame:
    nrows = config.dev_limit.max_rows_per_file

    frames = []
    for raw_path in config.paths:
        path = config.resolve_path(raw_path)
        frame = pd.read_csv(path, encoding=config.encoding, nrows=nrows)
        frame["_source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
