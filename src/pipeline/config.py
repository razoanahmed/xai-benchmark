"""Load a dataset's YAML settings file into a typed config object.

Every dataset (Sepsis, Sparkov, CIC-IDS2017) gets one YAML file under
configs/. This module is the only place that knows the YAML schema —
loaders.py, cleaning.py, and splitting.py all work off the DatasetConfig
object this produces, never off raw dicts or dataset names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DevLimit:
    """Caps applied while developing the pipeline, so test runs stay fast.

    Set both to null in the YAML (or omit dev_limit entirely) for a real,
    full-size run.
    """

    max_files: int | None = None
    max_rows_per_file: int | None = None


@dataclass
class SplitConfig:
    method: str  # "group" | "pretabulated" | "filename_group"
    test_size: float | None = None
    seed: int | None = None
    train_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str
    format: str  # "psv_dir" | "csv_pretabulated" | "csv_multi"
    target: str
    time_column: str
    group_column: str
    split: SplitConfig
    paths: list[str] = field(default_factory=list)
    train_path: str | None = None
    test_path: str | None = None
    file_glob: str = "*"
    separator: str = ","
    encoding: str = "utf-8"
    id_from_filename_pattern: str | None = None
    dev_limit: DevLimit = field(default_factory=DevLimit)
    clean_params: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else REPO_ROOT / path


REQUIRED_KEYS = ["name", "format", "target", "time_column", "group_column", "split"]


def load_config(config_path: str | Path) -> DatasetConfig:
    config_path = Path(config_path)
    with open(config_path) as config_file:
        raw: dict[str, Any] = yaml.safe_load(config_file)

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{config_path}: missing required keys {missing}")

    split_raw = raw["split"]
    split = SplitConfig(
        method=split_raw["method"],
        test_size=split_raw.get("test_size"),
        seed=split_raw.get("seed"),
        train_files=split_raw.get("train_files", []),
        test_files=split_raw.get("test_files", []),
    )

    dev_limit_raw = raw.get("dev_limit") or {}
    dev_limit = DevLimit(
        max_files=dev_limit_raw.get("max_files"),
        max_rows_per_file=dev_limit_raw.get("max_rows_per_file"),
    )

    return DatasetConfig(
        name=raw["name"],
        format=raw["format"],
        target=raw["target"],
        time_column=raw["time_column"],
        group_column=raw["group_column"],
        split=split,
        paths=raw.get("paths", []),
        train_path=raw.get("train_path"),
        test_path=raw.get("test_path"),
        file_glob=raw.get("file_glob", "*"),
        separator=raw.get("separator", ","),
        encoding=raw.get("encoding", "utf-8"),
        id_from_filename_pattern=raw.get("id_from_filename_pattern"),
        dev_limit=dev_limit,
        clean_params=raw.get("clean") or {},
    )
