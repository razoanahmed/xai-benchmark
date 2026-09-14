"""Stage 6.5 prep: write each dataset's fully processed train/test tables to
CSV, plus a small JSON sidecar of column dtypes and the selected-feature
list, so the tables can be uploaded to S3 without re-running the pipeline.

This runs the exact same load -> split -> clean -> reduce_features chain as
run.py, and writes exactly what models.py fed into X_train/X_test in Stage 6
(selected features + target + time/group columns). Every stochastic step in
that chain is seeded (split assignment, MI sampling, Sparkov's negative
downsampling -- see configs/*.yaml), so re-running this against the same
configs reproduces the Stage 6 tables row-for-row.

CSV alone loses dtype information (e.g. a zero-padded patient ID like
"00001" would silently become the integer 1 on reload). The dtype JSON
sidecar fixes that -- load_processed() below reads it back and passes it as
pandas' dtype= argument.

Usage:
    python -m src.pipeline.export configs/sepsis.yaml
    python -m src.pipeline.export configs/sparkov.yaml
    python -m src.pipeline.export configs/cic_ids2017.yaml
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .cleaning import clean
from .config import REPO_ROOT, load_config
from .features import reduce_features
from .loaders import load_raw
from .splitting import split

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def export_processed(config_path: str) -> None:
    config = load_config(config_path)

    print(f"[{config.name}] loading raw data...")
    df = load_raw(config)

    train_df, test_df = split(df, config)
    train_df, test_df = clean(train_df, test_df, config)
    train_df, test_df, selected = reduce_features(train_df, test_df, config)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / f"{config.name}_train.csv"
    test_path = PROCESSED_DIR / f"{config.name}_test.csv"
    meta_path = PROCESSED_DIR / f"{config.name}_dtypes.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "dtypes": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
                "selected_features": selected,
                "target": config.target,
            },
            indent=2,
        )
    )

    print(f"[{config.name}] wrote {len(train_df):,} train rows -> {train_path}")
    print(f"[{config.name}] wrote {len(test_df):,} test rows -> {test_path}")
    print(f"[{config.name}] selected features ({len(selected)}): {selected}")


def load_processed(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reload a dataset's processed CSVs with the exact dtypes they were
    saved with, plus the metadata (selected features, target column).
    """
    meta = json.loads((PROCESSED_DIR / f"{name}_dtypes.json").read_text())
    train_df = pd.read_csv(PROCESSED_DIR / f"{name}_train.csv", dtype=meta["dtypes"])
    test_df = pd.read_csv(PROCESSED_DIR / f"{name}_test.csv", dtype=meta["dtypes"])
    return train_df, test_df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a dataset YAML config, e.g. configs/sepsis.yaml")
    args = parser.parse_args()
    export_processed(args.config)


if __name__ == "__main__":
    main()
