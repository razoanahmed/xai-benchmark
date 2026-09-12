"""Stage 4 hook.

clean() is currently a no-op on purpose. Stage 3 only needs the pipeline
to have a cleaning step that every dataset passes through; Stage 4 fills
this in per dataset (sep handling already lives in loaders.py, but
negative-downsampling for Sparkov, duplicate/infinite-value handling for
CIC-IDS2017, and forward-filling sparse labs for Sepsis all belong here).
"""

from __future__ import annotations

import pandas as pd

from .config import DatasetConfig


def clean(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    return df
