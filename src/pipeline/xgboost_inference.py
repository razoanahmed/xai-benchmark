"""XGBoost-only inference, kept free of any torch import -- see
xgboost_worker.py's docstring for why importing torch before XGBoost
touches a numpy array segfaults on this machine. inference.py only ever
imports this module for XGBoost cells, and never in a process that also
loads torch_inference.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import xgboost as xgb

from .config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


def load_xgboost_predict_fn(dataset_name: str) -> Callable[[np.ndarray], np.ndarray]:
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / dataset_name / "xgboost.json")

    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X.astype(np.float32))[:, 1]

    return predict
