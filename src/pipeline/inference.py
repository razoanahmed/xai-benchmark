"""Stage 7: reconstruct a trained model as a plain predict_proba callable,
uniform across all three model types, so SHAP/LIME/Permutation Importance
never need to know which model they're explaining.

Imports are lazy and per-branch on purpose: importing torch (pulled in by
the lstm/ft_transformer path) before XGBoost touches a numpy array
segfaults on this machine (see xgboost_worker.py's docstring). A caller
that only ever asks for "xgboost" in a given process must never trigger a
torch import as a side effect of this module simply being imported.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]


def load_predict_fn(dataset_name: str, model_type: str, n_features: int) -> PredictFn:
    """Returns a function X (n_rows, n_features) float32 -> probabilities
    (n_rows,) of the positive class, for the given trained model. Loads
    strictly from the real models/<dataset>/ path (never models/dev/) --
    Stage 7 must only ever explain the full-scale Stage 6 models.
    """
    if model_type == "xgboost":
        from .xgboost_inference import load_xgboost_predict_fn

        return load_xgboost_predict_fn(dataset_name)
    if model_type == "lstm":
        from .torch_inference import load_lstm_predict_fn

        return load_lstm_predict_fn(dataset_name, n_features)
    if model_type == "ft_transformer":
        from .torch_inference import load_ft_transformer_predict_fn

        return load_ft_transformer_predict_fn(dataset_name, n_features)
    raise ValueError(f"Unknown model type: {model_type!r}")
