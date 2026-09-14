"""LSTM/FT-Transformer inference. Never imported in the same process as
xgboost_inference.py -- importing torch (which this module and models.py
both do at module level) before XGBoost touches a numpy array segfaults on
this machine (see xgboost_worker.py's docstring). inference.py's lazy
per-branch imports keep the two apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rtdl_revisiting_models import FTTransformer

from .models import MODELS_DIR, _get_device, _LSTMClassifier, _predict_proba, load_model_config


def load_lstm_predict_fn(dataset_name: str, n_features: int) -> Callable[[np.ndarray], np.ndarray]:
    params = load_model_config()["lstm"]
    return _load(MODELS_DIR / dataset_name / "lstm.pt", _LSTMClassifier(n_features, params["hidden_size"], params["num_layers"]))


def load_ft_transformer_predict_fn(dataset_name: str, n_features: int) -> Callable[[np.ndarray], np.ndarray]:
    params = load_model_config()["ft_transformer"]
    backbone_kwargs = FTTransformer.get_default_kwargs(n_blocks=params["n_blocks"])
    model = FTTransformer(n_cont_features=n_features, cat_cardinalities=[], d_out=1, **backbone_kwargs)

    def forward(x_batch: torch.Tensor) -> torch.Tensor:
        return model(x_batch, None).squeeze(-1)

    return _load(MODELS_DIR / dataset_name / "ft_transformer.pt", model, forward=forward)


def build_raw_module(dataset_name: str, model_type: str, n_features: int):
    """Returns (model, forward, device, scaler_mean, scaler_scale) -- the
    raw, loaded torch.nn.Module (not wrapped in a predict_fn), for callers
    that need to differentiate through it directly (SHAP's GradientExplainer
    in explain.py). forward(x_batch: Tensor) -> Tensor is the same
    single-argument calling convention _predict_proba already uses
    internally, so FT-Transformer's two-argument forward() is hidden either
    way.
    """
    params_key = "lstm" if model_type == "lstm" else "ft_transformer"
    params = load_model_config()[params_key]
    device = _get_device()

    if model_type == "lstm":
        model = _LSTMClassifier(n_features, params["hidden_size"], params["num_layers"]).to(device)
        forward = model
        checkpoint_path = MODELS_DIR / dataset_name / "lstm.pt"
    else:
        backbone_kwargs = FTTransformer.get_default_kwargs(n_blocks=params["n_blocks"])
        model = FTTransformer(n_cont_features=n_features, cat_cardinalities=[], d_out=1, **backbone_kwargs).to(device)

        def forward(x_batch: torch.Tensor) -> torch.Tensor:
            return model(x_batch, None).squeeze(-1)

        checkpoint_path = MODELS_DIR / dataset_name / "ft_transformer.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, forward, device, checkpoint["scaler_mean"], checkpoint["scaler_scale"]


def _load(checkpoint_path: Path, model: torch.nn.Module, forward=None) -> Callable[[np.ndarray], np.ndarray]:
    device = _get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]

    def predict(X: np.ndarray) -> np.ndarray:
        X_scaled = (X.astype(np.float64) - scaler_mean) / scaler_scale
        return _predict_proba(model, X_scaled, device, forward=forward)

    return predict
