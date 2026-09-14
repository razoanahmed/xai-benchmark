"""Stage 6: train XGBoost, LSTM, and FT-Transformer on each dataset's
reduced feature set.

No accuracy tuning (hard rule 2) -- every model uses the same fixed
hyperparameters from configs/models.yaml regardless of dataset, and this
module has no dataset-specific branching at all (unlike cleaning.py and
features.py): the same training code runs on all three datasets' tables,
which are already uniform by this point (binary target, all-numeric
features, 25-or-fewer columns).

LSTM and FT-Transformer both treat each row as one independent sample, not
a multi-step sequence -- decided with the user so every model explains the
exact same instance unit, which the cross-model XAI comparison this study
is built around depends on. FT-Transformer's own forward() signature only
accepts a single row's features anyway (no sequence dimension), which is
part of why.

XGBoost training happens in a separate standalone subprocess (plain
subprocess.run, data passed via temp files), not in-process like the
other two -- see xgboost_worker.py's docstring for why (a segfault caused
purely by import order between torch and xgboost on this machine).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import REPO_ROOT, DatasetConfig

MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
MODELS_DIR = REPO_ROOT / "models"
MODEL_TYPES = ["xgboost", "lstm", "ft_transformer"]


def _is_dev_run(config: DatasetConfig) -> bool:
    return config.dev_limit.max_files is not None or config.dev_limit.max_rows_per_file is not None


def _output_dir(config: DatasetConfig) -> Path:
    """Dev-limited runs (configs/dev/*.yaml) write here instead of the real
    models/<dataset>/ path, so a smoke-test run can never overwrite the real,
    full-scale models a dev run happens to share a dataset name with -- this
    is exactly how the full-scale Stage 6 models got silently clobbered by a
    later smoke-test run before this fix.
    """
    base = MODELS_DIR / "dev" if _is_dev_run(config) else MODELS_DIR
    return base / config.name


@dataclass
class TrainResult:
    model_type: str
    dataset: str
    n_train_rows: int
    train_seconds: float
    train_accuracy: float
    train_auc: float
    test_accuracy: float
    test_auc: float


def load_model_config() -> dict[str, Any]:
    with open(MODELS_CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    config: DatasetConfig,
) -> list[TrainResult]:
    model_config = load_model_config()
    return [_train_one(model_type, train_df, test_df, selected_features, config, model_config) for model_type in MODEL_TYPES]


def _train_one(
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    config: DatasetConfig,
    model_config: dict[str, Any],
) -> TrainResult:
    X_train = train_df[selected_features].to_numpy(dtype=np.float32)
    y_train = train_df[config.target].to_numpy(dtype=np.int64)
    X_test = test_df[selected_features].to_numpy(dtype=np.float32)
    y_test = test_df[config.target].to_numpy(dtype=np.int64)

    out_dir = _output_dir(config)
    seed = model_config["seed"]
    start = time.time()
    if model_type == "xgboost":
        # Saves straight to its final destination itself -- see _train_xgboost.
        train_pred, test_pred = _train_xgboost(X_train, y_train, X_test, model_config["xgboost"], seed, out_dir)
    elif model_type == "lstm":
        model, train_pred, test_pred = _train_lstm(X_train, y_train, X_test, model_config["lstm"], seed)
        _save_model(model, out_dir, model_type)
    elif model_type == "ft_transformer":
        model, train_pred, test_pred = _train_ft_transformer(X_train, y_train, X_test, model_config["ft_transformer"], seed)
        _save_model(model, out_dir, model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
    elapsed = time.time() - start

    return TrainResult(
        model_type=model_type,
        dataset=config.name,
        n_train_rows=len(X_train),
        train_seconds=elapsed,
        train_accuracy=accuracy_score(y_train, train_pred > 0.5),
        train_auc=_safe_auc(y_train, train_pred),
        test_accuracy=accuracy_score(y_test, test_pred > 0.5),
        test_auc=_safe_auc(y_test, test_pred),
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, params: dict[str, Any], seed: int, out_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Runs xgboost_worker.py as a genuine standalone subprocess, which
    never imports torch -- see that module's docstring for why. Data
    crosses the process boundary via temp .npy/.json files rather than
    multiprocessing's pickling, which hung indefinitely on this machine
    for reasons not worth chasing further.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        np.save(work_dir / "X_train.npy", X_train)
        np.save(work_dir / "y_train.npy", y_train)
        np.save(work_dir / "X_test.npy", X_test)
        (work_dir / "params.json").write_text(json.dumps({"params": params, "seed": seed}))

        subprocess.run(
            [sys.executable, "-m", "src.pipeline.xgboost_worker", str(work_dir)],
            cwd=REPO_ROOT,
            check=True,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(work_dir / "model.json", out_dir / "xgboost.json")

        train_pred = np.load(work_dir / "train_pred.npy")
        test_pred = np.load(work_dir / "test_pred.npy")
    return train_pred, test_pred


class _LSTMClassifier(nn.Module):
    """Sequence length is always 1 -- see module docstring for why."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, n_features) -> (batch, seq_len=1, n_features)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def _train_lstm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[tuple[nn.Module, StandardScaler], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    device = _get_device()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _LSTMClassifier(X_train.shape[1], params["hidden_size"], params["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    _run_training_loop(model, optimizer, loss_fn, X_train_s, y_train, params, device)

    train_pred = _predict_proba(model, X_train_s, device)
    test_pred = _predict_proba(model, X_test_s, device)
    return (model, scaler), train_pred, test_pred


def _train_ft_transformer(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[tuple[nn.Module, StandardScaler], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    device = _get_device()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    backbone_kwargs = FTTransformer.get_default_kwargs(n_blocks=params["n_blocks"])
    model = FTTransformer(n_cont_features=X_train.shape[1], cat_cardinalities=[], d_out=1, **backbone_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    def forward(x_batch: torch.Tensor) -> torch.Tensor:
        return model(x_batch, None).squeeze(-1)

    _run_training_loop(model, optimizer, loss_fn, X_train_s, y_train, params, device, forward=forward)

    train_pred = _predict_proba(model, X_train_s, device, forward=forward)
    test_pred = _predict_proba(model, X_test_s, device, forward=forward)
    return (model, scaler), train_pred, test_pred


def _run_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    device: torch.device,
    forward=None,
) -> None:
    forward = forward or model

    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    model.train()
    for _ in range(params["epochs"]):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(forward(x_batch), y_batch)
            loss.backward()
            optimizer.step()


def _predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, forward=None, batch_size: int = 4096) -> np.ndarray:
    """Batched, not one giant forward pass -- FT-Transformer's attention
    cost scales with how many rows go through at once, and a single pass
    over a large test set (seen on CIC-IDS2017, ~100k rows) exhausted MPS
    memory.
    """
    forward = forward or model
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32).to(device)
            chunks.append(torch.sigmoid(forward(x_batch)).cpu().numpy())
    return np.concatenate(chunks)


def _save_model(model: tuple[nn.Module, StandardScaler], out_dir: Path, model_type: str) -> None:
    """For lstm/ft_transformer only -- xgboost saves itself directly to its
    final destination in _train_xgboost, since its model object never
    safely exists in this (torch-loaded) process.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    net, scaler = model
    torch.save(
        {"state_dict": net.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
        out_dir / f"{model_type}.pt",
    )
