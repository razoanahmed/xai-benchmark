"""XGBoost training, kept in its own module that NEVER imports torch, run
as a genuine standalone subprocess (not a multiprocessing.Pool worker --
that hung indefinitely on this machine for reasons not worth chasing
further; a plain `subprocess.run` calling this file as a script is simpler
and was verified to work).

Importing torch before XGBoost touches a numpy array (even just building a
DMatrix) reliably segfaults on this machine -- confirmed with faulthandler:
the crash is inside xgboost/data.py's _meta_from_numpy, triggered purely by
import order, not by any particular data or hyperparameters. Likely a
numpy C-API/ABI conflict between this exact torch 2.13.0 / xgboost 3.2.0 /
numpy 2.4.6 combination on macOS arm64.

models.py's _train_xgboost writes X_train/y_train/X_test to temp .npy
files, runs `python -m src.pipeline.xgboost_worker <tmpdir>` as a fresh
subprocess (which never imports torch), and reads the results back from
the same directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


def train(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    return model, train_pred, test_pred


def _main() -> None:
    work_dir = Path(sys.argv[1])

    X_train = np.load(work_dir / "X_train.npy")
    y_train = np.load(work_dir / "y_train.npy")
    X_test = np.load(work_dir / "X_test.npy")
    params = json.loads((work_dir / "params.json").read_text())

    model, train_pred, test_pred = train(X_train, y_train, X_test, params["params"], params["seed"])

    model.save_model(work_dir / "model.json")
    np.save(work_dir / "train_pred.npy", train_pred)
    np.save(work_dir / "test_pred.npy", test_pred)


if __name__ == "__main__":
    _main()
