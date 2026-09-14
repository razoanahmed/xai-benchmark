"""Stage 7: run one (dataset, model, XAI method) cell and produce a
per-feature global importance ranking, plus the raw per-instance
attributions where the method produces them (SHAP, LIME).

Row caps (hard rule 3, refined by the Stage 2 timing test in CLAUDE.md):
  - SHAP:                    500 rows for XGBoost, 200 for LSTM/FT-Transformer
                             (SHAP's cost is the dominant driver for the two
                             neural models -- see CLAUDE.md's Stage 2 notes).
  - LIME:                    500 rows, every model type.
  - Permutation Importance:  500 rows, n_repeats=5 (Stage 2 confirmed PI is
                             not the cost bottleneck at either n_repeats=2
                             or 5; 5 gives a more stable estimate for
                             effectively the same cost).

SHAP is computed differently per model type on purpose, matching Stage 2's
decision to use the fast exact algorithm for each: shap.TreeExplainer on
XGBoost's raw booster (native, unscaled feature space -- XGBoost never used
a scaler), shap.GradientExplainer on the raw torch module for LSTM/FT-
Transformer (necessarily in the model's own STANDARDIZED input space, since
GradientExplainer differentiates through the actual network, and the
network was trained on scaled inputs). This means SHAP magnitudes aren't
comparable across model types -- but hard rule 7 already forbids comparing
raw attribution values across methods, and Stage 10 only ever compares
per-feature RANKS within one (dataset, model) cell at a time, so this
doesn't affect anything the study actually measures. LIME and Permutation
Importance both work through the uniform predict_fn from inference.py, so
those two stay in raw, un-scaled feature units across all three model
types.

Every _explain_* function returns (scores: np.ndarray shaped (n_features,),
raw: pd.DataFrame | None) -- run_cell() turns scores into the final ranked
importance table with real feature names.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score

from .export import load_processed
from .inference import load_predict_fn

ROW_CAPS = {
    "shap": {"xgboost": 500, "lstm": 200, "ft_transformer": 200},
    "lime": {"xgboost": 500, "lstm": 500, "ft_transformer": 500},
    "permutation_importance": {"xgboost": 500, "lstm": 500, "ft_transformer": 500},
}
PI_N_REPEATS = 5
LIME_BACKGROUND_CAP = 50_000  # rows sampled from train for LIME's internal feature-distribution stats
SHAP_GRADIENT_BACKGROUND = 100  # rows sampled from train as the SHAP baseline for GradientExplainer


@dataclass
class ExplainResult:
    dataset: str
    model_type: str
    method: str
    n_rows_explained: int
    elapsed_seconds: float
    importance: pd.DataFrame  # columns: feature, importance (ranked, descending)
    raw_attributions: pd.DataFrame | None = None  # rows x features, only for shap/lime
    extra: dict[str, Any] = field(default_factory=dict)


def run_cell(dataset_name: str, model_type: str, method: str, seed: int = 42, n_cap: int | None = None) -> ExplainResult:
    train_df, test_df, meta = load_processed(dataset_name)
    selected, target = meta["selected_features"], meta["target"]

    n_cap = n_cap if n_cap is not None else ROW_CAPS[method][model_type]
    X_train = train_df[selected].to_numpy(dtype=np.float32)

    # A random sample, not test_df.head(n_cap): the test tables are sorted
    # (Sepsis: by patient/ICULOS; CIC-IDS2017: chronological), so "the first
    # n rows" is a systematically biased slice, not a representative one --
    # for Sepsis that means SHAP/LIME's whole "global" importance would
    # reflect just the first 1-2 patients in ID order. For CIC-IDS2017 it's
    # worse than unrepresentative: attacks cluster later in each file (see
    # CLAUDE.md's dev_limit note), so the first 500 test rows were all
    # BENIGN, and permutation importance's roc_auc scoring crashed outright
    # (ValueError: only one class present). Found via all three CIC-IDS2017
    # Permutation Importance cells failing in the Stage 7 sweep.
    rng = np.random.default_rng(seed)
    n_sample = min(n_cap, len(test_df))
    sample_idx = rng.choice(len(test_df), size=n_sample, replace=False)
    X_explain = test_df[selected].to_numpy(dtype=np.float32)[sample_idx]
    y_explain = test_df[target].to_numpy(dtype=np.int64)[sample_idx]

    start = time.time()
    if method == "shap":
        scores, raw = _explain_shap(dataset_name, model_type, len(selected), X_train, X_explain, seed)
    elif method == "lime":
        predict_fn = load_predict_fn(dataset_name, model_type, len(selected))
        scores, raw = _explain_lime(predict_fn, X_train, X_explain, selected, seed)
    elif method == "permutation_importance":
        predict_fn = load_predict_fn(dataset_name, model_type, len(selected))
        scores, raw = _explain_permutation_importance(predict_fn, X_explain, y_explain, seed)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    elapsed = time.time() - start

    return ExplainResult(
        dataset=dataset_name,
        model_type=model_type,
        method=method,
        n_rows_explained=len(X_explain),
        elapsed_seconds=elapsed,
        importance=_rank(selected, scores),
        raw_attributions=raw,
    )


def _rank(feature_names: list[str], scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"feature": feature_names, "importance": scores})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _explain_shap(
    dataset_name: str, model_type: str, n_features: int, X_train: np.ndarray, X_explain: np.ndarray, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    if model_type == "xgboost":
        import shap
        import xgboost as xgb

        from .xgboost_inference import MODELS_DIR

        model = xgb.XGBClassifier()
        model.load_model(MODELS_DIR / dataset_name / "xgboost.json")

        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(X_explain))
        if shap_values.ndim == 3:  # some shap versions return (n_classes, n_rows, n_features)
            shap_values = shap_values[-1]
        return _finish_shap(shap_values)

    # lstm / ft_transformer: GradientExplainer on the raw torch module, in
    # the model's own scaled input space (see module docstring for why).
    import shap
    import torch

    from .torch_inference import build_raw_module

    model, forward, device, scaler_mean, scaler_scale = build_raw_module(dataset_name, model_type, n_features)

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(X_train), size=min(SHAP_GRADIENT_BACKGROUND, len(X_train)), replace=False)
    background = torch.tensor((X_train[background_idx] - scaler_mean) / scaler_scale, dtype=torch.float32, device=device)
    explain_tensor = torch.tensor((X_explain - scaler_mean) / scaler_scale, dtype=torch.float32, device=device)

    class _Wrapper(torch.nn.Module):
        """GradientExplainer indexes outputs[:, idx], so it needs a 2D
        (batch, 1) tensor -- forward() itself returns 1D (batch,)."""

        def forward(self, x):
            return forward(x).unsqueeze(-1)

    explainer = shap.GradientExplainer(_Wrapper(), background)
    shap_values = np.asarray(explainer.shap_values(explain_tensor))
    if shap_values.ndim == 3:  # (n_rows, n_features, 1) for a single-output head
        shap_values = shap_values.squeeze(-1)
    return _finish_shap(shap_values)


def _finish_shap(shap_values: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    raw = pd.DataFrame(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return mean_abs, raw


def _explain_lime(predict_fn, X_train, X_explain, feature_names, seed) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if len(X_train) > LIME_BACKGROUND_CAP:
        idx = rng.choice(len(X_train), size=LIME_BACKGROUND_CAP, replace=False)
        X_train = X_train[idx]

    def predict_proba(X: np.ndarray) -> np.ndarray:
        p = predict_fn(X)
        return np.column_stack([1 - p, p])

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["0", "1"], mode="classification", random_state=seed
    )

    rows = []
    for i in range(len(X_explain)):
        exp = explainer.explain_instance(X_explain[i], predict_proba, num_features=len(feature_names))
        weights = dict(exp.as_map()[1])  # {feature_index: weight}, class 1
        rows.append([weights.get(j, 0.0) for j in range(len(feature_names))])

    raw = pd.DataFrame(rows, columns=feature_names)
    mean_abs = raw.abs().mean(axis=0).to_numpy()
    return mean_abs, raw


def _explain_permutation_importance(predict_fn, X_explain, y_explain, seed) -> tuple[np.ndarray, None]:
    rng = np.random.default_rng(seed)
    n_features = X_explain.shape[1]
    baseline_score = roc_auc_score(y_explain, predict_fn(X_explain))

    importances = np.zeros((PI_N_REPEATS, n_features))
    for repeat in range(PI_N_REPEATS):
        for j in range(n_features):
            X_permuted = X_explain.copy()
            X_permuted[:, j] = rng.permutation(X_permuted[:, j])
            permuted_score = roc_auc_score(y_explain, predict_fn(X_permuted))
            importances[repeat, j] = baseline_score - permuted_score

    return importances.mean(axis=0), None


def save_result(result: ExplainResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.importance.to_csv(output_dir / "importance.csv", index=False)
    if result.raw_attributions is not None:
        result.raw_attributions.to_csv(output_dir / "raw_attributions.csv", index=False)

    meta = {
        "dataset": result.dataset,
        "model_type": result.model_type,
        "method": result.method,
        "n_rows_explained": result.n_rows_explained,
        "elapsed_seconds": result.elapsed_seconds,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one Stage 7 (dataset, model, XAI method) cell.")
    parser.add_argument("dataset")
    parser.add_argument("model_type", choices=["xgboost", "lstm", "ft_transformer"])
    parser.add_argument("method", choices=["shap", "lime", "permutation_importance"])
    parser.add_argument("output_dir")
    parser.add_argument("--n-cap", type=int, default=None, help="Override the row cap (for quick n=5 timing tests)")
    args = parser.parse_args()

    result = run_cell(args.dataset, args.model_type, args.method, n_cap=args.n_cap)
    save_result(result, args.output_dir)
    print(
        f"[{result.dataset}/{result.model_type}/{result.method}] "
        f"n_rows={result.n_rows_explained} elapsed={result.elapsed_seconds:.1f}s "
        f"top feature: {result.importance.iloc[0]['feature']}"
    )


if __name__ == "__main__":
    main()
