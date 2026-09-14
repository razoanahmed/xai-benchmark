"""Stage 8 Family C (docs/METRICS.md section 3): M9-M12 model quality,
computed on the full held-out test split for one (dataset, model) pair.

A control, not a headline result -- exists only so a Family A disagreement
can be checked against "one model is just worse" before being reported as
an architecture-dependent finding (METRICS.md's Reporting Rule).

Run as its own process per (dataset, model), like explain.py's cells --
inference.py's xgboost/torch split only protects a process that imports
just one of the two; running all 9 pairs in one Python process would hit
the same torch-before-XGBoost segfault documented in Stage 6.

Usage:
    python -m src.pipeline.quality_metrics <dataset> <model_type> <output_json>
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from .export import load_processed
from .inference import load_predict_fn


def compute_quality(dataset: str, model_type: str) -> dict:
    _, test_df, meta = load_processed(dataset)
    selected, target = meta["selected_features"], meta["target"]

    X_test = test_df[selected].to_numpy(dtype=np.float32)
    y_test = test_df[target].to_numpy(dtype=np.int64)

    predict_fn = load_predict_fn(dataset, model_type, len(selected))
    y_score = predict_fn(X_test)
    y_pred = (y_score > 0.5).astype(np.int64)

    return {
        "dataset": dataset,
        "model": model_type,
        "n_test_rows": len(y_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_positive": float(f1_score(y_test, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
    }


def main() -> None:
    dataset, model_type, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    result = compute_quality(dataset, model_type)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{dataset}/{model_type}] acc={result['accuracy']:.3f} f1={result['f1_positive']:.3f} "
          f"auc={result['roc_auc']:.3f} pr_auc={result['pr_auc']:.3f}")


if __name__ == "__main__":
    main()
