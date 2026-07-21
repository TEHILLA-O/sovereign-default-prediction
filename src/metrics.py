from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold(y_true, y_prob, metric: str = "f1"):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def evaluate_predictions(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_default": precision_score(y_true, y_pred, zero_division=0),
        "recall_default": recall_score(y_true, y_pred, zero_division=0),
        "f1_default": f1_score(y_true, y_pred, zero_division=0),
        "support_default": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }

    try:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        result["roc_auc"] = np.nan

    try:
        result["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        result["pr_auc"] = np.nan

    result["report"] = classification_report(y_true, y_pred, zero_division=0)
    return result
