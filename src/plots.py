from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(
    y_true,
    y_prob,
    title: str,
    output_path: Path,
    threshold: float | None = None,
    roc_auc: float | None = None,
) -> Path:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if roc_auc is None or np.isnan(roc_auc):
        roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")

    if threshold is not None and len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(
            fpr[idx],
            tpr[idx],
            color="#d62728",
            s=60,
            zorder=5,
            label=f"F1 tuned threshold = {threshold:.2f}",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
