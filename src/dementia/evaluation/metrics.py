"""ROC-AUC, balanced accuracy, sensitivity, specificity, F1, confusion counts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

METRICS = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def compute_metrics(y, score, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "roc_auc": auc,
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
