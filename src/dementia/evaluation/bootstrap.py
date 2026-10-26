"""Stratified participant-level bootstrap CIs and paired AUC differences."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from dementia.evaluation.metrics import METRICS, compute_metrics


def _draws(y, n: int, seed: int):
    """Resample positives and negatives separately so both classes are always present."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, score, n: int = 2000, seed: int = 42, threshold: float = 0.5) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    vals = {m: [] for m in METRICS}
    for idx in _draws(y, n, seed):
        r = compute_metrics(y[idx], score[idx], threshold)
        for m in METRICS:
            vals[m].append(r[m])
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in vals.items()}


def paired_auc_difference(y, score_a, score_b, n: int = 2000, seed: int = 42) -> dict:
    """AUC(a) - AUC(b) with a CI from resampling the same participants for both models."""
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    diffs = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in _draws(y, n, seed)]
    return {
        "auc_difference": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
