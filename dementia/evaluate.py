"""Evaluate every locked model once on held-out participants, with bootstrap CIs (step: evaluate)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from dementia.data import load_participants
from dementia.features import build_matrix, load_features
from dementia.train import check_lock

METRICS = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def compute_metrics(y, score, threshold=0.5) -> dict:
    y, pred = np.asarray(y), (np.asarray(score) >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _resamples(y, n, seed):
    """Stratified participant bootstrap: resample each class separately so both are always present."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def bootstrap_ci(y, score, n=2000, seed=42, threshold=0.5) -> dict:
    vals = [compute_metrics(y[i], score[i], threshold) for i in _resamples(y, n, seed)]
    return {m: tuple(np.percentile([v[m] for v in vals], [2.5, 97.5])) for m in METRICS}


def paired_auc_difference(y, a, b, n=2000, seed=42) -> dict:
    """AUC(a) - AUC(b), resampling the same participants for both models."""
    d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in _resamples(y, n, seed)]
    return {"auc_difference": roc_auc_score(y, a) - roc_auc_score(y, b), "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)}


def evaluate(cfg) -> pd.DataFrame:
    done = cfg.out("results", "test_metrics.csv")
    if done.exists():
        raise SystemExit("Held-out evaluation already done; results are in outputs/results. Do not re-tune on the test set.")
    lock = check_lock(cfg)
    test = load_participants(cfg, "test")
    y = test.label.to_numpy()
    experiments = json.loads(cfg.out("models", "experiments.json").read_text())
    feats = {e: load_features(cfg, e, test.subject_id) for e in lock["encoders"]}
    thr, n_boot = cfg["threshold"], cfg["bootstrap"]

    scores, rows = {}, []
    for x in experiments:
        m = joblib.load(cfg.out("models", f"{x['name']}.joblib"))
        X = test[["age"]].to_numpy(float) if m["representation"] == "age" else build_matrix(m["representation"], m["positions"], feats)
        s = m["pipeline"].predict_proba(X)[:, 1]
        scores[x["name"]] = s
        met, ci = compute_metrics(y, s, thr), bootstrap_ci(y, s, n_boot, cfg["seed"], thr)
        row = {"model": x["name"], "n_test": len(y), **met}
        for k in METRICS:
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = ci[k]
        rows.append(row)
        _plot_confusion(met, x["name"], cfg.out("results", f"confusion_{x['name']}.png"))

    table = pd.DataFrame(rows)
    pd.DataFrame({"subject_id": test.subject_id, "label": y, **{f"score_{k}": v for k, v in scores.items()}}).to_csv(
        cfg.out("results", "test_predictions.csv"), index=False
    )
    budgets = sorted({x["budget"] for x in experiments if x["representation"] == "fusion"})
    diffs = [
        {"model_a": f"fusion_{b}", "model_b": f"{o}_{b}", **paired_auc_difference(y, scores[f"fusion_{b}"], scores[f"{o}_{b}"], n_boot, cfg["seed"])}
        for b in budgets
        for o in ("cnn", "vit")
        if f"{o}_{b}" in scores
    ]
    pd.DataFrame(diffs, columns=["model_a", "model_b", "auc_difference", "ci_low", "ci_high"]).to_csv(
        cfg.out("results", "paired_auc_differences.csv"), index=False
    )
    _plot_roc(y, scores, cfg.out("results", "test_roc.png"))
    table.to_csv(done, index=False)
    cfg.out("results", "evaluation_notes.json").write_text(json.dumps({
        "n_test": int(len(y)), "n_cdr_positive": int(y.sum()), "threshold": thr, "selected_model": lock["selected"],
        "bootstrap": f"{n_boot} stratified participant resamples, 95% percentile intervals, conditional on the fitted models",
        "caveats": ["single dataset, no external validation", "comparisons are descriptive, not multiplicity-adjusted",
                    "a CI crossing 0 does not show fusion is better", "CDR group is not a diagnosis"],
    }, indent=2))
    print(table[["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "balanced_accuracy", "sensitivity", "specificity"]].round(3).to_string(index=False))
    return table


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_confusion(m, name, path):
    plt = _plt()
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["CDR 0", "CDR > 0"], yticklabels=["CDR 0", "CDR > 0"],
           xlabel="Predicted", ylabel="Recorded", title=name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_roc(y, scores, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, s):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC (one point per participant)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
