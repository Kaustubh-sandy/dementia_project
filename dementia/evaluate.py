"""Evaluate every locked model once on held-out participants, with bootstrap CIs (step: evaluate)."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from dementia.data import age_by_class, load_participants
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
    # Predeclared comparisons: fusion vs each encoder at the same budget, and the largest budget vs each smaller one.
    budgets = sorted({x["budget"] for x in experiments if x["representation"] != "age"})
    pairs = [("representation", f"fusion_{b}", f"{o}_{b}") for b in budgets for o in ("cnn", "vit")]
    pairs += [("budget", f"{r}_{budgets[-1]}", f"{r}_{b}") for r in ("cnn", "vit", "fusion") for b in budgets[:-1]]
    diffs = [
        {"comparison": kind, "model_a": a, "model_b": b, **paired_auc_difference(y, scores[a], scores[b], n_boot, cfg["seed"])}
        for kind, a, b in pairs
        if a in scores and b in scores
    ]
    pd.DataFrame(diffs, columns=["comparison", "model_a", "model_b", "auc_difference", "ci_low", "ci_high"]).to_csv(
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


def report(cfg) -> pd.DataFrame:
    """Join dev CV, held-out metrics and measured cost into one table; pick the cheapest model that retains performance."""
    results = cfg.out("results")
    missing = [f for f in ("dev_cv_results.csv", "test_metrics.csv", "timings_summary.csv") if not (results / f).exists()]
    if missing:
        raise SystemExit(f"report needs {missing}: run train, evaluate and benchmark first")
    dev = pd.read_csv(results / "dev_cv_results.csv").reindex(
        columns=["name", "representation", "budget", "n_features", "best_C", "cv_auc", "cv_auc_std", "fit_s"])
    test = pd.read_csv(results / "test_metrics.csv").rename(columns={"model": "name"})[
        ["name", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "balanced_accuracy", "sensitivity", "specificity", "f1"]]
    cost = pd.read_csv(results / "timings_summary.csv").rename(columns={"model": "name"})[
        ["name", "device", "warm_total_s", "warm_encode_s", "cold_total_s", "peak_gpu_mb"]]
    table = dev.merge(test, on="name").merge(cost, on="name", how="left")
    table.to_csv(results / "cost_vs_auc.csv", index=False)

    # Decided on development CV AUC only; the test AUC of the chosen model is reported, never used to choose.
    margin = cfg.get("retain_margin", 0.02)
    mri = table[table.representation != "age"]
    best = mri.loc[mri.cv_auc.idxmax()]
    keep = mri[mri.cv_auc >= best.cv_auc - margin].dropna(subset=["warm_total_s"])
    if keep.empty:
        raise SystemExit("No timings for the retained models: run `python run.py benchmark`")
    cheap = keep.sort_values("warm_total_s", kind="stable").iloc[0]
    summary = {
        "rule": f"cheapest warm s/participant among MRI models with dev CV AUC >= best - {margin}",
        "best_dev_model": best["name"], "best_dev_cv_auc": float(best.cv_auc),
        "cheapest_retaining_model": cheap["name"], "its_dev_cv_auc": float(cheap.cv_auc),
        "its_warm_s_per_participant": float(cheap.warm_total_s),
        "its_test_auc": float(cheap.roc_auc), "its_test_auc_ci": [float(cheap.roc_auc_ci_low), float(cheap.roc_auc_ci_high)],
        "speedup_vs_best_dev_model": float(best.warm_total_s / cheap.warm_total_s) if pd.notna(best.warm_total_s) else None,
        "age_by_class": age_by_class(load_participants(cfg), results / "age_by_class.png"),
    }
    (results / "report.json").write_text(json.dumps(summary, indent=2))
    _plot_cost_vs_auc(table, cheap["name"], results / "cost_vs_auc.png")
    _plot_auc_vs_slices(mri, results / "auc_vs_slices.png")
    print(table[["name", "cv_auc", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "warm_total_s", "peak_gpu_mb", "fit_s"]]
          .round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return table


MARKERS = {"cnn": "o", "vit": "s", "fusion": "^"}


def _auc_err(df):
    return [df.roc_auc - df.roc_auc_ci_low, df.roc_auc_ci_high - df.roc_auc]


def _plot_cost_vs_auc(table, cheapest, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep, g in table[table.representation != "age"].groupby("representation"):
        ax.errorbar(g.warm_total_s, g.roc_auc, yerr=_auc_err(g), fmt=MARKERS.get(rep, "o"), capsize=3, label=rep)
        for r in g.itertuples():
            ax.annotate(f"{r.budget}", (r.warm_total_s, r.roc_auc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    age = table[table.representation == "age"]
    if not age.empty:
        ax.axhline(age.roc_auc.iloc[0], color="grey", ls="--", lw=1, label="age only")
    c = table[table.name == cheapest].iloc[0]
    ax.scatter(c.warm_total_s, c.roc_auc, s=250, facecolors="none", edgecolors="red", label=f"cheapest retaining: {cheapest}")
    ax.set(xscale="log", xlabel="Warm prediction time per participant (s, log scale)", ylabel="Held-out ROC-AUC (95% CI)",
           title="Accuracy vs cost (numbers = slices)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_auc_vs_slices(mri, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for rep, g in mri.sort_values("budget").groupby("representation"):
        a1.errorbar(g.budget, g.roc_auc, yerr=_auc_err(g), marker=MARKERS.get(rep, "o"), capsize=3, label=rep)
        a2.plot(g.budget, g.warm_total_s, marker=MARKERS.get(rep, "o"), label=rep)
    budgets = sorted(mri.budget.unique())
    a1.set(xticks=budgets, xlabel="Slices per participant", ylabel="Held-out ROC-AUC (95% CI)", title="How many slices are needed?")
    a2.set(xticks=budgets, xlabel="Slices per participant", ylabel="Warm s per participant", title="Cost per slice budget")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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
