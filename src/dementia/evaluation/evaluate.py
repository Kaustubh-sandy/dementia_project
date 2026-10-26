"""One-shot locked held-out evaluation; refuses to rerun."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from dementia.data.manifest import load_participants
from dementia.evaluation.bootstrap import bootstrap_ci, paired_auc_difference
from dementia.evaluation.metrics import METRICS, compute_metrics
from dementia.modeling.pooling import build_matrix, encoders_for, load_features


def evaluate(cfg) -> pd.DataFrame:
    notes_path = cfg.out("results", "evaluation_notes.json")
    if notes_path.exists():
        raise SystemExit("Held-out evaluation already done. Read outputs/results; do not re-tune on the test set.")
    if not cfg.out("models", "lock.json").exists():
        raise SystemExit("Run `dementia train` first (it locks the protocol).")

    test = load_participants(cfg, "test")
    y = test.label.to_numpy()
    experiments = json.loads(cfg.out("models", "experiments.json").read_text())
    n_boot = cfg["bootstrap"]["n"]

    cache, scores, rows, pred_cols = {}, {}, [], {"subject_id": test.subject_id, "label": y}
    for exp in experiments:
        bundle = joblib.load(cfg.out("models", exp["model_file"]))
        rep = bundle["representation"]
        if rep == "age":
            X = test[["age"]].to_numpy(dtype=float)
        else:
            for e in encoders_for(rep):
                cache.setdefault(e, load_features(cfg, e, test.subject_id))
            X = build_matrix(rep, bundle["positions"], cnn=cache.get("cnn"), vit=cache.get("vit"))
        s = bundle["pipeline"].predict_proba(X)[:, 1]
        scores[exp["name"]] = s
        pred_cols[f"score_{exp['name']}"] = s
        m = compute_metrics(y, s)
        ci = bootstrap_ci(y, s, n_boot, cfg["seed"])
        row = {"model": exp["name"], "representation": rep, "budget": exp["budget"], "n_test": len(y)}
        for k in METRICS:
            row[k] = m[k]
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = ci[k]
        row.update({k: m[k] for k in ("tp", "tn", "fp", "fn")})
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.out("results", "test_metrics.csv"), index=False)
    pd.DataFrame(pred_cols).to_csv(cfg.out("results", "test_predictions.csv"), index=False)

    diffs = []
    for e in experiments:
        if e["representation"] != "fusion":
            continue
        for other in ("cnn", "vit"):
            name = f"{other}_{e['budget']}"
            if name in scores:
                d = paired_auc_difference(y, scores[e["name"]], scores[name], n_boot, cfg["seed"])
                diffs.append({"model_a": e["name"], "model_b": name, **d})
    pd.DataFrame(diffs, columns=["model_a", "model_b", "auc_difference", "ci_low", "ci_high"]).to_csv(
        cfg.out("results", "paired_auc_differences.csv"), index=False
    )

    _roc_plot(cfg, y, scores)
    notes_path.write_text(
        json.dumps(
            {
                "n_test": int(len(y)),
                "n_positive": int(y.sum()),
                "threshold": 0.5,
                "bootstrap": f"{n_boot} stratified participant-level resamples, 95% percentile CI",
                "caveats": [
                    "Single dataset; no external validation.",
                    "Comparisons are descriptive, not corrected for multiple testing.",
                    "Similar point estimates do not prove equivalence.",
                    "CDR group is not a confirmed diagnosis.",
                ],
            },
            indent=2,
        )
    )
    print(table[["model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "balanced_accuracy"]].round(3).to_string(index=False))
    return table


def _roc_plot(cfg, y, scores):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC (participant level)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.out("results", "test_roc.png"), dpi=120)
    plt.close(fig)
