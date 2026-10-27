"""Train scaler + logistic regression for each suite model plus age-only, then lock (step: train)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dementia.data import load_participants
from dementia.features import build_matrix, load_features

REP_RANK = {"cnn": 0, "vit": 1, "fusion": 2}


def fit_model(X, y, folds, c_grid, seed=42) -> GridSearchCV:
    """Tune C by ROC-AUC on the saved folds; the scaler is refitted inside every training fold."""
    cv = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in np.unique(folds)]
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]
    )
    return GridSearchCV(pipe, {"clf__C": c_grid}, cv=cv, scoring="roc_auc", refit=True).fit(X, y)


def select_best(results: list[dict]) -> dict:
    """Highest dev CV AUC among MRI models; ties prefer the simpler representation, then fewer slices."""
    mri = [r for r in results if r["representation"] in REP_RANK]
    return min(mri, key=lambda r: (-round(r["cv_auc"], 9), REP_RANK[r["representation"]], r["budget"]))


def lock_payload(cfg, encoders) -> dict:
    """Everything the held-out evaluation depends on; any change after training breaks the lock."""
    return {
        "participants_sha256": hashlib.sha256(cfg.out("participants.csv").read_bytes()).hexdigest(),
        "feature_hashes": {e: cfg.feature_hash(e) for e in encoders},
        "slice_budgets": {str(k): v for k, v in cfg["slice_budgets"].items()},
        "threshold": cfg["threshold"],
    }


def check_lock(cfg) -> dict:
    path = cfg.out("models", "lock.json")
    if not path.exists():
        raise SystemExit("Models are not trained/locked yet: run `python run.py train` first")
    lock = json.loads(path.read_text())
    if {k: lock[k] for k in lock_payload(cfg, lock["encoders"])} != lock_payload(cfg, lock["encoders"]):
        raise SystemExit("participants.csv, slices, geometry or threshold changed since training. Restore them.")
    return lock


def train(cfg, suite: str | None = None) -> pd.DataFrame:
    suite = suite or cfg["suite"]
    lock = cfg.out("models", "lock.json")
    if lock.exists():
        raise SystemExit("Models are already locked for this output_dir. Change output_dir to start a new run.")
    dev = load_participants(cfg, "dev")
    y, folds = dev.label.to_numpy(), dev.fold.to_numpy()
    encoders = cfg.encoders(suite)
    feats = {e: load_features(cfg, e, dev.subject_id) for e in encoders}

    jobs = [(f"{r}_{b}", r, b, cfg.positions(b)) for r, b in cfg.experiments(suite)]
    jobs.append(("age_only", "age", 0, None))
    results = []
    for name, rep, budget, pos in jobs:
        X = dev[["age"]].to_numpy(float) if rep == "age" else build_matrix(rep, pos, feats)
        t = time.perf_counter()
        search = fit_model(X, y, folds, cfg["C_grid"], cfg["seed"])
        fit_s = time.perf_counter() - t
        joblib.dump(
            {"name": name, "representation": rep, "budget": budget, "positions": pos, "pipeline": search.best_estimator_},
            cfg.out("models", f"{name}.joblib"),
        )
        pd.DataFrame(search.cv_results_).to_csv(cfg.out("results", f"cv_{name}.csv"), index=False)
        i = search.best_index_
        results.append(
            {
                "name": name,
                "representation": rep,
                "budget": budget,
                "n_features": X.shape[1],
                "best_C": search.best_params_["clf__C"],
                "cv_auc": float(search.best_score_),
                "cv_auc_std": float(search.cv_results_["std_test_score"][i]),
                "fit_s": fit_s,  # whole C search: grid x folds + final refit
                "refit_s": float(search.refit_time_),  # final fit on all development participants
            }
        )
        print(f"[train] {name:10s} CV AUC {results[-1]['cv_auc']:.3f} ± {results[-1]['cv_auc_std']:.3f}  C={results[-1]['best_C']}  fit {fit_s:.2f}s")

    best = select_best(results)
    table = pd.DataFrame(results)
    table.to_csv(cfg.out("results", "dev_cv_results.csv"), index=False)
    mri = table[table.representation != "age"]
    budgets = mri.assign(cv=mri.cv_auc.map("{:.3f}".format) + " ± " + mri.cv_auc_std.map("{:.3f}".format))
    budgets = budgets.pivot(index="representation", columns="budget", values="cv")
    budgets.to_csv(cfg.out("results", "dev_budget_table.csv"))
    print(f"\n[train] dev CV AUC by representation x slices:\n{budgets.to_string()}\n")
    cfg.out("models", "experiments.json").write_text(json.dumps(results, indent=2))
    lock.write_text(
        json.dumps(
            {
                "locked_at": datetime.now().isoformat(),
                "suite": suite,
                "selected": best["name"],
                "selection_rule": "highest dev CV AUC; ties: cnn < vit < fusion, then fewer slices",
                "encoders": encoders,
                "n_dev": len(dev),
                "cv_folds": int(np.unique(folds).size),
                **lock_payload(cfg, encoders),
            },
            indent=2,
        )
    )
    print(f"\n[train] suite={suite}; selected model for the app: {best['name']} (dev CV AUC {best['cv_auc']:.3f}; not a final estimate)")
    return table
