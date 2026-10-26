"""Train scaler + logistic regression for every representation x budget, plus age-only (step: train)."""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dementia.data import load_participants
from dementia.features import ENCODERS_FOR, build_matrix, load_features

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


def train(cfg) -> pd.DataFrame:
    lock = cfg.out("models", "lock.json")
    if lock.exists():
        raise SystemExit("Models are already locked for this output_dir. Change output_dir to start a new run.")
    dev = load_participants(cfg, "dev")
    y, folds = dev.label.to_numpy(), dev.fold.to_numpy()
    encoders = sorted({e for r in cfg["representations"] for e in ENCODERS_FOR[r]})
    feats = {e: load_features(cfg, e, dev.subject_id) for e in encoders}

    jobs = [(f"{r}_{b}", r, int(b), cfg.positions(b)) for r in cfg["representations"] for b in cfg["budgets"]]
    jobs.append(("age_only", "age", 0, None))
    results = []
    for name, rep, budget, pos in jobs:
        X = dev[["age"]].to_numpy(float) if rep == "age" else build_matrix(rep, pos, feats)
        search = fit_model(X, y, folds, cfg["C_grid"], cfg["seed"])
        joblib.dump(
            {"name": name, "representation": rep, "budget": budget, "positions": pos, "pipeline": search.best_estimator_},
            cfg.out("models", f"{name}.joblib"),
        )
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
            }
        )
        print(f"[train] {name:10s} CV AUC {results[-1]['cv_auc']:.3f} ± {results[-1]['cv_auc_std']:.3f}  C={results[-1]['best_C']}")

    best = select_best(results)
    table = pd.DataFrame(results)
    table.to_csv(cfg.out("results", "dev_cv_results.csv"), index=False)
    cfg.out("models", "experiments.json").write_text(json.dumps(results, indent=2))
    lock.write_text(json.dumps({"locked_at": datetime.now().isoformat(), "selected": best["name"]}, indent=2))
    print(f"\n[train] selected model for the app: {best['name']} (dev CV AUC {best['cv_auc']:.3f}; not a final estimate)")
    return table
