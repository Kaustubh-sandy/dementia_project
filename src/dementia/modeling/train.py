"""StandardScaler + balanced L2 logistic regression on saved folds; nine experiments + age-only baseline."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dementia.data.manifest import load_participants
from dementia.imaging.slices import validate_budgets
from dementia.modeling.pooling import build_matrix, encoders_for, load_features
from dementia.modeling.select import select_best


def fit_model(X, y, fold, c_grid, seed=42):
    """GridSearchCV over C on the saved folds. Scaler is fitted inside each training fold only."""
    cv = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in np.unique(fold)]
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed)),
        ]
    )
    search = GridSearchCV(pipe, {"clf__C": list(c_grid)}, cv=cv, scoring="roc_auc", refit=True, n_jobs=1)
    search.fit(X, y)
    return search


def train(cfg, full: bool = False) -> pd.DataFrame:
    cfg.require_ready()
    validate_budgets(cfg)
    lock = cfg.out("models", "lock.json")
    if lock.exists():
        raise SystemExit(f"{lock} exists: this run is locked. Use a new output_dir for a different protocol.")

    mcfg = cfg["models"]["full" if full else "baseline"]
    dev = load_participants(cfg, "dev")
    y, fold = dev.label.to_numpy(), dev.fold.to_numpy()
    needed = sorted({e for r in mcfg["representations"] for e in encoders_for(r)})
    feats = {e: load_features(cfg, e, dev.subject_id) for e in needed}

    experiments, rows = [], []

    def record(name, rep, budget, positions, X):
        t = time.perf_counter()
        s = fit_model(X, y, fold, cfg["models"]["C_grid"], cfg["seed"])
        secs = time.perf_counter() - t
        res = pd.DataFrame(s.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
        res.columns = ["C", "cv_auc_mean", "cv_auc_std"]
        res.to_csv(cfg.out("results", f"cv_{name}.csv"), index=False)
        path = cfg.out("models", f"{name}.joblib")
        bundle = {
            "name": name,
            "representation": rep,
            "budget": budget,
            "positions": positions,
            "slice_indices": [cfg.slice_indices[p] for p in positions] if positions else None,
            "best_C": float(s.best_params_["clf__C"]),
            "pipeline": s.best_estimator_,
        }
        joblib.dump(bundle, path)
        exp = {
            "name": name,
            "representation": rep,
            "budget": budget,
            "best_C": bundle["best_C"],
            "cv_auc": float(s.best_score_),
            "cv_auc_std_at_best": float(res.loc[res.C == bundle["best_C"], "cv_auc_std"].iloc[0]),
            "n_features": int(X.shape[1]),
            "fit_seconds": round(secs, 3),
            "model_file": path.name,
        }
        experiments.append(exp)
        rows.append(exp)
        print(f"[train] {name}: CV AUC {exp['cv_auc']:.3f} (C={exp['best_C']}) in {secs:.1f}s")

    for rep in mcfg["representations"]:
        for b in mcfg["budgets"]:
            pos = cfg.budget_positions(b)
            X = build_matrix(rep, pos, cnn=feats.get("cnn"), vit=feats.get("vit"))
            record(f"{rep}_{b}", rep, int(b), pos, X)
    record("age_only", "age", 0, None, dev[["age"]].to_numpy(dtype=float))

    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    table.to_csv(cfg.out("results", "development_comparison.csv"), index=False)

    best = select_best(experiments)
    cfg.out("models", "experiments.json").write_text(json.dumps(experiments, indent=2))
    cfg.out("models", "selected.json").write_text(json.dumps(best, indent=2))

    import sklearn
    import torch

    parts_hash = hashlib.sha256(cfg.out("manifests", "participants.csv").read_bytes()).hexdigest()
    lock.write_text(
        json.dumps(
            {
                "locked_at": datetime.now(timezone.utc).isoformat(),
                "mode": "full" if full else "baseline",
                "feature_hashes": {e: cfg.feature_hash(e) for e in needed},
                "participants_sha256": parts_hash,
                "selected": best["name"],
                "versions": {
                    "python": platform.python_version(),
                    "sklearn": sklearn.__version__,
                    "torch": torch.__version__,
                },
            },
            indent=2,
        )
    )
    print(f"\nSelected for the app (highest development CV AUC): {best['name']}")
    print("CV AUC is a development estimate, not final performance. Locked; next: extract test features, then evaluate.")
    return table
