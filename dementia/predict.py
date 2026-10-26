"""predict_volume(): the one prediction path used by the CLI, verify, benchmark and the app."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from dementia.config import ENCODERS_FOR
from dementia.data import load_participants
from dementia.features import _MODELS, build_matrix, encode, get_encoder, load_features
from dementia.preprocess import load_slices, to_model_input
from dementia.train import check_lock

GROUPS = {0: "CDR-zero group", 1: "CDR-positive group"}


def mri_models(cfg) -> list[str]:
    return [x["name"] for x in json.loads(cfg.out("models", "experiments.json").read_text()) if x["representation"] != "age"]


def load_model(cfg, name: str | None = None) -> dict:
    lock = check_lock(cfg)
    name = name or lock["selected"]
    if name not in mri_models(cfg):
        raise SystemExit(f"Unknown MRI model {name!r}; choose from {mri_models(cfg)}")
    return joblib.load(cfg.out("models", f"{name}.joblib"))


def predict_volume(image_path, cfg, model: dict) -> dict:
    """MRI -> budget's slices -> frozen encoder(s) -> mean pool -> classifier. Never reads CDR or age."""
    t0 = time.perf_counter()
    indices = cfg.budget_indices(model["budget"])
    slices = load_slices(image_path, cfg, indices)
    x = to_model_input(slices, cfg["image_size"])
    t1 = time.perf_counter()
    device = cfg.device()
    feat = np.concatenate([encode(get_encoder(e, device), x, device, cfg["batch_size"]).mean(0) for e in ENCODERS_FOR[model["representation"]]])
    t2 = time.perf_counter()
    score = float(model["pipeline"].predict_proba(feat[None])[0, 1])
    label = int(score >= cfg["threshold"])
    return {
        "model": model["name"], "score": score, "label": label, "group": GROUPS[label],
        "slices": slices, "slice_indices": indices,
        "seconds": time.perf_counter() - t0, "seconds_preprocess": t1 - t0, "seconds_encode": t2 - t1,
    }


def verify_prediction(cfg, tolerance: float = 1e-4) -> dict:
    """Live prediction on a dev scan must match the prediction from its cached training features."""
    model = load_model(cfg)
    row = load_participants(cfg, "dev").iloc[0]
    feats = {e: load_features(cfg, e, [row.subject_id]) for e in ENCODERS_FOR[model["representation"]]}
    cached = float(model["pipeline"].predict_proba(build_matrix(model["representation"], model["positions"], feats))[0, 1])
    live = predict_volume(row.image_path, cfg, model)["score"]
    result = {"subject_id": row.subject_id, "model": model["name"], "cached_score": cached, "live_score": live,
              "absolute_difference": abs(live - cached), "tolerance": tolerance, "passed": abs(live - cached) <= tolerance}
    cfg.out("results", "prediction_consistency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Live and cached predictions disagree: preprocessing or model mismatch. Fix before the demo.")
    return result


def benchmark(cfg, repeats: int = 3) -> pd.DataFrame:
    """Real (uncached) per-scan prediction time for every trained MRI model: one cold run, then warm runs."""
    path = load_participants(cfg, "dev").image_path.iloc[0]
    device = cfg.device()
    rows = []
    for name in mri_models(cfg):
        model = load_model(cfg, name)
        for r in range(repeats):
            if r == 0:
                _MODELS.clear()  # cold run includes loading the encoder(s)
            t = time.perf_counter()
            res = predict_volume(path, cfg, model)
            if device == "cuda":
                import torch

                torch.cuda.synchronize()
            rows.append({"model": name, "run": "cold" if r == 0 else "warm", "device": device, "slices": len(res["slice_indices"]),
                         "total_s": time.perf_counter() - t, "preprocess_s": res["seconds_preprocess"], "encode_s": res["seconds_encode"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out("results", "timings.csv"), index=False)
    print(df.groupby(["model", "run"])[["total_s", "encode_s"]].median().round(3).to_string())
    return df
