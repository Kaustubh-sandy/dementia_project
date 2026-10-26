"""predict_volume(): the one prediction path used by the CLI, the benchmark and the app."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from dementia.data import load_participants
from dementia.features import ENCODERS_FOR, _MODELS, encode, get_encoder
from dementia.preprocess import load_slices, to_model_input

GROUPS = {0: "CDR-zero group", 1: "CDR-positive group"}


def load_model(cfg, name: str | None = None) -> dict:
    lock = cfg.out("models", "lock.json")
    if not lock.exists():
        raise SystemExit("No trained models: run `python run.py train` first")
    name = name or json.loads(lock.read_text())["selected"]
    model = joblib.load(cfg.out("models", f"{name}.joblib"))
    if model["representation"] == "age":
        raise SystemExit("age_only is a baseline, not an MRI model")
    return model


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
    label = int(score >= 0.5)
    return {
        "model": model["name"], "score": score, "label": label, "group": GROUPS[label],
        "slices": slices, "slice_indices": indices,
        "seconds": time.perf_counter() - t0, "seconds_preprocess": t1 - t0, "seconds_encode": t2 - t1,
    }


def benchmark(cfg, repeats: int = 3) -> pd.DataFrame:
    """Real (uncached) per-scan prediction time for every trained MRI model: one cold run, then warm runs."""
    path = load_participants(cfg, "dev").image_path.iloc[0]
    names = [x["name"] for x in json.loads(cfg.out("models", "experiments.json").read_text()) if x["representation"] != "age"]
    rows = []
    for name in names:
        model = load_model(cfg, name)
        for r in range(repeats):
            if r == 0:
                _MODELS.clear()  # cold run includes loading the encoder(s)
            t = time.perf_counter()
            res = predict_volume(path, cfg, model)
            if cfg.device() == "cuda":
                import torch

                torch.cuda.synchronize()
            rows.append({"model": name, "run": "cold" if r == 0 else "warm", "device": cfg.device(),
                         "total_s": time.perf_counter() - t, "preprocess_s": res["seconds_preprocess"], "encode_s": res["seconds_encode"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out("results", "timings.csv"), index=False)
    print(df.groupby(["model", "run"])[["total_s", "encode_s"]].median().round(3).to_string())
    return df
