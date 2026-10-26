"""predict_volume(): single entry point used by the CLI and the Streamlit app."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np

from dementia.features.encoders import build_encoder, encode
from dementia.imaging.preprocess import load_slices, to_model_input
from dementia.modeling.pooling import encoders_for

GROUPS = {0: "CDR-zero group", 1: "CDR-positive group"}


def load_bundle(cfg, name: str | None = None) -> dict:
    mdir = cfg.out("models")
    if name is None:
        sel = mdir / "selected.json"
        if not sel.exists():
            raise SystemExit("No selected model: run `dementia train` first.")
        name = json.loads(sel.read_text())["name"]
    path = mdir / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    bundle = joblib.load(path)
    if bundle["representation"] == "age":
        raise SystemExit("age_only is a baseline, not an MRI model.")
    return bundle


def predict_volume(image_path, cfg, name: str | None = None, bundle: dict | None = None) -> dict:
    """MRI -> chosen slices -> frozen encoder(s) -> mean pool -> classifier. Never reads CDR or age."""
    t0 = time.perf_counter()
    bundle = bundle or load_bundle(cfg, name)
    device = cfg.device()
    slice_idx = bundle["slice_indices"]
    slices = load_slices(image_path, cfg, slice_idx)
    t_pre = time.perf_counter()

    x = to_model_input(slices, cfg["preprocess"]["size"])
    parts = []
    for enc in encoders_for(bundle["representation"]):
        model = build_encoder(enc, cfg["encoders"][enc], device)
        parts.append(encode(model, x, device, cfg["batch_size"]).mean(axis=0))
    t_enc = time.perf_counter()

    feat = np.concatenate(parts)[None, :]
    score = float(bundle["pipeline"].predict_proba(feat)[0, 1])
    t1 = time.perf_counter()
    label = int(score >= 0.5)
    return {
        "model": bundle["name"],
        "score": score,
        "label": label,
        "group": GROUPS[label],
        "slices": slices,
        "slice_indices": slice_idx,
        "seconds": t1 - t0,
        "seconds_preprocess": t_pre - t0,
        "seconds_encode": t_enc - t_pre,
        "seconds_classify": t1 - t_enc,
    }
