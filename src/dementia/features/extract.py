"""Per-participant feature extraction to outputs/features/<encoder>/<id>.npz; resumable, hash-validated."""
from __future__ import annotations

import time

import numpy as np

from dementia.config import FEATURE_DIMS
from dementia.data.manifest import load_participants
from dementia.features.encoders import build_encoder, encode
from dementia.imaging.preprocess import load_slices, to_model_input


def encode_path(cfg, encoder: str, image_path, indices=None, model=None):
    """Full chain for one scan: load -> preprocess -> frozen encoder. Returns (n, d) features."""
    device = cfg.device()
    model = model or build_encoder(encoder, cfg["encoders"][encoder], device)
    x = to_model_input(load_slices(image_path, cfg, indices), cfg["preprocess"]["size"])
    return encode(model, x, device, cfg["batch_size"])


def _compatible(path, cfg, encoder, subject_id) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return (
                str(z["config_hash"]) == cfg.feature_hash(encoder)
                and str(z["subject_id"]) == subject_id
                and z["features"].shape == (16, FEATURE_DIMS[encoder])
            )
    except Exception:
        return False


def extract(cfg, encoder: str, split: str = "dev", limit: int | None = None) -> None:
    cfg.require_ready()
    if split == "test" and not cfg.out("models", "lock.json").exists():
        raise SystemExit("Test features may be extracted only after `dementia train` has locked the run.")
    parts = load_participants(cfg, split)
    if limit:
        parts = parts.head(limit)
    device = cfg.device()
    t0 = time.perf_counter()
    model = build_encoder(encoder, cfg["encoders"][encoder], device)
    print(f"[{encoder}] encoder ready on {device} in {time.perf_counter() - t0:.1f}s")

    times = []
    for i, row in enumerate(parts.itertuples(), 1):
        out = cfg.out("features", encoder, f"{row.subject_id}.npz")
        if _compatible(out, cfg, encoder, row.subject_id):
            continue
        t = time.perf_counter()
        feats = encode_path(cfg, encoder, row.image_path, model=model)
        np.savez(
            out,
            features=feats,
            slice_indices=np.array(cfg.slice_indices),
            config_hash=cfg.feature_hash(encoder),
            subject_id=row.subject_id,
        )
        times.append(time.perf_counter() - t)
        print(f"[{encoder}] {i}/{len(parts)} {row.subject_id} {feats.shape} {times[-1]:.1f}s")
    if times:
        print(f"[{encoder}] median {np.median(times):.2f}s per participant over {len(times)} new extractions")
    else:
        print(f"[{encoder}] nothing to do: all {len(parts)} caches are up to date")
