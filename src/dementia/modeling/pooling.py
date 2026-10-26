"""Mean-pool slice features per budget; build CNN / ViT / fusion (concat) matrices."""
from __future__ import annotations

import numpy as np

from dementia.config import FEATURE_DIMS


def load_features(cfg, encoder: str, subject_ids) -> np.ndarray:
    """(N,16,d) array in the order of `subject_ids`, matched by ID and validated against the config hash."""
    arrs, missing = [], []
    for sid in subject_ids:
        p = cfg.out("features", encoder, f"{sid}.npz")
        if not p.exists():
            missing.append(sid)
            continue
        with np.load(p, allow_pickle=False) as z:
            if str(z["config_hash"]) != cfg.feature_hash(encoder) or str(z["subject_id"]) != sid:
                raise SystemExit(f"Stale feature cache for {sid} ({encoder}); re-run extract.")
            arrs.append(z["features"])
    if missing:
        raise SystemExit(f"Missing {encoder} features for {len(missing)} participants, e.g. {missing[:3]}. Run extract.")
    out = np.stack(arrs)
    assert out.shape[1:] == (16, FEATURE_DIMS[encoder])
    return out


def pool(feats: np.ndarray, positions) -> np.ndarray:
    """Mean over the chosen slice positions: (N,16,d) -> (N,d). Also accepts (n_slices,d) -> (d,)."""
    positions = list(positions)
    return feats[..., positions, :].mean(axis=-2)


def build_matrix(rep: str, positions, cnn=None, vit=None) -> np.ndarray:
    if rep == "cnn":
        return pool(cnn, positions)
    if rep == "vit":
        return pool(vit, positions)
    if rep == "fusion":
        return np.concatenate([pool(cnn, positions), pool(vit, positions)], axis=-1)
    raise ValueError(f"Unknown representation {rep!r}")


def encoders_for(rep: str) -> list[str]:
    return {"cnn": ["cnn"], "vit": ["vit"], "fusion": ["cnn", "vit"]}[rep]
