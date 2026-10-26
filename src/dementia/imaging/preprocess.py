"""Single deterministic preprocessing, shared by training, benchmark, CLI predict and the app."""
from __future__ import annotations

import numpy as np

from dementia.imaging.io import check_geometry, load_volume

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalise_volume(vol: np.ndarray, percentiles=(1, 99)) -> np.ndarray:
    """Clip to the brain-voxel percentiles of this volume and scale to [0,1]; background stays 0."""
    mask = vol > 0
    if not mask.any():
        raise ValueError("Empty brain mask")
    lo, hi = np.percentile(vol[mask], percentiles)
    if hi <= lo:
        raise ValueError("Degenerate intensity range")
    out = np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out


def extract_slices(vol: np.ndarray, indices, axis: int, rot90_k: int = 0) -> np.ndarray:
    """Axial planes at `indices` as an (n, H, W) array, rotated so anatomy is upright."""
    if not all(0 <= i < vol.shape[axis] for i in indices):
        raise ValueError(f"Slice index out of bounds for axis {axis} of size {vol.shape[axis]}")
    return np.stack([np.rot90(np.take(vol, i, axis=axis), rot90_k) for i in indices]).astype(np.float32)


def load_slices(path, cfg, indices=None) -> np.ndarray:
    """Load MRI -> validate -> normalise -> axial slices in [0,1]. Defaults to all 16 configured indices."""
    g = cfg["geometry"]
    vol, zooms = load_volume(path)
    check_geometry(vol, zooms, g)
    vol = normalise_volume(vol, cfg["preprocess"]["percentiles"])
    return extract_slices(vol, cfg.slice_indices if indices is None else indices, g["axial_axis"], g.get("rot90_k", 0))


def to_model_input(slices: np.ndarray, size: int = 224):
    """(n,H,W) in [0,1] -> float tensor (n,3,size,size): square pad, resize, 3 channels, ImageNet normalise."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(slices)).float().unsqueeze(1)
    h, w = t.shape[-2:]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    t = F.pad(t, (left, s - w - left, top, s - h - top))
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    t = t.clamp(0, 1).repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std
