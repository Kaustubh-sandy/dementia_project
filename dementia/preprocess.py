"""One deterministic MRI -> model-input function, shared by training, evaluation, prediction and the app."""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_volume(path, expected_shape=None) -> np.ndarray:
    """Analyze .hdr (+ .img beside it) -> 3-D float32 array."""
    import nibabel as nib

    vol = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if expected_shape and tuple(vol.shape) != tuple(expected_shape):
        raise ValueError(f"{path}: shape {vol.shape} != expected {tuple(expected_shape)}")
    if not np.isfinite(vol).all():
        raise ValueError(f"{path}: non-finite voxels")
    return vol


def normalise_volume(vol: np.ndarray, percentiles=(1, 99)) -> np.ndarray:
    """Clip to this volume's brain-voxel 1st/99th percentiles, scale to [0,1]; background stays 0."""
    mask = vol > 0
    if not mask.any():
        raise ValueError("empty brain mask")
    lo, hi = np.percentile(vol[mask], percentiles)
    if hi <= lo:
        raise ValueError("degenerate intensity range")
    out = np.clip((vol - lo) / (hi - lo), 0, 1).astype(np.float32)
    out[~mask] = 0
    return out


def extract_slices(vol: np.ndarray, indices, axis: int, rot90_k: int = 0) -> np.ndarray:
    """(n, H, W) axial planes, rotated so anatomy is upright."""
    if not all(0 <= i < vol.shape[axis] for i in indices):
        raise ValueError("slice index out of bounds")
    return np.stack([np.rot90(np.take(vol, i, axis=axis), rot90_k) for i in indices]).astype(np.float32)


def load_slices(path, cfg, indices=None) -> np.ndarray:
    """MRI path -> normalised axial slices in [0,1] (default: all 16 configured indices)."""
    vol = normalise_volume(load_volume(path, cfg["expected_shape"]))
    return extract_slices(vol, indices or cfg["slice_indices"], cfg["axial_axis"], cfg["rot90_k"])


def to_model_input(slices: np.ndarray, size: int = 224):
    """(n,H,W) in [0,1] -> (n,3,size,size) tensor: pad to square, resize, 3 channels, ImageNet normalise."""
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


def save_montage(slices: np.ndarray, indices, path, title=""):
    """QC image of the selected slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 8 if len(slices) > 8 else len(slices)
    rows = int(np.ceil(len(slices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, s, i in zip(axes.ravel(), slices, indices):
        ax.imshow(s, cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(i), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
