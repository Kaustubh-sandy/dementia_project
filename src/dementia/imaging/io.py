"""Load Analyze .hdr/.img with NiBabel; validate shape/zooms/finite values against config geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_volume(path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return (3-D float32 volume, voxel sizes). `path` is the .hdr; the .img must sit beside it."""
    import nibabel as nib

    path = Path(path)
    if not path.exists() or not path.with_suffix(".img").exists():
        raise FileNotFoundError(f"Need both {path.name} and its .img in the same folder: {path.parent}")
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {vol.shape}")
    if not np.isfinite(vol).all():
        raise ValueError("Volume contains non-finite values")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return vol, zooms


def check_geometry(vol: np.ndarray, zooms, geometry: dict) -> None:
    """Fail loudly if a volume is not on the verified common grid."""
    shape = geometry.get("expected_shape")
    if shape and tuple(vol.shape) != tuple(shape):
        raise ValueError(f"Shape {vol.shape} != expected {tuple(shape)}")
    z = geometry.get("expected_zooms")
    if z and not np.allclose(zooms, z, atol=1e-3):
        raise ValueError(f"Voxel size {zooms} != expected {tuple(z)}")
