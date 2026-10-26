"""Choose the 16 axial slice indices (development data only) and the nested 4/8 subsets."""
from __future__ import annotations

import numpy as np


def coverage_profile(vol: np.ndarray, axis: int) -> np.ndarray:
    """Fraction of brain (non-zero) voxels in each plane along `axis`."""
    other = tuple(i for i in range(3) if i != axis)
    return (vol > 0).mean(axis=other)


def suggest_slice_range(profiles: list[np.ndarray], min_frac: float = 0.5) -> tuple[int, int]:
    """Range where every subject's coverage is at least `min_frac` of the cohort-median peak coverage."""
    p = np.stack(profiles)
    keep = (p >= min_frac * np.median(p.max(axis=1))).all(axis=0)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("No slice range with consistent brain coverage")
    return int(idx[0]), int(idx[-1])


def evenly_spaced(lo: int, hi: int, n: int = 16) -> list[int]:
    idx = np.linspace(lo, hi, n).round().astype(int).tolist()
    if len(set(idx)) != n:
        raise ValueError("Range too small for unique slice indices")
    return idx


def validate_budgets(cfg) -> None:
    """Nested subsets: every 4-slice position is in the 8-slice set, which is within the 16."""
    b = {k: set(cfg.budget_positions(k)) for k in cfg["slice_budgets"]}
    ks = sorted(b)
    for small, big in zip(ks, ks[1:]):
        assert b[small] <= b[big], f"budget {small} is not nested in {big}"
    assert all(max(v) < 16 for v in b.values())
