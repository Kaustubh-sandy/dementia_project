"""Three-plane montages, slice montages and geometry JSON for manual orientation verification."""
from __future__ import annotations

import json

import numpy as np

from dementia.data.manifest import load_participants
from dementia.imaging.io import load_volume
from dementia.imaging.preprocess import extract_slices, normalise_volume
from dementia.imaging.slices import coverage_profile, evenly_spaced, suggest_slice_range


def _plot_planes(vol, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    for a, k in zip(ax, range(3)):
        sl = np.take(vol, vol.shape[k] // 2, axis=k)
        a.imshow(sl.T, cmap="gray", origin="lower")
        a.set_title(f"fixing array axis {k} = {vol.shape[k] // 2}")
        a.set_xlabel(f"array axis {[i for i in range(3) if i != k][0]}")
        a.set_ylabel(f"array axis {[i for i in range(3) if i != k][1]}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _plot_montage(slices, indices, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 8, figsize=(16, 4.5))
    for a, s, i in zip(ax.ravel(), slices, indices):
        a.imshow(s, cmap="gray", vmin=0, vmax=1)
        a.set_title(str(i), fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def inspect_subject(cfg, subject_id: str | None = None) -> dict:
    parts = load_participants(cfg, "dev")
    row = parts[parts.subject_id == subject_id].iloc[0] if subject_id else parts.iloc[0]
    if row.split != "dev":
        raise SystemExit("Inspect development participants only.")
    vol, zooms = load_volume(row.image_path)
    sid = row.subject_id
    info = {
        "subject_id": sid,
        "shape": list(vol.shape),
        "zooms": list(zooms),
        "nonzero_fraction": float((vol > 0).mean()),
        "intensity_min_max": [float(vol.min()), float(vol.max())],
        "note": "Analyze orientation is unreliable; judge axes from the PNG, not from the header.",
    }
    cfg.out("qc", f"{sid}_geometry.json").write_text(json.dumps(info, indent=2))
    _plot_planes(vol, cfg.out("qc", f"{sid}_planes.png"), sid)
    g = cfg["geometry"]
    if cfg.slice_indices:
        sl = extract_slices(normalise_volume(vol), cfg.slice_indices, g["axial_axis"], g.get("rot90_k", 0))
        _plot_montage(sl, cfg.slice_indices, cfg.out("qc", f"{sid}_slices.png"))
    print(json.dumps(info, indent=2))
    print(f"Wrote QC images under {cfg.out('qc')}")
    return info


def coverage_report(cfg, n: int = 10) -> list[int]:
    """Label-blind slice-range suggestion from the first n development volumes."""
    parts = load_participants(cfg, "dev").head(n)
    axis = cfg["geometry"]["axial_axis"]
    profiles = [coverage_profile(load_volume(p)[0], axis) for p in parts.image_path]
    lo, hi = suggest_slice_range(profiles)
    idx = evenly_spaced(lo, hi, 16)
    print(f"Brain coverage consistent over axis-{axis} indices {lo}..{hi} ({n} dev volumes).")
    print("Candidate slice_indices (confirm against the QC montage before using):")
    print(idx)
    return idx
