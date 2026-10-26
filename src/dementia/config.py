"""Load YAML config; resolve paths against the project root; hash settings that invalidate caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"
FEATURE_DIMS = {"cnn": 512, "vit": 768}


class Config:
    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_CONFIG).resolve()
        self.raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))

    def __getitem__(self, key):
        return self.raw[key]

    def resolve(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (ROOT / p).resolve()

    @property
    def out_dir(self) -> Path:
        return self.resolve(self.raw["output_dir"])

    def out(self, *parts) -> Path:
        """Path under output_dir; its parent directory is created."""
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def slice_indices(self) -> list[int]:
        return [int(i) for i in self.raw["slice_indices"]]

    def budget_positions(self, budget: int) -> list[int]:
        return [int(i) for i in self.raw["slice_budgets"][int(budget)]]

    def budget_slice_indices(self, budget: int) -> list[int]:
        idx = self.slice_indices
        return [idx[p] for p in self.budget_positions(budget)]

    def device(self) -> str:
        dev = self.raw.get("device", "auto")
        if dev == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return dev

    def feature_hash(self, encoder: str) -> str:
        """Changes whenever anything that alters cached features changes."""
        g = self.raw["geometry"]
        payload = {
            "encoder": encoder,
            "weights": self.raw["encoders"][encoder],
            "preprocess": self.raw["preprocess"],
            "geometry": {k: g.get(k) for k in ("axial_axis", "rot90_k", "expected_shape")},
            "slice_indices": self.slice_indices,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def require_ready(self):
        """Geometry must be verified and 16 valid slice indices chosen before any extraction."""
        g = self.raw["geometry"]
        if not g.get("verified"):
            raise SystemExit("geometry.verified is false: run `dementia inspect`, check orientation, then set it true.")
        idx = self.slice_indices
        if len(idx) != 16 or len(set(idx)) != 16:
            raise SystemExit("slice_indices must contain 16 unique raw indices.")
        shape = g.get("expected_shape")
        if shape and not all(0 <= i < shape[g["axial_axis"]] for i in idx):
            raise SystemExit("slice_indices out of bounds for expected_shape.")


def load_config(path=None) -> Config:
    return Config(path)
