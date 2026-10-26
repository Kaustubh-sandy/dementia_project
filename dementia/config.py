"""Load config.yaml; resolve paths; hash settings that invalidate feature caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIMS = {"cnn": 512, "vit": 768}
ENCODERS_FOR = {"cnn": ["cnn"], "vit": ["vit"], "fusion": ["cnn", "vit"]}


class Config(dict):
    def __init__(self, path=None):
        path = Path(path or ROOT / "config.yaml").resolve()
        super().__init__(yaml.safe_load(path.read_text(encoding="utf-8")))
        self.base = path.parent

    def path(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.base / p).resolve()

    def out(self, *parts) -> Path:
        """Path under output_dir; parent folder is created."""
        p = self.path(self["output_dir"]).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def positions(self, budget) -> list[int]:
        return [int(i) for i in self["slice_budgets"][int(budget)]]

    def budget_indices(self, budget) -> list[int]:
        return [self["slice_indices"][p] for p in self.positions(budget)]

    def experiments(self, suite: str | None = None) -> list[tuple[str, int]]:
        """(representation, budget) pairs of a suite; age_only is added by train()."""
        s = self["suites"][suite or self["suite"]]
        return [(r, int(b)) for r in s["representations"] for b in s["budgets"]]

    def encoders(self, suite: str | None = None) -> list[str]:
        return sorted({e for r, _ in self.experiments(suite) for e in ENCODERS_FOR[r]})

    def device(self) -> str:
        if self["device"] != "auto":
            return self["device"]
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def feature_hash(self, encoder: str) -> str:
        keys = ("expected_shape", "axial_axis", "rot90_k", "slice_indices", "image_size")
        payload = {"encoder": encoder, **{k: self[k] for k in keys}}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
