"""Pick the deployment model by development CV AUC; ties prefer simpler representation and fewer slices."""
from __future__ import annotations

REP_RANK = {"cnn": 0, "vit": 1, "fusion": 2}


def select_best(experiments: list[dict]) -> dict:
    mri = [e for e in experiments if e["representation"] in REP_RANK]
    if not mri:
        raise ValueError("No MRI experiments to select from")
    return min(mri, key=lambda e: (-round(e["cv_auc"], 9), REP_RANK[e["representation"]], e["budget"]))
