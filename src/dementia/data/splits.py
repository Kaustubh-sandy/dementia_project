"""Participant-level stratified 80/20 split and development CV folds; persisted, never regenerated."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_splits(df: pd.DataFrame, seed: int, test_size: float, cv_folds: int) -> pd.DataFrame:
    """Add `split` (dev/test) and `fold` (0..k-1 for dev, -1 for test). One row per participant."""
    assert df.subject_id.is_unique, "one row per participant required"
    dev, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    assert set(dev.subject_id).isdisjoint(set(test.subject_id))
    k = min(cv_folds, int(dev["label"].value_counts().min()))
    if k < 2:
        raise SystemExit("Too few participants in a class for cross-validation.")
    if k < cv_folds:
        print(f"[splits] reduced CV folds from {cv_folds} to {k} because of class counts")
    out = df.copy()
    out["split"] = np.where(out.subject_id.isin(test.subject_id), "test", "dev")
    out["fold"] = -1
    dev = out[out.split == "dev"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, val) in enumerate(skf.split(dev, dev["label"])):
        out.loc[dev.index[val], "fold"] = f
    return out
