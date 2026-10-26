"""Synthetic-data tests only (leakage, split disjointness, shapes, pooling). Never report as accuracy."""
import numpy as np
import pandas as pd
import pytest

from dementia.data.manifest import select_cohort
from dementia.data.splits import make_splits
from dementia.evaluation.bootstrap import bootstrap_ci, paired_auc_difference
from dementia.evaluation.metrics import compute_metrics
from dementia.imaging.preprocess import extract_slices, normalise_volume
from dementia.modeling.pooling import build_matrix, pool
from dementia.modeling.select import select_best
from dementia.modeling.train import fit_model


def _participants(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"subject_id": [f"S{i}" for i in range(n)], "label": rng.integers(0, 2, n)})


def test_split_disjoint_stratified_and_folds_cover_dev():
    df = make_splits(_participants(), seed=42, test_size=0.2, cv_folds=5)
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    assert set(dev.subject_id).isdisjoint(test.subject_id)
    assert (test.fold == -1).all() and set(dev.fold) == set(range(5))
    for _, g in dev.groupby("fold"):
        assert g.label.nunique() == 2
    assert df.equals(make_splits(_participants(), 42, 0.2, 5))  # deterministic


def test_cohort_rules():
    meta = pd.DataFrame(
        {
            "session_id": ["OAS1_0001_MR1", "OAS1_0001_MR2", "OAS1_0002_MR1", "OAS1_0003_MR1", "OAS1_0004_MR1", "OAS1_0005_MR1"],
            "age": [70, 70, 55, 80, 66, 90],
            "cdr": [0, 0, 0, np.nan, 0.5, 2],
        }
    )
    inv = pd.DataFrame({"session_id": meta.session_id[[0, 1, 2, 3, 4]], "image_path": "x.hdr"})
    elig, excl = select_cohort(meta, inv, {"min_age": 60, "valid_cdr": [0, 0.5, 1, 2]})
    assert sorted(elig.subject_id) == ["OAS1_0001", "OAS1_0004"]
    assert elig.set_index("subject_id").label.to_dict() == {"OAS1_0001": 0, "OAS1_0004": 1}
    reasons = dict(excl)
    assert reasons["OAS1_0001_MR2"] == "later_session"
    assert reasons["OAS1_0002_MR1"] == "age_below_min"
    assert reasons["OAS1_0003_MR1"] == "cdr_missing_or_invalid"
    assert reasons["OAS1_0005_MR1"] == "scan_not_found"


def test_normalise_keeps_background_zero_and_range():
    vol = np.zeros((10, 10, 10), np.float32)
    vol[2:8, 2:8, 2:8] = np.random.default_rng(0).uniform(100, 900, (6, 6, 6))
    out = normalise_volume(vol)
    assert out.min() == 0 and out.max() <= 1 and (out[vol == 0] == 0).all()
    with pytest.raises(ValueError):
        normalise_volume(np.zeros((4, 4, 4), np.float32))


def test_extract_slices_shape_and_bounds():
    vol = np.random.rand(8, 9, 10).astype(np.float32)
    assert extract_slices(vol, [1, 5], axis=2, rot90_k=1).shape == (2, 9, 8)
    with pytest.raises(ValueError):
        extract_slices(vol, [10], axis=2)


def test_pooling_and_fusion_shapes():
    cnn, vit = np.random.rand(5, 16, 512), np.random.rand(5, 16, 768)
    pos = [0, 4, 11, 15]
    assert np.allclose(pool(cnn, pos), cnn[:, pos].mean(axis=1))
    assert build_matrix("fusion", pos, cnn=cnn, vit=vit).shape == (5, 1280)
    assert pool(np.array([[2.0, 4.0], [4.0, 8.0]]), [0, 1]).tolist() == [3.0, 6.0]  # worked example in the guide


def test_select_best_ties_prefer_simple():
    exps = [
        {"name": "fusion_4", "representation": "fusion", "budget": 4, "cv_auc": 0.9},
        {"name": "cnn_16", "representation": "cnn", "budget": 16, "cv_auc": 0.9},
        {"name": "cnn_8", "representation": "cnn", "budget": 8, "cv_auc": 0.9},
        {"name": "age_only", "representation": "age", "budget": 0, "cv_auc": 0.99},
    ]
    assert select_best(exps)["name"] == "cnn_8"


def test_fit_model_learns_signal_with_fold_scaler():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 50)
    X = rng.normal(size=(100, 20))
    X[:, 0] += 2.0 * y
    fold = np.arange(100) % 5
    s = fit_model(X, y, fold, [0.001, 0.01, 0.1, 1.0])
    assert s.best_score_ > 0.8 and "scale" in s.best_estimator_.named_steps


def test_metrics_and_bootstrap():
    y = np.array([0] * 20 + [1] * 20)
    s = np.r_[np.linspace(0.0, 0.6, 20), np.linspace(0.4, 1.0, 20)]
    m = compute_metrics(y, s)
    assert m["tp"] + m["fn"] == 20 and m["tn"] + m["fp"] == 20 and 0.8 < m["roc_auc"] <= 1
    ci = bootstrap_ci(y, s, n=200)
    assert ci["roc_auc"][0] <= m["roc_auc"] <= ci["roc_auc"][1]
    d = paired_auc_difference(y, s, 1 - s, n=200)
    assert d["auc_difference"] > 0 and d["ci_low"] <= d["auc_difference"] <= d["ci_high"]


def test_budgets_are_nested_in_shipped_configs():
    from dementia.config import ROOT, load_config
    from dementia.imaging.slices import validate_budgets

    for name in ("base", "full"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        validate_budgets(cfg)
        cfg.require_ready()
        assert cfg.budget_slice_indices(4) == [41, 65, 107, 131]


def test_model_input_shape_and_normalisation():
    torch = pytest.importorskip("torch")
    from dementia.imaging.preprocess import to_model_input

    x = to_model_input(np.random.rand(3, 208, 176).astype(np.float32), 224)
    assert x.shape == (3, 3, 224, 224) and torch.isfinite(x).all()
    assert torch.allclose(x[:, 0] * 0.229 + 0.485, (x[:, 1] * 0.224 + 0.456), atol=1e-4)  # identical channels
