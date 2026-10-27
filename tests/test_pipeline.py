"""Synthetic-data checks of the pipeline logic. Never report these numbers as project results."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from dementia.config import ROOT, Config
from dementia.data import make_splits, select_cohort
from dementia.evaluate import bootstrap_ci, compute_metrics, evaluate, paired_auc_difference, report
from dementia.features import build_matrix, pool
from dementia.preprocess import extract_slices, normalise_volume
from dementia.train import check_lock, fit_model, select_best, train


def test_split_is_disjoint_stratified_and_deterministic():
    df = pd.DataFrame({"subject_id": [f"S{i}" for i in range(60)], "label": np.tile([0, 1], 30)})
    out = make_splits(df, 42, 0.2, 5)
    dev, test = out[out.split == "dev"], out[out.split == "test"]
    assert set(dev.subject_id).isdisjoint(test.subject_id)
    assert (test.fold == -1).all() and set(dev.fold) == set(range(5))
    assert all(g.label.nunique() == 2 for _, g in dev.groupby("fold"))
    assert out.equals(make_splits(df, 42, 0.2, 5))


def test_cohort_rules():
    meta = pd.DataFrame(
        {
            "session_id": ["OAS1_0001_MR1", "OAS1_0001_MR2", "OAS1_0002_MR1", "OAS1_0003_MR1", "OAS1_0004_MR1", "OAS1_0005_MR1"],
            "age": [70, 70, 55, 80, 66, 90],
            "cdr": [0, 0, 0, np.nan, 0.5, 2],
        }
    )
    scans = pd.DataFrame({"session_id": meta.session_id[:5], "image_path": "x.hdr"})
    elig, excl = select_cohort(meta, scans, 60, [0, 0.5, 1, 2])
    assert elig.set_index("subject_id").label.to_dict() == {"OAS1_0001": 0, "OAS1_0004": 1}
    reasons = dict(excl)
    assert reasons["OAS1_0001_MR2"] == "later_session"
    assert reasons["OAS1_0002_MR1"] == "age_missing_or_below_min"
    assert reasons["OAS1_0003_MR1"] == "cdr_missing_or_invalid"
    assert reasons["OAS1_0005_MR1"] == "scan_not_found"


def test_normalise_and_slices():
    vol = np.zeros((10, 12, 14), np.float32)
    vol[2:8, 2:8, 2:8] = np.random.default_rng(0).uniform(100, 900, (6, 6, 6))
    out = normalise_volume(vol)
    assert out.min() == 0 and out.max() <= 1 and (out[vol == 0] == 0).all()
    assert extract_slices(out, [3, 5], axis=2, rot90_k=1).shape == (2, 12, 10)
    with pytest.raises(ValueError):
        extract_slices(out, [14], axis=2)


def test_pooling_and_fusion():
    feats = {"cnn": np.random.rand(5, 16, 512), "vit": np.random.rand(5, 16, 768)}
    pos = [0, 4, 11, 15]
    assert np.allclose(pool(feats["cnn"], pos), feats["cnn"][:, pos].mean(1))
    assert build_matrix("fusion", pos, feats).shape == (5, 1280)


def test_training_learns_synthetic_signal():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 50)
    X = rng.normal(size=(100, 20))
    X[:, 0] += 2 * y
    s = fit_model(X, y, np.arange(100) % 5, [0.01, 0.1, 1.0])
    assert s.best_score_ > 0.8 and "scale" in s.best_estimator_.named_steps


def test_select_best_prefers_simple_on_ties_and_ignores_age():
    rs = [
        {"name": "fusion_4", "representation": "fusion", "budget": 4, "cv_auc": 0.9},
        {"name": "cnn_16", "representation": "cnn", "budget": 16, "cv_auc": 0.9},
        {"name": "cnn_8", "representation": "cnn", "budget": 8, "cv_auc": 0.9},
        {"name": "age_only", "representation": "age", "budget": 0, "cv_auc": 0.99},
    ]
    assert select_best(rs)["name"] == "cnn_8"


def test_metrics_and_bootstrap():
    y = np.array([0] * 20 + [1] * 20)
    s = np.r_[np.linspace(0, 0.6, 20), np.linspace(0.4, 1, 20)]
    m = compute_metrics(y, s)
    assert m["tp"] + m["fn"] == 20 and m["tn"] + m["fp"] == 20
    lo, hi = bootstrap_ci(y, s, 200)["roc_auc"]
    assert lo <= m["roc_auc"] <= hi
    assert paired_auc_difference(y, s, 1 - s, 200)["auc_difference"] > 0


def test_config_suites_and_budgets():
    cfg = Config()
    b = {k: set(cfg.positions(k)) for k in cfg["slice_budgets"]}
    assert b[4] <= b[8] <= b[16] and len(set(cfg["slice_indices"])) == 16
    assert cfg.budget_indices(4) == [41, 65, 107, 131]
    assert cfg.experiments("review") == [("cnn", 16), ("vit", 16), ("fusion", 16)]
    assert len(cfg.experiments("full")) == 9
    assert cfg.encoders("cnn") == ["cnn"] and cfg.encoders("review") == ["cnn", "vit"]


@pytest.fixture
def synthetic_run(tmp_path):
    """A config whose output_dir holds a fake manifest and fake cached features (no MRI, no torch)."""
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw.update(output_dir=str(tmp_path / "out"), bootstrap=100)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    cfg = Config(tmp_path / "config.yaml")
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"subject_id": [f"OAS1_{i:04d}" for i in range(n)], "label": np.tile([0, 1], n // 2)})
    df["session_id"] = df.subject_id + "_MR1"
    df["image_path"] = "unused.hdr"
    df["age"] = rng.integers(60, 90, n)
    df["cdr"] = df.label * 0.5
    df = make_splits(df, 42, 0.2, 5)
    df.to_csv(cfg.out("participants.csv"), index=False)
    for enc, d in (("cnn", 512), ("vit", 768)):
        for sid, y in zip(df.subject_id, df.label):
            f = rng.normal(size=(16, d)).astype(np.float32)
            f[:, :5] += y  # weak synthetic signal
            np.savez(cfg.out("features", enc, f"{sid}.npz"), features=f, config_hash=cfg.feature_hash(enc))
    return cfg


def test_train_evaluate_end_to_end_and_lock(synthetic_run):
    cfg = synthetic_run
    table = train(cfg, "review")
    assert set(table.name) == {"cnn_16", "vit_16", "fusion_16", "age_only"}
    assert (table.fit_s > 0).all() and (table.refit_s > 0).all()
    with pytest.raises(SystemExit):
        train(cfg, "review")  # locked
    metrics = evaluate(cfg)
    assert len(metrics) == 4 and metrics.roc_auc.between(0, 1).all()
    assert (cfg.out("results", "paired_auc_differences.csv")).exists()
    with pytest.raises(SystemExit):
        evaluate(cfg)  # evaluate runs once
    cfg.out("participants.csv").write_text(cfg.out("participants.csv").read_text() + "\n")
    with pytest.raises(SystemExit):
        check_lock(cfg)  # edited manifest is rejected


def test_full_suite_budget_comparisons_and_report(synthetic_run):
    cfg = synthetic_run
    table = train(cfg, "full")
    assert len(table) == 10  # 3 representations x 3 budgets + age_only
    budgets = pd.read_csv(cfg.out("results", "dev_budget_table.csv"), index_col=0)
    assert budgets.shape == (3, 3)
    evaluate(cfg)
    diffs = pd.read_csv(cfg.out("results", "paired_auc_differences.csv"))
    assert set(diffs.comparison) == {"representation", "budget"}
    assert {("cnn_16", "cnn_4"), ("fusion_16", "fusion_8")} <= set(zip(diffs.model_a, diffs.model_b))

    # Fake measured cost: more slices and more encoders cost more. report must not need torch.
    mri = table[table.representation != "age"]
    cost = mri.budget * mri.representation.map({"cnn": 1.0, "vit": 2.0, "fusion": 3.0}) / 100
    pd.DataFrame({"model": mri.name, "representation": mri.representation, "budget": mri.budget, "device": "cpu",
                  "warm_total_s": cost, "warm_encode_s": cost, "peak_gpu_mb": np.nan, "cold_total_s": cost + 1}).to_csv(
        cfg.out("results", "timings_summary.csv"), index=False)
    out = report(cfg)
    assert len(out) == 10 and cfg.out("results", "cost_vs_auc.png").exists() and cfg.out("results", "auc_vs_slices.png").exists()
    rep = json.loads(cfg.out("results", "report.json").read_text())
    kept = mri[mri.cv_auc >= mri.cv_auc.max() - cfg["retain_margin"]]
    assert rep["cheapest_retaining_model"] == kept.name.iloc[int(np.argmin(cost[kept.index]))]
    assert "dev_mannwhitney_p" in rep["age_by_class"]


def test_model_input():
    torch = pytest.importorskip("torch")
    from dementia.preprocess import to_model_input

    x = to_model_input(np.random.rand(3, 208, 176).astype(np.float32))
    assert x.shape == (3, 3, 224, 224) and torch.isfinite(x).all()
