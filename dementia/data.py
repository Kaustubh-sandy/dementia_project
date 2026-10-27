"""Find scans, match clinical labels, apply cohort rules, split participants (step: prepare)."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SESSION_RE = re.compile(r"(OAS1_\d{4}_MR\d+)")
COLUMNS = ["subject_id", "session_id", "image_path", "age", "cdr", "label", "split", "fold"]


def find_scans(cfg) -> pd.DataFrame:
    rows = []
    for hdr in sorted(cfg.path(cfg["data_root"]).glob(cfg["image_glob"])):
        m = SESSION_RE.search(hdr.name)
        img = hdr.with_suffix(".img")
        if m and img.exists() and img.stat().st_size > 0:
            rows.append({"session_id": m.group(1), "image_path": str(hdr)})
        else:
            print(f"[prepare] skipping unusable scan {hdr}")
    scans = pd.DataFrame(rows, columns=["session_id", "image_path"])
    if scans.empty:
        raise SystemExit(f"No scans matched {cfg['image_glob']!r} under {cfg.path(cfg['data_root'])}")
    if scans.session_id.duplicated().any():
        raise SystemExit(f"Same session found twice: {scans[scans.session_id.duplicated()].session_id.tolist()}")
    return scans


def read_clinical(cfg) -> pd.DataFrame:
    path = cfg.path(cfg["clinical_file"])
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    out = pd.DataFrame(
        {
            "session_id": df["ID"].astype(str).str.strip(),
            "age": pd.to_numeric(df["Age"], errors="coerce"),
            "cdr": pd.to_numeric(df["CDR"], errors="coerce"),  # 'N/A' -> NaN
        }
    )
    if not out.session_id.str.fullmatch(r"OAS1_\d{4}_MR\d+").all() or out.session_id.duplicated().any():
        raise SystemExit("Clinical ID column must hold unique session IDs like OAS1_0001_MR1")
    return out


def select_cohort(meta: pd.DataFrame, scans: pd.DataFrame, min_age, valid_cdr):
    """Return (one row per eligible participant, list of (session_id, reason) exclusions)."""
    reason = pd.Series("", index=meta.index)
    reason[meta.age.isna() | (meta.age < min_age)] = "age_missing_or_below_min"
    reason[(reason == "") & ~meta.cdr.isin(valid_cdr)] = "cdr_missing_or_invalid"
    reason[(reason == "") & ~meta.session_id.isin(scans.session_id)] = "scan_not_found"
    excl = list(zip(meta.session_id[reason != ""], reason[reason != ""]))
    excl += [(s, "scan_without_clinical_row") for s in set(scans.session_id) - set(meta.session_id)]

    elig = meta[reason == ""].merge(scans, on="session_id")
    elig["subject_id"] = elig.session_id.str.rsplit("_MR", n=1).str[0]
    elig["visit"] = elig.session_id.str.rsplit("_MR", n=1).str[1].astype(int)
    elig = elig.sort_values(["subject_id", "visit"])
    later = elig.duplicated("subject_id")
    excl += [(s, "later_session") for s in elig.session_id[later]]
    elig = elig[~later].copy()
    elig["label"] = (elig.cdr > 0).astype(int)
    return elig, excl


def make_splits(df: pd.DataFrame, seed: int, test_size: float, cv_folds: int) -> pd.DataFrame:
    """Stratified participant split (dev/test) plus CV folds inside dev (test rows get fold -1)."""
    dev, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    k = min(cv_folds, int(dev["label"].value_counts().min()))
    out = df.copy()
    out["split"] = np.where(out.subject_id.isin(test.subject_id), "test", "dev")
    out["fold"] = -1
    dev = out[out.split == "dev"]
    for f, (_, val) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(dev, dev["label"])):
        out.loc[dev.index[val], "fold"] = f
    return out


def _md5(path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def prepare(cfg) -> pd.DataFrame:
    if cfg.out("models", "lock.json").exists():
        raise SystemExit("Training is locked for this output_dir; the split cannot change now.")
    elig, excl = select_cohort(read_clinical(cfg), find_scans(cfg), cfg["min_age"], cfg["valid_cdr"])
    if elig.image_path.str.replace(".hdr", ".img").map(_md5).duplicated().any():
        raise SystemExit("Two participants have byte-identical images")
    if len(elig) < 20 or elig.label.nunique() < 2:
        raise SystemExit(f"Only {len(elig)} eligible participants: extract more discs first")

    parts = make_splits(elig, cfg["seed"], cfg["test_size"], cfg["cv_folds"])[COLUMNS]
    parts = parts.sort_values("subject_id").reset_index(drop=True)
    parts.to_csv(cfg.out("participants.csv"), index=False)
    pd.DataFrame(excl, columns=["session_id", "reason"]).to_csv(cfg.out("exclusions.csv"), index=False)

    summary = {
        "participants": len(parts),
        "dev_cdr0": int(((parts.split == "dev") & (parts.label == 0)).sum()),
        "dev_cdr_pos": int(((parts.split == "dev") & (parts.label == 1)).sum()),
        "test_cdr0": int(((parts.split == "test") & (parts.label == 0)).sum()),
        "test_cdr_pos": int(((parts.split == "test") & (parts.label == 1)).sum()),
        "cv_folds": int(parts.fold.max()) + 1,
        "exclusions": pd.Series([r for _, r in excl]).value_counts().to_dict(),
        "age_by_class": age_by_class(parts, cfg.out("qc", "age_by_class.png")),
    }
    cfg.out("prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return parts


def age_by_class(parts: pd.DataFrame, plot_path=None) -> dict:
    """Descriptive age summary per split and class (age is a known confounder), plus a dev Mann-Whitney U test."""
    from scipy.stats import mannwhitneyu

    out = {}
    for (split, label), g in parts.groupby(["split", "label"]):
        out[f"{split}_{'cdr_pos' if label else 'cdr0'}"] = {
            "n": len(g), "mean": round(float(g.age.mean()), 1), "sd": round(float(g.age.std()), 1),
            "min": float(g.age.min()), "max": float(g.age.max()),
        }
    dev = parts[parts.split == "dev"]
    out["dev_mannwhitney_p"] = float(mannwhitneyu(dev.age[dev.label == 0], dev.age[dev.label == 1]).pvalue)
    if plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.arange(parts.age.min() - 0.5, parts.age.max() + 2.5, 2)
        for label, name in ((0, "CDR 0"), (1, "CDR > 0")):
            ax.hist(parts.age[parts.label == label], bins=bins, alpha=0.6, label=f"{name} (n={int((parts.label == label).sum())})")
        ax.set(xlabel="Age (years)", ylabel="Participants", title="Age by class (all included participants)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    return out


def load_participants(cfg, split: str | None = None) -> pd.DataFrame:
    path = cfg.out("participants.csv")
    if not path.exists():
        raise SystemExit("Run `python run.py prepare` first")
    df = pd.read_csv(path)
    return df if split is None else df[df.split == split].reset_index(drop=True)
