"""Join inventory to clinical table, apply cohort rules, write participants/exclusions/preparation files."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from dementia.data.inventory import build_inventory
from dementia.data.splits import make_splits

PARTICIPANT_COLS = ["subject_id", "session_id", "image_path", "age", "cdr", "label", "split", "fold"]


def read_metadata(cfg) -> pd.DataFrame:
    path = cfg.resolve(cfg["metadata_file"])
    if not path.exists():
        raise SystemExit(f"Clinical table not found: {path}")
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    c = cfg["columns"]
    missing = [c[k] for k in ("id", "age", "cdr") if c[k] not in df.columns]
    if missing:
        raise SystemExit(f"Columns {missing} not in clinical table; found {list(df.columns)}")
    out = pd.DataFrame(
        {
            "session_id": df[c["id"]].astype(str).str.strip(),
            "age": pd.to_numeric(df[c["age"]], errors="coerce"),
            "cdr": pd.to_numeric(df[c["cdr"]], errors="coerce"),  # 'N/A' -> NaN
        }
    )
    if not out.session_id.str.fullmatch(r"OAS1_\d{4}_MR\d+").all():
        raise SystemExit("Expected original session IDs such as OAS1_0001_MR1 in the ID column.")
    if out.session_id.duplicated().any():
        raise SystemExit("Duplicate session IDs in the clinical table.")
    return out


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_cohort(meta: pd.DataFrame, inv_ok: pd.DataFrame, rules: dict):
    """Return (eligible one-row-per-participant frame, list of (session_id, reason))."""
    excl = []
    for sid in sorted(set(inv_ok.session_id) - set(meta.session_id)):
        excl.append((sid, "scan_without_clinical_row"))

    reasons = pd.Series("", index=meta.index)
    reasons[meta.age.isna()] = "age_missing"
    reasons[(reasons == "") & (meta.age < rules["min_age"])] = "age_below_min"
    reasons[(reasons == "") & ~meta.cdr.isin(rules["valid_cdr"])] = "cdr_missing_or_invalid"
    reasons[(reasons == "") & ~meta.session_id.isin(inv_ok.session_id)] = "scan_not_found"
    excl += list(zip(meta.session_id[reasons != ""], reasons[reasons != ""]))

    elig = meta[reasons == ""].merge(inv_ok[["session_id", "image_path"]], on="session_id")
    elig["subject_id"] = elig.session_id.str.rsplit("_MR", n=1).str[0]
    elig["mr"] = elig.session_id.str.rsplit("MR", n=1).str[1].astype(int)
    elig = elig.sort_values(["subject_id", "mr"])
    later = elig.duplicated("subject_id", keep="first")
    excl += [(s, "later_session") for s in elig.session_id[later]]
    elig = elig[~later].copy()
    elig["label"] = (elig.cdr > 0).astype(int)
    return elig, excl


def build_participants(cfg, force: bool = False) -> pd.DataFrame:
    target = cfg.out("manifests", "participants.csv")
    if target.exists() and not (force and not cfg.out("models", "lock.json").exists()):
        raise SystemExit(f"{target} exists. Use a new output_dir (or --force before training has locked the run).")

    meta = read_metadata(cfg)
    inv = build_inventory(cfg)
    inv_ok = inv[inv.status == "ok"]
    elig, excl = select_cohort(meta, inv_ok, cfg["cohort"])
    excl += [(s, "scan_files_unusable") for s in inv[inv.status != "ok"].session_id.dropna()]

    md5 = elig.image_path.str.replace(".hdr", ".img", regex=False).map(_md5)
    if md5.duplicated().any():
        raise SystemExit("Identical image bytes under different IDs: " + str(list(elig.subject_id[md5.duplicated(keep=False)])))
    if elig.label.nunique() < 2 or len(elig) < 20:
        raise SystemExit(f"Only {len(elig)} eligible participants / {elig.label.nunique()} class(es): extract more discs first.")

    parts = make_splits(elig, cfg["seed"], cfg["split"]["test_size"], cfg["split"]["cv_folds"])
    parts = parts[PARTICIPANT_COLS].sort_values("subject_id").reset_index(drop=True)
    assert parts.subject_id.is_unique and parts.notna().all().all()

    parts.to_csv(target, index=False)
    ex = pd.DataFrame(excl, columns=["session_id", "reason"])
    ex.to_csv(cfg.out("manifests", "exclusions.csv"), index=False)
    summary = {
        "participants": len(parts),
        "by_split_label": {f"{s}_label{l}": int(n) for (s, l), n in parts.groupby(["split", "label"]).size().items()},
        "cv_folds": int(parts.fold.max()) + 1,
        "exclusion_reasons": ex.reason.value_counts().to_dict(),
        "scans_found": int(len(inv_ok)),
    }
    cfg.out("manifests", "preparation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return parts


def load_participants(cfg, split: str | None = None) -> pd.DataFrame:
    path = cfg.out("manifests", "participants.csv")
    if not path.exists():
        raise SystemExit("Run `dementia prepare` first.")
    df = pd.read_csv(path)
    return df if split is None else df[df.split == split].reset_index(drop=True)
