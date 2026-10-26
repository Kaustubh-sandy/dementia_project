"""Find *_t88_masked_gfc.hdr/.img pairs under data_root; write manifests/inventory.csv."""
from __future__ import annotations

import re

import pandas as pd

SESSION_RE = re.compile(r"(OAS1_\d{4}_MR\d+)")
COLUMNS = ["session_id", "image_path", "img_path", "status"]


def build_inventory(cfg) -> pd.DataFrame:
    root = cfg.resolve(cfg["data_root"])
    rows = []
    for hdr in sorted(root.glob(cfg["image_glob"])):
        m = SESSION_RE.search(hdr.name)
        img = hdr.with_suffix(".img")
        if not m:
            status = "no_session_id"
        elif not img.exists() or img.stat().st_size == 0:
            status = "missing_img"
        else:
            status = "ok"
        rows.append(
            {"session_id": m.group(1) if m else None, "image_path": str(hdr), "img_path": str(img), "status": status}
        )
    if not rows:
        raise SystemExit(f"No scans matched {cfg['image_glob']!r} under {root}. Check data_root/image_glob.")
    inv = pd.DataFrame(rows, columns=COLUMNS)
    ok = inv[inv.status == "ok"]
    dup = ok.session_id.duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"Duplicate scans for sessions: {sorted(ok[dup].session_id.unique())}")
    inv.to_csv(cfg.out("manifests", "inventory.csv"), index=False)
    return inv
