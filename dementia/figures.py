"""Presentation figures: dataset examples, preprocessing, slice budgets, feature heatmaps (step: figures).

Writes only to <output_dir>/presentation/ and uses development participants only. Descriptive, not results.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dementia.data import load_participants
from dementia.features import load_features
from dementia.preprocess import IMAGENET_MEAN, IMAGENET_STD, extract_slices, load_volume, normalise_volume, to_model_input

GROUP = {0: "CDR 0", 1: "CDR > 0"}
MIDDLE = 89  # axial index shown when one slice is enough


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    _plt().close(fig)


def pick_pair(dev: pd.DataFrame) -> pd.DataFrame:
    """One CDR 0 and one CDR > 0 participant with the smallest age gap (ties: lowest IDs)."""
    neg = dev[dev.label == 0].sort_values("subject_id")
    pos = dev[dev.label == 1].sort_values("subject_id")
    gaps = np.abs(neg.age.to_numpy()[:, None] - pos.age.to_numpy()[None, :])
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    return pd.DataFrame([neg.iloc[i], pos.iloc[j]]).reset_index(drop=True)


def _title(r) -> str:
    return f"{r.subject_id}  |  age {int(r.age)}  |  recorded CDR {r.cdr:g}"


def mri_examples(cfg, pair, vols, out):
    plt, idx = _plt(), cfg["slice_indices"]
    fig, axes = plt.subplots(2, len(idx), figsize=(1.3 * len(idx), 3.4))
    for row, (r, vol) in enumerate(zip(pair.itertuples(), vols)):
        for col, (ax, s) in enumerate(zip(axes[row], extract_slices(vol, idx, cfg["axial_axis"], cfg["rot90_k"]))):
            ax.imshow(s, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 1:
                ax.set_xlabel(f"axial {idx[col]}", fontsize=7)
        axes[row][0].set_ylabel(GROUP[int(r.label)], fontsize=9)
        axes[row][len(idx) // 2].set_title(_title(r), fontsize=9)
    fig.suptitle("OASIS-1 skull-stripped, atlas-aligned MRI: the 16 axial slices the model uses", fontsize=11)
    _save(fig, out / "mri_examples.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6))
    for ax, r, vol in zip(axes, pair.itertuples(), vols):
        ax.imshow(extract_slices(vol, [MIDDLE], cfg["axial_axis"], cfg["rot90_k"])[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(_title(r), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Axial slice {MIDDLE} (ventricle level)", fontsize=11)
    _save(fig, out / "mri_examples_middle.png")


def preprocessing(cfg, r, out):
    plt = _plt()
    raw = load_volume(r.image_path, cfg["expected_shape"])
    norm = normalise_volume(raw)
    raw_s = extract_slices(raw, [MIDDLE], cfg["axial_axis"], cfg["rot90_k"])[0]
    norm_s = extract_slices(norm, [MIDDLE], cfg["axial_axis"], cfg["rot90_k"])
    x = to_model_input(norm_s, cfg["image_size"])[0, 0].numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1.3]})
    panels = [(raw_s, f"1. Raw scan voxels\n{raw_s.shape[0]}x{raw_s.shape[1]}, scanner units", None),
              (norm_s[0], "2. Brain-normalised\nclip to brain 1st-99th percentile, scale to [0, 1]", (0, 1)),
              (x, f"3. Model input\npadded to square, resized to {cfg['image_size']}x{cfg['image_size']}, x3 channels", (0, 1))]
    for ax, (img, title, lim) in zip(axes[0], panels):
        im = ax.imshow(img, cmap="gray", vmin=lim and lim[0], vmax=lim and lim[1])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    mask = raw > 0
    axes[1][0].hist(raw[mask], bins=100, color="grey")
    axes[1][0].set(title="Raw brain-voxel intensities", xlabel="scanner units")
    axes[1][1].hist(norm[mask], bins=100, color="tab:blue")
    axes[1][1].set(title="After normalisation", xlabel="[0, 1]")
    axes[1][2].axis("off")
    axes[1][2].text(0, 0.5, "Then: ImageNet mean/std normalisation\n(the values the frozen ResNet / ViT expect).\n"
                    "Shown image 3 is before that last step.", fontsize=9, va="center")
    fig.suptitle(f"Preprocessing of one slice (axial {MIDDLE}) — {_title(r)}", fontsize=11)
    _save(fig, out / "preprocessing_before_after.png")


def slice_budgets(cfg, r, vol, out):
    plt = _plt()
    budgets = sorted(cfg["slice_budgets"], key=int)
    n = max(len(cfg.budget_indices(b)) for b in budgets)
    fig, axes = plt.subplots(len(budgets), n + 1, figsize=(1.2 * (n + 1), 1.6 * len(budgets)))
    for row, b in enumerate(budgets):
        idx = cfg.budget_indices(b)
        for col in range(n):
            ax = axes[row][col + 1]
            ax.axis("off")
            if col < len(idx):
                ax.imshow(extract_slices(vol, [idx[col]], cfg["axial_axis"], cfg["rot90_k"])[0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(str(idx[col]), fontsize=7)
        # Sagittal reference with the chosen axial levels.
        sag = np.take(vol, vol.shape[0] // 2, axis=0).T  # rows = axial index
        ax = axes[row][0]
        ax.imshow(sag, cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
        for i in idx:
            ax.axhline(i, color="red", lw=0.6)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(f"{b} slices", fontsize=9)
    fig.suptitle(f"Slice budgets (4 is inside 8 is inside 16); left: sagittal view with the chosen levels — {r.subject_id}", fontsize=10)
    _save(fig, out / "slice_budgets.png")


def feature_figures(cfg, pair, dev, out):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plt = _plt()
    feats = {e: load_features(cfg, e, dev.subject_id) for e in ("cnn", "vit")}  # (N, 16, d)
    rows_of = {sid: i for i, sid in enumerate(dev.subject_id)}
    table = []
    for enc, d in (("cnn", 512), ("vit", 768)):
        f = feats[enc]
        mu, sd = f.reshape(-1, d).mean(0), f.reshape(-1, d).std(0) + 1e-8  # dev statistics per feature dimension
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        for ax, r in zip(axes, pair.itertuples()):
            m = f[rows_of[r.subject_id]]
            im = ax.imshow((m - mu) / sd, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
            ax.set_yticks(range(16), [str(i) for i in cfg["slice_indices"]], fontsize=6)
            ax.set_ylabel("axial slice")
            ax.set_title(_title(r), fontsize=9)
            pooled = m.mean(0)
            table.append({"subject_id": r.subject_id, "group": GROUP[int(r.label)], "encoder": enc, "per_scan_shape": f"{m.shape[0]}x{m.shape[1]}",
                          "pooled_dim": d, "mean": pooled.mean(), "std": pooled.std(), "min": pooled.min(), "max": pooled.max(),
                          **{f"f{k}": pooled[k] for k in range(5)}})
        axes[1].set_xlabel(f"feature dimension (0-{d - 1})")
        fig.colorbar(im, ax=axes, label="z-score vs development set")
        name = "ResNet-18 (CNN)" if enc == "cnn" else "ViT-B/16"
        fig.suptitle(f"{name} features: 16 slices x {d} numbers per participant (frozen encoder output)", fontsize=11)
        _save(fig, out / f"features_{enc}_heatmap.png")

    # What the fusion classifier sees: mean over slices, CNN then ViT.
    fused = np.concatenate([feats["cnn"].mean(1), feats["vit"].mean(1)], axis=1)  # (N, 1280)
    z = StandardScaler().fit_transform(fused)
    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow(z[[rows_of[s] for s in pair.subject_id]], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.axvline(511.5, color="black", lw=1.5)
    ax.set_yticks([0, 1], [f"{r.subject_id} ({GROUP[int(r.label)]})" for r in pair.itertuples()], fontsize=8)
    ax.set_xticks([256, 512 + 384], ["CNN part (512)", "ViT part (768)"])
    fig.colorbar(im, ax=ax, label="z-score")
    ax.set_title("Fusion vector = mean-pooled CNN (512) + ViT (768) = 1280 numbers per participant: the classifier's input", fontsize=10)
    _save(fig, out / "features_pooled.png")

    pca = PCA(2, random_state=cfg["seed"]).fit(z)
    xy = pca.transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in (0, 1):
        m = dev.label.to_numpy() == label
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.7, label=f"{GROUP[label]} (n={m.sum()})")
    ev = pca.explained_variance_ratio_ * 100
    ax.set(xlabel=f"PC1 ({ev[0]:.1f}% variance)", ylabel=f"PC2 ({ev[1]:.1f}% variance)",
           title="Development participants' fusion features, 2-D PCA (descriptive)")
    ax.legend()
    _save(fig, out / "features_pca.png")
    pd.DataFrame(table).round(4).to_csv(out / "features_table.csv", index=False)


NOTES = """Figures for the guide (development participants only; descriptive, not results).
mri_examples.png              the 16 axial slices the model uses, one CDR 0 and one CDR > 0 participant of similar age
mri_examples_middle.png       one larger slice (axial 89) for the same two people
preprocessing_before_after.png raw voxels -> brain-normalised [0,1] -> 224x224 model input, with intensity histograms
slice_budgets.png             which slices the 4 / 8 / 16 budgets use, with their levels on a sagittal view
features_cnn_heatmap.png      frozen ResNet-18 output: 16 slices x 512 numbers per person (z-scored per feature)
features_vit_heatmap.png      frozen ViT-B/16 output: 16 slices x 768 numbers per person (z-scored per feature)
features_pooled.png           mean over slices, CNN 512 + ViT 768 = the 1280-number fusion vector the classifier gets
features_pca.png              all development participants' fusion vectors in 2-D (PCA), coloured by CDR group
features_table.csv            shapes and summary numbers of the pooled vectors for the two example people
"""


def figures(cfg) -> None:
    out = cfg.out("presentation")
    out.mkdir(parents=True, exist_ok=True)
    dev = load_participants(cfg, "dev")
    pair = pick_pair(dev)
    vols = [normalise_volume(load_volume(p, cfg["expected_shape"])) for p in pair.image_path]
    mri_examples(cfg, pair, vols, out)
    preprocessing(cfg, pair.iloc[1], out)
    slice_budgets(cfg, pair.iloc[1], vols[1], out)
    feature_figures(cfg, pair, dev, out)
    (out / "README.txt").write_text(NOTES)
    ex = pair[["subject_id", "session_id", "age", "cdr", "label"]].to_dict("records")
    (out / "examples.json").write_text(json.dumps(ex, indent=2, default=float))
    print(f"[figures] examples: {ex}\n[figures] written to {out}")
