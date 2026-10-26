"""Frozen ResNet-18 / ViT-B/16 encoders, cached per-participant slice features, pooling (step: extract)."""
from __future__ import annotations

import time

import numpy as np

from dementia.config import ENCODERS_FOR, FEATURE_DIMS
from dementia.data import load_participants
from dementia.preprocess import load_slices, to_model_input

_MODELS: dict = {}


def get_encoder(name: str, device: str):
    """ImageNet-pretrained encoder with its classification head removed; frozen and in eval mode."""
    if (name, device) in _MODELS:
        return _MODELS[(name, device)]
    from torch import nn
    from torchvision import models

    if name == "cnn":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()  # 512-d
    else:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()  # 768-d
    model.eval().requires_grad_(False).to(device)
    _MODELS[(name, device)] = model
    return model


def encode(model, x, device: str, batch_size: int) -> np.ndarray:
    import torch

    with torch.inference_mode():
        out = [model(x[i : i + batch_size].to(device)).float().cpu().numpy() for i in range(0, len(x), batch_size)]
    return np.concatenate(out)


def encode_scan(cfg, encoder: str, image_path, indices=None) -> np.ndarray:
    """One scan -> (n_slices, d) features."""
    device = cfg.device()
    x = to_model_input(load_slices(image_path, cfg, indices), cfg["image_size"])
    return encode(get_encoder(encoder, device), x, device, cfg["batch_size"])


def _cache_ok(path, cfg, encoder) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return str(z["config_hash"]) == cfg.feature_hash(encoder) and z["features"].shape == (16, FEATURE_DIMS[encoder])


def extract(cfg, encoder: str, split: str = "dev", limit: int | None = None) -> None:
    """Cache 16-slice features for every participant in `split`. Safe to interrupt and rerun."""
    if split == "test":
        from dementia.train import check_lock

        check_lock(cfg)  # test features only after the protocol is fixed
    parts = load_participants(cfg, split).head(limit) if limit else load_participants(cfg, split)
    t = time.perf_counter()
    get_encoder(encoder, cfg.device())
    print(f"[extract {encoder}] model loaded on {cfg.device()} in {time.perf_counter() - t:.1f}s")
    times = []
    for i, row in enumerate(parts.itertuples(), 1):
        path = cfg.out("features", encoder, f"{row.subject_id}.npz")
        if _cache_ok(path, cfg, encoder):
            continue
        t = time.perf_counter()
        feats = encode_scan(cfg, encoder, row.image_path)
        np.savez(path, features=feats, config_hash=cfg.feature_hash(encoder))
        times.append(time.perf_counter() - t)
        print(f"[extract {encoder}] {i}/{len(parts)} {row.subject_id} {feats.shape} {times[-1]:.2f}s")
    if times:
        print(f"[extract {encoder}] median {np.median(times):.2f}s per participant")
    else:
        print(f"[extract {encoder}] all {len(parts)} {split} caches already up to date")


def doctor(cfg) -> None:
    """Check the environment: versions, CUDA, and a real forward pass of both pretrained encoders."""
    import json
    import platform

    import sklearn
    import torch
    import torchvision

    info = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
            "device": cfg.device()}
    if info["device"] == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    for enc, dim in FEATURE_DIMS.items():
        t = time.perf_counter()
        out = encode(get_encoder(enc, info["device"]), torch.zeros(2, 3, 224, 224), info["device"], 2)
        assert out.shape == (2, dim) and np.isfinite(out).all(), f"{enc} forward pass failed"
        info[f"{enc}_check"] = f"ok, {dim}-d features, loaded+ran in {time.perf_counter() - t:.1f}s"
    cfg.out("environment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if not info["cuda_available"]:
        print("Running on CPU. For the GPU, install the CUDA build of torch/torchvision from pytorch.org.")


def load_features(cfg, encoder: str, subject_ids) -> np.ndarray:
    """(N, 16, d) in the given participant order."""
    arrs = []
    for sid in subject_ids:
        path = cfg.out("features", encoder, f"{sid}.npz")
        if not _cache_ok(path, cfg, encoder):
            raise SystemExit(f"Missing or stale {encoder} features for {sid}: run extract --encoder {encoder}")
        with np.load(path) as z:
            arrs.append(z["features"])
    return np.stack(arrs)


def pool(feats: np.ndarray, positions) -> np.ndarray:
    """Mean over the chosen slice positions: (N,16,d) -> (N,d)."""
    return feats[:, list(positions), :].mean(axis=1)


def build_matrix(rep: str, positions, feats: dict) -> np.ndarray:
    """CNN (512), ViT (768) or fusion = concat (1280), one row per participant."""
    return np.concatenate([pool(feats[e], positions) for e in ENCODERS_FOR[rep]], axis=1)
