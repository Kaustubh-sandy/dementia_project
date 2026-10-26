"""Frozen ResNet-18 (512-d) and ViT-B/16 (768-d) with heads removed, eval mode, no grad."""
from __future__ import annotations

_CACHE: dict = {}


def build_encoder(name: str, weights_name: str, device: str):
    import torch
    from torch import nn
    from torchvision import models

    key = (name, weights_name, device)
    if key in _CACHE:
        return _CACHE[key]
    if name == "cnn":
        model = models.resnet18(weights=getattr(models.ResNet18_Weights, weights_name))
        model.fc = nn.Identity()
    elif name == "vit":
        model = models.vit_b_16(weights=getattr(models.ViT_B_16_Weights, weights_name))
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown encoder {name!r}")
    model.eval().requires_grad_(False)
    model.to(torch.device(device))
    _CACHE[key] = model
    return model


def clear_encoder_cache():
    _CACHE.clear()


def encode(model, x, device: str, batch_size: int = 4):
    """x: (n,3,224,224) tensor -> (n,d) float32 numpy array."""
    import numpy as np
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            out.append(model(x[i : i + batch_size].to(device)).float().cpu().numpy())
    feats = np.concatenate(out).astype(np.float32)
    assert np.isfinite(feats).all(), "non-finite features"
    return feats
