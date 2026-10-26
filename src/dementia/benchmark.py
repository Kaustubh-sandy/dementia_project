"""Time uncached load/preprocess/encode/predict per representation and slice budget; log hardware."""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import psutil

from dementia.data.manifest import load_participants
from dementia.features.encoders import build_encoder, clear_encoder_cache, encode
from dementia.imaging.preprocess import load_slices, to_model_input
from dementia.modeling.pooling import encoders_for
from dementia.predict import load_bundle, predict_volume


def _sync(device):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def benchmark(cfg, subjects: int = 2, repeats: int = 2) -> None:
    cfg.require_ready()
    import sklearn
    import torch

    device = cfg.device()
    paths = load_participants(cfg, "dev").image_path.head(subjects).tolist()
    experiments = json.loads(cfg.out("models", "experiments.json").read_text())
    mri = [e for e in experiments if e["representation"] != "age"]
    proc = psutil.Process()
    enc_rows, e2e_rows = [], []

    for enc in sorted({e for x in mri for e in encoders_for(x["representation"])}):
        t = time.perf_counter()
        model = build_encoder(enc, cfg["encoders"][enc], device)
        _sync(device)
        enc_rows.append(dict(encoder=enc, budget="", stage="model_load", repeat=0, seconds=time.perf_counter() - t))
        for b in sorted({x["budget"] for x in mri}):
            idx = cfg.budget_slice_indices(b)
            encode(model, to_model_input(load_slices(paths[0], cfg, idx)), device, cfg["batch_size"])  # warm-up
            for p in paths:
                for r in range(repeats):
                    t0 = time.perf_counter()
                    sl = load_slices(p, cfg, idx)
                    x = to_model_input(sl, cfg["preprocess"]["size"])
                    t1 = time.perf_counter()
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    encode(model, x, device, cfg["batch_size"])
                    _sync(device)
                    t2 = time.perf_counter()
                    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
                    for stage, s in (("load_preprocess", t1 - t0), ("encode", t2 - t1)):
                        enc_rows.append(
                            dict(
                                encoder=enc, budget=b, stage=stage, repeat=r, seconds=s, device=device,
                                batch_size=cfg["batch_size"], peak_gpu_mb=peak,
                                process_rss_mb=proc.memory_info().rss / 2**20,
                            )
                        )
    for x in mri:
        bundle = load_bundle(cfg, x["name"])
        for r in range(repeats):
            if r == 0:
                clear_encoder_cache()  # cold: includes encoder loading
            res = predict_volume(paths[0], cfg, bundle=bundle)
            e2e_rows.append(
                dict(model=x["name"], run="cold" if r == 0 else "warm", seconds=res["seconds"],
                     preprocess=res["seconds_preprocess"], encode=res["seconds_encode"], device=device)
            )

    enc_df, e2e_df = pd.DataFrame(enc_rows), pd.DataFrame(e2e_rows)
    enc_df.to_csv(cfg.out("results", "encoder_timings.csv"), index=False)
    e2e_df.to_csv(cfg.out("results", "end_to_end_timings.csv"), index=False)
    cfg.out("results", "benchmark_environment.json").write_text(
        json.dumps(
            {
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "cpu_count": psutil.cpu_count(),
                "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "subjects": subjects,
                "repeats": repeats,
                "note": "process_rss_mb is resident memory after a run, not peak RAM.",
            },
            indent=2,
        )
    )
    print(enc_df[enc_df.stage == "encode"].groupby(["encoder", "budget"]).seconds.median().round(3).to_string())
    print(e2e_df.groupby(["model", "run"]).seconds.median().round(3).to_string())
