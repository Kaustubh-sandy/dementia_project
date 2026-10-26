"""Pipeline entry point.

    python run.py prepare                 # match scans to CDR labels, 80/20 participant split, CV folds
    python run.py qc                      # save slice montages of a few dev participants
    python run.py extract [--encoder cnn|vit] [--limit N]      # cache dev features
    python run.py train                   # CV-tune and fit all models, lock them
    python run.py extract --split test    # test features (only after train)
    python run.py evaluate                # one-shot held-out evaluation
    python run.py benchmark               # real per-scan prediction time
    python run.py predict --image PATH.hdr [--model cnn_16]
    python run.py all                     # prepare -> extract -> train -> extract test -> evaluate -> benchmark
"""
import argparse

from dementia.config import Config
from dementia.features import ENCODERS_FOR


def needed_encoders(cfg):
    return sorted({e for r in cfg["representations"] for e in ENCODERS_FOR[r]})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prepare", "qc", "extract", "train", "evaluate", "benchmark", "predict", "all"])
    p.add_argument("--config", default=None)
    p.add_argument("--encoder", choices=["cnn", "vit"], help="extract only this encoder (default: all needed)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int, help="extract only the first N participants (pilot timing)")
    p.add_argument("--image", help="predict: path to *_t88_masked_gfc.hdr")
    p.add_argument("--model", help="predict: model name, default = selected model")
    a = p.parse_args()
    cfg = Config(a.config)

    from dementia import data, evaluate, features, predict, train

    if a.step in ("prepare", "all"):
        data.prepare(cfg)
    if a.step == "qc":
        from dementia.preprocess import load_slices, save_montage

        for row in data.load_participants(cfg, "dev").head(5).itertuples():
            save_montage(load_slices(row.image_path, cfg), cfg["slice_indices"], cfg.out("qc", f"{row.subject_id}.png"), row.subject_id)
        print(f"QC montages written to {cfg.out('qc')}")
    if a.step in ("extract", "all"):
        for enc in [a.encoder] if a.encoder else needed_encoders(cfg):
            features.extract(cfg, enc, a.split, a.limit)
    if a.step in ("train", "all"):
        print(train.train(cfg).round(3).to_string(index=False))
    if a.step == "all":
        for enc in needed_encoders(cfg):
            features.extract(cfg, enc, "test")
    if a.step in ("evaluate", "all"):
        evaluate.evaluate(cfg)
    if a.step in ("benchmark", "all"):
        predict.benchmark(cfg)
    if a.step == "predict":
        if not a.image:
            p.error("predict needs --image")
        r = predict.predict_volume(a.image, cfg, predict.load_model(cfg, a.model))
        print(f"{r['model']}: {r['group']}  score={r['score']:.3f}  ({r['seconds']:.2f}s)")
        print("Research prediction only; not clinically validated and not a probability of dementia.")


if __name__ == "__main__":
    main()
