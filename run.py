"""Pipeline entry point. Run the steps in this order:

    python run.py doctor                   # versions, CUDA, real CNN/ViT forward pass
    python run.py prepare                  # match scans to CDR labels, 80/20 participant split, CV folds
    python run.py audit                    # every dev scan loads and gives 16 usable slices (+ montages)
    python run.py extract --limit 2        # pilot: seconds per participant
    python run.py extract                  # cache dev features for the encoders the suite needs
    python run.py train [--suite review|full|cnn]   # CV-tune + fit all models, then lock
    python run.py audit --split test
    python run.py extract --split test     # test features (only after train)
    python run.py evaluate                 # one-shot held-out evaluation
    python run.py verify                   # live prediction == cached-feature prediction
    python run.py benchmark                # real per-scan prediction time
    python run.py predict --image PATH.hdr [--model cnn_16]
"""
import argparse

from dementia.config import Config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["doctor", "prepare", "audit", "extract", "train", "evaluate", "verify", "benchmark", "predict"])
    p.add_argument("--config", default=None, help="default: config.yaml next to run.py")
    p.add_argument("--suite", choices=["review", "full", "cnn"], help="train/extract: override `suite` in config")
    p.add_argument("--encoder", choices=["cnn", "vit"], help="extract: only this encoder (default: all the suite needs)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int, help="extract: only the first N participants (pilot timing)")
    p.add_argument("--repeats", type=int, default=3, help="benchmark repeats")
    p.add_argument("--image", help="predict: path to *_t88_masked_gfc.hdr (its .img must sit beside it)")
    p.add_argument("--model", help="predict: model name, default = selected model")
    a = p.parse_args()
    cfg = Config(a.config)

    if a.step == "doctor":
        from dementia.features import doctor

        doctor(cfg)
    elif a.step == "prepare":
        from dementia.data import prepare

        prepare(cfg)
    elif a.step == "audit":
        from dementia.preprocess import audit

        audit(cfg, a.split)
    elif a.step == "extract":
        from dementia.features import extract

        if a.split == "test":
            from dementia.train import check_lock

            encoders = check_lock(cfg)["encoders"]  # exactly what the locked models need
        else:
            encoders = cfg.encoders(a.suite)
        for enc in [a.encoder] if a.encoder else encoders:
            extract(cfg, enc, a.split, a.limit)
    elif a.step == "train":
        from dementia.train import train

        print(train(cfg, a.suite).round(3).to_string(index=False))
    elif a.step == "evaluate":
        from dementia.evaluate import evaluate

        evaluate(cfg)
    elif a.step == "verify":
        from dementia.predict import verify_prediction

        verify_prediction(cfg)
    elif a.step == "benchmark":
        from dementia.predict import benchmark

        benchmark(cfg, a.repeats)
    elif a.step == "predict":
        from dementia.predict import load_model, predict_volume

        if not a.image:
            p.error("predict needs --image")
        r = predict_volume(a.image, cfg, load_model(cfg, a.model))
        print(f"{r['model']}: {r['group']}  score={r['score']:.3f}  ({r['seconds']:.2f}s)")
        print("Research prediction only; not clinically validated and not a probability of dementia.")


if __name__ == "__main__":
    main()
