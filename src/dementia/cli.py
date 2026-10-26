"""Subcommands: prepare, inspect, extract, train, evaluate, benchmark, predict."""
from __future__ import annotations

import argparse

from dementia.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="dementia", description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (default configs/base.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="build inventory, participants, split")
    s.add_argument("--force", action="store_true", help="rebuild manifests (only before training has locked the run)")

    s = sub.add_parser("inspect", help="QC images / geometry / slice-range suggestion")
    s.add_argument("--subject")
    s.add_argument("--coverage", type=int, metavar="N", help="suggest slice range from first N dev volumes")

    s = sub.add_parser("extract", help="frozen-encoder features, cached")
    s.add_argument("--encoder", required=True, choices=["cnn", "vit"])
    s.add_argument("--split", default="dev", choices=["dev", "test"])
    s.add_argument("--limit", type=int)

    s = sub.add_parser("train", help="CV training; locks the protocol")
    s.add_argument("--full", action="store_true", help="9 configs instead of CNN-16 baseline")

    sub.add_parser("evaluate", help="one-shot held-out evaluation")

    s = sub.add_parser("benchmark", help="measure real compute cost")
    s.add_argument("--subjects", type=int, default=2)
    s.add_argument("--repeats", type=int, default=2)

    s = sub.add_parser("predict", help="predict one scan")
    s.add_argument("--image", required=True, help="path to *_t88_masked_gfc.hdr")
    s.add_argument("--model", default=None)

    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "prepare":
        from dementia.data.manifest import build_participants

        build_participants(cfg, a.force)
    elif a.cmd == "inspect":
        from dementia.imaging.qc import coverage_report, inspect_subject

        if a.coverage:
            coverage_report(cfg, a.coverage)
        else:
            inspect_subject(cfg, a.subject)
    elif a.cmd == "extract":
        from dementia.features.extract import extract

        extract(cfg, a.encoder, a.split, a.limit)
    elif a.cmd == "train":
        from dementia.modeling.train import train

        print(train(cfg, a.full).round(3).to_string(index=False))
    elif a.cmd == "evaluate":
        from dementia.evaluation.evaluate import evaluate

        evaluate(cfg)
    elif a.cmd == "benchmark":
        from dementia.benchmark import benchmark

        benchmark(cfg, a.subjects, a.repeats)
    elif a.cmd == "predict":
        from dementia.predict import predict_volume

        r = predict_volume(a.image, cfg, a.model)
        print(f"model={r['model']}  group={r['group']}  score={r['score']:.3f}  time={r['seconds']:.2f}s")
        print("Research prediction only; the score is not a probability of dementia and not clinically validated.")


if __name__ == "__main__":
    main()
