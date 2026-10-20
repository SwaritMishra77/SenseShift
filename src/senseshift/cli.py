"""Console entry point for acquisition, windowing, training and report creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import TRANSITION_SECONDS, acquire, build
from .experiment import DEFAULT_MODELS, DEFAULT_SEEDS, run


def main():
    p = argparse.ArgumentParser(prog="senseshift")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("acquire", help="verify UCI archive and extract original accelerometer files")
    a.add_argument("--archive", type=Path, default=Path("data/wisdm-uci-507.zip"))
    a.add_argument("--raw", type=Path, default=Path("data/raw"))
    a.add_argument("--download", action="store_true")
    b = sub.add_parser("prepare", help="build nonoverlapping 10-second windows")
    b.add_argument("--raw", type=Path, default=Path("data/raw"))
    b.add_argument("--cache", type=Path, default=Path("data/processed/windows.npz"))
    b.add_argument("--transition-seconds", type=float, default=TRANSITION_SECONDS,
                   help="skip this many seconds after each continuous activity starts")
    e = sub.add_parser("run", help="run subject-held-out experiments")
    e.add_argument("--cache", type=Path, default=Path("data/processed/windows.npz"))
    e.add_argument("--output", type=Path, default=Path("artifacts/results.json"))
    e.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    e.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    e.add_argument("--epochs", type=int, default=12)
    e.add_argument("--save-models", action="store_true")
    r = sub.add_parser("report", help="render local HTML and charts from measured results")
    r.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
    r.add_argument("--out", type=Path, default=Path("docs/index.html"))
    u = sub.add_parser("audit", help="write data-quality and held-out contrast audit")
    sub.add_parser("export-models", help="save and verify the seed-11 forest checkpoints")
    x = sub.add_parser("replay", help="predict one held-out window with a saved seed-11 model")
    x.add_argument("--model", choices=("rf", "rf_raw", "watch_rf_fallback", "cnn", "transformer"),
                   default="watch_rf_fallback")
    x.add_argument("--device", choices=("phone", "watch"), default="watch")
    x.add_argument("--subject", type=int)
    x.add_argument("--window", type=int, default=0)
    args = p.parse_args()
    if args.command == "acquire":
        print(json.dumps(acquire(args.archive, args.raw, download=args.download), indent=2))
    elif args.command == "prepare":
        meta = build(args.raw, args.cache, transition_seconds=args.transition_seconds)
        print(json.dumps({k: v for k, v in meta.items() if k != "per_file"}, indent=2))
    elif args.command == "run":
        run(args.cache, args.output, seeds=args.seeds, models=args.models,
            max_epochs=args.epochs, save_models=args.save_models)
    elif args.command == "report":
        from .report import render
        render(args.results, args.out)
    elif args.command == "audit":
        from .audit import write_audits
        print(json.dumps(write_audits(), indent=2))
    elif args.command == "export-models":
        from .export import export_forests
        print(json.dumps(export_forests(), indent=2))
    elif args.command == "replay":
        from .inference import replay
        print(json.dumps(replay(model_name=args.model, device=args.device,
                                subject=args.subject, window=args.window), indent=2))


if __name__ == "__main__":
    main()
