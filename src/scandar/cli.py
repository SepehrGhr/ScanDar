"""Command line entry point: ``scandar <command>``.

Data preparation, verification, training, evaluation and enhancement inference
all work. The end-to-end scanner does not exist yet — it needs the corner
detector — and says so plainly rather than failing with a traceback.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scandar",
        description="Document scanning and enhancement — phone photo in, clean scan out.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="cache the scans and write data/splits.json")
    prep.add_argument("--seed", type=int, default=1234, help="split seed (default: 1234)")
    prep.add_argument("--long-side", type=int, default=1600, help="cached scan long side in px")
    prep.add_argument("--force", action="store_true", help="rebuild the cache from scratch")

    freeze = sub.add_parser("freeze-eval", help="write the frozen synthetic evaluation sets")
    freeze.add_argument("--config", default=None, help="defaults to configs/base.yaml")
    freeze.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")
    freeze.add_argument("--seed", type=int, default=None, help="overrides data.split_seed")
    freeze.add_argument("--task", nargs="*", default=None, help="enhance, corner (default: both)")
    freeze.add_argument("--force", action="store_true", help="regenerate even if up to date")

    check = sub.add_parser("sanity", help="verify the data, the layout and the environment")
    check.add_argument("--strict", action="store_true", help="fail on checks that are only warnings")

    train = sub.add_parser("train", help="train a model from a config")
    train.add_argument("--config", required=True)
    train.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")

    ev = sub.add_parser("evaluate", help="score a trained model")
    ev.add_argument("--config", required=True)
    ev.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")

    enhance = sub.add_parser("enhance", help="restore an already-rectified page")
    enhance.add_argument("--input", required=True)
    enhance.add_argument("--output", required=True)
    enhance.add_argument("--checkpoint", required=True, help="a best.pt from a training run")
    enhance.add_argument("--max-side", type=int, default=None, help="cap the working resolution")

    scan = sub.add_parser("scan", help="photo in, clean scan out (bonus)")
    scan.add_argument("--input", required=True)
    scan.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "prepare-data":
        from . import prepare

        prepare.run(seed=args.seed, long_side=args.long_side, force=args.force)
        return 0

    if args.command == "freeze-eval":
        from .config import load_config
        from .io import paths
        from .prepare import freeze_eval_sets

        config_path = args.config or paths.repo / "configs" / "base.yaml"
        config = load_config(config_path, overrides=args.overrides)
        freeze_eval_sets(config, seed=args.seed, force=args.force, tasks=args.task)
        return 0

    if args.command == "sanity":
        from . import checks

        return 0 if checks.run(strict=args.strict) else 1

    if args.command == "train":
        from . import train as train_module

        forwarded = ["--config", args.config]
        if args.overrides:
            forwarded += ["--set", *args.overrides]
        return train_module.main(forwarded)

    if args.command == "evaluate":
        from . import evaluate as evaluate_module

        forwarded = ["--config", args.config]
        if args.overrides:
            forwarded += ["--set", *args.overrides]
        return evaluate_module.main(forwarded)

    if args.command == "enhance":
        from .pipelines import enhance_file

        written = enhance_file(args.input, args.output, args.checkpoint, max_side=args.max_side)
        print(f"wrote {written}")
        return 0

    if args.command == "scan":
        raise SystemExit(
            "The end-to-end scanner is not built yet — it needs the corner detector, "
            "which is not built either. The enhancement half works today: "
            "`scandar enhance --input page.jpg --output scan.png --checkpoint <best.pt>` "
            "takes an already-rectified page."
        )

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
