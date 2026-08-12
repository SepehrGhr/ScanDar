"""Command line entry point: ``scandar <command>``.

Everything implemented so far is data preparation and verification; training,
evaluation and the scanner are not built yet and say so plainly rather than
failing with a traceback.
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

    check = sub.add_parser("sanity", help="verify the data, the layout and the environment")
    check.add_argument("--strict", action="store_true", help="fail on checks that are only warnings")

    train = sub.add_parser("train", help="train a model from a config")
    train.add_argument("--config", required=True)
    train.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")

    ev = sub.add_parser("evaluate", help="score a trained model")
    ev.add_argument("--config", required=True)
    ev.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")

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

    if args.command == "sanity":
        from . import checks

        return 0 if checks.run(strict=args.strict) else 1

    if args.command == "train":
        from . import train as train_module

        return train_module.main(["--config", args.config, *args.overrides])

    if args.command == "evaluate":
        from . import evaluate as evaluate_module

        return evaluate_module.main(["--config", args.config, *args.overrides])

    if args.command == "scan":
        raise SystemExit(
            "The end-to-end scanner is not built yet. Until it is, the two stages "
            "are separate: document enhancement and corner detection."
        )

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
