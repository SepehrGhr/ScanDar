"""Command line entry point: ``scandar <command>``.

Data preparation, verification, training, evaluation and all three inference
pipelines — ``enhance`` for a rectified page, ``detect`` for a raw photo's
corners, and ``scan``, which chains them: a photo in, a clean scan out, with
nobody clicking anything.
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

    detect = sub.add_parser("detect", help="find the four page corners in a raw photo")
    detect.add_argument("--input", required=True)
    detect.add_argument("--output", required=True, help="where to write the overlay")
    detect.add_argument("--checkpoint", required=True, help="a corner detector's best.pt")
    detect.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail loudly on a degenerate quad instead of falling back to Canny",
    )

    scan = sub.add_parser("scan", help="photo in, clean scan out (bonus)")
    scan.add_argument("--input", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--detector", default=None, help="a corner detector's best.pt")
    scan.add_argument("--enhancer", default=None, help="an enhancement network's best.pt")
    scan.add_argument(
        "--scanner",
        default=None,
        help="a fine-tuned end-to-end run's best.pt, which carries both halves",
    )
    scan.add_argument(
        "--warp",
        choices=("cv2", "torch"),
        default="cv2",
        help="which implementation flattens the page (they agree; torch is the differentiable one)",
    )
    scan.add_argument("--rectified", default=None, help="also write the flattened page here")
    scan.add_argument("--width", type=int, default=1024, help="rectified page width in px")
    scan.add_argument(
        "--keep-aspect",
        action="store_true",
        help="estimate the page shape from the quad instead of assuming A4",
    )
    scan.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail loudly on a degenerate quad instead of falling back to Canny",
    )

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

    if args.command == "detect":
        from .pipelines import detect_corners_file

        result = detect_corners_file(
            args.input, args.output, args.checkpoint, fallback=not args.no_fallback
        )
        corners = ", ".join(f"({x:.0f}, {y:.0f})" for x, y in result["corners"])
        print(f"corners via the {result['source']} path: {corners}")
        if result["problem"]:
            print(f"  the model's quad was rejected: {result['problem']}")
        print(f"wrote {result['written']}")
        return 0

    if args.command == "scan":
        from .pipelines import scan_file

        if not args.scanner and not (args.detector and args.enhancer):
            raise SystemExit(
                "give --scanner <a fine-tuned end-to-end run's best.pt>, or both --detector "
                "and --enhancer"
            )
        result = scan_file(
            args.input,
            args.output,
            args.detector,
            args.enhancer,
            scanner_checkpoint=args.scanner,
            save_rectified=args.rectified,
            out_width=args.width,
            aspect=None if args.keep_aspect else "a4",
            fallback=not args.no_fallback,
            warp=args.warp,
        )
        print(f"corners via the {result['source']} path, page flattened with {result['warp']}")
        if result["problem"]:
            print(f"  the model's quad was rejected: {result['problem']}")
        if result.get("written_rectified"):
            print(f"wrote {result['written_rectified']}")
        print(f"wrote {result['written']}")
        return 0

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
