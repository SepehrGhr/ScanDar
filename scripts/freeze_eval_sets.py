#!/usr/bin/env python
"""Generate the validation and test sets once, with a fixed seed, and store them.

    python scripts/freeze_eval_sets.py [--config configs/base.yaml] [--force]

The synthetic dataset invents a fresh sample on every ``__getitem__``, so without
this the validation curve would measure the dice as much as the model, and two
models could never be compared on the same images (brief §2.3).

Idempotent: re-running is free unless the requested count or seed has changed, or
``--force`` is given.
"""

import argparse

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.config import load_config
from scandar.io import paths
from scandar.prepare import freeze_eval_sets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(paths.repo / "configs" / "base.yaml"))
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")
    parser.add_argument("--seed", type=int, default=None, help="overrides data.split_seed")
    parser.add_argument("--force", action="store_true", help="regenerate even if up to date")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    freeze_eval_sets(config, seed=args.seed, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
