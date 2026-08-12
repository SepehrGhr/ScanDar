#!/usr/bin/env python
"""Cache the clean scans and write the train/val/test manifest.

    python scripts/prepare_data.py [--seed 1234] [--long-side 1600] [--force]

Idempotent: re-run it whenever new background photos land.
"""

import argparse

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.prepare import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1234, help="split seed (default: 1234)")
    parser.add_argument("--long-side", type=int, default=1600, help="cached scan long side in px")
    parser.add_argument("--force", action="store_true", help="rebuild the cache from scratch")
    args = parser.parse_args()

    run(seed=args.seed, long_side=args.long_side, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
