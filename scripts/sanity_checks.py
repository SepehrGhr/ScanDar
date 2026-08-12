#!/usr/bin/env python
"""Verify the environment, the data layout, the scan cache and the splits.

    python scripts/sanity_checks.py [--strict]

Exits non-zero if anything failed, so it can guard a training run.
"""

import argparse

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.checks import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()
    return 0 if run(strict=args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
