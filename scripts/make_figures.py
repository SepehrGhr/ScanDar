#!/usr/bin/env python
"""Render the report's figures from what the runs left on disk.

    python scripts/make_figures.py                      # every run in outputs/runs
    python scripts/make_figures.py --run enhance_baseline
    python scripts/make_figures.py --compare enhance_baseline enhance_sharp

Figures are regenerated from ``metrics.csv``, never drawn by hand, so a figure in
the report can always be traced back to the run that produced it.

Only the training curves so far — the graph the brief asks for by name *(brief
§3.2)*. The rest of the figure set (dataset sheets, the degradation strip, the
"spot the fake" panel, corner overlays, the end-to-end storyboard) is written as
the results it draws come into existence.
"""

import argparse

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.io import paths
from scandar.viz import compare_runs, training_curves

#: Patch-level PSNR of the degraded input on the frozen enhancement validation
#: set — the line a restoration model has to clear to be earning its parameters.
#: Measured once; recompute it if the generator changes.
VAL_INPUT_PSNR = 15.60


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", nargs="*", default=None, help="run names (default: all of them)")
    parser.add_argument("--compare", nargs="*", default=None, help="overlay these runs' PSNR")
    parser.add_argument("--out", default=None, help="default: reports/figures/enhancement")
    args = parser.parse_args()

    out_dir = paths.figures / "enhancement" if args.out is None else __import__(
        "pathlib"
    ).Path(args.out)

    names = args.run or sorted(p.name for p in paths.runs.iterdir() if (p / "metrics.csv").exists())
    if not names:
        raise SystemExit(f"no trained runs found in {paths.runs}")

    for name in names:
        written = training_curves(
            paths.run_dir(name), out_dir / f"{name}_curves.png", baseline_psnr=VAL_INPUT_PSNR
        )
        print(f"curves   : {written}")

    if args.compare:
        written = compare_runs(
            [paths.run_dir(name) for name in args.compare],
            out_dir / ("compare_" + "_vs_".join(args.compare) + ".png"),
            labels=list(args.compare),
        )
        print(f"compare  : {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
