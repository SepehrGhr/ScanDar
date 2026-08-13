#!/usr/bin/env python
"""Put each model beside its dropout arm and let the table answer §6.

    python scripts/compare_dropout.py
    python scripts/compare_dropout.py --pair corner_reg corner_reg_dropout
    python scripts/compare_dropout.py --split val

The brief asks what dropout does to the gap between training and unseen data
*(brief §6)*. That is two numbers per run, not one: the metric on the test bucket,
and the distance between the training and test buckets — a regulariser that helps
is supposed to close the second even if it costs a little of the first.

Reads what ``evaluate.py`` already wrote into ``reports/tables/``, so nothing is
recomputed here and no number in this table can disagree with the per-run one.
Run ``evaluate.py`` on every arm first; arms with no table yet are skipped with a
note rather than failing, so this is usable while the runs are still going.

Writes ``reports/tables/dropout_study.csv`` and ``.md``.

A caution that belongs next to the output: the synthetic half of this comparison
is the half that can be measured today. The question the brief actually poses is
whether the **synthetic-to-real** gap shrinks, and the real-photo column needs the
reference scans and the corner annotations that are still owed. Those are scored
from these same checkpoints when they land — nothing here has to be retrained.
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.io import paths

#: (baseline, dropout arm). The wide-dropout arms are the placement sweep and may
#: not have been run; missing ones drop out of the table quietly.
DEFAULT_PAIRS = (
    ("enhance_realistic", "enhance_dropout"),
    ("enhance_realistic", "enhance_dropout_wide"),
    ("corner_reg", "corner_reg_dropout"),
    ("corner_heat", "corner_heat_dropout"),
    ("corner_heat", "corner_heat_dropout_wide"),
)

#: The metric each task is judged on, and which direction is better.
TASKS = {
    "restoration": {
        "metrics": [("psnr", "PSNR (dB)", "max", 2), ("ssim", "SSIM", "max", 4)],
        "variant": "enhanced",
    },
    "corners": {
        "metrics": [
            ("corner_err_px", "corner error (px @256)", "min", 2),
            ("pck", "PCK@2%", "max", 3),
            ("quad_iou", "quad IoU", "max", 4),
        ],
        "variant": "detector",
    },
}


def read_table(name: str, directory: Path | None = None) -> tuple[str, list[dict]] | None:
    """Find whichever table ``evaluate.py`` wrote for this run, or nothing."""
    directory = Path(directory or paths.tables)
    for suffix in TASKS:
        path = directory / f"{name}_{suffix}.csv"
        if path.exists():
            with open(path, newline="", encoding="utf-8") as handle:
                return suffix, list(csv.DictReader(handle))
    return None


def score(rows: list[dict], suffix: str, split: str) -> dict[str, float]:
    """The model's own row for one split — never the baseline row beside it."""
    wanted = TASKS[suffix]["variant"]
    for row in rows:
        if row["split"] == split and row["variant"] == wanted:
            return {key: float(row[key]) for key, *_ in TASKS[suffix]["metrics"]}
    raise SystemExit(f"no {split!r} {wanted!r} row in the table")


def compare(pairs, split: str = "Test", directory: Path | None = None) -> list[dict]:
    """One row per (model, arm), carrying the arm's score and its two deltas."""
    out = []
    for base_name, arm_name in pairs:
        base, arm = read_table(base_name, directory), read_table(arm_name, directory)
        if base is None:
            print(f"skipped  : {base_name} — no evaluation table yet")
            continue
        if arm is None:
            print(f"skipped  : {arm_name} — no evaluation table yet")
            continue
        suffix, base_rows = base
        arm_suffix, arm_rows = arm
        if arm_suffix != suffix:
            raise SystemExit(f"{base_name} and {arm_name} are not the same task")

        for key, label, direction, digits in TASKS[suffix]["metrics"]:
            base_split = score(base_rows, suffix, split)[key]
            arm_split = score(arm_rows, suffix, split)[key]
            base_train = score(base_rows, suffix, "Training")[key]
            arm_train = score(arm_rows, suffix, "Training")[key]
            # Signed so a positive gap always means "worse on data it did not
            # train on", whichever way the metric itself runs. That is the number
            # a regulariser is supposed to shrink, and reading it off a column
            # whose sign flips between rows is how a null result gets misread.
            base_gap = base_train - base_split if direction == "max" else base_split - base_train
            arm_gap = arm_train - arm_split if direction == "max" else arm_split - arm_train
            # "Better" is direction-dependent, so say it once here and let every
            # reader downstream read a sign instead of remembering which is which.
            improvement = arm_split - base_split if direction == "max" else base_split - arm_split
            out.append(
                {
                    "model": base_name,
                    "arm": arm_name,
                    "metric": label,
                    "baseline": round(base_split, digits),
                    "dropout": round(arm_split, digits),
                    "change": round(improvement, digits),
                    "better": "dropout" if improvement > 0 else "baseline",
                    # Signed the same way in both runs, so the two are comparable
                    # even where the metric's own direction is not.
                    "baseline_train_gap": round(base_gap, digits),
                    "dropout_train_gap": round(arm_gap, digits),
                }
            )
    return out


def markdown_table(rows, split: str) -> str:
    lines = [
        f"### Dropout, {split.lower()} split",
        "",
        "| Model | arm | metric | baseline | with dropout | change | train→test gap "
        "(baseline → dropout) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        sign = "+" if row["change"] > 0 else ""
        lines.append(
            f"| {row['model']} | {row['arm']} | {row['metric']} | {row['baseline']} "
            f"| {row['dropout']} | {sign}{row['change']} "
            f"| {row['baseline_train_gap']} → {row['dropout_train_gap']} |"
        )
    lines += [
        "",
        "`change` is signed so that **positive always means dropout did better**, whichever "
        "direction the metric runs in, and so is the train→test gap: **positive means worse on "
        "data the run did not train on**. That gap is the quantity a regulariser is supposed to "
        "shrink, and the one that was already near zero before any dropout was added.",
        "",
        "This is the synthetic half of the study. The real-photo column — the gap the brief "
        "actually asks about — is scored from these same checkpoints once the reference scans "
        "and the corner annotations exist.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pair", nargs=2, action="append", metavar=("BASELINE", "ARM"),
        help="compare these two runs; repeatable, defaults to every known pair",
    )
    parser.add_argument("--split", default="test", help="which frozen bucket to compare on")
    parser.add_argument("--tables", default=None, help="where evaluate.py wrote its tables")
    parser.add_argument("--out", default=None, help="where to write, default: reports/tables")
    args = parser.parse_args()

    label = "Validation" if args.split == "val" else args.split.capitalize()
    pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
    source = Path(args.tables) if args.tables else paths.tables
    directory = Path(args.out) if args.out else paths.tables
    rows = compare(pairs, split=label, directory=source)
    if not rows:
        raise SystemExit("nothing to compare yet — train and evaluate at least one dropout arm")

    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "dropout_study.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = directory / "dropout_study.md"
    md_path.write_text(markdown_table(rows, label), encoding="utf-8")

    print(markdown_table(rows, label))
    print(f"table    : {csv_path}")
    print(f"markdown : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
