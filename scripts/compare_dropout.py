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


# ---------------------------------------------------------------------------
# the matched-epoch comparison, for an arm that was stopped short
# ---------------------------------------------------------------------------
#: What each task's per-epoch log is judged on, and which direction is better.
CURVE_METRICS = {
    "val_psnr": ("validation PSNR (dB)", "max", 2),
    "val_corner_err": ("validation corner error (px @256)", "min", 2),
}


def read_curve(name: str) -> list[dict]:
    """A run's per-epoch log, as written by the trainer."""
    path = paths.run_dir(name) / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — has {name} been trained?")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def curve_agrees_with_table(name: str, key: str, curve: list[dict], directory=None) -> float | None:
    """How far a run's last logged validation number is from its evaluated one.

    A per-epoch log is only comparable across runs if the number in it means the
    same thing in both, and that is not guaranteed by the column name. ``corner_heat``
    is the case in point: it was trained before the corner extraction was fixed, so
    its log records 6.26 px where re-evaluating the same checkpoint with today's
    code gives 0.70. Its curve cannot be read against a curve logged after the fix,
    and nothing about the two files says so.

    Re-evaluating the run *is* the test — ``evaluate.py`` recomputes with current
    code, so a log that disagrees with its own table was written by different code.
    Returns the ratio, or ``None`` when there is no table to check against.
    """
    suffix = "corners" if key == "val_corner_err" else "restoration"
    table = read_table_for(name, suffix, directory)
    if table is None:
        return None
    column = {"val_corner_err": "corner_err_px", "val_psnr": "psnr"}[key]
    variant = TASKS[suffix]["variant"]
    for row in table:
        if row["split"] == "Validation" and row["variant"] == variant:
            logged, evaluated = float(curve[-1][key]), float(row[column])
            return max(logged, evaluated) / max(min(logged, evaluated), 1e-9)
    return None


def read_table_for(name: str, suffix: str, directory=None):
    path = Path(directory or paths.tables) / f"{name}_{suffix}.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def at_epoch(curve: list[dict], epoch: int) -> dict:
    for row in curve:
        if int(row["epoch"]) == epoch:
            return row
    raise SystemExit(f"no epoch {epoch} in the log — it reached epoch {curve[-1]['epoch']}")


def compare_curves(pairs, epoch: int | None = None) -> list[dict]:
    """Baseline against arm at the same epoch of the same schedule.

    An arm stopped at epoch 10 of a 20-epoch schedule cannot have its final
    numbers set beside a baseline that ran all 20 — half the difference would be
    the schedule. It *can* be set beside the baseline's own epoch 10, because up
    to that point the two runs are identical in every respect including the
    learning rate, which is why the short arms keep `epochs: 20` and stop with
    `train.stop_after_epoch` instead of declaring a shorter run.

    Both curves are validation, measured on the frozen buckets on identical
    patches, which is what makes them comparable at all.
    """
    out = []
    for base_name, arm_name in pairs:
        try:
            base_curve, arm_curve = read_curve(base_name), read_curve(arm_name)
        except SystemExit as exc:
            print(f"skipped  : {exc}")
            continue

        key = next((k for k in CURVE_METRICS if k in arm_curve[0]), None)
        if key is None:
            raise SystemExit(f"{arm_name} logs neither of {sorted(CURVE_METRICS)}")
        label, direction, digits = CURVE_METRICS[key]

        # Refuse the comparison outright when a log does not agree with its own
        # re-evaluation. Printing a number with a caveat under it invites the
        # number to be quoted without the caveat, and this one would be wrong by
        # a factor of nine.
        stale = [
            (run, ratio)
            for run in (base_name, arm_name)
            for ratio in [curve_agrees_with_table(run, key, read_curve(run))]
            if ratio is not None and ratio > 1.5
        ]
        if stale:
            for run, ratio in stale:
                print(
                    f"refused  : {base_name} vs {arm_name} — {run}'s log disagrees with its own "
                    f"evaluation by {ratio:.1f}x, so it was written by different code and its "
                    "curve is not comparable with a curve written by this one. Compare these two "
                    "on the evaluation tables instead, and mind the epoch difference."
                )
            continue

        # The last epoch the *arm* reached, which is the last one both ran.
        matched = epoch or min(int(arm_curve[-1]["epoch"]), int(base_curve[-1]["epoch"]))
        base_row, arm_row = at_epoch(base_curve, matched), at_epoch(arm_curve, matched)
        base_value, arm_value = float(base_row[key]), float(arm_row[key])
        improvement = arm_value - base_value if direction == "max" else base_value - arm_value

        # The train-to-validation distance at that same epoch, from the losses,
        # which every task logs. This is the quantity §6 is actually about, and
        # unlike the metric it is barely sensitive to how far the run got.
        base_gap = float(base_row["val_loss"]) - float(base_row["train_loss"])
        arm_gap = float(arm_row["val_loss"]) - float(arm_row["train_loss"])

        out.append(
            {
                "model": base_name,
                "arm": arm_name,
                "epoch": matched,
                "metric": label,
                "baseline": round(base_value, digits),
                "dropout": round(arm_value, digits),
                "change": round(improvement, digits),
                "better": "dropout" if improvement > 0 else "baseline",
                "baseline_val_minus_train_loss": round(base_gap, 5),
                "dropout_val_minus_train_loss": round(arm_gap, 5),
            }
        )
    return out


def markdown_curve_table(rows) -> str:
    lines = [
        "### Dropout, at a matched epoch of an identical schedule",
        "",
        "| Model | arm | epoch | metric | baseline | with dropout | change | val−train loss "
        "(baseline → dropout) |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        sign = "+" if row["change"] > 0 else ""
        lines.append(
            f"| {row['model']} | {row['arm']} | {row['epoch']} | {row['metric']} "
            f"| {row['baseline']} | {row['dropout']} | {sign}{row['change']} "
            f"| {row['baseline_val_minus_train_loss']} → {row['dropout_val_minus_train_loss']} |"
        )
    lines += [
        "",
        "Both columns are validation on the frozen bucket, read off each run's own per-epoch log "
        "at the **same epoch of the same schedule** — same learning rate, same number of samples "
        "seen, same everything but dropout. An arm stopped early can be compared this way; its "
        "final numbers cannot be compared with a baseline that ran to the end.",
        "",
        "The last column is validation loss minus training loss at that epoch — the overfitting "
        "gap dropout exists to close. It is the honest headline for a shortened run, because "
        "unlike the accuracy column it barely depends on how far the run got.",
        "",
        "**Truncation is biased against dropout on accuracy**: dropout slows convergence, so an "
        "arm judged halfway through a schedule looks worse than it would at the end. That makes "
        "a null result on the gap column the safe conclusion and a small accuracy loss the "
        "unsafe one.",
    ]
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--curves", action="store_true",
        help="compare the per-epoch logs at a matched epoch instead of the final tables — "
             "the comparison to use when an arm was stopped short of its schedule",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="with --curves: which epoch to compare at (default: the last one both reached)",
    )
    args = parser.parse_args()

    if args.curves:
        pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
        rows = compare_curves(pairs, args.epoch)
        if not rows:
            raise SystemExit("nothing to compare yet — train at least one dropout arm")
        directory = Path(args.out) if args.out else paths.tables
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "dropout_study_matched_epoch.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        md_path = directory / "dropout_study_matched_epoch.md"
        md_path.write_text(markdown_curve_table(rows), encoding="utf-8")
        print(markdown_curve_table(rows))
        print(f"table    : {csv_path}\nmarkdown : {md_path}")
        return 0

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
