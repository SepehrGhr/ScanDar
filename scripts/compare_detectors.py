#!/usr/bin/env python
"""Put the corner detectors head to head and let the numbers decide.

    python scripts/compare_detectors.py
    python scripts/compare_detectors.py --runs corner_heat corner_reg corner_heat_aux

The brief asks for both formulations to be built and for the *experiments* to
settle which one wins, on mean localisation error and on a stricter
all-four-within-a-threshold success rate, "supported with numbers and failure-case
visualizations" *(brief §5)*. This produces all three: the table, the PCK curve
across every threshold rather than the one that happens to flatter somebody, and
a gallery of each detector's worst cases with the truth drawn beside the guess.

It reads what ``evaluate.py`` already wrote, so nothing is recomputed and no
number here can disagree with the per-run tables. Run ``evaluate.py`` for each
detector first.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.io import paths, read_json

DEFAULT_RUNS = ("corner_heat", "corner_reg")
GALLERY_SAMPLES = 5


# ---------------------------------------------------------------------------
# reading what evaluate.py left behind
# ---------------------------------------------------------------------------
def read_table(name: str, suffix: str, directory: Path | None = None) -> list[dict]:
    path = Path(directory or paths.tables) / f"{name}_{suffix}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python evaluate.py --config configs/{name}.yaml` first"
        )
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def comparison_rows(runs, split: str = "Test", directory: Path | None = None) -> list[dict]:
    """One row per detector, plus the shared classical baseline once.

    Once, because the baseline is the same detector scored on the same photos in
    every run's table — quoting it per run would put three identical rows under a
    two-detector comparison and invite the reader to think they meant something.
    """
    rows, baseline = [], None
    for name in runs:
        for entry in read_table(name, "corners", directory):
            if entry["split"] != split:
                continue
            if entry["variant"] == "detector":
                rows.append(dict(entry, run=name))
            elif baseline is None:
                baseline = dict(entry, run="classical")
    if baseline is not None:
        rows.append(baseline)
    return rows


def markdown_table(rows, split: str) -> str:
    lines = [
        f"### Corner detection, {split.lower()} split",
        "",
        "| Detector | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = row["run"] if row["variant"] == "detector" else "*classical baseline*"
        lines.append(
            f"| {label} | {float(row['corner_err_px']):.2f} ± {float(row['corner_err_std']):.2f} "
            f"| {float(row['corner_err_pct']):.2f}% | {float(row['pck']):.3f} "
            f"| {float(row['quad_iou']):.4f} |"
        )
    lines += [
        "",
        "Mean Euclidean distance between predicted and true corners, averaged over the four "
        "corners of a photo and then over photos, in the detector's own 256x256 input space. "
        "PCK is the fraction of photos with **all four** corners inside 2% of the image "
        "diagonal. The baseline is Canny + findContours + approxPolyDP on the identical input.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# failure gallery
# ---------------------------------------------------------------------------
def worst_samples(name: str, split: str = "test", count: int = GALLERY_SAMPLES) -> list[dict]:
    rows = [r for r in read_table(name, "corners_per_sample") if r["split"] == split]
    return sorted(rows, key=lambda r: -float(r["corner_err"]))[:count]


def failure_gallery(runs, out_path: Path, split: str = "test") -> Path:
    """Each detector's worst photos, with the truth drawn beside the prediction.

    A number says a detector is wrong; only a picture says *how*. The two failures
    worth telling apart are a page found in the wrong place and a page found in
    the right place with one corner adrift, and they want different fixes.

    The label under each panel names the generator options that fired, because
    the point of a failure gallery is to notice that the same option keeps
    appearing under it.
    """
    import matplotlib.pyplot as plt
    import torch

    from scandar.config import load_config
    from scandar.datasets import frozen_dataset
    from scandar.device import get_device
    from scandar.geometry import denormalize_corners
    from scandar.model import corners_from_output, load_model
    from scandar.pipelines import draw_corners
    from scandar.viz import COLORS, use_style

    use_style()
    device = get_device()
    figure, axes = plt.subplots(
        len(runs), GALLERY_SAMPLES, figsize=(3.1 * GALLERY_SAMPLES, 4.4 * len(runs))
    )
    axes = np.atleast_2d(axes)

    manifest = {
        entry["id"]: entry
        for entry in read_json(paths.frozen_set("corner", split) / "manifest.json")["samples"]
    }

    for row, name in enumerate(runs):
        checkpoint = paths.run_dir(name) / "best.pt"
        model, config = load_model(checkpoint, device=device)
        dataset = frozen_dataset(load_config(f"configs/{name}.yaml"), split,
                                 task="corner", mode="page")
        index_of = {entry["id"]: i for i, entry in enumerate(dataset.entries)}
        size = int(config.get("data", {}).get("corner_input", 256))

        for column, sample in enumerate(worst_samples(name, split)):
            item = dataset[index_of[sample["id"]]]
            with torch.no_grad():
                predicted = corners_from_output(model(item["image"][None].to(device)).float())
            photo = (item["image"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)

            # Truth first, prediction over it, so an exact hit reads as the
            # prediction covering the truth rather than as one of them missing.
            canvas = draw_corners(photo, denormalize_corners(item["corners"].numpy(), (size, size)),
                                 color=(60, 130, 246), labels=False)
            canvas = draw_corners(canvas, denormalize_corners(predicted[0].cpu().numpy(),
                                                              (size, size)),
                                  color=(220, 80, 60), labels=False)

            options = manifest[sample["id"]]["params"]["page"]
            tags = [key for key in ("distractor", "curl", "tint") if options.get(key) is not None]
            axis = axes[row, column]
            axis.imshow(canvas)
            axis.set_title(
                f"{sample['id']}   {float(sample['corner_err']):.1f} px\n"
                f"{', '.join(tags) if tags else 'plain page'}",
                fontsize=8,
            )
            axis.axis("off")
        axes[row, 0].set_ylabel(name)
        # A row label that survives axis("off"), which removes the y label.
        axes[row, 0].text(
            -0.06, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
            va="center", ha="right", fontsize=11, color=COLORS["train"],
        )

    figure.suptitle(
        f"Worst {GALLERY_SAMPLES} {split} photos per detector — "
        "blue is the truth, red is the prediction",
        fontsize=12,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--split", default="test", help="which frozen bucket to compare on")
    parser.add_argument("--no-gallery", action="store_true", help="table and curve only")
    parser.add_argument(
        "--name", default=None,
        help="a name for this comparison's outputs, so a second comparison does not overwrite "
             "the first — the detector comparison keeps the unprefixed names",
    )
    args = parser.parse_args()

    # The head-to-head between the two formulations is *the* corner comparison and
    # owns the plain filenames; anything else asks for a name and gets its own.
    table_name = args.name or "corner_comparison"
    figure_prefix = f"{args.name}_" if args.name else ""

    label = args.split.capitalize() if args.split != "val" else "Validation"
    rows = comparison_rows(args.runs, split=label)
    if not rows:
        raise SystemExit(f"no {label} rows in the tables for {', '.join(args.runs)}")

    table = markdown_table(rows, label)
    print("\n" + table)

    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"{table_name}.md").write_text(table, encoding="utf-8")
    with open(paths.tables / f"{table_name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from scandar.viz import pck_curves

    figures = paths.figures / "corners"
    curve = pck_curves(
        {name: [r for r in read_table(name, "pck_curve") if r["split"] == args.split]
         for name in args.runs},
        figures / f"{figure_prefix}pck_curves.png",
    )

    print(f"table   : {paths.tables / f'{table_name}.md'}")
    print(f"curve   : {curve}")
    if not args.no_gallery:
        gallery = failure_gallery(args.runs, figures / f"{figure_prefix}failure_cases.png", args.split)
        print(f"gallery : {gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
