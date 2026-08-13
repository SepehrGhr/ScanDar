"""Figures.  *(brief, "Visualization of Results")*

One shared matplotlib style — consistent palette, 200 dpi, readable in print — so
every figure in the report looks like it came from the same project.

Built so far: the training curve. The brief asks for it by name — "we will plot
the loss on both the training and validation sets against the number of epochs...
Analyzing this graph is essential for diagnosing the model's behaviour" *(brief
§3.2)* — and it is the one figure that has to exist as soon as a model has been
trained, rather than at the end with the rest.

Still to build, when the results they draw exist: dataset contact sheets, the
degradation pipeline as a step-by-step strip, the "spot the fake" panel, the loss
ablation as zoomed text crops, real-photo triplets, corner prediction overlays and
error curves, the dropout study's gap chart, and the end-to-end storyboard.
"""

from __future__ import annotations

import csv
from pathlib import Path

#: One palette, used everywhere, so two figures never disagree about what
#: "validation" is coloured. Chosen to stay distinguishable in greyscale print
#: and for the two main series to differ in lightness, not only in hue.
COLORS = {
    "train": "#2b6cb0",
    "val": "#c05621",
    "accent": "#2f855a",
    "muted": "#a0aec0",
    "baseline": "#718096",
}

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}


def use_style() -> None:
    """Apply the project style to the current matplotlib session."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


def read_metrics(run_dir: Path | str) -> list[dict]:
    """The per-epoch rows a training run wrote, with numbers as numbers."""
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — has this run trained yet?")

    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def training_curves(run_dir: Path | str, out_path: Path | str | None = None, baseline_psnr=None):
    """Training and validation loss against epoch, with the PSNR beside it.

    The left panel is the graph the brief asks for and the one that diagnoses the
    run: two curves that fall together and flatten together are a model limited by
    capacity or by the data; a validation curve that turns back up while the
    training curve keeps falling is overfitting.

    The right panel carries the metric the run was actually steered on — PSNR for
    a restoration, corner localisation error for a detector — and the do-nothing
    baseline as a horizontal line, because a curve without the line the model has
    to clear does not say whether the model is any good. ``baseline_psnr`` is
    named for the restoration case it was written for; it is really "the line
    this run has to clear", whichever metric the right panel ends up showing.
    """
    import matplotlib.pyplot as plt

    use_style()
    rows = read_metrics(run_dir)
    run_dir = Path(run_dir)
    epochs = [r["epoch"] for r in rows]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.plot(epochs, [r["train_loss"] for r in rows], color=COLORS["train"], label="training")
    left.plot(epochs, [r["val_loss"] for r in rows], color=COLORS["val"], label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.set_title(f"{run_dir.name} — loss")
    left.set_yscale("log")
    left.legend()

    # Whichever headline metric this run recorded. The two tasks are steered on
    # different numbers, and one of them gets better by going down.
    for key, unit, label, baseline_label in (
        ("val_psnr", "PSNR (dB)", "validation PSNR", "degraded input ({:.2f} dB)"),
        ("val_corner_err", "corner error (px)", "validation corner error",
         "classical detector ({:.2f} px)"),
    ):
        if not any(key in r for r in rows):
            continue
        right.plot(epochs, [r[key] for r in rows], color=COLORS["accent"], label=label)
        if baseline_psnr is not None:
            right.axhline(baseline_psnr, color=COLORS["baseline"], linestyle="--", linewidth=1.2,
                          label=baseline_label.format(baseline_psnr))
        right.set_xlabel("epoch")
        right.set_ylabel(unit)
        right.set_title(label)
        right.legend()
        break

    out_path = Path(out_path) if out_path else run_dir / "training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def compare_runs(run_dirs, out_path: Path | str, metric: str = "val_psnr", labels=None):
    """One metric from several runs on one axis — how an ablation is read."""
    import matplotlib.pyplot as plt

    use_style()
    figure, axes = plt.subplots(figsize=(7, 4.2))
    palette = [COLORS["train"], COLORS["val"], COLORS["accent"], COLORS["muted"], "#6b46c1"]

    for index, run_dir in enumerate(run_dirs):
        rows = read_metrics(run_dir)
        if metric not in rows[0]:
            continue
        axes.plot(
            [r["epoch"] for r in rows],
            [r[metric] for r in rows],
            color=palette[index % len(palette)],
            label=(labels[index] if labels else Path(run_dir).name),
        )

    axes.set_xlabel("epoch")
    axes.set_ylabel(metric.replace("_", " "))
    axes.set_title(f"{metric.replace('_', ' ')} across runs")
    axes.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path
