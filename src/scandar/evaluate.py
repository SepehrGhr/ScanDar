"""Evaluation.  *(brief §3.3 and §5)*

The brief names this file explicitly. It produces the numbers the report is built
on.

    python evaluate.py --checkpoint outputs/runs/enhance_baseline/best.pt
    python evaluate.py --config configs/enhance.yaml --limit 20

Built so far: the restoration table. PSNR and SSIM on the synthetic **training,
validation and test** buckets, on whole pages rectified at 1024x1448, with the
degraded-input baseline computed *first*. Each row answers a different question.
Training says how well the model fit what it was shown — a poor number there is a
capacity or optimisation problem, not a generalisation one. Validation is the
number the run was steered on, so it is optimistic by construction. Test is the
honest headline: source scans and background surfaces the model has never seen in
any form. A large training-to-test gap is overfitting; small gaps with poor
numbers everywhere are underfitting. And if the model's scores are not clearly
above the do-nothing baseline, it is not earning its parameters.

The "training" row is measured on a frozen sample of the training *distribution*,
not on data the model literally saw. With a generator that invents a fresh sample
every time it is asked, no sample is ever seen twice, so there is nothing else
the row could honestly mean — and it still answers the question it is there for,
because it holds the training scans and the training backgrounds.

Pages are restored through exactly the pipeline that runs at inference time —
overlapping tiles, cosine-blended *(brief §3.4)* — so the table scores what the
project actually ships rather than a more convenient variant of it.

And the corner-detection table *(brief §5)*: mean localisation error in pixels
and as a percentage of the image diagonal, the stricter all-four-within-a-
threshold success rate swept into a PCK curve, and quad IoU as the proxy for what
a corner error costs the rectification downstream. It carries its own no-model
baseline, the way the restoration table does — the classical Canny detector, run
on the very same 256x256 input the network sees. A learned detector that cannot
beat a rectangle-finder from before neural networks is not earning its
parameters either.

And the end-to-end table *(brief §7)*: the whole chain run twice over the same
photographs, once on the corners the detector found and once on the true ones,
which prices the detection step in decibels — beside the comparison between those
predicted corners and the true ones, which is what the bonus is graded on in its
own right. It can be produced before any fine-tuning has happened, from the two
finished runs, because "the two networks bolted together" is the baseline the
fine-tuned chain has to beat.

Which table gets produced is decided by the model's ``output_kind``, so one
command scores any checkpoint this project produces.

Not built yet: the OCR readability comparison against the commercial scanning
app, which needs the reference scans and the transcripts, and the real-photo
corner numbers, which need the annotation export. None of the three has been
captured yet.

Tables are written to ``reports/tables/`` in both CSV and Markdown, so the report
never contains a hand-copied number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, load_config, parse_override
from .datasets import frozen_dataset
from .device import amp_enabled, describe_device, get_device, recommended_workers
from .io import paths, write_json
from .metrics import (
    PCK_THRESHOLD_PCT,
    MetricAccumulator,
    corner_metrics,
    pck_curve,
    psnr,
    ssim_metric,
)
from .model import clamp_image, corners_from_output, load_model
from .pipelines import tiled_forward

__all__ = ["evaluate_enhancement", "evaluate_corners", "evaluate_scan", "main"]

SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}


def evaluate_enhancement(
    model,
    config: Config,
    splits=("train", "val", "test"),
    device=None,
    tile: int = 512,
    overlap: int = 192,
    limit: int | None = None,
    amp: bool | None = None,
    progress: bool = True,
) -> dict:
    """Score whole rectified pages, with the do-nothing baseline alongside.

    Returns ``{split: {"input": {...}, "model": {...}, "per_sample": [...]}}``.
    The baseline costs nothing extra — the degraded input is already in the batch
    — and computing it in the same loop, on the same images, is what makes the
    comparison exact rather than approximately fair.
    """
    device = device if device is not None else get_device()
    amp = amp_enabled(device) if amp is None else (amp and amp_enabled(device))
    workers = recommended_workers(config.train.get("num_workers", "auto"))
    model.eval().to(device)

    results: dict[str, dict] = {}
    for split in splits:
        dataset = frozen_dataset(config, split, task="enhance", mode="page")
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

        baseline = MetricAccumulator()
        scored = MetricAccumulator()
        per_sample = []

        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # The baseline first, as the brief asks — and on the very same pages.
            input_psnr = psnr(inputs, targets, reduction="none")
            input_ssim = ssim_metric(inputs, targets, reduction="none")
            baseline.add("psnr", input_psnr)
            baseline.add("ssim", input_ssim)

            with torch.no_grad():
                if tile and tile > 0:
                    restored = tiled_forward(
                        model, inputs.cpu(), tile=tile, overlap=overlap, device=device, amp=amp
                    ).to(device)
                else:
                    with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                        restored = model(inputs)
                    restored = clamp_image(restored.float())

            model_psnr = psnr(restored, targets, reduction="none")
            model_ssim = ssim_metric(restored, targets, reduction="none")
            scored.add("psnr", model_psnr)
            scored.add("ssim", model_ssim)

            per_sample.append(
                {
                    "id": batch["id"][0],
                    "scan": batch["scan"][0],
                    "input_psnr": round(float(input_psnr[0]), 4),
                    "input_ssim": round(float(input_ssim[0]), 4),
                    "psnr": round(float(model_psnr[0]), 4),
                    "ssim": round(float(model_ssim[0]), 4),
                }
            )
            if progress and (index + 1) % 25 == 0:
                print(
                    f"  {split:<5} {index + 1:>4}/{limit or len(dataset)}"
                    f"  psnr {scored.mean('psnr'):.2f} dB  (input {baseline.mean('psnr'):.2f})"
                )

        results[split] = {
            "n": len(per_sample),
            "input": baseline.summary(),
            "model": scored.summary(),
            "per_sample": per_sample,
        }
    return results


# ---------------------------------------------------------------------------
# corner detection  (brief §5)
# ---------------------------------------------------------------------------
def evaluate_corners(
    model,
    config: Config,
    splits=("train", "val", "test"),
    device=None,
    limit: int | None = None,
    baseline: bool = True,
    threshold_pct: float = PCK_THRESHOLD_PCT,
    progress: bool = True,
) -> dict:
    """Score a corner detector on the frozen synthetic buckets *(brief §5)*.

    Everything is measured in the detector's own 256x256 input space. That is the
    only space in which two detectors, or two photographs of different sizes, are
    comparable at all — and the percentage-of-diagonal column carries the number
    over to any other resolution.

    The classical detector runs on the identical resized input, not on the
    original photo, so the baseline column answers "what does the network add to
    what this input already gives away" rather than "what would a different
    pipeline achieve".
    """
    import numpy as np

    from .geometry import normalize_corners, order_corners
    from .pipelines import classical_corners

    device = device if device is not None else get_device()
    workers = recommended_workers(config.train.get("num_workers", "auto"))
    input_size = int(config.data.get("corner_input", 256))
    kind = str(getattr(model, "output_kind", "coords"))
    model.eval().to(device)

    results: dict[str, dict] = {}
    for split in splits:
        dataset = frozen_dataset(config, split, task="corner", mode="page")
        loader = DataLoader(
            dataset,
            batch_size=int(config.train.get("batch_size", 16)),
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

        scored = MetricAccumulator()
        classical = MetricAccumulator()
        per_sample: list[dict] = []
        predicted_all: list[torch.Tensor] = []
        target_all: list[torch.Tensor] = []
        fallbacks = 0
        seen = 0

        for batch in loader:
            if limit is not None:
                room = limit - seen
                if room <= 0:
                    break
                if room < len(batch["id"]):
                    # Trim rather than round up to a whole batch: `--limit 20`
                    # scoring 32 samples would be a quietly different experiment
                    # from the one that was asked for.
                    batch = {key: value[:room] for key, value in batch.items()}
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["corners"].to(device)
            with torch.no_grad():
                outputs = model(images)
            # The same extraction the pipeline ships, via the same function:
            # scoring a heatmap model with a different read-back than the one it
            # is deployed with measures a network nobody will ever run.
            predicted = corners_from_output(outputs.float()).detach().cpu()
            targets = targets.cpu()

            metrics = corner_metrics(predicted, targets, size=input_size, threshold_pct=threshold_pct)
            scored.update(metrics)
            predicted_all.append(predicted)
            target_all.append(targets)

            for index in range(images.shape[0]):
                row = {
                    "id": batch["id"][index],
                    "scan": batch["scan"][index],
                    "corner_err": round(float(metrics["corner_err"][index]), 4),
                    "corner_pct": round(float(metrics["corner_pct"][index]), 4),
                    "corner_worst": round(float(metrics["corner_worst"][index]), 4),
                    "pck": int(metrics["pck"][index]),
                    "quad_iou": round(float(metrics["quad_iou"][index]), 4),
                }
                if baseline:
                    photo = (
                        images[index].cpu().numpy().transpose(1, 2, 0) * 255.0
                    ).round().astype(np.uint8)
                    found = classical_corners(photo, working_side=input_size)
                    if found is None:
                        fallbacks += 1
                        # No detection is not a free pass: it is scored as the
                        # whole frame, which is what a pipeline would be left
                        # holding. Dropping the sample would flatter the baseline
                        # by measuring it only where it succeeded.
                        found = order_corners(
                            np.array(
                                [[0, 0], [input_size - 1, 0],
                                 [input_size - 1, input_size - 1], [0, input_size - 1]],
                                dtype=np.float32,
                            )
                        )
                    guess = torch.from_numpy(
                        normalize_corners(found, (input_size, input_size))
                    )[None]
                    classical.update(
                        corner_metrics(guess, targets[index : index + 1], size=input_size,
                                       threshold_pct=threshold_pct)
                    )
                    row["baseline_corner_err"] = round(
                        float(classical.values["corner_err"][-1]), 4
                    )
                per_sample.append(row)
            seen += images.shape[0]

            if progress and seen % 100 < images.shape[0]:
                print(
                    f"  {split:<5} {seen:>4}/{limit or len(dataset)}"
                    f"  error {scored.mean('corner_err'):.2f} px"
                    f"  pck {scored.mean('pck'):.3f}"
                )

        predicted_all = torch.cat(predicted_all) if predicted_all else torch.zeros(0, 4, 2)
        target_all = torch.cat(target_all) if target_all else torch.zeros(0, 4, 2)
        results[split] = {
            "n": len(per_sample),
            "model": scored.summary(),
            "baseline": classical.summary() if baseline else {},
            "baseline_undetected": fallbacks,
            "pck_curve": pck_curve(predicted_all, target_all, size=input_size),
            "per_sample": per_sample,
        }
    return results


def corner_table_rows(results: dict) -> list[dict]:
    """One row per split per variant, long-form, which is what a CSV wants."""
    rows = []
    for split, entry in results.items():
        variants = [("detector", entry["model"])]
        if entry.get("baseline"):
            variants.append(("classical baseline", entry["baseline"]))
        for variant, scores in variants:
            rows.append(
                {
                    "split": SPLIT_LABELS.get(split, split),
                    "variant": variant,
                    "n": entry["n"],
                    "corner_err_px": round(scores.get("corner_err", float("nan")), 3),
                    "corner_err_std": round(scores.get("corner_err_std", float("nan")), 3),
                    "corner_err_pct": round(scores.get("corner_pct", float("nan")), 4),
                    "worst_corner_px": round(scores.get("corner_worst", float("nan")), 3),
                    "pck": round(scores.get("pck", float("nan")), 4),
                    "quad_iou": round(scores.get("quad_iou", float("nan")), 4),
                }
            )
    return rows


def corner_markdown_table(results: dict, title: str = "", threshold_pct=PCK_THRESHOLD_PCT) -> str:
    """The corner comparison table, with the no-model baseline beneath each row."""
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        f"| Split | corner error (px @256) | % of diagonal | PCK@{threshold_pct:g}% | quad IoU |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for split, entry in results.items():
        model = entry["model"]
        lines.append(
            f"| {SPLIT_LABELS.get(split, split)} "
            f"| {model.get('corner_err', float('nan')):.2f} ± "
            f"{model.get('corner_err_std', float('nan')):.2f} "
            f"| {model.get('corner_pct', float('nan')):.2f}% "
            f"| {model.get('pck', float('nan')):.3f} "
            f"| {model.get('quad_iou', float('nan')):.4f} |"
        )
        if entry.get("baseline"):
            base = entry["baseline"]
            lines.append(
                f"| *— classical baseline* "
                f"| *{base.get('corner_err', float('nan')):.2f}* "
                f"| *{base.get('corner_pct', float('nan')):.2f}%* "
                f"| *{base.get('pck', float('nan')):.3f}* "
                f"| *{base.get('quad_iou', float('nan')):.4f}* |"
            )
    lines += [
        "",
        "Mean Euclidean distance between predicted and true corners, averaged over the four "
        "corners of each photo and then over photos, measured in the detector's own 256x256 "
        f"input space. PCK is the fraction of photos where **all four** corners land within "
        f"{threshold_pct:g}% of the image diagonal. The baseline rows are Canny + findContours "
        "+ approxPolyDP on the identical input, with an undetected page scored as the whole "
        "frame.",
    ]
    return "\n".join(lines) + "\n"


def write_corner_tables(results: dict, name: str, directory: Path | None = None) -> dict[str, Path]:
    """CSV, Markdown, the PCK curve and the per-sample scores."""
    import csv

    directory = Path(directory or paths.tables)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}

    rows = corner_table_rows(results)
    csv_path = directory / f"{name}_corners.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = csv_path

    markdown_path = directory / f"{name}_corners.md"
    markdown_path.write_text(corner_markdown_table(results, title=name), encoding="utf-8")
    written["markdown"] = markdown_path

    # The curve, not just the single threshold: the crossing point between two
    # detectors is more informative than either's score at a threshold someone
    # chose, and choosing it after seeing the numbers is how a comparison stops
    # being one.
    curve = [dict(row, split=split) for split, e in results.items() for row in e["pck_curve"]]
    if curve:
        curve_path = directory / f"{name}_pck_curve.csv"
        with open(curve_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        written["pck_curve"] = curve_path

    per_sample = [dict(row, split=split) for split, e in results.items() for row in e["per_sample"]]
    if per_sample:
        detail_path = directory / f"{name}_corners_per_sample.csv"
        with open(detail_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
        written["per_sample"] = detail_path

    return written


# ---------------------------------------------------------------------------
# the end-to-end chain  (brief §7, the bonus)
# ---------------------------------------------------------------------------
def evaluate_scan(
    detector,
    enhancer,
    config: Config,
    splits=("test",),
    device=None,
    limit: int | None = None,
    tile: int = 512,
    overlap: int = 192,
    out_width: int = 1024,
    threshold_pct: float = PCK_THRESHOLD_PCT,
    progress: bool = True,
) -> dict:
    """Score the whole chain, twice, on the frozen synthetic photos *(brief §7)*.

    The chain is run **once with the corners the detector found and once with the
    true ones**, on the same photographs, through the same rectification and the
    same tiled enhancement. That pairing is the point of this table. The
    annotated-corner arm is what the enhancement network can do when the page is
    handed to it correctly — the enhancement table's number, reached through the
    scanner's own code path — and the gap between the two arms is the price of
    the detector's error, in decibels, which is a far more useful thing to know
    than either arm alone.

    Alongside it, and on the same photos, the comparison the bonus is graded on
    in its own right: **predicted corners against the true corners**, in the
    detector's 256x256 space so the numbers read against the detector table.

    Both restoration arms are scored against one fixed target — the clean scan
    rectified with the true corners — so a wrong warp is punished as
    misalignment, which is exactly what it is downstream.

    A note the report must carry: these are the *enhancement* frozen buckets,
    which is the right choice because they are the only ones whose restoration
    target is achievable, and they contain **no distractor sheets**. The detector
    therefore scores better here than it does on its own bucket.
    """
    from .datasets import frozen_dataset
    from .geometry import normalize_corners
    from .metrics import corner_metrics
    from .pipelines import A4_ASPECT, scan_document

    device = device if device is not None else get_device()
    input_size = int(config.data.get("corner_input", 256))
    rect_size = tuple(config.data.get("rect_size", (1024, 1448)))
    detector.eval().to(device)
    enhancer.eval().to(device)

    results: dict[str, dict] = {}
    for split in splits:
        dataset = frozen_dataset(config, split, task="scan", mode="page")
        arms = {"predicted": MetricAccumulator(), "annotated": MetricAccumulator()}
        baseline = MetricAccumulator()
        corners_scored = MetricAccumulator()
        per_sample: list[dict] = []
        sources: dict[str, int] = {}

        count = len(dataset.entries) if limit is None else min(limit, len(dataset.entries))
        for index in range(count):
            sample = dataset.sample_at(index)
            entry = dataset.entries[index]
            photo = sample.photo
            canvas = (photo.shape[1], photo.shape[0])

            # The target and the do-nothing baseline, both from the true corners
            # and both at the resolution the chain writes at.
            height = max(1, round(out_width / A4_ASPECT))
            degraded, target = sample.rectify((out_width, height))
            target_t = _image_tensor(target, device)
            baseline.add("psnr", psnr(_image_tensor(degraded, device), target_t, reduction="none"))
            baseline.add(
                "ssim", ssim_metric(_image_tensor(degraded, device), target_t, reduction="none")
            )

            row = {"id": entry["id"], "scan": entry["scan"]}
            predicted_norm = None
            for arm, given in (("predicted", None), ("annotated", sample.corners)):
                result = scan_document(
                    photo,
                    detector if given is None else None,
                    enhancer,
                    device=device,
                    input_size=input_size,
                    out_width=out_width,
                    aspect="a4",
                    corners=given,
                    tile=tile,
                    overlap=overlap,
                )
                scanned = _image_tensor(result["scan"], device)
                arm_psnr = psnr(scanned, target_t, reduction="none")
                arm_ssim = ssim_metric(scanned, target_t, reduction="none")
                arms[arm].add("psnr", arm_psnr)
                arms[arm].add("ssim", arm_ssim)
                row[f"{arm}_psnr"] = round(float(arm_psnr[0]), 4)
                row[f"{arm}_ssim"] = round(float(arm_ssim[0]), 4)
                if given is None:
                    sources[result["source"]] = sources.get(result["source"], 0) + 1
                    row["source"] = result["source"]
                    predicted_norm = torch.from_numpy(result["normalised"])[None]

            truth = torch.from_numpy(normalize_corners(sample.corners, canvas))[None]
            metrics = corner_metrics(
                predicted_norm, truth, size=input_size, threshold_pct=threshold_pct
            )
            corners_scored.update(metrics)
            row.update(
                {
                    "corner_err": round(float(metrics["corner_err"][0]), 4),
                    "corner_pct": round(float(metrics["corner_pct"][0]), 4),
                    "pck": int(metrics["pck"][0]),
                    "quad_iou": round(float(metrics["quad_iou"][0]), 4),
                    "psnr_cost": round(row["annotated_psnr"] - row["predicted_psnr"], 4),
                }
            )
            per_sample.append(row)

            if progress and (index + 1) % 10 == 0:
                print(
                    f"  {split:<5} {index + 1:>4}/{count}"
                    f"  psnr {arms['predicted'].mean('psnr'):.2f} dB"
                    f"  (true corners {arms['annotated'].mean('psnr'):.2f})"
                    f"  corners {corners_scored.mean('corner_err'):.2f} px"
                )

        results[split] = {
            "n": len(per_sample),
            "rect_size": [int(rect_size[0]), int(rect_size[1])],
            "input": baseline.summary(),
            "predicted": arms["predicted"].summary(),
            "annotated": arms["annotated"].summary(),
            "corners": corners_scored.summary(),
            "corner_sources": sources,
            "per_sample": per_sample,
        }
    return results


def _image_tensor(image, device) -> torch.Tensor:
    """An RGB uint8 page as the ``(1, 3, H, W)`` float tensor the metrics take."""
    import numpy as np

    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).to(device).to(torch.float32).div_(255.0)[None]


def scan_markdown_table(results: dict, threshold_pct=PCK_THRESHOLD_PCT) -> str:
    """The bonus's two tables: what the chain produced, and how it found the page."""
    lines = [
        "| Split | corners | PSNR (dB) | SSIM | true corners | cost of detection |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for split, entry in results.items():
        predicted, annotated, base = entry["predicted"], entry["annotated"], entry["input"]
        cost = predicted.get("psnr", float("nan")) - annotated.get("psnr", float("nan"))
        label = SPLIT_LABELS.get(split, split)
        lines.append(
            f"| {label} | detected "
            f"| {predicted.get('psnr', float('nan')):.2f} ± "
            f"{predicted.get('psnr_std', float('nan')):.2f} "
            f"| {predicted.get('ssim', float('nan')):.4f} "
            f"| {annotated.get('psnr', float('nan')):.2f} "
            f"| **{cost:+.2f} dB** |"
        )
        lines.append(
            f"| {label} | *degraded input, true corners* "
            f"| *{base.get('psnr', float('nan')):.2f}* "
            f"| *{base.get('ssim', float('nan')):.4f}* | | |"
        )
    lines += [
        "",
        "Photo in, clean scan out, with no human input: the detector finds the page, the "
        "chain flattens it and the enhancement network restores it. Both arms are scored "
        "against the same target — the clean scan rectified with the true corners — so a "
        "misplaced corner is punished as the misalignment it is. The last column is what "
        "the detection step costs against being handed the page correctly.",
        "",
        f"| Split | corner error (px @256) | % of diagonal | PCK@{threshold_pct:g}% | quad IoU |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for split, entry in results.items():
        corners = entry["corners"]
        lines.append(
            f"| {SPLIT_LABELS.get(split, split)} "
            f"| {corners.get('corner_err', float('nan')):.2f} ± "
            f"{corners.get('corner_err_std', float('nan')):.2f} "
            f"| {corners.get('corner_pct', float('nan')):.2f}% "
            f"| {corners.get('pck', float('nan')):.3f} "
            f"| {corners.get('quad_iou', float('nan')):.4f} |"
        )
    lines += [
        "",
        "The corners the chain actually used, against the true ones, on the same photos it "
        "was scored on above. These are the **enhancement** frozen buckets, which carry no "
        "distractor sheet, so a detector scores better here than on its own bucket — the "
        "numbers are not interchangeable with the detector table's.",
    ]
    return "\n".join(lines) + "\n"


def scan_table_rows(results: dict) -> list[dict]:
    """Long-form rows: one per split per arm."""
    rows = []
    for split, entry in results.items():
        for variant, key in (
            ("degraded input", "input"),
            ("chain, detected corners", "predicted"),
            ("chain, true corners", "annotated"),
        ):
            scores = entry[key]
            rows.append(
                {
                    "split": SPLIT_LABELS.get(split, split),
                    "variant": variant,
                    "n": entry["n"],
                    "psnr": round(scores.get("psnr", float("nan")), 3),
                    "psnr_std": round(scores.get("psnr_std", float("nan")), 3),
                    "ssim": round(scores.get("ssim", float("nan")), 4),
                    "corner_err_px": round(entry["corners"].get("corner_err", float("nan")), 3)
                    if key == "predicted"
                    else "",
                    "pck": round(entry["corners"].get("pck", float("nan")), 4)
                    if key == "predicted"
                    else "",
                    "quad_iou": round(entry["corners"].get("quad_iou", float("nan")), 4)
                    if key == "predicted"
                    else "",
                }
            )
    return rows


def write_scan_tables(results: dict, name: str, directory: Path | None = None) -> dict[str, Path]:
    """CSV, Markdown and the per-sample scores for the end-to-end chain."""
    import csv

    directory = Path(directory or paths.tables)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}

    rows = scan_table_rows(results)
    csv_path = directory / f"{name}_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = csv_path

    markdown_path = directory / f"{name}_scan.md"
    markdown_path.write_text(
        f"### {name}\n\n" + scan_markdown_table(results), encoding="utf-8"
    )
    written["markdown"] = markdown_path

    per_sample = [dict(row, split=split) for split, e in results.items() for row in e["per_sample"]]
    if per_sample:
        detail_path = directory / f"{name}_scan_per_sample.csv"
        with open(detail_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
        written["per_sample"] = detail_path

    return written


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_rows(results: dict) -> list[dict]:
    """Long-form rows: one per split per variant, which is what a CSV wants."""
    rows = []
    for split, entry in results.items():
        for variant, key in (("degraded input", "input"), ("enhanced", "model")):
            scores = entry[key]
            rows.append(
                {
                    "split": SPLIT_LABELS.get(split, split),
                    "variant": variant,
                    "n": entry["n"],
                    "psnr": round(scores.get("psnr", float("nan")), 3),
                    "psnr_std": round(scores.get("psnr_std", float("nan")), 3),
                    "ssim": round(scores.get("ssim", float("nan")), 4),
                    "ssim_std": round(scores.get("ssim_std", float("nan")), 4),
                }
            )
    return rows


def markdown_table(results: dict, title: str = "") -> str:
    """The brief's table, with the do-nothing baseline on the line above it."""
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, entry in results.items():
        model, baseline = entry["model"], entry["input"]
        gain = model.get("psnr", float("nan")) - baseline.get("psnr", float("nan"))
        lines.append(
            f"| {SPLIT_LABELS.get(split, split)} "
            f"| {model.get('psnr', float('nan')):.2f} ± {model.get('psnr_std', float('nan')):.2f} "
            f"| {model.get('ssim', float('nan')):.4f} ± {model.get('ssim_std', float('nan')):.4f} "
            f"| {baseline.get('psnr', float('nan')):.2f} "
            f"| {baseline.get('ssim', float('nan')):.4f} "
            f"| **{gain:+.2f} dB** |"
        )
    lines += [
        "",
        "Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — "
        "the same path the inference pipeline takes. The baseline columns are the degraded "
        "input measured against the same clean targets, before any enhancement.",
    ]
    return "\n".join(lines) + "\n"


def write_tables(results: dict, name: str, directory: Path | None = None) -> dict[str, Path]:
    """CSV, Markdown and per-sample scores, so no number is ever hand-copied."""
    import csv

    directory = Path(directory or paths.tables)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}

    rows = table_rows(results)
    csv_path = directory / f"{name}_restoration.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = csv_path

    markdown_path = directory / f"{name}_restoration.md"
    markdown_path.write_text(markdown_table(results, title=name), encoding="utf-8")
    written["markdown"] = markdown_path

    per_sample = [dict(row, split=split) for split, e in results.items() for row in e["per_sample"]]
    if per_sample:
        detail_path = directory / f"{name}_restoration_per_sample.csv"
        with open(detail_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
        written["per_sample"] = detail_path

    return written


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _find_checkpoint(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if not args.config:
        raise SystemExit("give either --checkpoint or --config")
    config = load_config(args.config, overrides=args.overrides)
    name = str(config.get("run", {}).get("name") or Path(config["_config_path"]).stem)
    candidate = paths.run_dir(name) / args.weights
    if not candidate.exists():
        raise SystemExit(f"no checkpoint at {candidate} — has {name} been trained yet?")
    return candidate


def _load_untrained_scanner(config: Config, device):
    """The chain assembled from two finished runs, with no fine-tuning on top.

    This is the *before* column of the bonus's before-and-after, and it has to be
    scoreable without a scanner checkpoint existing — the whole question is
    whether fine-tuning the chain end to end improves on simply bolting the two
    trained networks together, and half of that comparison is the bolt-together.
    """
    from .model import build_model

    model = build_model(config)
    if not hasattr(model, "load_components"):
        raise SystemExit(
            f"{config.get('run', {}).get('name')} is not an end-to-end run — "
            "give --checkpoint instead"
        )
    for role, path in model.load_components().items():
        print(f"init      : {role} from {path}")
    return model.to(device).eval()


def _run_detector_evaluation(model, config: Config, args, device, name: str, run_dir: Path) -> int:
    """Score a corner detector and write its tables. Shared by every entry path.

    Including the one that reaches inside a chained scanner, so a fine-tuned
    detector is measured on exactly the bucket and by exactly the code its
    baseline was.
    """
    input_size = int(config.data.get("corner_input", 256))
    kind = str(getattr(model, "output_kind", "coords"))
    print(f"detector  : {type(model).__name__}, {kind} at {input_size}x{input_size}\n")
    corner_results = evaluate_corners(
        model,
        config,
        splits=tuple(args.splits),
        device=device,
        limit=args.limit,
        baseline=not args.no_baseline,
    )
    print("\n" + corner_markdown_table(corner_results))
    written = write_corner_tables(corner_results, name, directory=args.out)
    for label, path in written.items():
        print(f"{label:<10}: {path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "evaluation.json",
        {
            "run": name,
            "task": "corner",
            "input_size": input_size,
            "splits": {
                split: {
                    key: entry[key]
                    for key in ("n", "model", "baseline", "baseline_undetected", "pck_curve")
                }
                for split, entry in corner_results.items()
            },
        },
    )
    return 0


def _run_scan_evaluation(
    scanner,
    config: Config,
    args,
    device,
    name: str,
    run_dir: Path,
    json_name: str = "evaluation_scan.json",
) -> int:
    """Score a chained scanner and write its tables. Shared by both entry paths.

    *json_name* separates the two arms that share a run directory: the fine-tuned
    chain's numbers and the assembled chain's baseline are both about
    ``corner_heat_e2e``, and one silently overwriting the other would destroy
    exactly the before-and-after the bonus is reported as.
    """
    splits = tuple(args.splits)
    print(f"scanner   : {type(scanner.detector).__name__} -> {type(scanner.enhancer).__name__}\n")
    results = evaluate_scan(
        scanner.detector,
        scanner.enhancer,
        config,
        splits=splits,
        device=device,
        limit=args.limit,
        tile=args.tile,
        overlap=args.overlap,
        out_width=args.width,
    )
    print("\n" + scan_markdown_table(results))
    written = write_scan_tables(results, name, directory=args.out)
    for label, path in written.items():
        print(f"{label:<10}: {path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / json_name,
        {
            "task": "scan",
            "run": name,
            "out_width": args.width,
            "splits": {
                split: {
                    key: entry[key]
                    for key in ("n", "input", "predicted", "annotated", "corners", "corner_sources")
                }
                for split, entry in results.items()
            },
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Score a trained ScanDar model."
    )
    parser.add_argument("--checkpoint", default=None, help="path to best.pt or last.pt")
    parser.add_argument("--config", default=None, help="find the checkpoint from a config instead")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="key.path=value")
    parser.add_argument("--weights", default="best.pt", help="which checkpoint in the run dir")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="score only the first N pages")
    parser.add_argument("--tile", type=int, default=512, help="tile size; 0 runs one whole-page pass")
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the classical corner detector's baseline column",
    )
    parser.add_argument("--out", default=None, help="table directory (default reports/tables)")
    parser.add_argument("--name", default=None, help="table name (default: the run name)")
    parser.add_argument(
        "--width", type=int, default=1024, help="rectified page width for the end-to-end table"
    )
    parser.add_argument(
        "--assembled",
        action="store_true",
        help="score the chain as assembled from its two finished runs, ignoring any fine-tune",
    )
    parser.add_argument(
        "--part",
        choices=("scanner", "detector"),
        default="scanner",
        help="score the whole chain, or only the detector inside it, on the corner buckets",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    device = get_device()

    # The end-to-end chain is scoreable before it has ever been fine-tuned —
    # that is the "before" half of the comparison the bonus asks for — so a
    # missing checkpoint is a legitimate state here rather than an error, and
    # --assembled asks for that baseline once the fine-tuned checkpoint exists.
    if args.config and not args.checkpoint:
        config = load_config(args.config, overrides=args.overrides)
        name = str(config.get("run", {}).get("name") or Path(config["_config_path"]).stem)
        trained = (paths.run_dir(name) / args.weights).exists()
        if str(config.get("task", "")) == "scan" and (args.assembled or not trained):
            print(
                "checkpoint: "
                + ("ignored (--assembled)" if trained else "none yet")
                + " — scoring the chain as assembled, not fine-tuned"
            )
            print(f"device    : {describe_device(device)}\n")
            scanner = _load_untrained_scanner(config, device)
            if args.part == "detector":
                return _run_detector_evaluation(
                    scanner.detector, config, args, device,
                    name=args.name or f"{name}_assembled", run_dir=paths.run_dir(name),
                )
            return _run_scan_evaluation(
                scanner, config, args, device,
                name=args.name or f"{name}_assembled", run_dir=paths.run_dir(name),
                json_name="evaluation_scan_assembled.json",
            )

    checkpoint_path = _find_checkpoint(args)
    model, config = load_model(checkpoint_path, device=device)
    # The checkpoint carries the config it trained with, which is the one that has
    # to be used to rebuild the model. Overrides still apply on top, so an
    # evaluation can be re-pointed at a different data root without retraining.
    for override in args.overrides:
        key, value = parse_override(override)
        config.set_path(key, value)

    name = args.name or checkpoint_path.parent.name
    kind = str(getattr(model, "output_kind", "restoration"))
    print(f"checkpoint: {checkpoint_path}")
    print(f"device    : {describe_device(device)}")

    if kind == "scan" and args.part == "scanner":
        return _run_scan_evaluation(
            model, config, args, device, name=name, run_dir=checkpoint_path.parent
        )

    if kind == "scan":
        # The detector on its own bucket, which is *harder* than the chain's:
        # distractor sheets, tinted stock, curled pages. Fine-tuning the chain
        # happens on the enhancement distribution, which has none of those, so
        # "did the fine-tune cost the detector anything on the world it was
        # trained for" is a question that has to be asked separately.
        return _run_detector_evaluation(
            model.detector, config, args, device, name=name, run_dir=checkpoint_path.parent
        )

    if kind in ("coords", "heatmaps"):
        return _run_detector_evaluation(
            model, config, args, device, name=name, run_dir=checkpoint_path.parent
        )

    print(f"pages     : {'whole-page pass' if not args.tile else f'{args.tile}px tiles, {args.overlap}px overlap'}\n")

    results = evaluate_enhancement(
        model,
        config,
        splits=tuple(args.splits),
        device=device,
        tile=args.tile,
        overlap=args.overlap,
        limit=args.limit,
    )

    print("\n" + markdown_table(results))
    written = write_tables(results, name, directory=args.out)
    for kind, path in written.items():
        print(f"{kind:<10}: {path}")

    write_json(
        checkpoint_path.parent / "evaluation.json",
        {
            "checkpoint": str(checkpoint_path),
            "tile": args.tile,
            "overlap": args.overlap,
            "splits": {
                split: {"n": e["n"], "input": e["input"], "model": e["model"]}
                for split, e in results.items()
            },
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
