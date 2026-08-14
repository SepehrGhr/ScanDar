#!/usr/bin/env python
"""Real photos vs the commercial app — the evaluation the synthetic table cannot do  *(brief §3.3, §5)*.

    python scripts/evaluate_real.py

Two things happen here, both blocked until now on data only the author could
produce (the Roboflow export and the CamScanner references):

**Corner detection on real photos** *(brief §5)*. `data/real/corners.json`
(written by `scripts/parse_roboflow.py`) carries hand-labelled corners for every
annotated real photo. The shipped detector (`corner_heat`) runs on each one
through the exact §5.1 pipeline — `pipelines.detect_corners`, classical fallback
included — and is scored against those labels the same way the synthetic table
scores it: mean corner error in pixels and as a percentage of the photo's own
diagonal, the worst-of-four PCK success rate, and quad IoU. A classical-detector
baseline runs alongside it, on the same photos.

**Enhancement vs CamScanner** *(brief §3.3)*, on the photos a reference scan
exists for. No clean target exists for a real photo, so this is not a PSNR
table: each photo is rectified with its *annotated* corners (never the
detector's own — this isolates the enhancement network from the corner
question above), enhanced, and placed next to the CamScanner reference,
resized to match. Readability is scored by Tesseract's own word confidence and
word count on all three variants — rectified input, our output, the reference
— with identical preprocessing, plus character/word error rate against a
hand-typed transcript for the one photo that has one (`Image18`, printed
English). Triplet figures with a zoomed inset go to
`reports/figures/real/`.

Tables land in `reports/tables/real_corners.{csv,md}` and
`reports/tables/real_enhancement_ocr.{csv,md}`, in the same CSV+Markdown shape
`evaluate.py` writes everything else in.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.device import get_device
from scandar.geometry import order_corners, quad_iou
from scandar.io import imread_rgb, paths, read_json
from scandar.model import load_model
from scandar.ocr import cer, ocr_confidence, ocr_text, wer
from scandar.pipelines import classical_corners, detect_corners, enhance_document, rectify_document

PCK_THRESHOLD_PCT = 2.0


# ---------------------------------------------------------------------------
# corner detection on the real photos  (brief §5)
# ---------------------------------------------------------------------------
def _corner_row(predicted: np.ndarray, target: np.ndarray, width: int, height: int) -> dict:
    predicted = order_corners(predicted)
    target = order_corners(target)
    per_corner = np.linalg.norm(predicted - target, axis=1)
    diagonal = float(np.hypot(width, height))
    mean_err = float(per_corner.mean())
    return {
        "corner_err": mean_err,
        "corner_pct": 100.0 * mean_err / diagonal,
        "corner_worst": float(per_corner.max()),
        "pck": float(per_corner.max() <= diagonal * PCK_THRESHOLD_PCT / 100.0),
        "quad_iou": quad_iou(predicted, target),
    }


def evaluate_real_corners(detector, corners: dict, photos_dir: Path, device) -> dict:
    per_sample = []
    fallbacks = 0
    baseline_rows = []

    for stem in sorted(corners):
        photo_path = photos_dir / f"{stem}.jpg"
        photo = imread_rgb(photo_path)
        height, width = photo.shape[:2]
        target = np.asarray(corners[stem], dtype=np.float32)

        result = detect_corners(photo, detector, device=device, fallback=True)
        if result["source"] == "classical":
            fallbacks += 1
        row = {"id": stem, "source": result["source"]}
        row.update(_corner_row(result["corners"], target, width, height))
        per_sample.append(row)

        found = classical_corners(photo)
        if found is None:
            found = order_corners(
                np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                          dtype=np.float32)
            )
        baseline_rows.append(_corner_row(found, target, width, height))

    def _summary(rows: list[dict]) -> dict:
        return {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "std": float(np.std([r[key] for r in rows])),
            }
            for key in ("corner_err", "corner_pct", "corner_worst", "quad_iou")
        } | {"pck": float(np.mean([r["pck"] for r in rows]))}

    return {
        "n": len(per_sample),
        "model": _summary(per_sample),
        "baseline": _summary(baseline_rows),
        "fallbacks": fallbacks,
        "per_sample": per_sample,
    }


def write_real_corner_tables(results: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    csv_path = directory / "real_corners.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "corner_err_px", "corner_pct", "corner_worst_px", "pck@2%", "quad_iou"])
        for variant, summary in (("corner_heat", results["model"]), ("classical", results["baseline"])):
            writer.writerow([
                variant,
                round(summary["corner_err"]["mean"], 3),
                round(summary["corner_pct"]["mean"], 3),
                round(summary["corner_worst"]["mean"], 3),
                round(summary["pck"], 4),
                round(summary["quad_iou"]["mean"], 4),
            ])

    per_sample_path = directory / "real_corners_per_sample.csv"
    with per_sample_path.open("w", newline="") as handle:
        fieldnames = ["id", "source", "corner_err", "corner_pct", "corner_worst", "pck", "quad_iou"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["per_sample"]:
            writer.writerow({key: row[key] for key in fieldnames})

    model, baseline = results["model"], results["baseline"]
    lines = [
        "### Corner detection on the real, Roboflow-labelled photos",
        "",
        f"{results['n']} annotated photos. The detector fell back to the classical "
        f"path on {results['fallbacks']}/{results['n']}.",
        "",
        "| Variant | corner error (px) | % of diagonal | PCK@2% | quad IoU |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| corner_heat | {model['corner_err']['mean']:.2f} ± {model['corner_err']['std']:.2f} "
        f"| {model['corner_pct']['mean']:.2f}% | {model['pck']:.3f} | {model['quad_iou']['mean']:.4f} |",
        f"| *classical baseline* | *{baseline['corner_err']['mean']:.2f}* "
        f"| *{baseline['corner_pct']['mean']:.2f}%* | *{baseline['pck']:.3f}* "
        f"| *{baseline['quad_iou']['mean']:.4f}* |",
        "",
        "Mean Euclidean distance between predicted and true corners in each photo's own "
        "pixel space (photos are ~1920x2560 but not all identical), against the four "
        "hand-labelled corners of every annotated real photo — not the synthetic buckets. "
        "PCK is the fraction of photos where all four corners land within 2% of the photo's "
        "own diagonal. The baseline is Canny + findContours + approxPolyDP on the full-resolution "
        "photo, with an undetected page scored as the whole frame.",
    ]
    (directory / "real_corners.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# enhancement vs CamScanner, with OCR readability  (brief §3.3)
# ---------------------------------------------------------------------------
def _zoom_box(width: int, height: int) -> tuple[int, int, int, int]:
    """A fixed fractional crop — the upper-third text band every A4 page has."""
    x0, y0 = int(width * 0.08), int(height * 0.10)
    x1, y1 = int(width * 0.55), int(height * 0.30)
    return x0, y0, x1, y1


def _triplet_figure(input_img, ours_img, reference_img, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    height, width = ours_img.shape[:2]
    box = _zoom_box(width, height)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles = ["rectified input", "ours", "CamScanner reference"]
    for col, (image, title) in enumerate(zip((input_img, ours_img, reference_img), titles)):
        axes[0, col].imshow(image)
        axes[0, col].set_title(title, fontsize=11)
        axes[0, col].axis("off")
        x0, y0, x1, y1 = box
        axes[0, col].add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e74c3c", linewidth=1.5)
        )
        crop = image[y0:y1, x0:x1]
        axes[1, col].imshow(crop)
        axes[1, col].axis("off")
    axes[1, 0].set_title("zoom", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_real_enhancement(
    enhancer, corners: dict, photos_dir: Path, reference_dir: Path,
    transcripts_dir: Path, figures_dir: Path, device,
) -> list[dict]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    reference_files = {
        path.stem.removesuffix("_CS"): path for path in reference_dir.glob("*") if path.is_file()
    }
    stems = sorted(set(corners) & set(reference_files))

    for stem in stems:
        photo = imread_rgb(photos_dir / f"{stem}.jpg")
        quad = np.asarray(corners[stem], dtype=np.float32)

        rectified = rectify_document(photo, quad, out_width=1024, aspect="a4")
        ours = enhance_document(rectified, enhancer, device=device)
        reference = imread_rgb(reference_files[stem])
        reference = cv2.resize(reference, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_AREA)

        _triplet_figure(rectified, ours, reference, figures_dir / f"triplet_{stem}.png")

        row = {"id": stem}
        for label, image in (("input", rectified), ("ours", ours), ("reference", reference)):
            stats = ocr_confidence(image)
            row[f"{label}_confidence"] = round(stats["mean_confidence"], 2)
            row[f"{label}_words"] = stats["word_count"]

        transcript_path = transcripts_dir / f"{stem}.txt"
        if transcript_path.is_file():
            reference_text = transcript_path.read_text()
            for label, image in (("input", rectified), ("ours", ours), ("reference", reference)):
                hypothesis = ocr_text(image)
                row[f"{label}_cer"] = round(cer(reference_text, hypothesis), 4)
                row[f"{label}_wer"] = round(wer(reference_text, hypothesis), 4)

        rows.append(row)
        print(f"  {stem}: conf in={row['input_confidence']:.0f} ours={row['ours_confidence']:.0f} "
              f"ref={row['reference_confidence']:.0f}")

    return rows


def write_real_enhancement_table(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row})
    fieldnames = ["id"] + [key for key in fieldnames if key != "id"]
    csv_path = directory / "real_enhancement_ocr.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _mean(key: str) -> float:
        values = [row[key] for row in rows if key in row]
        return float(np.mean(values)) if values else float("nan")

    lines = [
        "### Enhancement vs CamScanner, on the photos with a reference scan",
        "",
        f"{len(rows)} photos rectified with their annotated corners (never the detector's own — "
        "this isolates the enhancement network). Tesseract's own mean word confidence "
        "(0-100) and word count, identical preprocessing on all three variants.",
        "",
        "| Photo | input conf | input words | ours conf | ours words | reference conf | reference words |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['input_confidence']:.0f} | {row['input_words']} "
            f"| {row['ours_confidence']:.0f} | {row['ours_words']} "
            f"| {row['reference_confidence']:.0f} | {row['reference_words']} |"
        )
    lines += [
        "",
        f"**Mean word confidence** — input {_mean('input_confidence'):.1f}, "
        f"ours {_mean('ours_confidence'):.1f}, reference {_mean('reference_confidence'):.1f}.",
    ]
    cer_rows = [row for row in rows if "input_cer" in row]
    if cer_rows:
        lines += ["", "Character/word error rate against a hand-typed transcript:", ""]
        lines += ["| Photo | input CER | input WER | ours CER | ours WER | reference CER | reference WER |",
                   "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in cer_rows:
            lines.append(
                f"| {row['id']} | {row['input_cer']:.3f} | {row['input_wer']:.3f} "
                f"| {row['ours_cer']:.3f} | {row['ours_wer']:.3f} "
                f"| {row['reference_cer']:.3f} | {row['reference_wer']:.3f} |"
            )
    (directory / "real_enhancement_ocr.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default="corner_heat", help="run name under outputs/runs")
    parser.add_argument("--enhancer", default="enhance_realistic", help="run name under outputs/runs")
    args = parser.parse_args(argv)

    device = get_device()
    print(f"device: {device}")

    corners = read_json(paths.real_corners)
    print(f"{len(corners)} annotated real photos")

    print("\n-- corner detection --")
    detector, _ = load_model(paths.runs / args.detector / "best.pt", device=device)
    corner_results = evaluate_real_corners(detector, corners, paths.real_photos, device)
    write_real_corner_tables(corner_results, paths.tables)
    model = corner_results["model"]
    print(f"corner_heat: {model['corner_err']['mean']:.2f} px  PCK@2% {model['pck']:.3f}  "
          f"(classical {corner_results['baseline']['corner_err']['mean']:.2f} px)")

    print("\n-- enhancement vs CamScanner --")
    enhancer, _ = load_model(paths.runs / args.enhancer / "best.pt", device=device)
    enhancement_rows = evaluate_real_enhancement(
        enhancer, corners, paths.real_photos, paths.real_reference,
        paths.real_transcripts, paths.figures / "real", device,
    )
    write_real_enhancement_table(enhancement_rows, paths.tables)

    print(f"\nwrote reports/tables/real_corners.{{csv,md}}, "
          f"reports/tables/real_enhancement_ocr.{{csv,md}}, "
          f"reports/figures/real/triplet_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
