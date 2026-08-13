#!/usr/bin/env python
"""Find the page corners in a pile of your own photos, into one folder.

    python scripts/detect_batch.py --input data/real/photos
    python scripts/detect_batch.py --input a.jpg b.jpg c.jpg
    python scripts/detect_batch.py --input photos/ --out outputs/corners/my_test

For each photo: the four corners are predicted, drawn on the photo, and the page
is flattened with them. The flattened page is the panel to actually look at — a
quad drawn on a photo always looks roughly right, and a page that comes out
skewed, cropped or upside down tells you immediately that it was not
*(brief §5.1)*.

**The corners land in ``corners.json`` in exactly the format
``enhance_batch.py`` reads.** Point that script at this folder and it will
restore the pages without opening a single window:

    python scripts/detect_batch.py  --input data/real/photos --out outputs/corners/real
    python scripts/enhance_batch.py --input data/real/photos --out outputs/corners/real

which is the same thing ``enhance_batch.py --detect`` does in one command, split
in two so the corners can be inspected — and corrected — in between.

For a heatmap detector the maps themselves are written out too. They are the most
informative failure diagnostic in the project: a corner that is wrong because the
map is *diffuse* is a model that does not know, and a corner that is wrong because
the map has *two peaks* is a model that found the wrong page. Those two want
opposite fixes, and nothing but the map distinguishes them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.device import get_device
from scandar.io import collect_inputs, imread_rgb, imwrite_rgb, paths
from scandar.model import load_model
from scandar.pipelines import detect_corners, draw_corners, rectify_document

DEFAULT_RUN = "corner_heat"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def heatmap_overlay(photo: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """The four maps summed and laid over the photo, in the photo's own frame.

    Summed rather than shown as four panels, because what matters at a glance is
    *where the model is looking*, and four separate thumbnails of a mostly-black
    128x128 map answer that far worse than one picture does.
    """
    import cv2

    combined = heatmaps.max(axis=0)
    span = float(combined.max()) - float(combined.min())
    combined = (combined - float(combined.min())) / (span if span > 1e-9 else 1.0)
    heat = cv2.applyColorMap((combined * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(photo, 0.55, heat, 0.45, 0.0)


def result_figure(photo, result, page, out_path: Path, title: str) -> Path:
    """One page per photo: the overlay, where the model looked, and the result."""
    import matplotlib.pyplot as plt

    from scandar.viz import use_style

    use_style()
    panels = [(draw_corners(photo, result["corners"]), f"predicted corners ({result['source']})")]
    if result.get("heatmaps") is not None:
        panels.append((heatmap_overlay(photo, result["heatmaps"]), "where the model looked"))
    panels.append((page, "flattened with those corners — the panel that tells the truth"))

    figure, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 6),
                                constrained_layout=True)
    for axis, (image, caption) in zip(np.atleast_1d(axes), panels):
        axis.imshow(image)
        axis.set_title(caption, fontsize=10)
        axis.axis("off")

    figure.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def contact_sheet(results, out_path: Path, columns: int = 4) -> Path | None:
    """Every detection at a glance — the fastest way to spot the one that failed."""
    import matplotlib.pyplot as plt

    from scandar.viz import COLORS, use_style

    if not results:
        return None
    use_style()
    columns = min(columns, len(results))
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 4.6 * rows))
    for axis, entry in zip(np.atleast_1d(axes).ravel(), results):
        axis.imshow(imread_rgb(entry["overlay"]))
        # The path is in the title because a detection that came from the
        # fallback is a different claim from one the network made, and on a sheet
        # of twenty thumbnails that is the only place to say so.
        axis.set_title(
            f"{Path(entry['input']).name}\n{entry['source']}",
            fontsize=8,
            color=COLORS["train"] if entry["source"] == "model" else COLORS["val"],
        )
        axis.axis("off")
    for axis in np.atleast_1d(axes).ravel()[len(results):]:
        axis.axis("off")
    figure.suptitle(f"{len(results)} photos, corners detected", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", nargs="+", required=True, help="files, directories or globs")
    parser.add_argument("--out", default=None,
                        help="results folder (default: outputs/corners/<run name>)")
    parser.add_argument("--checkpoint", default=None,
                        help=f"default: outputs/runs/{DEFAULT_RUN}/best.pt")
    parser.add_argument("--force", action="store_true",
                        help="redo photos that already have results in the folder")
    parser.add_argument("--no-fallback", action="store_true",
                        help="do not fall back to the classical detector on a bad quad")
    parser.add_argument("--no-figures", action="store_true", help="write only the overlays")
    parser.add_argument("--width", type=int, default=1024, help="flattened page width in px")
    parser.add_argument("--aspect", default=None, help="page width/height, or 'a4'")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint or paths.runs / DEFAULT_RUN / "best.pt")
    if not checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint}\n"
            "Train one (`python train.py --config configs/corner_heat.yaml`), or pass "
            "--checkpoint <path to a best.pt>"
        )

    out_dir = Path(args.out) if args.out else paths.out / "corners" / checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    corners_path = out_dir / "corners.json"
    saved = json.loads(corners_path.read_text(encoding="utf-8")) if corners_path.exists() else {}

    photos = collect_inputs(args.input)
    if not photos:
        raise SystemExit("no images found")

    device = get_device()
    model, config = load_model(checkpoint, device=device)
    input_size = int(config.get("data", {}).get("corner_input", 256))

    print(f"model   : {checkpoint}  ({type(model).__name__}, {input_size}x{input_size} input)")
    print(f"device  : {device}")
    print(f"results : {out_dir}")
    print(f"photos  : {len(photos)}\n")

    results = []
    counts: dict[str, int] = {}
    for photo_path in photos:
        stem = photo_path.stem
        overlay_path = out_dir / f"{stem}_corners.png"
        if overlay_path.exists() and not args.force:
            print(f"  {stem:<20} already done, skipping (--force to redo)")
            continue

        photo = imread_rgb(photo_path)
        result = detect_corners(photo, model, device=device, input_size=input_size,
                                fallback=not args.no_fallback)
        page = rectify_document(photo, result["corners"], out_width=args.width,
                                aspect=args.aspect)

        imwrite_rgb(overlay_path, draw_corners(photo, result["corners"]))
        imwrite_rgb(out_dir / f"{stem}_rectified.png", page)
        figure_path = None
        if not args.no_figures:
            figure_path = result_figure(photo, result, page, out_dir / f"{stem}_figure.png",
                                        f"{stem}   —   {checkpoint.parent.name}")

        counts[result["source"]] = counts.get(result["source"], 0) + 1
        saved[stem] = np.asarray(result["corners"]).tolist()
        corners_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        results.append({
            "input": str(photo_path),
            "overlay": str(overlay_path),
            "rectified": str(out_dir / f"{stem}_rectified.png"),
            "figure": str(figure_path) if figure_path else None,
            "corners": np.asarray(result["corners"]).tolist(),
            "source": result["source"],
            "problem": result["problem"],
            "confidence": result["confidence"],
        })

        confidence = "" if result["confidence"] is None else f"  peak {result['confidence']:.2f}"
        note = "" if result["source"] == "model" else f"  <- {result['source']} path"
        print(f"  {stem:<20} {photo.shape[1]}x{photo.shape[0]}{confidence}{note}")

    if not results:
        print("\nnothing to do")
        return 0

    sheet = contact_sheet(results, out_dir / "contact_sheet.png")
    (out_dir / "manifest.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "input_size": input_size,
                    "sources": counts, "results": results}, indent=2),
        encoding="utf-8",
    )

    print(f"\n{len(results)} photo(s) processed")
    print("  paths taken   : " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    print(f"  contact sheet : {sheet}")
    print(f"  corners       : {corners_path}")
    print(f"  everything in : {out_dir}")
    if counts.get("model", 0) < len(results):
        print(
            "\n  Some photos did not come from the network. That is the guardrail doing its\n"
            "  job, not a crash — look at those overlays first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
