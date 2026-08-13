#!/usr/bin/env python
"""Run the enhancement network over a pile of your own photos, into one folder.

    python scripts/enhance_batch.py --input data/real/photos
    python scripts/enhance_batch.py --input a.jpg b.jpg c.jpg
    python scripts/enhance_batch.py --input photos/ --out outputs/scans/my_test

For each photo a window opens: click the four corners of the page, in any order,
then close it. The page is flattened, restored, and written out with a figure
showing what happened at each step. Everything lands in one folder per model, so
a whole set of results can be looked through in one place and handed to someone
else as-is.

**The corners are remembered.** They go into ``corners.json`` beside the results,
so re-running after retraining reuses them and asks for nothing — which is the
point, because comparing two models on the same photos is only fair if the
rectification is identical, and clicking forty corners twice by hand would not
be. ``--reclick`` throws them away and asks again.

``--detect`` replaces the clicking with the trained corner detector, and no window
opens at all:

    python scripts/enhance_batch.py --input data/real/photos --detect

That is the whole scanner — photo in, clean scan out, no human input. It is worth
keeping both paths: the clicked corners are the closest thing this project has to
ground truth on a real photo, so running the same photos both ways is what prices
exactly what a corner error costs the enhancement stage *(brief §7)*.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.device import get_device
from scandar.io import collect_inputs, imread_rgb, imwrite_rgb, paths
from scandar.model import load_model
from scandar.pipelines import detect_corners, draw_corners, enhance_document, rectify_document

DEFAULT_RUN = "enhance_realistic"
DEFAULT_DETECTOR = "corner_heat"


# ---------------------------------------------------------------------------
# corners
# ---------------------------------------------------------------------------
def require_display() -> None:
    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "no display to open a window on (running over ssh?).\n"
            "Pick the corners on a machine with a screen, or copy a corners.json in beside "
            "the results and re-run."
        )


def detect_batch_corners(needed, args, saved: dict, corners_path: Path) -> dict:
    """Fill in the corners with the trained detector instead of with a human.

    Written into the same ``corners.json`` the clicking writes, so the two paths
    are interchangeable and a later run cannot tell — or care — which one produced
    them. Which path each photo took *is* recorded, because a page found by the
    classical fallback is a different claim from one the network found, and the
    difference matters when a result looks wrong.
    """
    checkpoint = Path(args.detector or paths.runs / DEFAULT_DETECTOR / "best.pt")
    if not checkpoint.exists():
        raise SystemExit(
            f"no corner detector at {checkpoint}\n"
            "Train one (`python train.py --config configs/corner_heat.yaml`), pass "
            "--detector <path>, or drop --detect and click the corners instead."
        )

    device = get_device()
    model, config = load_model(checkpoint, device=device)
    input_size = int(config.get("data", {}).get("corner_input", 256))
    print(f"  detecting corners with {checkpoint} on {device}\n")

    sources = {}
    for photo_path, stem, _ in needed:
        result = detect_corners(imread_rgb(photo_path), model, device=device,
                                input_size=input_size)
        saved[stem] = np.asarray(result["corners"]).tolist()
        sources[stem] = result["source"]
        corners_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        note = "" if result["source"] == "model" else f"  <- {result['source']} path"
        print(f"  {stem:<20} corners found{note}")
    return sources


def click_corners(photo: np.ndarray, title: str) -> np.ndarray | None:
    """Four clicks on one photo. Returns None if the window was closed early.

    Order does not matter — the pipeline sorts the quad into the project's
    canonical TL, TR, BR, BL itself, because nobody clicking corners at a demo
    reliably starts at the top left.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 11))
    axes.imshow(photo)
    axes.set_title(f"{title}\nclick the four corners of the page, then close the window",
                   fontsize=10)
    axes.axis("off")
    figure.tight_layout()
    picked = figure.ginput(4, timeout=0, show_clicks=True)
    plt.close(figure)
    return np.array(picked, dtype=np.float32) if len(picked) == 4 else None


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def densest_ink_window(image: np.ndarray, height: int, width: int):
    """Where on the page is there most writing? That is what a reader will judge.

    A zoom panel cropped from the middle of the page lands on blank margin about
    as often as not, which makes a figure that says nothing about legibility.
    """
    import cv2

    darkness = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Zero-padded rather than the default reflection: reflecting a page's own
    # writing back across its edge invents ink that is not there and pulls the
    # window towards the margins, which is the one place it must not go.
    ink = cv2.boxFilter(
        darkness.astype(np.float32), -1, (width, height),
        normalize=False, borderType=cv2.BORDER_CONSTANT,
    )
    half_h, half_w = height // 2, width // 2
    interior = ink[half_h:-half_h or None, half_w:-half_w or None]
    if interior.size == 0:
        return slice(0, height), slice(0, width)
    y, x = np.unravel_index(int(interior.argmax()), interior.shape)
    y, x = y + half_h, x + half_w
    top = max(0, min(y - half_h, image.shape[0] - height))
    left = max(0, min(x - half_w, image.shape[1] - width))
    return slice(top, top + height), slice(left, left + width)


def result_figure(photo, quad, page, enhanced, out_path: Path, title: str,
                  corner_caption: str = "photo, with the corners you picked") -> Path:
    """One page per photo: what went in, what was flattened, what came out.

    The zoom row is the row that matters. Whole pages at thumbnail size all look
    fine; whether the handwriting is *readable* only shows at full resolution, and
    putting the before and after crops side by side is the only honest way to
    show it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from scandar.viz import use_style

    use_style()
    # A wide, short crop: a line of handwriting is wide and short, and it fits a
    # full-width row without the panels being scaled down to illegibility, which
    # would defeat the point of showing them at full resolution.
    zoom_h = int(min(page.shape[0] / 6, 220))
    zoom_w = int(min(page.shape[1], zoom_h * 3.6))
    rows, cols = densest_ink_window(enhanced, zoom_h, zoom_w)

    # constrained_layout rather than hand-tuned spacing: the three top panels have
    # different aspect ratios (a 4:3 photo beside two A4 pages) and fixed padding
    # that suits one set of shapes clips another.
    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = GridSpec(2, 6, figure=figure, height_ratios=[2.3, 1.0])

    for index, (image, caption) in enumerate((
        (draw_corners(photo, quad), corner_caption),
        (page, "flattened page — what the network sees"),
        (enhanced, "restored — what it produced"),
    )):
        axes = figure.add_subplot(grid[0, index * 2 : index * 2 + 2])
        axes.imshow(image)
        axes.set_title(caption, fontsize=10)
        axes.axis("off")

    for index, (image, caption) in enumerate((
        (page[rows, cols], "before, at full resolution"),
        (enhanced[rows, cols], "after, at full resolution"),
    )):
        axes = figure.add_subplot(grid[1, index * 3 : index * 3 + 3])
        axes.imshow(image)
        axes.set_title(caption, fontsize=10)
        axes.axis("off")

    figure.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def contact_sheet(results, out_path: Path, columns: int = 4) -> Path | None:
    """Every restored page at a glance, for judging a whole set at once."""
    import matplotlib.pyplot as plt

    from scandar.viz import use_style

    if not results:
        return None
    use_style()
    columns = min(columns, len(results))
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 4.3 * rows))
    for axis, entry in zip(np.atleast_1d(axes).ravel(), results):
        axis.imshow(imread_rgb(entry["scan"]))
        axis.set_title(Path(entry["input"]).name, fontsize=8)
        axis.axis("off")
    for axis in np.atleast_1d(axes).ravel()[len(results):]:
        axis.axis("off")
    figure.suptitle(f"{len(results)} restored pages", fontsize=12)
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
                        help="results folder (default: outputs/scans/<run name>)")
    parser.add_argument("--checkpoint", default=None,
                        help=f"default: outputs/runs/{DEFAULT_RUN}/best.pt")
    parser.add_argument("--detect", action="store_true",
                        help="find the corners with the trained detector instead of clicking")
    parser.add_argument("--detector", default=None,
                        help=f"detector checkpoint (default: outputs/runs/{DEFAULT_DETECTOR}/best.pt)")
    parser.add_argument("--reclick", action="store_true",
                        help="ignore saved corners and pick them again")
    parser.add_argument("--force", action="store_true",
                        help="redo photos that already have results in the folder")
    parser.add_argument("--no-rectify", action="store_true",
                        help="the inputs are already flat pages — never open a window")
    parser.add_argument("--width", type=int, default=1024, help="flattened page width in px")
    parser.add_argument("--aspect", default=None, help="page width/height, or 'a4'")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=192)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint or paths.runs / DEFAULT_RUN / "best.pt")
    if not checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint}\n"
            "Train one first, or pass --checkpoint <path to a best.pt>"
        )

    out_dir = Path(args.out) if args.out else paths.out / "scans" / checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    corners_path = out_dir / "corners.json"
    saved = {} if args.reclick or not corners_path.exists() else json.loads(
        corners_path.read_text(encoding="utf-8")
    )

    photos = collect_inputs(args.input)
    if not photos:
        raise SystemExit("no images found")

    print(f"model   : {checkpoint}")
    print(f"results : {out_dir}")
    print(f"photos  : {len(photos)}\n")

    # Everything that needs a click is collected before the model is loaded, so
    # the GPU is not sitting idle while a human decides where a corner is.
    todo = []
    for photo_path in photos:
        stem = photo_path.stem
        scan_path = out_dir / f"{stem}_scan.png"
        if scan_path.exists() and not args.force and not args.reclick:
            print(f"  {stem:<20} already done, skipping (--force to redo)")
            continue
        todo.append((photo_path, stem, scan_path))

    if not todo:
        print("\nnothing to do")
        return 0

    detector_paths = {}
    if not args.no_rectify:
        needed = [t for t in todo if t[1] not in saved]
        if needed and args.detect:
            detector_paths = detect_batch_corners(needed, args, saved, corners_path)
        elif needed:
            require_display()
            print(f"  picking corners for {len(needed)} photo(s) — close each window when done\n")
            for index, (photo_path, stem, _) in enumerate(needed, start=1):
                picked = click_corners(imread_rgb(photo_path), f"[{index}/{len(needed)}]  {stem}")
                if picked is None:
                    print(f"  {stem:<20} skipped (window closed before four clicks)")
                    continue
                saved[stem] = picked.tolist()
                corners_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    device = get_device()
    model, _ = load_model(checkpoint, device=device)
    print(f"\n  running on {device}\n")

    results = []
    for photo_path, stem, scan_path in todo:
        photo = imread_rgb(photo_path)
        if args.no_rectify:
            quad = np.float32([[0, 0], [photo.shape[1] - 1, 0],
                               [photo.shape[1] - 1, photo.shape[0] - 1], [0, photo.shape[0] - 1]])
            page = photo
        elif stem in saved:
            quad = np.asarray(saved[stem], dtype=np.float32)
            page = rectify_document(photo, quad, out_width=args.width, aspect=args.aspect)
        else:
            continue  # its window was closed early; already reported

        enhanced = enhance_document(page, model, device=device,
                                    tile=args.tile, overlap=args.overlap)
        imwrite_rgb(scan_path, enhanced)
        imwrite_rgb(out_dir / f"{stem}_rectified.png", page)
        source = detector_paths.get(stem, "clicked")
        figure_path = result_figure(
            photo, quad, page, enhanced, out_dir / f"{stem}_figure.png",
            f"{stem}   —   {checkpoint.parent.name}",
            corner_caption="photo, with the corners you picked" if source == "clicked"
            else f"photo, with the corners the detector found ({source} path)",
        )
        results.append({"input": str(photo_path), "scan": str(scan_path),
                        "figure": str(figure_path), "corners": np.asarray(quad).tolist(),
                        "corners_from": detector_paths.get(stem, "clicked")})
        print(f"  {stem:<20} {page.shape[1]}x{page.shape[0]}  ->  {scan_path.name}")

    if results:
        sheet = contact_sheet(results, out_dir / "contact_sheet.png")
        (out_dir / "manifest.json").write_text(
            json.dumps({"checkpoint": str(checkpoint), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\n{len(results)} page(s) restored")
        print(f"  contact sheet : {sheet}")
        print(f"  everything in : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
