#!/usr/bin/env python
"""Run the enhancement network on a photo of your own.

    python scripts/enhance_photo.py --input photo.jpg
    python scripts/enhance_photo.py --input photo.jpg --corners "310,500 780,440 850,1160 340,1215"
    python scripts/enhance_photo.py --input already_flat_page.png --no-rectify

The network restores *flattened pages* and nothing else, so a raw photo has to
have its page cut out and squared up first. Given a whole photo it will earnestly
try to turn the desk and the wall into white paper too — which is not a bug, it is
a model doing exactly what it was trained to do, on input it was never shown.

Until the corner detector is built, the four corners come from you. Run without
``--corners`` and a window opens: click the four corners of the page, in any
order, and close it. With ``--corners`` no window is needed, which is also how
this runs over ssh or in a notebook.

Writes three files next to the output: the flattened page, the enhanced result,
and a side-by-side comparison to look at.
"""

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.device import get_device
from scandar.io import imread_rgb, imwrite_rgb, paths
from scandar.model import load_model
from scandar.pipelines import enhance_document, rectify_document

DEFAULT_CHECKPOINT = paths.runs / "enhance_baseline" / "best.pt"


def parse_corners(text: str) -> np.ndarray:
    """``"x,y x,y x,y x,y"`` -> a (4, 2) array. Order does not matter."""
    points = [p for p in text.replace(";", " ").split() if p]
    if len(points) != 4:
        raise SystemExit(f"--corners needs exactly four x,y pairs, got {len(points)}")
    try:
        return np.array([[float(v) for v in p.split(",")] for p in points], dtype=np.float32)
    except ValueError as exc:
        raise SystemExit(f'--corners must look like "310,500 780,440 850,1160 340,1215": {exc}')


def click_corners(photo: np.ndarray) -> np.ndarray:
    """Open a window and collect four clicks.

    Deliberately forgiving about the order — the pipeline sorts the quad into the
    project's canonical TL, TR, BR, BL itself, because a human clicking corners
    under time pressure at a demo will not reliably start top-left.
    """
    import os

    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "no display to open a window on (running over ssh?).\n"
            'Pass the corners instead:  --corners "x,y x,y x,y x,y"'
        )
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - any import failure means no window
        raise SystemExit(
            f"cannot open a window for picking corners ({exc}).\n"
            'Pass them instead:  --corners "x,y x,y x,y x,y"'
        )

    figure, axes = plt.subplots(figsize=(9, 12))
    axes.imshow(photo)
    axes.set_title("Click the four corners of the page (any order), then close the window")
    axes.axis("off")
    picked = figure.ginput(4, timeout=0, show_clicks=True)
    plt.close(figure)
    if len(picked) != 4:
        raise SystemExit(f"need four corners, got {len(picked)} — click exactly four")
    return np.array(picked, dtype=np.float32)


def side_by_side(images, height: int = 720) -> np.ndarray:
    import cv2

    scaled = []
    for image in images:
        scale = height / image.shape[0]
        scaled.append(
            cv2.resize(
                image,
                (max(1, int(round(image.shape[1] * scale))), height),
                interpolation=cv2.INTER_AREA,
            )
        )
    gap = np.full((height, 12, 3), 255, dtype=np.uint8)
    strip = [scaled[0]]
    for image in scaled[1:]:
        strip += [gap, image]
    return np.concatenate(strip, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="a photo, or an already-flat page")
    parser.add_argument("--output", default=None, help="default: <input>_scan.png beside the input")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--corners", default=None, help='"x,y x,y x,y x,y" in photo pixels')
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="the input is already a flat page — skip the corner step entirely",
    )
    parser.add_argument("--width", type=int, default=1024, help="flattened page width in px")
    parser.add_argument(
        "--aspect",
        default=None,
        help="width/height of the page; 'a4' for a steeply-shot A4 sheet "
        "(default: estimated from the corners you gave)",
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=192)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint}\n"
            "Train one first:  python train.py --config configs/enhance.yaml"
        )

    source = Path(args.input)
    photo = imread_rgb(source)
    print(f"input      : {source}  ({photo.shape[1]}x{photo.shape[0]})")

    if args.no_rectify:
        page = photo
        print("rectify    : skipped, treating the input as an already-flat page")
    else:
        corners = parse_corners(args.corners) if args.corners else click_corners(photo)
        page = rectify_document(photo, corners, out_width=args.width, aspect=args.aspect)
        listed = " ".join(f"{x:.0f},{y:.0f}" for x, y in corners)
        print(f"corners    : {listed}")
        print(f"rectify    : -> {page.shape[1]}x{page.shape[0]}")

    device = get_device()
    model, _ = load_model(checkpoint, device=device)
    print(f"model      : {checkpoint}  on {device}")

    enhanced = enhance_document(
        page, model, device=device, tile=args.tile, overlap=args.overlap
    )

    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_scan.png")
    imwrite_rgb(output, enhanced)
    written = [output]
    if not args.no_rectify:
        written.append(imwrite_rgb(output.with_name(f"{output.stem}_rectified.png"), page))
    written.append(
        imwrite_rgb(
            output.with_name(f"{output.stem}_comparison.png"),
            side_by_side([page, enhanced]),
        )
    )
    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
