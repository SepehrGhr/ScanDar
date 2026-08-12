#!/usr/bin/env python
"""Render the synthetic generator's output so it can be inspected by eye.

    python scripts/preview_synth.py [--split train] [--count 12] [--out DIR]

The brief asks for exactly this before any model is trained (§4.4): look at a
batch of generated samples, check the corners land on the page, check the
rectified input and the clean target line up, and put a few next to the real
photos — if a stranger can tell instantly which is which, the degradations are
not realistic enough yet.

Writes into ``outputs/previews/`` by default, which is not tracked; the polished
figures for the report are a separate job.
"""

import argparse

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

import numpy as np

from scandar.config import load_config
from scandar.io import imwrite_rgb, list_images, paths
from scandar.seed import rng_for
from scandar.synth import build_sources

CORNER_COLORS = ((255, 64, 64), (64, 200, 64), (64, 140, 255), (255, 200, 0))  # TL TR BR BL
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def fit(image, width, height, pad=6):
    """Letterbox *image* into a ``width x height`` tile on a dark background."""
    import cv2

    tile = np.full((height, width, 3), 24, dtype=np.uint8)
    inner_w, inner_h = width - 2 * pad, height - 2 * pad
    scale = min(inner_w / image.shape[1], inner_h / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return tile


def label(tile, text, origin=(8, 26)):
    import cv2

    cv2.putText(tile, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 80), 1, cv2.LINE_AA)
    return tile


def grid(tiles, columns):
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def draw_corners(photo, corners):
    """Overlay the quad and its four labelled corners."""
    import cv2

    canvas = photo.copy()
    points = np.round(corners).astype(np.int32)
    cv2.polylines(canvas, [points], True, (255, 255, 255), 3, cv2.LINE_AA)
    for index, (point, color, name) in enumerate(zip(points, CORNER_COLORS, CORNER_NAMES)):
        cv2.circle(canvas, tuple(point), 16, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, name, (point[0] + 22, point[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA,
        )
        del index
    return canvas


def composites_sheet(sources, count, seed_key):
    tiles = []
    for index in range(count):
        sample = sources.compose(rng_for(seed_key, "composite", index))
        tile = fit(draw_corners(sample.photo, sample.corners), 300, 400)
        background = sample.params["background"]
        name = background.get("source", background.get("texture", "?")).removesuffix(".jpg")
        tiles.append(label(tile, name[:26]))
    return grid(tiles, 6 if count > 6 else count)


def steps_strip(sources, seed_key):
    """The degradation, stage by stage — the figure the report asks for."""
    sample = sources.compose(rng_for(seed_key, "steps"), keep_clean=True, collect_steps=True)
    tiles = [label(fit(sample.clean_photo, 260, 350), "composite")]
    for name, image in sample.steps[1:]:
        tiles.append(label(fit(image, 260, 350), name.replace("_", " ")))
    return grid(tiles, 5)


def pairs_sheet(sources, count, rect_size, seed_key):
    """Rectified input, clean target, and the difference between them."""
    import cv2

    rows = []
    for index in range(count):
        sample = sources.compose(rng_for(seed_key, "pair", index), keep_clean=True)
        degraded, target = sample.rectify(rect_size)
        aligned, _ = sample.rectify(rect_size, source=sample.clean_photo)

        difference = cv2.absdiff(aligned, target)
        amplified = np.clip(difference.astype(np.int32) * 4, 0, 255).astype(np.uint8)
        shift, _ = cv2.phaseCorrelate(
            cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY).astype(np.float64),
            cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float64),
        )
        rows.extend(
            [
                label(fit(degraded, 300, 420), "degraded input"),
                label(fit(target, 300, 420), "clean target"),
                label(fit(amplified, 300, 420), f"|diff| x4  shift {np.hypot(*shift):.2f}px"),
            ]
        )
    return grid(rows, 3)


def patches_sheet(sources, count, patch_size, rect_size, seed_key):
    tiles = []
    for index in range(count // 2):
        rng = rng_for(seed_key, "patch", index)
        sample = sources.compose(rng)
        for _ in range(2):
            degraded, target, _ = sample.random_patch(rng, patch_size, rect_size)
            tiles.append(label(fit(degraded, 260, 260), "in"))
            tiles.append(label(fit(target, 260, 260), "target"))
    return grid(tiles, 8)


def real_comparison_sheet(sources, seed_key):
    """Synthetic and real photos side by side — the brief's "spot the fake" test."""
    from scandar.io import imread_rgb

    real = list_images(paths.real_photos)[:6]
    tiles = []
    for index, path in enumerate(real):
        tiles.append(label(fit(imread_rgb(path), 300, 400), f"real: {path.stem}"))
    for index in range(len(real)):
        sample = sources.compose(rng_for(seed_key, "spot", index))
        tiles.append(label(fit(sample.photo, 300, 400), "synthetic"))
    return grid(tiles, max(1, len(real)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(paths.repo / "configs" / "base.yaml"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", default="preview")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    directory = args.out or (paths.out / "previews")
    sources = build_sources(config, args.split, task="corner")
    enhance_sources = build_sources(config, args.split, task="enhance")
    rect_size = tuple(config.data.rect_size)

    sheets = {
        "composites.jpg": composites_sheet(sources, args.count, args.seed),
        "degradation_steps.jpg": steps_strip(sources, args.seed),
        "rectified_pairs.jpg": pairs_sheet(enhance_sources, 3, rect_size, args.seed),
        "patches.jpg": patches_sheet(enhance_sources, 16, config.data.patch_size, rect_size, args.seed),
        "spot_the_fake.jpg": real_comparison_sheet(sources, args.seed),
    }
    for name, sheet in sheets.items():
        written = imwrite_rgb(f"{directory}/{name}", sheet, quality=92)
        print(f"  {written}  {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
