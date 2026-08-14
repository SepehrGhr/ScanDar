#!/usr/bin/env python
"""Parse the Roboflow export into ``data/real/corners.json``  *(brief §2.2)*.

    python scripts/parse_roboflow.py

Reads every ``*.json`` in ``data/real/annotations/`` and writes one ordered
``(4, 2)`` array per photo, in the photo's own original pixels, keyed by the
stem the rest of the project uses (``Image7``, not Roboflow's mangled export
filename).

**The export is not COCO Keypoints.** The brief asks for that format, but this
project's Roboflow project was labelled with the polygon tool, so what lands
here is an instance-segmentation annotation: one four-vertex polygon per photo,
``annotation["segmentation"][0]``, rather than a ``keypoints`` array. Both encode
the same thing — four ordered points — so this parser reads whichever is present
and treats them identically. A photo Roboflow has no annotation for (not
labelled, or deliberately dropped from the dataset) is simply absent from the
output; nothing here decides which photos are "in", the annotation export does.

**Corner order is not trusted blindly.** Every quad is run through
:func:`~scandar.geometry.order_corners`, and if that reorders more than one
point the photo is flagged on stdout rather than silently corrected — a
disagreement this large usually means a genuinely ambiguous or mislabelled
page (see ``order_corners``'s own note about pages rotated past 45°), and is
worth a human looking at the photo once rather than trusting the canonicaliser
to have guessed right.

**Scaling.** Roboflow records the pixel size it labelled at (``images[].width``/
``height``). If that ever disagrees with the actual file on disk — a re-export,
a re-compress, anything — the points are rescaled with
:func:`~scandar.geometry.scale_points`, the same half-pixel-correct rescale used
everywhere else corners follow a resize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from scandar.geometry import order_corners, scale_points
from scandar.io import imread_rgb, paths, write_json


def _photo_stem(image_entry: dict) -> str | None:
    """The project's own filename for an annotated image, e.g. ``Image7``."""
    extra = image_entry.get("extra") or {}
    name = extra.get("name") or image_entry.get("file_name")
    if not name:
        return None
    return Path(name).stem


def _quad_from_annotation(annotation: dict) -> np.ndarray | None:
    """Four ``(x, y)`` points from either a ``keypoints`` or a polygon export."""
    keypoints = annotation.get("keypoints")
    if keypoints:
        flat = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)  # x, y, visibility
        if flat.shape[0] >= 4:
            return flat[:4, :2]

    segmentation = annotation.get("segmentation")
    if segmentation:
        polygon = np.asarray(segmentation[0], dtype=np.float64).reshape(-1, 2)
        if polygon.shape[0] == 4:
            return polygon
    return None


def parse_roboflow(annotations_dir: Path, photos_dir: Path) -> dict[str, np.ndarray]:
    corners: dict[str, np.ndarray] = {}
    exports = sorted(annotations_dir.glob("*.json"))
    if not exports:
        raise FileNotFoundError(f"no *.json export found in {annotations_dir}")

    for export_path in exports:
        payload = json.loads(export_path.read_text())
        images = {entry["id"]: entry for entry in payload.get("images", [])}

        for annotation in payload.get("annotations", []):
            image = images.get(annotation["image_id"])
            if image is None:
                continue
            stem = _photo_stem(image)
            if stem is None:
                continue

            quad = _quad_from_annotation(annotation)
            if quad is None:
                print(f"  ! {stem}: annotation has neither keypoints nor a 4-point polygon, skipped")
                continue

            photo_path = photos_dir / f"{stem}.jpg"
            if not photo_path.is_file():
                print(f"  ! {stem}: labelled but no matching file in {photos_dir}, skipped")
                continue
            actual_height, actual_width = imread_rgb(photo_path).shape[:2]
            labelled_width = float(image.get("width", actual_width))
            labelled_height = float(image.get("height", actual_height))
            if (labelled_width, labelled_height) != (actual_width, actual_height):
                quad = scale_points(
                    quad, actual_width / labelled_width, actual_height / labelled_height
                )

            ordered = order_corners(quad)
            moved = int(np.sum(np.any(np.abs(ordered - quad) > 1.0, axis=1)))
            if moved > 1:
                print(
                    f"  ! {stem}: order_corners moved {moved}/4 points by >1px — "
                    f"flagged for manual review, not auto-corrected"
                )

            corners[stem] = ordered

    return corners


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=paths.real_annotations)
    parser.add_argument("--photos", type=Path, default=paths.real_photos)
    parser.add_argument("--out", type=Path, default=paths.real_corners)
    args = parser.parse_args(argv)

    corners = parse_roboflow(args.annotations, args.photos)
    if not corners:
        print("no annotated photos found — nothing written")
        return 1

    payload = {stem: corners[stem].tolist() for stem in sorted(corners)}
    write_json(args.out, payload)
    print(f"wrote {len(payload)} photos' corners -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
