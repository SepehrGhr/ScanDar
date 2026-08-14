#!/usr/bin/env python
"""Render the pictures the top-level README shows, into ``assets/``.

    python scripts/make_readme_assets.py
    python scripts/make_readme_assets.py --only hero
    python scripts/make_readme_assets.py --photo Image13 --gallery Image2 Image9 Image19

The README is the first thing anyone sees, so its pictures are generated rather
than screenshotted: the hero strip and the gallery run the shipped checkpoints
over real photographs through the same pipeline ``scandar scan`` uses, and the
rest are the report's own figures, downscaled to a size a browser will not choke
on. Nothing here is drawn by hand, so every panel can be traced back to the run
that produced it.

Panel captions are drawn in a mid grey on a transparent background, which is the
one colour that stays readable whether the reader's site is in light or dark
mode.

Needs the trained checkpoints under ``outputs/runs/``. ``--only copies`` skips
them and just refreshes the figures the report already wrote.
"""

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from detect_batch import heatmap_overlay
from scandar.device import get_device
from scandar.io import imread_rgb, paths
from scandar.model import load_model
from scandar.pipelines import draw_corners, scan_document

#: The shipped pair: the detector that won the comparison *(brief §5)* and the
#: enhancement network trained on the realistic canvas *(brief §3)*.
DETECTOR = "corner_heat"
ENHANCER = "enhance_realistic"

#: A steeply photographed page on a wood desk, in shadow — the hero has to show
#: a photograph nobody would call easy, or it is showing nothing.
HERO_PHOTO = "Image9"
GALLERY_PHOTOS = ("Image13", "Image2", "Image19")

#: Readable on white and on near-black, which no darker or lighter grey is.
LABEL = "#8b93a1"

#: Rendering resolution for the composed strips, and the size they are capped
#: at afterwards. A README column is under a thousand CSS pixels wide, so
#: anything past this is weight the reader pays for and never sees.
DPI = 130
MAX_SIDE = 1500


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def _shrink(path: Path) -> Path:
    """Cap a rendered panel at :data:`MAX_SIDE` and squeeze the PNG."""
    from PIL import Image

    image = Image.open(path)
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)
    print(f"  {path.relative_to(paths.repo)}  ({path.stat().st_size // 1024} KB)")
    return path


def strip(panels, out_path: Path, height: float = 3.4, dpi: int = DPI) -> Path:
    """One row of images, each keeping its own aspect ratio, captions above.

    *panels* is a sequence of ``(image, caption)``. Widths are allocated from
    the images' own aspect ratios, so a 3:4 photograph and a 1:1.414 page sit
    side by side at the same height instead of being squeezed to a common box.
    """
    import matplotlib.pyplot as plt

    from scandar.viz import use_style

    use_style()
    ratios = [image.shape[1] / image.shape[0] for image, _ in panels]
    figure, axes = plt.subplots(
        1, len(panels),
        figsize=(height * sum(ratios) + 0.25 * len(panels), height + 0.35),
        gridspec_kw={"width_ratios": ratios},
        constrained_layout=True,
    )
    for axis, (image, caption) in zip(np.atleast_1d(axes), panels):
        axis.imshow(image)
        axis.set_title(caption, fontsize=9, color=LABEL, pad=6)
        axis.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(figure)
    return _shrink(out_path)


def grid(panels, out_path: Path, columns: int, height: float = 3.2, dpi: int = DPI) -> Path:
    """The same, wrapped over several rows — for the gallery."""
    import matplotlib.pyplot as plt

    from scandar.viz import use_style

    use_style()
    rows = -(-len(panels) // columns)
    ratios = [panels[i][0].shape[1] / panels[i][0].shape[0] for i in range(min(columns, len(panels)))]
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(height * sum(ratios) + 0.25 * columns, rows * (height + 0.35)),
        gridspec_kw={"width_ratios": ratios},
        constrained_layout=True,
    )
    flat = np.atleast_1d(axes).ravel()
    for axis, (image, caption) in zip(flat, panels):
        axis.imshow(image)
        axis.set_title(caption, fontsize=9, color=LABEL, pad=6)
    for axis in flat:
        axis.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(figure)
    return _shrink(out_path)


# ---------------------------------------------------------------------------
# the panels that need the models
# ---------------------------------------------------------------------------
def _load_pair(detector: str, enhancer: str, device):
    detector_model, _ = load_model(paths.run_dir(detector) / "best.pt", device=device)
    enhancer_model, _ = load_model(paths.run_dir(enhancer) / "best.pt", device=device)
    return detector_model, enhancer_model


def _scan(name: str, detector_model, enhancer_model, device) -> tuple[np.ndarray, dict]:
    photo = imread_rgb(paths.real_photos / f"{name}.jpg")
    result = scan_document(photo, detector_model, enhancer_model, device=device)
    return photo, result


def hero(out_dir: Path, name: str, detector_model, enhancer_model, device) -> None:
    """Photo in, scan out, in the four steps the chain actually takes."""
    photo, result = _scan(name, detector_model, enhancer_model, device)
    strip(
        [
            (photo, "the photograph"),
            (draw_corners(photo, result["corners"]), f"corners found  ({result['source']})"),
            (result["rectified"], "flattened"),
            (result["scan"], "restored"),
        ],
        out_dir / "hero.png",
    )


def corner_panel(out_dir: Path, name: str, detector_model, enhancer_model, device) -> None:
    """Where the heatmap detector looked, beside what it did with the answer."""
    photo, result = _scan(name, detector_model, enhancer_model, device)
    if result.get("heatmaps") is None:
        print("  (no heatmaps — the detector is not a heatmap model)")
        return
    strip(
        [
            (draw_corners(photo, result["corners"]), "predicted corners"),
            (heatmap_overlay(photo, result["heatmaps"]), "the four heatmaps, summed"),
            (result["rectified"], "flattened with those corners"),
        ],
        out_dir / "corner-detection.png",
    )


def gallery(out_dir: Path, names, detector_model, enhancer_model, device) -> None:
    """Several photographs and their scans — the before/after that is the point."""
    panels = []
    for name in names:
        photo, result = _scan(name, detector_model, enhancer_model, device)
        panels.append((photo, name))
        panels.append((result["scan"], "scanned"))
    grid(panels, out_dir / "gallery.png", columns=len(panels))


# ---------------------------------------------------------------------------
# the panels the report already drew
# ---------------------------------------------------------------------------
#: (source, name in assets/, longest side, fraction of the height to keep).
#: Report figures are 200 dpi and far larger than any browser needs; a README
#: that takes ten seconds to paint is a README nobody scrolls. The fraction
#: takes the top rows off a contact sheet that is taller than a screen — a
#: reader scrolling past twelve pages of the same comparison has stopped reading
#: it after the second.
COPIES = (
    ("previews/degradation_steps.jpg", "degradation.jpg", 1600, 1.0),
    ("previews/spot_the_fake.jpg", "spot-the-fake.jpg", 1600, 1.0),
    ("previews/composites.jpg", "synthetic-samples.jpg", 1600, 1.0),
    ("previews/rectified_pairs.jpg", "training-pairs.jpg", 1500, 0.34),
    ("figures/enhancement/realistic_test_pages.png", "enhancement-pages.jpg", 1400, 0.5),
    ("figures/enhancement/realistic_test_text_zoom.png", "enhancement-zoom.jpg", 1500, 0.5),
    ("figures/enhancement/enhance_realistic_curves.png", "training-curves.png", 1500, 1.0),
    ("figures/corners/pck_curves.png", "pck-curves.png", 1200, 1.0),
    ("figures/corners/failure_cases.png", "corner-failures.jpg", 1500, 1.0),
    ("figures/real/triplet_Image7.png", "vs-camscanner.jpg", 1600, 1.0),
)


def copies(out_dir: Path) -> None:
    """Downscale the report's own figures into the README's asset folder."""
    from PIL import Image

    roots = {"previews": paths.out / "previews", "figures": paths.figures}
    for source, name, longest, keep in COPIES:
        head, _, tail = source.partition("/")
        path = roots[head] / tail
        if not path.exists():
            print(f"  missing, skipped: {path}")
            continue

        image = Image.open(path).convert("RGB")
        if keep < 1.0:
            image = image.crop((0, 0, image.width, round(image.height * keep)))
        scale = min(1.0, longest / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )

        out_path = out_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".jpg":
            image.save(out_path, quality=86, optimize=True, progressive=True)
        else:
            image.save(out_path, optimize=True)
        print(f"  {out_path.relative_to(paths.repo)}  ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=None, help="default: assets/")
    parser.add_argument("--photo", default=HERO_PHOTO, help="the hero photograph")
    parser.add_argument("--gallery", nargs="*", default=list(GALLERY_PHOTOS))
    parser.add_argument("--detector", default=DETECTOR)
    parser.add_argument("--enhancer", default=ENHANCER)
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=["hero", "corners", "gallery", "copies"],
        help="default: all of them",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else paths.repo / "assets"
    wanted = set(args.only or ["hero", "corners", "gallery", "copies"])

    if wanted & {"hero", "corners", "gallery"}:
        device = get_device()
        print(f"models on {device}")
        detector_model, enhancer_model = _load_pair(args.detector, args.enhancer, device)
        if "hero" in wanted:
            hero(out_dir, args.photo, detector_model, enhancer_model, device)
        if "corners" in wanted:
            corner_panel(out_dir, args.photo, detector_model, enhancer_model, device)
        if "gallery" in wanted:
            gallery(out_dir, args.gallery, detector_model, enhancer_model, device)

    if "copies" in wanted:
        copies(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
