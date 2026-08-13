"""Filesystem layout and image I/O.

Every path in the project is resolved through :data:`paths`, and every root can be
overridden by an environment variable. That is what lets the same notebook run
unchanged on the local RTX 3060 and on a Colab runtime where the data lives in
Drive::

    export SCANDAR_DATA=/content/drive/MyDrive/scandar/data
    export SCANDAR_OUT=/content/drive/MyDrive/scandar/outputs

Images are handled as **RGB uint8 HWC** arrays throughout the project. OpenCV's
native order is BGR, so the conversion happens exactly once, here, at the I/O
boundary — every other module can then assume RGB and stop thinking about it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _find_repo_root() -> Path:
    """Nearest ancestor holding ``pyproject.toml``, falling back to the layout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]  # src/scandar/io.py -> src/scandar -> src -> repo


REPO_ROOT = _find_repo_root()
DATA_ROOT = Path(os.environ.get("SCANDAR_DATA", REPO_ROOT / "data")).expanduser()
OUT_ROOT = Path(os.environ.get("SCANDAR_OUT", REPO_ROOT / "outputs")).expanduser()


@dataclass(frozen=True)
class Paths:
    """Every location the project reads or writes, in one place."""

    repo: Path
    data: Path
    out: Path

    # --- synthetic training material -------------------------------------
    @property
    def scans(self) -> Path:
        return self.data / "scans"

    @property
    def scans_cache(self) -> Path:
        return self.data / "cache" / "scans_1600"

    @property
    def backgrounds(self) -> Path:
        """All background photos live here; the train/held-out split is a manifest.

        Splitting by manifest rather than by directory keeps one copy of every
        file and makes the split re-derivable with a different seed.
        """
        return self.data / "backgrounds"

    # --- the real, evaluation-only bucket --------------------------------
    @property
    def real_photos(self) -> Path:
        return self.data / "real" / "photos"

    @property
    def real_reference(self) -> Path:
        return self.data / "real" / "reference"

    @property
    def real_annotations(self) -> Path:
        return self.data / "real" / "annotations"

    @property
    def real_transcripts(self) -> Path:
        return self.data / "real" / "transcripts"

    @property
    def real_corners(self) -> Path:
        return self.data / "real" / "corners.json"

    # --- frozen evaluation sets and manifests ----------------------------
    @property
    def frozen(self) -> Path:
        return self.data / "frozen"

    def frozen_set(self, task: str, split: str) -> Path:
        """One frozen evaluation bucket.

        Frozen sets are stored **per task**, not shared. The corner detector is
        trained against coloured page stock, a second sheet of paper in the frame
        and pages that will not lie flat; the enhancement network must never see
        any of those, because its target is the flat clean scan and no model can
        invert a tint it was never shown. One shared set of composited photos
        cannot serve both, so there are two.
        """
        return self.frozen / task / split

    @property
    def splits(self) -> Path:
        return self.data / "splits.json"

    # --- outputs ----------------------------------------------------------
    @property
    def runs(self) -> Path:
        return self.out / "runs"

    @property
    def figures(self) -> Path:
        return self.repo / "reports" / "figures"

    @property
    def tables(self) -> Path:
        return self.repo / "reports" / "tables"

    def run_dir(self, name: str) -> Path:
        return self.runs / name


paths = Paths(repo=REPO_ROOT, data=DATA_ROOT, out=OUT_ROOT)


# ---------------------------------------------------------------------------
# image I/O
# ---------------------------------------------------------------------------
def list_images(directory: Path | str) -> list[Path]:
    """Image files in *directory*, sorted naturally so ``2.jpg`` precedes ``10.jpg``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=natural_key)


def natural_key(path: Path | str):
    """Sort key that reads embedded digit runs as numbers (``Image2`` < ``Image10``)."""
    import re

    text = Path(path).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def imread_rgb(path: Path | str) -> np.ndarray:
    """Read an image as RGB uint8 HWC.

    Goes through ``np.fromfile`` + ``cv2.imdecode`` rather than ``cv2.imread`` so
    that non-ASCII paths work on every platform.
    """
    import cv2

    path = Path(path)
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise OSError(f"empty or unreadable file: {path}")
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not decode as an image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path | str, image: np.ndarray, quality: int = 95) -> Path:
    """Write an RGB uint8 HWC array, creating parent directories as needed."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = to_uint8(image)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, encoded = cv2.imencode(path.suffix, bgr, params)
    if not ok:
        raise OSError(f"could not encode image for: {path}")
    encoded.tofile(str(path))
    return path


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. Already-float input passes through."""
    if image.dtype == np.float32:
        return image
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8 [0, 255], rounding and clipping rather than truncating."""
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)


def read_json(path: Path | str):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload, indent: int = 2) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    return path
