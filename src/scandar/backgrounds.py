"""Backgrounds to composite documents onto.

Two sources, mixed:

* **real** background-only photos from ``data/backgrounds/`` — carpet, wood desk,
  marble, tile, cluttered table. Split into train and held-out groups so a surface
  the model trained on never reappears in validation or test.
* **procedural** textures generated with NumPy and OpenCV — wood grain, woven
  fabric, marble veining, tiling and fractal noise — for variety beyond what one
  apartment can supply.

Backgrounds may be flipped and rotated freely; documents may not.

The procedural half is not a stand-in for the real photos, it is insurance. Twenty
surfaces is a small vocabulary, and the fifteen that end up in the training split
are all lit by the same two lamps; a detector that has only ever seen those can
learn *this desk* instead of *a page on a desk*. The procedural textures cost
nothing to generate, never repeat, and keep the held-out real surfaces genuinely
held out.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .io import imread_rgb, list_images, paths

#: Backgrounds are cached in memory at this long side. The composite is built at
#: canvas resolution and then degraded by a 2-4x downscale, so detail beyond this
#: cannot survive to the model; the full 1920x2560 originals would just cost every
#: dataloader worker a few hundred megabytes.
CACHE_LONG_SIDE = 1600


class BackgroundBank:
    """The real background photos belonging to one side of the split.

    Decoding a phone JPEG takes longer than everything else in a sample put
    together, so images are decoded once per worker process and kept. Twenty
    photos at 1200x1600 is about 115 MB — worth it, and bounded by *cache_size*
    if the collection ever grows.
    """

    def __init__(
        self,
        names: list[str] | None = None,
        directory: Path | str | None = None,
        long_side: int = CACHE_LONG_SIDE,
        cache_size: int = 32,
    ) -> None:
        self.directory = Path(directory) if directory is not None else paths.backgrounds
        if names is None:
            names = [p.name for p in list_images(self.directory)]
        self.names = list(names)
        self.long_side = long_side
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BackgroundBank({len(self.names)} photos from {self.directory})"

    def load(self, name: str) -> np.ndarray:
        """The named photo, downscaled to the cache resolution."""
        import cv2

        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        image = imread_rgb(self.directory / name)
        height, width = image.shape[:2]
        scale = self.long_side / max(height, width)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        self._cache[name] = image
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image

    def warm(self) -> int:
        """Decode every photo now. Returns how many are held.

        Same reasoning as :meth:`~scandar.prepare.ScanBank.warm`: pay the JPEG
        decodes once, in the parent, and let forked dataloader workers inherit
        them rather than each rebuilding the whole bank.
        """
        for name in self.names[: self.cache_size]:
            self.load(name)
        return len(self._cache)

    def sample(self, rng, size) -> tuple[np.ndarray, dict]:
        """A random crop of a random photo, at ``size = (width, height)``."""
        if not self.names:
            raise ValueError(f"no background photos in {self.directory}")
        name = str(rng.choice(self.names))
        image, params = random_view(self.load(name), rng, size)
        params["source"] = name
        return image, params


def random_view(image: np.ndarray, rng, size) -> tuple[np.ndarray, dict]:
    """Crop, rotate and flip *image* into a ``(width, height)`` background.

    The crop keeps the target aspect ratio so the texture is never stretched, and
    the scale varies so the same carpet appears both as a coarse weave and as a
    distant pattern.
    """
    import cv2

    out_width, out_height = int(size[0]), int(size[1])

    turns = int(rng.integers(0, 4))
    view = np.rot90(image, turns)
    flip_x, flip_y = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    if flip_x:
        view = view[:, ::-1]
    if flip_y:
        view = view[::-1]

    height, width = view.shape[:2]
    aspect = out_width / out_height
    scale = float(rng.uniform(0.45, 1.0))
    crop_height = min(height, width / aspect) * scale
    crop_width = crop_height * aspect
    top = float(rng.uniform(0, max(0.0, height - crop_height)))
    left = float(rng.uniform(0, max(0.0, width - crop_width)))

    crop = view[
        int(top) : int(top) + max(1, int(crop_height)),
        int(left) : int(left) + max(1, int(crop_width)),
    ]
    shrinking = crop.shape[1] > out_width
    resized = cv2.resize(
        np.ascontiguousarray(crop),
        (out_width, out_height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    return resized, {
        "kind": "photo",
        "turns": turns,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "crop_scale": round(scale, 3),
    }


# ---------------------------------------------------------------------------
# procedural textures
# ---------------------------------------------------------------------------
def fbm(rng, size, octaves: int = 5, persistence: float = 0.55) -> np.ndarray:
    """Fractal noise in ``[0, 1]``: random grids at doubling frequency, summed.

    Built from ``cv2.resize`` with cubic interpolation rather than a gradient
    noise implementation — same smooth multi-scale look, ten lines, and it uses
    only what the course allows.
    """
    import cv2

    width, height = int(size[0]), int(size[1])
    field = np.zeros((height, width), dtype=np.float32)
    amplitude, total, cells = 1.0, 0.0, 2
    for _ in range(octaves):
        rows = max(2, cells)
        # Keep the cells roughly square so the noise is not stretched.
        columns = max(2, int(round(cells * width / height)))
        grid = rng.random((rows, columns)).astype(np.float32)
        field += amplitude * cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        total += amplitude
        amplitude *= persistence
        cells *= 2
    field /= max(total, 1e-6)
    low, high = float(field.min()), float(field.max())
    return (field - low) / max(high - low, 1e-6)


def _ramp(values: np.ndarray, colors) -> np.ndarray:
    """Map a ``[0, 1]`` field through a list of RGB stops.

    The ramp is evaluated at 256 levels and then looked up, rather than
    interpolated per pixel: over a two-megapixel field that is the difference
    between 7 ms and 100 ms, and the input is quantised to 8 bits at the end of
    this function's life anyway.
    """
    colors = np.asarray(colors, dtype=np.float32)
    stops = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    levels = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    table = np.stack([np.interp(levels, stops, colors[:, c]) for c in range(3)], axis=1)

    indices = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.take(table.astype(np.float32), indices, axis=0)


def _coordinates(size) -> tuple[np.ndarray, np.ndarray]:
    """Normalised column and row ramps, shaped to broadcast against ``(H, W)``."""
    width, height = int(size[0]), int(size[1])
    us = (np.arange(width, dtype=np.float32) / max(width - 1, 1))[None, :]
    vs = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, None]
    return us, vs


def wood_grain(rng, size) -> np.ndarray:
    """Growth rings distorted by noise, plus fine longitudinal streaks."""
    us, vs = _coordinates(size)
    along = us if rng.random() < 0.5 else vs
    # Gentle turbulence and a high ring frequency: growth rings drift slowly and
    # sit close together. Turn the turbulence up and the grain stops reading as
    # wood and starts reading as sand dunes.
    turbulence = fbm(rng, size, octaves=4)
    frequency = float(rng.uniform(9.0, 24.0))
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * (along + 0.09 * turbulence))
    streaks = fbm(rng, size, octaves=6, persistence=0.7)
    value = np.clip(0.66 * rings + 0.34 * streaks, 0.0, 1.0)
    tone = float(rng.uniform(0.75, 1.10))
    return _ramp(value, [[0.30, 0.21, 0.13], [0.53, 0.39, 0.25], [0.72, 0.58, 0.42]]) * tone


def marble(rng, size) -> np.ndarray:
    """Veining: a sine of the coordinate, heavily perturbed by turbulence."""
    us, vs = _coordinates(size)
    turbulence = fbm(rng, size, octaves=6)
    angle = float(rng.uniform(0, np.pi))
    coordinate = np.cos(angle) * us + np.sin(angle) * vs
    veins = 0.5 + 0.5 * np.sin(
        2 * np.pi * float(rng.uniform(1.5, 4.0)) * (coordinate + 0.40 * turbulence)
    )
    # A high power turns the sine's broad hump into the thin bright vein that
    # makes stone look like stone rather than like a contour map.
    value = np.clip(veins ** float(rng.uniform(3.0, 7.0)), 0.0, 1.0)
    return _ramp(value, [[0.55, 0.54, 0.52], [0.86, 0.85, 0.83], [0.96, 0.96, 0.95]])


def woven_fabric(rng, size) -> np.ndarray:
    """A carpet or tablecloth: colour blotches under a regular weave."""
    import cv2

    us, vs = _coordinates(size)
    width, height = int(size[0]), int(size[1])
    period = float(rng.uniform(3.0, 9.0))
    weave = 0.5 + 0.5 * np.sin(2 * np.pi * us * width / period) * np.sin(
        2 * np.pi * vs * height / period
    )
    blotches = fbm(rng, size, octaves=4)
    value = np.clip(0.65 * blotches + 0.35 * weave, 0.0, 1.0)

    palette = [
        [[0.45, 0.12, 0.10], [0.72, 0.35, 0.20], [0.90, 0.80, 0.62]],  # persian red
        [[0.16, 0.20, 0.28], [0.35, 0.42, 0.50], [0.66, 0.70, 0.74]],  # slate blue
        [[0.30, 0.26, 0.18], [0.58, 0.50, 0.36], [0.84, 0.78, 0.64]],  # sand
        [[0.14, 0.24, 0.18], [0.30, 0.46, 0.34], [0.62, 0.72, 0.60]],  # moss
    ]
    colors = palette[int(rng.integers(0, len(palette)))]
    texture = _ramp(value, colors)
    # A little directional smearing sells the pile of a carpet.
    return cv2.GaussianBlur(texture, (0, 0), 0.6, 1.6)


def tiled_floor(rng, size) -> np.ndarray:
    """A stone tile grid: per-tile shading, grout lines between."""
    import cv2

    width, height = int(size[0]), int(size[1])
    period = int(rng.uniform(0.18, 0.45) * min(width, height))
    grout = max(2, int(period * float(rng.uniform(0.02, 0.06))))

    stone = fbm(rng, size, octaves=5)
    columns = max(1, width // period + 1)
    rows = max(1, height // period + 1)
    per_tile = rng.uniform(0.82, 1.12, size=(rows, columns)).astype(np.float32)
    shading = cv2.resize(per_tile, (width, height), interpolation=cv2.INTER_NEAREST)

    value = np.clip(stone * shading, 0.0, 1.0)
    texture = _ramp(value, [[0.52, 0.48, 0.42], [0.78, 0.74, 0.66], [0.92, 0.90, 0.86]])

    xs = np.arange(width)[None, :] % period
    ys = np.arange(height)[:, None] % period
    lines = ((xs < grout) | (ys < grout)).astype(np.float32)[..., None]
    grout_color = np.array(rng.uniform(0.55, 0.75, size=3), dtype=np.float32)
    return texture * (1.0 - lines) + grout_color * lines


def painted_surface(rng, size) -> np.ndarray:
    """A plain matte wall or desktop — the hardest background, because it is featureless.

    Built as a grey plus a small colour offset rather than three independent
    channel values: a desk is off-white, beige, grey or occasionally a strong
    colour like the red table in the real photos, and almost never the saturated
    magenta that three independent uniform draws produce most of the time.
    """
    value = fbm(rng, size, octaves=3)
    grey = float(rng.uniform(0.28, 0.86))
    chroma = 0.22 if rng.random() < 0.12 else 0.05  # the occasional coloured table
    base = np.clip(grey + rng.uniform(-chroma, chroma, size=3), 0.05, 0.98).astype(np.float32)
    contrast = float(rng.uniform(0.02, 0.12))
    return np.clip(
        base[None, None, :] * (1.0 - contrast + 2 * contrast * value[..., None]), 0.0, 1.0
    )


TEXTURES = {
    "wood": wood_grain,
    "marble": marble,
    "woven": woven_fabric,
    "tiled": tiled_floor,
    "painted": painted_surface,
}


#: Procedural textures are generated at half the canvas size and enlarged once.
#: Every surface here is out of focus behind an in-focus page, and the whole
#: composite then goes through a 2-4x downscale in the degradation pipeline, so
#: detail at full resolution cannot reach the model — while generating it costs
#: four times the work. It also matches the real photos, whose crops are often
#: enlarged from less than the canvas size.
TEXTURE_SCALE = 2


def procedural_background(rng, size, kind: str | None = None) -> tuple[np.ndarray, dict]:
    """Generate a synthetic surface at ``size = (width, height)``."""
    import cv2

    if kind is None:
        kind = str(rng.choice(list(TEXTURES)))
    width, height = int(size[0]), int(size[1])
    small = (max(64, width // TEXTURE_SCALE), max(64, height // TEXTURE_SCALE))
    texture = TEXTURES[kind](rng, small)

    # One tint and one exposure at the end: the same generator then supplies a
    # warm oak and a bleached one without five more parameters inside every
    # texture function. Both are deliberately narrow, and a partial pull towards
    # grey follows, because the thing that gives a generated surface away is
    # never its pattern — it is being more colourful than any real desk.
    tint = rng.uniform(0.94, 1.06, size=3).astype(np.float32)
    exposure = float(rng.uniform(0.68, 1.02))
    image = np.clip(texture * tint * exposure, 0.0, 1.0)

    desaturation = float(rng.uniform(0.0, 0.45))
    luma = (image * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
    image = image * (1.0 - desaturation) + luma * desaturation
    image = (image * 255.0 + 0.5).astype(np.uint8)
    if small != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image, {"kind": "procedural", "texture": kind}


def sample_background(
    rng,
    size,
    bank: BackgroundBank | None = None,
    procedural_prob: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """A background of ``size = (width, height)``, real or procedural.

    Falls back to procedural textures when no real photos are available, so the
    generator — and its tests — run on a fresh clone before any photo is shot.
    """
    if bank is None or len(bank) == 0 or rng.random() < procedural_prob:
        return procedural_background(rng, size)
    return bank.sample(rng, size)
