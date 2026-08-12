"""The degradation pipeline — OpenCV and NumPy only.  *(brief §4)*

The brief is explicit: no third-party augmentation libraries. Every transform here
is built from OpenCV primitives and course techniques.

Applied in the brief's §4.3 order, each stage randomising *every* parameter within
a range and returning the values it sampled, so the step-by-step figure and any
debugging come for free:

1. ``downscale_upscale``    — 2-4x, simulating distance and sensor limits
2. ``brightness_contrast``, ``color_cast`` — light source and time of day
3. ``illumination_gradient``, ``soft_shadows`` — the characteristic defect of real
   document photos; the shadow shapes include the elongated photographer-arm form
   that dominates this project's real test photos
4. ``random_blur``          — Gaussian, or motion blur from a rotated line kernel
5. ``gaussian_noise``
6. ``jpeg_recompress``      — quality 30-80 via ``imencode``/``imdecode``

Photometric degradations touch the **input only**, never the target. Deliberately
no flipping: mirrored text is not something a document scanner should learn to
"restore".

Every stage takes an **RGB uint8** image and a ``numpy`` generator, and returns the
image alongside the parameters it drew. Nothing reads the global random state, so
a sample is reproducible from its key alone.

*Why 8-bit and not float.* A phone stores 8 bits per channel, so every stage here
is modelling something that happens to 8-bit data; carrying float32 through the
chain would buy precision the final JPEG throws away anyway. It also lets the
point operations collapse into a 256-entry lookup table and the multiplicative
ones into ``cv2.multiply`` with saturation, which is the difference between 240 ms
and 130 ms per sample — and the generator runs once per training sample, forever.

The brief also warns in the other direction — degrade the page past recovery and
the model is being asked to hallucinate rather than restore. That is what
:data:`SEVERITY_SCALE` is for: ``mild``, ``medium`` and ``hard`` stretch the same
ranges around the same centres, so the severity study changes one number.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

# Blurring a mask at full resolution with sigma 60 costs a 361-tap separable pass
# over two million pixels. The mask is smooth by construction, so it is drawn and
# blurred small and then scaled up; this is the sigma that survives the shrink.
_WORKING_SIGMA = 4.0

SEVERITY_SCALE = {"mild": 0.55, "medium": 1.0, "hard": 1.45}


def _around(centre: float, span: tuple[float, float], scale: float) -> tuple[float, float]:
    """Stretch a range around a fixed centre — for gains that sit near 1.0."""
    return (centre + (span[0] - centre) * scale, centre + (span[1] - centre) * scale)


def _from_zero(span: tuple[float, float], scale: float) -> tuple[float, float]:
    """Stretch a range that measures a magnitude, where 0 means "no degradation"."""
    return (span[0] * scale, span[1] * scale)


@dataclass(frozen=True)
class DegradationConfig:
    """Every range the pipeline samples from.

    The defaults *are* the ``medium`` reference. ``severity`` widens or narrows
    them; a config file may also pin any individual range, which then becomes the
    reference that severity scales.
    """

    severity: str = "medium"

    # sensor and distance
    downscale: tuple[float, float] = (2.0, 4.0)

    # light source and time of day
    brightness: tuple[float, float] = (-0.12, 0.12)
    contrast: tuple[float, float] = (0.82, 1.20)
    color_gain: tuple[float, float] = (0.88, 1.12)

    # the shape of the light in the room
    illumination_amplitude: tuple[float, float] = (0.06, 0.30)
    vignette_prob: float = 0.5
    vignette_strength: tuple[float, float] = (0.05, 0.35)

    # things between the lamp and the page — including the photographer
    shadow_count: tuple[int, int] = (0, 3)
    shadow_strength: tuple[float, float] = (0.15, 0.60)
    shadow_blur: tuple[float, float] = (10.0, 60.0)
    hard_shadow_prob: float = 0.20
    arm_shadow_prob: float = 0.35

    # a lamp reflecting off the paper
    specular_prob: float = 0.25
    specular_strength: tuple[float, float] = (0.10, 0.45)

    # camera shake and sensor noise
    blur_sigma: tuple[float, float] = (0.4, 2.0)
    motion_blur_prob: float = 0.35
    motion_length: tuple[int, int] = (3, 15)
    noise_sigma: tuple[float, float] = (0.004, 0.030)

    # storage
    jpeg_quality: tuple[int, int] = (30, 80)

    # ------------------------------------------------------------------
    def scaled(self, severity: str) -> "DegradationConfig":
        """This config, with every range stretched to *severity*."""
        if severity not in SEVERITY_SCALE:
            raise ValueError(
                f"unknown severity {severity!r}; expected one of {sorted(SEVERITY_SCALE)}"
            )
        k = SEVERITY_SCALE[severity]
        if k == 1.0:
            return replace(self, severity=severity)
        return replace(
            self,
            severity=severity,
            downscale=_around(1.0, self.downscale, k),
            brightness=_from_zero(self.brightness, k),
            contrast=_around(1.0, self.contrast, k),
            color_gain=_around(1.0, self.color_gain, k),
            illumination_amplitude=_from_zero(self.illumination_amplitude, k),
            vignette_strength=_from_zero(self.vignette_strength, k),
            shadow_count=(self.shadow_count[0], int(round(self.shadow_count[1] * k))),
            shadow_strength=_from_zero(self.shadow_strength, k),
            specular_strength=_from_zero(self.specular_strength, k),
            blur_sigma=_from_zero(self.blur_sigma, k),
            motion_length=(self.motion_length[0], int(round(self.motion_length[1] * k))),
            noise_sigma=_from_zero(self.noise_sigma, k),
            # Quality counts down from perfect, so it scales from 100, not from 0 —
            # with a floor, because JPEG quality 0 is not a photo any more.
            jpeg_quality=tuple(
                int(np.clip(round(100 - (100 - q) * k), 5, 100)) for q in self.jpeg_quality
            ),
        )

    @classmethod
    def from_config(cls, config) -> "DegradationConfig":
        """Build from a YAML ``degradation:`` block (or any mapping).

        Unknown keys are an error rather than a shrug: a typo in a config file
        that silently keeps the default is exactly the kind of bug that makes an
        ablation table meaningless.
        """
        mapping = dict(config or {})
        severity = mapping.pop("severity", "medium")
        known = {f.name for f in fields(cls)} - {"severity"}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(
                f"unknown degradation setting(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        typed = {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in mapping.items()
        }
        return cls(**typed).scaled(severity)


DEFAULT = DegradationConfig()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def as_uint8(image: np.ndarray) -> np.ndarray:
    """The pipeline's working type: RGB uint8, rounded rather than truncated."""
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _uniform(rng, span) -> float:
    return float(rng.uniform(span[0], span[1]))


def _pixel(point) -> tuple[int, int]:
    """A drawing coordinate OpenCV will accept — plain ints, never numpy scalars."""
    return int(point[0]), int(point[1])


def apply_lut(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Apply a per-channel intensity curve given as 256 output values per channel.

    Every point operation in the pipeline — brightness, contrast, colour cast — is
    a function of one pixel value, so it is exactly a lookup table, and evaluating
    it 256 times instead of five million times is free accuracy as well as speed.
    """
    import cv2

    lut = np.clip(np.rint(table), 0, 255).astype(np.uint8).reshape(1, 256, -1)
    return cv2.LUT(image, lut)


def apply_gain(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiply a uint8 image by a single-channel float gain map, with saturation."""
    import cv2

    return cv2.multiply(image, cv2.merge([gain, gain, gain]), dtype=cv2.CV_8U)


def _smooth_field(rng, size, cells: int, low: float, high: float) -> np.ndarray:
    """A random low-frequency field: a tiny grid blown up with cubic interpolation.

    A handful of control points is all it takes — real room lighting varies over
    the scale of the whole page, not pixel to pixel.
    """
    import cv2

    grid = rng.uniform(low, high, size=(cells, cells)).astype(np.float32)
    field = cv2.resize(grid, (int(size[0]), int(size[1])), interpolation=cv2.INTER_CUBIC)
    # Cubic interpolation overshoots its control points; keep the field in range.
    return np.clip(field, min(low, high), max(low, high))


def blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a single-channel mask, doing the work at whatever scale *sigma* allows.

    A soft shadow is smooth by definition, so it is drawn small, blurred small and
    scaled back up. The result is indistinguishable from the full-resolution blur
    and an order of magnitude cheaper — which matters when this runs once per
    shadow, per sample, inside a dataloader worker.
    """
    import cv2

    if sigma <= 0.5:
        return mask
    height, width = mask.shape[:2]
    scale = float(np.clip(_WORKING_SIGMA / sigma, 0.12, 1.0))
    if scale < 1.0:
        small = cv2.resize(
            mask,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (0, 0), sigma * scale)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(mask, (0, 0), sigma)


# ---------------------------------------------------------------------------
# the stages, in the brief's §4.3 order
# ---------------------------------------------------------------------------
def downscale_upscale(image, rng, cfg: DegradationConfig = DEFAULT):
    """Shrink and re-enlarge: the page was photographed from across the room."""
    import cv2

    image = as_uint8(image)
    factor = _uniform(rng, cfg.downscale)
    height, width = image.shape[:2]
    small = (max(8, int(round(width / factor))), max(8, int(round(height / factor))))
    up = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))

    # INTER_AREA on the way down (it averages, rather than point-sampling, the
    # pixels being merged) and a smooth filter on the way back up: this is what
    # turns a crisp pen stroke into the soft one a distant phone photo records.
    shrunk = cv2.resize(image, small, interpolation=cv2.INTER_AREA)
    out = cv2.resize(shrunk, (width, height), interpolation=up)
    return out, {
        "factor": round(factor, 3),
        "upscale": "cubic" if up == cv2.INTER_CUBIC else "linear",
    }


def brightness_contrast(image, rng, cfg: DegradationConfig = DEFAULT):
    """A brighter or dimmer room, and a flatter or harsher one."""
    image = as_uint8(image)
    brightness = _uniform(rng, cfg.brightness)
    contrast = _uniform(rng, cfg.contrast)

    levels = np.arange(256, dtype=np.float32) / 255.0
    curve = ((levels - 0.5) * contrast + 0.5 + brightness) * 255.0
    table = np.repeat(curve[:, None], 3, axis=1)
    return apply_lut(image, table), {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
    }


def color_cast(image, rng, cfg: DegradationConfig = DEFAULT):
    """Warm tungsten or cool daylight, as independent gains on red and blue."""
    image = as_uint8(image)
    red = _uniform(rng, cfg.color_gain)
    blue = _uniform(rng, cfg.color_gain)

    levels = np.arange(256, dtype=np.float32)
    table = np.stack([levels * red, levels, levels * blue], axis=1)
    return apply_lut(image, table), {"red_gain": round(red, 4), "blue_gain": round(blue, 4)}


def illumination_field(size, rng, cfg: DegradationConfig = DEFAULT):
    """A smooth random light field over a ``(width, height)`` frame.

    Three shapes, because a room offers all three: a lamp off to one side (a
    linear ramp), a lamp overhead (radial falloff), and the general case of
    several light sources and reflections (a coarse random grid). This is the
    defect the enhancement network exists to undo, so it is worth more than one
    formula.
    """
    width, height = int(size[0]), int(size[1])
    amplitude = _uniform(rng, cfg.illumination_amplitude)
    mode = str(rng.choice(["grid", "grid", "linear", "radial"]))
    params = {"mode": mode, "amplitude": round(amplitude, 4)}

    # Row and column ramps, broadcast rather than materialised as two full grids:
    # at canvas resolution that is the difference between 8 MB and 30 MB per call.
    us = (np.arange(width, dtype=np.float32) / max(width - 1, 1))[None, :]
    vs = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, None]

    if mode == "grid":
        cells = int(rng.integers(3, 5))
        field = _smooth_field(rng, (width, height), cells, 1.0 - amplitude, 1.0 + amplitude)
        params["cells"] = cells
    elif mode == "linear":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        ramp = np.cos(angle) * us + np.sin(angle) * vs
        ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)
        field = 1.0 - amplitude + 2.0 * amplitude * ramp
        params["angle_deg"] = round(float(np.degrees(angle)), 1)
    else:
        cx, cy = float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85))
        radius = np.hypot((us - cx) * width, (vs - cy) * height)
        field = 1.0 + amplitude - 2.0 * amplitude * (radius / max(float(radius.max()), 1e-6))
        params["centre"] = [round(cx, 3), round(cy, 3)]

    if rng.random() < cfg.vignette_prob:
        strength = _uniform(rng, cfg.vignette_strength)
        nx, ny = (us - 0.5) * 2.0, (vs - 0.5) * 2.0
        field = field * (1.0 - strength * np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0))
        params["vignette"] = round(strength, 4)

    return np.ascontiguousarray(np.broadcast_to(field, (height, width)), dtype=np.float32), params


def illumination_gradient(image, rng, cfg: DegradationConfig = DEFAULT):
    """Multiply the image by :func:`illumination_field`."""
    image = as_uint8(image)
    field, params = illumination_field((image.shape[1], image.shape[0]), rng, cfg)
    return apply_gain(image, field), params


def shadow_field(size, rng, cfg: DegradationConfig = DEFAULT):
    """A multiplicative attenuation map holding every shadow cast on the frame.

    Three families, all seen in this project's own test photos: a blob cast by an
    object on the desk, a straight-edged shadow from a nearby wall or book, and
    the photographer's own arm reaching over the page — an elongated capsule with
    a wider blob at the end. A model trained on one shadow shape learns that
    shape, not shadows *(brief §4.4)*.

    Shadows combine with ``maximum`` rather than by adding, so that two of them
    overlapping darken the page like two shadows and not like a hole.
    """
    import cv2

    width, height = int(size[0]), int(size[1])
    count = int(rng.integers(cfg.shadow_count[0], cfg.shadow_count[1] + 1))
    if count == 0:
        return None, {"count": 0, "shapes": []}

    shapes = []
    accumulated = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        mask = np.zeros((height, width), dtype=np.float32)
        kind = "arm" if rng.random() < cfg.arm_shadow_prob else str(rng.choice(["blob", "edge"]))

        if kind == "blob":
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            vertices = int(rng.integers(3, 7))
            angles = np.sort(rng.uniform(0, 2 * np.pi, vertices))
            radii = rng.uniform(0.12, 0.45, vertices) * min(width, height)
            points = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
        elif kind == "edge":
            # A half-plane: the shadow of something large and straight, off-frame.
            angle = float(rng.uniform(0, 2 * np.pi))
            offset = float(rng.uniform(0.15, 0.85))
            direction = np.array([np.cos(angle), np.sin(angle)])
            through = np.array([width * offset, height * offset])
            normal = np.array([-direction[1], direction[0]])
            far = 2.0 * max(width, height)
            corners = np.stack(
                [
                    through + direction * far,
                    through - direction * far,
                    through - direction * far + normal * far,
                    through + direction * far + normal * far,
                ]
            )
            cv2.fillPoly(mask, [corners.astype(np.int32)], 1.0)
        else:
            # Forearm plus hand: two thick segments and a blob at the far end.
            start = np.array([rng.uniform(-0.2, 1.2) * width, rng.uniform(-0.2, 1.2) * height])
            angle = float(rng.uniform(0, 2 * np.pi))
            length = float(rng.uniform(0.45, 1.1)) * max(width, height)
            elbow = start + np.array([np.cos(angle), np.sin(angle)]) * length * 0.6
            bend = angle + float(rng.uniform(-0.6, 0.6))
            wrist = elbow + np.array([np.cos(bend), np.sin(bend)]) * length * 0.4
            thickness = max(3, int(rng.uniform(0.08, 0.20) * min(width, height)))
            cv2.line(mask, _pixel(start), _pixel(elbow), 1.0, thickness)
            cv2.line(mask, _pixel(elbow), _pixel(wrist), 1.0, max(2, int(thickness * 0.8)))
            cv2.circle(mask, _pixel(wrist), max(2, int(thickness * 0.75)), 1.0, -1)

        hard = rng.random() < cfg.hard_shadow_prob
        sigma = float(rng.uniform(1.0, 4.0)) if hard else _uniform(rng, cfg.shadow_blur)
        strength = _uniform(rng, cfg.shadow_strength)
        accumulated = np.maximum(accumulated, blur_mask(mask, sigma) * strength)
        shapes.append({"kind": kind, "sigma": round(sigma, 2), "strength": round(strength, 3)})

    return 1.0 - accumulated, {"count": count, "shapes": shapes}


def soft_shadows(image, rng, cfg: DegradationConfig = DEFAULT):
    """Composite blurred shadow shapes over the photo."""
    image = as_uint8(image)
    attenuation, params = shadow_field((image.shape[1], image.shape[0]), rng, cfg)
    if attenuation is None:
        return image, params
    return apply_gain(image, attenuation), params


def specular_highlight(image, rng, cfg: DegradationConfig = DEFAULT):
    """An occasional bright reflection, the way a lamp bounces off glossy paper."""
    import cv2

    image = as_uint8(image)
    if rng.random() >= cfg.specular_prob:
        return image, {"present": False}

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    centre = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
    axes = (int(rng.uniform(0.06, 0.28) * width), int(rng.uniform(0.06, 0.28) * height))
    cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = blur_mask(mask, float(max(axes)) * 0.35)
    strength = _uniform(rng, cfg.specular_strength)

    # Screen rather than add, so the highlight saturates towards white instead of
    # clipping into a flat disc with a visible edge.
    headroom = cv2.bitwise_not(image)  # 255 - image, for a uint8 image
    out = cv2.add(image, apply_gain(headroom, mask * strength))
    return out, {
        "present": True,
        "strength": round(strength, 3),
        "centre": [round(centre[0] / width, 3), round(centre[1] / height, 3)],
    }


def random_blur(image, rng, cfg: DegradationConfig = DEFAULT):
    """Out-of-focus or hand-shake blur — Gaussian, or a rotated line kernel."""
    import cv2

    image = as_uint8(image)
    if rng.random() < cfg.motion_blur_prob:
        low, high = cfg.motion_length[0], max(cfg.motion_length[0] + 1, cfg.motion_length[1])
        length = int(rng.integers(low, high + 1))
        angle = float(rng.uniform(0.0, 180.0))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        total = kernel.sum()
        if total <= 1e-6:  # a degenerate 1-px kernel rotated off its own support
            return image, {"kind": "none"}
        out = cv2.filter2D(image, -1, kernel / total, borderType=cv2.BORDER_REFLECT101)
        return out, {"kind": "motion", "length": length, "angle_deg": round(angle, 1)}

    sigma = _uniform(rng, cfg.blur_sigma)
    out = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    return out, {"kind": "gaussian", "sigma": round(sigma, 3)}


def gaussian_noise(image, rng, cfg: DegradationConfig = DEFAULT):
    """Sensor noise, independent per channel.

    Drawn with ``cv2.randn`` rather than NumPy: it is twice as fast over five
    million samples, and it is an OpenCV primitive, which is what the brief asks
    the pipeline to be built from. OpenCV's generator is global state, so it is
    re-seeded here from the sample's own generator — the noise field stays a
    function of the sample key and nothing else.
    """
    import cv2

    image = as_uint8(image)
    sigma = _uniform(rng, cfg.noise_sigma)
    noise = np.empty(image.shape, dtype=np.int16)
    cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))
    cv2.randn(noise, 0.0, sigma * 255.0)
    return cv2.add(image, noise, dtype=cv2.CV_8U), {"sigma": round(sigma, 5)}


def jpeg_recompress(image, rng, cfg: DegradationConfig = DEFAULT):
    """Re-encode at a random quality — phones do not store raw sensor data.

    The round trip goes through BGR properly rather than encoding RGB in place:
    JPEG derives luma from the channels by fixed weights, so feeding it swapped
    channels would put the chroma subsampling artefacts on the wrong ones.
    """
    import cv2

    image = as_uint8(image)
    quality = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:  # pragma: no cover - OpenCV only fails here on an unwritable buffer
        return image, {"quality": None}
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), {
        "quality": quality
    }


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
#: The stages in the brief's §4.3 order. Brightness/contrast and colour cast are
#: separate functions but one conceptual step, as are illumination and shadows.
STAGES = (
    ("downscale_upscale", downscale_upscale),
    ("brightness_contrast", brightness_contrast),
    ("color_cast", color_cast),
    ("illumination_gradient", illumination_gradient),
    ("soft_shadows", soft_shadows),
    ("specular_highlight", specular_highlight),
    ("random_blur", random_blur),
    ("gaussian_noise", gaussian_noise),
    ("jpeg_recompress", jpeg_recompress),
)


def degrade(
    image: np.ndarray,
    rng,
    cfg: DegradationConfig = DEFAULT,
    collect_steps: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run the whole pipeline over an RGB image.

    Returns the degraded image as uint8 RGB — byte-for-byte what a frozen sample
    written to disk would hold — plus a dict of every parameter that was sampled.

    ``collect_steps`` additionally returns the image after each stage under
    ``params["steps"]``: that is the step-by-step strip the report needs, and it
    costs nothing when it is off.
    """
    current = as_uint8(image)
    params: dict = {"severity": cfg.severity}
    steps: list[tuple[str, np.ndarray]] = [("input", current)] if collect_steps else []

    for name, stage in STAGES:
        current, stage_params = stage(current, rng, cfg)
        params[name] = stage_params
        if collect_steps:
            steps.append((name, current))

    if collect_steps:
        params["steps"] = steps
    return current, params
