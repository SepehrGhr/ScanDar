"""Synthetic sample generation.  *(brief §1.3 and §4)*

The key insight of the project: the training set never needs annotating. Choose
four random target points in a background image, warp a clean scan onto them, and
those four points *are* the corner labels — pixel-perfect, at zero annotation cost.
Because the homography is known, warping the degraded photo back also yields a
perfectly aligned (degraded input, clean target) pair. The label generator and the
data generator are the same function.

:func:`compose_sample` returns a :class:`Sample` holding what the corner detector
sees — ``photo``, the degraded composite, and ``corners``, ordered TL, TR, BR, BL —
together with the scan and the homography it was built from. What the enhancement
network sees is *derived* from those, on demand:

``sample.rectify(size)``      the whole page, flattened: (degraded input, clean target)
``sample.random_patch(...)``  one 256x256 crop of the same pair, warped directly

Deriving rather than storing matters twice over. A whole rectified page costs 4 MB
that patch training would throw away, and — more importantly — input and target
come out of the *same* homography chain, so they are aligned by construction
rather than by agreement between two pieces of code. The sanity checks measure the
residual shift with phase correlation and hold it under half a pixel; the brief
warns twice that a few pixels of drift punish the model for errors it did not make.

Two things the composite does that a naive paste does not:

* the page mask is **feathered**, and the page casts a soft **drop shadow** on the
  surface. Without them the page meets the background at an impossibly clean seam
  — a one-pixel step no real camera produces, and the easiest possible shortcut
  for a corner detector to learn instead of learning what a page looks like.
* the scan is **downscaled with INTER_AREA to roughly its size in the canvas**
  before being warped. ``warpPerspective`` only point-samples; shrinking a
  1600 px scan into a 700 px page with it would alias the text into moiré that no
  phone camera would ever record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import degrade as degrade_module
from .backgrounds import BackgroundBank, sample_background
from .degrade import DegradationConfig
from .geometry import (
    homography,
    order_corners,
    quad_problem,
    rect_corners,
    scale_points,
    translation,
)
from .prepare import ScanBank, load_splits

PAPER_WHITE = (0.92, 0.92, 0.90)

#: Placement extras that make the page stop looking like the clean scan. They are
#: legitimate for the corner detector, whose label is four points, and poison for
#: the enhancement network, whose target *is* the clean scan: a tinted or bulged
#: input paired with a flat white target teaches the model to invent a colour
#: correction it was never shown how to get right. Nothing may switch them on for
#: an enhancement sample, which is why they are stripped structurally rather than
#: left to a default that a config file can quietly overwrite.
CORNER_ONLY_OPTIONS = ("page_tint_prob", "page_dark_prob", "distractor_prob", "curl_prob")


@dataclass(frozen=True)
class SynthOptions:
    """How the page is placed on the background, and what else is in the frame.

    The last three fields are **corner-detector only**. They deliberately break
    the assumption the enhancement network is trained on — that the clean scan is
    exactly what the page should look like — so they are off by default and
    :meth:`for_enhancement` keeps them off.
    """

    canvas: tuple[int, int] = (1152, 1536)
    landscape_prob: float = 0.10

    page_scale: tuple[float, float] = (0.42, 0.90)
    rotation_deg: float = 30.0
    keystone: tuple[float, float] = (0.78, 1.00)
    corner_jitter: float = 0.06
    center_jitter: float = 0.06
    max_tries: int = 40

    feather_sigma: tuple[float, float] = (0.6, 1.6)
    drop_shadow_prob: float = 0.85
    drop_shadow_offset: tuple[float, float] = (4.0, 26.0)
    drop_shadow_sigma: tuple[float, float] = (6.0, 30.0)
    drop_shadow_strength: tuple[float, float] = (0.10, 0.45)

    procedural_prob: float = 0.25

    # --- corner-detector only ---------------------------------------------
    page_tint_prob: float = 0.0
    page_dark_prob: float = 0.0
    distractor_prob: float = 0.0
    curl_prob: float = 0.0

    @classmethod
    def for_corners(cls, **overrides) -> "SynthOptions":
        """Placement with the robustness extras on.

        The real test photos include a blue notebook cover, a dark printed card,
        a page lying on top of other paper and a spread that will not lie flat.
        None of those are modelled by a flat white rectangle, and the detector is
        the model that has to survive them.
        """
        defaults = {
            "page_tint_prob": 0.25,
            "page_dark_prob": 0.10,
            "distractor_prob": 0.20,
            "curl_prob": 0.20,
        }
        return cls(**{**defaults, **overrides})

    @classmethod
    def for_enhancement(cls, **overrides) -> "SynthOptions":
        """Placement with the extras off, because the target must stay the scan."""
        return replace(cls(**overrides), **{name: 0.0 for name in CORNER_ONLY_OPTIONS})

    @classmethod
    def from_config(cls, config, task: str = "corner") -> "SynthOptions":
        """Build from a loaded config's ``synth:`` block and ``data.canvas``.

        One ``synth:`` block serves both tasks, so the corner-only extras are
        stripped again *after* the config is applied. Leaving them to be merely
        defaulted-off would mean a single shared config file silently trained the
        enhancement network on tinted and bulged pages.
        """
        settings = dict(config.get("synth") or {})
        canvas = config.get("data", {}).get("canvas")
        if canvas is not None:
            settings["canvas"] = tuple(canvas)
        unknown = set(settings) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown synth setting(s): {', '.join(sorted(unknown))}")
        typed = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in settings.items()
        }
        if task == "corner":
            return replace(cls.for_corners(), **typed)
        return cls.for_enhancement(**typed)


@dataclass
class Sample:
    """One generated photo and everything needed to derive its training pairs."""

    photo: np.ndarray  # uint8 RGB HWC — the degraded composite
    corners: np.ndarray  # (4, 2) float32 canvas pixel indices, TL TR BR BL
    scan: np.ndarray  # uint8 RGB HWC — the clean page this was built from
    H: np.ndarray  # 3x3, scan pixel indices -> canvas pixel indices
    params: dict
    clean_photo: np.ndarray | None = None  # the composite before degradation
    steps: list[tuple[str, np.ndarray]] | None = None  # per-stage, for the figure

    @property
    def canvas_size(self) -> tuple[int, int]:
        return int(self.photo.shape[1]), int(self.photo.shape[0])

    def rectification(self, size) -> np.ndarray:
        """The homography flattening the page onto a ``(width, height)`` rectangle."""
        return homography(self.corners, rect_corners(int(size[0]), int(size[1])))

    def rectify(self, size, source: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(degraded input, clean target)`` for the whole page at ``(width, height)``.

        *source* overrides which composite is flattened; passing ``clean_photo``
        is how the sanity check measures alignment without the degradation noise
        getting in the way of the measurement.
        """
        import cv2

        size = (int(size[0]), int(size[1]))
        to_rect = self.rectification(size)
        photo = self.photo if source is None else source
        degraded = cv2.warpPerspective(photo, to_rect, size, flags=cv2.INTER_LINEAR)
        target = cv2.warpPerspective(self.scan, to_rect @ self.H, size, flags=cv2.INTER_CUBIC)
        return degraded, target

    def rectify_patch(self, box, rect_size) -> tuple[np.ndarray, np.ndarray]:
        """One patch of :meth:`rectify`, warped straight out of the canvas.

        ``box`` is ``(x, y, size)`` in the coordinates of the rectified page.
        Composing the crop into the homography instead of flattening the whole
        page and slicing it is roughly twenty times less work per sample, and it
        is exact: it is the same matrix with a translation on the end.
        """
        import cv2

        x, y, patch = int(box[0]), int(box[1]), int(box[2])
        to_patch = translation(-x, -y) @ self.rectification(rect_size)
        degraded = cv2.warpPerspective(self.photo, to_patch, (patch, patch), flags=cv2.INTER_LINEAR)
        target = cv2.warpPerspective(
            self.scan, to_patch @ self.H, (patch, patch), flags=cv2.INTER_CUBIC
        )
        return degraded, target

    def random_patch(
        self,
        rng,
        patch_size: int,
        rect_size,
        tries: int = 4,
        min_std: float = 0.045,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        """A random patch, preferring one with something written on it.

        A uniformly random crop of an A4 page is very often blank margin. Blank
        patches are not useless — flattening the paper to an even white *is* half
        the job — but a training set made mostly of them spends its capacity on
        the easy half. Up to *tries* boxes are drawn and the first with enough
        contrast wins, which biases towards text without ever excluding blanks.
        """
        import cv2

        width, height = int(rect_size[0]), int(rect_size[1])
        patch = min(int(patch_size), width, height)
        to_rect = self.rectification((width, height))
        scan_to_rect = to_rect @ self.H

        box = (0, 0, patch)
        target = None
        for attempt in range(max(1, tries)):
            x = int(rng.integers(0, max(1, width - patch + 1)))
            y = int(rng.integers(0, max(1, height - patch + 1)))
            candidate = cv2.warpPerspective(
                self.scan, translation(-x, -y) @ scan_to_rect, (patch, patch), flags=cv2.INTER_CUBIC
            )
            box, target = (x, y, patch), candidate
            if candidate.std() / 255.0 >= min_std or attempt == tries - 1:
                break

        degraded = cv2.warpPerspective(
            self.photo,
            translation(-box[0], -box[1]) @ to_rect,
            (patch, patch),
            flags=cv2.INTER_LINEAR,
        )
        return degraded, target, box


# ---------------------------------------------------------------------------
# placing the page
# ---------------------------------------------------------------------------
def sample_quad(rng, canvas, page_aspect: float, options: SynthOptions) -> tuple[np.ndarray, dict]:
    """Pick where the page lands, by rejection sampling *(brief §4.1)*.

    An axis-aligned rectangle at a random scale, squeezed along one edge into a
    trapezoid (a page photographed from an angle rather than from overhead),
    rotated, then nudged corner by corner. Anything that comes out concave, too
    thin, too small or off the edge of the canvas is thrown away and redrawn —
    which is cheaper to write and to explain than trying to sample only from the
    valid set.
    """
    width, height = float(canvas[0]), float(canvas[1])
    reason = None

    for _ in range(max(1, options.max_tries)):
        scale = float(rng.uniform(*options.page_scale))
        page_height = scale * height
        page_width = page_height * page_aspect
        half_w, half_h = page_width / 2.0, page_height / 2.0

        keystone = float(rng.uniform(*options.keystone))
        near_edge = "top" if rng.random() < 0.5 else "bottom"
        top_scale = keystone if near_edge == "top" else 1.0
        bottom_scale = 1.0 if near_edge == "top" else keystone
        quad = np.array(
            [
                [-half_w * top_scale, -half_h],
                [+half_w * top_scale, -half_h],
                [+half_w * bottom_scale, +half_h],
                [-half_w * bottom_scale, +half_h],
            ],
            dtype=np.float64,
        )

        angle = np.radians(float(rng.uniform(-options.rotation_deg, options.rotation_deg)))
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
        )
        quad = quad @ rotation.T

        jitter = options.corner_jitter * min(page_width, page_height)
        quad += rng.uniform(-jitter, jitter, size=(4, 2))

        centre = np.array([width / 2.0, height / 2.0])
        centre += rng.uniform(-1, 1, size=2) * options.center_jitter * np.array([width, height])
        quad += centre

        reason = quad_problem(quad, (width, height), margin=2.0)
        if reason is None:
            return order_corners(quad), {
                "scale": round(scale, 4),
                "rotation_deg": round(float(np.degrees(angle)), 2),
                "keystone": round(keystone, 4),
                "keystone_edge": near_edge,
            }

    # Every draw was rejected — an aspect ratio the canvas cannot hold at any of
    # the sampled scales, say. Fall back to a centred, unrotated page, so the
    # generator degrades into something usable instead of raising mid-epoch.
    scale = float(np.mean(options.page_scale)) * 0.8
    page_height = scale * height
    page_width = page_height * page_aspect
    centre = np.array([width / 2.0, height / 2.0])
    quad = np.array(
        [
            [-page_width / 2, -page_height / 2],
            [+page_width / 2, -page_height / 2],
            [+page_width / 2, +page_height / 2],
            [-page_width / 2, +page_height / 2],
        ]
    ) + centre
    return order_corners(quad), {
        "scale": round(scale, 4),
        "rotation_deg": 0.0,
        "keystone": 1.0,
        "keystone_edge": "none",
        "fallback": reason,
    }


def curl_page(page: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Bend the page with a sinusoidal displacement that vanishes at its border.

    Real pages photographed on a desk are rarely flat, and a page that will not
    lie flat is one of the failure cases in this project's own test photos.
    Because the displacement is zero along the whole boundary, the four corners
    do not move — so the labels stay exact and only the interior bulges.
    """
    import cv2

    height, width = page.shape[:2]
    amplitude = float(rng.uniform(0.010, 0.035)) * height
    waves_x = int(rng.integers(1, 3))
    waves_y = int(rng.integers(1, 3))

    us = (np.arange(width, dtype=np.float32) / max(width - 1, 1))[None, :]
    vs = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, None]
    envelope = np.sin(np.pi * us) * np.sin(np.pi * vs)

    map_x = np.ascontiguousarray(
        us * (width - 1) + 0.35 * amplitude * np.sin(np.pi * waves_y * vs) * envelope,
        dtype=np.float32,
    )
    map_y = np.ascontiguousarray(
        vs * (height - 1) + amplitude * np.sin(np.pi * waves_x * us) * envelope, dtype=np.float32
    )
    warped = cv2.remap(
        page, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    return warped, {"amplitude_px": round(amplitude, 2), "waves": [waves_x, waves_y]}


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------
def compose_sample(
    scan: np.ndarray,
    rng,
    options: SynthOptions | None = None,
    backgrounds: BackgroundBank | None = None,
    degradation: DegradationConfig = degrade_module.DEFAULT,
    keep_clean: bool = False,
    collect_steps: bool = False,
) -> Sample:
    """Composite one clean *scan* onto a random background and degrade it.

    Everything random comes from *rng*, so the same generator and the same key
    reproduce the same sample on any machine — which is what makes the frozen
    evaluation sets frozen and an interrupted run resumable.
    """
    import cv2

    options = options or SynthOptions()
    width, height = int(options.canvas[0]), int(options.canvas[1])
    if rng.random() < options.landscape_prob:
        width, height = height, width
    canvas_size = (width, height)

    background, background_params = sample_background(
        rng, canvas_size, backgrounds, options.procedural_prob
    )

    scan_height, scan_width = scan.shape[:2]
    corners, placement = sample_quad(rng, canvas_size, scan_width / scan_height, options)
    H = homography(rect_corners(scan_width, scan_height), corners)

    # --- the page itself ---------------------------------------------------
    page, page_corners, page_params = _prepare_page(scan, corners, rng, options)
    to_canvas = homography(page_corners, corners)
    warped = cv2.warpPerspective(page, to_canvas, canvas_size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full(page.shape[:2], 255, dtype=np.uint8), to_canvas, canvas_size, flags=cv2.INTER_LINEAR
    )

    composite = background

    # --- a second sheet, peeking out from under the page -------------------
    if rng.random() < options.distractor_prob:
        composite, distractor_params = _add_distractor(composite, corners, rng, options)
        page_params["distractor"] = distractor_params
    else:
        page_params["distractor"] = None

    # --- the page's own shadow on the surface ------------------------------
    shadow_params = None
    if rng.random() < options.drop_shadow_prob:
        composite, shadow_params = _cast_drop_shadow(composite, mask, rng, options)

    # --- alpha composite, with a feathered edge ----------------------------
    # `warped` comes out of warpPerspective already multiplied by its own
    # coverage (outside the page it faded to the zero border), so colour and
    # coverage are blurred by the *same* kernel and the premultiplied form is
    # added directly. Blurring only the mask and then multiplying a second time
    # would darken every edge pixel — a one-pixel black outline around the page,
    # which is both wrong and a gift to a corner detector looking for a shortcut.
    feather = float(rng.uniform(*options.feather_sigma))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
    premultiplied = cv2.GaussianBlur(warped, (0, 0), feather)
    clean = cv2.add(degrade_module.apply_gain(composite, 1.0 - alpha), premultiplied)

    photo, degradation_params = degrade_module.degrade(
        clean, rng, degradation, collect_steps=collect_steps
    )
    steps = degradation_params.pop("steps", None)

    params = {
        "canvas": [width, height],
        "placement": placement,
        "corners": np.asarray(corners, dtype=float).round(2).tolist(),
        "background": background_params,
        "page": page_params,
        "feather_sigma": round(feather, 3),
        "drop_shadow": shadow_params,
        "degradation": degradation_params,
    }
    return Sample(
        photo=photo,
        corners=np.asarray(corners, dtype=np.float32),
        scan=scan,
        H=H,
        params=params,
        clean_photo=clean if keep_clean else None,
        steps=steps,
    )


@dataclass
class Sources:
    """Everything :func:`compose_sample` draws on, for one side of the split."""

    scans: ScanBank
    backgrounds: BackgroundBank
    options: SynthOptions
    degradation: DegradationConfig

    def warm(self) -> "Sources":
        """Decode every scan and background now, and return self.

        Call this in the parent process before handing the dataset to a
        DataLoader. Forked workers then inherit the decoded images copy-on-write;
        without it each worker decodes the whole split itself, which the profiler
        shows costing more than the compositing does.
        """
        self.scans.warm()
        self.backgrounds.warm()
        return self

    def compose(self, rng, **kwargs) -> Sample:
        """One sample from a random scan, with the scan id recorded in its params."""
        scan_id = str(rng.choice(self.scans.ids))
        sample = compose_sample(
            self.scans.load(scan_id),
            rng,
            options=self.options,
            backgrounds=self.backgrounds,
            degradation=self.degradation,
            **kwargs,
        )
        sample.params["scan"] = scan_id
        return sample


def build_sources(config, split: str, task: str = "corner", splits: dict | None = None) -> Sources:
    """Assemble the generator inputs for *split* from a loaded config.

    Backgrounds follow the scans: the training split draws on the training
    surfaces, and both evaluation splits draw on the held-out ones, so a carpet
    the model trained on never turns up in a validation score.
    """
    splits = splits if splits is not None else load_splits()
    if split not in splits["scans"]:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(splits['scans'])}")

    background_names = splits["backgrounds"]["train" if split == "train" else "heldout"]
    return Sources(
        scans=ScanBank(splits["scans"][split]),
        backgrounds=BackgroundBank(background_names),
        options=SynthOptions.from_config(config, task=task),
        degradation=DegradationConfig.from_config(config.get("degradation")),
    )


def _prepare_page(scan, corners, rng, options: SynthOptions) -> tuple[np.ndarray, np.ndarray, dict]:
    """The scan as it will be drawn: resampled to its size on the canvas, and bent.

    Returns the page image, **the four points of it that correspond to the scan's
    own corners**, and the parameters drawn. Those points are not always the
    image's own corners: shrinking with ``cv2.resize`` maps the continuous image
    extent, while a homography maps pixel indices, and the half-pixel between the
    two conventions is a systematic scale error — small, always in the same
    direction, and directly on top of the sub-pixel alignment the enhancement
    pair depends on. :func:`~scandar.geometry.scale_points` moves the corners the
    way ``resize`` moved the pixels, and the mismatch disappears.
    """
    import cv2

    params: dict = {}
    page = scan
    scan_height, scan_width = scan.shape[:2]
    source = rect_corners(scan_width, scan_height)

    # Shrink with INTER_AREA first if the page lands smaller than the scan.
    # warpPerspective interpolates but does not average, so handing it a 2.5x
    # reduction would alias every pen stroke.
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    target_width = float(np.mean([edges[0], edges[2]]))
    target_height = float(np.mean([edges[1], edges[3]]))
    if scan_width > 1.3 * target_width and target_width >= 16 and target_height >= 16:
        drawn = (int(round(target_width)), int(round(target_height)))
        page = cv2.resize(page, drawn, interpolation=cv2.INTER_AREA)
        source = scale_points(source, drawn[0] / scan_width, drawn[1] / scan_height)
        params["drawn_at"] = list(drawn)

    if rng.random() < options.curl_prob:
        page, params["curl"] = curl_page(page, rng)
    else:
        params["curl"] = None

    tint = None
    if rng.random() < options.page_tint_prob:
        # Coloured stock: a blue notebook cover, cream paper, a form printed on
        # pink. The enhancement network never sees this — its target is the scan.
        tint = rng.uniform(0.80, 1.05, size=3)
        if rng.random() < options.page_dark_prob:
            tint = tint * float(rng.uniform(0.35, 0.70))  # a dark printed card
        page = np.clip(page.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    params["tint"] = None if tint is None else [round(float(v), 3) for v in tint]

    return page, source, params


def _cast_drop_shadow(composite, mask, rng, options: SynthOptions):
    """Darken the surface where the page blocks the light."""
    import cv2

    angle = float(rng.uniform(0, 2 * np.pi))
    offset = float(rng.uniform(*options.drop_shadow_offset))
    sigma = float(rng.uniform(*options.drop_shadow_sigma))
    strength = float(rng.uniform(*options.drop_shadow_strength))

    shift = np.float32([[1, 0, np.cos(angle) * offset], [0, 1, np.sin(angle) * offset]])
    height, width = mask.shape[:2]
    shifted = cv2.warpAffine(mask, shift, (width, height))
    blurred = degrade_module.blur_mask(shifted.astype(np.float32) / 255.0, sigma)

    composite = degrade_module.apply_gain(composite, 1.0 - strength * blurred)
    return composite, {
        "angle_deg": round(float(np.degrees(angle)), 1),
        "offset_px": round(offset, 1),
        "sigma": round(sigma, 1),
        "strength": round(strength, 3),
    }


def _add_distractor(composite, corners, rng, options: SynthOptions):
    """Another sheet of paper in the frame, drawn *under* the page.

    A detector trained only on "one bright rectangle on a dark surface" solves
    that problem instead of the real one. Keeping the distractor behind the page
    means it can peek out from underneath — a stack of paper, which is what a
    desk actually looks like — without ever occluding a corner and making the
    label ambiguous.
    """
    import cv2

    height, width = composite.shape[:2]
    centre = np.asarray(corners, dtype=np.float64).mean(axis=0)
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    size = float(np.mean(edges)) * float(rng.uniform(0.7, 1.15))
    angle = float(rng.uniform(0, np.pi))

    offset = rng.uniform(-1, 1, size=2) * size * 0.55
    half = np.array([size * 0.45, size * 0.62])
    quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64) * half
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    quad = quad @ rotation.T + centre + offset

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)

    paper = np.array(PAPER_WHITE, dtype=np.float32) * float(rng.uniform(0.72, 1.02)) * 255.0
    sheet = np.empty_like(composite)
    sheet[:] = np.clip(np.rint(paper), 0, 255).astype(np.uint8)
    composite = cv2.add(
        degrade_module.apply_gain(composite, 1.0 - mask), degrade_module.apply_gain(sheet, mask)
    )
    return composite, {"size_px": round(size, 1), "angle_deg": round(float(np.degrees(angle)), 1)}
