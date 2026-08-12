"""Homographies, quadrilaterals and corner ordering.

Shared by both mandatory tasks: the synthetic generator picks a destination quad
and derives the corner labels from it, and the corner detector is scored against
those same quads.

**Corner order is TL, TR, BR, BL everywhere.** Mixed-up corners silently break
both the evaluation metric and the rectification, which is why every quad that
enters the project goes through :func:`order_corners` once and is never reordered
again.

Two coordinate conventions live here, and keeping them apart is what makes the
(degraded input, clean target) pairs line up to a fraction of a pixel:

*Pixel indices.* Homographies operate on pixel **indices**, because that is what
``cv2.warpPerspective`` samples: output pixel ``(x, y)`` is fetched from
``H⁻¹·(x, y, 1)``. So the four corners of a ``W x H`` image are ``(0, 0)``,
``(W-1, 0)``, ``(W-1, H-1)``, ``(0, H-1)`` — see :func:`rect_corners`.

*Normalised coordinates.* Corner labels are stored as ``(x + 0.5) / W`` rather
than ``x / W``. ``cv2.resize`` maps the *continuous* image extent, not the index
range, so only the half-pixel form is exactly invariant under a resize — and a
corner label that does not survive a resize is a wrong label *(brief §2.2)*.
"""

from __future__ import annotations

import numpy as np

# Interior angles below this are not a page seen at an angle, they are a sliver.
MIN_INTERIOR_ANGLE_DEG = 20.0


# ---------------------------------------------------------------------------
# ordering and construction
# ---------------------------------------------------------------------------
def order_corners(points) -> np.ndarray:
    """Canonicalise four points to TL, TR, BR, BL.

    Sorting by angle around the centroid fixes the *winding* for any convex quad
    — in image coordinates, where y grows downward, ascending angle runs
    clockwise on screen — and the corner nearest the origin then fixes where the
    ring starts. The naive "split by y, then sort by x" recipe agrees with this
    for gently tilted pages and disagrees exactly where it matters, on the
    strongly rotated ones.

    Ordering a page rotated by more than 45° is genuinely ambiguous: nothing in
    four bare points says which edge is the top. The generator keeps rotation
    inside ±30° for that reason, and the real photos are all upright.
    """
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)

    delta = points - points.mean(axis=0)
    ring = points[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]

    # Start the ring at the top-left. `lexsort` breaks the tie between two equally
    # "top-left" corners on y, so the result is deterministic for a diamond.
    scores = ring.sum(axis=1)
    start = int(np.lexsort((ring[:, 1], scores))[0])
    return np.roll(ring, -start, axis=0).astype(np.float32)


def rect_corners(width: int, height: int) -> np.ndarray:
    """The four corners of a ``width x height`` image, as pixel indices, TL-first."""
    w, h = float(width - 1), float(height - 1)
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)


def rect_size_for(shape, out_width: int, multiple_of: int | None = None) -> tuple[int, int]:
    """Output ``(width, height)`` that preserves the aspect ratio of *shape*.

    *shape* is anything with ``(height, width)`` leading dimensions — an image
    array's ``.shape`` works. ``multiple_of`` rounds the height so a fully
    convolutional network can downsample it without a remainder.
    """
    height, width = shape[0], shape[1]
    out_height = int(round(out_width * height / width))
    if multiple_of:
        out_width = max(multiple_of, int(round(out_width / multiple_of)) * multiple_of)
        out_height = max(multiple_of, int(round(out_height / multiple_of)) * multiple_of)
    return int(out_width), int(out_height)


# ---------------------------------------------------------------------------
# homographies
# ---------------------------------------------------------------------------
def homography(src, dst) -> np.ndarray:
    """The 3x3 transform taking the four points *src* onto the four points *dst*."""
    import cv2

    src = np.asarray(src, dtype=np.float32).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def warp_points(H, points) -> np.ndarray:
    """Apply a homography to an ``(N, 2)`` array of points."""
    import cv2

    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def translation(dx: float, dy: float) -> np.ndarray:
    """A 3x3 translation, for composing a crop into a homography chain.

    Warping straight into a patch is far cheaper than warping a whole page and
    then slicing it, and composing the crop in is what makes that exact.
    """
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# scaling and normalisation
# ---------------------------------------------------------------------------
def scale_points(points, scale_x: float, scale_y: float) -> np.ndarray:
    """Rescale pixel-index points the way ``cv2.resize`` rescales the pixels.

    The half-pixel terms are not pedantry: without them a corner drifts by up to
    half a pixel per resize, always in the same direction, and the drift lands in
    the localisation error of every model that is scored afterwards.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] + 0.5) * scale_x - 0.5
    out[:, 1] = (points[:, 1] + 0.5) * scale_y - 0.5
    return out.astype(np.float32)


def normalize_corners(points, size) -> np.ndarray:
    """Pixel indices -> ``[0, 1]``, given ``size = (width, height)`` *(brief §2.2)*."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = float(size[0]), float(size[1])
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] + 0.5) / width
    out[:, 1] = (points[:, 1] + 0.5) / height
    return out.astype(np.float32)


def denormalize_corners(points, size) -> np.ndarray:
    """``[0, 1]`` -> pixel indices, the exact inverse of :func:`normalize_corners`."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = float(size[0]), float(size[1])
    out = np.empty_like(points)
    out[:, 0] = points[:, 0] * width - 0.5
    out[:, 1] = points[:, 1] * height - 0.5
    return out.astype(np.float32)


def resize_with_corners(image: np.ndarray, corners, size) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image to ``size = (width, height)`` and move its corners with it.

    The two halves of this operation are the ones the brief warns about being
    separated, so they are one function and there is no way to call half of it.
    """
    import cv2

    height, width = image.shape[:2]
    out_width, out_height = int(size[0]), int(size[1])
    # INTER_AREA is the right filter when shrinking; anything else point-samples
    # and turns a background texture into aliasing the detector can latch onto.
    shrinking = out_width < width or out_height < height
    resized = cv2.resize(
        image,
        (out_width, out_height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    moved = scale_points(corners, out_width / width, out_height / height)
    return resized, moved


# ---------------------------------------------------------------------------
# validity and overlap
# ---------------------------------------------------------------------------
def quad_area(quad) -> float:
    """Absolute area of a quadrilateral, by the shoelace formula."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def quad_problem(
    quad,
    canvas,
    margin: float = 2.0,
    min_edge_fraction: float = 0.05,
    area_fraction: tuple[float, float] = (0.05, 0.95),
    min_angle_deg: float = MIN_INTERIOR_ANGLE_DEG,
) -> str | None:
    """Why *quad* is unusable as a page outline, or ``None`` if it is fine.

    Returning the reason rather than a bare ``False`` is what makes a rejection
    loop debuggable: when the generator suddenly rejects nine tries out of ten,
    the counter says whether the pages are too small or the tilt is too extreme.
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    width, height = float(canvas[0]), float(canvas[1])

    if not np.isfinite(quad).all():
        return "non-finite coordinates"

    if (
        quad[:, 0].min() < margin
        or quad[:, 1].min() < margin
        or quad[:, 0].max() > width - 1 - margin
        or quad[:, 1].max() > height - 1 - margin
    ):
        return "outside the canvas"

    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < min_edge_fraction * min(width, height):
        return f"shortest edge {lengths.min():.0f}px is too short"

    # Convexity: with a consistent winding every turn goes the same way. Written
    # out rather than via np.cross, which deprecated 2-D vectors in NumPy 2.
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return "not convex"

    # Interior angle at corner i sits between edge i-1 reversed and edge i.
    incoming = -np.roll(edges, 1, axis=0)
    cosines = np.sum(incoming * edges, axis=1) / (np.roll(lengths, 1) * lengths)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    if angles.min() < min_angle_deg:
        return f"sharpest corner {angles.min():.0f}° is a sliver"

    ratio = quad_area(quad) / (width * height)
    if not area_fraction[0] <= ratio <= area_fraction[1]:
        return f"covers {ratio:.0%} of the canvas"

    return None


def is_valid_quad(quad, canvas, **kwargs) -> bool:
    """Whether *quad* is a plausible page outline inside a ``(width, height)`` canvas."""
    return quad_problem(quad, canvas, **kwargs) is None


def quad_iou(a, b, max_side: int = 1024) -> float:
    """Intersection over union of two quadrilaterals.

    Corner-by-corner distance says how far the prediction is; this says how much
    of the *page* the prediction would actually rectify, which is the number the
    downstream enhancement stage cares about.

    Rasterised rather than solved analytically: polygon clipping for the general
    non-convex case is a lot of code to get subtly wrong, and both quads are
    downscaled to at most *max_side* first, so the cost is bounded.
    """
    import cv2

    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0

    both = np.concatenate([a, b])
    origin = np.floor(both.min(axis=0)) - 1.0
    extent = np.ceil(both.max(axis=0)) + 1.0 - origin
    scale = min(1.0, max_side / max(extent.max(), 1.0))
    size = np.maximum(np.ceil(extent * scale), 2.0).astype(int)

    masks = []
    for quad in (a, b):
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round((quad - origin) * scale).astype(np.int32)], 1)
        masks.append(mask.astype(bool))

    union = int(np.count_nonzero(masks[0] | masks[1]))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(masks[0] & masks[1]) / union)


def corner_errors(predicted, target) -> np.ndarray:
    """Per-corner Euclidean distance, in whatever units the points are given in."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(predicted - target, axis=1)
