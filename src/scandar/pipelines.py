"""Inference pipelines for unseen data.  *(brief §3.4, §5.1 and §7)*

Built so far:

``enhance_document(image)``  — brief §3.4
    Takes an unseen *rectified* document image: preprocess, run the network
    fully-convolutionally in overlapping tiles with cosine blending, return an
    8-bit image at the size that came in.

``detect_corners(photo, model)``  — brief §5.1
    Takes an unseen *raw* photo: preprocess, predict the four corners, map them
    back to the original resolution, put them in canonical order, check that what
    came out is a usable page outline, and hand back an overlay. If the quad is
    degenerate it falls back to a classical Canny detector and says which path
    ran.

``rectify_document(photo, corners)``
    Flattens a page out of a raw photo given its four corners, which is the step
    that has to happen before the enhancement network sees anything. **It is not
    optional.** The network was trained on flattened pages and on nothing else, so
    handed a whole photo it faithfully tries to turn the desk, the floor and the
    wall into white paper with ink on them.

Not built yet: ``scan_document(photo)`` *(brief §7, the bonus)*, which composes
the two — photo in, clean scan out, no human input. Corner ordering matters
there: a permuted quad flips or rotates the page.

**Why there is a classical fallback at all.** The grade is decided live, on
photographs nobody has seen, and a neural detector that has never met a
particular background can return four points that are not a quadrilateral in any
useful sense — crossed, collapsed, or a sliver. Everything downstream then
produces something worse than nothing: a homography built from a degenerate quad
smears the page across the output. Canny plus ``findContours`` plus
``approxPolyDP`` finds a bright convex quadrilateral on a darker surface without
having been trained on anything, which is a genuinely different failure mode, so
the two rarely fail together. It is cheap insurance, and it is built from the
course's own techniques.

**Why tiles.** The network is fully convolutional, so a whole 1024x1448 page can
be pushed through in one pass, and on a machine with the memory to spare that is
what happens. But a page photographed from close up can be several thousand
pixels on a side, and the activations of a U-Net at that resolution do not fit in
6 GB. Tiling caps the memory at the tile size regardless of the page.

**Why the tiles overlap and are blended.** Independent tiles butt against each
other at a seam the eye finds immediately: the network's answer for a pixel
depends on its neighbourhood, and two tiles disagree slightly about a pixel near
their shared border. Overlapping them and cross-fading over the overlap with a
raised-cosine window makes the transition continuous — the weights sum to one
everywhere, so the result is a genuine weighted average and not a brightening
band down the middle of the page.

**How much overlap.** Not a round number picked by feel: the network's receptive
field is 189x189, measured by pushing a gradient back from one output pixel, so
every output pixel wants 86 pixels of real context on each side. With the 64-pixel
overlap this started with, a pixel in the middle of the blend sat 32 pixels from
both tiles' edges — inside the context radius of *both* contributors, so the
cross-fade was averaging two answers at exactly the place each was least sure of.
At 192 the midpoint sits 96 pixels in, past the 86 it needs, and on a 1024x1448
page it costs nothing at all: still twelve tiles, because the last tile in each
direction is flush against the edge either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import imread_rgb, imwrite_rgb, to_uint8
from .model import SOFT_ARGMAX_WINDOW, clamp_image

__all__ = [
    "enhance_document",
    "rectify_document",
    "tiled_forward",
    "enhance_file",
    "detect_corners",
    "detect_corners_file",
    "classical_corners",
    "predict_corners",
    "draw_corners",
]

#: A4, the shape every scan in this project was made from. Used when a quad's own
#: aspect ratio cannot be trusted — under a steep perspective the projected edge
#: lengths are a poor estimate of the real ones.
A4_ASPECT = 1.0 / 1.4142


def rectify_document(photo, corners, out_width: int = 1024, aspect=None) -> np.ndarray:
    """Flatten the page bounded by *corners* out of *photo*.

    *corners* is any four ``(x, y)`` points in photo pixels; they are put into the
    project's canonical TL, TR, BR, BL order first, because a permuted quad
    rotates or mirrors the page and every downstream step then quietly works on a
    sideways document.

    ``aspect`` is width over height. Left as ``None`` it is estimated from the
    quad's own edges, which is what a scanning app does and is right for a page
    photographed from a modest angle. Pass ``"a4"`` for a page you know is A4 and
    was shot steeply enough that its projected edges understate one dimension —
    foreshortening makes the far edge shorter, and an aspect read off it squashes
    the result.
    """
    import cv2

    from .geometry import homography, order_corners, rect_corners

    if isinstance(photo, (str, Path)):
        photo = imread_rgb(photo)
    quad = order_corners(np.asarray(corners, dtype=np.float32).reshape(4, 2))

    if aspect in ("a4", "A4"):
        ratio = A4_ASPECT
    elif aspect is not None:
        ratio = float(aspect)
    else:
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        width = float(np.mean([edges[0], edges[2]]))   # top and bottom
        height = float(np.mean([edges[1], edges[3]]))  # right and left
        ratio = width / max(height, 1e-6)

    out_width = int(out_width)
    out_height = max(1, int(round(out_width / max(ratio, 1e-6))))
    transform = homography(quad, rect_corners(out_width, out_height))
    return cv2.warpPerspective(
        photo, transform, (out_width, out_height), flags=cv2.INTER_CUBIC
    )


def _cosine_window(height: int, width: int, overlap: int) -> np.ndarray:
    """A separable raised-cosine ramp: 0 at the tile edge, 1 by ``overlap`` in.

    Only the overlap region is ramped. A window that tapered across the whole
    tile would give the tile centre almost all the weight and waste most of what
    was computed.
    """
    def ramp(length: int) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        taper = min(overlap, length // 2)
        if taper > 0:
            edge = 0.5 - 0.5 * np.cos(np.pi * (np.arange(taper) + 0.5) / taper)
            weights[:taper] = edge
            weights[-taper:] = edge[::-1]
        return weights

    return np.outer(ramp(height), ramp(width)).astype(np.float32)


def _starts(extent: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering ``extent``, with the last one flush against the end."""
    if extent <= tile:
        return [0]
    positions = list(range(0, extent - tile + 1, stride))
    if positions[-1] != extent - tile:
        positions.append(extent - tile)
    return positions


@torch.no_grad()
def tiled_forward(
    model,
    image: torch.Tensor,
    tile: int = 512,
    overlap: int = 192,
    device=None,
    amp: bool = False,
) -> torch.Tensor:
    """Run *model* over a large image in blended overlapping tiles.

    *image* is ``(3, H, W)`` or ``(1, 3, H, W)`` float in [0, 1]; the result has
    the same shape. Accumulating the weighted output and the weights separately
    and dividing at the end means the seams are correct without the tile order
    mattering, and it costs one extra single-channel buffer.
    """
    if image.dim() == 3:
        image = image[None]
    if image.shape[0] != 1:
        raise ValueError("tiled inference takes one image at a time")

    device = device if device is not None else next(model.parameters()).device
    height, width = image.shape[-2:]
    tile = int(min(tile, height, width))
    stride = max(1, tile - int(overlap))

    accumulated = torch.zeros(1, image.shape[1], height, width, dtype=torch.float32)
    weights = torch.zeros(1, 1, height, width, dtype=torch.float32)

    for top in _starts(height, tile, stride):
        for left in _starts(width, tile, stride):
            patch = image[..., top : top + tile, left : left + tile].to(device)
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                predicted = model(patch)
            predicted = clamp_image(predicted.float()).cpu()

            window = torch.from_numpy(_cosine_window(tile, tile, int(overlap)))[None, None]
            accumulated[..., top : top + tile, left : left + tile] += predicted * window
            weights[..., top : top + tile, left : left + tile] += window

    return accumulated / weights.clamp_min(1e-8)


def enhance_document(
    image,
    model,
    device=None,
    tile: int = 512,
    overlap: int = 192,
    max_side: int | None = None,
    amp: bool = False,
) -> np.ndarray:
    """A rectified document photo in, a clean scan out  *(brief §3.4)*.

    1. **Preprocess** — read the image if a path was given, and turn RGB uint8
       HWC into the float CHW tensor in [0, 1] the network was trained on. There
       is no mean/standard-deviation normalisation to apply: the target is an
       image in [0, 1], so the input lives in the same space, and with no
       pretrained weights anywhere in the project there are no external constants
       to match.
    2. **Predict** — fully convolutionally, in blended overlapping tiles.
    3. **Post-process** — back to the original dimensions and to 8 bits.
    4. Return the array, ready to be written or shown.

    ``max_side`` caps the resolution the network sees and is off by default.
    Restoration is a local job and the model was trained at the scale of a page
    rectified to 1024x1448; shrinking a larger page before enhancing it and
    stretching the result back is a way to keep that scale, at the cost of the
    detail the shrink threw away. Left off, the page is enhanced at its own
    resolution, which is what the tiling exists to make affordable.
    """
    import cv2

    if isinstance(image, (str, Path)):
        image = imread_rgb(image)
    if image.dtype != np.uint8:
        image = to_uint8(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB HWC image, got shape {image.shape}")

    original_height, original_width = image.shape[:2]
    working = image
    if max_side and max(original_height, original_width) > max_side:
        scale = max_side / max(original_height, original_width)
        working = cv2.resize(
            image,
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    tensor = torch.from_numpy(np.ascontiguousarray(working.transpose(2, 0, 1)))
    tensor = tensor.to(torch.float32).div_(255.0)

    was_training = model.training
    model.eval()
    restored = tiled_forward(model, tensor, tile=tile, overlap=overlap, device=device, amp=amp)
    if was_training:
        model.train()

    output = to_uint8(restored[0].permute(1, 2, 0).numpy())
    if output.shape[:2] != (original_height, original_width):
        output = cv2.resize(
            output, (original_width, original_height), interpolation=cv2.INTER_CUBIC
        )
    return output


def enhance_file(
    input_path,
    output_path,
    checkpoint,
    device=None,
    **kwargs,
) -> Path:
    """``enhance_document`` end to end over files, for scripts and the demo."""
    from .device import get_device
    from .model import load_model

    device = device if device is not None else get_device()
    model, _ = load_model(checkpoint, device=device)
    enhanced = enhance_document(input_path, model, device=device, **kwargs)
    return imwrite_rgb(output_path, enhanced)


# ---------------------------------------------------------------------------
# corner detection  (brief §5.1)
# ---------------------------------------------------------------------------
#: How forgiving the check on a predicted quad is. Looser than the generator's
#: own placement rules on purpose: those describe pages this project *creates*,
#: while these describe pages it must not refuse to *believe in*. A page may be
#: photographed close enough to fill the frame or to run a corner just off it, and
#: rejecting that would send a perfectly good detection to the fallback. What is
#: still rejected is a quad no page can be: crossed, collapsed, or a sliver.
QUAD_TOLERANCE = {
    "margin": -24.0,          # a corner may sit a little outside the frame
    "min_edge_fraction": 0.04,
    "area_fraction": (0.02, 1.0),
    "min_angle_deg": 20.0,
}


@torch.no_grad()
def predict_corners(
    photo: np.ndarray,
    model,
    device=None,
    input_size: int = 256,
    soft_argmax_window: int | None = SOFT_ARGMAX_WINDOW,
) -> dict:
    """Run a corner detector on one photo. Steps 1 and 2 of *(brief §5.1)*.

    **Preprocess** exactly as the training dataset did — resize to ``input_size``
    square with an area filter, RGB uint8 HWC to float CHW in [0, 1], no
    mean/standard-deviation normalisation, because the model was trained without
    one. **Predict**, and read coordinates back in the normalised form the labels
    live in, which is the form that survives being mapped onto any resolution.

    Both formulations come out of here identically — normalised ``(4, 2)`` — which
    is what lets everything downstream, including the comparison the brief asks
    for, be written once.
    """
    import cv2

    from .model import heatmap_peaks, soft_argmax2d

    device = device if device is not None else next(model.parameters()).device
    resized = cv2.resize(photo, (input_size, input_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
    tensor = tensor.to(torch.float32).div_(255.0)[None].to(device)

    was_training = model.training
    model.eval()
    output = model(tensor)
    if was_training:
        model.train()

    kind = str(getattr(model, "output_kind", "coords"))
    result = {"kind": kind, "confidence": None, "heatmaps": None}
    if kind == "heatmaps":
        # A window around the peak, unlike during training — and it is worth six
        # times the accuracy, not a rounding difference. See
        # `model.corners_from_output` for the measurement.
        coords = soft_argmax2d(output.float(), window=soft_argmax_window)
        peaks, values = heatmap_peaks(output.float())
        result["confidence"] = float(values.mean())
        result["heatmaps"] = output[0].float().cpu().numpy()
        result["peaks"] = peaks[0].cpu().numpy()
    elif kind == "coords":
        coords = output.float()
    else:
        raise ValueError(f"{type(model).__name__} is not a corner detector (output {kind!r})")

    result["normalised"] = coords[0].cpu().numpy().astype(np.float32)
    return result


def classical_corners(photo: np.ndarray, working_side: int = 768) -> np.ndarray | None:
    """Find a page-shaped quadrilateral with Canny, contours and a polygon fit.

    The guardrail, and no more than that: a rectangle-finder of the kind a
    scanning app used before neural networks, built from the course's own
    techniques. It works when the page is a bright convex shape against a darker
    surface, which is most photographs, and it fails on a busy background or a
    page whose edges run out of the frame — a failure mode uncorrelated with the
    detector's, which is exactly why it is worth having.

    Returns ordered TL, TR, BR, BL corners in *photo* pixels, or ``None``.

    Everything happens on a downscaled copy: at 768 pixels the paper's texture and
    the background's grain stop generating edges of their own, and the contours
    that survive are the ones that belong to objects.
    """
    import cv2

    from .geometry import is_valid_quad, order_corners, scale_points

    height, width = photo.shape[:2]
    scale = min(1.0, working_side / max(height, width))
    small = (
        cv2.resize(photo, (max(1, round(width * scale)), max(1, round(height * scale))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else photo
    )
    canvas = (small.shape[1], small.shape[0])

    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    # Thresholds from the image's own median rather than fixed numbers: a dim
    # photo and a bright one have nothing in common on an absolute scale.
    median = float(np.median(grey))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(grey, low, high)
    # Close the gaps a shadow or a low-contrast edge leaves, so the page's outline
    # is one closed contour rather than four disconnected sides.
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        for tolerance in (0.02, 0.03, 0.04, 0.05, 0.015, 0.01):
            approximation = cv2.approxPolyDP(contour, tolerance * perimeter, True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            quad = approximation.reshape(4, 2).astype(np.float64)
            if not is_valid_quad(quad, canvas, **QUAD_TOLERANCE):
                continue
            ordered = order_corners(quad)
            return scale_points(ordered, width / canvas[0], height / canvas[1])
    return None


def detect_corners(
    photo,
    model=None,
    device=None,
    input_size: int = 256,
    fallback: bool = True,
    verbose: bool = False,
    **predict_kwargs,
) -> dict:
    """A raw photo in, its four page corners out  *(brief §5.1)*.

    1. **Preprocess** — read the photo if a path was given, resize and normalise
       it the way the training data was.
    2. **Predict** the four corners with the trained detector.
    3. **Map** them back to the original resolution. The labels are normalised by
       width and height, so this is one multiplication — and it is the step the
       brief warns about, because a coordinate scaled by the wrong factor is a
       wrong label that looks like a bad model.
    4. **Order and check.** Canonical TL, TR, BR, BL, then a look at whether the
       result is a quadrilateral a page could be. If it is not, and *fallback* is
       on, the classical detector gets a turn.

    Returns a dict carrying ``corners`` in the photo's own pixels, the normalised
    form, ``source`` — which of the two paths produced the answer — and, for a
    heatmap model, the maps themselves and their peak heights. ``source`` is
    returned rather than logged so that a caller scoring a whole set can count how
    often the fallback ran; that count is worth knowing before presentation day.
    """
    from .geometry import denormalize_corners, normalize_corners, order_corners, quad_problem

    if isinstance(photo, (str, Path)):
        photo = imread_rgb(photo)
    if photo.ndim != 3 or photo.shape[2] != 3:
        raise ValueError(f"expected an RGB HWC photo, got shape {photo.shape}")
    height, width = photo.shape[:2]
    canvas = (width, height)

    result = {
        "source": "none",
        "problem": None,
        "confidence": None,
        "heatmaps": None,
        "kind": None,
    }
    if model is not None:
        prediction = predict_corners(
            photo, model, device=device, input_size=input_size, **predict_kwargs
        )
        result.update(
            {key: prediction.get(key) for key in ("confidence", "heatmaps", "kind", "peaks")}
        )
        quad = order_corners(denormalize_corners(prediction["normalised"], canvas))
        problem = quad_problem(quad, canvas, **QUAD_TOLERANCE)
        if problem is None:
            result.update({"corners": quad, "source": "model"})
        else:
            result["problem"] = problem

    if "corners" not in result and fallback:
        quad = classical_corners(photo)
        if quad is not None:
            result.update({"corners": quad, "source": "classical"})

    if "corners" not in result:
        # The frame itself. A rectification of the whole photo is wrong, but it is
        # wrong in a way a human can see and correct in one glance, which beats
        # raising in the middle of a demonstration.
        from .geometry import rect_corners

        result.update({"corners": rect_corners(width, height), "source": "frame"})

    result["corners"] = np.asarray(result["corners"], dtype=np.float32)
    result["normalised"] = normalize_corners(result["corners"], canvas)
    result["size"] = canvas
    if verbose:
        detail = f" ({result['problem']})" if result["problem"] else ""
        print(f"corners from the {result['source']} path{detail}")
    return result


def draw_corners(
    photo: np.ndarray,
    corners,
    color=(46, 204, 113),
    thickness: int | None = None,
    labels: bool = True,
) -> np.ndarray:
    """The overlay the brief asks for: the quad and its four labelled corners.

    Sizes itself to the photo — a 3-pixel line and a 12-pixel font are invisible
    on a 2560-pixel photograph and unreadable when it is scaled down for a report.
    Labelling the corners rather than only marking them is deliberate: the failure
    this project is most exposed to is a *permuted* quad, and a picture with four
    anonymous dots on it looks identical whether the ordering is right or wrong.
    """
    import cv2

    canvas = photo.copy()
    quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    scale = max(photo.shape[:2]) / 1000.0
    thickness = int(thickness or max(2, round(3 * scale)))
    radius = max(3, round(7 * scale))

    cv2.polylines(canvas, [np.round(quad).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    for index, (x, y) in enumerate(quad):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(canvas, centre, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, radius, (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)
        if labels:
            cv2.putText(
                canvas,
                ("TL", "TR", "BR", "BL")[index],
                (centre[0] + radius * 2, centre[1] - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, 0.8 * scale),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return canvas


def detect_corners_file(input_path, output_path, checkpoint, device=None, **kwargs) -> dict:
    """``detect_corners`` end to end over files, with the overlay written out.

    The input size comes from the checkpoint's own config rather than from a
    default, because a detector trained at one resolution and run at another is
    wrong in a way that looks like a mediocre model rather than like a bug.
    """
    from .device import get_device
    from .model import load_model

    device = device if device is not None else get_device()
    model, config = load_model(checkpoint, device=device)
    photo = imread_rgb(input_path) if isinstance(input_path, (str, Path)) else input_path
    result = detect_corners(
        photo,
        model,
        device=device,
        input_size=int(config.get("data", {}).get("corner_input", 256)),
        **kwargs,
    )
    result["written"] = imwrite_rgb(output_path, draw_corners(photo, result["corners"]))
    return result
