"""Inference pipelines for unseen data.  *(brief §3.4, §5.1 and §7)*

Built so far:

``enhance_document(image)``  — brief §3.4
    Takes an unseen *rectified* document image: preprocess, run the network
    fully-convolutionally in overlapping tiles with cosine blending, return an
    8-bit image at the size that came in.

``rectify_document(photo, corners)``
    Flattens a page out of a raw photo given its four corners, which is the step
    that has to happen before the enhancement network sees anything. **It is not
    optional.** The network was trained on flattened pages and on nothing else, so
    handed a whole photo it faithfully tries to turn the desk, the floor and the
    wall into white paper with ink on them. Until the corner detector exists the
    four corners have to be supplied by hand; afterwards ``detect_corners`` fills
    them in and the two compose into ``scan_document``.

Not built yet: ``detect_corners(photo)`` *(brief §5.1)*, which needs the corner
detectors, and ``scan_document(photo)`` *(brief §7, the bonus)*, which composes
the two — photo in, clean scan out, no human input. Corner ordering matters
there: a permuted quad flips or rotates the page.

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
from .model import clamp_image

__all__ = ["enhance_document", "rectify_document", "tiled_forward", "enhance_file"]

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
