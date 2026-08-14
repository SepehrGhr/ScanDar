"""The warp, in torch, with a gradient.  *(brief §7, the bonus)*

Everything else in this project flattens a page with ``cv2.warpPerspective``,
which is fast, exact and completely opaque to autograd. That is fine for
generating data and for inference, and it is fatal for the bonus: chaining the
corner detector into the enhancement network only means something if the
enhancement loss can be pushed back *through* the flattening and onto the
predicted corners. So the same operation is written a second time here, out of
two differentiable pieces:

``homography_from_points(src, dst)``
    The four-point solve ``cv2.getPerspectiveTransform`` does, as an 8x8 linear
    system solved with :func:`torch.linalg.solve` — differentiable in both the
    source and the destination points.
``warp_perspective(images, matrices, size)``
    Sampling, as ``F.grid_sample`` over a grid built from the matrix.
``rectify(images, corners, size)`` / ``rectify_patch(...)``
    The two the rest of the project actually calls: flatten the page bounded by
    four corners, either whole or into one crop of it, and the crop is composed
    *into* the homography exactly as :meth:`scandar.synth.Sample.rectify_patch`
    composes it in numpy.

**The brief's no-third-party-libraries rule binds the degradation pipeline, not
this.** kornia is explicitly permitted for the bonus warp. It is not used
anyway: the whole differentiable path is two dozen lines of torch, and adding a
dependency to avoid writing them would be the wrong trade in a project whose
point is that the pieces are built rather than imported.

**Conventions, both of which are load-bearing.**

*Pixel indices, and ``align_corners=True``.* Corners are in the same pixel-index
convention as :mod:`scandar.geometry` — the four corners of a ``W x H`` image are
``(0, 0)`` to ``(W-1, H-1)`` — and ``grid_sample`` with ``align_corners=True``
maps -1 onto pixel 0 and +1 onto pixel ``W-1``, which is that convention exactly.
With ``align_corners=False`` the same matrix would sample half a pixel off, in a
direction that looks like a slightly worse model rather than like a bug.

*Float32, always.* The 8x8 solve and the grid's gradient are the two places in
this project where fp16 genuinely falls over: the system is built from products
of coordinates that reach into the millions on a 2560-pixel photo, and half
precision has three decimal digits to spend on them. Both functions therefore
cast to fp32 and disable autocast around themselves, the way the restoration
loss already does for SSIM's variance products.

**How the gradient reaches a corner**, since the number that comes out is easy to
misread. ``grid_sample`` differentiates with respect to the sampling grid, and
that derivative is the *image gradient at the sample point* — so moving a corner
changes the loss only through the intensity slope of what lies under the
resampled page. It is Lucas-Kanade's derivative, arrived at from the other
direction. Where the page is blank white paper the gradient is genuinely zero and
that is not a bug; it is why the anchor of this chain is the fixed target rather
than anything in the warp.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = [
    "rect_corners_like",
    "denormalize_corners",
    "normalize_corners",
    "homography_from_points",
    "transform_points",
    "perspective_grid",
    "warp_perspective",
    "rectify",
    "rectify_patch",
]


def _as_batch(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Accept ``(4, 2)`` or ``(B, 4, 2)``; say which came in."""
    if points.dim() == 2:
        return points[None], True
    if points.dim() == 3:
        return points, False
    raise ValueError(f"expected (4, 2) or (B, 4, 2) points, got {tuple(points.shape)}")


def rect_corners_like(
    width: int,
    height: int,
    batch: int = 1,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """``(B, 4, 2)`` corners of a ``width x height`` rectangle, TL TR BR BL.

    The torch twin of :func:`scandar.geometry.rect_corners`, and it agrees with
    it down to the ``-1``: these are pixel *indices*, so the bottom-right corner
    of a 1024-wide image is 1023, not 1024.
    """
    w, h = float(width - 1), float(height - 1)
    corners = torch.tensor(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], device=device, dtype=dtype
    )
    return corners[None].expand(int(batch), 4, 2)


def denormalize_corners(corners: torch.Tensor, size) -> torch.Tensor:
    """``[0, 1]`` corners onto the pixel indices of a ``(width, height)`` image.

    The exact inverse of :func:`scandar.geometry.normalize_corners`, half pixel
    included, and differentiable — which is the only reason it is written again
    here instead of being called through numpy.

    ``size`` may be a pair, or a ``(B, 2)`` tensor of per-sample sizes, because a
    batch of real photographs is not all one size.
    """
    corners = corners.float()
    if isinstance(size, torch.Tensor):
        extent = size.to(corners.device, corners.dtype).view(-1, 1, 2)
    else:
        extent = torch.tensor(
            [float(size[0]), float(size[1])], device=corners.device, dtype=corners.dtype
        ).view(1, 1, 2)
    return corners * extent - 0.5


def normalize_corners(corners: torch.Tensor, size) -> torch.Tensor:
    """Pixel indices back to ``[0, 1]``, the inverse of the above."""
    corners = corners.float()
    if isinstance(size, torch.Tensor):
        extent = size.to(corners.device, corners.dtype).view(-1, 1, 2)
    else:
        extent = torch.tensor(
            [float(size[0]), float(size[1])], device=corners.device, dtype=corners.dtype
        ).view(1, 1, 2)
    return (corners + 0.5) / extent


def homography_from_points(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """The 3x3 transform taking four *src* points onto four *dst* points.

    ``cv2.getPerspectiveTransform`` in torch, and it returns the same matrix to
    within floating-point noise — there is a test that says so, because "the
    differentiable path agrees with the one that generated the data" is the
    property everything downstream rests on.

    The standard direct linear transform. A homography has eight degrees of
    freedom, so ``h33`` is fixed at 1 and each correspondence contributes two
    rows to an 8x8 system:

        [x  y  1  0  0  0  -u x  -u y] h = u
        [0  0  0  x  y  1  -v x  -v y] h = v

    Solved rather than least-squared: with exactly four points the system is
    square, and :func:`torch.linalg.solve` differentiates cleanly through it.
    """
    src, squeeze = _as_batch(src)
    dst, _ = _as_batch(dst)
    if src.shape != dst.shape:
        raise ValueError(f"src {tuple(src.shape)} and dst {tuple(dst.shape)} disagree")
    if src.shape[1] != 4:
        raise ValueError(f"a homography needs exactly four points, got {src.shape[1]}")

    src = src.float()
    dst = dst.float()
    batch = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rows_x = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    rows_y = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    # Interleave so the two rows of a correspondence stay together. It changes
    # nothing mathematically and makes the matrix readable when it is printed.
    system = torch.stack([rows_x, rows_y], dim=2).reshape(batch, 8, 8)
    target = torch.stack([u, v], dim=-1).reshape(batch, 8, 1)

    solution = torch.linalg.solve(system, target).reshape(batch, 8)
    matrix = torch.cat([solution, torch.ones(batch, 1, device=src.device, dtype=src.dtype)], dim=1)
    matrix = matrix.reshape(batch, 3, 3)
    return matrix[0] if squeeze else matrix


def transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a batch of homographies to ``(B, N, 2)`` points."""
    points, squeeze = _as_batch(points) if points.dim() != 3 else (points, False)
    matrix = matrix if matrix.dim() == 3 else matrix[None]
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    mapped = homogeneous @ matrix.transpose(-1, -2)
    out = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-8)
    return out[0] if squeeze else out


def perspective_grid(
    matrix: torch.Tensor,
    out_size,
    source_size,
    offset=(0.0, 0.0),
) -> torch.Tensor:
    """The sampling grid ``F.grid_sample`` wants, for one perspective warp.

    *matrix* maps **output** pixel indices to **source** pixel indices — the
    inverse direction from the one a forward warp is usually written in, and the
    direction ``warpPerspective`` samples in too. *offset* shifts the output
    origin, which is how a crop is taken without flattening the whole page first:
    output pixel ``(px, py)`` stands for page pixel ``(px + offset_x, py +
    offset_y)``. It may be one pair for the whole batch or ``(B, 2)``, one origin
    per sample, because the patch sampler puts every sample's crop somewhere
    different. The offset is a constant either way, so composing it in changes
    nothing about the gradient that reaches the corners.
    """
    out_width, out_height = int(out_size[0]), int(out_size[1])
    matrix = matrix if matrix.dim() == 3 else matrix[None]
    batch = matrix.shape[0]
    device, dtype = matrix.device, matrix.dtype

    origins = torch.as_tensor(offset, device=device, dtype=dtype).reshape(-1, 2).detach()
    if origins.shape[0] == 1:
        origins = origins.expand(batch, 2)
    if origins.shape[0] != batch:
        raise ValueError(f"{origins.shape[0]} output origins for {batch} matrices")

    xs = torch.arange(out_width, device=device, dtype=dtype)
    ys = torch.arange(out_height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)[None]
    flat = flat + origins[:, None, :]

    mapped = transform_points(matrix, flat).reshape(batch, out_height, out_width, 2)

    # Pixel indices to grid_sample's [-1, 1], under align_corners=True: index 0
    # is -1 and index W-1 is +1. A one-pixel image would divide by zero, so the
    # denominator is floored.
    source_width = max(int(source_size[0]) - 1, 1)
    source_height = max(int(source_size[1]) - 1, 1)
    scale = torch.tensor([2.0 / source_width, 2.0 / source_height], device=device, dtype=dtype)
    return mapped * scale - 1.0


def warp_perspective(
    images: torch.Tensor,
    matrix: torch.Tensor,
    out_size,
    offset=(0.0, 0.0),
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """``cv2.warpPerspective`` with a gradient. ``(B, C, H, W)`` in and out.

    *matrix* maps output pixel indices to input pixel indices, as
    :func:`perspective_grid` describes.

    ``padding_mode="border"`` rather than ``"zeros"``: a predicted quad may reach
    a little outside the photo — the guardrail in the inference pipeline
    tolerates that deliberately, because a page can be photographed with a corner
    just off the frame — and a band of black along the edge of the rectified page
    would be read by the enhancement network as ink to preserve.
    """
    if images.dim() != 4:
        raise ValueError(f"expected (B, C, H, W) images, got {tuple(images.shape)}")
    with torch.autocast(images.device.type, enabled=False):
        images = images.float()
        height, width = images.shape[-2:]
        grid = perspective_grid(matrix.float(), out_size, (width, height), offset=offset)
        if grid.shape[0] != images.shape[0]:
            if grid.shape[0] != 1:
                raise ValueError(
                    f"{grid.shape[0]} matrices for {images.shape[0]} images"
                )
            grid = grid.expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(
            images, grid, mode=mode, padding_mode=padding_mode, align_corners=True
        )


def _page_to_source(corners: torch.Tensor, images: torch.Tensor, rect_size) -> torch.Tensor:
    """The homography from rectified-page pixels to source-photo pixels.

    *corners* are in the source photo's own pixel indices. This is the inverse
    direction of "flattening", which is exactly what the sampler needs: for each
    pixel of the flat page, where in the photo does it come from.
    """
    batch = images.shape[0]
    rect = rect_corners_like(
        int(rect_size[0]), int(rect_size[1]), batch, images.device, torch.float32
    )
    return homography_from_points(rect, corners.float())


def rectify(
    images: torch.Tensor,
    corners: torch.Tensor,
    rect_size,
    normalised: bool = True,
    size=None,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Flatten the page bounded by *corners*, differentiably.

    The torch twin of :meth:`scandar.synth.Sample.rectify`. *corners* are
    ``(B, 4, 2)`` in canonical TL, TR, BR, BL order — **no reordering happens
    here**, because :func:`scandar.geometry.order_corners` is numpy and would cut
    the graph, and because the heatmap detector's four channels are already in
    that order by construction. Ordering is an inference-time guardrail and lives
    in :func:`scandar.pipelines.scan_document`.

    *size* is each photo's true ``(width, height)``, for a batch that was padded
    up to a common shape because it mixed portrait and landscape photographs.
    Left out, the tensor's own shape is used, which is right whenever nothing was
    padded.
    """
    if normalised:
        height, width = images.shape[-2:]
        corners = denormalize_corners(corners, size if size is not None else (width, height))
    matrix = _page_to_source(corners, images, rect_size)
    # Bilinear while training — it is what the gradient is cheapest and best
    # behaved through — and bicubic when this is standing in for
    # ``cv2.warpPerspective`` at inference, which is asked for cubic.
    return warp_perspective(images, matrix, rect_size, mode=mode)


def rectify_patch(
    images: torch.Tensor,
    corners: torch.Tensor,
    rect_size,
    box,
    patch_size: int,
    normalised: bool = True,
    size=None,
) -> torch.Tensor:
    """One ``patch_size`` crop of :func:`rectify`, warped straight out of the photo.

    *box* is ``(B, 2)`` or a pair — the crop's ``(x, y)`` origin in the
    coordinates of the page rectified at *rect_size*. The crop is composed into
    the sampling grid rather than taken afterwards, which is what makes training
    on patches affordable: a whole 1024x1448 page through the enhancement network
    with a backward pass does not fit on a 6 GB card, and a 256x256 crop of it
    costs what the enhancement baseline already paid per step.

    Because the origin is a constant, the gradient reaching the corners is the
    same one a full-page warp would deliver, restricted to the pixels this crop
    looked at.
    """
    if images.dim() != 4:
        raise ValueError(f"expected (B, C, H, W) images, got {tuple(images.shape)}")
    batch = images.shape[0]
    if normalised:
        height, width = images.shape[-2:]
        corners = denormalize_corners(corners, size if size is not None else (width, height))
    matrix = _page_to_source(corners, images, rect_size)

    patch = int(patch_size)
    with torch.autocast(images.device.type, enabled=False):
        grid = perspective_grid(
            matrix.float(),
            (patch, patch),
            (images.shape[-1], images.shape[-2]),
            offset=torch.as_tensor(box, device=images.device, dtype=torch.float32),
        )
        if grid.shape[0] != batch:
            raise ValueError(f"{grid.shape[0]} crop origins for {batch} images")
        return F.grid_sample(
            images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )
