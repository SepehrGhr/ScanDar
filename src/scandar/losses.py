"""Loss functions, implemented from scratch.  *(brief §3.2)*

A plain pixel-wise MSE is known to produce blurry restorations, and blur is
precisely the enemy when the goal is readable text. The reason is worth stating
because the whole design of this module follows from it: squared error is
minimised by the *conditional mean*, so wherever the network is unsure exactly
where a stroke edge falls, the cheapest answer is to hedge — to output the
average of every plausible edge position, which is a smear. L1 is minimised by
the conditional *median*, which commits to one answer instead of averaging over
them, and that alone recovers a good deal of sharpness.

The loss this project trains on is a combination, in the spirit of the
image-restoration literature:

    L = 1.0 * L1  +  0.5 * (1 - MS-SSIM)  +  0.25 * L1(Sobel(x), Sobel(y))

L1 keeps the intensities honest without MSE's blur-favouring averaging, MS-SSIM
scores local structure the way an eye does and at several scales at once, and the
gradient term puts the penalty exactly where legibility lives — the edges of the
strokes. Every weight is read from the config, so the ablation the brief asks for
(MSE / L1 / L1+MS-SSIM / L1+MS-SSIM+gradient) is four config files and no code.

``gaussian_window`` / ``ssim`` / ``ms_ssim``
    Gaussian-window SSIM (11x11, sigma 1.5) and its five-scale variant, written
    here rather than imported — the same maths is needed for the evaluation
    metric anyway, and one implementation cannot drift from the other.
``sobel_loss``
    L1 between Sobel edge maps, through fixed convolution kernels.
``CombinedRestorationLoss`` / ``build_loss``
    The weighted sum, reporting each term separately so the training log shows
    which one is actually moving.

Corner detection has its own family *(brief §5)*, because it optimises a
different kind of thing:

``heatmap_mse``
    The pixel-wise loss on the heatmaps the brief asks for, with an optional
    weight on the cells the Gaussian actually occupies.
``wing_loss``
    The alternative to a plain coordinate L1, for the ablation. L1 gives the same
    gradient to a 40-pixel error and a 1-pixel one; the Wing loss keeps L1's
    behaviour on the large errors and amplifies the small ones, which is where a
    corner detector spends the second half of its training.
``CornerLoss``
    One class serving both approaches, because they differ only in which terms
    carry a weight: coordinates for approach A, heatmaps for approach B, and both
    at once for the variant that puts a coordinate loss through the soft-argmax.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "gaussian_window",
    "ssim",
    "ms_ssim",
    "sobel_edges",
    "sobel_loss",
    "SSIM",
    "MSSSIM",
    "CombinedRestorationLoss",
    "heatmap_mse",
    "wing_loss",
    "CornerLoss",
    "build_loss",
]

#: The five-scale weights from Wang et al.'s multi-scale SSIM paper, which are
#: what everyone else reports MS-SSIM with. Deviating from them would make this
#: project's numbers incomparable to any other.
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    """A separable Gaussian as a depthwise convolution weight, ``(C, 1, k, k)``.

    Depthwise, because SSIM is computed per channel and then averaged; mixing the
    channels inside the window would measure something else entirely.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    line = torch.exp(-(coords**2) / (2.0 * sigma**2))
    line = line / line.sum()
    window = torch.outer(line, line)
    return window.expand(channels, 1, size, size).contiguous()


def _filter(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, window, groups=x.shape[1])


def _ssim_maps(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    data_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SSIM map and the contrast-structure map, both un-pooled.

    The convolutions are unpadded on purpose. Padding the window would compare
    every border pixel against invented neighbours and quietly bias the score,
    and since MS-SSIM needs the contrast-structure term at every scale anyway,
    both maps come out of one pass.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_pred = _filter(pred, window)
    mu_target = _filter(target, window)
    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_cross = mu_pred * mu_target

    sigma_pred = _filter(pred * pred, window) - mu_pred_sq
    sigma_target = _filter(target * target, window) - mu_target_sq
    sigma_cross = _filter(pred * target, window) - mu_cross

    contrast = (2 * sigma_cross + c2) / (sigma_pred + sigma_target + c2)
    luminance = (2 * mu_cross + c1) / (mu_pred_sq + mu_target_sq + c1)
    return luminance * contrast, contrast


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Structural similarity, in [-1, 1] and 1.0 for identical images.

    ``reduction="none"`` returns one number per image, which is what the
    evaluation table needs — a mean and a standard deviation over a set require
    the per-image scores, not the average of the batch.
    """
    pred, target = pred.float(), target.float()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    smallest = min(pred.shape[-2:])
    if smallest < window_size:
        raise ValueError(f"image side {smallest} is smaller than the {window_size}px SSIM window")

    window = gaussian_window(window_size, sigma, pred.shape[1]).to(pred.device, pred.dtype)
    per_pixel, _ = _ssim_maps(pred, target, window, data_range)
    per_image = per_pixel.flatten(1).mean(dim=1)
    return per_image.mean() if reduction == "mean" else per_image


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: tuple[float, ...] = MS_SSIM_WEIGHTS,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-scale SSIM: contrast-structure at each scale, luminance at the last.

    Five scales need an image of at least ``(window - 1) * 2^4 + 1`` = 161 pixels
    on the short side, which the 256x256 training patches clear comfortably. A
    smaller input silently dropping to fewer scales would make two runs
    incomparable, so the number of scales is reduced *and the weights are
    renormalised*, which keeps the result on the same 0-to-1 footing.
    """
    pred, target = pred.float(), target.float()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    levels = len(weights)
    smallest = min(pred.shape[-2:])
    while levels > 1 and smallest < (window_size - 1) * 2 ** (levels - 1) + 1:
        levels -= 1
    scale_weights = torch.tensor(weights[:levels], dtype=torch.float32, device=pred.device)
    scale_weights = scale_weights / scale_weights.sum()

    window = gaussian_window(window_size, sigma, pred.shape[1]).to(pred.device, pred.dtype)
    values = []
    for level in range(levels):
        per_pixel, contrast = _ssim_maps(pred, target, window, data_range)
        chosen = per_pixel if level == levels - 1 else contrast
        # A negative contrast term would become NaN under a fractional power.
        # Clamping it away is what every reference implementation does.
        values.append(chosen.flatten(1).mean(dim=1).clamp_min(1e-6))
        if level < levels - 1:
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)

    stacked = torch.stack(values, dim=0)  # (levels, batch)
    per_image = torch.prod(stacked ** scale_weights[:, None], dim=0)
    return per_image.mean() if reduction == "mean" else per_image


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Horizontal and vertical Sobel responses, stacked on the channel axis.

    The kernels are fixed, applied depthwise so each colour channel keeps its own
    edges, and the input is padded by replication — zero padding would draw a
    bright artificial edge around the whole patch and put a quarter of the loss
    on the border.
    """
    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=x.dtype, device=x.device
    )
    kernel_y = kernel_x.t()
    channels = x.shape[1]
    weight = torch.stack([kernel_x, kernel_y]).repeat(channels, 1, 1).unsqueeze(1)
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, weight, groups=channels)


def sobel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between the Sobel edge maps — the term that protects legibility."""
    return F.l1_loss(sobel_edges(pred.float()), sobel_edges(target.float()))


class SSIM(nn.Module):
    """SSIM as a module, so ``1 - SSIM`` can be used as a loss directly."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ssim(pred, target, self.data_range, self.window_size, self.sigma)


class MSSSIM(SSIM):
    """The five-scale variant, same interface."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ms_ssim(pred, target, self.data_range, self.window_size, self.sigma)


class CombinedRestorationLoss(nn.Module):
    """The weighted sum, reporting every term it was asked for.

    ``forward`` returns ``(total, parts)``, where *parts* holds each *unweighted*
    term as a plain float. Unweighted, because the point of logging them is to see
    which one is actually moving, and a term scaled by its own weight makes a
    weight change look like progress.

    A weight of zero drops its term entirely rather than multiplying it by zero:
    MS-SSIM is the expensive one in the sum, and the ablation runs that exclude it
    should not pay for it.
    """

    def __init__(
        self,
        l1: float = 1.0,
        mse: float = 0.0,
        msssim: float = 0.5,
        ssim: float = 0.0,
        sobel: float = 0.25,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.weights = {
            "l1": float(l1),
            "mse": float(mse),
            "msssim": float(msssim),
            "ssim": float(ssim),
            "sobel": float(sobel),
        }
        if not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("every loss weight is zero — there is nothing to optimise")
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    @property
    def active(self) -> list[str]:
        return [name for name, weight in self.weights.items() if weight > 0]

    def extra_repr(self) -> str:
        return " + ".join(f"{self.weights[name]:g}*{name}" for name in self.active)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Mixed precision is a throughput trick, not a numerical claim: the
        # variance products inside SSIM are small enough that fp16 loses real
        # precision on them, so the loss is computed in fp32 regardless of the
        # autocast context the forward pass ran under.
        pred = pred.float()
        target = target.float()

        terms: dict[str, torch.Tensor] = {}
        weights = self.weights
        if weights["l1"] > 0:
            terms["l1"] = F.l1_loss(pred, target)
        if weights["mse"] > 0:
            terms["mse"] = F.mse_loss(pred, target)
        if weights["msssim"] > 0:
            terms["msssim"] = 1.0 - ms_ssim(
                pred, target, self.data_range, self.window_size, self.sigma
            )
        if weights["ssim"] > 0:
            terms["ssim"] = 1.0 - ssim(pred, target, self.data_range, self.window_size, self.sigma)
        if weights["sobel"] > 0:
            terms["sobel"] = sobel_loss(pred, target)

        total = sum(weights[name] * value for name, value in terms.items())
        parts = {name: float(value.detach()) for name, value in terms.items()}
        return total, parts


# ---------------------------------------------------------------------------
# corner detection  (brief §5)
# ---------------------------------------------------------------------------
def heatmap_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 1.0,
    positive_threshold: float = 0.1,
) -> torch.Tensor:
    """Pixel-wise squared error on the heatmaps, as the brief describes it.

    A Gaussian of sigma 3 covers around 60 of a 128x128 map's 16 384 cells, so
    **99.6% of the target is background** and "predict zero everywhere" is a
    respectable score under a plain mean. It is usually still enough — the squared
    error at the peak is large — but if a run flattens out at exactly that
    solution, ``positive_weight`` scales the error on the cells the blob occupies
    and pulls it back out. It is 1.0 by default, which is the plain pixel-wise
    loss and nothing else.
    """
    pred, target = pred.float(), target.float()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    squared = (pred - target) ** 2
    if positive_weight != 1.0:
        weights = torch.where(
            target > positive_threshold,
            torch.full_like(target, float(positive_weight)),
            torch.ones_like(target),
        )
        return (squared * weights).sum() / weights.sum()
    return squared.mean()


def wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    width: float = 10.0,
    curvature: float = 2.0,
    scale: float = 256.0,
) -> torch.Tensor:
    """The Wing loss on coordinates — logarithmic when close, L1 when far.

    ``w * ln(1 + |e| / eps)`` for errors under *width*, and ``|e| - C`` beyond it,
    with *C* chosen to make the two halves meet. The point is the gradient: L1
    treats a 40-pixel error and a 1-pixel error as equally urgent, so once the
    detector is roughly right there is nothing pushing it to be precisely right.
    The logarithmic region gives small errors a gradient that *grows* as the error
    shrinks, which is where the last pixel of accuracy comes from.

    *width* and *curvature* are in pixels of a ``scale``-wide input, because that
    is the space the published values (10 and 2) were chosen in and the space a
    reader thinks in. The result is divided by *scale* again on the way out, so a
    Wing term and an L1 term with the same weight are the same size — otherwise
    swapping one for the other would silently change the effective learning rate
    as well as the loss shape.
    """
    if width <= 0 or curvature <= 0:
        raise ValueError("wing loss needs a positive width and curvature")
    error = (pred.float() - target.float()).abs() * scale
    constant = width - width * math.log1p(width / curvature)
    curved = width * torch.log1p(error / curvature)
    return torch.where(error < width, curved, error - constant).mean() / scale


class CornerLoss(nn.Module):
    """The loss for both corner detectors *(brief §5)*.

    One class, because the two approaches differ in which terms carry a weight
    rather than in how the loss is assembled — and because that makes the
    comparison between them a config file instead of a second code path that
    could differ in some detail nobody notices.

    ``coord_l1`` / ``coord_l2`` / ``coord_wing``
        On the normalised coordinates. For approach A they are the whole loss.
    ``heatmap``
        Pixel-wise MSE on the Gaussian maps. Approach B's whole loss by default.
    Both at once
        A coordinate term on a heatmap model is applied to the **soft-argmax** of
        the predicted maps, so the gradient flows back through the extraction the
        detector will actually be read with at inference. That is the variant
        worth trying if the heatmaps look right and the coordinates come out
        slightly off: it optimises the number that gets reported rather than a
        proxy for it.

    ``forward`` returns ``(total, parts)`` like the restoration loss, with every
    term reported unweighted, and accepts either a bare tensor of labels or the
    ``{"corners": ..., "heatmaps": ...}`` mapping the corner datasets produce.

    **Every weight defaults to zero**, so a config that names a term it did not
    mean gets an empty loss and an immediate error rather than a silent default.
    That failure has happened once in this project already, in the generator's
    options, and it cost a set of quietly wrong evaluation samples.
    """

    def __init__(
        self,
        coord_l1: float = 0.0,
        coord_l2: float = 0.0,
        coord_wing: float = 0.0,
        heatmap: float = 0.0,
        wing_width: float = 10.0,
        wing_curvature: float = 2.0,
        coord_scale: float = 256.0,
        positive_weight: float = 1.0,
        soft_argmax_mode: str = "normalised",
        soft_argmax_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = {
            "coord_l1": float(coord_l1),
            "coord_l2": float(coord_l2),
            "coord_wing": float(coord_wing),
            "heatmap": float(heatmap),
        }
        if not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("every loss weight is zero — there is nothing to optimise")
        self.wing_width = float(wing_width)
        self.wing_curvature = float(wing_curvature)
        self.coord_scale = float(coord_scale)
        self.positive_weight = float(positive_weight)
        self.soft_argmax_mode = str(soft_argmax_mode)
        self.soft_argmax_beta = float(soft_argmax_beta)

    @property
    def active(self) -> list[str]:
        return [name for name, weight in self.weights.items() if weight > 0]

    @property
    def wants_coordinates(self) -> bool:
        return any(self.weights[name] > 0 for name in ("coord_l1", "coord_l2", "coord_wing"))

    def extra_repr(self) -> str:
        return " + ".join(f"{self.weights[name]:g}*{name}" for name in self.active)

    def forward(self, pred: torch.Tensor, target):
        from .model import soft_argmax2d

        labels = target if isinstance(target, dict) else None
        weights = self.weights
        terms: dict[str, torch.Tensor] = {}

        if pred.dim() == 4:  # heatmaps
            maps = labels["heatmaps"] if labels is not None else target
            if weights["heatmap"] > 0:
                terms["heatmap"] = heatmap_mse(pred, maps, self.positive_weight)
            coords = (
                soft_argmax2d(pred, mode=self.soft_argmax_mode, beta=self.soft_argmax_beta)
                if self.wants_coordinates
                else None
            )
        elif pred.dim() == 3:  # coordinates
            coords = pred
            if weights["heatmap"] > 0:
                raise ValueError("a coordinate model cannot be trained with a heatmap loss")
        else:
            raise ValueError(f"expected (B, K, 2) or (B, K, H, W), got {tuple(pred.shape)}")

        if coords is not None:
            true = labels["corners"] if labels is not None else target
            true = true.to(coords.dtype)
            if weights["coord_l1"] > 0:
                terms["coord_l1"] = F.l1_loss(coords, true)
            if weights["coord_l2"] > 0:
                terms["coord_l2"] = F.mse_loss(coords, true)
            if weights["coord_wing"] > 0:
                terms["coord_wing"] = wing_loss(
                    coords, true, self.wing_width, self.wing_curvature, self.coord_scale
                )

        total = sum(weights[name] * value for name, value in terms.items())
        parts = {name: float(value.detach()) for name, value in terms.items()}
        return total, parts


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
LOSSES = {
    "restoration": CombinedRestorationLoss,
    "corner": CornerLoss,
}


def build_loss(config):
    """The loss described by a config's ``loss:`` block.

    ``name`` picks the family — restoration by default, since that is the only
    one that existed when the key was introduced and every enhancement config
    omits it. Unknown keys are an error rather than a shrug: a weight silently
    ignored because it was misspelled produces a run that did not test what its
    filename says it tested.
    """
    settings = dict(config.get("loss") or {})
    name = str(settings.pop("name", "restoration")).lower()
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(LOSSES)}")
    try:
        return LOSSES[name](**settings)
    except TypeError as exc:
        raise TypeError(f"bad loss: settings for {name!r}: {exc}") from exc
