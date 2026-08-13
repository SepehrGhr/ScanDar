"""Evaluation metrics.  *(brief §3.3 and §5)*

Restoration:

``psnr`` / ``ssim_metric``
    Reported on all three synthetic buckets, alongside the "do nothing" baseline —
    the metrics of the degraded input itself. If the model's scores are not
    clearly above that line, it is not earning its parameters.
``MetricAccumulator``
    Collects **per-image** scores so the table can carry a mean *and* a standard
    deviation. Averaging batch averages would silently weight a short last batch
    the same as a full one, and would make the standard deviation meaningless.

Corner detection *(brief §5)*:

``corner_error``
    Mean Euclidean distance between predicted and true corners — the brief's own
    headline metric. Reported in pixels of the network's 256x256 input **and** as
    a percentage of that space's diagonal, because a pixel count means nothing
    without the size it was measured at and the percentage survives any resize.
``pck``
    The stricter success metric the brief asks for alongside it: the fraction of
    images where **all four** corners land within a threshold. Three good corners
    and one bad one is a failed rectification, and a mean over corners hides
    exactly that; swept over a range of thresholds it becomes a curve, which is
    the honest way to compare two detectors without picking the threshold that
    flatters one of them.
``quad_iou_batch``
    Overlap between the predicted page and the true one, wrapping
    :func:`~scandar.geometry.quad_iou`. Corner error says how far the prediction
    is; this says how much of the page it would actually rectify, which is what
    the enhancement stage downstream cares about.

The SSIM used here is the one in :mod:`scandar.losses`, not a second
implementation of the same formula. Two copies of a metric drift, and the day
they disagree is the day the reported number stops matching the trained
objective. Both are hand-written; neither is imported from a library.
"""

from __future__ import annotations

import math

import torch

from .losses import ssim as _ssim

__all__ = [
    "psnr",
    "ssim_metric",
    "corner_errors_px",
    "corner_error",
    "pck",
    "pck_curve",
    "quad_iou_batch",
    "corner_metrics",
    "MetricAccumulator",
    "summarise",
]

#: Thresholds for the PCK curve, as a percentage of the image diagonal. Spanning
#: two orders of magnitude on purpose: the low end separates two good detectors
#: from each other, the high end separates a detector that is merely imprecise
#: from one that has lost the page.
PCK_THRESHOLDS_PCT = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0)

#: The threshold quoted as *the* success rate when only one number fits. 2% of a
#: 362-pixel diagonal is 7.2 pixels at 256, which is about a millimetre on an A4
#: page — small enough that the rectification is visually exact.
PCK_THRESHOLD_PCT = 2.0


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Peak signal-to-noise ratio in decibels, computed per image.

    Per image, and not over the whole batch at once: PSNR is a logarithm of a
    mean, so the average of per-image scores is not the score of the pooled
    error, and it is the former that every paper reports.

    Identical images give an infinite ratio, which is arithmetically right and
    also the reason no evaluation set should contain a pair the model can copy.
    """
    pred, target = pred.float(), target.float()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.dim() == 3:
        pred, target = pred[None], target[None]

    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    per_image = 10.0 * torch.log10(data_range**2 / mse)
    return per_image.mean() if reduction == "mean" else per_image


def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Structural similarity, per image — the same estimator the loss uses."""
    if pred.dim() == 3:
        pred, target = pred[None], target[None]
    return _ssim(pred, target, data_range=data_range, reduction=reduction)


# ---------------------------------------------------------------------------
# corner detection  (brief §5)
# ---------------------------------------------------------------------------
def _pixel_size(size, batch: int, device, dtype) -> torch.Tensor:
    """``(B, 2)`` of (width, height) from a side length or per-sample sizes."""
    if isinstance(size, torch.Tensor):
        sizes = size.to(device=device, dtype=dtype)
    elif isinstance(size, (int, float)):
        sizes = torch.tensor([[float(size), float(size)]], device=device, dtype=dtype)
    else:
        sizes = torch.as_tensor(size, device=device, dtype=dtype)
    if sizes.dim() == 1:
        sizes = sizes[None]
    if sizes.shape[0] == 1:
        sizes = sizes.expand(batch, 2)
    return sizes


def corner_errors_px(predicted: torch.Tensor, target: torch.Tensor, size=256) -> torch.Tensor:
    """Per-corner Euclidean error in pixels. ``(B, K, 2)`` normalised -> ``(B, K)``.

    *size* is the space the error is measured in: a side length for the square
    input the detector sees, or a ``(B, 2)`` tensor of the original photo's
    ``(width, height)`` to measure in the photo's own pixels instead. The first is
    what the comparison table uses, because it is the same space for every sample
    and therefore the only one two detectors can be compared in.
    """
    predicted, target = predicted.float(), target.float()
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(predicted.shape)} vs {tuple(target.shape)}")
    if predicted.dim() == 2:
        predicted, target = predicted[None], target[None]
    sizes = _pixel_size(size, predicted.shape[0], predicted.device, predicted.dtype)
    delta = (predicted - target) * sizes[:, None, :]
    return delta.norm(dim=-1)


def corner_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    size=256,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean corner localisation error per image, in pixels *(brief §5)*.

    Averaged over the four corners of an image first, then over images — the
    brief's "average Euclidean distance between predicted and true corners".
    ``reduction="none"`` keeps the per-image scores, which is what an error bar,
    a histogram and a failure list are all made of.
    """
    per_image = corner_errors_px(predicted, target, size).mean(dim=-1)
    return per_image.mean() if reduction == "mean" else per_image


def _diagonal(size, batch: int, device, dtype) -> torch.Tensor:
    sizes = _pixel_size(size, batch, device, dtype)
    return sizes.pow(2).sum(dim=-1).sqrt()


def pck(errors: torch.Tensor, threshold: float, reduction: str = "mean") -> torch.Tensor:
    """Fraction of images whose **worst** corner is within *threshold* pixels.

    All four, deliberately. A page rectified from three good corners and one bad
    one is a bent page, so the metric that decides whether a detection succeeded
    has to be decided by the worst corner rather than by the average of them.
    """
    if errors.dim() == 1:
        errors = errors[None]
    hit = (errors.max(dim=-1).values <= float(threshold)).float()
    return hit.mean() if reduction == "mean" else hit


def pck_curve(
    predicted: torch.Tensor,
    target: torch.Tensor,
    size=256,
    thresholds_pct=PCK_THRESHOLDS_PCT,
) -> list[dict]:
    """The success rate swept over thresholds — one row per threshold.

    Reported in both units at once: the threshold as a percentage of the image
    diagonal, and the same threshold in pixels, so the curve can be read by
    someone who thinks in either.
    """
    errors = corner_errors_px(predicted, target, size)
    diagonal = float(_diagonal(size, errors.shape[0], errors.device, errors.dtype).mean())
    rows = []
    for percent in thresholds_pct:
        pixels = diagonal * float(percent) / 100.0
        rows.append(
            {
                "threshold_pct": float(percent),
                "threshold_px": round(pixels, 3),
                "pck": round(float(pck(errors, pixels)), 4),
            }
        )
    return rows


def quad_iou_batch(predicted: torch.Tensor, target: torch.Tensor, size=256) -> torch.Tensor:
    """Per-image intersection over union of the predicted and true page outlines.

    Rasterised one sample at a time by :func:`~scandar.geometry.quad_iou`, on
    denormalised coordinates — the measure is a ratio, so any consistent pixel
    space gives the same answer.
    """
    from .geometry import quad_iou

    if predicted.dim() == 2:
        predicted, target = predicted[None], target[None]
    sizes = _pixel_size(size, predicted.shape[0], predicted.device, torch.float32)
    scaled_pred = (predicted.detach().float() * sizes[:, None, :]).cpu().numpy()
    scaled_true = (target.detach().float() * sizes[:, None, :]).cpu().numpy()
    values = [quad_iou(a, b) for a, b in zip(scaled_pred, scaled_true)]
    return torch.tensor(values, dtype=torch.float32)


def corner_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    size=256,
    threshold_pct: float = PCK_THRESHOLD_PCT,
    with_iou: bool = True,
) -> dict[str, torch.Tensor]:
    """Every corner metric at once, per image, ready for the accumulator.

    Per image rather than per batch, because a mean of batch means is not a mean,
    and because the table wants a standard deviation and the failure gallery wants
    to know which samples were the bad ones.
    """
    per_corner = corner_errors_px(predicted, target, size)
    batch = per_corner.shape[0]
    diagonal = _diagonal(size, batch, per_corner.device, per_corner.dtype)
    per_image = per_corner.mean(dim=-1)

    # Each sample against its *own* diagonal. Identical to a shared threshold when
    # every sample is the same square, and correct when they are not.
    limit = diagonal * float(threshold_pct) / 100.0
    metrics = {
        "corner_err": per_image,
        "corner_pct": 100.0 * per_image / diagonal,
        "corner_worst": per_corner.max(dim=-1).values,
        "pck": (per_corner.max(dim=-1).values <= limit).float(),
    }
    if with_iou:
        metrics["quad_iou"] = quad_iou_batch(predicted, target, size)
    return metrics


class MetricAccumulator:
    """Running mean and standard deviation over per-image scores.

    Values are kept, not just summed. A few thousand floats cost nothing and they
    are what a histogram or a per-sample failure list is made of later.
    """

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, values) -> None:
        bucket = self.values.setdefault(name, [])
        if isinstance(values, torch.Tensor):
            bucket.extend(values.detach().flatten().tolist())
        elif isinstance(values, (list, tuple)):
            bucket.extend(float(v) for v in values)
        else:
            bucket.append(float(values))

    def update(self, mapping: dict) -> None:
        for name, values in mapping.items():
            self.add(name, values)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __len__(self) -> int:
        return max((len(v) for v in self.values.values()), default=0)

    def mean(self, name: str) -> float:
        values = self.values.get(name) or []
        finite = [v for v in values if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    def std(self, name: str) -> float:
        """Population standard deviation, which is what "mean ± std" means here."""
        values = [v for v in self.values.get(name, []) if math.isfinite(v)]
        if len(values) < 2:
            return float("nan")
        average = sum(values) / len(values)
        return math.sqrt(sum((v - average) ** 2 for v in values) / len(values))

    def summary(self) -> dict[str, float]:
        """``{"psnr": mean, "psnr_std": std, ...}``, ready for a CSV row."""
        out: dict[str, float] = {}
        for name in self.values:
            out[name] = self.mean(name)
            out[f"{name}_std"] = self.std(name)
        return out


def summarise(accumulator: MetricAccumulator, keys=None) -> str:
    """One readable line for the training log."""
    keys = list(keys or accumulator.values)
    return "  ".join(f"{key} {accumulator.mean(key):.4f}" for key in keys if key in accumulator)
