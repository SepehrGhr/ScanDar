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

The SSIM used here is the one in :mod:`scandar.losses`, not a second
implementation of the same formula. Two copies of a metric drift, and the day
they disagree is the day the reported number stops matching the trained
objective. Both are hand-written; neither is imported from a library.

Not built yet: the corner-detection metrics. Mean Euclidean corner error in
pixels and as a percentage of the image diagonal, the stricter
all-four-within-a-threshold success rate swept into a curve, and quad IoU as the
proxy for what a corner error costs the rectification downstream. The building
block for the last one already exists as
:func:`~scandar.geometry.quad_iou`; the rest arrive with the detectors.
"""

from __future__ import annotations

import math

import torch

from .losses import ssim as _ssim

__all__ = ["psnr", "ssim_metric", "MetricAccumulator", "summarise"]


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
