"""The losses and the metrics.

Both are written from scratch, which means both can be wrong in ways no shape
assertion catches. The tests here are mostly *identities* — an image compared
with itself, a known ordering between two comparisons — because those are what
pin a similarity measure down without hard-coding numbers that would only ever be
compared against this implementation's own output.
"""

import math

import pytest
import torch
from torch.nn import functional as F

from scandar.config import Config
from scandar.losses import (
    CombinedRestorationLoss,
    build_loss,
    gaussian_window,
    ms_ssim,
    sobel_edges,
    sobel_loss,
    ssim,
)
from scandar.metrics import MetricAccumulator, psnr, ssim_metric


@pytest.fixture(scope="module")
def image():
    """Something with structure in it — SSIM on uniform noise is not a test."""
    torch.manual_seed(0)
    canvas = torch.full((2, 3, 256, 256), 0.9)
    canvas[..., 40:60, 20:200] = 0.1
    canvas[..., 100:220, 90:110] = 0.2
    return canvas + 0.02 * torch.randn(2, 3, 256, 256)


# --- identities ------------------------------------------------------------
def test_an_image_is_identical_to_itself(image):
    assert float(ssim(image, image)) == pytest.approx(1.0, abs=1e-5)
    assert float(ms_ssim(image, image)) == pytest.approx(1.0, abs=1e-5)
    assert float(sobel_loss(image, image)) == 0.0
    assert math.isinf(float(psnr(image, image)))


def test_similarity_is_symmetric(image):
    other = image + 0.05 * torch.randn_like(image)
    assert float(ssim(image, other)) == pytest.approx(float(ssim(other, image)), abs=1e-6)


def test_more_noise_scores_worse(image):
    a = image + 0.02 * torch.randn_like(image)
    b = image + 0.20 * torch.randn_like(image)
    assert float(ssim(image, a)) > float(ssim(image, b))
    assert float(psnr(image, a)) > float(psnr(image, b))
    assert float(ms_ssim(image, a)) > float(ms_ssim(image, b))


def test_blur_is_punished(image):
    """The property the whole loss design rests on: a blurred restoration has to
    score worse than a sharp one, or nothing stops the network smearing."""
    blurred = F.avg_pool2d(image, 7, 1, 3)
    assert float(ssim(image, blurred)) < 1.0
    assert float(sobel_loss(image, blurred)) > 0.0


def test_psnr_matches_its_definition():
    a = torch.zeros(1, 3, 8, 8)
    b = torch.full((1, 3, 8, 8), 0.1)
    assert float(psnr(a, b)) == pytest.approx(20.0, abs=1e-4)  # mse 0.01 -> 20 dB


def test_the_gaussian_window_is_a_normalised_gaussian():
    window = gaussian_window(11, 1.5, channels=3)
    assert window.shape == (3, 1, 11, 11)
    assert float(window[0].sum()) == pytest.approx(1.0, abs=1e-6)
    assert int(window[0, 0].argmax()) == 11 * 5 + 5  # peaks in the middle


def test_sobel_finds_an_edge_where_the_edge_is():
    """Channel 0 is the x-derivative and channel 1 the y-derivative, and mixing
    the two up would be invisible in the loss — it is symmetric in them."""
    vertical_edge = torch.zeros(1, 1, 16, 16)
    vertical_edge[:, :, :, 8:] = 1.0  # left half dark, right half light
    d_x, d_y = sobel_edges(vertical_edge)[:, 0], sobel_edges(vertical_edge)[:, 1]
    assert float(d_x.abs().max()) > float(d_y.abs().max())

    horizontal_edge = vertical_edge.transpose(-1, -2)
    d_x, d_y = sobel_edges(horizontal_edge)[:, 0], sobel_edges(horizontal_edge)[:, 1]
    assert float(d_y.abs().max()) > float(d_x.abs().max())


# --- reductions ------------------------------------------------------------
def test_per_image_scores_come_back_one_per_image(image):
    other = image + 0.05 * torch.randn_like(image)
    for metric in (ssim, ms_ssim):
        assert metric(image, other, reduction="none").shape == (2,)
    assert psnr(image, other, reduction="none").shape == (2,)
    assert ssim_metric(image, other, reduction="none").shape == (2,)


def test_a_single_image_without_a_batch_axis_is_accepted(image):
    assert psnr(image[0], image[0] * 0.9).dim() == 0


def test_ms_ssim_falls_back_to_fewer_scales_on_a_small_image():
    """Five scales need 161 pixels. Refusing would make the tests unable to use
    small images; silently changing the answer would make two runs incomparable,
    so the weights are renormalised over the scales that fit."""
    small = torch.rand(1, 3, 32, 32)
    assert float(ms_ssim(small, small)) == pytest.approx(1.0, abs=1e-5)


def test_an_image_smaller_than_the_window_is_refused():
    with pytest.raises(ValueError, match="smaller than"):
        ssim(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))


def test_mismatched_shapes_are_refused(image):
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim(image, image[..., :128, :])


# --- the combined loss -----------------------------------------------------
def test_the_combined_loss_is_zero_for_a_perfect_restoration(image):
    total, parts = CombinedRestorationLoss()(image, image)
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert set(parts) == {"l1", "msssim", "sobel"}


def test_only_the_terms_with_a_weight_are_computed(image):
    _, parts = CombinedRestorationLoss(l1=1.0, msssim=0.0, sobel=0.0)(image, image * 0.9)
    assert set(parts) == {"l1"}


def test_the_reported_terms_are_unweighted(image):
    """So that a weight change does not look like progress in the log."""
    target = image * 0.9
    _, plain = CombinedRestorationLoss(l1=1.0, msssim=0.0, sobel=0.0)(image, target)
    _, scaled = CombinedRestorationLoss(l1=7.0, msssim=0.0, sobel=0.0)(image, target)
    assert plain["l1"] == pytest.approx(scaled["l1"])


def test_the_total_is_the_weighted_sum(image):
    target = image + 0.1 * torch.randn_like(image)
    loss = CombinedRestorationLoss(l1=1.0, msssim=0.5, sobel=0.25)
    total, parts = loss(image, target)
    expected = parts["l1"] + 0.5 * parts["msssim"] + 0.25 * parts["sobel"]
    assert float(total) == pytest.approx(expected, rel=1e-5)


def test_a_loss_with_no_weights_at_all_is_refused():
    with pytest.raises(ValueError, match="nothing to optimise"):
        CombinedRestorationLoss(l1=0.0, mse=0.0, msssim=0.0, ssim=0.0, sobel=0.0)


def test_build_loss_reads_a_config():
    loss = build_loss(Config({"loss": {"l1": 0.0, "mse": 1.0, "msssim": 0.0, "sobel": 0.0}}))
    assert loss.active == ["mse"]


def test_the_loss_is_differentiable(image):
    prediction = image.clone().requires_grad_(True)
    total, _ = CombinedRestorationLoss()(prediction, image * 0.8)
    total.backward()
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0


def test_the_loss_survives_half_precision_inputs(image):
    """The forward pass runs under autocast on CUDA; the SSIM variance products
    do not survive fp16, so the loss casts back up on its own."""
    total, _ = CombinedRestorationLoss()(image.half(), image.half() * 0.9)
    assert torch.isfinite(total)


# --- the accumulator -------------------------------------------------------
def test_the_accumulator_keeps_per_image_scores():
    accumulator = MetricAccumulator()
    accumulator.add("psnr", torch.tensor([10.0, 20.0, 30.0]))
    assert accumulator.mean("psnr") == pytest.approx(20.0)
    assert accumulator.std("psnr") == pytest.approx(math.sqrt(200 / 3))
    assert accumulator.summary()["psnr"] == pytest.approx(20.0)


def test_the_accumulator_ignores_infinities():
    """An identical pair scores an infinite PSNR, which would otherwise poison
    the mean of a whole evaluation set."""
    accumulator = MetricAccumulator()
    accumulator.add("psnr", [10.0, float("inf"), 30.0])
    assert accumulator.mean("psnr") == pytest.approx(20.0)
