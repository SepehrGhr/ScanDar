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
    CornerLoss,
    build_loss,
    gaussian_window,
    heatmap_mse,
    ms_ssim,
    sobel_edges,
    sobel_loss,
    ssim,
    wing_loss,
)
from scandar.metrics import (
    MetricAccumulator,
    corner_error,
    corner_metrics,
    pck,
    pck_curve,
    psnr,
    quad_iou_batch,
    ssim_metric,
)


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


# ---------------------------------------------------------------------------
# corner detection: the losses  (brief §5)
# ---------------------------------------------------------------------------
@pytest.fixture
def quads():
    """Two page outlines, and a prediction that is off by a known amount."""
    truth = torch.tensor(
        [
            [[0.15, 0.20], [0.80, 0.12], [0.88, 0.85], [0.10, 0.90]],
            [[0.30, 0.35], [0.70, 0.30], [0.75, 0.70], [0.25, 0.65]],
        ]
    )
    return truth + torch.tensor([0.01, -0.005]), truth


def test_the_wing_loss_is_zero_on_a_perfect_prediction(quads):
    _, truth = quads
    assert float(wing_loss(truth, truth)) == pytest.approx(0.0)


def test_the_wing_loss_amplifies_the_gradient_of_a_small_error():
    """Its entire purpose. L1 pushes as hard on a nearly-right answer as on a
    hopeless one, so nothing drives the last pixel of accuracy."""
    gradients = {}
    for error in (0.5 / 256, 30.0 / 256):
        prediction = torch.tensor([[[error, 0.0]]], requires_grad=True)
        wing_loss(prediction, torch.zeros(1, 1, 2)).backward()
        gradients[error] = float(prediction.grad[0, 0, 0])
    assert gradients[0.5 / 256] > 3 * gradients[30.0 / 256]


def test_the_wing_loss_stays_on_l1_footing(quads):
    """Scaled back down on the way out, so swapping it in for an L1 does not
    silently change the effective learning rate as well as the loss shape."""
    prediction, truth = quads
    assert 0.2 < float(wing_loss(prediction, truth)) / float(F.l1_loss(prediction, truth)) < 10


def test_the_wing_loss_refuses_a_degenerate_shape():
    with pytest.raises(ValueError, match="positive width"):
        wing_loss(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), width=0.0)


def test_heatmap_mse_is_zero_on_itself():
    maps = torch.rand(2, 4, 32, 32)
    assert float(heatmap_mse(maps, maps)) == 0.0


def test_weighting_the_blob_changes_what_an_empty_prediction_costs():
    """A Gaussian covers well under a percent of its map, so "predict nothing"
    scores respectably under a plain mean. The knob exists for the run where that
    turns out to be a local minimum rather than a curiosity."""
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 14:18, 14:18] = 1.0
    empty = torch.zeros_like(target)
    assert heatmap_mse(empty, target, positive_weight=20.0) > 5 * heatmap_mse(empty, target)


# --- the loss that serves both approaches ---------------------------------
def test_the_corner_loss_reports_every_term_unweighted(quads):
    prediction, truth = quads
    loss = CornerLoss(coord_l1=1.0, coord_wing=0.5)
    total, parts = loss(prediction, truth)
    assert set(parts) == {"coord_l1", "coord_wing"}
    assert float(total) == pytest.approx(parts["coord_l1"] + 0.5 * parts["coord_wing"], rel=1e-5)
    assert loss.active == ["coord_l1", "coord_wing"]


def test_the_corner_loss_takes_labels_either_way(quads):
    """A coordinate model is handed a tensor and a heatmap model a mapping; the
    trainer should not have to care which."""
    prediction, truth = quads
    loss = CornerLoss(coord_l1=1.0)
    bare, _ = loss(prediction, truth)
    mapped, _ = loss(prediction, {"corners": truth, "heatmaps": torch.zeros(2, 4, 8, 8)})
    assert float(bare) == pytest.approx(float(mapped))


def test_an_auxiliary_coordinate_loss_reaches_the_heatmaps():
    """The variant that optimises the number actually reported: the gradient has
    to travel back through the soft-argmax into the maps themselves."""
    maps = torch.rand(1, 4, 16, 16, requires_grad=True)
    corners = torch.tensor([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]])
    total, parts = CornerLoss(heatmap=1.0, coord_l1=0.01)(
        maps, {"corners": corners, "heatmaps": torch.zeros(1, 4, 16, 16)}
    )
    total.backward()
    assert set(parts) == {"heatmap", "coord_l1"}
    assert maps.grad is not None and float(maps.grad.abs().sum()) > 0


def test_a_heatmap_loss_on_a_coordinate_model_is_refused(quads):
    prediction, truth = quads
    with pytest.raises(ValueError, match="cannot be trained with a heatmap loss"):
        CornerLoss(heatmap=1.0)(prediction, truth)


def test_a_corner_loss_with_no_weights_is_refused():
    """Every weight defaults to zero on purpose, so a misspelled key in a config
    produces an error rather than a silently different experiment."""
    with pytest.raises(ValueError, match="nothing to optimise"):
        CornerLoss()


def test_build_loss_picks_the_family_by_name():
    corner = build_loss(Config({"loss": {"name": "corner", "heatmap": 1.0}}))
    assert isinstance(corner, CornerLoss) and corner.active == ["heatmap"]
    assert isinstance(build_loss(Config({"loss": {"l1": 1.0}})), CombinedRestorationLoss)
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss(Config({"loss": {"name": "triplet"}}))


# ---------------------------------------------------------------------------
# corner detection: the metrics  (brief §5)
# ---------------------------------------------------------------------------
def test_the_corner_error_is_zero_on_the_truth(quads):
    _, truth = quads
    assert float(corner_error(truth, truth)) == 0.0
    assert float(quad_iou_batch(truth, truth).min()) == pytest.approx(1.0, abs=1e-3)


def test_the_corner_error_is_measured_in_the_space_it_is_told_about():
    """Normalised coordinates carry no unit of their own, so the space has to
    come from outside — and a pixel count quoted without it means nothing."""
    truth = torch.zeros(1, 4, 2)
    predicted = torch.full((1, 4, 2), 0.1)
    at_256 = float(corner_error(predicted, truth, size=256))
    at_512 = float(corner_error(predicted, truth, size=512))
    assert at_256 == pytest.approx(0.1 * 256 * math.sqrt(2))
    assert at_512 == pytest.approx(2 * at_256)


def test_the_success_metric_needs_all_four_corners():
    """Three good corners and one bad one is a bent page, and a mean over the
    four would hide exactly that."""
    errors = torch.tensor([[1.0, 1.0, 1.0, 40.0], [1.0, 2.0, 1.5, 1.0]])
    assert float(pck(errors, threshold=5.0)) == pytest.approx(0.5)
    assert pck(errors, threshold=5.0, reduction="none").tolist() == [0.0, 1.0]


def test_the_pck_curve_only_climbs():
    truth = torch.rand(8, 4, 2) * 0.6 + 0.2
    predicted = truth + torch.randn(8, 4, 2) * 0.01
    curve = [row["pck"] for row in pck_curve(predicted, truth)]
    assert curve == sorted(curve)
    assert curve[-1] == 1.0


def test_the_corner_metrics_arrive_per_image(quads):
    prediction, truth = quads
    metrics = corner_metrics(prediction, truth, size=256)
    assert set(metrics) == {"corner_err", "corner_pct", "corner_worst", "pck", "quad_iou"}
    assert all(value.shape == (2,) for value in metrics.values())
    assert float(metrics["corner_worst"][0]) >= float(metrics["corner_err"][0])
