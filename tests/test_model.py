"""The architecture's contract.

The tests here are about the properties the rest of the project leans on: the
enhancement network is **fully convolutional**, so it can train on 256x256
patches and restore a whole page, and its output is a valid image at whatever
size went in. Both are the kind of thing that breaks silently — a shape mismatch
at 1024x1448 would only surface on the day someone tried to enhance a real page.

The corner detectors add one more: reading a coordinate back off a heatmap has to
be *right*, not merely plausible. An extraction that cannot recover the corner a
Gaussian was drawn around puts a floor under approach B's accuracy that has
nothing to do with how well the model was trained, and nothing in a loss curve
would ever say so.
"""

import numpy as np
import pytest
import torch

from scandar.config import Config
from scandar.model import (
    ConvBlock,
    CornerHeatNet,
    CornerRegNet,
    DocUNet,
    build_model,
    clamp_image,
    corners_from_output,
    count_parameters,
    heatmap_peaks,
    soft_argmax2d,
)


@pytest.fixture(scope="module")
def model():
    net = DocUNet(base=8, depth=3)
    net.eval()
    return net


# --- shapes ----------------------------------------------------------------
@pytest.mark.parametrize("size", [(64, 64), (96, 128), (57, 91), (8, 8)])
def test_output_matches_input_size(model, size):
    """Including sizes that are not multiples of the downsampling factor.

    1024x1448 — the page size this project rectifies to — is not a multiple of
    16, and neither is whatever a grader hands the model on the day.
    """
    with torch.no_grad():
        out = model(torch.rand(1, 3, *size))
    assert out.shape == (1, 3, *size)


def test_output_is_a_valid_image(model):
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 64))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_a_page_sized_input_works():
    """The proof that patch training and whole-page inference are the same model."""
    net = DocUNet(base=4, depth=4).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 256, 362))
    assert out.shape == (1, 3, 256, 362)


def test_batch_entries_are_independent(model):
    """In eval mode, since BatchNorm deliberately couples them while training."""
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        together = model(torch.cat([a, b]))
        alone = model(a)
    assert torch.allclose(together[:1], alone, atol=1e-5)


# --- the heads -------------------------------------------------------------
def test_the_residual_head_starts_near_the_identity():
    """Its whole point: begin at "change nothing", not at "invent a page"."""
    net = DocUNet(base=8, depth=2, residual_output=True).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = net(image)
    assert float((out - image).abs().max()) < 0.1


def test_clamp_image_only_bites_outside_the_range():
    inside = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(clamp_image(inside), inside)
    assert torch.equal(clamp_image(torch.tensor([-0.3, 1.4])), torch.tensor([0.0, 1.0]))


# --- skip connections ------------------------------------------------------
def test_dropping_the_skips_drops_the_decoder_inputs():
    """The ablation has to be a real architectural change, not a flag that lies."""
    with_skips = count_parameters(DocUNet(base=8, depth=3, skips=True))
    without = count_parameters(DocUNet(base=8, depth=3, skips=False))
    assert without < with_skips
    with torch.no_grad():
        out = DocUNet(base=8, depth=3, skips=False).eval()(torch.rand(1, 3, 64, 64))
    assert out.shape == (1, 3, 64, 64)


# --- construction ----------------------------------------------------------
def test_conv_block_drops_the_bias_when_a_norm_follows():
    """A bias immediately before a normalisation layer is subtracted straight
    back out — free parameters that do nothing but slow the run down."""
    convs = [m for m in ConvBlock(3, 8, norm="batch").modules() if isinstance(m, torch.nn.Conv2d)]
    assert all(conv.bias is None for conv in convs)
    convs = [m for m in ConvBlock(3, 8, norm="none").modules() if isinstance(m, torch.nn.Conv2d)]
    assert all(conv.bias is not None for conv in convs)


def test_dropout_is_off_unless_it_is_asked_for():
    """The brief forbids it in the first version of every model, and the later
    study is only meaningful if dropout is the single thing that changed."""
    plain = [m for m in DocUNet(base=4, depth=2).modules() if isinstance(m, torch.nn.Dropout2d)]
    assert plain == []
    regularised = [
        m for m in DocUNet(base=4, depth=2, dropout=0.2).modules()
        if isinstance(m, torch.nn.Dropout2d)
    ]
    assert regularised and all(m.p == 0.2 for m in regularised)


def test_build_model_reads_a_config():
    config = Config({"model": {"name": "docunet", "base": 8, "depth": 2}})
    net = build_model(config)
    assert isinstance(net, DocUNet) and net.base == 8 and net.depth == 2


def test_build_model_rejects_what_it_does_not_understand():
    with pytest.raises(ValueError, match="unknown model"):
        build_model(Config({"model": {"name": "resnet"}}))
    with pytest.raises(TypeError):
        build_model(Config({"model": {"name": "docunet", "widht": 32}}))


def test_the_default_size_is_the_one_the_report_quotes():
    """~7.8 M parameters, in fp32 a 31 MB checkpoint. If this changes, the
    architecture changed, and every trained checkpoint stops loading."""
    assert 7.5e6 < count_parameters(DocUNet()) < 8.0e6


# ---------------------------------------------------------------------------
# corner detection, approach A: direct coordinate regression
# ---------------------------------------------------------------------------
def test_the_regressor_emits_four_corners_inside_the_frame():
    net = CornerRegNet(base=8, stages=3, grid=4, hidden=32).eval()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 4, 2)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_the_regressor_starts_at_an_average_page():
    """Its bias is initialised at a plausible quad, so an untrained network
    answers "a page, roughly centred" rather than answering noise. The residual
    head of the enhancement network starts at the identity for the same reason."""
    from scandar.model import PRIOR_QUAD

    net = CornerRegNet(base=8, stages=3, grid=4, hidden=32).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 64, 64))[0]
    assert torch.allclose(out, torch.tensor(PRIOR_QUAD), atol=0.02)


def test_the_regressor_can_drop_the_sigmoid_for_the_ablation():
    net = CornerRegNet(base=8, stages=2, grid=4, hidden=16, output_activation="none").eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 4, 2)
    with pytest.raises(ValueError, match="output_activation"):
        CornerRegNet(output_activation="softmax")


def test_the_regressor_flattens_rather_than_pools():
    """The decision the class exists to defend: a coordinate is made of *where*
    an activation fired, and global pooling is the operation that discards that.
    A wider grid must therefore mean a bigger head, not the same one."""
    small = count_parameters(CornerRegNet(base=8, stages=3, grid=2, hidden=32))
    large = count_parameters(CornerRegNet(base=8, stages=3, grid=8, hidden=32))
    assert large > small * 2


def test_dropout_in_the_fully_connected_layers_is_opt_in():
    """The classic place for it (brief §6), and still empty until that study."""
    plain = CornerRegNet(base=4, stages=2, grid=2, hidden=8)
    assert not [m for m in plain.modules() if isinstance(m, torch.nn.Dropout)]
    regularised = CornerRegNet(base=4, stages=2, grid=2, hidden=8, fc_dropout=0.3)
    assert [m for m in regularised.modules() if isinstance(m, torch.nn.Dropout)]


# ---------------------------------------------------------------------------
# corner detection, approach B: heatmap regression
# ---------------------------------------------------------------------------
def test_the_heatmap_net_emits_one_map_per_corner_at_half_resolution():
    net = CornerHeatNet(base=8, depth=3).eval()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 4, 32, 32)


@pytest.mark.parametrize("stride,expected", [(1, 64), (2, 32), (4, 16)])
def test_the_output_stride_decides_the_map_size(stride, expected):
    net = CornerHeatNet(base=4, depth=3, out_stride=stride).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 64, 64))
    assert out.shape[-2:] == (expected, expected)


def test_the_heatmap_net_takes_a_size_that_is_not_a_multiple_of_its_depth():
    """A grader's photo is whatever size their phone made it."""
    net = CornerHeatNet(base=4, depth=3).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 70, 100))
    assert out.shape == (1, 4, 35, 50)


def test_an_impossible_output_stride_is_refused():
    with pytest.raises(ValueError, match="power of two"):
        CornerHeatNet(out_stride=3)
    with pytest.raises(ValueError, match="encoder levels"):
        CornerHeatNet(depth=2, out_stride=8)


def test_the_heatmap_head_is_linear_unless_asked_otherwise():
    """Targets are Gaussians in [0, 1], but MSE against a saturating output learns
    slowly exactly where it matters, so the default head is not squashed."""
    torch.manual_seed(0)
    linear = CornerHeatNet(base=4, depth=2).eval()
    squashed = CornerHeatNet(base=4, depth=2, head_activation="sigmoid").eval()
    with torch.no_grad():
        assert float(squashed(torch.rand(1, 3, 32, 32)).min()) >= 0.0
        # A linear head is free to answer outside [0, 1]; only its initialisation
        # keeps it near zero, so this asserts the absence of a clamp, not a value.
        assert linear.head_activation == "none"


# ---------------------------------------------------------------------------
# reading a coordinate off a heatmap
# ---------------------------------------------------------------------------
def _blob(size, x, y, sigma=3.0, amplitude=1.0):
    """One Gaussian in a ``(size, size)`` map, centred on normalised (x, y)."""
    grid = np.arange(size, dtype=np.float32) + 0.5
    dx = (grid[None, :] - x * size) ** 2
    dy = (grid[:, None] - y * size) ** 2
    return amplitude * np.exp(-(dx + dy) / (2 * sigma**2))


def test_the_soft_argmax_recovers_the_corner_its_gaussian_was_drawn_around():
    """The property everything downstream rests on: the extraction must invert
    the drawing. Sub-pixel, because the label was placed sub-pixel."""
    wanted = [(0.3, 0.4), (0.62, 0.18), (0.75, 0.81), (0.5, 0.5)]
    maps = torch.from_numpy(np.stack([_blob(128, x, y) for x, y in wanted]))[None]
    got = soft_argmax2d(maps)[0]
    assert float((got - torch.tensor(wanted)).abs().max()) * 256 < 0.3


def test_the_soft_argmax_is_differentiable():
    """Which is the whole reason it exists — a coordinate loss on a heatmap model,
    and the bonus part's end-to-end fine-tuning, both need the gradient."""
    maps = torch.from_numpy(_blob(32, 0.4, 0.6))[None, None].clone().requires_grad_(True)
    soft_argmax2d(maps).sum().backward()
    assert maps.grad is not None and float(maps.grad.abs().sum()) > 0


def test_a_window_keeps_a_second_page_from_dragging_the_answer():
    """The generator puts a distractor sheet in a fifth of its corner samples, so
    a bimodal map is not hypothetical. A global centre of mass over two blobs
    lands between them, which is on neither page."""
    two = _blob(64, 0.25, 0.25) + _blob(64, 0.75, 0.75, amplitude=0.7)
    maps = torch.from_numpy(two)[None, None]
    global_guess = soft_argmax2d(maps)[0, 0]
    windowed = soft_argmax2d(maps, window=11)[0, 0]
    assert float(global_guess[0]) > 0.35  # dragged toward the second blob
    assert torch.allclose(windowed, torch.tensor([0.25, 0.25]), atol=0.01)


def test_a_map_with_nothing_in_it_answers_the_middle():
    """Rather than dividing by zero and returning the top-left cell as if it were
    an answer."""
    got = soft_argmax2d(torch.zeros(1, 4, 16, 16))
    assert torch.allclose(got, torch.full((1, 4, 2), 0.5))


def test_the_softmax_mode_needs_a_temperature_to_be_sharp():
    """Documented behaviour, not a bug: at beta = 1 a map whose peak is 1.0 is
    nearly flat under a softmax, so the estimate is pulled toward the centre."""
    maps = torch.from_numpy(_blob(64, 0.2, 0.2))[None, None]
    lukewarm = soft_argmax2d(maps, mode="softmax", beta=1.0)[0, 0]
    sharp = soft_argmax2d(maps, mode="softmax", beta=200.0)[0, 0]
    assert float(lukewarm[0]) > float(sharp[0]) > 0.19
    assert torch.allclose(sharp, torch.tensor([0.2, 0.2]), atol=0.01)


def test_hard_peaks_are_quantised_to_the_grid_and_report_their_height():
    maps = torch.from_numpy(_blob(16, 0.51, 0.51, sigma=1.0, amplitude=0.8))[None, None]
    coords, values = heatmap_peaks(maps)
    assert coords.shape == (1, 1, 2) and values.shape == (1, 1)
    # The height reported is the brightest *cell*, which sits below the blob's
    # true amplitude of 0.8 precisely because the grid does not sample its centre.
    assert float(values[0, 0]) == pytest.approx(float(maps.max()))
    assert 0.6 < float(values[0, 0]) < 0.8
    # Cell centres are (i + 0.5) / n, so nothing lands between them.
    assert float(coords[0, 0, 0] * 16 - 0.5) == pytest.approx(round(0.51 * 16 - 0.5))


def test_the_extraction_refuses_something_that_is_not_a_stack_of_maps():
    with pytest.raises(ValueError, match=r"\(B, K, H, W\)"):
        soft_argmax2d(torch.rand(4, 2))


# ---------------------------------------------------------------------------
# one extraction, shared by the trainer, the table and the pipeline
# ---------------------------------------------------------------------------
def test_a_faint_background_wrecks_a_global_centre_of_mass():
    """The bug this shared function exists to prevent, as a test.

    A linear head's background is not zero, it is small positive noise — and
    spread over 16 384 cells, a little noise everywhere outweighs the blob. The
    first trained detector measured 6.83 px this way and 1.06 px windowed: the
    same weights, a factor of six, and no epoch count would have closed it.
    """
    blob = _blob(128, 0.2, 0.25)
    noisy = torch.from_numpy(blob + 0.01)[None, None]  # 1% of the peak, everywhere

    dragged = soft_argmax2d(noisy, window=None)[0, 0]
    windowed = corners_from_output(noisy)[0, 0]

    assert float(dragged[0]) > 0.35  # hauled a third of the way to the centre
    assert torch.allclose(windowed, torch.tensor([0.2, 0.25]), atol=0.01)


def test_the_shared_extraction_passes_coordinates_straight_through():
    """A regression model already emits what everyone downstream wants, so this
    is the identity for it — which is why the corner_reg runs are unaffected."""
    coords = torch.rand(3, 4, 2)
    assert corners_from_output(coords) is coords


def test_the_shared_extraction_can_be_told_to_go_global():
    """Which is right while *training* a coordinate term through it: the gradient
    should reach the whole map, including the background it must learn to
    suppress."""
    maps = torch.from_numpy(_blob(64, 0.3, 0.7) + 0.02)[None, None]
    assert not torch.allclose(corners_from_output(maps), corners_from_output(maps, window=None))


def test_the_shared_extraction_refuses_a_shape_it_cannot_read():
    with pytest.raises(ValueError, match=r"\(B, K, 2\) or \(B, K, H, W\)"):
        corners_from_output(torch.rand(4, 2))


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_build_model_makes_either_detector():
    reg = build_model(Config({"model": {"name": "cornerregnet", "base": 8, "stages": 3}}))
    heat = build_model(Config({"model": {"name": "cornerheatnet", "base": 8, "depth": 2}}))
    assert isinstance(reg, CornerRegNet) and isinstance(heat, CornerHeatNet)


def test_every_model_says_what_it_produces():
    """The trainer, the losses and the metrics all dispatch on this, so a model
    that forgot to declare it would be routed as a restoration network."""
    assert DocUNet(base=4, depth=1).output_kind == "restoration"
    assert CornerRegNet(base=4, stages=1, grid=2, hidden=8).output_kind == "coords"
    assert CornerHeatNet(base=4, depth=1).output_kind == "heatmaps"
