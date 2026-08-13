"""The architecture's contract.

The tests here are about the two properties the rest of the project leans on:
the network is **fully convolutional**, so it can train on 256x256 patches and
restore a whole page, and its output is a valid image at whatever size went in.
Both are the kind of thing that breaks silently — a shape mismatch at 1024x1448
would only surface on the day someone tried to enhance a real page.
"""

import pytest
import torch

from scandar.config import Config
from scandar.model import ConvBlock, DocUNet, build_model, clamp_image, count_parameters


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
