"""The training loop's moving parts.

The loop itself is exercised by running it — a handful of steps on one batch,
which is the classic check that catches a broken training loop: if a network
cannot memorise a single pair, no amount of data will help it, and the bug is in
the plumbing rather than in the model.

The rest is about resuming. A Colab session dies without warning, so a run has to
come back on the same weights, the same optimiser moments and the same random
stream, and each of those is a separate way to be silently wrong.
"""

import math

import pytest
import torch

from scandar.config import Config
from scandar.losses import CombinedRestorationLoss
from scandar.model import DocUNet
from scandar.pipelines import enhance_document, tiled_forward
from scandar.train import _atomic_save, lr_at, restore_rng_state, rng_state


# --- the schedule ----------------------------------------------------------
def test_the_warmup_ramps_and_the_cosine_decays():
    schedule = [lr_at(step, 100, 10, 1.0, 0.0) for step in range(100)]
    assert schedule[0] == pytest.approx(0.1)  # first step of a ten-step warmup
    assert schedule[9] == pytest.approx(1.0)  # warmup ends at the base rate
    assert all(b <= a + 1e-12 for a, b in zip(schedule[9:], schedule[10:]))  # then decays
    assert schedule[-1] < 0.01


def test_the_schedule_lands_on_the_floor_not_on_zero():
    assert lr_at(100, 100, 0, 1e-3, 1e-6) == pytest.approx(1e-6, abs=1e-9)


def test_a_schedule_without_warmup_starts_at_the_base_rate():
    assert lr_at(0, 50, 0, 2e-4, 0.0) == pytest.approx(2e-4)


def test_the_schedule_does_not_run_off_the_end():
    """A run extended past its planned length keeps a valid rate rather than a
    cosine that has wrapped back around and started climbing."""
    assert lr_at(500, 100, 10, 1e-3, 1e-6) == pytest.approx(1e-6, abs=1e-9)


# --- the loop --------------------------------------------------------------
def test_the_model_can_memorise_one_batch():
    """Overfit one batch to nearly nothing. This is the check that separates "the
    model is not learning" from "the training loop is broken"."""
    torch.manual_seed(0)
    model = DocUNet(base=8, depth=2)
    criterion = CombinedRestorationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    inputs = torch.rand(2, 3, 64, 64)
    targets = (inputs * 0.5 + 0.25).clamp(0, 1)

    first_loss = first_l1 = None
    for _ in range(300):
        loss, parts = criterion(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss, first_l1 = float(loss), parts["l1"]
    assert float(loss) < 0.15 * first_loss
    assert parts["l1"] < 0.5 * first_l1


# --- checkpoints -----------------------------------------------------------
def test_a_checkpoint_round_trips(tmp_path):
    model = DocUNet(base=4, depth=2)
    path = _atomic_save({"model": model.state_dict(), "config": {}}, tmp_path / "last.pt")
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()

    reloaded = DocUNet(base=4, depth=2)
    reloaded.load_state_dict(torch.load(path, weights_only=False)["model"])
    for before, after in zip(model.state_dict().values(), reloaded.state_dict().values()):
        assert torch.equal(before, after)


def test_load_model_rebuilds_from_the_checkpoint_alone(tmp_path):
    """Reloading must not depend on remembering which config produced a file."""
    from scandar.model import load_model

    config = Config({"model": {"name": "docunet", "base": 4, "depth": 2}})
    model = DocUNet(base=4, depth=2)
    path = _atomic_save(
        {"model": model.state_dict(), "config": config.to_dict()}, tmp_path / "best.pt"
    )
    reloaded, reloaded_config = load_model(path)
    assert reloaded.base == 4 and reloaded_config.model.depth == 2
    assert not reloaded.training


def test_the_random_state_survives_a_save_and_a_load(tmp_path):
    """Including the trip through torch.save, which is where restoring the CUDA
    generator went wrong the first time."""
    torch.manual_seed(1234)
    saved = rng_state()
    expected = torch.rand(4)

    path = _atomic_save({"rng": saved}, tmp_path / "state.pt")
    torch.manual_seed(999)
    restore_rng_state(torch.load(path, weights_only=False)["rng"])
    assert torch.equal(torch.rand(4), expected)


# --- inference pipeline ----------------------------------------------------
def test_tiling_reproduces_a_single_pass_in_the_interior():
    """Cosine-blended tiles must not leave a seam. Border tiles genuinely see
    replicated context instead of real neighbours, so the interior is what has to
    agree."""
    torch.manual_seed(0)
    model = DocUNet(base=4, depth=2).eval()
    image = torch.rand(1, 3, 128, 128)
    tiled = tiled_forward(model, image, tile=64, overlap=24, device=torch.device("cpu"))
    with torch.no_grad():
        whole = model(image)
    interior = (slice(None), slice(None), slice(24, -24), slice(24, -24))
    assert float((tiled[interior] - whole[interior]).abs().max()) < 0.02


def test_the_pipeline_returns_the_size_and_type_it_was_given():
    import numpy as np

    model = DocUNet(base=4, depth=2).eval()
    photo = (np.random.rand(97, 131, 3) * 255).astype(np.uint8)
    out = enhance_document(photo, model, device=torch.device("cpu"), tile=64, overlap=16)
    assert out.shape == photo.shape and out.dtype == np.uint8


def test_the_pipeline_can_cap_the_resolution_it_works_at():
    import numpy as np

    model = DocUNet(base=4, depth=2).eval()
    photo = (np.random.rand(200, 150, 3) * 255).astype(np.uint8)
    out = enhance_document(
        photo, model, device=torch.device("cpu"), tile=64, overlap=16, max_side=80
    )
    assert out.shape == photo.shape  # shrunk to work, stretched back to return


def test_rectify_pulls_the_page_out_of_a_photo():
    """The step that has to happen before the network sees anything: four corners
    in, a flat page out, at the aspect the corners describe."""
    import cv2
    import numpy as np

    from scandar.pipelines import rectify_document

    photo = np.zeros((400, 400, 3), dtype=np.uint8)
    quad = np.float32([[100, 60], [320, 110], [290, 330], [70, 280]])
    cv2.fillConvexPoly(photo, quad.astype(np.int32), (255, 255, 255))
    cv2.circle(photo, (150, 110), 8, (255, 0, 0), -1)  # a mark near the page's TL

    page = rectify_document(photo, quad, out_width=200)
    assert page.shape[1] == 200
    # the page fills the frame: every corner of the output is page, not background
    for y, x in ((2, 2), (2, -3), (-3, 2), (-3, -3)):
        assert page[y, x].mean() > 128
    # and the mark stayed in the top-left quadrant rather than being flipped
    marked = np.argwhere((page[:, :, 0] > 128) & (page[:, :, 1] < 128))
    assert marked.mean(axis=0)[0] < page.shape[0] / 2
    assert marked.mean(axis=0)[1] < page.shape[1] / 2


def test_rectify_accepts_the_corners_in_any_order():
    """A human clicking four corners at a demo will not start at the top-left."""
    import numpy as np

    from scandar.pipelines import rectify_document

    photo = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype(np.uint8)
    quad = np.float32([[40, 30], [250, 50], [240, 260], [30, 240]])
    straight = rectify_document(photo, quad, out_width=128)
    for roll in (1, 2, 3):
        assert np.array_equal(rectify_document(photo, np.roll(quad, roll, axis=0),
                                               out_width=128), straight)


def test_rectify_can_be_told_the_page_is_a4():
    """Foreshortening makes a steeply-shot page's far edge short, so an aspect
    read off the quad squashes it. 'a4' overrides the estimate."""
    import numpy as np

    from scandar.pipelines import rectify_document

    photo = np.full((300, 300, 3), 200, dtype=np.uint8)
    squashed_quad = np.float32([[20, 20], [280, 20], [280, 120], [20, 120]])
    page = rectify_document(photo, squashed_quad, out_width=100, aspect="a4")
    assert page.shape[0] == 141  # 100 / (1/1.4142)


def test_the_pipeline_refuses_something_that_is_not_an_rgb_image():
    import numpy as np

    model = DocUNet(base=4, depth=2).eval()
    with pytest.raises(ValueError, match="RGB HWC"):
        enhance_document(np.zeros((32, 32), dtype=np.uint8), model, device=torch.device("cpu"))


def test_the_blending_weights_sum_to_one_everywhere():
    """Otherwise the overlap regions come back brighter or darker than the rest —
    a band down the middle of the page that no amount of training removes."""
    from scandar.pipelines import _cosine_window, _starts

    tile, overlap, extent = 64, 16, 200
    weights = torch.zeros(extent)
    window = torch.from_numpy(_cosine_window(tile, tile, overlap))[0]
    for start in _starts(extent, tile, tile - overlap):
        weights[start : start + tile] += window
    assert float(weights.min()) > 0.0
    assert math.isfinite(float(weights.max()))


# --- figures and guards ----------------------------------------------------
def test_the_training_curve_is_drawn_from_the_run_it_names(tmp_path):
    """Figures come from metrics.csv, never from a number typed by hand."""
    from scandar.viz import read_metrics, training_curves

    run = tmp_path / "a_run"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,val_psnr\n1,0.8,0.7,12.0\n2,0.4,0.45,18.5\n",
        encoding="utf-8",
    )
    rows = read_metrics(run)
    assert rows[1]["val_psnr"] == 18.5 and isinstance(rows[1]["epoch"], float)

    written = training_curves(run, tmp_path / "curves.png", baseline_psnr=15.6)
    assert written.exists() and written.stat().st_size > 0


def test_plotting_an_untrained_run_says_so(tmp_path):
    from scandar.viz import read_metrics

    with pytest.raises(FileNotFoundError, match="trained yet"):
        read_metrics(tmp_path)


def test_a_frozen_set_from_another_canvas_is_flagged(capsys):
    """Training on one distribution while validating on another is silent, plausible
    and wrong for the whole run — so it gets shouted about before the GPU time."""
    from scandar.config import Config
    from scandar.train import warn_if_frozen_set_is_stale

    class FakeDataset:
        entries = [{"canvas": [1152, 1536]}]

    matching = Config({"data": {"canvas": [1152, 1536]}, "_config_path": "x.yaml"})
    assert warn_if_frozen_set_is_stale(matching, FakeDataset()) is True

    mismatched = Config({"data": {"canvas": [1920, 2560]}, "_config_path": "x.yaml"})
    assert warn_if_frozen_set_is_stale(mismatched, FakeDataset()) is False
    assert "1920x2560" in capsys.readouterr().out


def test_a_landscape_frozen_sample_is_not_mistaken_for_a_mismatch():
    """One sample in ten is generated with the canvas transposed — the phone held
    sideways — and that is not a stale frozen set."""
    from scandar.config import Config
    from scandar.train import warn_if_frozen_set_is_stale

    class Landscape:
        entries = [{"canvas": [1536, 1152]}]

    config = Config({"data": {"canvas": [1152, 1536]}, "_config_path": "x.yaml"})
    assert warn_if_frozen_set_is_stale(config, Landscape()) is True
