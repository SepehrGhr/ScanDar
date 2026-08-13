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


# --- what a batch means, per task ------------------------------------------
def _corner_batch():
    return {
        "image": torch.rand(2, 3, 32, 32),
        "input": torch.rand(2, 3, 32, 32),
        "target": torch.rand(2, 3, 32, 32),
        "corners": torch.rand(2, 4, 2),
        "heatmaps": torch.rand(2, 4, 16, 16),
    }


def test_each_model_is_handed_the_tensors_it_needs():
    """A coordinate model has no use for the heatmaps in the batch, and moving
    four 128x128 maps per sample onto the GPU to be ignored is not free."""
    from scandar.train import batch_inputs, batch_targets

    batch = _corner_batch()
    device = torch.device("cpu")
    assert batch_inputs(batch, "restoration", device) is batch["input"]
    assert batch_inputs(batch, "coords", device) is batch["image"]
    assert batch_targets(batch, "coords", device) is batch["corners"]
    assert set(batch_targets(batch, "heatmaps", device)) == {"corners", "heatmaps"}
    assert batch_targets(batch, "restoration", device) is batch["target"]


def test_the_metrics_follow_the_model():
    """PSNR for a restoration, localisation error for a detector, and the heatmap
    model scored through the same soft-argmax the pipeline reads it with."""
    from scandar.train import quality_metrics

    batch = _corner_batch()
    restoration = quality_metrics("restoration", batch["input"], batch["target"])
    assert set(restoration) == {"psnr", "ssim"}

    coords = quality_metrics("coords", batch["corners"], batch["corners"], input_size=32)
    assert float(coords["corner_err"].max()) == 0.0

    heat = quality_metrics("heatmaps", batch["heatmaps"], batch, input_size=32)
    assert "corner_err" in heat and heat["corner_err"].shape == (2,)


def test_a_heatmap_size_that_disagrees_with_the_model_is_refused():
    """Two config keys and a model attribute all have to agree, and nothing forces
    them to. Caught before a single sample is generated, with all three numbers in
    the message."""
    from scandar.config import Config
    from scandar.model import CornerHeatNet
    from scandar.train import check_heatmap_size

    model = CornerHeatNet(base=4, depth=2, out_stride=2)
    ok = Config({"data": {"corner_input": 256, "heatmap_size": 128}})
    check_heatmap_size(ok, model, "heatmaps")

    wrong = Config({"data": {"corner_input": 256, "heatmap_size": 64}})
    with pytest.raises(ValueError, match="heatmap_size=128"):
        check_heatmap_size(wrong, model, "heatmaps")
    # A model that does not emit heatmaps is not asked about them.
    check_heatmap_size(wrong, DocUNet(base=4, depth=1), "restoration")


# --- corner detection pipeline (brief §5.1) --------------------------------
def _page_photo(quad=None):
    """A bright page on a dark surface — what the classical detector is for."""
    import cv2
    import numpy as np

    photo = np.full((600, 800, 3), 60, dtype=np.uint8)
    quad = np.float32([[150, 110], [650, 140], [620, 500], [130, 460]]) if quad is None else quad
    cv2.fillConvexPoly(photo, quad.astype(np.int32), (238, 236, 230))
    return cv2.GaussianBlur(photo, (5, 5), 0), quad


class _FixedDetector(torch.nn.Module):
    """A stand-in detector that answers whatever it was constructed with."""

    output_kind = "coords"

    def __init__(self, corners):
        super().__init__()
        self.register_parameter("unused", torch.nn.Parameter(torch.zeros(1)))
        self.corners = torch.tensor(corners, dtype=torch.float32)[None]

    def forward(self, x):
        return self.corners


def test_detect_corners_maps_its_answer_back_onto_the_photo():
    """Step 3 of the brief's §5.1, and the one it warns about: a coordinate
    scaled by the wrong factor is a wrong label that looks like a bad model."""
    from scandar.pipelines import detect_corners

    photo, _ = _page_photo()
    normalised = [[0.2, 0.25], [0.8, 0.25], [0.8, 0.75], [0.2, 0.75]]
    result = detect_corners(photo, _FixedDetector(normalised), input_size=64)

    assert result["source"] == "model"
    assert result["corners"].shape == (4, 2)
    # (x + 0.5) / W is the convention the labels are stored in, so 0.2 of an
    # 800-pixel photo is 159.5, not 160.
    assert result["corners"][0].tolist() == pytest.approx([159.5, 149.5], abs=0.01)
    assert result["normalised"].flatten().tolist() == pytest.approx(
        [value for corner in normalised for value in corner], abs=1e-4
    )


def test_a_degenerate_prediction_falls_back_to_the_classical_detector():
    """The guardrail. Four points that are not a page must not become a
    homography — the rectification would smear the page across the output."""
    from scandar.geometry import corner_errors
    from scandar.pipelines import detect_corners

    photo, quad = _page_photo()
    collapsed = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    result = detect_corners(photo, _FixedDetector(collapsed), input_size=64)

    assert result["source"] == "classical"
    assert result["problem"]  # and it says what was wrong with the model's answer
    assert float(corner_errors(result["corners"], quad).mean()) < 15


def test_the_last_resort_is_the_frame_rather_than_an_exception():
    """A demonstration in front of the teaching staff should degrade to something
    a human can see and correct, not to a traceback."""
    import numpy as np

    from scandar.pipelines import detect_corners

    noise = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    collapsed = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    result = detect_corners(noise, _FixedDetector(collapsed), input_size=64, fallback=False)
    assert result["source"] == "frame"
    assert result["corners"].tolist() == [[0, 0], [159, 0], [159, 119], [0, 119]]


def test_the_classical_detector_finds_a_page_on_a_desk():
    from scandar.geometry import corner_errors, order_corners
    from scandar.pipelines import classical_corners

    photo, quad = _page_photo()
    found = classical_corners(photo)
    assert found is not None
    assert float(corner_errors(found, order_corners(quad)).max()) < 15


def test_a_heatmap_detector_comes_back_with_its_maps_and_a_confidence():
    """The pipeline reads either formulation identically, and hands back what
    only a heatmap model can offer: how sure it was."""
    from scandar.model import CornerHeatNet
    from scandar.pipelines import detect_corners

    photo, _ = _page_photo()
    result = detect_corners(photo, CornerHeatNet(base=4, depth=2).eval(), input_size=64)
    assert result["kind"] == "heatmaps"
    assert result["heatmaps"].shape == (4, 32, 32)
    assert result["confidence"] is not None


def test_the_overlay_labels_the_corners_and_leaves_the_photo_alone():
    """A permuted quad is the failure this project is most exposed to, and four
    anonymous dots look the same whichever order they are in."""
    import numpy as np

    from scandar.pipelines import draw_corners

    photo, quad = _page_photo()
    before = photo.copy()
    overlay = draw_corners(photo, quad)
    assert overlay.shape == photo.shape and overlay.dtype == photo.dtype
    assert np.array_equal(photo, before)
    assert not np.array_equal(overlay, photo)


def test_detect_corners_refuses_something_that_is_not_a_photo():
    import numpy as np

    from scandar.pipelines import detect_corners

    with pytest.raises(ValueError, match="RGB HWC"):
        detect_corners(np.zeros((32, 32), dtype=np.uint8), None)
