"""The end-to-end scanner: the differentiable warp, the chain, its loss.  *(brief §7)*

Two questions run through all of it.

**Does the torch warp agree with the OpenCV one?** It has to, exactly, because the
training pairs everything else in this project was built and measured on came out
of ``cv2.warpPerspective``, and a differentiable twin that resamples half a pixel
elsewhere would train the detector to correct for the difference.

**Does the gradient actually reach the corners?** That single assertion is what
"end to end" means here, and it is what the bonus is graded on. It is cheap, it
is unambiguous, and it fails the moment somebody puts a numpy call — an
``order_corners``, a ``.numpy()``, a detach for convenience — into the chain.
"""

import numpy as np
import pytest
import torch

from scandar import warp
from scandar.backgrounds import BackgroundBank
from scandar.datasets import SyntheticScanDataset, scan_item
from scandar.degrade import DegradationConfig
from scandar.geometry import homography, normalize_corners, rect_corners
from scandar.losses import ScanLoss, build_loss
from scandar.model import EndToEndScanner, build_model
from scandar.prepare import ScanBank
from scandar.synth import Sources, SynthOptions

CANVAS = (288, 384)
RECT = (128, 181)
PATCH = 64
QUAD = np.float32([[24.0, 30.0], [250.0, 12.0], [262.0, 350.0], [16.0, 330.0]])


@pytest.fixture
def photo():
    """Something with texture everywhere: a warp is only measurable on detail."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (CANVAS[1], CANVAS[0], 3), dtype=np.uint8)
    return image


@pytest.fixture
def sources(tmp_path):
    """Three tiny ruled 'scans' and procedural backgrounds, as elsewhere."""
    import cv2

    from scandar.io import imwrite_rgb

    cache = tmp_path / "cache"
    cache.mkdir()
    for index in range(3):
        page = np.full((424, 300, 3), 240, dtype=np.uint8)
        for row in range(20 + index * 4, 410, 24):
            cv2.line(page, (18, row), (282, row), (30, 30, 45), 2)
        imwrite_rgb(cache / f"{index}.png", page)

    return Sources(
        scans=ScanBank([str(i) for i in range(3)], directory=cache),
        backgrounds=BackgroundBank([], directory=tmp_path / "backgrounds"),
        options=SynthOptions(canvas=CANVAS, procedural_prob=1.0),
        degradation=DegradationConfig(),
    )


def _tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None] / 255.0


# --- the warp, against the one that made the data --------------------------
def test_the_homography_solve_agrees_with_opencv():
    """The same four-point solve, to floating-point noise. Everything downstream
    rests on this: the detector is fine-tuned through this matrix and scored
    against labels generated through OpenCV's."""
    source = rect_corners(200, 300)
    theirs = homography(source, QUAD)
    ours = warp.homography_from_points(torch.from_numpy(source), torch.from_numpy(QUAD))
    assert np.abs(theirs - ours.numpy()).max() < 1e-5


def test_rectifying_agrees_with_warp_perspective(photo):
    """The differentiable path reproduces the pipeline that generated every
    training pair in the project."""
    import cv2

    theirs = cv2.warpPerspective(
        photo, homography(QUAD, rect_corners(*RECT)), RECT, flags=cv2.INTER_LINEAR
    )
    ours = warp.rectify(
        _tensor(photo), torch.from_numpy(normalize_corners(QUAD, CANVAS))[None], RECT
    )
    difference = np.abs(ours[0].permute(1, 2, 0).numpy() * 255.0 - theirs.astype(np.float32))
    # A grey level, on pure noise, where OpenCV's fixed-point interpolation
    # weights and torch's floating-point ones disagree by construction.
    assert difference.mean() < 1.0


def test_a_patch_is_the_crop_it_claims_to_be(photo):
    """Composing the crop into the homography has to give the same pixels as
    flattening the whole page and slicing it — that is the only reason it is
    allowed to be twenty times cheaper."""
    corners = torch.from_numpy(normalize_corners(QUAD, CANVAS))[None]
    whole = warp.rectify(_tensor(photo), corners, RECT)
    box = (24, 40)
    patch = warp.rectify_patch(
        _tensor(photo), corners, RECT, torch.tensor([box], dtype=torch.float32), 48
    )
    sliced = whole[..., box[1] : box[1] + 48, box[0] : box[0] + 48]
    assert torch.allclose(patch, sliced, atol=2e-3)


def test_the_identity_warp_returns_the_image(photo):
    """align_corners=True and the pixel-index convention, checked together: a page
    whose corners are the frame's corners must come back unchanged."""
    frame = torch.from_numpy(normalize_corners(rect_corners(*CANVAS), CANVAS))[None]
    same = warp.rectify(_tensor(photo), frame, CANVAS)
    # Not bit-exact: the sampling coordinates come out of a linear solve, so they
    # miss the cell centres by a part in ten million and the bilinear weights
    # follow. A quarter of a hundredth of a grey level is that, and not a
    # half-pixel convention error, which would show up here as a visible blur.
    assert torch.allclose(same, _tensor(photo), atol=1e-4)


def test_each_sample_is_cropped_at_its_own_origin(photo):
    """The batch is warped in one call, and every sample's crop lands somewhere
    different — a shared grid would silently give them all the first one's."""
    corners = torch.from_numpy(normalize_corners(QUAD, CANVAS))[None].expand(2, 4, 2)
    images = _tensor(photo).expand(2, 3, CANVAS[1], CANVAS[0])
    boxes = torch.tensor([[0.0, 0.0], [30.0, 50.0]])
    patches = warp.rectify_patch(images, corners, RECT, boxes, 32)
    assert not torch.allclose(patches[0], patches[1])


def test_the_warp_is_differentiable_in_the_corners(photo):
    """**The assertion the bonus is graded on.** Push an image loss back through
    the sampler and the homography solve, and the gradient at the corners is
    finite and not zero — which is what "the loss is computed end to end" means.
    """
    corners = torch.from_numpy(normalize_corners(QUAD, CANVAS))[None].requires_grad_(True)
    rectified = warp.rectify(_tensor(photo), corners, (64, 90))
    torch.nn.functional.mse_loss(rectified, torch.zeros_like(rectified)).backward()
    assert corners.grad is not None
    assert torch.isfinite(corners.grad).all()
    assert float(corners.grad.abs().sum()) > 0


# --- the chained model -----------------------------------------------------
def _scanner(**overrides):
    settings = {
        "name": "scanner",
        "detector": {"name": "cornerheatnet", "base": 4, "depth": 2, "out_stride": 2},
        "enhancer": {"name": "docunet", "base": 4, "depth": 2},
        "rect_size": RECT,
        "patch_size": 32,
    }
    settings.update(overrides)
    return build_model({"model": settings})


def _scan_batch(photo, batch=2):
    corners = torch.from_numpy(normalize_corners(QUAD, CANVAS))[None].expand(batch, 4, 2)
    source = torch.from_numpy(np.ascontiguousarray(photo.transpose(2, 0, 1)))[None]
    return {
        "image": torch.rand(batch, 3, 64, 64),
        "source": source.expand(batch, 3, CANVAS[1], CANVAS[0]),
        "box": torch.tensor([[8, 12, 32]] * batch, dtype=torch.int32),
        "size": torch.tensor([[CANVAS[0], CANVAS[1]]] * batch, dtype=torch.int32),
        "corners": corners,
        "target": torch.rand(batch, 3, 32, 32),
    }


def test_the_chain_produces_a_restored_patch(photo):
    scanner = _scanner()
    out = scanner(_scan_batch(photo))
    assert out["corners"].shape == (2, 4, 2)
    assert out["rectified"].shape == (2, 3, 32, 32)
    assert out["restored"].shape == (2, 3, 32, 32)
    assert 0.0 <= float(out["restored"].min()) and float(out["restored"].max()) <= 1.0


def test_the_enhancement_loss_reaches_the_detector(photo):
    """The whole bonus in one assertion: nothing supervises the corners, and the
    detector still gets a gradient — through the enhancer, the warp and the
    homography solve."""
    scanner = _scanner()
    batch = _scan_batch(photo)
    output = scanner(batch)
    loss, _ = ScanLoss()(output, {"target": batch["target"], "corners": batch["corners"]})
    loss.backward()

    grads = [p.grad for p in scanner.detector.parameters() if p.grad is not None]
    assert grads, "no gradient reached the detector at all"
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0


def test_a_frozen_enhancer_neither_learns_nor_drifts(photo):
    """Frozen means frozen: no gradients, and — the part that is easy to miss —
    batch normalisation left in eval so its running statistics do not follow the
    warped patches."""
    scanner = _scanner()
    scanner.train()
    assert not scanner.enhancer.training and scanner.detector.training
    assert all(not p.requires_grad for p in scanner.enhancer.parameters())

    batch = _scan_batch(photo)
    scanner(batch)["restored"].mean().backward()
    assert all(p.grad is None for p in scanner.enhancer.parameters())


def test_an_unfrozen_enhancer_is_trained_too(photo):
    scanner = _scanner(freeze_enhancer=False)
    scanner.train()
    assert scanner.enhancer.training
    batch = _scan_batch(photo)
    scanner(batch)["restored"].mean().backward()
    assert any(p.grad is not None for p in scanner.enhancer.parameters())


def test_the_chain_refuses_halves_that_are_the_wrong_way_round():
    with pytest.raises(ValueError, match="corner detector"):
        _scanner(detector={"name": "docunet", "base": 4, "depth": 2})
    with pytest.raises(ValueError, match="restore"):
        _scanner(enhancer={"name": "cornerregnet", "base": 4, "stages": 2, "grid": 4})


def test_the_chain_says_what_it_wants_instead_of_crashing_on_a_tensor():
    with pytest.raises(TypeError, match="dict of tensors"):
        _scanner()(torch.rand(1, 3, 64, 64))


def test_a_coordinate_detector_also_chains(photo):
    """The chain reads its corners through ``corners_from_output``, which is the
    identity for a regression head — so approach A works here unchanged, and the
    comparison could be run through the bonus too."""
    scanner = _scanner(
        detector={"name": "cornerregnet", "base": 4, "stages": 2, "grid": 4, "hidden": 16}
    )
    out = scanner(_scan_batch(photo))
    assert out["restored"].shape == (2, 3, 32, 32)


def test_the_components_are_only_loaded_when_asked(tmp_path):
    """A scanner checkpoint has to be reloadable on a machine where the runs it
    was assembled from do not exist, so the constructor never touches them."""
    scanner = _scanner(detector_weights=str(tmp_path / "nothing.pt"))
    with pytest.raises(FileNotFoundError, match="detector"):
        scanner.load_components()
    assert scanner.load_components(strict=False) == {}


# --- the loss --------------------------------------------------------------
def test_the_scan_loss_is_the_enhancement_loss_by_default():
    loss = build_loss({"loss": {"name": "scan"}})
    assert isinstance(loss, ScanLoss)
    assert loss.corner is None
    assert loss.active == ["l1", "msssim", "sobel"]


def test_the_corner_anchor_is_opt_in_and_reported_separately():
    loss = ScanLoss(coord_l1=0.5)
    prediction = {"restored": torch.rand(2, 3, 32, 32), "corners": torch.rand(2, 4, 2)}
    target = {"target": torch.rand(2, 3, 32, 32), "corners": torch.rand(2, 4, 2)}
    total, parts = loss(prediction, target)
    assert "coord_l1" in parts and "l1" in parts
    assert "coord_l1" in loss.active
    assert torch.isfinite(total)


def test_the_scan_loss_refuses_a_bare_tensor():
    with pytest.raises(TypeError, match="output dict"):
        ScanLoss()(torch.rand(2, 3, 8, 8), {"target": torch.rand(2, 3, 8, 8)})


# --- the dataset -----------------------------------------------------------
def test_a_scan_sample_carries_both_labels(sources):
    dataset = SyntheticScanDataset(
        sources,
        "train",
        length=4,
        input_size=64,
        heatmap_size=32,
        source_side=max(CANVAS),
        rect_size=RECT,
        patch_size=PATCH,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64) and item["image"].dtype == torch.float32
    assert item["target"].shape == (3, PATCH, PATCH)
    assert item["corners"].shape == (4, 2)
    assert float(item["corners"].min()) >= 0.0 and float(item["corners"].max()) <= 1.0
    # The large photo travels in 8 bits: a float copy of a real 1920x2560 canvas
    # is 59 MB per sample through the loader.
    assert item["source"].dtype == torch.uint8
    assert item["box"].shape == (3,)


def test_the_scan_sample_warps_to_its_own_target(sources):
    """The chain's anchor: fed the *true* corners, the differentiable warp lands
    on the clean target the loss is computed against. If these two drift apart,
    every scan run is optimising alignment instead of restoration."""
    from scandar.seed import rng_for

    sample = sources.compose(rng_for("scan-test", 0))
    item = scan_item(
        sample,
        rng_for("scan-test", 0, "patch"),
        input_size=64,
        heatmap_size=32,
        heatmap_sigma=2.0,
        source_side=max(CANVAS),
        rect_size=RECT,
        patch_size=PATCH,
    )
    box = tuple(int(v) for v in item["box"])
    rectified = warp.rectify_patch(
        item["source"][None].float() / 255.0,
        item["corners"][None],
        RECT,
        torch.tensor([[box[0], box[1]]], dtype=torch.float32),
        PATCH,
    )
    degraded, _ = sample.rectify_patch(box, RECT)
    difference = np.abs(
        rectified[0].permute(1, 2, 0).numpy() * 255.0 - degraded.astype(np.float32)
    )
    assert difference.mean() < 1.0


def test_the_scan_dataset_is_reproducible_from_its_key(sources):
    dataset = SyntheticScanDataset(
        sources, "train", length=4, input_size=64, heatmap_size=32,
        source_side=max(CANVAS), rect_size=RECT, patch_size=PATCH,
    )
    assert torch.equal(dataset[1]["corners"], dataset[1]["corners"])
    assert not torch.equal(dataset[0]["corners"], dataset[1]["corners"])


def test_a_batch_of_two_orientations_collates_and_still_warps(photo):
    """The phone is held sideways in a tenth of the generator's samples, so a
    batch can hold a portrait photo and a landscape one. They are padded to a
    common shape — and the padding must be *invisible*: each sample's corners
    belong to its own size, not to the padded tensor's, or a tenth of every batch
    carries a systematic scale error."""
    from scandar.datasets import collate_photos

    landscape = np.ascontiguousarray(photo.transpose(1, 0, 2))
    items = []
    for image in (photo, landscape):
        height, width = image.shape[:2]
        quad = np.float32(
            [[0.1 * width, 0.1 * height], [0.8 * width, 0.05 * height],
             [0.85 * width, 0.9 * height], [0.15 * width, 0.85 * height]]
        )
        items.append(
            {
                "source": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "corners": torch.from_numpy(normalize_corners(quad, (width, height))),
                "size": torch.tensor([width, height], dtype=torch.int32),
                "box": torch.tensor([8, 12, 32], dtype=torch.int32),
            }
        )

    batch = collate_photos(items)
    side = max(photo.shape[0], photo.shape[1])
    assert batch["source"].shape == (2, 3, side, side)

    together = warp.rectify_patch(
        batch["source"].float() / 255.0,
        batch["corners"],
        RECT,
        batch["box"][:, :2].float(),
        32,
        size=batch["size"],
    )
    for index, item in enumerate(items):
        alone = warp.rectify_patch(
            item["source"][None].float() / 255.0,
            item["corners"][None],
            RECT,
            item["box"][None, :2].float(),
            32,
        )
        assert torch.allclose(together[index], alone[0], atol=1e-5)


def test_collating_untouched_batches_is_the_default(photo):
    """Nothing else in the project has a variable-sized tensor, so the shared
    collate has to be exactly the default whenever the shapes already agree."""
    from scandar.datasets import collate_photos

    items = [{"image": torch.rand(3, 8, 8), "corners": torch.rand(4, 2)} for _ in range(3)]
    batch = collate_photos(items)
    assert batch["image"].shape == (3, 3, 8, 8) and batch["corners"].shape == (3, 4, 2)


# --- the trainer's plumbing ------------------------------------------------
def test_the_trainer_hands_the_chain_all_three_of_its_inputs(photo):
    from scandar.train import batch_inputs, batch_size_of, batch_targets, quality_metrics

    batch = _scan_batch(photo)
    device = torch.device("cpu")
    inputs = batch_inputs(batch, "scan", device)
    assert set(inputs) == {"image", "source", "box", "size"}
    assert batch_size_of(inputs) == 2
    assert set(batch_targets(batch, "scan", device)) == {"target", "corners"}

    metrics = quality_metrics(
        "scan",
        {"restored": batch["target"], "corners": batch["corners"]},
        {"target": batch["target"], "corners": batch["corners"]},
        input_size=64,
    )
    # Both halves on one line: the point of the bonus is what one costs the other.
    assert {"psnr", "ssim", "corner_err", "quad_iou"} <= set(metrics)
    assert float(metrics["corner_err"].max()) == 0.0


# --- the inference chain ---------------------------------------------------
def test_scan_document_flattens_and_restores_without_a_click():
    """The §7 pipeline on a synthetic desk: a page in a photo, a flat page out,
    with the corner ordering enforced on the way through."""
    import cv2

    from scandar.model import DocUNet
    from scandar.pipelines import scan_document

    photo = np.full((400, 320, 3), 50, dtype=np.uint8)
    quad = np.float32([[40, 30], [280, 55], [265, 350], [30, 320]])
    cv2.fillConvexPoly(photo, quad.astype(np.int32), (235, 233, 228))

    result = scan_document(
        photo, enhancer=DocUNet(base=4, depth=2), corners=quad[[2, 0, 3, 1]], out_width=64
    )
    assert result["source"] == "given"
    assert result["scan"].shape == (int(round(64 * 1.4142)), 64, 3)
    assert result["rectified"].shape == result["scan"].shape
    # Reordered on the way in, so the page comes out the right way up.
    assert np.allclose(result["corners"][0], quad[0], atol=1.0)


def test_both_warps_flatten_the_same_page(photo):
    """The project has two implementations of "rectify a page" — OpenCV's, which
    made every training pair, and the differentiable one the bonus trains
    through. They are offered as alternatives at inference, so they had better be
    the same operation."""
    from scandar.pipelines import rectify_document

    with_cv2 = rectify_document(photo, QUAD, out_width=96, aspect="a4", backend="cv2")
    with_torch = rectify_document(
        photo, QUAD, out_width=96, aspect="a4", backend="torch", device=torch.device("cpu")
    )
    assert with_cv2.shape == with_torch.shape and with_torch.dtype == np.uint8
    # Bicubic in both, so the disagreement is only in how each rounds its
    # interpolation weights — measured on pure noise, which is the worst case
    # any real photograph could put to them.
    assert np.abs(with_cv2.astype(float) - with_torch.astype(float)).mean() < 2.0


def test_an_unknown_warp_backend_is_refused(photo):
    from scandar.pipelines import rectify_document

    with pytest.raises(ValueError, match="warp backend"):
        rectify_document(photo, QUAD, backend="numpy")


def test_a_fine_tuned_scanner_runs_the_inference_chain():
    """A chain saved by the end-to-end fine-tune goes straight back into the
    pipeline it was fine-tuned for, with no unpacking at the call site."""
    import cv2

    from scandar.pipelines import scan_document

    photo = np.full((400, 320, 3), 50, dtype=np.uint8)
    cv2.fillConvexPoly(
        photo, np.int32([[40, 30], [280, 55], [265, 350], [30, 320]]), (235, 233, 228)
    )
    scanner = _scanner()
    result = scan_document(photo, scanner, out_width=64, tile=64)
    assert result["scan"].shape == (int(round(64 * 1.4142)), 64, 3)
    assert result["warp"] == "cv2"

    through_torch = scan_document(photo, scanner, out_width=64, tile=64, warp="torch")
    assert through_torch["warp"] == "torch"
    assert through_torch["scan"].shape == result["scan"].shape


def test_scan_document_needs_a_detector_or_corners():
    with pytest.raises(ValueError, match="detector"):
        scan_document_missing()


def scan_document_missing():
    from scandar.pipelines import scan_document

    return scan_document(np.zeros((32, 32, 3), dtype=np.uint8))


def test_scan_document_reports_the_path_its_corners_came_from():
    """The fallback counter, which is the number worth knowing before a live
    demonstration rather than during one."""
    import cv2

    from scandar.pipelines import scan_document

    photo = np.full((400, 320, 3), 50, dtype=np.uint8)
    cv2.fillConvexPoly(
        photo, np.int32([[40, 30], [280, 55], [265, 350], [30, 320]]), (235, 233, 228)
    )

    class _Degenerate(torch.nn.Module):
        output_kind = "coords"

        def __init__(self):
            super().__init__()
            # The pipeline reads the device off the model's parameters, so a
            # stand-in needs one even if it never uses it.
            self.register_parameter("unused", torch.nn.Parameter(torch.zeros(1)))

        def forward(self, x):  # four points piled on one another
            return torch.full((1, 4, 2), 0.5)

    result = scan_document(photo, _Degenerate(), out_width=64)
    assert result["source"] in ("classical", "frame")
    assert result["problem"]


# --- the config ------------------------------------------------------------
def test_the_chain_is_assembled_from_the_baselines_and_nothing_else():
    """The fine-tune must start from the models the report already measured. The
    two halves are spelled out in the scan config because they come from files
    with nothing in common, and spelled-out is exactly the kind of duplication
    that drifts, so it is asserted rather than trusted."""
    from scandar.config import load_config
    from scandar.io import paths

    scan = load_config(paths.repo / "configs" / "scan_e2e.yaml")
    detector = load_config(paths.repo / "configs" / "corner_heat.yaml")
    enhancer = load_config(paths.repo / "configs" / "enhance.yaml")

    def without_name(block):
        return {k: v for k, v in dict(block).items() if k != "name"}

    assert without_name(scan.model["detector"]) == without_name(detector.model)
    assert without_name(scan.model["enhancer"]) == without_name(enhancer.model)
    assert scan.model["detector_weights"] == "corner_heat"
    assert scan.model["enhancer_weights"] == "enhance_realistic"
    # Fine-tuning either detector of the matched pair would invalidate the
    # comparison that answers §5, so this run has to have its own name.
    assert scan.run["name"] not in ("corner_heat", "corner_reg")
