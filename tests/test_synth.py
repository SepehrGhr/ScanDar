"""The synthetic generator.

The properties tested here are the ones the whole project rests on, and every one
of them fails silently if it breaks:

* the four points the generator chose really are where the page ended up;
* the rectified input and the clean target line up to a fraction of a pixel;
* the same key always produces the same sample, which is what makes the frozen
  evaluation sets frozen and an interrupted training run resumable;
* the extras that make a page stop looking like its scan can never reach the
  enhancement task, whose target *is* that scan.
"""

import numpy as np
import pytest

from scandar.degrade import DegradationConfig
from scandar.geometry import quad_problem, rect_corners, warp_points
from scandar.seed import rng_for
from scandar.synth import CORNER_ONLY_OPTIONS, SynthOptions, compose_sample, curl_page, sample_quad

CANVAS = (288, 384)


@pytest.fixture
def scan():
    """A stand-in for a clean A4 scan: pale paper with dark ruled lines."""
    import cv2

    image = np.full((424, 300, 3), 240, dtype=np.uint8)
    for row in range(24, 400, 26):
        cv2.line(image, (20, row), (280, row), (35, 35, 45), 2)
    for column in range(30, 280, 60):
        cv2.line(image, (column, 20), (column, 400), (60, 40, 120), 1)
    return image


@pytest.fixture
def options():
    # No real backgrounds and no scaled-down canvas: the tests should exercise the
    # generator, not the contents of data/.
    return SynthOptions(canvas=CANVAS, procedural_prob=1.0)


def make(scan, options, key="s", **kwargs):
    return compose_sample(scan, rng_for(key), options=options, **kwargs)


# --- placement -------------------------------------------------------------
def test_quads_are_always_valid_and_ordered(options):
    for index in range(120):
        quad, params = sample_quad(rng_for("quad", index), CANVAS, 300 / 424, options)
        assert quad_problem(quad, CANVAS) is None, params
        assert quad.shape == (4, 2)
        # TL is up and left of BR, always.
        assert quad[0, 0] < quad[2, 0] and quad[0, 1] < quad[2, 1]


def test_placement_falls_back_rather_than_failing(options):
    """A page that cannot fit must still yield something usable."""
    impossible = SynthOptions(canvas=CANVAS, page_scale=(3.0, 4.0), max_tries=5)
    quad, params = sample_quad(rng_for("impossible"), CANVAS, 300 / 424, impossible)
    assert "fallback" in params and quad.shape == (4, 2)


# --- the labels ------------------------------------------------------------
def test_the_corners_are_where_the_page_actually_is(scan, options):
    """The label has to be the page, not merely near it.

    Checked against the pixels rather than against the homography that produced
    them: the scan's paper is much lighter than any background the generator
    draws, so the quad's interior must be bright and the ring just outside it
    must not be.
    """
    import cv2

    for index in range(6):
        sample = make(scan, options, key=f"where{index}", keep_clean=True)
        inside = np.zeros(sample.clean_photo.shape[:2], dtype=np.uint8)
        cv2.fillPoly(inside, [np.round(sample.corners).astype(np.int32)], 255)
        eroded = cv2.erode(inside, np.ones((9, 9), np.uint8))
        outside = cv2.bitwise_not(cv2.dilate(inside, np.ones((15, 15), np.uint8)))

        grey = cv2.cvtColor(sample.clean_photo, cv2.COLOR_RGB2GRAY)
        assert grey[eroded > 0].mean() > grey[outside > 0].mean() + 25


def test_the_homography_maps_the_scan_onto_the_corners(scan, options):
    sample = make(scan, options, key="H")
    mapped = warp_points(sample.H, rect_corners(scan.shape[1], scan.shape[0]))
    assert np.allclose(mapped, sample.corners, atol=1e-2)


# --- alignment -------------------------------------------------------------
def test_the_rectified_pair_is_aligned_to_within_half_a_pixel(scan, options):
    """The brief warns twice about this one, so it is measured, not assumed."""
    from scandar.checks import alignment_shift

    for index in range(6):
        sample = make(scan, options, key=f"align{index}", keep_clean=True)
        aligned, target = sample.rectify((256, 362), source=sample.clean_photo)
        assert alignment_shift(aligned, target) < 0.5


def test_a_patch_is_the_matching_crop_of_the_whole_page(scan, options):
    """Warping straight into a patch has to equal flattening and then slicing."""
    sample = make(scan, options, key="patch")
    rect_size = (256, 362)
    _, whole = sample.rectify(rect_size)
    _, patch = sample.rectify_patch((40, 60, 128), rect_size)

    reference = whole[60:188, 40:168]
    assert patch.shape == reference.shape
    assert float(np.abs(patch.astype(float) - reference.astype(float)).mean()) < 1.0


def test_random_patches_stay_inside_the_page(scan, options):
    sample = make(scan, options, key="boxes")
    rng = rng_for("boxes")
    for _ in range(10):
        degraded, target, (x, y, size) = sample.random_patch(rng, 96, (256, 362))
        assert 0 <= x <= 256 - 96 and 0 <= y <= 362 - 96
        assert degraded.shape == target.shape == (96, 96, 3)


def test_patch_sampling_prefers_something_over_nothing(scan, options):
    """Ink-seeking is a bias, not a filter — but it has to actually bias."""
    sample = make(scan, options, key="ink")
    blank = np.full_like(scan, 240)
    empty = compose_sample(blank, rng_for("ink"), options=options)

    with_text = np.mean(
        [sample.random_patch(rng_for("t", i), 96, (256, 362))[1].std() for i in range(12)]
    )
    without = np.mean(
        [empty.random_patch(rng_for("t", i), 96, (256, 362))[1].std() for i in range(12)]
    )
    assert with_text > without


# --- reproducibility -------------------------------------------------------
def test_the_same_key_gives_the_same_sample(scan, options):
    first = compose_sample(scan, rng_for("same"), options=options)
    second = compose_sample(scan, rng_for("same"), options=options)
    other = compose_sample(scan, rng_for("other"), options=options)

    assert np.array_equal(first.photo, second.photo)
    assert np.array_equal(first.corners, second.corners)
    assert not np.array_equal(first.photo, other.photo)


def test_params_describe_the_sample_and_serialise(scan, options):
    import json

    sample = make(scan, options, key="params")
    json.dumps(sample.params)
    assert set(sample.params) >= {"canvas", "placement", "corners", "background", "degradation"}
    assert sample.params["degradation"]["severity"] == "medium"


# --- keeping the two tasks apart ------------------------------------------
def test_the_enhancement_task_never_gets_the_corner_only_extras():
    """A shared config file must not be able to switch these on for enhancement.

    A tinted or bulged page paired with the flat clean scan would ask the model
    to invent a colour and a shape it was never shown how to derive.
    """
    config = {
        "data": {"canvas": [288, 384]},
        "synth": {name: 1.0 for name in CORNER_ONLY_OPTIONS},
    }
    enhancement = SynthOptions.from_config(config, task="enhance")
    assert all(getattr(enhancement, name) == 0.0 for name in CORNER_ONLY_OPTIONS)
    assert enhancement.canvas == (288, 384)

    corner = SynthOptions.from_config(config, task="corner")
    assert all(getattr(corner, name) == 1.0 for name in CORNER_ONLY_OPTIONS)


def test_from_config_rejects_an_unknown_setting():
    with pytest.raises(ValueError, match="rotaton_deg"):
        SynthOptions.from_config({"synth": {"rotaton_deg": 10}})


def test_curling_a_page_leaves_its_corners_alone(scan):
    """The curl is a displacement that vanishes on the border, so labels stay exact."""
    curled, params = curl_page(scan, rng_for("curl"))
    assert curled.shape == scan.shape
    assert params["amplitude_px"] > 0
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert np.array_equal(curled[corner], scan[corner])
    assert not np.array_equal(curled, scan)


def test_severity_reaches_the_generator(scan, options):
    mild = compose_sample(
        scan, rng_for("sev"), options=options, degradation=DegradationConfig().scaled("mild")
    )
    assert mild.params["degradation"]["severity"] == "mild"


def test_landscape_canvases_are_generated(scan):
    always = SynthOptions(canvas=CANVAS, landscape_prob=1.0, procedural_prob=1.0)
    sample = compose_sample(scan, rng_for("landscape"), options=always)
    assert sample.canvas_size == (CANVAS[1], CANVAS[0])
