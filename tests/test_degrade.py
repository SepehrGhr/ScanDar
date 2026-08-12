"""The degradation pipeline.

Two things have to hold for every stage: it must actually change the image (a
transform that quietly does nothing is worse than no transform, because the
report will still claim it ran), and it must change it *reproducibly* from the
sample's own generator, or the frozen evaluation sets stop being frozen.

A third, subtler property gets its own test: severity has to be monotone. The
brief warns against degrading the text past recovery, and the guard against that
is only meaningful if `mild` really is milder than `hard`.
"""

import json

import numpy as np
import pytest

from scandar.degrade import (
    DEFAULT,
    SEVERITY_SCALE,
    STAGES,
    DegradationConfig,
    blur_mask,
    brightness_contrast,
    color_cast,
    degrade,
    downscale_upscale,
    gaussian_noise,
    illumination_gradient,
    jpeg_recompress,
    random_blur,
    soft_shadows,
)


@pytest.fixture
def page():
    """A small page-like image: light paper with dark strokes on it."""
    import cv2

    image = np.full((240, 180, 3), 236, dtype=np.uint8)
    for row in range(20, 220, 18):
        cv2.line(image, (16, row), (164, row), (30, 30, 40), 2)
    return image


ALL_STAGES = [pytest.param(stage, id=name) for name, stage in STAGES]


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_every_stage_keeps_the_contract(stage, page):
    out, params = stage(page, np.random.default_rng(0), DEFAULT)
    assert out.dtype == np.uint8, "the pipeline works in the 8 bits a phone stores"
    assert out.shape == page.shape
    assert isinstance(params, dict)
    json.dumps(params)  # params travel into the frozen manifest, so they must serialise


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_every_stage_is_reproducible_from_its_generator(stage, page):
    first, params = stage(page, np.random.default_rng(11), DEFAULT)
    second, again = stage(page, np.random.default_rng(11), DEFAULT)
    assert np.array_equal(first, second) and params == again


@pytest.mark.parametrize(
    "stage",
    [downscale_upscale, brightness_contrast, color_cast, illumination_gradient, random_blur,
     gaussian_noise, jpeg_recompress],
)
def test_stages_that_must_always_do_something(stage, page):
    """Shadows and highlights are sampled in, these are not — they always apply."""
    out, _ = stage(page, np.random.default_rng(2), DEFAULT)
    assert not np.array_equal(out, page)


def test_shadows_only_darken(page):
    # Fifty draws, so that the "no shadow this time" branch is not the only one seen.
    for seed in range(50):
        out, params = soft_shadows(page, np.random.default_rng(seed), DEFAULT)
        assert out.max() <= page.max()
        if params["count"]:
            assert {shape["kind"] for shape in params["shapes"]} <= {"blob", "edge", "arm"}


def test_the_pipeline_runs_every_stage_in_the_briefs_order(page):
    out, params = degrade(page, np.random.default_rng(3))
    assert out.dtype == np.uint8 and out.shape == page.shape
    assert [name for name, _ in STAGES] == [k for k in params if k != "severity"]
    json.dumps(params)


def test_collecting_steps_gives_one_image_per_stage(page):
    _, params = degrade(page, np.random.default_rng(4), collect_steps=True)
    names = [name for name, _ in params["steps"]]
    assert names == ["input"] + [name for name, _ in STAGES]
    assert all(image.shape == page.shape for _, image in params["steps"])


def test_the_same_key_degrades_identically(page):
    first, _ = degrade(page, np.random.default_rng(5))
    second, _ = degrade(page, np.random.default_rng(5))
    third, _ = degrade(page, np.random.default_rng(6))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_severity_is_monotone(page):
    """`mild` must really be milder, or the guard on over-degradation means nothing."""
    damage = {}
    for severity in SEVERITY_SCALE:
        config = DegradationConfig().scaled(severity)
        errors = [
            float(np.abs(degrade(page, np.random.default_rng(seed), config)[0].astype(float)
                         - page.astype(float)).mean())
            for seed in range(12)
        ]
        damage[severity] = float(np.mean(errors))
    assert damage["mild"] < damage["medium"] < damage["hard"]


def test_severity_never_produces_an_impossible_jpeg_quality():
    for severity in SEVERITY_SCALE:
        low, high = DegradationConfig().scaled(severity).jpeg_quality
        assert 1 <= low <= high <= 100


def test_config_reads_a_yaml_block_and_rejects_typos():
    config = DegradationConfig.from_config({"severity": "mild", "blur_sigma": [1.0, 3.0]})
    assert config.severity == "mild"
    assert config.blur_sigma == (1.0 * SEVERITY_SCALE["mild"], 3.0 * SEVERITY_SCALE["mild"])

    with pytest.raises(ValueError, match="blur_sigmaa"):
        DegradationConfig.from_config({"blur_sigmaa": [1.0, 3.0]})
    with pytest.raises(ValueError, match="severity"):
        DegradationConfig.from_config({"severity": "brutal"})


def test_blurring_a_mask_small_matches_blurring_it_large():
    """The shortcut that makes wide shadows affordable has to be a shortcut only."""
    import cv2

    mask = np.zeros((400, 300), dtype=np.float32)
    cv2.circle(mask, (150, 200), 70, 1.0, -1)

    fast = blur_mask(mask, 40.0)
    exact = cv2.GaussianBlur(mask, (0, 0), 40.0)
    assert float(np.abs(fast - exact).max()) < 0.02


def test_jpeg_quality_lands_where_it_is_asked_to(page):
    """Low quality must visibly cost more than high quality, or the stage is a no-op."""
    low = DegradationConfig(jpeg_quality=(20, 20))
    high = DegradationConfig(jpeg_quality=(95, 95))
    damage = [
        float(np.abs(jpeg_recompress(page, np.random.default_rng(1), config)[0].astype(float)
                     - page.astype(float)).mean())
        for config in (high, low)
    ]
    assert damage[0] < damage[1]
