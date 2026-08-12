"""The split is the one data decision that silently invalidates every number.

If two degraded versions of the same page land on opposite sides, the test score
stops measuring generalisation and starts measuring memorisation — and nothing in
the training logs would show it.
"""

import numpy as np
import pytest

from scandar import prepare
from scandar.io import imread_rgb, imwrite_rgb, list_images, paths


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """A miniature data root: 12 tiny 'scans' and 8 'backgrounds'."""
    from scandar.io import Paths

    fake = Paths(repo=paths.repo, data=tmp_path / "data", out=tmp_path / "out")
    fake.scans.mkdir(parents=True)
    fake.backgrounds.mkdir(parents=True)

    rng = np.random.default_rng(0)
    for i in range(1, 13):
        imwrite_rgb(fake.scans / f"{i}.png", rng.integers(0, 255, (140, 100, 3), dtype=np.uint8))
    for i in range(1, 9):
        imwrite_rgb(fake.backgrounds / f"bg{i}.png", rng.integers(0, 255, (60, 60, 3), dtype=np.uint8))

    monkeypatch.setattr(prepare, "paths", fake)
    return fake


def test_splits_are_disjoint_and_complete(fake_data):
    splits = prepare.make_splits(seed=7, n_val=2, n_test=2)
    train, val, test = (set(splits["scans"][k]) for k in ("train", "val", "test"))

    assert len(train) == 8 and len(val) == 2 and len(test) == 2
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == {str(i) for i in range(1, 13)}


def test_splits_are_deterministic_in_the_seed(fake_data):
    assert prepare.make_splits(seed=7)["scans"] == prepare.make_splits(seed=7)["scans"]
    assert prepare.make_splits(seed=7)["scans"] != prepare.make_splits(seed=8)["scans"]


def test_backgrounds_split_without_overlap(fake_data):
    splits = prepare.make_splits(seed=7)
    bg_train = set(splits["backgrounds"]["train"])
    bg_heldout = set(splits["backgrounds"]["heldout"])
    assert bg_heldout and not (bg_train & bg_heldout)
    assert len(bg_train) + len(bg_heldout) == 8


def test_adding_backgrounds_later_does_not_reshuffle_the_scans(fake_data):
    """Scans are permuted before backgrounds, so collecting more surfaces later
    cannot quietly move a scan from train to test."""
    before = prepare.make_splits(seed=7)["scans"]
    rng = np.random.default_rng(1)
    for i in range(9, 15):
        imwrite_rgb(fake_data.backgrounds / f"bg{i}.png", rng.integers(0, 255, (60, 60, 3), dtype=np.uint8))
    assert prepare.make_splits(seed=7)["scans"] == before


def test_cache_downscales_and_preserves_aspect_ratio(fake_data):
    info = prepare.cache_scans(long_side=70)
    assert info["written"] == 12

    cached = list_images(fake_data.scans_cache)
    assert len(cached) == 12
    for path in cached:
        image = imread_rgb(path)
        height, width = image.shape[:2]
        assert max(height, width) == 70
        assert width / height == pytest.approx(100 / 140, rel=0.02)
    # Lossless, because these images are the ground-truth targets.
    assert all(p.suffix == ".png" for p in cached)


def test_cache_skips_work_on_a_second_run(fake_data):
    prepare.cache_scans(long_side=70)
    assert prepare.cache_scans(long_side=70)["written"] == 0
    assert prepare.cache_scans(long_side=70, force=True)["written"] == 12
