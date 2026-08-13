"""The dataset contracts.

These are the tensors a training loop will actually be handed, so the tests are
about shape, range and ordering — the transposed-tensor and unscaled-label class
of bug that the brief warns about twice and that costs an afternoon each time.

They run against a miniature data root built in a temporary directory rather than
against ``data/``, so they pass on a fresh clone with no photos in it yet.
"""

import numpy as np
import pytest
import torch

from scandar.backgrounds import BackgroundBank
from scandar.datasets import (
    FrozenSyntheticDataset,
    SyntheticCornerDataset,
    SyntheticEnhanceDataset,
    corner_item,
    gaussian_heatmaps,
    to_tensor,
)
from scandar.degrade import DegradationConfig
from scandar.geometry import denormalize_corners
from scandar.io import imwrite_rgb
from scandar.prepare import ScanBank, freeze_split
from scandar.synth import Sources, SynthOptions

CANVAS = (288, 384)
RECT = (128, 181)
PATCH = 64


@pytest.fixture
def sources(tmp_path):
    """Three tiny 'scans' and no background photos, so the textures are procedural."""
    import cv2

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


# --- plumbing --------------------------------------------------------------
def test_to_tensor_is_chw_float_in_zero_one():
    image = np.zeros((7, 5, 3), dtype=np.uint8)
    image[0, 0] = (255, 128, 0)
    tensor = to_tensor(image)

    assert tensor.shape == (3, 7, 5) and tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert float(tensor[0, 0, 0]) == pytest.approx(1.0)
    assert float(tensor[2, 0, 0]) == pytest.approx(0.0)


def test_heatmaps_peak_on_their_own_corner():
    corners = np.array([[0.2, 0.3], [0.8, 0.25], [0.75, 0.9], [0.15, 0.85]], dtype=np.float32)
    maps = gaussian_heatmaps(corners, (64, 64), sigma=2.0)

    assert maps.shape == (4, 64, 64) and maps.dtype == np.float32
    for index, (x, y) in enumerate(denormalize_corners(corners, (64, 64))):
        peak = np.unravel_index(int(maps[index].argmax()), maps[index].shape)
        assert abs(peak[1] - x) <= 1.0 and abs(peak[0] - y) <= 1.0
        assert float(maps[index].max()) == pytest.approx(1.0, abs=0.05)


def test_corner_item_keeps_the_original_size_with_the_sample():
    """Predictions have to be mapped back, and guessing the size later is how
    corners end up scaled by the wrong factor."""
    photo = np.zeros((384, 288, 3), dtype=np.uint8)
    corners = np.array([[20, 30], [260, 25], [265, 350], [15, 355]], dtype=np.float32)
    item = corner_item(photo, corners, input_size=64, heatmap_size=32, heatmap_sigma=1.5)

    assert item["size"].tolist() == [288, 384]
    assert torch.allclose(item["corners_px"], torch.from_numpy(corners))
    assert item["image"].shape == (3, 64, 64)
    assert item["heatmaps"].shape == (4, 32, 32)


# --- the synthetic datasets ------------------------------------------------
def test_corner_dataset_contract(sources):
    dataset = SyntheticCornerDataset(
        sources, "train", length=4, input_size=64, heatmap_size=32, heatmap_sigma=1.5
    )
    assert len(dataset) == 4

    item = dataset[0]
    assert item["image"].shape == (3, 64, 64) and item["image"].dtype == torch.float32
    assert item["corners"].shape == (4, 2)
    assert 0.0 <= float(item["corners"].min()) and float(item["corners"].max()) <= 1.0
    assert item["heatmaps"].shape == (4, 32, 32)
    assert item["scan"] in sources.scans.ids


def test_enhance_dataset_returns_aligned_pairs(sources):
    dataset = SyntheticEnhanceDataset(
        sources, "train", length=4, patch_size=PATCH, rect_size=RECT, patches_per_photo=1
    )
    item = dataset[0]
    assert item["input"].shape == item["target"].shape == (3, PATCH, PATCH)
    assert item["target"].dtype == torch.float32
    # The target is a clean scan crop: mostly bright paper, never a saturated mess.
    assert float(item["target"].mean()) > 0.5


def test_enhance_dataset_page_mode_returns_the_whole_rectified_page(sources):
    dataset = SyntheticEnhanceDataset(sources, "val", length=2, mode="page", rect_size=RECT)
    item = dataset[0]
    assert item["input"].shape == (3, RECT[1], RECT[0])
    assert item["target"].shape == (3, RECT[1], RECT[0])


def test_enhance_dataset_rejects_an_unknown_mode(sources):
    with pytest.raises(ValueError, match="mode"):
        SyntheticEnhanceDataset(sources, "train", length=1, mode="pages")


def test_patches_from_one_photo_share_it_but_differ(sources):
    """The amortisation must reuse the photo without returning the same patch."""
    dataset = SyntheticEnhanceDataset(
        sources, "train", length=8, patch_size=PATCH, rect_size=RECT, patches_per_photo=4
    )
    first, second = dataset[0], dataset[1]
    assert first["scan"] == second["scan"]
    assert first["box"].tolist() != second["box"].tolist()

    # A new group composites a new photo.
    assert dataset[4]["box"].tolist() != first["box"].tolist()


def test_samples_are_reproducible_and_epoch_dependent(sources):
    dataset = SyntheticCornerDataset(sources, "train", length=4, input_size=64, heatmap_size=32)
    first = dataset[0]
    assert torch.equal(dataset[0]["image"], first["image"])

    dataset.set_epoch(1)
    assert not torch.equal(dataset[0]["image"], first["image"])

    dataset.set_epoch(0)
    assert torch.equal(dataset[0]["image"], first["image"])


def test_two_indices_are_two_different_samples(sources):
    dataset = SyntheticCornerDataset(sources, "train", length=4, input_size=64, heatmap_size=32)
    assert not torch.equal(dataset[0]["image"], dataset[1]["image"])


def test_warming_the_banks_decodes_everything_up_front(sources):
    """Called in the parent, so forked workers inherit the images, not the decodes."""
    assert sources.warm() is sources
    assert len(sources.scans._cache) == len(sources.scans.ids)


def test_a_dataloader_can_batch_it(sources):
    from torch.utils.data import DataLoader

    dataset = SyntheticEnhanceDataset(
        sources, "train", length=4, patch_size=PATCH, rect_size=RECT, patches_per_photo=2
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    assert batch["input"].shape == (4, 3, PATCH, PATCH)
    assert batch["target"].shape == (4, 3, PATCH, PATCH)


# --- the frozen sets -------------------------------------------------------
def test_freezing_is_reproducible_and_readable(tmp_path, sources, monkeypatch):
    """The point of freezing: regenerating the set gives back the same bytes."""
    import scandar.synth as synth_module

    monkeypatch.setattr(synth_module, "build_sources", lambda *a, **k: sources)
    config = {"data": {"canvas": list(CANVAS)}}

    directory = tmp_path / "frozen" / "enhance" / "val"
    kwargs = dict(seed=7, directory=directory, task="enhance")
    first = freeze_split(config, "val", count=3, force=True, **kwargs)
    bytes_first = [(directory / e["photo"]).read_bytes() for e in first["samples"]]

    second = freeze_split(config, "val", count=3, force=True, **kwargs)
    bytes_second = [(directory / e["photo"]).read_bytes() for e in second["samples"]]
    assert bytes_first == bytes_second
    assert [e["corners"] for e in first["samples"]] == [e["corners"] for e in second["samples"]]

    dataset = FrozenSyntheticDataset(
        directory, task="enhance", scans=sources.scans, rect_size=RECT
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert item["input"].shape == (3, RECT[1], RECT[0])
    assert item["id"] == "val_0000"

    corner_directory = tmp_path / "frozen" / "corner" / "val"
    freeze_split(config, "val", count=3, seed=7, directory=corner_directory, task="corner")
    corners = FrozenSyntheticDataset(
        corner_directory, task="corner", scans=sources.scans, input_size=64, heatmap_size=32
    )[0]
    assert corners["image"].shape == (3, 64, 64)
    assert 0.0 <= float(corners["corners"].min()) and float(corners["corners"].max()) <= 1.0


def test_a_frozen_set_refuses_to_serve_the_other_task(tmp_path, sources, monkeypatch):
    """The corner task's photos carry tinted stock and curl; the enhancement
    target is the flat clean scan, so scoring one on the other's samples asks the
    model to undo something it was never shown. The mismatch is refused rather
    than quietly measured."""
    import scandar.synth as synth_module

    monkeypatch.setattr(synth_module, "build_sources", lambda *a, **k: sources)
    directory = tmp_path / "frozen" / "corner" / "val"
    freeze_split({"data": {"canvas": list(CANVAS)}}, "val", count=2, seed=7,
                 directory=directory, task="corner")

    with pytest.raises(ValueError, match="separate frozen sets"):
        FrozenSyntheticDataset(directory, task="enhance", scans=sources.scans)


def test_frozen_patches_are_frozen(tmp_path, sources, monkeypatch):
    """The per-epoch validation curve is patch-level, so the patches themselves
    have to be as fixed as the pages are."""
    import scandar.synth as synth_module

    monkeypatch.setattr(synth_module, "build_sources", lambda *a, **k: sources)
    directory = tmp_path / "frozen" / "enhance" / "val"
    freeze_split({"data": {"canvas": list(CANVAS)}}, "val", count=2, seed=7,
                 directory=directory, task="enhance")

    def build():
        return FrozenSyntheticDataset(
            directory, task="enhance", scans=sources.scans, rect_size=RECT,
            mode="patch", patch_size=PATCH, patches_per_page=3,
        )

    first, second = build(), build()
    assert len(first) == 6  # two pages, three patches each
    assert first[0]["input"].shape == (3, PATCH, PATCH)
    for index in range(len(first)):
        assert torch.equal(first[index]["input"], second[index]["input"])
        assert torch.equal(first[index]["target"], second[index]["target"])
    # Different patches of the same page, not the same patch three times.
    boxes = {tuple(first[i]["box"].tolist()) for i in range(3)}
    assert len(boxes) > 1


def test_freezing_is_idempotent(tmp_path, sources, monkeypatch):
    import scandar.synth as synth_module

    monkeypatch.setattr(synth_module, "build_sources", lambda *a, **k: sources)
    config = {"data": {"canvas": list(CANVAS)}}
    directory = tmp_path / "frozen" / "test"

    freeze_split(config, "test", count=2, seed=7, directory=directory)
    stamps = {p.name: p.stat().st_mtime_ns for p in directory.glob("photo_*.jpg")}
    freeze_split(config, "test", count=2, seed=7, directory=directory)
    assert {p.name: p.stat().st_mtime_ns for p in directory.glob("photo_*.jpg")} == stamps


def test_a_missing_frozen_set_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze_eval_sets"):
        FrozenSyntheticDataset(tmp_path / "nothing")
