"""The parts of the command-line scripts that are not just plumbing.

Scripts in this project are meant to be thin, and most of them are. The batch
enhancer is the exception: it decides which files to work on and where on a page
to crop a comparison, and both are easy to get quietly wrong -- a glob that
silently matches nothing, or a zoom panel that lands on blank margin and shows
the reader nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from enhance_batch import collect_inputs, densest_ink_window  # noqa: E402


@pytest.fixture
def photos(tmp_path):
    from scandar.io import imwrite_rgb

    for name in ("Image2", "Image10", "Image1"):  # deliberately out of order
        imwrite_rgb(tmp_path / f"{name}.jpg", np.full((32, 24, 3), 200, dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    return tmp_path


# --- choosing what to work on ----------------------------------------------
def test_a_directory_yields_its_images_in_natural_order(photos):
    found = collect_inputs([str(photos)])
    assert [p.stem for p in found] == ["Image1", "Image2", "Image10"]


def test_non_images_are_left_alone(photos):
    assert all(p.suffix == ".jpg" for p in collect_inputs([str(photos)]))


def test_files_directories_and_globs_can_be_mixed(photos, tmp_path):
    from scandar.io import imwrite_rgb

    other = tmp_path / "elsewhere"
    other.mkdir()
    imwrite_rgb(other / "extra.png", np.zeros((8, 8, 3), dtype=np.uint8))
    found = collect_inputs([str(photos), str(other / "extra.png")])
    assert [p.name for p in found] == ["extra.png", "Image1.jpg", "Image2.jpg", "Image10.jpg"]


def test_the_same_file_named_twice_is_only_done_once(photos):
    one = str(photos / "Image1.jpg")
    assert len(collect_inputs([one, one, str(photos / "Image1.jpg")])) == 1


def test_a_pattern_matching_nothing_says_so_instead_of_doing_nothing(tmp_path):
    """Silently restoring zero photos and reporting success is the worst outcome."""
    with pytest.raises(SystemExit, match="no such file"):
        collect_inputs([str(tmp_path / "nope") + "/*.jpg"])


# --- choosing where to zoom -------------------------------------------------
def test_the_zoom_lands_on_the_writing_not_on_the_margin():
    """A crop from the middle of a page is blank paper about half the time, which
    makes a before/after panel that says nothing about legibility. The contract is
    that the window captures the writing, not that it lands on a particular pixel."""
    page = np.full((400, 600, 3), 250, dtype=np.uint8)
    page[300:360, 400:560] = 20  # the only ink on the page, in the bottom right
    rows, cols = densest_ink_window(page, 80, 200)

    ink = page[:, :, 0] < 128
    captured = ink[rows, cols].sum() / ink.sum()
    assert captured > 0.95, f"the zoom window only caught {captured:.0%} of the writing"


def test_the_zoom_prefers_the_densest_writing_when_there_is_more_than_one_patch():
    page = np.full((400, 600, 3), 250, dtype=np.uint8)
    page[40:70, 40:120] = 90     # a faint scribble top-left
    page[300:360, 380:560] = 10  # the real block of text, bottom-right
    rows, cols = densest_ink_window(page, 90, 220)
    assert rows.start > 200 and cols.start > 300


def test_the_zoom_window_stays_inside_the_page():
    page = np.full((120, 160, 3), 250, dtype=np.uint8)
    page[:10, :10] = 0  # ink jammed into the very corner
    rows, cols = densest_ink_window(page, 60, 100)
    assert rows.start >= 0 and cols.start >= 0
    assert rows.stop <= page.shape[0] and cols.stop <= page.shape[1]


def test_a_blank_page_still_returns_a_usable_window():
    rows, cols = densest_ink_window(np.full((90, 90, 3), 255, dtype=np.uint8), 40, 40)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (40, 40)


# --- showing where the detector looked --------------------------------------
def test_the_heatmap_overlay_keeps_the_photo_s_frame():
    """The maps are 128x128 and the photo is whatever the phone made; an overlay
    that does not resize back is four blobs in the corner of a black square."""
    from detect_batch import heatmap_overlay

    photo = np.full((300, 200, 3), 120, dtype=np.uint8)
    maps = np.zeros((4, 32, 32), dtype=np.float32)
    maps[0, 4, 4] = 1.0
    blended = heatmap_overlay(photo, maps)
    assert blended.shape == photo.shape and blended.dtype == np.uint8


def test_the_overlay_is_brightest_where_the_model_looked():
    """Otherwise it is a decoration rather than a diagnostic."""
    from detect_batch import heatmap_overlay

    photo = np.full((128, 128, 3), 120, dtype=np.uint8)
    maps = np.zeros((4, 32, 32), dtype=np.float32)
    maps[0, 24:28, 4:8] = 1.0  # bottom-left in map coordinates
    blended = heatmap_overlay(photo, maps).astype(np.int32).sum(axis=2)
    hot = np.unravel_index(int(blended.argmax()), blended.shape)
    assert hot[0] > 64 and hot[1] < 64


def test_a_flat_map_does_not_divide_by_its_own_zero_range():
    """An untrained model, or one that found nothing, emits a constant map."""
    from detect_batch import heatmap_overlay

    photo = np.full((64, 64, 3), 200, dtype=np.uint8)
    blended = heatmap_overlay(photo, np.zeros((4, 16, 16), dtype=np.float32))
    assert np.isfinite(blended).all() and blended.shape == photo.shape
