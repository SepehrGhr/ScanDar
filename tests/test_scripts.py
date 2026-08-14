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


# --- putting the detectors head to head -------------------------------------
def _write_corner_table(directory, name, rows):
    import csv

    path = directory / f"{name}_corners.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_comparison_takes_one_row_per_detector_and_one_baseline(tmp_path):
    """The classical baseline is the same detector scored twice, so quoting it
    once per run would put three identical rows in a two-detector table."""
    import compare_detectors

    def row(variant, err):
        return {"split": "Test", "variant": variant, "n": "200", "corner_err_px": err,
                "corner_err_std": "1.0", "corner_err_pct": "0.3", "worst_corner_px": "2.0",
                "pck": "0.9", "quad_iou": "0.98"}

    for name, err in (("a_run", "1.0"), ("b_run", "3.0")):
        _write_corner_table(tmp_path, name, [row("detector", err),
                                             row("classical baseline", "40.0"),
                                             dict(row("detector", "9.9"), split="Validation")])

    rows = compare_detectors.comparison_rows(["a_run", "b_run"], "Test", directory=tmp_path)
    assert [r["run"] for r in rows] == ["a_run", "b_run", "classical"]
    assert "a_run" in compare_detectors.markdown_table(rows, "Test")


def test_comparing_a_run_that_was_never_evaluated_says_so(tmp_path):
    import compare_detectors

    with pytest.raises(SystemExit, match="evaluate.py"):
        compare_detectors.comparison_rows(["never_run"], directory=tmp_path)


def test_the_pck_curve_is_drawn_from_the_numbers_evaluate_wrote(tmp_path):
    """Two detectors on one axis, which is how the brief's comparison is read."""
    from scandar.viz import pck_curves

    curves = {
        "heat": [{"threshold_pct": 0.5, "pck": 0.8}, {"threshold_pct": 2.0, "pck": 0.95}],
        "reg": [{"threshold_pct": 0.5, "pck": 0.1}, {"threshold_pct": 2.0, "pck": 0.83}],
    }
    written = pck_curves(curves, tmp_path / "pck.png")
    assert written.exists() and written.stat().st_size > 0


# --- the dropout study's before-and-after table -----------------------------
def _write_restoration_table(directory, name, rows):
    import csv

    path = directory / f"{name}_restoration.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_dropout_table_signs_every_change_the_same_way(tmp_path):
    """A table where "+0.4" means better in one row and worse in the next is a
    table that will be read wrong, and the whole study is a question of sign:
    the metrics run in opposite directions and the expected result is a null."""
    import compare_dropout

    def restoration(psnr, split="Test", variant="enhanced"):
        return {"split": split, "variant": variant, "n": "200", "psnr": psnr,
                "psnr_std": "2.0", "ssim": "0.95", "ssim_std": "0.02"}

    def corners(err, split="Test", variant="detector"):
        return {"split": split, "variant": variant, "n": "200", "corner_err_px": err,
                "corner_err_std": "1.0", "corner_err_pct": "0.3", "worst_corner_px": "2.0",
                "pck": "0.9", "quad_iou": "0.98"}

    # PSNR is better when larger, corner error when smaller. Both arms below are
    # worse than their baseline, so both changes must come out negative.
    _write_restoration_table(tmp_path, "base_e", [restoration("26.0"),
                                                  restoration("27.0", split="Training")])
    _write_restoration_table(tmp_path, "drop_e", [restoration("25.0"),
                                                  restoration("25.2", split="Training")])
    _write_corner_table(tmp_path, "base_c", [corners("1.0"), corners("0.8", split="Training")])
    _write_corner_table(tmp_path, "drop_c", [corners("2.0"), corners("1.9", split="Training")])

    rows = compare_dropout.compare([("base_e", "drop_e"), ("base_c", "drop_c")],
                                   directory=tmp_path)
    psnr = next(r for r in rows if r["metric"].startswith("PSNR"))
    error = next(r for r in rows if r["metric"].startswith("corner error"))
    assert psnr["change"] == -1.0 and error["change"] == -1.0
    assert psnr["better"] == error["better"] == "baseline"
    # And the train-to-test gap is signed the same way: positive is worse unseen.
    assert psnr["baseline_train_gap"] == 1.0 and error["baseline_train_gap"] == 0.2


def test_an_arm_that_has_not_been_evaluated_yet_is_skipped_not_fatal(tmp_path, capsys):
    """The arms are trained one at a time over several sessions, so the table has
    to be readable while the rest are still running."""
    import compare_dropout

    assert compare_dropout.compare([("nothing", "nowhere")], directory=tmp_path) == []
    assert "no evaluation table yet" in capsys.readouterr().out


def test_a_log_that_disagrees_with_its_own_evaluation_is_not_compared(tmp_path):
    """A per-epoch log is only comparable across runs if the number in it means
    the same thing in both, and the column name does not promise that. The
    heatmap detector's baseline trained before the corner extraction was fixed:
    its log says 6.26 px where re-evaluating the same weights says 0.70. Printing
    that difference as a dropout effect would be off by a factor of nine."""
    import compare_dropout

    _write_corner_table(tmp_path, "stale_run", [
        {"split": "Validation", "variant": "detector", "n": "200", "corner_err_px": "0.70",
         "corner_err_std": "0.8", "corner_err_pct": "0.2", "worst_corner_px": "1.4",
         "pck": "0.98", "quad_iou": "0.98"},
    ])
    curve = [{"epoch": "1", "val_corner_err": "6.26", "val_loss": "0.1", "train_loss": "0.1"}]

    ratio = compare_dropout.curve_agrees_with_table("stale_run", "val_corner_err", curve, tmp_path)
    assert ratio == pytest.approx(6.26 / 0.70, rel=1e-3)
    assert ratio > 1.5  # the threshold the comparison refuses above

    # A run whose log and table agree is compared as normal.
    _write_corner_table(tmp_path, "fresh_run", [
        {"split": "Validation", "variant": "detector", "n": "200", "corner_err_px": "2.72",
         "corner_err_std": "1.8", "corner_err_pct": "0.75", "worst_corner_px": "4.7",
         "pck": "0.87", "quad_iou": "0.96"},
    ])
    fresh = [{"epoch": "1", "val_corner_err": "2.72", "val_loss": "0.1", "train_loss": "0.1"}]
    assert compare_dropout.curve_agrees_with_table("fresh_run", "val_corner_err", fresh, tmp_path) < 1.5
