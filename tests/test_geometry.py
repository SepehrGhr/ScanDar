"""Corner ordering and coordinate conventions.

Two families of bug live here and neither one announces itself. Corners ordered
differently on different images rotate the rectification silently; corners that
are not rescaled with their image are simply wrong labels, and the model dutifully
learns the error. Both are cheap to test and expensive to find later.
"""

import numpy as np
import pytest

from scandar.geometry import (
    corner_errors,
    denormalize_corners,
    homography,
    is_valid_quad,
    normalize_corners,
    order_corners,
    quad_area,
    quad_iou,
    quad_problem,
    rect_corners,
    rect_size_for,
    resize_with_corners,
    scale_points,
    translation,
    warp_points,
)

CANVAS = (400, 600)


def tilted_quad(angle_deg=12.0, half=(90.0, 130.0), centre=(200.0, 300.0), jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    angle = np.radians(angle_deg)
    quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * np.asarray(half)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    quad = quad @ rotation.T + np.asarray(centre)
    if jitter:
        quad = quad + rng.uniform(-jitter, jitter, size=(4, 2))
    return quad


# --- ordering --------------------------------------------------------------
def test_ordering_is_top_left_first_and_clockwise():
    ordered = order_corners(tilted_quad(angle_deg=0.0))
    assert ordered[0].tolist() == [110.0, 170.0]  # TL
    assert ordered[1].tolist() == [290.0, 170.0]  # TR
    assert ordered[2].tolist() == [290.0, 430.0]  # BR
    assert ordered[3].tolist() == [110.0, 430.0]  # BL


@pytest.mark.parametrize("angle", [-30.0, -12.0, 0.0, 12.0, 30.0, 44.0])
def test_ordering_survives_any_permutation(angle):
    quad = tilted_quad(angle_deg=angle, jitter=8.0, seed=1)
    canonical = order_corners(quad)
    rng = np.random.default_rng(2)
    for _ in range(20):
        shuffled = order_corners(quad[rng.permutation(4)])
        assert np.allclose(shuffled, canonical, atol=1e-4)


def test_ordering_is_idempotent():
    ordered = order_corners(tilted_quad(angle_deg=25.0, jitter=6.0, seed=3))
    assert np.allclose(order_corners(ordered), ordered)


# --- homographies ----------------------------------------------------------
def test_homography_maps_the_corners_it_was_given():
    source = rect_corners(320, 480)
    destination = order_corners(tilted_quad(angle_deg=18.0, jitter=10.0, seed=4))
    mapped = warp_points(homography(source, destination), source)
    assert np.allclose(mapped, destination, atol=1e-3)


def test_rectification_is_the_inverse_of_placement():
    """Warping a page onto a quad and flattening it again is the identity."""
    source = rect_corners(320, 480)
    quad = order_corners(tilted_quad(angle_deg=-20.0, jitter=10.0, seed=5))
    onto = homography(source, quad)
    back = homography(quad, source)
    assert np.allclose(warp_points(back, warp_points(onto, source)), source, atol=1e-3)


def test_translation_composes_a_crop_into_a_homography():
    quad = order_corners(tilted_quad(angle_deg=8.0, seed=6))
    to_rect = homography(quad, rect_corners(256, 256))
    cropped = translation(-40, -25) @ to_rect
    assert np.allclose(warp_points(cropped, quad), warp_points(to_rect, quad) - [40, 25], atol=1e-3)


# --- scaling and normalisation --------------------------------------------
def test_normalisation_round_trips():
    quad = tilted_quad(seed=7)
    assert np.allclose(denormalize_corners(normalize_corners(quad, CANVAS), CANVAS), quad, atol=1e-3)


def test_normalised_corners_are_invariant_under_a_resize():
    """The whole point of normalising: the label stops depending on the size."""
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    quad = tilted_quad(seed=8)

    before = normalize_corners(quad, (400, 600))
    _, moved = resize_with_corners(image, quad, (256, 256))
    after = normalize_corners(moved, (256, 256))
    assert np.allclose(before, after, atol=1e-5)


def test_resize_moves_corners_the_way_opencv_moves_pixels():
    """A landmark at a known pixel must land on that pixel after the resize."""
    import cv2

    image = np.zeros((600, 400, 3), dtype=np.uint8)
    image[300, 200] = 255
    resized, moved = resize_with_corners(image, [[200.0, 300.0]], (200, 300))

    peak = np.unravel_index(int(resized[..., 0].argmax()), resized.shape[:2])
    assert abs(moved[0, 0] - peak[1]) <= 0.5 and abs(moved[0, 1] - peak[0]) <= 0.5
    del cv2


def test_scale_points_matches_the_half_pixel_convention():
    # A corner at the centre of a 400 px axis stays at the centre of a 200 px one.
    assert scale_points([[199.5, 0.0]], 0.5, 0.5)[0, 0] == pytest.approx(99.5)


# --- validity and overlap --------------------------------------------------
def test_a_sensible_page_quad_is_valid():
    assert is_valid_quad(tilted_quad(angle_deg=15.0, jitter=8.0, seed=9), CANVAS)


@pytest.mark.parametrize(
    "quad, expect",
    [
        ([[10, 10], [390, 10], [390, 590], [-5, 590]], "outside"),
        ([[100, 100], [300, 100], [300, 104], [100, 104]], "short"),
        ([[100, 100], [300, 100], [120, 130], [100, 300]], "convex"),
        ([[195, 295], [205, 295], [205, 305], [195, 305]], "short"),
        ([[4, 4], [395, 4], [395, 595], [4, 595]], "canvas"),
    ],
)
def test_bad_quads_are_rejected_with_a_reason(quad, expect):
    problem = quad_problem(order_corners(quad), CANVAS)
    assert problem is not None and expect in problem


def test_quad_area_and_iou():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    assert quad_area(square) == pytest.approx(10000.0)
    assert quad_iou(square, square) == pytest.approx(1.0, abs=0.01)
    assert quad_iou(square, square + [500, 500]) == 0.0
    # Half-overlapping squares: intersection 1/2, union 3/2 -> 1/3.
    assert quad_iou(square, square + [50, 0]) == pytest.approx(1 / 3, abs=0.02)


def test_rect_size_for_preserves_aspect_and_can_snap():
    assert rect_size_for((1414, 1000, 3), 1024) == (1024, 1448)  # A4, 1:sqrt(2)
    width, height = rect_size_for((1600, 1131, 3), 1024, multiple_of=16)
    assert width % 16 == 0 and height % 16 == 0


def test_corner_errors_are_per_corner_distances():
    errors = corner_errors([[0, 0], [3, 4], [0, 0], [0, 0]], np.zeros((4, 2)))
    assert errors.tolist() == [0.0, 5.0, 0.0, 0.0]
