"""Homographies, quadrilaterals and corner ordering.  *(Phase 1)*

Shared by both mandatory tasks: the synthetic generator picks a destination quad
and derives the corner labels from it, and the corner detector is scored against
those same quads.

Planned surface:

``order_corners(pts)``
    Canonicalise four points to TL, TR, BR, BL. Consistent ordering is not
    optional — mixed-up corners silently break both the evaluation metric and the
    rectification in the bonus part.
``homography(src, dst)`` / ``warp_points(H, pts)``
    Thin wrappers over ``cv2.getPerspectiveTransform`` / ``cv2.perspectiveTransform``.
``is_valid_quad(quad, canvas)``
    Rejection-sampling guard: convex, minimum interior angle, minimum edge length,
    inside the canvas with a margin, plausible area ratio.
``quad_iou(a, b)``
    Overlap between predicted and true page quads — the metric that actually
    predicts how good the rectification will be.
``rect_size_for(shape, out_width)``
    Output size that preserves the source page's aspect ratio.
"""
