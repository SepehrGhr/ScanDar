"""Inference pipelines for unseen data.  *(brief §3.4, §5.1 and §7)*

Three entry points, matching the three deliverables:

``enhance_document(image)``  — brief §3.4
    Takes an unseen *rectified* document image: preprocess, run the network
    fully-convolutionally in overlapping tiles with cosine blending (so a whole
    page can be restored at full resolution by a model trained on 256x256
    patches), resize back to the original dimensions, return 8-bit output.

``detect_corners(photo)``  — brief §5.1
    Takes an unseen *raw* photo: preprocess, predict the four corners with the
    better of the two detectors, map the coordinates back to the original
    resolution, return them ordered with an overlay for visualisation. Falls back
    to a classical Canny + contour + ``approxPolyDP`` detector if the predicted
    quad comes out degenerate, and records which path ran.

``scan_document(photo)``  — brief §7, the bonus
    The two above, composed: photo in, clean scan out, no human input. Corners ->
    homography -> high-resolution rectification -> enhancement. Corner ordering
    matters here: a permuted quad flips or rotates the page.
"""
