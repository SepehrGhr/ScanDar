"""Synthetic sample generation.  *(Phase 1, brief §1.3 and §4)*

The key insight of the project: the training set never needs annotating. Choose
four random target points in a background image, warp a clean scan onto them, and
those four points *are* the corner labels — pixel-perfect, at zero annotation cost.
Because the homography is known, warping the degraded photo back also yields a
perfectly aligned (degraded input, clean target) pair. The label generator and the
data generator are the same function.

``compose_sample()`` returns, for one sample:

``photo``        the degraded composite — what the corner detector sees
``clean_photo``  the same composite before photometric degradation, for figures
``corners``      the four destination points, ordered TL, TR, BR, BL
``H``            scan -> canvas homography
``rect_input``   the degraded photo warped back flat — what the enhancer sees
``rect_target``  the matching crop of the clean scan — what the enhancer predicts
``params``       every degradation parameter that was sampled

Alignment between ``rect_input`` and ``rect_target`` is exact by construction, and
asserted in the sanity checks: a few pixels of drift would punish the model for
errors it did not make.
"""
