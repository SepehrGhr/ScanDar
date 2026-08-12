"""Evaluation metrics.  *(Phases 2 and 4, brief §3.3 and §5)*

Restoration:

``psnr`` / ``ssim``
    Reported on all three synthetic buckets, alongside the "do nothing" baseline —
    the metrics of the degraded input itself. If the model's scores are not clearly
    above that line, it is not earning its parameters.

Corner detection:

``corner_error``
    Mean Euclidean distance between predicted and true corners, in pixels and as a
    percentage of the image diagonal so the number survives a resize.
``pck``
    The stricter success metric: the fraction of images where *all four* corners
    land within a threshold. Swept over thresholds, it becomes the curve that
    settles the regression-vs-heatmap argument.
``quad_iou``
    Overlap of the predicted and true page quads — closest proxy for how much a
    corner error will cost the rectification downstream.
"""
