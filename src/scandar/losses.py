"""Loss functions, implemented from scratch.  *(Phase 2, brief §3.2)*

A plain pixel-wise MSE is known to produce blurry restorations, and blur is
precisely the enemy when the goal is readable text. The answer this project takes
is a combination, in the spirit of the image-restoration literature:

    L = 1.0 * L1  +  0.5 * (1 - MS-SSIM)  +  0.25 * L1(Sobel(x), Sobel(y))

L1 keeps the intensities honest without MSE's blur-favouring averaging, MS-SSIM
scores structure the way an eye does, and the gradient term puts the penalty
exactly where legibility lives — the edges of the strokes.

``SSIM`` / ``MSSSIM``
    Gaussian-window SSIM (11x11, sigma 1.5) and its five-scale variant, written
    here rather than imported, since the same maths is needed for the evaluation
    metric anyway.
``sobel_loss``
    L1 between Sobel edge maps, via fixed convolution kernels.
``CombinedRestorationLoss``
    The weighted sum above, with weights read from the config so the ablation
    (MSE / L1 / L1+SSIM / L1+SSIM+gradient) is a matter of swapping config files.
``heatmap_mse`` / ``coord_l1`` / ``wing_loss``
    Corner-detection losses for approaches A and B.
"""
