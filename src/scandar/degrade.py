"""The degradation pipeline — OpenCV and NumPy only.  *(Phase 1, brief §4)*

The brief is explicit: no third-party augmentation libraries. Every transform here
is built from OpenCV primitives and course techniques.

Applied in the brief's §4.3 order, each stage randomising *every* parameter within
a range and returning the values it sampled, so the step-by-step figure and any
debugging come for free:

1. ``downscale_upscale``    — 2-4x, simulating distance and sensor limits
2. ``brightness_contrast``, ``color_cast`` — light source and time of day
3. ``illumination_gradient``, ``soft_shadows`` — the characteristic defect of real
   document photos; the shadow shapes include the elongated photographer-arm form
   that dominates this project's real test photos
4. ``random_blur``          — Gaussian, or motion blur from a rotated line kernel
5. ``gaussian_noise``
6. ``jpeg_recompress``      — quality 30-80 via ``imencode``/``imdecode``

Photometric degradations touch the **input only**, never the target. Deliberately
no flipping: mirrored text is not something a document scanner should learn to
"restore".
"""
