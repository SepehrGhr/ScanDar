"""Backgrounds to composite documents onto.  *(Phase 1)*

Two sources, mixed:

* **real** background-only photos from ``data/backgrounds/`` — carpet, wood desk,
  marble, tile, cluttered table. Split into train and held-out groups so a surface
  the model trained on never reappears in validation or test.
* **procedural** textures generated with NumPy and OpenCV — wood grain, woven
  fabric, marble veining, tiling and fractal noise — for variety beyond what one
  apartment can supply.

Backgrounds may be flipped and rotated freely; documents may not.
"""
