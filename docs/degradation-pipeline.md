# The degradation pipeline

`src/scandar/degrade.py`

In this project augmentation and dataset generation are the same thing: the richness of this pipeline
directly determines what the enhancement network can fix. A model that has never seen a shadow during
training will not remove one at test time *(brief §4)*.

**The brief forbids third-party augmentation libraries** — no albumentations, no kornia. Every
transform here is built from OpenCV primitives and NumPy.

```python
from scandar.degrade import DegradationConfig, degrade
from scandar.seed import rng_for

photo, params = degrade(clean_composite, rng_for("demo"), DegradationConfig())
params["random_blur"]        # {'kind': 'motion', 'length': 14, 'angle_deg': 85.5}
params["soft_shadows"]       # {'count': 2, 'shapes': [{'kind': 'arm', ...}, ...]}
```

Every stage takes an RGB uint8 image and a NumPy generator, and returns the image **together with the
parameters it sampled**. That is not decoration: the step-by-step figure, the frozen-set manifest and
every debugging session come out of it for free, and it means a sample can be described exactly after
the fact.

## The nine stages

Applied in the brief's §4.3 order. Every parameter is randomised within a range on every call —
*"a model trained on one shadow direction learns that shadow direction, not shadows"*.

| # | Stage | What it models | How |
|---|---|---|---|
| 0 | *(the perspective warp)* | the photo was taken at an angle | done by the [generator](synthetic-generator.md), because it also produces the labels |
| 1 | `downscale_upscale` | distance and limited sensor resolution | shrink by 2–4× with `INTER_AREA`, enlarge back with linear or cubic |
| 2 | `brightness_contrast` | a brighter or dimmer, flatter or harsher room | one 256-entry lookup table |
| 3 | `color_cast` | warm tungsten vs cool daylight | independent gains on red and blue, again as a table |
| 4 | `illumination_gradient` | the shape of the light in the room | multiply by a smooth random field |
| 5 | `soft_shadows` | things between the lamp and the page | blurred polygons, half-planes and arms |
| 6 | `specular_highlight` | a lamp reflecting off glossy paper | a blurred ellipse, screened rather than added, so it saturates towards white instead of clipping into a flat disc |
| 7 | `random_blur` | camera shake, imperfect focus | Gaussian, or motion blur from a rotated line kernel |
| 8 | `gaussian_noise` | sensor noise | `cv2.randn`, independent per channel |
| 9 | `jpeg_recompress` | phones do not store raw sensor data | `imencode`/`imdecode` at quality 30–80 |

Two stages deserve more than a table row.

### Illumination gradients

The characteristic defect of a real document photo, and the thing the enhancement network mostly
exists to undo — so it is worth more than one formula. `illumination_field` samples one of three
shapes, because a room offers all three:

* **grid** — a 3×3 or 4×4 random field blown up with cubic interpolation. Several light sources and
  reflections; a handful of control points is all it takes, because real room lighting varies over the
  scale of a whole page, not pixel to pixel.
* **linear** — a ramp in a random direction. A lamp off to one side.
* **radial** — falloff from a random centre. A lamp overhead.

Half the time a vignette is multiplied on top.

### Shadows

`shadow_field` draws 0–3 shapes into a mask, blurs each with σ between 10 and 60 (occasionally 1–4,
for a hard-edged shadow), and combines them with `maximum` rather than by adding — two shadows
overlapping should darken the page like two shadows, not like a hole. Three families, all present in
this project's own test photos:

* **blob** — a 3–6 vertex polygon. Something on the desk.
* **edge** — a half-plane. Something large and straight, off-frame.
* **arm** — two thick segments and a circle at the far end. The photographer reaching over the page,
  which dominates the real set.

Wide blurs are expensive, so `blur_mask` draws and blurs the mask at reduced resolution and scales it
back up. A soft shadow is smooth by definition, so the result is indistinguishable from the
full-resolution blur and an order of magnitude cheaper — which matters when it runs once per shadow,
per sample, inside a dataloader worker.

## Severity

The brief warns in both directions: too little degradation and the task is trivial; too much and *"the
text is destroyed entirely, leaving the model nothing to recover"*.

`DegradationConfig` holds every range, and its defaults **are** the `medium` reference.
`severity: mild | medium | hard` stretches all of them at once around the same centres, so the severity
study is a one-line change and every range moves together. Gains near 1.0 stretch around 1.0;
magnitudes stretch from 0; JPEG quality counts down from 100 rather than up from 0, with a floor,
because quality 0 is not a photo any more.

A config file may also pin any individual range, which then becomes the reference that severity
scales. Unknown keys raise rather than being ignored — a typo that silently keeps the default is
exactly the kind of bug that makes an ablation table meaningless.

`scripts/sanity_checks.py` measures the result rather than trusting it: the mean Sobel magnitude of the
degraded input over that of the clean target. Over the shipped ranges that sits near 0.21 (0.30 mild,
0.18 hard), and a sample below 0.04 fails the check.

## Why the pipeline works in 8 bits

A phone stores 8 bits per channel, so every stage here models something that happens to 8-bit data;
carrying float32 through the chain would buy precision that the final JPEG discards anyway. Working in
uint8 also lets the point operations collapse into `cv2.LUT` and the multiplicative ones into
`cv2.multiply` with saturation — which took a sample from 450 ms to 140 ms. The generator runs once per
training sample, forever, so that is not a micro-optimisation.

Concretely: brightness, contrast and colour cast are all functions of a single pixel value, so each is
*exactly* a 256-entry table (`apply_lut`), evaluated 256 times instead of five million. Illumination
and shadows are per-pixel multiplications, applied through `apply_gain`.

## What the pipeline deliberately does not do

**No flipping.** Mirrored text is not something a document scanner should ever learn to "restore"
*(brief §4.1)*. Backgrounds may be flipped freely; the document may not.

**No photometric degradation of the target.** Only the input is degraded. The geometric part is
inverted *exactly*, using the known homography, before the degraded image is paired with the clean
target — see [conventions](conventions.md#2-two-coordinate-systems-kept-apart-deliberately) for how
the sub-pixel part of that is kept honest.

## Seeing it

`python scripts/preview_synth.py` writes `outputs/previews/degradation_steps.jpg`: the composite
followed by one panel per stage, so each stage's contribution is visible on its own. It is produced by
`degrade(..., collect_steps=True)`, which costs nothing when it is off.
