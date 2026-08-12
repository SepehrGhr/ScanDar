# The synthetic generator

`src/scandar/synth.py` · `src/scandar/backgrounds.py` · `src/scandar/geometry.py`

The training set is never annotated. Four points are *chosen* in a background image, a clean scan is
warped onto them, and those four points **are** the corner labels — pixel-perfect, at zero annotation
cost. Because the homography is known, the degraded photo can be warped back to give a perfectly
aligned (degraded input, clean target) pair for the enhancement network. The label generator and the
data generator are the same function *(brief §1.3)*.

```python
from scandar.config import load_config
from scandar.seed import rng_for
from scandar.synth import build_sources

sources = build_sources(load_config("configs/base.yaml"), "train", task="corner")
sample = sources.compose(rng_for("demo", 0))

sample.photo            # uint8 RGB — the degraded composite
sample.corners          # (4, 2) float32, TL TR BR BL, in canvas pixels
sample.rectify((1024, 1448))          # -> (degraded input, clean target)
sample.random_patch(rng, 256, (1024, 1448))   # -> one 256² pair and its box
```

## One sample, step by step

`compose_sample()` is the whole story. Everything below happens in that one function.

### 1. Choose the canvas

`1152 × 1536` by default — the same 3:4 aspect as the real phone photos (1920 × 2560), small enough to
warp on the fly. With probability `landscape_prob` the dimensions swap, because one of the real photos
was taken with the phone held sideways. Keeping the synthetic canvas at the *same* aspect ratio as the
real photos matters: both get squashed into a square 256 × 256 input for the corner detector, so the
distortion is identical for training and for testing.

### 2. Choose the background

`backgrounds.sample_background` draws either a real background photo or a procedural texture
(`procedural_prob`, default 0.25).

Real photos come from `BackgroundBank`, which decodes each one once and keeps it — decoding a phone
JPEG costs more than everything else in a sample put together. `random_view` then takes a random crop
at a random scale, with rotation in quarter turns and free flipping. **Backgrounds may be flipped;
documents may not** — mirrored text is not something a scanner should learn to restore.

Procedural textures are five generators built from NumPy and OpenCV: `wood_grain`, `marble`,
`woven_fabric`, `tiled_floor` and `painted_surface`, all layered on `fbm` — fractal noise made by
summing random grids at doubling frequency through `cv2.resize` with cubic interpolation. They are not
a stand-in for the real photos, they are insurance: twenty surfaces is a small vocabulary and the
fifteen in the training split are lit by the same two lamps, so a detector that has only ever seen
those can learn *this desk* instead of *a page on a desk*.

Two details that matter more than the pattern generators themselves:

* **Textures are generated at half size** and enlarged once. Every surface is out of focus behind an
  in-focus page and the composite then goes through a 2–4× downscale, so detail at full resolution
  cannot reach the model — while generating it costs four times the work.
* **Colour is deliberately muted.** Three independent uniform draws per channel produce saturated
  magenta far more often than any real desk; `painted_surface` samples a grey plus a small chroma
  offset instead, and every texture then gets a narrow tint, a modest exposure and a partial pull
  towards its own luma. The thing that gives a generated surface away is never its pattern — it is
  being more colourful than anything in the room.

### 3. Choose where the page lands

`sample_quad` builds a destination quadrilateral by **rejection sampling**:

1. an axis-aligned rectangle at `page_scale` of the canvas height, in the scan's own aspect ratio;
2. one edge squeezed by `keystone` — a page photographed from an angle rather than from overhead;
3. rotated by up to `rotation_deg` (±30°, the limit past which "top-left" stops being defined);
4. each corner nudged independently by up to `corner_jitter` of the page size, which is what turns a
   trapezoid into a general perspective quad;
5. the centre displaced by `center_jitter`.

The result goes through `geometry.quad_problem`, which rejects anything concave, too small, too thin,
off the edge of the canvas, or with an interior angle below 20°, and **returns the reason** rather
than a bare `False` — when the generator suddenly rejects nine tries out of ten, the reason says
whether the pages are too small or the tilt is too extreme. Up to `max_tries` draws; if every one is
rejected the generator falls back to a centred, unrotated page rather than raising mid-epoch.

Rejection sampling is used because it is far easier to write, and to explain, than trying to sample
directly from the set of valid quads.

### 4. Composite the page onto the background

Three things here that a naive paste does not do.

**Pre-shrink the scan.** `warpPerspective` interpolates but does not average, so shrinking a 1600 px
scan into a 700 px page with it would alias every pen stroke into moiré that no phone camera would
ever record. The scan is resampled to roughly its size on the canvas with `INTER_AREA` first — and its
source corners are moved with `geometry.scale_points`, because `resize` and `warpPerspective` use
[different pixel conventions](conventions.md#2-two-coordinate-systems-kept-apart-deliberately).

**Feather the edge, and use premultiplied alpha.** `warpPerspective` already returns the page
multiplied by its own coverage — outside the page it faded to the zero border — so colour and coverage
are blurred with the *same* kernel and the premultiplied result is **added**:

```python
alpha = cv2.GaussianBlur(mask / 255.0, (0, 0), feather)
premultiplied = cv2.GaussianBlur(warped, (0, 0), feather)
clean = cv2.add(apply_gain(background, 1.0 - alpha), premultiplied)
```

Blurring only the mask and then multiplying the colour by it a second time would darken every edge
pixel — a one-pixel black outline around the page. Wrong, and a gift to a corner detector looking for
a shortcut.

**Cast a shadow.** The page mask is shifted, blurred and used to darken the surface underneath.
Without a feathered edge and a drop shadow, the page meets the background at a perfectly clean
one-pixel step that no camera produces, and a detector will happily learn to find *that* instead of
learning what a page looks like.

### 5. Degrade

The composite goes through the [degradation pipeline](degradation-pipeline.md), which returns the
degraded photo and every parameter it sampled.

## Deriving the training pairs

`compose_sample` returns a `Sample`, not a dict of finished tensors. What the enhancement network sees
is derived from it on demand:

| | |
|---|---|
| `sample.rectify(size)` | the whole page flattened: `(degraded input, clean target)` |
| `sample.rectify_patch(box, rect_size)` | one crop of the same pair, warped straight out of the canvas |
| `sample.random_patch(rng, ...)` | a random box, preferring one with ink on it |

Deriving rather than storing matters twice over. A whole rectified page costs 4 MB that patch training
would throw away — and, more importantly, input and target come out of the **same homography chain**,
so they are aligned by construction rather than by agreement between two pieces of code:

```
target :  scan  --H-->  canvas  --H_rect-->  rect     (composed: H_rect @ H)
input  :  photo         --------H_rect-->    rect
```

For a patch, a translation is composed onto the end of that chain and only the patch is warped — about
twenty times less work per sample than flattening a whole page and slicing it, and exact, because it
is the same matrix with a translation on it.

`random_patch` draws up to `patch_tries` boxes and keeps the first with enough contrast. A uniformly
random crop of an A4 page is very often blank margin; blank patches are not useless — flattening the
paper to an even white *is* half the job — but a training set made mostly of them spends its capacity
on the easy half. This biases towards text without ever excluding blanks.

## The two tasks see deliberately different worlds

Some of the real photos are not a flat white rectangle: there is a blue notebook cover, a dark printed
card, a page resting on other paper, and a spread that will not lie flat. The corner detector has to
survive all of them, so `SynthOptions.for_corners()` enables four extras:

| Option | What it adds |
|---|---|
| `page_tint_prob` | coloured paper stock — cream, pink, blue |
| `page_dark_prob` | a dark printed card rather than paper |
| `distractor_prob` | a second sheet **underneath** the page, so it can peek out without ever occluding a corner and making the label ambiguous |
| `curl_prob` | a sinusoidal `cv2.remap` displacement that vanishes on the border, so the page bulges but its four corners do not move and the labels stay exact |

The enhancement network sees **none** of them, because its target *is* the clean scan: a tinted or
bulged input paired with a flat white target would ask the model to invent a colour correction and a
geometry it was never shown how to derive. One `synth:` block in the config serves both tasks, so
`SynthOptions.from_config` strips those four fields again *after* applying the config rather than
merely defaulting them off — which is not paranoia, it is what the first version of this code got
wrong, visibly, in the difference maps.

## Verification

`scripts/preview_synth.py` writes five sheets into `outputs/previews/`, which is what the brief asks
for before any model is trained *(brief §4.4)*:

| Sheet | Answers |
|---|---|
| `composites.jpg` | do the corner labels land on the page? |
| `degradation_steps.jpg` | what does each degradation stage actually do? |
| `rectified_pairs.jpg` | do input and target line up? (each panel prints its measured shift) |
| `patches.jpg` | is the training pair legible and aligned at patch scale? |
| `spot_the_fake.jpg` | put six synthetic photos next to six real ones — can a stranger tell instantly? |

`scripts/sanity_checks.py` turns the same questions into numbers: valid and ordered quads, alignment
under half a pixel, normalised corners inside `[0, 1]`, corners surviving a resize, same key giving the
same photo, and enough of the target's edge energy surviving the degradation to leave something to
restore.

## Cost

One composited photo costs about 140 ms (enhancement options) to 180 ms (corner options) on the
development machine; a 256² patch cut out of one costs about 2 ms. That ratio is why
`SyntheticEnhanceDataset` cuts several patches from each photo — see
[datasets](datasets-and-splits.md#throughput).
