# Conventions

Six rules the rest of the project assumes without restating. Each one is here because breaking it
fails *silently* — no exception, no obviously wrong picture, just numbers that quietly mean something
other than what they claim.

## 1. Corner order is TL, TR, BR, BL. Everywhere.

Every quadrilateral entering the project goes through `geometry.order_corners` once and is never
reordered again. Labels that mix up top-left and bottom-right across images break both the evaluation
metric and the rectification, and the brief warns about it twice *(brief §1.2, §7)*.

`order_corners` sorts the four points by angle around their centroid, which fixes the *winding* for any
convex quad — in image coordinates, where y grows downward, ascending angle runs clockwise on screen —
and then rotates the ring so it starts at the corner nearest the origin. The naive "split by y, then
sort by x" recipe agrees with this for gently tilted pages and disagrees exactly where it matters, on
strongly rotated ones.

Ordering a page rotated by more than 45° is genuinely ambiguous: nothing in four bare points says
which edge is the top. The generator keeps rotation inside ±30° for that reason, and the real photos
are all upright.

## 2. Two coordinate systems, kept apart deliberately

This is the subtlest thing in the codebase and it is worth a minute.

**Homographies work on pixel indices.** `cv2.warpPerspective` fills output pixel `(x, y)` by sampling
the source at `H⁻¹·(x, y, 1)`, with integer pixel coordinates. So the four corners of a `W × H` image
are `(0,0)`, `(W-1,0)`, `(W-1,H-1)`, `(0,H-1)` — that is `geometry.rect_corners`.

**`cv2.resize` maps the continuous extent**, not the index range: a feature at index `i` lands at
`(i + 0.5)·s − 0.5` after a resize by factor `s`. That is `geometry.scale_points`.

The two differ by half a pixel, always in the same direction. Two consequences:

* Normalised corner labels are `(x + 0.5) / W`, not `x / W`. Only that form is *exactly* invariant
  under a resize, which is the entire point of normalising *(brief §2.2)*. `geometry.normalize_corners`
  and `denormalize_corners` are exact inverses of each other.
* When the generator shrinks a scan with `INTER_AREA` before warping it (see the
  [generator notes](synthetic-generator.md)), it moves the source corners with `scale_points` rather
  than reusing the image's own corners. Skipping that step costs about 0.1 px of systematic drift
  between the degraded input and the clean target — small, but sitting directly on top of the
  sub-pixel alignment the enhancement pair depends on.

`geometry.resize_with_corners` does both halves of a resize in one call, so there is no way for a
caller to do only one of them.

Measured alignment between the rectified input and the clean target, over the real scans: **0.01 px
mean, 0.03 px worst case**. `scripts/sanity_checks.py` reports it on every run.

## 3. Images are RGB uint8 HWC

OpenCV's native order is BGR. The conversion happens exactly once, in `io.imread_rgb` /
`io.imwrite_rgb`, at the I/O boundary — every other module can then assume RGB and stop thinking about
it. The degradation pipeline works in 8 bits too, for reasons covered in
[its own notes](degradation-pipeline.md).

Tensors are the exception, and only at the very edge: `datasets.to_tensor` produces float32 **CHW** in
`[0, 1]`, which is what the models consume.

## 4. Randomness comes from a key, never from global state

`seed.rng_for(*keys)` returns a NumPy generator derived from a *stable* hash of whatever identifies the
sample. Python's builtin `hash()` is salted per process — `hash("a")` differs between interpreter
runs — so it cannot be used for this, and `seed.stable_hash` uses BLAKE2b instead.

Nothing in the generator reads the global random state. Three things depend on that:

* the frozen evaluation sets regenerate **byte-identically**, which `sanity_checks.py` verifies by
  re-encoding a few samples and comparing them to the stored files;
* a run interrupted by a Colab timeout resumes onto exactly the samples it would have seen;
* a dataloader with eight workers cannot hand back the same "random" degradation eight times.

The one exception is `degrade.gaussian_noise`, which uses `cv2.randn` because it is twice as fast over
five million values and OpenCV is what the brief asks the pipeline to be built from. OpenCV's generator
is global state, so it is re-seeded from the sample's own generator immediately before use — the noise
field stays a function of the sample key and nothing else.

## 5. Split by source scan, never by generated sample

Two degraded versions of the same page must never land on opposite sides of a split, or the test score
stops measuring generalisation and starts measuring memorisation — and nothing in the training logs
would show it. 50 scans become 40 train / 5 validation / 5 test, recorded in `data/splits.json`.

Background photos are split the same way, so a surface the model trained on does not reappear in
validation or test. Scans are permuted *before* backgrounds, so collecting more surfaces later and
re-running `prepare_data.py` cannot quietly move a scan from train to test.

## 6. `data/real/` is evaluation-only

The 19 real phone photos and the commercial reference scans are never trained on, and the degradation
pipeline never touches them — they arrive degraded by reality *(brief §2.3)*. They are the only
preview of what the teaching staff will hand the model at the presentation.
