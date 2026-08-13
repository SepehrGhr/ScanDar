# Predictions, written before the runs

The brief asks for this explicitly *(brief §5.1)*: **"Think about *why* the two approaches might
behave differently before running the experiments, and write your prediction down. Was your
prediction right?"** A prediction written after seeing the results is not a prediction, so this file
is committed before either corner detector has been trained, and git holds the timestamp. Nothing
here may be edited once the numbers exist — the verdict goes in the report, next to what was
predicted, whichever way it falls.

**State of the world at the time of writing.** The enhancement network is trained and evaluated. Both
corner detectors are implemented and neither has run a single training step. No corner metric has
been computed on anything.

---

## 1. Direct coordinate regression vs heatmap regression *(brief §5)*

### The setup being predicted

Both detectors see the identical sample: one composited photo resized to 256×256, corners normalised
to [0, 1] in TL, TR, BR, BL order. `CornerRegNet` is a five-stage convolutional encoder flattened
into two fully connected layers emitting eight numbers. `CornerHeatNet` is the same encoder–decoder
family as the enhancement network, emitting four 128×128 Gaussian heatmaps read back with a
soft-argmax. Same data, same schedule, same optimiser; the formulation is the only difference.

Errors below are **mean Euclidean corner error in the network's 256×256 space**, which is also
quoted as a percentage of that space's 362-pixel diagonal so it survives any resize.

### The prediction, in one line

**Heatmap regression wins on accuracy by roughly a factor of two, converges several times faster, and
loses on exactly one axis: the rare sample where two page-like objects are in frame.**

### Why — the mechanisms this rests on

**1. The heatmap keeps the problem where the evidence is.** A corner is a local visual event: two
page edges meeting over a background. In a heatmap network the output pixel at (x, y) is spatially
aligned with the input pixel at (2x, 2y), so "there is a corner here" is a local decision made on top
of local evidence, and translation equivariance comes free from the convolutions. Direct regression
throws that alignment away at the flatten: position now has to be encoded in the *values* of a
16 384-element vector, and a fully connected layer has to learn the entire map from feature index to
coordinate from scratch. Nothing in the architecture knows that moving the page ten pixels right
should move the answer ten pixels right.

**2. The supervision is denser by four orders of magnitude.** Coordinate L1 hands back eight scalars
per sample. Heatmap MSE hands back 4 × 128 × 128 = 65 536 supervised values, each one telling a
specific location whether a corner is there. Per gradient step the heatmap network is being told
vastly more, which is a convergence-rate argument before it is an accuracy argument.

**3. Regression's one structural advantage is sub-pixel precision.** Its output is continuous by
construction. The heatmap's is not: a hard argmax at 128² quantises to 2 pixels of 256-space, which
is already comparable to the error being predicted. Soft-argmax removes the quantisation in
principle, but it is an expectation over the whole map, so it is pulled by the tails of the
distribution and by any second mode. **If heatmaps lose, this is where they lose it**, and the
diagnostic is that hard-argmax and soft-argmax errors will be close to each other rather than the
latter being clearly better.

**4. The two formulations fail differently, and the difference is predictable.** Fully connected
layers regress toward the mean of what they have seen, so regression's failure is quiet: an unusual
viewpoint gets pulled toward a plausible average quad. It will look sane and be wrong by a moderate
amount, everywhere, on the same sample. A heatmap fails loudly and locally: one corner's map goes
diffuse or bimodal while the other three stay sharp. That matters because the generator deliberately
puts a **second sheet of paper in ~20% of corner samples**, and a distractor is exactly the input
that produces two modes — and soft-argmax over two modes returns a point *between* them, which is on
neither page. Expect the heatmap detector's worst samples to be distractor samples, and expect them
to be worse than regression's worst samples.

**5. Therefore the strict metric will flatter regression relative to the mean.** PCK — all four
corners inside a threshold — is decided by a sample's *worst* corner, so a heavier tail costs more
there than in a mean. The heatmap detector should still win it, by a smaller margin than the mean
error suggests.

### The numbers being predicted

On the frozen synthetic test bucket, which is the one place both get an exact label:

| | `CornerHeatNet` | `CornerRegNet` |
|---|---:|---:|
| mean corner error, px at 256 | **2.0 – 4.0** | **5.0 – 9.0** |
| the same, % of the diagonal | 0.55 – 1.1 % | 1.4 – 2.5 % |
| PCK, all four corners within 2% of the diagonal (7.2 px) | ≥ 0.85 | 0.35 – 0.65 |
| mean quad IoU against the true page | ≥ 0.97 | ≥ 0.93 |
| worst 5% of samples, mean error | 8 – 20 px | 12 – 25 px |

And three statements that are not about the final number:

* **Convergence.** The heatmap detector reaches regression's *final* error inside the first fifth of
  its run. Regression's validation curve is the slower and the noisier of the two.
* **Both are good enough to rectify with.** Even the pessimistic end of regression's range is under
  2.5% of the diagonal, and quad IoU stays above 0.93, so neither detector is predicted to break the
  end-to-end scanner. The comparison decides which one to ship, not whether shipping is possible.
* **Ease of training.** Regression is the fiddlier of the two: 8.4 M of its parameters are in a
  single fully connected layer fed by a flattened feature map, and its output must be squashed into
  [0, 1], so it is the one more likely to need a learning-rate change to behave. If either run needs
  its schedule adjusted, it will be that one.

### On the real photos *(when the annotations land)*

The Roboflow keypoint export has not arrived, so this half cannot be measured yet. The prediction is
recorded anyway, because it is the more interesting one:

* Both detectors lose **at least a factor of two** of accuracy going from synthetic to real. The
  synthetic pages are composited with a feathered edge and a drawn drop shadow onto flat backgrounds;
  real pages sit in real light with real contact shadows and occasionally will not lie flat.
* **The gap between the two narrows on real photos**, and may close. The heatmap advantage is largely
  an advantage in precision, and on real photos both will be dominated by a different error — finding
  the right *object* at all — which is where regression's global view of the frame is least
  disadvantaged.
* The specific real photos that will hurt: the bound notebook and the two-page spread (no single
  four-corner page exists in the frame), and the dark printed card (the page-versus-background
  contrast the detectors lean on is inverted).

### What would falsify this

Stated concretely, so the report cannot quietly reinterpret it afterwards:

* Regression matching heatmaps within 1 px of mean error would falsify mechanisms 1 and 2 — it would
  say the task is globally constrained enough (a rectangle, four corners, always in frame, always
  convex) that a learned global prior is nearly as good as local evidence.
* Heatmaps *losing* would most likely mean the 128² output resolution is the binding constraint,
  diagnosable by hard-argmax and soft-argmax scoring alike, and fixable by predicting at 256² or by
  adding a coordinate loss through the soft-argmax rather than by abandoning the formulation.
* Heatmaps winning the mean but **losing** PCK would confirm mechanisms 4 and 5 more strongly than
  the predicted outcome does, and would make the distractor samples the headline of the failure
  analysis.

---

## 2. Dropout, on all three models *(brief §6)*

Recorded here because the evidence for it already exists and the runs do not.

The enhancement baseline showed a **0.75 dB gap between its training and test buckets** and finished
with training and validation loss within 3% of each other. There is essentially nothing there for
regularisation to recover, because a generator that never repeats a sample gives the model nothing to
memorise. So:

* **Dropout buys approximately nothing on the synthetic metrics** for the enhancement network, and
  slightly *hurts* them — a network with capacity to spare, trained on unlimited data, pays dropout's
  variance cost without collecting its variance benefit.
* If dropout helps anywhere, it helps on the **real photos**, and for a reason that is not
  overfitting in the usual sense: the synthetic-to-real gap is a distribution shift, and a model
  forced not to rely on any single feature is less brittle when the features shift.
* The corner detectors are the more likely to benefit, and `CornerRegNet` most of all, because the
  fully connected layers are the one place in this project where a genuinely over-parameterised map
  is learned from a finite view of the data. That is also the classic place to put dropout, which is
  where it goes.

The brief asks specifically whether the gap between synthetic validation and real-photo test shrinks.
The prediction is: **for the enhancement network, marginally and possibly not measurably; for
`CornerRegNet`, yes, visibly.**
