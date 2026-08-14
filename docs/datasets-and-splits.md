# Datasets, splits and frozen evaluation sets

`src/scandar/datasets.py` · `src/scandar/prepare.py`

## The splits

50 clean scans become **40 train / 5 validation / 5 test**, split by *source scan* and recorded in
`data/splits.json` with a fixed seed. Two degraded versions of the same page must never land on
opposite sides *(brief §2.3)*.

Background photos are split the same way — 15 training surfaces, 5 held out — so a carpet the model
trained on never turns up in a validation score. Both evaluation splits draw on the held-out
surfaces. The split is a *manifest*, not a directory move: one copy of every file, and the split stays
re-derivable with a different seed.

Scans are permuted before backgrounds, so collecting more surfaces later and re-running
`prepare_data.py` cannot quietly move a scan from train to test.

`prepare_data.py` also caches every scan downscaled to 1600 px on the long side, as **PNG**: these
images are the ground-truth targets, and re-encoding them as JPEG would bake fresh compression
artefacts into the very thing the network is asked to reproduce.

## The datasets

All three build on `Sources` — a scan bank, a background bank, the placement options and the
degradation config for one split. `build_sources(config, split, task)` assembles it from a loaded
config and the split manifest.

### `SyntheticEnhanceDataset`

```python
{"input":  float32 (3, 256, 256) in [0, 1],     # degraded, rectified
 "target": float32 (3, 256, 256) in [0, 1],     # the clean scan, same pixels
 "scan": "17", "box": (x, y, size), "index": i}
```

`mode="patch"` cuts a 256 × 256 crop out of the page rectified at 1024 × 1448; `mode="page"` returns
the whole rectified page for evaluation.

**Patches, not whole pages, is the single most important architectural call in the project.** An A4
page squeezed into 256 × 256 leaves a pen stroke under a pixel wide, and no loss function recovers what
the resize threw away. The network is fully convolutional, so it trains on crops and infers on whole
pages in overlapping tiles.

### `SyntheticCornerDataset`

```python
{"image":      float32 (3, 256, 256) in [0, 1],
 "corners":    float32 (4, 2) in [0, 1],        # normalised, TL TR BR BL
 "heatmaps":   float32 (4, 128, 128),           # one Gaussian blob per corner
 "corners_px": float32 (4, 2),                  # original photo pixels
 "size":       int32 (2,)}                      # the photo's (width, height)
```

Both formulations the brief asks for — direct coordinate regression and heatmap regression — train
from this one dataset on identical samples, so the comparison is between the models and nothing else
*(brief §5)*.

Heatmap peaks are placed at the *sub-pixel* position of the corner rather than at the nearest cell, so
a soft-argmax read back off the target reproduces the label instead of a quantised version of it.

The original size travels with every sample because predictions have to be mapped back to the photo
they came from, and reconstructing that afterwards is how corners end up scaled by the wrong factor.

### `SyntheticScanDataset`

```python
{"image":   float32 (3, 256, 256) in [0, 1],   # what the detector reads
 "source":  uint8   (3, 2560, 1920),           # what the differentiable warp resamples
 "corners": float32 (4, 2) in [0, 1],
 "target":  float32 (3, 256, 256) in [0, 1],   # the clean scan, one window of the flat page
 "box":     int32 (3,)}                        # where that window sits, (x, y, size)
```

For the [end-to-end scanner](end-to-end-scanner.md) *(brief §7)*, which needs a sample carrying both
tasks' labels at once — neither of the two above does. It is a third view of the same `Sample`, so it
needed no generator work.

Two things are unusual here and both are deliberate. The large photo travels in **8 bits**, because a
float copy of a 1920×2560 canvas is 59 MB per sample through the loader and it is converted on the
GPU instead; and it is **not shrunk**, because the page would then be upsampled into the target,
which is the pathology `enhance_realistic` exists to avoid. There is no degraded input in the sample:
the model warps its own, out of the photo, through the corners the detector predicted.

### `FrozenSyntheticDataset`

Reads the validation and test samples generated once and written to disk. Serves either task — and
the scanner, which has no bucket of its own and borrows the enhancement one, because those photos are
the only ones whose restoration target is achievable. They carry no distractor sheet, so a detector
scores better there than on its own bucket; that caveat has to travel with the numbers.

## Preprocessing choices

**Corners are normalised to `[0, 1]` and always rescaled together with their image.** A corner label
that is not transformed with its image is a wrong label *(brief §2.2)*. Both halves happen inside
`geometry.resize_with_corners`, which exists as one function precisely so no caller can do half of it.
The half-pixel convention is covered in [conventions](conventions.md).

**Images stop at `[0, 1]` and are not standardised by a mean and a standard deviation.** The
restoration network's target *is* an image in `[0, 1]` behind a sigmoid, so its input belongs in the
same space; and with no pretrained weights anywhere in the project there is no external normalisation
to match. Copying a `transforms.Normalize` line whose constants would mean nothing here is worse than
saying why it is absent.

## Frozen evaluation sets

The dataset invents a fresh sample on every `__getitem__`, so a naive implementation scores every epoch
on different images and the validation curve measures the dice as much as the model. Validation and
test are generated **once** from a fixed seed and written to disk *(brief §2.3)*: 200 samples each,
about 72 MB.

Only the composited photo is stored. Everything else — the rectified input, the clean target, the
heatmaps — is *derived* from it using the recorded corners and the cached scan, exactly as the
on-the-fly datasets derive them. That keeps the frozen set to one file per sample, and it guarantees
that the evaluation path and the training path cannot drift apart, because they are the same code.

`manifest.json` carries, per sample: the corners, the source scan, the canvas size, and every
degradation parameter that was drawn — so "what was actually in the test set" is a question with an
answer.

Photos are stored as quality-96 JPEG rather than PNG, which matters not at all: the last stage of the
degradation pipeline is itself a quality 30–80 re-encode, and 400 PNGs of degraded photo texture would
be well over a gigabyte.

**The frozen set is checked, not trusted.** Because a sample is a pure function of its key,
`sanity_checks.py` regenerates a few of them, re-encodes at the same quality, and compares the bytes.
If the generator changes, that check fails loudly — which is the point: numbers measured on an old
frozen set are not comparable with numbers from a new one.

```
data/frozen/<task>/<split>/
├── manifest.json
├── photo_0000.jpg
└── ...
```

**One set per task, and this is not tidiness.** The corner detector is deliberately trained on a
harder world than the enhancement network: coloured page stock, a dark printed card, a second sheet
of paper peeking out from underneath, a page that will not lie flat. The enhancement network must
never see any of those, because its *target* is the flat clean scan — pair a tinted input with an
untinted target and you are asking the model to invert a colour cast it was never shown how to
derive, and to unbend a curl the rectification does not undo.

The first version of the frozen sets was generated once, with the corner options on, and used for
both tasks. About 40% of the enhancement validation samples were unmatched pairs, and the effect was
already visible in the do-nothing baseline: 15.0 dB on plain pages against 14.3 dB on tinted or
curled ones. On a *trained* model the gap would have been far larger, because the model can improve a
plain page and can do nothing at all about a tint. The sanity checks now refuse an enhancement set
containing either.

There is a frozen **train** bucket too. That sounds odd until you notice that the generator never
produces the same sample twice, so there is no set of files the model was trained on: "performance on
the training set" can only mean performance on the training *distribution* — the training scans and
the training backgrounds — and a fixed draw from it is exactly what measures that *(brief §3.3)*.

For the enhancement task the frozen pages serve two purposes. `mode="page"` gives whole rectified
pages at 1024 × 1448 for the results table; `mode="patch"` cuts deterministic crops for the per-epoch
validation curve, so that the training and validation curves on the brief's graph are the same
quantity rather than two different ones sharing an axis.

## Reproducibility

Every sample is derived from `rng_for(seed, task, split, epoch, index)` and nothing reads the global
random state. `set_epoch(e)` moves the dataset to a new stream, the way a `DistributedSampler` does —
call it before iterating, once per epoch.

## Throughput

Composing and degrading a 1152 × 1536 photo costs about 140 ms; cutting a 256 × 256 patch out of one
costs about 2 ms. Generating a whole photo per patch would leave the GPU idle almost all of the time.

So `SyntheticEnhanceDataset` groups consecutive indices: `patches_per_photo` of them share one
composited photo, cut at different places on the page. This **requires `shuffle=False`** in the
DataLoader — which is not a compromise, because every index invents a fresh random sample, so there is
no fixed order to break. With `shuffle=False` each worker receives a contiguous run of indices, which
is what makes the one-entry cache hit. Set it to 1 for one photo per patch, at four times the cost.

`Sources.warm()` decodes every scan and background up front, in the parent process, so forked
dataloader workers inherit them copy-on-write instead of each rebuilding the cache — worth calling
before handing the dataset to a DataLoader, especially without `persistent_workers`.

Measured on the development machine (16 cores), at batch 16, through the real training loop:

| | samples/s |
|---|---|
| data loader alone, no workers | 31 |
| data loader alone, 8 workers | 84 |
| the model's forward and backward alone, mixed precision | ~83 |
| the model's forward and backward alone, fp32 | ~26 |
| **loader and model together, 8 workers** | **~34** |

The loader does parallelise — an earlier measurement taken before there was a training loop to
profile suggested otherwise, and does not reproduce against one. What is left is two stages of
roughly equal cost competing for the same cores, which is why the combined figure is well under
either one alone. Mixed precision is not optional: it is a threefold difference in the model step by
itself. At ~34 samples per second, 24 000 steps at batch 16 is a little over three hours.
