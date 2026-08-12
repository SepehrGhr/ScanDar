# ScanDar

**A document scanner built from scratch: a phone photo of a page goes in, a clean scan comes out.**

Everyone has photographed a document and got back something tilted, dim, shadowed and barely
readable. Apps like CamScanner fix that by finding the page, flattening it, and enhancing it into
something scanner-like. This repository builds that machinery — two convolutional networks, designed
and trained from nothing, with no pre-trained weights and no imported architectures.

| | |
|---|---|
| **Enhancement network** | An encoder–decoder with skip connections that maps a degraded, rectified page to a clean, evenly lit scan. |
| **Corner detector** | Finds the four page corners in a raw photo — implemented *twice*, as direct coordinate regression and as heatmap regression, so the experiments decide which formulation wins rather than the author. |
| **End-to-end scanner** | The two composed: detect corners → warp → enhance. No human input. |

The interesting part is that **no training image was ever annotated**. Clean scans are
perspective-warped onto random backgrounds and put through a degradation pipeline built only from
OpenCV primitives — perspective, resolution loss, colour casts, illumination gradients, soft shadows,
blur, sensor noise, JPEG artefacts. The four points chosen for the warp *are* the corner labels, and
because the homography is known, warping the degraded photo back gives a pixel-perfect
(degraded, clean) training pair. The label generator and the data generator are the same function.

---

## Status

Built incrementally; this table is kept honest.

| Component | State |
|---|:--|
| Project layout, configuration, splits and sanity checks | **done** |
| Synthetic generator, degradation pipeline, datasets, frozen evaluation sets | **done** |
| Enhancement network: architecture, training, loss ablation, evaluation | not started |
| Real-photo study: OCR readability against a commercial scanning app | not started |
| Corner detection: coordinate regression vs heatmap regression, head to head | not started |
| Dropout study: does it close the synthetic-to-real gap? | not started |
| The end-to-end scanner | not started |
| Demo app, figures, written report | not started |

No results are claimed here yet — this section fills in with numbers and figures as the work lands.

## Quickstart

```bash
conda env create -f environment.yml && conda activate scandar
pip install -e ".[dev]"

python scripts/prepare_data.py     # cache the scans, write the split manifest
python scripts/freeze_eval_sets.py # generate the frozen synthetic val/test sets
python scripts/preview_synth.py    # render generated samples to look at
python scripts/sanity_checks.py    # verify the environment, the data and the generator
pytest                             # unit tests
```

Training on Colab instead of locally: run `notebooks/00_colab_bootstrap.ipynb`. It points
`SCANDAR_DATA` and `SCANDAR_OUT` at Drive, and everything else runs unchanged.

## Layout

```
configs/      experiment configs; every run is a file, never an edited constant
data/         scans, real photos, backgrounds, annotations   (see data/README.md)
docs/         how each part works, without reading the code   (see docs/README.md)
src/scandar/  the package
scripts/      prepare_data · freeze_eval_sets · preview_synth · sanity_checks · (more to come)
notebooks/    Colab bootstrap, then one notebook per part of the study
tests/        unit tests
reports/      figures, tables and the written report
```

[`docs/`](docs/README.md) explains the machinery in prose: the
[conventions](docs/conventions.md) everything rests on, the
[synthetic generator](docs/synthetic-generator.md), the
[degradation pipeline](docs/degradation-pipeline.md), and the
[datasets and splits](docs/datasets-and-splits.md).

The assignment brief names three files specifically; here is where they are:

| Brief | File |
|---|---|
| "implemented … in the `model.py` file" | `src/scandar/model.py` |
| "the implementation will be contained in `train.py`" | `src/scandar/train.py`, runnable from the root as `python train.py` |
| "implement the evaluation step in the `evaluate.py` file" | `src/scandar/evaluate.py`, runnable as `python evaluate.py` |

## Decisions worth knowing

**The enhancement network trains on patches, not whole pages.** An A4 page squeezed into 256×256
turns a pen stroke into less than one pixel, and no loss function recovers that. Instead the network
trains on 256×256 crops of pages rectified at 1024×1448, and — being fully convolutional — restores
a whole page at full resolution at inference time, in overlapping tiles.

**The split is by source scan, not by generated sample.** Two degraded versions of the same page must
never land on opposite sides, or the test score stops measuring generalisation. 50 scans → 40/5/5,
and the background photos are split the same way so a surface seen in training never reappears in
evaluation.

**Validation and test sets are frozen.** The dataset invents a fresh sample on every `__getitem__`,
so a naive implementation would score every epoch on different images and the validation curve would
measure the dice as much as the model. Val and test are generated once from a fixed seed and written
to disk.

**The page is composited, not pasted.** Its mask is feathered and it casts a soft shadow on the
surface underneath. Without those, the page meets the background at a perfectly clean one-pixel step
that no camera produces — and a corner detector will happily learn to find *that* instead of learning
what a page looks like. For the same reason the scan is resampled down to its size on the canvas with
an averaging filter before it is warped, rather than letting `warpPerspective` point-sample a
1600 px scan into a 700 px page and alias the text.

**The two tasks are trained on deliberately different worlds.** The corner detector also sees
coloured and dark page stock, a second sheet of paper underneath, and pages that will not lie flat —
all cases the real photos contain. The enhancement network sees none of them, because its target *is*
the clean scan, and a tinted input paired with a flat white target would ask it to invent a colour
correction it was never shown how to derive. The code strips those options for the enhancement task
structurally, so a shared config file cannot switch them on by accident.

**The real photos are never trained on, and never degraded.** They arrive degraded by reality. They
are the only preview of what the graders will hand the model on presentation day.

**The first version of every model carries no regularisation** — no dropout, no weight decay — so the
dropout study later isolates the one variable it is about.

## Data

19 real phone photos, deliberately varied in lighting, viewpoint, background and camera shake, plus
50 clean scans that become an unlimited synthetic training set. Full provenance, and the rules about
what may be trained on, are in [`data/README.md`](data/README.md). Image files are not tracked by
git; the layout, the annotations and the split manifest are.

## Credits

Course project for a Convolutional Neural Networks course, directed by Arshia Akbari Alagha.
Clean document scans provided by the teaching staff; real photos, corner annotations and background
images by the author.

Licensed under the MIT License.
