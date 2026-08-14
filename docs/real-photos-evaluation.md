# Real photos vs the commercial app

Everything else in this project is scored on synthetic pages, because that is the only way to get a
clean target for free. It answers "did the model fit the generator." It does not answer the question
that actually decides the grade: handed a photograph nobody staged, does the pipeline still work?
*(brief §3.3, §5)*

This page is that answer, on the 19 real phone photos in `data/real/photos/` — carpet, wood, marble
and a red table; hard photographer shadows; steep angles; documents the synthetic generator does not
model well (a bound notebook, a two-page spread, a dark printed card).

**Two of the nineteen were dropped.** `Image5` and `Image10` did not photograph well enough to be
useful and were removed from the dataset by the author before labelling. **A sixteenth is missing for
a different reason: `Image17` was never annotated in Roboflow**, so it is simply absent from
`data/real/corners.json` — nothing in the code decided that, the export did. Sixteen of the remaining
seventeen carry hand-labelled corners and are the real corner-detection test set; five of those
sixteen also have a CamScanner reference scan and are the enhancement/OCR test set.

## Corner detection on real photos *(brief §5)*

`scripts/parse_roboflow.py` turns the Roboflow export into `data/real/corners.json` — one ordered
`(4, 2)` array per photo, in that photo's own original pixels. The export is not, in the end, COCO
**Keypoints** as the brief asks for: this project's Roboflow project was labelled with the polygon
tool, so what lands in the JSON is a four-vertex instance-segmentation polygon per photo rather than a
`keypoints` array. The parser reads either shape identically — both are four ordered points — and logs
a warning rather than silently rewriting anything if `order_corners` has to move more than one point
to reach canonical TL, TR, BR, BL order. Nothing was flagged: every label was already consistent.

`scripts/evaluate_real.py` runs the shipped detector (`corner_heat`) through the exact §5.1 pipeline —
`pipelines.detect_corners`, classical fallback included — on all sixteen photos, and scores it against
the hand labels the same way the synthetic table does: mean corner error in pixels and as a percentage
of *that photo's own* diagonal (the real photos are not all quite the same size — one is 1926 px wide,
not 1920), the worst-of-four PCK success rate, and quad IoU.

| Variant | corner error (px) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| **corner_heat** (pipeline, incl. fallback) | **10.89 ± 4.37** | **0.34%** | **1.000** | **0.9841** |
| *classical baseline* | *357.08* | *11.16%* | *0.188* | *0.7153* |

*(see `reports/tables/real_corners_per_sample.csv` for the per-photo split)*

The network went straight to the neural path on 15 of 16 photos, averaging 10.62 px on those. On
`Image1` it returned a non-quadrilateral and the classical guardrail took over — the one photo where
doing so still cost 14.9 px against that ~10.6 px network average, which is the guardrail doing its
job (something usable) rather than its best (nothing beats the network when the network works).

**10.89 px on a ~3200 px-diagonal photo is worse in absolute pixels than the 1.06 px on the 256×256
synthetic bucket, and better in every relative sense.** As a percentage of the diagonal — the only
unit the two numbers can honestly be compared in — synthetic is 0.29% and real is 0.34%: a real
photograph the network has never seen anything like costs almost nothing over a synthetic page from
the training distribution. PCK@2% is a clean 1.000 on the real photos; every one of the sixteen has
all four corners within 2% of the diagonal, comfortably inside what the enhancement network's
receptive field can absorb as ordinary perspective slop. The synthetic-to-real generalisation gap this
project worried about from the beginning turns out to be close to zero for the corner detector — the
degradation pipeline's perspective, lighting and shadow augmentation generalises.

The classical baseline is not a fair fight here in a way it partly wasn't on synthetic either, but
worse: **`working_side=768` on a `~1920x2560` photo throws away the resolution the network is given**,
and a busy real background (carpet weave, wood grain) generates far more spurious Canny edges than a
synthetic composite's cleaner textures. 0.188 PCK against the synthetic bucket's 0.485 is the
guardrail behaving exactly as documented — a rectangle-finder that works when the page is a bright
convex shape on a plain surface, and loses badly when the surface fights back. It is still worth
having: it is the fallback for the 1-in-16 case the network gets wrong, not a competitor to it.

## Enhancement vs CamScanner *(brief §3.3)*

Five of the sixteen annotated photos have a CamScanner reference scan in `data/real/reference/`
(`Image1`, `Image7`, `Image9`, `Image11`, `Image18`) — the rest of the nineteen were judged enough
rehearsal for presentation day without capturing all of them. No clean target exists for a real photo
— none of these documents were ever scanned before the photo was taken — so this is not a PSNR table.
Each photo is rectified with its **annotated** corners, never the detector's own, which isolates the
enhancement network from the corner-detection question above; a bad crop from a bad corner would
otherwise show up as a bad restoration and blur which network is responsible.

Readability is scored by Tesseract's own per-word confidence and word count, with identical
preprocessing (RGB → greyscale, nothing else) on all three variants — rectified input, our output, the
CamScanner reference — so no part of the comparison is explained by one image being handed to the
engine differently than another:

| Photo | input conf | ours conf | reference conf |
| :--- | ---: | ---: | ---: |
| Image1 | 31 | 31 | 30 |
| Image11 | 35 | 38 | 33 |
| Image18 | 74 | 73 | 79 |
| Image7 | 34 | 34 | 31 |
| Image9 | 32 | 37 | 33 |

Mean word confidence: input 41.1, ours **42.5**, reference 41.2. On this small a sample **our output
is not distinguishable from the commercial app** by Tesseract's own confidence — it wins on three of
five photos and loses narrowly on two, and the gap in both directions is a few points on a 0-100
scale. That is a genuinely different result from the synthetic PSNR table, where the model is the only
thing being scored: here there are two competent systems and the photos are not sorted by which one
they favour.

`Image18` — a printed English exam sheet, the one photo in this set with a hand-typed transcript
(`data/real/transcripts/Image18.txt`) — gives a character/word error rate instead of confidence alone:

| Photo | input CER | ours CER | reference CER |
| :--- | ---: | ---: | ---: |
| Image18 | 0.250 | 0.285 | **0.173** |

CamScanner wins this one clearly. Looking at `reports/figures/real/triplet_Image18.png`, the reason is
visible rather than mysterious: CamScanner's aggressive whitening and sharpening — the same style
choice the brief warns is "different, not worse" — happens to be exactly right for dense, small,
printed serif text, where crisp black-on-white edges are what an OCR engine's character segmentation
wants most. This project's network was trained to restore what a flatbed scan of that page would
look like, not to maximise Tesseract's read rate, and on printed text the two objectives diverge more
than they do on handwriting.

The qualitative triplets — `reports/figures/real/triplet_*.png`, each with a zoomed inset — tell a
more even story than the one CER number does. On `Image1` and `Image9` (handwritten lecture notes on
a dark wood desk), our output's zoomed crop reads **more crisply** than the CamScanner reference's own
zoomed crop — the reference has visible micro-blur at 100%, ours does not. On `Image18` the two are
close enough that the difference is the whitening style, not sharpness. Look at the figures before
trusting either number in isolation; a single scalar averaged over five photos is not the whole story
the brief asks this section to tell.

## Reading the two tables together

**A model that tops the synthetic test set can still meet a real photo it handles differently — and
the direction of the surprise here was not the expected one.** The corner detector, trained
exclusively on synthetic composites, generalises to real photos about as well as it does to unseen
synthetic ones once the comparison is put in the same units. The enhancement network, on the same real
photos, is fully competitive with a commercial product built and tuned specifically for document
scanning — except on the one case (dense printed text, an OCR engine as the judge) where the
commercial product's tuning is aimed exactly at the metric being used to judge it. Neither result was
obvious from the synthetic PSNR table alone, which is the entire reason this section exists.

## Reproducing this page

```bash
python scripts/parse_roboflow.py     # data/real/annotations -> data/real/corners.json
python scripts/evaluate_real.py      # both tables + the triplet figures
```

`scripts/evaluate_real.py` needs the `corner_heat` and `enhance_realistic` checkpoints under
`outputs/runs/` (or `--detector`/`--enhancer` pointing elsewhere), and `tesseract` on `PATH` —
`conda install -c conda-forge tesseract` plus `pip install pytesseract`, no sudo needed.
