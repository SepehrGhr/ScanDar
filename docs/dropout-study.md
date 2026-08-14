# The dropout study

The brief asks one question here *(brief §6)*: add dropout, retrain, and say what it does — in
particular whether the gap between what the model scores on synthetic data and what it scores on real
photographs gets smaller. This page is what was set up to answer it, written before any arm had
trained, so that the design can be read separately from the result.

Everything the study needs already existed. `dropout` is wired through `ConvBlock` and both U-Nets,
`bottleneck_dropout` selects the bottleneck alone, and `CornerRegNet` takes `fc_dropout` separately
from the convolutional rate — the brief names the fully connected layers as the classic place, so
those are deliberately two knobs. Each arm is therefore a config file that inherits its baseline and
changes one line.

## The one rule the study rests on

**An arm may differ from its baseline by dropout and by nothing else.** Same photos, same schedule,
same optimiser, same seed, same canvas, same loss, and weight decay still zero — which is why the
first version of every model in this project was built without any regularisation at all.

The trap is the schedule. The trained enhancement baseline, `enhance_realistic`, was run with
`--set train.epochs=20` rather than the 60 that `base.yaml` carries, so an arm left at the inherited
default would train three times as long as the thing it is compared against and the table would be
measuring the schedule. The enhancement arms pin `train.epochs: 20` in the file. A test
(`tests/test_config.py`) asserts, for every arm, that the only difference from its baseline is the
dropout key it is named for.

## The arms

| Config | Baseline | What changes | Cost on the 3060 |
|---|---|---|---|
| `enhance_dropout.yaml` | `enhance_realistic` | `bottleneck_dropout: 0.2` | ~2.0 h |
| `enhance_dropout_wide.yaml` | `enhance_realistic` | `dropout: 0.1` in every `ConvBlock` | ~2.0 h |
| `corner_reg_dropout.yaml` | `corner_reg` | `fc_dropout: 0.3` | ~2.4 h |
| `corner_heat_dropout.yaml` | `corner_heat` | `bottleneck_dropout: 0.2` | ~2.4 h |
| `corner_heat_dropout_wide.yaml` | `corner_heat` | `dropout: 0.1` in every `ConvBlock` | ~2.4 h |

The first three answer the brief's question — one sensible rate per model. The two `_wide` arms are a
placement sweep on top of it and are the first thing to cut when GPU time is short.

**Why the bottleneck for the U-Nets.** Dropout in an encoder–decoder with skip connections is not one
decision. The skips carry high-resolution detail straight across to the decoder, so dropping channels
in the encoder damages exactly the signal the fine detail is rebuilt from, and the decoder's last
block sits one 1×1 convolution away from the pixels. The bottleneck is where the representation is
global, most redundant and furthest from the output. The `_wide` arms put a smaller rate everywhere
instead, which is the other half of that argument.

**Why 0.3 in the regressor's head.** `CornerRegNet` flattens an 8×8×256 grid into a 512-unit layer:
8.4 M weights, four fifths of the model, learned as one dense map from feature position to
coordinate. It is the only genuinely over-parameterised map in the project, and the only place where
a classic dropout rate is the obvious rate.

Note that `Dropout2d` drops whole feature *channels*, not scattered pixels — neighbouring activations
in a convolutional map are strongly correlated, so dropping individual ones leaks most of the signal
through and regularises far less than the rate suggests. That is why the convolutional rates here are
smaller than the fully connected one.

## What the numbers already say, before any arm runs

`reports/PREDICTIONS.md` §2 was committed in advance and predicts a null result on the synthetic
metrics. The evidence for that is in the baselines themselves:

| Baseline | training bucket | test bucket | gap |
|---|---|---|---|
| `enhance_realistic` | 26.81 dB | 26.67 dB | **0.14 dB** |
| `corner_heat` | 0.79 px | 1.06 px | **0.27 px** |
| `corner_reg` | 3.22 px | 3.16 px | **−0.06 px** (test is *better*) |

There is essentially no overfitting for a regulariser to remove. The generator never repeats a
sample, so there is nothing to memorise: 128 000 training samples means 128 000 distinct photographs.
The expected outcome is that dropout costs a little accuracy and buys nothing back, and that a null
result reported with the numbers that make it unsurprising is the honest answer to §6.

## The half of the question that cannot be measured yet

The brief asks about the **synthetic-to-real** gap, and that gap is not a number yet — it needs the
reference scans from a commercial scanning app and the exported corner annotations for the 19 real
photos, both still outstanding. Running the study now is fine: the synthetic column is complete on
its own, and the real column is scored later **from these same checkpoints**, with nothing retrained.
What the write-up must not do is claim §6 has been answered when only the synthetic half has been
measured.

## Running it

Every arm is one command, and the arms are independent — run them in any order, one at a time.

```bash
python train.py    --config configs/corner_reg_dropout.yaml
python evaluate.py --config configs/corner_reg_dropout.yaml
```

On Colab, add the wall-clock guard and re-run the identical command to resume; see
[running on Colab](running-on-colab.md).

### Running the arms at half length, without giving up the comparison

Three full arms are about 6.4 hours of GPU time. There is a way to spend half of that and still say
something defensible, and one way to spend half of it and say nothing at all — they differ by one
config key.

**The wrong way** is `train.epochs: 10`. The learning rate follows a cosine over the whole declared
schedule, so a 10-epoch run anneals to its floor by epoch 10 while the 20-epoch baseline is still at
1.1e-4 there. Every comparison then confounds dropout with the schedule, and the arm's final number
is not comparable with anything.

**The right way** is to leave `epochs: 20` alone and stop the run early:

```bash
python train.py --config configs/corner_reg_dropout.yaml --set train.stop_after_epoch=10
```

The learning rate, the data, the seed and the sample count are then *identical* to the baseline's
first ten epochs, so the arm can be read against **the baseline's own epoch 10**, which is already
recorded in `outputs/runs/<baseline>/metrics.csv`. Nothing has to be retrained to produce the other
side of the comparison.

```bash
python scripts/compare_dropout.py --curves            # matched-epoch table
python scripts/make_figures.py --compare corner_heat corner_heat_dropout
```

**What a half-length arm may and may not claim.** It may claim what happened to the
train-to-validation gap at a matched epoch — which is the quantity §6 is actually about, and which
barely depends on how far the run got. It may claim the two curves' shape. It may **not** quote its
final test PSNR or corner error beside a baseline that trained twice as long, because half of that
difference is the schedule.

And the direction of the bias is worth stating in the write-up, because it is the honest reason the
conclusion survives: **truncation is biased against dropout.** Dropout slows convergence, so an arm
judged halfway looks worse than it would at the end. A null or slightly-negative result under that
bias is therefore weak evidence that dropout hurts, and strong evidence that it does not help — and
"does not help" is the claim this study is making, for the reason the table of baseline gaps above
gives: there is no overfitting here to remove. A grader can check that reasoning against the numbers
in `metrics.csv`, which is the point of writing it down this way rather than quietly running shorter
runs and reporting the endpoints.

If the budget stretches later, the arms resume: drop `stop_after_epoch` and re-run the identical
command, and the run continues to epoch 20 on the schedule it was always following.

## Reading the result

```bash
python scripts/compare_dropout.py                       # the before-and-after table
python scripts/compare_detectors.py --runs corner_heat corner_heat_dropout corner_reg corner_reg_dropout
python scripts/make_figures.py --compare enhance_realistic enhance_dropout
```

`compare_dropout.py` reads what `evaluate.py` wrote and produces `reports/tables/dropout_study.csv`
and `.md`, one row per model and metric, with two columns that matter:

* **change** — signed so that positive always means dropout did better, whichever direction the
  metric itself runs in. A table where `+0.4` means better in one row and worse in the next is a
  table that gets misread, and this study is entirely a question of sign.
* **train→test gap** — each run's own training-bucket score minus its test score, signed so positive
  means worse on data it did not train on. This is the quantity a regulariser is supposed to shrink,
  and per the table above it was already near zero.

Arms that have not been evaluated yet are skipped with a note rather than failing, so the table is
readable while the remaining arms are still training.
