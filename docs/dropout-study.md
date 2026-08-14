# The dropout study

The brief asks one question here *(brief §6)*: add dropout, retrain, and say what it does — in
particular whether the gap between what the model scores on synthetic data and what it scores on real
photographs gets smaller. This page was written before any arm had trained, so that the design can be
read separately from the result; [the result](#the-result) is at the bottom, and it is a null one.

**In one line: dropout changes nothing worth having on the synthetic metrics, and the model the
prediction named as most likely to benefit is the one it measurably hurt.**

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

---

## The result

Three arms ran, all shortened with `stop_after_epoch` as described above: `corner_reg_dropout` to
epoch 10, `corner_heat_dropout` and `enhance_dropout` to epoch 8, against baselines that ran the
full 20. About 3 hours of GPU time in total. The two `_wide` placement arms were not run.

### At a matched epoch, which is the comparison that is controlled

Identical schedule, identical data, identical seed, identical learning rate at that step — dropout is
the only difference. Validation, on the frozen bucket, from each run's own per-epoch log:

| Model | epoch | metric | baseline | with dropout | change | val−train loss |
| :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| `enhance_realistic` → `enhance_dropout` | 8 | validation PSNR | 25.35 dB | **25.42 dB** | +0.07 | 0.0036 → 0.0022 |
| `corner_reg` → `corner_reg_dropout` | 10 | validation corner error | 4.56 px | **5.66 px** | −1.11 | −0.0014 → −0.0060 |

**The enhancement network does not care.** 0.07 dB is inside the run-to-run noise of this setup, and
the sign is if anything the wrong way round for the prediction, which expected a small loss. The
val−train loss distance was already 0.0036 — three parts in a thousand — and dropout moved it to
0.0022, which is a smaller number and not a meaningful one: neither value describes a model that is
overfitting.

**The coordinate regressor is worse by a quarter of its own error**, at the same step, with the same
data. That is the clearest single result in this study, and it is the opposite of what was predicted.

`corner_heat` could not be compared this way, and the comparison tool refuses to try: that baseline
trained before the corner extraction was fixed, so its log records 6.26 px where re-evaluating the
same checkpoint with current code gives 0.70 px. Its per-epoch curve and a curve logged today do not
measure the same quantity, and the 8.9× disagreement between the log and its own re-evaluation is how
the tool detects that without being told.

### On the frozen test bucket, where the schedules do not match

Every row here compares a 20-epoch baseline against an 8- or 10-epoch arm, so **the accuracy columns
are not evidence about dropout** — they are mostly evidence about training length, and they are
recorded for completeness rather than for argument. The gap column is the one that survives, because
it is computed within a single run.

| Run | epochs | test | train→test gap |
| :--- | ---: | ---: | ---: |
| `enhance_realistic` | 20 | 26.67 dB / 0.9533 SSIM | 0.14 dB |
| `enhance_dropout` | 8 | 25.60 dB / 0.9444 SSIM | 0.15 dB |
| `corner_reg` | 20 | 3.16 px / PCK 0.830 | −0.06 px |
| `corner_reg_dropout` | 10 | 6.12 px / PCK 0.405 | −0.03 px |
| `corner_heat` | 20 | 1.06 px / PCK 0.955 | 0.27 px |
| `corner_heat_dropout` | 8 | 1.91 px / PCK 0.905 | −0.17 px |

The gap does not close, because there was no gap. It was 0.14 dB and it stayed 0.15; it was already
*negative* for the regressor — better on test than on training — and stayed negative. A regulariser
cannot recover generalisation that was never lost, and this is what that looks like when it is
measured rather than assumed.

Two things in that table are worth reading on their own terms. `corner_heat_dropout` reaches 1.91 px
in **8** epochs against the baseline's 1.06 px in 20, which says more about how quickly the heatmap
formulation converges than about dropout. And `corner_reg_dropout`'s PCK falls to 0.405 while its
mean error only doubles: the failures are concentrated, not spread — a few photos where one corner
goes badly wrong, which is exactly the shape of damage that injecting noise into a low-dimensional
coordinate estimate produces.

### Scoring the prediction

`reports/PREDICTIONS.md` §2 was committed before any of this ran, and it is scored here as written,
the same way the detector comparison was.

| Predicted | Outcome |
| :--- | :--- |
| Dropout buys approximately nothing on the synthetic metrics for the enhancement network | **Right.** +0.07 dB at a matched epoch. |
| …and slightly *hurts* them | **Wrong**, though only just: the sign came out marginally positive rather than marginally negative. Both readings mean "nothing". |
| The corner detectors are more likely to benefit, and `CornerRegNet` most of all | **Wrong, and wrong about the model it was most confident about.** The regressor is the one arm that clearly lost: −1.11 px at a matched epoch. |
| The gap between train and unseen data shrinks | **Not testable as posed** — the gap was ~0 in every baseline before dropout, so there was nothing to shrink. |
| If dropout helps anywhere it helps on the real photos | **Still untested**, and blocked on the reference scans and the corner annotations. |

The reasoning behind the prediction was that `CornerRegNet`'s fully connected head — 8.4 M weights,
four fifths of the model — is the one genuinely over-parameterised map in the project, so it should
be the one dropout rescues. The parameter count was right and the inference from it was wrong, for a
reason worth keeping:

**Over-parameterised is not the same as over-fitting.** That head is large because it has to learn a
dense map from feature position to coordinate, and it is trained on a generator that never repeats a
sample, so it never gets the chance to memorise anything. What dropout does there is not remove
memorisation — there is none — but inject multiplicative noise directly into a *low-dimensional
continuous* estimate. Eight numbers come out; there is no spatial pooling downstream to average the
noise away, and no redundancy to route around it. In the two U-Nets the same rate lands on a
bottleneck of 512 heavily redundant channels with skip connections carrying detail around it, which
is why they shrug it off.

So the brief's "classic place for dropout" is, in this project, the worst place for it — and the
reason is not that dropout is a bad idea but that this model was never doing the thing dropout fixes.

### What would change this answer

Nothing in the synthetic half. The result is a null, the mechanism for the null is measured
(no train-to-test gap anywhere, in any baseline, before any arm ran), and running the arms to 20
epochs would sharpen the accuracy columns without touching that conclusion.

The real-photo half is a different question and is still open. When the reference scans and the
Roboflow export land, the same checkpoints get scored on the 19 real photos and the
synthetic-to-real gap becomes a number for both the baselines and the arms — no retraining. That is
the half where the prediction's one untested claim lives, and it is the half the brief actually asks
about, so the write-up says so rather than presenting this page as a complete answer to §6.
