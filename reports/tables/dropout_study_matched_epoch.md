### Dropout, at a matched epoch of an identical schedule

| Model | arm | epoch | metric | baseline | with dropout | change | val−train loss (baseline → dropout) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| enhance_realistic | enhance_dropout | 8 | validation PSNR (dB) | 25.35 | 25.42 | +0.07 | 0.00359 → 0.00222 |
| corner_reg | corner_reg_dropout | 10 | validation corner error (px @256) | 4.56 | 5.66 | -1.11 | -0.00143 → -0.00601 |

Both columns are validation on the frozen bucket, read off each run's own per-epoch log at the **same epoch of the same schedule** — same learning rate, same number of samples seen, same everything but dropout. An arm stopped early can be compared this way; its final numbers cannot be compared with a baseline that ran to the end.

The last column is validation loss minus training loss at that epoch — the overfitting gap dropout exists to close. It is the honest headline for a shortened run, because unlike the accuracy column it barely depends on how far the run got.

**Truncation is biased against dropout on accuracy**: dropout slows convergence, so an arm judged halfway through a schedule looks worse than it would at the end. That makes a null result on the gap column the safe conclusion and a small accuracy loss the unsafe one.
