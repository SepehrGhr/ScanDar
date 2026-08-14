### Dropout, test split

| Model | arm | metric | baseline | with dropout | change | train→test gap (baseline → dropout) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| enhance_realistic | enhance_dropout | PSNR (dB) | 26.67 | 25.6 | -1.07 | 0.14 → 0.15 |
| enhance_realistic | enhance_dropout | SSIM | 0.9533 | 0.9444 | -0.0089 | -0.0005 → -0.0005 |
| corner_reg | corner_reg_dropout | corner error (px @256) | 3.16 | 6.12 | -2.96 | -0.06 → -0.03 |
| corner_reg | corner_reg_dropout | PCK@2% | 0.83 | 0.405 | -0.425 | 0.015 → -0.005 |
| corner_reg | corner_reg_dropout | quad IoU | 0.9577 | 0.9168 | -0.0409 | -0.0017 → -0.0029 |
| corner_heat | corner_heat_dropout | corner error (px @256) | 1.06 | 1.91 | -0.85 | 0.27 → -0.17 |
| corner_heat | corner_heat_dropout | PCK@2% | 0.955 | 0.905 | -0.05 | 0.02 → 0.02 |
| corner_heat | corner_heat_dropout | quad IoU | 0.983 | 0.9722 | -0.0108 | 0.0033 → -0.0042 |

`change` is signed so that **positive always means dropout did better**, whichever direction the metric runs in, and so is the train→test gap: **positive means worse on data the run did not train on**. That gap is the quantity a regulariser is supposed to shrink, and the one that was already near zero before any dropout was added.

This is the synthetic half of the study. The real-photo column — the gap the brief actually asks about — is scored from these same checkpoints once the reference scans and the corner annotations exist.
