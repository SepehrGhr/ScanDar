### enhance_realistic

| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training | 26.81 ± 2.98 | 0.9528 ± 0.0291 | 14.70 | 0.8466 | **+12.12 dB** |
| Validation | 26.71 ± 2.55 | 0.9519 ± 0.0290 | 15.03 | 0.8542 | **+11.68 dB** |
| Test | 26.67 ± 2.54 | 0.9533 ± 0.0256 | 15.30 | 0.8489 | **+11.36 dB** |

Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — the same path the inference pipeline takes. The baseline columns are the degraded input measured against the same clean targets, before any enhancement.
