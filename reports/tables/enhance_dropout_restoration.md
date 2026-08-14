### enhance_dropout

| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training | 25.75 ± 2.49 | 0.9439 ± 0.0304 | 14.70 | 0.8466 | **+11.05 dB** |
| Validation | 25.70 ± 2.29 | 0.9447 ± 0.0306 | 15.03 | 0.8542 | **+10.67 dB** |
| Test | 25.60 ± 2.15 | 0.9444 ± 0.0274 | 15.30 | 0.8489 | **+10.30 dB** |

Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — the same path the inference pipeline takes. The baseline columns are the degraded input measured against the same clean targets, before any enhancement.
