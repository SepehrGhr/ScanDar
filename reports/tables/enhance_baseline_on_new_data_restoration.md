### enhance_baseline_on_new_data

| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Test | 26.46 ± 1.46 | 0.9566 ± 0.0155 | 15.30 | 0.8489 | **+11.16 dB** |

Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — the same path the inference pipeline takes. The baseline columns are the degraded input measured against the same clean targets, before any enhancement.
