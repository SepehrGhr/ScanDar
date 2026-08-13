### enhance_baseline

| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training | 24.24 ± 2.41 | 0.9225 ± 0.0365 | 14.24 | 0.8137 | **+10.00 dB** |
| Validation | 23.96 ± 1.95 | 0.9200 ± 0.0324 | 14.59 | 0.8240 | **+9.37 dB** |
| Test | 23.49 ± 1.84 | 0.9175 ± 0.0303 | 14.70 | 0.8085 | **+8.79 dB** |

Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — the same path the inference pipeline takes. The baseline columns are the degraded input measured against the same clean targets, before any enhancement.
