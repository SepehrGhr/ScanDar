### enhance_sharp

| Split | PSNR (dB) | SSIM | baseline PSNR | baseline SSIM | gain |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training | 23.94 ± 2.32 | 0.9173 ± 0.0375 | 14.24 | 0.8137 | **+9.70 dB** |
| Validation | 23.86 ± 1.90 | 0.9175 ± 0.0326 | 14.59 | 0.8240 | **+9.27 dB** |
| Test | 23.36 ± 1.81 | 0.9140 ± 0.0311 | 14.70 | 0.8085 | **+8.66 dB** |

Whole pages rectified at 1024x1448, restored in cosine-blended overlapping tiles — the same path the inference pipeline takes. The baseline columns are the degraded input measured against the same clean targets, before any enhancement.
