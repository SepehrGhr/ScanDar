### corner_heat_e2e

| Split | corners | PSNR (dB) | SSIM | true corners | cost of detection |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Test | detected | 19.01 ± 1.77 | 0.8375 | 26.70 | **-7.70 dB** |
| Test | *degraded input, true corners* | *15.30* | *0.8489* | | |

Photo in, clean scan out, with no human input: the detector finds the page, the chain flattens it and the enhancement network restores it. Both arms are scored against the same target — the clean scan rectified with the true corners — so a misplaced corner is punished as the misalignment it is. The last column is what the detection step costs against being handed the page correctly.

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Test | 0.66 ± 0.56 | 0.18% | 0.990 | 0.9879 |

The corners the chain actually used, against the true ones, on the same photos it was scored on above. These are the **enhancement** frozen buckets, which carry no distractor sheet, so a detector scores better here than on its own bucket — the numbers are not interchangeable with the detector table's.
