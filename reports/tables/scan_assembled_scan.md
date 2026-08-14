### scan_assembled

| Split | corners | PSNR (dB) | SSIM | true corners | cost of detection |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Test | detected | 18.97 ± 1.76 | 0.8363 | 26.70 | **-7.73 dB** |
| Test | *degraded input, true corners* | *15.30* | *0.8489* | | |

Photo in, clean scan out, with no human input: the detector finds the page, the chain flattens it and the enhancement network restores it. Both arms are scored against the same target — the clean scan rectified with the true corners — so a misplaced corner is punished as the misalignment it is. The last column is what the detection step costs against being handed the page correctly.

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Test | 0.68 ± 0.54 | 0.19% | 0.995 | 0.9876 |

The corners the chain actually used, against the true ones, on the same photos it was scored on above. These are the **enhancement** frozen buckets, which carry no distractor sheet, so a detector scores better here than on its own bucket — the numbers are not interchangeable with the detector table's.
