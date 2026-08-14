### corner_heat_e2e

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Test | 1.10 ± 2.74 | 0.30% | 0.950 | 0.9826 |
| *— classical baseline* | *41.66* | *11.51%* | *0.485* | *0.6582* |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of each photo and then over photos, measured in the detector's own 256x256 input space. PCK is the fraction of photos where **all four** corners land within 2% of the image diagonal. The baseline rows are Canny + findContours + approxPolyDP on the identical input, with an undetected page scored as the whole frame.
