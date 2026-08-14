### corner_heat_dropout

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Training | 2.08 ± 5.51 | 0.57% | 0.925 | 0.9680 |
| *— classical baseline* | *43.52* | *12.02%* | *0.410* | *0.6398* |
| Validation | 1.28 ± 3.06 | 0.35% | 0.960 | 0.9787 |
| *— classical baseline* | *36.67* | *10.13%* | *0.545* | *0.6927* |
| Test | 1.91 ± 6.08 | 0.53% | 0.905 | 0.9722 |
| *— classical baseline* | *41.66* | *11.51%* | *0.485* | *0.6582* |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of each photo and then over photos, measured in the detector's own 256x256 input space. PCK is the fraction of photos where **all four** corners land within 2% of the image diagonal. The baseline rows are Canny + findContours + approxPolyDP on the identical input, with an undetected page scored as the whole frame.
