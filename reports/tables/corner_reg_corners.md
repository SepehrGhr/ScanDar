### corner_reg

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Training | 3.22 ± 3.88 | 0.89% | 0.845 | 0.9560 |
| *— classical baseline* | *43.52* | *12.02%* | *0.410* | *0.6398* |
| Validation | 2.72 ± 1.78 | 0.75% | 0.870 | 0.9617 |
| *— classical baseline* | *36.67* | *10.13%* | *0.545* | *0.6927* |
| Test | 3.16 ± 3.02 | 0.87% | 0.830 | 0.9577 |
| *— classical baseline* | *41.66* | *11.51%* | *0.485* | *0.6582* |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of each photo and then over photos, measured in the detector's own 256x256 input space. PCK is the fraction of photos where **all four** corners land within 2% of the image diagonal. The baseline rows are Canny + findContours + approxPolyDP on the identical input, with an undetected page scored as the whole frame.
