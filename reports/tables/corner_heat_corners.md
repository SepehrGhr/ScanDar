### corner_heat

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Training | 0.79 ± 1.23 | 0.22% | 0.975 | 0.9863 |
| *— classical baseline* | *43.52* | *12.02%* | *0.410* | *0.6398* |
| Validation | 0.70 ± 0.84 | 0.19% | 0.980 | 0.9877 |
| *— classical baseline* | *36.67* | *10.13%* | *0.545* | *0.6927* |
| Test | 1.06 ± 2.44 | 0.29% | 0.955 | 0.9830 |
| *— classical baseline* | *41.66* | *11.51%* | *0.485* | *0.6582* |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of each photo and then over photos, measured in the detector's own 256x256 input space. PCK is the fraction of photos where **all four** corners land within 2% of the image diagonal. The baseline rows are Canny + findContours + approxPolyDP on the identical input, with an undetected page scored as the whole frame.
