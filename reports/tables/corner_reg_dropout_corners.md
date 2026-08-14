### corner_reg_dropout

| Split | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| Training | 6.15 ± 4.16 | 1.70% | 0.400 | 0.9139 |
| *— classical baseline* | *43.52* | *12.02%* | *0.410* | *0.6398* |
| Validation | 5.66 ± 2.52 | 1.56% | 0.390 | 0.9198 |
| *— classical baseline* | *36.67* | *10.13%* | *0.545* | *0.6927* |
| Test | 6.12 ± 3.99 | 1.69% | 0.405 | 0.9168 |
| *— classical baseline* | *41.66* | *11.51%* | *0.485* | *0.6582* |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of each photo and then over photos, measured in the detector's own 256x256 input space. PCK is the fraction of photos where **all four** corners land within 2% of the image diagonal. The baseline rows are Canny + findContours + approxPolyDP on the identical input, with an undetected page scored as the whole frame.
