### Corner detection, test split

| Detector | corner error (px @256) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| corner_heat | 1.06 ± 2.44 | 0.29% | 0.955 | 0.9830 |
| corner_reg | 3.16 ± 3.02 | 0.87% | 0.830 | 0.9577 |
| *classical baseline* | 41.66 ± 46.13 | 11.51% | 0.485 | 0.6582 |

Mean Euclidean distance between predicted and true corners, averaged over the four corners of a photo and then over photos, in the detector's own 256x256 input space. PCK is the fraction of photos with **all four** corners inside 2% of the image diagonal. The baseline is Canny + findContours + approxPolyDP on the identical input.
