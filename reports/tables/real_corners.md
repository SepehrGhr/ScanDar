### Corner detection on the real, Roboflow-labelled photos

16 annotated photos. The detector fell back to the classical path on 1/16.

| Variant | corner error (px) | % of diagonal | PCK@2% | quad IoU |
| :--- | ---: | ---: | ---: | ---: |
| corner_heat | 10.89 ± 4.37 | 0.34% | 1.000 | 0.9841 |
| *classical baseline* | *357.08* | *11.16%* | *0.188* | *0.7153* |

Mean Euclidean distance between predicted and true corners in each photo's own pixel space (photos are ~1920x2560 but not all identical), against the four hand-labelled corners of every annotated real photo — not the synthetic buckets. PCK is the fraction of photos where all four corners land within 2% of the photo's own diagonal. The baseline is Canny + findContours + approxPolyDP on the full-resolution photo, with an undetected page scored as the whole frame.
