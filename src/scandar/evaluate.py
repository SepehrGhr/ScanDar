"""Evaluation.  *(brief §3.3 and §5)*

The brief names this file explicitly. It produces the numbers the report is built
on:

* PSNR and SSIM on the synthetic **training, validation and test** buckets, with
  the degraded-input baseline computed *first* — each row answers a different
  question, and the gap between the first and last is the overfitting story;
* corner localisation error, all-four-within-threshold success rate, and quad IoU
  for both detectors, on the synthetic test set and on the Roboflow-labelled real
  photos;
* the OCR readability comparison against the commercial scanning app.

Tables are written to ``reports/tables/`` in both CSV and Markdown, so the report
never contains a hand-copied number.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise SystemExit(
        "Evaluation is not implemented yet — there is no trained model to score."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
