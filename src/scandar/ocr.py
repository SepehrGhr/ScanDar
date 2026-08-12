"""OCR-based readability evaluation.  *(Phase 3, brief §3.3)*

The real photos have no clean target — those documents were never scanned — so
PSNR and SSIM cannot be extended to them. Readability answers the two questions
that actually matter instead: did the enhancement make the page more legible than
the raw photo, and how close did it get to the commercial app?

Each metric is computed three times per photo — on the rectified input, on our
output, and on the reference scan — with identical preprocessing, so the
comparison is fair.

``ocr_text`` / ``ocr_confidence``
    Tesseract, with the engine's own per-word confidence and word count for every
    photo.
``cer`` / ``wer``
    Character and word error rate against hand-typed transcripts, computed with a
    plain Levenshtein distance. Only meaningful for the printed documents: no OCR
    engine reads handwritten Persian, so the handwritten majority is compared by
    confidence alone.
"""
