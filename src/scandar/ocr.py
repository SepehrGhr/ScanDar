"""OCR-based readability evaluation.  *(brief §3.3)*

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

from __future__ import annotations

import numpy as np

__all__ = ["ocr_text", "ocr_confidence", "levenshtein", "cer", "wer"]


def _prepare(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> greyscale uint8, the one preprocessing step every variant gets.

    Tesseract works on greyscale internally regardless; doing the conversion here
    rather than leaving it to pytesseract means the rectified input, our output
    and the reference scan are handed *exactly* the same array shape and dtype,
    so nothing about the comparison can be explained by a preprocessing
    difference between the three.
    """
    import cv2

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def ocr_text(image: np.ndarray, lang: str = "eng") -> str:
    """Run Tesseract and return the recognised text."""
    import pytesseract

    return pytesseract.image_to_string(_prepare(image), lang=lang)


def ocr_confidence(image: np.ndarray, lang: str = "eng") -> dict:
    """Mean per-word confidence and word count, Tesseract's own numbers.

    Confidence is averaged only over tokens Tesseract actually returned text
    for — its ``image_to_data`` emits a ``-1`` confidence row for whitespace and
    layout regions that carry no text, and including those would understate a
    page that recognised well but has a lot of blank margin.
    """
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(_prepare(image), lang=lang, output_type=Output.DICT)
    confidences = [
        float(conf)
        for conf, text in zip(data["conf"], data["text"])
        if str(text).strip() and float(conf) >= 0
    ]
    return {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "word_count": len(confidences),
    }


def levenshtein(a: list | str, b: list | str) -> int:
    """Edit distance between two sequences (characters or words), hand-written.

    Classic O(len(a) * len(b)) DP, one row kept at a time. Used for both
    :func:`cer` (sequence of characters) and :func:`wer` (sequence of words) —
    the algorithm does not care what the tokens are.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current[j] = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


def _normalise(text: str) -> str:
    """Collapse whitespace (line breaks, tabs, repeated spaces) to single spaces.

    Layout is not part of what CER/WER is meant to score — a transcript typed
    with different line wrapping than the OCR output should not be penalised for
    it — so both are flattened before the edit distance is computed.
    """
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance over characters / reference length.

    Undefined for an empty reference; the caller should not ask for a CER
    against a transcript that has not been written.
    """
    reference = _normalise(reference)
    hypothesis = _normalise(hypothesis)
    if len(reference) == 0:
        raise ValueError("cer needs a non-empty reference transcript")
    return levenshtein(list(reference), list(hypothesis)) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over words / reference word count."""
    reference_words = _normalise(reference).split(" ")
    hypothesis_words = _normalise(hypothesis).split(" ")
    if len(reference_words) == 0 or reference_words == [""]:
        raise ValueError("wer needs a non-empty reference transcript")
    return levenshtein(reference_words, hypothesis_words) / len(reference_words)
