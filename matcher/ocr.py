"""Optional OCR.

Text inside the screenshot is one of the three retrieval signals, and often
the decisive one: a website screenshot usually carries the product name, an
Instagram screenshot the handle and caption.

Tesseract is an optional system dependency. If it is absent the pipeline runs
on two signals instead of three and says so in the audit log, rather than
failing or pretending it read something.
"""
from __future__ import annotations

import functools

from PIL import Image


@functools.lru_cache(maxsize=1)
def available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def extract(image: Image.Image, max_chars: int = 400) -> str | None:
    """Best-effort text extraction. Returns None when OCR is unavailable."""
    if not available():
        return None
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
    except Exception:
        return None
    cleaned = " ".join(text.split())
    return cleaned[:max_chars] or None
