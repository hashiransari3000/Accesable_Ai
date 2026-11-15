"""Local OCR text recognition for AccessAble AI.

Wraps EasyOCR, loaded only on demand: the reader (and its
~85 MB of model weights, cached once by EasyOCR into ~/.EasyOCR) is not
created until the user actually requests text reading. The reader is
then cached for the process lifetime and reused.

Returned states are explicit so the UI never has to guess:
  - error:    unreadable image, reader/load failure, OCR failure
  - empty:    a clean run that found no text ("no text found", not an error)
  - success:  extracted text plus per-region confidence where available

Language support: English only for now. Other languages must be
explicitly tested before being offered (safety docs section 5).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

DEFAULT_LANGUAGES = ["en"]

_READER_CACHE: dict[tuple, object] = {}


@dataclass
class TextRegion:
    """One detected text region with its model confidence."""

    text: str
    confidence: float  # 0..1, model score (not a correctness guarantee)


@dataclass
class TextRecognitionResult:
    """Everything the UI needs after one OCR run."""

    text: str = ""
    regions: list[TextRegion] = field(default_factory=list)
    processing_ms: int = 0
    languages: list[str] = field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    error: str | None = None
    is_empty: bool = False  # clean run, genuinely no text found


def get_reader(languages: list[str] | None = None):
    """Create and cache the EasyOCR reader; heavy work happens once.

    Raises RuntimeError with technical detail; extract_text() wraps it
    into an actionable, user-facing message.
    """
    languages = languages or DEFAULT_LANGUAGES
    key = tuple(languages)
    if key in _READER_CACHE:
        return _READER_CACHE[key]
    try:
        import easyocr

        reader = easyocr.Reader(languages, gpu=False, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"{exc.__class__.__name__}: {exc}") from exc
    _READER_CACHE[key] = reader
    return reader


def is_reader_loaded(languages: list[str] | None = None) -> bool:
    languages = languages or DEFAULT_LANGUAGES
    return tuple(languages) in _READER_CACHE


def _decode_image(image_bytes: bytes):
    """Decode bytes to a PIL image; raise ValueError if broken."""
    from PIL import Image

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"The image could not be decoded for OCR ({exc.__class__.__name__}). "
            "Please try a different JPG or PNG."
        ) from exc


def _join_regions(regions: list[TextRegion]) -> str:
    """Join region texts in reading order, one line per region."""
    return "\n".join(r.text.strip() for r in regions if r.text.strip())


def extract_text(
    image_bytes: bytes, languages: list[str] | None = None
) -> TextRecognitionResult:
    """Run OCR once and return a fully-specified result.

    Never raises: failures are reported through result.error.
    """
    import time

    result = TextRecognitionResult(languages=list(languages or DEFAULT_LANGUAGES))

    try:
        _decode_image(image_bytes)
    except ValueError as exc:
        result.error = str(exc)
        return result

    try:
        reader = get_reader(result.languages)
    except Exception as exc:
        result.error = (
            "Could not load the OCR model. Check your internet connection "
            f"for the one-time download ({exc}), disk space, and that "
            "'easyocr' is installed (pip install -r requirements.txt)."
        )
        return result

    try:
        start = time.perf_counter()
        # EasyOCR supports str paths, raw bytes, or numpy arrays.
        raw = reader.readtext(image_bytes, detail=1)
        result.processing_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        result.error = (
            f"OCR failed while processing this image "
            f"({exc.__class__.__name__}: {exc}). Try a clearer, well-lit "
            "image with horizontal text."
        )
        return result

    regions = [
        TextRegion(text=str(item[1]).strip(), confidence=float(item[2]))
        for item in raw
        if str(item[1]).strip()
    ]
    result.regions = regions
    result.text = _join_regions(regions)
    result.is_empty = not bool(result.text)
    return result
