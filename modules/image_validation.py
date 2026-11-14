"""Image input validation for AccessAble AI.

Pure, testable logic used by the Scene Assistant and Read Text pages.
The UI layer calls `validate_upload(...)` and renders the returned
state; this module never touches Streamlit.

Rules (error prevention, HCI):
  - Allowed formats: JPEG, PNG (also accepted by the file picker).
  - Max size checked BEFORE decoding (default 10 MB).
  - Corrupted/undecodable files are rejected with an actionable message.
"""

from __future__ import annotations

import io

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png")

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def check_extension(file_name: str, allowed: set[str] | None = None) -> str | None:
    """Return an error message if the extension is not allowed, else None."""
    allowed = allowed or ALLOWED_EXTENSIONS
    name = (file_name or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if not ext or ext not in allowed:
        pretty = ", ".join(sorted(allowed))
        return f"Unsupported file type: '{file_name}'. Please upload one of: {pretty}."
    return None


def check_size(size_bytes: int, max_bytes: int = DEFAULT_MAX_BYTES) -> str | None:
    """Return an error message if the file is too large, else None."""
    if size_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return (
            f"File is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {limit_mb:.0f} MB."
        )
    return None


def check_decodable(data: bytes) -> str | None:
    """Return an error message if the bytes are not a decodable image."""
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:  # narrow at call boundary: any decode failure
        return f"The file could not be opened as an image ({exc.__class__.__name__}). Please try a different JPG or PNG."
    return None


def validate_upload(
    file_name: str,
    size_bytes: int,
    data: bytes | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str | None:
    """Run all validation steps in order; return first error or None.

    Order matters: cheap checks first (extension, size), then decode.
    `data` is optional so extension/size can be validated before the
    (potentially large) bytes are read into memory.
    """
    error = check_extension(file_name)
    if error:
        return error
    error = check_size(size_bytes, max_bytes)
    if error:
        return error
    if data is not None:
        error = check_decodable(data)
        if error:
            return error
    return None
