"""Tests for the upload validation logic (TEST_PLAN items M2-M4, M5-ish)."""

import io

import pytest
from PIL import Image

from modules.image_validation import (
    DEFAULT_MAX_BYTES,
    check_decodable,
    check_extension,
    check_size,
    validate_upload,
)


def _png_bytes(size=(4, 4), color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="JPEG")
    return buf.getvalue()


# ---- Extension checks ------------------------------------------------------

def test_valid_extensions_pass():
    assert check_extension("photo.jpg") is None
    assert check_extension("photo.JPEG") is None
    assert check_extension("scan.png") is None


def test_unsupported_extension_rejected():
    msg = check_extension("movie.mp4")
    assert msg is not None and "Unsupported file type" in msg


def test_missing_extension_rejected():
    assert check_extension("noext") is not None


# ---- Size checks -----------------------------------------------------------

def test_size_within_limit_passes():
    assert check_size(100, DEFAULT_MAX_BYTES) is None


def test_oversized_file_rejected():
    msg = check_size(DEFAULT_MAX_BYTES + 1)
    assert msg is not None and "too large" in msg


# ---- Decode checks ---------------------------------------------------------

def test_png_and_jpeg_decode():
    assert check_decodable(_png_bytes()) is None
    assert check_decodable(_jpeg_bytes()) is None


def test_corrupted_bytes_rejected():
    msg = check_decodable(b"not an image at all")
    assert msg is not None and "could not be opened" in msg


# ---- Combined order --------------------------------------------------------

def test_validate_upload_valid_image_passes():
    assert validate_upload("a.png", 10, _png_bytes()) is None


def test_extension_checked_before_size():
    # Oversized AND wrong extension -> extension error wins (cheap first)
    msg = validate_upload("a.gif", DEFAULT_MAX_BYTES + 5)
    assert msg is not None and "Unsupported file type" in msg


def test_size_checked_before_decode():
    # Wrong-extension-less? ensure size error surfaces when ext is fine
    msg = validate_upload("a.png", DEFAULT_MAX_BYTES + 5, b"")
    assert msg is not None and "too large" in msg


def test_decode_checked_last():
    # Correct extension + size, corrupt content -> decode error
    msg = validate_upload("a.png", 10, b"corrupt")
    assert msg is not None and "could not be opened" in msg


@pytest.mark.parametrize(
    "name,size,data,expect",
    [
        ("ok.png", 10, None, None),            # data optional
        ("ok.jpg", 10, _jpeg_bytes(), None),
        ("bad.png", 10, b"xx", "could not be opened"),
        ("big.png", DEFAULT_MAX_BYTES + 1, None, "too large"),
        ("x.txt", 10, None, "Unsupported file type"),
    ],
)
def test_validate_upload_matrix(name, size, data, expect):
    result = validate_upload(name, size, data)
    if expect is None:
        assert result is None
    else:
        assert expect in result
