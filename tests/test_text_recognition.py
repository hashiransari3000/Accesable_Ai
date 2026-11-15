"""Tests for the OCR module with a stubbed heavy reader."""

import io
import sys

import pytest
from PIL import Image

from modules import text_recognition as tr


class _StubReader:
    """Stands in for easyocr.Reader.readtext."""

    def __init__(self, output):
        self._output = output

    def readtext(self, source, detail=1):
        return self._output


def _png_bytes(color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stub_reader(monkeypatch, request):
    output = request.param if hasattr(request, "param") else []
    monkeypatch.setitem(
        tr._READER_CACHE, tuple(tr.DEFAULT_LANGUAGES), _StubReader(output)
    )
    return output


# ---- O2: normal output -> joined text + confidence ---------------------------

def test_normal_extraction_joins_regions(stub_reader):
    stub_reader.append(([[0, 0], [10, 0], [10, 10], [0, 10]], "HELLO", 0.93))
    stub_reader.append(([[0, 20], [10, 20], [10, 30], [0, 30]], "WORLD", 0.81))
    result = tr.extract_text(_png_bytes())
    assert result.error is None
    assert not result.is_empty
    assert result.text == "HELLO\nWORLD"
    assert [r.confidence for r in result.regions] == [0.93, 0.81]


# ---- O1: empty output is "no text found", NOT an error ------------------------

def test_empty_output_is_not_error(stub_reader):
    result = tr.extract_text(_png_bytes())
    assert result.error is None
    assert result.is_empty is True
    assert result.text == ""


def test_whitespace_only_is_empty(stub_reader):
    stub_reader.append(([[0, 0], [5, 0], [5, 5], [0, 5]], "   ", 0.9))
    result = tr.extract_text(_png_bytes())
    assert result.error is None
    assert result.is_empty is True


# ---- O3: reader failure -> actionable error -----------------------------------

def test_reader_load_failure_is_actionable(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("FileNotFoundError: model missing")
    monkeypatch.setattr(tr, "get_reader", boom)
    result = tr.extract_text(_png_bytes())
    assert result.error is not None
    assert "Could not load the OCR model" in result.error


def test_ocr_runtime_failure_is_actionable(stub_reader, monkeypatch):
    class BrokenReader:
        def readtext(self, *a, **k):
            raise ValueError("bad input")

    monkeypatch.setitem(tr._READER_CACHE, tuple(tr.DEFAULT_LANGUAGES), BrokenReader())
    result = tr.extract_text(_png_bytes())
    assert result.error is not None
    assert "OCR failed" in result.error


# ---- Corrupt image --------------------------------------------------------------

def test_corrupted_image_reports_decode_error(monkeypatch):
    monkeypatch.setitem(
        tr._READER_CACHE, tuple(tr.DEFAULT_LANGUAGES), _StubReader([])
    )
    result = tr.extract_text(b"not an image")
    assert result.error is not None
    assert "could not be decoded" in result.error


# ---- O4: lazy loading - reader is NOT created until requested -------------------

def test_reader_lazy_and_cached(monkeypatch):
    calls = {"n": 0}

    class FakeEasyOCRModule:
        @staticmethod
        def Reader(langs, gpu=False, verbose=False):
            calls["n"] += 1
            return _StubReader([])

    monkeypatch.setattr(
        sys, "modules", {**sys.modules, "easyocr": FakeEasyOCRModule}
    )
    monkeypatch.setattr(tr, "_READER_CACHE", {})
    assert tr.is_reader_loaded() is False
    tr.get_reader()          # first call constructs
    tr.get_reader()          # second call must reuse the cache
    assert calls["n"] == 1
    assert tr.is_reader_loaded() is True
