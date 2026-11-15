"""Tests for open-vocabulary prompt parsing and safe failure paths."""

from modules import open_vocabulary as ov


def test_parse_vocabulary_basic():
    assert ov.parse_vocabulary("Door, stairs ,  RAMP") == ["door", "stairs", "ramp"]


def test_parse_vocabulary_dedupes_and_handles_newlines():
    text = "door\ndoor, Door\nramp"
    assert ov.parse_vocabulary(text) == ["door", "ramp"]


def test_parse_vocabulary_collapses_whitespace():
    assert ov.parse_vocabulary("  fire   exit  ,  hand  rail ") == [
        "fire exit",
        "hand rail",
    ]


def test_parse_vocabulary_caps_at_max_classes():
    names = ", ".join(f"object {i}" for i in range(ov.MAX_CLASSES + 15))
    assert len(ov.parse_vocabulary(names)) == ov.MAX_CLASSES


def test_parse_vocabulary_empty():
    assert ov.parse_vocabulary("  , ,\n") == []
    assert ov.parse_vocabulary("") == []


def test_empty_class_list_fails_without_loading_model():
    result = ov.analyze_image(b"irrelevant", ["", "  "])
    assert "at least one object name" in result.error
    assert ov.is_model_loaded(ov.DEFAULT_MODEL_PATH) is False


def test_invalid_image_fails_without_loading_model():
    result = ov.analyze_image(b"not-an-image", ["door"])
    assert result.error
    assert "at least one object name" not in result.error
    assert ov.is_model_loaded(ov.DEFAULT_MODEL_PATH) is False


def test_unknown_model_path_reports_load_error():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    result = ov.analyze_image(
        buf.getvalue(),
        ["door"],
        model_path="/nonexistent/nope.pt",
    )
    assert result.error and "Could not load" in result.error
