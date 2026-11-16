"""Tests for hosted/local runtime detection."""

from __future__ import annotations

from modules import runtime


def test_loopback_urls_are_local():
    assert not runtime.is_remote_url("http://localhost:8501")
    assert not runtime.is_remote_url("http://127.0.0.1:8501/")
    assert not runtime.is_remote_url("http://[::1]:8501")
    assert not runtime.is_remote_url("")


def test_public_urls_are_remote():
    assert runtime.is_remote_url("https://accessable.streamlit.app/")
    assert runtime.is_remote_url("http://192.168.1.20:8501")
    assert runtime.is_remote_url("https://hashir17-accessable.hf.space")


def test_explicit_hosted_flag(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("IS_STREAMLIT_CLOUD", raising=False)
    monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)

    monkeypatch.setenv("ACCESSABLE_HOSTED", "true")
    assert runtime.is_hosted()

    monkeypatch.delenv("ACCESSABLE_HOSTED")
    monkeypatch.setenv("SPACE_ID", "someone/space")
    assert runtime.is_hosted()
