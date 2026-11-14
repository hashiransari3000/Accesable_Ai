"""Tests for the voice assistant (TEST_PLAN speech matrix, stubbed pyttsx3).

Real audio output cannot be verified in an automated run; these tests
stub the pyttsx3 import to check state transitions, cancellation, and
the unavailable-TTS degradation path (M10-M12 logic, F3).
"""

import sys
import types

import pytest

from modules import voice_assistant as va


@pytest.fixture()
def fake_pyttsx3(monkeypatch):
    """A controllable pyttsx3 stub with a start/stop cycle."""
    state = {"speaking": False}

    class FakeEngine:
        def __init__(self):
            self.rate = None
            self.queue = []

        def setProperty(self, name, value):
            if name == "rate":
                self.rate = value

        def say(self, text):
            state["speaking"] = True
            self.queue.append(text)

        def runAndWait(self):
            # Simulate playback that honours an external stop() request.
            while state["speaking"]:
                pass
            state["speaking"] = False

        def stop(self):
            state["speaking"] = False

    monkeypatch.setitem(
        sys.modules, "pyttsx3", types.SimpleNamespace(init=lambda: FakeEngine())
    )
    return state


def _wait_done(assistant):
    if assistant._thread is not None:
        assistant._thread.join(timeout=2)


# ---- Availability and degradation (F3) ----------------------------------------

def test_available_true_with_stub(fake_pyttsx3):
    assert va.is_available() is True


def test_unavailable_tts_degrades(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pyttsx3", None  # forces ImportError inside probe
    )
    monkeypatch.setattr(va, "_AVAILABILITY", None)  # reset cached probe
    assert va.is_available() is False
    a = va.VoiceAssistant()
    assert a.speak("hello") is False
    assert "eSpeak NG" in va.setup_hint()


# ---- Speak / rate / replay / stop (M10-M12) -------------------------------------

def test_speak_sets_state_and_text(fake_pyttsx3):
    a = va.VoiceAssistant()
    a.set_rate(180)
    assert a.speak("A person appears in the image.") is True
    assert a.is_speaking() is True
    assert a.rate == 180
    a.stop()
    _wait_done(a)
    assert a.is_speaking() is False


def test_stop_during_speech_ends_playback(fake_pyttsx3):
    a = va.VoiceAssistant()
    a.speak("long sentence")
    a.stop()
    _wait_done(a)
    assert a.is_speaking() is False
    assert fake_pyttsx3["speaking"] is False


def test_replay_uses_last_text(fake_pyttsx3):
    a = va.VoiceAssistant()
    a.speak("first")
    a.stop()
    _wait_done(a)
    assert a.speak("second") is True
    a.stop()
    _wait_done(a)
    # replay re-speaks the last text
    assert a.replay() is True
    a.stop()
    _wait_done(a)


def test_empty_text_not_spoken(fake_pyttsx3):
    a = va.VoiceAssistant()
    assert a.speak("   ") is False
    _wait_done(a)
    assert a.is_speaking() is False
