"""Offline voice assistant for AccessAble AI.

Wraps pyttsx3 (eSpeak NG backend on Linux) for fully offline speech -
no network, no account, no cost (spec 3.3).

Design notes:
  - Speech runs in a background daemon thread because Streamlit scripts
    block on UI code; runAndWait() would freeze the page otherwise.
  - A fresh engine is created per utterance inside the thread: pyttsx3
    engines are not safe to share across threads, and per-utterance
    engines are cheap to create.
  - stop() interrupts the current utterance from the main thread.
  - Every failure path degrades to text: is_available() is False ->
    the UI shows the text summary plus a setup hint instead of crashing
    (spec: never make audio the only channel).
"""

from __future__ import annotations

import threading

DEFAULT_RATE = 160  # words/min-ish default, adjustable 100-220

_AVAILABILITY: bool | None = None  # cached probe result


def is_available() -> bool:
    """True if a TTS engine can be initialized on this system.

    The probe result is cached for the process lifetime; engine
    initialization is not free and Streamlit reruns are frequent.
    """
    global _AVAILABILITY
    if _AVAILABILITY is None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.stop()
            _AVAILABILITY = True
        except Exception:
            _AVAILABILITY = False
    return _AVAILABILITY


def setup_hint() -> str:
    """Actionable hint shown when TTS is unavailable."""
    return (
        "Speech is unavailable on this system. The text summary below "
        "stays fully available. To enable offline speech on Fedora/Linux, "
        "install the eSpeak NG backend (sudo dnf install espeak-ng) and "
        "make sure 'pyttsx3' is installed in the virtual environment."
    )


class VoiceAssistant:
    """User-triggered speech with speak / stop / replay / rate control."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._engine = None
        self._rate: int = DEFAULT_RATE
        self._last_text: str = ""

    # -- internal worker ----------------------------------------------------
    def _speak_blocking(self, text: str, rate: int) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            self._engine = engine
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # Degrade to text; the UI reads state via is_speaking().
            pass
        finally:
            try:
                if self._engine is not None:
                    self._engine.stop()
            except Exception:
                pass
            self._engine = None

    # -- public API -----------------------------------------------------------
    @property
    def rate(self) -> int:
        return self._rate

    def set_rate(self, rate: int) -> None:
        self._rate = int(rate)

    def speak(self, text: str) -> bool:
        """Speak text in the background. Returns False if TTS is down."""
        if not text.strip():
            return False
        try:
            import pyttsx3  # noqa: F401  (availability probe)
        except Exception:
            return False
        self.stop()  # one utterance at a time (no overlap)
        self._last_text = text
        self._thread = threading.Thread(
            target=self._speak_blocking, args=(text, self._rate), daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Interrupt the current utterance, if any."""
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def replay(self) -> bool:
        """Speak the most recent text again."""
        return self.speak(self._last_text)

    def is_speaking(self) -> bool:
        thread = self._thread
        return not (thread is None or not thread.is_alive())

    def has_last_text(self) -> bool:
        """True if there is a previous utterance to replay."""
        return bool(self._last_text.strip())


# Single shared instance for the whole app process.
assistant = VoiceAssistant()
