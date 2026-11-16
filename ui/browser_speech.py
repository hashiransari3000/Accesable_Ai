"""Browser-based speech output for hosted deployments.

The local build uses pyttsx3 + eSpeak NG, which plays audio on the machine
running the app. On a remote server that audio is inaudible to the visitor,
so hosted mode asks the browser to speak with the Web Speech API instead.
Voices depend on the visitor's browser and operating system; the text
version is always visible, so speech is never the only channel.
"""

from __future__ import annotations

import json

import streamlit.components.v1 as components


def _rate(rate: int) -> float:
    return max(0.5, min(2.0, rate / 160))


def _run(script: str) -> None:
    components.html(f"<script>{script}</script>", height=0, scrolling=False)


def speak(text: str, rate: int = 160) -> None:
    """Ask the browser to read text aloud, cancelling any prior utterance."""
    _run(
        "try {"
        "  window.speechSynthesis.cancel();"
        f"  const u = new SpeechSynthesisUtterance({json.dumps(text)});"
        f"  u.rate = {_rate(rate)};"
        "  window.speechSynthesis.speak(u);"
        "} catch (e) {}"
    )


def stop() -> None:
    """Cancel any browser speech currently in progress."""
    _run("try { window.speechSynthesis.cancel(); } catch (e) {}")
