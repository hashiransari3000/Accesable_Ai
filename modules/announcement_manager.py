"""Announcement building for AccessAble AI speech output.

Turns raw detections into a short, prioritized, human-friendly summary
(spec 3.3: never overwhelm the user with a long list) and provides a
repetition guard with a cooldown for continuous sources like the camera
(Phase 5) so the same unchanged object is not announced every frame.

Pure logic, no I/O, fully unit-testable (TEST_PLAN A1-A6).
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.object_detection import Detection
from modules.position_analysis import describe_position

# Cap on spoken items: concise beats complete (spec 3.3).
DEFAULT_MAX_ITEMS = 4

# Classes that matter most for a person moving through a space get a
# priority boost; everything else ranks by confidence alone.
PRIORITY_BOOST = {
    "person": 0.15,
    "chair": 0.05,
    "couch": 0.05,
    "bed": 0.05,
    "dining table": 0.05,
    "door": 0.05,
    "bicycle": 0.05,
    "car": 0.05,
}


@dataclass
class Summary:
    """The spoken summary plus everything the UI needs around it."""

    summary: str = ""
    phrases: list[str] = None  # type: ignore[assignment]
    spoken_count: int = 0
    total_count: int = 0
    truncated: bool = False


def _phrase_for(det: Detection, width: int, height: int) -> str:
    position = describe_position(det, width, height, include_vertical=True)
    position = position.replace("appears in the ", "")
    label = det.label.replace("_", " ")
    if det.proximity:
        lead = "at mid-distance" if det.proximity == "mid-distance" else det.proximity
        return f"A {label} appears {lead} in the {position}."
    return f"A {label} appears in the {position}."


def prioritize(
    detections: list[Detection],
    width: int,
    height: int,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> list[Detection]:
    """Rank by usefulness (priority class + confidence), cap the length."""
    ranked = sorted(
        detections,
        key=lambda d: d.confidence + PRIORITY_BOOST.get(d.label, 0.0),
        reverse=True,
    )
    return ranked[:max_items]


def summarize(
    detections: list[Detection],
    width: int,
    height: int,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> Summary:
    """Build the concise spoken summary for one analysis."""
    if not detections:
        return Summary(
            summary="No objects were detected in this image.",
            phrases=[],
            spoken_count=0,
            total_count=0,
        )
    chosen = prioritize(detections, width, height, max_items)
    phrases = [_phrase_for(d, width, height) for d in chosen]
    truncated = len(detections) > len(chosen)
    summary = " ".join(phrases)
    if truncated:
        summary += (
            f" {len(detections) - len(chosen)} more object(s) were detected; "
            "see the list on screen."
        )
    return Summary(
        summary=summary,
        phrases=phrases,
        spoken_count=len(chosen),
        total_count=len(detections),
        truncated=truncated,
    )


class RepetitionGuard:
    """Cooldown-based deduplication for continuous sources (camera).

    Given a stream of detection keys (labels) with timestamps, decide
    which are NEW enough to announce: a key is announced only once per
    cooldown window. Position changes count as new because the phrase
    includes the region.
    """

    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_announced: dict[str, float] = {}

    def filter_new(
        self, keys: list[str], now: float
    ) -> tuple[list[str], list[str]]:
        """Return (keys_to_announce, keys_suppressed).

        Updates internal state only for announced keys, so a suppressed
        object may be announced later if its cooldown expires.
        """
        to_announce: list[str] = []
        suppressed: list[str] = []
        for key in keys:
            last = self._last_announced.get(key)
            if last is None or (now - last) >= self.cooldown_seconds:
                to_announce.append(key)
                self._last_announced[key] = now
            else:
                suppressed.append(key)
        return to_announce, suppressed

    def reset(self) -> None:
        """Clear state (user pressed stop, image cleared, etc.)."""
        self._last_announced.clear()
