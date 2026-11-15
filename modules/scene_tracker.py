"""Change detection for camera auto-briefing.

The camera can run inference periodically. Announcing every frame would
be overwhelming (and spam speech), so this module remembers what was
seen and reports only meaningful changes:

  - NEW objects are reported once, after being seen in consecutive
    updates (`confirm_new`), so one-frame detection flicker is ignored.
  - GONE objects are reported after being missing for consecutive
    updates (`forget_after`), so momentary misses are ignored.
  - Objects are identified by label plus a coarse 3x3 screen region, so
    moving across the view reads as "gone here / new there" rather than
    a stream of identical announcements.

Pure logic, no I/O - fully unit-testable. The UI must present every
spoken announcement as text as well (audio is never the only channel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.object_detection import Detection

_COL_NAMES = ("left", "center", "right")
_ROW_NAMES = ("upper", "middle", "lower")


def grid_position(det: Detection, width: int, height: int) -> str:
    """Coarse 3x3 position name, e.g. 'upper left', 'center'."""
    if width <= 0 or height <= 0:
        return "in view"
    x1, y1, x2, y2 = det.bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = min(2, max(0, int(cx * 3 / width)))
    row = min(2, max(0, int(cy * 3 / height)))
    if row == 1 and col == 1:
        return "center"
    if row == 1:
        return _COL_NAMES[col]
    if col == 1:
        return f"{_ROW_NAMES[row]} center"
    return f"{_ROW_NAMES[row]} {_COL_NAMES[col]}"


def _key_for(det: Detection, width: int, height: int) -> tuple[str, str]:
    return (det.label.replace("_", " "), grid_position(det, width, height))


def _phrase_for(det: Detection, width: int, height: int) -> str:
    label = det.label.replace("_", " ")
    return f"{label} ({grid_position(det, width, height)})"


@dataclass
class ChangeReport:
    """What changed since the previous observation."""

    new: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.gone)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.new:
            parts.append("New: " + ", ".join(self.new) + ".")
        if self.gone:
            parts.append("Gone: " + ", ".join(self.gone) + ".")
        return " ".join(parts)


@dataclass
class _Entry:
    hits: int = 1
    missed: int = 0
    announced: bool = False


class SceneMemory:
    """Stateful diff engine for one continuous camera session."""

    def __init__(self, confirm_new: int = 2, forget_after: int = 2):
        self.confirm_new = max(1, confirm_new)
        self.forget_after = max(1, forget_after)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._phrases: dict[tuple[str, str], str] = {}

    def reset(self) -> None:
        self._entries.clear()
        self._phrases.clear()

    def update(
        self, detections: list[Detection], width: int, height: int
    ) -> ChangeReport:
        """Compare against the previous observation and return the diff."""
        observed: dict[tuple[str, str], str] = {}
        for det in detections:
            key = _key_for(det, width, height)
            observed.setdefault(key, _phrase_for(det, width, height))

        report = ChangeReport()

        for key, phrase in observed.items():
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
                self._phrases[key] = phrase
            else:
                entry.hits += 1
                entry.missed = 0
            if not entry.announced and entry.hits >= self.confirm_new:
                entry.announced = True
                report.new.append(phrase)

        for key in list(self._entries):
            if key in observed:
                continue
            entry = self._entries[key]
            entry.missed += 1
            if entry.missed >= self.forget_after:
                if entry.announced:
                    report.gone.append(self._phrases.get(key, key[0]))
                del self._entries[key]
                self._phrases.pop(key, None)

        report.new.sort()
        report.gone.sort()
        return report
