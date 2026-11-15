"""Tests for change detection used by camera auto-briefing."""

from modules.object_detection import Detection
from modules.scene_tracker import ChangeReport, SceneMemory, grid_position

W, H = 300, 300


def _d(label, bbox=(0, 0, 100, 100), conf=0.8):
    return Detection(label=label, confidence=conf, bbox=bbox)


def test_grid_position_names():
    assert grid_position(_d("x", (0, 0, 100, 100)), W, H) == "upper left"
    assert grid_position(_d("x", (100, 100, 200, 200)), W, H) == "center"
    assert grid_position(_d("x", (200, 200, 300, 300)), W, H) == "lower right"
    assert grid_position(_d("x", (100, 0, 200, 100)), W, H) == "upper center"
    assert grid_position(_d("x", (0, 100, 100, 200)), W, H) == "left"


def test_new_object_confirmed_after_two_updates():
    memory = SceneMemory(confirm_new=2, forget_after=2)
    first = memory.update([_d("chair")], W, H)
    assert not first.has_changes
    second = memory.update([_d("chair")], W, H)
    assert second.new == ["chair (upper left)"]
    assert second.gone == []
    third = memory.update([_d("chair")], W, H)
    assert not third.has_changes  # announced once only


def test_one_frame_flicker_is_ignored():
    memory = SceneMemory()
    memory.update([_d("chair")], W, H)
    memory.update([_d("chair")], W, H)  # confirmed
    assert not memory.update([], W, H).gone  # one miss tolerated
    report = memory.update([_d("chair")], W, H)
    assert not report.gone
    assert not report.new  # returned before being forgotten - no re-announce


def test_gone_reported_after_forget_after_misses():
    memory = SceneMemory()
    memory.update([_d("chair")], W, H)
    memory.update([_d("chair")], W, H)
    memory.update([], W, H)
    report = memory.update([], W, H)
    assert report.gone == ["chair (upper left)"]


def test_movement_reads_as_gone_then_new():
    memory = SceneMemory()
    left = _d("chair", (0, 0, 100, 100))
    right = _d("chair", (200, 0, 300, 100))
    memory.update([left], W, H)
    memory.update([left], W, H)
    memory.update([right], W, H)
    report = memory.update([right], W, H)
    assert report.new == ["chair (upper right)"]
    assert report.gone == ["chair (upper left)"]


def test_unconfirmed_object_is_not_announced_when_gone():
    memory = SceneMemory()
    memory.update([_d("chair")], W, H)  # seen once, not confirmed
    report = memory.update([], W, H)
    assert not report.gone


def test_summary_is_text_equivalent():
    report = ChangeReport(new=["chair (left)"], gone=["door (right)"])
    assert report.summary == "New: chair (left). Gone: door (right)."
    assert not ChangeReport().has_changes


def test_reset_forgets_everything():
    memory = SceneMemory()
    memory.update([_d("chair")], W, H)
    memory.update([_d("chair")], W, H)
    memory.reset()
    assert memory.update([_d("chair")], W, H).new == []
