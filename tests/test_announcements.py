"""Tests for announcement prioritization and repetition guard (A1-A6, cooldown)."""

from modules.announcement_manager import (
    RepetitionGuard,
    prioritize,
    summarize,
)
from modules.object_detection import Detection

W, H = 600, 400


def _d(label, conf):
    return Detection(label=label, confidence=conf, bbox=(10, 10, 50, 50))


# ---- Prioritization (A1) ------------------------------------------------------

def test_person_boosted_to_front():
    # person 0.90 + boost 0.15 = 1.05 beats chair 0.95 + 0.05 = 1.00
    dets = [_d("chair", 0.95), _d("person", 0.90)]
    top = prioritize(dets, W, H, max_items=2)
    assert top[0].label == "person"


def test_confidence_ordering_without_boost():
    dets = [_d("low", 0.40), _d("high", 0.90)]
    top = prioritize(dets, W, H, max_items=2)
    assert [d.label for d in top] == ["high", "low"]


def test_max_items_caps_list(A=None):
    dets = [_d(f"obj{i}", 0.5 + i / 100) for i in range(8)]
    assert len(prioritize(dets, W, H, max_items=4)) == 4  # A5 cap


# ---- summarize (A4, A5, P7 phrasing) ------------------------------------------

def test_empty_detections_give_clean_summary():
    s = summarize([], W, H)
    assert s.summary and "No objects" in s.summary
    assert s.phrases == []


def test_summary_phrases_are_concise_and_cautious():
    s = summarize([_d("person", 0.8)], W, H)
    assert len(s.phrases) == 1
    phrase = s.phrases[0]
    assert phrase.startswith("A person")
    assert "side" in phrase or "area" in phrase
    for banned in ("meters", "safe", "collision"):
        assert banned not in phrase


def test_summary_flags_truncation():
    dets = [_d(f"obj{i}", 0.9 - i / 100) for i in range(6)]
    s = summarize(dets, W, H, max_items=4)
    assert s.truncated is True
    assert "more object(s)" in s.summary


# ---- RepetitionGuard (A2, A3, cooldown) ---------------------------------------

def test_repeated_key_within_cooldown_suppressed():
    g = RepetitionGuard(cooldown_seconds=10.0)
    announced, _ = g.filter_new(["person"], now=0.0)
    assert announced == ["person"]
    announced, suppressed = g.filter_new(["person"], now=5.0)
    assert announced == [] and suppressed == ["person"]


def test_key_after_cooldown_announced_again():
    g = RepetitionGuard(cooldown_seconds=10.0)
    g.filter_new(["person"], now=0.0)
    announced, _ = g.filter_new(["person"], now=10.5)
    assert announced == ["person"]


def test_new_object_always_announced():
    g = RepetitionGuard(cooldown_seconds=10.0)
    g.filter_new(["person"], now=0.0)
    announced, _ = g.filter_new(["chair"], now=1.0)
    assert announced == ["chair"]


def test_mixed_new_and_repeated():
    g = RepetitionGuard(cooldown_seconds=10.0)
    g.filter_new(["a", "b"], now=0.0)
    announced, suppressed = g.filter_new(["a", "b", "c"], now=20.0)
    assert announced == ["a", "b", "c"] and suppressed == []


def test_reset_clears_cooldowns():
    g = RepetitionGuard(cooldown_seconds=100.0)
    g.filter_new(["person"], now=0.0)
    g.reset()
    announced, _ = g.filter_new(["person"], now=1.0)
    assert announced == ["person"]


def test_summary_uses_relative_proximity_word():
    det = _d("chair", 0.8)
    det.proximity = "close"
    s = summarize([det], W, H)
    assert "close" in s.phrases[0]
    for banned in ("meters", "metres", "safe", "collision", "distance"):
        assert banned not in s.phrases[0]
