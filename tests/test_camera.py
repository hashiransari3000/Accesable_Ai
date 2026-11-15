"""Tests for the camera module with a stubbed cv2.

cv2 hardware cannot exist in CI; these tests stub cv2.VideoCapture to
verify lifecycle, throttling, error paths, and resource cleanup (M16-M18).
"""

import sys
import time
import types
from typing import ClassVar

import numpy as np
import pytest

from modules import camera as cam


class FakeCapture:
    """Records release() calls; feeds synthetic frames."""

    opened: bool = True
    frames: int = 10**9  # effectively endless
    fail_from: int = 10**9  # start failing after this many reads
    delay: float = 0.0

    instances: ClassVar[list] = []

    def __init__(self, index):
        self.index = index
        self.released = False
        self.reads = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return FakeCapture.opened and not self.released

    def read(self):
        self.reads += 1
        if FakeCapture.delay:
            time.sleep(FakeCapture.delay)
        if self.reads > FakeCapture.fail_from or self.released:
            return False, None
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def _reset():
    FakeCapture.instances = []
    FakeCapture.opened = True
    FakeCapture.fail_from = 10**9
    FakeCapture.delay = 0.0
    FakeCapture.frames = 10**9
    yield


@pytest.fixture()
def fake_cv2(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "cv2", types.SimpleNamespace(VideoCapture=FakeCapture)
    )
    yield


def _wait_frames(c, n=1, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if c.get_frame_count() >= n:
            return True
        time.sleep(0.01)
    return False


# ---- No camera / denied access (M17/M18 fallback) ------------------------------

def test_unavailable_device_reports_error_and_does_not_start(fake_cv2):
    FakeCapture.opened = False
    c = cam.CameraCapture()
    error = c.start()
    assert error is not None
    assert "No camera was detected" in error
    assert c.is_active() is False
    # device object was still released (cleanup on error)
    assert FakeCapture.instances[0].released is True


def test_cv2_init_crash_reports_error(fake_cv2, monkeypatch):
    def boom(index):
        raise RuntimeError("driver crash")

    monkeypatch.setitem(
        sys.modules, "cv2", types.SimpleNamespace(VideoCapture=boom)
    )
    c = cam.CameraCapture()
    error = c.start()
    assert error is not None
    assert "could not be initialized" in error


# ---- Start / frame delivery / active state (M16) --------------------------------

def test_start_and_frames_flow(fake_cv2):
    c = cam.CameraCapture()
    assert c.start() is None
    assert c.is_active() is True
    assert _wait_frames(c, 2)
    frame = c.get_frame()
    assert frame is not None and frame.shape == (4, 4, 3)
    c.stop()
    assert c.is_active() is False


def test_get_frame_returns_copy(fake_cv2):
    c = cam.CameraCapture()
    c.start()
    assert _wait_frames(c, 1)
    f1 = c.get_frame()
    f1[:] = 255  # mutate returned array
    f2 = c.get_frame()
    assert f2 is not None and not f2.all()  # internal frame untouched
    c.stop()


# ---- stop() cleanup (M17: resource cleanup) --------------------------------------

def test_stop_releases_device_and_is_idempotent(fake_cv2):
    c = cam.CameraCapture()
    c.start()
    assert _wait_frames(c, 1)
    c.stop()
    assert FakeCapture.instances[0].released is True
    assert c.get_frame() is None
    c.stop()  # second call must not raise
    assert not c.is_active()


# ---- Device lost mid-stream (M18) -------------------------------------------------

def test_persistent_read_failures_stop_camera_with_error(fake_cv2):
    FakeCapture.fail_from = 3  # 3 good frames, then persistent failures
    FakeCapture.delay = 0.001
    c = cam.CameraCapture()
    assert c.start() is None
    assert _wait_frames(c, 3, timeout=3.0)
    deadline = time.time() + 3.0
    while time.time() < deadline and c.is_active():
        time.sleep(0.02)
    assert c.is_active() is False
    error = c.get_error()
    assert error is not None and "stopped delivering frames" in error


# ---- Throttling --------------------------------------------------------------------

def test_grab_loop_holds_target_fps(fake_cv2):
    FakeCapture.delay = 0.0
    c = cam.CameraCapture(target_fps=10)  # 100 ms per frame
    c.start()
    _wait_frames(c, 1)
    time.sleep(0.25)
    c.stop()
    count = c.get_frame_count()
    # ~0.25 s at 10 fps -> about 2-3 frames; definitely not dozens
    assert 1 <= count <= 6


def test_restart_after_stop(fake_cv2):
    c = cam.CameraCapture()
    c.start()
    _wait_frames(c, 1)
    c.stop()
    assert c.start() is None
    assert c.is_active() is True
    c.stop()
