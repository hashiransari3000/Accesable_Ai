"""Controlled camera capture for AccessAble AI.

Design (spec 3.1, 11, and Phase 5 requirements):
  - Activation requires a deliberate user action; the app never starts
    the camera on page load.
  - A background daemon thread grabs frames at a modest target FPS
    (default 15) and keeps only the latest frame: capture rate is
    separate from inference rate, which is user-triggered per snapshot.
  - stop() is always safe to call: it sets the stop flag, releases the
    device, and joins the thread. Resources are released on stop, on
    page change (the UI calls stop on navigation), and on errors.
  - Failures (no device, permission denied, device lost) are reported
    through a readable error string so the UI can show fallback
    instructions for image upload instead of crashing.
"""

from __future__ import annotations

import threading
import time

DEFAULT_DEVICE_INDEX = 0
DEFAULT_TARGET_FPS = 15  # preview rate; inference is per-snapshot, on demand

_MAX_CONSECUTIVE_READ_FAILURES = 30  # ~2 s at 15 fps before giving up


class CameraCapture:
    """Threaded, cancellable frame grabber around cv2.VideoCapture."""

    def __init__(
        self,
        device_index: int = DEFAULT_DEVICE_INDEX,
        target_fps: int = DEFAULT_TARGET_FPS,
    ):
        self._device_index = device_index
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._capture = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None
        self._error: str | None = None
        self._frame_count = 0

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> str | None:
        """Start grabbing frames. Returns an error message on failure.

        On success returns None; the device stays open until stop().
        """
        if self.is_active():
            return None
        self._error = None
        self._latest_frame = None
        self._frame_count = 0

        try:
            import cv2

            capture = cv2.VideoCapture(self._device_index)
            # Give the driver a moment, then check availability.
            if not capture.isOpened():
                capture.release()
                self._error = (
                    "No camera was detected or access was denied. Check that "
                    "a webcam is connected, that its permission is allowed "
                    "for this user, and that no other app is using it. You "
                    "can still use the image upload above."
                )
                return self._error
        except Exception as exc:
            self._error = (
                f"Camera could not be initialized ({exc.__class__.__name__}: "
                f"{exc}). Image upload remains available above."
            )
            return self._error

        self._capture = capture
        self._running = True
        self._thread = threading.Thread(
            target=self._grab_loop, name="accessable-camera", daemon=True
        )
        self._thread.start()
        return None

    def stop(self) -> None:
        """Stop grabbing and release the device. Always safe to call."""
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            capture = self._capture
            self._capture = None
            self._latest_frame = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass  # device already gone; nothing else to clean up

    def is_active(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and self._running)

    # -- frame access -----------------------------------------------------------
    def get_frame(self):
        """Return the latest frame (a copy) or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_error(self) -> str | None:
        return self._error

    # -- worker ------------------------------------------------------------------
    def _grab_loop(self) -> None:
        import cv2

        failures = 0
        while self._running:
            start = time.perf_counter()
            ok, frame = self._capture.read()  # type: ignore[union-attr]
            if not ok or frame is None:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    self._error = (
                        "The camera stopped delivering frames (it may have "
                        "been disconnected). Camera mode stopped; image "
                        "upload remains available."
                    )
                    self._running = False
                    break
                time.sleep(self._frame_interval)
                continue
            failures = 0
            with self._lock:
                self._latest_frame = frame
                self._frame_count += 1
            # Hold the target FPS (throttle; spec: avoid excessive CPU use).
            elapsed = time.perf_counter() - start
            delay = self._frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)
