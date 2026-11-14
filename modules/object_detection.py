"""YOLO object detection service for AccessAble AI.

Responsibilities:
  - Lazily load a small pretrained YOLO nano model and cache it (the model
    is never reloaded per interaction - performance constraint #11).
  - Run CPU inference on image bytes.
  - Return structured detections (label, confidence, bbox) plus an
    annotated preview image.

This module knows nothing about Streamlit. The UI calls
`analyze_image(...)` and renders the returned dataclass.

Model provenance (recorded per TEST_PLAN/documentation requirements):
  - Model:    YOLO11n (nano variant)
  - Package:  ultralytics 8.4.212
  - Licence:  AGPL-3.0 (Ultralytics). Used here under its terms for an
              educational project; see docs for details.
  - Source:   downloaded automatically from Ultralytics assets on first
              use into models/yolo11n.pt (~5 MB), cached afterwards,
              works offline. Weights are never committed to Git.
  - Classes:  80 COCO classes only. The model cannot detect objects
              outside this label set (safety docs note this).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

# --- Configuration ----------------------------------------------------------
DEFAULT_MODEL_PATH = "models/yolo11n.pt"
DEFAULT_CONFIDENCE = 0.40
DEFAULT_INPUT_SIZE = 640  # documented inference size (spec section 11)

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class Detection:
    """One detected object, fully structured."""

    label: str
    confidence: float  # model score, NOT a probability of safety
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels


@dataclass
class DetectionResult:
    """Everything the UI needs after one analysis run."""

    detections: list[Detection] = field(default_factory=list)
    annotated_png: bytes | None = None  # annotated preview, PNG bytes
    image_size: tuple[int, int] = (0, 0)  # width, height
    model_name: str = ""
    inference_ms: int = 0
    error: str | None = None


# --- Pure post-processing helpers (unit-tested without any model) -----------

def filter_by_confidence(
    detections: list[Detection], threshold: float
) -> list[Detection]:
    """Keep only detections with confidence >= threshold, best first."""
    kept = [d for d in detections if d.confidence >= threshold]
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def clamp_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clamp a box to image bounds; return None if it is fully outside."""
    x1, y1, x2, y2 = bbox
    cx1, cy1 = max(0, min(x1, width)), max(0, min(y1, height))
    cx2, cy2 = max(0, min(x2, width)), max(0, min(y2, height))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return (cx1, cy1, cx2, cy2)


def format_label(label: str, confidence: float) -> str:
    """Human-readable label, e.g. 'chair (82%)'."""
    return f"{label} ({confidence * 100:.0f}%)"


# --- Model loading (lazy, cached, CPU) --------------------------------------

def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the YOLO model. Downloads on first use (~5 MB).

    Raises RuntimeError with technical detail on failure; analyze_image()
    wraps this into an actionable, user-facing message.
    """
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)  # downloads to model_path if missing
    except Exception as exc:
        raise RuntimeError(f"{exc.__class__.__name__}: {exc}") from exc
    _MODEL_CACHE[model_path] = model
    return model


def is_model_loaded(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    return model_path in _MODEL_CACHE


# --- Image preparation and annotation ----------------------------------------

def _decode_image(image_bytes: bytes):
    """Decode image bytes to an RGB PIL image; raise ValueError if broken."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"The image could not be decoded for detection "
            f"({exc.__class__.__name__}). Please try a different file."
        ) from exc
    return img


def annotate_image(image_bytes: bytes, detections: list[Detection]) -> bytes:
    """Draw bounding boxes + labels on the image; return PNG bytes.

    Kept deliberately simple: solid rectangle + filled label background so
    the overlay stays readable (avoid clutter, spec 3.1).
    """
    import cv2
    import numpy as np
    from PIL import Image

    pil = _decode_image(image_bytes)
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 112, 122), 2)
        text = format_label(det.label, det.confidence)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 6 if y1 - th - 6 > 0 else y1 + 2
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + base), (0, 112, 122), -1)
        cv2.putText(
            frame, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode the annotated preview image.")
    return buf.tobytes()


# --- Main entry point --------------------------------------------------------

def analyze_image(
    image_bytes: bytes,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    model_path: str = DEFAULT_MODEL_PATH,
) -> DetectionResult:
    """Full pipeline: decode -> inference -> threshold -> annotate.

    Returns a DetectionResult; failures are reported via `result.error`
    so the UI never crashes on a bad image or a broken model setup.
    """
    import time

    result = DetectionResult(model_name="yolo11n")

    try:
        pil = _decode_image(image_bytes)
    except ValueError as exc:
        result.error = str(exc)
        return result
    result.image_size = pil.size  # (width, height)

    try:
        model = _load_model(model_path)
    except Exception as exc:
        result.error = (
            "Could not load the detection model. Check your internet "
            f"connection for the one-time download ({exc}), disk space, "
            "and that 'ultralytics' is installed "
            "(pip install -r requirements.txt)."
        )
        return result

    try:
        start = time.perf_counter()
        raw = model.predict(
            pil, conf=confidence_threshold, imgsz=DEFAULT_INPUT_SIZE,
            device="cpu", verbose=False,
        )
        result.inference_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:  # inference failure on this particular image
        result.error = (
            f"Detection failed while processing this image "
            f"({exc.__class__.__name__}: {exc}). Try a smaller or different "
            "image."
        )
        return result

    detections: list[Detection] = []
    for box in raw[0].boxes:
        vals = box.xyxy[0]
        if hasattr(vals, "tolist"):
            vals = vals.tolist()
        x1, y1, x2, y2 = (int(v) for v in vals)
        clamped = clamp_bbox((x1, y1, x2, y2), pil.width, pil.height)
        if clamped is None:
            continue
        detections.append(
            Detection(
                label=raw[0].names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=clamped,
            )
        )
    result.detections = filter_by_confidence(detections, confidence_threshold)

    try:
        result.annotated_png = annotate_image(image_bytes, result.detections)
    except Exception as exc:
        # Annotation is cosmetic: report detections even if drawing fails.
        result.annotated_png = None
        result.error = (
            f"Detections succeeded but the annotated preview could not be "
            f"drawn ({exc.__class__.__name__}). The list below is still valid."
        )
    return result
