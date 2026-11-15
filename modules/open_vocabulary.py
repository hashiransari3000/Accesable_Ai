"""Open-vocabulary detection with YOLO-World for AccessAble AI.

The standard nano model only knows its 80 COCO training classes, which
excludes things that matter for accessible navigation (doors, stairs,
curbs, ramps, elevators, handrails). YOLO-World accepts arbitrary text
prompts, so the user can describe what to look for.

Honest limitations (documented in SAFETY_AND_LIMITATIONS):
  - Open-vocabulary detection is less reliable than fixed-class YOLO:
    scores are lower, and unusual or abstract prompts may yield false
    positives or nothing at all.
  - Results still describe only the uploaded image - never live space,
    distance, or safety.
  - The model (~25 MB) plus its CLIP text encoder (ViT-B/32, ~350 MB)
    download once on first use (to models/ and weights/clip/), then work
    fully offline.
"""

from __future__ import annotations

import time

from modules import object_detection as od

DEFAULT_MODEL_PATH = "models/yolov8s-worldv2.pt"
MAX_CLASSES = 30

DEFAULT_VOCABULARY = (
    "person, door, stairs, step, crosswalk, curb, ramp, elevator, handrail, "
    "pillar, bench, chair, table, sign, bicycle, car, stroller, luggage"
)

_MODEL_CACHE: dict[str, object] = {}
_ACTIVE_VOCAB: dict[str, tuple[str, ...]] = {}


def parse_vocabulary(text: str) -> list[str]:
    """Split user text into clean class prompts (lowercase, de-duplicated)."""
    names: list[str] = []
    for chunk in text.replace("\n", ",").split(","):
        name = " ".join(chunk.strip().lower().split())
        if name and name not in names:
            names.append(name)
    return names[:MAX_CLASSES]


def is_model_loaded(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    return model_path in _MODEL_CACHE


def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"{exc.__class__.__name__}: {exc}") from exc
    _MODEL_CACHE[model_path] = model
    return model


def analyze_image(
    image_bytes: bytes,
    classes: list[str],
    confidence_threshold: float = 0.20,
    model_path: str = DEFAULT_MODEL_PATH,
) -> od.DetectionResult:
    """Detect custom classes in one image; failures land in result.error."""
    result = od.DetectionResult(model_name="yolov8s-worldv2")

    cleaned = parse_vocabulary(", ".join(classes))
    if not cleaned:
        result.error = (
            "Add at least one object name to look for (for example: door, "
            "stairs, ramp)."
        )
        return result

    try:
        pil = od._decode_image(image_bytes)
    except ValueError as exc:
        result.error = str(exc)
        return result
    result.image_size = pil.size  # (width, height)

    try:
        model = _load_model(model_path)
    except Exception as exc:
        result.error = (
            "Could not load the open-vocabulary model. Check your internet "
            f"connection for the one-time download ({exc}), disk space, and "
            "that 'ultralytics' is installed (pip install -r requirements.txt)."
        )
        return result

    try:
        with od.INFERENCE_LOCK:
            vocab = tuple(cleaned)
            if _ACTIVE_VOCAB.get(model_path) != vocab:
                model.set_classes(cleaned)
                _ACTIVE_VOCAB[model_path] = vocab
            start = time.perf_counter()
            raw = model.predict(
                pil, conf=confidence_threshold, imgsz=od.DEFAULT_INPUT_SIZE,
                device="cpu", verbose=False,
            )
            result.inference_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        result.error = (
            f"Open-vocabulary detection failed while processing this image "
            f"({exc.__class__.__name__}: {exc}). Try a different image or "
            "fewer object names."
        )
        return result

    result.detections = od.detections_from_raw(
        raw[0], pil.width, pil.height, confidence_threshold
    )

    try:
        result.annotated_png = od.annotate_image(image_bytes, result.detections)
    except Exception as exc:
        result.annotated_png = None
        result.error = (
            f"Detections succeeded but the annotated preview could not be "
            f"drawn ({exc.__class__.__name__}). The list below is still valid."
        )
    return result
