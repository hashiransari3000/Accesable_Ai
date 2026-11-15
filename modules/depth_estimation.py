"""Monocular relative-depth estimation for AccessAble AI.

Adds a coarse proximity band ("very close" ... "far") to each detection
using Depth Anything V2 Small (Hugging Face transformers), running on CPU.

What this IS: a relative ordering of things WITHIN one image - useful to
say "the chair is closer than the door" or to prioritize near objects.

What this is NOT: distance measurement. Monocular depth has no metric
scale, no calibration, and no safety meaning. The UI must always say
"relative", never metres, and never imply obstacle-avoidance capability.

The model (~100 MB) downloads once on first use (Hugging Face cache),
then runs fully offline. Inference on CPU: roughly 0.5-1.5 s per image.
"""

from __future__ import annotations

import io

MODEL_NAME = "depth-anything/Depth-Anything-V2-small-hf"

# Normalized depth: 1.0 = nearest in this view, 0.0 = farthest.
BAND_THRESHOLDS = (
    (0.70, "very close"),
    (0.45, "close"),
    (0.22, "mid-distance"),
    (0.0, "far"),
)

_MODEL_CACHE: dict[str, tuple] = {}


def is_loaded(model_name: str = MODEL_NAME) -> bool:
    return model_name in _MODEL_CACHE


def _load_model(model_name: str = MODEL_NAME):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    _MODEL_CACHE[model_name] = (processor, model)
    return processor, model


def proximity_band(value: float) -> str:
    """Map a normalized depth (1 = nearest) to a spoken band label."""
    for threshold, label in BAND_THRESHOLDS:
        if value >= threshold:
            return label
    return "far"


def annotate_result(result, image_bytes: bytes) -> str | None:
    """Fill result.detections[*].proximity; return an error string on failure.

    Detection results stay valid even when depth fails - the caller shows
    a warning but keeps the object list.
    """
    if not result.detections:
        return None

    import time

    import torch
    from PIL import Image
    from torch.nn import functional

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return (
            "Relative proximity could not be estimated because the image "
            f"could not be re-read ({exc.__class__.__name__})."
        )

    try:
        processor, model = _load_model()
    except Exception as exc:
        return (
            "The depth model could not be loaded. Check your internet "
            f"connection for the one-time ~100 MB download ({exc}) and disk "
            "space. Detections above are still valid without proximity."
        )

    try:
        start = time.perf_counter()
        inputs = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        depth = outputs.predicted_depth
        depth = functional.interpolate(
            depth.unsqueeze(1),
            size=(pil.height, pil.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        arr = depth.numpy()
        result.depth_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        return (
            f"Relative proximity estimation failed "
            f"({exc.__class__.__name__}: {exc}). Detections are still valid."
        )

    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo if hi > lo else 1.0

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        region = arr[y1:y2, x1:x2]
        if region.size == 0:
            continue
        normalized = (float(region.mean()) - lo) / span
        det.proximity = proximity_band(normalized)
    return None


__all__ = [
    "BAND_THRESHOLDS",
    "MODEL_NAME",
    "annotate_result",
    "is_loaded",
    "proximity_band",
]
