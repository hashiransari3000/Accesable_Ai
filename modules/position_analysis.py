"""Coarse relative-position estimation for AccessAble AI.

Pure functions. Given a bounding box and the image dimensions, classify
where the object APPEARS IN THE IMAGE using the box centre. This says
nothing about the physical world: a 2D image gives no reliable distance,
depth, or collision risk (safety docs, spec section 3.2).

All phrasing is deliberately cautious ("appears on the left side of the
image"), never a claim about the user's surroundings.
"""

from __future__ import annotations

from modules.object_detection import Detection

REGIONS_H = ("left", "center", "right")
REGIONS_V = ("upper", "middle", "lower")


def horizontal_region(
    bbox: tuple[int, int, int, int], width: int
) -> str:
    """Return 'left' | 'center' | 'right' for the box centre.

    The image is split into thirds. A box whose centre falls in the
    middle third counts as 'center'; boxes near boundaries resolve to
    whichever third contains the centre (P2/P4 in TEST_PLAN).
    """
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2
    if width <= 0:
        return "center"  # safe default for degenerate input (P6)
    third = width / 3
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "center"


def vertical_region(
    bbox: tuple[int, int, int, int], height: int
) -> str:
    """Return 'upper' | 'middle' | 'lower' for the box centre."""
    _, y1, _, y2 = bbox
    cy = (y1 + y2) / 2
    if height <= 0:
        return "middle"
    third = height / 3
    if cy < third:
        return "upper"
    if cy > 2 * third:
        return "lower"
    return "middle"


def describe_position(
    detection: Detection,
    width: int,
    height: int,
    include_vertical: bool = False,
) -> str:
    """Cautious, user-friendly description of where the box appears."""
    h = horizontal_region(detection.bbox, width)
    v = vertical_region(detection.bbox, height)
    if include_vertical:
        return f"appears in the {v} {h} area of the image"
    return f"appears on the {h} side of the image"
