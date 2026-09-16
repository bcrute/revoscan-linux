"""Offline fixed confidence-tapered direction combination; no tracking integration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from scanner_normal_hybrid_review import Vector, combine


def combine_confidence(
    original: Sequence[float] | None, field: dict[str, Any], *, maximum_weight: float
) -> tuple[Vector | None, dict[str, Any]]:
    """Use only declared support/RMS/agreement factors; preserve PCA availability."""
    if type(maximum_weight) not in (int, float) or maximum_weight not in (0.5, 1.0):
        raise ValueError("Only declared maximum weights 0.5/1 are supported")
    if not isinstance(field, dict):
        raise ValueError("A recorded organized field is required")
    radius = field.get("radius")
    if type(radius) is not int or radius not in (5, 10):
        raise ValueError("Only declared image radii 5/10 are supported")
    if field.get("status") not in (
        "usable",
        "border",
        "center-missing-or-band",
        "insufficient-support",
        "degenerate",
        "nonplanar",
    ):
        raise ValueError("Unknown organized-field status")
    if (field.get("status") == "usable") != (field.get("normal") is not None):
        raise ValueError("Usable status and available direction must agree")
    first, _ = combine(original, None, weight=1.0)
    second, _ = combine(original, field.get("normal"), weight=1.0)
    info = dict(
        mode="missing", weight=0.0, coverage=None, planarity=None, agreement=None, angle=None
    )
    if first is None:
        return None, info
    info["mode"] = "fallback"
    if field.get("status") != "usable":
        if field.get("normal") is not None:
            raise ValueError("Unavailable field cannot provide a direction")
        return first, info
    if second is None:
        raise ValueError("Usable field must have a unit direction")
    count, minimum = field.get("support_count"), field.get("required_support")
    area = (2 * radius + 1) ** 2
    if (
        type(count) is not int
        or type(minimum) is not int
        or minimum != (3 * area + 3) // 4
        or not minimum <= count <= area
    ):
        raise ValueError("Invalid original support counts")
    rms = field.get("perpendicular_rms")
    if type(rms) not in (int, float):
        raise ValueError("Invalid usable-plane perpendicular RMS")
    checked_rms = cast(float, rms)
    if not math.isfinite(checked_rms) or not 0 <= checked_rms <= 0.25:
        raise ValueError("Invalid usable-plane perpendicular RMS")
    cosine = abs(math.fsum(a * b for a, b in zip(first, second, strict=True))) / (
        math.hypot(*first) * math.hypot(*second)
    )
    angle = math.degrees(math.acos(min(1.0, cosine)))
    coverage = (count - minimum) / (area - minimum)
    planarity = max(0.0, 1 - checked_rms / 0.25) ** 2
    agreement = max(0.0, 1 - (angle / 30) ** 2) ** 2
    weight = maximum_weight * coverage * planarity * agreement
    info.update(
        weight=weight, coverage=coverage, planarity=planarity, agreement=agreement, angle=angle
    )
    if weight == 0:
        return first, info
    mixed = tuple((1 - weight) * a + weight * b for a, b in zip(first, second, strict=True))
    length = math.hypot(*mixed)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Tapered combination is nonfinite or degenerate")
    info["mode"] = "weighted"
    return (mixed[0] / length, mixed[1] / length, mixed[2] / length), info
