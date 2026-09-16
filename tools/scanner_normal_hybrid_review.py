"""Pure diagnostic normal combination; no tracking integration or missing-normal creation."""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = tuple[float, float, float]


def _unit(value: Sequence[float] | None) -> Vector | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("A unit direction requires three numeric components")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Finite numeric direction components required")
    if not math.isclose(math.hypot(*value), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Direction must already be unit length")
    return value[0], value[1], value[2]


def combine(
    original: Sequence[float] | None, organized: Sequence[float] | None, *, weight: float
) -> tuple[Vector | None, str]:
    """Preserve original availability; align signs before a fixed normalized blend."""
    if type(weight) not in (int, float) or weight not in (0.5, 1.0):
        raise ValueError("Only declared weights 0.5 and 1 are permitted")
    first, second = _unit(original), _unit(organized)
    if first is None:
        return None, "missing"
    if second is None:
        return first, "fallback"
    sign = -1.0 if math.fsum(a * b for a, b in zip(first, second, strict=True)) < 0 else 1.0
    mixed = tuple((1 - weight) * a + weight * sign * b for a, b in zip(first, second, strict=True))
    length = math.hypot(*mixed)
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Combination is nonfinite or degenerate")
    return (mixed[0] / length, mixed[1] / length, mixed[2] / length), (
        "organized" if weight == 1 else "blend"
    )
