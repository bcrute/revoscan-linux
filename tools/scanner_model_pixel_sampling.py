"""Offline native-pixel model sampling; no application or trajectory integration."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scanner_pixel_grid import ranks

from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.geometry import Point
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH


def pixels(cloud: Sequence[Point], *, capacity: int) -> tuple[int, ...]:
    """Recover native camera pixels, refusing non-pixel coordinates."""
    if type(capacity) is not int or capacity not in (900, 1800) or len(cloud) > capacity:
        raise ValueError("Native pixel sampling requires the explicit 900/1800 retained cap")
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    selected = []
    for point in cloud:
        if len(point) != 3 or not all(math.isfinite(x) for x in point) or point[2] <= 0:
            raise ValueError("Native pixel sampling requires finite positive-depth points")
        u, v = fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("Native pixel projection is nonfinite")
        column, row = round(u), round(v)
        if (
            abs(u - column) > 1e-8
            or abs(v - row) > 1e-8
            or not 0 <= column < FRAME_WIDTH
            or not 0 <= row < FRAME_HEIGHT
        ):
            raise ValueError("Model sampling requires original native pixel coordinates")
        selected.append(row * FRAME_WIDTH + column)
    if len(set(selected)) != len(selected):
        raise ValueError("Native model cloud contains duplicate pixels")
    return tuple(selected)


def indices(cloud: Sequence[Point], limit: int, *, capacity: int) -> tuple[int, ...]:
    """Same exact count as M8, lowest fixed S3 pixel ranks in original input order."""
    if type(limit) is not int or not 1 <= limit <= 1800:
        raise ValueError("Native pixel model budget must be an integer in 1..1800")
    native = pixels(cloud, capacity=capacity)
    if len(cloud) <= limit:
        return tuple(range(len(cloud)))
    keys = ranks(np.asarray(native, dtype=np.int64))
    chosen = sorted(range(len(cloud)), key=lambda i: int(keys[i]))[:limit]
    return tuple(sorted(chosen))
