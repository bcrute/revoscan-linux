"""S2 fixed-rank pixel sampling; diagnostic only, no application integration."""

from __future__ import annotations

import numpy as np

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import (
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
)
from scanner3d.preparation import (
    DepthBand,
    PreparedCloud,
    _filter_thinned,
    ensure_depth_band,
    ensure_target_points,
)
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.v4l2_capture import FRAME_WIDTH


def ranks(indices: np.ndarray) -> np.ndarray:
    # Fixed, bijective 32-bit finalizer. uint64 products avoid implicit-overflow
    # warnings; masking explicitly implements the declared 32-bit arithmetic.
    x: np.ndarray = indices.astype(np.uint64)
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return np.asarray(x ^ (x >> 16), dtype=np.uint64)


def select_pixels(indices: np.ndarray, target_points: int) -> np.ndarray:
    ensure_target_points(target_points)
    step = max(1, len(indices) // target_points)
    count = (len(indices) + step - 1) // step
    if count == len(indices):
        return indices.copy()
    chosen = np.argpartition(ranks(indices), count - 1)[:count]
    return np.sort(indices[chosen])


def prepare_frame(
    raw: bytes, *, target_points: int, depth_band: DepthBand | None = None
) -> PreparedCloud:
    ensure_target_points(target_points)
    band = ensure_depth_band(depth_band)
    ensure_frame_length(raw)
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    scale = ensure_scale_usable(DEPTH_SCALE_METRES)
    samples = np.frombuffer(raw, dtype="<u2")
    indices = np.flatnonzero(samples)
    z = samples[indices].astype(np.float64) * scale
    rows, columns = np.divmod(indices, FRAME_WIDTH)
    x = (columns - cx) * z / fx
    y = (rows - cy) * z / fy
    finite = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
    clipped = np.flatnonzero(finite & (z > band.near) & (z < band.far))
    selected_pixels = select_pixels(indices[clipped], target_points)
    selected = np.searchsorted(indices, selected_pixels)
    points = [(float(x[i]), float(y[i]), float(z[i])) for i in selected]
    return _filter_thinned(
        points,
        target_points=target_points,
        deprojected_points=int(finite.sum()),
        clipped_points=len(clipped),
        band=band,
    )
