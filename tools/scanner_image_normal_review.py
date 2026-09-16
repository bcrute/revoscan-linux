"""Read-only raw-image normal directions; no tracker, pose fitting or GPU hooks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.geometry import Point
from scanner3d.registration import voxel_downsample
from scanner3d.simulate import FACTORY_INTRINSICS


def representatives(points: tuple[Point, ...]) -> tuple[tuple[Point, ...], tuple[int, ...]]:
    if not 1 <= len(points) <= 3600 or any(
        len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points
    ):
        raise ValueError("Image review requires 1..3600 finite source points")
    centers = voxel_downsample(points, 1.5)
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(points):
        groups.setdefault(tuple(math.floor(x / 1.5) for x in p), []).append(i)
    selected = tuple(
        min(group, key=lambda i: (math.dist(points[i], center), i))
        for center, group in zip(centers, groups.values(), strict=True)
    )
    return centers, selected


def original_pixels(raw: bytes, points: tuple[Point, ...]) -> np.ndarray:
    if not isinstance(raw, bytes) or len(raw) != 512000 or not 1 <= len(points) <= 3600:
        raise ValueError("Exact immutable raw frame and 1..3600 measured points required")
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (len(points), 3) or not np.isfinite(p).all() or np.any(p[:, 2] <= 0):
        raise ValueError("Finite camera coordinates with positive Z required")
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    u = np.rint(p[:, 0] * fx / p[:, 2] + cx).astype(np.int64)
    v = np.rint(p[:, 1] * fy / p[:, 2] + cy).astype(np.int64)
    if np.any((u < 0) | (u >= 640) | (v < 0) | (v >= 400)):
        raise ValueError("Measured point projects outside original image")
    pixels = v * 640 + u
    z = np.frombuffer(raw, dtype="<u2")[pixels].astype(np.float64) * DEPTH_SCALE_METRES
    actual = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    if not np.array_equal(actual, p):
        raise ValueError("Point is not the exact original measured pixel deprojection")
    return pixels


def image_directions(raw: bytes, pixels: np.ndarray) -> list[dict[str, Any]]:
    if (
        not isinstance(raw, bytes)
        or len(raw) != 512000
        or pixels.ndim != 1
        or not 1 <= len(pixels) <= 3600
        or pixels.dtype.kind not in "iu"
        or np.any(pixels >= 256000)
        or np.any(pixels < 0)
    ):
        raise ValueError("Exact raw bytes and 1..3600 integer image indices required")
    pixels = pixels.astype(np.int64)
    codes = np.frombuffer(raw, dtype="<u2")
    v, u = np.divmod(pixels, 640)
    interior = (u > 0) & (u < 639) & (v > 0) & (v < 399)
    rows: list[dict[str, Any]] = [
        dict(pixel=int(index), status="border", normal=None, tangent_lengths=None)
        for index in pixels
    ]
    positions = np.flatnonzero(interior)
    neighbors = pixels[positions, None] + np.array([-641, -640, -639, -1, 0, 1, 639, 640, 641])
    samples = codes[neighbors].astype(np.float64)
    z = samples * DEPTH_SCALE_METRES
    valid = np.all((z > 60) & (z < 300), axis=1)
    span = z.max(axis=1) - z.min(axis=1)
    # Match M26's code-difference arithmetic, not subtraction after rounding.
    bounded = (samples.max(axis=1) - samples.min(axis=1)) * DEPTH_SCALE_METRES <= 1.5
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    nv, nu = np.divmod(neighbors, 640)
    xyz = np.stack(((nu - cx) * z / fx, (nv - cy) * z / fy, z), axis=2)
    horizontal = xyz[:, 5] - xyz[:, 3]
    vertical = xyz[:, 7] - xyz[:, 1]
    cross = np.cross(horizontal, vertical)
    norms = np.linalg.norm(cross, axis=1)
    lengths = np.stack(
        (np.linalg.norm(horizontal, axis=1), np.linalg.norm(vertical, axis=1)), axis=1
    )
    for local, position in enumerate(positions):
        row = rows[int(position)]
        row["depth_span"] = float(span[local])
        if not valid[local]:
            row["status"] = "missing-or-band"
        elif not bounded[local]:
            row["status"] = "depth-span"
        elif not math.isfinite(float(norms[local])) or norms[local] <= 0:
            row["status"] = "degenerate"
        else:
            row.update(
                status="usable",
                normal=(cross[local] / norms[local]).tolist(),
                tangent_lengths=lengths[local].tolist(),
            )
    return rows


def unoriented_angle(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    values = np.asarray((a, b), dtype=np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all():
        raise ValueError("Two finite normal vectors required")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Normal vectors must be nonzero")
    cosine = abs(float((values[0] / lengths[0]) @ (values[1] / lengths[1])))
    return math.degrees(math.acos(min(1.0, cosine)))
