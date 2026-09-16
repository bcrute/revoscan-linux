"""Offline image-local plane estimates; no tracker, hardware or GPU integration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.simulate import FACTORY_INTRINSICS

RADII = (2, 5, 10)


def plane_directions(raw: bytes, pixels: np.ndarray, *, radius: int) -> list[dict[str, Any]]:
    """Estimate directions from original support; unavailable results stay explicit."""
    if (
        not isinstance(raw, bytes)
        or len(raw) != 512000
        or type(radius) is not int
        or radius not in RADII
        or not isinstance(pixels, np.ndarray)
        or pixels.ndim != 1
        or not 1 <= len(pixels) <= 3600
        or pixels.dtype.kind not in "iu"
        or np.any(pixels < 0)
        or np.any(pixels >= 256000)
    ):
        raise ValueError("Exact raw bytes, integer native pixels and a declared radius required")
    pixels = pixels.astype(np.int64)
    codes = np.frombuffer(raw, dtype="<u2")
    v, u = np.divmod(pixels, 640)
    area = (2 * radius + 1) ** 2
    minimum = math.ceil(3 * area / 4)
    rows: list[dict[str, Any]] = [
        dict(
            pixel=int(pixel),
            radius=radius,
            status="border",
            normal=None,
            support_count=None,
            required_support=minimum,
            eigenvalues=None,
            centroid=None,
            perpendicular_rms=None,
        )
        for pixel in pixels
    ]
    positions = np.flatnonzero(
        (u >= radius) & (u < 640 - radius) & (v >= radius) & (v < 400 - radius)
    )
    offsets = np.array(
        [dv * 640 + du for dv in range(-radius, radius + 1) for du in range(-radius, radius + 1)],
        dtype=np.int64,
    )
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    for start in range(0, len(positions), 256):
        batch = positions[start : start + 256]
        neighborhood = pixels[batch, None] + offsets
        samples = codes[neighborhood].astype(np.float64)
        center = codes[pixels[batch]].astype(np.float64)
        z = samples * DEPTH_SCALE_METRES
        valid = (
            (z > 60) & (z < 300) & (np.abs(samples - center[:, None]) * DEPTH_SCALE_METRES <= 3.0)
        )
        counts = valid.sum(axis=1)
        nv, nu = np.divmod(neighborhood, 640)
        xyz = np.stack(((nu - cx) * z / fx, (nv - cy) * z / fy, z), axis=2)
        means = (xyz * valid[:, :, None]).sum(axis=1) / np.maximum(counts, 1)[:, None]
        centered = (xyz - means[:, None, :]) * valid[:, :, None]
        cov = np.einsum("bni,bnj->bij", centered, centered) / np.maximum(counts, 1)[:, None, None]
        values, vectors = np.linalg.eigh(cov)
        for local, position in enumerate(batch):
            row = rows[int(position)]
            row["support_count"] = int(counts[local])
            if not 60 < center[local] * DEPTH_SCALE_METRES < 300:
                row["status"] = "center-missing-or-band"
                continue
            if counts[local] < minimum:
                row["status"] = "insufficient-support"
                continue
            eigenvalues = values[local]
            row.update(eigenvalues=eigenvalues.tolist(), centroid=means[local].tolist())
            if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= 1e-12 * max(
                float(eigenvalues[2]), 1.0
            ):
                row["status"] = "degenerate"
                continue
            rms = math.sqrt(max(0.0, float(eigenvalues[0])))
            row["perpendicular_rms"] = rms
            if rms > 0.25:
                row["status"] = "nonplanar"
                continue
            normal = vectors[local, :, 0]
            if float(normal @ means[local]) > 0:
                normal = -normal
            row.update(status="usable", normal=normal.tolist())
    return rows
