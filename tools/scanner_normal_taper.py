"""M24 tapered local-normal covariance, a closed offline-only experiment."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

from scanner_factorial_config import Factors
from scanner_source_capacity import settings as capacity_settings

from scanner3d import icp_plane, tracking
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid, normalize
from scanner3d.gpu_normals import GpuNormals
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-20", "taper-20")
CAPACITIES = ("ordinary-900", "dense-1800")
ORIGINAL_ICP = icp_plane.icp_point_to_plane
ORIGINAL_ESTIMATE = GpuNormals.estimate_normals
_ACTIVE: ContextVar[bool] = ContextVar("m24_normal_scale_active", default=False)


class TaperDispatchFailed(BaseException):
    """Fatal diagnostic failure bypasses the live wrapper's ordinary CPU fallback."""


def tapered_patch(
    point: Point, neighbors: Sequence[Point]
) -> tuple[list[list[float]], dict[str, Any]]:
    """One fixed kernel; actual weighted support, no sensor-noise interpretation."""
    patch = (point, *neighbors)
    if (
        not 1 <= len(patch) <= 21
        or any(len(p) != 3 for p in patch)
        or not all(math.isfinite(v) for p in patch for v in p)
    ):
        raise ValueError("M24 requires one finite query and at most twenty finite neighbors")
    squared = [
        (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 + (p[2] - point[2]) ** 2 for p in patch
    ]
    radius_squared = max(squared)
    if not math.isfinite(radius_squared):
        raise ValueError("M24 squared support radius overflowed")
    weights = (
        [1.0] * len(patch)
        if radius_squared == 0
        else [max(0.0, 1.0 - value / radius_squared) ** 2 for value in squared]
    )
    total = math.fsum(weights)
    centre = tuple(
        math.fsum(w * p[a] for w, p in zip(weights, patch, strict=True)) / total for a in range(3)
    )
    matrix = [[0.0] * 3 for _ in range(3)]
    for p, w in zip(patch, weights, strict=True):
        delta = [p[a] - centre[a] for a in range(3)]
        for a in range(3):
            for b in range(a, 3):
                matrix[a][b] += (delta[a] * delta[b]) * w
    for a in range(3):
        for b in range(a):
            matrix[a][b] = matrix[b][a]
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError("M24 covariance overflowed")
    return matrix, dict(
        radius_squared=radius_squared,
        positive_neighbors=sum(w > 0 for w in weights[1:]),
        zero_weight_neighbors=sum(w == 0 for w in weights[1:]),
        requested_patch_neighbors=len(neighbors),
        effective_count=total * total / math.fsum(w * w for w in weights),
        weight_sum=total,
        weights=weights,
        centre=list(centre),
    )


def tapered_field(
    points: Sequence[Point],
    eigensolve: Callable[..., Any],
) -> tuple[icp_plane.NormalField, dict[str, Any]]:
    """Fixed SciPy neighbors and CPU covariance; injected batch eigensolve only."""
    if not 6 <= len(points) <= 2048:
        raise ValueError("M24 requires a bounded field of 6..2048 points")
    _, compute = icp_plane._resolve_backend("scipy")
    assert compute is not None
    indices = compute.k_nearest_neighbors(points, 20)
    patches = [
        tapered_patch(p, [points[j] for j in row]) for p, row in zip(points, indices, strict=True)
    ]
    solutions = eigensolve([matrix for matrix, _ in patches])
    estimates = []
    counts = dict(sparse=0, coincident=0, isotropic=0)
    for point, (_, data), (values, vectors) in zip(points, patches, solutions, strict=True):
        if not all(math.isfinite(v) for v in [*values, *(v for row in vectors for v in row)]):
            raise RuntimeError("M24 eigensolver returned non-finite values")
        order = sorted(range(3), key=lambda i: values[i])
        smallest, largest = values[order[0]], values[order[2]]
        normal = None
        if data["positive_neighbors"] < icp_plane.MIN_NORMAL_NEIGHBOURS:
            counts["sparse"] += 1
        elif largest <= 0.0:
            counts["coincident"] += 1
        elif smallest / largest > icp_plane.NORMAL_ISOTROPY_RATIO:
            counts["isotropic"] += 1
        else:
            normal = normalize((vectors[0][order[0]], vectors[1][order[0]], vectors[2][order[0]]))
        estimates.append(icp_plane.NormalEstimate(point, normal, data["positive_neighbors"]))
    limitations = (
        "M24 diagnostic tapered covariance; not measured sensor uncertainty or normal truth.",
        "Twenty neighbors requested; counts include only positive-weight other contributors.",
        "Normals are unoriented; original sparse/coincident/isotropy classifications remain.",
        f"Refusals: {counts['sparse']} sparse, {counts['coincident']} coincident, "
        f"{counts['isotropic']} isotropic.",
    )
    summary: dict[str, Any] = dict(classifications=counts)
    for name in (
        "radius_squared",
        "positive_neighbors",
        "zero_weight_neighbors",
        "effective_count",
    ):
        values = [data[name] for _, data in patches]
        summary[name] = dict(
            minimum=min(values), median=statistics.median(values), maximum=max(values)
        )
    return icp_plane.NormalField(tuple(estimates), 20, limitations), summary


def settings(
    capacity: str, profile: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy, str]:
    if (
        capacity not in CAPACITIES
        or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")
    ):
        raise ValueError("M24 requires its closed capacity/normal-taper/grid/solver roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, solver, None)
    return target, policy, source


class NormalTaperExperiment:
    """Twenty-neighbor observation or scoped tapered covariance; no hidden fallback."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError("M24 requires a declared normal-taper profile")
        self.profile = profile
        self.neighbours = 20
        self.records: list[dict[str, Any]] = []
        self.field_records: list[dict[str, Any]] = []
        self._original: Callable[..., icp_plane.IcpPlaneResult] | None = None

    def estimate(
        self,
        gpu: GpuNormals,
        points: Sequence[Point],
        *,
        neighbours: int = 10,
        backend: Any = "scipy",
    ) -> icp_plane.NormalField:
        if not _ACTIVE.get() or self._original is None:
            raise RuntimeError("M24 field requires its active nonnested context")
        if len(self.field_records) >= 90000:
            raise RuntimeError("M24 exceeds its 90000 field record bound")
        record: dict[str, Any] = dict(
            ordinal=len(self.field_records),
            profile=self.profile,
            points=len(points),
            neighbors=neighbours,
            status="running",
            points_sha256=hashlib.sha256(
                json.dumps(points, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
        )
        self.field_records.append(record)
        try:
            if neighbours != 20 or backend != "scipy" or not 6 <= len(points) <= 2048:
                raise RuntimeError(
                    "M24 field requires twenty-neighbor SciPy support and 6..2048 points"
                )
            before = gpu.calls["jacobi"]
            if self.profile == "observe-20":
                field = ORIGINAL_ESTIMATE(gpu, points, neighbours=neighbours, backend=backend)
                statistics_record = None
            else:
                field, statistics_record = tapered_field(points, gpu.jacobi)
            dispatches = gpu.calls["jacobi"] - before
            if (
                dispatches != 1
                or field.requested_neighbours != 20
                or tuple(e.point for e in field.estimates) != tuple(points)
            ):
                raise RuntimeError(
                    "M24 requires one actual dispatch and unchanged field point identities"
                )
            record.update(
                status="completed",
                field=field.as_dict(),
                statistics=statistics_record,
                jacobi_dispatches=dispatches,
                point_identity_exact=True,
            )
            return field
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            if isinstance(error, Exception):
                raise TaperDispatchFailed(record["error"]) from error
            raise

    def fit(
        self, source: Sequence[Point], target: Sequence[Point], **kwargs: Any
    ) -> icp_plane.IcpPlaneResult:
        if self._original is None or not _ACTIVE.get():
            raise RuntimeError("M24 requires its active nonnested context")
        if len(self.records) >= 90000:
            raise RuntimeError("M24 exceeds the declared 90000 fit-call bound")
        record: dict[str, Any] = dict(
            ordinal=len(self.records),
            profile=self.profile,
            status="running",
            source_points=len(source),
            target_points=len(target),
            requested_normal_neighbours=kwargs.get("normal_neighbours", 10),
            actual_normal_neighbours=self.neighbours,
        )
        self.records.append(record)
        try:
            if (
                record["requested_normal_neighbours"] != 10
                or kwargs.get("backend") != "scipy"
                or kwargs.get("max_iterations", 50) != 50
                or kwargs.get("voxel_size") != 1.5
                or kwargs.get("keep_fraction") != 0.8
                or not isinstance(kwargs.get("initial"), Rigid)
                or icp_plane._NORMAL_ACCELERATOR.get() is None
            ):
                raise RuntimeError("M24 requires original ten-neighbor request and GPU provider")
            effective = kwargs | dict(normal_neighbours=20)
            result = self._original(source, target, **effective)
            if (
                result.normal_neighbours != 20
                or result.voxel_size != 1.5
                or not 1 <= result.iterations <= 50
            ):
                raise RuntimeError("M24 returned ICP metadata does not name actual neighbor count")
            record.update(status="completed", result=result.as_dict())
            return result
        except (ScannerError, ValueError, ArithmeticError) as error:
            record.update(status="refused", error=f"{type(error).__name__}: {error}")
            raise
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if (
            _ACTIVE.get()
            or self._original is not None
            or vars(tracking)["icp_point_to_plane"] is not ORIGINAL_ICP
            or GpuNormals.estimate_normals is not ORIGINAL_ESTIMATE
        ):
            raise RuntimeError("M24 refuses nested context or a nonoriginal tracker ICP hook")
        token = _ACTIVE.set(True)
        self._original = ORIGINAL_ICP

        def estimate(
            gpu: GpuNormals, points: Sequence[Point], **kwargs: Any
        ) -> icp_plane.NormalField:
            return self.estimate(gpu, points, **kwargs)

        try:
            with (
                patch.object(tracking, "icp_point_to_plane", self.fit),
                patch.object(GpuNormals, "estimate_normals", estimate),
            ):
                yield
        finally:
            self._original = None
            _ACTIVE.reset(token)
