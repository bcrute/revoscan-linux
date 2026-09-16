"""R2 native-frame model normals. Offline diagnosis only, no GUI integration.

Rejected on both good controls on 8 September 2026: original checked pairs
regressed beyond the frozen one-percentage-point guard. Retained to reproduce
that failure, not recommended for live tracking; see reliability-reference-model.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from scanner3d import icp_plane
from scanner3d.geometry import Point, Rigid
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import (
    DEFAULT_POLICY,
    MODEL_ORIGIN,
    IncrementalTracker,
    TrackCandidate,
    _sampled,
)

Provider = Callable[[tuple[Point, ...], int], icp_plane.NormalField]


@dataclass(frozen=True)
class NativeFrame:
    index: int
    points: tuple[Point, ...]
    pose: Rigid


class NativeCache:
    """Bounded native fields, keyed by the actual provider as well as the cloud."""

    def __init__(self) -> None:
        self.estimate = lru_cache(maxsize=icp_plane.NORMAL_CACHE_ENTRIES)(self._estimate)

    @staticmethod
    def _estimate(
        points: tuple[Point, ...], neighbours: int, provider: Provider | None
    ) -> icp_plane.NormalField:
        field = (
            provider(points, neighbours)
            if provider is not None
            else icp_plane.estimate_normals(points, neighbours=neighbours, backend="scipy")
        )
        if (
            field.requested_neighbours != neighbours
            or tuple(e.point for e in field.estimates) != points
        ):
            raise ValueError("Native estimator changed point coordinates/order/count or neighbours")
        for estimate in field.estimates:
            n = estimate.normal
            if n is not None and (
                len(n) != 3 or not all(math.isfinite(x) for x in n) or math.hypot(*n) == 0.0
            ):
                raise ValueError("Native normal is nonfinite, zero or malformed")
        return field


def reference_points(
    frames: Sequence[NativeFrame], budget: int
) -> tuple[tuple[Point, ...], tuple[tuple[int, Point], ...]]:
    """Exactly the existing model's staggered sampling, with its origins retained."""
    if not frames or len({f.index for f in frames}) != len(frames):
        raise ValueError("Distinct nonempty native frames required")
    if not 1 <= budget <= 1800 or len(frames) > 8:
        raise ValueError("Ordinary model bounds: at most eight frames and 1800 points")
    per_frame = max(1, budget // len(frames))
    points: list[Point] = []
    origins: list[tuple[int, Point]] = []
    for position, frame in enumerate(frames):
        if not frame.points or len(frame.points) > 900:
            raise ValueError("Each native retained cloud needs 1..900 points")
        selected = _sampled(frame.points, per_frame, offset=position)
        points.extend(frame.pose.apply_all(selected))
        origins.extend((frame.index, p) for p in selected)
    return tuple(points[:budget]), tuple(origins[:budget])


def voxel_field(
    points: tuple[Point, ...],
    estimates: Sequence[icp_plane.NormalEstimate],
    owners: Sequence[int],
    *,
    cell: float,
    neighbours: int,
) -> tuple[icp_plane.NormalField, tuple[int, ...]]:
    """Keep original centroids; closest contributor's normal, first-index ties."""
    if tuple(e.point for e in estimates) != points or len(owners) != len(points):
        raise ValueError("Point/normal/owner mapping differs")
    sampled = voxel_downsample(points, cell)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, point in enumerate(points):
        key = tuple(math.floor(x / cell) for x in point)
        groups.setdefault(key, []).append(index)
    selected = [
        min(group, key=lambda i: (math.dist(points[i], centre), i))
        for centre, group in zip(sampled, groups.values(), strict=True)
    ]
    field = icp_plane.NormalField(
        tuple(
            icp_plane.NormalEstimate(centre, estimates[i].normal, estimates[i].neighbours)
            for centre, i in zip(sampled, selected, strict=True)
        ),
        neighbours,
        (
            "Offline model normals: native-frame PCA directions rotated into model coordinates.",
            "Unchanged voxel centroids; nearest contributor's direction, original-index tie break.",
            "Missing native normals remain missing; no quality verdict or pose correction.",
        ),
    )
    return field, tuple(owners[i] for i in selected)


def model_field(
    frames: Sequence[NativeFrame],
    expected: tuple[Point, ...],
    cache: NativeCache,
    *,
    budget: int = 1800,
    cell: float = 1.5,
    neighbours: int = 10,
) -> tuple[icp_plane.NormalField, tuple[int, ...]]:
    points, origins = reference_points(frames, budget)
    if points != expected:
        raise ValueError("Reconstructed model points differ from the ordinary model")
    provider = icp_plane._NORMAL_ACCELERATOR.get()
    native = {}
    for frame in frames:
        field = cache.estimate(frame.points, neighbours, provider)
        rotation = Rigid.of(frame.pose.rotation, (0.0, 0.0, 0.0))
        for estimate in field.estimates:
            n = estimate.normal
            native[(frame.index, estimate.point)] = icp_plane.NormalEstimate(
                frame.pose.apply(estimate.point),
                None if n is None else rotation.apply(n),
                estimate.neighbours,
            )
    estimates = [native[origin] for origin in origins]
    return voxel_field(points, estimates, [f for f, _ in origins], cell=cell, neighbours=neighbours)


class NativeModelTracker(IncrementalTracker):
    """Closed ordinary-model normal substitution; every other candidate is unchanged."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            self._backend != "scipy"
            or self._policy != DEFAULT_POLICY
            or not self._point_to_plane
            or self._voxel_size != 1.5
            or self._max_correspondence_distance != 12.0
            or self._keep_fraction != 0.8
        ):
            raise ValueError(
                "R2 requires the unchanged SciPy/default-policy point-to-plane settings"
            )
        self.native_cache = NativeCache()
        self.model_normal_records: list[dict[str, Any]] = []

    def _candidate(self, *args: Any, **kwargs: Any) -> TrackCandidate | None:
        if kwargs["origin"] != MODEL_ORIGIN:
            return super()._candidate(*args, **kwargs)
        if kwargs.get("coarse_to_fine", False) or len(self.model_normal_records) >= 3000:
            raise ValueError("R2 is ordinary-model only with a 3000-call diagnostic bound")
        frames = []
        for index in kwargs["reference_frames"]:
            item = self._accepted[index]
            if item.cloud is None:
                raise ValueError("Model normal provenance requires a retained native cloud")
            frames.append(NativeFrame(index, item.cloud, item.pose))
        expected = tuple(kwargs["reference"])
        field, _ = model_field(frames, expected, self.native_cache)
        sampled = tuple(e.point for e in field.estimates)
        assert sampled == voxel_downsample(expected, self._voxel_size)

        def provided(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
            if points != sampled or neighbours != field.requested_neighbours:
                raise ValueError("R2 normal context received a different target or neighbourhood")
            return field

        record: dict[str, Any] = dict(
            model_call=len(self.model_normal_records),
            accepted_frame_index=args[0] if args else kwargs["index"],
            reference_frames=kwargs["reference_frames"],
            model_points=len(expected),
            voxel_points=len(sampled),
            usable_native_normals=len(field.usable),
            status="running",
        )
        self.model_normal_records.append(record)
        try:
            with icp_plane.normal_acceleration(provided):
                result = super()._candidate(*args, **kwargs)
            record["status"] = "completed"
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
