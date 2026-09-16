"""M4: measured model-voxel representatives, offline only; no qualification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from scanner_model_normals import NativeCache, NativeFrame, model_field, reference_points

from scanner3d import icp_plane
from scanner3d.geometry import Point
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import DEFAULT_POLICY, MODEL_ORIGIN, IncrementalTracker, TrackCandidate

PROFILES = ("observe-centroid", "observed-point")
_ACTIVE: ContextVar[bool] = ContextVar("m4_model_representation", default=False)


@dataclass(frozen=True)
class Representatives:
    centroids: tuple[Point, ...]
    measured: tuple[Point, ...]
    indices: tuple[int, ...]
    owners: tuple[int, ...]
    distances: tuple[float, ...]


def representatives(
    points: tuple[Point, ...], owners: Sequence[int], *, cell: float = 1.5
) -> Representatives:
    """Existing voxel order/count; nearest contributor with original-index ties."""
    if (
        not 1 <= len(points) <= 1800
        or len(points) != len(owners)
        or not math.isfinite(cell)
        or cell <= 0
        or any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points)
    ):
        raise ValueError("Finite bounded model points, matching owners and positive cell required")
    centroids = voxel_downsample(points, cell)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, point in enumerate(points):
        groups.setdefault(tuple(math.floor(x / cell) for x in point), []).append(index)
    selected = tuple(
        min(group, key=lambda i: (math.dist(points[i], centre), i))
        for centre, group in zip(centroids, groups.values(), strict=True)
    )
    measured = tuple(points[i] for i in selected)
    return Representatives(
        centroids,
        measured,
        selected,
        tuple(owners[i] for i in selected),
        tuple(math.dist(a, b) for a, b in zip(centroids, measured, strict=True)),
    )


class ModelRepresentationTracker(IncrementalTracker):
    """Only the ordinary model ICP target changes; judge/reference clouds do not."""

    def __init__(self, *, model_profile: str, normal_mode: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model_profile not in PROFILES or normal_mode not in ("pooled", "native"):
            raise ValueError("Closed model-representation and normal profiles required")
        if (
            self._backend != "scipy"
            or self._policy != DEFAULT_POLICY
            or not self._point_to_plane
            or self._voxel_size != 1.5
            or self._max_correspondence_distance != 12.0
            or self._keep_fraction != 0.8
        ):
            raise ValueError("M4 requires unchanged SciPy/default-policy point-to-plane settings")
        self.model_profile = model_profile
        self.normal_mode = normal_mode
        self.native_cache = NativeCache()
        self.model_representation_records: list[dict[str, Any]] = []
        self.model_normal_records: list[dict[str, Any]] | None = (
            [] if normal_mode == "native" else None
        )

    def _candidate(self, *args: Any, **kwargs: Any) -> TrackCandidate | None:
        if kwargs["origin"] != MODEL_ORIGIN:
            return super()._candidate(*args, **kwargs)
        if kwargs.get("coarse_to_fine", False) or len(self.model_representation_records) >= 3000:
            raise ValueError("M4 is ordinary-model only with a 3000-call diagnostic bound")
        if _ACTIVE.get():
            raise RuntimeError("Nested M4 model contexts are not supported")
        source = tuple(args[1] if len(args) > 1 else kwargs["source"])
        expected = tuple(kwargs["reference"])
        if not source or len(expected) < self._policy.min_track_points:
            return super()._candidate(*args, **kwargs)
        frames = []
        for index in kwargs["reference_frames"]:
            accepted = self._accepted[index]
            if accepted.cloud is None:
                raise ValueError("Model representation requires retained native provenance")
            frames.append(NativeFrame(index, accepted.cloud, accepted.pose))
        actual, origins = reference_points(frames, 1800)
        if actual != expected:
            raise ValueError("Reconstructed model points differ from ordinary target")
        reps = representatives(expected, [index for index, _ in origins])
        sampled = reps.measured if self.model_profile == "observed-point" else reps.centroids
        record: dict[str, Any] = dict(
            profile=self.model_profile,
            normal_mode=self.normal_mode,
            model_call=len(self.model_representation_records),
            accepted_frame_index=args[0] if args else kwargs["index"],
            reference_frames=kwargs["reference_frames"],
            model_points=len(expected),
            voxel_points=len(sampled),
            selected_contributor_indices=reps.indices,
            selected_contributor_owners=reps.owners,
            centroid_displacements=reps.distances,
            source_downsampling_unchanged=False,
            target_substitution_verified=False,
            status="running",
        )
        self.model_representation_records.append(record)
        calls = 0
        interception_errors: list[str] = []
        if id(vars(icp_plane)["voxel_downsample"]) != id(voxel_downsample):
            raise RuntimeError("M4 requires the original ICP voxel provider outside its context")

        def provided_voxels(points: Sequence[Point], cell: float) -> tuple[Point, ...]:
            nonlocal calls
            calls += 1
            if cell != 1.5 or calls > 2 or tuple(points) != (source if calls == 1 else expected):
                interception_errors.append("M4 received unexpected ICP downsampling call")
                raise RuntimeError(interception_errors[-1])
            if calls == 1:
                result = voxel_downsample(points, cell)
                record["source_downsampling_unchanged"] = True
                return result
            record["target_substitution_verified"] = True
            return sampled

        field = None
        if self.normal_mode == "native":
            native, owners = model_field(frames, expected, self.native_cache)
            if tuple(e.point for e in native.estimates) != reps.centroids or owners != reps.owners:
                raise ValueError("Native directions and representative provenance disagree")
            field = icp_plane.NormalField(
                tuple(
                    icp_plane.NormalEstimate(point, estimate.normal, estimate.neighbours)
                    for point, estimate in zip(sampled, native.estimates, strict=True)
                ),
                native.requested_neighbours,
                native.limitations
                if self.model_profile == "observe-centroid"
                else (
                    "Offline native directions at measured model-voxel contributors.",
                    "No missing direction replaced; no quality qualification.",
                ),
            )
            record["usable_native_normals"] = len(field.usable)
            assert self.model_normal_records is not None
            self.model_normal_records.append(record)

        def provided_normals(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
            assert field is not None
            if points != sampled or neighbours != field.requested_neighbours:
                interception_errors.append("M4 native normal mapping or neighbourhood changed")
                raise RuntimeError(interception_errors[-1])
            return field

        token = _ACTIVE.set(True)
        try:
            with (
                patch.object(icp_plane, "voxel_downsample", provided_voxels),
                nullcontext() if field is None else icp_plane.normal_acceleration(provided_normals),
            ):
                result = super()._candidate(*args, **kwargs)
            if calls != 2 or interception_errors:
                raise RuntimeError(
                    f"M4 target interception was not verified: {calls}, {interception_errors}"
                )
            record["status"] = "completed"
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            _ACTIVE.reset(token)
