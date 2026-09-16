"""M11 voxel-before-budget ordinary model, closed offline-only experiment."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Any, ClassVar

from scanner_dense_model_sampling import EXPECTED_POLICY
from scanner_factorial_config import Factors

from scanner3d.geometry import Point, Rigid
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker, TrackingPolicy, _sampled

PROFILES = ("observe-stride", "voxel-first-model")
CAPACITIES = ("ordinary-900", "dense-1800")


def settings(
    capacity: str, profile: str, factors: Factors, solver_profile: str
) -> tuple[int, TrackingPolicy, str | None]:
    if (
        capacity not in CAPACITIES
        or profile not in PROFILES
        or factors.values != ("grid", "pooled", "residual")
        or solver_profile not in ("observe-origin", "centered")
    ):
        raise ValueError("M11 requires its closed capacity/profile/grid/solver roster")
    if capacity == "ordinary-900":
        return 900, DEFAULT_POLICY, None
    return 1800, EXPECTED_POLICY, "source-1800"


def indices(size: int, limit: int = 1800) -> tuple[int, ...]:
    """Bounded midpoint strata on the joint first-encounter voxel list."""
    if type(size) is not int or not 0 <= size <= 14400 or type(limit) is not int or limit != 1800:
        raise ValueError("M11 requires a bounded union and the unchanged 1800 model cap")
    if size <= limit:
        return tuple(range(size))
    return tuple(((2 * j + 1) * size) // (2 * limit) for j in range(limit))


def point_digest(points: Sequence[Point]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<ddd", *p) for p in points)).hexdigest()


class VoxelModelTracker(IncrementalTracker):
    """Only ordinary model construction changes; source and recovery are inherited."""

    expected_policy: ClassVar[TrackingPolicy] = DEFAULT_POLICY

    def __init__(self, *, sampling_profile: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            sampling_profile not in PROFILES
            or self._backend != "scipy"
            or self._policy != self.expected_policy
            or not self._point_to_plane
            or self._voxel_size != 1.5
            or self._max_correspondence_distance != 12.0
            or self._keep_fraction != 0.8
        ):
            raise ValueError("M11 requires unchanged SciPy/declared-policy settings")
        self.sampling_profile = sampling_profile
        self.model_sampling_records: list[dict[str, Any]] = []
        self._inside_model_sampling = False

    def _model_reference(
        self, previous: Any, predicted: Rigid, *, exclude: frozenset[int] = frozenset()
    ) -> tuple[tuple[Point, ...], tuple[int, ...]]:
        if self._inside_model_sampling or len(self.model_sampling_records) >= 3000:
            raise RuntimeError("M11 refuses nested reference calls or more than 3000 calls")
        record: dict[str, Any] = dict(
            profile=self.sampling_profile, accepted_frame_index=len(self._accepted),
            status="running", original_reference_verified=False, contributors=[],
        )
        self.model_sampling_records.append(record)
        self._inside_model_sampling = True
        try:
            requested = self._reference_frames(previous, predicted, exclude=exclude)
            original = super()._model_reference(previous, predicted, exclude=exclude)
            points, frames = original
            if tuple(requested) != frames or len(frames) > 8 or len(set(frames)) != len(frames):
                raise RuntimeError("M11 requires exact unchanged reference membership")
            record.update(reference_frames=frames, original_points=len(points))
            reconstructed: list[Point] = []
            union: list[Point] = []
            voxel_frames: dict[tuple[int, ...], set[int]] = {}
            if frames:
                limit = self._policy.model_reference_points // len(frames)
                for position, frame in enumerate(frames):
                    accepted = self._accepted[frame]
                    cloud = accepted.cloud
                    if (
                        not cloud or len(cloud) > self._policy.retained_cloud_points
                        or any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in cloud)
                    ):
                        raise RuntimeError("M11 requires finite retained native cloud provenance")
                    before = _sampled(cloud, limit, offset=position)
                    reconstructed.extend(accepted.pose.apply_all(before))
                    world = tuple(accepted.pose.apply_all(cloud))
                    if any(not all(math.isfinite(x) for x in p) for p in world):
                        raise RuntimeError("M11 requires finite saved world coordinates")
                    union.extend(world)
                    for point in world:
                        key = tuple(math.floor(x / 1.5) for x in point)
                        voxel_frames.setdefault(key, set()).add(frame)
                    record["contributors"].append(dict(
                        frame=frame, position=position, retained_points=len(cloud),
                        limit=limit, original_points=len(before),
                    ))
            if tuple(reconstructed) != points:
                raise RuntimeError("M11 original reference reconstruction differs")
            record["original_reference_verified"] = True
            if len(union) > 14400:
                raise RuntimeError("M11 union exceeds declared temporary CPU bound")
            voxels = voxel_downsample(union, 1.5) if union else ()
            if len(voxels) != len(voxel_frames):
                raise RuntimeError("M11 contributor voxel keys differ from original reducer")
            if voxels and voxel_downsample(voxels, 1.5) != voxels:
                raise RuntimeError("M11 full union voxel reduction is not exactly idempotent")
            selected = indices(len(voxels))
            replacement = tuple(voxels[i] for i in selected)
            if replacement and voxel_downsample(replacement, 1.5) != replacement:
                raise RuntimeError("M11 selected voxel reduction is not exactly idempotent")
            support = list(voxel_frames.values())
            for contributor in record["contributors"]:
                represented = sum(contributor["frame"] in support[i] for i in selected)
                contributor["selected_voxel_contributions"] = represented
                if not represented:
                    raise RuntimeError("M11 declared reference has no selected contribution")
            result = (
                original if self.sampling_profile == "observe-stride" else (replacement, frames)
            )
            record.update(
                status="completed", union_points=len(union), available_voxels=len(voxels),
                voxel_first_points=len(replacement), returned_points=len(result[0]),
                full_and_selected_voxels_exactly_idempotent=True,
                union_points_sha256=point_digest(union), voxels_sha256=point_digest(voxels),
                selected_points_sha256=point_digest(replacement),
                selected_indices_sha256=hashlib.sha256(
                    ",".join(map(str, selected)).encode("ascii")
                ).hexdigest(),
            )
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._inside_model_sampling = False


class DenseVoxelModelTracker(VoxelModelTracker):
    """Identical joint-voxel rule, requiring exactly M7's explicit source capacity."""

    expected_policy = EXPECTED_POLICY
