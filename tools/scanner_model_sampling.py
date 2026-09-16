"""M8 ordinary-model exact-budget stratification, an offline-only experiment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, ClassVar

from scanner_factorial_config import Factors

from scanner3d.geometry import Point, Rigid
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker, TrackingPolicy, _sampled

PROFILES = ("observe-stride", "stratified-model")


def validate(
    profile: str,
    factors: Factors,
    solver_profile: str | None,
    model_profile: str | None,
    source_profile: str | None,
) -> None:
    if (
        profile not in PROFILES
        or factors.sampling not in ("stride", "grid")
        or factors.model_normals != "pooled"
        or factors.convergence != "residual"
        or solver_profile not in ("observe-origin", "centered")
        or model_profile is not None
        or source_profile is not None
    ):
        raise ValueError(
            "M8 requires closed model-sampling/factor/solver roster and ordinary capacity"
        )


def indices(size: int, limit: int, *, position: int, frames: int) -> tuple[int, ...]:
    """One measured point per uniform stratum, staggered across reference frames."""
    if (
        not all(type(n) is int for n in (size, limit, position, frames))
        or not 0 <= size <= 900
        or not 1 <= limit <= 1800
        or not 1 <= frames <= 8
        or not 0 <= position < frames
    ):
        raise ValueError("M8 requires bounded integer sizes, limits and reference positions")
    if size <= limit:
        return tuple(range(size))
    return tuple(
        ((2 * frames * j + 2 * position + 1) * size) // (2 * frames * limit) for j in range(limit)
    )


class ModelSamplingTracker(IncrementalTracker):
    """Only _model_reference is specialized; no source/recovery/global sampling change."""

    expected_policy: ClassVar[TrackingPolicy] = DEFAULT_POLICY
    profiles: ClassVar[tuple[str, ...]] = PROFILES
    selected_count_key: ClassVar[str] = "stratified_points"

    @staticmethod
    def sample_indices(size: int, limit: int, *, position: int, frames: int) -> tuple[int, ...]:
        return indices(size, limit, position=position, frames=frames)

    def selection_indices(
        self, cloud: Sequence[Point], limit: int, *, position: int, frames: int
    ) -> tuple[int, ...]:
        return self.sample_indices(len(cloud), limit, position=position, frames=frames)

    def __init__(self, *, sampling_profile: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            sampling_profile not in self.profiles
            or self._backend != "scipy"
            or self._policy != self.expected_policy
            or not self._point_to_plane
            or self._voxel_size != 1.5
            or self._max_correspondence_distance != 12.0
            or self._keep_fraction != 0.8
        ):
            raise ValueError(
                "M8 requires closed profile and unchanged SciPy/declared-policy settings"
            )
        self.sampling_profile = sampling_profile
        self.model_sampling_records: list[dict[str, Any]] = []
        self._inside_model_sampling = False

    def _model_reference(
        self, previous: Any, predicted: Rigid, *, exclude: frozenset[int] = frozenset()
    ) -> tuple[tuple[Point, ...], tuple[int, ...]]:
        if self._inside_model_sampling or len(self.model_sampling_records) >= 3000:
            raise RuntimeError("M8 refuses nested reference calls or more than 3000 calls")
        record: dict[str, Any] = dict(
            profile=self.sampling_profile,
            accepted_frame_index=len(self._accepted),
            status="running",
            original_reference_verified=False,
            contributors=[],
        )
        self.model_sampling_records.append(record)
        self._inside_model_sampling = True
        try:
            requested = self._reference_frames(previous, predicted, exclude=exclude)
            original = super()._model_reference(previous, predicted, exclude=exclude)
            points, frames = original
            if tuple(requested) != frames or len(frames) > 8 or len(set(frames)) != len(frames):
                raise RuntimeError("M8 requires exact unchanged nonempty reference membership")
            record["reference_frames"] = frames
            record["original_points"] = len(points)
            reconstructed: list[Point] = []
            replacement: list[Point] = []
            if frames:
                limit = self._policy.model_reference_points // len(frames)
                for position, frame in enumerate(frames):
                    accepted = self._accepted[frame]
                    cloud = accepted.cloud
                    if (
                        not cloud
                        or len(cloud) > self._policy.retained_cloud_points
                        or any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in cloud)
                    ):
                        raise RuntimeError("M8 requires finite retained native cloud provenance")
                    selected = self.selection_indices(
                        cloud, limit, position=position, frames=len(frames)
                    )
                    before = _sampled(cloud, limit, offset=position)
                    reconstructed.extend(accepted.pose.apply_all(before))
                    replacement.extend(accepted.pose.apply_all(tuple(cloud[i] for i in selected)))
                    record["contributors"].append(
                        dict(
                            frame=frame,
                            position=position,
                            retained_points=len(cloud),
                            limit=limit,
                            original_points=len(before),
                            **{self.selected_count_key: len(selected)},
                            selected_indices_sha256=hashlib.sha256(
                                ",".join(map(str, selected)).encode("ascii")
                            ).hexdigest(),
                        )
                    )
            if tuple(reconstructed) != points:
                raise RuntimeError("M8 original reference reconstruction differs")
            record["original_reference_verified"] = True
            if len(replacement) > self._policy.model_reference_points:
                raise RuntimeError("M8 replacement exceeds unchanged model point budget")
            result = (
                original
                if self.sampling_profile == "observe-stride"
                else (tuple(replacement), frames)
            )
            record.update(
                status="completed",
                returned_points=len(result[0]),
                **{self.selected_count_key: len(replacement)},
            )
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._inside_model_sampling = False
