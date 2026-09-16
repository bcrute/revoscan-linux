"""M14 stable retained pixel identities at exactly the original retained count."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

from scanner_capacity_split import settings as capacity_settings
from scanner_factorial_config import Factors
from scanner_model_pixel_sampling import indices

from scanner3d import tracking
from scanner3d.geometry import Point, Rigid
from scanner3d.tracking import IncrementalTracker, KeyframeDecision, TrackingPolicy

PROFILES = ("retained-stride", "retained-pixel-grid")
SOURCE_PROFILE = "prepared-1800-source-1800-retained-900"
ORIGINAL_SAMPLED = tracking._sampled
EXPECTED_POLICY = capacity_settings(
    SOURCE_PROFILE, Factors("grid", "pooled", "residual"), "observe-origin", None
)[1]


def settings(
    profile: str, normal: str, factors: Factors, solver: str
) -> tuple[int, TrackingPolicy]:
    if (
        profile not in PROFILES or normal not in ("observe-10", "normal-20")
        or factors.values != ("grid", "pooled", "residual")
        or solver not in ("observe-origin", "centered")
    ):
        raise ValueError("M14 requires its closed retention/normal/grid/solver roster")
    return capacity_settings(SOURCE_PROFILE, factors, solver, None)


class RetainedPixelTracker(IncrementalTracker):
    """Only retention inside admission changes; input fitting/model sampling are inherited."""

    def __init__(self, *, retention_profile: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            retention_profile not in PROFILES or self._policy != EXPECTED_POLICY
            or self._backend != "scipy" or not self._point_to_plane
            or self._voxel_size != 1.5 or self._max_correspondence_distance != 12.0
            or self._keep_fraction != .8
        ):
            raise ValueError("M14 requires exact declared policy and original alignment settings")
        self.retention_profile = retention_profile
        self.retention_records: list[dict[str, Any]] = []
        self._inside_retention = False

    def _admit(
        self, index: int, points: tuple[Point, ...], pose: Rigid, *, overlap: float, anchor: bool
    ) -> tuple[KeyframeDecision, ...]:
        if (
            self._inside_retention or len(self.retention_records) >= 3000
            or tracking._sampled is not ORIGINAL_SAMPLED
        ):
            raise RuntimeError("M14 refuses nested admission, nonoriginal sampler or overflow")
        record: dict[str, Any] = dict(
            status="running", profile=self.retention_profile, accepted_frame_index=index,
            prepared_points=len(points), sampler_calls=0,
        )
        self.retention_records.append(record)
        self._inside_retention = True
        returned: tuple[Point, ...] | None = None
        selected: tuple[int, ...] = ()
        replacement: tuple[Point, ...] = ()

        def sampled(
            cloud: tuple[Point, ...], limit: int, *, offset: int = 0
        ) -> tuple[Point, ...]:
            nonlocal returned
            record["sampler_calls"] += 1
            if record["sampler_calls"] != 1 or cloud is not points or limit != 900 or offset != 0:
                raise RuntimeError("M14 requires exactly the original retained-cloud sampler call")
            before = ORIGINAL_SAMPLED(cloud, limit, offset=offset)
            if len(before) != len(replacement):
                raise RuntimeError("M14 retained sample counts differ")
            returned = before if self.retention_profile == "retained-stride" else replacement
            record.update(
                original_retained_points=len(before), returned_points=len(returned),
                proposed_retained_points=len(replacement),
                selected_indices_sha256=hashlib.sha256(
                    ",".join(map(str, selected)).encode("ascii")
                ).hexdigest(),
            )
            return returned

        try:
            # Validate native provenance before the base admission mutates state.
            selected = indices(points, len(ORIGINAL_SAMPLED(points, 900)), capacity=1800)
            replacement = tuple(points[i] for i in selected)
            with patch.object(tracking, "_sampled", sampled):
                decisions = super()._admit(index, points, pose, overlap=overlap, anchor=anchor)
            if record["sampler_calls"] != 1 or self._accepted[index].cloud is not returned:
                raise RuntimeError("M14 retained-cloud evidence does not match actual admission")
            record["status"] = "completed"
            return decisions
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._inside_retention = False
