"""M10 ordinary-model native pixel-grid selection; closed offline capacities."""

from __future__ import annotations

from collections.abc import Sequence

from scanner_dense_model_sampling import EXPECTED_POLICY
from scanner_factorial_config import Factors
from scanner_model_pixel_sampling import indices
from scanner_model_sampling import ModelSamplingTracker

from scanner3d.geometry import Point
from scanner3d.tracking import DEFAULT_POLICY, TrackingPolicy

PROFILES = ("observe-stride", "pixel-grid-model")
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
        raise ValueError("M10 requires its closed capacity/profile/grid/solver roster")
    if capacity == "ordinary-900":
        return 900, DEFAULT_POLICY, None
    return 1800, EXPECTED_POLICY, "source-1800"


class PixelModelTracker(ModelSamplingTracker):
    """Preserve native coordinates/order and original model membership at 900."""

    profiles = PROFILES
    selected_count_key = "pixel_grid_points"

    def selection_indices(
        self, cloud: Sequence[Point], limit: int, *, position: int, frames: int
    ) -> tuple[int, ...]:
        if (
            type(position) is not int
            or type(frames) is not int
            or not 1 <= frames <= 8
            or not 0 <= position < frames
        ):
            raise ValueError("M10 requires a bounded original model-frame position")
        return indices(cloud, limit, capacity=self._policy.retained_cloud_points)


class DensePixelModelTracker(PixelModelTracker):
    """Identical native pixel-grid rule, requiring exactly M7's explicit capacity."""

    expected_policy = EXPECTED_POLICY
