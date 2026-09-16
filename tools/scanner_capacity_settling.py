"""M15 closed density/pose-settling combination, offline diagnostic only."""

from __future__ import annotations

from dataclasses import replace

import scanner_source_capacity
from scanner_factorial_config import Factors

from scanner3d.icp_plane import NORMAL_CACHE_MAX_POINTS
from scanner3d.tracking import DEFAULT_POLICY, TrackingPolicy

PROFILES = ("observe-900", "source-1800")
ORIGINAL_SETTINGS = scanner_source_capacity.settings


def settings(
    profile: str, factors: Factors, solver_profile: str | None, model_profile: str | None
) -> tuple[int, TrackingPolicy]:
    if (
        profile not in PROFILES or factors.sampling != "grid"
        or factors.model_normals != "pooled"
        or factors.convergence not in ("residual", "settling")
        or solver_profile not in ("observe-origin", "centered") or model_profile is not None
    ):
        raise ValueError("M15 requires its closed density/grid/pooled/stopping/solver roster")
    if profile == "observe-900":
        return 900, DEFAULT_POLICY
    if NORMAL_CACHE_MAX_POINTS < 1800 or DEFAULT_POLICY.model_reference_points != 1800:
        raise ValueError("M15 requires original target budget and real GPU eligibility")
    return 1800, replace(DEFAULT_POLICY, candidate_source_points=1800,
                         retained_cloud_points=1800)
