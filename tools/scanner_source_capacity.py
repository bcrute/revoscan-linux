"""M7 closed incoming/retained-cloud capacity, never a production setting."""

from __future__ import annotations

from dataclasses import replace

from scanner_factorial_config import Factors

from scanner3d.icp_plane import NORMAL_CACHE_MAX_POINTS
from scanner3d.tracking import DEFAULT_POLICY, TrackingPolicy

PROFILES = ("observe-900", "source-1800")


def settings(
    profile: str,
    factors: Factors,
    solver_profile: str | None,
    model_profile: str | None,
) -> tuple[int, TrackingPolicy]:
    if (
        profile not in PROFILES
        or factors.sampling not in ("stride", "grid")
        or factors.model_normals != "pooled"
        or factors.convergence != "residual"
        or solver_profile not in ("observe-origin", "centered")
        or model_profile is not None
    ):
        raise ValueError("M7 requires its closed source/factor/solver roster and ordinary model")
    if profile == "observe-900":
        return 900, DEFAULT_POLICY
    if NORMAL_CACHE_MAX_POINTS < 1800 or DEFAULT_POLICY.model_reference_points != 1800:
        raise ValueError("M7 requires existing 1800-point target capacity and real GPU eligibility")
    policy = replace(DEFAULT_POLICY, candidate_source_points=1800, retained_cloud_points=1800)
    return 1800, policy
