"""M12 separate preparation/source/retained capacities, offline-only closed roster."""

from __future__ import annotations

from dataclasses import replace

import scanner_source_capacity
from scanner_factorial_config import Factors

from scanner3d.icp_plane import NORMAL_CACHE_MAX_POINTS
from scanner3d.tracking import DEFAULT_POLICY, TrackingPolicy

PROFILES = (
    "observe-900", "source-1800", "prepared-1800-source-900-retained-900",
    "prepared-1800-source-1800-retained-900", "prepared-1800-source-900-retained-1800",
)
ORIGINAL_SETTINGS = scanner_source_capacity.settings


def settings(
    profile: str, factors: Factors, solver_profile: str | None, model_profile: str | None
) -> tuple[int, TrackingPolicy]:
    if (
        profile not in PROFILES or factors.values != ("grid", "pooled", "residual")
        or solver_profile not in ("observe-origin", "centered") or model_profile is not None
    ):
        raise ValueError("M12 requires its closed preparation/source/retention/solver roster")
    if profile in ("observe-900", "source-1800"):
        return ORIGINAL_SETTINGS(profile, factors, solver_profile, model_profile)
    if NORMAL_CACHE_MAX_POINTS < 1800 or DEFAULT_POLICY.model_reference_points != 1800:
        raise ValueError("M12 requires original target budget and GPU eligibility")
    source, retained = {
        "prepared-1800-source-900-retained-900": (900, 900),
        "prepared-1800-source-1800-retained-900": (1800, 900),
        "prepared-1800-source-900-retained-1800": (900, 1800),
    }[profile]
    return 1800, replace(DEFAULT_POLICY, candidate_source_points=source,
                         retained_cloud_points=retained)
