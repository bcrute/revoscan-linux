"""M9 closed dense-capacity/model-stratification combination; never production."""

from __future__ import annotations

from dataclasses import replace

from scanner_factorial_config import Factors
from scanner_model_sampling import PROFILES as PROFILES
from scanner_model_sampling import ModelSamplingTracker
from scanner_source_capacity import settings

from scanner3d.tracking import DEFAULT_POLICY

EXPECTED_POLICY = replace(DEFAULT_POLICY, candidate_source_points=1800, retained_cloud_points=1800)


def validate(
    profile: str,
    factors: Factors,
    solver_profile: str | None,
    model_profile: str | None,
    source_profile: str | None,
) -> None:
    if profile not in PROFILES or source_profile != "source-1800" or model_profile is not None:
        raise ValueError("M9 requires a closed model-sampling profile and source-1800 only")
    points, policy = settings(source_profile, factors, solver_profile, model_profile)
    if points != 1800 or policy != EXPECTED_POLICY:
        raise ValueError("M9 capacity differs from the exact M7 policy")


def indices(size: int, limit: int, *, position: int, frames: int) -> tuple[int, ...]:
    """Exactly M8's integer strata, admitting the declared M7 retained cloud size."""
    if (
        not all(type(n) is int for n in (size, limit, position, frames))
        or not 0 <= size <= 1800
        or not 1 <= limit <= 1800
        or not 1 <= frames <= 8
        or not 0 <= position < frames
    ):
        raise ValueError("M9 requires bounded integer sizes, limits and reference positions")
    if size <= limit:
        return tuple(range(size))
    return tuple(
        ((2 * frames * j + 2 * position + 1) * size) // (2 * frames * limit) for j in range(limit)
    )


class DenseModelSamplingTracker(ModelSamplingTracker):
    """M8 model logic with only the exact, explicit M7 retained/source capacity."""

    expected_policy = EXPECTED_POLICY

    @staticmethod
    def sample_indices(size: int, limit: int, *, position: int, frames: int) -> tuple[int, ...]:
        return indices(size, limit, position=position, frames=frames)
