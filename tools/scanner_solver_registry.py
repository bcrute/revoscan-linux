"""Closed offline solver dispatch, distinct from production registration."""

from __future__ import annotations

from scanner_generalized_icp import GeneralizedExperiment
from scanner_reciprocal_icp import PROFILES as RECIPROCAL_PROFILES
from scanner_reciprocal_icp import ReciprocalExperiment
from scanner_solver_mechanisms import SolverExperiment
from scanner_symmetric_icp import SymmetricExperiment
from scanner_weighted_icp import PROFILES as WEIGHTED_PROFILES
from scanner_weighted_icp import WeightedExperiment

PROFILES = (
    "observe-origin",
    "centered",
    "symmetric",
    *WEIGHTED_PROFILES,
    "generalized-covariance",
    *RECIPROCAL_PROFILES,
)


def create_experiment(
    profile: str,
) -> SolverExperiment | SymmetricExperiment | WeightedExperiment | ReciprocalExperiment:
    if profile not in PROFILES:
        raise ValueError(f"Unknown closed solver mechanism {profile!r}")
    if profile in WEIGHTED_PROFILES:
        return WeightedExperiment(profile)
    if profile in RECIPROCAL_PROFILES:
        return ReciprocalExperiment(profile)
    if profile == "generalized-covariance":
        return GeneralizedExperiment()
    return SymmetricExperiment() if profile == "symmetric" else SolverExperiment(profile)
