"""Closed offline solver mechanisms; never imported by production tracking."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import patch

import numpy as np

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis

PROFILES = ("observe-origin", "centered")
MAX_RECORDED_STEPS = 450_000
_ACTIVE: ContextVar[bool] = ContextVar("offline_solver_mechanism_active", default=False)


def centered_step(
    matched: Sequence[tuple[Point, Point, Point]], current: Rigid
) -> tuple[Rigid, dict[str, Any]]:
    """Same linear objective/pivot rule; finite increment around matched centroid."""
    triples = np.asarray(matched, dtype=np.float64)
    if triples.ndim != 3 or triples.shape[1:] != (3, 3) or len(triples) < 6:
        raise ValueError("At least six point/target/normal triples are required")
    if not np.isfinite(triples).all():
        raise ValueError("Finite point/target/normal triples are required")
    source, target, normals = triples[:, 0], triples[:, 1], triples[:, 2]
    if np.any(np.linalg.norm(normals, axis=1) <= 0):
        raise ValueError("Every matched normal must be nonzero")
    moved = source @ np.asarray(current.rotation).T + np.asarray(current.translation)
    centre = moved.mean(axis=0)
    local = moved - centre
    radius = float(np.sqrt(np.mean(np.sum(local * local, axis=1))))
    if radius <= 0:
        raise icp_plane.IcpPlaneGeometryUnusable("Zero matched-source extent")
    jacobian = np.concatenate((np.cross(local, normals), normals), axis=1)
    residuals = np.sum((moved - target) * normals, axis=1)
    matrix = np.add.accumulate(jacobian[:, :, None] * jacobian[:, None, :], axis=0)[-1]
    rhs = np.add.accumulate(-jacobian * residuals[:, None], axis=0)[-1]
    solution = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    rotation_vector = (solution[0], solution[1], solution[2])
    angle = math.sqrt(sum(value * value for value in rotation_vector))
    rotation = (
        IDENTITY_MATRIX
        if angle < MIN_VECTOR_NORM
        else rotation_about_axis(rotation_vector, angle).rotation
    )
    translation = np.asarray(solution[3:]) + centre - np.asarray(rotation) @ centre
    displacement = (float(translation[0]), float(translation[1]), float(translation[2]))
    result = Rigid.of(rotation, displacement).compose(current)
    # Unit-balanced local information is descriptive, never a new acceptance gate.
    normalized = np.concatenate((np.cross(local / radius, normals), normals), axis=1)
    eigenvalues = np.linalg.eigvalsh(normalized.T @ normalized)
    return result, dict(
        matched=len(matched),
        centre=centre.tolist(),
        rms_radius=radius,
        rotation_step_rad=angle,
        centroid_translation_step=solution[3:],
        information_eigenvalues=eigenvalues.tolist(),
        information_condition=(
            float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None
        ),
        before_rmse=float(np.sqrt(np.mean(residuals * residuals))),
    )


class SolverExperiment:
    """Process-local opt-in, bounded trace, transparent observation mode."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown closed solver mechanism {profile!r}")
        self.profile = profile
        self.records: list[dict[str, Any]] = []
        self.source_index: int | None = None
        self._original: Callable[..., Rigid] | None = None

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("Solver experiment requires source indices in 0..2999")
        self.source_index = index

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if self._original is None or not _ACTIVE.get():
            raise RuntimeError("Solver experiment must run inside its active context")
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("Closed solver experiment exceeded its step-record bound")
        row: dict[str, Any] = dict(source_index=self.source_index, profile=self.profile)
        self.records.append(row)
        original = centered = None
        try:
            # Observation must not swallow or replace an original solver refusal.
            original = self._original(matched, current, accelerated=accelerated)
        except (icp_plane.IcpPlaneNotAllowed, ValueError, ArithmeticError) as error:
            row["original_error"] = f"{type(error).__name__}: {error}"
            if self.profile == "observe-origin":
                raise
        try:
            centered, data = centered_step(matched, current)
            row.update(data)
        except (icp_plane.IcpPlaneNotAllowed, ValueError, ArithmeticError) as error:
            row["centered_error"] = f"{type(error).__name__}: {error}"
            if self.profile == "centered":
                raise
        source = np.asarray([triple[0] for triple in matched])
        target = np.asarray([triple[1] for triple in matched])
        normals = np.asarray([triple[2] for triple in matched])
        moved_by = {}
        for label, value in (("original", original), ("centered", centered)):
            if value is None:
                continue
            moved = source @ np.asarray(value.rotation).T + np.asarray(value.translation)
            moved_by[label] = moved
            residuals = np.sum((moved - target) * normals, axis=1)
            row[label + "_after_rmse"] = float(np.sqrt(np.mean(residuals * residuals)))
        if len(moved_by) == 2:
            distances = np.linalg.norm(moved_by["original"] - moved_by["centered"], axis=1)
            row["paired_step_difference_median"] = float(np.median(distances))
            row["paired_step_difference_max"] = float(np.max(distances))
        result = original if self.profile == "observe-origin" else centered
        assert result is not None
        row["status"] = "completed"
        return result

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._original is not None:
            raise RuntimeError("Solver mechanisms cannot be nested")
        token = _ACTIVE.set(True)
        self._original = icp_plane._solve_point_to_plane
        try:
            with patch.object(icp_plane, "_solve_point_to_plane", self.step):
                yield
        finally:
            self._original = None
            _ACTIVE.reset(token)
