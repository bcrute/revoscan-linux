"""M23 fixed plane/3-D residual mixture, offline only and with original refusal guard."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
from scanner_factorial_config import Factors
from scanner_solver_mechanisms import _ACTIVE, MAX_RECORDED_STEPS
from scanner_source_capacity import settings as capacity_settings

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-plane", "plane-plus-point-0.1")
CAPACITIES = ("ordinary-900", "dense-1800")
MODES = ("observe-origin", "centered")
POINT_WEIGHT = 0.1
ORIGINAL_STEP = icp_plane._solve_point_to_plane


def settings(
    capacity: str, profile: str, factors: Factors, mode: str
) -> tuple[int, TrackingPolicy, str]:
    if (
        capacity not in CAPACITIES
        or profile not in PROFILES
        or mode not in MODES
        or factors.values != ("grid", "pooled", "residual")
    ):
        raise ValueError("M23 requires its closed mixture/capacity/grid/update roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, mode, None)
    return target, policy, source


def point_jacobian(relative: Any) -> np.ndarray:
    """Map (omega,v) to omega cross relative + v, independently per point."""
    points = np.asarray(relative, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("M23 requires a finite Nx3 relative-point array")
    result = np.zeros((len(points), 3, 6))
    x, y, z = points.T
    result[:, 0, 1], result[:, 0, 2] = z, -y
    result[:, 1, 0], result[:, 1, 2] = -z, x
    result[:, 2, 0], result[:, 2, 1] = y, -x
    result[:, :, 3:] = np.eye(3)
    return result


def costs(displacements: np.ndarray, normals: np.ndarray) -> dict[str, float]:
    if (
        displacements.ndim != 2
        or displacements.shape[1:] != (3,)
        or normals.shape != displacements.shape
        or not len(displacements)
        or not np.isfinite(displacements).all()
        or not np.isfinite(normals).all()
    ):
        raise ValueError("M23 requires matching nonempty finite Nx3 costs")
    plane = float(np.mean(np.sum(displacements * normals, axis=1) ** 2))
    point = float(np.mean(np.sum(displacements * displacements, axis=1)))
    values = dict(plane_mse=plane, point_mse=point, mixed_mse=plane + POINT_WEIGHT * point)
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("M23 requires finite nonnegative frozen-pair costs")
    return values


def mixed_step(
    matched: Sequence[tuple[Point, Point, Point]], current: Rigid, mode: str
) -> tuple[Rigid, dict[str, Any]]:
    values = np.asarray(matched, dtype=float)
    if (
        mode not in MODES
        or values.ndim != 3
        or values.shape[1:] != (3, 3)
        or not 6 <= len(values) <= 2048
        or not np.isfinite(values).all()
        or np.any(np.linalg.norm(values[:, 2], axis=1) <= 0)
    ):
        raise ValueError("M23 requires its update and 6..2048 finite normal triples")
    source, target, normals = values[:, 0], values[:, 1], values[:, 2]
    moved = source @ np.asarray(current.rotation).T + np.asarray(current.translation)
    centre = moved.mean(axis=0) if mode == "centered" else np.zeros(3)
    plane_j = np.concatenate((np.cross(moved - centre, normals), normals), axis=1)
    point_j = point_jacobian(moved - centre)
    displacement = moved - target
    residuals = np.sum(displacement * normals, axis=1)
    plane_products = plane_j[:, :, None] * plane_j[:, None, :]
    unweighted_matrix = np.add.accumulate(plane_products, axis=0)[-1]
    icp_plane._solve_6x6(unweighted_matrix.tolist(), [0.0] * 6)
    point_products = np.sum(point_j[:, :, :, None] * point_j[:, :, None, :], axis=1)
    matrix = np.add.accumulate(plane_products + POINT_WEIGHT * point_products, axis=0)[-1]
    point_rhs = np.sum(point_j * displacement[:, :, None], axis=1)
    rhs = np.add.accumulate(-plane_j * residuals[:, None] - POINT_WEIGHT * point_rhs, axis=0)[-1]
    delta = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    vector = (delta[0], delta[1], delta[2])
    angle = math.sqrt(sum(v * v for v in vector))
    rotation = (
        IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else rotation_about_axis(vector, angle).rotation
    )
    translation = np.asarray(delta[3:]) + centre - np.asarray(rotation) @ centre
    result = Rigid.of(
        rotation, (float(translation[0]), float(translation[1]), float(translation[2]))
    ).compose(current)
    after = source @ np.asarray(result.rotation).T + np.asarray(result.translation)
    return result, dict(
        matched=len(values),
        point_weight=POINT_WEIGHT,
        centre=centre.tolist(),
        delta=delta,
        original_plane_information_guard_passed=True,
        rotation_step_rad=angle,
        costs_before=costs(displacement, normals),
        costs_after=costs(after - target, normals),
        frozen_pairs_for_both_costs=True,
        final_transform=result.as_dict(),
    )


class MixedMetricExperiment:
    """Only the finite step changes; all external metrics and tracker guards remain."""

    def __init__(self, profile: str, mode: str) -> None:
        if profile != "plane-plus-point-0.1" or mode not in MODES:
            raise ValueError("M23 changed solver requires its declared mixture and update")
        self.metric_profile, self.profile = profile, mode
        self.source_index: int | None = None
        self.records: list[dict[str, Any]] = []
        self._active = False

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("M23 source index must be in 0..2999")
        self.source_index = index

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if not self._active or not _ACTIVE.get():
            raise RuntimeError("M23 step requires its active solver context")
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("M23 exceeds the original step-record bound")
        row: dict[str, Any] = dict(
            source_index=self.source_index,
            profile=self.profile,
            metric_profile=self.metric_profile,
            kind="step",
            status="running",
        )
        self.records.append(row)
        try:
            result, data = mixed_step(matched, current, self.profile)
            row.update(data, status="completed")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._active or icp_plane._solve_point_to_plane is not ORIGINAL_STEP:
            raise RuntimeError("M23 refuses nested contexts or a nonoriginal step hook")
        token = _ACTIVE.set(True)
        self._active = True
        try:
            with patch.object(icp_plane, "_solve_point_to_plane", self.step):
                yield
        finally:
            self._active = False
            _ACTIVE.reset(token)
