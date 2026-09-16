"""M20 closed residual-based robust weighting, offline diagnostic only."""

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
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis
from scanner3d.tracking import TrackingPolicy

PROFILES = ("observe-l2", "huber-scale", "cauchy-scale")
CAPACITIES = ("ordinary-900", "dense-1800")
MODES = ("observe-origin", "centered")
NORMAL_ABSOLUTE_MEDIAN = 0.6744897501960817
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
        raise ValueError("M20 requires its closed kernel/capacity/grid/update roster")
    source = "observe-900" if capacity == "ordinary-900" else "source-1800"
    target, policy = capacity_settings(source, factors, mode, None)
    return target, policy, source


def robust_weights(residuals: Any, profile: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(residuals, dtype=float)
    if (
        profile not in PROFILES
        or values.ndim != 1
        or not 6 <= len(values) <= 2048
        or not np.isfinite(values).all()
    ):
        raise ValueError("M20 requires a declared kernel and 6..2048 finite residuals")
    absolute = np.abs(values)
    median = float(np.median(absolute))
    scale = max(DEPTH_SCALE_METRES, median / NORMAL_ABSOLUTE_MEDIAN)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("M20 requires finite positive residual scale")
    ratio = absolute / scale
    if profile == "observe-l2":
        raw = np.ones(len(values))
    elif profile == "huber-scale":
        raw = 1.0 / np.maximum(1.0, ratio)
    else:
        raw = 1.0 / np.square(np.hypot(1.0, ratio))
    total = float(raw.sum())
    if not np.isfinite(raw).all() or np.any(raw <= 0) or not math.isfinite(total):
        raise ValueError("M20 refuses nonfinite or nonpositive weights")
    weights = raw * (len(raw) / total)
    return weights, dict(
        residual_scale=scale,
        median_absolute_residual=median,
        scale_floor=DEPTH_SCALE_METRES,
        scale_floor_active=scale == DEPTH_SCALE_METRES,
        raw_weight_min=float(raw.min()),
        raw_weight_max=float(raw.max()),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
        weight_sum=float(weights.sum()),
        zero_weight=int(np.sum(weights == 0)),
        downweighted=int(np.sum(raw < 1)),
        effective_weighted_count=float(weights.sum() ** 2 / (weights @ weights)),
    )


def robust_cost(residuals: Any, scale: float, profile: str) -> float:
    values = np.asarray(residuals, dtype=float)
    if (
        profile not in PROFILES
        or values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("M20 requires finite residuals and positive frozen scale")
    ratio = np.abs(values) / scale
    if profile == "observe-l2":
        costs = 0.5 * values * values
    elif profile == "huber-scale":
        costs = scale * scale * np.where(ratio <= 1, 0.5 * ratio * ratio, ratio - 0.5)
    else:
        costs = scale * scale * np.log(np.hypot(1.0, ratio))
    result = float(np.mean(costs))
    if not math.isfinite(result):
        raise ValueError("M20 requires finite frozen-scale objective")
    return result


def robust_step(
    matched: Sequence[tuple[Point, Point, Point]],
    current: Rigid,
    profile: str,
    mode: str,
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
        raise ValueError("M20 requires its update and 6..2048 finite normal triples")
    source, target, normals = values[:, 0], values[:, 1], values[:, 2]
    moved = source @ np.asarray(current.rotation).T + np.asarray(current.translation)
    centre = moved.mean(axis=0) if mode == "centered" else np.zeros(3)
    jacobian = np.concatenate((np.cross(moved - centre, normals), normals), axis=1)
    residuals = np.sum((moved - target) * normals, axis=1)
    weights, data = robust_weights(residuals, profile)
    products = jacobian[:, :, None] * jacobian[:, None, :]
    unweighted_matrix = np.add.accumulate(products, axis=0)[-1]
    # Same singularity rule on the unchanged constraints before weighting.
    icp_plane._solve_6x6(unweighted_matrix.tolist(), [0.0] * 6)
    matrix = np.add.accumulate(products * weights[:, None, None], axis=0)[-1]
    rhs = np.add.accumulate(-jacobian * (weights * residuals)[:, None], axis=0)[-1]
    delta = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    vector = (delta[0], delta[1], delta[2])
    angle = math.sqrt(sum(value * value for value in vector))
    rotation = (
        IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else rotation_about_axis(vector, angle).rotation
    )
    translation = np.asarray(delta[3:]) + centre - np.asarray(rotation) @ centre
    result = Rigid.of(
        rotation, (float(translation[0]), float(translation[1]), float(translation[2]))
    ).compose(current)
    after = source @ np.asarray(result.rotation).T + np.asarray(result.translation)
    after_residuals = np.sum((after - target) * normals, axis=1)
    return result, dict(
        **data,
        matched=len(values),
        centre=centre.tolist(),
        delta=delta,
        rotation_step_rad=angle,
        before_unweighted_target_plane_rmse=float(np.sqrt(np.mean(residuals * residuals))),
        after_unweighted_target_plane_rmse=float(
            np.sqrt(np.mean(after_residuals * after_residuals))
        ),
        robust_cost_before=robust_cost(residuals, data["residual_scale"], profile),
        robust_cost_after=robust_cost(after_residuals, data["residual_scale"], profile),
        frozen_scale_for_both_costs=True,
        final_transform=result.as_dict(),
    )


class RobustResidualExperiment:
    """Only the finite step changes; original iteration/gates report raw residuals."""

    def __init__(self, profile: str, mode: str) -> None:
        if profile not in PROFILES[1:] or mode not in MODES:
            raise ValueError("M20 changed solver requires its declared kernel and update")
        self.kernel_profile = profile
        self.profile = mode
        self.source_index: int | None = None
        self.records: list[dict[str, Any]] = []
        self._active = False

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("M20 source index must be in 0..2999")
        self.source_index = index

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if not self._active or not _ACTIVE.get():
            raise RuntimeError("M20 step requires its active solver context")
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("M20 exceeds the original step-record bound")
        row: dict[str, Any] = dict(
            source_index=self.source_index,
            profile=self.profile,
            kernel_profile=self.kernel_profile,
            kind="step",
            status="running",
        )
        self.records.append(row)
        try:
            result, data = robust_step(matched, current, self.kernel_profile, self.profile)
            row.update(data, status="completed")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._active or icp_plane._solve_point_to_plane is not ORIGINAL_STEP:
            raise RuntimeError("M20 refuses nested contexts or a nonoriginal step hook")
        token = _ACTIVE.set(True)
        self._active = True
        try:
            with patch.object(icp_plane, "_solve_point_to_plane", self.step):
                yield
        finally:
            self._active = False
            _ACTIVE.reset(token)
