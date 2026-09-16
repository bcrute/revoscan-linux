"""Offline M2: independently implemented symmetric ICP, Rusinkiewicz 2019 eqs. 10-11.

https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2019_ASO/symm_icp.pdf
The existing target-plane residual, correspondence distance/trim and tracker
gates stay unchanged. Only correspondences with two usable normals may fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
from scanner_solver_mechanisms import _ACTIVE, MAX_RECORDED_STEPS

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis


def symmetric_step(
    matched: Sequence[tuple[Point, Point, Point, Point]], current: Rigid
) -> tuple[Rigid, dict[str, Any]]:
    """Matched tuples: native source, target, native source normal, target normal."""
    values = np.asarray(matched, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 3) or len(values) < 6:
        raise ValueError("At least six point/point/normal/normal quadruples required")
    if not np.isfinite(values).all():
        raise ValueError("Finite symmetric correspondences required")
    source, target = values[:, 0], values[:, 1]
    source_normals, target_normals = values[:, 2], values[:, 3].copy()
    if any(np.any(np.linalg.norm(n, axis=1) <= 0) for n in (source_normals, target_normals)):
        raise ValueError("Symmetric correspondences need two nonzero normals")
    rotation = np.asarray(current.rotation)
    moved = source @ rotation.T + np.asarray(current.translation)
    moved_normals = source_normals @ rotation.T
    # Canonical target sign makes the exact orthogonal case deterministic too.
    rows = np.arange(len(values))
    dominant = np.argmax(np.abs(target_normals), axis=1)
    target_normals[target_normals[rows, dominant] < 0] *= -1
    dots = np.sum(moved_normals * target_normals, axis=1)
    moved_normals[dots < 0] *= -1
    orthogonal = dots == 0
    source_dominant = np.argmax(np.abs(moved_normals), axis=1)
    moved_normals[orthogonal & (moved_normals[rows, source_dominant] < 0)] *= -1
    normals = moved_normals + target_normals
    p_centre, q_centre = moved.mean(axis=0), target.mean(axis=0)
    p, q = moved - p_centre, target - q_centre
    jacobian = np.concatenate((np.cross(p + q, normals), normals), axis=1)
    residuals = np.sum((p - q) * normals, axis=1)
    matrix = np.add.accumulate(jacobian[:, :, None] * jacobian[:, None, :], axis=0)[-1]
    rhs = np.add.accumulate(-jacobian * residuals[:, None], axis=0)[-1]
    solution = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    vector = (solution[0], solution[1], solution[2])
    magnitude = math.sqrt(sum(value * value for value in vector))
    half_angle = math.atan(magnitude)
    half_rotation = (
        IDENTITY_MATRIX
        if magnitude < MIN_VECTOR_NORM
        else rotation_about_axis(vector, half_angle).rotation
    )
    half = np.asarray(half_rotation)
    complete_rotation = half @ half
    displacement = np.asarray(solution[3:]) * math.cos(half_angle)
    translation = q_centre + half @ displacement - complete_rotation @ p_centre
    matrix_tuple = (
        (
            float(complete_rotation[0, 0]),
            float(complete_rotation[0, 1]),
            float(complete_rotation[0, 2]),
        ),
        (
            float(complete_rotation[1, 0]),
            float(complete_rotation[1, 1]),
            float(complete_rotation[1, 2]),
        ),
        (
            float(complete_rotation[2, 0]),
            float(complete_rotation[2, 1]),
            float(complete_rotation[2, 2]),
        ),
    )
    result = Rigid.of(
        (matrix_tuple[0], matrix_tuple[1], matrix_tuple[2]),
        (float(translation[0]), float(translation[1]), float(translation[2])),
    ).compose(current)
    return result, dict(
        matched=len(values),
        half_angle_rad=half_angle,
        source_centre=p_centre.tolist(),
        target_centre=q_centre.tolist(),
        paired_normal_abs_cosine_median=float(np.median(np.abs(dots))),
        symmetric_linear_residual_rmse=float(np.sqrt(np.mean(residuals * residuals))),
        limitation="Symmetric fit diagnostic; externally reported residual remains target-plane.",
    )


class SymmetricExperiment:
    """Capture the original GPU provider before any native-target override."""

    profile = "symmetric"

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.source_index: int | None = None
        self._provider: Callable[..., icp_plane.NormalField] | None = None
        self._iteration: Callable[..., icp_plane._FitPlane] | None = None
        self._normals: dict[Point, Point] | None = None

    def mark_frame(self, index: int) -> None:
        if not 0 <= index < 3000:
            raise ValueError("Source indices must be in 0..2999")
        self.source_index = index

    def _record(self, **fields: Any) -> dict[str, Any]:
        if len(self.records) >= MAX_RECORDED_STEPS:
            raise RuntimeError("Closed symmetric experiment exceeded its record bound")
        row = dict(profile=self.profile, source_index=self.source_index, **fields)
        self.records.append(row)
        return row

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if self._normals is None:
            raise RuntimeError("Symmetric step outside its source-normal context")
        row = self._record(kind="step", status="running")
        try:
            full = [(p, q, self._normals[p], n) for p, q, n in matched]
            result, details = symmetric_step(full, current)
            row.update(details, status="completed")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def iterate(
        self, source: Sequence[Point], correspond: icp_plane._Correspond, **kwargs: Any
    ) -> icp_plane._FitPlane:
        if self._provider is None or self._iteration is None or self._normals is not None:
            raise RuntimeError("Symmetric iteration requires a unique active provider context")
        if not 6 <= len(source) <= icp_plane.NORMAL_CACHE_MAX_POINTS:
            raise ValueError("Symmetric source-normal budget is 6..2048 points")
        row = self._record(kind="fit", status="running", source_points=len(source))
        try:
            field = self._provider(tuple(source), 10)
            if (
                tuple(e.point for e in field.estimates) != tuple(source)
                or field.requested_neighbours != 10
            ):
                raise ValueError(
                    "Source normal provider returned a different cloud or neighbourhood"
                )
            self._normals = {e.point: e.normal for e in field.usable if e.normal is not None}
            row["source_normals_missing"] = len(field.estimates) - len(field.usable)

            def symmetric_correspond(
                src: Sequence[Point], transform: Rigid
            ) -> tuple[list[tuple[float, Point, Point, Point]], int]:
                pairs, rejected = correspond(src, transform)
                eligible = [pair for pair in pairs if pair[1] in (self._normals or {})]
                return eligible, rejected + len(pairs) - len(eligible)

            result = self._iteration(source, symmetric_correspond, **kwargs)
            row.update(
                status="completed",
                iterations=result.iterations,
                externally_reported_target_plane_rmse=result.rmse,
            )
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._normals = None

    @contextmanager
    def active(self) -> Iterator[None]:
        if _ACTIVE.get() or self._iteration is not None:
            raise RuntimeError("Solver mechanisms cannot be nested")
        provider = icp_plane._NORMAL_ACCELERATOR.get()
        if provider is None:
            raise RuntimeError("Symmetric experiment requires the active normal provider")
        token = _ACTIVE.set(True)
        self._provider = provider
        self._iteration = icp_plane._iterate
        try:
            with (
                patch.object(icp_plane, "_iterate", self.iterate),
                patch.object(icp_plane, "_solve_point_to_plane", self.step),
            ):
                yield
        finally:
            self._provider = self._iteration = self._normals = None
            _ACTIVE.reset(token)
