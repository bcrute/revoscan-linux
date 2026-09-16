"""M5: GICP-inspired covariance step, not a reproduction of the full paper solver.

Metric: Segal et al. 2009, Generalized-ICP, equation 2. The author's reference
gicp.cpp uses covariance epsilon .0004. Own implementation: unchanged ten-neighbour
GPU normals, one frozen-covariance centered Gauss-Newton step per old iteration,
and unchanged external target-plane residual/stopping. No nonlinear inner solve.
https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf
https://github.com/avsegal/gicp/blob/master/gicp.cpp
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scanner_solver_mechanisms import centered_step
from scanner_symmetric_icp import SymmetricExperiment

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis

COVARIANCE_EPSILON = 0.0004


def paired_precision(source_normals: Any, target_normals: Any) -> np.ndarray:
    source, target = (np.asarray(n, dtype=float) for n in (source_normals, target_normals))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Paired three-component normal arrays required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Finite paired normals required")
    sn, tn = (np.linalg.norm(n, axis=1) for n in (source, target))
    if not np.isfinite(sn).all() or not np.isfinite(tn).all() or np.any(sn <= 0) or np.any(tn <= 0):
        raise ValueError("Nonzero paired normals required")
    source, target = source / sn[:, None], target / tn[:, None]
    covariance = 2 * np.eye(3)[None, :, :] + (COVARIANCE_EPSILON - 1) * (
        source[:, :, None] * source[:, None, :] + target[:, :, None] * target[:, None, :]
    )
    return np.linalg.inv(covariance)


def covariance_step(
    matched: Sequence[tuple[Point, Point, Point, Point]], current: Rigid
) -> tuple[Rigid, dict[str, Any]]:
    """One frozen-metric step, retaining the original plane-information refusal."""
    values = np.asarray(matched, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4, 3) or len(values) < 6:
        raise ValueError("At least six point/point/normal/normal quadruples required")
    if not np.isfinite(values).all():
        raise ValueError("Finite generalized correspondences required")
    source, target, sn, tn = (values[:, i] for i in range(4))
    # Do not manufacture observable motion from the new tangential covariance.
    # Its finite update is discarded; this retains M1's original 6x6 plane gate.
    centered_step([(p, q, n) for p, q, _, n in matched], current)
    rotation = np.asarray(current.rotation)
    moved = source @ rotation.T + np.asarray(current.translation)
    moved_normals = sn @ rotation.T
    precision = paired_precision(moved_normals, tn)
    centre = moved.mean(axis=0)
    local = moved - centre
    jacobian = np.zeros((len(values), 3, 6))
    jacobian[:, 0, 1], jacobian[:, 0, 2] = local[:, 2], -local[:, 1]
    jacobian[:, 1, 0], jacobian[:, 1, 2] = -local[:, 2], local[:, 0]
    jacobian[:, 2, 0], jacobian[:, 2, 1] = local[:, 1], -local[:, 0]
    jacobian[:, :, 3:] = np.eye(3)
    residual = moved - target
    jt_w = np.swapaxes(jacobian, 1, 2) @ precision
    matrix = np.add.accumulate(jt_w @ jacobian, axis=0)[-1]
    rhs = np.add.accumulate(-(jt_w @ residual[:, :, None])[:, :, 0], axis=0)[-1]
    solution = icp_plane._solve_6x6(matrix.tolist(), rhs.tolist())
    vector = (solution[0], solution[1], solution[2])
    angle = math.hypot(*vector)
    increment_rotation = (
        IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else rotation_about_axis(vector, angle).rotation
    )
    ir = np.asarray(increment_rotation)
    translation = np.asarray(solution[3:]) + centre - ir @ centre
    result = Rigid.of(
        increment_rotation, (float(translation[0]), float(translation[1]), float(translation[2]))
    ).compose(current)
    after = source @ np.asarray(result.rotation).T + np.asarray(result.translation) - target
    after_precision = paired_precision(sn @ np.asarray(result.rotation).T, tn)

    def cost(delta: np.ndarray, metric: np.ndarray) -> float:
        return float(np.mean(np.einsum("ni,nij,nj->n", delta, metric, delta)))

    return result, dict(
        matched=len(values),
        covariance_epsilon=COVARIANCE_EPSILON,
        original_plane_information_guard_passed=True,
        rotation_step_rad=angle,
        covariance_cost_before=cost(residual, precision),
        covariance_cost_after_frozen_metric=cost(after, precision),
        covariance_cost_after_rotated_metric=cost(after, after_precision),
        limitation="One frozen-metric step; external residual and stopping remain target-plane.",
    )


class GeneralizedExperiment(SymmetricExperiment):
    """Reuse the proven M2 source-normal context; no symmetric step is applied."""

    profile = "generalized-covariance"

    def step(
        self,
        matched: Sequence[tuple[Point, Point, Point]],
        current: Rigid,
        *,
        accelerated: bool = False,
    ) -> Rigid:
        if self._normals is None:
            raise RuntimeError("Generalized step outside its source-normal context")
        row = self._record(kind="step", status="running")
        try:
            result, details = covariance_step(
                [(p, q, self._normals[p], n) for p, q, n in matched], current
            )
            row.update(details, status="completed")
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
