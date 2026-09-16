"""M16 rotation-aware covariance gradient and bounded descent; offline only.

Own derivative of Generalized-ICP equation 2, not its full inner optimizer.
https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scanner_generalized_icp import COVARIANCE_EPSILON, GeneralizedExperiment, paired_precision
from scanner_solver_mechanisms import centered_step

from scanner3d import icp_plane
from scanner3d.geometry import IDENTITY_MATRIX, MIN_VECTOR_NORM, Point, Rigid, rotation_about_axis

ARMIJO = 1e-4
MAX_LINE_SEARCH = 16
PROFILES = ("generalized-covariance", "rotation-aware-covariance")
BASES = ("stride-pooled-residual", "rank-pooled-settling",
         "grid-pooled-residual", "rank-native-residual")
Match = tuple[Point, Point, Point, Point]


def arrays(matched: Sequence[Match]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matched, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (4, 3) or len(values) < 6:
        raise ValueError("At least six point/point/normal/normal quadruples required")
    if not np.isfinite(values).all():
        raise ValueError("Finite covariance correspondences required")
    source, target, sn, tn = (values[:, i] for i in range(4))
    snorm, tnorm = np.linalg.norm(sn, axis=1), np.linalg.norm(tn, axis=1)
    if not np.isfinite(snorm).all() or not np.isfinite(tnorm).all():
        raise ValueError("Finite normal magnitudes required")
    if np.any(snorm <= 0) or np.any(tnorm <= 0):
        raise ValueError("Nonzero source and target normals required")
    return source, target, sn / snorm[:, None], tn / tnorm[:, None]


def objective(matched: Sequence[Match], current: Rigid) -> float:
    source, target, sn, tn = arrays(matched)
    rotation = np.asarray(current.rotation)
    residual = source @ rotation.T + np.asarray(current.translation) - target
    precision = paired_precision(sn @ rotation.T, tn)
    return float(np.mean(np.einsum("ni,nij,nj->n", residual, precision, residual)))


def system(matched: Sequence[Match], current: Rigid) -> dict[str, Any]:
    """Left centered half-gradient, including covariance rotation derivative."""
    source, target, sn, tn = arrays(matched)
    rotation = np.asarray(current.rotation)
    moved = source @ rotation.T + np.asarray(current.translation)
    normals = sn @ rotation.T
    precision = paired_precision(normals, tn)
    centre = moved.mean(axis=0)
    local = moved - centre
    residual = moved - target
    weighted = (precision @ residual[:, :, None])[:, :, 0]
    covariance_term = (COVARIANCE_EPSILON - 1) * np.sum(
        normals * weighted, axis=1)[:, None] * np.cross(normals, weighted)
    rows = np.concatenate((np.cross(local, weighted) - covariance_term, weighted), axis=1)
    gradient = np.add.accumulate(rows, axis=0)[-1] / len(source)
    jacobian = np.zeros((len(source), 3, 6))
    jacobian[:, 0, 1], jacobian[:, 0, 2] = local[:, 2], -local[:, 1]
    jacobian[:, 1, 0], jacobian[:, 1, 2] = -local[:, 2], local[:, 0]
    jacobian[:, 2, 0], jacobian[:, 2, 1] = local[:, 1], -local[:, 0]
    jacobian[:, :, 3:] = np.eye(3)
    products = np.swapaxes(jacobian, 1, 2) @ precision @ jacobian
    matrix = np.add.accumulate(products, axis=0)[-1] / len(source)
    cost = float(np.mean(np.einsum("ni,nij,nj->n", residual, precision, residual)))
    if not math.isfinite(cost) or cost < 0 or not np.isfinite(gradient).all():
        raise ValueError("Finite nonnegative covariance objective and gradient required")
    if not np.isfinite(matrix).all():
        raise ValueError("Finite covariance preconditioner required")
    return dict(half_gradient=gradient, matrix=matrix, centre=centre, cost=cost)


def increment(current: Rigid, centre: np.ndarray, direction: np.ndarray, alpha: float) -> Rigid:
    vector = tuple(float(v) for v in direction[:3] * alpha)
    angle = math.hypot(*vector)
    rotation = (IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else
                rotation_about_axis((vector[0], vector[1], vector[2]), angle).rotation)
    translation = direction[3:] * alpha + centre - np.asarray(rotation) @ centre
    point = (float(translation[0]), float(translation[1]), float(translation[2]))
    return Rigid.of(rotation, point).compose(current)


def descent_step(
    matched: Sequence[Match], current: Rigid, *, trace: dict[str, Any] | None = None
) -> tuple[Rigid, dict[str, Any]]:
    details: dict[str, Any] = {} if trace is None else trace
    details.update(matched=len(matched), covariance_epsilon=COVARIANCE_EPSILON,
                   armijo_coefficient=ARMIJO, line_search_limit=MAX_LINE_SEARCH, trials=[])
    terms = system(matched, current)
    # Retain M5's plane-information refusal; discard this diagnostic finite step.
    centered_step([(p, q, n) for p, q, _, n in matched], current)
    gradient, before = terms["half_gradient"], terms["cost"]
    details.update(original_plane_information_guard_passed=True,
        covariance_cost_before=before, half_gradient=gradient.tolist(),
        source_centre=terms["centre"].tolist())
    if np.all(gradient == 0):
        details.update(step_status="exact_zero_gradient", covariance_cost_after=before,
                       directional_derivative=0.0, selected_alpha=0.0)
        return current, details
    direction = np.asarray(icp_plane._solve_6x6(
        terms["matrix"].tolist(), (-gradient).tolist()))
    derivative = float(2 * gradient @ direction)
    if not np.isfinite(direction).all() or not math.isfinite(derivative) or derivative >= 0:
        raise icp_plane.IcpPlaneNotAllowed("Covariance direction is not a finite descent direction")
    details.update(direction=direction.tolist(), directional_derivative=derivative)
    for ordinal in range(MAX_LINE_SEARCH):
        alpha = 2.0 ** -ordinal
        proposed = increment(current, terms["centre"], direction, alpha)
        after = objective(matched, proposed)
        bound = before + ARMIJO * alpha * derivative
        acceptable = math.isfinite(after) and 0 <= after <= bound
        details["trials"].append(dict(alpha=alpha, cost=after if math.isfinite(after) else None,
                                     armijo_bound=bound, accepted=acceptable))
        if acceptable:
            details.update(step_status="armijo_descent", covariance_cost_after=after,
                           selected_alpha=alpha)
            return proposed, details
    raise icp_plane.IcpPlaneNotAllowed("Covariance line search exhausted without accepted descent")


class RotationAwareExperiment(GeneralizedExperiment):
    """Same source-normal eligibility/context as M5, with truthful new profile."""

    profile = "rotation-aware-covariance"

    def step(
        self, matched: Sequence[tuple[Point, Point, Point]], current: Rigid,
        *, accelerated: bool = False,
    ) -> Rigid:
        if self._normals is None:
            raise RuntimeError("Rotation-aware covariance step outside source-normal context")
        row = self._record(kind="step", status="running")
        try:
            result, _ = descent_step(
                [(p, q, self._normals[p], n) for p, q, n in matched], current, trace=row)
            row["status"] = "completed"
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
