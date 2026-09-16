"""Offline graph experiment using measured-cloud surface displacement in one unit."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from scanner3d.geometry import Point, Rigid
from scanner3d.posegraph import PoseEdge
from scanner3d.posegraph_lm import se3_exp


def moment_landmarks(points: Sequence[Point]) -> np.ndarray:
    """Four points with exactly the same centroid and population covariance."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 3 or not np.isfinite(cloud).all():
        raise ValueError("At least three finite 3-D source points required")
    centre = cloud.mean(axis=0)
    centred = cloud - centre
    covariance = centred.T @ centred / len(cloud)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0:
        raise ValueError("Coincident source points supply no surface extent")
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0)))
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return cast(np.ndarray, centre + tetrahedron @ root.T)


def apply(pose: Rigid, points: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, points @ np.asarray(pose.rotation).T + np.asarray(pose.translation))


def edge_displacements(poses: Sequence[Rigid], edge: PoseEdge, landmarks: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        apply(poses[edge.i], landmarks) - apply(poses[edge.j], apply(edge.transform, landmarks)),
    )


def validated_landmarks(
    original: list[Rigid],
    edges: list[PoseEdge],
    clouds: Sequence[Sequence[Point]],
    *,
    max_nodes: int = 128,
) -> list[np.ndarray]:
    if (
        max_nodes not in (128, 256)
        or not 2 <= len(original) <= max_nodes
        or len(clouds) != len(original)
        or not edges
    ):
        raise ValueError("Bounded nodes with measured source clouds and edges required")
    if any(
        type(e.i) is not int
        or type(e.j) is not int
        or e.i == e.j
        or not 0 <= e.i < len(original)
        or not 0 <= e.j < len(original)
        or not math.isfinite(e.weight)
        or e.weight <= 0
        for e in edges
    ):
        raise ValueError("Finite positive-weight edges between distinct valid nodes required")
    reached = {0}
    while True:
        grown = (
            reached
            | {e.i for e in edges if e.j in reached}
            | {e.j for e in edges if e.i in reached}
        )
        if grown == reached:
            break
        reached = grown
    if len(reached) != len(original):
        raise ValueError("Every node must have an edge path to fixed anchor 0")
    return [moment_landmarks(c) for c in clouds]


def optimise(
    original: list[Rigid], edges: list[PoseEdge], clouds: Sequence[Sequence[Point]]
) -> tuple[list[Rigid], dict[str, Any]]:
    landmarks = validated_landmarks(original, edges, clouds)
    sparsity = lil_matrix((len(edges) * 12, (len(original) - 1) * 6), dtype=np.int8)
    for index, edge in enumerate(edges):
        for node in (edge.i, edge.j):
            if node:
                sparsity[index * 12 : (index + 1) * 12, (node - 1) * 6 : node * 6] = 1

    def poses(parameters: np.ndarray) -> list[Rigid]:
        return [original[0]] + [
            old.compose(se3_exp(parameters[(i - 1) * 6 : i * 6].tolist()))
            for i, old in enumerate(original[1:], 1)
        ]

    evaluations = 0

    def residual(parameters: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        current = poses(parameters)
        return np.concatenate(
            [
                (edge_displacements(current, e, landmarks[e.i]) * math.sqrt(e.weight / 4)).ravel()
                for e in edges
            ]
        )

    initial = np.zeros((len(original) - 1) * 6)
    initial_residual = residual(initial)
    solved = least_squares(
        residual,
        initial,
        jac_sparsity=sparsity.tocsr(),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100,
        loss="linear",
    )
    corrected = poses(solved.x)
    measured = [
        float(
            np.sqrt(np.mean(np.sum(edge_displacements(corrected, e, landmarks[e.i]) ** 2, axis=1)))
        )
        for e in edges
    ]
    report = dict(
        metric="prepared_source_cloud_mean_squared_surface_displacement",
        anchor=0,
        nodes=len(original),
        edges=len(edges),
        converged=bool(solved.success),
        status=int(solved.status),
        message=str(solved.message),
        nfev=solved.nfev,
        actual_residual_evaluations=evaluations,
        optimality=float(solved.optimality),
        initial_sum_squared=float(initial_residual @ initial_residual),
        final_sum_squared=float(2 * solved.cost),
        source_moment_landmarks=[p.tolist() for p in landmarks],
        per_edge_surface_rms=measured,
        worst_edge_surface_rms=max(measured),
        limitations="Offline heuristic edge weights, no robust outlier loss, no physical accuracy "
        "claim. Four landmarks reproduce each prepared cloud's quadratic displacement metric; "
        "they do not establish certainty of a measured edge.",
    )
    return corrected, report


def point_jacobian(pose: Rigid, points: np.ndarray) -> np.ndarray:
    """World-space point derivative for a local/right SE(3) pose update."""
    rotation = np.asarray(pose.rotation)
    cross = np.zeros((len(points), 3, 3))
    x, y, z = points.T
    cross[:, 0, 1], cross[:, 0, 2] = -z, y
    cross[:, 1, 0], cross[:, 1, 2] = z, -x
    cross[:, 2, 0], cross[:, 2, 1] = -y, x
    return np.concatenate(
        [np.broadcast_to(rotation, cross.shape), -np.einsum("ij,njk->nik", rotation, cross)], axis=2
    ).reshape(-1, 6)


def edge_system(
    poses: list[Rigid], edge: PoseEdge, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = math.sqrt(edge.weight / 4)
    return (
        edge_displacements(poses, edge, points).ravel() * scale,
        point_jacobian(poses[edge.i], points) * scale,
        -point_jacobian(poses[edge.j], apply(edge.transform, points)) * scale,
    )


def normal_system(
    poses: list[Rigid], edges: list[PoseEdge], landmarks: list[np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    size = 6 * (len(poses) - 1)
    hessian, gradient = np.zeros((size, size)), np.zeros(size)
    cost = 0.0
    for edge in edges:
        residual, ji, jj = edge_system(poses, edge, landmarks[edge.i])
        cost += float(residual @ residual)
        blocks = [(node, j) for node, j in ((edge.i, ji), (edge.j, jj)) if node]
        for node, jacobian in blocks:
            span = slice(6 * (node - 1), 6 * node)
            gradient[span] += jacobian.T @ residual
            for other, other_jacobian in blocks:
                other_span = slice(6 * (other - 1), 6 * other)
                hessian[span, other_span] += jacobian.T @ other_jacobian
    return cost, hessian, gradient


def analytic_optimise(
    original: list[Rigid],
    edges: list[PoseEdge],
    clouds: Sequence[Sequence[Point]],
    *,
    max_nodes: int = 128,
) -> tuple[list[Rigid], dict[str, Any]]:
    landmarks = validated_landmarks(original, edges, clouds, max_nodes=max_nodes)
    current = list(original)
    cost, hessian, gradient = normal_system(current, edges, landmarks)
    initial_cost = cost
    damping, growth = 1e-3, 2.0
    history: list[dict[str, Any]] = []
    converged = False
    reason = "100_trial_budget_exhausted"
    for iteration in range(100):
        diagonal = np.maximum(np.diag(hessian), 1e-12)
        scaled_gradient = float(np.max(np.abs(gradient) / np.sqrt(diagonal)))
        if (
            not math.isfinite(cost)
            or not np.isfinite(hessian).all()
            or not np.isfinite(gradient).all()
        ):
            raise ValueError("Nonfinite surface normal equations")
        if scaled_gradient <= 1e-8:
            converged, reason = True, "scaled_gradient_converged"
            break
        delta = np.linalg.solve(hessian + damping * np.diag(diagonal), -gradient)
        trial = [original[0]] + [
            p.compose(se3_exp(delta[(i - 1) * 6 : i * 6].tolist()))
            for i, p in enumerate(current[1:], 1)
        ]
        trial_cost, trial_hessian, trial_gradient = normal_system(trial, edges, landmarks)
        predicted = float(-2 * gradient @ delta - delta @ hessian @ delta)
        actual = cost - trial_cost
        accepted = math.isfinite(trial_cost) and actual > 0 and predicted > 0
        history.append(
            dict(
                iteration=iteration,
                cost=cost,
                trial_cost=trial_cost,
                scaled_gradient=scaled_gradient,
                damping=damping,
                accepted=accepted,
            )
        )
        if accepted:
            ratio = actual / predicted
            settled = actual <= 1e-8 * max(cost, 1e-12)
            current, cost, hessian, gradient = trial, trial_cost, trial_hessian, trial_gradient
            damping = max(1e-12, damping * max(1 / 3, 1 - (2 * ratio - 1) ** 3))
            growth = 2.0
            if settled:
                converged, reason = True, "relative_cost_converged"
                break
        else:
            damping *= growth
            growth *= 2
    measured = [
        float(np.sqrt(np.mean(np.sum(edge_displacements(current, e, landmarks[e.i]) ** 2, axis=1))))
        for e in edges
    ]
    return current, dict(
        metric="prepared_source_cloud_mean_squared_surface_displacement",
        solver="analytic_joint_diagonal_scaled_marquardt",
        anchor=0,
        nodes=len(original),
        edges=len(edges),
        converged=converged,
        reason=reason,
        iterations_run=len(history),
        history=history,
        initial_sum_squared=initial_cost,
        final_sum_squared=cost,
        per_edge_surface_rms=measured,
        worst_edge_surface_rms=max(measured),
        source_moment_landmarks=[p.tolist() for p in landmarks],
        limitations="Offline heuristic weights, no outlier loss; measured surface RMS is not "
        "independent truth. All held-out geometric gates remain separate.",
    )
