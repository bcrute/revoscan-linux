"""Measured-cloud metric is physically meaningful independently of the solver."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d.geometry import Rigid
from scanner3d.posegraph import PoseEdge
from scanner3d.posegraph_lm import se3_exp


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_surface_graph")


def cloud():
    points = np.random.default_rng(47).normal(size=(113, 3)) * [10, 20, 5] + [20, -10, 200]
    return [tuple(row) for row in points]


def test_four_landmarks_reproduce_full_cloud_displacement_exactly(graph):
    points = cloud()
    landmarks = graph.moment_landmarks(points)
    poses = [se3_exp([2, 3, 4, 0.1, -0.2, 0.3]), se3_exp([-3, 5, 2, -0.2, 0.1, 0])]
    edge = PoseEdge(1, 0, se3_exp([1, 2, 3, 0.04, 0.01, -0.03]), 1)
    full = graph.edge_displacements(poses, edge, np.asarray(points))
    reduced = graph.edge_displacements(poses, edge, landmarks)
    assert np.mean(np.sum(full**2, axis=1)) == pytest.approx(
        np.mean(np.sum(reduced**2, axis=1)), rel=1e-12
    )


def test_change_of_sensor_coordinate_origin_does_not_change_physical_metric(graph):
    points = cloud()
    poses = [Rigid.identity(), se3_exp([4, -3, 1, 0.02, 0.1, -0.04])]
    edge = PoseEdge(1, 0, se3_exp([3, -2, 0, 0.01, 0.09, -0.02]), 1)
    basis = se3_exp([1000, -900, 700, 0.3, -0.2, 0.1])
    rebased_poses = [p.compose(basis.inverse()) for p in poses]
    rebased_edge = PoseEdge(1, 0, basis.compose(edge.transform).compose(basis.inverse()), 1)
    before = graph.edge_displacements(poses, edge, graph.moment_landmarks(points))
    after = graph.edge_displacements(
        rebased_poses, rebased_edge, graph.moment_landmarks(tuple(basis.apply_all(points)))
    )
    assert np.mean(before**2) == pytest.approx(np.mean(after**2), rel=1e-10)


@pytest.mark.parametrize("solver", ["optimise", "analytic_optimise"])
def test_graph_recovers_known_noncommuting_poses_and_keeps_anchor(graph, solver):
    truth = [se3_exp([i * 3, i * 2, -i, i * 0.03, i * 0.02, -i * 0.01]) for i in range(5)]
    start = [truth[0]] + [
        p.compose(se3_exp([i * 0.3, 0.2, -0.1, 0.002, -0.001, 0.003]))
        for i, p in enumerate(truth[1:], 1)
    ]
    edges = [
        PoseEdge(i, j, truth[j].inverse().compose(truth[i]), 1)
        for i in range(1, 5)
        for j in range(i)
    ]
    result, report = getattr(graph, solver)(start, edges, [cloud()] * 5)
    assert report["converged"] and result[0] == start[0]
    assert report["worst_edge_surface_rms"] < 1e-3
    for actual, expected in zip(result, truth, strict=True):
        assert graph.apply(actual, np.asarray(cloud())) == pytest.approx(
            graph.apply(expected, np.asarray(cloud())), abs=1e-3
        )


def test_degenerate_or_nonfinite_clouds_are_not_valid_geometry(graph):
    for points in ([], [(0, 0, 0)] * 4, [(float("nan"), 1, 2)] * 4):
        with pytest.raises(ValueError):
            graph.moment_landmarks(points)


def test_disconnected_graph_cannot_invent_unanchored_poses(graph):
    with pytest.raises(ValueError, match="path"):
        graph.optimise([Rigid.identity()] * 3, [PoseEdge(1, 0, Rigid.identity(), 1)], [cloud()] * 3)


def test_analytic_edge_jacobians_match_independent_finite_differences(graph):
    poses = [se3_exp([3, -2, 10, -0.3, 0.1, 0.04]), se3_exp([-1, 4, 9, 0.2, 0.03, -0.1])]
    edge = PoseEdge(1, 0, se3_exp([4, -1, 3, 0.02, -0.03, 0.01]), 0.3)
    points = graph.moment_landmarks(cloud())
    _, ji, jj = graph.edge_system(poses, edge, points)
    step = 1e-6
    for node, jacobian in ((1, ji), (0, jj)):
        numeric = np.zeros((12, 6))
        for coordinate in range(6):
            delta = np.zeros(6)
            delta[coordinate] = step
            plus, minus = list(poses), list(poses)
            plus[node] = poses[node].compose(se3_exp(delta.tolist()))
            minus[node] = poses[node].compose(se3_exp((-delta).tolist()))
            numeric[:, coordinate] = (
                graph.edge_displacements(plus, edge, points).ravel()
                - graph.edge_displacements(minus, edge, points).ravel()
            )
        numeric *= np.sqrt(edge.weight / 4) / (2 * step)
        assert jacobian == pytest.approx(numeric, rel=1e-6, abs=1e-7)


def test_perfect_surface_graph_stays_exactly_unchanged(graph):
    poses = [se3_exp([i, -i, i * 2, 0.03 * i, 0.01 * i, 0.02 * i]) for i in range(4)]
    edges = [PoseEdge(i, i - 1, poses[i - 1].inverse().compose(poses[i]), 1) for i in range(1, 4)]
    result, report = graph.analytic_optimise(poses, edges, [cloud()] * 4)
    assert report["converged"] and report["iterations_run"] == 0
    assert result == poses


def test_expanded_node_bound_requires_explicit_closed_protocol(graph):
    poses = [Rigid.identity()] * 129
    edges = [PoseEdge(i, i - 1, Rigid.identity(), 1) for i in range(1, len(poses))]
    clouds = [cloud()] * len(poses)
    with pytest.raises(ValueError, match="Bounded"):
        graph.validated_landmarks(poses, edges, clouds)
    assert len(graph.validated_landmarks(poses, edges, clouds, max_nodes=256)) == 129
    with pytest.raises(ValueError, match="Bounded"):
        graph.validated_landmarks(poses, edges, clouds, max_nodes=1024)
