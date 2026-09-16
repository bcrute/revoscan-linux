"""Tracing must preserve the solver; information cannot hide planar freedom."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid
from scanner3d.posegraph_lm import se3_exp
from scanner3d.tracking import DEFAULT_POLICY, convergence_tolerance


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_registration_audit")


def corner():
    return (
        [(a, b, 200.0) for a in range(0, 60, 4) for b in range(0, 60, 4)]
        + [(0.0, a, 200.0 + b) for a in range(0, 60, 4) for b in range(0, 60, 4)]
        + [(a, 0.0, 200.0 + b) for a in range(0, 60, 4) for b in range(0, 60, 4)]
    )


def test_trace_is_exact_and_restores_solver(audit):
    target = corner()
    transform = se3_exp([0.4, -0.2, 0.3, 0.001, 0.002, -0.001])
    source = tuple(transform.inverse().apply_all(target))
    step = icp_plane._solve_point_to_plane
    expected = icp_plane.icp_point_to_plane(
        source,
        target,
        initial=Rigid.identity(),
        max_correspondence_distance=12.0,
        voxel_size=1.5,
        keep_fraction=0.8,
        backend="scipy",
        tolerance=convergence_tolerance(target, fraction=DEFAULT_POLICY.convergence_noise_fraction),
    )
    result = audit.traced_fit(source, target)
    assert result["result"] == expected.as_dict()
    assert icp_plane._solve_point_to_plane is step
    assert len(result["trace"]) == expected.iterations
    assert result["information"]["rank"] == 6


def test_failure_restores_original_solver(audit):
    step = icp_plane._solve_point_to_plane
    result = audit.traced_fit([], corner())
    assert "error" in result and "result" not in result
    assert icp_plane._solve_point_to_plane is step


def test_plane_reports_only_three_observable_directions(audit):
    points = np.array([(a, b, 200) for a in range(10) for b in range(10)], dtype=float)
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    result = audit.information(points, points, normals)
    assert result["rank"] == 3
    assert result["condition"] is None and result["linear_weak_direction_sigma"] is None


def test_information_metric_is_translation_and_uniform_scale_invariant(audit):
    rng = np.random.default_rng(8)
    points = rng.normal(size=(100, 3)) * [10, 20, 5]
    normals = rng.normal(size=points.shape)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    first = audit.information(points, points, normals)
    moved = points * 0.001 + [4, 5, 6]
    second = audit.information(moved, moved, normals)
    assert first["eigenvalues"] == pytest.approx(second["eigenvalues"], rel=1e-10)


def test_nonfinite_information_is_rejected(audit):
    with pytest.raises(ValueError, match="Finite"):
        audit.information(np.full((8, 3), np.nan), np.ones((8, 3)), np.ones((8, 3)))
