"""M3 weighted fit mathematics; original residual/gates stay unweighted."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis
from tests.test_scanner_solver_mechanisms import fixture_pairs, moved


@pytest.fixture
def weighted(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_weighted_icp")


def test_uniform_weights_are_exact_centered_update(weighted):
    centered = importlib.import_module("scanner_solver_mechanisms")
    pairs, _ = fixture_pairs()
    a, _ = centered.centered_step(pairs, Rigid.identity())
    b, details = weighted.weighted_step(pairs, Rigid.identity(), [1.0] * len(pairs), balance=False)
    assert a == b
    assert details["weight_sum"] == len(pairs)
    assert details["effective_weighted_count"] == len(pairs)


@pytest.mark.parametrize("balance", [False, True])
def test_weighted_known_transform_recovery(weighted, balance):
    pairs, truth = fixture_pairs()
    confidence = np.linspace(0.1, 1.0, len(pairs)).tolist()
    pose = Rigid.identity()
    for _ in range(10):
        pose, details = weighted.weighted_step(pairs, pose, confidence, balance=balance)
    points = [p for p, _, _ in pairs]
    assert np.max(np.linalg.norm(moved(pose, points) - moved(truth, points), axis=1)) < 1e-8
    assert details["zero_weight"] == 0


def test_unequal_orthogonal_direction_counts_get_equal_total_weight(weighted):
    normals = np.asarray([(1, 0, 0)] * 60 + [(0, 1, 0)] * 30 + [(0, 0, 1)] * 10)
    result = weighted.matched_weights(normals, np.ones(100), balance=True)
    assert np.allclose([result[:60].sum(), result[60:90].sum(), result[90:].sum()], 100 / 3)
    assert result[-1] == pytest.approx(result[0] * 6)


@pytest.mark.parametrize("balance", [False, True])
def test_normal_sign_and_rigid_coordinate_invariance(weighted, balance):
    pairs, _ = fixture_pairs()
    confidence = np.linspace(0.1, 1.0, len(pairs)).tolist()
    frame = Rigid.of(
        rotation_about_axis((0.4, 0.2, -1.0), 0.65).rotation, (10000.0, -70000.0, 23000.0)
    )
    rotate = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    changed = [
        (frame.apply(p), frame.apply(q), rotate.apply(tuple(-x for x in n))) for p, q, n in pairs
    ]
    a, _ = weighted.weighted_step(pairs, Rigid.identity(), confidence, balance=balance)
    b, _ = weighted.weighted_step(changed, Rigid.identity(), confidence, balance=balance)
    expected = frame.compose(a).compose(frame.inverse())
    points = [p for p, _, _ in changed]
    assert np.max(np.linalg.norm(moved(b, points) - moved(expected, points), axis=1)) < 1e-7


def test_source_planarity_does_not_invent_a_normal_for_a_line(weighted):
    line = [(float(i), 0.0, 0.0) for i in range(31)]
    plane = [(float(x), float(y), 0.0) for x in range(7) for y in range(7)]
    assert weighted.planarity_weights(line) == [0.0] * len(line)
    weights = weighted.planarity_weights(plane)
    assert all(0 < value <= 1 for value in weights)


@pytest.mark.parametrize("bad", [[-1.0] * 6, [float("nan")] * 6, [1.0] * 5])
def test_invalid_weights_refused(weighted, bad):
    with pytest.raises(ValueError):
        weighted.matched_weights(np.ones((6, 3)), bad, balance=False)


def test_zero_weights_and_singular_normal_directions_refused(weighted):
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        weighted.matched_weights(np.ones((6, 3)), np.zeros(6), balance=False)
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        weighted.matched_weights(np.ones((6, 3)), np.ones(6), balance=True)


def test_weighting_does_not_make_flat_geometry_usable(weighted):
    pairs = [
        ((float(x), float(y), 8.0), (float(x), float(y), 9.0), (0.0, 0.0, 1.0))
        for x in range(5)
        for y in range(5)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        weighted.weighted_step(pairs, Rigid.identity(), [1.0] * len(pairs), balance=False)


@pytest.mark.parametrize("profile", ["planarity", "normal-balance", "planarity-balance"])
def test_iteration_reports_the_original_unweighted_residual(weighted, profile):
    pairs, _ = fixture_pairs()
    points = tuple(p for p, _, _ in pairs)
    original_step, original_iteration = icp_plane._solve_point_to_plane, icp_plane._iterate
    experiment = weighted.WeightedExperiment(profile)
    with experiment.active():
        experiment.mark_frame(3)
        result = icp_plane._iterate(
            points,
            lambda *_a: ([(i / 1000, p, q, n) for i, (p, q, n) in enumerate(pairs)], 0),
            initial=Rigid.identity(),
            max_iterations=1,
            threshold=1e-4,
            keep_fraction=0.8,
            gate=12.0,
            accelerated=True,
        )
    kept = pairs[: icp_plane.kept_count(len(pairs), 0.8)]
    expected = icp_plane._point_to_plane_rmse(
        [(result.transform.apply(p), q, n) for p, q, n in kept]
    )
    assert result.rmse == expected
    assert (
        icp_plane._solve_point_to_plane is original_step
        and icp_plane._iterate is original_iteration
    )
    assert all(row["source_index"] == 3 for row in experiment.records)
    assert experiment._confidence is None and not experiment._cache


def test_closed_modes_bounds_and_context_cleanup(weighted):
    with pytest.raises(ValueError):
        weighted.WeightedExperiment("retuned")
    a = weighted.WeightedExperiment("planarity")
    b = weighted.WeightedExperiment("normal-balance")
    original = icp_plane._iterate
    with pytest.raises(ValueError):
        a.mark_frame(3000)
    with pytest.raises(RuntimeError, match="injected"), a.active():
        with pytest.raises(RuntimeError, match="nested"), b.active():
            pass
        raise RuntimeError("injected")
    assert icp_plane._iterate is original


def test_source_confidence_cache_is_bounded_and_cleared(weighted, monkeypatch):
    calls = []

    def confidence(points):
        calls.append(points)
        return [0.5] * len(points)

    monkeypatch.setattr(weighted, "planarity_weights", confidence)
    monkeypatch.setattr(
        icp_plane,
        "_iterate",
        lambda *_a, **_k: icp_plane._FitPlane(Rigid.identity(), 0.0, 1, True, 8, 0, 0, 1.0),
    )
    experiment = weighted.WeightedExperiment("planarity")
    with experiment.active():
        for index in range(40):
            points = tuple((float(i), float(index), 3.0) for i in range(11))
            icp_plane._iterate(points, None)
            assert len(experiment._cache) <= 32
        icp_plane._iterate(points, None)
    assert len(calls) == 40 and not experiment._cache
