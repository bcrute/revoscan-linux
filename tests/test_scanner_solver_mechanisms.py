"""Centered finite updates and transparent, bounded offline instrumentation."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis


@pytest.fixture
def mechanism(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_solver_mechanisms")


def fixture_pairs():
    rng = np.random.default_rng(76231)
    directions = rng.normal(size=(180, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    axes = np.array([31.0, 47.0, 22.0])
    target = directions * axes + [130.0, -20.0, 210.0]
    normals = directions / axes
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    rotation = rotation_about_axis((0.2, -0.5, 0.8), 0.045).rotation
    truth = Rigid.of(rotation, (0.3, -0.7, 1.1))
    source = truth.inverse().apply_all([tuple(p) for p in target])
    return [(p, tuple(q), tuple(n)) for p, q, n in zip(source, target, normals, strict=True)], truth


def moved(pose, points):
    return np.asarray(list(pose.apply_all(points)))


def test_known_transform_converges_with_original_correspondences(mechanism):
    pairs, truth = fixture_pairs()
    pose = Rigid.identity()
    for _ in range(8):
        pose, data = mechanism.centered_step(pairs, pose)
    points = [p for p, _, _ in pairs]
    assert np.max(np.linalg.norm(moved(pose, points) - moved(truth, points), axis=1)) < 1e-8
    assert data["information_condition"] > 1
    assert data["matched"] == len(pairs)


@pytest.mark.parametrize("shift", [(100.0, -170.0, 240.0), (100000.0, -50000.0, 70000.0)])
def test_update_is_coordinate_translation_equivariant(mechanism, shift):
    pairs, _ = fixture_pairs()
    initial = rotation_about_axis((1.0, 2.0, -1.0), 0.012)
    shifted_frame = Rigid.of(Rigid.identity().rotation, shift)
    shifted_pairs = [(shifted_frame.apply(p), shifted_frame.apply(q), n) for p, q, n in pairs]
    result, _ = mechanism.centered_step(pairs, initial)
    shifted_result, _ = mechanism.centered_step(
        shifted_pairs, shifted_frame.compose(initial).compose(shifted_frame.inverse())
    )
    expected = shifted_frame.compose(result).compose(shifted_frame.inverse())
    points = [p for p, _, _ in shifted_pairs]
    assert (
        np.max(np.linalg.norm(moved(shifted_result, points) - moved(expected, points), axis=1))
        < 1e-7
    )


def test_update_is_coordinate_rotation_equivariant(mechanism):
    pairs, _ = fixture_pairs()
    frame = rotation_about_axis((1.0, -3.0, 2.0), 0.73)
    rotated = [(frame.apply(p), frame.apply(q), frame.apply(n)) for p, q, n in pairs]
    result, _ = mechanism.centered_step(pairs, Rigid.identity())
    transformed, _ = mechanism.centered_step(rotated, Rigid.identity())
    expected = frame.compose(result).compose(frame.inverse())
    points = [p for p, _, _ in rotated]
    assert (
        np.max(np.linalg.norm(moved(transformed, points) - moved(expected, points), axis=1)) < 1e-8
    )


def test_original_linear_rotation_matches_but_finite_translation_has_second_order_term(mechanism):
    pairs, _ = fixture_pairs()
    original = icp_plane._solve_point_to_plane(pairs, Rigid.identity(), accelerated=True)
    centered, data = mechanism.centered_step(pairs, Rigid.identity())
    assert np.allclose(original.rotation, centered.rotation, rtol=0, atol=1e-10)
    rotation = np.asarray(centered.rotation)
    theta = centered.rotation_angle
    vector = (
        theta
        / (2 * np.sin(theta))
        * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    )
    centre = np.asarray(data["centre"])
    expected = centre - rotation @ centre + np.cross(vector, centre)
    actual = np.asarray(centered.translation) - np.asarray(original.translation)
    assert np.allclose(actual, expected, rtol=0, atol=1e-8)
    assert np.linalg.norm(actual) > 0.01


def test_normal_sign_does_not_change_solution(mechanism):
    pairs, _ = fixture_pairs()
    flipped = [(p, q, tuple(-v for v in n)) for p, q, n in pairs]
    a, _ = mechanism.centered_step(pairs, Rigid.identity())
    b, _ = mechanism.centered_step(flipped, Rigid.identity())
    assert a == b


def test_unconstrained_plane_is_still_refused(mechanism):
    pairs = [
        ((float(x), float(y), 8.0), (float(x), float(y), 9.0), (0.0, 0.0, 1.0))
        for x in range(5)
        for y in range(5)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        mechanism.centered_step(pairs, Rigid.identity())


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [(1, 2, 3)] * 6,
        [((0, 0, 0), (0, 0, 0), (0, 0, 0))] * 6,
        [((float("nan"), 0, 0), (0, 0, 0), (0, 0, 1))] * 6,
    ],
)
def test_malformed_input_refused(mechanism, bad):
    with pytest.raises(ValueError):
        mechanism.centered_step(bad, Rigid.identity())


def test_observation_returns_original_exactly_and_restores_patch(mechanism):
    pairs, _ = fixture_pairs()
    original = icp_plane._solve_point_to_plane
    expected = original(pairs, Rigid.identity(), accelerated=True)
    observer = mechanism.SolverExperiment("observe-origin")
    observer.mark_frame(14)
    with observer.active():
        actual = icp_plane._solve_point_to_plane(pairs, Rigid.identity(), accelerated=True)
    assert actual == expected and icp_plane._solve_point_to_plane is original
    assert observer.records[0]["source_index"] == 14
    assert observer.records[0]["paired_step_difference_median"] > 0


def test_observation_does_not_replace_original_failure(mechanism, monkeypatch):
    def fail(*_a, **_k):
        raise icp_plane.IcpPlaneGeometryUnusable("injected")

    monkeypatch.setattr(icp_plane, "_solve_point_to_plane", fail)
    observer = mechanism.SolverExperiment("observe-origin")
    with observer.active(), pytest.raises(icp_plane.IcpPlaneGeometryUnusable, match="injected"):
        icp_plane._solve_point_to_plane(*[fixture_pairs()[0], Rigid.identity()])
    assert icp_plane._solve_point_to_plane is fail
    assert "original_error" in observer.records[0]


def test_context_nesting_and_exception_restore(mechanism):
    original = icp_plane._solve_point_to_plane
    a = mechanism.SolverExperiment("centered")
    b = mechanism.SolverExperiment("observe-origin")
    with pytest.raises(RuntimeError, match="injected"), a.active():
        with pytest.raises(RuntimeError, match="nested"), b.active():
            pass
        assert icp_plane._solve_point_to_plane == a.step
        raise RuntimeError("injected")
    assert icp_plane._solve_point_to_plane is original
    with b.active():
        assert icp_plane._solve_point_to_plane == b.step


def test_closed_profile_and_step_bounds(mechanism, monkeypatch):
    with pytest.raises(ValueError):
        mechanism.SolverExperiment("retuned")
    observer = mechanism.SolverExperiment("centered")
    with pytest.raises(ValueError):
        observer.mark_frame(3000)
    with pytest.raises(RuntimeError, match="active"):
        observer.step(fixture_pairs()[0], Rigid.identity())
    monkeypatch.setattr(mechanism, "MAX_RECORDED_STEPS", 0)
    with observer.active(), pytest.raises(RuntimeError, match="bound"):
        icp_plane._solve_point_to_plane(fixture_pairs()[0], Rigid.identity())
