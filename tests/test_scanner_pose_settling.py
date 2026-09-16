"""Residual plateau alone cannot certify a pose that continues to move."""

import importlib
from pathlib import Path

import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid
from scanner3d.posegraph_lm import se3_exp


@pytest.fixture
def settling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_pose_settling")


def test_constant_residual_with_large_pose_steps_is_not_converged(settling, monkeypatch):
    source = [(float(i), 0.0, 200.0) for i in range(10)]

    def correspond(cloud, transform):
        return [(0.0, p, p, (0.0, 0.0, 1.0)) for p in cloud], 0

    def moving_step(matched, current, **kwargs):
        return se3_exp([1, 0, 0, 0, 0, 0]).compose(current)

    monkeypatch.setattr(icp_plane, "_solve_point_to_plane", moving_step)
    options = dict(
        initial=Rigid.identity(), max_iterations=5, threshold=0.0065, keep_fraction=0.8, gate=12.0
    )
    previous = icp_plane._iterate(source, correspond, **options)
    guarded = settling.iterate(source, correspond, **options)
    assert previous.converged and previous.iterations == 2
    assert not guarded.converged and guarded.iterations == 5
    assert guarded.rmse == previous.rmse == 0


def test_same_arithmetic_until_old_stop_then_exact_rigid_recovery(settling, monkeypatch):
    from tests.test_scanner_registration_audit import corner

    target = corner()
    truth = se3_exp([0.3, -0.2, 0.1, 0.001, -0.002, 0.001])
    source = tuple(truth.inverse().apply_all(target))
    step = icp_plane._solve_point_to_plane
    paths = []
    current = []

    def recorded(matched, before, **kwargs):
        result = step(matched, before, **kwargs)
        current.append(result)
        return result

    monkeypatch.setattr(icp_plane, "_solve_point_to_plane", recorded)
    for enhanced in (False, True):
        current = []
        if enhanced:
            with settling.pose_settling():
                result = icp_plane.icp_point_to_plane(
                    source, target, backend="scipy", voxel_size=1.5
                )
        else:
            result = icp_plane.icp_point_to_plane(source, target, backend="scipy", voxel_size=1.5)
        paths.append(current)
    assert paths[1][: len(paths[0])] == paths[0]
    assert result.converged
    assert result.transform.apply(source[80]) == pytest.approx(target[80], abs=0.01)


def test_context_restores_solver_after_exception(settling):
    previous = icp_plane._iterate
    with pytest.raises(RuntimeError), settling.pose_settling():
        assert icp_plane._iterate is settling.iterate
        raise RuntimeError("test interruption")
    assert icp_plane._iterate is previous


def test_unusable_geometry_is_still_refused(settling):
    with settling.pose_settling(), pytest.raises(icp_plane.IcpPlaneNotAllowed):
        icp_plane.icp_point_to_plane([], [], backend="scipy")
