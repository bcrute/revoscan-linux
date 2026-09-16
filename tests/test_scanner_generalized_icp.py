"""Closed M5 covariance geometry and honest inherited reporting/refusals."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis
from tests.test_scanner_symmetric_icp import distance, pairs_for


@pytest.fixture
def generalized(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_generalized_icp")


def test_parallel_and_orthogonal_normals_have_expected_precision(generalized):
    source = [(0, 0, 1), (0, 0, 1)]
    target = [(0, 0, 1), (1, 0, 0)]
    actual = generalized.paired_precision(source, target)
    e = generalized.COVARIANCE_EPSILON
    np.testing.assert_allclose(actual[0], np.diag([0.5, 0.5, 1 / (2 * e)]), rtol=1e-12)
    np.testing.assert_allclose(actual[1], np.diag([1 / (1 + e), 0.5, 1 / (1 + e)]))


@pytest.mark.parametrize("angle", [0.0, 0.045, 0.35])
def test_known_correspondences_converge_to_transform(generalized, angle):
    pairs, truth = pairs_for(angle)
    pose = Rigid.identity()
    for _ in range(15):
        pose, details = generalized.covariance_step(pairs, pose)
    assert distance(pose, truth, [p for p, _, _, _ in pairs]) < 1e-8
    assert details["original_plane_information_guard_passed"]
    assert details["covariance_cost_after_rotated_metric"] < 1e-15


def test_normal_sign_and_magnitude_do_not_change_precision(generalized):
    source = np.array([(0.1, 0.2, 0.9), (0.3, 1.0, 0.4)])
    target = np.array([(0.2, 0.1, 0.9), (0.2, 1.0, 0.4)])
    a = generalized.paired_precision(source, target)
    b = generalized.paired_precision(-source * 3, target * 2)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_individual_normal_signs_leave_step_unchanged(generalized):
    pairs, _ = pairs_for(0.045)
    flipped = [
        (p, q, tuple(-v for v in sn) if i % 2 else sn, tuple(-v for v in tn) if i % 3 else tn)
        for i, (p, q, sn, tn) in enumerate(pairs)
    ]
    a, _ = generalized.covariance_step(pairs, Rigid.identity())
    b, _ = generalized.covariance_step(flipped, Rigid.identity())
    assert distance(a, b, [p for p, _, _, _ in pairs]) < 1e-10


def test_rigid_coordinate_change_commutes_with_step(generalized):
    pairs, _ = pairs_for(0.045)
    frame = Rigid.of(
        rotation_about_axis((1.0, 2.0, 3.0), 0.71).rotation, (10000.0, -20000.0, 40000.0)
    )
    normal_frame = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    transformed = [
        (frame.apply(p), frame.apply(q), normal_frame.apply(sn), normal_frame.apply(tn))
        for p, q, sn, tn in pairs
    ]
    a, _ = generalized.covariance_step(pairs, Rigid.identity())
    b, _ = generalized.covariance_step(transformed, Rigid.identity())
    expected = frame.compose(a).compose(frame.inverse())
    assert distance(expected, b, [p for p, _, _, _ in transformed]) < 1e-7


def test_tangential_regularization_does_not_override_plane_refusal(generalized):
    pairs = [
        ((float(x), float(y), 8.0), (float(x), float(y), 9.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        for x in range(5)
        for y in range(5)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        generalized.covariance_step(pairs, Rigid.identity())


@pytest.mark.parametrize(
    "source,target",
    [
        ([], []),
        ([(0, 0, 0)], [(1, 0, 0)]),
        ([(1, 0, 0)], []),
        ([(float("nan"), 0, 0)], [(1, 0, 0)]),
    ],
)
def test_invalid_normal_arrays_refuse(generalized, source, target):
    with pytest.raises(ValueError):
        generalized.paired_precision(source, target)


def test_source_context_is_required_and_failures_are_preserved(generalized):
    experiment = generalized.GeneralizedExperiment()
    with pytest.raises(RuntimeError, match="context"):
        experiment.step([], Rigid.identity())
    experiment._normals = {}
    with pytest.raises(ValueError):
        experiment.step([], Rigid.identity())
    assert experiment.records[-1]["status"] == "failed"
    assert experiment.records[-1]["profile"] == "generalized-covariance"


@pytest.mark.parametrize("all_missing", [False, True])
def test_inherited_context_missing_data_and_unweighted_residual(generalized, all_missing):
    pairs, _ = pairs_for(0.045)
    points = tuple(p for p, _, _, _ in pairs)
    original_step, original_iteration = icp_plane._solve_point_to_plane, icp_plane._iterate

    def provider(source, neighbours):
        assert source == points and neighbours == 10
        return icp_plane.NormalField(
            tuple(
                icp_plane.NormalEstimate(p, None if all_missing or i == 0 else sn, 10)
                for i, (p, _, sn, _) in enumerate(pairs)
            ),
            10,
            (),
        )

    def correspond(source, pose):
        return [(float(i) / 1000, p, q, tn) for i, (p, q, _, tn) in enumerate(pairs)], 0

    def target_only(*_):
        raise AssertionError("Source normals requested from target-only provider")

    experiment = generalized.GeneralizedExperiment()
    with icp_plane.normal_acceleration(provider), experiment.active():
        experiment.mark_frame(7)
        with icp_plane.normal_acceleration(target_only):
            parameters = dict(
                initial=Rigid.identity(),
                max_iterations=1,
                threshold=1e-4,
                keep_fraction=0.8,
                gate=12.0,
                accelerated=True,
            )
            if all_missing:
                with pytest.raises(icp_plane.IcpPlaneCorrespondencesInsufficient):
                    icp_plane._iterate(points, correspond, **parameters)
            else:
                result = icp_plane._iterate(points, correspond, **parameters)
                count = icp_plane.kept_count(len(points) - 1, 0.8)
                expected = icp_plane._point_to_plane_rmse(
                    [(result.transform.apply(p), q, tn) for p, q, _, tn in pairs[1 : count + 1]]
                )
                assert result.rmse == pytest.approx(expected, rel=1e-12, abs=1e-12)
                assert result.rejected == 1 and result.correspondences == count
    assert icp_plane._solve_point_to_plane is original_step
    assert icp_plane._iterate is original_iteration
    assert experiment._normals is None and experiment._provider is None
    assert all(r["source_index"] == 7 for r in experiment.records)
    assert experiment.records[0]["source_normals_missing"] == (len(points) if all_missing else 1)
