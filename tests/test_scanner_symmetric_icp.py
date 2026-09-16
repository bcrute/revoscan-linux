"""Analytic M2 equations and bounded normal/context provenance."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis


@pytest.fixture
def symmetric(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_symmetric_icp")


def pairs_for(angle):
    rng = np.random.default_rng(24319)
    directions = rng.normal(size=(240, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    axes = np.asarray([25.0, 47.0, 31.0])
    target = directions * axes + [130.0, -20.0, 210.0]
    normals = directions / axes
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    truth = Rigid.of(rotation_about_axis((0.3, -0.7, 0.2), angle).rotation, (1.0, 2.0, -0.5))
    inverse = truth.inverse()
    normal_rotation = Rigid.of(inverse.rotation, (0.0, 0.0, 0.0))
    return [
        (inverse.apply(tuple(q)), tuple(q), normal_rotation.apply(tuple(n)), tuple(n))
        for q, n in zip(target, normals, strict=True)
    ], truth


def distance(a, b, points):
    return np.max(
        np.linalg.norm(
            np.asarray(list(a.apply_all(points))) - np.asarray(list(b.apply_all(points))), axis=1
        )
    )


@pytest.mark.parametrize("angle", [0.0, 0.045, 0.35, 0.65])
def test_exact_correspondences_recover_truth_in_one_step(symmetric, angle):
    pairs, truth = pairs_for(angle)
    result, _ = symmetric.symmetric_step(pairs, Rigid.identity())
    assert distance(result, truth, [p for p, _, _, _ in pairs]) < 1e-8


def test_nonidentity_initial_does_not_change_transform_convention(symmetric):
    pairs, truth = pairs_for(0.35)
    initial = rotation_about_axis((1.0, 2.0, 3.0), 0.08)
    result, _ = symmetric.symmetric_step(pairs, initial)
    assert distance(result, truth, [p for p, _, _, _ in pairs]) < 1e-8


def test_swapped_correspondences_return_inverse(symmetric):
    pairs, truth = pairs_for(0.35)
    swapped = [(q, p, nq, np_) for p, q, np_, nq in pairs]
    result, _ = symmetric.symmetric_step(swapped, Rigid.identity())
    assert distance(result, truth.inverse(), [p for p, _, _, _ in swapped]) < 1e-8


@pytest.mark.parametrize("source_flip,target_flip", [(True, False), (False, True), (True, True)])
def test_individual_normal_signs_do_not_change_placement(symmetric, source_flip, target_flip):
    pairs, _ = pairs_for(0.35)
    flipped = [
        (
            p,
            q,
            tuple(-x if source_flip and i % 2 else x for x in np_),
            tuple(-x if target_flip and i % 3 else x for x in nq),
        )
        for i, (p, q, np_, nq) in enumerate(pairs)
    ]
    a, _ = symmetric.symmetric_step(pairs, Rigid.identity())
    b, _ = symmetric.symmetric_step(flipped, Rigid.identity())
    assert distance(a, b, [p for p, _, _, _ in pairs]) < 1e-9


def test_arbitrary_rigid_coordinate_change(symmetric):
    pairs, _ = pairs_for(0.35)
    frame = Rigid.of(
        rotation_about_axis((2.0, 1.0, 3.0), 0.73).rotation, (10000.0, -32000.0, 71000.0)
    )
    normal_frame = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    changed = [
        (frame.apply(p), frame.apply(q), normal_frame.apply(np_), normal_frame.apply(nq))
        for p, q, np_, nq in pairs
    ]
    a, _ = symmetric.symmetric_step(pairs, Rigid.identity())
    b, _ = symmetric.symmetric_step(changed, Rigid.identity())
    expected = frame.compose(a).compose(frame.inverse())
    assert distance(expected, b, [p for p, _, _, _ in changed]) < 1e-7


def test_flat_unconstrained_scene_is_refused(symmetric):
    pairs = [
        ((float(x), float(y), 8.0), (float(x), float(y), 9.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        for x in range(5)
        for y in range(5)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        symmetric.symmetric_step(pairs, Rigid.identity())


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [(1, 2, 3)] * 6,
        [((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 1))] * 6,
        [((float("inf"), 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1))] * 6,
    ],
)
def test_invalid_correspondences_refused(symmetric, bad):
    with pytest.raises(ValueError):
        symmetric.symmetric_step(bad, Rigid.identity())


def test_provider_required_and_both_patches_restored(symmetric):
    step, iteration = icp_plane._solve_point_to_plane, icp_plane._iterate
    experiment = symmetric.SymmetricExperiment()
    with pytest.raises(RuntimeError, match="provider"), experiment.active():
        pass
    with (
        icp_plane.normal_acceleration(lambda p, n: icp_plane.estimate_normals(p, neighbours=n)),
        pytest.raises(RuntimeError, match="injected"),
        experiment.active(),
    ):
        assert icp_plane._solve_point_to_plane == experiment.step
        assert icp_plane._iterate == experiment.iterate
        raise RuntimeError("injected")
    assert icp_plane._solve_point_to_plane is step and icp_plane._iterate is iteration


def test_source_provider_survives_nested_target_override_and_counts_missing(symmetric):
    pairs, _ = pairs_for(0.045)
    points = tuple(p for p, _, _, _ in pairs)
    calls = []

    def source_provider(source, neighbours):
        calls.append((source, neighbours))
        return icp_plane.NormalField(
            tuple(
                icp_plane.NormalEstimate(p, np_ if i else None, 10)
                for i, (p, _, np_, _) in enumerate(pairs)
            ),
            10,
            (),
        )

    def target_only(*_a):
        raise AssertionError("Source requested from target-only provider")

    def correspond(_source, _transform):
        return [(float(i) / 1000, p, q, nq) for i, (p, q, _, nq) in enumerate(pairs)], 0

    experiment = symmetric.SymmetricExperiment()
    with icp_plane.normal_acceleration(source_provider), experiment.active():
        experiment.mark_frame(6)
        with icp_plane.normal_acceleration(target_only):
            result = icp_plane._iterate(
                points,
                correspond,
                initial=Rigid.identity(),
                max_iterations=1,
                threshold=1e-4,
                keep_fraction=0.8,
                gate=12.0,
                accelerated=True,
            )
    assert calls == [(points, 10)]
    assert result.rejected == 1
    assert result.correspondences == icp_plane.kept_count(len(points) - 1, 0.8)
    assert experiment.records[0]["source_normals_missing"] == 1
    assert all(row["source_index"] == 6 for row in experiment.records)
    assert experiment._normals is None


def test_all_missing_source_normals_refuse_without_fabrication(symmetric):
    pairs, _ = pairs_for(0.045)
    points = tuple(p for p, _, _, _ in pairs)

    def provider(source, neighbours):
        return icp_plane.NormalField(
            tuple(icp_plane.NormalEstimate(p, None, 10) for p in source), neighbours, ()
        )

    experiment = symmetric.SymmetricExperiment()
    with (
        icp_plane.normal_acceleration(provider),
        experiment.active(),
        pytest.raises(icp_plane.IcpPlaneCorrespondencesInsufficient),
    ):
        icp_plane._iterate(
            points,
            lambda *_a: ([(0.0, p, q, nq) for p, q, _, nq in pairs], 0),
            initial=Rigid.identity(),
            max_iterations=2,
            threshold=1e-4,
            keep_fraction=0.8,
            gate=12.0,
        )
    assert experiment.records[0]["source_normals_missing"] == len(points)
    assert experiment.records[0]["status"] == "failed"
    assert experiment._normals is None


def test_closed_registry_and_cross_mechanism_nesting(symmetric):
    registry = importlib.import_module("scanner_solver_registry")
    with pytest.raises(ValueError):
        registry.create_experiment("retuned")
    a = registry.create_experiment("centered")
    b = registry.create_experiment("symmetric")
    with a.active(), pytest.raises(RuntimeError, match="nested"), b.active():
        pass


def test_existing_settling_iteration_is_preserved(symmetric, monkeypatch):
    settling = importlib.import_module("scanner_pose_settling")
    monkeypatch.setattr(icp_plane, "_iterate", settling.iterate)
    experiment = symmetric.SymmetricExperiment()
    with icp_plane.normal_acceleration(lambda *_a: None), experiment.active():
        assert experiment._iteration is settling.iterate
    assert icp_plane._iterate is settling.iterate
