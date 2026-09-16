"""M24 weighted covariance and diagnostic scope; real GPU parity is separate."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.tracking import DEFAULT_POLICY


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_normal_taper")


def points():
    return tuple(
        (1.1 * x, 1.3 * y, 100 + 0.07 * x * x + 0.15 * y * y + 0.03 * x * y)
        for x in range(-6, 7)
        for y in range(-6, 7)
    )


def arguments():
    return dict(
        backend="scipy",
        voxel_size=1.5,
        keep_fraction=0.8,
        max_correspondence_distance=12.0,
        initial=Rigid.identity(),
    )


def cpu_gpu(module):
    """CPU-reference test double only; no real GPU or OpenGL context is claimed."""
    gpu = object.__new__(module.GpuNormals)
    gpu.np = np
    gpu.calls = dict(jacobi=0)

    def jacobi(matrices):
        gpu.calls["jacobi"] += 1
        return [icp_plane._jacobi(np.asarray(m).tolist()) for m in matrices]

    gpu.jacobi = jacobi
    return gpu


def test_patch_matches_independent_weighted_matrix(module):
    patch = np.random.default_rng(912).normal(size=(21, 3))
    actual, data = module.tapered_patch(tuple(patch[0]), [tuple(p) for p in patch[1:]])
    distances = np.sum((patch - patch[0]) ** 2, axis=1)
    weights = np.maximum(0.0, 1.0 - distances / distances.max()) ** 2
    center = np.sum(patch * weights[:, None], axis=0) / weights.sum()
    delta = patch - center
    expected = delta.T @ (weights[:, None] * delta)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(data["centre"], center, atol=1e-14)
    np.testing.assert_allclose(data["weights"], weights, atol=1e-14)
    assert data["positive_neighbors"] == 19 and data["zero_weight_neighbors"] == 1
    assert 1 < data["effective_count"] <= 20


def test_replacing_zero_weight_boundary_does_not_change_covariance(module):
    inner = [(i / 20.0, (i % 4) / 8.0, (i % 5) / 10.0) for i in range(1, 20)]
    a = module.tapered_patch((0.0, 0.0, 0.0), [*inner, (3.0, 0.0, 0.0)])
    b = module.tapered_patch((0.0, 0.0, 0.0), [*inner, (0.0, 0.0, -3.0)])
    assert a == b and a[1]["weights"][-1] == 0


@pytest.mark.parametrize("count", [0, 3, 20])
def test_coincident_patch_is_not_given_invented_spread(module, count):
    matrix, data = module.tapered_patch((4.0, 7.0, 9.0), [(4.0, 7.0, 9.0)] * count)
    assert matrix == [[0.0] * 3] * 3
    assert data["radius_squared"] == 0 and data["positive_neighbors"] == count
    assert data["weights"] == [1.0] * (count + 1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_patch_rejected(module, bad):
    with pytest.raises(ValueError):
        module.tapered_patch((bad, 0.0, 0.0), [(0.0, 1.0, 0.0)])


@pytest.mark.parametrize("count", [0, 5, 2049])
def test_field_bounds(module, count):
    with pytest.raises(ValueError, match=r"6\.\.2048"):
        module.tapered_field([(0.0, 0.0, 0.0)] * count, lambda rows: [])


def test_rigid_covariance_equivariance_and_scale(module):
    patch = points()[:21]
    matrix, data = module.tapered_patch(patch[0], patch[1:])
    spin = rotation_about_axis((1.0, 2.0, 3.0), 0.31)
    transform = Rigid.of(spin.rotation, (10.0, -20.0, 15.0))
    moved = [transform.apply(p) for p in patch]
    actual, shifted = module.tapered_patch(moved[0], moved[1:])
    r = np.asarray(spin.rotation)
    np.testing.assert_allclose(actual, r @ np.asarray(matrix) @ r.T, atol=1e-11)
    np.testing.assert_allclose(shifted["weights"], data["weights"], atol=1e-12)
    scaled = [tuple(7 * v for v in p) for p in patch]
    larger, scale_data = module.tapered_patch(scaled[0], scaled[1:])
    np.testing.assert_allclose(larger, 49 * np.asarray(matrix), atol=1e-10)
    np.testing.assert_allclose(scale_data["weights"], data["weights"], atol=1e-12)


@pytest.mark.parametrize("profile", ["observe-20", "taper-20"])
def test_field_observation_identity_or_independent_cpu_result(module, monkeypatch, profile):
    gpu = cpu_gpu(module)
    expected = (
        module.ORIGINAL_ESTIMATE(gpu, points(), neighbours=20, backend="scipy")
        if profile == "observe-20"
        else module.tapered_field(points(), gpu.jacobi)[0]
    )
    if profile == "observe-20":
        old = module.ORIGINAL_ESTIMATE

        def original(gpu, cloud, **kwargs):
            assert cloud == points() and kwargs == dict(neighbours=20, backend="scipy")
            old(gpu, cloud, **kwargs)
            return expected

        monkeypatch.setattr(module, "ORIGINAL_ESTIMATE", original)
        monkeypatch.setattr(module.GpuNormals, "estimate_normals", original)
    experiment = module.NormalTaperExperiment(profile)
    with experiment.active():
        actual = gpu.estimate_normals(points(), neighbours=20, backend="scipy")
    assert actual == expected
    if profile == "observe-20":
        assert actual is expected
    row = experiment.field_records[0]
    assert row["status"] == "completed" and row["jacobi_dispatches"] == 1
    assert row["point_identity_exact"] and row["neighbors"] == 20
    assert actual.requested_neighbours == 20
    assert all(e.neighbours <= (20 if profile == "observe-20" else 19) for e in actual.estimates)


@pytest.mark.parametrize("count", [6, 30])
def test_all_coincident_normals_are_refused(module, count):
    field, stats = module.tapered_field([(1.0, 2.0, 3.0)] * count, cpu_gpu(module).jacobi)
    assert len(field.estimates) == count and not field.usable
    assert stats["classifications"] == dict(sparse=0, coincident=count, isotropic=0)


@pytest.mark.parametrize("profile", ["observe-20", "taper-20"])
def test_only_twenty_neighbor_argument_changes_original_fit(module, monkeypatch, profile):
    calls = []
    result = SimpleNamespace(
        normal_neighbours=20, voxel_size=1.5, iterations=3, as_dict=lambda: dict(original=True)
    )

    def original(source, target, **kwargs):
        calls.append((source, target, kwargs))
        return result

    monkeypatch.setattr(module, "ORIGINAL_ICP", original)
    monkeypatch.setattr(tracking, "icp_point_to_plane", original)
    experiment = module.NormalTaperExperiment(profile)
    source, target = points(), points()[:90]
    args = arguments()
    with experiment.active(), icp_plane.normal_acceleration(lambda p, k: None):
        actual = tracking.icp_point_to_plane(source, target, **args)
    assert actual is result and calls[0][0] is source and calls[0][1] is target
    assert calls[0][2] == args | dict(normal_neighbours=20)
    assert experiment.records[0]["status"] == "completed"
    assert not module._ACTIVE.get() and experiment._original is None
    assert tracking.icp_point_to_plane is original
    assert module.GpuNormals.estimate_normals is module.ORIGINAL_ESTIMATE


@pytest.mark.parametrize("profile", ["observe-20", "taper-20"])
def test_real_icp_matches_independently_installed_normal_provider(module, profile):
    gpu = cpu_gpu(module)
    spin = rotation_about_axis((1.0, 2.0, -1.0), 0.015)
    transform = Rigid.of(spin.rotation, (0.2, -0.1, 0.03))
    target = tuple(transform.apply(p) for p in points())

    def reference(cloud, count):
        assert count == 20
        if profile == "observe-20":
            return icp_plane.estimate_normals(cloud, neighbours=count, backend="scipy")
        return module.tapered_field(cloud, gpu.jacobi)[0]

    with icp_plane.normal_acceleration(reference):
        expected = module.ORIGINAL_ICP(points(), target, normal_neighbours=20, **arguments())
    experiment = module.NormalTaperExperiment(profile)
    with (
        experiment.active(),
        icp_plane.normal_acceleration(
            lambda p, k: gpu.estimate_normals(p, neighbours=k, backend="scipy")
        ),
    ):
        actual = tracking.icp_point_to_plane(points(), target, **arguments())
    assert actual == expected and actual.converged
    assert experiment.records[0]["result"] == actual.as_dict()
    assert experiment.field_records and all(
        r["status"] == "completed" for r in experiment.field_records
    )


@pytest.mark.parametrize("failure", ["bad-count", "bad-backend", "no-dispatch", "nan"])
def test_field_failure_is_fatal_not_a_silent_cpu_fallback(module, failure):
    gpu = cpu_gpu(module)
    experiment = module.NormalTaperExperiment("taper-20")
    if failure == "no-dispatch":
        gpu.jacobi = lambda matrices: [icp_plane._jacobi(m) for m in matrices]
    elif failure == "nan":
        gpu.jacobi = lambda matrices: [([float("nan")] * 3, [[0.0] * 3] * 3) for m in matrices]
    fallback = []
    with pytest.raises(module.TaperDispatchFailed), experiment.active():
        try:
            gpu.estimate_normals(
                points(),
                neighbours=10 if failure == "bad-count" else 20,
                backend="python" if failure == "bad-backend" else "scipy",
            )
        except Exception:
            fallback.append("not allowed")
    assert not fallback and experiment.field_records[0]["status"] == "failed"
    assert (
        not module._ACTIVE.get() and module.GpuNormals.estimate_normals is module.ORIGINAL_ESTIMATE
    )


@pytest.mark.parametrize(
    "failure",
    [
        "inactive",
        "nested",
        "fit-bound",
        "field-bound",
        "no-provider",
        "wrong-neighbors",
        "wrong-voxel",
        "wrong-keep",
        "wrong-iterations",
    ],
)
def test_scope_and_original_fit_constraints(module, failure):
    experiment = module.NormalTaperExperiment("taper-20")
    if failure == "inactive":
        with pytest.raises(RuntimeError, match="active"):
            experiment.fit(points(), points(), **arguments())
        return
    with experiment.active():
        if failure == "nested":
            with pytest.raises(RuntimeError), module.NormalTaperExperiment("observe-20").active():
                pass
            return
        if failure == "field-bound":
            experiment.field_records = [{}] * 90000
            with pytest.raises(RuntimeError, match="90000"):
                cpu_gpu(module).estimate_normals(points(), neighbours=20, backend="scipy")
            return
        if failure == "no-provider":
            with pytest.raises(RuntimeError, match="GPU provider"):
                experiment.fit(points(), points(), **arguments())
            return
        if failure == "fit-bound":
            experiment.records = [{}] * 90000
        overrides = {
            "wrong-neighbors": dict(normal_neighbours=20),
            "wrong-voxel": dict(voxel_size=3.0),
            "wrong-keep": dict(keep_fraction=1.0),
            "wrong-iterations": dict(max_iterations=100),
        }
        with icp_plane.normal_acceleration(lambda p, k: None), pytest.raises(RuntimeError):
            experiment.fit(points(), points(), **(arguments() | overrides.get(failure, {})))
    assert not module._ACTIVE.get()


@pytest.mark.parametrize("profile", ["observe-20", "taper-20"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
def test_policy_and_normal_kernel_are_the_only_declared_differences(module, profile, capacity):
    count = 900 if capacity == "ordinary-900" else 1800
    target, policy, source = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), "centered"
    )
    assert target == count and source == ("observe-900" if count == 900 else "source-1800")
    assert policy.as_dict() == DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=count, retained_cloud_points=count
    )


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_records_actual_fields_and_preserves_reports(
    module, monkeypatch, tmp_path, capacity, static, failure
):
    adapter = importlib.import_module("scanner_normal_taper_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    gpu = cpu_gpu(module)
    output = tmp_path / "output"
    content = b'{"unchanged": true}'
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "taper-20", factors, "centered")

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["skip_complete"] == 12
            assert kwargs["depth_band"] == (60, 300) and kwargs["diagnostic_scene"]
        with icp_plane.normal_acceleration(
            lambda p, k: gpu.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            result = tracking.icp_point_to_plane(points(), points(), **arguments())
            assert result.converged
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if not static:
            (output / "stride-1.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    kwargs = dict(
        normal_profile="taper-20",
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320,
        skip_complete=0 if static else 12,
        depth_band=None if static else (60, 300),
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected adapter failure"):
            adapter.run(tmp_path / "capture", output, factors, **kwargs)
    else:
        adapter.run(tmp_path / "capture", output, factors, **kwargs)
    report = json.loads((output / "normal-taper-report.json").read_bytes())
    assert report["status"] == ("failed" if failure else "completed")
    assert (
        report["actual_normal_neighbours"] == 20
        and report["normal_kernel"] == "squared-distance-taper"
    )
    assert report["field_records"][0]["jacobi_dispatches"] == 1
    assert report["fit_records"][0]["result"]["normal_neighbours"] == 20
    names = ("report.json",) if static else ("report.json", "stride-1.json")
    assert report["underlying_reports"] == [
        dict(path=str(output / name), sha256=hashlib.sha256(content).hexdigest()) for name in names
    ]
    assert (
        report["tracking_policy"] == expected[1].as_dict()
        and report["preparation_target_points"] == expected[0]
    )
    assert (
        replay.IncrementalTracker is factory
        and not adapter._ACTIVE.get()
        and not module._ACTIVE.get()
    )
    assert all((output / name).read_bytes() == content for name in names)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-20", "taper-20"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_actual_tracker_and_solver_contexts_match_explicit_provider(
    module, capacity, profile, mode
):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    cloud = tuple(corner())
    _, policy, _ = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), mode
    )
    gpu = cpu_gpu(module)

    def provider(cloud, count):
        assert count == 20
        if profile == "observe-20":
            return icp_plane.estimate_normals(cloud, neighbours=count, backend="scipy")
        return module.tapered_field(cloud, gpu.jacobi)[0]

    def explicit(source, target, **kwargs):
        return module.ORIGINAL_ICP(source, target, **(kwargs | dict(normal_neighbours=20)))

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    expected_solver = solvers.SolverExperiment(mode)
    with (
        icp_plane.normal_acceleration(provider),
        expected_solver.active(),
        patch.object(tracking, "icp_point_to_plane", explicit),
    ):
        expected = [expected_tracker.offer(cloud).as_dict() for _ in range(4)]
    tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    actual_solver = solvers.SolverExperiment(mode)
    experiment = module.NormalTaperExperiment(profile)
    with (
        experiment.active(),
        actual_solver.active(),
        icp_plane.normal_acceleration(
            lambda p, k: gpu.estimate_normals(p, neighbours=k, backend="scipy")
        ),
    ):
        actual = [tracker.offer(cloud).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert actual_solver.records == expected_solver.records
    assert experiment.records and all(r["status"] == "completed" for r in experiment.records)
    assert experiment.field_records and all(
        r["status"] == "completed" for r in experiment.field_records
    )


def test_cli_requires_and_forwards_explicit_normal_taper(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_normal_taper_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--configuration",
        "grid-pooled-residual",
        "--capacity",
        "ordinary-900",
        "--mechanism",
        "centered",
        "--static",
        "--frames",
        "320",
    ]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--normal-taper", "taper-20"])
    adapter.main()
    assert calls[0][1]["normal_profile"] == "taper-20"


@pytest.mark.parametrize(
    "changes",
    [
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False),
        dict(static=False, frames=99, skip_complete=12, depth_band=(60, 300)),
        dict(static=False, frames=3001, skip_complete=12, depth_band=(60, 300)),
        dict(static=False, skip_complete=12, depth_band=(61, 300)),
        dict(normal_profile="taper-10"),
        dict(capacity_profile="unknown"),
    ],
)
def test_invalid_adapter_roster_does_not_create_outputs(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_normal_taper_replay")
    kwargs = dict(
        normal_profile="taper-20",
        capacity_profile="ordinary-900",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError):
        adapter.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **(kwargs | changes),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "base", ["stride-pooled-residual", "grid-native-residual", "grid-pooled-settling"]
)
def test_unrelated_factor_changes_are_rejected(module, base):
    with pytest.raises(ValueError):
        module.settings("ordinary-900", "taper-20", module.Factors(*base.split("-")), "centered")


@pytest.mark.parametrize("isotropic", [False, True])
def test_positive_support_and_original_isotropy_rule(module, isotropic):
    cloud = tuple(tuple(p) for p in np.random.default_rng(74).normal(size=(6, 3)))
    eigenvalues = [1.0, 1.0, 1.0] if isotropic else [0.0, 1.0, 2.0]
    field, summary = module.tapered_field(
        cloud, lambda matrices: [(eigenvalues, np.eye(3).tolist()) for _ in matrices]
    )
    assert all(e.neighbours == 4 for e in field.estimates)
    assert len(field.usable) == (0 if isotropic else 6)
    assert summary["classifications"] == dict(
        sparse=0, coincident=0, isotropic=6 if isotropic else 0
    )


def test_equal_radius_shell_is_honestly_sparse_after_taper(module):
    shell = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    field, stats = module.tapered_field([(0.0, 0.0, 0.0), *shell], cpu_gpu(module).jacobi)
    assert field.estimates[0].normal is None and field.estimates[0].neighbours == 0
    assert stats["classifications"]["sparse"] >= 1
