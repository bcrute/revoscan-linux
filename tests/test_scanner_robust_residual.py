"""M20 robust weights, finite equations and unchanged external fit semantics."""

import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid, rotation_about_axis
from tests.test_scanner_solver_mechanisms import fixture_pairs, moved


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_robust_residual")


@pytest.mark.parametrize("profile", ["observe-l2", "huber-scale", "cauchy-scale"])
def test_weights_follow_declared_equations_and_signs(module, profile):
    residuals = np.array([0.0, 0.005, -0.01, 0.02, -0.03, 5.0])
    weights, data = module.robust_weights(residuals, profile)
    scale = max(module.DEPTH_SCALE_METRES, 0.015 / module.NORMAL_ABSOLUTE_MEDIAN)
    absolute = np.abs(residuals)
    raw = (
        np.ones(6)
        if profile == "observe-l2"
        else np.array(
            [
                min(1.0, scale / value)
                if profile == "huber-scale" and value
                else 1.0 / (1.0 + (value / scale) ** 2)
                for value in absolute
            ]
        )
    )
    assert data["residual_scale"] == pytest.approx(scale)
    assert np.allclose(weights, raw * 6 / raw.sum(), atol=1e-15, rtol=1e-15)
    flipped, other = module.robust_weights(-residuals, profile)
    assert np.array_equal(weights, flipped) and data == other
    assert data["weight_sum"] == pytest.approx(6.0)
    assert 0 < data["effective_weighted_count"] <= 6
    assert data["zero_weight"] == 0


@pytest.mark.parametrize("profile", ["observe-l2", "huber-scale", "cauchy-scale"])
def test_zero_residuals_use_raw_code_floor_and_uniform_weights(module, profile):
    weights, data = module.robust_weights(np.zeros(6), profile)
    assert weights.tolist() == [1.0] * 6
    assert data["scale_floor_active"] and data["residual_scale"] == module.DEPTH_SCALE_METRES
    assert module.robust_cost(np.zeros(6), data["residual_scale"], profile) == 0


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_known_transform_and_arbitrary_normal_signs(module, profile, mode):
    pairs, truth = fixture_pairs()
    pose = Rigid.identity()
    for _ in range(10):
        pose, data = module.robust_step(pairs, pose, profile, mode)
    points = [p for p, _, _ in pairs]
    assert np.max(np.linalg.norm(moved(pose, points) - moved(truth, points), axis=1)) < 1e-8
    flipped = [
        (p, q, tuple((-1 if i % 2 else 1) * v for v in n)) for i, (p, q, n) in enumerate(pairs)
    ]
    a, ad = module.robust_step(pairs, Rigid.identity(), profile, mode)
    b, bd = module.robust_step(flipped, Rigid.identity(), profile, mode)
    assert a == b and ad == bd
    assert data["frozen_scale_for_both_costs"]


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_weighted_step_matches_independent_dense_equations(module, profile, mode):
    pairs, _ = fixture_pairs()
    current = rotation_about_axis((0.2, -0.7, 0.1), 0.012)
    actual, data = module.robust_step(pairs, current, profile, mode)
    values = np.asarray(pairs)
    p = moved(current, [p for p, _, _ in pairs])
    centre = p.mean(axis=0) if mode == "centered" else np.zeros(3)
    jacobian = np.c_[np.cross(p - centre, values[:, 2]), values[:, 2]]
    residuals = np.sum((p - values[:, 1]) * values[:, 2], axis=1)
    scale = max(module.DEPTH_SCALE_METRES, np.median(np.abs(residuals)) / 0.6744897501960817)
    w = (
        1 / np.maximum(1, np.abs(residuals) / scale)
        if profile == "huber-scale"
        else 1 / (1 + (residuals / scale) ** 2)
    )
    delta = np.linalg.solve(jacobian.T @ (w[:, None] * jacobian), -jacobian.T @ (w * residuals))
    rot = rotation_about_axis(tuple(delta[:3]), np.linalg.norm(delta[:3])).rotation
    translation = delta[3:] + centre - np.asarray(rot) @ centre
    expected = Rigid.of(rot, tuple(translation)).compose(current)
    assert np.allclose(data["delta"], delta, rtol=1e-8, atol=1e-8)
    points = [p for p, _, _ in pairs]
    assert np.max(np.linalg.norm(moved(actual, points) - moved(expected, points), axis=1)) < 1e-8


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_sparse_deliberate_outliers_have_less_influence(module, profile, mode):
    pairs, _ = fixture_pairs()
    changed = [
        (q, tuple(np.asarray(q) + (6 * np.asarray(n) if i < 12 else 0)), n)
        for i, (_, q, n) in enumerate(pairs)
    ]
    ordinary, _ = module.robust_step(changed, Rigid.identity(), "observe-l2", mode)
    robust, _ = module.robust_step(changed, Rigid.identity(), profile, mode)
    points = [p for p, _, _ in changed]
    ordinary_error = np.mean(np.linalg.norm(moved(ordinary, points) - np.asarray(points), axis=1))
    robust_error = np.mean(np.linalg.norm(moved(robust, points) - np.asarray(points), axis=1))
    assert robust_error < ordinary_error / 100


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
def test_centered_coordinate_equivariance(module, profile):
    pairs, _ = fixture_pairs()
    frame = Rigid.of(
        rotation_about_axis((1.0, 2.0, -1.0), 0.4).rotation, (10000.0, -70000.0, 23000.0)
    )
    axis = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    changed = [(frame.apply(p), frame.apply(q), axis.apply(n)) for p, q, n in pairs]
    a, _ = module.robust_step(pairs, Rigid.identity(), profile, "centered")
    b, _ = module.robust_step(changed, Rigid.identity(), profile, "centered")
    expected = frame.compose(a).compose(frame.inverse())
    points = [p for p, _, _ in changed]
    assert np.max(np.linalg.norm(moved(b, points) - moved(expected, points), axis=1)) < 1e-7


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_unconstrained_plane_remains_refused(module, profile, mode):
    pairs = [
        ((float(x), float(y), 8.0), (float(x), float(y), 9.0), (0.0, 0.0, 1.0))
        for x in range(5)
        for y in range(5)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        module.robust_step(pairs, Rigid.identity(), profile, mode)


@pytest.mark.parametrize(
    "bad", [[], [1.0] * 5, [1.0] * 2049, [[1.0] * 6], [float("nan")] * 6, [float("inf")] * 6]
)
def test_malformed_residuals_refused(module, bad):
    with pytest.raises(ValueError):
        module.robust_weights(bad, "huber-scale")


@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_iterator_reports_original_unweighted_residual_and_restores(module, profile, mode):
    pairs, _ = fixture_pairs()
    experiment = module.RobustResidualExperiment(profile, mode)
    original_step, original_iterate = icp_plane._solve_point_to_plane, icp_plane._iterate
    experiment.mark_frame(14)
    with experiment.active():
        result = icp_plane._iterate(
            tuple(p for p, _, _ in pairs),
            lambda *_a: ([(i / 1000, p, q, n) for i, (p, q, n) in enumerate(pairs)], 0),
            initial=Rigid.identity(),
            max_iterations=50,
            threshold=0.0065,
            keep_fraction=0.8,
            gate=12.0,
            accelerated=True,
        )
    kept = pairs[: icp_plane.kept_count(len(pairs), 0.8)]
    expected = icp_plane._point_to_plane_rmse(
        [(result.transform.apply(p), q, n) for p, q, n in kept]
    )
    assert result.rmse == expected
    assert experiment.records and all(r["source_index"] == 14 for r in experiment.records)
    assert (
        icp_plane._solve_point_to_plane is original_step and icp_plane._iterate is original_iterate
    )
    assert not module._ACTIVE.get() and not experiment._active


@pytest.mark.parametrize("guard", ["inactive", "nested", "foreign-hook", "bound", "failure"])
def test_scope_work_bounds_and_exception_cleanup(module, monkeypatch, guard):
    experiment = module.RobustResidualExperiment("huber-scale", "centered")
    pairs, _ = fixture_pairs()
    original = icp_plane._solve_point_to_plane
    if guard == "inactive":
        with pytest.raises(RuntimeError):
            experiment.step(pairs, Rigid.identity())
    elif guard == "foreign-hook":
        monkeypatch.setattr(icp_plane, "_solve_point_to_plane", lambda *a, **k: None)
        with pytest.raises(RuntimeError), experiment.active():
            pass
        return
    elif guard == "nested":
        with experiment.active(), pytest.raises(RuntimeError), experiment.active():
            pass
    else:
        if guard == "bound":
            monkeypatch.setattr(module, "MAX_RECORDED_STEPS", 0)
        else:

            def fail(*a, **k):
                raise RuntimeError("injected")

            monkeypatch.setattr(module, "robust_step", fail)
        with pytest.raises(RuntimeError), experiment.active():
            icp_plane._solve_point_to_plane(pairs, Rigid.identity())
        if guard == "failure":
            assert experiment.records[0]["status"] == "failed"
    assert icp_plane._solve_point_to_plane is original and not module._ACTIVE.get()


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_whole_tracker_equals_explicit_robust_step_composition(module, capacity, profile, mode):
    from tests.test_scanner_registration_audit import corner

    points = tuple(corner())
    _, policy, _ = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), mode
    )

    def provider(cloud, count):
        return icp_plane.estimate_normals(cloud, neighbours=count, backend="scipy")

    def explicit(matched, current, **kwargs):
        return module.robust_step(matched, current, profile, mode)[0]

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    with (
        icp_plane.normal_acceleration(provider),
        patch.object(icp_plane, "_solve_point_to_plane", explicit),
    ):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    actual_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    experiment = module.RobustResidualExperiment(profile, mode)
    with icp_plane.normal_acceleration(provider), experiment.active():
        actual = [actual_tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert experiment.records and all(r["status"] == "completed" for r in experiment.records)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-l2", "huber-scale", "cauchy-scale"])
@pytest.mark.parametrize("static", [False, True])
def test_adapter_factory_and_unchanged_report_bytes(
    module, monkeypatch, tmp_path, capacity, profile, static
):
    adapter = importlib.import_module("scanner_robust_residual_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = adapter.scanner_solver_registry.create_experiment
    original_step = icp_plane._solve_point_to_plane
    factors = module.Factors("grid", "pooled", "residual")
    output = tmp_path / "output"
    content = b'{"original_report":true}'
    pairs, _ = fixture_pairs()
    expected = module.settings(capacity, profile, factors, "centered")
    original_experiment = original_factory("centered")
    with original_experiment.active():
        expected_pose = icp_plane._solve_point_to_plane(pairs, Rigid.identity(), accelerated=True)
    original_experiment.records.clear()
    if profile == "observe-l2":

        def original(profile_name):
            assert profile_name == "centered"
            return original_experiment

        monkeypatch.setattr(adapter, "ORIGINAL_FACTORY", original)
        monkeypatch.setattr(adapter.scanner_solver_registry, "create_experiment", original)
        original_factory = original

    def delegated(capture, destination, *args, **kwargs):
        assert destination == output and capture == tmp_path / "capture"
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        experiment = adapter.scanner_solver_registry.create_experiment("centered")
        if profile == "observe-l2":
            assert experiment is original_experiment
        experiment.mark_frame(2)
        with experiment.active():
            actual = icp_plane._solve_point_to_plane(pairs, Rigid.identity(), accelerated=True)
        if profile == "observe-l2":
            assert actual == expected_pose
        output.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (output / name).write_bytes(content)

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    adapter.run(
        tmp_path / "capture",
        output,
        factors,
        kernel_profile=profile,
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320,
    )
    report = json.loads((output / "robust-residual-report.json").read_bytes())
    assert report["status"] == "completed" and report["solver_factory_calls"] == 1
    assert report["step_records"][0]["source_index"] == 2
    assert report["tracking_policy"] == expected[1].as_dict()
    assert report["preparation_target_points"] == expected[0]
    assert not report["quality_qualification_allowed"]
    assert len(report["underlying_reports"]) == (1 if static else 2)
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in report["underlying_reports"]
    )
    assert adapter.scanner_solver_registry.create_experiment is original_factory
    assert icp_plane._solve_point_to_plane is original_step and not adapter._ACTIVE.get()


@pytest.mark.parametrize(
    "failure",
    ["before-factory", "after-step", "twice", "wrong-profile", "no-records", "missing-report"],
)
def test_adapter_failures_restore_factory_and_preserve_failure(
    module, monkeypatch, tmp_path, failure
):
    adapter = importlib.import_module("scanner_robust_residual_replay")
    original = adapter.scanner_solver_registry.create_experiment
    output = tmp_path / "output"

    def delegated(*args, **kwargs):
        output.mkdir()
        if failure == "before-factory":
            raise RuntimeError("injected before factory")
        experiment = adapter.scanner_solver_registry.create_experiment(
            "observe-origin" if failure == "wrong-profile" else "centered"
        )
        if failure == "twice":
            adapter.scanner_solver_registry.create_experiment("centered")
        if failure != "no-records":
            with experiment.active():
                icp_plane._solve_point_to_plane(fixture_pairs()[0], Rigid.identity())
        if failure == "after-step":
            raise RuntimeError("injected after step")
        if failure != "missing-report":
            (output / "report.json").write_text("{}")

    monkeypatch.setattr(adapter.static_replay, "run_static", delegated)
    with pytest.raises(RuntimeError):
        adapter.run(
            tmp_path / "capture",
            output,
            module.Factors("grid", "pooled", "residual"),
            kernel_profile="huber-scale",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=True,
            frames=320,
        )
    report = json.loads((output / "robust-residual-report.json").read_bytes())
    assert report["status"] == "failed" and report["error"]
    assert adapter.scanner_solver_registry.create_experiment is original
    assert not adapter._ACTIVE.get() and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "changes",
    [
        dict(capacity_profile="unknown"),
        dict(kernel_profile="tukey"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False, frames=99),
        dict(static=False, frames=3001),
    ],
)
def test_nonclosed_adapter_inputs_never_create_output(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_robust_residual_replay")
    options = dict(
        capacity_profile="dense-1800",
        kernel_profile="huber-scale",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    options.update(changes)
    with pytest.raises(ValueError):
        adapter.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **options,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "profile,mode",
    [("observe-l2", "centered"), ("unknown", "centered"), ("huber-scale", "unknown")],
)
def test_changed_context_closed_roster(module, profile, mode):
    with pytest.raises(ValueError):
        module.RobustResidualExperiment(profile, mode)


@pytest.mark.parametrize("index", [-1, 3000])
def test_frame_bounds(module, index):
    experiment = module.RobustResidualExperiment("huber-scale", "centered")
    with pytest.raises(ValueError):
        experiment.mark_frame(index)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [(1, 2, 3)] * 6,
        [((0, 0, 0), (0, 0, 0), (0, 0, 0))] * 6,
        [((float("nan"), 0, 0), (0, 0, 0), (0, 0, 1))] * 6,
    ],
)
def test_malformed_triples(module, bad):
    with pytest.raises(ValueError):
        module.robust_step(bad, Rigid.identity(), "huber-scale", "centered")


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_frozen_cost_scale(module, scale):
    with pytest.raises(ValueError):
        module.robust_cost([1.0] * 6, scale, "cauchy-scale")
