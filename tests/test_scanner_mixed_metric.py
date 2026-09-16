"""M23 independent derivatives/least-squares, guard preservation and true reports."""

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
    return importlib.import_module("scanner_mixed_metric")


@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_stacked_least_squares_and_frozen_costs(module, mode):
    pairs, _ = fixture_pairs()
    current = Rigid.of(rotation_about_axis((0.2, -0.7, 0.1), 0.012).rotation, (0.1, -0.3, 0.2))
    actual, data = module.mixed_step(pairs, current, mode)
    p = moved(current, [p for p, _, _ in pairs])
    centre = p.mean(axis=0) if mode == "centered" else np.zeros(3)
    rows, residuals = [], []
    for point, (_, q, n) in zip(p, pairs, strict=True):
        relative, normal = point - centre, np.array(n)
        rows.append(np.r_[np.cross(relative, normal), normal])
        residuals.append((point - q) @ normal)
        for axis in np.eye(3):
            rows.append(np.sqrt(0.1) * np.r_[np.cross(relative, axis), axis])
            residuals.append(np.sqrt(0.1) * ((point - q) @ axis))
    delta = np.linalg.lstsq(np.array(rows), -np.array(residuals), rcond=None)[0]
    assert np.allclose(data["delta"], delta, rtol=1e-8, atol=1e-8)
    for key, pose in (("costs_before", current), ("costs_after", actual)):
        displacement = moved(pose, [p for p, _, _ in pairs]) - np.array([q for _, q, _ in pairs])
        plane = np.mean(
            [float(d @ n) ** 2 for d, (_, _, n) in zip(displacement, pairs, strict=True)]
        )
        point = np.mean([float(d @ d) for d in displacement])
        assert data[key] == pytest.approx(
            dict(plane_mse=plane, point_mse=point, mixed_mse=plane + 0.1 * point)
        )
    assert data["original_plane_information_guard_passed"] and data["frozen_pairs_for_both_costs"]


def test_point_jacobian_matches_finite_difference_rodrigues(module):
    points = np.array([[1.2, -3.4, 200.0], [-12.0, 24.0, -36.0]])
    jacobian = module.point_jacobian(points)
    eps = 1e-6
    for coordinate in range(6):
        axis = np.eye(3)[coordinate % 3]
        if coordinate < 3:
            high = moved(rotation_about_axis(tuple(axis), eps), points)
            low = moved(rotation_about_axis(tuple(axis), -eps), points)
        else:
            high, low = points + eps * axis, points - eps * axis
        assert np.allclose(jacobian[:, :, coordinate], (high - low) / (2 * eps), atol=1e-7)


@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_known_transform_and_normal_signs(module, mode):
    pairs, truth = fixture_pairs()
    pose = Rigid.identity()
    for _ in range(10):
        pose, _ = module.mixed_step(pairs, pose, mode)
    points = [p for p, _, _ in pairs]
    assert np.max(np.linalg.norm(moved(pose, points) - moved(truth, points), axis=1)) < 1e-8
    flipped = [
        (p, q, tuple((-1 if i % 2 else 1) * v for v in n)) for i, (p, q, n) in enumerate(pairs)
    ]
    assert module.mixed_step(pairs, Rigid.identity(), mode) == module.mixed_step(
        flipped, Rigid.identity(), mode
    )


def test_centered_coordinate_equivariance(module):
    pairs, _ = fixture_pairs()
    frame = Rigid.of(
        rotation_about_axis((1.0, 2.0, -1.0), 0.4).rotation, (10000.0, -70000.0, 23000.0)
    )
    axis = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    changed = [(frame.apply(p), frame.apply(q), axis.apply(n)) for p, q, n in pairs]
    a, _ = module.mixed_step(pairs, Rigid.identity(), "centered")
    b, _ = module.mixed_step(changed, Rigid.identity(), "centered")
    expected = frame.compose(a).compose(frame.inverse())
    points = [p for p, _, _ in changed]
    assert np.max(np.linalg.norm(moved(b, points) - moved(expected, points), axis=1)) < 1e-7


@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_original_plane_singularity_not_filled_by_point_term(module, mode):
    pairs = [
        ((float(x), float(y), 100.0), (float(x), float(y), 100.1), (0.0, 0.0, 1.0))
        for x in range(4)
        for y in range(4)
    ]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        module.mixed_step(pairs, Rigid.identity(), mode)


@pytest.mark.parametrize(
    "fault", ["empty", "few", "oversized", "nan", "zero-normal", "shape", "mode"]
)
def test_invalid_step_refuses(module, fault):
    pairs, _ = fixture_pairs()
    if fault == "empty":
        pairs = []
    elif fault == "few":
        pairs = pairs[:5]
    elif fault == "oversized":
        pairs = [pairs[0]] * 2049
    elif fault == "nan":
        pairs[0] = ((float("nan"), 0.0, 0.0), pairs[0][1], pairs[0][2])
    elif fault == "zero-normal":
        pairs[0] = (pairs[0][0], pairs[0][1], (0.0, 0.0, 0.0))
    elif fault == "shape":
        pairs = [(1.0, 2.0, 3.0)] * 6
    with pytest.raises(ValueError):
        module.mixed_step(pairs, Rigid.identity(), "unknown" if fault == "mode" else "centered")


def test_step_scope_records_failures_and_bounds(module):
    pairs, _ = fixture_pairs()
    ex = module.MixedMetricExperiment("plane-plus-point-0.1", "centered")
    with pytest.raises(RuntimeError):
        ex.step(pairs, Rigid.identity())
    ex.mark_frame(42)
    with ex.active():
        with pytest.raises(RuntimeError), ex.active():
            pass
        ex.step(pairs, Rigid.identity())
        with pytest.raises(ValueError):
            ex.step([], Rigid.identity())
        assert [r["status"] for r in ex.records] == ["completed", "failed"]
        assert all(r["source_index"] == 42 for r in ex.records)
        ex.records = [None] * module.MAX_RECORDED_STEPS
        with pytest.raises(RuntimeError, match="bound"):
            ex.step(pairs, Rigid.identity())
    assert icp_plane._solve_point_to_plane is module.ORIGINAL_STEP and not module._ACTIVE.get()
    for index in (-1, 3000):
        with pytest.raises(ValueError):
            ex.mark_frame(index)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_real_tracker_matches_explicit_changed_step(module, capacity, mode):
    from tests.test_scanner_registration_audit import corner

    points = tuple(corner())
    factors = module.Factors("grid", "pooled", "residual")
    _, policy, _ = module.settings(capacity, "plane-plus-point-0.1", factors, mode)

    def cpu(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    def explicit(matched, current, **kwargs):
        return module.mixed_step(matched, current, mode)[0]

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    with (
        icp_plane.normal_acceleration(cpu),
        patch.object(icp_plane, "_solve_point_to_plane", explicit),
    ):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    ex = module.MixedMetricExperiment("plane-plus-point-0.1", mode)
    with icp_plane.normal_acceleration(cpu), ex.active():
        actual = [tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert ex.records and all(r["status"] == "completed" for r in ex.records)


@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
@pytest.mark.parametrize("angle", [0.0, 0.04])
def test_actual_iterator_reports_original_plane_rmse(module, mode, angle):
    from tests.test_scanner_registration_audit import corner

    target = tuple(corner())
    truth = Rigid.of(
        rotation_about_axis((0.3, 0.5, 0.7), angle).rotation, (angle * 10, -angle * 4, angle * 3)
    )
    source = tuple(truth.inverse().apply_all(target))
    options = dict(
        backend="scipy",
        initial=Rigid.identity(),
        voxel_size=1.5,
        keep_fraction=0.8,
        tolerance=0.001,
        max_correspondence_distance=12.0,
    )
    ex = module.MixedMetricExperiment("plane-plus-point-0.1", mode)
    actual_matches = []

    def observed(matched, current, **kwargs):
        actual_matches.append(matched)
        return ex.step(matched, current, **kwargs)

    with ex.active(), patch.object(icp_plane, "_solve_point_to_plane", observed):
        actual = icp_plane.icp_point_to_plane(source, target, **options)
    residuals = [(np.array(actual.transform.apply(p)) - q) @ n for p, q, n in actual_matches[-1]]
    assert len(actual_matches) == actual.iterations
    assert actual.rmse == pytest.approx(float(np.sqrt(np.mean(np.square(residuals)))), abs=1e-12)
    assert actual.converged and actual.normal_neighbours == 10


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("profile", ["observe-plane", "plane-plus-point-0.1"])
def test_adapter_actual_factory_object_and_preserved_reports(
    module, monkeypatch, tmp_path, capacity, static, profile
):
    adapter = importlib.import_module("scanner_mixed_metric_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    output, factors = tmp_path / "output", module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, profile, factors, "centered")
    pairs, _ = fixture_pairs()
    content = b'{"original": true}'
    original_objects = []
    original_factory = adapter.ORIGINAL_FACTORY

    def factory(mode):
        value = original_factory(mode)
        original_objects.append(value)
        return value

    monkeypatch.setattr(adapter, "ORIGINAL_FACTORY", factory)
    monkeypatch.setattr(adapter.scanner_solver_registry, "create_experiment", factory)

    def delegated(capture, destination, *args, **kwargs):
        assert destination == output and kwargs["diagnostic_source_profile"] == expected[2]
        value = adapter.scanner_solver_registry.create_experiment("centered")
        if profile == "observe-plane":
            assert value is original_objects[0]
        else:
            assert not original_objects and isinstance(value, module.MixedMetricExperiment)
        with value.active():
            value.step(pairs, Rigid.identity())
        output.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (output / name).write_bytes(content)

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    adapter.run(
        tmp_path / "capture",
        output,
        factors,
        metric_profile=profile,
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320,
    )
    report = json.loads((output / "mixed-metric-report.json").read_bytes())
    assert report["status"] == "completed" and report["solver_factory_calls"] == 1
    assert report["point_weight"] == (0.0 if profile == "observe-plane" else 0.1)
    assert report["tracking_policy"] == expected[1].as_dict() and report["step_records"]
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in report["underlying_reports"]
    )
    assert len(report["underlying_reports"]) == (1 if static else 2)
    assert adapter.scanner_solver_registry.create_experiment is factory
    assert not adapter._ACTIVE.get() and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "failure",
    ["before-factory", "after-step", "twice", "wrong-profile", "no-records", "missing-report"],
)
def test_adapter_failures_restore_and_retain_error(module, monkeypatch, tmp_path, failure):
    adapter = importlib.import_module("scanner_mixed_metric_replay")
    original, output = adapter.scanner_solver_registry.create_experiment, tmp_path / "output"

    def delegated(*args, **kwargs):
        output.mkdir()
        if failure == "before-factory":
            raise RuntimeError("injected before factory")
        ex = adapter.scanner_solver_registry.create_experiment(
            "observe-origin" if failure == "wrong-profile" else "centered"
        )
        if failure == "twice":
            adapter.scanner_solver_registry.create_experiment("centered")
        if failure != "no-records":
            with ex.active():
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
            metric_profile="plane-plus-point-0.1",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=True,
            frames=320,
        )
    report = json.loads((output / "mixed-metric-report.json").read_bytes())
    assert report["status"] == "failed" and report["error"]
    assert adapter.scanner_solver_registry.create_experiment is original
    assert not adapter._ACTIVE.get() and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "changes",
    [
        dict(capacity_profile="unknown"),
        dict(metric_profile="plane-plus-point-1.0"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False, frames=99),
        dict(static=False, frames=3001),
    ],
)
def test_nonclosed_adapter_inputs_create_nothing(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_mixed_metric_replay")
    options = dict(
        capacity_profile="dense-1800",
        metric_profile="plane-plus-point-0.1",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError):
        adapter.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **(options | changes),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "base", ["stride-pooled-residual", "grid-native-residual", "grid-pooled-settling"]
)
def test_other_factors_rejected(module, base):
    with pytest.raises(ValueError):
        module.settings(
            "dense-1800", "plane-plus-point-0.1", module.Factors(*base.split("-")), "centered"
        )


@pytest.mark.parametrize(
    "profile,mode",
    [("observe-plane", "centered"), ("unknown", "centered"), ("plane-plus-point-0.1", "unknown")],
)
def test_changed_context_roster_closed(module, profile, mode):
    with pytest.raises(ValueError):
        module.MixedMetricExperiment(profile, mode)


def test_cli_requires_explicit_metric(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_mixed_metric_replay")
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
        "dense-1800",
        "--mechanism",
        "centered",
        "--frames",
        "628",
        "--skip-complete",
        "12",
        "--depth-band",
        "60",
        "300",
    ]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--metric", "plane-plus-point-0.1"])
    adapter.main()
    assert calls[0][1]["metric_profile"] == "plane-plus-point-0.1"
    assert calls[0][1]["depth_band"] == (60.0, 300.0)
