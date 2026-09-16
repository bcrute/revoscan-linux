"""M22 exact policy/routing and scoped adapter evidence, without scanner access."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_model_capacity")


def experiment(module, profile="model-3600", capacity="dense-1800", solver="centered"):
    return module.ModelCapacityExperiment(
        capacity, profile, module.Factors("grid", "pooled", "residual"), solver
    )


def arguments():
    return dict(
        backend="scipy",
        initial=Rigid.identity(),
        voxel_size=1.5,
        keep_fraction=0.8,
        tolerance=0.001,
        max_correspondence_distance=12.0,
    )


def provider(points, neighbours):
    return icp_plane.NormalField(
        tuple(icp_plane.NormalEstimate(p, (0.0, 0.0, 1.0), neighbours) for p in points),
        neighbours,
        (),
    )


def fake_fit(a, b, **kwargs):
    icp_plane._NORMAL_ACCELERATOR.get()(tuple(b), 10)
    values = dict(voxel_size=1.5, normal_neighbours=10, iterations=2)
    return SimpleNamespace(**values, as_dict=lambda: values)


def install(module, monkeypatch, function=fake_fit):
    monkeypatch.setattr(module, "ORIGINAL_ICP", function)
    monkeypatch.setattr(tracking, "icp_point_to_plane", function)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-1800", "model-3600"])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_only_model_policy_changes_and_settings_are_scoped(module, capacity, profile, solver):
    ex = experiment(module, profile, capacity, solver)
    count = 900 if capacity == "ordinary-900" else 1800
    assert ex.policy.as_dict() == tracking.DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=count,
        retained_cloud_points=count,
        model_reference_points=1800 if profile == "observe-1800" else 3600,
    )
    with ex.active():
        actual = module.source_capacity.settings(ex.source_profile, ex.factors, solver, None)
        assert actual == (count, ex.policy) and ex.settings_calls == 1
        assert ex.cache_limit == icp_plane.NORMAL_CACHE_MAX_POINTS
    assert module.source_capacity.settings is module.ORIGINAL_SETTINGS
    assert icp_plane.NORMAL_CACHE_MAX_POINTS == 2048 and icp_plane.NORMAL_CACHE_ENTRIES == 32


@pytest.mark.parametrize("fault", ["bypass", "double", "field-points", "neighbors", "metadata"])
def test_provider_and_result_contract_cannot_be_bypassed(module, monkeypatch, fault):
    def original(a, b, **kwargs):
        if fault != "bypass":
            fake_fit(a, b, **kwargs)
        if fault == "double":
            fake_fit(a, b, **kwargs)
        return SimpleNamespace(
            voxel_size=1.5, normal_neighbours=10, iterations=51 if fault == "metadata" else 2
        )

    def field(p, k):
        return provider(
            p[::-1] if fault == "field-points" else p, 20 if fault == "neighbors" else k
        )

    install(module, monkeypatch, original)
    ex = experiment(module)
    points = [(float(i), 0.0, 100.0) for i in range(6)]
    with ex.active(), icp_plane.normal_acceleration(field), pytest.raises(RuntimeError):
        tracking.icp_point_to_plane(points, points, **arguments())
    assert ex.records[0]["status"] == "failed" and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "changes",
    [
        dict(normal_neighbours=20),
        dict(max_iterations=100),
        dict(voxel_size=3.0),
        dict(keep_fraction=0.9),
        dict(backend="python"),
        dict(initial=None),
    ],
)
def test_nonoriginal_fit_options_fail_closed(module, monkeypatch, changes):
    install(module, monkeypatch)
    ex = experiment(module)
    with ex.active(), icp_plane.normal_acceleration(provider), pytest.raises(RuntimeError):
        tracking.icp_point_to_plane([], [], **(arguments() | changes))
    assert not ex.records[0]["provider_calls"]


def test_nested_hooks_bounds_and_wrong_settings_rejected(module, monkeypatch):
    ex = experiment(module)
    with pytest.raises(RuntimeError, match="active"):
        ex.fit([], [], **arguments())
    with ex.active():
        with pytest.raises(RuntimeError, match="nested"), experiment(module).active():
            pass
        with pytest.raises(RuntimeError, match="policy"):
            module.source_capacity.settings("observe-900", ex.factors, ex.solver, None)
        with pytest.raises(RuntimeError, match="GPU provider"):
            tracking.icp_point_to_plane([], [], **arguments())
        ex.records = [None] * 90000
        with pytest.raises(RuntimeError, match="90000"):
            ex.fit([], [], **arguments())
    for name, value in (("NORMAL_CACHE_MAX_POINTS", 4096), ("NORMAL_CACHE_ENTRIES", 64)):
        with monkeypatch.context() as context:
            context.setattr(icp_plane, name, value)
            with pytest.raises(RuntimeError, match="nonoriginal"), experiment(module).active():
                pass


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-1800", "model-3600"])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_real_tracker_matches_explicit_policy(module, capacity, profile, solver):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    ex = experiment(module, profile, capacity, solver)
    points = tuple(corner())

    def cpu(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)
    expected_solver = solvers.SolverExperiment(solver)
    with (
        icp_plane.normal_acceleration(cpu),
        expected_solver.active(),
        patch.object(icp_plane, "NORMAL_CACHE_MAX_POINTS", ex.cache_limit),
    ):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    actual_tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)
    actual_solver = solvers.SolverExperiment(solver)
    with icp_plane.normal_acceleration(cpu), actual_solver.active(), ex.active():
        actual = [actual_tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert actual_solver.records == expected_solver.records
    assert ex.records and all(r["status"] == "completed" for r in ex.records)


@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_real_large_field_bypasses_provider_without_scoped_routing(module, solver):
    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    target = tuple(
        p
        for dx, dy in ((0, 0), (80, 0), (0, 80), (80, 80))
        for x, y, z in corner()
        for p in [(x + dx, y + dy, z)]
    )
    source = target[::3]
    calls = []

    def cpu(cloud, neighbours):
        calls.append(len(cloud))
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    with icp_plane.normal_acceleration(cpu), solvers.SolverExperiment(solver).active():
        expected = icp_plane.icp_point_to_plane(source, target, **arguments())
    assert not calls
    ex = experiment(module, solver=solver)
    with icp_plane.normal_acceleration(cpu), solvers.SolverExperiment(solver).active(), ex.active():
        actual = tracking.icp_point_to_plane(source, target, **arguments())
    assert actual == expected and len(calls) == 1 and calls[0] > 2048
    assert ex.records[0]["provider_calls"][0]["above_original_cache_limit"]
    assert icp_plane.NORMAL_CACHE_MAX_POINTS == 2048 and icp_plane.NORMAL_CACHE_ENTRIES == 32


@pytest.mark.parametrize("profile,count", [("observe-1800", 1800), ("model-3600", 3600)])
def test_actual_result_identity_and_field_route(module, monkeypatch, profile, count):
    returned, calls = [], []

    def original(a, b, **kwargs):
        calls.append((a, b, kwargs))
        returned.append(fake_fit(a, b, **kwargs))
        return returned[-1]

    install(module, monkeypatch, original)
    source = [(float(i), 0.0, 100.0) for i in range(6)]
    target = [(float(i), 1.0, 100.0) for i in range(count)]
    options = arguments()
    ex = experiment(module, profile)
    with ex.active(), icp_plane.normal_acceleration(provider):
        before = icp_plane._NORMAL_ACCELERATOR.get()
        actual = tracking.icp_point_to_plane(source, target, **options)
        assert icp_plane._NORMAL_ACCELERATOR.get() is before
    assert actual is returned[0] and len(calls) == 1
    assert calls[0][0] is source and calls[0][1] is target and calls[0][2] == options
    record = ex.records[0]
    assert record["status"] == "completed" and record["result"] == actual.as_dict()
    call = record["provider_calls"][0]
    assert call["points"] == count and call["point_identity_exact"]
    assert call["above_original_cache_limit"] == (count > 2048)
    assert (
        call["points_sha256"]
        == hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest()
    )
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()


@pytest.mark.parametrize("at_provider", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("data"),
        ArithmeticError("math"),
        icp_plane.IcpPlaneCorrespondencesInsufficient("few"),
        RuntimeError("internal"),
    ],
)
def test_failure_retained_and_scopes_restored(module, monkeypatch, at_provider, error):
    def fail(*args, **kwargs):
        raise error

    install(module, monkeypatch, fake_fit if at_provider else fail)
    ex = experiment(module)
    points = [(float(i), 0.0, 100.0) for i in range(6)]
    with icp_plane.normal_acceleration(fail if at_provider else provider):
        original_provider = icp_plane._NORMAL_ACCELERATOR.get()
        with pytest.raises(type(error), match=str(error)), ex.active():
            tracking.icp_point_to_plane(points, points, **arguments())
        assert icp_plane._NORMAL_ACCELERATOR.get() is original_provider
    record = ex.records[0]
    assert record["status"] == ("failed" if isinstance(error, RuntimeError) else "refused")
    assert str(error) in record["error"]
    assert len(record["provider_calls"]) == int(at_provider)
    assert all(c["status"] == record["status"] for c in record["provider_calls"])
    assert icp_plane.NORMAL_CACHE_MAX_POINTS == 2048 and not module._ACTIVE.get()
    assert module.source_capacity.settings is module.ORIGINAL_SETTINGS


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_reports_and_exact_actual_policy(
    module, monkeypatch, tmp_path, capacity, static, failure
):
    adapter = importlib.import_module("scanner_model_capacity_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    output = tmp_path / "output"
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "model-3600", factors, "centered")
    content = b'{"original": true}'
    install(module, monkeypatch)

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        actual_policy = module.source_capacity.settings(expected[2], factors, "centered", None)
        assert actual_policy == expected[:2] and icp_plane.NORMAL_CACHE_MAX_POINTS == 4096
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        points = [(float(i), 0.0, 100.0) for i in range(6)]
        with icp_plane.normal_acceleration(provider):
            tracking.icp_point_to_plane(points, points, **arguments())
        output.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (output / name).write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    options = dict(
        budget_profile="model-3600",
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320,
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            adapter.run(tmp_path / "capture", output, factors, **options)
    else:
        adapter.run(tmp_path / "capture", output, factors, **options)
    metadata = json.loads((output / "model-capacity-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["settings_calls"] == 1 and metadata["normal_cache_max_points"] == 4096
    assert metadata["tracking_policy"] == expected[1].as_dict()
    assert metadata["preparation_target_points"] == expected[0]
    assert not metadata["quality_qualification_allowed"]
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in metadata["underlying_reports"]
    )
    assert len(metadata["underlying_reports"]) == (1 if static else 2)
    assert replay.IncrementalTracker is factory and not adapter._ACTIVE.get()
    assert not module._ACTIVE.get() and icp_plane.NORMAL_CACHE_MAX_POINTS == 2048


@pytest.mark.parametrize(
    "changes",
    [
        dict(capacity_profile="unknown"),
        dict(budget_profile="model-7200"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False, frames=99),
        dict(static=False, frames=3001),
    ],
)
def test_adapter_nonclosed_inputs_create_nothing(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_model_capacity_replay")
    options = dict(
        capacity_profile="dense-1800",
        budget_profile="model-3600",
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
        module.settings("dense-1800", "model-3600", module.Factors(*base.split("-")), "centered")


def test_cli_requires_explicit_budget(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_model_capacity_replay")
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
    monkeypatch.setattr("sys.argv", [*args, "--model-capacity", "model-3600"])
    adapter.main()
    assert calls[0][1]["budget_profile"] == "model-3600"
    assert calls[0][1]["depth_band"] == (60.0, 300.0)


def test_no_output_reuse_or_patched_tracker_factory(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_model_capacity_replay")
    options = dict(
        capacity_profile="dense-1800",
        budget_profile="model-3600",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError, match="new output"):
        adapter.run(
            tmp_path / "capture", tmp_path, module.Factors("grid", "pooled", "residual"), **options
        )
    monkeypatch.setattr(adapter.static_replay, "IncrementalTracker", lambda **k: None)
    with pytest.raises(RuntimeError, match="factory"):
        adapter.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **options,
        )
    assert not (tmp_path / "output").exists()
