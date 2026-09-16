"""M18 level/seed/metadata identity and explicit failures, no scanner needed."""

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
    return importlib.import_module("scanner_multiscale")


def install(module, monkeypatch, function):
    monkeypatch.setattr(module, "ORIGINAL_ICP", function)
    monkeypatch.setattr(tracking, "icp_point_to_plane", function)


def arguments():
    return dict(backend="scipy", initial=Rigid.identity(), voxel_size=1.5,
        keep_fraction=.8, tolerance=.001, max_correspondence_distance=12.)


def result(voxel, ordinal):
    transform = Rigid.of(Rigid.identity().rotation, (float(ordinal), 0., 0.))
    values = dict(voxel_size=voxel, normal_neighbours=10, iterations=ordinal+1,
                  transform=transform.as_dict(), converged=False)
    return SimpleNamespace(voxel_size=voxel, normal_neighbours=10, iterations=ordinal+1,
        transform=transform, as_dict=lambda: values)


@pytest.mark.parametrize("profile", ["observe-single", "pyramid-4-2-1"])
def test_level_order_seed_chain_final_object_and_unchanged_arguments(module, monkeypatch, profile):
    calls, returned = [], []
    source, target = [(0.,0.,100.)], [(1.,2.,100.)]
    options = arguments()

    def original(a, b, **kwargs):
        calls.append((a,b,kwargs))
        value = result(kwargs["voxel_size"], len(calls))
        returned.append(value)
        return value

    install(module, monkeypatch, original)
    experiment = module.MultiscaleExperiment(profile)
    with experiment.active(), icp_plane.normal_acceleration(lambda p,k: None):
        actual = tracking.icp_point_to_plane(source, target, **options)
    assert actual is returned[-1]
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()
    assert experiment._original is None
    assert [k["voxel_size"] for _,_,k in calls] == list(experiment.voxels)
    for i, (a,b,kwargs) in enumerate(calls):
        assert a is source and b is target
        expected = options if profile == "observe-single" else options | dict(
            voxel_size=experiment.voxels[i],
            initial=options["initial"] if i == 0 else returned[i-1].transform)
        assert kwargs == expected
    record = experiment.records[0]
    assert record["status"] == "completed"
    assert record["total_completed_iterations"] == sum(r.iterations for r in returned)
    assert [r["result"] for r in record["levels"]] == [r.as_dict() for r in returned]


@pytest.mark.parametrize("level", [0,1,2])
@pytest.mark.parametrize("error", [ValueError("data"), ArithmeticError("math"),
    icp_plane.IcpPlaneCorrespondencesInsufficient("few"), RuntimeError("internal")])
def test_no_coarse_or_fine_failure_is_hidden(module, monkeypatch, level, error):
    calls = []

    def original(a,b,**kwargs):
        ordinal = len(calls)
        calls.append(kwargs)
        if ordinal == level:
            raise error
        return result(kwargs["voxel_size"], ordinal)

    install(module, monkeypatch, original)
    experiment = module.MultiscaleExperiment("pyramid-4-2-1")
    with (pytest.raises(type(error)) as caught, experiment.active(),
          icp_plane.normal_acceleration(lambda p,k: None)):
        tracking.icp_point_to_plane([], [], **arguments())
    assert caught.value is error and len(calls) == level+1
    record = experiment.records[0]
    status = "failed" if isinstance(error, RuntimeError) else "refused"
    assert record["status"] == record["levels"][-1]["status"] == status
    assert all(r["status"] == "completed" for r in record["levels"][:-1])
    assert not module._ACTIVE.get() and tracking.icp_point_to_plane is original


@pytest.mark.parametrize("changes", [dict(backend="python"), dict(voxel_size=3.),
    dict(keep_fraction=.9), dict(normal_neighbours=20), dict(max_iterations=100),
    dict(initial=object())])
def test_foreign_fit_parameters_are_not_silently_accepted(module, changes):
    experiment = module.MultiscaleExperiment("pyramid-4-2-1")
    with (experiment.active(), icp_plane.normal_acceleration(lambda p,k: None),
          pytest.raises(RuntimeError, match="original fit parameters")):
        tracking.icp_point_to_plane([], [], **(arguments() | changes))
    assert experiment.records[0]["status"] == "failed"


@pytest.mark.parametrize("guard", ["inactive", "nested", "foreign-hook", "no-provider", "bound"])
def test_scope_and_work_bounds(module, monkeypatch, guard):
    experiment = module.MultiscaleExperiment("observe-single")
    if guard == "inactive":
        with pytest.raises(RuntimeError):
            experiment.fit([], [], **arguments())
    elif guard == "foreign-hook":
        monkeypatch.setattr(tracking, "icp_point_to_plane", lambda *a,**k: None)
        with pytest.raises(RuntimeError), experiment.active():
            pass
    else:
        with experiment.active():
            if guard == "nested":
                other = module.MultiscaleExperiment("observe-single")
                with pytest.raises(RuntimeError), other.active():
                    pass
            elif guard == "no-provider":
                with pytest.raises(RuntimeError, match="GPU provider"):
                    experiment.fit([], [], **arguments())
            else:
                experiment.records = [{}] * 90000
                with pytest.raises(RuntimeError, match="90000"):
                    experiment.fit([], [], **arguments())
    assert not module._ACTIVE.get()


@pytest.mark.parametrize("field,value", [("voxel_size", 3.), ("normal_neighbours", 20),
                                       ("iterations", 0), ("iterations", 51)])
def test_bad_result_metadata_refuses_before_return(module, monkeypatch, field, value):
    returned = result(1.5, 1)
    setattr(returned, field, value)
    install(module, monkeypatch, lambda *a,**k: returned)
    experiment = module.MultiscaleExperiment("observe-single")
    with (experiment.active(), icp_plane.normal_acceleration(lambda p,k: None),
          pytest.raises(RuntimeError, match="metadata")):
        tracking.icp_point_to_plane([], [], **arguments())
    assert experiment.records[0]["status"] == "failed"


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-single", "pyramid-4-2-1"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
def test_real_tracker_matches_explicit_pyramid(module, capacity, profile, solver_profile):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    points = tuple(corner())
    _, policy, _ = module.settings(capacity, profile,
        module.Factors("grid", "pooled", "residual"), solver_profile)
    experiment = module.MultiscaleExperiment(profile)
    original = tracking.icp_point_to_plane

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    def explicit(source, target, **kwargs):
        initial = kwargs["initial"]
        for voxel in experiment.voxels:
            fitted = original(source, target, **(kwargs | dict(initial=initial, voxel_size=voxel)))
            initial = fitted.transform
        return fitted

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    expected_solver = solvers.SolverExperiment(solver_profile)
    with (icp_plane.normal_acceleration(provider), expected_solver.active(),
          patch.object(tracking, "icp_point_to_plane", explicit)):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    actual_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    actual_solver = solvers.SolverExperiment(solver_profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = [actual_tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert actual_solver.records == expected_solver.records
    assert experiment.records and all(r["status"] == "completed" for r in experiment.records)


@pytest.mark.parametrize("changes", [dict(capacity_profile="unknown"),
    dict(pyramid_profile="pyramid-8-4-2-1"), dict(solver_profile="symmetric"),
    dict(frames=319), dict(skip_complete=1), dict(depth_band=(60,300)),
    dict(static=False, frames=99), dict(static=False, frames=3001)])
def test_adapter_rejects_nonclosed_inputs_before_work(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_multiscale_replay")
    options = dict(capacity_profile="dense-1800", pyramid_profile="pyramid-4-2-1",
        solver_profile="centered", static=True, frames=320)
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "capture", tmp_path / "output",
            module.Factors("grid", "pooled", "residual"), **(options | changes))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("base", ["stride-pooled-residual", "grid-pooled-settling",
                                 "grid-native-residual"])
def test_other_factors_rejected(module, base):
    with pytest.raises(ValueError):
        module.settings("ordinary-900", "pyramid-4-2-1",
            module.Factors(*base.split("-")), "centered")


def test_cli_requires_explicit_pyramid(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_multiscale_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *a,**k: calls.append((a,k)))
    args = ["replay", str(tmp_path / "capture"), "--output", str(tmp_path / "output"),
        "--configuration", "grid-pooled-residual", "--capacity", "dense-1800",
        "--mechanism", "centered", "--frames", "628", "--skip-complete", "12",
        "--depth-band", "60", "300"]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--pyramid", "pyramid-4-2-1"])
    adapter.main()
    assert calls[0][1]["pyramid_profile"] == "pyramid-4-2-1"
    assert calls[0][1]["depth_band"] == (60.,300.)


@pytest.mark.parametrize("profile", ["observe-single", "pyramid-4-2-1"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
@pytest.mark.parametrize("angle", [0., .04])
def test_real_numerics_equal_explicit_level_chain(module, profile, solver_profile, angle):
    from scanner3d.geometry import rotation_about_axis
    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    target = tuple(corner())
    rotation = rotation_about_axis((.3,.5,.7), angle)
    truth = Rigid.of(rotation.rotation, (angle*10, -angle*4, angle*3))
    source = tuple(truth.inverse().apply_all(target))
    options = arguments()
    experiment = module.MultiscaleExperiment(profile)
    expected_solver = solvers.SolverExperiment(solver_profile)
    expected_levels = []
    initial = options["initial"]

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    with icp_plane.normal_acceleration(provider), expected_solver.active():
        for voxel in experiment.voxels:
            expected = icp_plane.icp_point_to_plane(source, target,
                **(options | dict(voxel_size=voxel, initial=initial)))
            expected_levels.append(expected.as_dict())
            initial = expected.transform
    actual_solver = solvers.SolverExperiment(solver_profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = tracking.icp_point_to_plane(source, target, **options)
    assert actual == expected and actual_solver.records == expected_solver.records
    assert [r["result"] for r in experiment.records[0]["levels"]] == expected_levels
    assert actual.converged


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-single", "pyramid-4-2-1"])
def test_capacity_changes_no_other_policy(module, capacity, profile):
    target, policy, source = module.settings(capacity, profile,
        module.Factors("grid", "pooled", "residual"), "centered")
    count = 900 if capacity == "ordinary-900" else 1800
    assert target == count
    assert policy.as_dict() == tracking.DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=count, retained_cloud_points=count)
    assert source == ("observe-900" if count == 900 else "source-1800")


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_reports_and_closed_scope(module, monkeypatch, tmp_path,
                                                   capacity, static, failure):
    adapter = importlib.import_module("scanner_multiscale_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    output = tmp_path / "output"
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "pyramid-4-2-1", factors, "centered")
    content = b'{"original": true}'
    install(module, monkeypatch, lambda *a,**k: result(k["voxel_size"], 1))

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        with icp_plane.normal_acceleration(lambda p,k: None):
            tracking.icp_point_to_plane([], [], **arguments())
        output.mkdir()
        for name in (("report.json",) if static else ("report.json", "stride-1.json")):
            (output / name).write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    options = dict(pyramid_profile="pyramid-4-2-1", capacity_profile=capacity,
                   solver_profile="centered", static=static, frames=320)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            adapter.run(tmp_path / "capture", output, factors, **options)
    else:
        adapter.run(tmp_path / "capture", output, factors, **options)
    metadata = json.loads((output / "pyramid-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["voxel_schedule"] == [6.,3.,1.5]
    assert metadata["maximum_iterations_per_requested_fit"] == 150
    assert metadata["tracking_policy"] == expected[1].as_dict()
    assert metadata["preparation_target_points"] == expected[0]
    assert not metadata["quality_qualification_allowed"]
    assert all(r["sha256"] == hashlib.sha256(content).hexdigest()
               for r in metadata["underlying_reports"])
    assert len(metadata["underlying_reports"]) == (1 if static else 2)
    assert replay.IncrementalTracker is factory and not adapter._ACTIVE.get()
    assert not module._ACTIVE.get()
