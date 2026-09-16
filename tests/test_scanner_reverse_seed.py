"""M21 swap/inverse/forward metadata identity and explicit failures, no scanner."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid, rotation_about_axis


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_reverse_seed")


def install(module, monkeypatch, function):
    monkeypatch.setattr(module, "ORIGINAL_ICP", function)
    monkeypatch.setattr(tracking, "icp_point_to_plane", function)


def arguments():
    return dict(
        backend="scipy",
        initial=Rigid.identity(),
        voxel_size=1.5,
        keep_fraction=0.8,
        tolerance=0.001,
        max_correspondence_distance=12.0,
    )


def result(voxel, ordinal):
    transform = Rigid.of(
        rotation_about_axis((0.3, 0.5, -0.7), 0.04 * ordinal).rotation, (float(ordinal), -0.3, 0.9)
    )
    values = dict(
        voxel_size=voxel,
        normal_neighbours=10,
        iterations=ordinal + 1,
        transform=transform.as_dict(),
        converged=False,
    )
    return SimpleNamespace(
        voxel_size=voxel,
        normal_neighbours=10,
        iterations=ordinal + 1,
        transform=transform,
        as_dict=lambda: values,
    )


@pytest.mark.parametrize("profile", ["observe-forward", "reverse-forward"])
def test_direction_order_inverse_seeds_final_object_and_arguments(module, monkeypatch, profile):
    calls, returned = [], []
    source = [(0.0, 0.0, 100.0), (1.0, 0.0, 100.0)]
    target = [(1.0, 2.0, 100.0), (2.0, 2.0, 100.0), (3.0, 2.0, 100.0)]
    options = arguments()
    options["initial"] = Rigid.of(
        rotation_about_axis((0.2, -0.7, 0.3), 0.17).rotation, (0.8, -0.2, 1.4)
    )

    def original(a, b, **kwargs):
        calls.append((a, b, kwargs))
        value = result(kwargs["voxel_size"], len(calls))
        returned.append(value)
        return value

    install(module, monkeypatch, original)
    experiment = module.ReverseSeedExperiment(profile)
    with experiment.active(), icp_plane.normal_acceleration(lambda p, k: None):
        actual = tracking.icp_point_to_plane(source, target, **options)
    assert actual is returned[-1]
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()
    assert experiment._original is None
    assert len(calls) == len(experiment.directions)
    for i, (a, b, kwargs) in enumerate(calls):
        reverse = experiment.directions[i] == "reverse"
        assert a is (target if reverse else source) and b is (source if reverse else target)
        expected = (
            options
            if profile == "observe-forward"
            else options
            | dict(
                initial=options["initial"].inverse()
                if i == 0
                else returned[i - 1].transform.inverse()
            )
        )
        assert kwargs == expected
    record = experiment.records[0]
    assert record["status"] == "completed"
    assert record["total_completed_iterations"] == sum(r.iterations for r in returned)
    assert [r["direction"] for r in record["stages"]] == list(experiment.directions)
    for stage in record["stages"]:
        reverse = stage["direction"] == "reverse"
        assert stage["source_points"] == len(target if reverse else source)
        assert stage["target_points"] == len(source if reverse else target)
    assert [r["result"] for r in record["stages"]] == [r.as_dict() for r in returned]


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("data"),
        ArithmeticError("math"),
        icp_plane.IcpPlaneCorrespondencesInsufficient("few"),
        RuntimeError("internal"),
    ],
)
def test_no_reverse_or_forward_failure_is_hidden(module, monkeypatch, level, error):
    calls = []

    def original(a, b, **kwargs):
        ordinal = len(calls)
        calls.append(kwargs)
        if ordinal == level:
            raise error
        return result(kwargs["voxel_size"], ordinal)

    install(module, monkeypatch, original)
    experiment = module.ReverseSeedExperiment("reverse-forward")
    with (
        pytest.raises(type(error)) as caught,
        experiment.active(),
        icp_plane.normal_acceleration(lambda p, k: None),
    ):
        tracking.icp_point_to_plane([], [], **arguments())
    assert caught.value is error and len(calls) == level + 1
    record = experiment.records[0]
    status = "failed" if isinstance(error, RuntimeError) else "refused"
    assert record["status"] == record["stages"][-1]["status"] == status
    assert all(r["status"] == "completed" for r in record["stages"][:-1])
    assert not module._ACTIVE.get() and tracking.icp_point_to_plane is original


@pytest.mark.parametrize(
    "changes",
    [
        dict(backend="python"),
        dict(voxel_size=3.0),
        dict(keep_fraction=0.9),
        dict(normal_neighbours=20),
        dict(max_iterations=100),
        dict(initial=object()),
    ],
)
def test_foreign_fit_parameters_are_not_silently_accepted(module, changes):
    experiment = module.ReverseSeedExperiment("reverse-forward")
    with (
        experiment.active(),
        icp_plane.normal_acceleration(lambda p, k: None),
        pytest.raises(RuntimeError, match="original fit parameters"),
    ):
        tracking.icp_point_to_plane([], [], **(arguments() | changes))
    assert experiment.records[0]["status"] == "failed"


@pytest.mark.parametrize("guard", ["inactive", "nested", "foreign-hook", "no-provider", "bound"])
def test_scope_and_work_bounds(module, monkeypatch, guard):
    experiment = module.ReverseSeedExperiment("observe-forward")
    if guard == "inactive":
        with pytest.raises(RuntimeError):
            experiment.fit([], [], **arguments())
    elif guard == "foreign-hook":
        monkeypatch.setattr(tracking, "icp_point_to_plane", lambda *a, **k: None)
        with pytest.raises(RuntimeError), experiment.active():
            pass
    else:
        with experiment.active():
            if guard == "nested":
                other = module.ReverseSeedExperiment("observe-forward")
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


@pytest.mark.parametrize(
    "field,value",
    [("voxel_size", 3.0), ("normal_neighbours", 20), ("iterations", 0), ("iterations", 51)],
)
def test_bad_result_metadata_refuses_before_return(module, monkeypatch, field, value):
    returned = result(1.5, 1)
    setattr(returned, field, value)
    install(module, monkeypatch, lambda *a, **k: returned)
    experiment = module.ReverseSeedExperiment("observe-forward")
    with (
        experiment.active(),
        icp_plane.normal_acceleration(lambda p, k: None),
        pytest.raises(RuntimeError, match="metadata"),
    ):
        tracking.icp_point_to_plane([], [], **arguments())
    assert experiment.records[0]["status"] == "failed"


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
def test_real_tracker_matches_explicit_reverse_seed(module, capacity, profile, solver_profile):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    points = tuple(corner())
    _, policy, _ = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), solver_profile
    )
    experiment = module.ReverseSeedExperiment(profile)
    original = tracking.icp_point_to_plane

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    def explicit(source, target, **kwargs):
        if profile == "observe-forward":
            return original(source, target, **kwargs)
        reverse = original(target, source, **(kwargs | dict(initial=kwargs["initial"].inverse())))
        return original(source, target, **(kwargs | dict(initial=reverse.transform.inverse())))

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    expected_solver = solvers.SolverExperiment(solver_profile)
    with (
        icp_plane.normal_acceleration(provider),
        expected_solver.active(),
        patch.object(tracking, "icp_point_to_plane", explicit),
    ):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    actual_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    actual_solver = solvers.SolverExperiment(solver_profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = [actual_tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert actual_solver.records == expected_solver.records
    assert experiment.records and all(r["status"] == "completed" for r in experiment.records)


@pytest.mark.parametrize(
    "changes",
    [
        dict(capacity_profile="unknown"),
        dict(seed_profile="undeclared-three-pass"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False, frames=99),
        dict(static=False, frames=3001),
    ],
)
def test_adapter_rejects_nonclosed_inputs_before_work(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_reverse_seed_replay")
    options = dict(
        capacity_profile="dense-1800",
        seed_profile="reverse-forward",
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
    "base", ["stride-pooled-residual", "grid-pooled-settling", "grid-native-residual"]
)
def test_other_factors_rejected(module, base):
    with pytest.raises(ValueError):
        module.settings(
            "ordinary-900", "reverse-forward", module.Factors(*base.split("-")), "centered"
        )


def test_cli_requires_explicit_reverse_seed(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_reverse_seed_replay")
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
    monkeypatch.setattr("sys.argv", [*args, "--reverse-seed", "reverse-forward"])
    adapter.main()
    assert calls[0][1]["seed_profile"] == "reverse-forward"
    assert calls[0][1]["depth_band"] == (60.0, 300.0)


@pytest.mark.parametrize("profile", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
@pytest.mark.parametrize("angle", [0.0, 0.04])
def test_real_numerics_equal_explicit_direction_chain(module, profile, solver_profile, angle):
    from scanner3d.geometry import rotation_about_axis
    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    target = tuple(corner())
    rotation = rotation_about_axis((0.3, 0.5, 0.7), angle)
    truth = Rigid.of(rotation.rotation, (angle * 10, -angle * 4, angle * 3))
    source = tuple(truth.inverse().apply_all(target))
    options = arguments()
    experiment = module.ReverseSeedExperiment(profile)
    expected_solver = solvers.SolverExperiment(solver_profile)
    expected_levels = []
    initial = options["initial"] if profile == "observe-forward" else options["initial"].inverse()

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    with icp_plane.normal_acceleration(provider), expected_solver.active():
        for direction in experiment.directions:
            a, b = (target, source) if direction == "reverse" else (source, target)
            expected = icp_plane.icp_point_to_plane(a, b, **(options | dict(initial=initial)))
            expected_levels.append(expected.as_dict())
            initial = expected.transform.inverse() if direction == "reverse" else expected.transform
    actual_solver = solvers.SolverExperiment(solver_profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = tracking.icp_point_to_plane(source, target, **options)
    assert actual == expected and actual_solver.records == expected_solver.records
    assert [r["result"] for r in experiment.records[0]["stages"]] == expected_levels
    assert actual.converged


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-forward", "reverse-forward"])
def test_capacity_changes_no_other_policy(module, capacity, profile):
    target, policy, source = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), "centered"
    )
    count = 900 if capacity == "ordinary-900" else 1800
    assert target == count
    assert policy.as_dict() == tracking.DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=count, retained_cloud_points=count
    )
    assert source == ("observe-900" if count == 900 else "source-1800")


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_reports_and_closed_scope(
    module, monkeypatch, tmp_path, capacity, static, failure
):
    adapter = importlib.import_module("scanner_reverse_seed_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    output = tmp_path / "output"
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "reverse-forward", factors, "centered")
    content = b'{"original": true}'
    install(module, monkeypatch, lambda *a, **k: result(k["voxel_size"], 1))

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        with icp_plane.normal_acceleration(lambda p, k: None):
            tracking.icp_point_to_plane([], [], **arguments())
        output.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (output / name).write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    options = dict(
        seed_profile="reverse-forward",
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
    metadata = json.loads((output / "reverse-seed-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["direction_schedule"] == ["reverse", "forward"]
    assert metadata["maximum_iterations_per_requested_fit"] == 100
    assert metadata["tracking_policy"] == expected[1].as_dict()
    assert metadata["preparation_target_points"] == expected[0]
    assert not metadata["quality_qualification_allowed"]
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in metadata["underlying_reports"]
    )
    assert len(metadata["underlying_reports"]) == (1 if static else 2)
    assert replay.IncrementalTracker is factory and not adapter._ACTIVE.get()
    assert not module._ACTIVE.get()
