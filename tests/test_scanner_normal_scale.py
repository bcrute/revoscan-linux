"""M13 argument-level normal-scale intervention and paired-report provenance."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.tracking import DEFAULT_POLICY


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_normal_scale")


def install_original(module, monkeypatch, function):
    monkeypatch.setattr(module, "ORIGINAL_ICP", function)
    monkeypatch.setattr(tracking, "icp_point_to_plane", function)


@pytest.mark.parametrize("profile", ["observe-10", "normal-20"])
def test_exact_delegation_object_identity_and_only_declared_argument_changes(
    module, monkeypatch, profile
):
    calls = []
    count = 10 if profile == "observe-10" else 20
    result = SimpleNamespace(normal_neighbours=count, as_dict=lambda: dict(normal_neighbours=count))
    source, target = [(0., 0., 1.)], [(1., 1., 1.)]

    def original(a, b, **kwargs):
        calls.append((a, b, kwargs))
        return result

    install_original(module, monkeypatch, original)
    experiment = module.NormalScaleExperiment(profile)
    arguments = dict(backend="scipy", initial=object(), tolerance=0.1,
                     max_correspondence_distance=12., voxel_size=1.5, keep_fraction=.8)
    with icp_plane.normal_acceleration(lambda p,k: None), experiment.active():
        actual = tracking.icp_point_to_plane(source, target, **arguments)
    assert actual is result and calls[0][0] is source and calls[0][1] is target
    assert calls[0][2] == (arguments if count == 10 else arguments | dict(normal_neighbours=20))
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()
    assert experiment._original is None
    assert experiment.records[0]["actual_normal_neighbours"] == count
    assert experiment.records[0]["status"] == "completed"


@pytest.mark.parametrize("error", [ValueError("data"), ArithmeticError("arithmetic"),
    icp_plane.IcpPlaneCorrespondencesInsufficient("few"), RuntimeError("internal")])
def test_original_refusal_or_failure_is_preserved_with_cleanup(module, monkeypatch, error):
    def original(*args, **kwargs):
        raise error

    install_original(module, monkeypatch, original)
    experiment = module.NormalScaleExperiment("normal-20")
    with (
        pytest.raises(type(error)) as caught,
        icp_plane.normal_acceleration(lambda p,k: None), experiment.active()
    ):
        tracking.icp_point_to_plane([], [], backend="scipy")
    assert caught.value is error
    expected = "failed" if isinstance(error, RuntimeError) else "refused"
    assert experiment.records[0]["status"] == expected
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()


@pytest.mark.parametrize("failure", ["inactive", "nested", "no-provider", "wrong-count", "bound"])
def test_invalid_scope_or_count_is_explicit(module, failure):
    experiment = module.NormalScaleExperiment("observe-10")
    if failure == "inactive":
        with pytest.raises(RuntimeError):
            experiment.fit([], [], backend="scipy")
        return
    with experiment.active():
        if failure == "nested":
            with pytest.raises(RuntimeError), module.NormalScaleExperiment("normal-20").active():
                pass
        elif failure == "no-provider":
            with pytest.raises(RuntimeError, match="GPU provider"):
                experiment.fit([], [], backend="scipy")
        else:
            if failure == "bound":
                experiment.records = [{}] * 90000
            with icp_plane.normal_acceleration(lambda p,k: None), pytest.raises(RuntimeError):
                experiment.fit([], [], backend="scipy", normal_neighbours=11)
    assert not module._ACTIVE.get()


def test_returned_neighbor_count_must_be_truthful(module, monkeypatch):
    install_original(module, monkeypatch, lambda *a,**k: SimpleNamespace(normal_neighbours=10))
    experiment = module.NormalScaleExperiment("normal-20")
    with (
        experiment.active(), icp_plane.normal_acceleration(lambda p,k: None),
        pytest.raises(RuntimeError, match="metadata")
    ):
        experiment.fit([], [], backend="scipy")
    assert experiment.records[0]["status"] == "failed"


def test_real_icp_receives_twenty_and_matches_explicit_cpu_reference(module):
    points = tuple((x*2., y*2., 100. + .02*x*x + .06*y*y + .01*x*y)
                   for x in range(-6,7) for y in range(-6,7))
    counts = []

    def provider(cloud, count):
        counts.append(count)
        return icp_plane.estimate_normals(cloud, neighbours=count, backend="scipy")

    arguments = dict(backend="scipy", voxel_size=1.5, max_correspondence_distance=12.,
                     keep_fraction=.8)
    expected = icp_plane.icp_point_to_plane(points, points, normal_neighbours=20, **arguments)
    experiment = module.NormalScaleExperiment("normal-20")
    with experiment.active(), icp_plane.normal_acceleration(provider):
        actual = tracking.icp_point_to_plane(points, points, **arguments)
    assert actual == expected and counts == [20]
    assert actual.normal_neighbours == 20
    assert experiment.records[0]["result"] == actual.as_dict()


def test_cli_requires_explicit_normal_scale(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_normal_scale_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *a,**k: calls.append((a,k)))
    arguments = ["scanner_normal_scale_replay.py", str(tmp_path / "capture"), "--output",
        str(tmp_path / "output"), "--configuration", "grid-pooled-residual", "--mechanism",
        "centered", "--capacity", "dense-1800", "--frames", "320"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*arguments, "--normal-scale", "normal-20"])
    adapter.main()
    assert calls[0][1]["normal_profile"] == "normal-20"


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-10", "normal-20"])
def test_actual_source_capacity_and_unchanged_policy(module, capacity, profile):
    target, policy, source = module.settings(capacity, profile,
        module.Factors("grid", "pooled", "residual"), "centered")
    count = 900 if capacity == "ordinary-900" else 1800
    assert target == count
    assert policy.as_dict() == DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=count, retained_cloud_points=count)
    assert source == ("observe-900" if count == 900 else "source-1800")


@pytest.mark.parametrize("capacity,profile,base,solver", [
    ("unknown", "normal-20", "grid-pooled-residual", "centered"),
    ("ordinary-900", "normal-15", "grid-pooled-residual", "centered"),
    ("ordinary-900", "normal-20", "stride-pooled-residual", "centered"),
    ("ordinary-900", "normal-20", "grid-pooled-residual", "symmetric"),
])
def test_nonclosed_settings_rejected(module, capacity, profile, base, solver):
    with pytest.raises(ValueError):
        module.settings(capacity, profile, module.Factors(*base.split("-")), solver)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_parameters_report_bytes_and_restoration(
    module, monkeypatch, tmp_path, capacity, static, failure
):
    adapter = importlib.import_module("scanner_normal_scale_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = replay.IncrementalTracker
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "normal-20", factors, "centered")
    output = tmp_path / "output"
    content = b'{"unchanged": true}'
    result = SimpleNamespace(normal_neighbours=20, as_dict=lambda: dict(normal_neighbours=20))
    def original(*args, **kwargs):
        return result
    install_original(module, monkeypatch, original)

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        with icp_plane.normal_acceleration(lambda p,k: None):
            assert tracking.icp_point_to_plane([], [], backend="scipy") is result
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    arguments = dict(normal_profile="normal-20", capacity_profile=capacity,
                     solver_profile="centered", static=static, frames=320)
    if failure:
        with pytest.raises(RuntimeError, match="injected adapter failure"):
            adapter.run(tmp_path / "capture", output, factors, **arguments)
    else:
        adapter.run(tmp_path / "capture", output, factors, **arguments)
    assert tracking.icp_point_to_plane is original and replay.IncrementalTracker is original_factory
    assert not adapter._ACTIVE.get() and not module._ACTIVE.get()
    assert (output / "report.json").read_bytes() == content
    report = json.loads((output / "normal-scale-report.json").read_text())
    assert report["status"] == ("failed" if failure else "completed")
    assert report["actual_normal_neighbours"] == 20
    assert report["fit_records"][0]["actual_normal_neighbours"] == 20
    assert report["tracking_policy"] == expected[1].as_dict()
    assert report["preparation_target_points"] == expected[0]
    assert report["underlying_reports"] == [dict(path=str(output / "report.json"),
        sha256=hashlib.sha256(content).hexdigest())]
    assert not report["quality_qualification_allowed"]
