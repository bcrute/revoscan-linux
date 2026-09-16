"""M17 component composition, unchanged report bytes and actual fit arguments."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_voxel_normal_replay")


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-10", "normal-20"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_exact_composition_metadata_and_report_bytes(
    module, monkeypatch, tmp_path, capacity, profile, static, failure
):
    normal = importlib.import_module("scanner_normal_scale")
    count = 10 if profile == "observe-10" else 20
    result = SimpleNamespace(normal_neighbours=count, as_dict=lambda: dict(normal_neighbours=count))
    calls = []

    def original(source, target, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(normal, "ORIGINAL_ICP", original)
    monkeypatch.setattr(tracking, "icp_point_to_plane", original)
    capture, output = tmp_path / "capture", tmp_path / "output"
    factors = module.Factors("grid", "pooled", "residual")
    content = b'{"original": "not rewritten"}'
    settings = module.voxel_settings(capacity, module.MODEL_PROFILE, factors, "centered")

    def delegated(actual_capture, actual_output, actual_factors, **kwargs):
        assert (actual_capture, actual_output, actual_factors) == (capture, output, factors)
        assert kwargs == dict(
            sampling_profile="voxel-first-model",
            capacity_profile=capacity,
            solver_profile="centered",
            static=static,
            frames=320,
            skip_complete=0 if static else 12,
            depth_band=None if static else (60, 300),
        )
        with icp_plane.normal_acceleration(lambda p, k: None):
            assert tracking.icp_point_to_plane([], [], backend="scipy") is result
        output.mkdir()
        for name in ("report.json", "model-sampling-report.json"):
            (output / name).write_bytes(content)
        if not static:
            (output / "stride-1.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected")

    monkeypatch.setattr(module.voxel_replay, "run", delegated)
    options = dict(
        capacity_profile=capacity,
        normal_profile=profile,
        solver_profile="centered",
        static=static,
        frames=320,
        skip_complete=0 if static else 12,
        depth_band=None if static else (60, 300),
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            module.run(capture, output, factors, **options)
    else:
        module.run(capture, output, factors, **options)
    assert tracking.icp_point_to_plane is original and not module._ACTIVE.get()
    assert not normal._ACTIVE.get() and not module.voxel_replay._ACTIVE.get()
    assert calls == [
        dict(backend="scipy") if count == 10 else dict(backend="scipy", normal_neighbours=20)
    ]
    metadata = json.loads((output / "voxel-normal-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["normal_profile"] == profile and metadata["capacity_profile"] == capacity
    assert metadata["sampling_profile"] == "voxel-first-model"
    assert metadata["preparation_target_points"] == settings[0]
    assert metadata["tracking_policy"] == settings[1].as_dict()
    assert metadata["diagnostic_source_profile"] == settings[2]
    assert not metadata["quality_qualification_allowed"]
    assert metadata["fit_records"][0]["actual_normal_neighbours"] == count
    expected = {str(output / "report.json"), str(output / "model-sampling-report.json")}
    if not static:
        expected.add(str(output / "stride-1.json"))
    assert {r["path"] for r in metadata["underlying_reports"]} == expected
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in metadata["underlying_reports"]
    )
    assert all(Path(path).read_bytes() == content for path in expected)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-10", "normal-20"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
def test_real_model_tracker_matches_explicit_normal_count(
    module, capacity, profile, solver_profile
):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    voxel = importlib.import_module("scanner_voxel_model")
    solvers = importlib.import_module("scanner_solver_mechanisms")
    points = tuple(corner())
    factors = module.Factors("grid", "pooled", "residual")
    _, policy, _ = module.voxel_settings(capacity, module.MODEL_PROFILE, factors, solver_profile)
    tracker_type = (
        voxel.VoxelModelTracker if capacity == "ordinary-900" else voxel.DenseVoxelModelTracker
    )
    count = 10 if profile == "observe-10" else 20
    original = tracking.icp_point_to_plane

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    def explicit(source, target, **kwargs):
        return original(source, target, **(kwargs | dict(normal_neighbours=count)))

    expected_tracker = tracker_type(
        backend="scipy", policy=policy, sampling_profile=module.MODEL_PROFILE
    )
    expected_solver = solvers.SolverExperiment(solver_profile)
    expected = []
    with (
        icp_plane.normal_acceleration(provider),
        patch.object(tracking, "icp_point_to_plane", explicit),
        expected_solver.active(),
    ):
        for index in range(4):
            expected_solver.mark_frame(index)
            expected.append(expected_tracker.offer(points).as_dict())
    actual_tracker = tracker_type(
        backend="scipy", policy=policy, sampling_profile=module.MODEL_PROFILE
    )
    actual_solver = solvers.SolverExperiment(solver_profile)
    experiment = module.NormalScaleExperiment(profile)
    actual = []
    with icp_plane.normal_acceleration(provider), experiment.active(), actual_solver.active():
        for index in range(4):
            actual_solver.mark_frame(index)
            actual.append(actual_tracker.offer(points).as_dict())
    assert actual == expected and all(row["accepted"] for row in actual)
    assert actual_tracker.model_sampling_records == expected_tracker.model_sampling_records
    assert actual_tracker.model_sampling_records
    assert actual_solver.records == expected_solver.records
    assert experiment.records and all(
        r["actual_normal_neighbours"] == count for r in experiment.records
    )
    assert tracking.icp_point_to_plane is original


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity_profile": "unknown"},
        {"normal_profile": "normal-15"},
        {"solver_profile": "symmetric"},
        {"frames": 319},
        {"skip_complete": 1},
        {"depth_band": (60, 300)},
        {"static": False, "frames": 99},
        {"static": False, "frames": 3001},
    ],
)
def test_closed_inputs_before_any_output(module, tmp_path, changes):
    options = dict(
        capacity_profile="dense-1800",
        normal_profile="normal-20",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError):
        module.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **(options | changes),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("guard", ["output", "nested-adapter", "nested-normal", "foreign-hook"])
def test_scope_guards_restore_original_state(module, monkeypatch, tmp_path, guard):
    from contextlib import nullcontext

    output = tmp_path / "output"
    token = None
    if guard == "output":
        output.mkdir()
    if guard == "nested-adapter":
        token = module._ACTIVE.set(True)
    if guard == "foreign-hook":
        monkeypatch.setattr(tracking, "icp_point_to_plane", lambda *a, **k: None)
    scope = (
        module.NormalScaleExperiment("normal-20").active()
        if guard == "nested-normal"
        else nullcontext()
    )
    try:
        with scope, pytest.raises((ValueError, RuntimeError)):
            module.run(
                tmp_path / "capture",
                output,
                module.Factors("grid", "pooled", "residual"),
                capacity_profile="dense-1800",
                normal_profile="normal-20",
                solver_profile="centered",
                static=True,
                frames=320,
            )
    finally:
        if token is not None:
            module._ACTIVE.reset(token)
    assert not module._ACTIVE.get()


def test_missing_actual_fits_is_not_success(module, monkeypatch, tmp_path):
    output = tmp_path / "output"

    def delegated(*args, **kwargs):
        output.mkdir()
        (output / "report.json").write_text("{}")
        (output / "model-sampling-report.json").write_text("{}")

    monkeypatch.setattr(module.voxel_replay, "run", delegated)
    with pytest.raises(RuntimeError, match="actual tracker fit"):
        module.run(
            tmp_path / "capture",
            output,
            module.Factors("grid", "pooled", "residual"),
            capacity_profile="dense-1800",
            normal_profile="normal-20",
            solver_profile="centered",
            static=True,
            frames=320,
        )
    report = json.loads((output / "voxel-normal-report.json").read_bytes())
    assert report["status"] == "failed" and report["fit_records"] == []
    assert not module._ACTIVE.get()


def test_cli_requires_explicit_support_profile(module, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--configuration",
        "grid-pooled-residual",
        "--mechanism",
        "centered",
        "--capacity",
        "dense-1800",
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
        module.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--normal-scale", "normal-20"])
    module.main()
    assert calls[0][1] == dict(
        capacity_profile="dense-1800",
        normal_profile="normal-20",
        solver_profile="centered",
        static=False,
        frames=628,
        skip_complete=12,
        depth_band=(60.0, 300.0),
    )
