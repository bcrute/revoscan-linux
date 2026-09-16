"""M15 exact capacity/stopping composition, metadata and scope guards."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scanner3d.tracking import DEFAULT_POLICY


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_capacity_settling")


@pytest.mark.parametrize("profile,limit", [("observe-900", 900), ("source-1800", 1800)])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
@pytest.mark.parametrize("stopping", ["residual", "settling"])
def test_exact_caps_and_original_policy(module, profile, limit, solver, stopping):
    factors = module.Factors("grid", "pooled", stopping)
    target, policy = module.settings(profile, factors, solver, None)
    assert target == limit
    assert policy.as_dict() == DEFAULT_POLICY.as_dict() | dict(
        candidate_source_points=limit, retained_cloud_points=limit)
    if profile == "observe-900":
        assert policy is DEFAULT_POLICY
    if stopping == "residual":
        assert (target, policy) == module.ORIGINAL_SETTINGS(profile, factors, solver, None)


@pytest.mark.parametrize("profile,base,solver,model", [
    ("unknown", "grid-pooled-settling", "centered", None),
    ("source-1800", "stride-pooled-settling", "centered", None),
    ("source-1800", "grid-native-settling", "centered", None),
    ("source-1800", "rank-pooled-residual", "centered", None),
    ("source-1800", "grid-pooled-settling", "symmetric", None),
    ("source-1800", "grid-pooled-settling", "centered", "measured-voxel"),
])
def test_nonclosed_roster_refused(module, profile, base, solver, model):
    with pytest.raises(ValueError):
        module.settings(profile, module.Factors(*base.split("-")), solver, model)


@pytest.mark.parametrize("profile", ["observe-900", "source-1800"])
@pytest.mark.parametrize("stopping", ["residual", "settling"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_exact_delegation_report_and_cleanup(
    module, monkeypatch, tmp_path, profile, stopping, static, failure
):
    adapter = importlib.import_module("scanner_capacity_settling_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = replay.IncrementalTracker
    output = tmp_path / "output"
    content = b'{"original": "untouched"}'
    factors = module.Factors("grid", "pooled", stopping)
    expected = module.settings(profile, factors, "centered", None)

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == profile
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if static:
            assert args == (factors,)
        else:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
            assert kwargs["diagnostic_factors"] == factors.values
            assert kwargs["skip_complete"] == 12 and kwargs["depth_band"] == (60, 300)
        assert module.scanner_source_capacity.settings(
            profile, factors, "centered", None
        ) == expected
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    arguments = dict(capacity_profile=profile, solver_profile="centered", static=static,
                     frames=320, skip_complete=0 if static else 12,
                     depth_band=None if static else (60, 300))
    if failure:
        with pytest.raises(RuntimeError, match="injected failure"):
            adapter.run(tmp_path / "capture", output, factors, **arguments)
    else:
        adapter.run(tmp_path / "capture", output, factors, **arguments)
    assert not adapter._ACTIVE.get()
    assert module.scanner_source_capacity.settings is module.ORIGINAL_SETTINGS
    assert replay.IncrementalTracker is original_factory
    assert (output / "report.json").read_bytes() == content
    report = json.loads((output / "capacity-settling-report.json").read_text())
    assert report["status"] == ("failed" if failure else "completed")
    assert report["capacity_settings_calls"] == 1
    assert report["tracking_policy"] == expected[1].as_dict()
    assert report["preparation_target_points"] == expected[0]
    assert report["diagnostic_source_profile"] == profile
    assert report["diagnostic_factors"] == factors.as_dict()
    assert report["convergence_rule"] == factors.convergence_rule
    assert report["underlying_reports"] == [dict(path=str(output / "report.json"),
        sha256=hashlib.sha256(content).hexdigest())]
    assert not report["quality_qualification_allowed"]


@pytest.mark.parametrize("calls", [0, 2])
def test_missing_or_repeated_hook_fails(module, monkeypatch, tmp_path, calls):
    adapter = importlib.import_module("scanner_capacity_settling_replay")
    factors = module.Factors("grid", "pooled", "settling")

    def delegated(*args, **kwargs):
        for _ in range(calls):
            module.scanner_source_capacity.settings("source-1800", factors, "centered", None)

    monkeypatch.setattr(adapter.static_replay, "run_static", delegated)
    with pytest.raises(RuntimeError):
        adapter.run(tmp_path / "capture", tmp_path / "output", factors,
                    capacity_profile="source-1800", solver_profile="centered",
                    static=True, frames=320)
    assert not adapter._ACTIVE.get()
    assert module.scanner_source_capacity.settings is module.ORIGINAL_SETTINGS


@pytest.mark.parametrize("profile", ["observe-origin", "centered"])
def test_actual_settling_solver_composition_and_restoration(module, profile):
    from scanner3d import icp_plane
    from scanner3d.posegraph_lm import se3_exp
    from tests.test_scanner_registration_audit import corner

    settling = importlib.import_module("scanner_pose_settling")
    mechanisms = importlib.import_module("scanner_solver_mechanisms")
    original_step, original_iterate = icp_plane._solve_point_to_plane, icp_plane._iterate
    target = corner()
    truth = se3_exp([0.3, -0.2, 0.1, 0.001, -0.002, 0.001])
    source = tuple(truth.inverse().apply_all(target))
    with settling.pose_settling():
        reference = icp_plane.icp_point_to_plane(
            source, target, backend="scipy", voxel_size=1.5)
    experiment = mechanisms.SolverExperiment(profile)
    experiment.mark_frame(0)
    with settling.pose_settling(), experiment.active():
        assert icp_plane._iterate is settling.iterate
        assert icp_plane._solve_point_to_plane == experiment.step
        result = icp_plane.icp_point_to_plane(source, target, backend="scipy", voxel_size=1.5)
    assert icp_plane._iterate is original_iterate
    assert icp_plane._solve_point_to_plane is original_step
    assert result.converged and experiment.records
    assert all(r["profile"] == profile and r["source_index"] == 0 for r in experiment.records)
    assert result.transform.apply(source[80]) == pytest.approx(target[80], abs=0.01)
    if profile == "observe-origin":
        assert result == reference


@pytest.mark.parametrize("static,frames,skip,band", [
    (True, 319, 0, None), (True, 320, 1, None), (True, 320, 0, (60, 300)),
    (False, 99, 0, None), (False, 3001, 0, None),
])
def test_input_bounds_fail_before_delegation(module, tmp_path, static, frames, skip, band):
    adapter = importlib.import_module("scanner_capacity_settling_replay")
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "capture", tmp_path / "output",
            module.Factors("grid", "pooled", "settling"), capacity_profile="source-1800",
            solver_profile="centered", static=static, frames=frames,
            skip_complete=skip, depth_band=band)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("guard", ["output", "nested", "capacity_hook", "tracker"])
def test_existing_state_and_foreign_hooks_refused(module, monkeypatch, tmp_path, guard):
    adapter = importlib.import_module("scanner_capacity_settling_replay")
    output = tmp_path / "output"
    token = None
    if guard == "output":
        output.mkdir()
    elif guard == "nested":
        token = adapter._ACTIVE.set(True)
    elif guard == "capacity_hook":
        monkeypatch.setattr(module.scanner_source_capacity, "settings", lambda *a: None)
    else:
        monkeypatch.setattr(adapter.static_replay, "IncrementalTracker", lambda *a: None)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            adapter.run(tmp_path / "capture", output,
                module.Factors("grid", "pooled", "settling"), capacity_profile="source-1800",
                solver_profile="centered", static=True, frames=320)
    finally:
        if token is not None:
            adapter._ACTIVE.reset(token)


@pytest.mark.parametrize("stopping", ["residual", "settling"])
def test_cli_preserves_explicit_stopping_and_capacity(module, monkeypatch, tmp_path, stopping):
    adapter = importlib.import_module("scanner_capacity_settling_replay")
    captured = []
    monkeypatch.setattr(adapter, "run", lambda *a, **k: captured.append((a, k)))
    monkeypatch.setattr("sys.argv", ["replay", str(tmp_path / "capture"),
        "--output", str(tmp_path / "output"), "--configuration", f"grid-pooled-{stopping}",
        "--mechanism", "centered", "--capacity", "source-1800", "--frames", "628",
        "--skip-complete", "12", "--depth-band", "60", "300"])
    adapter.main()
    args, kwargs = captured[0]
    assert args[2].convergence == stopping
    assert kwargs == dict(capacity_profile="source-1800", solver_profile="centered",
        static=False, frames=628, skip_complete=12, depth_band=(60.0, 300.0))
