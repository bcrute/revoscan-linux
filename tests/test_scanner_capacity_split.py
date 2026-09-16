"""M12 point-cap decomposition without hidden policy or metadata changes."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scanner3d.tracking import DEFAULT_POLICY


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_capacity_split")


@pytest.mark.parametrize("profile,prepare,source,retained", [
    ("observe-900", 900, 900, 900), ("source-1800", 1800, 1800, 1800),
    ("prepared-1800-source-900-retained-900", 1800, 900, 900),
    ("prepared-1800-source-1800-retained-900", 1800, 1800, 900),
    ("prepared-1800-source-900-retained-1800", 1800, 900, 1800),
])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_exact_caps_and_every_other_policy_field(
    module, profile, prepare, source, retained, solver
):
    target, policy = module.settings(profile, module.Factors("grid", "pooled", "residual"),
                                     solver, None)
    expected = DEFAULT_POLICY.as_dict() | dict(candidate_source_points=source,
                                              retained_cloud_points=retained)
    assert target == prepare and policy.as_dict() == expected
    if profile == "observe-900":
        assert policy is DEFAULT_POLICY
    if profile in ("observe-900", "source-1800"):
        assert (target, policy) == module.ORIGINAL_SETTINGS(profile,
            module.Factors("grid", "pooled", "residual"), solver, None)


@pytest.mark.parametrize("profile,base,solver,model", [
    ("unknown", "grid-pooled-residual", "centered", None),
    ("source-1800", "stride-pooled-residual", "centered", None),
    ("source-1800", "grid-native-residual", "centered", None),
    ("source-1800", "grid-pooled-settling", "centered", None),
    ("source-1800", "grid-pooled-residual", "symmetric", None),
    ("source-1800", "grid-pooled-residual", "centered", "measured-voxel"),
])
def test_nonclosed_roster_refused(module, profile, base, solver, model):
    with pytest.raises(ValueError):
        module.settings(profile, module.Factors(*base.split("-")), solver, model)


@pytest.mark.parametrize("profile", ["observe-900", "source-1800",
    "prepared-1800-source-900-retained-900", "prepared-1800-source-1800-retained-900",
    "prepared-1800-source-900-retained-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_exact_parameters_report_bytes_and_cleanup(
    module, monkeypatch, tmp_path, profile, static, failure
):
    adapter = importlib.import_module("scanner_capacity_split_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = replay.IncrementalTracker
    output = tmp_path / "output"
    content = b'{"original": "untouched"}'
    factors = module.Factors("grid", "pooled", "residual")
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
        assert module.scanner_source_capacity.settings(
            profile, factors, "centered", None
        ) == expected
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    arguments = dict(capacity_profile=profile, solver_profile="centered", static=static, frames=320)
    if failure:
        with pytest.raises(RuntimeError, match="injected failure"):
            adapter.run(tmp_path / "capture", output, factors, **arguments)
    else:
        adapter.run(tmp_path / "capture", output, factors, **arguments)
    assert not adapter._ACTIVE.get()
    assert module.scanner_source_capacity.settings is module.ORIGINAL_SETTINGS
    assert replay.IncrementalTracker is original_factory
    assert (output / "report.json").read_bytes() == content
    report = json.loads((output / "capacity-split-report.json").read_text())
    assert report["status"] == ("failed" if failure else "completed")
    assert report["capacity_settings_calls"] == 1
    assert report["tracking_policy"] == expected[1].as_dict()
    assert report["preparation_target_points"] == expected[0]
    assert report["diagnostic_source_profile"] == profile
    assert report["underlying_reports"] == [dict(path=str(output / "report.json"),
        sha256=hashlib.sha256(content).hexdigest())]
    assert not report["quality_qualification_allowed"]


@pytest.mark.parametrize("calls", [0, 2])
def test_missing_or_repeated_capacity_hook_fails_and_restores(module, monkeypatch, tmp_path, calls):
    adapter = importlib.import_module("scanner_capacity_split_replay")
    factors = module.Factors("grid", "pooled", "residual")

    def delegated(*args, **kwargs):
        for _ in range(calls):
            module.scanner_source_capacity.settings("observe-900", factors, "centered", None)

    monkeypatch.setattr(adapter.static_replay, "run_static", delegated)
    with pytest.raises(RuntimeError):
        adapter.run(tmp_path / "capture", tmp_path / "output", factors,
                    capacity_profile="observe-900", solver_profile="centered",
                    static=True, frames=320)
    assert not adapter._ACTIVE.get()
    assert module.scanner_source_capacity.settings is module.ORIGINAL_SETTINGS
