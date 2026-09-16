"""Historical framing wrapper; archived GPU routing checks remain separate."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_reverse_seed_stage_b")


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("profile", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_actual_adapter_preserves_only_declared_framing_change(
    module, monkeypatch, tmp_path, framing, profile, capacity, mode
):
    calls = []
    output = tmp_path / "output"
    legacy = framing == "legacy-project"
    factors = module.Factors("grid", "pooled", "residual")
    target, policy, source = module.settings(capacity, profile, factors, mode)
    cloud = tuple(
        (x * 2.0, y * 2.0, 100.0 + 0.02 * x * x + 0.06 * y * y + 0.01 * x * y)
        for x in range(-6, 7)
        for y in range(-6, 7)
    )

    def original(capture, destination, frames, strides, **kwargs):
        calls.append((capture, destination, frames, strides, kwargs))
        assert kwargs == dict(
            skip_complete=0 if legacy else 12,
            depth_band=None if legacy else (60.0, 300.0),
            diagnostic_scene=not legacy,
            normals_backend="gpu",
            diagnostic_factors=factors.values,
            diagnostic_solver_profile=mode,
            diagnostic_source_profile=source,
        )
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            result = tracking.icp_point_to_plane(
                cloud,
                cloud,
                backend="scipy",
                initial=Rigid.identity(),
                voxel_size=1.5,
                keep_fraction=0.8,
                max_correspondence_distance=12.0,
            )
        assert result.converged
        destination.mkdir()
        content = json.dumps(dict(diagnostic_scene=not legacy, exact_arguments=True)).encode()
        for name in ("report.json", "stride-1.json"):
            (destination / name).write_bytes(content)

    monkeypatch.setattr(module, "ORIGINAL_MOTION", original)
    monkeypatch.setattr(module.motion, "run", original)
    module.run(
        tmp_path / "capture",
        output,
        framing=framing,
        seed_profile=profile,
        capacity_profile=capacity,
        solver_profile=mode,
        frames=188,
    )
    assert len(calls) == 1 and calls[0][:4] == (tmp_path / "capture", output, 188, (1,))
    report = json.loads((output / "framing-report.json").read_bytes())
    seed = json.loads((output / "reverse-seed-report.json").read_bytes())
    assert report["status"] == seed["status"] == "completed"
    assert (
        report["diagnostic_scene"] is not legacy and report["tracking_policy"] == policy.as_dict()
    )
    assert report["preparation_target_points"] == target and len(
        report["legacy_delegations"]
    ) == int(legacy)
    assert seed["direction_schedule"] == (
        ["forward"] if profile == "observe-forward" else ["reverse", "forward"]
    )
    assert seed["fit_records"][0]["status"] == "completed"
    assert len(report["underlying_reports"]) == 3
    for item in report["underlying_reports"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert module.motion.run is original and not module._ACTIVE.get()
    assert not module.original._ACTIVE.get() and not report["quality_qualification_allowed"]


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
def test_failed_delegation_restores_scope_and_retains_report_bytes(
    module, monkeypatch, tmp_path, framing
):
    output = tmp_path / "output"

    def original(*args, **kwargs):
        output.mkdir()
        (output / "report.json").write_bytes(b'{"partial": true}')
        raise RuntimeError("injected delegate failure")

    monkeypatch.setattr(module, "ORIGINAL_MOTION", original)
    monkeypatch.setattr(module.motion, "run", original)
    with pytest.raises(RuntimeError, match="injected delegate failure"):
        module.run(
            tmp_path / "capture",
            output,
            framing=framing,
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            frames=188,
        )
    report = json.loads((output / "framing-report.json").read_bytes())
    assert report["status"] == "failed" and "injected delegate failure" in report["error"]
    assert (output / "report.json").read_bytes() == b'{"partial": true}'
    assert len(report["legacy_delegations"]) == int(framing == "legacy-project")
    assert (
        module.motion.run is original
        and not module._ACTIVE.get()
        and not module.original._ACTIVE.get()
    )
    for item in report["underlying_reports"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(framing="unknown"),
        dict(frames=99),
        dict(frames=3001),
        dict(seed_profile="unknown"),
        dict(capacity_profile="unknown"),
        dict(solver_profile="symmetric"),
    ],
)
def test_nonclosed_wrapper_inputs_create_nothing(module, tmp_path, changes):
    kwargs = dict(
        framing="legacy-project",
        seed_profile="reverse-forward",
        capacity_profile="ordinary-900",
        solver_profile="centered",
        frames=188,
    )
    with pytest.raises(ValueError):
        module.run(tmp_path / "capture", tmp_path / "output", **(kwargs | changes))
    assert not (tmp_path / "output").exists()


def test_nested_or_modified_runner_is_rejected(module, monkeypatch, tmp_path):
    kwargs = dict(
        framing="legacy-project",
        seed_profile="reverse-forward",
        capacity_profile="ordinary-900",
        solver_profile="centered",
        frames=188,
    )
    token = module._ACTIVE.set(True)
    try:
        with pytest.raises(ValueError):
            module.run(tmp_path / "capture", tmp_path / "output", **kwargs)
    finally:
        module._ACTIVE.reset(token)
    monkeypatch.setattr(module.motion, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="original motion runner"):
        module.run(tmp_path / "capture", tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


def test_cli_requires_explicit_historical_framing(module, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--reverse-seed",
        "reverse-forward",
        "--capacity",
        "ordinary-900",
        "--mechanism",
        "centered",
        "--frames",
        "188",
    ]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--framing", "legacy-project"])
    module.main()
    assert calls[0][1]["framing"] == "legacy-project" and calls[0][1]["frames"] == 188
