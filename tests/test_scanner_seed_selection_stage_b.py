"""Conditional M25 Stage B composition; test doubles are not GPU qualification."""

import hashlib
import importlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_seed_selection_stage_b")


def options(**changes):
    return (
        dict(
            selection_profile="adjacent-first",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            framing="legacy-project",
            frames=100,
        )
        | changes
    )


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("selection", ["observe-model-first", "adjacent-first"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_all_adapter_contracts(
    module, monkeypatch, tmp_path, framing, selection, seed, capacity, mode
):
    from tests.test_scanner_registration_audit import corner

    calls = []
    legacy = framing == "legacy-project"
    factors = module.Factors("grid", "pooled", "residual")
    target, policy, source = module.settings(capacity, seed, factors, mode)
    preference = list(module.SeedSelectionExperiment(selection).preference)

    def motion_double(capture, destination, count, strides, **kwargs):
        calls.append((capture, destination, count, strides, kwargs))
        assert kwargs == dict(
            skip_complete=0 if legacy else 12,
            depth_band=None if legacy else (60.0, 300.0),
            diagnostic_scene=not legacy,
            normals_backend="gpu",
            diagnostic_factors=factors.values,
            diagnostic_solver_profile=mode,
            diagnostic_source_profile=source,
        )
        assert list(tracking.PREFERENCE_ORDER) == preference
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            result = tracking.icp_point_to_plane(
                corner(),
                corner(),
                backend="scipy",
                initial=Rigid.identity(),
                voxel_size=1.5,
                keep_fraction=0.8,
                max_correspondence_distance=12.0,
            )
        assert result.converged
        metadata = dict(
            status="completed",
            tracking_policy=policy.as_dict(),
            preparation_target_points=target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=source,
            diagnostic_solver_profile=mode,
            diagnostic_model_profile=None,
            candidate_preference=preference,
            quality_qualification_allowed=False,
        )
        document = metadata | dict(runs=[dict(metadata)])
        destination.mkdir()
        (destination / "report.json").write_text(json.dumps(document))
        (destination / "stride-1.json").write_text(json.dumps(metadata))

    monkeypatch.setattr(module.original, "ORIGINAL_MOTION", motion_double)
    monkeypatch.setattr(module.original.motion, "run", motion_double)
    output = tmp_path / "output"
    module.run(
        tmp_path / "capture",
        output,
        **options(
            framing=framing,
            selection_profile=selection,
            seed_profile=seed,
            capacity_profile=capacity,
            solver_profile=mode,
        ),
    )
    assert len(calls) == 1 and calls[0][2:4] == (100, (1,))
    report = json.loads((output / "seed-selection-stage-b-report.json").read_bytes())
    assert report["status"] == "completed" and not report["quality_qualification_allowed"]
    assert report["preference_observations"] == [preference, preference]
    assert len(report["underlying_reports"]) == 4
    for item in report["underlying_reports"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    selection_module = importlib.import_module("scanner_seed_selection")
    assert tracking.PREFERENCE_ORDER is selection_module.ORIGINAL_PREFERENCE
    assert module.original.motion.run is motion_double
    assert not module.original._ACTIVE.get() and not selection_module._ACTIVE.get()


@pytest.mark.parametrize(
    "change",
    [
        dict(framing="unknown"),
        dict(frames=99),
        dict(frames=3001),
        dict(selection_profile="unknown"),
        dict(seed_profile="unknown"),
        dict(capacity_profile="unknown"),
        dict(solver_profile="symmetric"),
    ],
)
def test_nonclosed_input_creates_nothing(module, tmp_path, change):
    with pytest.raises(ValueError):
        module.run(tmp_path / "capture", tmp_path / "output", **options(**change))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("guard", ["existing", "foreign", "nested"])
def test_existing_outputs_foreign_delegate_and_nesting(module, monkeypatch, tmp_path, guard):
    output = tmp_path / "output"
    if guard == "existing":
        output.mkdir()
        (output / "keep").write_bytes(b"owner evidence")
    if guard == "foreign":
        monkeypatch.setattr(module.original, "run", lambda *a, **k: pytest.fail("foreign call"))
    scope = (
        module.SeedSelectionExperiment("adjacent-first").active()
        if guard == "nested"
        else nullcontext()
    )
    with scope, pytest.raises((ValueError, RuntimeError)):
        module.run(tmp_path / "capture", output, **options())
    if guard == "existing":
        assert list(output.iterdir()) == [output / "keep"]
        assert (output / "keep").read_bytes() == b"owner evidence"
    else:
        assert not output.exists()


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_failure_preserves_bytes_and_restores(
    module, monkeypatch, tmp_path, framing, error_type
):
    output = tmp_path / "output"

    def failing(*args, **kwargs):
        output.mkdir()
        (output / "report.json").write_bytes(b'{"partial": true}')
        raise error_type("injected failure")

    monkeypatch.setattr(module.original, "ORIGINAL_MOTION", failing)
    monkeypatch.setattr(module.original.motion, "run", failing)
    with pytest.raises(error_type, match="injected failure"):
        module.run(tmp_path / "capture", output, **options(framing=framing))
    report = json.loads((output / "seed-selection-stage-b-report.json").read_bytes())
    assert report["status"] == "failed" and "injected failure" in report["error"]
    assert (output / "report.json").read_bytes() == b'{"partial": true}'
    assert len(report["underlying_reports"]) == 3
    selection_module = importlib.import_module("scanner_seed_selection")
    assert tracking.PREFERENCE_ORDER is selection_module.ORIGINAL_PREFERENCE
    assert not selection_module._ACTIVE.get() and not module.original._ACTIVE.get()
    assert module.original.motion.run is failing


@pytest.mark.parametrize("absent", ["framing", "selection", "reverse-seed"])
def test_cli_requires_explicit_profiles(module, monkeypatch, tmp_path, absent):
    calls = []
    monkeypatch.setattr(module, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--capacity",
        "ordinary-900",
        "--mechanism",
        "centered",
        "--frames",
        "100",
    ]
    required = {
        "framing": "legacy-project",
        "selection": "adjacent-first",
        "reverse-seed": "reverse-forward",
    }
    monkeypatch.setattr(
        "sys.argv",
        args + [v for k, value in required.items() if k != absent for v in (f"--{k}", value)],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
    monkeypatch.setattr(
        "sys.argv", args + [v for k, value in required.items() for v in (f"--{k}", value)]
    )
    module.main()
    assert calls[0][1] == options()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-framing",
        "missing-stride",
        "policy",
        "run-preference",
        "qualification",
        "framing-qualification",
        "framing-band",
        "framing-mode",
        "framing-status",
        "seed-fits",
    ],
)
def test_damaged_evidence_cannot_complete(module, monkeypatch, tmp_path, damage):
    output = tmp_path / "output"
    factors = module.Factors("grid", "pooled", "residual")
    target, policy, source = module.settings("ordinary-900", "reverse-forward", factors, "centered")
    original_bytes = {}

    def damaged(capture, destination, **kwargs):
        document = dict(
            status="completed",
            tracking_policy=policy.as_dict(),
            preparation_target_points=target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=source,
            diagnostic_solver_profile="centered",
            diagnostic_model_profile=None,
            quality_qualification_allowed=False,
            candidate_preference=["adjacent", "model", "keyframe"],
        )
        document["runs"] = [dict(document)]
        framing = dict(
            status="completed",
            quality_qualification_allowed=False,
            framing="legacy-project",
            capture=str(capture),
            requested_frames=100,
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            diagnostic_solver_profile="centered",
            configuration=factors.identifier,
            diagnostic_scene=False,
            skip_complete=0,
            depth_band_override=None,
            preparation_target_points=target,
            tracking_policy=policy.as_dict(),
        )
        seed = dict(
            status="completed",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            fit_records=[dict(status="completed")],
        )
        if damage == "policy":
            document["tracking_policy"] = {}
        elif damage == "run-preference":
            document["runs"][0]["candidate_preference"] = []
        elif damage == "qualification":
            document["quality_qualification_allowed"] = True
        elif damage == "framing-qualification":
            framing["quality_qualification_allowed"] = True
        elif damage == "framing-band":
            framing["depth_band_override"] = [60.0, 300.0]
        elif damage == "framing-mode":
            framing["diagnostic_scene"] = True
        elif damage == "framing-status":
            framing["status"] = "failed"
        elif damage == "seed-fits":
            seed["fit_records"] = []
        reports = {
            "report.json": document,
            "stride-1.json": document["runs"][0],
            "framing-report.json": framing,
            "reverse-seed-report.json": seed,
        }
        if damage in ("missing-framing", "missing-stride"):
            del reports["framing-report.json" if damage == "missing-framing" else "stride-1.json"]
        destination.mkdir()
        for name, value in reports.items():
            original_bytes[name] = json.dumps(value).encode()
            (destination / name).write_bytes(original_bytes[name])

    monkeypatch.setattr(module, "ORIGINAL_REPLAY", damaged)
    monkeypatch.setattr(module.original, "run", damaged)
    with pytest.raises(RuntimeError):
        module.run(tmp_path / "capture", output, **options())
    report = json.loads((output / "seed-selection-stage-b-report.json").read_bytes())
    assert report["status"] == "failed"
    assert len(report["underlying_reports"]) == len(original_bytes)
    assert all((output / name).read_bytes() == raw for name, raw in original_bytes.items())


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("selection", ["observe-model-first", "adjacent-first"])
def test_actual_motion_runner_keeps_outer_preference(
    module, monkeypatch, tmp_path, framing, selection
):
    """Actual runner/tracker/fits; synthetic preparation and CPU-reference normals."""
    from scanner3d.preparation import DepthBand
    from tests.test_scanner_registration_audit import corner

    motion = module.original.motion
    assert motion.run is module.original.ORIGINAL_MOTION
    capture = tmp_path / "capture"
    cloud = SimpleNamespace(points=tuple(corner()), deprojected_points=675, clipped_points=675)
    band = DepthBand(60.0, 300.0)
    calls = []

    def load_double(*args, **kwargs):
        calls.append((args, kwargs))
        # Payloads and points deliberately are independent synthetic test inputs.
        # This tests routing, not deprojection, geometry scores, or GPU hardware.
        return (
            [capture / str(i) for i in range(100)],
            [bytes(512000)] * 100,
            band,
            [dict(index=i) for i in range(100)],
        )

    @contextmanager
    def cpu_reference_double(_owner, message):
        message("GPU-assisted tracking: UNIT TEST DOUBLE, CPU reference, no hardware")
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            yield

    grid = importlib.import_module("scanner_pixel_grid")
    monkeypatch.setattr(motion, "load_inputs", load_double)
    monkeypatch.setattr(motion, "live_normals", cpu_reference_double)
    monkeypatch.setattr(grid, "prepare_frame", lambda *a, **k: cloud)
    monkeypatch.setattr(
        motion,
        "check_coherence",
        lambda *a, **k: SimpleNamespace(
            as_dict=lambda: dict(unit_test_double=True, physical_geometry_claim=False)
        ),
    )
    output = tmp_path / "output"
    module.run(capture, output, **options(framing=framing, selection_profile=selection))
    assert calls == [
        (
            (
                capture,
                100,
                0 if framing == "legacy-project" else 12,
                None if framing == "legacy-project" else (60.0, 300.0),
            ),
            dict(diagnostic_scene=framing != "legacy-project", start_offset=0),
        )
    ]
    report = json.loads((output / "report.json").read_bytes())
    run = report["runs"][0]
    preference = list(module.SeedSelectionExperiment(selection).preference)
    assert report["candidate_preference"] == run["candidate_preference"] == preference
    assert run["accepted"] == run["offered"] == 100
    assert len(run["decisions"]) == len(run["poses"]) == 100
    assert not report["quality_qualification_allowed"]
    assert any("UNIT TEST DOUBLE" in m for m in run["messages"])
    assert motion.run is module.original.ORIGINAL_MOTION
