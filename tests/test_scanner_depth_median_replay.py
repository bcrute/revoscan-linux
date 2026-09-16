"""M26 metadata/adapter composition; synthetic tests are not recorded-GPU qualification."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scanner3d.preparation import DepthBand


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_depth_median_replay")


def options(static=False, **changes):
    return (
        dict(
            preparation_profile="valid-nine-median3",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=static,
            frames=320 if static else 100,
            skip_complete=0 if static else 12,
            depth_band=None if static else (60.0, 300.0),
        )
        | changes
    )


def install_delegate(module, monkeypatch, damage=None):
    depth = importlib.import_module("scanner_depth_median")
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()

    def delegate(capture, output, factors, **kwargs):
        static = kwargs["static"]
        target, policy, source = module.settings(
            kwargs["capacity_profile"], kwargs["seed_profile"], factors, kwargs["solver_profile"]
        )
        prepare = depth.static_replay.prepare_grid if static else depth.grid.prepare_frame
        cloud = prepare(raw, target_points=target, depth_band=DepthBand(60, 300))
        metadata = dict(
            status="completed",
            quality_qualification_allowed=False,
            tracking_policy=policy.as_dict(),
            preparation_target_points=target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=source,
            diagnostic_solver_profile=kwargs["solver_profile"],
            diagnostic_model_profile=None,
            candidate_preference=["model", "adjacent", "keyframe"],
            offered=1,
            accepted=1,
            decisions=[dict(source_index=0, prepared_points=len(cloud.points))],
        )
        document = dict(metadata, input_frames=[dict(sha256=hashlib.sha256(raw).hexdigest())])
        if not static:
            document["runs"] = [dict(metadata)]
        seed = dict(
            status="completed",
            seed_profile=kwargs["seed_profile"],
            capacity_profile=kwargs["capacity_profile"],
            fit_records=[dict(status="completed")],
        )
        if damage == "policy":
            document["tracking_policy"] = {}
        elif damage == "input":
            document["input_frames"][0]["sha256"] = "wrong"
        elif damage == "preference":
            document["candidate_preference"] = ["adjacent", "model", "keyframe"]
        elif damage == "qualification":
            document["quality_qualification_allowed"] = True
        elif damage == "count":
            (document if static else document["runs"][0])["offered"] = 2
        elif damage == "points":
            (document if static else document["runs"][0])["decisions"][0]["prepared_points"] = 0
        elif damage == "seed":
            seed["fit_records"] = []
        output.mkdir()
        (output / "report.json").write_text(json.dumps(document))
        (output / "reverse-seed-report.json").write_text(json.dumps(seed))
        if not static and damage != "missing-stride":
            (output / "stride-1.json").write_text(json.dumps(document["runs"][0]))

    monkeypatch.setattr(module, "ORIGINAL_REPLAY", delegate)
    monkeypatch.setattr(module.original, "run", delegate)


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("profile", ["observe-raw", "valid-nine-median3"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_all_adapter_contracts(
    module, monkeypatch, tmp_path, static, profile, seed, capacity, mode
):
    install_delegate(module, monkeypatch)
    output = tmp_path / "output"
    module.run(
        tmp_path / "capture",
        output,
        **options(
            static,
            preparation_profile=profile,
            seed_profile=seed,
            capacity_profile=capacity,
            solver_profile=mode,
        ),
    )
    report = json.loads((output / "depth-median-report.json").read_bytes())
    assert report["status"] == "completed" and not report["quality_qualification_allowed"]
    assert len(report["preparation_records"]) == 1
    assert len(report["underlying_reports"]) == (2 if static else 3)
    for row in report["underlying_reports"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    "damage",
    ["policy", "input", "preference", "qualification", "count", "points", "seed", "missing-stride"],
)
def test_damaged_metadata_is_failed_evidence(module, monkeypatch, tmp_path, damage):
    install_delegate(module, monkeypatch, damage)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        module.run(tmp_path / "capture", output, **options())
    report = json.loads((output / "depth-median-report.json").read_bytes())
    assert report["status"] == "failed" and len(report["preparation_records"]) == 1


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("profile", ["observe-raw", "valid-nine-median3"])
def test_actual_runners_with_numerical_tracker_and_synthetic_preparation(
    module, monkeypatch, tmp_path, static, profile
):
    """Actual runner/seed/solver/tracker; CPU-normal and preparation/score doubles."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scanner3d import icp_plane
    from tests.test_scanner_registration_audit import corner

    depth = importlib.import_module("scanner_depth_median")
    motion = module.original.motion_replay
    stationary = module.original.static_replay
    count = 320 if static else 100
    raw_array = np.full((400, 640), 20000, dtype="<u2")
    raw_array[200, 320] = 20100
    raw = raw_array.tobytes()
    inputs = [dict(sha256=hashlib.sha256(raw).hexdigest()) for _ in range(count)]
    cloud = SimpleNamespace(
        points=tuple(corner()), deprojected_points=256000, clipped_points=256000
    )
    seen = []

    def grid_double(derived, **kwargs):
        seen.append(hashlib.sha256(derived).hexdigest())
        return cloud

    @contextmanager
    def cpu_reference_double(_owner, message):
        message("GPU-assisted tracking: UNIT TEST DOUBLE, CPU reference, no hardware")
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            yield

    monkeypatch.setattr(depth, "ORIGINAL_GRID", grid_double)
    monkeypatch.setattr(depth.grid, "prepare_frame", grid_double)
    monkeypatch.setattr(stationary, "prepare_grid", grid_double)
    monkeypatch.setattr(stationary, "load_static", lambda *a: ([raw] * count, inputs, {}))
    monkeypatch.setattr(stationary, "static_checks", lambda *a: [dict(unit_test_double=True)])
    monkeypatch.setattr(stationary, "live_normals", cpu_reference_double)
    monkeypatch.setattr(
        motion, "load_inputs", lambda *a, **k: ([], [raw] * count, DepthBand(60, 300), inputs)
    )
    monkeypatch.setattr(motion, "live_normals", cpu_reference_double)
    monkeypatch.setattr(
        motion,
        "check_coherence",
        lambda *a, **k: SimpleNamespace(as_dict=lambda: dict(unit_test_double=True)),
    )
    output = tmp_path / "output"
    module.run(tmp_path / "capture", output, **options(static, preparation_profile=profile))
    document = json.loads((output / "report.json").read_bytes())
    report = json.loads((output / "depth-median-report.json").read_bytes())
    result = document if static else document["runs"][0]
    assert report["status"] == document["status"] == "completed"
    assert result["offered"] == result["accepted"] == count
    assert len(report["preparation_records"]) == len(seen) == count
    assert all(
        r["derived_raw_sha256"] == s
        for r, s in zip(report["preparation_records"], seen, strict=True)
    )
    if profile == "observe-raw":
        assert set(seen) == {hashlib.sha256(raw).hexdigest()}
    else:
        assert all(r["filtering"]["changed_pixels"] == 1 for r in report["preparation_records"])
        assert all(s != hashlib.sha256(raw).hexdigest() for s in seen)
    seed = json.loads((output / "reverse-seed-report.json").read_bytes())
    assert seed["status"] == "completed" and len(seed["fit_records"]) > 0
    assert depth.grid.prepare_frame is stationary.prepare_grid is grid_double
    assert depth._ACTIVE.get() is None


@pytest.mark.parametrize(
    "change",
    [
        dict(preparation_profile="unknown"),
        dict(seed_profile="unknown"),
        dict(capacity_profile="unknown"),
        dict(solver_profile="symmetric"),
        dict(frames=99),
        dict(frames=3001),
        dict(skip_complete=0),
        dict(depth_band=(61, 300)),
        dict(static=True),
        dict(static=True, frames=320, skip_complete=0),
    ],
)
def test_closed_inputs_do_not_create_output(module, tmp_path, change):
    with pytest.raises(ValueError):
        module.run(tmp_path / "capture", tmp_path / "output", **(options() | change))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", [RuntimeError, KeyboardInterrupt])
def test_partial_adapter_failure_keeps_report_bytes(module, monkeypatch, tmp_path, kind):
    output = tmp_path / "output"

    def failing(*args, **kwargs):
        output.mkdir()
        (output / "report.json").write_bytes(b'{"partial":true}')
        raise kind("injected replay failure")

    monkeypatch.setattr(module, "ORIGINAL_REPLAY", failing)
    monkeypatch.setattr(module.original, "run", failing)
    with pytest.raises(kind, match="injected replay failure"):
        module.run(tmp_path / "capture", output, **options())
    report = json.loads((output / "depth-median-report.json").read_bytes())
    assert report["status"] == "failed" and len(report["underlying_reports"]) == 1
    assert (output / "report.json").read_bytes() == b'{"partial":true}'
    depth = importlib.import_module("scanner_depth_median")
    assert depth.grid.prepare_frame is depth.static_replay.prepare_grid is depth.ORIGINAL_GRID


@pytest.mark.parametrize("guard", ["existing", "delegate", "preference", "nested"])
def test_foreign_or_existing_state_is_rejected(module, monkeypatch, tmp_path, guard):
    from contextlib import nullcontext

    output = tmp_path / "output"
    if guard == "existing":
        output.mkdir()
        (output / "keep").write_bytes(b"owner evidence")
    if guard == "delegate":
        monkeypatch.setattr(module.original, "run", lambda *a, **k: pytest.fail("foreign delegate"))
    if guard == "preference":
        monkeypatch.setattr(module.tracking, "PREFERENCE_ORDER", ("adjacent", "model", "keyframe"))
    scope = (
        module.DepthMedianExperiment("observe-raw").active() if guard == "nested" else nullcontext()
    )
    with scope, pytest.raises((ValueError, RuntimeError)):
        module.run(tmp_path / "capture", output, **options())
    if guard == "existing":
        assert list(output.iterdir()) == [output / "keep"]
        assert (output / "keep").read_bytes() == b"owner evidence"
    else:
        assert not output.exists()


@pytest.mark.parametrize("absent", ["preparation", "reverse-seed"])
def test_cli_requires_both_profiles(module, monkeypatch, tmp_path, absent):
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
        "--skip-complete",
        "12",
        "--depth-band",
        "60",
        "300",
    ]
    required = {"preparation": "valid-nine-median3", "reverse-seed": "reverse-forward"}
    monkeypatch.setattr(
        "sys.argv",
        args + [v for key, value in required.items() if key != absent for v in (f"--{key}", value)],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
    monkeypatch.setattr(
        "sys.argv", args + [v for key, value in required.items() for v in (f"--{key}", value)]
    )
    module.main()
    assert calls[0][1] == options()
