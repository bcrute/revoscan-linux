"""Conditional M27 historical framing; explicit doubles are not GPU qualification."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scanner3d.preparation import DepthBand
from scanner3d.tracking import IncrementalTracker
from tests.test_scanner_organized_model import curved_raw


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_organized_model_stage_b")


def options(framing="legacy-project", **changes):
    return (
        dict(
            normal_profile="organized-confidence",
            radius=5,
            maximum_weight=1.0,
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            framing=framing,
            frames=100,
        )
        | changes
    )


def install(module, monkeypatch, damage=None):
    adapter = importlib.import_module("scanner_organized_model")
    raw = curved_raw()

    def delegate(capture, output, **kwargs):
        factors = module.Factors("grid", "pooled", "residual")
        ex = adapter._ACTIVE.get()
        cloud = ex.prepare(raw, target_points=ex.target, depth_band=DepthBand(60, 300))
        tracker = IncrementalTracker(backend="scipy", policy=ex.policy)
        decision = tracker.offer(cloud.points)
        assert decision.accepted
        # Explicit model/seed metadata doubles; the actual numerical adapter is
        # tested separately. These rows are never geometric evidence.
        query = dict(
            direction="forward",
            seed_fit_ordinal=0,
            seed_stage_ordinal=0,
            neighbours=10,
            usable=20,
            actual_sha256="test-double",
            baseline=dict(usable=20, sha256="test-double"),
        )
        ex.models.append(dict(status="completed", queries=[query], unit_test_double=True))
        metadata = dict(
            status="completed",
            quality_qualification_allowed=False,
            tracking_policy=ex.policy.as_dict(),
            preparation_target_points=ex.target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=ex.source_profile,
            diagnostic_solver_profile="centered",
            diagnostic_model_profile=None,
            candidate_preference=["model", "adjacent", "keyframe"],
            offered=1,
            accepted=1,
            decisions=[
                dict(source_index=0, prepared_points=len(cloud.points), decision=decision.as_dict())
            ],
        )
        document = dict(metadata, input_frames=[dict(sha256=hashlib.sha256(raw).hexdigest())])
        document["runs"] = [dict(metadata)]
        seed = dict(
            status="completed",
            seed_profile=kwargs["seed_profile"],
            capacity_profile=kwargs["capacity_profile"],
            fit_records=[
                dict(
                    status="completed",
                    stages=[dict(direction="forward", status="completed", unit_test_double=True)],
                )
            ],
        )
        if damage == "policy":
            document["tracking_policy"] = {}
        elif damage == "input":
            document["input_frames"][0]["sha256"] = "wrong"
        elif damage == "qualification":
            document["quality_qualification_allowed"] = True
        elif damage == "preference":
            document["candidate_preference"] = ["adjacent", "model", "keyframe"]
        elif damage == "admission":
            document["runs"][0]["accepted"] = 2
        elif damage == "model":
            ex.models[0]["status"] = "running"
        elif damage == "no-query":
            ex.models[0]["queries"] = []
        elif damage == "accepted-index":
            ex.preparations[0]["accepted_index"] = 8
        elif damage == "seed":
            seed["fit_records"] = []
        elif damage == "stage":
            seed["fit_records"][0]["stages"][0]["direction"] = "reverse"
        elif damage == "availability":
            query["usable"] = 21
        elif damage == "fit-index":
            query["seed_fit_ordinal"] = -1
        elif damage == "stage-index":
            query["seed_stage_ordinal"] = True
        output.mkdir()
        (output / "report.json").write_text(json.dumps(document))
        (output / "reverse-seed-report.json").write_text(json.dumps(seed))
        if damage != "missing-stride":
            (output / "stride-1.json").write_text(json.dumps(document["runs"][0]))
        legacy = kwargs["framing"] == "legacy-project"
        framing_report = dict(
            status="completed",
            quality_qualification_allowed=False,
            framing=kwargs["framing"],
            capture=str(capture),
            requested_frames=kwargs["frames"],
            seed_profile=kwargs["seed_profile"],
            capacity_profile=kwargs["capacity_profile"],
            diagnostic_solver_profile="centered",
            configuration=factors.identifier,
            diagnostic_scene=not legacy,
            skip_complete=0 if legacy else 12,
            depth_band_override=None if legacy else [60.0, 300.0],
            preparation_target_points=ex.target,
            tracking_policy=ex.policy.as_dict(),
            underlying_reports=[],
            unit_test_double=True,
        )
        for name in ("report.json", "stride-1.json", "reverse-seed-report.json"):
            path = output / name
            if path.exists():
                framing_report["underlying_reports"].append(
                    dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                )
        if damage == "framing-band":
            framing_report["depth_band_override"] = [50.0, 300.0]
        elif damage == "framing-mode":
            framing_report["diagnostic_scene"] = legacy
        elif damage == "framing-status":
            framing_report["status"] = "failed"
        elif damage == "framing-hash":
            framing_report["underlying_reports"][0]["sha256"] = "wrong"
        elif damage == "framing-links":
            framing_report["underlying_reports"] = []
        elif damage == "framing-qualification":
            framing_report["quality_qualification_allowed"] = True
        elif damage == "framing-link-path":
            framing_report["underlying_reports"][0]["path"] = str(output.parent / "foreign.json")
        if damage != "missing-framing":
            (output / "framing-report.json").write_text(json.dumps(framing_report))

    monkeypatch.setattr(module, "ORIGINAL_REPLAY", delegate)
    monkeypatch.setattr(module.original, "run", delegate)


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("profile", ["observe-pooled", "organized-confidence"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("radius", [5, 10])
@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_all_closed_metadata_combinations(
    module, monkeypatch, tmp_path, framing, profile, seed, capacity, radius, weight
):
    install(module, monkeypatch)
    output = tmp_path / "output"
    module.run(
        tmp_path / "capture",
        output,
        **options(
            framing,
            normal_profile=profile,
            seed_profile=seed,
            capacity_profile=capacity,
            radius=radius,
            maximum_weight=weight,
        ),
    )
    report = json.loads((output / "organized-model-stage-b-report.json").read_bytes())
    assert report["status"] == "completed" and report["hooks_restored"]
    assert report["quality_qualification_allowed"] is False
    assert len(report["preparation_records"]) == len(report["model_records"]) == 1
    assert report["model_records"][0]["unit_test_double"]
    assert report["framing"] == framing and len(report["underlying_reports"]) == 4
    for row in report["underlying_reports"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    "damage",
    [
        "policy",
        "input",
        "qualification",
        "preference",
        "admission",
        "model",
        "no-query",
        "accepted-index",
        "seed",
        "stage",
        "availability",
        "missing-stride",
        "fit-index",
        "stage-index",
        "missing-framing",
        "framing-band",
        "framing-mode",
        "framing-status",
        "framing-hash",
        "framing-links",
        "framing-qualification",
        "framing-link-path",
    ],
)
def test_damaged_metadata_is_explicit_failed_evidence(module, monkeypatch, tmp_path, damage):
    install(module, monkeypatch, damage)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        module.run(tmp_path / "capture", output, **options())
    report = json.loads((output / "organized-model-stage-b-report.json").read_bytes())
    assert report["status"] == "failed" and report["hooks_restored"]


@pytest.mark.parametrize(
    "changes",
    [
        {"solver_profile": "observe-origin"},
        {"framing": "unknown"},
        {"frames": 99},
        {"frames": 3001},
        {"normal_profile": "unknown"},
        {"radius": 2},
        {"maximum_weight": 0},
        {"capacity_profile": "unknown"},
        {"seed_profile": "unknown"},
    ],
)
def test_unsupported_inputs_do_not_create_output(module, tmp_path, changes):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        module.run(tmp_path / "capture", output, **options(**changes))
    assert not output.exists()


@pytest.mark.parametrize("framing", ["diagnostic-raw", "legacy-project"])
@pytest.mark.parametrize("profile", ["observe-pooled", "organized-confidence"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
def test_actual_runners_with_raw_backed_preparation_and_cpu_normal_double(
    module, monkeypatch, tmp_path, framing, profile, seed
):
    """Original runner/seed/solver/tracker and actual raw grid; loaders/scores/GPU are doubles."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scanner3d import icp_plane

    raw = curved_raw()
    count = 100
    inputs = [dict(sha256=hashlib.sha256(raw).hexdigest()) for _ in range(count)]
    motion = module.original.motion
    loads = []

    @contextmanager
    def cpu_double(_owner, message):
        message("GPU-assisted tracking: UNIT TEST DOUBLE, CPU normals, no hardware")
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            yield

    def load_double(*args, **kwargs):
        loads.append((args, kwargs))
        return [], [raw] * count, DepthBand(60, 300), inputs

    monkeypatch.setattr(motion, "load_inputs", load_double)
    monkeypatch.setattr(motion, "live_normals", cpu_double)
    monkeypatch.setattr(
        motion,
        "check_coherence",
        lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: dict(unit_test_double=True)),
    )
    output = tmp_path / "output"
    module.run(
        tmp_path / "capture", output, **options(framing, normal_profile=profile, seed_profile=seed)
    )
    document = json.loads((output / "report.json").read_bytes())
    report = json.loads((output / "organized-model-stage-b-report.json").read_bytes())
    run = document["runs"][0]
    assert run["offered"] == run["accepted"] == count
    assert report["status"] == document["status"] == "completed"
    assert report["hooks_restored"] and len(report["preparation_records"]) == count
    assert report["model_records"] and all(
        r["status"] == "completed" for r in report["model_records"]
    )
    if profile == "observe-pooled":
        assert all(
            q["actual_sha256"] == q["baseline"]["sha256"]
            for r in report["model_records"]
            for q in r["queries"]
        )
    assert all(r["accepted_index"] == i for i, r in enumerate(report["preparation_records"]))
    legacy = framing == "legacy-project"
    assert len(loads) == 1
    assert loads[0][0][2:] == (0, None) if legacy else loads[0][0][2:] == (12, (60.0, 300.0))
    assert loads[0][1] == dict(diagnostic_scene=not legacy, start_offset=0)
    assert module.original.motion.run is module.original.ORIGINAL_MOTION
    assert not module.original._ACTIVE.get()


@pytest.mark.parametrize("guard", ["existing", "foreign", "nested"])
def test_existing_output_foreign_delegate_and_nesting(module, monkeypatch, tmp_path, guard):
    from contextlib import nullcontext

    output = tmp_path / "output"
    if guard == "existing":
        output.mkdir()
        (output / "keep").write_bytes(b"owner evidence")
    if guard == "foreign":
        monkeypatch.setattr(module.original, "run", lambda *a, **k: pytest.fail("foreign call"))
    scope = (
        module.OrganizedModelExperiment(
            "observe-pooled", "ordinary-900", "observe-forward", radius=5, maximum_weight=1.0
        ).active()
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
def test_partial_failure_keeps_original_bytes_and_restores(
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
        module.run(tmp_path / "capture", output, **options(framing))
    report = json.loads((output / "organized-model-stage-b-report.json").read_bytes())
    assert report["status"] == "failed" and report["hooks_restored"]
    assert "injected failure" in report["error"]
    assert (output / "report.json").read_bytes() == b'{"partial": true}'
    assert len(report["underlying_reports"]) == 3
    adapter = importlib.import_module("scanner_organized_model")
    assert adapter._ACTIVE.get() is None and adapter._FIT.get() is None
    assert not adapter._MODEL.get() and not module.original._ACTIVE.get()
    assert IncrementalTracker.offer is adapter.ORIGINAL_OFFER
    assert IncrementalTracker._candidate is adapter.ORIGINAL_CANDIDATE
    assert module.original.motion.run is failing


@pytest.mark.parametrize("absent", ["framing", "normal-profile", "reverse-seed"])
def test_cli_requires_explicit_profiles(module, monkeypatch, tmp_path, absent):
    calls = []
    monkeypatch.setattr(module, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay", str(tmp_path / "capture"), "--output", str(tmp_path / "output"),
        "--capacity", "ordinary-900", "--mechanism", "centered", "--frames", "100",
        "--radius", "5", "--maximum-weight", "1.0",
    ]
    required = {
        "framing": "legacy-project", "normal-profile": "organized-confidence",
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
