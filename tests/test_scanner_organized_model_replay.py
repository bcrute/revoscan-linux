"""Replay metadata contracts with labelled doubles, not recorded/GPU qualification."""

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
    return importlib.import_module("scanner_organized_model_replay")


def options(static=False, **changes):
    return (
        dict(
            normal_profile="organized-confidence",
            radius=5,
            maximum_weight=1.0,
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


def install(module, monkeypatch, damage=None):
    adapter = importlib.import_module("scanner_organized_model")
    raw = curved_raw()

    def delegate(capture, output, factors, **kwargs):
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
        if not kwargs["static"]:
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
            (document if kwargs["static"] else document["runs"][0])["accepted"] = 2
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
        if not kwargs["static"] and damage != "missing-stride":
            (output / "stride-1.json").write_text(json.dumps(document["runs"][0]))

    monkeypatch.setattr(module, "ORIGINAL_REPLAY", delegate)
    monkeypatch.setattr(module.original, "run", delegate)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("profile", ["observe-pooled", "organized-confidence"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("radius", [5, 10])
@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_all_closed_metadata_combinations(
    module, monkeypatch, tmp_path, static, profile, seed, capacity, radius, weight
):
    install(module, monkeypatch)
    output = tmp_path / "output"
    module.run(
        tmp_path / "capture",
        output,
        **options(
            static,
            normal_profile=profile,
            seed_profile=seed,
            capacity_profile=capacity,
            radius=radius,
            maximum_weight=weight,
        ),
    )
    report = json.loads((output / "organized-model-report.json").read_bytes())
    assert report["status"] == "completed" and report["hooks_restored"]
    assert report["quality_qualification_allowed"] is False
    assert len(report["preparation_records"]) == len(report["model_records"]) == 1
    assert report["model_records"][0]["unit_test_double"]
    assert len(report["underlying_reports"]) == (2 if static else 3)
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
    ],
)
def test_damaged_metadata_is_explicit_failed_evidence(module, monkeypatch, tmp_path, damage):
    install(module, monkeypatch, damage)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        module.run(tmp_path / "capture", output, **options())
    report = json.loads((output / "organized-model-report.json").read_bytes())
    assert report["status"] == "failed" and report["hooks_restored"]


@pytest.mark.parametrize(
    "changes",
    [
        {"solver_profile": "observe-origin"},
        {"skip_complete": 0},
        {"depth_band": (50.0, 300.0)},
        {"frames": 99},
        {"frames": 3001},
        {"normal_profile": "unknown"},
        {"radius": 2},
        {"maximum_weight": 0},
        {"capacity_profile": "unknown"},
        {"seed_profile": "unknown"},
        {"static": True, "frames": 100},
    ],
)
def test_unsupported_inputs_do_not_create_output(module, tmp_path, changes):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        module.run(tmp_path / "capture", output, **options(**changes))
    assert not output.exists()


@pytest.mark.parametrize("static", [False, True])
def test_actual_runners_with_raw_backed_preparation_and_cpu_normal_double(
    module, monkeypatch, tmp_path, static
):
    """Original runner/seed/solver/tracker and actual raw grid; loaders/scores/GPU are doubles."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scanner3d import icp_plane

    raw = curved_raw()
    count = 320 if static else 100
    inputs = [dict(sha256=hashlib.sha256(raw).hexdigest()) for _ in range(count)]
    stationary, motion = module.original.static_replay, module.original.motion_replay

    @contextmanager
    def cpu_double(_owner, message):
        message("GPU-assisted tracking: UNIT TEST DOUBLE, CPU normals, no hardware")
        with icp_plane.normal_acceleration(
            lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
        ):
            yield

    monkeypatch.setattr(stationary, "load_static", lambda *args: ([raw] * count, inputs, {}))
    monkeypatch.setattr(stationary, "static_checks", lambda *args: [dict(unit_test_double=True)])
    monkeypatch.setattr(stationary, "live_normals", cpu_double)
    monkeypatch.setattr(
        motion,
        "load_inputs",
        lambda *args, **kwargs: ([], [raw] * count, DepthBand(60, 300), inputs),
    )
    monkeypatch.setattr(motion, "live_normals", cpu_double)
    monkeypatch.setattr(
        motion,
        "check_coherence",
        lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: dict(unit_test_double=True)),
    )
    output = tmp_path / "output"
    module.run(tmp_path / "capture", output, **options(static, normal_profile="observe-pooled"))
    document = json.loads((output / "report.json").read_bytes())
    report = json.loads((output / "organized-model-report.json").read_bytes())
    run = document if static else document["runs"][0]
    assert run["offered"] == run["accepted"] == count
    assert report["status"] == document["status"] == "completed"
    assert report["hooks_restored"] and len(report["preparation_records"]) == count
    assert report["model_records"] and all(
        r["status"] == "completed" for r in report["model_records"]
    )
    assert all(
        q["actual_sha256"] == q["baseline"]["sha256"]
        for r in report["model_records"]
        for q in r["queries"]
    )
    assert all(r["accepted_index"] == i for i, r in enumerate(report["preparation_records"]))
