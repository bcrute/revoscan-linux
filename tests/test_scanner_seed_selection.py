"""M25 changes a preference, never the original evidence required for agreement."""

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scanner3d import tracking
from scanner3d.geometry import IDENTITY_MATRIX, Rigid
from tests.test_tracking import _candidate_at, _grid, _state


@pytest.mark.parametrize("selection", ["observe-model-first", "adjacent-first"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("mode", ["observe-origin", "centered"])
def test_real_tracker_composition_exact(module, selection, seed, capacity, mode):
    from unittest.mock import patch

    from scanner3d import icp_plane
    from tests.test_scanner_registration_audit import corner

    reverse = importlib.import_module("scanner_reverse_seed")
    solvers = importlib.import_module("scanner_solver_mechanisms")
    factors = reverse.Factors("grid", "pooled", "residual")
    _, policy, _ = reverse.settings(capacity, seed, factors, mode)
    points = tuple(corner())
    selection_experiment = module.SeedSelectionExperiment(selection)
    before_seed, after_seed = (
        reverse.ReverseSeedExperiment(seed),
        reverse.ReverseSeedExperiment(seed),
    )
    before_solver, after_solver = solvers.SolverExperiment(mode), solvers.SolverExperiment(mode)
    before, after = (
        tracking.IncrementalTracker(backend="scipy", policy=policy),
        tracking.IncrementalTracker(backend="scipy", policy=policy),
    )

    def provider(cloud, neighbours):
        return icp_plane.estimate_normals(cloud, neighbours=neighbours, backend="scipy")

    with (
        icp_plane.normal_acceleration(provider),
        before_seed.active(),
        before_solver.active(),
        patch.object(tracking, "PREFERENCE_ORDER", selection_experiment.preference),
    ):
        expected = [before.offer(points).as_dict() for _ in range(4)]
    with (
        icp_plane.normal_acceleration(provider),
        after_seed.active(),
        after_solver.active(),
        selection_experiment.active(),
    ):
        actual = [after.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(d["accepted"] for d in actual)
    assert (
        before_seed.records == after_seed.records and before_solver.records == after_solver.records
    )
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_seed_selection")


def test_closed_profile(module):
    with pytest.raises(ValueError, match="closed selection"):
        module.SeedSelectionExperiment("lowest-residual")


@pytest.mark.parametrize("missing", ["selection", "reverse-seed"])
def test_cli_requires_both_explicit_profiles(module, monkeypatch, tmp_path, missing):
    adapter = importlib.import_module("scanner_seed_selection_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "test",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--capacity",
        "ordinary-900",
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
    pairs = {
        "selection": ["--selection", "adjacent-first"],
        "reverse-seed": ["--reverse-seed", "reverse-forward"],
    }
    absent = args + [arg for name, values in pairs.items() if name != missing for arg in values]
    monkeypatch.setattr("sys.argv", absent)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", args + pairs["selection"] + pairs["reverse-seed"])
    adapter.main()
    assert len(calls) == 1 and calls[0][1]["selection_profile"] == "adjacent-first"
    assert calls[0][1]["seed_profile"] == "reverse-forward"
    assert calls[0][1]["depth_band"] == (60.0, 300.0) and calls[0][1]["skip_complete"] == 12


@pytest.mark.parametrize(
    "guard", ["existing-output", "foreign-delegate", "nested-observation", "nested-change"]
)
def test_adapter_never_overwrites_or_nests(module, monkeypatch, tmp_path, guard):
    from contextlib import nullcontext

    adapter = importlib.import_module("scanner_seed_selection_replay")
    output = tmp_path / "output"
    if guard == "existing-output":
        output.mkdir()
        (output / "keep").write_bytes(b"owner evidence")
    if guard == "foreign-delegate":
        monkeypatch.setattr(
            adapter.original, "run", lambda *a, **k: pytest.fail("foreign delegate called")
        )
    scope = (
        module.SeedSelectionExperiment(
            "observe-model-first" if guard == "nested-observation" else "adjacent-first"
        ).active()
        if guard.startswith("nested")
        else nullcontext()
    )
    with scope, pytest.raises((ValueError, RuntimeError)):
        adapter.run(
            tmp_path / "capture",
            output,
            selection_profile="adjacent-first",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=True,
            frames=320,
        )
    assert (
        (
            sorted(p.name for p in output.iterdir()) == ["keep"]
            and (output / "keep").read_bytes() == b"owner evidence"
        )
        if guard == "existing-output"
        else not output.exists()
    )
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-stride",
        "policy",
        "qualification",
        "preference",
        "run-preference",
        "seed",
        "missing-fits",
        "source-profile",
        "status",
    ],
)
def test_corrupt_reports_never_earn_completed_status(module, monkeypatch, tmp_path, damage):
    adapter = importlib.import_module("scanner_seed_selection_replay")
    output = tmp_path / "output"
    factors = adapter.Factors("grid", "pooled", "residual")
    target, policy, source = adapter.settings(
        "ordinary-900", "reverse-forward", factors, "centered"
    )
    expected_order = list(module.SeedSelectionExperiment("adjacent-first").preference)
    original_bytes = {}

    def damaged(capture, destination, *args, **kwargs):
        destination.mkdir()
        document = dict(
            status="completed",
            tracking_policy=policy.as_dict(),
            quality_qualification_allowed=False,
            preparation_target_points=target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=source,
            diagnostic_solver_profile="centered",
            diagnostic_model_profile=None,
            candidate_preference=expected_order,
        )
        document["runs"] = [dict(document)]
        seed = dict(
            status="completed",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            fit_records=[dict(status="completed")],
        )
        if damage == "policy":
            document["tracking_policy"] = {}
        elif damage == "qualification":
            document["quality_qualification_allowed"] = True
        elif damage == "preference":
            document["candidate_preference"] = list(module.ORIGINAL_PREFERENCE)
        elif damage == "run-preference":
            document["runs"][0]["candidate_preference"] = []
        elif damage == "seed":
            seed["seed_profile"] = "observe-forward"
        elif damage == "missing-fits":
            seed["fit_records"] = []
        elif damage == "source-profile":
            document["diagnostic_source_profile"] = "source-1800"
        elif damage == "status":
            document["status"] = "failed"
        for name, value in (
            ("report.json", document),
            ("stride-1.json", document),
            ("reverse-seed-report.json", seed),
        ):
            if damage == "missing-stride" and name == "stride-1.json":
                continue
            raw = json.dumps(value).encode()
            (destination / name).write_bytes(raw)
            original_bytes[name] = raw

    monkeypatch.setattr(adapter, "ORIGINAL_REPLAY", damaged)
    monkeypatch.setattr(adapter.original, "run", damaged)
    with pytest.raises(RuntimeError):
        adapter.run(
            tmp_path / "capture",
            output,
            selection_profile="adjacent-first",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=False,
            frames=628,
            skip_complete=12,
            depth_band=(60.0, 300.0),
        )
    report = json.loads((output / "seed-selection-report.json").read_bytes())
    assert report["status"] == "failed" and not report["quality_qualification_allowed"]
    assert len(report["underlying_reports"]) == len(original_bytes)
    for name, raw in original_bytes.items():
        assert (output / name).read_bytes() == raw
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize(
    "changes",
    [
        dict(selection_profile="oracle"),
        dict(seed_profile="third-pass"),
        dict(capacity_profile="3600"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60.0, 300.0)),
        dict(static=False, frames=628),
        dict(static=False, frames=99, skip_complete=12, depth_band=(60.0, 300.0)),
        dict(static=False, frames=3001, skip_complete=12, depth_band=(60.0, 300.0)),
    ],
)
def test_adapter_nonclosed_options_create_nothing(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_seed_selection_replay")
    options = dict(
        selection_profile="adjacent-first",
        seed_profile="reverse-forward",
        capacity_profile="ordinary-900",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "capture", tmp_path / "output", **(options | changes))
    assert not (tmp_path / "output").exists()
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize("static", [False, True])
def test_actual_delegate_failure_preserves_partial_reports(module, monkeypatch, tmp_path, static):
    adapter = importlib.import_module("scanner_seed_selection_replay")
    replay = adapter.original.static_replay if static else adapter.original.motion_replay
    output = tmp_path / "output"
    content = b'{"partial_original_evidence": true}'

    def failed(*args, **kwargs):
        output.mkdir()
        (output / "report.json").write_bytes(content)
        raise RuntimeError("injected delegate failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", failed)
    with pytest.raises(RuntimeError, match="injected delegate failure"):
        adapter.run(
            tmp_path / "capture",
            output,
            selection_profile="adjacent-first",
            seed_profile="reverse-forward",
            capacity_profile="ordinary-900",
            solver_profile="centered",
            static=static,
            frames=320 if static else 628,
            skip_complete=0 if static else 12,
            depth_band=None if static else (60.0, 300.0),
        )
    report = json.loads((output / "seed-selection-report.json").read_bytes())
    assert report["status"] == "failed" and "injected delegate failure" in report["error"]
    assert (output / "report.json").read_bytes() == content
    assert len(report["preference_observations"]) == 1
    assert len(report["underlying_reports"]) == 2
    for item in report["underlying_reports"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()
    assert not adapter.original._ACTIVE.get()


@pytest.mark.parametrize("error", [RuntimeError("test"), KeyboardInterrupt(), SystemExit(7)])
def test_every_exception_restores_original_scope(module, error):
    with pytest.raises(type(error)), module.SeedSelectionExperiment("adjacent-first").active():
        raise error
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize("profile", ["observe-model-first", "adjacent-first"])
def test_nested_and_foreign_scopes_refused(module, monkeypatch, profile):
    experiment = module.SeedSelectionExperiment(profile)
    with experiment.active():
        with (
            pytest.raises(RuntimeError, match="nonnested"),
            module.SeedSelectionExperiment(profile).active(),
        ):
            pytest.fail("nested preference was admitted")
        assert tracking.PREFERENCE_ORDER is experiment.preference
    foreign = tuple(reversed(module.ORIGINAL_PREFERENCE))
    monkeypatch.setattr(tracking, "PREFERENCE_ORDER", foreign)
    with pytest.raises(RuntimeError, match="original preference"), experiment.active():
        pytest.fail("foreign preference was overwritten")
    assert tracking.PREFERENCE_ORDER is foreign and not module._ACTIVE.get()


def test_mutation_during_scope_is_reported_and_restored(module):
    with (
        pytest.raises(RuntimeError, match="changed during"),
        module.SeedSelectionExperiment("adjacent-first").active(),
    ):
        tracking.PREFERENCE_ORDER = tuple(reversed(module.ORIGINAL_PREFERENCE))
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("selection", ["observe-model-first", "adjacent-first"])
@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
def test_actual_adapter_truthful_reports(
    module, monkeypatch, tmp_path, static, selection, seed, capacity
):
    from scanner3d import icp_plane
    from tests.test_scanner_registration_audit import corner

    adapter = importlib.import_module("scanner_seed_selection_replay")
    solvers = importlib.import_module("scanner_solver_mechanisms")
    factors = adapter.Factors("grid", "pooled", "residual")
    target, policy, source = adapter.settings(capacity, seed, factors, "centered")
    output = tmp_path / "output"
    expected_order = list(module.SeedSelectionExperiment(selection).preference)
    calls = []

    def delegated(capture, destination, *args, **kwargs):
        calls.append((capture, destination, args, kwargs))
        assert list(tracking.PREFERENCE_ORDER) == expected_order
        assert adapter.original.static_replay.IncrementalTracker is tracking.IncrementalTracker
        assert adapter.original.motion_replay.IncrementalTracker is tracking.IncrementalTracker
        expected = dict(diagnostic_solver_profile="centered", diagnostic_source_profile=source)
        if static:
            assert args == (factors,)
        else:
            assert args == (628, (1,))
            expected.update(
                skip_complete=12,
                depth_band=(60.0, 300.0),
                diagnostic_scene=True,
                normals_backend="gpu",
                diagnostic_factors=factors.values,
            )
        assert kwargs == expected
        points = tuple(corner())
        with (
            icp_plane.normal_acceleration(
                lambda p, k: icp_plane.estimate_normals(p, neighbours=k, backend="scipy")
            ),
            solvers.SolverExperiment("centered").active(),
        ):
            tracking.icp_point_to_plane(
                points,
                points,
                backend="scipy",
                initial=Rigid.identity(),
                voxel_size=1.5,
                keep_fraction=0.8,
                max_correspondence_distance=12.0,
            )
        document = dict(
            status="completed",
            tracking_policy=policy.as_dict(),
            quality_qualification_allowed=False,
            preparation_target_points=target,
            diagnostic_factors=factors.as_dict(),
            diagnostic_source_profile=source,
            diagnostic_solver_profile="centered",
            diagnostic_model_profile=None,
        )
        if not static:
            document["candidate_preference"] = expected_order
            document["runs"] = [dict(document)]
        destination.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (destination / name).write_text(json.dumps(document))

    replay = adapter.original.static_replay if static else adapter.original.motion_replay
    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    adapter.run(
        tmp_path / "capture",
        output,
        selection_profile=selection,
        seed_profile=seed,
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320 if static else 628,
        skip_complete=0 if static else 12,
        depth_band=None if static else (60.0, 300.0),
    )
    report = json.loads((output / "seed-selection-report.json").read_bytes())
    seed_report = json.loads((output / "reverse-seed-report.json").read_bytes())
    assert len(calls) == 1 and report["status"] == seed_report["status"] == "completed"
    assert report["preference_observations"] == [expected_order, expected_order]
    assert report["candidate_preference"] == expected_order
    assert report["original_static_preference_field_present"] is (False if static else None)
    assert len(report["underlying_reports"]) == (2 if static else 3)
    for item in report["underlying_reports"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()
    assert not adapter.original._ACTIVE.get() and not report["quality_qualification_allowed"]


@pytest.mark.parametrize("profile", ["observe-model-first", "adjacent-first"])
def test_original_functions_and_policy_untouched(module, profile):
    original = (
        tracking.IncrementalTracker.offer,
        tracking.IncrementalTracker._decide,
        tracking.icp_point_to_plane,
        tracking.DEFAULT_POLICY.as_dict(),
    )
    experiment = module.SeedSelectionExperiment(profile)
    with experiment.active():
        assert tracking.PREFERENCE_ORDER is experiment.preference
        assert (
            tracking.IncrementalTracker.offer,
            tracking.IncrementalTracker._decide,
            tracking.icp_point_to_plane,
            tracking.DEFAULT_POLICY.as_dict(),
        ) == original
        if profile == "observe-model-first":
            assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE
    assert tracking.PREFERENCE_ORDER is module.ORIGINAL_PREFERENCE and not module._ACTIVE.get()


@pytest.mark.parametrize("profile", ["observe-model-first", "adjacent-first"])
@pytest.mark.parametrize("case", ["agree", "ambiguous", "tighter", "model-only"])
def test_actual_arbitration_keeps_every_agreement_and_rescue_guard(module, profile, case):
    points = _grid()
    tracker = tracking.IncrementalTracker(policy=tracking.TrackingPolicy(recovery_enabled=False))
    assert tracker.offer(points).accepted
    before = _state(tracker)
    offsets = {
        "agree": (0.1, 0.2, 0.3),
        "ambiguous": (-4.68, 0.0, 4.68),
        "tighter": (0.0, 4.7, -1.6),
        "model-only": (0.0, 0.0, 0.0),
    }[case]
    candidates = tuple(
        _candidate_at(origin, Rigid.of(IDENTITY_MATRIX, (offset, 0.0, 0.0)))
        for origin, offset in zip(
            (tracking.ADJACENT_ORIGIN, tracking.MODEL_ORIGIN, tracking.KEYFRAME_ORIGIN),
            offsets,
            strict=True,
        )
    )
    candidates = (
        replace(candidates[0], edge=replace(candidates[0].edge, rmse=1.0)),
        candidates[1],
        candidates[2],
    )
    if case == "model-only":
        candidates = (candidates[1],)
    with module.SeedSelectionExperiment(profile).active():
        decision = tracker._decide(1, points, points, tracker._accepted[-1], candidates)
    assert decision.candidates == candidates
    if case in ("ambiguous", "model-only"):
        assert not decision.accepted and _state(tracker) == before
        assert decision.failure_type == (
            tracking.TRACK_DISAGREEMENT_TYPE
            if case == "ambiguous"
            else tracking.TRACK_RESCUE_UNSUPPORTED_TYPE
        )
    else:
        assert decision.accepted and not decision.rescue
        expected = (
            tracking.ADJACENT_ORIGIN
            if case == "tighter" or profile == "adjacent-first"
            else tracking.MODEL_ORIGIN
        )
        assert decision.origin == expected
        assert decision.pose == next(c.pose for c in candidates if c.origin == expected)
        assert decision.agreeing == (
            (tracking.ADJACENT_ORIGIN, tracking.KEYFRAME_ORIGIN)
            if case == "tighter"
            else tuple(c.origin for c in candidates)
        )
