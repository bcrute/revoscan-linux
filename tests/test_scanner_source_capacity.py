"""M7 changes two cloud-capacity fields, never quality thresholds or model budget."""

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def capacity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_source_capacity")


@pytest.mark.parametrize("sampler", ["stride", "grid"])
@pytest.mark.parametrize("solver", ["observe-origin", "centered"])
def test_exact_observation_and_two_changed_policy_fields(capacity, sampler, solver):
    factors = capacity.Factors(sampler, "pooled", "residual")
    points, baseline = capacity.settings("observe-900", factors, solver, None)
    assert points == 900 and baseline is capacity.DEFAULT_POLICY
    points, dense = capacity.settings("source-1800", factors, solver, None)
    assert points == 1800 and dense.model_reference_points == 1800
    before, after = baseline.as_dict(), dense.as_dict()
    assert {key for key in before if before[key] != after[key]} == {
        "candidate_source_points",
        "retained_cloud_points",
    }
    assert dense.candidate_source_points == dense.retained_cloud_points == 1800
    assert dense.max_retained_clouds == baseline.max_retained_clouds


@pytest.mark.parametrize(
    "values",
    [
        ("guess", ("stride", "pooled", "residual"), "observe-origin", None),
        ("source-1800", ("rank", "pooled", "residual"), "observe-origin", None),
        ("source-1800", ("grid", "native", "residual"), "observe-origin", None),
        ("source-1800", ("grid", "pooled", "settling"), "observe-origin", None),
        ("source-1800", ("grid", "pooled", "residual"), "generalized-covariance", None),
        ("source-1800", ("grid", "pooled", "residual"), "observe-origin", "observed-point"),
    ],
)
def test_out_of_roster_combinations_are_refused(capacity, values):
    profile, factors, solver, model = values
    with pytest.raises(ValueError, match="M7 requires"):
        capacity.settings(profile, capacity.Factors(*factors), solver, model)


def test_insufficient_gpu_target_capacity_refuses(capacity, monkeypatch):
    monkeypatch.setattr(capacity, "NORMAL_CACHE_MAX_POINTS", 1024)
    with pytest.raises(ValueError, match="GPU"):
        capacity.settings("source-1800", capacity.Factors(), "observe-origin", None)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("sampler", ["stride", "grid"])
@pytest.mark.parametrize(
    "profile,points", [(None, 900), ("observe-900", 900), ("source-1800", 1800)]
)
def test_runner_uses_declared_capacity_and_preserves_failure_metadata(
    capacity, monkeypatch, tmp_path, static, sampler, profile, points
):
    replay = importlib.import_module(
        "scanner_factorial_replay" if static else "scanner_motion_replay"
    )
    factors = capacity.Factors(sampler, "pooled", "residual")
    band = replay.DepthBand(60, 300)
    policies = []
    requests = []

    def tracker(**kwargs):
        policies.append(kwargs["policy"])
        return SimpleNamespace(poses=())

    @contextmanager
    def normals(provider, message):
        message("GPU-assisted tracking: test provider only; not a hardware test")
        yield

    def prepare(raw, *, target_points, depth_band):
        assert raw == b"raw" and depth_band == band
        requests.append(target_points)
        raise RuntimeError("injected preparation failure")

    monkeypatch.setattr(replay, "IncrementalTracker", tracker)
    monkeypatch.setattr(replay, "live_normals", normals)
    if static:
        monkeypatch.setattr(replay, "load_static", lambda path: ([b"raw"], [], {}))
        monkeypatch.setattr(
            replay, "prepare_frame" if sampler == "stride" else "prepare_grid", prepare
        )
    else:
        monkeypatch.setattr(
            replay, "load_inputs", lambda *args, **kwargs: (None, [b"raw"], band, [])
        )
        monkeypatch.setattr(replay, "source_tracking_loss", lambda path: None)
        prepare_module = (
            replay if sampler == "stride" else importlib.import_module("scanner_pixel_grid")
        )
        monkeypatch.setattr(prepare_module, "prepare_frame", prepare)
    output = tmp_path / "output"
    parameters = dict(diagnostic_solver_profile="observe-origin", diagnostic_source_profile=profile)
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        if static:
            replay.run_static(tmp_path / "capture", output, factors, **parameters)
        else:
            replay.run(
                tmp_path / "capture",
                output,
                100,
                (1,),
                diagnostic_scene=True,
                diagnostic_factors=factors.values,
                **parameters,
            )
    expected = (
        capacity.DEFAULT_POLICY
        if profile is None
        else capacity.settings(profile, factors, "observe-origin", None)[1]
    )
    assert policies == [expected] and requests == [points]
    document = json.loads((output / "report.json").read_text())
    assert document["status"] == "failed"
    assert document["error"] == "RuntimeError: injected preparation failure"
    for report in [document] if static else [document, document["failed_run"]]:
        assert report["diagnostic_source_profile"] == profile
        assert report["preparation_target_points"] == points
        assert report["tracking_policy"] == expected.as_dict()
        assert not report["quality_qualification_allowed"]
        assert report.get("decisions", []) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"normals_backend": "cpu"},
        {"diagnostic_factors": None},
        {"diagnostic_solver_profile": "reciprocal-origin"},
        {"diagnostic_solver_profile": None},
        {"diagnostic_model_profile": "observed-point"},
        {"diagnostic_source_profile": "guess"},
        {"diagnostic_factors": ("rank", "pooled", "residual")},
    ],
)
def test_motion_api_refuses_out_of_roster_before_loading(capacity, tmp_path, overrides):
    replay = importlib.import_module("scanner_motion_replay")
    parameters = (
        dict(
            diagnostic_factors=capacity.Factors().values,
            diagnostic_solver_profile="observe-origin",
            diagnostic_source_profile="source-1800",
        )
        | overrides
    )
    with pytest.raises(ValueError, match="M7 requires"):
        replay.run(tmp_path / "absent", tmp_path / "output", 100, (1,), **parameters)
    assert not (tmp_path / "output").exists()


def test_motion_api_refuses_other_stride_and_legacy_profile(capacity, tmp_path):
    replay = importlib.import_module("scanner_motion_replay")
    parameters = dict(
        diagnostic_factors=capacity.Factors().values,
        diagnostic_solver_profile="observe-origin",
        diagnostic_source_profile="source-1800",
    )
    with pytest.raises(ValueError, match="M7 requires"):
        replay.run(tmp_path / "absent", tmp_path / "output", 100, (2,), **parameters)
    with pytest.raises(ValueError, match="legacy diagnostic"):
        replay.run(
            tmp_path / "absent",
            tmp_path / "output",
            100,
            (1,),
            diagnostic_reference_profile="pixel-grid",
            **parameters,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("static", [False, True])
def test_cli_dispatch_preserves_source_profile(capacity, monkeypatch, tmp_path, static):
    entry = importlib.import_module("scanner_mechanism_replay")
    calls = []
    monkeypatch.setattr(
        entry,
        "run_static" if static else "run_motion",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "scanner_mechanism_replay.py",
            str(tmp_path / "capture"),
            "--output",
            str(tmp_path / "output"),
            "--configuration",
            "grid-pooled-residual",
            "--mechanism",
            "centered",
            "--source-capacity",
            "source-1800",
            "--frames",
            "320",
        ]
        + (["--static"] if static else []),
    )
    entry.main()
    assert len(calls) == 1
    assert calls[0][1]["diagnostic_source_profile"] == "source-1800"
    assert calls[0][1]["diagnostic_solver_profile"] == "centered"


@pytest.mark.parametrize("solver", ["reciprocal-origin", "reciprocal-centered"])
def test_reciprocal_registry_remains_separate_from_source_roster(capacity, solver):
    registry = importlib.import_module("scanner_solver_registry")
    reciprocal = registry.create_experiment(solver)
    assert reciprocal.profile == solver
    with pytest.raises(ValueError, match="M7 requires"):
        capacity.settings("source-1800", capacity.Factors(), solver, None)
