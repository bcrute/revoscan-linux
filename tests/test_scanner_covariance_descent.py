"""M16 true objective derivative, bounded descent and original refusal semantics."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis
from tests.test_scanner_symmetric_icp import distance, pairs_for


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_covariance_descent")


@pytest.mark.parametrize("angle", [0.0, 0.045, 0.35])
@pytest.mark.parametrize("nonidentity", [False, True])
def test_analytic_full_gradient_matches_central_differences(module, angle, nonidentity):
    matched, _ = pairs_for(angle)
    current = (Rigid.of(rotation_about_axis((1, 2, 3), 0.12).rotation, (0.4, -0.2, 0.3))
               if nonidentity else Rigid.identity())
    terms = module.system(matched, current)
    assert terms["cost"] == module.objective(matched, current)
    numerical = []
    for component in range(6):
        direction = np.eye(6)[component]
        step = 1e-6 if component < 3 else 1e-5
        positive = module.increment(current, terms["centre"], direction, step)
        negative = module.increment(current, terms["centre"], direction, -step)
        numerical.append((module.objective(matched, positive) -
                          module.objective(matched, negative)) / (2 * step))
    np.testing.assert_allclose(2 * terms["half_gradient"], numerical, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("angle", [0.0, 0.045, 0.35])
def test_known_transform_and_every_accepted_step_descends(module, angle):
    matched, truth = pairs_for(angle)
    current = Rigid.identity()
    for _ in range(100):
        if distance(current, truth, [p for p, _, _, _ in matched]) < 1e-8:
            break
        updated, details = module.descent_step(matched, current)
        assert details["covariance_cost_after"] <= details["covariance_cost_before"]
        assert details["covariance_cost_before"] == module.objective(matched, current)
        assert details["covariance_cost_after"] == module.objective(matched, updated)
        assert details["original_plane_information_guard_passed"]
        assert len(details["trials"]) <= 16
        if details["trials"]:
            assert all(not row["accepted"] for row in details["trials"][:-1])
            assert details["trials"][-1]["accepted"]
        current = updated
    assert distance(current, truth, [p for p, _, _, _ in matched]) < 1e-8


def test_normal_sign_and_magnitude_preserve_gradient_and_step(module):
    matched, _ = pairs_for(0.045)
    changed = [(p, q, tuple(v * (-3 if i % 2 else 2) for v in sn),
                tuple(v * (-2 if i % 3 else 3) for v in tn))
               for i, (p, q, sn, tn) in enumerate(matched)]
    original, _ = module.descent_step(matched, Rigid.identity())
    transformed, _ = module.descent_step(changed, Rigid.identity())
    assert distance(original, transformed, [p for p, _, _, _ in matched]) < 1e-9


def test_rigid_coordinate_change_commutes_with_step(module):
    matched, _ = pairs_for(0.045)
    frame = Rigid.of(rotation_about_axis((1, 2, 3), 0.71).rotation,
                     (10000.0, -20000.0, 40000.0))
    normal_frame = Rigid.of(frame.rotation, (0.0, 0.0, 0.0))
    changed = [(frame.apply(p), frame.apply(q), normal_frame.apply(sn), normal_frame.apply(tn))
               for p, q, sn, tn in matched]
    original, _ = module.descent_step(matched, Rigid.identity())
    transformed, _ = module.descent_step(changed, Rigid.identity())
    expected = frame.compose(original).compose(frame.inverse())
    assert distance(expected, transformed, [p for p, _, _, _ in changed]) < 1e-7


def test_forced_backtracking_uses_first_acceptable_trial(module, monkeypatch):
    matched, _ = pairs_for(0.045)
    original = module.objective
    before = original(matched, Rigid.identity())
    calls = 0

    def delayed_acceptance(points, pose):
        nonlocal calls
        calls += 1
        return before + 1 if calls < 3 else original(points, pose)

    monkeypatch.setattr(module, "objective", delayed_acceptance)
    _, details = module.descent_step(matched, Rigid.identity())
    assert calls == 3 and details["selected_alpha"] == 0.25
    assert [row["accepted"] for row in details["trials"]] == [False, False, True]


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), 1e30])
def test_exhaustion_preserves_all_trials_and_refuses(module, monkeypatch, cost):
    matched, _ = pairs_for(0.045)
    monkeypatch.setattr(module, "objective", lambda *args: cost)
    trace = {}
    with pytest.raises(icp_plane.IcpPlaneNotAllowed, match="exhausted"):
        module.descent_step(matched, Rigid.identity(), trace=trace)
    assert len(trace["trials"]) == 16
    assert [row["alpha"] for row in trace["trials"]] == [2.0 ** -i for i in range(16)]
    assert not any(row["accepted"] for row in trace["trials"])
    assert "covariance_cost_after" not in trace


def test_exact_zero_gradient_returns_actual_pose_object(module):
    matched, _ = pairs_for(0.045)
    exact = [(p, p, sn, sn) for p, _, sn, _ in matched]
    current = Rigid.identity()
    updated, details = module.descent_step(exact, current)
    assert updated is current and details["step_status"] == "exact_zero_gradient"
    assert details["trials"] == [] and details["covariance_cost_after"] == 0


def test_plane_information_refusal_not_regularized_away(module):
    matched = [((float(x), float(y), 8.0), (float(x), float(y), 9.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) for x in range(5) for y in range(5)]
    with pytest.raises(icp_plane.IcpPlaneGeometryUnusable):
        module.descent_step(matched, Rigid.identity())


@pytest.mark.parametrize("invalid", ["empty", "short", "nan", "zero_source", "zero_target"])
def test_invalid_input_refused(module, invalid):
    matched, _ = pairs_for(0.045)
    if invalid == "empty":
        matched = []
    elif invalid == "short":
        matched = matched[:5]
    else:
        p, q, sn, tn = matched[0]
        matched[0] = ((float("nan"), p[1], p[2]) if invalid == "nan" else p, q,
                      (0.0, 0.0, 0.0) if invalid == "zero_source" else sn,
                      (0.0, 0.0, 0.0) if invalid == "zero_target" else tn)
    with pytest.raises(ValueError):
        module.descent_step(matched, Rigid.identity())


def test_inherited_step_records_keep_failure_and_profile(module, monkeypatch):
    matched, _ = pairs_for(0.045)
    experiment = module.RotationAwareExperiment()
    with pytest.raises(RuntimeError, match="context"):
        experiment.step([], Rigid.identity())
    experiment._normals = {p: sn for p, _, sn, _ in matched}
    monkeypatch.setattr(module, "objective", lambda *args: float("nan"))
    with pytest.raises(icp_plane.IcpPlaneNotAllowed, match="exhausted"):
        experiment.step([(p, q, tn) for p, q, _, tn in matched], Rigid.identity())
    record = experiment.records[-1]
    assert record["status"] == "failed" and record["profile"] == "rotation-aware-covariance"
    assert len(record["trials"]) == 16 and all(row["cost"] is None for row in record["trials"])


@pytest.mark.parametrize("profile", ["generalized-covariance", "rotation-aware-covariance"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_exact_delegation_and_hash_binding(module, monkeypatch, tmp_path, profile,
                                                  static, failure):
    import hashlib
    import json

    adapter = importlib.import_module("scanner_covariance_descent_replay")
    factors = adapter.Factors("grid", "pooled", "residual")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_tracker = replay.IncrementalTracker
    output, capture = tmp_path / "output", tmp_path / "capture"
    content = b'{"underlying": "not rewritten"}'

    def delegated(actual_capture, actual_output, *args, **kwargs):
        assert actual_capture == capture and actual_output == output
        assert kwargs["diagnostic_solver_profile"] == profile
        assert "diagnostic_source_profile" not in kwargs
        assert "diagnostic_model_profile" not in kwargs
        if static:
            assert args == (factors,)
        else:
            assert args == (320, (1,))
            assert kwargs["normals_backend"] == "gpu" and kwargs["diagnostic_scene"]
            assert kwargs["diagnostic_factors"] == factors.values
            assert kwargs["skip_complete"] == 12 and kwargs["depth_band"] == (60, 300)
        experiment = adapter.scanner_solver_registry.create_experiment(profile)
        expected_type = (adapter.GeneralizedExperiment if profile == "generalized-covariance"
                         else adapter.RotationAwareExperiment)
        assert type(experiment) is expected_type and experiment.profile == profile
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if not static:
            (output / "stride-1.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    options = dict(solver_profile=profile, static=static, frames=320,
                   skip_complete=0 if static else 12, depth_band=None if static else (60, 300))
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            adapter.run(capture, output, factors, **options)
    else:
        adapter.run(capture, output, factors, **options)
    assert adapter.scanner_solver_registry.create_experiment is adapter.ORIGINAL_FACTORY
    assert replay.IncrementalTracker is original_tracker and not adapter._ACTIVE.get()
    assert (output / "report.json").read_bytes() == content
    metadata = json.loads((output / "covariance-gradient-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["solver_factory_calls"] == 1
    assert metadata["diagnostic_solver_profile"] == profile
    assert metadata["diagnostic_factors"] == factors.as_dict()
    assert metadata["tracking_policy"] == adapter.DEFAULT_POLICY.as_dict()
    assert metadata["diagnostic_source_profile"] is metadata["diagnostic_model_profile"] is None
    assert not metadata["quality_qualification_allowed"]
    expected_reports = {str(output / "report.json")}
    if not static:
        expected_reports.add(str(output / "stride-1.json"))
    assert {r["path"] for r in metadata["underlying_reports"]} == expected_reports
    assert all(r["sha256"] == hashlib.sha256(content).hexdigest()
               for r in metadata["underlying_reports"])


@pytest.mark.parametrize("all_missing", [False, True])
def test_rotation_aware_inherited_context_missing_data_and_unweighted_residual(module, all_missing):
    pairs, _ = pairs_for(0.045)
    points = tuple(p for p, _, _, _ in pairs)
    original_step, original_iteration = icp_plane._solve_point_to_plane, icp_plane._iterate

    def provider(source, neighbours):
        assert source == points and neighbours == 10
        return icp_plane.NormalField(
            tuple(
                icp_plane.NormalEstimate(p, None if all_missing or i == 0 else sn, 10)
                for i, (p, _, sn, _) in enumerate(pairs)
            ),
            10,
            (),
        )

    def correspond(source, pose):
        return [(float(i) / 1000, p, q, tn) for i, (p, q, _, tn) in enumerate(pairs)], 0

    def target_only(*_):
        raise AssertionError("Source normals requested from target-only provider")

    experiment = module.RotationAwareExperiment()
    with icp_plane.normal_acceleration(provider), experiment.active():
        experiment.mark_frame(7)
        with icp_plane.normal_acceleration(target_only):
            parameters = dict(
                initial=Rigid.identity(),
                max_iterations=1,
                threshold=1e-4,
                keep_fraction=0.8,
                gate=12.0,
                accelerated=True,
            )
            if all_missing:
                with pytest.raises(icp_plane.IcpPlaneCorrespondencesInsufficient):
                    icp_plane._iterate(points, correspond, **parameters)
            else:
                result = icp_plane._iterate(points, correspond, **parameters)
                count = icp_plane.kept_count(len(points) - 1, 0.8)
                expected = icp_plane._point_to_plane_rmse(
                    [(result.transform.apply(p), q, tn) for p, q, _, tn in pairs[1 : count + 1]]
                )
                assert result.rmse == pytest.approx(expected, rel=1e-12, abs=1e-12)
                assert result.rejected == 1 and result.correspondences == count
    assert icp_plane._solve_point_to_plane is original_step
    assert icp_plane._iterate is original_iteration
    assert experiment._normals is None and experiment._provider is None
    assert all(r["source_index"] == 7 for r in experiment.records)
    assert experiment.records[0]["source_normals_missing"] == (len(points) if all_missing else 1)


@pytest.mark.parametrize("calls", [0, 2])
def test_factory_missing_or_repeated_is_explicit_failure(module, monkeypatch, tmp_path, calls):
    adapter = importlib.import_module("scanner_covariance_descent_replay")

    def delegated(*args, **kwargs):
        for _ in range(calls):
            adapter.scanner_solver_registry.create_experiment("rotation-aware-covariance")

    monkeypatch.setattr(adapter.static_replay, "run_static", delegated)
    with pytest.raises(RuntimeError):
        adapter.run(tmp_path / "capture", tmp_path / "output",
            adapter.Factors("grid", "pooled", "residual"),
            solver_profile="rotation-aware-covariance", static=True, frames=320)
    assert not adapter._ACTIVE.get()
    assert adapter.scanner_solver_registry.create_experiment is adapter.ORIGINAL_FACTORY


@pytest.mark.parametrize("base,profile,frames,skip,band", [
    ("rank-pooled-residual", "rotation-aware-covariance", 320, 0, None),
    ("grid-pooled-residual", "centered", 320, 0, None),
    ("grid-pooled-residual", "rotation-aware-covariance", 319, 0, None),
    ("grid-pooled-residual", "rotation-aware-covariance", 320, 12, None),
    ("grid-pooled-residual", "rotation-aware-covariance", 320, 0, (60, 300)),
])
def test_closed_adapter_roster(module, tmp_path, base, profile, frames, skip, band):
    adapter = importlib.import_module("scanner_covariance_descent_replay")
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "capture", tmp_path / "output", adapter.Factors(*base.split("-")),
            solver_profile=profile, static=True, frames=frames, skip_complete=skip, depth_band=band)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("guard", ["output", "nested", "factory", "tracker"])
def test_adapter_does_not_overwrite_or_nest(module, monkeypatch, tmp_path, guard):
    adapter = importlib.import_module("scanner_covariance_descent_replay")
    output = tmp_path / "output"
    token = None
    if guard == "output":
        output.mkdir()
    elif guard == "nested":
        token = adapter._ACTIVE.set(True)
    elif guard == "factory":
        monkeypatch.setattr(adapter.scanner_solver_registry, "create_experiment", lambda *a: None)
    else:
        monkeypatch.setattr(adapter.static_replay, "IncrementalTracker", lambda *a: None)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            adapter.run(tmp_path / "capture", output, adapter.Factors("grid", "pooled", "residual"),
                solver_profile="rotation-aware-covariance", static=True, frames=320)
    finally:
        if token is not None:
            adapter._ACTIVE.reset(token)


@pytest.mark.parametrize("profile", ["generalized-covariance", "rotation-aware-covariance"])
def test_cli_preserves_true_solver_profile(module, monkeypatch, tmp_path, profile):
    adapter = importlib.import_module("scanner_covariance_descent_replay")
    captured = []
    monkeypatch.setattr(adapter, "run", lambda *a, **k: captured.append((a, k)))
    monkeypatch.setattr("sys.argv", ["replay", str(tmp_path / "capture"),
        "--output", str(tmp_path / "output"), "--configuration", "rank-pooled-settling",
        "--mechanism", profile, "--frames", "628", "--skip-complete", "12",
        "--depth-band", "60", "300"])
    adapter.main()
    args, kwargs = captured[0]
    assert args[2].identifier == "rank-pooled-settling"
    assert kwargs == dict(solver_profile=profile, static=False, frames=628,
                         skip_complete=12, depth_band=(60.0, 300.0))
