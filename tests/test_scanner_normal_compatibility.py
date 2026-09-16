"""M19 independent eligibility, normal direction and exact scoped composition."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scanner3d import icp_plane, tracking
from scanner3d.geometry import Rigid, rotation_about_axis


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_normal_compatibility")


def fixture_pairs():
    points = [(float(i), 0.0, 100.0) for i in range(6)]
    axes = [
        (1.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        None,
        (1.0, 0.0, 0.0),
    ]
    normals = dict(zip(points, axes, strict=True))
    pairs = [(float(i), p, (p[0] + 0.1, 0.0, 100.0), (1.0, 0.0, 0.0)) for i, p in enumerate(points)]
    return normals, pairs


@pytest.mark.parametrize(
    "profile,positions",
    [
        ("observe-pairs", [0, 1, 2, 3, 4, 5]),
        ("source-normal-required", [0, 1, 2, 3, 5]),
        ("normal-angle-45", [0, 1, 3, 5]),
    ],
)
def test_missing_perpendicular_antiparallel_and_inclusive_45(module, profile, positions):
    normals, pairs = fixture_pairs()
    selected, removed, counts = module.compatible_pairs(normals, Rigid.identity(), pairs, profile)
    assert selected == [pairs[i] for i in positions]
    assert all(a is b for a, b in zip(selected, [pairs[i] for i in positions], strict=True))
    assert removed == 6 - len(positions)
    assert counts["source_normals_missing"] == counts["angle_incompatible"] == 1
    assert counts["angle_compatible"] == 4 and counts["source_normal_available"] == 5
    if profile == "observe-pairs":
        assert selected is pairs


@pytest.mark.parametrize("profile", ["observe-pairs", "source-normal-required", "normal-angle-45"])
def test_empty_pairs_preserve_no_imputed_match(module, profile):
    normals, _ = fixture_pairs()
    pairs = []
    selected, removed, counts = module.compatible_pairs(normals, Rigid.identity(), pairs, profile)
    assert selected == [] and removed == 0 and counts["absolute_cosine_percentiles"] is None
    if profile == "observe-pairs":
        assert selected is pairs


def test_source_axes_rotate_but_do_not_translate(module):
    normals, pairs = fixture_pairs()
    rotation = rotation_about_axis((0.0, 0.0, 1.0), np.pi / 2)
    moved = Rigid.of(rotation.rotation, (1000.0, -500.0, 25.0))
    first = module.compatible_pairs(normals, rotation, pairs, "normal-angle-45")
    second = module.compatible_pairs(normals, moved, pairs, "normal-angle-45")
    assert first == second and pairs[2] in first[0]
    assert all(pairs[i] not in first[0] for i in (0, 1, 5))


@pytest.mark.parametrize(
    "source_scale,target_scale", [(2.0, 1.0), (-2.0, 1.0), (1.0, -8.0), (-4.0, -8.0)]
)
def test_sign_and_scale_invariance(module, source_scale, target_scale):
    normals, pairs = fixture_pairs()
    expected = module.compatible_pairs(normals, Rigid.identity(), pairs, "normal-angle-45")
    scaled_normals = {
        p: None if n is None else tuple(x * source_scale for x in n) for p, n in normals.items()
    }
    scaled_pairs = [(d, p, q, tuple(x * target_scale for x in n)) for d, p, q, n in pairs]
    actual = module.compatible_pairs(
        scaled_normals, Rigid.identity(), scaled_pairs, "normal-angle-45"
    )
    assert [p for _, p, _, _ in actual[0]] == [p for _, p, _, _ in expected[0]]
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize(
    "invalid",
    [
        "profile",
        "short-field",
        "duplicate",
        "foreign-point",
        "source-zero",
        "source-nan",
        "source-short",
        "target-zero",
        "target-nan",
        "negative-distance",
    ],
)
def test_malformed_pairs_never_pass_silently(module, invalid):
    normals, pairs = fixture_pairs()
    profile = "normal-angle-45"
    if invalid == "profile":
        profile = "normal-angle-30"
    elif invalid == "short-field":
        normals.pop(pairs[-1][1])
        pairs.pop()
    elif invalid == "duplicate":
        pairs[1] = pairs[0]
    elif invalid == "foreign-point":
        d, p, q, n = pairs[0]
        pairs[0] = (d, (99.0, 0.0, 100.0), q, n)
    elif invalid.startswith("source-"):
        normals[pairs[0][1]] = {
            "source-zero": (0.0, 0.0, 0.0),
            "source-nan": (np.nan, 0.0, 0.0),
            "source-short": (1.0, 0.0),
        }[invalid]
    else:
        d, p, q, n = pairs[0]
        pairs[0] = (
            (-1.0 if invalid == "negative-distance" else d),
            p,
            q,
            (0.0, 0.0, 0.0)
            if invalid == "target-zero"
            else (np.nan, 0.0, 0.0)
            if invalid == "target-nan"
            else n,
        )
    with pytest.raises(ValueError):
        module.compatible_pairs(normals, Rigid.identity(), pairs, profile)


def fake_field(normals):
    return SimpleNamespace(
        estimates=[SimpleNamespace(point=p, normal=n) for p, n in normals.items()],
        requested_neighbours=10,
        as_dict=lambda: dict(requested_neighbours=10),
    )


@pytest.mark.parametrize("profile", ["observe-pairs", "source-normal-required", "normal-angle-45"])
def test_iterator_returns_original_result_and_accounts_rejections(module, monkeypatch, profile):
    normals, pairs = fixture_pairs()
    source = tuple(normals)
    returned = SimpleNamespace(iterations=2, rmse=0.1, transform=Rigid.identity())
    calls = []
    options = dict(
        max_iterations=50,
        keep_fraction=0.8,
        initial=Rigid.identity(),
        threshold=0.0065,
        gate=12.0,
        accelerated=True,
    )

    def original(src, correspond, **kwargs):
        assert src is source and kwargs == options
        calls.append(correspond(src, Rigid.identity()))
        return returned

    monkeypatch.setattr(module, "ORIGINAL_ITERATE", original)
    monkeypatch.setattr(icp_plane, "_iterate", original)
    experiment = module.NormalCompatibilityExperiment(profile)
    with experiment.active(), icp_plane.normal_acceleration(lambda p, k: fake_field(normals)):
        actual = icp_plane._iterate(source, lambda s, t: (pairs, 7), **options)
    selected, removed, _ = module.compatible_pairs(normals, Rigid.identity(), pairs, profile)
    assert actual is returned and calls == [(selected, 7 + removed)]
    if profile == "observe-pairs":
        assert calls[0][0] is pairs
    assert icp_plane._iterate is original and experiment._iteration is None
    assert not experiment._inside_fit and not module._ACTIVE.get()
    assert experiment.records[0]["calls"][0]["additional_rejected"] == removed


@pytest.mark.parametrize("profile", ["observe-pairs", "source-normal-required", "normal-angle-45"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
@pytest.mark.parametrize("angle", [0.0, 0.04])
def test_real_fit_matches_explicit_filter(module, profile, solver_profile, angle):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    target = tuple(corner())
    truth = Rigid.of(
        rotation_about_axis((0.3, 0.5, 0.7), angle).rotation, (angle * 10, -angle * 4, angle * 3)
    )
    source = tuple(truth.inverse().apply_all(target))
    original = icp_plane._iterate

    def provider(points, count):
        return icp_plane.estimate_normals(points, neighbours=count, backend="scipy")

    def explicit(src, correspond, **kwargs):
        field = provider(tuple(src), 10)
        normals = {e.point: e.normal for e in field.estimates}

        def filtered(points, transform):
            pairs, rejected = correspond(points, transform)
            selected, removed, _ = module.compatible_pairs(normals, transform, pairs, profile)
            return selected, rejected + removed

        return original(src, filtered, **kwargs)

    options = dict(
        backend="scipy",
        initial=Rigid.identity(),
        voxel_size=1.5,
        max_correspondence_distance=12.0,
        keep_fraction=0.8,
    )
    expected_solver = solvers.SolverExperiment(solver_profile)
    with (
        icp_plane.normal_acceleration(provider),
        expected_solver.active(),
        patch.object(icp_plane, "_iterate", explicit),
    ):
        expected = tracking.icp_point_to_plane(source, target, **options)
    actual_solver = solvers.SolverExperiment(solver_profile)
    experiment = module.NormalCompatibilityExperiment(profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = tracking.icp_point_to_plane(source, target, **options)
    assert actual == expected and actual_solver.records == expected_solver.records
    assert actual.converged and len(experiment.records) == 1
    assert experiment.records[0]["status"] == "completed"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("data"),
        ArithmeticError("math"),
        icp_plane.IcpPlaneCorrespondencesInsufficient("few"),
        RuntimeError("internal"),
    ],
)
def test_iterator_failure_propagates_and_restores(module, monkeypatch, error):
    normals, pairs = fixture_pairs()

    def original(src, correspond, **kwargs):
        correspond(src, Rigid.identity())
        raise error

    monkeypatch.setattr(module, "ORIGINAL_ITERATE", original)
    monkeypatch.setattr(icp_plane, "_iterate", original)
    experiment = module.NormalCompatibilityExperiment("normal-angle-45")
    with (
        experiment.active(),
        icp_plane.normal_acceleration(lambda p, k: fake_field(normals)),
        pytest.raises(type(error)) as caught,
    ):
        icp_plane._iterate(
            tuple(normals), lambda s, t: (pairs, 0), max_iterations=50, keep_fraction=0.8
        )
    assert caught.value is error and icp_plane._iterate is original
    assert not experiment._inside_fit and not module._ACTIVE.get()
    assert experiment.records[0]["status"] == (
        "failed" if isinstance(error, RuntimeError) else "refused"
    )


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("profile", ["observe-pairs", "source-normal-required", "normal-angle-45"])
@pytest.mark.parametrize("solver_profile", ["observe-origin", "centered"])
def test_real_tracker_and_normal_context_composition(module, capacity, profile, solver_profile):
    from unittest.mock import patch

    from tests.test_scanner_registration_audit import corner

    solvers = importlib.import_module("scanner_solver_mechanisms")
    points = tuple(corner())
    _, policy, _ = module.settings(
        capacity, profile, module.Factors("grid", "pooled", "residual"), solver_profile
    )
    original = icp_plane._iterate

    def provider(cloud, count):
        return icp_plane.estimate_normals(cloud, neighbours=count, backend="scipy")

    def explicit(src, correspond, **kwargs):
        normals = {e.point: e.normal for e in provider(tuple(src), 10).estimates}

        def filtered(points, transform):
            pairs, rejected = correspond(points, transform)
            selected, removed, _ = module.compatible_pairs(normals, transform, pairs, profile)
            return selected, rejected + removed

        return original(src, filtered, **kwargs)

    expected_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    expected_solver = solvers.SolverExperiment(solver_profile)
    with (
        icp_plane.normal_acceleration(provider),
        expected_solver.active(),
        patch.object(icp_plane, "_iterate", explicit),
    ):
        expected = [expected_tracker.offer(points).as_dict() for _ in range(4)]
    actual_tracker = tracking.IncrementalTracker(backend="scipy", policy=policy)
    actual_solver = solvers.SolverExperiment(solver_profile)
    experiment = module.NormalCompatibilityExperiment(profile)
    with icp_plane.normal_acceleration(provider), actual_solver.active(), experiment.active():
        actual = [actual_tracker.offer(points).as_dict() for _ in range(4)]
    assert actual == expected and all(r["accepted"] for r in actual)
    assert actual_solver.records == expected_solver.records
    assert experiment.records and all(r["status"] == "completed" for r in experiment.records)


@pytest.mark.parametrize("capacity", ["ordinary-900", "dense-1800"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_arguments_report_bytes_and_scope(
    module, monkeypatch, tmp_path, capacity, static, failure
):
    adapter = importlib.import_module("scanner_normal_compatibility_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    factors = module.Factors("grid", "pooled", "residual")
    expected = module.settings(capacity, "normal-angle-45", factors, "centered")
    output = tmp_path / "output"
    content = b'{"unchanged": true}'
    normals, pairs = fixture_pairs()

    def original(src, correspond, **kwargs):
        correspond(src, Rigid.identity())
        return SimpleNamespace(iterations=2, rmse=0.1, transform=Rigid.identity())

    monkeypatch.setattr(module, "ORIGINAL_ITERATE", original)
    monkeypatch.setattr(icp_plane, "_iterate", original)

    def delegated(capture, destination, *args, **kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == expected[2]
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320, (1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        with icp_plane.normal_acceleration(lambda p, k: fake_field(normals)):
            icp_plane._iterate(
                tuple(normals), lambda s, t: (pairs, 0), max_iterations=50, keep_fraction=0.8
            )
        output.mkdir()
        for name in ("report.json",) if static else ("report.json", "stride-1.json"):
            (output / name).write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    options = dict(
        compatibility_profile="normal-angle-45",
        capacity_profile=capacity,
        solver_profile="centered",
        static=static,
        frames=320,
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            adapter.run(tmp_path / "capture", output, factors, **options)
    else:
        adapter.run(tmp_path / "capture", output, factors, **options)
    metadata = json.loads((output / "normal-compatibility-report.json").read_bytes())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["angle_filter_enabled"] and metadata["source_normal_required"]
    assert metadata["tracking_policy"] == expected[1].as_dict()
    assert metadata["preparation_target_points"] == expected[0]
    assert metadata["normal_filter_records"][0]["calls"][0]["selected"] == 4
    assert not metadata["quality_qualification_allowed"]
    assert len(metadata["underlying_reports"]) == (1 if static else 2)
    assert all(
        r["sha256"] == hashlib.sha256(content).hexdigest() for r in metadata["underlying_reports"]
    )
    assert replay.IncrementalTracker is factory and icp_plane._iterate is original
    assert not adapter._ACTIVE.get() and not module._ACTIVE.get()


@pytest.mark.parametrize("guard", ["inactive", "nested", "foreign-hook", "no-provider", "bound"])
def test_context_provider_and_work_guards(module, monkeypatch, guard):
    experiment = module.NormalCompatibilityExperiment("observe-pairs")
    normals, pairs = fixture_pairs()
    options = dict(max_iterations=50, keep_fraction=0.8)
    if guard == "inactive":
        with pytest.raises(RuntimeError):
            experiment.iterate(tuple(normals), lambda s, t: (pairs, 0), **options)
    elif guard == "foreign-hook":
        monkeypatch.setattr(icp_plane, "_iterate", lambda *a, **k: None)
        with pytest.raises(RuntimeError), experiment.active():
            pass
    else:
        with experiment.active():
            if guard == "nested":
                other = module.NormalCompatibilityExperiment("normal-angle-45")
                with pytest.raises(RuntimeError), other.active():
                    pass
            else:
                if guard == "bound":
                    experiment.records = [{}] * 90000
                with pytest.raises(RuntimeError):
                    experiment.iterate(tuple(normals), lambda s, t: (pairs, 0), **options)
    assert not module._ACTIVE.get() and not experiment._inside_fit


@pytest.mark.parametrize("defect", ["order", "count", "neighbors"])
def test_source_field_identity_is_mandatory(module, defect):
    normals, pairs = fixture_pairs()
    field = fake_field(normals)
    if defect == "order":
        field.estimates.reverse()
    elif defect == "count":
        field.estimates.pop()
    else:
        field.requested_neighbours = 20
    experiment = module.NormalCompatibilityExperiment("observe-pairs")
    with (
        experiment.active(),
        icp_plane.normal_acceleration(lambda p, k: field),
        pytest.raises(ValueError, match="different points or neighbors"),
    ):
        icp_plane._iterate(
            tuple(normals), lambda s, t: (pairs, 0), max_iterations=50, keep_fraction=0.8
        )
    assert experiment.records[0]["status"] == "refused"
    assert not module._ACTIVE.get()


@pytest.mark.parametrize(
    "changes",
    [
        dict(capacity_profile="unknown"),
        dict(compatibility_profile="normal-angle-30"),
        dict(solver_profile="symmetric"),
        dict(frames=319),
        dict(skip_complete=1),
        dict(depth_band=(60, 300)),
        dict(static=False, frames=99),
        dict(static=False, frames=3001),
    ],
)
def test_adapter_nonclosed_inputs_do_not_start_work(module, tmp_path, changes):
    adapter = importlib.import_module("scanner_normal_compatibility_replay")
    options = dict(
        capacity_profile="dense-1800",
        compatibility_profile="normal-angle-45",
        solver_profile="centered",
        static=True,
        frames=320,
    )
    with pytest.raises(ValueError):
        adapter.run(
            tmp_path / "capture",
            tmp_path / "output",
            module.Factors("grid", "pooled", "residual"),
            **(options | changes),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "base", ["stride-pooled-residual", "grid-pooled-settling", "grid-native-residual"]
)
def test_other_factors_rejected(module, base):
    with pytest.raises(ValueError):
        module.settings(
            "ordinary-900", "normal-angle-45", module.Factors(*base.split("-")), "centered"
        )


def test_cli_requires_explicit_filter(module, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_normal_compatibility_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *a, **k: calls.append((a, k)))
    args = [
        "replay",
        str(tmp_path / "capture"),
        "--output",
        str(tmp_path / "output"),
        "--configuration",
        "grid-pooled-residual",
        "--capacity",
        "dense-1800",
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
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*args, "--normal-compatibility", "normal-angle-45"])
    adapter.main()
    assert calls[0][1]["compatibility_profile"] == "normal-angle-45"
    assert calls[0][1]["depth_band"] == (60.0, 300.0)
