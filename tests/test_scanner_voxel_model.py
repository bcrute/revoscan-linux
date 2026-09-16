"""M11 closed joint-voxel ordering and report provenance, not scanner qualification."""

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d.geometry import Rigid
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker


@pytest.fixture(params=["ordinary-900", "dense-1800"])
def setup(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("scanner_voxel_model")
    capacity, policy, source = module.settings(
        request.param, "voxel-first-model", module.Factors("grid", "pooled", "residual"), "centered"
    )
    tracker = module.VoxelModelTracker if capacity == 900 else module.DenseVoxelModelTracker
    return SimpleNamespace(module=module, capacity=capacity, policy=policy,
                           source=source, name=request.param, tracker=tracker)


def cloud(size):
    return tuple((i % 40 * 1.7, (i // 40) % 40 * 1.7, 100.0 + i // 1600 * 1.7)
                 for i in range(size))


def make_tracker(setup, monkeypatch, profile="voxel-first-model"):
    tracker = setup.tracker(backend="scipy", policy=setup.policy, sampling_profile=profile)
    clouds = [cloud(setup.capacity - 20 * i) for i in range(3)]
    poses = [replace(Rigid.identity(), translation=(0.0, 0.0, 5.0 * i)) for i in range(3)]
    tracker._accepted = [SimpleNamespace(index=i, cloud=c, pose=p)
                         for i, (c, p) in enumerate(zip(clouds, poses, strict=True))]
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: [0, 1, 2])
    return tracker, clouds, poses


@pytest.mark.parametrize("size", [0, 1, 899, 1800, 1801, 5399, 14400])
def test_exact_midpoint_budget_and_order(setup, size):
    selected = setup.module.indices(size)
    assert len(selected) == min(size, 1800)
    assert selected == tuple(sorted(set(selected)))
    assert all(0 <= i < size for i in selected)
    assert selected == (tuple(range(size)) if size <= 1800 else
                        tuple(((2 * j + 1) * size) // 3600 for j in range(1800)))


@pytest.mark.parametrize("size,limit", [(True, 1800), (-1, 1800), (14401, 1800),
                                       (900.0, 1800), (900, 900), (900, 1800.0)])
def test_nonclosed_budgets_refused(setup, size, limit):
    with pytest.raises(ValueError):
        setup.module.indices(size, limit)


@pytest.mark.parametrize("profile", ["observe-stride", "voxel-first-model"])
def test_exact_original_or_voxel_union_with_full_provenance(setup, monkeypatch, profile):
    tracker, clouds, poses = make_tracker(setup, monkeypatch, profile)
    original_method = IncrementalTracker._model_reference
    originals = []

    def original(*args, **kwargs):
        value = original_method(*args, **kwargs)
        originals.append(value)
        return value

    monkeypatch.setattr(IncrementalTracker, "_model_reference", original)
    actual = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    union = tuple(p for c, pose in zip(clouds, poses, strict=True) for p in pose.apply_all(c))
    voxels = voxel_downsample(union, 1.5)
    expected = tuple(voxels[i] for i in setup.module.indices(len(voxels)))
    if profile == "observe-stride":
        assert actual is originals[0]
    else:
        assert actual == (expected, (0, 1, 2)) and len(expected) == 1800
    record = tracker.model_sampling_records[0]
    assert record["status"] == "completed" and record["original_reference_verified"]
    assert record["union_points"] == len(union)
    assert record["available_voxels"] == len(voxels)
    assert record["voxel_first_points"] == 1800
    assert record["selected_points_sha256"] == setup.module.point_digest(expected)
    assert record["full_and_selected_voxels_exactly_idempotent"]
    assert all(c["selected_voxel_contributions"] > 0 for c in record["contributors"])
    assert [a.cloud for a in tracker._accepted] == clouds
    for name in ("offer", "_candidate", "_admit", "_reference_frames"):
        assert getattr(setup.tracker, name) is getattr(IncrementalTracker, name)


def test_empty_and_shared_voxels_retain_truthful_frame_provenance(setup, monkeypatch):
    tracker, _, _ = make_tracker(setup, monkeypatch)
    for item in tracker._accepted:
        item.cloud = ((0.1, 0.2, 100.1), (0.2, 0.3, 100.2))
        item.pose = Rigid.identity()
    result = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert len(result[0]) == 1 and result[1] == (0, 1, 2)
    assert all(c["selected_voxel_contributions"] == 1
               for c in tracker.model_sampling_records[-1]["contributors"])
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: [])
    assert tracker._model_reference(tracker._accepted[-1], Rigid.identity()) == ((), ())


@pytest.mark.parametrize("failure", ["missing", "oversized", "nonfinite", "nested", "bound"])
def test_bad_provenance_or_call_bounds_refused(setup, monkeypatch, failure):
    tracker, _, _ = make_tracker(setup, monkeypatch)
    if failure == "missing":
        tracker._accepted[0].cloud = None
    elif failure == "oversized":
        tracker._accepted[0].cloud = cloud(setup.capacity + 1)
    elif failure == "nonfinite":
        tracker._accepted[0].cloud = ((float("nan"), 0.0, 100.0),)
    elif failure == "nested":
        tracker._inside_model_sampling = True
    else:
        tracker.model_sampling_records = [{}] * 3000
    with pytest.raises(RuntimeError):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    if failure not in ("nested", "bound"):
        assert tracker.model_sampling_records[-1]["status"] == "failed"
        assert not tracker._inside_model_sampling


def test_profile_policy_and_settings_are_closed(setup):
    other = DEFAULT_POLICY if setup.capacity == 1800 else setup.module.EXPECTED_POLICY
    with pytest.raises(ValueError):
        setup.tracker(backend="scipy", policy=other, sampling_profile="voxel-first-model")
    with pytest.raises(ValueError):
        setup.tracker(backend="scipy", policy=setup.policy, sampling_profile="pixel-grid-model")
    with pytest.raises(ValueError):
        setup.module.settings(setup.name, "voxel-first-model",
                              setup.module.Factors("stride", "pooled", "residual"), "centered")


def test_missing_selected_reference_support_is_not_counted(setup, monkeypatch):
    tracker, _, _ = make_tracker(setup, monkeypatch)
    monkeypatch.setattr(setup.module, "indices", lambda size: (0,))
    with pytest.raises(RuntimeError, match="no selected contribution"):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert tracker.model_sampling_records[-1]["status"] == "failed"
    assert not tracker._inside_model_sampling


def test_non_idempotent_reduction_is_not_silently_accepted(setup, monkeypatch):
    tracker, _, _ = make_tracker(setup, monkeypatch)
    calls = 0

    def changed(points, cell):
        nonlocal calls
        calls += 1
        result = voxel_downsample(points, cell)
        return result if calls == 1 else result[:-1]

    monkeypatch.setattr(setup.module, "voxel_downsample", changed)
    with pytest.raises(RuntimeError, match="not exactly idempotent"):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert tracker.model_sampling_records[-1]["status"] == "failed"
    assert not tracker._inside_model_sampling


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_report_bytes_and_actual_capacity_metadata(
    setup, monkeypatch, tmp_path, static, failure
):
    adapter = importlib.import_module("scanner_voxel_model_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    factory = replay.IncrementalTracker
    output = tmp_path / "output"
    content = b'{"original": "unchanged"}'

    def delegated(capture, destination, *args, **kwargs):
        assert destination == output and capture == tmp_path / "capture"
        assert kwargs["diagnostic_solver_profile"] == "centered"
        assert kwargs["diagnostic_source_profile"] == setup.source
        if not static:
            assert args == (320, (1,))
            assert kwargs["diagnostic_scene"] and kwargs["normals_backend"] == "gpu"
        tracker = replay.IncrementalTracker(backend="scipy", policy=setup.policy)
        assert type(tracker) is setup.tracker
        assert tracker.sampling_profile == "voxel-first-model"
        tracker.model_sampling_records.append(dict(profile="voxel-first-model",
            status="failed" if failure else "completed", original_reference_verified=not failure))
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected underlying failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    parameters = dict(sampling_profile="voxel-first-model", capacity_profile=setup.name,
                      solver_profile="centered", static=static, frames=320)
    factors = setup.module.Factors("grid", "pooled", "residual")
    if failure:
        with pytest.raises(RuntimeError, match="injected underlying failure"):
            adapter.run(tmp_path / "capture", output, factors, **parameters)
    else:
        adapter.run(tmp_path / "capture", output, factors, **parameters)
    assert replay.IncrementalTracker is factory and not adapter._ACTIVE.get()
    assert (output / "report.json").read_bytes() == content
    metadata = json.loads((output / "model-sampling-report.json").read_text())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["sampling_profile"] == "voxel-first-model"
    assert metadata["capacity_profile"] == setup.name
    assert metadata["tracking_policy"] == setup.policy.as_dict()
    assert metadata["preparation_target_points"] == setup.capacity
    assert metadata["diagnostic_source_profile"] == setup.source
    assert not metadata["quality_qualification_allowed"]
    assert metadata["underlying_reports"] == [dict(path=str(output / "report.json"),
        sha256=hashlib.sha256(content).hexdigest())]


@pytest.mark.parametrize("kwargs", [dict(capacity_profile="unknown"),
                                   dict(sampling_profile="stratified-model"),
                                   dict(solver_profile="symmetric"), dict(frames=99),
                                   dict(static=True, frames=319),
                                   dict(static=True, skip_complete=12)])
def test_adapter_refuses_undeclared_controls_before_loading(setup, tmp_path, kwargs):
    adapter = importlib.import_module("scanner_voxel_model_replay")
    parameters = dict(sampling_profile="voxel-first-model", capacity_profile=setup.name,
                      solver_profile="centered", static=False, frames=320) | kwargs
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "absent", tmp_path / "output",
                    setup.module.Factors("grid", "pooled", "residual"), **parameters)
    assert not (tmp_path / "output").exists()


def test_adapter_cli_requires_explicit_capacity(setup, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_voxel_model_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    arguments = ["scanner_voxel_model_replay.py", str(tmp_path / "capture"), "--output",
                 str(tmp_path / "output"), "--configuration", "grid-pooled-residual",
                 "--mechanism", "centered", "--model-sampling", "voxel-first-model",
                 "--frames", "320"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*arguments, "--capacity", setup.name])
    adapter.main()
    assert calls[0][1]["capacity_profile"] == setup.name
