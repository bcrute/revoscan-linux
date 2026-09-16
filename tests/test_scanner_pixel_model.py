"""M10 model-only native-pixel selection at both explicit capacities."""

import hashlib
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.geometry import Rigid
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker


@pytest.fixture(params=["ordinary-900", "dense-1800"])
def setup(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("scanner_pixel_model")
    capacity, policy, source = module.settings(
        request.param, "pixel-grid-model", module.Factors("grid", "pooled", "residual"), "centered"
    )
    tracker = module.PixelModelTracker if capacity == 900 else module.DensePixelModelTracker
    return SimpleNamespace(module=module, capacity=capacity, policy=policy,
                           source=source, name=request.param, tracker=tracker)


def native_cloud(size):
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    return tuple(((i % 90 - cx) * (100.0 + i % 7) / fx,
                  (i // 90 - cy) * (100.0 + i % 7) / fy,
                  100.0 + i % 7) for i in range(size))


def make_tracker(setup, monkeypatch, profile="pixel-grid-model"):
    tracker = setup.tracker(backend="scipy", sampling_profile=profile, policy=setup.policy)
    sizes = (751, 773, 797) if setup.capacity == 900 else (1251, 1373, 1597)
    clouds = [native_cloud(n) for n in sizes]
    poses = [Rigid.identity(), replace(Rigid.identity(), translation=(1.0, 2.0, 3.0)),
             replace(Rigid.identity(), translation=(-1.0, 1.0, 2.0))]
    tracker._accepted = [SimpleNamespace(index=i, cloud=cloud, pose=pose)
                         for i, (cloud, pose) in enumerate(zip(clouds, poses, strict=True))]
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: [0, 1, 2])
    return tracker, clouds, poses


@pytest.mark.parametrize("profile", ["observe-stride", "pixel-grid-model"])
def test_native_model_preserves_exact_members_coordinates_and_original_order(
    setup, monkeypatch, profile
):
    tracker, clouds, poses = make_tracker(setup, monkeypatch, profile)
    method = IncrementalTracker._model_reference
    originals = []

    def original(*args, **kwargs):
        result = method(*args, **kwargs)
        originals.append(result)
        return result

    monkeypatch.setattr(IncrementalTracker, "_model_reference", original)
    actual = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    if profile == "observe-stride":
        assert actual is originals[0]
    else:
        expected = tuple(point for cloud, pose in zip(clouds, poses, strict=True)
                         for point in pose.apply_all(tuple(cloud[i] for i in
                             setup.module.indices(cloud, 600, capacity=setup.capacity))))
        assert actual == (expected, (0, 1, 2)) and len(expected) == 1800
    record = tracker.model_sampling_records[0]
    assert record["original_reference_verified"] and record["status"] == "completed"
    assert record["profile"] == profile and record["pixel_grid_points"] == 1800
    assert "stratified_points" not in record
    assert sum(c["pixel_grid_points"] for c in record["contributors"]) == 1800
    assert [a.cloud for a in tracker._accepted] == clouds
    assert tracker._policy == setup.policy
    for name in ("offer", "_candidate", "_admit", "_reference_frames"):
        assert getattr(setup.tracker, name) is getattr(IncrementalTracker, name)


def test_world_frame_changes_do_not_change_native_pixel_selection(setup, monkeypatch):
    tracker, _, _ = make_tracker(setup, monkeypatch)
    first = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    world = Rigid(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (3.0, -8.0, 2.0))
    for accepted in tracker._accepted:
        accepted.pose = world.compose(accepted.pose)
    second = tracker._model_reference(tracker._accepted[-1], world)
    assert second[1] == first[1]
    assert all(math.dist(a, b) < 1e-12 for a, b in
               zip(second[0], world.apply_all(first[0]), strict=True))


@pytest.mark.parametrize("failure", ["missing", "oversized", "duplicate", "nonpixel"])
def test_invalid_native_provenance_fails_with_record_and_restored_state(
    setup, monkeypatch, failure
):
    tracker, clouds, _ = make_tracker(setup, monkeypatch)
    if failure == "missing":
        tracker._accepted[0].cloud = None
    elif failure == "oversized":
        tracker._accepted[0].cloud = native_cloud(setup.capacity + 1)
    elif failure == "duplicate":
        tracker._accepted[0].cloud = (*clouds[0][:-1], clouds[0][0])
    else:
        p = clouds[0][0]
        tracker._accepted[0].cloud = ((p[0] + 0.1, p[1], p[2]), *clouds[0][1:])
    with pytest.raises((ValueError, RuntimeError)):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert tracker.model_sampling_records[0]["status"] == "failed"
    assert not tracker._inside_model_sampling


def test_profiles_and_capacities_cannot_leak(setup):
    with pytest.raises(ValueError):
        setup.tracker(backend="scipy", policy=setup.policy, sampling_profile="stratified-model")
    other = DEFAULT_POLICY if setup.capacity == 1800 else setup.module.EXPECTED_POLICY
    with pytest.raises(ValueError):
        setup.tracker(backend="scipy", policy=other, sampling_profile="pixel-grid-model")
    tracker = setup.tracker(
        backend="scipy", policy=setup.policy, sampling_profile="pixel-grid-model"
    )
    for position, frames in ((0, 0), (8, 8), (0, 9), (True, 2)):
        with pytest.raises(ValueError):
            tracker.selection_indices(native_cloud(900), 225, position=position, frames=frames)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_report_bytes_and_actual_capacity_metadata(
    setup, monkeypatch, tmp_path, static, failure
):
    adapter = importlib.import_module("scanner_pixel_model_replay")
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
        assert tracker.sampling_profile == "pixel-grid-model"
        tracker.model_sampling_records.append(dict(profile="pixel-grid-model",
            status="failed" if failure else "completed", original_reference_verified=not failure))
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected underlying failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    parameters = dict(sampling_profile="pixel-grid-model", capacity_profile=setup.name,
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
    assert metadata["sampling_profile"] == "pixel-grid-model"
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
    adapter = importlib.import_module("scanner_pixel_model_replay")
    parameters = dict(sampling_profile="pixel-grid-model", capacity_profile=setup.name,
                      solver_profile="centered", static=False, frames=320) | kwargs
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "absent", tmp_path / "output",
                    setup.module.Factors("grid", "pooled", "residual"), **parameters)
    assert not (tmp_path / "output").exists()


def test_adapter_cli_requires_explicit_capacity(setup, monkeypatch, tmp_path):
    adapter = importlib.import_module("scanner_pixel_model_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    arguments = ["scanner_pixel_model_replay.py", str(tmp_path / "capture"), "--output",
                 str(tmp_path / "output"), "--configuration", "grid-pooled-residual",
                 "--mechanism", "centered", "--model-sampling", "pixel-grid-model",
                 "--frames", "320"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit):
        adapter.main()
    assert not calls
    monkeypatch.setattr("sys.argv", [*arguments, "--capacity", setup.name])
    adapter.main()
    assert calls[0][1]["capacity_profile"] == setup.name
