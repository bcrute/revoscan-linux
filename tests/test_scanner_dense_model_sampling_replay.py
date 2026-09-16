"""M9 provenance, ordinary-model boundaries and paired adapter evidence."""

import hashlib
import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d.geometry import Rigid
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker, _sampled


@pytest.fixture
def dense(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_dense_model_sampling")


def setup_tracker(dense, monkeypatch, profile="stratified-model"):
    tracker = dense.DenseModelSamplingTracker(
        backend="scipy", sampling_profile=profile, policy=dense.EXPECTED_POLICY
    )
    clouds = [
        tuple((float(i), float(i % 17), float(i % 13)) for i in range(n))
        for n in (1251, 1373, 1597)
    ]
    poses = [
        Rigid.identity(),
        replace(Rigid.identity(), translation=(1.0, 2.0, 3.0)),
        replace(Rigid.identity(), translation=(-1.0, 1.0, 2.0)),
    ]
    tracker._accepted = [
        SimpleNamespace(index=i, cloud=cloud, pose=pose)
        for i, (cloud, pose) in enumerate(zip(clouds, poses, strict=True))
    ]
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: [0, 1, 2])
    return tracker, clouds, poses


@pytest.mark.parametrize("profile", ["observe-stride", "stratified-model"])
def test_only_existing_model_members_are_evenly_resampled(dense, monkeypatch, profile):
    tracker, clouds, poses = setup_tracker(dense, monkeypatch, profile)
    original_method = IncrementalTracker._model_reference
    originals = []

    def base(*args, **kwargs):
        original = original_method(*args, **kwargs)
        originals.append(original)
        return original

    monkeypatch.setattr(IncrementalTracker, "_model_reference", base)
    actual = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    if profile == "observe-stride":
        assert actual is originals[0]
    else:
        assert actual[1] == originals[0][1] == (0, 1, 2)
        assert len(actual[0]) == 1800 > len(originals[0][0])
        expected = tuple(
            point
            for p, (cloud, pose) in enumerate(zip(clouds, poses, strict=True))
            for point in pose.apply_all(
                tuple(cloud[i] for i in dense.indices(len(cloud), 600, position=p, frames=3))
            )
        )
        assert actual[0] == expected
    record = tracker.model_sampling_records[0]
    assert record["status"] == "completed" and record["original_reference_verified"]
    assert record["stratified_points"] == 1800
    assert record["returned_points"] == len(actual[0])
    assert not tracker._inside_model_sampling
    assert [a.cloud for a in tracker._accepted] == clouds
    assert importlib.import_module("scanner_model_sampling")._sampled is _sampled
    assert tracker._policy == dense.EXPECTED_POLICY
    for name in ("offer", "_candidate", "_admit"):
        assert getattr(dense.DenseModelSamplingTracker, name) is getattr(IncrementalTracker, name)


def test_empty_reference_observation_is_exact(dense, monkeypatch):
    tracker, _, _ = setup_tracker(dense, monkeypatch, "observe-stride")
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: [])
    assert tracker._model_reference(tracker._accepted[-1], Rigid.identity()) == ((), ())
    assert tracker.model_sampling_records[0]["original_reference_verified"]


def test_model_selection_is_equivariant_under_a_changed_world_frame(dense, monkeypatch):
    tracker, _, _ = setup_tracker(dense, monkeypatch)
    first = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    world = Rigid(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (3.0, -8.0, 2.0))
    for accepted in tracker._accepted:
        accepted.pose = world.compose(accepted.pose)
    second = tracker._model_reference(tracker._accepted[-1], world)
    assert second[1] == first[1]
    assert all(
        math.dist(a, b) < 1e-12 for a, b in zip(second[0], world.apply_all(first[0]), strict=True)
    )


@pytest.mark.parametrize("failure", ["membership", "coordinates", "missing", "nonfinite", "error"])
def test_reference_failures_preserve_evidence_and_restore_state(dense, monkeypatch, failure):
    tracker, _, _ = setup_tracker(dense, monkeypatch)
    base = IncrementalTracker._model_reference

    def broken(*args, **kwargs):
        if failure == "error":
            raise RuntimeError("injected")
        points, frames = base(*args, **kwargs)
        if failure == "membership":
            return points, frames[:-1]
        if failure == "coordinates":
            return points[:-1], frames
        return points, frames

    monkeypatch.setattr(IncrementalTracker, "_model_reference", broken)
    if failure == "missing":
        tracker._accepted[0].cloud = None
    if failure == "nonfinite":
        tracker._accepted[0].cloud = ((float("nan"), 0.0, 0.0),)
    with pytest.raises(RuntimeError):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert not tracker._inside_model_sampling
    assert tracker.model_sampling_records[0]["status"] == "failed"


def test_context_and_record_bounds_do_not_leak_state(dense, monkeypatch):
    tracker, _, _ = setup_tracker(dense, monkeypatch)
    tracker._inside_model_sampling = True
    with pytest.raises(RuntimeError, match="nested"):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert tracker._inside_model_sampling and not tracker.model_sampling_records
    tracker._inside_model_sampling = False
    tracker.model_sampling_records[:] = [{} for _ in range(3000)]
    with pytest.raises(RuntimeError, match="3000"):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert not tracker._inside_model_sampling and len(tracker.model_sampling_records) == 3000


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sampling_profile="guess"),
        dict(keep_fraction=1.0),
        dict(voxel_size=2.0),
        dict(policy=replace(DEFAULT_POLICY, candidate_source_points=1800)),
    ],
)
def test_changed_tracker_parameters_refuse(dense, kwargs):
    with pytest.raises(ValueError, match="M8 requires"):
        dense.DenseModelSamplingTracker(
            **(
                dict(
                    backend="scipy", sampling_profile="stratified-model",
                    policy=dense.EXPECTED_POLICY,
                ) | kwargs
            )
        )


def test_dense_capacity_must_be_explicit_and_cannot_leak_into_m8(dense):
    ordinary = importlib.import_module("scanner_model_sampling")
    with pytest.raises(ValueError, match="declared-policy"):
        dense.DenseModelSamplingTracker(backend="scipy", sampling_profile="observe-stride")
    with pytest.raises(ValueError, match="declared-policy"):
        ordinary.ModelSamplingTracker(
            backend="scipy", sampling_profile="observe-stride", policy=dense.EXPECTED_POLICY
        )
    tracker = ordinary.ModelSamplingTracker(backend="scipy", sampling_profile="observe-stride")
    assert tracker._policy is DEFAULT_POLICY
    with pytest.raises(ValueError, match="bounded integer"):
        tracker.sample_indices(901, 225, position=0, frames=8)


def test_oversized_dense_retained_cloud_is_recorded_and_refused(dense, monkeypatch):
    tracker, _, _ = setup_tracker(dense, monkeypatch)
    tracker._accepted[0].cloud = tuple((float(i), 0.0, 1.0) for i in range(1801))
    with pytest.raises(RuntimeError, match="finite retained native"):
        tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    assert tracker.model_sampling_records[0]["status"] == "failed"
    assert not tracker._inside_model_sampling


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_adapter_preserves_original_reports_and_binds_separate_metadata(
    dense, monkeypatch, tmp_path, static, failure
):
    adapter = importlib.import_module("scanner_dense_model_sampling_replay")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = replay.IncrementalTracker
    output = tmp_path / "output"
    original_content = b'{"underlying": "original bytes are never rewritten"}'

    def delegated(capture, destination, *args, **kwargs):
        assert destination == output and capture == tmp_path / "capture"
        assert kwargs["diagnostic_solver_profile"] == "centered"
        assert kwargs["diagnostic_source_profile"] == "source-1800"
        if not static:
            assert args == (320, (1,))
            assert kwargs["diagnostic_scene"] and kwargs["normals_backend"] == "gpu"
        tracker = replay.IncrementalTracker(backend="scipy", policy=dense.EXPECTED_POLICY)
        assert isinstance(tracker, dense.DenseModelSamplingTracker)
        assert tracker.sampling_profile == "stratified-model"
        tracker.model_sampling_records.append(
            dict(
                profile="stratified-model",
                status="failed" if failure else "completed",
                original_reference_verified=not failure,
            )
        )
        output.mkdir()
        (output / "report.json").write_bytes(original_content)
        if failure:
            raise RuntimeError("injected original runner failure")

    monkeypatch.setattr(replay, "run_static" if static else "run", delegated)
    parameters = dict(
        sampling_profile="stratified-model", solver_profile="centered", static=static, frames=320
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected original runner failure"):
            adapter.run(tmp_path / "capture", output, dense.Factors(), **parameters)
    else:
        adapter.run(tmp_path / "capture", output, dense.Factors(), **parameters)
    assert replay.IncrementalTracker is original_factory
    assert not adapter._ACTIVE.get()
    assert (output / "report.json").read_bytes() == original_content
    metadata = json.loads((output / "model-sampling-report.json").read_text())
    assert metadata["status"] == ("failed" if failure else "completed")
    assert metadata["model_sampling_records"][0]["original_reference_verified"] == (not failure)
    assert metadata["sampling_profile"] == "stratified-model"
    assert metadata["tracking_policy"] == dense.EXPECTED_POLICY.as_dict()
    assert metadata["underlying_reports"] == [
        dict(path=str(output / "report.json"), sha256=hashlib.sha256(original_content).hexdigest())
    ]
    assert not metadata["quality_qualification_allowed"]
    assert metadata["diagnostic_source_profile"] == "source-1800"
    assert metadata["preparation_target_points"] == 1800


@pytest.mark.parametrize(
    "overrides",
    [
        dict(frames=99),
        dict(frames=3001),
        dict(static=True, frames=319),
        dict(static=True, skip_complete=12),
        dict(static=True, depth_band=(60, 300)),
        dict(sampling_profile="guess"),
        dict(solver_profile="reciprocal-origin"),
    ],
)
def test_adapter_refuses_out_of_roster_before_loading(dense, tmp_path, overrides):
    adapter = importlib.import_module("scanner_dense_model_sampling_replay")
    parameters = (
        dict(
            sampling_profile="stratified-model", solver_profile="centered", static=False, frames=320
        )
        | overrides
    )
    with pytest.raises(ValueError):
        adapter.run(tmp_path / "absent", tmp_path / "output", dense.Factors(), **parameters)
    assert not (tmp_path / "output").exists()


def test_adapter_refuses_existing_output_and_nested_scope(dense, tmp_path):
    adapter = importlib.import_module("scanner_dense_model_sampling_replay")
    parameters = dict(
        sampling_profile="stratified-model", solver_profile="centered", static=False, frames=320
    )
    with pytest.raises(ValueError, match="new output"):
        adapter.run(tmp_path / "absent", tmp_path, dense.Factors(), **parameters)
    token = adapter._ACTIVE.set(True)
    try:
        with pytest.raises(ValueError, match="nonnested"):
            adapter.run(tmp_path / "absent", tmp_path / "output", dense.Factors(), **parameters)
        assert adapter._ACTIVE.get()
    finally:
        adapter._ACTIVE.reset(token)


@pytest.mark.parametrize("static", [False, True])
def test_adapter_cli_selects_explicit_profile(dense, monkeypatch, tmp_path, static):
    adapter = importlib.import_module("scanner_dense_model_sampling_replay")
    calls = []
    monkeypatch.setattr(adapter, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "scanner_dense_model_sampling_replay.py",
            str(tmp_path / "capture"),
            "--output",
            str(tmp_path / "output"),
            "--configuration",
            "grid-pooled-residual",
            "--mechanism",
            "centered",
            "--model-sampling",
            "stratified-model",
            "--frames",
            "320",
        ]
        + (["--static"] if static else []),
    )
    adapter.main()
    assert len(calls) == 1
    assert calls[0][1]["static"] == static
    assert calls[0][1]["sampling_profile"] == "stratified-model"
