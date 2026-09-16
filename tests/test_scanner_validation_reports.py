"""Offline audit invariants: chosen frames, hashes, and non-misleading geometry."""

import hashlib
import importlib
import json
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


@pytest.fixture
def revisit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_revisit_audit")


@pytest.mark.parametrize("drift,coherent", [(0, True), (5, False)])
@pytest.mark.parametrize("dense", [False, True])
def test_actual_best_pair_is_used_instead_of_median_lag(
    revisit, tmp_path, drift, coherent, dense, monkeypatch
):
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    checksum = hashlib.sha256(raw).hexdigest()
    inputs = [dict(path=str(tmp_path / f"frame_{i:04d}.raw"), sha256=checksum) for i in range(6)]
    for i in (0, 3, 5) if dense else (0, 5):
        Path(inputs[i]["path"]).write_bytes(raw)
    identity = dict(rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0, 0, 0])
    record = dict(
        diagnostic_scene=True,
        diagnostic_reference_profile="retention64",
        tracking_policy={"max_keyframes": 64},
        normals_backend="gpu",
        quality_qualification_allowed=False,
        source_scene_validity={"reason": "deliberately unqualified test scene"},
        capture=str(tmp_path),
        depth_band=dict(near=60, far=300),
        input_frames=inputs,
        runs=[
            dict(
                stride=2,
                offered=3,
                accepted=3,
                accepted_source_indices=[0, 3, 5],
                poses=[identity, identity, {**identity, "translation": [0, 0, drift]}],
                coherence=dict(
                    coherent=True,
                    period_frames=1,
                    period_estimate=dict(
                        samples=[
                            dict(frame=0, best_frame=2, credible=True, median_depth_difference=0)
                        ]
                    ),
                ),
            )
        ],
    )
    estimate = record["runs"][0]["coherence"]["period_estimate"]

    def image_match(frames, *, depth_scale, samples):
        assert dense and samples == 25 and depth_scale > 0
        assert frames == [raw, raw, raw]
        return SimpleNamespace(as_dict=lambda: estimate)

    monkeypatch.setattr(revisit, "image_period", image_match)
    result = revisit.audit(record, dense=dense)
    assert result["quality_qualification_allowed"] is False
    report = result["runs"][0]
    assert report["quality_qualification_allowed"] is False
    assert report["source_scene_validity"] == record["source_scene_validity"]
    assert report["tracking_policy"] == record["tracking_policy"]
    assert report["diagnostic_reference_profile"] == "retention64"
    assert report["normals_backend"] == "gpu"
    assert report["pairs"][0]["source_indices"] == [0, 5]
    assert report["individually_matched_coherent"] is coherent
    # The unused median-lag frame is deliberately absent: it must not be read.
    assert Path(inputs[3]["path"]).exists() is dense
    inputs[5]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        revisit.audit(record, dense=dense)
    if dense:
        inputs[5]["sha256"] = checksum
        inputs[3]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="hash changed"):
            revisit.audit(record, dense=True)


def test_plane_report_rejects_unsupported_geometry(revisit):
    planes = importlib.import_module("stationary_plane_report")
    with pytest.raises(ValueError, match="Insufficient"):
        planes.fitted_plane(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="convincingly planar"):
        planes.fitted_plane(np.random.default_rng(4).normal(size=(1000, 3)))
    yy, xx = np.indices((10, 10))
    points = np.column_stack([xx.ravel(), yy.ravel(), (2 * xx + 3 * yy + 1).ravel()])
    _, _, errors = planes.fitted_plane(points)
    assert np.max(np.abs(errors)) < 1e-12


def test_dense_revisit_inconclusive_does_not_inherit_sparse_pass(revisit, tmp_path):
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    rows = []
    identity = dict(rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0, 0, 0])
    for i in range(2):
        path = tmp_path / f"frame_{i:04d}.raw"
        path.write_bytes(raw)
        rows.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(
        capture=str(tmp_path),
        depth_band=dict(near=60, far=300),
        input_frames=rows,
        runs=[
            dict(
                stride=1,
                offered=2,
                accepted=2,
                accepted_source_indices=[0, 1],
                poses=[identity, identity],
                coherence=dict(coherent=True, period_estimate=None),
            )
        ],
    )
    result = revisit.audit(record, dense=True)["runs"][0]
    assert result["legacy_fixed_lag_coherent"] is True
    assert result["individually_matched_coherent"] is None
    assert result["weakest_individually_matched_agreement"] is None
    assert result["audit_period_estimate"] is None
    assert result["pairs"] == []
    assert record["runs"][0]["coherence"] == dict(coherent=True, period_estimate=None)


def test_plane_report_retains_bad_patch_and_continues_other_regions(revisit, tmp_path):
    planes = importlib.import_module("stationary_plane_report")
    depth = np.full((20, 20), 100.0)
    frequency = np.ones_like(depth)
    frequency[:10, :10] = 0  # Too few reference points; must remain in report.
    np.savez(
        tmp_path / "spatial.npz",
        reference_mm=depth,
        reference_frequency=frequency,
        median_0_mm=depth,
        frequency_0=frequency,
        mad_0_mm=np.zeros_like(depth),
    )
    (tmp_path / "report.json").write_text(
        json.dumps(dict(runs=[dict(timing_us=20000, exposure_us=20000)]))
    )
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps(dict(regions={"bad": [0, 0, 10, 10], "good": [10, 10, 20, 20]})))
    result = planes.run(tmp_path, regions)["regions"]
    assert result["bad"]["status"] == "invalid_reference_patch"
    assert result["bad"]["runs"] == []
    assert "Insufficient" in result["bad"]["error"]
    assert result["good"]["runs"][0]["flatness_rms_mm"] < 1e-10


def test_evidence_manifest_hashes_once_and_refuses_overwrite(revisit, tmp_path):
    evidence = importlib.import_module("scanner_evidence_manifest")
    raw = tmp_path / "frame.raw"
    raw.write_bytes(b"preserved measurement")
    output = tmp_path / "manifest.json"
    report = evidence.manifest([tmp_path, tmp_path], output)
    assert report["file_count"] == 1
    assert report["total_bytes"] == len(raw.read_bytes())
    assert report["files"] == [
        dict(
            path=str(raw),
            bytes=raw.stat().st_size,
            sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
    ]
    saved = output.read_bytes()
    with pytest.raises(ValueError, match="new output"):
        evidence.manifest([tmp_path], output)
    assert output.read_bytes() == saved


def test_evidence_manifest_rejects_symlinks(revisit, tmp_path):
    evidence = importlib.import_module("scanner_evidence_manifest")
    (tmp_path / "link").symlink_to(tmp_path / "absent")
    output = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="symlinks"):
        evidence.manifest([tmp_path], output)
    assert not output.exists()


@pytest.fixture
def raw_motion_source(revisit, tmp_path):
    replay = importlib.import_module("scanner_motion_replay")
    capture = tmp_path / "raw"
    capture.mkdir()
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    rows = []
    for i in range(136):
        name = f"frame_{i:04d}.raw"
        (capture / name).write_bytes(raw)
        rows.append(
            dict(
                path=name,
                delivery_index=i,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                monotonic_s=10 + i * 0.2,
            )
        )
    (capture / "capture-progress.json").write_text(
        json.dumps(dict(capture_error=None, frames=rows))
    )
    (tmp_path / "report.json").write_text(
        json.dumps(dict(status="completed", restoration_verified=True))
    )
    return replay, capture


def test_motion_loader_preserves_actual_timestamps_and_settling(raw_motion_source):
    replay, capture = raw_motion_source
    paths, raws, band, inputs = replay.load_inputs(capture, 100, 12, (60, 300))
    assert len(raws) == 100 and paths[0].name == "frame_0012.raw"
    assert inputs[0]["monotonic_s"] == pytest.approx(12.4)
    assert band.as_dict() == dict(near=60, far=300)


def test_starting_view_offset_is_diagnostic_only_and_preserves_provenance(raw_motion_source):
    replay, capture = raw_motion_source
    with pytest.raises(ValueError, match="diagnostic-scene"):
        replay.load_inputs(capture, 100, 12, (60, 300), start_offset=8)
    paths, _, _, inputs = replay.load_inputs(
        capture, 100, 12, (60, 300), diagnostic_scene=True, start_offset=8
    )
    assert paths[0].name == "frame_0020.raw"
    assert inputs[0]["delivery_index"] == 20
    assert inputs[0]["monotonic_s"] == pytest.approx(14.0)
    for offset in (-1, 3001, 100):
        with pytest.raises(ValueError):
            replay.load_inputs(
                capture, 100, 12, (60, 300), diagnostic_scene=True, start_offset=offset
            )


@pytest.mark.parametrize(
    "problem", ["failed", "unrestored", "hash", "time", "short_settle", "band", "scene"]
)
def test_motion_loader_refuses_invalid_provenance(raw_motion_source, problem):
    replay, capture = raw_motion_source
    meta_path = capture / "capture-progress.json"
    metadata = json.loads(meta_path.read_text())
    if problem in ("failed", "unrestored"):
        (capture.parent / "report.json").write_text(
            json.dumps(
                dict(
                    status="failed" if problem == "failed" else "completed",
                    restoration_verified=False,
                )
            )
        )
    if problem == "hash":
        metadata["frames"][12]["sha256"] = "0" * 64
    if problem == "time":
        metadata["frames"][13]["monotonic_s"] = metadata["frames"][12]["monotonic_s"]
    if problem == "scene":
        (capture.parent / "quality-validity.json").write_text(
            json.dumps(dict(quality_qualification_allowed=False))
        )
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        replay.load_inputs(
            capture,
            100,
            0 if problem == "short_settle" else 12,
            None if problem == "band" else (60, 300),
        )
    if problem != "scene":
        with pytest.raises(ValueError):
            replay.load_inputs(
                capture,
                100,
                0 if problem == "short_settle" else 12,
                None if problem == "band" else (60, 300),
                diagnostic_scene=True,
            )


@pytest.mark.parametrize(
    "profile,normals_backend",
    [
        (profile, mode)
        for profile in (
            "default",
            "retention64",
            "no-model",
            "stable64",
            "keyframe64",
            "untrimmed",
            "point-to-point",
            "return-health",
        )
        for mode in ("cpu", "gpu")
        if profile != "point-to-point" or mode == "cpu"
    ],
)
@pytest.mark.parametrize("flagged_scene", [False, True])
def test_diagnostic_scene_and_reference_profile_preserve_qualification(
    raw_motion_source, monkeypatch, normals_backend, profile, flagged_scene
):
    replay, capture = raw_motion_source
    original_preference = replay.tracking_module.PREFERENCE_ORDER
    expected_preference = (
        ("keyframe", "model", "adjacent") if profile == "keyframe64" else original_preference
    )
    validity = dict(quality_qualification_allowed=False, reason="base at image edge")
    if flagged_scene:
        (capture.parent / "quality-validity.json").write_text(json.dumps(validity))
        with pytest.raises(ValueError, match="scene-validity"):
            replay.load_inputs(capture, 100, 12, (60, 300))
    _, raws, _, inputs = replay.load_inputs(
        capture, 100, 12, (60, 300), diagnostic_scene=flagged_scene
    )
    assert len(raws) == len(inputs) == 100

    @contextmanager
    def mock_normals(_surface, message):
        assert normals_backend == "gpu", "CPU replay must not open a GPU context"
        message("GPU-assisted tracking: mock only")
        yield

    class RefusingTracker:
        def __init__(self, *, backend, policy, point_to_plane, keep_fraction):
            assert backend == "scipy"
            assert policy == replay.reference_policy(profile)
            assert point_to_plane is (profile != "point-to-point")
            assert keep_fraction == (1.0 if profile == "untrimmed" else 0.8)
            self.poses = []

        def offer(self, _points, **kwargs):
            assert expected_preference == replay.tracking_module.PREFERENCE_ORDER
            assert ("return_health" in kwargs) is (profile == "return-health")
            return SimpleNamespace(accepted=False, as_dict=lambda: {"accepted": False})

    monkeypatch.setattr(replay, "live_normals", mock_normals)
    monkeypatch.setattr(replay, "IncrementalTracker", RefusingTracker)
    monkeypatch.setattr(replay, "StableReferenceTracker", RefusingTracker)
    monkeypatch.setattr(replay, "ReturnHealthReferenceTracker", RefusingTracker)
    monkeypatch.setattr(
        replay,
        "prepare_frame",
        lambda *_a, **_kw: SimpleNamespace(points=[], deprojected_points=0, clipped_points=0),
    )
    monkeypatch.setattr(
        replay, "check_coherence", lambda *_a, **_kw: SimpleNamespace(as_dict=lambda: {})
    )
    output = capture.parent / "diagnostic-replay"
    replay.run(
        capture,
        output,
        100,
        (1,),
        skip_complete=12,
        depth_band=(60, 300),
        diagnostic_scene=flagged_scene,
        start_offset=8 if flagged_scene else 0,
        normals_backend=normals_backend,
        diagnostic_reference_profile=profile,
    )
    assert original_preference == replay.tracking_module.PREFERENCE_ORDER
    for name in ("report.json", "stride-1.json"):
        report = json.loads((output / name).read_text())
        assert report["diagnostic_scene"] is flagged_scene
        assert report["start_offset"] == (8 if flagged_scene else 0)
        assert report["normals_backend"] == normals_backend
        assert report["diagnostic_reference_profile"] == profile
        assert report["tracking_policy"] == replay.reference_policy(profile).as_dict()
        assert report["candidate_preference"] == list(expected_preference)
        assert report["alignment_keep_fraction"] == (1.0 if profile == "untrimmed" else 0.8)
        assert report["alignment_method"] == (
            "point_to_point" if profile == "point-to-point" else "point_to_plane"
        )
        assert report["quality_qualification_allowed"] is (
            not flagged_scene and profile == "default"
        )
        assert report["source_scene_validity"] == (validity if flagged_scene else None)


def test_reference_ablations_are_closed_and_leave_gates_unchanged(revisit):
    replay = importlib.import_module("scanner_motion_replay")
    default = replay.DEFAULT_POLICY
    assert replay.reference_policy("default") is default
    for name, changed in (
        ("retention64", {"max_keyframes": 64, "max_retained_clouds": 96}),
        ("keyframe64", {"max_keyframes": 64, "max_retained_clouds": 96}),
        ("untrimmed", {}),
        ("point-to-point", {}),
        ("return-health", {}),
        ("no-model", {"use_model": False}),
        (
            "stable64",
            {"max_keyframes": 64, "max_retained_clouds": 96, "protected_recent_keyframes": 0},
        ),
    ):
        policy = replay.reference_policy(name)
        actual = {
            field.name: getattr(policy, field.name)
            for field in fields(default)
            if getattr(default, field.name) != getattr(policy, field.name)
        }
        assert actual == changed
    with pytest.raises(ValueError, match="reference profile"):
        replay.reference_policy("relaxed-gates")


def test_point_to_point_gpu_mode_is_refused_before_reading_inputs(revisit, tmp_path, monkeypatch):
    replay = importlib.import_module("scanner_motion_replay")

    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid mode must not read inputs or initialize GPU compute")

    monkeypatch.setattr(replay, "load_inputs", unexpected)
    monkeypatch.setattr(replay, "live_normals", unexpected)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="requires cpu"):
        replay.run(
            tmp_path / "missing-source",
            output,
            100,
            (1,),
            normals_backend="gpu",
            diagnostic_reference_profile="point-to-point",
        )
    assert not output.exists()


def test_stable_archive_preserves_old_reference_and_separate_recent_cloud(revisit):
    replay = importlib.import_module("scanner_motion_replay")
    from scanner3d.geometry import Rigid

    tracker = replay.StableReferenceTracker(policy=replay.reference_policy("stable64"))
    identity = Rigid.identity()
    cloud = ((0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (0.0, 10.0, 100.0))
    for i in range(64):
        pose = Rigid.of(identity.rotation, (8.0 * i, 0.0, 0.0))
        tracker._admit(i, cloud, pose, overlap=0.5, anchor=i == 0)
    assert tracker.keyframes == tuple(range(64))
    # A nearby but non-stationary revisit is redundant with old frame 10.
    decisions = tracker._admit(
        64, cloud, Rigid.of(identity.rotation, (81.0, 0.0, 0.0)), overlap=0.5, anchor=False
    )
    assert any(d.action == "evict" and d.frame == 64 for d in decisions)
    assert tracker.keyframes == tuple(range(64))
    assert tracker._accepted[-1].cloud is not None
    # A genuinely new distant view can replace a redundant older pair.
    tracker._admit(
        65, cloud, Rigid.of(identity.rotation, (520.0, 0.0, 0.0)), overlap=0.5, anchor=False
    )
    assert 65 in tracker.keyframes and tracker.keyframes[0] == 0
    assert len(tracker.keyframes) == 64
    assert tracker.retained_clouds <= tracker.policy.max_retained_clouds


def test_replay_failure_preserves_partial_poses_and_restores_preference(
    raw_motion_source, monkeypatch
):
    replay, capture = raw_motion_source
    original = replay.tracking_module.PREFERENCE_ORDER

    class FailingTracker:
        def __init__(self, **_kwargs):
            self.poses = []

        def offer(self, _points):
            assert replay.tracking_module.PREFERENCE_ORDER == ("keyframe", "model", "adjacent")
            if self.poses:
                raise RuntimeError("synthetic coordinate failure")
            self.poses.append(replay.Rigid.identity())
            return SimpleNamespace(accepted=True, as_dict=lambda: {"accepted": True})

    monkeypatch.setattr(replay, "IncrementalTracker", FailingTracker)
    monkeypatch.setattr(
        replay,
        "prepare_frame",
        lambda *_a, **_k: SimpleNamespace(points=[], deprojected_points=0, clipped_points=0),
    )
    output = capture.parent / "failed-partial"
    with pytest.raises(RuntimeError, match="coordinate failure"):
        replay.run(
            capture,
            output,
            100,
            (1,),
            skip_complete=12,
            depth_band=(60, 300),
            normals_backend="cpu",
            diagnostic_reference_profile="keyframe64",
        )
    assert original == replay.tracking_module.PREFERENCE_ORDER
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["runs"] == []
    partial = report["failed_run"]
    assert partial["offered_before_error"] == partial["accepted_before_error"] == 1
    assert partial["source_index_in_progress"] == 1
    assert partial["accepted_source_indices"] == [0]
    assert len(partial["poses"]) == len(partial["decisions"]) == 1
    assert partial["quality_qualification_allowed"] is False


@pytest.mark.parametrize("score,expected", [(1.0, None), (0.5, False)])
def test_revisit_partial_projection_never_becomes_clean_pass(
    revisit, tmp_path, monkeypatch, score, expected
):
    raw = np.full((400, 640), 30000, dtype="<u2").tobytes()
    rows = []
    identity = dict(rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0, 0, 0])
    for i in range(3):
        path = tmp_path / f"frame_{i:04d}.raw"
        path.write_bytes(raw)
        rows.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(
        capture=str(tmp_path),
        depth_band=dict(near=60, far=300),
        input_frames=rows,
        runs=[
            dict(
                stride=1,
                offered=3,
                accepted=3,
                accepted_source_indices=[0, 1, 2],
                poses=[identity] * 3,
                coherence=dict(
                    coherent=True,
                    pair_selection="actual_image_match",
                    period_estimate=dict(
                        samples=[
                            dict(frame=0, best_frame=i, credible=True, median_depth_difference=0)
                            for i in (1, 2)
                        ]
                    ),
                ),
            )
        ],
    )
    answers = iter((score, None))

    def project(*_args, **_kwargs):
        value = next(answers)
        return SimpleNamespace(as_dict=lambda: {"agreement": value})

    monkeypatch.setattr(revisit, "period_agreement", project)
    result = revisit.audit(record)["runs"][0]
    assert result["legacy_fixed_lag_coherent"] is None
    assert result["source_coherence_coherent"] is True
    assert result["individually_matched_coherent"] is expected
    assert result["uncheckable_pair_count"] == 1


def test_motion_framing_retains_empty_and_border_cases(revisit):
    framing = importlib.import_module("scanner_motion_framing")
    raw = np.zeros((10, 12), dtype=np.uint16)
    assert framing.measurements(raw)["bbox_xyxy"] is None
    raw[0:4, 3:8] = 20000
    result = framing.measurements(raw)
    assert result["bbox_xyxy"] == [3, 0, 8, 4]
    assert result["component_pixels"] == 20 and result["border_pixels"] == 5
