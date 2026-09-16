"""Reference-only dropout experiment: no production gate or pose-contract changes."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d.geometry import Rigid
from scanner3d.return_health import ReturnHealthTracker, return_health
from scanner3d.tracking import IncrementalTracker


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_motion_replay")


CLOUD = ((0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (0.0, 10.0, 100.0))


def admit(tracker, index, health):
    """Exercise reference lifecycle in isolation, without claiming an ICP result."""
    tracker._current_health = health
    tracker._offers_since_accepted += 1
    try:
        return tracker._admit(
            index,
            CLOUD,
            Rigid.of(Rigid.identity().rotation, (8.0 * index, 0.0, 0.0)),
            overlap=0.5,
            anchor=index == 0,
        )
    finally:
        tracker._current_health = None


def test_warning_is_adjacent_only_then_released_from_model_and_recovery(replay):
    tracker = replay.ReturnHealthReferenceTracker()
    for i in range(9):
        admit(tracker, i, return_health(0.3))
    old_keys = tracker.keyframes
    admit(tracker, 9, return_health(0.03))
    assert tracker.keyframes == old_keys and tracker.frames == 10
    assert tracker._accepted[-1].cloud is not None
    assert tracker._velocity is not None  # actual accepted continuity is preserved
    previous = tracker._accepted[-1]
    assert 9 not in tracker._reference_frames(previous, previous.pose)
    partitions = tracker._recovery_partitions(previous)
    assert partitions and all(9 not in p.frames for p, _ in partitions)
    admit(tracker, 10, return_health(0.3))
    assert tracker._accepted[9].cloud is None
    assert tracker._accepted[10].cloud is not None
    assert tracker.frames == len(tracker.poses) == 11
    previous = tracker._accepted[-1]
    selected = tracker._selected_keyframes(previous, previous.pose)
    excluded = frozenset(p.index for p in selected)
    model = tracker._reference_frames(previous, previous.pose, exclude=excluded)
    assert not (set(model) & (set(excluded) | {9, 10}))
    partitions = tracker._recovery_partitions(previous)
    assert partitions and all(9 not in p.frames for p, _ in partitions)
    assert len({f for p, _ in partitions for f in p.frames}) == sum(
        len(p.frames) for p, _ in partitions
    )


def test_warning_anchor_is_explicitly_preserved_and_flags_are_bounded(replay):
    tracker = replay.ReturnHealthReferenceTracker()
    first = tracker.offer(CLOUD, return_health=return_health(0.03))
    assert first.accepted and first.origin == "anchor" and tracker.keyframes == (0,)
    for i in range(1, 150):
        admit(tracker, i, return_health(0.03 if i % 2 else 0.3))
        assert len(tracker._withheld_clouds) <= 1
        assert tracker.retained_clouds <= tracker.policy.max_retained_clouds
        assert len(tracker.keyframes) <= tracker.policy.max_keyframes
        assert all(j == 0 or j % 2 == 0 for j in tracker.keyframes)
        assert tracker._accepted[0].cloud is not None
        assert tracker._accepted[-1].cloud is not None


def test_refused_warning_does_not_mark_next_accepted_index(replay):
    tracker = replay.ReturnHealthReferenceTracker(point_to_plane=False)
    tracker.offer(CLOUD, return_health=return_health(0.3))
    state = tracker.record()
    rejected = tracker.offer((), return_health=return_health(0.0))
    assert not rejected.accepted and rejected.frame == 1
    assert tracker.record() == state
    assert tracker._withheld_clouds == set() and tracker._current_health is None
    admit(tracker, 1, return_health(0.3))
    assert 1 in tracker.keyframes
    assert tracker._velocity is None  # refusal broke one-step motion continuity


def test_unlabelled_offer_or_seed_cannot_mutate_state(replay):
    tracker = replay.ReturnHealthReferenceTracker()
    with pytest.raises(ValueError, match="raw health"):
        tracker.offer(CLOUD)
    with pytest.raises(ValueError, match="seeded bridges"):
        tracker.seed(CLOUD, Rigid.identity())
    assert tracker.frames == 0 and tracker._offers_since_accepted == 0


def test_no_warning_archive_and_cloud_state_exactly_match_default(replay):
    baseline = IncrementalTracker()
    candidate = replay.ReturnHealthReferenceTracker()
    for i in range(100):
        pose = Rigid.of(Rigid.identity().rotation, (8.0 * i, 0.0, 0.0))
        baseline._offers_since_accepted += 1
        expected = baseline._admit(i, CLOUD, pose, overlap=0.5, anchor=i == 0)
        assert admit(candidate, i, return_health(0.3)) == expected
        assert candidate.record() == baseline.record()
        assert candidate.poses == baseline.poses
        assert candidate._accepted == baseline._accepted


def test_no_warning_real_alignment_decisions_exactly_match_default(replay):
    pytest.importorskip("scipy")
    from tests.test_tracking import _camera, _view

    baseline = IncrementalTracker(backend="scipy")
    candidate = replay.ReturnHealthReferenceTracker(backend="scipy")
    for i in range(8):
        cloud = _view(_camera(i))
        expected = baseline.offer(cloud)
        actual = candidate.offer(cloud, return_health=return_health(0.3))
        assert actual.as_dict() == expected.as_dict()
        assert candidate.poses == baseline.poses


def test_raw_health_is_causal_and_existing_history_is_not_good_frame_only():
    health = ReturnHealthTracker()
    for _ in range(15):
        assert not health.update(0.3).is_warning
    first = health.update(0.1)
    assert first.is_warning and first.baseline_fraction == 0.3
    for _ in range(15):
        last = health.update(0.1)
    assert not last.is_warning and health.baseline == 0.1
    assert health.frames == 15


@pytest.mark.parametrize("project_marker,sidecar", [(True, True), (True, False), (False, True)])
def test_failed_source_provenance_is_hashed_and_never_erased(
    replay, tmp_path, project_marker, sidecar
):
    project = tmp_path / "scan.scanner3d"
    project.write_text(json.dumps({"capture": {"tracking_lost": project_marker}}))
    loss = tmp_path / "tracking-loss.json"
    if sidecar:
        loss.write_text(json.dumps({"failed_frames": [185, 186, 187]}))
    result = replay.source_tracking_loss(tmp_path)
    assert result["project_tracking_lost"] is project_marker
    assert result["evidence"][project.name]["sha256"] == hashlib.sha256(
        project.read_bytes()
    ).hexdigest()
    assert (result["tracking_loss"] is not None) is sidecar
    if sidecar:
        assert result["evidence"][loss.name]["sha256"] == hashlib.sha256(
            loss.read_bytes()
        ).hexdigest()


def test_default_replay_of_failed_source_remains_unqualified(replay, tmp_path, monkeypatch):
    project = tmp_path / "scan.scanner3d"
    project.write_text(json.dumps({"capture": {"tracking_lost": True}}))
    raw = bytes(512000)
    monkeypatch.setattr(
        replay,
        "load_inputs",
        lambda *_a, **_kw: ([], [raw] * 100, replay.DepthBand(60, 300), []),
    )
    monkeypatch.setattr(
        replay,
        "prepare_frame",
        lambda *_a, **_kw: SimpleNamespace(points=[], deprojected_points=0, clipped_points=0),
    )
    monkeypatch.setattr(
        replay, "check_coherence", lambda *_a, **_kw: SimpleNamespace(as_dict=lambda: {})
    )
    output = tmp_path / "replay"
    replay.run(tmp_path, output, 100, (1,), normals_backend="cpu")
    for name in ("report.json", "stride-1.json"):
        report = json.loads((output / name).read_text())
        assert report["diagnostic_reference_profile"] == "default"
        assert report["quality_qualification_allowed"] is False
        assert report["source_tracking_loss"]["project_tracking_lost"] is True
    result = json.loads((output / "stride-1.json").read_text())
    assert result["offered"] == 3 and result["tracking_stopped"] is True


def test_linked_loss_provenance_is_refused(replay, tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    (tmp_path / "tracking-loss.json").symlink_to(target)
    with pytest.raises(ValueError, match="Linked tracking provenance"):
        replay.source_tracking_loss(tmp_path)
