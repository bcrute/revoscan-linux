"""Closing a tracking loss, and every way a recovery can fail to close one.

What this is about
------------------
Task 020 made a lost capture say so and refuse. Refusing is honest and it is
not what the operator wants: the frames before the loss are real, the object is
still there, and the scan can be pointed back at surface it has already seen -
which is exactly the situation a PAUSE boundary already handles. So a recovery
is an ordinary ``segment-*`` directory carrying one thing more: a ``recovery``
object BINDING it to the loss it closes.

The binding is the whole subject of this file. A bridge that merely happens to
follow a loss record proves nothing - the frames between them are frames no
pose was ever accepted for - so a reader closing a loss on adjacency alone
would be deciding where somebody's scan resumes from directory order. Most of
what is below is therefore a valid recovered capture with one thing changed,
and every one of those refuses.

The live half - when the capture starts looking, what it holds back, what it
publishes and what the window says - is in ``test_gui_tracking_recovery.py``.
Frames here are plain deterministic bytes, as they are in
``test_capture_sequence.py``: nothing in this module knows what depth data is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scanner3d.capture_sequence import (
    BRIDGE_RECORD_NAME,
    FRAME_TEMPLATE,
    RECOVERY_EXCLUDED_LIMITATION,
    RECOVERY_KIND,
    RECOVERY_LIMITATIONS,
    RECOVERY_RECORD_KEY,
    RECOVERY_RELOCALIZED_LIMITATION,
    RECOVERY_SCHEMA_VERSION,
    TRACKING_LOSS_RECORD_NAME,
    AlignmentFailure,
    BridgeRecordUnusable,
    CaptureSequenceUnreadable,
    SegmentPublicationRefused,
    TrackingLossUnresolved,
    publish_recovery_segment,
    publish_segment,
    publish_tracking_loss,
    read_capture_sequence,
    read_tracking_loss,
)
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.pointcloud import sha256_of
from scanner3d.relocalize import MIN_TENTATIVE_FRAMES

# --- building capture directories by hand --------------------------------------


def frame_bytes(index: int) -> bytes:
    return bytes([index % 251]) * 64


def write_frames(directory: Path, indices: range | list[int]) -> None:
    for index in indices:
        (directory / FRAME_TEMPLATE.format(index=index)).write_bytes(frame_bytes(index))


def transform(seed: float) -> Rigid:
    """Neither identity nor axis aligned, so a dropped rotation would show."""
    return rotation_about_axis(
        (0.4, 0.6, 0.6931), math.radians(10.0 + seed), through=(1.0, -2.0, 5.0)
    )


def local_edge(frame: int) -> dict[str, object]:
    return {
        "frame": frame,
        "rmse": 0.02,
        "converged": True,
        "iterations": 11,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
        "accepted": True,
        "refusals": [],
    }


def accepted_evidence(batch_count: int = MIN_TENTATIVE_FRAMES) -> dict[str, object]:
    return {
        "accepted": True,
        "thresholds": {"min_tentative_frames": MIN_TENTATIVE_FRAMES},
        "candidates": [{"residual": 0.01}],
        "local_chain": {
            "gates": {"local_max_residual": 1.5},
            "edges": [local_edge(frame) for frame in range(1, max(batch_count, 1))],
            "anchor_frames": 1,
        },
    }


def failures(*indices: int) -> list[AlignmentFailure]:
    return [
        AlignmentFailure(
            frame_index=index,
            failure_type="RegistrationError",
            message=f"point-to-plane singular in translation z at frame {index}",
        )
        for index in indices
    ]


def lose_tracking(run: Path, *, last_aligned: int) -> Path:
    """Write the three failed frames and publish the record that names them.

    Both halves, always. A loss record whose failed frames are not on disk
    describes some other capture, and a fixture that produced one would be
    testing the reader against a directory the writer cannot make.
    """
    failed = [last_aligned + 1, last_aligned + 2, last_aligned + 3]
    write_frames(run, failed)
    return publish_tracking_loss(
        run, last_aligned_frame_index=last_aligned, failures=failures(*failed)
    )


def recover(capture: Path, run: Path, *, start: int, count: int = MIN_TENTATIVE_FRAMES) -> Path:
    """Publish one accepted recovery of the loss recorded in ``run``."""
    loss = read_tracking_loss(run)
    assert loss is not None
    publication = publish_recovery_segment(
        capture,
        loss_dir=run,
        loss=loss,
        frames=[(start + offset, frame_bytes(start + offset)) for offset in range(count)],
        transform=transform(3.0),
        evidence=accepted_evidence(count),
        limitations=["a recovery, joined by a recorded relocalization"],
    )
    return publication.directory


def pause_segment(capture: Path, *, predecessor: int, count: int = MIN_TENTATIVE_FRAMES) -> Path:
    """One ordinary accepted pause boundary, with no recovery in it."""
    start = predecessor + 1
    publication = publish_segment(
        capture,
        frames=[(start + offset, frame_bytes(start + offset)) for offset in range(count)],
        predecessor_index=predecessor,
        transform=transform(1.0),
        evidence=accepted_evidence(count),
        limitations=["an ordinary pause boundary"],
    )
    return publication.directory


def recovered_capture(tmp_path: Path) -> Path:
    """Frames 0-4 aligned, 5-7 lost, and 8-10 recovered against frame 4."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    recover(tmp_path, tmp_path, start=8)
    return tmp_path


def record_of(segment: Path) -> dict[str, object]:
    payload = json.loads((segment / BRIDGE_RECORD_NAME).read_text())
    assert isinstance(payload, dict)
    return payload


def rewrite(segment: Path, record: dict[str, object]) -> None:
    (segment / BRIDGE_RECORD_NAME).write_text(json.dumps(record))


def break_recovery(segment: Path, **changes: object) -> None:
    """Change one field of a KNOWN-GOOD recovery, leaving everything else."""
    record = record_of(segment)
    recovery = record[RECOVERY_RECORD_KEY]
    assert isinstance(recovery, dict)
    for key, value in changes.items():
        if value is ...:
            del recovery[key]
        else:
            recovery[key] = value
    rewrite(segment, record)


# --- a recovered capture reads back as one sequence ----------------------------


def test_a_recovered_capture_hands_over_the_prefix_and_the_recovered_frames(
    tmp_path: Path,
) -> None:
    """The valid prefix, then the recovered segment, and nothing in between.

    Frames 5, 6 and 7 were written while tracking was lost. They are on disk,
    they are named in the loss record, and they are NOT in the sequence: no
    pose was ever accepted for them and none was estimated afterwards.
    """
    capture = recovered_capture(tmp_path)

    sequence = read_capture_sequence(capture)

    assert [frame.index for frame in sequence.frames] == [0, 1, 2, 3, 4, 8, 9, 10]
    assert [frame.index for frame in sequence.excluded] == [5, 6, 7]
    assert all(frame.path.exists() for frame in sequence.excluded)
    assert len(sequence.recoveries) == 1
    recovery = sequence.recoveries[0]
    assert recovery.run is None
    assert recovery.excluded_indices == (5, 6, 7)
    assert recovery.loss.last_aligned_frame_index == 4
    assert recovery.segment == sequence.bridges[0].segment


def test_the_recovered_seam_is_an_ordinary_bridge_boundary(tmp_path: Path) -> None:
    """Whatever reconstructs this must not fit ICP across the loss.

    The pair either side of the seam - frame 4 and frame 8 - is exactly the
    pair that must not be aligned, and the boundary is how a reconstruction is
    told so. It is expressed as a POSITION in the frame list, and the excluded
    frames are not in that list, so the two are adjacent there.
    """
    capture = recovered_capture(tmp_path)

    sequence = read_capture_sequence(capture)
    boundaries = sequence.boundaries()

    assert len(boundaries) == 1
    assert boundaries[0].position == 5
    assert sequence.frames[4].index == 4
    assert sequence.frames[5].index == 8
    assert boundaries[0].transform == transform(3.0)


def test_the_capture_and_the_boundary_both_say_what_was_left_out(tmp_path: Path) -> None:
    """A caveat nobody reads is not a limitation. Both places carry it."""
    capture = recovered_capture(tmp_path)

    sequence = read_capture_sequence(capture)

    stated = "\n".join(sequence.limitations)
    assert "lost tracking 1 time(s)" in stated
    assert "5, 6, 7" in stated
    assert "NOT part of this sequence" in stated
    assert RECOVERY_EXCLUDED_LIMITATION in sequence.bridges[0].limitations
    assert RECOVERY_RELOCALIZED_LIMITATION in sequence.bridges[0].limitations
    assert RECOVERY_EXCLUDED_LIMITATION in sequence.boundaries()[0].limitations


def test_the_recovery_carries_the_relocalization_evidence_a_pause_boundary_does(
    tmp_path: Path,
) -> None:
    """A recovery is not a weaker bridge. It clears the same evidence gates."""
    capture = recovered_capture(tmp_path)

    sequence = read_capture_sequence(capture)

    evidence = sequence.bridges[0].evidence
    assert evidence["accepted"] is True
    assert evidence["thresholds"]
    assert evidence["candidates"]
    chain = evidence["local_chain"]
    assert isinstance(chain, dict)
    assert chain["anchor_frames"] == 1


def test_striding_keeps_both_frames_of_a_recovered_seam(tmp_path: Path) -> None:
    """The stride policy is per run, so the bridged pair survives every stride."""
    capture = recovered_capture(tmp_path)

    sequence = read_capture_sequence(capture).strided(3)

    indices = [frame.index for frame in sequence.frames]
    assert 4 in indices and 8 in indices
    assert sequence.boundaries()[0].position == indices.index(8)


# --- the loss is not closed by presence, order or plausibility -----------------


def test_a_loss_with_nothing_after_it_still_refuses_exactly_as_before(
    tmp_path: Path,
) -> None:
    """Task 020's containment, unchanged. This is the case that must not move."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)

    with pytest.raises(TrackingLossUnresolved, match="recorded a tracking loss"):
        read_capture_sequence(tmp_path)


def test_an_ordinary_pause_segment_does_not_close_a_loss(tmp_path: Path) -> None:
    """The exact shape a reader closing on adjacency alone would accept.

    A perfectly valid pause boundary, published after a loss, bridging from the
    frame the loss names. It carries no binding, so it closes nothing.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    # Bridging from frame 7, the last frame on disk, so that it is a boundary
    # the reader would otherwise be perfectly happy with.
    pause_segment(tmp_path, predecessor=7)

    with pytest.raises(TrackingLossUnresolved, match="carries no recovery record"):
        read_capture_sequence(tmp_path)


def test_a_recovery_with_no_loss_before_it_refuses(tmp_path: Path) -> None:
    """A recovery is the answer to a loss. One standing alone is refused.

    Not read as an ordinary pause boundary either: a capture holding something
    recovery-shaped is never handed over as though it held nothing.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    segment = recover(tmp_path, tmp_path, start=8)
    (tmp_path / TRACKING_LOSS_RECORD_NAME).unlink()

    with pytest.raises(BridgeRecordUnusable, match="recorded no tracking loss"):
        read_capture_sequence(tmp_path)
    assert (segment / BRIDGE_RECORD_NAME).exists()


def test_a_second_recovery_for_a_loss_already_closed_refuses(tmp_path: Path) -> None:
    """Duplicated, by copying the closing segment's record into a later one."""
    capture = recovered_capture(tmp_path)
    first = capture / "segment-0001"
    second = pause_segment(capture, predecessor=10)
    (second / BRIDGE_RECORD_NAME).write_bytes((first / BRIDGE_RECORD_NAME).read_bytes())

    with pytest.raises(BridgeRecordUnusable, match="recorded no tracking loss"):
        read_capture_sequence(capture)


def test_a_recovery_naming_a_loss_record_somewhere_else_refuses(tmp_path: Path) -> None:
    """The path is compared, never followed. Nothing outside the run is opened."""
    capture = recovered_capture(tmp_path)
    break_recovery(
        capture / "segment-0001",
        loss_record={
            "path": f"../{TRACKING_LOSS_RECORD_NAME}",
            "byte_count": 10,
            "sha256": "0" * 64,
        },
    )

    with pytest.raises(BridgeRecordUnusable, match="closes the loss recorded at"):
        read_capture_sequence(capture)


def test_a_loss_record_edited_after_the_recovery_was_published_refuses(
    tmp_path: Path,
) -> None:
    """The binding is by bytes for exactly this.

    A record rewritten to name a different last-aligned frame would otherwise
    still look closed, and the sequence would silently start somewhere else.
    """
    capture = recovered_capture(tmp_path)
    record = json.loads((capture / TRACKING_LOSS_RECORD_NAME).read_text())
    record["last_aligned_frame_index"] = 3
    record["first_failed_frame_index"] = 4
    record["triggering_frame_index"] = 6
    record["failures"] = [
        {"frame_index": index, "failure_type": "RegistrationError", "message": "edited"}
        for index in (4, 5, 6)
    ]
    (capture / TRACKING_LOSS_RECORD_NAME).write_text(json.dumps(record))

    with pytest.raises(BridgeRecordUnusable):
        read_capture_sequence(capture)


def test_a_loss_record_replaced_by_a_symlink_refuses(tmp_path: Path) -> None:
    """A capture's own record of its own loss is read as bytes inside it."""
    capture = recovered_capture(tmp_path)
    elsewhere = tmp_path.parent / "elsewhere.json"
    elsewhere.write_bytes((capture / TRACKING_LOSS_RECORD_NAME).read_bytes())
    (capture / TRACKING_LOSS_RECORD_NAME).unlink()
    (capture / TRACKING_LOSS_RECORD_NAME).symlink_to(elsewhere)

    with pytest.raises(CaptureSequenceUnreadable, match="symbolic link"):
        read_capture_sequence(capture)


def test_an_excluded_frame_replaced_by_a_symlink_refuses(tmp_path: Path) -> None:
    """The excluded frames are evidence, and evidence is read where it lies."""
    capture = recovered_capture(tmp_path)
    frame = capture / FRAME_TEMPLATE.format(index=6)
    elsewhere = tmp_path.parent / "frame.raw"
    elsewhere.write_bytes(frame.read_bytes())
    frame.unlink()
    frame.symlink_to(elsewhere)

    with pytest.raises(BridgeRecordUnusable, match="symbolic link"):
        read_capture_sequence(capture)


def test_an_excluded_frame_whose_bytes_changed_refuses(tmp_path: Path) -> None:
    """Evidence that has changed is not evidence."""
    capture = recovered_capture(tmp_path)
    (capture / FRAME_TEMPLATE.format(index=6)).write_bytes(b"x" * len(frame_bytes(6)))

    with pytest.raises(BridgeRecordUnusable, match="does not hash to the digest"):
        read_capture_sequence(capture)


def test_an_excluded_frame_that_has_gone_missing_refuses(tmp_path: Path) -> None:
    """A frame the recovery says is on disk and is not."""
    capture = recovered_capture(tmp_path)
    (capture / FRAME_TEMPLATE.format(index=6)).unlink()

    with pytest.raises(CaptureSequenceUnreadable):
        read_capture_sequence(capture)


def test_a_recovery_that_excludes_a_different_frame_refuses(tmp_path: Path) -> None:
    """The excluded frames are the streak the loss lists, in order."""
    capture = recovered_capture(tmp_path)
    record = record_of(capture / "segment-0001")
    recovery = record[RECOVERY_RECORD_KEY]
    assert isinstance(recovery, dict)
    entries = recovery["excluded_frames"]
    assert isinstance(entries, list)
    entries[1]["frame_index"] = 9
    entries[1]["file"] = FRAME_TEMPLATE.format(index=9)
    rewrite(capture / "segment-0001", record)

    with pytest.raises(BridgeRecordUnusable, match="excludes frame 9"):
        read_capture_sequence(capture)


def test_a_recovery_that_excludes_the_wrong_number_of_frames_refuses(
    tmp_path: Path,
) -> None:
    """Every frame written while tracking was lost is named, or the count is wrong."""
    capture = recovered_capture(tmp_path)
    record = record_of(capture / "segment-0001")
    recovery = record[RECOVERY_RECORD_KEY]
    assert isinstance(recovery, dict)
    entries = recovery["excluded_frames"]
    assert isinstance(entries, list)
    del entries[-1]
    rewrite(capture / "segment-0001", record)

    with pytest.raises(BridgeRecordUnusable, match="excludes 2 frame"):
        read_capture_sequence(capture)


def test_a_recovery_whose_excluded_frame_names_a_path_refuses(tmp_path: Path) -> None:
    """The name is rebuilt from the index, so nothing outside the run opens."""
    capture = recovered_capture(tmp_path)
    record = record_of(capture / "segment-0001")
    recovery = record[RECOVERY_RECORD_KEY]
    assert isinstance(recovery, dict)
    entries = recovery["excluded_frames"]
    assert isinstance(entries, list)
    entries[0]["file"] = "../../frame_0005.raw"
    rewrite(capture / "segment-0001", record)

    with pytest.raises(BridgeRecordUnusable, match="Nothing outside the run is opened"):
        read_capture_sequence(capture)


@pytest.mark.parametrize("version", [0, 2, "1", None, True])
def test_a_recovery_in_a_schema_this_reader_does_not_know_refuses(
    tmp_path: Path, version: object
) -> None:
    """A dialect this cannot read is refused, never guessed at."""
    capture = recovered_capture(tmp_path)
    break_recovery(capture / "segment-0001", schema_version=version)

    with pytest.raises(BridgeRecordUnusable, match="schema version"):
        read_capture_sequence(capture)


def test_a_recovery_of_the_wrong_kind_refuses(tmp_path: Path) -> None:
    capture = recovered_capture(tmp_path)
    break_recovery(capture / "segment-0001", kind="scanner3d.relocalization-bridge")

    with pytest.raises(BridgeRecordUnusable, match="says kind"):
        read_capture_sequence(capture)


@pytest.mark.parametrize("key", ["predecessor_frame_index", "segment_start_frame_index"])
def test_a_recovery_that_disagrees_with_its_own_bridge_refuses(tmp_path: Path, key: str) -> None:
    """Repeated inside the binding on purpose, so the two can be compared."""
    capture = recovered_capture(tmp_path)
    break_recovery(capture / "segment-0001", **{key: 99})

    with pytest.raises(BridgeRecordUnusable, match="A recovery that disagrees"):
        read_capture_sequence(capture)


def test_a_recovery_that_does_not_start_after_the_last_lost_frame_refuses(
    tmp_path: Path,
) -> None:
    """A gap either side of the streak is a frame silently left out of both."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    loss = read_tracking_loss(tmp_path)
    assert loss is not None

    with pytest.raises(SegmentPublicationRefused, match="continues at the very next"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=tmp_path,
            loss=loss,
            frames=[(index, frame_bytes(index)) for index in (9, 10, 11)],
            transform=transform(3.0),
            evidence=accepted_evidence(),
        )
    assert not list(tmp_path.glob("segment-*"))


@pytest.mark.parametrize("missing", [RECOVERY_EXCLUDED_LIMITATION, RECOVERY_RELOCALIZED_LIMITATION])
def test_a_recovery_missing_a_mandatory_limitation_refuses(tmp_path: Path, missing: str) -> None:
    """Those two sentences travel with the boundary. Without them it is refused."""
    capture = recovered_capture(tmp_path)
    kept = [item for item in RECOVERY_LIMITATIONS if item != missing]
    break_recovery(capture / "segment-0001", limitations=kept)

    with pytest.raises(BridgeRecordUnusable, match="does not state that"):
        read_capture_sequence(capture)


def test_a_recovery_that_is_not_an_object_refuses(tmp_path: Path) -> None:
    capture = recovered_capture(tmp_path)
    record = record_of(capture / "segment-0001")
    record[RECOVERY_RECORD_KEY] = "yes"
    rewrite(capture / "segment-0001", record)

    with pytest.raises(BridgeRecordUnusable, match="not an object"):
        read_capture_sequence(capture)


def test_a_recovery_binding_a_loss_that_names_frames_not_on_disk_refuses(
    tmp_path: Path,
) -> None:
    """The record is checked against the run's own frames before it can close it."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    recover(tmp_path, tmp_path, start=8)
    # One of the frames the loss named, removed after publication: the sequence
    # would otherwise silently start at a frame the record never described.
    (tmp_path / FRAME_TEMPLATE.format(index=7)).unlink()

    with pytest.raises(CaptureSequenceUnreadable):
        read_capture_sequence(tmp_path)


def test_a_gap_inside_the_recovered_run_refuses(tmp_path: Path) -> None:
    """A missing file is not a frame that was never taken.

    The gap is in the valid PREFIX, which is the one place the excluded frames
    could have hidden one: a check that looked only at the frames it hands over
    would see 0, 1, 2, 4 followed by 8 and find nothing wrong with either half.
    """
    capture = recovered_capture(tmp_path)
    (capture / FRAME_TEMPLATE.format(index=3)).unlink()

    with pytest.raises(CaptureSequenceUnreadable, match=r"missing frame\(s\) \[3\]"):
        read_capture_sequence(capture)


# --- more than one episode, and pauses around them -----------------------------


def test_two_losses_in_one_capture_are_recovered_independently(tmp_path: Path) -> None:
    """The second loss happens INSIDE the segment the first recovery published.

    Publishing both records at the capture root would overwrite the first - and
    an existing loss record is never overwritten - so the second episode would
    either be lost or fail the capture. Each run keeps its own.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    first = recover(tmp_path, tmp_path, start=8)
    write_frames(first, [11, 12])
    lose_tracking(first, last_aligned=12)
    second = recover(tmp_path, first, start=16)

    sequence = read_capture_sequence(tmp_path)

    assert [frame.index for frame in sequence.frames] == [
        0,
        1,
        2,
        3,
        4,
        8,
        9,
        10,
        11,
        12,
        16,
        17,
        18,
    ]
    assert [frame.index for frame in sequence.excluded] == [5, 6, 7, 13, 14, 15]
    assert [recovery.run for recovery in sequence.recoveries] == [None, first.name]
    assert [recovery.segment for recovery in sequence.recoveries] == [first.name, second.name]
    assert [boundary.position for boundary in sequence.boundaries()] == [5, 10]
    assert (first / TRACKING_LOSS_RECORD_NAME).exists()
    assert (tmp_path / TRACKING_LOSS_RECORD_NAME).exists()


def test_a_second_loss_left_unresolved_refuses_the_whole_capture(tmp_path: Path) -> None:
    """One recovered episode does not make the capture whole."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    first = recover(tmp_path, tmp_path, start=8)
    write_frames(first, [11, 12])
    lose_tracking(first, last_aligned=12)

    with pytest.raises(TrackingLossUnresolved, match=first.name):
        read_capture_sequence(tmp_path)


def test_ordinary_pauses_before_between_and_after_a_recovery_still_read(
    tmp_path: Path,
) -> None:
    """A pause is not a recovery and a recovery is not a pause.

    Both kinds of boundary in one capture, in the order an operator could
    actually produce them, and the reader keeps them apart: only the recovered
    seam excludes anything, and only the pauses are counted as repositions.
    """
    write_frames(tmp_path, range(5))
    paused = pause_segment(tmp_path, predecessor=4)
    write_frames(paused, [8, 9])
    lose_tracking(paused, last_aligned=9)
    recovered = recover(tmp_path, paused, start=13)
    write_frames(recovered, [16])
    after = pause_segment(tmp_path, predecessor=16)

    sequence = read_capture_sequence(tmp_path)

    assert [frame.index for frame in sequence.frames] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]
    assert [frame.index for frame in sequence.excluded] == [10, 11, 12]
    assert [bridge.segment for bridge in sequence.bridges] == [
        paused.name,
        recovered.name,
        after.name,
    ]
    assert len(sequence.recoveries) == 1
    assert sequence.recoveries[0].segment == recovered.name
    stated = "\n".join(sequence.limitations)
    assert "repositioned 2 time(s)" in stated
    assert "lost tracking 1 time(s)" in stated


# --- what the writer refuses before anything is staged -------------------------


def test_a_recovery_of_a_loss_that_never_aligned_anything_is_refused(
    tmp_path: Path,
) -> None:
    """There is no accepted model to place a segment against, so there is no
    recovery to publish."""
    write_frames(tmp_path, [0, 1, 2])
    publish_tracking_loss(tmp_path, last_aligned_frame_index=-1, failures=failures(0, 1, 2))
    loss = read_tracking_loss(tmp_path)
    assert loss is not None

    with pytest.raises(SegmentPublicationRefused, match="nothing to recover to"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=tmp_path,
            loss=loss,
            frames=[(index, frame_bytes(index)) for index in (3, 4, 5)],
            transform=transform(3.0),
            evidence=accepted_evidence(),
        )
    assert not list(tmp_path.glob("segment-*"))


def test_a_recovery_whose_excluded_frames_are_not_on_disk_is_refused(
    tmp_path: Path,
) -> None:
    """Measured off the disk, not off what the caller believes it wrote."""
    write_frames(tmp_path, range(5))
    publish_tracking_loss(tmp_path, last_aligned_frame_index=4, failures=failures(5, 6, 7))
    loss = read_tracking_loss(tmp_path)
    assert loss is not None

    with pytest.raises(SegmentPublicationRefused, match="could not be bound"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=tmp_path,
            loss=loss,
            frames=[(index, frame_bytes(index)) for index in (8, 9, 10)],
            transform=transform(3.0),
            evidence=accepted_evidence(),
        )
    assert not list(tmp_path.glob("segment-*"))


def test_a_published_recovery_names_the_loss_record_by_its_own_digest(
    tmp_path: Path,
) -> None:
    """What is written down, stated field by field rather than round-tripped."""
    capture = recovered_capture(tmp_path)

    recovery = record_of(capture / "segment-0001")[RECOVERY_RECORD_KEY]

    assert isinstance(recovery, dict)
    assert recovery["kind"] == RECOVERY_KIND
    assert recovery["schema_version"] == RECOVERY_SCHEMA_VERSION
    assert recovery["predecessor_frame_index"] == 4
    assert recovery["segment_start_frame_index"] == 8
    payload = (capture / TRACKING_LOSS_RECORD_NAME).read_bytes()
    assert recovery["loss_record"] == {
        "path": TRACKING_LOSS_RECORD_NAME,
        "byte_count": len(payload),
        "sha256": sha256_of(payload),
    }
    assert [entry["frame_index"] for entry in recovery["excluded_frames"]] == [5, 6, 7]
    assert [entry["sha256"] for entry in recovery["excluded_frames"]] == [
        sha256_of(frame_bytes(index)) for index in (5, 6, 7)
    ]


def test_a_recovery_bound_to_a_loss_that_has_since_changed_is_refused(
    tmp_path: Path,
) -> None:
    """The record on disk NOW is the authority, and a stale one is not believed.

    The publication derives the predecessor, the start index and the excluded
    frames from the ``loss`` it is handed, while the binding hashes the bytes at
    ``tracking-loss.json``. If those two are different records the segment could
    be published - and the capture could announce a recovery - carrying a
    binding the reader would then refuse. So the supplied loss is compared
    against the current record and a mismatch publishes nothing at all.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    stale = read_tracking_loss(tmp_path)
    assert stale is not None
    record = tmp_path / TRACKING_LOSS_RECORD_NAME
    mutated = json.loads(record.read_text())
    mutated["failures"][0]["message"] = "a different account of a different failure"
    record.write_text(json.dumps(mutated))

    with pytest.raises(SegmentPublicationRefused, match="no longer the loss"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=tmp_path,
            loss=stale,
            frames=[(index, frame_bytes(index)) for index in (8, 9, 10)],
            transform=transform(3.0),
            evidence=accepted_evidence(),
        )
    assert not list(tmp_path.glob("segment-*"))
    # And nothing was staged on the way to that refusal either.
    assert not list(tmp_path.glob(".*staging*"))


def test_a_recovery_of_a_loss_whose_record_was_removed_is_refused(tmp_path: Path) -> None:
    """A recovery closes a durable loss. With no record there is nothing to close."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    (tmp_path / TRACKING_LOSS_RECORD_NAME).unlink()

    with pytest.raises(SegmentPublicationRefused, match="no durable loss"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=tmp_path,
            loss=loss,
            frames=[(index, frame_bytes(index)) for index in (8, 9, 10)],
            transform=transform(3.0),
            evidence=accepted_evidence(),
        )
    assert not list(tmp_path.glob("segment-*"))


def test_the_binding_hashes_the_record_the_publication_actually_read(
    tmp_path: Path,
) -> None:
    """Re-formatted, not re-written: same loss, different bytes.

    The complement of the test above. A record that still MEANS the same thing
    is still the loss being closed, and the digest that goes into the binding
    has to be of the bytes that are there when it is published - not of the ones
    the caller happened to read earlier. If those two reads could disagree, this
    is the case where the disagreement would be silent.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    record = tmp_path / TRACKING_LOSS_RECORD_NAME
    before = record.read_bytes()
    record.write_text(json.dumps(json.loads(before.decode())))
    after = record.read_bytes()
    assert after != before

    segment = recover(tmp_path, tmp_path, start=8)

    binding = record_of(segment)[RECOVERY_RECORD_KEY]
    assert isinstance(binding, dict)
    assert binding["loss_record"]["sha256"] == sha256_of(after)
    assert binding["loss_record"]["byte_count"] == len(after)
    # And the reader, which hashes the same file for itself, agrees.
    assert len(read_capture_sequence(tmp_path).recoveries) == 1


def test_a_recovery_inside_a_segment_names_its_run_relative_location(
    tmp_path: Path,
) -> None:
    """A loss in a segment is bound by where it actually sits, not by the root."""
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    first = recover(tmp_path, tmp_path, start=8)
    write_frames(first, [11])
    lose_tracking(first, last_aligned=11)
    second = recover(tmp_path, first, start=15)

    recovery = record_of(second)[RECOVERY_RECORD_KEY]

    assert isinstance(recovery, dict)
    assert recovery["loss_record"]["path"] == f"{first.name}/{TRACKING_LOSS_RECORD_NAME}"


# --- the entry points inherit the decision, rather than repeating the check -----


def test_the_cli_reconstruct_refuses_a_capture_whose_loss_nothing_closes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One recovered episode does not excuse an unrecovered one.

    ``scanner3d reconstruct`` carries no tracking-loss check of its own; it
    catches :class:`CaptureSequenceUnreadable` around the single reader. So this
    is the assertion that the per-run decision reaches the operator, since a
    refusal the entry points do not surface is a refusal nobody sees.
    """
    from scanner3d.cli import main

    capture = tmp_path / "scan"
    capture.mkdir()
    write_frames(capture, range(5))
    lose_tracking(capture, last_aligned=4)
    first = recover(capture, capture, start=8)
    write_frames(first, [11])
    lose_tracking(first, last_aligned=11)

    code = main(["reconstruct", str(capture), "--backend", "python"])

    assert code != 0
    error = capsys.readouterr().err
    assert "tracking loss" in error
    assert first.name in error


def test_the_cli_reconstruct_accepts_a_capture_whose_losses_are_all_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the same inheritance: a recovered folder is readable.

    The frames here are 64 bytes of deterministic filler rather than depth
    data, so what the command reaches is the depth reader complaining about
    their LENGTH - which is the assertion: the capture was handed over, the
    excluded frames left out of it, and the refusal that used to stop this
    folder at the door is gone.
    """
    from scanner3d.cli import main
    from scanner3d.depth_metric import DepthFrameSizeMismatch

    capture = tmp_path / "scan"
    capture.mkdir()
    recovered_capture(capture)

    with pytest.raises(DepthFrameSizeMismatch):
        main(["reconstruct", str(capture), "--backend", "python"])

    error = capsys.readouterr().err
    assert "tracking loss" not in error
    assert TRACKING_LOSS_RECORD_NAME not in error


def test_a_recovery_of_a_loss_that_cuts_into_a_published_batch_is_refused(
    tmp_path: Path,
) -> None:
    """Every frame of a published batch was accepted by the relocalization that
    published it, so a loss cannot begin before the end of one.

    Refused by the WRITER as well as by the reader, and that is the point of the
    pre-flight: publishing a recovery the reader would then refuse would leave a
    capture holding a loss that looks closed.
    """
    write_frames(tmp_path, range(5))
    lose_tracking(tmp_path, last_aligned=4)
    segment = recover(tmp_path, tmp_path, start=8)
    # A loss inside that segment naming a last-aligned frame from the middle of
    # its own published batch, which is frames 8, 9 and 10.
    write_frames(segment, [11, 12])
    publish_tracking_loss(segment, last_aligned_frame_index=9, failures=failures(10, 11, 12))
    loss = read_tracking_loss(segment)
    assert loss is not None

    with pytest.raises(SegmentPublicationRefused, match="cannot begin before the end of one"):
        publish_recovery_segment(
            tmp_path,
            loss_dir=segment,
            loss=loss,
            frames=[(index, frame_bytes(index)) for index in (13, 14, 15)],
            transform=transform(4.0),
            evidence=accepted_evidence(),
        )
    assert sorted(path.name for path in tmp_path.glob("segment-*")) == [segment.name]
