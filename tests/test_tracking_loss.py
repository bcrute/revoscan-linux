"""The durable record of a lost scan, and the refusal that follows it.

What this is about
------------------
On 2026-08-28 a capture aligned 51 frames, failed on the 53rd, and then failed
on every frame after it for another two minutes. Nothing stopped. The model on
screen simply stopped changing, 120-odd more frames reached the disk looking
exactly like frames that had aligned, and the folder afterwards was
indistinguishable from a healthy one.

The containment for that is two files' worth of behaviour and this module tests
the durable half: three consecutive live-alignment failures publish one
``tracking-loss.json`` at the capture root, and
:func:`~scanner3d.capture_sequence.read_capture_sequence` refuses that directory
from then on. The live half - when the record is written, and what the capture
stops doing - is in ``test_gui_tracking_loss.py``.

The refusal is the point, so most of this file is about the ways a directory can
carry something loss-shaped that is not a loss record. Every one of them refuses.
None of them falls back to reading the capture as though it were fine, because
that fallback IS the original failure: a broken sequence that reads back as a
whole one and reconstructs into a confident, wrong model.

Frames here are plain deterministic bytes, for the same reason they are in
``test_capture_sequence.py``: nothing in this module knows what depth data is,
only what files, JSON and refusals are.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scanner3d.capture_sequence import (
    FRAME_TEMPLATE,
    TRACKING_LOSS_KIND,
    TRACKING_LOSS_MESSAGE_LIMIT,
    TRACKING_LOSS_NO_BRIDGE_LIMITATION,
    TRACKING_LOSS_RECORD_NAME,
    TRACKING_LOSS_SCHEMA_VERSION,
    TRACKING_LOSS_STREAK,
    TRACKING_LOSS_TEXT_LIMIT,
    TRACKING_LOSS_TRUNCATION_MARK,
    AlignmentFailure,
    CaptureSequenceUnreadable,
    TrackingLossPublicationRefused,
    TrackingLossRecordUnusable,
    TrackingLossUnresolved,
    publish_tracking_loss,
    read_capture_sequence,
    read_tracking_loss,
)

# --- building capture directories by hand --------------------------------------


def write_frames(directory: Path, count: int) -> None:
    for index in range(count):
        (directory / FRAME_TEMPLATE.format(index=index)).write_bytes(bytes([index % 251]) * 64)


def failures(*indices: int) -> list[AlignmentFailure]:
    """One failure per frame index, in the order the capture saw them."""
    return [
        AlignmentFailure(
            frame_index=index,
            failure_type="RegistrationError",
            message=f"point-to-plane singular in translation z at frame {index}",
        )
        for index in indices
    ]


def lost_capture(directory: Path, *, frames: int = 8, last_aligned: int = 2) -> Path:
    """A capture that wrote ``frames`` frames and recorded a loss after three."""
    write_frames(directory, frames)
    return publish_tracking_loss(
        directory,
        last_aligned_frame_index=last_aligned,
        failures=failures(last_aligned + 1, last_aligned + 2, last_aligned + 3),
    )


def record_of(directory: Path) -> dict[str, object]:
    payload = json.loads((directory / TRACKING_LOSS_RECORD_NAME).read_text())
    assert isinstance(payload, dict)
    return payload


def rewrite(directory: Path, record: dict[str, object]) -> None:
    (directory / TRACKING_LOSS_RECORD_NAME).write_text(json.dumps(record))


# --- the record round-trips ----------------------------------------------------


def test_a_published_loss_reads_back_with_every_field_the_refusal_needs(
    tmp_path: Path,
) -> None:
    """The record is what a reader has months later, so it carries the whole story.

    Named field by field rather than by comparing to a blob: each of these is
    something the refusal or the operator needs, and a record that silently
    stopped carrying one would still round-trip through a shape comparison.
    """
    path = lost_capture(tmp_path, last_aligned=51)

    assert path.name == TRACKING_LOSS_RECORD_NAME
    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    assert loss.threshold == TRACKING_LOSS_STREAK
    assert loss.last_aligned_frame_index == 51
    assert loss.first_failed_frame_index == 52
    assert loss.triggering_frame_index == 54
    assert [failure.frame_index for failure in loss.failures] == [52, 53, 54]
    assert [failure.failure_type for failure in loss.failures] == ["RegistrationError"] * 3
    assert "singular in translation z" in loss.failures[0].message
    assert any("no pose bridges" in item.lower() for item in loss.limitations)


def test_the_record_names_what_classified_the_loss_and_does_not_claim_the_hardware(
    tmp_path: Path,
) -> None:
    """Nothing measured this. A record read later as a hardware fault is a lie.

    The projector was lit and the camera was delivering frames the whole time;
    what failed was registration, in software, on the host. A record that left
    that unsaid would send somebody looking at the scanner.
    """
    lost_capture(tmp_path)
    loss = read_tracking_loss(tmp_path)
    assert loss is not None

    said = loss.classified_by.lower()
    assert "scanner3d" in said
    assert "alignment" in said
    assert "projector" in said and "camera" in said
    assert "no hardware or object measurement" in said


def test_a_long_failure_message_is_truncated_rather_than_written_whole(
    tmp_path: Path,
) -> None:
    """A backend exception can carry a whole array repr. The record is bounded."""
    write_frames(tmp_path, 6)
    publish_tracking_loss(
        tmp_path,
        last_aligned_frame_index=0,
        failures=[
            AlignmentFailure(frame_index=index, failure_type="ValueError", message="x" * 5000)
            for index in (1, 2, 3)
        ],
    )

    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    message = loss.failures[0].message
    assert len(message) <= TRACKING_LOSS_MESSAGE_LIMIT + len(" [truncated]")
    assert message.endswith("[truncated]")


def test_publishing_a_loss_touches_no_frame_the_capture_already_wrote(
    tmp_path: Path,
) -> None:
    """The valid prefix is evidence. Nothing here renumbers, moves or rewrites it."""
    write_frames(tmp_path, 8)
    before = {path.name: path.read_bytes() for path in tmp_path.glob("frame_*.raw")}

    publish_tracking_loss(tmp_path, last_aligned_frame_index=2, failures=failures(3, 4, 5))

    after = {path.name: path.read_bytes() for path in tmp_path.glob("frame_*.raw")}
    assert after == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        *sorted(before),
        TRACKING_LOSS_RECORD_NAME,
    ]


def test_a_capture_that_never_aligned_anything_records_minus_one(tmp_path: Path) -> None:
    """Losing tracking before the first alignment is a loss like any other.

    ``-1`` is how the record says "no frame in this capture ever aligned", and
    it has to be readable back as that rather than refused as out of range.
    """
    write_frames(tmp_path, 4)
    publish_tracking_loss(tmp_path, last_aligned_frame_index=-1, failures=failures(0, 1, 2))

    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    assert loss.last_aligned_frame_index == -1
    assert "no frames at all" in loss.refusal(tmp_path)


# --- the record is written once, completely ------------------------------------


def test_an_existing_record_is_never_overwritten(tmp_path: Path) -> None:
    """A capture loses tracking once. A second record is a second answer."""
    lost_capture(tmp_path, last_aligned=2)
    first = (tmp_path / TRACKING_LOSS_RECORD_NAME).read_bytes()

    with pytest.raises(TrackingLossPublicationRefused, match="already exists"):
        publish_tracking_loss(tmp_path, last_aligned_frame_index=40, failures=failures(41, 42, 43))

    assert (tmp_path / TRACKING_LOSS_RECORD_NAME).read_bytes() == first


def test_a_publication_that_cannot_write_refuses_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    """Complete or absent. A half-written record is a capture nothing can classify.

    A read-only capture directory is the honest way to produce the failure: the
    staging directory cannot even be made, which is the earliest point the
    publication can fall over.
    """
    capture = tmp_path / "scan"
    capture.mkdir()
    write_frames(capture, 5)
    capture.chmod(0o500)
    try:
        with pytest.raises(TrackingLossPublicationRefused):
            publish_tracking_loss(capture, last_aligned_frame_index=1, failures=failures(2, 3, 4))
    finally:
        capture.chmod(0o700)

    assert sorted(path.name for path in capture.iterdir()) == [
        FRAME_TEMPLATE.format(index=index) for index in range(5)
    ]
    assert read_tracking_loss(capture) is None


def test_a_publication_with_no_failures_refuses(tmp_path: Path) -> None:
    """A record with no evidence is one nothing can interpret."""
    write_frames(tmp_path, 4)
    with pytest.raises(TrackingLossPublicationRefused, match="none were given"):
        publish_tracking_loss(tmp_path, last_aligned_frame_index=1, failures=[])
    assert read_tracking_loss(tmp_path) is None


def test_a_publication_the_reader_would_refuse_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """The writer runs the reader's own checks first.

    A record that publishes and is then refused forever after would be worse
    than no record: the capture would be refused for a reason that has nothing
    to do with what actually happened to it.
    """
    write_frames(tmp_path, 6)
    with pytest.raises(
        TrackingLossPublicationRefused, match="not one this reader would accept back"
    ):
        publish_tracking_loss(
            # Last aligned AFTER the first failure, which cannot be true.
            tmp_path,
            last_aligned_frame_index=4,
            failures=failures(2, 3, 4),
        )
    assert read_tracking_loss(tmp_path) is None


# --- the reader refuses a lost capture -----------------------------------------


def test_a_capture_with_a_loss_record_is_refused_as_one_sequence(tmp_path: Path) -> None:
    """The whole directory, not the frames after the loss. There is no valid whole."""
    lost_capture(tmp_path, frames=12, last_aligned=2)

    with pytest.raises(TrackingLossUnresolved) as caught:
        read_capture_sequence(tmp_path)

    said = str(caught.value)
    assert "tracking loss" in said
    # The three things the operator has to be told: the prefix is real, the
    # evidence is on disk, and nothing was deleted.
    assert "frames up to and including frame 2" in said
    assert TRACKING_LOSS_RECORD_NAME in said
    assert "nothing has been deleted" in said
    # And the frames really are all still there.
    assert len(list(tmp_path.glob("frame_*.raw"))) == 12


def test_the_refusal_is_a_capture_sequence_refusal_so_every_caller_already_handles_it(
    tmp_path: Path,
) -> None:
    """The CLI and the window catch one exception class between them.

    If this were not a :class:`CaptureSequenceUnreadable` the CLI's ``reconstruct``
    would abort with a traceback instead of the message it prints for every other
    unreadable capture, and the check would have to be duplicated at both entry
    points - which is the thing the single reader exists to prevent.
    """
    lost_capture(tmp_path)
    with pytest.raises(CaptureSequenceUnreadable):
        read_capture_sequence(tmp_path)


def test_a_lost_capture_that_also_has_a_published_segment_is_still_refused(
    tmp_path: Path,
) -> None:
    """A capture can be paused, resumed, and THEN lose tracking.

    The loss is checked before the segments precisely so that a directory which
    is interesting in two ways is refused for the one that matters.
    """
    write_frames(tmp_path, 5)
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    (segment / "bridge.json").write_text("{}")
    publish_tracking_loss(tmp_path, last_aligned_frame_index=4, failures=failures(5, 6, 7))

    with pytest.raises(TrackingLossUnresolved):
        read_capture_sequence(tmp_path)


# --- captures with no loss are untouched ---------------------------------------


def test_a_capture_with_no_loss_record_reads_back_exactly_as_it_always_did(
    tmp_path: Path,
) -> None:
    """Every scan ever taken still reconstructs. The check costs one missing stat."""
    write_frames(tmp_path, 7)

    assert read_tracking_loss(tmp_path) is None
    sequence = read_capture_sequence(tmp_path)
    assert [frame.index for frame in sequence.frames] == list(range(7))
    assert sequence.bridges == ()
    assert sequence.limitations == ()


def test_a_capture_holding_an_unrelated_json_file_is_not_a_lost_capture(
    tmp_path: Path,
) -> None:
    """Only the one name at the capture root means this. Nothing else is sniffed."""
    write_frames(tmp_path, 4)
    (tmp_path / "notes.json").write_text('{"kind": "scanner3d.tracking-loss"}')

    assert read_tracking_loss(tmp_path) is None
    assert len(read_capture_sequence(tmp_path).frames) == 4


# --- everything loss-shaped that is not a loss record --------------------------


def test_an_unparseable_record_refuses_rather_than_reading_as_no_loss(
    tmp_path: Path,
) -> None:
    """The fallback this must never take is "cannot read it, so it is fine"."""
    write_frames(tmp_path, 6)
    (tmp_path / TRACKING_LOSS_RECORD_NAME).write_text("{not json")

    with pytest.raises(TrackingLossRecordUnusable, match="not readable JSON"):
        read_tracking_loss(tmp_path)
    with pytest.raises(TrackingLossRecordUnusable):
        read_capture_sequence(tmp_path)


def test_a_record_that_is_not_a_json_object_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, 6)
    (tmp_path / TRACKING_LOSS_RECORD_NAME).write_text("[1, 2, 3]")
    with pytest.raises(TrackingLossRecordUnusable, match="JSON object"):
        read_capture_sequence(tmp_path)


def test_a_record_with_the_wrong_kind_refuses(tmp_path: Path) -> None:
    """Something else's JSON under this name is not read as a loss and not ignored."""
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    record["kind"] = "scanner3d.relocalization-bridge"
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="says kind"):
        read_capture_sequence(tmp_path)


def test_a_record_from_an_unsupported_schema_version_refuses(tmp_path: Path) -> None:
    """A loss stated in a dialect this cannot read is still a loss, so still refused."""
    lost_capture(tmp_path)
    for version in (0, TRACKING_LOSS_SCHEMA_VERSION - 1, TRACKING_LOSS_SCHEMA_VERSION + 1, True):
        record = record_of(tmp_path)
        record["schema_version"] = version
        rewrite(tmp_path, record)
        with pytest.raises(TrackingLossRecordUnusable, match="schema version"):
            read_capture_sequence(tmp_path)


def test_a_record_that_is_a_symlink_refuses_and_its_target_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture directory is a place an operator can edit.

    Following a link out of it would read bytes from somewhere this was never
    pointed, so the refusal happens on the ``lstat`` and no read is attempted at
    all - asserted by watching every ``read_bytes`` across the call, because an
    exception raised for some later reason would not prove the read never
    happened.
    """
    capture = tmp_path / "scan"
    capture.mkdir()
    write_frames(capture, 4)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"kind": TRACKING_LOSS_KIND}))
    (capture / TRACKING_LOSS_RECORD_NAME).symlink_to(elsewhere)

    read: list[Path] = []
    original = Path.read_bytes

    def spy(self: Path) -> bytes:
        read.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", spy)

    with pytest.raises(TrackingLossRecordUnusable, match="symbolic link"):
        read_capture_sequence(capture)
    assert elsewhere not in read


def test_a_record_that_is_a_directory_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, 4)
    (tmp_path / TRACKING_LOSS_RECORD_NAME).mkdir()
    with pytest.raises(TrackingLossRecordUnusable, match="not a regular file"):
        read_capture_sequence(tmp_path)


def test_a_record_that_cannot_be_read_refuses_rather_than_reading_as_no_loss(
    tmp_path: Path,
) -> None:
    """Unreadable is not absent. Only ``ENOENT`` means this capture has no loss."""
    lost_capture(tmp_path)
    path = tmp_path / TRACKING_LOSS_RECORD_NAME
    path.chmod(0o000)
    if os.access(path, os.R_OK):  # pragma: no cover - running as root
        pytest.skip("this process can read a mode 000 file")
    try:
        with pytest.raises(TrackingLossRecordUnusable, match="cannot be read"):
            read_capture_sequence(tmp_path)
    finally:
        path.chmod(0o600)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", "three"),
        ("threshold", True),
        ("last_aligned_frame_index", None),
        ("first_failed_frame_index", 1.5),
        ("triggering_frame_index", "54"),
    ],
)
def test_a_record_whose_own_numbers_are_not_numbers_refuses(
    tmp_path: Path, field: str, value: object
) -> None:
    """A record that cannot be checked against itself is never half-believed."""
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    record[field] = value
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable):
        read_capture_sequence(tmp_path)


def test_a_record_whose_last_aligned_frame_is_not_before_its_first_failure_refuses(
    tmp_path: Path,
) -> None:
    """The failures follow the last accepted pose. Both cannot be the same frame."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    record["last_aligned_frame_index"] = 3
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="cannot all be true at once"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_streak_ends_before_it_begins_refuses(tmp_path: Path) -> None:
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    record["triggering_frame_index"] = 3
    record["first_failed_frame_index"] = 5
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="cannot all be true at once"):
        read_capture_sequence(tmp_path)


def test_a_record_listing_fewer_failures_than_its_threshold_refuses(tmp_path: Path) -> None:
    """The failures ARE the interpretation. A record short of one cannot be read."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    record["failures"] = listed[:-1]
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="lists 2 failure"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_failures_are_out_of_order_refuses(tmp_path: Path) -> None:
    """A streak is distinct frames in the order they arrived."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    record["failures"] = list(reversed(listed))
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable):
        read_capture_sequence(tmp_path)


def test_a_record_that_disagrees_with_its_own_failure_list_refuses(tmp_path: Path) -> None:
    """The summary and the evidence are two statements of one fact."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    record["triggering_frame_index"] = 99
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="disagrees with its own evidence"):
        read_capture_sequence(tmp_path)


def test_a_record_with_a_failure_entry_missing_its_message_refuses(tmp_path: Path) -> None:
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    assert isinstance(listed[0], dict)
    del listed[0]["message"]
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="missing a frame index"):
        read_capture_sequence(tmp_path)


def test_a_record_that_says_nothing_about_what_classified_the_loss_refuses(
    tmp_path: Path,
) -> None:
    """Left unsaid, a loss record is read later as a hardware fault it never saw."""
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    record["classified_by"] = ""
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="what classified the loss"):
        read_capture_sequence(tmp_path)


def test_a_record_with_no_limitations_refuses(tmp_path: Path) -> None:
    """The first limitation - that no pose bridges the gap - IS the refusal."""
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    record["limitations"] = []
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="states 0 limitations"):
        read_capture_sequence(tmp_path)


def test_a_record_stating_any_threshold_but_the_one_policy_refuses(
    tmp_path: Path,
) -> None:
    """Schema 1 has exactly one threshold, and it is not a field to negotiate.

    A record that could name its own threshold could name one, publish itself
    with a single failure as evidence, and be believed. The policy is what makes
    the refusal mean something, so the reader enforces the policy rather than
    whatever number the file happens to carry.
    """
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    record["threshold"] = 2
    record["failures"] = listed[:2]
    record["triggering_frame_index"] = 4
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="states a threshold of 2"):
        read_capture_sequence(tmp_path)


def test_a_record_claiming_more_failures_than_the_policy_also_refuses(tmp_path: Path) -> None:
    """Both directions. Four is not the threshold either."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    assert isinstance(listed[-1], dict)
    extra = dict(listed[-1])
    extra["frame_index"] = 6
    record["threshold"] = 4
    record["failures"] = [*listed, extra]
    record["triggering_frame_index"] = 6
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="states a threshold of 4"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("field", ["failure_type", "message"])
def test_a_record_whose_failure_evidence_is_blank_refuses(tmp_path: Path, field: str) -> None:
    """Blank evidence explains nothing, and explaining is what the record is for."""
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    assert isinstance(listed[1], dict)
    listed[1][field] = "   "
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="is blank"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("field", ["failure_type", "message"])
def test_a_record_whose_failure_evidence_is_unbounded_refuses(tmp_path: Path, field: str) -> None:
    """The writer bounds both; a file carrying more was not written by it.

    A record is read by a person as well as by this reader, and a driver
    exception carrying a whole array repr would make it neither.
    """
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    assert isinstance(listed[0], dict)
    listed[0][field] = "x" * (TRACKING_LOSS_TEXT_LIMIT + 1)
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="past the"):
        read_capture_sequence(tmp_path)


def test_a_record_missing_a_frame_from_the_middle_of_its_streak_refuses(
    tmp_path: Path,
) -> None:
    """A streak is CONSECUTIVE frames, not merely sorted distinct ones.

    Every selected frame a live capture does not accept is recorded as a
    failure - a failed alignment, a frame that deprojects to nothing, a
    deprojection that raises - so there is no way for this writer to produce a
    run with a hole in it. A record showing one is describing a capture that
    accepted a frame in the middle of its own loss, which would mean the model
    moved and the streak should have reset.
    """
    lost_capture(tmp_path, last_aligned=2)
    record = record_of(tmp_path)
    listed = record["failures"]
    assert isinstance(listed, list)
    assert isinstance(listed[1], dict)
    # 3, 4, 5 becomes 3, 6, 7: still increasing, still distinct, still three,
    # and no longer a run.
    listed[1]["frame_index"] = 6
    assert isinstance(listed[2], dict)
    listed[2]["frame_index"] = 7
    record["triggering_frame_index"] = 7
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="not one consecutive run"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_provenance_has_been_replaced_by_a_hardware_claim_refuses(
    tmp_path: Path,
) -> None:
    """The dangerous edit is not a missing sentence, it is a swapped one.

    "The scanner reported a tracking fault" is a claim about hardware that
    nothing in this system ever observed, and no reader can tell an honest
    provenance from a rewritten one by reading its prose. Schema 1 has one
    publisher, so it accepts one statement.
    """
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    record["classified_by"] = "the Revopoint MINI projector reported a tracking fault"
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="not the one schema"):
        read_capture_sequence(tmp_path)


def test_a_record_that_drops_the_no_pose_bridge_limitation_refuses(tmp_path: Path) -> None:
    """That limitation is not documentation. It is WHY the directory is refused."""
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    listed = record["limitations"]
    assert isinstance(listed, list)
    remaining = [item for item in listed if item != TRACKING_LOSS_NO_BRIDGE_LIMITATION]
    assert remaining and len(remaining) < len(listed)
    record["limitations"] = remaining
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="no pose bridges"):
        read_capture_sequence(tmp_path)


def test_a_record_with_a_blank_limitation_refuses(tmp_path: Path) -> None:
    lost_capture(tmp_path)
    record = record_of(tmp_path)
    listed = record["limitations"]
    assert isinstance(listed, list)
    record["limitations"] = [*listed, "  "]
    rewrite(tmp_path, record)

    with pytest.raises(TrackingLossRecordUnusable, match="blank or longer"):
        read_capture_sequence(tmp_path)


def test_a_publication_whose_threshold_is_not_the_policy_writes_nothing(
    tmp_path: Path,
) -> None:
    """The writer is held to the same policy, before anything reaches the disk."""
    write_frames(tmp_path, 6)
    with pytest.raises(
        TrackingLossPublicationRefused, match="not one this reader would accept back"
    ):
        publish_tracking_loss(
            tmp_path,
            last_aligned_frame_index=1,
            failures=failures(2, 3),
            threshold=2,
        )
    assert read_tracking_loss(tmp_path) is None


def test_a_publication_that_drops_the_mandatory_limitation_writes_nothing(
    tmp_path: Path,
) -> None:
    """A record refused forever after is worse than a capture that failed loudly."""
    write_frames(tmp_path, 6)
    with pytest.raises(
        TrackingLossPublicationRefused, match="not one this reader would accept back"
    ):
        publish_tracking_loss(
            tmp_path,
            last_aligned_frame_index=1,
            failures=failures(2, 3, 4),
            limitations=["the frames before the loss are still on disk"],
        )
    assert read_tracking_loss(tmp_path) is None


def test_a_publication_of_failures_that_are_not_consecutive_writes_nothing(
    tmp_path: Path,
) -> None:
    """The writer cannot produce this, and it is refused if it ever could."""
    write_frames(tmp_path, 9)
    with pytest.raises(
        TrackingLossPublicationRefused, match="not one this reader would accept back"
    ):
        publish_tracking_loss(
            tmp_path,
            last_aligned_frame_index=1,
            failures=failures(2, 3, 7),
        )
    assert read_tracking_loss(tmp_path) is None


def test_a_long_failure_type_is_truncated_the_way_a_long_message_is(tmp_path: Path) -> None:
    """Both bounded fields are bounded by the writer, not only the obvious one."""
    write_frames(tmp_path, 6)
    publish_tracking_loss(
        tmp_path,
        last_aligned_frame_index=1,
        failures=[
            AlignmentFailure(
                frame_index=index,
                failure_type="E" * (TRACKING_LOSS_MESSAGE_LIMIT + 500),
                message="singular in translation z",
            )
            for index in (2, 3, 4)
        ],
    )

    loss = read_tracking_loss(tmp_path)
    assert loss is not None
    assert all(len(f.failure_type) <= TRACKING_LOSS_TEXT_LIMIT for f in loss.failures)
    assert all(f.failure_type.endswith(TRACKING_LOSS_TRUNCATION_MARK) for f in loss.failures)


# --- the CLI inherits the refusal, rather than repeating the check -------------


def test_the_cli_reconstruct_refuses_a_lost_capture_through_the_one_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``scanner3d reconstruct`` on a lost folder exits non-zero and says why.

    It carries no tracking-loss check of its own - it catches
    :class:`CaptureSequenceUnreadable` around the single reader, exactly as it
    already did for every other way a capture folder can be unreadable. This is
    the assertion that the inheritance actually works end to end, since a
    refusal the entry points do not surface is a refusal nobody sees.
    """
    from scanner3d.cli import main

    capture = tmp_path / "scan"
    capture.mkdir()
    lost_capture(capture, frames=10, last_aligned=2)

    code = main(["reconstruct", str(capture), "--backend", "python"])

    assert code != 0
    error = capsys.readouterr().err
    assert str(capture) in error
    assert "tracking loss" in error
    assert TRACKING_LOSS_RECORD_NAME in error
