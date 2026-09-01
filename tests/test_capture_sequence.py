"""What a capture folder actually holds, proven from the outside in.

:mod:`scanner3d.capture_sequence` exists because a glob is not enough once a
capture can be paused: see that module's own docstring for the 65-degree-drift
shaped argument for why fitting ICP across a reposition is exactly the failure
a reader must never produce, even by accident, even on a folder an operator
edited by hand. That gives this file two very different jobs.

The first is the honest path: a legacy capture (no ``segment-`` directory)
must read back as precisely ``sorted(glob("frame_*.raw"))`` always has, and a
capture with one or more published segments must read back with every frame
in global-index order, the right ``segment`` on each one, and a
:class:`~scanner3d.capture_sequence.BridgeConstraint` per boundary whose
transform, evidence and digests are exactly what was published - not merely
"close enough for a reconstruction to look right", because the whole point of
a verified bridge is that nothing downstream re-derives it.

The second job is larger: proving every way a capture directory can lie about
itself is refused rather than guessed at. The module's docstring names each
one - a name that is not the canonical one, a symlink, a duplicated index, a
predecessor that does not exist, unparseable or wrong-shaped JSON, a schema
version from the future - and this file has one test per claim, checking the
right exception subclass rather than just "raises something". A few of those
tests go further and prove a negative: that a maliciously named ``file`` entry
never causes a file outside the segment to be opened, and that a frame or
segment which is a symlink never causes the link's target to be read. Both
are checked by spying on :meth:`pathlib.Path.read_bytes` across the call and
asserting the forbidden path never appears in the calls made, not just by
checking that an exception was raised - an exception could be raised for the
wrong reason and still leave the bad read to have happened first.

Everything here is built from plain deterministic bytes
(``bytes([index % 251]) * 64``) rather than real depth data: this module knows
nothing about depth, only about files, digests and JSON, and a synthetic frame
that never varies makes every digest in every test reproducible by inspection.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from scanner3d import capture_sequence as capture_sequence_module
from scanner3d.capture_sequence import (
    BRIDGE_RECORD_NAME,
    FRAME_TEMPLATE,
    RECORD_KIND,
    SCHEMA_VERSION,
    SEGMENT_GLOB,
    STAGING_PREFIX,
    BridgeRecordUnusable,
    CaptureSequenceUnreadable,
    SegmentPublicationRefused,
    next_frame_index,
    publish_segment,
    read_capture_sequence,
)
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.pointcloud import sha256_of
from scanner3d.relocalize import MIN_TENTATIVE_FRAMES

# --- building capture directories by hand --------------------------------------


def frame_bytes(index: int) -> bytes:
    """A frame's payload: deterministic, distinct per index, never depth data."""
    return bytes([index % 251]) * 64


def write_frame(directory: Path, index: int) -> Path:
    path = directory / FRAME_TEMPLATE.format(index=index)
    path.write_bytes(frame_bytes(index))
    return path


def write_frames(directory: Path, indices: range | list[int]) -> None:
    for index in indices:
        write_frame(directory, index)


#: A non-trivial transform for round-trip tests: neither identity nor axis
#: aligned, so a bug that dropped rotation or swapped translation components
#: would not accidentally still look right.
def transform(seed: float) -> Rigid:
    return rotation_about_axis(
        (0.4, 0.6, 0.6931), math.radians(10.0 + seed), through=(1.0, -2.0, 5.0)
    )


def frame_entries(indices: list[int]) -> list[dict[str, object]]:
    return [
        {
            "file": FRAME_TEMPLATE.format(index=index),
            "frame_index": index,
            "byte_count": len(frame_bytes(index)),
            "sha256": sha256_of(frame_bytes(index)),
        }
        for index in indices
    ]


def local_edge(frame: int, **extra: object) -> dict[str, object]:
    """One accepted adjacent link, shaped exactly as
    :meth:`scanner3d.relocalize.LocalEdge.as_dict` writes one. ``extra``
    merges in on top so a test can break one field of a KNOWN-GOOD edge.
    """
    edge: dict[str, object] = {
        "frame": frame,
        "rmse": 0.02,
        "rmse_unit": "input coordinate units, whose physical meaning is not established",
        "converged": True,
        "iterations": 11,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
        "accepted": True,
        "refusals": [],
    }
    edge.update(extra)
    return edge


def local_chain(batch_count: int, **extra: object) -> dict[str, object]:
    """A valid local chain for a batch of ``batch_count`` frames: one anchor
    and a measured, accepted link into every frame after it - the shape
    :meth:`scanner3d.gui.TentativeSegment.chain_record` produces.
    """
    chain: dict[str, object] = {
        "gates": {"local_max_residual": 1.5, "local_min_matched_fraction": 0.35},
        "edges": [local_edge(frame) for frame in range(1, max(batch_count, 1))],
        "anchor_frames": 1,
    }
    chain.update(extra)
    return chain


def accepted_evidence(
    batch_count: int = MIN_TENTATIVE_FRAMES, **extra: object
) -> dict[str, object]:
    """A structurally valid accepted-evidence object, shaped as the reader
    requires: an object, ``accepted: true``, a ``thresholds`` object, a
    non-empty ``candidates`` list of objects, and a ``local_chain`` describing
    the batch it belongs to - the same shape :mod:`scanner3d.relocalize` and
    the capture window actually write (see their ``as_dict``/``thresholds``/
    ``chain_record``). ``extra`` merges in on top, for a test that wants to
    check a field is carried through the reader verbatim.
    """
    evidence: dict[str, object] = {
        "accepted": True,
        "thresholds": {"min_tentative_frames": MIN_TENTATIVE_FRAMES},
        "candidates": [{"residual": 0.01}],
        "local_chain": local_chain(batch_count),
    }
    evidence.update(extra)
    return evidence


def valid_record(
    *,
    predecessor: int,
    start: int,
    indices: list[int],
    rigid: Rigid | None = None,
    evidence: dict[str, object] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    """A record shaped exactly as :func:`publish_segment` writes one.

    Every malformed-record test below starts from this and breaks one field,
    so each test's diff from a KNOWN-GOOD record is exactly the thing it means
    to test.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "predecessor_frame_index": predecessor,
        "segment_start_frame_index": start,
        "frames": frame_entries(indices),
        "frame_order": "the batch is one consecutive run of global indices, in capture order",
        "bridge_transform": (rigid or transform(0.0)).as_dict(),
        "bridge_transform_convention": "irrelevant to the reader; carried verbatim",
        "evidence": evidence if evidence is not None else accepted_evidence(len(indices)),
        "limitations": limitations if limitations is not None else [],
    }


def frames_of(record: dict[str, object]) -> list[dict[str, object]]:
    """The ``"frames"`` list of a hand-built record, typed for mutation.

    ``valid_record`` returns ``dict[str, object]`` (it must, to hold a mix of
    strings, ints and nested structures), so every test that pokes a single
    frame entry to break it goes through this rather than indexing ``object``
    directly.
    """
    frames = record["frames"]
    assert isinstance(frames, list)
    return frames


def write_segment(directory: Path, record: dict[str, object], indices: list[int]) -> Path:
    """A segment directory holding exactly the frames named, plus its record.

    Deliberately writes only the frames the CALLER lists, at the names the
    caller gives them - so a test can omit a frame, or misname one, without
    this helper silently repairing it.
    """
    directory.mkdir()
    for index in indices:
        (directory / FRAME_TEMPLATE.format(index=index)).write_bytes(frame_bytes(index))
    (directory / BRIDGE_RECORD_NAME).write_text(json.dumps(record, indent=2))
    return directory


def read_bytes_spy(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path :meth:`pathlib.Path.read_bytes` is called on, for the
    duration of the test - the instrument the symlink and path-traversal
    refusal tests use to prove a forbidden file was never opened, rather than
    merely that *an* exception happened to come out.
    """
    opened: list[Path] = []
    original = Path.read_bytes

    def spy(self: Path) -> bytes:
        opened.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    return opened


# --- legacy captures: the glob this always was ----------------------------------


def test_a_legacy_capture_reads_back_as_the_sorted_glob_it_always_was(tmp_path: Path) -> None:
    """No ``segment-`` directory means no bridges, no limitations, and the
    frame order a plain ``sorted(glob("frame_*.raw"))`` would have produced -
    the property every pre-pause capture, and every capture taken before this
    module existed, depends on.
    """
    write_frames(tmp_path, [0, 3, 1, 2])
    expected = sorted(tmp_path.glob("frame_*.raw"))

    sequence = read_capture_sequence(tmp_path)

    assert sequence.paths == tuple(expected)
    assert [frame.index for frame in sequence.frames] == [0, 1, 2, 3]
    assert all(frame.segment is None for frame in sequence.frames)
    assert sequence.bridges == ()
    assert sequence.limitations == ()
    assert sequence.segmented is False


def test_strided_on_a_legacy_capture_is_the_plain_slice_it_always_was(tmp_path: Path) -> None:
    """``strided(n)`` on an unsegmented capture must equal ``paths[::n]``
    exactly - the per-run striding logic exists for segmented captures, and
    must not change behaviour when there is only one run.
    """
    write_frames(tmp_path, range(11))
    sequence = read_capture_sequence(tmp_path)

    for stride in (1, 2, 3, 4):
        assert sequence.strided(stride).paths == sequence.paths[::stride]


def test_frame_00007_beside_frame_0007_does_not_round_trip(tmp_path: Path) -> None:
    """A zero-padded duplicate is not a name :data:`FRAME_TEMPLATE` would ever
    produce for the index it parses to, so it must not be silently placed as
    if it were an ordinary frame with a longer name.
    """
    write_frame(tmp_path, 7)
    (tmp_path / "frame_00007.raw").write_bytes(frame_bytes(7))

    with pytest.raises(CaptureSequenceUnreadable, match="does not round-trip"):
        read_capture_sequence(tmp_path)


def test_a_root_frame_file_that_is_a_symlink_is_refused_and_its_target_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even outside any segment, a frame is read as bytes on disk, not
    followed through a link - proven by showing the link's target is never
    opened, not merely that reading fails somehow.
    """
    elsewhere = tmp_path / "elsewhere.raw"
    elsewhere.write_bytes(frame_bytes(999))
    write_frame(tmp_path, 0)
    (tmp_path / "frame_0001.raw").symlink_to(elsewhere)
    opened = read_bytes_spy(monkeypatch)

    with pytest.raises(BridgeRecordUnusable, match="symbolic link"):
        read_capture_sequence(tmp_path)

    assert elsewhere not in opened


def test_a_legacy_capture_with_gaps_in_its_indices_reads_back_unchanged(tmp_path: Path) -> None:
    """Root-only legacy gaps are preserved exactly, not treated as evidence of
    anything: a capture missing frames 1, 3 and 4 (an operator deleted a bad
    frame, or a very old capture never claimed to be gapless) has no
    ``segment-`` directory and so is read exactly as the plain sorted glob it
    always was - no bridges, and no limitation added on its behalf. This is
    the one side of the module's contract that must NOT get stricter: only
    the presence of bridge metadata triggers the new gates.
    """
    write_frames(tmp_path, [0, 2, 5, 9])
    expected = sorted(tmp_path.glob("frame_*.raw"))

    sequence = read_capture_sequence(tmp_path)

    assert sequence.paths == tuple(expected)
    assert [frame.index for frame in sequence.frames] == [0, 2, 5, 9]
    assert sequence.bridges == ()
    assert sequence.limitations == ()
    assert sequence.segmented is False


# --- publish_segment -> read_capture_sequence round trip -------------------------


def test_publish_then_read_round_trips_order_indices_transform_and_digests(
    tmp_path: Path,
) -> None:
    """The central promise of the pair: everything :func:`publish_segment`
    wrote is exactly what :func:`read_capture_sequence` hands back - frame
    order across root and segment, global indices, the segment name, the
    transform's rotation and translation numerically, and the record's own
    digests and byte counts.
    """
    write_frames(tmp_path, range(3))
    rigid = transform(5.0)
    evidence = accepted_evidence(rmse=0.042, correspondences=731)
    publication = publish_segment(
        tmp_path,
        frames=[(3, frame_bytes(3)), (4, frame_bytes(4)), (5, frame_bytes(5))],
        predecessor_index=2,
        transform=rigid,
        evidence=evidence,
        limitations=["a synthetic relocalization, never actually measured"],
    )

    assert publication.directory == tmp_path / "segment-0001"
    assert publication.predecessor_index == 2
    assert publication.start_index == 3

    sequence = read_capture_sequence(tmp_path)

    assert [frame.index for frame in sequence.frames] == [0, 1, 2, 3, 4, 5]
    assert [frame.segment for frame in sequence.frames] == [
        None,
        None,
        None,
        "segment-0001",
        "segment-0001",
        "segment-0001",
    ]
    assert sequence.segmented is True
    assert len(sequence.bridges) == 1
    bridge = sequence.bridges[0]
    assert bridge.segment == "segment-0001"
    assert bridge.predecessor_index == 2
    assert bridge.start_index == 3
    assert bridge.transform.rotation == rigid.rotation
    assert bridge.transform.translation == rigid.translation
    assert bridge.evidence == evidence
    assert bridge.limitations == ("a synthetic relocalization, never actually measured",)

    record = json.loads((publication.directory / BRIDGE_RECORD_NAME).read_text())
    for entry in record["frames"]:
        index = entry["frame_index"]
        payload = (publication.directory / entry["file"]).read_bytes()
        assert len(payload) == entry["byte_count"] == len(frame_bytes(index))
        assert entry["sha256"] == sha256_of(payload) == sha256_of(frame_bytes(index))

    boundaries = sequence.boundaries()
    assert len(boundaries) == 1
    assert boundaries[0].position == 3
    assert boundaries[0].source == "segment-0001"
    assert boundaries[0].transform.rotation == rigid.rotation


def test_frames_written_into_a_segment_after_publication_are_picked_up_in_order(
    tmp_path: Path,
) -> None:
    """A capture that keeps sweeping after a pause is accepted writes ordinary
    frames straight into the segment directory, exactly as it always wrote
    frames at the root. They must be read back in index order after the
    published batch and must not touch the bridge - the bridge describes the
    evidence the boundary was accepted from, not the frames taken since.
    """
    write_frames(tmp_path, range(2))
    publication = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(1.0),
        evidence=accepted_evidence(rmse=0.01),
    )
    # Written out of order, the way a live sweep would arrive; the reader
    # sorts by name/index regardless.
    write_frame(publication.directory, 6)
    write_frame(publication.directory, 5)

    sequence = read_capture_sequence(tmp_path)

    assert [frame.index for frame in sequence.frames] == [0, 1, 2, 3, 4, 5, 6]
    assert [frame.segment for frame in sequence.frames[2:]] == ["segment-0001"] * 5
    assert len(sequence.bridges) == 1
    assert sequence.bridges[0].start_index == 2
    assert sequence.bridges[0].predecessor_index == 1


# --- refusals: every way a segment can lie about itself ---------------------------


def test_a_missing_bridge_record_refuses(tmp_path: Path) -> None:
    """No ``bridge.json`` at all - the plainest way a segment can be unusable."""
    write_frames(tmp_path, range(2))
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    write_frame(segment, 2)

    with pytest.raises(BridgeRecordUnusable, match="cannot be examined"):
        read_capture_sequence(tmp_path)


def test_an_unparseable_bridge_record_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    write_frame(segment, 2)
    (segment / BRIDGE_RECORD_NAME).write_text("{not json")

    with pytest.raises(BridgeRecordUnusable, match="not readable JSON"):
        read_capture_sequence(tmp_path)


def test_a_record_with_the_wrong_kind_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    record["kind"] = "something-else"
    write_segment(tmp_path / "segment-0001", record, [2])

    with pytest.raises(BridgeRecordUnusable, match="kind"):
        read_capture_sequence(tmp_path)


def test_a_record_from_a_future_schema_version_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    record["schema_version"] = SCHEMA_VERSION + 1
    write_segment(tmp_path / "segment-0001", record, [2])

    with pytest.raises(BridgeRecordUnusable, match="schema version"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("version", [0, -1, SCHEMA_VERSION + 1])
def test_every_schema_version_other_than_the_exact_supported_one_refuses(
    tmp_path: Path, version: int
) -> None:
    """This reader supports EXACTLY ``SCHEMA_VERSION`` - not "up to it" - so
    zero and a negative number (which a naive ``<= SCHEMA_VERSION`` check
    would let through) are refused exactly as a version from the future is.
    """
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    record["schema_version"] = version
    write_segment(tmp_path / "segment-0001", record, [2])

    with pytest.raises(BridgeRecordUnusable, match="schema version"):
        read_capture_sequence(tmp_path)


def test_a_frame_the_record_lists_but_is_missing_from_disk_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    # write_segment is not used here: the record lists frame 2 but no frame
    # file is actually written for it.
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    (segment / BRIDGE_RECORD_NAME).write_text(json.dumps(record))

    with pytest.raises(BridgeRecordUnusable, match="cannot be examined"):
        read_capture_sequence(tmp_path)


def test_a_frame_whose_bytes_do_not_match_its_digest_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    write_segment(tmp_path / "segment-0001", record, [2])
    # Correct byte count, wrong digest: the byte-count check passes and the
    # digest check is what must catch this.
    (tmp_path / "segment-0001" / "frame_0002.raw").write_bytes(frame_bytes(99))

    with pytest.raises(BridgeRecordUnusable, match="does not hash to the digest"):
        read_capture_sequence(tmp_path)


def test_a_frame_whose_byte_count_disagrees_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2])
    entry = frames_of(record)[0]
    entry["byte_count"] = len(frame_bytes(2)) + 1
    write_segment(tmp_path / "segment-0001", record, [2])

    with pytest.raises(BridgeRecordUnusable, match="bytes and its boundary record claims"):
        read_capture_sequence(tmp_path)


def test_a_duplicate_global_index_between_root_and_segment_refuses(tmp_path: Path) -> None:
    """Frame 1 exists at the root and the segment's record also claims frame
    index 1 - global indices must be unique across the whole capture.
    """
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[1, 2])
    write_segment(tmp_path / "segment-0001", record, [1, 2])

    with pytest.raises(BridgeRecordUnusable, match="frame index 1 appears twice"):
        read_capture_sequence(tmp_path)


def test_a_duplicate_global_index_between_two_segments_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(2))
    write_segment(
        tmp_path / "segment-0001",
        valid_record(predecessor=1, start=2, indices=[2, 3, 4]),
        [2, 3, 4],
    )
    # segment-0002 reuses frame index 4, which segment-0001 already claimed.
    write_segment(
        tmp_path / "segment-0002",
        valid_record(predecessor=4, start=4, indices=[4]),
        [4],
    )

    with pytest.raises(BridgeRecordUnusable, match="frame index 4 appears twice"):
        read_capture_sequence(tmp_path)


def test_a_predecessor_that_is_not_the_frame_immediately_before_the_segment_refuses(
    tmp_path: Path,
) -> None:
    """The root ends at frame 1; a record claiming a predecessor of frame 0
    names a frame that is not where this capture actually paused. Sized at
    ``MIN_TENTATIVE_FRAMES`` frames so this is provably about the predecessor
    claim and not about the batch being too small.
    """
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=0, start=2, indices=[2, 3, 4])
    write_segment(tmp_path / "segment-0001", record, [2, 3, 4])

    with pytest.raises(BridgeRecordUnusable, match="bridges from frame 0"):
        read_capture_sequence(tmp_path)


def test_a_start_index_that_disagrees_with_the_first_published_frame_refuses(
    tmp_path: Path,
) -> None:
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=99, indices=[2, 3, 4])
    write_segment(tmp_path / "segment-0001", record, [2, 3, 4])

    with pytest.raises(BridgeRecordUnusable, match="claims to start at frame 99"):
        read_capture_sequence(tmp_path)


def test_a_start_index_that_skips_past_the_predecessor_refuses(tmp_path: Path) -> None:
    """The root ends at frame 1, so a boundary starting at frame 3 leaves a
    gap at frame 2 - the batch's first frame must be the very next global
    index, not merely somewhere after the predecessor.
    """
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=3, indices=[3, 4, 5])
    write_segment(tmp_path / "segment-0001", record, [3, 4, 5])

    with pytest.raises(BridgeRecordUnusable, match="not the very next global index"):
        read_capture_sequence(tmp_path)


def test_a_non_consecutive_published_batch_refuses(tmp_path: Path) -> None:
    """The batch a boundary publishes is the frames buffered while
    relocalizing - one consecutive run. Frames 2, 3, 5 skip 4.
    """
    write_frames(tmp_path, range(2))
    record = valid_record(predecessor=1, start=2, indices=[2, 3, 5])
    write_segment(tmp_path / "segment-0001", record, [2, 3, 5])

    with pytest.raises(BridgeRecordUnusable, match="not one consecutive run"):
        read_capture_sequence(tmp_path)


def test_a_frame_entry_naming_a_path_outside_the_segment_refuses_and_opens_nothing_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record entry naming ``../frame_0000.raw`` for its frame cannot match
    the canonical name for any index, so it is refused before anything is
    opened by that name - proven by showing the traversal target, which does
    hold a real, differently-valued frame at that path, is never read.
    """
    write_frames(tmp_path, range(1))  # frame_0000.raw at the root, a real file
    record = valid_record(predecessor=0, start=1, indices=[1])
    frames_of(record)[0]["file"] = "../frame_0000.raw"
    write_segment(tmp_path / "segment-0001", record, [1])
    opened = read_bytes_spy(monkeypatch)

    with pytest.raises(BridgeRecordUnusable, match="only name that index has"):
        read_capture_sequence(tmp_path)

    assert (tmp_path / "frame_0000.raw") not in opened


def test_a_frame_entry_naming_a_non_canonical_name_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1])
    frames_of(record)[0]["file"] = "frame_0001_final.raw"
    write_segment(tmp_path / "segment-0001", record, [1])

    with pytest.raises(BridgeRecordUnusable, match="only name that index has"):
        read_capture_sequence(tmp_path)


def test_a_bridge_transform_whose_rotation_is_not_a_rotation_matrix_refuses(
    tmp_path: Path,
) -> None:
    """A record cannot be built through :meth:`Rigid.of` with an invalid
    rotation - construction itself validates - so this is exactly the case a
    hand-edited or corrupted record on disk would produce, and it must be
    caught on read, not trusted because the JSON otherwise parses.
    """
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    record["bridge_transform"] = {
        "rotation": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation": [0.0, 0.0, 0.0],
    }
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="not a rigid transform"):
        read_capture_sequence(tmp_path)


# --- refusals: the evidence a boundary is accepted from must actually look like it ---


def test_a_record_with_no_evidence_at_all_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    del record["evidence"]
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="no evidence object"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_evidence_is_not_an_object_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    record["evidence"] = "accepted, trust me"
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="no evidence object"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_evidence_says_accepted_false_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(
        predecessor=0, start=1, indices=[1, 2, 3], evidence=accepted_evidence(accepted=False)
    )
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="does not say accepted: true"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_evidence_has_no_thresholds_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    evidence = accepted_evidence()
    del evidence["thresholds"]
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3], evidence=evidence)
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="names no thresholds object"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("candidates", [[], None])
def test_a_record_whose_evidence_has_no_candidates_refuses(
    tmp_path: Path, candidates: list[object] | None
) -> None:
    """An empty ``candidates`` list and an absent one are both refused - the
    evidence must actually name at least one candidate the boundary was
    chosen among, not merely have room for the key.
    """
    write_frames(tmp_path, range(1))
    evidence = accepted_evidence()
    if candidates is None:
        del evidence["candidates"]
    else:
        evidence["candidates"] = candidates
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3], evidence=evidence)
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="lists no candidates"):
        read_capture_sequence(tmp_path)


# --- refusals: limitations must be a plain list of strings, present or not at all ---


def test_a_record_with_no_limitations_key_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    del record["limitations"]
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="no limitations list"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_limitations_is_not_a_list_refuses(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    record["limitations"] = "nothing to see here"
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="no limitations list"):
        read_capture_sequence(tmp_path)


def test_a_record_whose_limitations_holds_a_non_string_refuses(tmp_path: Path) -> None:
    """A non-string element is refused outright, not silently filtered out -
    the old behaviour dropped anything that was not a string and kept going,
    which is exactly the kind of quiet loss this reader must not repeat.
    """
    write_frames(tmp_path, range(1))
    record = valid_record(predecessor=0, start=1, indices=[1, 2, 3])
    record["limitations"] = ["a real limitation", 42]
    write_segment(tmp_path / "segment-0001", record, [1, 2, 3])

    with pytest.raises(BridgeRecordUnusable, match="something other than a plain string"):
        read_capture_sequence(tmp_path)


# --- refusals: a batch below MIN_TENTATIVE_FRAMES was never a genuine acceptance ---


@pytest.mark.parametrize("count", [1, 2])
def test_a_batch_below_min_tentative_frames_refuses(tmp_path: Path, count: int) -> None:
    write_frames(tmp_path, range(1))
    indices = list(range(1, 1 + count))
    record = valid_record(predecessor=0, start=1, indices=indices)
    write_segment(tmp_path / "segment-0001", record, indices)

    with pytest.raises(BridgeRecordUnusable, match=f"fewer than {MIN_TENTATIVE_FRAMES}"):
        read_capture_sequence(tmp_path)


def test_a_batch_of_exactly_min_tentative_frames_is_accepted(tmp_path: Path) -> None:
    write_frames(tmp_path, range(1))
    indices = list(range(1, 1 + MIN_TENTATIVE_FRAMES))
    record = valid_record(predecessor=0, start=1, indices=indices)
    write_segment(tmp_path / "segment-0001", record, indices)

    sequence = read_capture_sequence(tmp_path)

    assert [frame.index for frame in sequence.frames] == [0, *indices]
    assert len(sequence.bridges) == 1


def test_a_segment_directory_that_is_a_symlink_refuses(tmp_path: Path) -> None:
    """Even a symlink pointing at an otherwise perfectly valid segment must be
    refused - a capture directory's own segment names are never followed
    through a link.
    """
    real = tmp_path / "real-segment"
    write_segment(real, valid_record(predecessor=-1, start=0, indices=[0]), [0])
    (tmp_path / "segment-0001").symlink_to(real, target_is_directory=True)

    with pytest.raises(BridgeRecordUnusable, match="symbolic link"):
        read_capture_sequence(tmp_path)


def test_a_frame_inside_a_segment_that_is_a_symlink_refuses_and_its_target_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_frames(tmp_path, range(1))
    elsewhere = tmp_path / "elsewhere.raw"
    elsewhere.write_bytes(frame_bytes(999))
    segment = tmp_path / "segment-0001"
    segment.mkdir()
    (segment / BRIDGE_RECORD_NAME).write_text(
        json.dumps(valid_record(predecessor=0, start=1, indices=[1]))
    )
    (segment / "frame_0001.raw").symlink_to(elsewhere)
    opened = read_bytes_spy(monkeypatch)

    with pytest.raises(BridgeRecordUnusable, match="symbolic link"):
        read_capture_sequence(tmp_path)

    assert elsewhere not in opened


# --- refusals: every segment-shaped entry is examined, never merely skipped ---


def test_a_plain_file_named_like_a_segment_refuses_rather_than_reading_as_legacy(
    tmp_path: Path,
) -> None:
    """A file named ``segment-0001`` is not a directory this reader will ever
    open, but it is also not nothing: the capture must not fall back to being
    read as though it held no segment at all, which is the exact failure mode
    a plain ``if _SEGMENT_NAME.match(name)`` filter used to produce.
    """
    write_frames(tmp_path, range(4))
    (tmp_path / "segment-0001").write_bytes(b"not a directory")

    with pytest.raises(BridgeRecordUnusable):
        read_capture_sequence(tmp_path)


def test_a_symlink_named_like_a_segment_to_a_real_segment_refuses(tmp_path: Path) -> None:
    """Even when the symlink's target is itself a perfectly valid, canonically
    named segment, the link standing in for the canonical name is refused -
    duplicate coverage of :func:`test_a_segment_directory_that_is_a_symlink_refuses`
    from the discovery side rather than the per-segment check, proving the
    refusal happens before a non-canonical/linked entry could ever be treated
    as absent.
    """
    write_frames(tmp_path, range(1))
    real = tmp_path / "real-segment"
    indices = list(range(1, 1 + MIN_TENTATIVE_FRAMES))
    write_segment(real, valid_record(predecessor=0, start=1, indices=indices), indices)
    (tmp_path / "segment-0001").symlink_to(real, target_is_directory=True)

    with pytest.raises(BridgeRecordUnusable, match="symbolic link"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("name", ["segment-1", "segment-abc", "segment-00001x"])
def test_a_non_canonically_named_segment_directory_refuses(tmp_path: Path, name: str) -> None:
    """A directory that ``SEGMENT_GLOB`` matches but whose name is not exactly
    ``segment-NNNN`` (a short index, a non-numeric one, or trailing junk after
    the digits) is refused, not skipped - and must not cause this capture to
    read back as though it were an unsegmented, legacy one.
    """
    write_frames(tmp_path, range(4))
    indices = list(range(4, 4 + MIN_TENTATIVE_FRAMES))
    write_segment(tmp_path / name, valid_record(predecessor=3, start=4, indices=indices), indices)

    with pytest.raises(CaptureSequenceUnreadable, match="not in the exact form"):
        read_capture_sequence(tmp_path)


def test_a_malformed_segment_refuses_outright_rather_than_falling_back_to_the_root_glob(
    tmp_path: Path,
) -> None:
    """The property the whole module exists to guarantee: a directory whose
    boundary evidence is broken must not degrade into "the frames it can make
    sense of", which here would be a healthy-looking, wrong,
    unsegmented-looking read of just the root frames. It must raise, in full,
    and the caller must be left with no :class:`CaptureSequence` at all - not
    a partial one, not the pre-pause frames alone.
    """
    write_frames(tmp_path, range(4))
    # Confirms the root alone WOULD read fine, so the refusal below is
    # provably about the segment and not some other mistake in this test.
    legacy = read_capture_sequence(tmp_path)
    assert [frame.index for frame in legacy.frames] == [0, 1, 2, 3]

    (tmp_path / "segment-0001").mkdir()
    (tmp_path / "segment-0001" / BRIDGE_RECORD_NAME).write_text("{ this is not json")
    write_frame(tmp_path / "segment-0001", 4)

    with pytest.raises(BridgeRecordUnusable):
        result = read_capture_sequence(tmp_path)
        pytest.fail(f"a CaptureSequence was returned despite the broken segment: {result!r}")


# --- publish_segment: atomicity and non-collision ---------------------------------


def test_publication_failing_on_the_last_frame_leaves_no_segment_and_no_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure that strikes after every earlier frame is already written to
    staging - the worst case, since the most work has been done - must still
    leave the capture directory exactly as it was: no ``segment-*``, no
    ``.segment-staging-*``, and the legacy read unaffected.
    """
    write_frames(tmp_path, range(2))
    before = sorted(entry.name for entry in tmp_path.iterdir())
    last_frame_name = FRAME_TEMPLATE.format(index=4)
    original_write = capture_sequence_module._write_exclusive

    def flaky(path: Path, payload: bytes) -> None:
        if path.name == last_frame_name:
            raise OSError("synthetic disk failure on the last frame")
        original_write(path, payload)

    monkeypatch.setattr(capture_sequence_module, "_write_exclusive", flaky)

    with pytest.raises(OSError, match="synthetic disk failure"):
        publish_segment(
            tmp_path,
            frames=[(2, frame_bytes(2)), (3, frame_bytes(3)), (4, frame_bytes(4))],
            predecessor_index=1,
            transform=transform(2.0),
            evidence=accepted_evidence(),
        )

    after = sorted(entry.name for entry in tmp_path.iterdir())
    assert after == before
    assert not list(tmp_path.glob(SEGMENT_GLOB))
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))

    sequence = read_capture_sequence(tmp_path)
    assert [frame.index for frame in sequence.frames] == [0, 1]
    assert sequence.bridges == ()


def test_publication_never_writes_into_an_existing_segment_directory(tmp_path: Path) -> None:
    """A taken segment name is stepped over, never adopted - the same
    discipline :mod:`scanner3d.reconstruction_export` uses for its own output
    directories, for the same reason: a name that already holds a result must
    never be the destination of a different one.
    """
    write_frames(tmp_path, range(2))
    existing = tmp_path / "segment-0001"
    write_segment(existing, valid_record(predecessor=-1, start=0, indices=[0]), [0])
    marker_before = (existing / "frame_0000.raw").read_bytes()
    record_before = (existing / BRIDGE_RECORD_NAME).read_text()

    publication = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(3.0),
        evidence=accepted_evidence(),
    )

    assert publication.directory == tmp_path / "segment-0002"
    assert (existing / "frame_0000.raw").read_bytes() == marker_before
    assert (existing / BRIDGE_RECORD_NAME).read_text() == record_before


def test_no_segment_is_visible_while_staging_is_fully_populated_but_unpublished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instant staging holds every frame and the verified record - the
    moment right before the one atomic rename that publishes it - a reader
    must see NO segment at all, and a capture with root frames must still
    read back as the plain root sequence. This is what makes the eventual
    ``segment-NNNN`` name mean something: it never exists half-made.
    """
    write_frames(tmp_path, range(2))
    original_rename = capture_sequence_module._no_replace_rename
    observed: dict[str, object] = {}

    def observing_rename(src: Path, dst: Path) -> None:
        observed["segment_entries"] = list(tmp_path.glob(SEGMENT_GLOB))
        observed["staging_entries"] = list(tmp_path.glob(f"{STAGING_PREFIX}*"))
        sequence = read_capture_sequence(tmp_path)
        observed["frames_seen"] = [frame.index for frame in sequence.frames]
        observed["bridges_seen"] = sequence.bridges
        original_rename(src, dst)

    monkeypatch.setattr(capture_sequence_module, "_no_replace_rename", observing_rename)

    publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(9.0),
        evidence=accepted_evidence(),
    )

    # The staging directory genuinely existed, fully populated, at this
    # instant - and SEGMENT_GLOB, which a reader uses to find segments, saw
    # nothing there at all.
    assert len(observed["staging_entries"]) == 1  # type: ignore[arg-type]
    assert observed["segment_entries"] == []
    assert observed["frames_seen"] == [0, 1]
    assert observed["bridges_seen"] == ()


def _collide_with_empty_directory(path: Path, outside: Path) -> None:
    path.mkdir()


def _collide_with_nonempty_directory(path: Path, outside: Path) -> None:
    path.mkdir()
    (path / "marker.txt").write_bytes(b"do not touch")


def _collide_with_plain_file(path: Path, outside: Path) -> None:
    path.write_bytes(b"do not touch")


def _collide_with_symlink_to_outside_directory(path: Path, outside: Path) -> None:
    path.symlink_to(outside, target_is_directory=True)


@pytest.mark.parametrize(
    "make_collision",
    [
        pytest.param(_collide_with_empty_directory, id="empty-directory"),
        pytest.param(_collide_with_nonempty_directory, id="nonempty-directory"),
        pytest.param(_collide_with_plain_file, id="plain-file"),
        pytest.param(_collide_with_symlink_to_outside_directory, id="symlink-outside"),
    ],
)
def test_a_collision_at_the_publication_instant_is_stepped_over_and_never_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_collision: object,
) -> None:
    """Something takes the name ``segment-0001`` at the exact instant
    publication has decided on it - after staging is verified, right as the
    no-replace rename is about to run. Whatever it is - an empty directory, a
    non-empty one, a plain file, or a symlink pointing outside the capture -
    publication must step to the next free name and must never overwrite,
    empty out, remove, or follow the colliding entry.
    """
    write_frames(tmp_path, range(2))
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "untouched.raw").write_bytes(b"outside content")

    first_candidate = tmp_path / "segment-0001"
    original_rename = capture_sequence_module._no_replace_rename
    collided = {"done": False}

    def colliding_rename(src: Path, dst: Path) -> None:
        if not collided["done"] and dst == first_candidate:
            make_collision(first_candidate, outside)  # type: ignore[operator]
            collided["done"] = True
        original_rename(src, dst)

    monkeypatch.setattr(capture_sequence_module, "_no_replace_rename", colliding_rename)

    publication = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(10.0),
        evidence=accepted_evidence(),
    )

    assert collided["done"] is True
    assert publication.directory == tmp_path / "segment-0002"

    if first_candidate.is_symlink():
        assert first_candidate.readlink() == outside
        assert (outside / "untouched.raw").read_bytes() == b"outside content"
        assert [entry.name for entry in outside.iterdir()] == ["untouched.raw"]
    elif first_candidate.is_dir():
        contents = sorted(entry.name for entry in first_candidate.iterdir())
        assert contents in ([], ["marker.txt"])
        if contents:
            assert (first_candidate / "marker.txt").read_bytes() == b"do not touch"
    else:
        assert first_candidate.read_bytes() == b"do not touch"


def test_a_failure_during_the_publish_rename_leaves_no_segment_and_no_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real (non-collision) failure raised by the publish rename itself
    must be treated exactly like a failure earlier in staging: the staging
    directory is removed and no ``segment-*`` name is left behind, published
    or half-published.
    """
    write_frames(tmp_path, range(2))
    before = sorted(entry.name for entry in tmp_path.iterdir())

    def failing_rename(src: Path, dst: Path) -> None:
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(capture_sequence_module, "_no_replace_rename", failing_rename)

    with pytest.raises(OSError, match="synthetic rename failure"):
        publish_segment(
            tmp_path,
            frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
            predecessor_index=1,
            transform=transform(11.0),
            evidence=accepted_evidence(),
        )

    after = sorted(entry.name for entry in tmp_path.iterdir())
    assert after == before
    assert not list(tmp_path.glob(SEGMENT_GLOB))
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))


# --- strided(): the per-segment policy on a real boundary ---------------------------


def _segmented_capture(tmp_path: Path) -> tuple[Path, set[int]]:
    """Root frames 0-9, two published boundaries and ordinary frames after
    each, 30 frames total. Returns the capture directory and the set of
    frame indices every published boundary bridges FROM - the ones
    :meth:`CaptureSequence.strided` must never drop.
    """
    write_frames(tmp_path, range(10))
    seg1 = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in range(10, 15)],
        predecessor_index=9,
        transform=transform(4.0),
        evidence=accepted_evidence(5),
    )
    write_frames(seg1.directory, range(15, 20))
    seg2 = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in range(20, 25)],
        predecessor_index=19,
        transform=transform(5.0),
        evidence=accepted_evidence(5),
    )
    write_frames(seg2.directory, range(25, 30))
    return tmp_path, {9, 19}


@pytest.mark.parametrize("stride", [2, 3, 5])
def test_strided_keeps_every_bridged_pair_adjacent_on_a_segmented_capture(
    tmp_path: Path, stride: int
) -> None:
    """For each tested stride, every bridge's predecessor and start frame
    both survive striding and remain adjacent in the strided list - the
    property :meth:`~scanner3d.capture_sequence.CaptureSequence.boundaries`
    depends on, checked here by calling it and letting it raise if that
    property does not hold.
    """
    directory, predecessors = _segmented_capture(tmp_path)
    sequence = read_capture_sequence(directory)
    assert {bridge.predecessor_index for bridge in sequence.bridges} == predecessors

    strided = sequence.strided(stride)
    kept = {frame.index for frame in strided.frames}
    for bridge in strided.bridges:
        assert bridge.predecessor_index in kept
        assert bridge.start_index in kept

    boundaries = strided.boundaries()  # raises if any bridged pair is not adjacent
    assert len(boundaries) == 2
    assert any(f"strided one in {stride}" in limitation for limitation in strided.limitations)


def test_strided_by_one_is_the_capture_unchanged(tmp_path: Path) -> None:
    directory, _ = _segmented_capture(tmp_path)
    sequence = read_capture_sequence(directory)
    assert sequence.strided(1) is sequence


def test_strided_rejects_a_stride_below_one(tmp_path: Path) -> None:
    write_frames(tmp_path, range(3))
    sequence = read_capture_sequence(tmp_path)
    with pytest.raises(CaptureSequenceUnreadable, match="stride must be at least 1"):
        sequence.strided(0)


# --- next_frame_index --------------------------------------------------------------


def test_next_frame_index_on_a_root_only_capture(tmp_path: Path) -> None:
    write_frames(tmp_path, [0, 1, 2, 5])
    assert next_frame_index(tmp_path) == 6


def test_next_frame_index_on_an_empty_capture(tmp_path: Path) -> None:
    assert next_frame_index(tmp_path) == 0


def test_next_frame_index_looks_inside_segment_directories_too(tmp_path: Path) -> None:
    directory, _ = _segmented_capture(tmp_path)
    assert next_frame_index(directory) == 30


def test_publish_segment_refuses_a_non_consecutive_offered_batch(tmp_path: Path) -> None:
    """The guard on the writing side, mirroring the reader's own: a caller
    offering a non-consecutive batch is refused before anything is written,
    same as a record claiming one would be refused on read. Sized at
    ``MIN_TENTATIVE_FRAMES`` frames so the failure is provably about the gap
    (1, 2, skip to 4) and not about the batch being too small.
    """
    write_frames(tmp_path, range(1))
    with pytest.raises(SegmentPublicationRefused, match="not one consecutive run"):
        publish_segment(
            tmp_path,
            frames=[(1, frame_bytes(1)), (2, frame_bytes(2)), (4, frame_bytes(4))],
            predecessor_index=0,
            transform=transform(6.0),
            evidence={},
        )
    assert not list(tmp_path.glob(SEGMENT_GLOB))


def test_publish_segment_refuses_a_batch_below_min_tentative_frames(tmp_path: Path) -> None:
    """The writer-side mirror of the reader's own floor: publishing fewer
    than ``MIN_TENTATIVE_FRAMES`` would write a record no reader could ever
    accept back, so it is refused before anything touches disk.
    """
    write_frames(tmp_path, range(1))
    with pytest.raises(SegmentPublicationRefused, match=f"fewer than {MIN_TENTATIVE_FRAMES}"):
        publish_segment(
            tmp_path,
            frames=[(1, frame_bytes(1))],
            predecessor_index=0,
            transform=transform(7.0),
            evidence={},
        )
    assert not list(tmp_path.glob(SEGMENT_GLOB))


def test_publish_segment_refuses_a_non_consecutive_predecessor(tmp_path: Path) -> None:
    """A batch that runs forwards and is internally consecutive is still
    refused if its first frame is not exactly one past the predecessor - the
    writer-side mirror of the reader's own boundary check.
    """
    write_frames(tmp_path, range(3))
    with pytest.raises(SegmentPublicationRefused, match="very next global index"):
        publish_segment(
            tmp_path,
            frames=[(index, frame_bytes(index)) for index in (5, 6, 7)],
            predecessor_index=2,
            transform=transform(8.0),
            evidence={},
        )
    assert not list(tmp_path.glob(SEGMENT_GLOB))


# --- an atomic no-replace rename, or no publication at all ------------------------


def test_publication_refuses_outright_when_no_atomic_no_replace_rename_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``renameat2(RENAME_NOREPLACE)`` means no boundary, not a weaker one.

    There used to be a fallback here - ``lstat`` the destination, then plain
    ``rename`` - and it was precisely the check-then-use race the no-replace
    primitive exists to close: POSIX ``rename`` silently replaces an existing
    EMPTY directory, so a name created in the gap between those two calls is a
    name the publication then eats. This holds the primitive unavailable and
    proves the publication refuses: no ``segment-*``, no staging residue, and
    - the point of the test - no ordinary rename attempted at any level.
    """
    write_frames(tmp_path, range(2))
    before = sorted(entry.name for entry in tmp_path.iterdir())

    def unavailable(src: Path, dst: Path) -> None:
        raise capture_sequence_module._RenameAt2Unavailable("synthetic: no renameat2 here")

    monkeypatch.setattr(capture_sequence_module, "_renameat2_noreplace", unavailable)

    renamed: list[tuple[str, str]] = []
    original_path_rename = Path.rename
    original_os_rename = os.rename

    def spy_path_rename(self: Path, target: object) -> Path:
        renamed.append((str(self), str(target)))
        return original_path_rename(self, target)  # type: ignore[arg-type]

    def spy_os_rename(src: object, dst: object, **kwargs: object) -> None:
        renamed.append((str(src), str(dst)))
        original_os_rename(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "rename", spy_path_rename)
    monkeypatch.setattr(os, "rename", spy_os_rename)

    with pytest.raises(SegmentPublicationRefused, match="no atomic no-replace rename"):
        publish_segment(
            tmp_path,
            frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
            predecessor_index=1,
            transform=transform(12.0),
            evidence=accepted_evidence(),
        )

    assert renamed == [], "a publication fell back to an ordinary, replacing rename"
    assert not list(tmp_path.glob(SEGMENT_GLOB))
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == before

    sequence = read_capture_sequence(tmp_path)
    assert [frame.index for frame in sequence.frames] == [0, 1]
    assert sequence.bridges == ()


# --- gaps: legacy captures may have them, segmented captures may not --------------


def test_a_missing_first_ordinary_frame_after_a_boundary_refuses(tmp_path: Path) -> None:
    """The batch ends at 4 and the next ordinary frame on disk is 6. Sorting
    hides that perfectly - the list still comes back in order - so it is
    checked explicitly, because a reconstruction would otherwise register 4
    against 6 as an adjacent pair.
    """
    write_frames(tmp_path, range(2))
    publication = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(13.0),
        evidence=accepted_evidence(),
    )
    write_frame(publication.directory, 6)

    with pytest.raises(BridgeRecordUnusable, match="one consecutive run continuing from it"):
        read_capture_sequence(tmp_path)


def test_a_missing_interior_ordinary_frame_inside_a_segment_refuses(tmp_path: Path) -> None:
    """Batch 2-4, then ordinary frames 5, 6 and 8: frame 7 has gone missing
    from the middle of the run.
    """
    write_frames(tmp_path, range(2))
    publication = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
        predecessor_index=1,
        transform=transform(14.0),
        evidence=accepted_evidence(),
    )
    for index in (5, 6, 8):
        write_frame(publication.directory, index)

    with pytest.raises(BridgeRecordUnusable, match="one consecutive run continuing from it"):
        read_capture_sequence(tmp_path)


def test_a_gap_in_the_first_segment_of_a_two_segment_capture_refuses(tmp_path: Path) -> None:
    """Two boundaries, and the ordinary run inside the FIRST one is missing a
    frame. The second segment is untouched and perfectly valid, which is
    exactly the case a per-segment check has to catch rather than a
    whole-capture count.
    """
    write_frames(tmp_path, range(3))
    first = publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (3, 4, 5)],
        predecessor_index=2,
        transform=transform(15.0),
        evidence=accepted_evidence(),
    )
    for index in (6, 7):
        write_frame(first.directory, index)
    publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (8, 9, 10)],
        predecessor_index=7,
        transform=transform(16.0),
        evidence=accepted_evidence(),
    )
    # Reads cleanly until frame 6 is taken out from under it.
    assert [frame.index for frame in read_capture_sequence(tmp_path).frames] == list(range(11))
    (first.directory / FRAME_TEMPLATE.format(index=6)).unlink()

    with pytest.raises(BridgeRecordUnusable, match="one consecutive run continuing from it"):
        read_capture_sequence(tmp_path)


def test_a_gap_among_the_root_frames_of_a_segmented_capture_refuses(tmp_path: Path) -> None:
    """Root gaps are legacy-compatible only for a capture with no bridges. A
    capture that carries boundary evidence was written by a writer that never
    skips an index, so a hole in its pre-pause frames is a missing file.
    """
    write_frames(tmp_path, range(4))
    publish_segment(
        tmp_path,
        frames=[(index, frame_bytes(index)) for index in (4, 5, 6)],
        predecessor_index=3,
        transform=transform(17.0),
        evidence=accepted_evidence(),
    )
    (tmp_path / FRAME_TEMPLATE.format(index=1)).unlink()

    with pytest.raises(CaptureSequenceUnreadable, match=r"missing frame\(s\) \[1\]"):
        read_capture_sequence(tmp_path)


# --- the persisted local chain is validated, not merely carried -------------------


def _segment_with_chain(tmp_path: Path, chain: object) -> None:
    """Root frames 0-1 and one hand-built three-frame segment whose evidence
    carries exactly ``chain`` as its ``local_chain``.
    """
    write_frames(tmp_path, range(2))
    evidence = accepted_evidence()
    evidence["local_chain"] = chain
    indices = [2, 3, 4]
    write_segment(
        tmp_path / "segment-0001",
        valid_record(predecessor=1, start=2, indices=indices, evidence=evidence),
        indices,
    )


def test_a_record_whose_evidence_carries_no_local_chain_refuses(tmp_path: Path) -> None:
    """A bridge says where a segment was put. Without a local chain nothing in
    the record says the published frames were ever a segment at all.
    """
    write_frames(tmp_path, range(2))
    evidence = accepted_evidence()
    del evidence["local_chain"]
    indices = [2, 3, 4]
    write_segment(
        tmp_path / "segment-0001",
        valid_record(predecessor=1, start=2, indices=indices, evidence=evidence),
        indices,
    )

    with pytest.raises(BridgeRecordUnusable, match="no local_chain object"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("chain", [[], "a chain", 3, None])
def test_a_local_chain_that_is_not_an_object_refuses(tmp_path: Path, chain: object) -> None:
    _segment_with_chain(tmp_path, chain)
    with pytest.raises(BridgeRecordUnusable, match="no local_chain object"):
        read_capture_sequence(tmp_path)


def test_a_local_chain_with_no_gates_object_refuses(tmp_path: Path) -> None:
    """Links measured against thresholds the record does not name are links
    this reader cannot re-judge."""
    chain = local_chain(3)
    del chain["gates"]
    _segment_with_chain(tmp_path, chain)

    with pytest.raises(BridgeRecordUnusable, match="names no gates object"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("anchors", [0, 2, True, "one", None])
def test_a_local_chain_without_exactly_one_anchor_refuses(tmp_path: Path, anchors: object) -> None:
    """A tentative segment has exactly one frame that starts it. Two anchors
    means two chains that were never joined; none means a chain with no start.
    """
    _segment_with_chain(tmp_path, local_chain(3, anchor_frames=anchors))
    with pytest.raises(BridgeRecordUnusable, match="anchor frame"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_a_local_chain_whose_edge_count_disagrees_with_the_batch_refuses(
    tmp_path: Path, count: int
) -> None:
    """Three published frames are exactly two adjacent links. Any other number
    is a chain describing a batch other than the one that was published.
    """
    _segment_with_chain(
        tmp_path, local_chain(3, edges=[local_edge(frame) for frame in range(1, count + 1)])
    )
    with pytest.raises(BridgeRecordUnusable, match="adjacent link"):
        read_capture_sequence(tmp_path)


def test_a_local_chain_whose_edges_are_not_sequential_refuses(tmp_path: Path) -> None:
    """Positions 1 and 3 are not a chain of three frames; the link into frame
    2 is simply not there, whatever the count says.
    """
    _segment_with_chain(tmp_path, local_chain(3, edges=[local_edge(1), local_edge(3)]))
    with pytest.raises(BridgeRecordUnusable, match="links of a chain are consecutive"):
        read_capture_sequence(tmp_path)


def test_a_local_chain_whose_edges_are_consecutive_but_shifted_refuses(tmp_path: Path) -> None:
    """Positions 2 and 3 are the right count and are consecutive, and are still
    not this batch's chain: the link into frame position 1 is missing, and
    position 3 is past the last frame of a three-frame batch.
    """
    _segment_with_chain(tmp_path, local_chain(3, edges=[local_edge(2), local_edge(3)]))
    with pytest.raises(BridgeRecordUnusable, match="starts at frame position 2"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("frame", [0, -1, True, "1", None])
def test_a_local_chain_edge_with_no_usable_frame_position_refuses(
    tmp_path: Path, frame: object
) -> None:
    _segment_with_chain(
        tmp_path, local_chain(3, edges=[{**local_edge(1), "frame": frame}, local_edge(2)])
    )
    with pytest.raises(BridgeRecordUnusable, match="frame position"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("accepted", [False, None, "yes", 1])
def test_a_local_chain_carrying_a_refused_edge_refuses(tmp_path: Path, accepted: object) -> None:
    """A link the capture itself dropped is not part of a published segment.
    A record that publishes one is describing a chain that never existed.
    """
    _segment_with_chain(
        tmp_path, local_chain(3, edges=[local_edge(1), local_edge(2, accepted=accepted)])
    )
    with pytest.raises(BridgeRecordUnusable, match="does not say accepted: true"):
        read_capture_sequence(tmp_path)


def test_a_local_chain_edge_accepted_without_converging_refuses(tmp_path: Path) -> None:
    _segment_with_chain(
        tmp_path, local_chain(3, edges=[local_edge(1), local_edge(2, converged=False)])
    )
    with pytest.raises(BridgeRecordUnusable, match="without its solve having converged"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("measurement", ["rmse", "matched_fraction", "used_fraction"])
@pytest.mark.parametrize("value", [None, "0.02", True, float("nan"), float("inf")])
def test_a_local_chain_edge_with_a_non_finite_measurement_refuses(
    tmp_path: Path, measurement: str, value: object
) -> None:
    """``null`` is what a non-finite value serializes to (see
    ``relocalize._json_number``), so an edge saying ``accepted: true`` over a
    ``null`` residual is an acceptance with no number behind it.
    """
    _segment_with_chain(
        tmp_path,
        local_chain(3, edges=[local_edge(1), local_edge(2, **{measurement: value})]),
    )
    # NaN and inf cannot survive a strict JSON round trip; the record this
    # writes holds them as the literals Python's json emits, which is exactly
    # what a hand-edited file could hold.
    with pytest.raises(BridgeRecordUnusable, match="not a finite measurement"):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize("field", ["iterations", "resolved_backend"])
def test_a_local_chain_edge_missing_a_structural_field_refuses(tmp_path: Path, field: str) -> None:
    edge = local_edge(2)
    del edge[field]
    _segment_with_chain(tmp_path, local_chain(3, edges=[local_edge(1), edge]))
    with pytest.raises(BridgeRecordUnusable):
        read_capture_sequence(tmp_path)


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ({"local_chain": None}, "no local_chain object"),
        ({"local_chain": {"gates": {}, "anchor_frames": 1, "edges": []}}, "adjacent link"),
        (
            {
                "local_chain": {
                    "gates": {},
                    "anchor_frames": 1,
                    "edges": [local_edge(1), local_edge(2, accepted=False)],
                }
            },
            "does not say accepted: true",
        ),
        (
            {
                "local_chain": {
                    "gates": {},
                    "anchor_frames": 1,
                    "edges": [local_edge(2), local_edge(3)],
                }
            },
            "starts at frame position 2",
        ),
    ],
)
def test_publish_segment_refuses_evidence_its_own_reader_would_reject(
    tmp_path: Path, broken: dict[str, object], message: str
) -> None:
    """The writer runs the reader's checks BEFORE staging exists.

    A publication whose evidence could never be read back is not a cosmetic
    flaw: those frames have left the tentative buffer and been counted, and
    the capture would be unreadable as a whole from that moment on. Refusing
    here leaves the capture directory holding exactly what it held before.
    """
    write_frames(tmp_path, range(2))
    before = sorted(entry.name for entry in tmp_path.iterdir())
    evidence = accepted_evidence()
    evidence.update(broken)

    with pytest.raises(SegmentPublicationRefused, match=message):
        publish_segment(
            tmp_path,
            frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
            predecessor_index=1,
            transform=transform(18.0),
            evidence=evidence,
        )

    assert sorted(entry.name for entry in tmp_path.iterdir()) == before
    assert not list(tmp_path.glob(SEGMENT_GLOB))
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))


def test_publish_segment_refuses_limitations_its_own_reader_would_reject(
    tmp_path: Path,
) -> None:
    """The same writer-side mirror, for the limitations a boundary carries
    verbatim into a reconstruction's own reported limitations.
    """
    write_frames(tmp_path, range(2))
    before = sorted(entry.name for entry in tmp_path.iterdir())

    with pytest.raises(SegmentPublicationRefused, match="other than a plain string"):
        publish_segment(
            tmp_path,
            frames=[(index, frame_bytes(index)) for index in (2, 3, 4)],
            predecessor_index=1,
            transform=transform(19.0),
            evidence=accepted_evidence(),
            limitations=[17],  # type: ignore[list-item]
        )

    assert sorted(entry.name for entry in tmp_path.iterdir()) == before
    assert not list(tmp_path.glob(SEGMENT_GLOB))
    assert not list(tmp_path.glob(f"{STAGING_PREFIX}*"))
