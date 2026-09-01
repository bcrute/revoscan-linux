"""What a capture folder actually holds, read once and read the same way.

Why this exists
---------------
Until a capture could be paused, "the frames" meant ``sorted(glob("frame_*.raw"))``
and every consumer wrote that line for itself: the window's Reconstruct, the
CLI's ``reconstruct``, the two places that count what a finished capture wrote.
Four copies of one assumption is survivable while the assumption holds.

Pausing breaks it. A capture that was repositioned mid-sweep has frames that are
NOT a single chain: somewhere in the middle there is a pair of consecutive
frames whose relationship was established by a global relocalization rather than
by ordinary adjacent alignment, and fitting that pair with ICP is the one thing
that must not happen. A glob returns those frames in the right order and says
nothing about the seam, so every consumer of a glob would silently do the wrong
thing on a paused capture and report a healthy-looking number for it.

So there is one reader. It returns the frames in order AND the verified bridge
constraints between them, and a consumer that has one cannot accidentally not
have the other.

The layout
----------
A capture directory holds its pre-pause frames at the root, exactly as every
capture always has::

    my-scan/
        frame_0000.raw
        frame_0001.raw
        ...
        frame_0040.raw
        segment-0001/
            bridge.json
            frame_0041.raw     <- the accepted tentative batch, published atomically
            frame_0042.raw
            frame_0043.raw
            frame_0044.raw     <- ordinary frames after acceptance, written normally
            ...

A legacy capture is simply one with no ``segment-`` directory, and it reads
back byte for byte as it always did. Frame indices are GLOBAL and unique across
the whole capture, so the order is the index order whichever directory a frame
sits in, and a segment's own name never has to be parsed to place its frames.

What a bridge record has to survive
-----------------------------------
The record is read back by a reconstruction that will use its transform INSTEAD
of fitting the pair, so a record that is wrong is worse than a record that is
missing: the missing one refuses, and the wrong one produces a confident model
joined at the wrong angle. Everything it claims is therefore checked against the
bytes on disk - each named raw frame's length and SHA-256 - and every way the
directory could be something other than what it looks like is refused rather
than worked around: a name that is not exactly the canonical one, a symlink, a
frame index that duplicates one already seen, a predecessor that does not exist,
an unparseable or wrong-shaped JSON, a schema version from the future.

None of that is defensive decoration. A capture directory is a place the
operator can open in a file manager, and a scan is worth keeping for years.

The other kind of break
-----------------------
A pause is a break the operator asked for and the scan then proved it had
crossed. A TRACKING LOSS is the break nobody asked for and nothing crossed: live
registration stops working, and the frames after it are related to the frames
before it by nothing at all. Such a run publishes one ``tracking-loss.json``
inside the directory it was writing into, and this reader hands that run over
only as far as the frame the record names - see :data:`TRACKING_LOSS_STREAK` and
:func:`read_tracking_loss`. The frames and the evidence stay; what is refused is
the claim that they are one sequence.

A loss can then be CLOSED, and only one way: the segment immediately after it
carries a ``recovery`` object binding it to that exact record, by the record's
own SHA-256 and by every excluded frame's, and its transform comes from the same
strict global relocalization a pause boundary uses. The excluded frames stay on
disk and stay out of the sequence. A loss nothing closes refuses the whole
capture, exactly as it did before recovery existed - see :class:`RecoveredLoss`
and :func:`publish_recovery_segment`.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno as errno_mod
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Matrix3, Point, Rigid
from scanner3d.pointcloud import sha256_of

SCHEMA_VERSION: Final = 1

#: What a raw depth frame is called, wherever in a capture it sits. One format
#: string for the writer and one expression for the reader, matched against the
#: WHOLE name so that a file whose index does not round-trip through the format
#: is refused rather than silently renumbered.
FRAME_TEMPLATE: Final = "frame_{index:04d}.raw"
FRAME_GLOB: Final = "frame_*.raw"
_FRAME_NAME = re.compile(r"^frame_(\d{4,})\.raw$")

#: The directory one accepted pause boundary publishes, and the record inside it.
SEGMENT_TEMPLATE: Final = "segment-{index:04d}"
SEGMENT_GLOB: Final = "segment-*"
_SEGMENT_NAME = re.compile(r"^segment-(\d{4,})$")
BRIDGE_RECORD_NAME: Final = "bridge.json"

#: Prefix of the half-written directory a publication stages into. Leading dot
#: so it is hidden from a file manager, and not matched by :data:`SEGMENT_GLOB`,
#: so a reader can never mistake staging for a published segment.
STAGING_PREFIX: Final = ".segment-staging-"

#: How many segment names one capture may take before publication refuses.
#: A pause costs the operator a reposition, so a capture with hundreds of them
#: is not a capture, it is a loop that has gone wrong.
SEGMENT_ATTEMPTS: Final = 999

#: What the record's ``kind`` says, so a file found loose is identifiable.
RECORD_KIND: Final = "scanner3d.relocalization-bridge"


class CaptureSequenceUnreadable(ScannerError):
    """A capture directory does not describe a sequence this will hand over.

    Raised instead of returning a partial answer, always. Half a sequence is
    indistinguishable from a whole one to the caller that gets it, and the whole
    reason this module exists is that guessing at a paused capture produces a
    model that looks fine and is wrong.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class BridgeRecordUnusable(CaptureSequenceUnreadable):
    """One segment's boundary evidence is missing, malformed or contradicted."""


class SegmentPublicationRefused(ScannerError):
    """A pause boundary could not be published, so nothing was written."""

    status = DoctorStatus.UNSUPPORTED_MODE


class TrackingLossRecordUnusable(CaptureSequenceUnreadable):
    """A capture holds something loss-record-shaped that cannot be read as one.

    A separate refusal from :class:`TrackingLossUnresolved` because the two say
    different things to the operator - "this capture is known to be broken" and
    "this capture claims something about itself that cannot be checked" - and
    the same thing to the caller, which is that no sequence is handed over.
    """


class TrackingLossUnresolved(CaptureSequenceUnreadable):
    """The capture recorded a live tracking loss that nothing has resolved.

    The frames are still there and the record is still there; what is NOT there
    is any pose relating the frames after the loss to the ones before it. So the
    directory is refused as one reconstructable sequence rather than handed over
    with a caveat: a caveat is something a caller can decline to read, and
    registering across that seam produces a confident, wrong model.
    """


class TrackingLossPublicationRefused(ScannerError):
    """The loss record could not be published, so the loss is not durable.

    Deliberately NOT a quiet outcome. A capture that has decided it is lost and
    cannot say so on disk is the exact state this whole mechanism exists to
    prevent, so the capture fails with this rather than carrying on unmarked.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


@dataclass(frozen=True, slots=True)
class SequenceFrame:
    """One raw frame, where it lives and what its global index is."""

    path: Path
    index: int
    #: The segment directory's name, or ``None`` for a frame at the capture root.
    #: Root frames are the pre-pause sweep and, for a legacy capture, all of it.
    segment: str | None

    @property
    def in_segment(self) -> bool:
        return self.segment is not None


@dataclass(frozen=True, slots=True)
class BridgeConstraint:
    """A verified relative transform across one accepted pause boundary.

    ``transform`` plays exactly the role an adjacent ICP fit plays everywhere
    else in this repository: it carries the SEGMENT-START frame's coordinates
    into the PREDECESSOR frame's, so a sequential pose chain composes it in the
    same place and the same direction it would compose a fitted pair. Stated in
    those words because the direction is the one thing here that is invisible
    when it is wrong.
    """

    segment: str
    predecessor_index: int
    start_index: int
    transform: Rigid
    #: The relocalization's own recorded evidence and thresholds, verbatim.
    evidence: Mapping[str, object]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "segment": self.segment,
            "predecessor_frame_index": self.predecessor_index,
            "segment_start_frame_index": self.start_index,
            "transform": self.transform.as_dict(),
            "evidence": dict(self.evidence),
            "limitations": list(self.limitations),
        }


#: What a reconstruction is told about a boundary once the frames it applies to
#: have been chosen: the POSITION in the frame list whose pair with the one
#: before it must not be fitted, and the transform to use instead.
@dataclass(frozen=True, slots=True)
class BridgeBoundary:
    """A bridge, expressed against a particular ordered list of frames."""

    position: int
    transform: Rigid
    source: str
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaptureSequence:
    """Every frame of one capture in order, and the seams between them."""

    directory: Path
    frames: tuple[SequenceFrame, ...]
    bridges: tuple[BridgeConstraint, ...]
    limitations: tuple[str, ...]
    #: Each tracking loss this capture recorded and then closed by an accepted
    #: relocalization. Empty for every capture that never lost tracking, which
    #: is what keeps a healthy or legacy capture reading back exactly as it did.
    #: Defaulted rather than required for the same reason: a caller that builds
    #: a sequence by hand is describing a capture with no losses in it.
    recoveries: tuple[RecoveredLoss, ...] = ()

    @property
    def excluded(self) -> tuple[SequenceFrame, ...]:
        """Frames on disk that no pose places, in global order.

        Not a gap and not a loss of data - they are the frames written while
        tracking was lost, kept exactly where they were as the evidence for it.
        Named here because a consumer counting files in the folder would
        otherwise have no way to explain why there are more of them than
        :attr:`frames`.
        """
        return tuple(frame for recovery in self.recoveries for frame in recovery.excluded)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(frame.path for frame in self.frames)

    @property
    def segmented(self) -> bool:
        """Whether this capture was ever repositioned mid-sweep."""
        return bool(self.bridges)

    def boundaries(self) -> tuple[BridgeBoundary, ...]:
        """Each bridge as a position in :attr:`frames`, ready for registration.

        A bridge whose two frames are not adjacent in this particular frame list
        cannot be expressed and is a programming error rather than a data one:
        :meth:`strided` is the only thing that drops frames and it is written
        never to separate a bridged pair.
        """
        at_index = {frame.index: position for position, frame in enumerate(self.frames)}
        boundaries: list[BridgeBoundary] = []
        for bridge in self.bridges:
            start = at_index.get(bridge.start_index)
            predecessor = at_index.get(bridge.predecessor_index)
            if start is None or predecessor is None or start - predecessor != 1:
                raise BridgeRecordUnusable(
                    f"{bridge.segment} bridges frame {bridge.predecessor_index} to frame "
                    f"{bridge.start_index}, and this frame list does not hold them as an "
                    "adjacent pair. Registering it would fit ICP across the reposition."
                )
            boundaries.append(
                BridgeBoundary(
                    position=start,
                    transform=bridge.transform,
                    source=bridge.segment,
                    limitations=bridge.limitations,
                )
            )
        return tuple(boundaries)

    def strided(self, stride: int) -> CaptureSequence:
        """One frame in ``stride``, keeping every frame a bridge depends on.

        The policy is PER SEGMENT and it is the reason this is a method rather
        than a slice at the call site. Striding the whole list uniformly will,
        for most strides, drop either the frame a bridge starts at or the one it
        starts FROM, and the result is a sequence whose seam is no longer an
        adjacent pair - at which point a reconstruction either has to refuse or,
        far worse, quietly fit ICP across the reposition.

        So each run of frames - the root sweep, then each segment - is strided
        within itself, which always keeps its own first frame, and the last
        frame of a run is kept as well when the next run bridges from it. Every
        bridged pair therefore stays adjacent, at every stride, and a legacy
        capture with no segments is the plain ``[::stride]`` it always was.
        """
        if stride < 1:
            raise CaptureSequenceUnreadable(f"stride must be at least 1, got {stride}")
        if stride == 1:
            return self
        predecessors = {bridge.predecessor_index for bridge in self.bridges}
        kept: list[SequenceFrame] = []
        for run in self._runs():
            keep = list(run[::stride])
            tail = run[-1]
            if tail.index in predecessors and (not keep or keep[-1].index != tail.index):
                keep.append(tail)
            kept.extend(keep)
        limitations = list(self.limitations)
        if stride > 1:
            limitations.append(
                f"Frames were strided one in {stride}, within each captured segment "
                "separately, so that every relocalization bridge kept both of the frames it "
                f"relates. {len(kept)} of {len(self.frames)} frames were registered."
            )
        return CaptureSequence(
            directory=self.directory,
            frames=tuple(kept),
            bridges=self.bridges,
            limitations=tuple(limitations),
            recoveries=self.recoveries,
        )

    def _runs(self) -> list[list[SequenceFrame]]:
        """The frames grouped by the directory they were written into, in order."""
        runs: list[list[SequenceFrame]] = []
        for frame in self.frames:
            if not runs or frame.segment != runs[-1][0].segment:
                runs.append([])
            runs[-1].append(frame)
        return runs


# --- reading ------------------------------------------------------------------


def _frame_index(path: Path) -> int | None:
    """The global index a raw frame's name states, or ``None`` if it states none.

    Matched against the whole name and re-formatted back, so a name that would
    not be produced by :data:`FRAME_TEMPLATE` for the index it parses to - a
    zero-padded duplicate such as ``frame_00007.raw`` beside ``frame_0007.raw`` -
    is not a frame this reader will place.
    """
    match = _FRAME_NAME.match(path.name)
    if match is None:
        return None
    index = int(match.group(1))
    return index if FRAME_TEMPLATE.format(index=index) == path.name else None


def _plain_file(path: Path, *, what: str) -> None:
    """Refuse anything that is not a regular file this process may read.

    ``lstat``, so a symlink is seen as a symlink rather than as whatever it
    points at. A capture directory sits under a path the operator chose and can
    edit, and a reader that followed a link out of it would hash bytes from
    somewhere it was never pointed.
    """
    try:
        info = path.lstat()
    except OSError as exc:
        raise BridgeRecordUnusable(f"{what} at {path} cannot be examined: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise BridgeRecordUnusable(
            f"{what} at {path} is a symbolic link. A capture's frames are read as bytes on "
            "disk, and a link is not followed out of the capture directory."
        )
    if not stat.S_ISREG(info.st_mode):
        raise BridgeRecordUnusable(f"{what} at {path} is not a regular file.")


def _plain_directory(path: Path, *, what: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BridgeRecordUnusable(f"{what} at {path} cannot be examined: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise BridgeRecordUnusable(
            f"{what} at {path} is a symbolic link rather than a directory this capture owns."
        )
    if not stat.S_ISDIR(info.st_mode):
        raise BridgeRecordUnusable(f"{what} at {path} is not a directory.")


def _rigid_from(payload: object, *, what: str) -> Rigid:
    """A :class:`~scanner3d.geometry.Rigid` out of a record, or a refusal.

    Built through :meth:`Rigid.of`, which validates, so a record carrying a
    matrix that is not a rotation - hand-edited, truncated, or produced by a
    version that got the convention wrong - is refused here rather than becoming
    a transform that silently scales the second half of somebody's scan.
    """
    if not isinstance(payload, Mapping):
        raise BridgeRecordUnusable(f"{what} is not an object.")
    rotation = payload.get("rotation")
    translation = payload.get("translation")
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
    ):
        raise BridgeRecordUnusable(f"{what} has no 3x3 rotation.")
    if not isinstance(translation, list) or len(translation) != 3:
        raise BridgeRecordUnusable(f"{what} has no three-component translation.")
    try:
        matrix: Matrix3 = tuple(tuple(float(value) for value in row) for row in rotation)  # type: ignore[assignment]
        point: Point = tuple(float(value) for value in translation)  # type: ignore[assignment]
    except (TypeError, ValueError) as exc:
        raise BridgeRecordUnusable(f"{what} holds a value that is not a number: {exc}") from exc
    try:
        return Rigid.of(matrix, point)
    except ScannerError as exc:
        raise BridgeRecordUnusable(f"{what} is not a rigid transform: {exc}") from exc


def _finite_number(payload: object) -> bool:
    """A JSON number that is actually a measurement.

    ``bool`` is excluded on purpose - it is an ``int`` in Python and ``true``
    is not a residual - and so is ``null``, which is what
    :func:`scanner3d.relocalize._json_number` writes for a non-finite value. A
    record whose edge reports ``"rmse": null`` is saying the solve produced no
    number; that is a refusal, not a measurement with a gap in it.
    """
    if not isinstance(payload, int | float) or isinstance(payload, bool):
        return False
    return math.isfinite(payload)


def _local_chain_from(
    evidence: Mapping[str, object], *, batch_count: int, what: str
) -> Mapping[str, object]:
    """The segment's own chain, checked against the batch it claims to describe.

    A bridge says where a segment was PUT. The local chain says whether the
    segment was a segment: one anchor frame, and a measured adjacent link into
    every frame after it. Until this was checked the reader accepted a record
    with the object missing entirely, or holding an empty edge list, or holding
    edges for frames that are not in the batch - which is a record claiming a
    global placement for a chain nothing can substantiate, and the transform it
    carries is then used INSTEAD of fitting the pair.

    So for schema 1 the shape is exact: a ``gates`` object naming what the links
    were judged against, exactly one anchor, and edges whose frame positions are
    exactly ``list(range(1, batch_count))`` - the right number of links is not
    enough, because a consecutive run that starts at 2 omits the link into the
    frame after the anchor and names a position past the end of the batch. Each
    edge must say
    ``accepted: true`` in so many words and carry finite measurements behind
    that word, because "accepted" with a ``null`` residual is the failure this
    is here to refuse.
    """
    chain = evidence.get("local_chain")
    if not isinstance(chain, Mapping):
        raise BridgeRecordUnusable(
            f"{what}'s evidence carries no local_chain object, so nothing in it says the "
            "published frames ever chained to one another."
        )
    if not isinstance(chain.get("gates"), Mapping):
        raise BridgeRecordUnusable(
            f"{what}'s local chain names no gates object, so its links were measured against "
            "nothing this reader can name."
        )
    anchors = chain.get("anchor_frames")
    if not isinstance(anchors, int) or isinstance(anchors, bool) or anchors != 1:
        raise BridgeRecordUnusable(
            f"{what}'s local chain claims {anchors!r} anchor frame(s); a tentative segment has "
            "exactly one frame that starts it and no other frame without a link."
        )
    edges = chain.get("edges")
    expected = batch_count - 1
    if not isinstance(edges, list) or len(edges) != expected:
        count = len(edges) if isinstance(edges, list) else edges
        raise BridgeRecordUnusable(
            f"{what}'s local chain holds {count!r} edge(s) for a batch of {batch_count} frames; "
            f"a chain of {batch_count} frames is exactly {expected} adjacent link(s)."
        )
    previous: int | None = None
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise BridgeRecordUnusable(f"{what}'s local chain has an edge that is not an object.")
        frame = edge.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1:
            raise BridgeRecordUnusable(
                f"{what}'s local chain has an edge at frame position {frame!r}; a link's "
                "position is the newer of its two frames, counted from one."
            )
        if previous is None and frame != 1:
            raise BridgeRecordUnusable(
                f"{what}'s local chain starts at frame position {frame}; the first link of a "
                f"segment joins position 1 to the anchor, so a chain of {batch_count} frames "
                f"is exactly the positions {list(range(1, batch_count))} and no shift of them."
            )
        if previous is not None and frame != previous + 1:
            raise BridgeRecordUnusable(
                f"{what}'s local chain steps from frame position {previous} to {frame}; the "
                "links of a chain are consecutive, and a jump is a link that is not there."
            )
        previous = frame
        if edge.get("accepted") is not True:
            raise BridgeRecordUnusable(
                f"{what}'s local chain carries an edge at frame position {frame} that does not "
                "say accepted: true. A refused link is not part of a published segment."
            )
        if not isinstance(edge.get("converged"), bool) or edge.get("converged") is not True:
            raise BridgeRecordUnusable(
                f"{what}'s local chain accepts the edge at frame position {frame} without its "
                "solve having converged."
            )
        iterations = edge.get("iterations")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            raise BridgeRecordUnusable(
                f"{what}'s local chain's edge at frame position {frame} reports no iteration count."
            )
        if not isinstance(edge.get("resolved_backend"), str):
            raise BridgeRecordUnusable(
                f"{what}'s local chain's edge at frame position {frame} does not name the "
                "backend that solved it."
            )
        for measurement in ("rmse", "matched_fraction", "used_fraction"):
            if not _finite_number(edge.get(measurement)):
                raise BridgeRecordUnusable(
                    f"{what}'s local chain's edge at frame position {frame} reports "
                    f"{measurement} as {edge.get(measurement)!r}, which is not a finite "
                    "measurement. An accepted link with no number behind it is refused."
                )
    return chain


def _evidence_from(
    record: Mapping[str, object], *, batch_count: int, what: str
) -> Mapping[str, object]:
    """The relocalization's own accepted evidence, checked rather than trusted.

    A bridge's transform is used INSTEAD of fitting the pair, so the record
    claiming that transform came from an accepted relocalization must actually
    look like one: an object, explicitly ``accepted``, carrying the thresholds
    it was accepted against and at least one candidate it was chosen among.
    Missing or thin evidence is not "evidence with less detail" - it is a
    record that never says why its transform should be trusted, and that is
    refused exactly as a missing record would be.
    """
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        raise BridgeRecordUnusable(f"{what} has no evidence object.")
    if evidence.get("accepted") is not True:
        raise BridgeRecordUnusable(f"{what}'s evidence does not say accepted: true.")
    thresholds = evidence.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise BridgeRecordUnusable(f"{what}'s evidence names no thresholds object.")
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise BridgeRecordUnusable(f"{what}'s evidence lists no candidates it was chosen among.")
    if any(not isinstance(candidate, Mapping) for candidate in candidates):
        raise BridgeRecordUnusable(f"{what}'s evidence has a candidate that is not an object.")
    _local_chain_from(evidence, batch_count=batch_count, what=what)
    return evidence


def _limitations_from(record: Mapping[str, object], *, what: str) -> tuple[str, ...]:
    """The record's own stated limitations, required rather than defaulted.

    A boundary's limitations are carried verbatim into a reconstruction's own
    reported limitations (see :class:`BridgeConstraint`), so a record that
    omits them, or names something other than a plain list of strings, is not
    "a boundary with nothing to say" - it is a record this reader cannot tell
    apart from one that silently dropped a limitation it should have kept.
    """
    stated = record.get("limitations")
    if not isinstance(stated, list):
        raise BridgeRecordUnusable(f"{what} has no limitations list.")
    if any(not isinstance(item, str) for item in stated):
        raise BridgeRecordUnusable(
            f"{what}'s limitations list holds something other than a plain string."
        )
    return tuple(stated)


def _read_record(directory: Path) -> Mapping[str, object]:
    path = directory / BRIDGE_RECORD_NAME
    _plain_file(path, what=f"the boundary record for {directory.name}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BridgeRecordUnusable(f"{path} cannot be read: {exc}") from exc
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeRecordUnusable(
            f"{path} is not readable JSON: {exc}. A boundary whose evidence cannot be parsed "
            "is refused; it is never treated as an ordinary adjacent pair."
        ) from exc
    if not isinstance(record, dict):
        raise BridgeRecordUnusable(f"{path} does not hold a JSON object.")
    if record.get("kind") != RECORD_KIND:
        raise BridgeRecordUnusable(
            f"{path} says kind {record.get('kind')!r} rather than {RECORD_KIND!r}."
        )
    version = record.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
        raise BridgeRecordUnusable(
            f"{path} states schema version {version!r}; this reader understands exactly "
            f"version {SCHEMA_VERSION} and does not guess at an earlier or a later one."
        )
    return record


def _batch_frames(
    directory: Path, record: Mapping[str, object], seen: set[int]
) -> list[SequenceFrame]:
    """The atomically published batch, each file verified against the record."""
    listed = record.get("frames")
    if not isinstance(listed, list) or not listed:
        raise BridgeRecordUnusable(f"{directory.name} lists no frames in its boundary record.")
    frames: list[SequenceFrame] = []
    for entry in listed:
        if not isinstance(entry, Mapping):
            raise BridgeRecordUnusable(f"{directory.name} has a frame entry that is not an object.")
        name = entry.get("file")
        index = entry.get("frame_index")
        byte_count = entry.get("byte_count")
        digest = entry.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or not isinstance(digest, str)
        ):
            raise BridgeRecordUnusable(
                f"{directory.name} has a frame entry missing a name, index, length or digest."
            )
        # The name is rebuilt from the index rather than trusted, which is what
        # closes path traversal: a record naming ``../../frame_0000.raw`` cannot
        # match ``frame_{index:04d}.raw`` for any index, and nothing outside
        # this directory is ever opened.
        expected = FRAME_TEMPLATE.format(index=index)
        if name != expected:
            raise BridgeRecordUnusable(
                f"{directory.name} names a frame {name!r} for index {index}, and the only name "
                f"that index has is {expected!r}. Nothing outside the segment is opened."
            )
        if index in seen:
            raise BridgeRecordUnusable(
                f"frame index {index} appears twice in {directory.parent.name}, in "
                f"{directory.name} and earlier. Global frame indices are unique."
            )
        path = directory / expected
        _plain_file(path, what=f"frame {index} of {directory.name}")
        payload = path.read_bytes()
        if len(payload) != byte_count:
            raise BridgeRecordUnusable(
                f"{path} is {len(payload)} bytes and its boundary record claims {byte_count}."
            )
        if sha256_of(payload) != digest:
            raise BridgeRecordUnusable(
                f"{path} does not hash to the digest its boundary record claims."
            )
        seen.add(index)
        frames.append(SequenceFrame(path=path, index=index, segment=directory.name))
    frames.sort(key=lambda frame: frame.index)
    indices = [frame.index for frame in frames]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise BridgeRecordUnusable(
            f"{directory.name} publishes frames {indices}, which are not one consecutive run. "
            "The atomic batch is the frames buffered while relocalizing, in order."
        )
    # Imported here rather than at module scope: relocalize imports pipeline,
    # and pipeline imports capture_sequence for BridgeBoundary, so a
    # module-scope import of relocalize here would be a cycle. The constant
    # itself is small and stable enough that a local import each call is a
    # fair price for not having to relocate it.
    from scanner3d.relocalize import MIN_TENTATIVE_FRAMES

    if len(frames) < MIN_TENTATIVE_FRAMES:
        raise BridgeRecordUnusable(
            f"{directory.name} publishes {len(frames)} frame(s); a relocalization is never "
            f"accepted from fewer than {MIN_TENTATIVE_FRAMES} tentative frames, so a batch "
            "under that count is not one a genuine acceptance could have produced."
        )
    return frames


def _later_frames(
    directory: Path, batch: Sequence[SequenceFrame], seen: set[int]
) -> list[SequenceFrame]:
    """Ordinary frames written into the segment after it was accepted.

    Not verified against the boundary record and deliberately so: the record
    describes the frames whose acceptance the bridge was decided from, and
    appending to it as the sweep continued would mean rewriting evidence that
    has already been published.

    They are still required to be a CHAIN, though, which is the one thing a
    unique index above the batch does not establish. A segment holding batch
    frames 10, 11, 12 and ordinary frames 13 and 15 has lost frame 14, and
    sorting hides that perfectly: the list comes back in order, the count looks
    plausible, and a reconstruction registers 13 against 15 as though they were
    a small step apart. Root-only legacy captures may have gaps - they were
    written before any of this existed and must keep reading back unchanged -
    but a capture carrying bridge metadata was written by a writer that never
    skips an index, so a gap in one is a missing file, not a style.
    """
    highest = batch[-1].index
    frames: list[SequenceFrame] = []
    for path in sorted(directory.glob(FRAME_GLOB)):
        index = _frame_index(path)
        if index is None:
            raise BridgeRecordUnusable(
                f"{path} is named like a frame and its index does not round-trip; this reader "
                "will not guess which frame it is."
            )
        if any(index == frame.index for frame in batch):
            continue
        if index <= highest:
            raise BridgeRecordUnusable(
                f"{path} sits below the batch {directory.name} published, which ends at frame "
                f"{highest}. Frames after an accepted boundary continue above it."
            )
        if index in seen:
            raise BridgeRecordUnusable(
                f"frame index {index} appears twice in {directory.parent.name}; global frame "
                "indices are unique."
            )
        _plain_file(path, what=f"frame {index} of {directory.name}")
        seen.add(index)
        frames.append(SequenceFrame(path=path, index=index, segment=directory.name))
    frames.sort(key=lambda frame: frame.index)
    indices = [frame.index for frame in frames]
    if indices and indices != list(range(highest + 1, highest + 1 + len(indices))):
        raise BridgeRecordUnusable(
            f"{directory.name} holds ordinary frames {indices} after a batch ending at frame "
            f"{highest}; the frames taken after a boundary was accepted are one consecutive run "
            "continuing from it, and a gap here is a frame that is missing rather than a frame "
            "that was never taken."
        )
    return frames


def read_capture_sequence(directory: Path) -> CaptureSequence:
    """Every frame of one capture, in order, with its verified bridges.

    The only authority on what a capture folder holds. A legacy capture - no
    segment directories, frames at the root - reads back as exactly the sorted
    glob it always was, with no bridges and no added limitation, which is what
    keeps every scan taken before pausing existed reconstructing identically.

    Raises rather than degrading. A directory holding boundary evidence that is
    missing, unparseable, self-contradictory, duplicated, or pointing anywhere
    other than at its own files is refused as a whole: silently falling back to
    the legacy glob would register the reposition with adjacent ICP, which is
    the precise failure this module was written to make impossible.

    A capture is read as ordered RUNS - the root sweep, then each segment - and
    each run may end in its own immutable ``tracking-loss.json``. A run that did
    is handed over only as far as the frame that record names, its failed frames
    are kept on disk and left out, and the segment immediately after it must
    close that exact loss by an accepted recovery bound to the record's own
    bytes. A loss nothing closes refuses the whole capture, exactly as it did
    before recovery existed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CaptureSequenceUnreadable(f"{directory} is not a directory.")

    # Before the frames, deliberately. A capture that recorded a tracking loss
    # is not a sequence with a flaw somewhere in it - it is a sequence that
    # stops at one frame and is then followed by frames nothing places - and
    # the refusal has to say THAT rather than whatever the file names happen to
    # look like. It is also the one check that must not be reachable only when
    # some other complaint about the directory happens not to fire first.
    #
    # A capture holding no segment at all cannot hold a recovery either, so its
    # loss is answered here and the refusal is word for word the one a
    # contained, unrecovered capture has always got.
    root_loss = read_tracking_loss(directory)
    if root_loss is not None and not any(directory.glob(SEGMENT_GLOB)):
        raise TrackingLossUnresolved(root_loss.refusal(directory))

    seen: set[int] = set()
    root: list[SequenceFrame] = []
    for path in sorted(directory.glob(FRAME_GLOB)):
        index = _frame_index(path)
        if index is None:
            raise CaptureSequenceUnreadable(
                f"{path} is named like a frame and its index does not round-trip through "
                f"{FRAME_TEMPLATE!r}; this reader will not guess which frame it is."
            )
        if index in seen:
            raise CaptureSequenceUnreadable(f"frame index {index} appears twice in {directory}.")
        _plain_file(path, what=f"frame {index}")
        seen.add(index)
        root.append(SequenceFrame(path=path, index=index, segment=None))
    root.sort(key=lambda frame: frame.index)

    # Every entry SEGMENT_GLOB matches is examined here, not filtered down to
    # the ones that look canonical: a capture that holds something
    # segment-shaped and WRONG must never fall back to being read as though it
    # held nothing, so a non-canonical name, a plain file, or a symlink
    # standing in for a segment is refused outright rather than skipped.
    segment_dirs: list[Path] = []
    for path in directory.glob(SEGMENT_GLOB):
        match = _SEGMENT_NAME.match(path.name)
        if match is None:
            raise CaptureSequenceUnreadable(
                f"{path} is named like a segment but not in the exact form "
                f"{SEGMENT_TEMPLATE!r}; a capture holding something segment-shaped is never "
                "read as though it held none."
            )
        _plain_directory(path, what=f"the segment directory {path.name}")
        segment_dirs.append(path)
    segment_dirs.sort(key=lambda path: int(_SEGMENT_NAME.match(path.name).group(1)))  # type: ignore[union-attr]

    frames: list[SequenceFrame] = []
    bridges: list[BridgeConstraint] = []
    limitations: list[str] = []
    recoveries: list[RecoveredLoss] = []
    #: The loss the NEXT segment has to close, or ``None`` while the capture is
    #: whole. Carried across runs rather than looked back for, so a recovery is
    #: matched against the loss IMMEDIATELY before it and never against an
    #: earlier one a later segment might reach back to.
    pending: _PendingLoss | None = None

    valid, failed = _run_after_loss(root, root_loss)
    frames.extend(valid)
    if root_loss is not None:
        pending = _PendingLoss(
            loss=root_loss,
            directory=directory,
            run=None,
            excluded=tuple(failed),
            prefix_end=valid[-1].index if valid else None,
        )

    for segment in segment_dirs:
        try:
            record = _read_record(segment)
        except BridgeRecordUnusable as exc:
            if pending is None:
                raise
            # A directory that is interesting in two ways is refused for the one
            # that matters. The run before this one recorded a tracking loss,
            # only a readable recovery record could close it, and this one
            # cannot be read at all - so what the caller is told is that the
            # loss stands, with the record's own complaint named after it.
            raise TrackingLossUnresolved(
                f"{pending.refusal(directory, segment.name)}\nThat segment's own boundary "
                f"record cannot be read either: {exc}"
            ) from exc
        claimed = record.get(RECOVERY_RECORD_KEY)
        # Before the batch and before the bridge, deliberately. Whether this
        # segment is even the right KIND of thing to follow what came before it
        # is the first question, and a capture whose loss nothing closes must be
        # refused for the loss rather than for whatever else is wrong with the
        # directory that was supposed to close it.
        if pending is not None and claimed is None:
            raise TrackingLossUnresolved(pending.refusal(directory, segment.name))
        if pending is None and claimed is not None:
            raise BridgeRecordUnusable(
                f"{segment.name} carries a {RECOVERY_RECORD_KEY} record, and the run before it "
                "recorded no tracking loss for it to close. A recovery is the answer to a "
                "loss; one standing on its own is refused rather than read as an ordinary "
                "pause boundary."
            )
        batch = _batch_frames(segment, record, seen)
        later = _later_frames(segment, batch, seen)

        start = record.get("segment_start_frame_index")
        predecessor = record.get("predecessor_frame_index")
        if not isinstance(start, int) or isinstance(start, bool) or start != batch[0].index:
            raise BridgeRecordUnusable(
                f"{segment.name} claims to start at frame {start!r} and its first published "
                f"frame is {batch[0].index}."
            )
        if not isinstance(predecessor, int) or isinstance(predecessor, bool):
            raise BridgeRecordUnusable(f"{segment.name} names no predecessor frame index.")
        before = frames[-1].index if frames else None
        if before is None or predecessor != before:
            raise BridgeRecordUnusable(
                f"{segment.name} bridges from frame {predecessor!r}, and the frame immediately "
                f"before it in this capture is "
                f"{before if before is not None else 'nothing at all'}. A boundary is only "
                "meaningful against the frame the capture actually paused at."
            )
        if pending is None:
            if start != predecessor + 1:
                raise BridgeRecordUnusable(
                    f"{segment.name} bridges from frame {predecessor} to frame {start}, which "
                    "is not the very next global index. A boundary's start is the frame "
                    "immediately after the one the capture paused at, with no gap between them."
                )
            recovery_limitations: tuple[str, ...] = ()
        else:
            if pending.loss.last_aligned_frame_index < 0:
                raise BridgeRecordUnusable(
                    f"{segment.name} claims to recover a loss whose record says nothing in this "
                    "capture ever aligned. There is no accepted model such a segment could have "
                    "been placed against, so the claim is refused rather than believed."
                )
            if start != pending.loss.triggering_frame_index + 1:
                raise BridgeRecordUnusable(
                    f"{segment.name} recovers a loss that ended at frame "
                    f"{pending.loss.triggering_frame_index} and starts at frame {start}. A "
                    "recovery continues at the very next global index, so that the frames it "
                    "leaves out are exactly the ones the loss record named and no others."
                )
            _check_lost_run(pending, what=segment.name)
            recovery_limitations = _recovery_binding(
                record,
                pending=pending,
                predecessor_index=predecessor,
                start_index=start,
                what=segment.name,
            )
            recoveries.append(
                RecoveredLoss(
                    run=pending.run,
                    record=pending.record,
                    loss=pending.loss,
                    segment=segment.name,
                    excluded=pending.excluded,
                )
            )
            pending = None
        transform = _rigid_from(record.get("bridge_transform"), what=f"{segment.name}'s transform")
        evidence = _evidence_from(record, batch_count=len(batch), what=f"{segment.name}")
        segment_limitations = _limitations_from(record, what=f"{segment.name}")
        bridges.append(
            BridgeConstraint(
                segment=segment.name,
                predecessor_index=predecessor,
                start_index=start,
                transform=transform,
                evidence=evidence,
                # The recovery's own limitations travel with the boundary, not
                # merely with the capture: whatever registers this seam is what
                # has to know that frames were left out of it.
                limitations=segment_limitations + recovery_limitations,
            )
        )
        segment_loss = read_tracking_loss(segment)
        valid, failed = _run_after_loss([*batch, *later], segment_loss)
        frames.extend(valid)
        if segment_loss is not None:
            pending = _PendingLoss(
                loss=segment_loss,
                directory=segment,
                run=segment.name,
                excluded=tuple(failed),
                prefix_end=valid[-1].index if valid else None,
                batch_end=batch[-1].index,
            )

    if pending is not None:
        raise TrackingLossUnresolved(pending.refusal(directory, None))

    excluded = [frame for recovery in recoveries for frame in recovery.excluded]
    if bridges:
        # A segmented capture is one consecutive run END TO END, root frames
        # included - and the frames a recovery excluded are part of that run,
        # because they were written and are still there. The per-segment check
        # above catches a gap after a boundary and the predecessor/start checks
        # catch a gap across one; this catches the remaining case, a gap among
        # the pre-pause root frames, which the writer that produced the bridges
        # cannot have made either. A legacy capture reaches none of this: with
        # no bridges there is nothing here to run, and its gaps read back
        # exactly as they always have.
        run = sorted([frame.index for frame in frames] + [frame.index for frame in excluded])
        if run != list(range(run[0], run[0] + len(run))):
            missing = sorted(set(range(run[0], run[-1] + 1)) - set(run))
            raise CaptureSequenceUnreadable(
                f"{directory} carries boundary evidence and is missing frame(s) "
                f"{missing}. Once a capture has been repositioned its frames are one "
                "consecutive global run through every segment; a gap is a file that has gone "
                "missing, and reading over it would register the frames either side of it as "
                "though they were adjacent."
            )
        pauses = len(bridges) - len(recoveries)
        if pauses:
            limitations.append(
                f"This capture was repositioned {pauses} time(s) mid-sweep. At each boundary "
                "the relationship between the last frame before the pause and the first frame "
                "after it comes from a recorded global relocalization, not from aligning that "
                "pair - the two frames are not a small step apart and fitting them would "
                "produce a confident, wrong answer."
            )
    if recoveries:
        indices = ", ".join(str(frame.index) for frame in excluded)
        limitations.append(
            f"This capture lost tracking {len(recoveries)} time(s) and recovered each time by "
            f"a recorded global relocalization against the model already accepted. "
            f"{len(excluded)} frame(s) - {indices} - were written while tracking was lost. They "
            "are still on disk, no pose was ever accepted for them, and they are deliberately "
            "NOT part of this sequence."
        )
        limitations.append(
            "At each recovered seam the relationship between the last frame before the loss "
            "and the first frame after it comes from that relocalization and not from aligning "
            "the pair, which are not a small step apart. Each recovery is bound to the loss "
            "record it closes by that record's own bytes."
        )
    return CaptureSequence(
        directory=directory,
        frames=tuple(frames),
        bridges=tuple(bridges),
        limitations=tuple(limitations),
        recoveries=tuple(recoveries),
    )


def next_frame_index(directory: Path) -> int:
    """One past the highest frame index anywhere in ``directory``.

    Used by a capture resuming into a folder rather than by a reader, and
    deliberately tolerant where :func:`read_capture_sequence` is strict: it is
    answering "what number is free", and a name it cannot parse is a name it
    simply does not count.
    """
    highest = -1
    for path in [*directory.glob(FRAME_GLOB), *directory.glob(f"{SEGMENT_GLOB}/{FRAME_GLOB}")]:
        index = _frame_index(path)
        if index is not None:
            highest = max(highest, index)
    return highest + 1


# --- writing ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentPublication:
    """One accepted pause boundary, as it now exists on disk."""

    directory: Path
    record: Path
    frames: tuple[Path, ...]
    predecessor_index: int
    start_index: int
    transform: Rigid


class _RenameAt2Unavailable(Exception):
    """Internal signal only: the syscall could not be used, not that it
    refused. A caller that catches this has no atomic primitive and refuses
    the publication; a caller that gets anything else - including
    ``FileExistsError`` - is looking at the rename's own real answer, which is
    a taken destination rather than a missing primitive.
    """


#: Linux ``RENAME_NOREPLACE``, from ``<linux/fs.h>``. The standard library
#: exposes no wrapper for ``renameat2``, so the flag is named here rather than
#: imported.
_RENAME_NOREPLACE: Final = 1

#: ``AT_FDCWD``, so a plain path (already resolved relative to this process's
#: current directory by the caller building it from ``capture_dir``) can be
#: passed to the ``*at`` syscall without opening a directory fd for it.
_AT_FDCWD: Final = -100


def _renameat2_noreplace(src: Path, dst: Path) -> None:
    """One attempt at the raw ``renameat2(..., RENAME_NOREPLACE)`` syscall.

    Raises :class:`_RenameAt2Unavailable` only when the syscall itself cannot
    be used here - the symbol is missing, or the kernel does not understand
    the flag (``ENOSYS``/``EINVAL``) - so the caller knows there is no atomic
    no-replace primitive to publish through and must refuse. Any other
    outcome, success or a real ``OSError`` (``FileExistsError`` when ``dst``
    is taken, in particular), is the rename's genuine answer.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        renameat2 = libc.renameat2
    except (OSError, AttributeError) as exc:
        raise _RenameAt2Unavailable(str(exc)) from exc
    result = renameat2(
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(src)),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(dst)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result == 0:
        return
    err = ctypes.get_errno()
    if err in (errno_mod.ENOSYS, errno_mod.EINVAL):
        raise _RenameAt2Unavailable(os.strerror(err))
    raise OSError(err, os.strerror(err), str(src))


def _no_replace_rename(src: Path, dst: Path) -> None:
    """Rename ``src`` onto ``dst``, refusing outright if ``dst`` names anything.

    Plain ``os.rename`` is not enough to publish a segment: POSIX ``rename()``
    silently REPLACES an existing EMPTY destination directory, and a name a
    reserved-then-abandoned publication left behind, or one an operator or
    another process placed there, is exactly the kind of empty directory that
    must never be treated as "free". ``renameat2(RENAME_NOREPLACE)`` (Linux,
    called as a raw syscall through :mod:`ctypes` because the standard library
    wraps ``rename`` but not this flag) makes the check and the rename one
    atomic kernel call: there is no instant at which a reader could observe
    the destination name as both absent and about to exist, and it refuses on
    ``dst`` regardless of what kind of thing is there - file, directory
    (empty or not), or symlink.

    If ``renameat2`` genuinely is not available on this kernel or libc there
    is NO fallback and the publication is refused. There used to be one -
    ``lstat`` then ``rename`` - and it was exactly the check-then-use race
    this function exists to close: a name created in the gap between the two
    calls is a name the following ``rename`` silently replaces, and an empty
    directory is precisely what a concurrent publication or an operator leaves
    there. "Almost atomic" is not a weaker version of this guarantee, it is
    the absence of it, so an unavailable primitive costs the operator one
    boundary rather than costing them the frames a replaced directory held.
    """
    try:
        _renameat2_noreplace(src, dst)
    except _RenameAt2Unavailable as exc:
        raise SegmentPublicationRefused(
            f"this kernel or libc offers no atomic no-replace rename ({exc}), and a segment is "
            "never published through a check-then-rename that could replace whatever appears "
            f"at {dst} in between. Nothing was published."
        ) from exc


def _publish_staging(capture_dir: Path, staging: Path) -> Path:
    """Make one fully-populated staging directory visible under a free name.

    The name is chosen and the directory is made to exist under it in the
    same no-replace rename, so a ``segment-NNNN`` name never exists - not
    even for an instant - before the directory behind it is the verified,
    fully written one. A taken name is stepped over rather than adopted, the
    same discipline :func:`scanner3d.reconstruction_export._reserve_output_dir`
    uses for reserving a name up front, applied here to publishing the
    finished result instead of to reserving an empty placeholder for it.
    """
    for attempt in range(1, SEGMENT_ATTEMPTS + 1):
        candidate = capture_dir / SEGMENT_TEMPLATE.format(index=attempt)
        try:
            _no_replace_rename(staging, candidate)
        except FileExistsError:
            continue
        return candidate
    raise SegmentPublicationRefused(
        f"every one of {SEGMENT_ATTEMPTS} segment names under {capture_dir} is taken; nothing "
        "was published and no earlier segment was written into."
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create ``path`` exclusively and write all of ``payload``, or leave nothing.

    ``O_EXCL | O_NOFOLLOW``, a loop resuming from what the kernel actually took,
    an fsync before this claims the file exists, and an unlink of anything
    partial - the same discipline every other writer in this repository uses,
    because a truncated raw frame is not a shorter frame, it is one whose digest
    will not match and whose whole segment is then refused.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise OSError(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
        os.fsync(fd)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    finally:
        os.close(fd)


def publish_segment(
    capture_dir: Path,
    *,
    frames: Sequence[tuple[int, bytes]],
    predecessor_index: int,
    transform: Rigid,
    evidence: Mapping[str, object],
    limitations: Iterable[str] = (),
    recovery: Mapping[str, object] | None = None,
    recovered: _PendingLoss | None = None,
) -> SegmentPublication:
    """Make one accepted pause boundary and its first batch visible, at once.

    ``frames`` are ``(global index, raw bytes)`` in capture order - the frames
    buffered while the resume was still tentative, which have never been on disk
    and never been counted. ``transform`` carries the SEGMENT-START frame's
    coordinates into the PREDECESSOR frame's, which is the direction a
    sequential pose chain composes.

    Everything is written into a hidden staging directory, read back and
    matched against the digests the record claims, and only then made visible
    under a ``segment-NNNN`` name by one no-replace rename (see
    :func:`_no_replace_rename`) that refuses rather than ever replacing
    whatever, if anything, already holds that name. So the segment either
    appears whole - record and every frame together, under a name nothing else
    held a moment before - or does not appear at all, and a run interrupted
    anywhere in here leaves the capture directory holding exactly the ordinary
    frames it held before, with no half-made ``segment-*`` name left behind.
    """
    capture_dir = Path(capture_dir)
    if not frames:
        raise SegmentPublicationRefused("a boundary publishes at least one frame; none were given")
    # Imported here rather than at module scope for the same reason
    # ``_batch_frames`` does: relocalize imports pipeline, which imports this
    # module for BridgeBoundary, so importing relocalize up here would cycle.
    from scanner3d.relocalize import MIN_TENTATIVE_FRAMES

    if len(frames) < MIN_TENTATIVE_FRAMES:
        raise SegmentPublicationRefused(
            f"the batch offers {len(frames)} frame(s); a relocalization is never accepted from "
            f"fewer than {MIN_TENTATIVE_FRAMES} tentative frames, so publishing fewer would "
            "write a record no reader could accept back"
        )
    indices = [index for index, _payload in frames]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise SegmentPublicationRefused(
            f"the batch offered indices {indices}, which are not one consecutive run"
        )
    if recovery is None and predecessor_index + 1 != indices[0]:
        # A PAUSE boundary's start is the very next global index. A RECOVERY's
        # is not, and that is the whole difference between them: the frames
        # written while tracking was lost sit in between, they stay on disk as
        # evidence, and they stay out of the sequence. Which frames those are is
        # checked below against the loss record itself rather than inferred from
        # the size of the gap.
        raise SegmentPublicationRefused(
            f"the batch starts at frame {indices[0]} and its predecessor is frame "
            f"{predecessor_index}; a boundary's start is the very next global index, with no "
            "gap between them"
        )

    artifacts = [
        {
            "file": FRAME_TEMPLATE.format(index=index),
            "frame_index": index,
            "byte_count": len(payload),
            "sha256": sha256_of(payload),
        }
        for index, payload in frames
    ]
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "predecessor_frame_index": predecessor_index,
        "segment_start_frame_index": indices[0],
        "frames": artifacts,
        "frame_order": "the batch is one consecutive run of global indices, in capture order",
        "bridge_transform": transform.as_dict(),
        "bridge_transform_convention": (
            "carries the segment-start frame's coordinates into the predecessor frame's, so a "
            "sequential pose chain composes it exactly where it would compose an adjacent fit"
        ),
        "evidence": dict(evidence),
        "limitations": list(limitations),
    }
    if recovery is not None:
        record[RECOVERY_RECORD_KEY] = dict(recovery)
    # The reader's own checks, run on the record BEFORE anything exists on
    # disk. A publication whose evidence its own reader would refuse is not a
    # boundary with a cosmetic flaw - it is frames that have left the tentative
    # buffer, been counted, and can never be read back, so the capture would be
    # unreadable as a whole from that moment on. Refusing here leaves the
    # segment tentative and the capture directory untouched, which is a state
    # the operator can still act on.
    try:
        _evidence_from(record, batch_count=len(frames), what="the boundary about to be published")
        _limitations_from(record, what="the boundary about to be published")
        if recovery is not None:
            # The binding, checked by the same functions that will check it on
            # the way back in, against the same loss record and the same
            # excluded frames it names. A recovery this reader would refuse is
            # worse than a pause boundary it would refuse: it would leave a
            # capture holding a loss that LOOKS closed and is not.
            if recovered is None:
                raise BridgeRecordUnusable(
                    "a recovery is published against the loss it closes, and none was given"
                )
            _check_lost_run(recovered, what="the recovery about to be published")
            _recovery_binding(
                record,
                pending=recovered,
                predecessor_index=predecessor_index,
                start_index=indices[0],
                what="the recovery about to be published",
            )
    except BridgeRecordUnusable as exc:
        raise SegmentPublicationRefused(
            f"the evidence offered for this boundary is not one this reader would accept back "
            f"({exc}); nothing was staged and nothing was published"
        ) from exc
    payload = json.dumps(record, indent=2, allow_nan=False).encode("utf-8")

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(dir=capture_dir, prefix=STAGING_PREFIX))
        # ``mkdtemp`` always makes 0o700, which is right for something private
        # and half-written and wrong for a published result; a renamed directory
        # carries the SOURCE's permissions. Suppressed rather than raised for
        # the same reason the export does it: a filesystem that will not take a
        # mode must not turn a correct publication into a failed one.
        with contextlib.suppress(OSError):
            staging.chmod(stat.S_IMODE(capture_dir.stat().st_mode))
        for index, raw in frames:
            _write_exclusive(staging / FRAME_TEMPLATE.format(index=index), raw)
        _write_exclusive(staging / BRIDGE_RECORD_NAME, payload)
        # Read back from disk, not from what was just handed in. The whole value
        # of the record is that a later reader can trust it without redoing the
        # capture, and that is only true if it was checked against the bytes.
        for artifact in artifacts:
            path = staging / str(artifact["file"])
            written = path.read_bytes()
            if len(written) != artifact["byte_count"] or sha256_of(written) != artifact["sha256"]:
                raise SegmentPublicationRefused(
                    f"{path.name} did not read back as the record describes it; nothing was "
                    "published."
                )
        # The name is chosen HERE, at the moment staging is fully populated and
        # verified, by one no-replace rename - never earlier. Until this call
        # returns, no ``segment-*`` name exists for this publication at all, so
        # there is nothing for a reader to see half-made, and no name for a
        # hard interruption to leave behind.
        directory = _publish_staging(capture_dir, staging)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return SegmentPublication(
        directory=directory,
        record=directory / BRIDGE_RECORD_NAME,
        frames=tuple(directory / FRAME_TEMPLATE.format(index=index) for index, _ in frames),
        predecessor_index=predecessor_index,
        start_index=indices[0],
        transform=transform,
    )


# --- tracking loss ------------------------------------------------------------
#
# A pause boundary is a break the operator ASKED for and the scan then proved it
# had crossed. A tracking loss is the other kind: a break nobody asked for and
# nothing has crossed. The capture of 2026-08-28 is the whole reason this exists
# - live registration failed on frame 53 and never recovered, the capture went
# on writing well over a hundred more frames, and the model on screen simply
# stopped changing. Every one of those frames looked, on disk, exactly like a
# frame that had aligned.
#
# So the loss is written down, once, and this reader refuses the whole directory
# afterwards. Containment, not recovery: nothing here estimates a pose, relaxes
# a gate or joins the two halves. The valid prefix and the failure evidence both
# stay on disk, and what is refused is the claim that they are ONE sequence.

#: How many ordinary live-alignment failures IN A ROW mean tracking has been
#: lost. Three, and consecutive: one failure is a bad frame, two is a bad
#: moment, and three in a row with nothing between them is the scan no longer
#: knowing where the object is. A single success anywhere in the run resets it,
#: which is what keeps an occasional dropout from ending a healthy capture.
#:
#: Deliberately NOT the cumulative failure count the live log already keeps.
#: That number answers "how has this capture been going" and rises on a scan
#: that is working; this one answers "is the chain still connected", and the
#: two must not be the same variable.
TRACKING_LOSS_STREAK: Final = 3

#: The record one lost capture publishes, at the capture root.
TRACKING_LOSS_RECORD_NAME: Final = "tracking-loss.json"

#: What the record's ``kind`` says, so a file found loose is identifiable, and
#: which schema this reader understands. Versioned separately from
#: :data:`SCHEMA_VERSION`: a bridge record and a loss record describe different
#: things and there is no reason one should drag the other's version along.
TRACKING_LOSS_KIND: Final = "scanner3d.tracking-loss"
TRACKING_LOSS_SCHEMA_VERSION: Final = 1

#: Prefix of the hidden directory the record is staged in before it is renamed
#: into place. Leading dot for the same reason :data:`STAGING_PREFIX` has one.
TRACKING_LOSS_STAGING_PREFIX: Final = ".tracking-loss-staging-"

#: Longest failure message kept, per failure. A record is read by a person and
#: by a reader; a driver or backend exception carrying a whole array repr would
#: make it neither. Truncation is marked in the text so a reader can see that
#: is what happened.
TRACKING_LOSS_MESSAGE_LIMIT: Final = 400

#: What a truncated message ends with, and therefore the longest any text a
#: record may carry can be. The reader enforces this bound rather than trusting
#: it: a record is read by a person, and an unbounded string in a file that
#: exists to be interpretable is a record that may not be.
TRACKING_LOSS_TRUNCATION_MARK: Final = " [truncated]"
TRACKING_LOSS_TEXT_LIMIT: Final = TRACKING_LOSS_MESSAGE_LIMIT + len(TRACKING_LOSS_TRUNCATION_MARK)

#: How many limitations one record may state. Bounded for the same reason the
#: text is: the mandatory one below is the content of the refusal, and a list
#: long enough to bury it is a list nothing reads to the end of.
TRACKING_LOSS_LIMITATION_COUNT: Final = 16

#: ``failure_type`` for the two ways a selected frame can fail to be placed
#: WITHOUT alignment ever being attempted on it. Named rather than borrowed
#: from an exception class, because neither of them is one: the first is a
#: frame that deprojected to nothing, and the second is the preparation of
#: that cloud raising before the aligner ever saw it. Both leave the model
#: exactly as stationary as a failed alignment does, so both count towards the
#: same streak - but a record that called them ``RegistrationError`` would be
#: telling a later reader that alignment was tried and failed, which is not
#: what happened.
TRACKING_INPUT_EMPTY_TYPE: Final = "EmptyTrackingInput"
TRACKING_INPUT_UNUSABLE_TYPE: Final = "UnusableTrackingInput"

#: Where the classification came from, stated in the record itself. The scanner
#: reported nothing: no projector, camera or object told this capture it was
#: lost, and a record that left that unsaid would be read later as a hardware
#: fault it never observed.
TRACKING_LOSS_CLASSIFIED_BY: Final = (
    "scanner3d live capture, from consecutive failures of adjacent point-to-plane "
    "alignment during the scan. The projector, the camera and the object reported "
    "nothing; no hardware or object measurement contributed to this classification."
)

#: The one limitation every schema-1 record MUST state, checked on the way in
#: and on the way out. It is not decoration: it is the entire reason the reader
#: refuses the directory, and a record that had dropped it would be a record
#: saying a loss happened without saying what is missing because of it.
TRACKING_LOSS_NO_BRIDGE_LIMITATION: Final = (
    "No pose bridges the failed continuation. Nothing relates the frames after the "
    "last aligned one to the frames before it, and no transform was estimated, "
    "interpolated or guessed to stand in for one."
)

#: Stated on every record, because they are true of every one of them and an
#: operator reading a record months later has only what it says.
TRACKING_LOSS_LIMITATIONS: Final = (
    TRACKING_LOSS_NO_BRIDGE_LIMITATION,
    "The frames written before the loss are untouched and remain on disk. They are "
    "evidence and a valid prefix, not an accepted continuation of one another "
    "across the loss.",
    "This record says tracking was lost. It does not say why the geometry stopped "
    "supporting alignment, and nothing here attempts to recover from it.",
)


@dataclass(frozen=True, slots=True)
class AlignmentFailure:
    """One live-alignment failure, as the capture saw it.

    ``failure_type`` is the exception's class name - or one of
    :data:`TRACKING_INPUT_EMPTY_TYPE` and :data:`TRACKING_INPUT_UNUSABLE_TYPE`
    for the two ways a frame fails to be placed before alignment is even
    attempted - and ``message`` is its text, kept apart so a later reader can
    group by kind without parsing prose. Both are what the LIVE path observed -
    the same frames re-read offline may well fail differently, or not at all,
    and this record never claims otherwise. Both are non-blank and bounded by
    :data:`TRACKING_LOSS_TEXT_LIMIT`, enforced by the reader rather than
    assumed of the writer.
    """

    frame_index: int
    failure_type: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "failure_type": self.failure_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class TrackingLoss:
    """One capture's durable statement that it stopped knowing where it was."""

    threshold: int
    #: Global index of the last frame that both reached disk and aligned, or
    #: ``-1`` for a capture that lost tracking before anything aligned at all.
    last_aligned_frame_index: int
    first_failed_frame_index: int
    #: The failure that reached the threshold, and therefore the frame at which
    #: the capture stopped adding to the model.
    triggering_frame_index: int
    failures: tuple[AlignmentFailure, ...]
    limitations: tuple[str, ...]
    classified_by: str

    def refusal(self, directory: Path) -> str:
        """What a caller asking for this directory as a sequence is told."""
        prefix = (
            f"frames up to and including frame {self.last_aligned_frame_index}"
            if self.last_aligned_frame_index >= 0
            else "no frames at all"
        )
        return (
            f"{directory} recorded a tracking loss and is refused as one reconstructable "
            f"sequence. Every one of the {self.threshold} consecutive frames "
            f"{self.first_failed_frame_index} through {self.triggering_frame_index} failed to be "
            "placed against the frame before it, and the scan stopped knowing where the object "
            f"was. The valid prefix is {prefix}, and "
            f"the failure evidence is in {TRACKING_LOSS_RECORD_NAME}; both are still on disk and "
            "nothing has been deleted. What does not exist is any pose relating the later frames "
            "to that prefix, so registering this directory as one sweep would fit a pair that "
            "was never a small step apart and produce a confident, wrong model. A loss is "
            "closed only by an accepted relocalization published as a recovery segment bound "
            "to this record; this capture holds none, and nothing here will estimate one."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRACKING_LOSS_SCHEMA_VERSION,
            "kind": TRACKING_LOSS_KIND,
            "threshold": self.threshold,
            "last_aligned_frame_index": self.last_aligned_frame_index,
            "first_failed_frame_index": self.first_failed_frame_index,
            "triggering_frame_index": self.triggering_frame_index,
            "failures": [failure.as_dict() for failure in self.failures],
            "classified_by": self.classified_by,
            "limitations": list(self.limitations),
        }


def _loss_int(record: Mapping[str, object], key: str, *, what: str) -> int:
    """One integer field of a loss record, or a refusal. ``bool`` is not one."""
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrackingLossRecordUnusable(
            f"{what} states {key} as {value!r}, which is not a frame index this reader will "
            "accept. A loss record whose own numbers cannot be read cannot be checked against "
            "itself, and is refused rather than half-believed."
        )
    return value


def _loss_failures(
    record: Mapping[str, object], *, threshold: int, what: str
) -> tuple[AlignmentFailure, ...]:
    """The captured failures, checked for shape and count rather than trusted.

    Exactly ``threshold`` of them, on numerically CONSECUTIVE frames, each
    carrying non-blank bounded text. A record naming fewer is one whose own
    threshold was not reached by what it lists; a record naming them out of
    order, or with a frame missing from the middle, is not describing a streak
    at all. Every selected frame a live capture does not accept is recorded as
    a failure, so a gap in the run means the record describes something other
    than what this writer produces - and a record that cannot be trusted about
    its own evidence cannot be trusted about where the valid prefix ends.
    """
    listed = record.get("failures")
    if not isinstance(listed, list):
        raise TrackingLossRecordUnusable(f"{what} carries no failures list.")
    if len(listed) != threshold:
        raise TrackingLossRecordUnusable(
            f"{what} states a threshold of {threshold} and lists {len(listed)} failure(s). The "
            "failures that reached the threshold are exactly what makes the refusal "
            "interpretable, and a record missing any of them is refused."
        )
    failures: list[AlignmentFailure] = []
    for entry in listed:
        if not isinstance(entry, Mapping):
            raise TrackingLossRecordUnusable(f"{what} has a failure entry that is not an object.")
        index = entry.get("frame_index")
        kind = entry.get("failure_type")
        message = entry.get("message")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(kind, str)
            or not isinstance(message, str)
        ):
            raise TrackingLossRecordUnusable(
                f"{what} has a failure entry missing a frame index, a type or a message."
            )
        for label, text in (("type", kind), ("message", message)):
            # Blank evidence is the shape of a record written by something that
            # had nothing to say, and the failure text is what makes the refusal
            # interpretable rather than merely obeyed.
            if not text.strip():
                raise TrackingLossRecordUnusable(
                    f"{what} has a failure entry for frame {index} whose {label} is blank. The "
                    "captured failure text is what a reader has months later, and a record that "
                    "carries none of it explains nothing."
                )
            if len(text) > TRACKING_LOSS_TEXT_LIMIT:
                raise TrackingLossRecordUnusable(
                    f"{what} has a failure entry for frame {index} whose {label} is "
                    f"{len(text)} characters, past the {TRACKING_LOSS_TEXT_LIMIT} this writer "
                    "ever produces. A record is read by a person, and this one was not written "
                    "by the capture that would be refused for it."
                )
        failures.append(AlignmentFailure(frame_index=index, failure_type=kind, message=message))
    indices = [failure.frame_index for failure in failures]
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise TrackingLossRecordUnusable(
            f"{what} lists failure frames {indices}, which are not one consecutive run of "
            f"frames; {expected} is what a streak beginning at frame {indices[0]} looks like. "
            "A live capture records every selected frame it does not accept, so a repeat, a "
            "frame out of order or a frame missing from the middle is not a streak this "
            "writer produced."
        )
    return tuple(failures)


def _tracking_loss_from(record: Mapping[str, object], *, what: str) -> TrackingLoss:
    """A :class:`TrackingLoss` out of a parsed record, or a refusal.

    Every check here is a way the record could contradict ITSELF, and each one
    is a refusal rather than a repair. There is no safe repair available: the
    record is the only thing that says which frames are the valid prefix, and a
    reader that guessed at that would be choosing where somebody's scan ends.
    """
    if record.get("kind") != TRACKING_LOSS_KIND:
        raise TrackingLossRecordUnusable(
            f"{what} says kind {record.get('kind')!r} rather than {TRACKING_LOSS_KIND!r}."
        )
    version = record.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != TRACKING_LOSS_SCHEMA_VERSION
    ):
        raise TrackingLossRecordUnusable(
            f"{what} states schema version {version!r}; this reader understands exactly version "
            f"{TRACKING_LOSS_SCHEMA_VERSION} and does not guess at an earlier or a later one. A "
            "capture that says it is lost in a dialect this cannot read is still refused."
        )
    threshold = _loss_int(record, "threshold", what=what)
    if threshold != TRACKING_LOSS_STREAK:
        raise TrackingLossRecordUnusable(
            f"{what} states a threshold of {threshold}. Schema "
            f"{TRACKING_LOSS_SCHEMA_VERSION} has exactly one threshold - "
            f"{TRACKING_LOSS_STREAK} consecutive failures - and a record claiming any other "
            "one was not written by the policy this reader enforces. Accepting it would let a "
            "record decide for itself how little evidence a refusal needs."
        )
    last_aligned = _loss_int(record, "last_aligned_frame_index", what=what)
    first_failed = _loss_int(record, "first_failed_frame_index", what=what)
    triggering = _loss_int(record, "triggering_frame_index", what=what)
    if last_aligned < -1:
        raise TrackingLossRecordUnusable(
            f"{what} names {last_aligned} as its last aligned frame; -1 is how a capture that "
            "never aligned anything says so, and there is nothing below that."
        )
    if not last_aligned < first_failed <= triggering:
        raise TrackingLossRecordUnusable(
            f"{what} places its last aligned frame at {last_aligned}, its first failure at "
            f"{first_failed} and the failure that reached the threshold at {triggering}. Those "
            "cannot all be true at once: the failures follow the last accepted pose, and the "
            "streak ends at or after it begins."
        )
    failures = _loss_failures(record, threshold=threshold, what=what)
    if failures[0].frame_index != first_failed or failures[-1].frame_index != triggering:
        raise TrackingLossRecordUnusable(
            f"{what} says the streak ran from frame {first_failed} to frame {triggering} and "
            f"lists failures from frame {failures[0].frame_index} to frame "
            f"{failures[-1].frame_index}. A record that disagrees with its own evidence is "
            "refused."
        )
    classified_by = record.get("classified_by")
    if not isinstance(classified_by, str) or not classified_by.strip():
        raise TrackingLossRecordUnusable(
            f"{what} does not say what classified the loss. A record that leaves that unsaid is "
            "read later as a hardware fault it never observed."
        )
    if classified_by != TRACKING_LOSS_CLASSIFIED_BY:
        # Exact, and this is the one field where that is the point. Schema 1 has
        # a single publisher and a single provenance, and the danger is not a
        # missing sentence but a REPLACED one - a record saying the scanner, the
        # projector or the object reported the loss would be read as a hardware
        # fault nothing here ever observed. A reader cannot tell a rewritten
        # provenance from an honest one by inspecting its prose, so it accepts
        # only the statement this version writes.
        raise TrackingLossRecordUnusable(
            f"{what} states a provenance that is not the one schema "
            f"{TRACKING_LOSS_SCHEMA_VERSION} publishes. This classification comes from "
            "consecutive live alignment failures and from nothing else, and a record claiming "
            "any other source - the projector, the camera, the object - is refused rather than "
            "believed."
        )
    limitations = record.get("limitations")
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise TrackingLossRecordUnusable(f"{what} has no limitations list of plain strings.")
    if not limitations or len(limitations) > TRACKING_LOSS_LIMITATION_COUNT:
        raise TrackingLossRecordUnusable(
            f"{what} states {len(limitations)} limitations, and a record states between 1 and "
            f"{TRACKING_LOSS_LIMITATION_COUNT} of them."
        )
    if any(not item.strip() or len(item) > TRACKING_LOSS_TEXT_LIMIT for item in limitations):
        raise TrackingLossRecordUnusable(
            f"{what} states a limitation that is blank or longer than the "
            f"{TRACKING_LOSS_TEXT_LIMIT} characters this writer ever produces."
        )
    if TRACKING_LOSS_NO_BRIDGE_LIMITATION not in limitations:
        raise TrackingLossRecordUnusable(
            f"{what} does not state that no pose bridges the failed continuation. That "
            "limitation is the whole content of the refusal - it is WHY the directory is not "
            "one sequence - and a loss record without it is refused rather than read as a "
            "loss with nothing missing."
        )
    return TrackingLoss(
        threshold=threshold,
        last_aligned_frame_index=last_aligned,
        first_failed_frame_index=first_failed,
        triggering_frame_index=triggering,
        failures=failures,
        limitations=tuple(limitations),
        classified_by=classified_by,
    )


def _read_tracking_loss(directory: Path) -> tuple[TrackingLoss, bytes] | None:
    """The record AND the exact bytes it was read from, or ``None``.

    One read, two answers, because a caller that has to hash the record it is
    binding to must hash the same bytes it took its meaning from. Reading the
    file twice - once for the meaning and once for the digest - is a window in
    which the two can disagree, and a recovery whose digest and semantics come
    from different records is exactly what :func:`publish_recovery_segment`
    exists to refuse.
    """
    path = Path(directory) / TRACKING_LOSS_RECORD_NAME
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TrackingLossRecordUnusable(
            f"the tracking-loss record at {path} cannot be examined: {exc}"
        ) from exc
    if stat.S_ISLNK(info.st_mode):
        raise TrackingLossRecordUnusable(
            f"{path} is a symbolic link. A capture's own record of its own loss is read as bytes "
            "inside the capture directory, and a link is not followed out of it."
        )
    if not stat.S_ISREG(info.st_mode):
        raise TrackingLossRecordUnusable(f"{path} is not a regular file.")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrackingLossRecordUnusable(f"{path} cannot be read: {exc}") from exc
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackingLossRecordUnusable(
            f"{path} is not readable JSON: {exc}. A capture carrying an unparseable statement "
            "that it is lost is refused; it is never read as a capture that is fine."
        ) from exc
    if not isinstance(record, dict):
        raise TrackingLossRecordUnusable(f"{path} does not hold a JSON object.")
    return _tracking_loss_from(record, what=str(path)), payload


def read_tracking_loss(directory: Path) -> TrackingLoss | None:
    """The capture's tracking-loss record, or ``None`` if it has none.

    ``None`` means one thing only: the record is not there. Everything else -
    a symlink standing in for it, a directory, an unreadable file, JSON that
    does not parse, a schema this does not know, a record that contradicts
    itself - raises :class:`TrackingLossRecordUnusable`, because a capture
    holding something loss-shaped that cannot be read must never be handed over
    as though it held nothing. That fallback is exactly how the silent failure
    this whole mechanism prevents would come back.
    """
    current = _read_tracking_loss(directory)
    return None if current is None else current[0]


def _loss_text(text: str) -> str:
    """One bounded string for the record, marked if it had to be cut.

    Applied to the failure TYPE as well as the message. A type is normally an
    exception class name and nowhere near the bound, but the reader enforces the
    bound on both, and a writer that enforced it on only one could produce a
    record it would then refuse to read back.
    """
    if len(text) <= TRACKING_LOSS_MESSAGE_LIMIT:
        return text
    return text[:TRACKING_LOSS_MESSAGE_LIMIT] + TRACKING_LOSS_TRUNCATION_MARK


def publish_tracking_loss(
    capture_dir: Path,
    *,
    last_aligned_frame_index: int,
    failures: Sequence[AlignmentFailure],
    threshold: int = TRACKING_LOSS_STREAK,
    limitations: Iterable[str] = TRACKING_LOSS_LIMITATIONS,
) -> Path:
    """Write one capture's loss record, completely or not at all.

    Staged into a hidden directory, fsynced, then moved onto its final name by
    the same no-replace rename a segment publishes through - so the record
    either exists whole or does not exist, and an existing one is never
    overwritten. A capture loses tracking once; a second record would be a
    second answer to a question that already has one.

    ``threshold`` and ``limitations`` are arguments so that this reads as one
    statement rather than as a constant repeated in two places, but they are not
    free: the reader accepts exactly :data:`TRACKING_LOSS_STREAK` and requires
    :data:`TRACKING_LOSS_NO_BRIDGE_LIMITATION`, and this refuses before writing
    anything if what it was handed would not read back.

    Raises :class:`TrackingLossPublicationRefused` for every failure, including
    a name already taken. The caller's contract is that a loss it cannot make
    durable fails the capture: an unmarked capture that has silently stopped
    tracking is the state this exists to prevent, and it is strictly worse than
    a capture that ended with an error the operator can see.
    """
    capture_dir = Path(capture_dir)
    trimmed = tuple(
        AlignmentFailure(
            frame_index=failure.frame_index,
            failure_type=_loss_text(failure.failure_type),
            message=_loss_text(failure.message),
        )
        for failure in failures
    )
    if not trimmed:
        raise TrackingLossPublicationRefused(
            "a tracking loss is published with the failures that reached its threshold; none "
            "were given, and a record with no evidence is one nothing can interpret"
        )
    loss = TrackingLoss(
        threshold=threshold,
        last_aligned_frame_index=last_aligned_frame_index,
        first_failed_frame_index=trimmed[0].frame_index,
        triggering_frame_index=trimmed[-1].frame_index,
        failures=trimmed,
        limitations=tuple(limitations),
        classified_by=TRACKING_LOSS_CLASSIFIED_BY,
    )
    record = loss.as_dict()
    # The reader's own checks, run before anything exists on disk, for the same
    # reason ``publish_segment`` runs them: a record this reader would refuse is
    # not a cosmetic flaw, it is a capture that is refused for the WRONG reason
    # forever after. Refusing here fails the capture with the real one instead.
    try:
        _tracking_loss_from(record, what="the tracking loss about to be published")
    except TrackingLossRecordUnusable as exc:
        raise TrackingLossPublicationRefused(
            f"the loss about to be recorded is not one this reader would accept back ({exc}); "
            "nothing was written"
        ) from exc
    try:
        payload = json.dumps(record, indent=2, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - shape already checked
        raise TrackingLossPublicationRefused(
            f"the loss record cannot be serialised: {exc}"
        ) from exc

    destination = capture_dir / TRACKING_LOSS_RECORD_NAME
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(dir=capture_dir, prefix=TRACKING_LOSS_STAGING_PREFIX))
        staged = staging / TRACKING_LOSS_RECORD_NAME
        _write_exclusive(staged, payload)
        with contextlib.suppress(OSError):
            staged.chmod(0o644)
        _no_replace_rename(staged, destination)
    except SegmentPublicationRefused as exc:
        raise TrackingLossPublicationRefused(
            f"the loss record could not be published atomically: {exc}"
        ) from exc
    except FileExistsError as exc:
        raise TrackingLossPublicationRefused(
            f"{destination} already exists. A capture loses tracking once, and an existing "
            "record is never overwritten - the one on disk is the one that stands."
        ) from exc
    except OSError as exc:
        raise TrackingLossPublicationRefused(
            f"{destination} could not be written: {exc}. Nothing partial was left behind."
        ) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return destination


# --- recovering from a tracking loss ------------------------------------------
#
# Task 020 contained a loss: the capture stopped, wrote one immutable record and
# this reader refused the folder. Containment is not the answer an operator
# wants, though - it is the answer that is honest while nothing better exists -
# and the better answer is the one a PAUSE boundary already gives: hold the
# frames back, find the accepted model again by global relocalization, and
# publish the batch and its evidence together or not at all.
#
# So a recovery is an ordinary ``segment-*`` directory. It carries everything a
# pause boundary carries - the batch, the transform, the relocalization
# evidence, the local chain - and one thing more: a ``recovery`` object that
# BINDS it to the exact loss it closes, by that record's location and its
# SHA-256, and that names the failed frames which stay on disk and stay OUT of
# the sequence.
#
# The binding is the whole of it. A bridge that merely happens to follow a loss
# record proves nothing: the frames between them are frames no pose was ever
# accepted for, and a reader that closed a loss on adjacency alone would hand
# back a sequence whose seam was decided by directory order. So the record must
# name the loss it closes, that name must hash to the bytes actually on disk,
# and every excluded frame must be there, be the length and the digest claimed,
# and be exactly the streak the loss record itself lists.

#: What a recovery record's ``kind`` says, and which dialect this reader knows.
#: Versioned apart from :data:`SCHEMA_VERSION` and
#: :data:`TRACKING_LOSS_SCHEMA_VERSION` for the same reason those are versioned
#: apart from each other: three different things, and no reason one should drag
#: another's version along.
RECOVERY_KIND: Final = "scanner3d.tracking-loss-recovery"
RECOVERY_SCHEMA_VERSION: Final = 1

#: Where the recovery object sits inside the ordinary bridge record. One record
#: per segment, still: a recovery IS a bridge, with a binding attached, and
#: splitting it into two files would create a moment where one exists and the
#: other does not.
RECOVERY_RECORD_KEY: Final = "recovery"

#: The two limitations every recovery record MUST state, checked on the way in
#: and on the way out. They are not decoration. The first is the fact an
#: operator counting files will otherwise get wrong - frames are on disk that
#: are deliberately not in the scan - and the second is what a reconstruction
#: must not forget when it reaches the seam.
RECOVERY_EXCLUDED_LIMITATION: Final = (
    "The frames the loss record names are still on disk and are NOT part of this "
    "sequence. They were written before tracking was declared lost, no pose was ever "
    "accepted for them, and none was estimated afterwards to bring them back in."
)
RECOVERY_RELOCALIZED_LIMITATION: Final = (
    "The frames after the loss were placed against the accepted model by a recorded "
    "global relocalization, not by aligning them to the frame before them. Nothing "
    "here recovers the poses the excluded frames would have had."
)
RECOVERY_LIMITATIONS: Final = (
    RECOVERY_EXCLUDED_LIMITATION,
    RECOVERY_RELOCALIZED_LIMITATION,
    "A recovery says where the scan picked the object up again. It does not say why "
    "tracking was lost, and the loss record beside it is still the evidence for that.",
)

#: How many limitations one recovery record may state, and how long each may be.
#: Bounded for the reason a loss record's are: the two mandatory ones above are
#: the content, and a list long enough to bury them is a list nothing reads to
#: the end of.
RECOVERY_LIMITATION_COUNT: Final = 16


@dataclass(frozen=True, slots=True)
class RecoveredLoss:
    """One tracking loss and the segment that closed it, as read back.

    Returned rather than merely checked, because "this capture lost tracking and
    recovered" is a fact about the data that a caller reporting on the capture
    has to be able to state. It is also the only place the EXCLUDED frames are
    named: they are on disk, they are not in :attr:`CaptureSequence.frames`, and
    a consumer counting files would otherwise be unable to explain the
    difference.
    """

    #: The run the loss was recorded in: ``None`` for the capture root, or a
    #: segment directory's name.
    run: str | None
    #: The loss record itself, inside that run.
    record: Path
    loss: TrackingLoss
    #: The recovery segment's directory name.
    segment: str
    #: The failed frames, still on disk and deliberately not in the sequence.
    excluded: tuple[SequenceFrame, ...]

    @property
    def excluded_indices(self) -> tuple[int, ...]:
        return tuple(frame.index for frame in self.excluded)


@dataclass(frozen=True, slots=True)
class _PendingLoss:
    """A run that ended in a loss, waiting for the segment that must close it."""

    loss: TrackingLoss
    directory: Path
    run: str | None
    #: The frames of that run after its valid prefix: the streak, still on disk.
    excluded: tuple[SequenceFrame, ...]
    #: The highest index the run's valid prefix reached, or ``None`` for a run
    #: that handed over nothing at all.
    prefix_end: int | None = None
    #: The last frame of the batch this run's own segment published, or ``None``
    #: for the capture root, which publishes no batch.
    batch_end: int | None = None

    @property
    def record(self) -> Path:
        return self.directory / TRACKING_LOSS_RECORD_NAME

    @property
    def where(self) -> str:
        return "the capture root" if self.run is None else self.run

    @property
    def location(self) -> str:
        """Where the record sits, relative to the capture directory.

        Rebuilt from the run this reader has already validated rather than read
        out of the record it is checking, which is what makes comparing the two
        worth anything - and is also why nothing outside the capture is ever
        opened on the strength of a string in a file.
        """
        if self.run is None:
            return TRACKING_LOSS_RECORD_NAME
        return f"{self.run}/{TRACKING_LOSS_RECORD_NAME}"

    def refusal(self, capture: Path, following: str | None) -> str:
        """What a caller is told when nothing closes this loss."""
        unresolved = (
            f"Nothing follows it that closes it: {following} is the next thing in this "
            "capture and it carries no recovery record binding that loss."
            if following is not None
            else "Nothing follows it at all."
        )
        return (
            f"{self.loss.refusal(capture)}\nThe loss was recorded in {self.where}. "
            f"{unresolved} A "
            "loss is closed only by an accepted relocalization that names this exact record "
            "and the frames it excludes; presence or order alone never closes one."
        )


def _run_after_loss(
    run_frames: Sequence[SequenceFrame], loss: TrackingLoss | None
) -> tuple[list[SequenceFrame], list[SequenceFrame]]:
    """Split one run at the last frame its loss record says was placed.

    The valid prefix, and everything after it. Nothing is checked here: a run
    with an unresolved loss is refused as a whole a moment later, and a run
    whose loss IS closed is checked in full by :func:`_check_lost_run` before
    the recovery that closes it is accepted. Splitting and validating are kept
    apart precisely so that a capture which merely recorded a loss and stopped -
    the only thing this could be before recovery existed - still refuses with
    the same refusal, in the same words, whatever else is odd about it.
    """
    if loss is None:
        return list(run_frames), []
    last = loss.last_aligned_frame_index
    return (
        [frame for frame in run_frames if frame.index <= last],
        [frame for frame in run_frames if frame.index > last],
    )


def _check_lost_run(pending: _PendingLoss, *, what: str) -> None:
    """Check a loss record against the frames of the run that published it.

    Only reached when something claims to CLOSE the loss, which is the moment it
    starts to matter: from here on the record decides which frames are in
    somebody's scan and which are evidence sitting beside it, and a record that
    cannot be checked against the run would be choosing that split from a file
    rather than from the data.

    Three things have to agree: the prefix ends exactly at the frame the record
    names, the frames after it are exactly the streak it lists - none missing,
    none extra, in order - and, in a segment, the loss begins after the batch
    that segment published rather than inside it, since every frame of a
    published batch was accepted by the relocalization that published it.
    """
    loss = pending.loss
    last = loss.last_aligned_frame_index
    if pending.prefix_end != last:
        held = "nothing" if pending.prefix_end is None else pending.prefix_end
        raise BridgeRecordUnusable(
            f"{what} closes a loss recorded in {pending.where} that names frame {last} as the "
            f"last frame that aligned, and the last frame at or below it in that run is {held}. "
            "A recovery is only meaningful against the frame the scan actually stopped at."
        )
    if pending.batch_end is not None and last < pending.batch_end:
        raise BridgeRecordUnusable(
            f"{what} closes a loss recorded in {pending.where} that places its last aligned "
            f"frame at {last}, inside the batch that run published, which ends at frame "
            f"{pending.batch_end}. Every frame of a published batch was accepted by the "
            "relocalization that published it, so a loss cannot begin before the end of one."
        )
    listed = [failure.frame_index for failure in loss.failures]
    present = [frame.index for frame in pending.excluded]
    if present != listed:
        raise BridgeRecordUnusable(
            f"{what} closes a loss recorded in {pending.where} that lists failed frames "
            f"{listed}, and the frames after that run's valid prefix are {present}. The failed "
            "frames are kept exactly where they were written, as evidence, so the two are the "
            "same list or the record is not describing this run."
        )


def _recovery_binding(
    record: Mapping[str, object],
    *,
    pending: _PendingLoss,
    predecessor_index: int,
    start_index: int,
    what: str,
) -> tuple[str, ...]:
    """Check that this segment closes THAT loss, and return what it states.

    Everything here is a way the recovery could be about something other than
    the loss immediately before it: a stale record copied from an earlier
    episode, a loss record edited after the fact, an excluded frame quietly
    swapped for a different one, indices that no longer agree with the bridge
    they sit beside. Each is a refusal, because the alternative is a sequence
    whose seam was decided by which directory happened to be next.
    """
    claim = record.get(RECOVERY_RECORD_KEY)
    if not isinstance(claim, Mapping):
        raise BridgeRecordUnusable(
            f"{what} carries a {RECOVERY_RECORD_KEY} entry that is not an object."
        )
    if claim.get("kind") != RECOVERY_KIND:
        raise BridgeRecordUnusable(
            f"{what}'s recovery says kind {claim.get('kind')!r} rather than {RECOVERY_KIND!r}."
        )
    version = claim.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != RECOVERY_SCHEMA_VERSION
    ):
        raise BridgeRecordUnusable(
            f"{what}'s recovery states schema version {version!r}; this reader understands "
            f"exactly version {RECOVERY_SCHEMA_VERSION} and does not guess at an earlier or a "
            "later one."
        )

    # The loss this claims to close, by where it is and by what it holds. The
    # path is COMPARED against the one rebuilt from the run, never used to open
    # anything, so a record naming somewhere else is refused rather than
    # followed there.
    bound = claim.get("loss_record")
    if not isinstance(bound, Mapping):
        raise BridgeRecordUnusable(
            f"{what}'s recovery names no loss_record object, so nothing in it says WHICH loss "
            "it closes."
        )
    location = bound.get("path")
    if location != pending.location:
        raise BridgeRecordUnusable(
            f"{what}'s recovery says it closes the loss recorded at {location!r}, and the loss "
            f"immediately before it in this capture is {pending.location!r}. A recovery closes "
            "the loss it follows or it closes nothing."
        )
    _plain_file(pending.record, what=f"the tracking-loss record {pending.location} closes by")
    try:
        payload = pending.record.read_bytes()
    except OSError as exc:
        raise BridgeRecordUnusable(f"{pending.record} cannot be read: {exc}") from exc
    byte_count = bound.get("byte_count")
    digest = bound.get("sha256")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count != len(payload)
    ):
        raise BridgeRecordUnusable(
            f"{what}'s recovery claims the loss record is {byte_count!r} bytes and "
            f"{pending.record} is {len(payload)}."
        )
    if not isinstance(digest, str) or sha256_of(payload) != digest:
        raise BridgeRecordUnusable(
            f"{what}'s recovery does not hash to the loss record beside it. The binding is by "
            "bytes on purpose: a loss record edited after a recovery was published would "
            "otherwise still look closed."
        )

    for key, expected in (
        ("predecessor_frame_index", predecessor_index),
        ("segment_start_frame_index", start_index),
    ):
        stated = claim.get(key)
        if not isinstance(stated, int) or isinstance(stated, bool) or stated != expected:
            raise BridgeRecordUnusable(
                f"{what}'s recovery states {key} as {stated!r} and the bridge beside it says "
                f"{expected}. A recovery that disagrees with its own bridge is refused."
            )

    _recovery_exclusions(claim, pending=pending, what=what)

    stated = claim.get("limitations")
    if not isinstance(stated, list) or any(not isinstance(item, str) for item in stated):
        raise BridgeRecordUnusable(f"{what}'s recovery has no limitations list of plain strings.")
    if not stated or len(stated) > RECOVERY_LIMITATION_COUNT:
        raise BridgeRecordUnusable(
            f"{what}'s recovery states {len(stated)} limitations, and a record states between 1 "
            f"and {RECOVERY_LIMITATION_COUNT} of them."
        )
    if any(not item.strip() or len(item) > TRACKING_LOSS_TEXT_LIMIT for item in stated):
        raise BridgeRecordUnusable(
            f"{what}'s recovery states a limitation that is blank or longer than the "
            f"{TRACKING_LOSS_TEXT_LIMIT} characters this writer ever produces."
        )
    for mandatory in (RECOVERY_EXCLUDED_LIMITATION, RECOVERY_RELOCALIZED_LIMITATION):
        if mandatory not in stated:
            raise BridgeRecordUnusable(
                f"{what}'s recovery does not state that {mandatory[:60]}... That limitation "
                "travels with the boundary into whatever reconstructs it, and a recovery "
                "without it is refused rather than read as one with nothing to declare."
            )
    return tuple(stated)


def _recovery_exclusions(claim: Mapping[str, object], *, pending: _PendingLoss, what: str) -> None:
    """The failed frames the recovery names, matched against the bytes on disk.

    They are the frames the sequence deliberately does not include, so this is
    the check that stops a recovery from quietly changing WHICH frames those
    are. Each is verified the same way a published batch is - the name rebuilt
    from the index, the length, the digest - because a frame that was swapped
    for another is exactly as invisible here as a swapped batch frame would be.
    """
    listed = claim.get("excluded_frames")
    if not isinstance(listed, list):
        raise BridgeRecordUnusable(
            f"{what}'s recovery names no excluded_frames list, so nothing in it says which "
            "frames stay out of the scan."
        )
    if len(listed) != len(pending.excluded):
        raise BridgeRecordUnusable(
            f"{what}'s recovery excludes {len(listed)} frame(s) and the loss it closes left "
            f"{len(pending.excluded)} on disk. Every frame written while tracking was lost is "
            "named, or the count of what is in the scan is wrong."
        )
    for entry, frame in zip(listed, pending.excluded, strict=True):
        if not isinstance(entry, Mapping):
            raise BridgeRecordUnusable(
                f"{what}'s recovery has an excluded frame entry that is not an object."
            )
        index = entry.get("frame_index")
        name = entry.get("file")
        byte_count = entry.get("byte_count")
        digest = entry.get("sha256")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(name, str)
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or not isinstance(digest, str)
        ):
            raise BridgeRecordUnusable(
                f"{what}'s recovery has an excluded frame entry missing a name, index, length "
                "or digest."
            )
        if index != frame.index:
            raise BridgeRecordUnusable(
                f"{what}'s recovery excludes frame {index} where the loss it closes failed on "
                f"frame {frame.index}. The excluded frames are the streak the loss record "
                "lists, in order."
            )
        expected = FRAME_TEMPLATE.format(index=index)
        if name != expected:
            raise BridgeRecordUnusable(
                f"{what}'s recovery names an excluded frame {name!r} for index {index}, and the "
                f"only name that index has is {expected!r}. Nothing outside the run is opened."
            )
        _plain_file(frame.path, what=f"excluded frame {index}")
        payload = frame.path.read_bytes()
        if len(payload) != byte_count:
            raise BridgeRecordUnusable(
                f"{frame.path} is {len(payload)} bytes and the recovery that excludes it claims "
                f"{byte_count}."
            )
        if sha256_of(payload) != digest:
            raise BridgeRecordUnusable(
                f"{frame.path} does not hash to the digest the recovery that excludes it "
                "claims. The excluded frames are evidence, and evidence that has changed is "
                "not evidence."
            )


def recovery_record(
    capture_dir: Path,
    *,
    loss_dir: Path,
    loss: TrackingLoss,
    predecessor_index: int,
    start_index: int,
    limitations: Iterable[str] = RECOVERY_LIMITATIONS,
    record_payload: bytes | None = None,
) -> dict[str, object]:
    """The binding a recovery segment publishes, measured off the disk.

    Built here rather than by the capture, because every number in it is a fact
    about files that already exist - the loss record's digest, each excluded
    frame's length and digest - and a caller that assembled it from what it
    believed it had written would be recording its own belief.

    ``record_payload`` is the loss record's bytes when the caller has ALREADY
    read them and taken ``loss`` from them, which is what
    :func:`publish_recovery_segment` does. Hashing those exact bytes rather than
    re-reading the file is the difference between a digest that describes the
    record this recovery means and one that describes whatever is at that path
    by the time the digest is taken.
    """
    loss_dir = Path(loss_dir)
    capture_dir = Path(capture_dir)
    record_path = loss_dir / TRACKING_LOSS_RECORD_NAME
    if record_payload is None:
        _plain_file(record_path, what="the tracking-loss record this recovery closes")
        payload = record_path.read_bytes()
    else:
        payload = record_payload
    location = (
        TRACKING_LOSS_RECORD_NAME
        if loss_dir == capture_dir
        else f"{loss_dir.name}/{TRACKING_LOSS_RECORD_NAME}"
    )
    excluded: list[dict[str, object]] = []
    for failure in loss.failures:
        frame = loss_dir / FRAME_TEMPLATE.format(index=failure.frame_index)
        _plain_file(frame, what=f"excluded frame {failure.frame_index}")
        raw = frame.read_bytes()
        excluded.append(
            {
                "file": frame.name,
                "frame_index": failure.frame_index,
                "byte_count": len(raw),
                "sha256": sha256_of(raw),
            }
        )
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "kind": RECOVERY_KIND,
        "loss_record": {
            "path": location,
            "byte_count": len(payload),
            "sha256": sha256_of(payload),
        },
        "predecessor_frame_index": predecessor_index,
        "segment_start_frame_index": start_index,
        "excluded_frames": excluded,
        "excluded_frame_policy": (
            "written before tracking was declared lost, kept exactly where they were as "
            "evidence, and deliberately not part of the sequence this capture hands over"
        ),
        "limitations": list(limitations),
    }


def publish_recovery_segment(
    capture_dir: Path,
    *,
    loss_dir: Path,
    loss: TrackingLoss,
    frames: Sequence[tuple[int, bytes]],
    transform: Rigid,
    evidence: Mapping[str, object],
    limitations: Iterable[str] = (),
    recovery_limitations: Iterable[str] = RECOVERY_LIMITATIONS,
) -> SegmentPublication:
    """Publish one accepted recovery: the held frames, the bridge and the binding.

    Exactly :func:`publish_segment`, with the binding attached, so the atomicity
    is the same atomicity: a hidden staging directory, every frame read back
    against its own digest, and one no-replace rename that either makes the
    whole thing visible under a free ``segment-NNNN`` name or leaves the capture
    directory as it was.

    The predecessor is the loss's last aligned frame - not the last frame on
    disk, which is the end of the failed streak - and the batch starts one past
    the frame that triggered the loss. Both are checked against the record this
    builds and against the loss itself, so a caller that got either wrong is
    refused before anything is staged.

    ``loss`` is checked against the record on disk rather than trusted: the
    current ``tracking-loss.json`` in ``loss_dir`` is the semantic AND the byte
    authority for what is published, and a supplied loss that no longer equals
    it is refused with nothing staged.
    """
    if not frames:
        raise SegmentPublicationRefused("a recovery publishes at least one frame; none were given")
    # The record ON DISK, right now, is both authorities: the meaning this
    # recovery is published against and the bytes it is bound to. A caller
    # holding a TrackingLoss it read at some earlier moment gets to be checked
    # against it rather than believed - because everything below derives the
    # predecessor, the start index and the excluded frames from the semantics
    # while the binding hashes the bytes, and if the two came from different
    # records the capture could publish a recovery its own reader would refuse
    # and announce a recovery that never closed anything.
    loss_dir = Path(loss_dir)
    try:
        current = _read_tracking_loss(loss_dir)
    except TrackingLossRecordUnusable as exc:
        raise SegmentPublicationRefused(
            f"the tracking-loss record this recovery would close cannot be read ({exc}); "
            "nothing was staged and nothing was published"
        ) from exc
    if current is None:
        raise SegmentPublicationRefused(
            f"there is no {TRACKING_LOSS_RECORD_NAME} in {loss_dir}, so there is no durable loss "
            "for this recovery to close; nothing was staged and nothing was published"
        )
    on_disk, record_payload = current
    if on_disk != loss:
        raise SegmentPublicationRefused(
            f"the tracking-loss record in {loss_dir} is no longer the loss this recovery was "
            "decided against. A recovery closes ONE record, exactly as it stands on disk when "
            "it is published; nothing was staged and nothing was published"
        )
    loss = on_disk
    predecessor_index = loss.last_aligned_frame_index
    start_index = frames[0][0]
    if predecessor_index < 0:
        raise SegmentPublicationRefused(
            "the loss being recovered says nothing in this capture ever aligned, so there is no "
            "accepted frame for a bridge to be measured against and nothing to recover to"
        )
    if start_index != loss.triggering_frame_index + 1:
        raise SegmentPublicationRefused(
            f"the batch starts at frame {start_index} and the loss it recovers from ended at "
            f"frame {loss.triggering_frame_index}; a recovery continues at the very next global "
            "index, so that the frames it excludes are exactly the ones the loss named"
        )
    # What the run the loss was recorded in actually holds, read off the disk
    # rather than taken from the caller. The frames it names are checked against
    # the record before anything is staged, for the reason the evidence is: a
    # recovery whose own reader would refuse it leaves the capture holding a
    # loss that looks closed, which is worse than one it could not close at all.
    run_indices = sorted(
        index
        for index in (_frame_index(path) for path in loss_dir.glob(FRAME_GLOB))
        if index is not None
    )
    run = None if loss_dir == Path(capture_dir) else loss_dir.name
    # Where that run's own published batch ends, for a loss recorded inside a
    # segment - read from its own boundary record, since a loss cannot begin
    # inside frames an earlier relocalization already accepted. The reader
    # enforces this too, and the point of doing it here is that it must not be
    # possible to PUBLISH a recovery the reader would then refuse.
    batch_end: int | None = None
    if run is not None:
        try:
            listed = _read_record(loss_dir).get("frames")
            if not isinstance(listed, list):
                raise BridgeRecordUnusable(f"{loss_dir.name} lists no frames of its own")
            batch_end = max(
                index
                for entry in listed
                if isinstance(entry, Mapping)
                and isinstance(index := entry.get("frame_index"), int)
                and not isinstance(index, bool)
            )
        except (BridgeRecordUnusable, ValueError) as exc:
            raise SegmentPublicationRefused(
                f"the run this loss was recorded in ({loss_dir.name}) does not hold a boundary "
                f"record this reader can place it against ({exc}); nothing was published"
            ) from exc
    pending = _PendingLoss(
        loss=loss,
        directory=loss_dir,
        run=run,
        excluded=tuple(
            SequenceFrame(
                path=loss_dir / FRAME_TEMPLATE.format(index=index), index=index, segment=run
            )
            for index in run_indices
            if index > predecessor_index
        ),
        prefix_end=max(
            (index for index in run_indices if index <= predecessor_index), default=None
        ),
        batch_end=batch_end,
    )
    try:
        binding = recovery_record(
            capture_dir,
            loss_dir=loss_dir,
            loss=loss,
            predecessor_index=predecessor_index,
            start_index=start_index,
            limitations=recovery_limitations,
            record_payload=record_payload,
        )
    except (BridgeRecordUnusable, OSError) as exc:
        raise SegmentPublicationRefused(
            f"the recovery could not be bound to the loss it closes ({exc}); nothing was staged "
            "and nothing was published"
        ) from exc
    return publish_segment(
        capture_dir,
        frames=frames,
        predecessor_index=predecessor_index,
        transform=transform,
        evidence=evidence,
        limitations=limitations,
        recovery=binding,
        recovered=pending,
    )
