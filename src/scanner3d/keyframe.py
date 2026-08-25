"""Chain a subset of frames, not all of them, so drift has fewer links to grow along.

Why this exists
---------------
Registration chains every captured frame: 2 to 1, 3 to 2, and so on. Each
pairwise alignment carries a small error, and those errors accumulate along the
chain - so drift grows with the NUMBER OF LINKS, not with elapsed time or with
how far the object turned.

Measured on a real two-lap capture, varying only the angular step, which varies
the number of links per revolution:

    deg/frame   frames/lap   drift/lap
       1.96        189         31.2 deg
       3.92         94         23.4 deg
       7.84         47         65.2 deg

Fewer links helped, until the step grew so large that individual alignments
started failing outright. That is the tension keyframing resolves: you want few
links AND reliable alignments, and you get both by keeping every frame for its
POINTS while chaining only a subset for its POSE.

The vendor's own software works this way. Disassembling RevoScan 6.0.3.5 shows
a keyframe pose graph at the centre of its registration stack
(``dumpKeyFramePoseGraph``, ``rebuildFromKeyFrames``, ``compactChunkToKeyFrame``).
This module closes the same gap.

Why the rule is measured motion, not a fixed interval
-----------------------------------------------------
"Every tenth frame" is wrong at both ends: it wastes links when the object
barely moves, and it drops below the alignable step when the object moves fast.
Accumulating :func:`~scanner3d.pipeline.frame_travel` - how far the surface
actually moved - and cutting a new keyframe once enough has accumulated adapts
to both without a setting per capture. A capture that pauses produces no new
keyframes; a fast one produces them more often.

What this does NOT do
---------------------
It does not discard anything. Every frame keeps contributing its points; a
non-keyframe simply borrows its pose from the keyframe it was measured against.

It does not make a bad alignment good. If the alignment between a frame and its
keyframe is wrong, keyframing spreads that error over the frames attached to it
rather than isolating it. Fewer links means fewer chances to go wrong, not a
guarantee that none do.

No physical unit is implied. Distances are in whatever the depth scale produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.pipeline import NOISE_FLOOR, TRAVEL_MAX_MULTIPLE, frame_travel

SCHEMA_VERSION: Final = 1

#: Surface travel to accumulate before cutting a new keyframe, as a multiple of
#: the sensor's noise floor.
#:
#: Set at the TOP of the band that registers well rather than in the middle.
#: The band exists because alignment degrades at both ends; keyframe spacing
#: wants to sit as wide as alignment reliably tolerates, since every link saved
#: is a link that cannot contribute error. Going wider than the band would buy
#: fewer links at the cost of alignments that fail, which is the trade that made
#: 7.84 deg/frame the worst row in the table above.
KEYFRAME_TRAVEL_MULTIPLE: Final = TRAVEL_MAX_MULTIPLE

#: Frames to force a keyframe after, however little has moved. A capture of a
#: stationary object would otherwise chain hundreds of frames to one keyframe,
#: and a single bad alignment there would take all of them with it.
MAX_FRAMES_PER_KEYFRAME: Final = 40


class KeyframeNotAllowed(ScannerError):
    """A keyframe selection was requested that this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


@dataclass(frozen=True, slots=True)
class FrameAttachment:
    """Where one frame's pose comes from.

    ``offset`` carries the frame from its keyframe's coordinates into its own.
    For a keyframe it is the identity, and ``keyframe`` is the frame itself -
    stated rather than special-cased so a caller never has to branch.
    """

    index: int
    keyframe: int
    offset: Rigid
    travel: float

    @property
    def is_keyframe(self) -> bool:
        return self.index == self.keyframe


@dataclass(frozen=True, slots=True)
class KeyframeSelection:
    """Which frames carry the pose chain, and where every other frame attaches."""

    attachments: tuple[FrameAttachment, ...]
    keyframes: tuple[int, ...]
    travel_threshold: float
    limitations: tuple[str, ...]

    @property
    def link_reduction(self) -> float:
        """How many times fewer links the keyframe chain has. 1.0 means none."""
        links = max(1, len(self.keyframes) - 1)
        return (len(self.attachments) - 1) / links if len(self.attachments) > 1 else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frames": len(self.attachments),
            "keyframes": len(self.keyframes),
            "keyframe_indices": list(self.keyframes),
            "travel_threshold": self.travel_threshold,
            "link_reduction": self.link_reduction,
            "limitations": list(self.limitations),
        }


def select_keyframes(
    clouds: Sequence[Sequence[Point]],
    align: object,
    *,
    travel_threshold: float | None = None,
    max_frames_per_keyframe: int = MAX_FRAMES_PER_KEYFRAME,
) -> KeyframeSelection:
    """Choose keyframes by accumulated surface motion, and attach the rest.

    ``align(source, target)`` must return the :class:`~scanner3d.geometry.Rigid`
    carrying ``source`` onto ``target``. It is injected rather than chosen here
    so this module stays about SELECTION - which matcher to use, and with which
    settings, is the caller's decision and is measured elsewhere.

    Each frame is aligned to the CURRENT KEYFRAME, not to its predecessor. That
    is the whole point: the chain runs keyframe to keyframe, and every other
    frame hangs off one of them by a single measured transform rather than by a
    running product of them.
    """
    if len(clouds) < 2:
        raise KeyframeNotAllowed(f"keyframe selection needs at least two frames, got {len(clouds)}")
    if max_frames_per_keyframe < 1:
        raise KeyframeNotAllowed(
            f"max_frames_per_keyframe must be at least 1, got {max_frames_per_keyframe}"
        )
    threshold = (
        travel_threshold if travel_threshold is not None else KEYFRAME_TRAVEL_MULTIPLE * NOISE_FLOOR
    )
    if not threshold > 0.0:
        raise KeyframeNotAllowed(f"travel_threshold must be positive, got {threshold}")
    if not callable(align):
        raise KeyframeNotAllowed("align must be callable")

    attachments: list[FrameAttachment] = [
        FrameAttachment(index=0, keyframe=0, offset=Rigid.identity(), travel=0.0)
    ]
    keyframes: list[int] = [0]
    since_keyframe = 0
    failures = 0

    for index in range(1, len(clouds)):
        current = keyframes[-1]
        try:
            offset = align(clouds[index], clouds[current])
        except ScannerError:
            # A frame that will not align to the current keyframe becomes a
            # keyframe itself: whatever changed, the old one no longer describes
            # what the scanner is looking at.
            failures += 1
            keyframes.append(index)
            attachments.append(
                FrameAttachment(index=index, keyframe=index, offset=Rigid.identity(), travel=0.0)
            )
            since_keyframe = 0
            continue

        travel = frame_travel(clouds[index], offset)
        attachments.append(
            FrameAttachment(index=index, keyframe=current, offset=offset, travel=travel)
        )
        since_keyframe += 1

        # Travel is measured from the KEYFRAME, not summed along the way, so it
        # cannot be inflated by a frame that wobbled back and forth without the
        # object having actually gone anywhere.
        if travel >= threshold or since_keyframe >= max_frames_per_keyframe:
            keyframes.append(index)
            attachments[-1] = FrameAttachment(
                index=index, keyframe=index, offset=Rigid.identity(), travel=travel
            )
            since_keyframe = 0

    limitations = [
        "Keyframing reduces the NUMBER of chained links, which is what drift accumulates "
        "along. It does not improve any individual alignment, and a wrong keyframe-to-"
        "keyframe transform now carries every frame attached to it.",
        "Travel is measured against the current keyframe rather than summed frame to frame, "
        "so an object that moves away and returns does not trigger a new keyframe. That is "
        "deliberate - nothing new was seen - but it means a keyframe can span more real "
        "motion than the threshold if the path doubled back.",
    ]
    if failures:
        limitations.append(
            f"{failures} frame(s) could not be aligned to the keyframe in force and became "
            "keyframes themselves. A capture with many of these is not being tracked; the "
            "count is worth reading before trusting the result."
        )
    if len(keyframes) == len(clouds):
        limitations.append(
            "EVERY frame became a keyframe, so nothing was saved. Either the object moved "
            "more than the threshold between every pair, or alignment is failing throughout."
        )
    return KeyframeSelection(
        attachments=tuple(attachments),
        keyframes=tuple(keyframes),
        travel_threshold=threshold,
        limitations=tuple(limitations),
    )


def poses_from_keyframes(
    selection: KeyframeSelection, keyframe_poses: Sequence[Rigid]
) -> tuple[Rigid, ...]:
    """Give every ORIGINAL frame a pose, from the poses of the keyframes.

    A non-keyframe's pose is its keyframe's pose composed with the offset that
    was measured directly between them - one transform, not a running product,
    which is the whole reason its error does not accumulate.
    """
    if len(keyframe_poses) != len(selection.keyframes):
        raise KeyframeNotAllowed(
            f"{len(keyframe_poses)} poses for {len(selection.keyframes)} keyframes: one each"
        )
    lookup = dict(zip(selection.keyframes, keyframe_poses, strict=True))
    return tuple(
        lookup[attachment.keyframe].compose(attachment.offset)
        for attachment in selection.attachments
    )
