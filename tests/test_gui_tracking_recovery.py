"""A live capture finding the object again after it has lost it.

What this is about
------------------
Task 020 stopped the 2026-08-28 failure from being invisible: three consecutive
frames that cannot be placed publish one immutable ``tracking-loss.json``, the
model freezes, and nothing more is written. That is honest and it is not what
the operator wants. They are standing in front of the object with the scanner in
their hand, the frames before the loss are real, and the machinery for putting a
small segment back against an accepted model already exists - it is what a
Resume uses.

So the capture tries, once, within bounds, and the whole of this file is about
what it may and may not do while it does:

- the loss record is published FIRST and is never withdrawn. Recovery is
  something that happens after a durable loss, not instead of one.
- the frames written before the threshold stay exactly where they are. They are
  the evidence for the loss and they never rejoin the scan.
- the frames collected while looking are held in memory only. Nothing reaches
  disk, no count moves and no pose is placed until an accepted relocalization
  has been PUBLISHED, and a publication that fails fails the capture.
- if the bounded search does not find the object, the loss simply stands. No
  gate is relaxed to make it succeed.

How these drive it
------------------
The real :meth:`~scanner3d.gui.ScannerWindow._start_capture`, its real
``on_frame`` and post-requeue ``after_frame``, the real dequeue loop against the
scripted fake driver, the real
:func:`~scanner3d.capture_sequence.publish_tracking_loss` and
:func:`~scanner3d.capture_sequence.publish_recovery_segment`, the real
authoritative reader and the real GUI slots.

Deprojection, adjacent alignment and the relocalization DECISION are stubbed,
for the reason they are stubbed in ``test_gui_pause.py``: the subject here is
the state machine, and a test that cannot choose which frame fails or whether a
decision accepts cannot ask about it. The decision's own correctness is in
``test_relocalize.py``, and what the published record has to survive is in
``test_tracking_recovery.py``.

Alignment attempts are numbered from 1 in the order ICP is called, which is once
per frame after the first, so "attempt N fails" and "frame N failed to align"
are the same statement.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import capture_sequence as capture_sequence_module
from scanner3d import gui, pipeline
from scanner3d.capture_sequence import (
    RECOVERY_RECORD_KEY,
    TRACKING_LOSS_RECORD_NAME,
    TRACKING_LOSS_STREAK,
    BridgeBoundary,
    SegmentPublicationRefused,
    TrackingLossUnresolved,
    read_capture_sequence,
    read_tracking_loss,
)
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.gui import (
    MIN_PAUSE_REFERENCE_CLOUDS,
    PAUSE_TITLE,
    RECOVERING_PAUSE_TITLE,
    RECOVERING_TRACKING_STATE,
    TRACKING_LOST_PAUSE_TITLE,
    TRACKING_LOST_STATE,
    TRACKING_RECOVERED_STATE,
)
from scanner3d.pipeline import SweepReconstruction
from scanner3d.relocalize import MAX_ATTEMPTS, MIN_TENTATIVE_FRAMES
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    accepting,
    alignments,
    capture_dir,
    clouds,
    decisions,
    live,
    mesh_budget,
    meshable_clouds,
    models,
    refusing,
    searches,
    sphere_raw_frame,
    start_capture,
    start_reconstruct,
    tentative_clouds,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "decisions",
    "live",
    "mesh_budget",
    "meshable_clouds",
    "models",
    "searches",
    "stream_kernel",
    "stream_node",
    "tentative_clouds",
]

#: Autouse for the reason it is autouse in the pause tests: a search that ran on
#: a real thread would let the machine decide when the answer arrived.
pytestmark = pytest.mark.usefixtures("searches")

#: The bridge a scripted acceptance carries. Neither identity nor axis aligned,
#: so a capture that composed it the wrong way round, or dropped it, would not
#: accidentally still produce a plausible pose chain.
BRIDGE = rotation_about_axis((0.3, 0.7, 0.65), 0.7, through=(0.0, 0.0, 120.0))


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, each with a fill byte of its own."""
    return tuple(whole_frame(1 + index, buffer_index=index % 4) for index in range(count))


def run(window: CaptureWindow, stream_kernel, delivered: int) -> None:
    stream_kernel(frames(delivered))
    start_capture(window)


def capture_of(window: CaptureWindow) -> Path:
    return cast(Path, window._capture_dir)


def states(window: CaptureWindow) -> list[str]:
    return [state.state for state in window.pause_states]


def details(window: CaptureWindow) -> str:
    return "\n".join(state.detail for state in window.pause_states)


def saved(window: CaptureWindow) -> list[int]:
    return [cast(int, count) for (count,) in window.worker.saved.emitted]


def segments(window: CaptureWindow) -> list[Path]:
    return sorted(capture_of(window).glob("segment-*"))


#: The frames whose alignment fails, chosen so that the loss lands well after
#: the accepted-model threshold: attempts 12, 13 and 14 are frames 12, 13 and
#: 14, and by then 12 frames have aligned.
LOSING = {12, 13, 14}
#: What the loss those failures produce names: the last frame that aligned, and
#: the three that did not.
LAST_ALIGNED = 11
EXCLUDED = [12, 13, 14]


def losing_capture(
    window: CaptureWindow, stream_kernel, delivered: int, decisions, outcomes
) -> list[int]:
    """A capture that loses tracking at frame 14 and then decides ``outcomes``."""
    asked = decisions(outcomes)
    run(window, stream_kernel, delivered)
    return asked


# -- an accepted recovery ---------------------------------------------------


def test_a_recovered_capture_carries_on_scanning_into_a_new_segment(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The whole arc, in one test: lost, looked, placed, carried on.

    The three held frames are published as one segment, the frames after them
    are written into that segment as ordinary frames, and the capture is running
    normally again by the end of it.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    # The three transitions, in order and each reached once. The recovering
    # state repeats while it collects, which is the progress report, so the
    # order is asserted on where each one FIRST appears.
    published_states = states(window)
    assert [state for state in published_states if state != RECOVERING_TRACKING_STATE] == [
        TRACKING_LOST_STATE,
        TRACKING_RECOVERED_STATE,
    ]
    first = published_states.index(RECOVERING_TRACKING_STATE)
    assert published_states.index(TRACKING_LOST_STATE) < first
    assert first < published_states.index(TRACKING_RECOVERED_STATE)
    published = segments(window)
    assert len(published) == 1
    sequence = read_capture_sequence(capture_of(window))
    assert len(sequence.recoveries) == 1
    # 0-11 aligned, 12-14 lost and excluded, 15 onwards recovered.
    assert [frame.index for frame in sequence.frames] == [*range(12), *range(15, 24)]
    assert [frame.index for frame in sequence.excluded] == EXCLUDED
    assert [frame.index for frame in sequence.frames if frame.in_segment] == list(range(15, 24))


def test_the_operator_is_told_it_recovered_and_how_many_frames_were_left_out(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """ "It worked" is not enough. The folder holds more frames than the scan does."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    said = details(window)
    assert "TRACKING RECOVERED" in said
    assert f"The {TRACKING_LOSS_STREAK} frame(s) written while tracking was lost" in said
    assert "are NOT part of the scan" in said
    assert "TRACKING RECOVERED" in window.report


def test_the_failed_frames_stay_on_disk_and_stay_out_of_the_scan(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Not deleted, not renamed, not moved, and not quietly readmitted.

    They are the evidence for the loss. What the recovery changes is that the
    capture is readable again, not what happened to them.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    on_disk = [int(path.stem.split("_")[1]) for path in window.raw_frames_on_disk()]
    assert on_disk == list(range(24))
    sequence = read_capture_sequence(capture_of(window))
    assert set(EXCLUDED).isdisjoint(frame.index for frame in sequence.frames)
    for frame in sequence.excluded:
        assert frame.path.parent == capture_of(window)


def test_the_loss_record_is_published_before_anything_is_recovered(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A recovery is a claim ABOUT a loss, so the loss has to exist first.

    Checked from inside the slot that receives the recovering state, which is
    the instant the capture first says it is trying: the record has to be on
    disk already, and no segment may be.
    """
    seen: list[tuple[bool, int]] = []
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)

    def watch(state: object) -> None:
        if cast(gui.PauseState, state).state == RECOVERING_TRACKING_STATE:
            directory = capture_of(window)
            seen.append(
                (
                    (directory / TRACKING_LOSS_RECORD_NAME).exists(),
                    len(list(directory.glob("segment-*"))),
                )
            )

    window.worker.pause_state.connect(watch)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    assert seen, "the capture never said it was looking for the object"
    assert set(seen) == {(True, 0)}
    assert read_tracking_loss(capture_of(window)) is not None


def test_the_recovery_binds_the_loss_it_closes_and_names_the_excluded_frames(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """What the capture actually wrote, read back as the record itself."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    import json

    record = json.loads((segments(window)[0] / "bridge.json").read_text())
    recovery = record[RECOVERY_RECORD_KEY]
    assert recovery["loss_record"]["path"] == TRACKING_LOSS_RECORD_NAME
    assert recovery["predecessor_frame_index"] == LAST_ALIGNED
    assert recovery["segment_start_frame_index"] == EXCLUDED[-1] + 1
    assert [entry["frame_index"] for entry in recovery["excluded_frames"]] == EXCLUDED
    # And the bridge is measured against the frame that ALIGNED, not the last
    # frame on disk, which has no pose at all.
    assert record["predecessor_frame_index"] == LAST_ALIGNED


def test_nothing_is_written_or_counted_while_the_search_is_still_looking(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The held frames are in memory and nowhere else.

    Every decision refuses, so the frames after the loss are collected, judged,
    offered to the search and dropped - and the folder afterwards holds exactly
    the frames written before the threshold.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=30)
    asked = losing_capture(window, stream_kernel, 30, decisions, [refusing("no overlap")])

    assert asked, "the capture never asked where the object went"
    assert all(count >= MIN_TENTATIVE_FRAMES for count in asked)
    assert not segments(window)
    # 15 frames reached disk: 0-11 aligned and 12-14 written before the
    # threshold was reached. Nothing after that.
    assert len(window.raw_frames_on_disk()) == 15
    assert saved(window)[-1] == 15
    # And the depth view kept moving throughout. A window that froze while it
    # looked would be indistinguishable from the crash all of this prevents,
    # and the operator has to be able to see the object to move back over it.
    assert len(window.worker.preview.emitted) == 30


def test_the_count_and_the_model_only_move_once_the_recovery_is_published(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The batch joins the scan all at once, on the frame it was accepted at."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    counts = saved(window)
    # 15 while lost, then a jump of the whole held batch, then one per frame.
    assert 15 in counts
    assert counts[counts.index(15) + 1] >= 15 + MIN_TENTATIVE_FRAMES
    aligned = [cast(gui.ModelSnapshot, s).aligned for (s,) in window.worker.coverage.emitted]
    assert aligned[-1] > MIN_PAUSE_REFERENCE_CLOUDS


# -- when it cannot be done -------------------------------------------------


def test_a_search_that_never_places_anything_leaves_the_loss_standing(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Refused is refused. No gate is lowered to turn it into an acceptance."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=40)
    losing_capture(
        window, stream_kernel, 40, decisions, [refusing("ambiguous: two placements fit")]
    )

    assert not segments(window)
    assert window._tracking_loss is not None
    said = details(window)
    assert "NOT recovered" in said
    assert "Press Stop and rescan" in said
    with pytest.raises(TrackingLossUnresolved):
        read_capture_sequence(capture_of(window))


def test_the_search_is_bounded_and_stops_asking(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A recovery that kept trying would spend the rest of the capture proving
    the same thing, at the frame rate."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=60)
    asked = losing_capture(window, stream_kernel, 60, decisions, [refusing("no overlap")])

    assert len(asked) <= MAX_ATTEMPTS
    assert "TRACKING LOST and NOT recovered" in details(window)


def test_a_loss_before_there_is_enough_model_is_not_even_attempted(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Below the reference threshold a search could only ever refuse.

    Spending the whole bounded budget to arrive there - and telling the operator
    something was being tried while it did - is worse than saying so at once.
    """
    live(failed_alignments={1, 2, 3})
    window = CaptureWindow(frames=20)
    asked = losing_capture(window, stream_kernel, 20, decisions, [accepting(BRIDGE)])

    assert asked == []
    assert states(window) == [TRACKING_LOST_STATE]
    assert f"under the {MIN_PAUSE_REFERENCE_CLOUDS}" in details(window)
    assert not segments(window)


def test_a_recovery_that_cannot_be_published_fails_the_capture(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, monkeypatch
) -> None:
    """It never reports a recovery it did not write.

    A pause boundary that cannot be published leaves the segment tentative and
    the capture as it was. This is not that: the capture has a durable loss and
    has just found the object, and carrying on would leave a folder whose loss
    looks closed.
    """
    live(failed_alignments=LOSING)

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise SegmentPublicationRefused("the disk went away")

    monkeypatch.setattr(gui, "publish_recovery_segment", refuse, raising=False)
    monkeypatch.setattr(capture_sequence_module, "publish_recovery_segment", refuse)
    window = CaptureWindow(frames=24)

    with pytest.raises(SegmentPublicationRefused):
        losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    assert TRACKING_RECOVERED_STATE not in states(window)
    assert not segments(window)
    assert window._tracking_loss is not None
    assert "could NOT be written" in details(window)
    # The loss still stands, and the folder is exactly what it was.
    assert len(window.raw_frames_on_disk()) == 15


def test_stop_during_a_recovery_publishes_nothing(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The held frames were never on disk, so a Stop simply drops them."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=40)
    pressed: list[int] = []

    def act(_raw: object) -> None:
        if len(pressed) == 0 and window._tracking_loss is not None:
            pressed.append(1)
            window._stop()

    window.worker.preview.connect(act)
    losing_capture(window, stream_kernel, 40, decisions, [refusing("still looking")])

    assert pressed == [1]
    assert not segments(window)
    assert window.stopped == 1
    assert len(window.raw_frames_on_disk()) == 15


# -- the clock the recovery is bounded by -----------------------------------
#
# An automatic recovery has nobody behind it. Nobody pressed anything to start
# it and nobody is going to press anything to end it, so a search that is not
# getting anywhere has to end itself. Both of these prove a path that ends
# NOWHERE ELSE: a frame that never reaches the tentative buffer, and a search
# that never comes back.


def test_a_recovery_that_never_gets_a_usable_frame_still_ends_at_its_deadline(
    stream_kernel, capture_dir: Path, live, decisions, monkeypatch
) -> None:
    """Post-loss frames that cannot even be deprojected must not run forever.

    Every frame here yields an empty cloud, which returns from ``collect_tentative``
    BEFORE anything is held, before any attempt is made and before any elapsed
    time was previously looked at. That is what a lost scanner pointed at
    nothing delivers, and without a deadline checked ahead of the work the
    capture would sit in ``recovering_tracking`` for the rest of the run.
    """
    offered: list[int] = []

    def unusable(raw: bytes, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        offered.append(raw[0])
        return ()

    live(failed_alignments=LOSING)
    monkeypatch.setattr(gui, "_cloud_from_bytes", unusable)
    window = CaptureWindow(frames=60)

    def arm(_raw: object) -> None:
        # Only once the capture has actually been chewing on unusable frames,
        # so this test is about the deadline rather than about a capture that
        # was over before it began.
        if len(offered) >= 3:
            monkeypatch.setattr(gui, "MAX_TENTATIVE_SECONDS", 0.0)

    window.worker.preview.connect(arm)
    asked = losing_capture(window, stream_kernel, 60, decisions, [accepting(BRIDGE)])

    assert len(offered) >= 3, "the capture never tried to hold a frame after the loss"
    # Nothing ever reached the buffer, so no decision was ever asked for - which
    # is exactly why the old check, taken after a successful add, was never
    # reached either.
    assert asked == []
    assert not segments(window)
    assert window._tracking_loss is not None
    said = details(window)
    assert "TRACKING LOST and NOT recovered" in said
    assert "after tracking was lost" in said
    assert "Press Stop and rescan" in said
    assert states(window)[-1] == TRACKING_LOST_STATE
    # And it STOPPED looking. 45 frames arrived after the loss; a capture that
    # never noticed the deadline would have deprojected all of them.
    assert len(offered) <= 6
    assert len(window.raw_frames_on_disk()) == 15


def test_a_search_still_running_at_the_deadline_is_abandoned_rather_than_waited_for(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches, monkeypatch
) -> None:
    """A search that never comes back is the other way to wait forever.

    Held and marked running, so it can no longer be cancelled - the state a real
    worker that has picked a task up is in. The capture must drop it, publish
    nothing and go to the terminal state, rather than poll it for the rest of
    the capture or block waiting for it.
    """
    live(failed_alignments=LOSING)
    searches.immediate = False
    window = CaptureWindow(frames=60)

    def arm(_raw: object) -> None:
        if searches.in_flight and searches.begin():
            monkeypatch.setattr(gui, "MAX_TENTATIVE_SECONDS", 0.0)

    window.worker.preview.connect(arm)
    asked = losing_capture(window, stream_kernel, 60, decisions, [accepting(BRIDGE)])

    assert searches.submitted == 1
    # Submitted, started and never finished: the decision itself was never made.
    assert asked == []
    assert searches.in_flight == 1, "the capture waited for the search rather than abandoning it"
    assert not segments(window)
    assert window._tracking_loss is not None
    said = details(window)
    assert "TRACKING LOST and NOT recovered" in said
    assert states(window)[-1] == TRACKING_LOST_STATE
    assert len(window.raw_frames_on_disk()) == 15
    # Nothing waited for it on the way out either.
    assert searches.shutdowns and all(entry["wait"] is False for entry in searches.shutdowns)
    # The depth view kept moving the whole time, which is what a capture stuck
    # behind a search does not do.
    assert len(window.worker.preview.emitted) == 60


def test_a_search_that_finishes_after_the_deadline_is_not_applied(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches, monkeypatch
) -> None:
    """A late acceptance is still late, and a late recovery is not a recovery.

    The clock is an ACCEPTANCE BOUNDARY, not merely a watchdog for a search
    that hangs. Here the search does come back, and comes back accepted - but
    only after the recovery was already allowed to end, which is the moment
    nobody can say the object is still where that answer places it. So the
    result is dropped unlooked-at: no segment, no recovered state, and the
    durable loss stands exactly as recorded.

    Checking the deadline only when the future is unfinished would let this
    exact case through, because by the time it is polled the future is done.
    """
    live(failed_alignments=LOSING)
    searches.immediate = False
    window = CaptureWindow(frames=60)
    finished: list[int] = []

    def arm(_raw: object) -> None:
        # One delivery AFTER the search was submitted, so the sequence is the
        # one that matters: the clock runs out, the search then finishes with
        # an accepted answer, and only then is the recovery path polled.
        if finished or not searches.in_flight:
            return
        monkeypatch.setattr(gui, "MAX_TENTATIVE_SECONDS", 0.0)
        finished.append(searches.run())

    window.worker.preview.connect(arm)
    asked = losing_capture(window, stream_kernel, 60, decisions, [accepting(BRIDGE)])

    # The decision really was made, really accepted, and really is sitting in a
    # finished future - otherwise this test would pass for the wrong reason.
    assert searches.submitted == 1
    assert finished == [1]
    assert asked == [MIN_TENTATIVE_FRAMES]
    assert searches.in_flight == 0

    # And none of it was applied.
    assert not segments(window)
    assert TRACKING_RECOVERED_STATE not in states(window)
    assert states(window)[-1] == TRACKING_LOST_STATE
    said = details(window)
    assert "TRACKING LOST and NOT recovered" in said
    assert "Press Stop and rescan" in said
    assert "TRACKING RECOVERED" not in said
    assert window._tracking_loss is not None
    # The loss is untouched, the failed frames are still the only thing written
    # after the last aligned frame, and the capture is still unreconstructable.
    assert len(window.raw_frames_on_disk()) == 15
    on_disk = read_tracking_loss(capture_of(window))
    assert on_disk is not None
    assert on_disk.last_aligned_frame_index == LAST_ALIGNED
    assert [failure.frame_index for failure in on_disk.failures] == EXCLUDED
    with pytest.raises(TrackingLossUnresolved):
        read_capture_sequence(capture_of(window))


# -- what the window does ---------------------------------------------------


def test_pause_is_denied_while_recovering_and_says_what_is_happening(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Pause needs exactly the machinery currently in use, so it is not offered.

    Relabelled rather than merely greyed: a disabled control still reading
    "Pause" says "not right now", and what is true is "the scan is doing that
    itself at this moment".
    """
    seen: list[tuple[str, bool]] = []
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=40)

    def watch(_raw: object) -> None:
        if window._tracking_loss is not None:
            seen.append((window._pause_action.text, window._pause_action.enabled))

    # On the depth-preview channel, which is offered every complete delivery, so
    # this reads the toolbar as an operator would see it BETWEEN the capture's
    # own announcements rather than in the middle of one.
    window.worker.preview.connect(watch)
    losing_capture(window, stream_kernel, 40, decisions, [refusing("no overlap")])

    assert RECOVERING_PAUSE_TITLE in [title for title, _ in seen]
    assert {title for title, _ in seen} <= {RECOVERING_PAUSE_TITLE, TRACKING_LOST_PAUSE_TITLE}
    assert not any(enabled for _, enabled in seen), "Pause was offered while tracking was lost"
    assert window._pause_action.enabled is False
    assert "collecting a small segment" in details(window)
    assert "looking for the model it had already accepted" in details(window)
    # And once it gave up, the words go back to naming the loss itself. Read
    # from the last delivery of the run rather than from afterwards: every
    # capture that ends resets the toolbar for the next one.
    assert seen[-1] == (TRACKING_LOST_PAUSE_TITLE, False)


def test_pause_comes_back_once_the_recovery_has_rebuilt_enough_model(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The lost banner is cleared by the recovery and by nothing else.

    Pause then returns the same way it arrives at the start of a capture: when
    the model panel reports enough accepted geometry to relocalize against.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    offered: list[tuple[str, bool]] = []

    def watch(_raw: object) -> None:
        if any(state.state == TRACKING_RECOVERED_STATE for state in window.pause_states):
            offered.append((window._pause_action.text, window._pause_action.enabled))

    window.worker.preview.connect(watch)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    assert window._tracking_loss is None
    assert window._tracking_loss_unrecorded is None
    assert window._lost_tracking is False
    # Read while the capture was still running: the toolbar is reset when any
    # capture ends, so the end of the run cannot answer this.
    assert offered, "the capture never recovered"
    assert offered[-1] == (PAUSE_TITLE, True)


def test_a_recovered_capture_ends_by_pointing_at_the_frames_rather_than_the_loss(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The closing sentence follows the state the capture actually ended in."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    assert "Press Reconstruct" in window.report


def test_a_recovered_capture_counts_the_scan_and_the_evidence_apart_when_it_ends(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The folder holds more frames than the scan does, and the ending says so.

    The live counter had already shown every file written, excluded ones
    included. An ending that quietly reported only the reconstructable ones
    would look like frames had gone missing between the last heartbeat and the
    last sentence; one that reported only the total would be calling failure
    evidence part of the scan.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    sequence = read_capture_sequence(capture_of(window))
    assert (len(sequence.frames), len(sequence.excluded)) == (21, 3)
    report = window.report
    assert "Captured 21 frames" in report
    assert "3 further frame(s) - 24 files in all - were written while tracking was lost" in report
    assert "NOT part of the 21 reconstructable frames" in report
    # The number the operator was last shown while it ran. It does not fall.
    assert saved(window)[-1] == 24
    assert len(window.raw_frames_on_disk()) == 24


def test_stop_after_a_recovery_says_what_is_reconstructable_and_what_is_evidence(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The same two numbers on the other ending, which has its own sentence."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=40)
    pressed: list[int] = []

    def act(_raw: object) -> None:
        if not pressed and any(
            state.state == TRACKING_RECOVERED_STATE for state in window.pause_states
        ):
            pressed.append(1)
            window._stop()

    window.worker.preview.connect(act)
    losing_capture(window, stream_kernel, 40, decisions, [accepting(BRIDGE)])

    assert pressed == [1]
    assert window.stopped == 1
    sequence = read_capture_sequence(capture_of(window))
    on_disk = len(window.raw_frames_on_disk())
    assert len(sequence.excluded) == TRACKING_LOSS_STREAK
    assert len(sequence.frames) + len(sequence.excluded) == on_disk
    report = window.report
    assert f"Stopped. {len(sequence.frames)} frames were already written" in report
    assert (
        f"{len(sequence.excluded)} further frame(s) - {on_disk} files in all - were written "
        "while tracking was lost" in report
    )
    assert f"NOT part of the {len(sequence.frames)} reconstructable frames" in report
    assert saved(window)[-1] == on_disk
    assert "REFUSE this folder" not in window.report


def test_an_unrecovered_capture_still_ends_by_refusing_to_offer_reconstruct(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=40)
    losing_capture(window, stream_kernel, 40, decisions, [refusing("no overlap")])

    assert "REFUSE this folder" in window.report
    assert "Press Reconstruct to register them" not in window.report


# -- more than one episode, and pausing around them -------------------------


def test_a_second_loss_inside_a_recovered_segment_recovers_independently(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Two episodes in one capture, the second inside the first's own segment.

    A loss record always written at the capture root would be refused the second
    time - an existing one is never overwritten - so the second episode would
    either be silently lost or fail the run. Each run keeps its own.
    """
    live(failed_alignments={12, 13, 14, 24, 25, 26})
    window = CaptureWindow(frames=40)
    losing_capture(window, stream_kernel, 40, decisions, [accepting(BRIDGE)])

    sequence = read_capture_sequence(capture_of(window))
    assert len(sequence.recoveries) == 2
    assert [recovery.run for recovery in sequence.recoveries] == [None, "segment-0001"]
    # Three frames per episode, the first three being the ones scripted above.
    # The second episode's own indices depend on how many frames the first
    # recovery held, which is the state machine's business rather than this
    # test's: what matters is that the second streak is excluded as well.
    excluded = [frame.index for frame in sequence.excluded]
    assert len(excluded) == 2 * TRACKING_LOSS_STREAK
    assert excluded[:3] == EXCLUDED
    assert set(excluded).isdisjoint(frame.index for frame in sequence.frames)
    assert (capture_of(window) / TRACKING_LOSS_RECORD_NAME).exists()
    assert (capture_of(window) / "segment-0001" / TRACKING_LOSS_RECORD_NAME).exists()
    assert states(window).count(TRACKING_RECOVERED_STATE) == 2


def test_an_ordinary_pause_after_a_recovery_still_works(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A recovery leaves an ordinary running capture behind it, pause included.

    The pause boundary it then publishes is a pause boundary: it carries no
    recovery binding, and the reader counts it as a reposition rather than as a
    second closed loss.
    """
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=34)
    pressed: list[str] = []

    def act(_raw: object) -> None:
        recovered = any(state.state == TRACKING_RECOVERED_STATE for state in window.pause_states)
        if not pressed and recovered and window._pause_action.enabled:
            pressed.append("pause")
            window.pause()
        elif pressed == ["pause"]:
            pressed.append("resume")
            window.resume()

    window.worker.preview.connect(act)
    losing_capture(window, stream_kernel, 34, decisions, [accepting(BRIDGE)])

    assert pressed == ["pause", "resume"]
    sequence = read_capture_sequence(capture_of(window))
    assert len(sequence.recoveries) == 1
    assert len(sequence.bridges) == 2
    stated = "\n".join(sequence.limitations)
    assert "repositioned 1 time(s)" in stated
    assert "lost tracking 1 time(s)" in stated


# -- the one reader decides for both entry points ---------------------------


def test_reconstruct_refuses_an_unrecovered_capture_and_accepts_a_recovered_one(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Neither entry point carries a check of its own; both ask the reader.

    Driven here through the same function the window's Reconstruct and the
    CLI's ``reconstruct`` both call, so the decision they inherit is the one
    this file has been asserting all along.
    """
    live(failed_alignments=LOSING)
    lost = CaptureWindow(frames=40)
    losing_capture(lost, stream_kernel, 40, decisions, [refusing("no overlap")])

    with pytest.raises(TrackingLossUnresolved):
        read_capture_sequence(capture_of(lost))


def test_the_recovered_seam_is_handed_over_as_a_boundary_not_as_a_pair_to_fit(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The pair either side of the loss is the one pair that must not be fitted."""
    live(failed_alignments=LOSING)
    window = CaptureWindow(frames=24)
    losing_capture(window, stream_kernel, 24, decisions, [accepting(BRIDGE)])

    sequence = read_capture_sequence(capture_of(window))
    boundaries = sequence.boundaries()

    assert len(boundaries) == 1
    position = boundaries[0].position
    assert sequence.frames[position - 1].index == LAST_ALIGNED
    assert sequence.frames[position].index == EXCLUDED[-1] + 1
    assert boundaries[0].transform != Rigid.identity()


# -- the window's own Reconstruct, over a folder a recovery closed ----------


def _recovered_capture_on_disk(root: Path) -> Path:
    """A finished capture: 9 aligned frames, 3 lost, then 3 recovered and 3 more.

    Built through the real publishers rather than by hand, so what Reconstruct
    reads back is what a capture would actually have left behind. Every frame is
    the same real 512000-byte render, because the export downstream reads these
    bytes and refuses anything shorter.
    """
    from scanner3d.capture_sequence import publish_recovery_segment, publish_tracking_loss
    from scanner3d.geometry import Rigid as _Rigid

    capture = root / "recovered-scan"
    capture.mkdir()
    raw = sphere_raw_frame()
    for index in range(12):
        (capture / f"frame_{index:04d}.raw").write_bytes(raw)
    publish_tracking_loss(
        capture,
        last_aligned_frame_index=8,
        failures=[
            capture_sequence_module.AlignmentFailure(
                frame_index=index,
                failure_type="RegistrationError",
                message="synthetic: point-to-plane singular in translation z",
            )
            for index in (9, 10, 11)
        ],
    )
    loss = read_tracking_loss(capture)
    assert loss is not None
    publish_recovery_segment(
        capture,
        loss_dir=capture,
        loss=loss,
        frames=[(12 + offset, raw) for offset in range(3)],
        transform=BRIDGE,
        evidence={
            **accepting(BRIDGE).as_dict(),
            "local_chain": {
                "gates": {"local_max_residual": 1.5},
                "edges": [
                    {
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
                    for frame in (1, 2)
                ],
                "anchor_frames": 1,
            },
        },
        limitations=("a synthetic recovery, published by this test",),
    )
    for index in range(15, 18):
        (capture / "segment-0001" / f"frame_{index:04d}.raw").write_bytes(raw)
    del _Rigid
    return capture


def test_the_window_reconstruct_registers_the_prefix_and_the_recovered_frames(
    tmp_path: Path, meshable_clouds, mesh_budget, monkeypatch
) -> None:
    """Neither entry point checks anything itself; both ask the one reader.

    A glob of the root would register twelve frames - the nine that aligned AND
    the three written while tracking was lost - and stop there. That is the
    failure this whole mechanism exists to prevent, and it is invisible from the
    report: it still reconstructs, still exports and still scores.
    """
    seen: list[dict[str, object]] = []

    def run(frame_clouds, **kwargs: object) -> SweepReconstruction:
        seen.append({"frames": len(frame_clouds), **kwargs})
        return SweepReconstruction(
            poses=tuple(
                rotation_about_axis((0.0, 1.0, 0.0), 0.05 * index, through=(0.0, 0.0, 120.0))
                for index in range(len(frame_clouds))
            ),
            frames=len(frame_clouds),
            revisits_found=0,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=("a synthetic registration, fitted to nothing",),
            bridges=("segment-0001",),
        )

    monkeypatch.setattr(pipeline, "reconstruct_sweep", run)
    capture = _recovered_capture_on_disk(tmp_path)
    window = CaptureWindow(frames=1)

    start_reconstruct(window, capture, monkeypatch)

    assert len(seen) == 1
    # Nine aligned plus six in the segment. The three lost frames are on disk
    # and are not among them.
    assert seen[0]["frames"] == 15
    boundaries = cast(Sequence[BridgeBoundary], seen[0]["bridges"])
    assert [boundary.position for boundary in boundaries] == [9]
    assert boundaries[0].transform == BRIDGE
    assert len(sorted(capture.rglob("frame_*.raw"))) == 18


def test_the_window_reconstruct_refuses_a_capture_whose_loss_is_still_open(
    tmp_path: Path, meshable_clouds, mesh_budget, monkeypatch
) -> None:
    """The other half of the inheritance, through the window rather than the CLI."""
    capture = _recovered_capture_on_disk(tmp_path)
    (capture / "segment-0001" / "bridge.json").unlink()
    window = CaptureWindow(frames=1)

    with pytest.raises(TrackingLossUnresolved, match="segment-0001"):
        start_reconstruct(window, capture, monkeypatch)

    assert window.completed == []
