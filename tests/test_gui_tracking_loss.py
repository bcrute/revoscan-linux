"""What a live capture does when it stops knowing where the object is.

The capture this is about
-------------------------
``/home/ben/scans/scan/20260828-105933``. Fifty-one frames saved and aligned by
11:00:08. The first live-alignment failure on displayed frame 53, one second
later. Then a failure on every frame after it - point-to-plane singular in
translation z at first, from geometry that had gone effectively planar, and
later no usable correspondence at all once the view had moved well past the
12-unit gate. The capture ran on to 176 frames. Depth and colour previews kept
moving, the frame counter kept climbing, and the model on screen simply stopped
changing. Nothing on disk, and nothing on screen, said why.

So the failure was not that alignment broke. Alignment breaking is ordinary and
a single frame that will not align is not worth stopping for. The failure was
that a capture went on producing 120-odd more frames which LOOK like a
continuation of the scan and are related to it by nothing at all.

What this file drives
---------------------
The real :meth:`~scanner3d.gui.ScannerWindow._start_capture`, its real
``on_frame``, the real dequeue loop against the scripted fake driver, the real
:func:`~scanner3d.capture_sequence.publish_tracking_loss` and the real GUI slots
that write the panel and move the toolbar. Deprojection, alignment and the
coverage build are stubbed for the reason they are stubbed in
``test_gui_live_model.py``: the question here is which frame the capture decides
on and what it stops doing, and a test that cannot choose which alignment fails
cannot ask it.

Alignment attempts are numbered from 1 in the order ICP is called, which is once
per frame after the first - frame 0 is the reference and is never aligned to
anything. So "attempt N fails" and "frame N failed to align" are the same
statement, and every scripted failure set below reads as frame numbers.

The record's own shape, and the reader's refusal of every way it can be wrong,
are in ``test_tracking_loss.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import capture_sequence as capture_sequence_module
from scanner3d import gui
from scanner3d.capture_sequence import (
    TRACKING_INPUT_EMPTY_TYPE,
    TRACKING_INPUT_UNUSABLE_TYPE,
    TRACKING_LOSS_RECORD_NAME,
    TRACKING_LOSS_STREAK,
    TrackingLossPublicationRefused,
    TrackingLossUnresolved,
    read_capture_sequence,
    read_tracking_loss,
)
from scanner3d.gui import (
    PAUSE_TITLE,
    PAUSE_TOOLTIP,
    TRACKING_LOSS_UNRECORDED_STATE,
    TRACKING_LOST_PAUSE_TITLE,
    TRACKING_LOST_STATE,
)
from scanner3d.tracking import TRACK_NO_CANDIDATE_TYPE
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    clouds,
    live,
    models,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "live",
    "models",
    "stream_kernel",
    "stream_node",
]


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, cycling the four buffers a real driver grants."""
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


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


def tracking_input(
    monkeypatch: pytest.MonkeyPatch,
    *,
    empty: set[int] | None = None,
    raising: set[int] | None = None,
    stop_on: tuple[CaptureWindow, int] | None = None,
) -> None:
    """Make named frames deproject to nothing, or fail to deproject at all.

    Wraps whatever stub is already installed - the ``live`` fixture's, always,
    since this is called from the test body - so every other frame still yields
    the ordinary three-point cloud and the alignment script still means what it
    says. Frame numbers are the global frame indices written to disk.

    ``stop_on`` presses Stop from inside the deprojection of one frame, which is
    the only way to land a cancellation in the middle of a frame the capture is
    already halfway through classifying.
    """
    empty = empty or set()
    raising = raising or set()
    base = gui._load_cloud

    def load(path: Path, target_points: int) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        if stop_on is not None and index == stop_on[1]:
            stop_on[0]._stop()
            raise ValueError(f"synthetic deprojection failure on frame {index}")
        if index in raising:
            raise ValueError(f"synthetic deprojection failure on frame {index}")
        if index in empty:
            return ()
        return base(path, target_points)

    monkeypatch.setattr(gui, "_load_cloud", load)


# -- the threshold ----------------------------------------------------------


def test_two_consecutive_failures_are_a_dropout_and_not_a_loss(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Exactly below the threshold, and the capture is untouched by it.

    Two frames in a row that will not align is a bad moment - a hand across the
    view, a burst of noise - and a scan that ended there would be unusable for
    the ordinary case this has to survive.
    """
    window = CaptureWindow(frames=6)
    live(failed_alignments={1, 2})

    run(window, stream_kernel, 6)

    assert window._tracking_loss is None
    assert TRACKING_LOST_STATE not in states(window)
    assert read_tracking_loss(capture_of(window)) is None
    assert len(window.raw_frames_on_disk()) == 6
    assert saved(window) == [1, 2, 3, 4, 5, 6]


def test_the_primary_status_counts_a_dropout_before_it_becomes_a_loss(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The operator gets the threshold and one action, not an ICP paragraph."""
    window = CaptureWindow(frames=6)
    live(failed_alignments={1, 2})

    run(window, stream_kernel, 6)

    statuses = [cast(gui.ScanStatus, payload[0]) for payload in window.worker.scan_status.emitted]
    titles = [status.title for status in statuses]
    assert "TRACKING WARNING — 1 OF 3" in titles
    assert "TRACKING WARNING — 2 OF 3" in titles
    restored = next(status for status in statuses if status.title == "TRACKING RESTORED")
    assert restored.summary == "Frame 3 aligned; the scan is connected again."
    assert restored.action == "Continue slowly with overlap to the previous view."
    assert all("candidate" not in status.summary.lower() for status in statuses)


def test_the_third_consecutive_failure_loses_tracking_at_that_frame(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Exactly at the threshold. The record names the frames, not a vague moment.

    Frame 0 is the reference and aligns by definition; frames 1, 2 and 3 fail in
    a row. The last accepted pose is therefore frame 0, the streak runs from
    frame 1 to frame 3, and the loss is entered on frame 3 - which is what makes
    the valid prefix a fact rather than an estimate.
    """
    window = CaptureWindow(frames=10)
    live(failed_alignments={1, 2, 3})

    run(window, stream_kernel, 10)

    loss = read_tracking_loss(capture_of(window))
    assert loss is not None
    assert loss.threshold == TRACKING_LOSS_STREAK
    assert loss.last_aligned_frame_index == 0
    assert loss.first_failed_frame_index == 1
    assert loss.triggering_frame_index == 3
    assert [failure.frame_index for failure in loss.failures] == [1, 2, 3]
    # The failure text the live path actually saw, kept so the refusal months
    # later can be interpreted rather than merely obeyed. It is the TRACKER's
    # account of the frame - which references were tried and what each of them
    # said - rather than the exception one solve happened to raise, because a
    # frame is placed by up to three references now and "ValueError" would not
    # say which of them failed or why.
    assert all("no reference placed this frame" in f.message for f in loss.failures)
    assert all(f.failure_type == TRACK_NO_CANDIDATE_TYPE for f in loss.failures)


def test_one_alignment_that_works_resets_the_streak(stream_kernel, capture_dir: Path, live) -> None:
    """Two, recover, two, recover. Never three in a row, so never a loss.

    The cumulative failure count reaches four here. A capture that decided on
    that number would end a scan that was working, which is why the streak is a
    separate variable from the count the live log already keeps.
    """
    window = CaptureWindow(frames=8)
    live(failed_alignments={1, 2, 4, 5})

    run(window, stream_kernel, 8)

    assert window._tracking_loss is None
    assert read_tracking_loss(capture_of(window)) is None
    assert len(window.raw_frames_on_disk()) == 8
    assert saved(window) == [1, 2, 3, 4, 5, 6, 7, 8]


# -- what stops, and what does not ------------------------------------------


def test_nothing_is_written_counted_or_modelled_after_the_loss(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The 120 extra frames of 2026-08-28, asserted as three counts that stop.

    Twenty frames are delivered and four reach the disk. A capture that stopped
    only the model, or only the counter, would look identical from the toolbar
    and would still be filling a folder with frames nothing places.
    """
    window = CaptureWindow(frames=20)
    live(failed_alignments={1, 2, 3})

    run(window, stream_kernel, 20)

    assert len(window.raw_frames_on_disk()) == 4
    assert saved(window) == [1, 2, 3, 4]
    # Frame 0 is the only one that ever aligned, which is below
    # MIN_COVERAGE_CLOUDS, so no model was ever built - and none was built
    # afterwards either.
    assert window.worker.coverage.emitted == []
    assert window._cloud.surfaces == []


def test_the_loss_is_entered_and_recorded_exactly_once(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Terminal means terminal: no second record, no second banner, no repeats."""
    window = CaptureWindow(frames=24)
    live(failed_alignments=set(range(1, 25)))

    run(window, stream_kernel, 24)

    assert states(window).count(TRACKING_LOST_STATE) == 1
    assert window.report.count("TRACKING LOST") == 1
    assert len(list(capture_of(window).glob("tracking-loss*"))) == 1


def test_the_previews_keep_running_and_stop_still_ends_the_capture(
    stream_kernel, capture_dir: Path, live
) -> None:
    """A frozen model must not mean a frozen window.

    The operator has to be able to see that the scanner is still delivering -
    otherwise the containment is indistinguishable from the crash it is
    preventing - and Stop has to be answered at the ordinary rate rather than
    at whatever rate a stalled capture happens to poll.
    """
    window = CaptureWindow(frames=30)
    live(failed_alignments=set(range(1, 31)))
    seen = [0]

    def press(_raw: object) -> None:
        seen[0] += 1
        if seen[0] == 12:
            window._stop()

    window.worker.preview.connect(press)
    run(window, stream_kernel, 30)

    # Deliveries continued well past the loss at frame 3, and the Stop pressed
    # on the twelfth of them was answered.
    assert seen[0] >= 12
    assert window.stopped == 1
    assert window.completed == []
    # Still only the four frames from before the loss.
    assert len(window.raw_frames_on_disk()) == 4


def test_stopping_after_a_loss_says_what_is_on_disk_and_does_not_offer_reconstruct(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The closing sentence is where the original failure was invisible.

    "N frames were already written and are safe to reconstruct" is true of a
    healthy stopped capture and wrong here: the frames are safe, and they are
    not one sweep.
    """
    window = CaptureWindow(frames=16)
    live(failed_alignments=set(range(1, 17)))
    seen = [0]

    def press(_raw: object) -> None:
        seen[0] += 1
        if seen[0] == 8:
            window._stop()

    window.worker.preview.connect(press)
    run(window, stream_kernel, 16)

    report = window.report
    assert "lost track of where the object was" in report
    assert "still on disk" in report
    assert TRACKING_LOSS_RECORD_NAME in report
    assert "REFUSE" in report
    assert "safe to reconstruct" not in report


def test_a_capture_that_runs_to_its_own_end_after_a_loss_ends_the_same_way(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The 2026-08-28 sweep ran on to its own stop. The ending has to be honest.

    Reaching the frame count is not recovery, so the completion path must not
    fall back to "Captured N frames... Press Reconstruct".
    """
    window = CaptureWindow(frames=10)
    live(failed_alignments=set(range(1, 11)))

    run(window, stream_kernel, 10)

    assert window.completed, "the capture should have reached its own end"
    assert "Press Reconstruct" not in window.report
    assert "lost track of where the object was" in window.report


# -- what the operator is told ----------------------------------------------


def test_the_operator_is_told_the_model_is_frozen_and_that_stop_is_the_safe_action(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Read literally, by somebody deciding whether to keep turning the turntable.

    Three things have to be in it: the model is frozen, the frames after it are
    not a continuation, and Stop is what to do. Anything less leaves the
    operator doing what the owner did - carrying on for another two minutes.
    """
    window = CaptureWindow(frames=12)
    live(failed_alignments={1, 2, 3})

    run(window, stream_kernel, 12)

    said = details(window)
    assert "TRACKING LOST" in said
    assert "FROZEN at frame 0" in said
    assert "NOT a valid continuation" in said
    assert "Press Stop" in said
    assert "safe on disk" in said
    # And it is on screen, not only in the log.
    assert window._pause_label.visible is True
    assert "TRACKING LOST" in window._pause_label.text
    assert window._scan_status_title.text == "SCAN FAILED — TRACKING LOST"
    assert window._scan_status_action.text == (
        "Next: Start a new scan; this folder cannot be reconstructed."
    )


def test_pause_is_disabled_and_relabelled_rather_than_left_offering_a_repair(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Pause places new frames against the accepted model. That is what broke.

    A control still reading "Pause" over a lost scan says "not right now", and
    what is true is "not at all". So it comes up, goes dead, and says why.
    """
    window = CaptureWindow(frames=12)
    live(failed_alignments={1, 2, 3})
    tooltips: list[str] = []
    window.worker.preview.connect(lambda _raw: tooltips.append(window._pause_action.tooltip))

    run(window, stream_kernel, 12)

    # What the control said WHILE the capture was lost, which is when it
    # matters. Both are asserted from the histories rather than from the end
    # state, because the end state is a capture that has finished and the
    # control is deliberately put back to neutral there.
    assert TRACKING_LOST_PAUSE_TITLE in window._pause_action.titles
    assert any("Unavailable" in tooltip and "Press Stop" in tooltip for tooltip in tooltips)
    assert window._pause_action.isChecked() is False
    assert window._pause_action.enabled is False
    # And once the capture is over the words describe a capture that is over.
    assert window._pause_action.text == PAUSE_TITLE
    assert window._pause_action.tooltip == PAUSE_TOOLTIP


def test_a_pause_pressed_after_a_loss_is_answered_rather_than_silently_ignored(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The toolbar disables it, so this is the route that goes round the toolbar.

    Answered rather than dropped: a control that does nothing at all is exactly
    what the operator will keep pressing.
    """
    window = CaptureWindow(frames=12)
    live(failed_alignments={1, 2, 3})
    seen = [0]

    def press(_raw: object) -> None:
        seen[0] += 1
        # Well after the loss at frame 3, and while the capture is still running.
        if seen[0] == 8:
            window.pause()

    window.worker.preview.connect(press)
    run(window, stream_kernel, 12)

    assert window._pause_action.isChecked() is False
    assert TRACKING_LOST_PAUSE_TITLE in window._pause_action.titles
    assert any("tracking is lost" in message for message in window._status.messages)
    # And the press changed nothing about the scan: no relocalization was
    # started and no further frame was written.
    assert len(window.raw_frames_on_disk()) == 4
    # The loss stays the last thing the capture published: the press produced
    # no "pausing at the next frame that aligns", which is the sentence that
    # would tell the operator to keep waiting.
    assert states(window)[-1] == TRACKING_LOST_STATE
    assert states(window).count(TRACKING_LOST_STATE) == 1


# -- the capture folder afterwards ------------------------------------------


def test_the_folder_a_lost_capture_leaves_is_refused_by_the_authoritative_reader(
    stream_kernel, capture_dir: Path, live
) -> None:
    """End to end: a live loss produces a folder the one reader will not hand over.

    Both reconstruction entry points - the CLI's ``reconstruct`` and the
    window's Reconstruct - go through this function and neither carries a check
    of its own, so this is where that refusal is asserted.
    """
    window = CaptureWindow(frames=14)
    live(failed_alignments={1, 2, 3})

    run(window, stream_kernel, 14)

    with pytest.raises(TrackingLossUnresolved) as caught:
        read_capture_sequence(capture_of(window))
    assert "frames up to and including frame 0" in str(caught.value)
    # The prefix and the evidence are both still there. Refusing is not deleting.
    assert len(window.raw_frames_on_disk()) == 4
    assert (capture_of(window) / TRACKING_LOSS_RECORD_NAME).is_file()


def test_a_capture_that_never_loses_tracking_reads_back_exactly_as_it_always_did(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The unaffected case, through the same path, so it cannot regress quietly."""
    window = CaptureWindow(frames=9)
    live()

    run(window, stream_kernel, 9)

    sequence = read_capture_sequence(capture_of(window))
    assert [frame.index for frame in sequence.frames] == list(range(9))
    assert sequence.bridges == ()
    assert sequence.limitations == ()


# -- a frame that cannot be placed at all -----------------------------------


def test_two_frames_that_deproject_to_nothing_are_a_dropout_and_not_a_loss(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly below the threshold, by the other route into it."""
    window = CaptureWindow(frames=6)
    live()
    tracking_input(monkeypatch, empty={1, 2})

    run(window, stream_kernel, 6)

    assert window._tracking_loss is None
    assert read_tracking_loss(capture_of(window)) is None
    assert len(window.raw_frames_on_disk()) == 6


def test_three_frames_that_deproject_to_nothing_lose_tracking(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-08-28 failure with a different cause and the same symptom.

    A frame that yields no usable cloud is never handed to alignment at all, so
    before this it extended nothing: the model stayed exactly as stationary as a
    run of singular ICP solves left it, and the capture would have gone on
    forever. The record says what actually happened rather than borrowing
    alignment's vocabulary for it.
    """
    window = CaptureWindow(frames=12)
    live()
    tracking_input(monkeypatch, empty={1, 2, 3})

    run(window, stream_kernel, 12)

    loss = read_tracking_loss(capture_of(window))
    assert loss is not None
    assert loss.last_aligned_frame_index == 0
    assert [failure.frame_index for failure in loss.failures] == [1, 2, 3]
    assert [f.failure_type for f in loss.failures] == [TRACKING_INPUT_EMPTY_TYPE] * 3
    assert all("no usable points" in failure.message for failure in loss.failures)
    # And it stopped: four frames on disk out of twelve delivered.
    assert len(window.raw_frames_on_disk()) == 4


def test_three_deprojections_that_raise_lose_tracking_and_never_leave_the_callback(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising deprojection used to escape a callback documented never to raise.

    It is now caught, counted as the tracking failure it is, and named as
    something other than an alignment failure - alignment was never attempted on
    these frames.
    """
    window = CaptureWindow(frames=12)
    live()
    tracking_input(monkeypatch, raising={1, 2, 3})

    run(window, stream_kernel, 12)

    loss = read_tracking_loss(capture_of(window))
    assert loss is not None
    assert [f.failure_type for f in loss.failures] == [TRACKING_INPUT_UNUSABLE_TYPE] * 3
    assert all("before alignment was attempted" in f.message for f in loss.failures)
    assert all("synthetic deprojection failure" in f.message for f in loss.failures)
    assert len(window.raw_frames_on_disk()) == 4


def test_one_usable_frame_resets_a_streak_of_unusable_ones(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two, recover, two, recover - and the two kinds mixed, because a streak is
    a streak whatever each failure was."""
    window = CaptureWindow(frames=8)
    live()
    tracking_input(monkeypatch, empty={1, 4}, raising={2, 5})

    run(window, stream_kernel, 8)

    assert window._tracking_loss is None
    assert read_tracking_loss(capture_of(window)) is None
    assert len(window.raw_frames_on_disk()) == 8


def test_a_failed_alignment_and_unusable_frames_make_one_streak(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three consecutive frames nothing could place, by three different routes.

    The streak counts frames that did not reach the model, not failures of one
    particular stage, because the operator's symptom is the same either way: a
    model that has stopped moving while the counter climbs.
    """
    window = CaptureWindow(frames=12)
    # Frame 1 is the only frame alignment is reached on here, since frames 2 and
    # 3 never produce a cloud to align.
    live(failed_alignments={1})
    tracking_input(monkeypatch, empty={2}, raising={3})

    run(window, stream_kernel, 12)

    loss = read_tracking_loss(capture_of(window))
    assert loss is not None
    assert [failure.frame_index for failure in loss.failures] == [1, 2, 3]
    assert [f.failure_type for f in loss.failures] == [
        # Not "ValueError": a frame no reference could place is classified by
        # the tracker, which knows which references it tried, rather than by
        # whichever exception one of those solves happened to raise.
        TRACK_NO_CANDIDATE_TYPE,
        TRACKING_INPUT_EMPTY_TYPE,
        TRACKING_INPUT_UNUSABLE_TYPE,
    ]
    assert len(window.raw_frames_on_disk()) == 4


def test_a_stop_landing_inside_a_frame_is_not_the_third_failure(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation is not a tracking failure and must never complete a streak.

    Two frames have genuinely failed. The third fails only because the operator
    pressed Stop in the middle of it - so a capture that counted it would tell
    the owner their scan lost tracking when what actually happened is that they
    stopped it.
    """
    window = CaptureWindow(frames=12)
    live(failed_alignments={1, 2})
    tracking_input(monkeypatch, stop_on=(window, 3))

    run(window, stream_kernel, 12)

    assert window._tracking_loss is None
    assert window._tracking_loss_unrecorded is None
    assert TRACKING_LOST_STATE not in states(window)
    assert read_tracking_loss(capture_of(window)) is None
    assert window.stopped == 1


# -- when the loss itself cannot be written ---------------------------------


def test_a_loss_that_cannot_be_recorded_fails_the_capture_instead_of_continuing(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one outcome worse than a lost capture is an unmarked lost capture.

    So a publication that fails does not leave the scan running and does not end
    it quietly: the capture stops and the job raises, which reaches the window's
    failure path and a dialog. The frames written before the loss are still
    untouched - failing loudly is not the same as undoing anything.
    """

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise TrackingLossPublicationRefused("the capture directory went read-only")

    monkeypatch.setattr(capture_sequence_module, "publish_tracking_loss", refuse)
    window = CaptureWindow(frames=20)
    live(failed_alignments=set(range(1, 21)))

    with pytest.raises(TrackingLossPublicationRefused, match="went read-only"):
        run(window, stream_kernel, 20)

    assert window.completed == []
    assert window.stopped == 0
    # Stopped at the loss rather than running on: only the four frames from
    # before it, and nothing after.
    assert len(window.raw_frames_on_disk()) == 4
    assert not (capture_of(window) / TRACKING_LOSS_RECORD_NAME).exists()
    assert "could NOT be written" in details(window)
    # The formal loss is never CLAIMED, because the thing it claims - a record
    # on disk - is not there. The state published says so in as many words, and
    # the window's recorded-loss field stays empty so no closing sentence sends
    # the operator to a file that does not exist.
    assert TRACKING_LOST_STATE not in states(window)
    assert states(window).count(TRACKING_LOSS_UNRECORDED_STATE) == 1
    assert window._tracking_loss is None
    assert window._tracking_loss_unrecorded is not None
    assert f"NO {TRACKING_LOSS_RECORD_NAME}" in details(window)
    # Pause is still taken away. Containment happened whatever the disk did.
    assert window._pause_action.text == TRACKING_LOST_PAUSE_TITLE
    assert not window._pause_action.enabled


def test_the_record_is_on_disk_before_the_loss_is_ever_reported(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Durable first, reported second - checked at the moment of the claim.

    A capture that announced the loss and then wrote the record would, for as
    long as the write took and forever if it failed, be telling the operator to
    go and read a file that is not there. So the disk is inspected from inside
    the slot that receives the state, which is the instant the claim is made.
    """
    window = CaptureWindow(frames=20)
    live(failed_alignments={1, 2, 3})
    seen: list[tuple[str, bool]] = []

    def watch(state: object) -> None:
        published = cast(gui.PauseState, state)
        seen.append((published.state, (capture_of(window) / TRACKING_LOSS_RECORD_NAME).exists()))

    window.worker.pause_state.connect(watch)

    run(window, stream_kernel, 20)

    assert (TRACKING_LOST_STATE, True) in seen
    assert not any(state == TRACKING_LOST_STATE and not existed for state, existed in seen)


def test_a_loss_that_cannot_be_recorded_still_stops_the_capture_writing(
    stream_kernel, capture_dir: Path, live, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment does not wait on the disk, and is not undone by it.

    The two halves are deliberately separable: stopping is always right and
    happens first, and REPORTING a loss is a claim about a file. This is the
    half that must hold either way - twenty frames delivered, four written, and
    nothing added to the model after the fourth.
    """

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise TrackingLossPublicationRefused("the capture directory went read-only")

    monkeypatch.setattr(capture_sequence_module, "publish_tracking_loss", refuse)
    window = CaptureWindow(frames=20)
    live(failed_alignments=set(range(1, 21)))

    with pytest.raises(TrackingLossPublicationRefused):
        run(window, stream_kernel, 20)

    assert saved(window) == [1, 2, 3, 4]
    assert len(window.raw_frames_on_disk()) == 4
    assert window.worker.coverage.emitted == []
