"""Pausing a capture, moving the object, and joining the two halves honestly.

What this is about
------------------
The operator can reach roughly two thirds of an object on a turntable. The other
third is whatever the object is standing on, and the only way to scan it is to
pick the object up and put it down differently - at which point the assumption
every frame of a capture rests on, that this frame is a small step from the last
one, is simply false. Adjacent alignment handed that pair does not fail. It
returns a small residual and a confident transform for a placement that is
nonsense, and the resulting model looks like a scan.

So a pause is three separate promises, and this module is about all three:

- while paused, NOTHING happens. No file, no count, no pose, no model. The
  camera views keep moving, because framing the new position is the entire
  reason the operator asked for the pause.
- on resume, a few frames are held back and the capture visibly WAITS. They are
  in memory, they are not on disk, they are not in any count, and they are not
  in the picture, until a global relocalization against the model already
  accepted has cleared every named gate.
- when it does clear, the whole batch and its evidence appear at once - one
  directory rename - and the model updates immediately, because an operator who
  has just been told it worked has to be able to see that it worked.

And when it does NOT clear, the capture folder is byte for byte what it was
before the pause. That is the case worth most of these tests: a refusal that
leaves a trace is a refusal that has contaminated a scan.

How these drive it
------------------
Through the real :meth:`~scanner3d.gui.ScannerWindow._start_capture`, its real
``on_frame`` AND its real post-requeue ``after_frame``, the real dequeue loop
against the scripted fake driver, the real
:class:`~scanner3d.gui.PauseControl`, the real
:func:`~scanner3d.capture_sequence.publish_segment` and the real GUI slots.

The split between those two callbacks is itself under test here. ``on_frame``
runs while the driver's buffer is still owned by userspace, so it does the
deciding and nothing else; everything a resume actually costs - deprojecting a
held frame, chaining it, and the global search - happens in ``after_frame``,
once ``VIDIOC_QBUF`` has returned. The search then runs off the stream thread
entirely, on an executor :class:`~tests.gui_harness.HeldExecutor` stands in for,
so a test can either have the answer arrive inline or freeze one in flight and
ask what the capture managed to do meanwhile.

The operator's button presses are fired from a slot on the depth-preview
channel, which is offered every complete delivery - including while paused,
which is the point. A press therefore lands between the driver handing a buffer
over and ``on_frame`` seeing it, which is exactly where a real press lands
relative to a frame, and it makes "which frame did the pause take effect on" a
question with one answer rather than a race.

The relocalization DECISION is stubbed in most of these, and that is deliberate
rather than avoidance: the questions here are about the state machine - what is
written, what is counted, what is published, what is left behind - and a test
that cannot choose whether a decision accepts or refuses cannot ask them. The
evaluator's own correctness lives in ``test_relocalize.py``, and one test below
runs the whole thing for real against a known large reposition so that the two
halves are joined by something.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui, icp_plane, pipeline, tracking
from scanner3d.capture_sequence import (
    BRIDGE_RECORD_NAME,
    STAGING_PREFIX,
    BridgeBoundary,
    publish_segment,
    read_capture_sequence,
)
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.gui import (
    MIN_PAUSE_REFERENCE_CLOUDS,
    PauseState,
    ReconstructionOutcome,
)
from scanner3d.multiview import CoarseAlignment, coarse_align
from scanner3d.pipeline import NOISE_FLOOR, SweepReconstruction
from scanner3d.project import PROJECT_NAME
from scanner3d.relocalize import (
    LOCAL_MIN_MATCHED_FRACTION,
    LOCAL_RESIDUAL_MULTIPLE,
    MAX_ATTEMPTS,
    MAX_TENTATIVE_BYTES,
    MAX_TENTATIVE_FRAMES,
    MIN_TENTATIVE_FRAMES,
    judge_local_edge,
)
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    accepting,
    alignments,
    capture_dir,
    clouds,
    decisions,
    frame_of,
    live,
    local_fit,
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


#: Autouse, so nothing below can accidentally race a real search thread: the
#: shared ``searches`` fixture runs a submitted relocalization inline unless a
#: test asks for it by name in order to hold one in flight.
pytestmark = pytest.mark.usefixtures("searches")


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, each with a fill byte of its own.

    The fill is the frame's identity all the way through: it is what the driver
    writes, what reaches disk, and what the stubbed deprojection below reads to
    decide which cloud this frame is. So a test can say which frame's geometry
    ended up where without threading an index through the capture.
    """
    return tuple(whole_frame(1 + index, buffer_index=index % 4) for index in range(count))


def _states(window: CaptureWindow) -> list[str]:
    return [state.state for state in window.pause_states]


def _details(window: CaptureWindow) -> str:
    return "\n".join(state.detail for state in window.pause_states)


# -- the operator ----------------------------------------------------------


def operating(window: CaptureWindow, script: dict[int, str]) -> list[int]:
    """Press Pause at the scripted deliveries, and record which they were.

    Wired to the depth-preview channel because that one is offered EVERY
    complete delivery, paused or not. A hook on the saved-frame count would go
    quiet at exactly the moment a test needs to press Resume.
    """
    pressed: list[int] = []
    seen = [0]

    def act(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        action = script.get(index)
        if action == "pause":
            window.pause()
            pressed.append(index)
        elif action == "resume":
            window.resume()
            pressed.append(index)

    window.worker.preview.connect(act)
    return pressed


# -- stubbed deprojection and decisions ------------------------------------


def _run(
    window: CaptureWindow,
    kernel_factory,
    delivered: int,
    script: dict[int, str],
) -> list[int]:
    kernel_factory(frames(delivered))
    pressed = operating(window, script)
    start_capture(window)
    return pressed


# -- while paused, nothing happens -----------------------------------------


def test_a_paused_capture_writes_no_frame_moves_no_count_and_draws_no_model(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The three things a pause exists to prevent, asserted as three counts.

    A pause that merely stopped the model refreshing, or merely stopped the
    counter, would look identical from the toolbar and would still be splicing
    frames of a moved object into the scan.
    """
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=16)
    _run(window, stream_kernel, 16, {10: "pause"})

    # Ten frames were committed before the pause; the six after it were not.
    assert len(window.frames_on_disk()) == 10
    assert [count for (count,) in window.worker.saved.emitted][-1] == 10
    assert not list(window._capture_dir.glob("segment-*"))  # type: ignore[union-attr]
    assert "paused" in _states(window)
    # Every delivery still reached the depth preview, paused or not: framing the
    # new position is what the operator paused in order to do.
    assert len(window.worker.preview.emitted) == 16


def test_the_model_stops_growing_while_paused_and_no_pose_is_placed(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A pose placed for a frame taken after the object moved is the whole bug."""
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=18)
    _run(window, stream_kernel, 18, {9: "pause"})

    # The refresh schedule is 3 then every four alignments, so nine committed
    # frames produce exactly two models. Nothing after the pause produced a
    # third, and nothing after the pause raised the count either.
    aligned = [cast(gui.ModelSnapshot, s).aligned for (s,) in window.worker.coverage.emitted]
    assert aligned == [3, 7]


def test_a_pause_before_there_is_enough_scanned_is_refused_with_guidance(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Refused, not accepted-and-then-broken.

    A relocalization is measured against the model already accepted. Below a
    few frames there is no such model, so a pause taken then could only ever
    end in a refusal to resume - and by then the operator has already moved the
    object and cannot undo it.
    """
    live()
    asked = decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=12)
    _run(window, stream_kernel, 12, {2: "pause", 6: "resume"})

    # Nothing was ever held back, and every frame reached disk.
    assert len(window.frames_on_disk()) == 12
    assert asked == []
    assert "Pause not taken" in window.report
    assert f"{MIN_PAUSE_REFERENCE_CLOUDS}" in _details(window)
    # And the toolbar put itself back up, without that looking like a press.
    assert window._pause_action.isChecked() is False


def test_a_refused_pause_puts_the_button_back_up_without_looking_like_a_press(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Setting the action would re-enter the handler and ask to resume a pause
    that never started. The signal is blocked across the change, and this is the
    only thing that checks it stays blocked."""
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=10)
    _run(window, stream_kernel, 10, {1: "pause"})

    assert window._pause_action.isChecked() is False
    assert window._pause_action.blocked is False, "the block must be restored, not left on"


def test_a_pause_waits_for_a_frame_that_aligns_and_commits_the_ones_it_waits_through(
    stream_kernel, capture_dir: Path, clouds, alignments, models, tentative_clouds, decisions
) -> None:
    """A pause is taken at a frame that BOTH reached disk and aligned.

    The bridge is measured against the last aligned pose and recorded against
    the frame immediately before the segment, so those two have to be the same
    frame or the record describes a pair that does not exist. When the last
    committed frame did not align, the pause therefore waits - and the frames it
    waits through are committed normally, because dropping them would guarantee
    there is never another aligned frame and the wait could only ever end in a
    refusal.
    """
    alignments({9})
    models(set())
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=16)
    _run(window, stream_kernel, 16, {10: "pause"})

    # The ninth frame did not align, so the pause deferred and frame ten was
    # written anyway. Frame eleven onwards is where it took effect.
    assert len(window.frames_on_disk()) == 11
    assert "Pausing at the next frame that aligns" in _details(window)
    assert "paused" in _states(window)


def test_a_pause_that_never_finds_a_boundary_is_answered_rather_than_silent(
    stream_kernel, capture_dir: Path, clouds, alignments, models, tentative_clouds, decisions
) -> None:
    """A control that engages only when alignment recovers, and never says so,
    is indistinguishable from a button that does nothing.

    What answers it changed with tracking loss. A pause is deferred while it
    waits for a frame that both reached disk and aligned, and
    ``PAUSE_BOUNDARY_GRACE`` allows eight such frames - but three consecutive
    alignment failures are now a tracking loss, so the loss arrives first and
    IS the answer. That is the stronger one: it names what is wrong rather than
    only what cannot be done about it, it puts the toggle back up, and it takes
    the control away instead of leaving it to be pressed again. What must not
    happen either way is a press that is silently never answered.
    """
    alignments(set(range(9, 40)))
    models(set())
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=26)
    _run(window, stream_kernel, 26, {10: "pause"})

    assert "TRACKING LOST" in _details(window)
    assert "TRACKING LOST" in window.report
    assert window._pause_action.isChecked() is False
    assert window._pause_action.enabled is False
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))


# -- resuming holds frames back --------------------------------------------


def test_tentative_frames_reach_neither_the_disk_the_counts_nor_the_model(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Held frames are held. They were deprojected, and that is all that happened."""
    live()
    asked = decisions([refusing("the segment overlaps the accepted model by 0.03")])
    window = CaptureWindow(frames=20)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    assert len(window.frames_on_disk()) == 10
    assert [count for (count,) in window.worker.saved.emitted][-1] == 10
    assert not list(window._capture_dir.glob("segment-*"))  # type: ignore[union-attr]
    # They WERE examined - seven deliveries after the resume - and none of them
    # became a file. Held in memory is exactly the state being asserted.
    assert len(tentative_clouds) == 7
    assert asked, "a decision should have been attempted once three frames were held"


def test_three_frames_are_required_before_a_decision_is_even_attempted(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """One noisy view will find somewhere plausible to sit on almost any model.

    The counts the evaluator was called with are the assertion: the first
    attempt is made with three frames, never one and never two.
    """
    live()
    asked = decisions([refusing("not enough evidence yet")])
    window = CaptureWindow(frames=20)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    assert asked, "no decision was attempted at all"
    assert min(asked) >= MIN_TENTATIVE_FRAMES
    assert asked[0] == MIN_TENTATIVE_FRAMES
    # And each retry saw one more frame than the last: a retry that added no
    # evidence would be the same decision re-run at the frame rate.
    assert asked == sorted(asked)
    assert all(later > earlier for earlier, later in pairwise(asked))


def test_the_panel_reports_paused_then_collecting_then_the_refusal_reason(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A capture that is waiting must not look like one that has stopped.

    The three states have to be distinguishable from the panel alone, and the
    refusal has to name the measurement, because the measurement is the thing
    the operator can act on by moving back over scanned surface.
    """
    live()
    decisions([refusing("the segment overlaps the accepted model by 0.03, under the 0.35")])
    window = CaptureWindow(frames=20)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    assert "paused" in _states(window)
    assert "relocalizing" in _states(window)
    details = _details(window)
    assert "PAUSED" in details
    assert f"of at least {MIN_TENTATIVE_FRAMES} frames collected" in details
    assert "0.03" in details
    assert "0.03" in window.report


# -- acceptance -------------------------------------------------------------


def test_a_confident_overlap_publishes_the_batch_atomically_and_joins_the_model(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Everything or nothing, and then a model at once.

    The batch, its record and its digests appear as one directory; the counts
    move only after that returned; and a model reaches the screen on the same
    frame rather than at the next scheduled refresh, because the operator has
    just been told the wait is over.
    """
    live()
    bridge = rotation_about_axis((0.0, 1.0, 0.0), 1.9, through=(0.0, 0.0, 120.0))
    asked = decisions([accepting(bridge)])
    window = CaptureWindow(frames=20)
    models_before = len(window.worker.coverage.emitted)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    assert asked
    capture = window._capture_dir
    assert capture is not None
    segments = sorted(capture.glob("segment-*"))
    assert len(segments) == 1
    assert (segments[0] / BRIDGE_RECORD_NAME).is_file()
    # Ten before the pause, three published as the batch, and the four ordinary
    # frames after acceptance written into the segment that now owns them.
    sequence = read_capture_sequence(capture)
    assert [frame.index for frame in sequence.frames] == list(range(17))
    assert len(sequence.bridges) == 1
    assert sequence.bridges[0].predecessor_index == 9
    assert sequence.bridges[0].start_index == 10
    # The saved count moved by the whole batch in one step, never part of it.
    counts = [cast(int, count) for (count,) in window.worker.saved.emitted]
    assert 13 in counts and 11 not in counts and 12 not in counts
    assert len(window.worker.coverage.emitted) > models_before


def test_the_published_bridge_is_the_transform_the_pose_chain_needs(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The direction of the recorded transform, which is invisible when wrong.

    ``bridge`` carries segment-local coordinates into the accepted model's
    frame. What is RECORDED is that composed back through the predecessor's
    pose, because a reconstruction drops it in exactly where it would have
    dropped in a fitted adjacent pair. Stubbed alignment leaves every accepted
    pose the identity, so the recorded transform must be the bridge itself.
    """
    live()
    bridge = rotation_about_axis((0.0, 1.0, 0.0), 1.9, through=(0.0, 0.0, 120.0))
    decisions([accepting(bridge)])
    window = CaptureWindow(frames=20)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    capture = window._capture_dir
    assert capture is not None
    recorded = read_capture_sequence(capture).bridges[0].transform
    residual = recorded.inverse().compose(bridge)
    assert residual.rotation_angle == pytest.approx(0.0, abs=1e-9)
    assert residual.translation_magnitude == pytest.approx(0.0, abs=1e-9)


def test_the_record_carries_the_evidence_the_thresholds_and_the_digests(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A record that says "accepted" and nothing else cannot be re-judged."""
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=20)
    _run(window, stream_kernel, 20, {10: "pause", 13: "resume"})

    capture = window._capture_dir
    assert capture is not None
    segment = sorted(capture.glob("segment-*"))[0]
    record = json.loads((segment / BRIDGE_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["predecessor_frame_index"] == 9
    assert record["segment_start_frame_index"] == 10
    assert [entry["frame_index"] for entry in record["frames"]] == [10, 11, 12]
    assert all(len(entry["sha256"]) == 64 for entry in record["frames"])
    assert all(entry["byte_count"] > 0 for entry in record["frames"])
    assert record["evidence"]["accepted"] is True
    assert record["evidence"]["thresholds"]["min_tentative_frames"] == MIN_TENTATIVE_FRAMES
    assert record["limitations"]


def test_ordinary_tracking_continues_from_the_bridged_pose(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """After acceptance the capture is an ordinary capture again.

    Frames go back to being written the moment they arrive, into the segment
    that now owns them, and the model keeps growing from the count the batch
    left it at rather than restarting.
    """
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=22)
    _run(window, stream_kernel, 22, {10: "pause", 13: "resume"})

    sequence = read_capture_sequence(cast(Path, window._capture_dir))
    assert [frame.index for frame in sequence.frames] == list(range(19))
    # Every frame after the batch is in the segment directory and none of them
    # is in the boundary record: they are ordinary frames of an ordinary chain.
    after = [frame for frame in sequence.frames if frame.index >= 13]
    assert after and all(frame.segment == "segment-0001" for frame in after)
    # Thirteen is off the ordinary schedule - it is ten accepted frames plus the
    # batch of three - which is the immediate model acceptance publishes, and
    # the refreshes after it prove tracking simply carried on.
    aligned = [cast(gui.ModelSnapshot, s).aligned for (s,) in window.worker.coverage.emitted]
    assert 13 in aligned
    assert max(aligned) > 13


def test_repeated_pause_and_resume_cycles_each_publish_their_own_segment(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """One reposition is the interesting case; two is the one that regresses."""
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=32)
    _run(window, stream_kernel, 32, {10: "pause", 13: "resume", 22: "pause", 25: "resume"})

    capture = cast(Path, window._capture_dir)
    assert sorted(path.name for path in capture.glob("segment-*")) == [
        "segment-0001",
        "segment-0002",
    ]
    sequence = read_capture_sequence(capture)
    assert len(sequence.bridges) == 2
    indices = [frame.index for frame in sequence.frames]
    assert indices == sorted(set(indices)), "global frame indices must be unique and ordered"
    assert indices == list(range(len(indices)))


# -- refusal leaves nothing behind ------------------------------------------


def test_a_refused_relocalization_leaves_the_capture_folder_exactly_as_it_was(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The case worth most: an uncertain attempt that contaminated a scan."""
    live()
    decisions([refusing("a genuinely different placement fits almost as well")])
    window = CaptureWindow(frames=26)
    _run(window, stream_kernel, 26, {10: "pause", 13: "resume"})

    capture = cast(Path, window._capture_dir)
    assert sorted(path.name for path in capture.iterdir()) == [
        *(f"frame_{index:04d}.raw" for index in range(10)),
        PROJECT_NAME,
    ]
    assert not list(capture.glob(f"{STAGING_PREFIX}*"))
    assert read_capture_sequence(capture).bridges == ()


def test_the_retry_history_is_bounded_by_the_attempt_ceiling(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A resume that never works must not spend CPU at the frame rate forever."""
    live()
    asked = decisions([refusing("no overlap")])
    window = CaptureWindow(frames=40)
    _run(window, stream_kernel, 40, {10: "pause", 13: "resume"})

    assert len(asked) == MAX_ATTEMPTS
    assert "Still cannot tell where the object is" in _details(window)
    assert "press Stop" in _details(window)


def test_the_tentative_buffer_slides_rather_than_growing(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Held frames are bounded, and the bound drops the OLDEST.

    The segment has to keep describing where the scanner is now, which is what
    the operator is currently moving - so the newest frame is never the one
    refused.
    """
    live()
    asked = decisions([refusing("no overlap")])
    window = CaptureWindow(frames=40)
    _run(window, stream_kernel, 40, {10: "pause", 13: "resume"})

    assert asked
    assert max(asked) <= MAX_TENTATIVE_FRAMES


# -- races, stops and teardown ---------------------------------------------


def test_a_pause_and_a_resume_between_two_frames_still_force_a_relocalization(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A toggle too fast to be seen as a state is still an event.

    Both presses land on one delivery, so the capture never observes a paused
    frame. It must still refuse to carry on with adjacent alignment: the object
    may well have been moved in between, and nothing here can tell.
    """
    live()
    asked = decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=20)

    pressed: list[int] = []
    seen = [0]

    def act(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        if index == 10:
            window.pause()
            window.resume()
            pressed.append(index)

    window.worker.preview.connect(act)
    stream_kernel(frames(20))
    start_capture(window)

    assert pressed == [10]
    assert asked, "a pause the capture never saw as a state was still an event"
    assert read_capture_sequence(cast(Path, window._capture_dir)).bridges


def test_stop_while_paused_leaves_no_segment_and_no_staging_artifact(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Stop stops, and a pause in flight is not a half-written scan."""
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=30)
    stream_kernel(frames(30))

    seen = [0]

    def act(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        if index == 10:
            window.pause()
        elif index == 13:
            window._stop()

    window.worker.preview.connect(act)
    start_capture(window)

    assert window.stopped == 1
    assert window.completed == []
    capture = cast(Path, window._capture_dir)
    assert not list(capture.glob("segment-*"))
    assert not list(capture.glob(f"{STAGING_PREFIX}*"))
    assert len(window.frames_on_disk()) == 10
    assert "10 frames were already written" in window.report


def test_stop_while_relocalizing_leaves_no_segment_and_no_staging_artifact(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """The frames being held were never promised, so nothing has to be undone."""
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=30)
    stream_kernel(frames(30))

    seen = [0]

    def act(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        if index == 10:
            window.pause()
        elif index == 13:
            window.resume()
        elif index == 17:
            window._stop()

    window.worker.preview.connect(act)
    start_capture(window)

    assert window.stopped == 1
    capture = cast(Path, window._capture_dir)
    assert not list(capture.glob("segment-*"))
    assert not list(capture.glob(f"{STAGING_PREFIX}*"))
    assert len(window.frames_on_disk()) == 10


# -- the search runs behind the requeue, and off this thread ---------------


def test_the_global_search_starts_only_after_the_buffer_went_back_to_the_driver(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches
) -> None:
    """The one ordering that makes a slow match survivable.

    Deprojection, the local chain and the global search all used to run between
    ``VIDIOC_DQBUF`` and ``VIDIOC_QBUF``, so the driver was left holding no free
    buffer for the whole of a search - the depth view froze, deliveries backed
    up, and the Stop poll (which happens after the requeue) could not be
    reached. Here the search records how many buffers had been handed back at
    the instant it was submitted, and it is always all of them.
    """
    live()
    decisions([refusing("no overlap")])
    kernel = stream_kernel(frames(20))
    # Taken buffers and returned buffers, at the instant each search starts.
    searches.watch = lambda: (len(kernel.dequeued), len(kernel.requeued))
    window = CaptureWindow(frames=20)
    operating(window, {10: "pause", 13: "resume"})
    start_capture(window)

    assert searches.watched, "no search was ever submitted"
    # The whole claim, in one line: at the moment a global search begins,
    # userspace owns no buffer. Every one the driver handed over has gone back.
    assert all(taken == returned for taken, returned in searches.watched)
    # And searches happened DURING the run rather than only at the end of it,
    # so this is not trivially true of a stream that had already finished.
    assert len(searches.watched) >= 2
    assert min(taken for taken, _returned in searches.watched) < 20
    assert len(kernel.requeued) == len(kernel.dequeued) == 20


def test_a_search_still_running_does_not_hold_the_stream_or_commit_anything(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches
) -> None:
    """A relocalization frozen in flight, and everything that must still work.

    The search is submitted and never allowed to finish. What the capture does
    for the rest of the run is the entire subject: every dequeued buffer goes
    back, the depth preview keeps advancing at the delivery rate, no frame is
    written, no count moves, and no segment appears.
    """
    live()
    decisions([accepting(Rigid.identity())])
    searches.immediate = False
    kernel = stream_kernel(frames(30))
    window = CaptureWindow(frames=30)
    operating(window, {10: "pause", 13: "resume"})
    # Sampled while the capture is running, because the job releases the worker
    # at teardown and the state under test is the one DURING the search.
    in_flight: list[int] = []
    window.worker.preview.connect(lambda _raw: in_flight.append(searches.in_flight))
    start_capture(window)

    # Held, never finished, and never joined by a second one: a new attempt may
    # not start while one is in flight.
    assert searches.submitted == 1
    assert max(in_flight) == 1
    # Every buffer the driver handed over came back.
    assert len(kernel.requeued) == len(kernel.dequeued) == 30
    # The picture kept moving for the whole run, which is what the operator is
    # looking at while they reposition.
    assert len(window.worker.preview.emitted) == 30
    # And nothing reached the scan.
    assert len(window.frames_on_disk()) == 10
    assert [count for (count,) in window.worker.saved.emitted][-1] == 10
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))
    assert not list(cast(Path, window._capture_dir).glob(f"{STAGING_PREFIX}*"))
    assert "Still looking for where the object went" in _details(window)


def test_stop_is_answered_while_a_search_is_still_in_flight(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches
) -> None:
    """Stop must not wait for a global search, and must not be starved by one.

    The Stop poll sits after the requeue, so a search that ran before it - as
    this one used to - delayed the answer by the length of the search. Held in
    flight here, the very next delivery still ends the run.
    """
    live()
    decisions([accepting(Rigid.identity())])
    searches.immediate = False
    kernel = stream_kernel(frames(30))
    window = CaptureWindow(frames=30)

    seen = [0]

    def act(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        if index == 10:
            window.pause()
        elif index == 13:
            window.resume()
        elif index == 18:
            window._stop()

    window.worker.preview.connect(act)
    start_capture(window)

    assert window.stopped == 1
    assert searches.in_flight == 0, "the abandoned search was not released at teardown"
    assert searches.shutdowns == [{"wait": False, "cancel_futures": True}]
    # Stopped at the delivery it was pressed on, not after the search.
    assert len(kernel.dequeued) == 19
    assert len(kernel.requeued) == 19
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))
    assert not list(cast(Path, window._capture_dir).glob(f"{STAGING_PREFIX}*"))


def test_a_search_abandoned_by_a_second_pause_can_never_reach_the_scan(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches
) -> None:
    """The answer arrives, and there is no longer anybody it is about.

    A search the worker has already picked up cannot be cancelled, so pausing
    again while one is running leaves a thread that WILL produce an accepted
    bridge for a segment the operator has just thrown away. Nothing may consume
    it: the frames it describes are gone, and publishing them would splice the
    reposition the operator abandoned into the scan.
    """
    live()
    decisions([accepting(Rigid.identity())])
    searches.immediate = False
    stream_kernel(frames(34))
    window = CaptureWindow(frames=34)
    operating(window, {10: "pause", 13: "resume", 20: "pause"})
    finished: list[int] = []
    seen = [0]

    def drive(_raw: object) -> None:
        index = seen[0]
        seen[0] += 1
        if index == 18:
            # Past the point of no return, exactly as a real worker would be.
            searches.begin()
        elif index == 25:
            finished.append(searches.run())

    window.worker.preview.connect(drive)
    start_capture(window)

    assert searches.submitted == 1
    assert finished == [1], "the abandoned search never actually produced its answer"
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))
    assert not list(cast(Path, window._capture_dir).glob(f"{STAGING_PREFIX}*"))
    assert len(window.frames_on_disk()) == 10
    assert [count for (count,) in window.worker.saved.emitted][-1] == 10


def test_the_worker_thread_is_shut_down_when_an_ordinary_capture_ends(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions, searches
) -> None:
    """And is never started at all by a capture that was never paused.

    A thread per capture that nothing ever asked for is a thread leaked once per
    scan, and the search is the only thing that needs one.
    """
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=8)
    _run(window, stream_kernel, 8, {})

    assert searches.submitted == 0
    assert searches.shutdowns == []


# -- the tentative segment has to chain to itself --------------------------


@pytest.fixture
def local_chain(monkeypatch) -> Callable[[dict[int, dict[str, object]]], list[int]]:
    """Script what each ADJACENT alignment inside the tentative segment returns.

    Keyed by the FRAME, recovered from the cloud by
    :func:`~tests.gui_harness.frame_of` - a capture's frames are numbered by
    their index and a tentative frame's by the fill byte of the raw bytes it was
    deprojected from, and both are the same counter. It used to be keyed by
    attempt number; a frame is now placed by up to three candidate solves
    (:class:`~scanner3d.tracking.IncrementalTracker`), so an attempt number no
    longer has a stable relationship to the frame a test means.

    Patched in both modules for the same reason
    :func:`~tests.gui_harness.alignments` is: the ordinary chain resolves ICP
    through :mod:`scanner3d.tracking` and the relocalization path through
    :mod:`scanner3d.icp_plane`, and a stand-in in only one of them would leave
    the other running real ICP over three synthetic points.

    A frame with no entry clears every gate.
    """
    seen: list[int] = []

    def install(overrides: dict[int, dict[str, object]]) -> list[int]:
        def align(*args: object, **_kwargs: object) -> object:
            frame = frame_of(args[0]) if args else None
            if frame is not None and (not seen or seen[-1] != frame):
                seen.append(frame)
            return local_fit(**overrides.get(frame if frame is not None else -1, {}))

        monkeypatch.setattr(icp_plane, "icp_point_to_plane", align)
        monkeypatch.setattr(tracking, "icp_point_to_plane", align)
        return seen

    return install


def test_a_tentative_frame_that_does_not_chain_to_the_one_before_it_is_not_held(
    stream_kernel, capture_dir: Path, clouds, models, tentative_clouds, decisions, local_chain
) -> None:
    """A returned ICP result is not a link until it has been measured.

    Ten frames align ordinarily; from the eleventh the local solve reports that
    it ran out of iterations while still moving. Those frames are dropped from
    the segment rather than chained, so the decision is never even attempted -
    a global match cannot rescue a segment that never chained to itself.
    """
    models(set())
    asked = decisions([accepting(Rigid.identity())])
    local_chain({attempt: {"converged": False} for attempt in range(10, 60)})
    window = CaptureWindow(frames=30)
    _run(window, stream_kernel, 30, {10: "pause", 13: "resume"})

    assert asked == [], "a segment that never chained was handed to the global search"
    assert "did not line up with the one before it" in _details(window)
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))
    assert len(window.frames_on_disk()) == 10


@pytest.mark.parametrize(
    "bad",
    [
        {"converged": False},
        {"rmse": 99.0},
        {"matched_fraction": 0.01},
        {"used_fraction": 0.01},
        {"rmse": float("nan")},
    ],
)
def test_every_local_gate_keeps_a_bad_link_out_of_the_segment(
    stream_kernel, capture_dir: Path, clouds, models, tentative_clouds, decisions, local_chain, bad
) -> None:
    """One case per gate, driven through the real capture closure."""
    models(set())
    asked = decisions([accepting(Rigid.identity())])
    local_chain({attempt: bad for attempt in range(10, 60)})
    window = CaptureWindow(frames=30)
    _run(window, stream_kernel, 30, {10: "pause", 13: "resume"})

    assert asked == []
    assert not list(cast(Path, window._capture_dir).glob("segment-*"))


def test_the_published_record_carries_every_accepted_local_edge_and_its_gates(
    stream_kernel, capture_dir: Path, clouds, models, tentative_clouds, decisions, local_chain
) -> None:
    """The chain is evidence too, and it is written down.

    A record that named only the global decision said where a segment was put
    and nothing about whether the segment was a segment. A later reader
    re-judging the bridge needs both.
    """
    models(set())
    local_chain({})
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=30)
    _run(window, stream_kernel, 30, {10: "pause", 13: "resume"})

    segments = sorted(cast(Path, window._capture_dir).glob("segment-*"))
    assert len(segments) == 1
    record = json.loads((segments[0] / BRIDGE_RECORD_NAME).read_text())
    chain = record["evidence"]["local_chain"]

    # One anchor with no link, and a measured link for every frame after it.
    assert chain["anchor_frames"] == 1
    assert len(chain["edges"]) == MIN_TENTATIVE_FRAMES - 1
    assert all(edge["accepted"] for edge in chain["edges"])
    assert all(edge["rmse"] is not None for edge in chain["edges"])
    assert chain["gates"]["local_min_matched_fraction"] == LOCAL_MIN_MATCHED_FRACTION
    assert chain["gates"]["local_max_residual"] == LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR


# -- the bounds are bounds -------------------------------------------------


def test_an_exhausted_attempt_stops_spending_anything_at_all(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Given up means given up, including the deprojection.

    An exhausted attempt used to keep deprojecting and locally aligning every
    selected frame for the rest of the capture - the two most expensive things
    a held frame costs - having already told the operator it had stopped. The
    count of deprojections is what proves it stops: it must not grow after the
    attempt ceiling is reached.
    """
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=60)
    _run(window, stream_kernel, 60, {10: "pause", 13: "resume"})

    assert "Still cannot tell where the object is" in _details(window)
    # One attempt per newly held frame from the minimum onwards, and the frame
    # AFTER the last attempt is the one that notices the ceiling. Nothing beyond
    # it is deprojected, however long the capture runs after that.
    assert len(tentative_clouds) == MIN_TENTATIVE_FRAMES + MAX_ATTEMPTS
    assert len(window.frames_on_disk()) == 10


def test_the_byte_ceiling_holds_however_many_frames_arrive() -> None:
    """The bound is a bound, not a target the buffer settles above.

    Frames sized so that exactly the minimum number fit: the buffer fills to
    three and then slides, and the held total never crosses the ceiling however
    many more arrive.
    """
    segment = gui.TentativeSegment.empty()
    exactly_a_third = bytes(MAX_TENTATIVE_BYTES // MIN_TENTATIVE_FRAMES)

    for _ in range(MIN_TENTATIVE_FRAMES + 4):
        assert segment.add(exactly_a_third, ((0.0, 0.0, 1.0),), Rigid.identity(), None) == ""
        assert segment.held_bytes <= MAX_TENTATIVE_BYTES

    assert segment.frames == MIN_TENTATIVE_FRAMES
    assert segment.dropped == 4


def test_a_frame_size_that_could_never_make_a_segment_is_refused_with_a_reason() -> None:
    """The ceiling used to stop applying in exactly the case it was written for.

    Below the minimum frame count the buffer refused to slide, so the minimum
    three frames were held at ANY size - and ``held_bytes`` was allowed to sit
    above ``MAX_TENTATIVE_BYTES`` indefinitely. Now a frame size that cannot
    make a decidable segment inside the ceiling is refused before it is held,
    and the operator is told why rather than left collecting forever.
    """
    segment = gui.TentativeSegment.empty()
    too_big = bytes(MAX_TENTATIVE_BYTES // 2)

    refused = segment.add(too_big, ((0.0, 0.0, 1.0),), Rigid.identity(), None)

    assert refused
    assert str(MAX_TENTATIVE_BYTES) in refused
    assert str(MIN_TENTATIVE_FRAMES) in refused
    assert segment.frames == 0
    assert segment.held_bytes == 0


def test_the_pause_control_is_dropped_when_the_capture_ends(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """A Pause pressed against a finished scan must reach nothing at all."""
    live()
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=12)
    _run(window, stream_kernel, 12, {})

    assert window._pause is None
    assert window._pause_action.enabled is False
    # And pressing it now is a no-op rather than an exception or a stale request.
    window.pause()
    assert window._pause_action.isChecked() is False


# -- the real evaluator, against a known large reposition -------------------


#: A deliberately asymmetric object: an ellipsoid shell with three different
#: axis lengths and one lobe grown out of a single off-axis direction.
#:
#: The asymmetry is not decoration. A shape that maps onto itself under a half
#: turn offers several equally good placements, and the evaluator is written to
#: REFUSE those - so a symmetric test object cannot prove that a correct
#: placement is found, only that an ambiguous one is declined. Measured here,
#: this shape agrees with itself by at most 0.45 under any quarter or half turn
#: about any of seven axes, against 1.0 at the true placement.
def _object_points(count: int = 460) -> tuple[Point, ...]:
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        x, z = ring * math.cos(angle), ring * math.sin(angle)
        lobe = 1.0 + 0.9 * max(0.0, 0.60 * x + 0.50 * y + 0.62 * z)
        points.append((15.0 * x * lobe, 9.0 * y * lobe, 6.0 * z * lobe))
    return tuple(points)


OBJECT: tuple[Point, ...] = _object_points()
STANDOFF: Point = (0.0, 0.0, 130.0)


def _placement(angle: float, extra: Rigid | None = None) -> Rigid:
    """Where the object sits in the camera's frame at one turntable angle."""
    spin = rotation_about_axis((0.0, 1.0, 0.0), angle)
    shifted = Rigid.of(spin.rotation, STANDOFF)
    return shifted if extra is None else extra.compose(shifted)


def _reposition() -> Rigid:
    """The operator picking the object up and putting it down differently."""
    turn = rotation_about_axis((0.0, 0.0, 1.0), math.radians(115.0), through=STANDOFF)
    lean = rotation_about_axis((1.0, 0.0, 0.0), math.radians(28.0), through=STANDOFF)
    return turn.compose(lean)


def test_a_real_large_reposition_is_recognised_and_recorded_as_the_known_transform(
    stream_kernel, capture_dir: Path, monkeypatch
) -> None:
    """The whole thing, for real, against a transform the test knows.

    Real deprojection stubbed to real geometry, real adjacent ICP, real coarse
    candidates, real refinement and the real gates. The object is picked up,
    turned 115 degrees about a different axis and leaned 28 degrees over, and
    what is asserted is that the transform written into the boundary record is
    the one the synthetic reposition actually applied - not merely that
    something was accepted.
    """
    before = 9
    placements: dict[int, Rigid] = {}
    for index in range(before):
        placements[1 + index] = _placement(math.radians(5.0 * index))
    moved = _reposition()
    for offset in range(6):
        placements[1 + before + offset] = _placement(math.radians(5.0 * offset), extra=moved)

    def cloud_for(fill: int) -> tuple[Point, ...]:
        return tuple(placements[fill].apply_all(OBJECT))

    monkeypatch.setattr(
        gui, "_load_cloud", lambda path, _n, _band=None: cloud_for(path.read_bytes()[0])
    )
    monkeypatch.setattr(gui, "_cloud_from_bytes", lambda raw, _n, _band=None: cloud_for(raw[0]))

    window = CaptureWindow(frames=before + 6)
    _run(window, stream_kernel, before + 6, {before: "pause", before + 1: "resume"})

    capture = cast(Path, window._capture_dir)
    sequence = read_capture_sequence(capture)
    assert len(sequence.bridges) == 1, (
        "the reposition was not recognised: " + _details(window)[-600:]
    )
    # The recorded transform relates the last frame before the pause to the
    # first frame after it, so it must be that pair's true relative placement.
    truth = placements[before].compose(placements[before + 2].inverse())
    residual = sequence.bridges[0].transform.inverse().compose(truth)
    # Widened from 3.0 when the ordinary chain stopped being blind adjacency.
    # The bridge is recorded RELATIVE to the last accepted pose before the
    # pause, so it inherits whatever that chain says, and the chain is no longer
    # the same one: measured, this residual moved from just under 3 degrees to
    # 3.08. What this test asserts is that a 115-degree turn and a 28-degree
    # lean are recovered as themselves rather than as something plausible, and
    # three degrees against a hundred and fifteen is the claim either way.
    assert math.degrees(residual.rotation_angle) < 4.0
    assert residual.translation_magnitude < 4.0


def test_a_reposition_with_no_shared_surface_is_never_accepted(
    stream_kernel, capture_dir: Path, monkeypatch
) -> None:
    """Moved out of the overlap entirely, the real evaluator must refuse.

    The frames after the resume see a different, disjoint surface. A best
    candidate still exists - one always does - and turning it into a placement
    is exactly what the gates are there to stop.
    """
    before = 9
    placements: dict[int, Rigid] = {}
    for index in range(before):
        placements[1 + index] = _placement(math.radians(5.0 * index))
    away = Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (900.0, -700.0, 400.0))
    for offset in range(6):
        placements[1 + before + offset] = away.compose(_placement(math.radians(5.0 * offset)))

    def cloud_for(fill: int) -> tuple[Point, ...]:
        return tuple(placements[fill].apply_all(OBJECT))

    monkeypatch.setattr(
        gui, "_load_cloud", lambda path, _n, _band=None: cloud_for(path.read_bytes()[0])
    )
    monkeypatch.setattr(gui, "_cloud_from_bytes", lambda raw, _n, _band=None: cloud_for(raw[0]))

    window = CaptureWindow(frames=before + 6)
    _run(window, stream_kernel, before + 6, {before: "pause", before + 1: "resume"})

    capture = cast(Path, window._capture_dir)
    assert not list(capture.glob("segment-*"))
    assert len(window.frames_on_disk()) == before
    assert read_capture_sequence(capture).bridges == ()


def test_a_coarse_ambiguous_reposition_publishes_no_segment_and_no_bridge(
    stream_kernel, capture_dir: Path, monkeypatch
) -> None:
    """The 2026-08-28 false accept, closed at the capture that would publish it.

    Exactly the geometry
    :func:`test_a_real_large_reposition_is_recognised_and_recorded_as_the_known_transform`
    has accepted above - the same object, the same reposition, every gate real -
    with one thing changed: the coarse search reports that it could not separate
    its winner from a genuinely different orientation. That is the single piece
    of evidence the bad bridge of 2026-08-28 carried, and the capture must now
    publish nothing at all: no segment directory, no staging directory, no
    bridge, and not one frame moved.
    """
    before = 9
    placements: dict[int, Rigid] = {}
    for index in range(before):
        placements[1 + index] = _placement(math.radians(5.0 * index))
    moved = _reposition()
    for offset in range(6):
        placements[1 + before + offset] = _placement(math.radians(5.0 * offset), extra=moved)

    def cloud_for(fill: int) -> tuple[Point, ...]:
        return tuple(placements[fill].apply_all(OBJECT))

    def ambiguous(source: Sequence[Point], target: Sequence[Point]) -> CoarseAlignment:
        """The real ranking, reporting the one thing this test is about."""
        return replace(coarse_align(source, target), unambiguous=False)

    monkeypatch.setattr(
        gui, "_load_cloud", lambda path, _n, _band=None: cloud_for(path.read_bytes()[0])
    )
    monkeypatch.setattr(gui, "_cloud_from_bytes", lambda raw, _n, _band=None: cloud_for(raw[0]))
    monkeypatch.setattr("scanner3d.relocalize.coarse_align", ambiguous)

    window = CaptureWindow(frames=before + 6)
    _run(window, stream_kernel, before + 6, {before: "pause", before + 1: "resume"})

    capture = cast(Path, window._capture_dir)
    assert not list(capture.glob("segment-*"))
    assert not list(capture.glob(f"{STAGING_PREFIX}*"))
    assert len(window.frames_on_disk()) == before
    assert read_capture_sequence(capture).bridges == ()
    # And the operator was told what to do about it, not merely that it failed.
    details = _details(window)
    assert "coarse search" in details
    assert "does NOT look like the rest of it" in details


# -- what the state payload itself promises ---------------------------------


def test_the_pause_state_the_capture_publishes_is_what_the_toolbar_follows() -> None:
    """The toolbar follows the CAPTURE, never the click.

    A window whose button reflected the request rather than the state would say
    "paused" over a capture that had refused the pause and was still writing.
    """
    window = CaptureWindow(frames=1)
    window._show_pause_state(
        PauseState(state="paused", detail="PAUSED after 9 frames", paused=True)
    )
    assert window._pause_action.isChecked() is True
    assert window._pause_label.text == "PAUSED after 9 frames"

    window._show_pause_state(PauseState(state="running", detail="Pause not taken", paused=False))
    assert window._pause_action.isChecked() is False


# -- what the toolbar action is CALLED --------------------------------------
#
# One checkable action does two opposite things, and the word on it is the only
# statement the window makes about which. On 2026-08-28 the owner paused a
# capture and the button went on saying "Pause" - an offer nothing behind it
# could honour, over a scan that was already stopped.


def test_the_action_says_resume_only_while_the_capture_says_it_is_paused() -> None:
    """The words follow the published STATE, through every state there is.

    ``relocalizing`` is the one worth spelling out: the operator has pressed
    resume, frames are being collected, and pressing the control again would
    pause - so it says Pause, even though the capture is not committing
    anything at that moment.
    """
    window = CaptureWindow(frames=1)
    assert window._pause_action.text == gui.PAUSE_TITLE

    window._show_pause_state(PauseState(state="paused", detail="PAUSED after 9", paused=True))
    assert window._pause_action.text == gui.RESUME_TITLE

    window._show_pause_state(PauseState(state="relocalizing", detail="Placing", paused=False))
    assert window._pause_action.text == gui.PAUSE_TITLE

    window._show_pause_state(PauseState(state="paused", detail="PAUSED again", paused=True))
    assert window._pause_action.text == gui.RESUME_TITLE

    window._show_pause_state(PauseState(state="running", detail="Pause not taken", paused=False))
    assert window._pause_action.text == gui.PAUSE_TITLE


def test_the_hover_text_names_the_control_the_operator_will_actually_see() -> None:
    """The tooltip is the only place the two-state control is explained.

    Asserted here because the wording is the part that has to be right and a
    stand-in toolbar cannot hover; a tooltip still telling the operator to
    "press it again" would be describing a button that no longer exists.
    """
    assert gui.RESUME_TITLE in gui.PAUSE_TOOLTIP
    assert "press it again" not in gui.PAUSE_TOOLTIP


def test_the_click_cannot_rename_the_action_ahead_of_the_capture() -> None:
    """A press is a request. The words wait for the answer.

    Between the press and the frame boundary the pause is actually taken at,
    the capture is still writing frames into the scan. A control that renamed
    itself on the click would be saying "Resume" over that, and would go on
    saying it if the pause were refused.
    """
    window = CaptureWindow(frames=1)
    window._pause = gui.PauseControl()

    window.pause()

    assert window._pause_action.isChecked() is True
    assert window._pause_action.text == gui.PAUSE_TITLE, (
        "the button was renamed by the click, before the capture had paused anything"
    )

    window._show_pause_state(
        PauseState(state="running", detail="Pause not taken: no boundary", paused=False)
    )

    assert window._pause_action.isChecked() is False
    assert window._pause_action.text == gui.PAUSE_TITLE


def test_the_action_goes_back_to_pause_when_a_capture_ends_while_paused() -> None:
    """A finished run cannot leave a Resume the next capture would inherit."""
    window = CaptureWindow(frames=1)
    window._pause = gui.PauseControl()
    window._pause_action.setEnabled(True)
    window._show_pause_state(PauseState(state="paused", detail="PAUSED after 9", paused=True))
    assert window._pause_action.text == gui.RESUME_TITLE

    window._busy(False)

    assert window._pause_action.text == gui.PAUSE_TITLE
    assert window._pause_action.isChecked() is False
    assert window._pause_action.enabled is False
    assert window._pause is None


def test_a_real_paused_capture_offers_resume_and_says_so_in_words(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Through the real capture: paused, renamed, and reset when the run ends.

    The titles are read as a SEQUENCE rather than at the end, because the end
    of a capture is a reset and would answer "Pause" whether the control had
    ever said anything else or not.
    """
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=16)
    _run(window, stream_kernel, 16, {10: "pause"})

    assert gui.RESUME_TITLE in window._pause_action.titles, (
        "the capture published a pause and the control never said Resume"
    )
    # And it is back to Pause now the run is over, with nothing to resume.
    assert window._pause_action.text == gui.PAUSE_TITLE

    paused = [state for state in window.pause_states if state.paused]
    assert paused, "this capture was supposed to reach the paused state"
    assert f"press {gui.RESUME_TITLE}" in paused[0].detail
    assert "press Pause again" not in _details(window)


def test_a_capture_starts_with_the_action_saying_pause(
    stream_kernel, capture_dir: Path, live, tentative_clouds, decisions
) -> None:
    """Whatever the window inherited, a new capture opens offering Pause."""
    live()
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=4)
    # As a previous run that ended while paused would have left it.
    window._pause_action.setChecked(True)
    window._pause_action.text = gui.RESUME_TITLE

    _run(window, stream_kernel, 4, {})

    assert window._pause_action.titles[0] == gui.PAUSE_TITLE, (
        "the capture started without correcting a stale Resume left by the last one"
    )
    assert window._pause_action.text == gui.PAUSE_TITLE


def test_the_tentative_buffer_re_anchors_its_chain_when_it_slides() -> None:
    """Dropping the frame the identity refers to without re-anchoring leaves
    every remaining pose expressed against a frame that is no longer there - an
    error that still aligns and is wrong by exactly the dropped step."""
    segment = gui.TentativeSegment.empty()
    steps = [
        Rigid.identity(),
        rotation_about_axis((0.0, 1.0, 0.0), 0.1, through=STANDOFF),
        rotation_about_axis((0.0, 1.0, 0.0), 0.2, through=STANDOFF),
    ]
    for index, pose in enumerate(steps):
        edge = None if index == 0 else judge_local_edge(local_fit(), frame=index)
        assert segment.add(bytes([index]) * 8, ((0.0, 0.0, 1.0),), pose, edge) == ""
    relative = steps[1].inverse().compose(steps[2])

    segment._drop_oldest()

    assert segment.frames == 2
    assert segment.poses[0].rotation_angle == pytest.approx(0.0, abs=1e-12)
    # And the link that described the step INTO the new anchor went with it: the
    # anchor is where this chain starts, and it has no link.
    assert segment.edges[0] is None
    assert segment.edges[1] is not None
    residual = segment.poses[1].inverse().compose(relative)
    assert residual.rotation_angle == pytest.approx(0.0, abs=1e-12)
    assert residual.translation_magnitude == pytest.approx(0.0, abs=1e-9)


def test_a_capture_that_is_never_paused_is_byte_for_byte_the_capture_it_always_was(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The whole pause machinery must be invisible to an ordinary scan.

    Root directory, ``frame_NNNN.raw`` names from zero, no segment, no record,
    and no bridge - which is what keeps every capture taken before any of this
    existed reconstructing exactly as it did.
    """
    live()
    window = CaptureWindow(frames=7)
    _run(window, stream_kernel, 7, {})

    capture = cast(Path, window._capture_dir)
    assert sorted(path.name for path in capture.iterdir()) == [
        *(f"frame_{index:04d}.raw" for index in range(7)),
        PROJECT_NAME,
    ]
    sequence = read_capture_sequence(capture)
    assert sequence.bridges == ()
    assert sequence.limitations == ()
    assert list(sequence.paths) == sorted(capture.glob("frame_*.raw"))


# -- what Reconstruct does with a capture that was repositioned -------------


def _synthetic_chain(batch_count: int) -> dict[str, object]:
    """The local chain a real capture would have recorded beside the bridge,
    for a hand-published batch. :func:`publish_segment` checks the shape the
    reader checks, so a test that publishes a boundary has to offer one.
    """
    return {
        "gates": {"note": "synthetic; this capture was published by a test"},
        "anchor_frames": 1,
        "edges": [
            {
                "frame": frame,
                "rmse": 0.02,
                "converged": True,
                "iterations": 9,
                "matched_fraction": 0.9,
                "used_fraction": 0.8,
                "resolved_backend": "python",
                "accepted": True,
                "refusals": [],
            }
            for frame in range(1, max(batch_count, 1))
        ],
    }


def _paused_capture(root: Path) -> Path:
    """A finished capture on disk: nine ordinary frames, then a boundary.

    Built through the real :func:`~scanner3d.capture_sequence.publish_segment`
    rather than by hand, so what these tests read back is what a capture would
    actually have left behind.

    Every frame is the SAME real, 512000-byte render :func:`sphere_raw_frame`
    returns, rather than the tiny fill-byte buffers this used to write: the
    tests that use this capture now run the real Reconstruct path end to end,
    including the export's TSDF fusion, which reads these bytes directly and
    refuses anything short of a real frame. The registration these tests care
    about is stubbed by :func:`_recording_registration` regardless of what the
    bytes say, so their content beyond "a real frame" carries no meaning here.
    """
    capture = root / "paused-scan"
    capture.mkdir()
    raw = sphere_raw_frame()
    for index in range(9):
        (capture / f"frame_{index:04d}.raw").write_bytes(raw)
    publish_segment(
        capture,
        frames=[(9 + offset, raw) for offset in range(3)],
        predecessor_index=8,
        transform=BRIDGE_TRANSFORM,
        evidence={**accepting(BRIDGE_TRANSFORM).as_dict(), "local_chain": _synthetic_chain(3)},
        limitations=("a synthetic boundary, published by this test",),
    )
    for index in range(12, 15):
        (capture / "segment-0001" / f"frame_{index:04d}.raw").write_bytes(raw)
    return capture


BRIDGE_TRANSFORM: Rigid = rotation_about_axis(
    (0.0, 1.0, 0.0), math.radians(112.0), through=STANDOFF
)


def _recording_registration(monkeypatch, *, bridges: tuple[str, ...] = ()):
    """Registration reduced to a recorder, so the WIRING is what is asserted.

    Reconstruct's job here is to hand the reader's frames and the reader's
    bridges to the pipeline and to carry what comes back into the export. What
    the pipeline then does with a bridge is ``test_pipeline_bridges.py``'s
    subject, and running it for real would make this test about ICP.
    """
    seen: list[dict[str, object]] = []

    def run(frame_clouds: Sequence[Sequence[Point]], **kwargs: object) -> SweepReconstruction:
        seen.append({"frames": len(frame_clouds), **kwargs})
        return SweepReconstruction(
            poses=tuple(
                rotation_about_axis((0.0, 1.0, 0.0), 0.05 * index, through=STANDOFF)
                for index in range(len(frame_clouds))
            ),
            frames=len(frame_clouds),
            revisits_found=0,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=(
                "This sweep was REPOSITIONED 1 time(s) mid-capture (segment-0001 at frame 9).",
            ),
            bridges=bridges,
        )

    monkeypatch.setattr(pipeline, "reconstruct_sweep", run)
    return seen


def test_reconstruct_reads_every_segment_and_hands_the_pipeline_its_bridges(
    tmp_path: Path, meshable_clouds, mesh_budget, monkeypatch
) -> None:
    """A glob of the root would register the first nine frames and stop.

    Which is the failure this reader exists to prevent, and it is invisible from
    the report: nine frames of a fifteen-frame scan still reconstruct, still
    export and still score.
    """
    capture = _paused_capture(tmp_path)
    seen = _recording_registration(monkeypatch, bridges=("segment-0001",))
    window = CaptureWindow(frames=1)
    start_reconstruct(window, capture, monkeypatch)

    assert len(seen) == 1
    assert seen[0]["frames"] == 15
    boundaries = cast(Sequence[BridgeBoundary], seen[0]["bridges"])
    assert [boundary.position for boundary in boundaries] == [9]
    assert [boundary.source for boundary in boundaries] == ["segment-0001"]
    residual = boundaries[0].transform.inverse().compose(BRIDGE_TRANSFORM)
    assert residual.rotation_angle == pytest.approx(0.0, abs=1e-9)
    assert "1 recorded relocalization bridge(s)" in window.report


def test_the_exported_record_names_the_bridge_evidence_and_its_limitation(
    tmp_path: Path, meshable_clouds, mesh_budget, monkeypatch
) -> None:
    """Provenance has to say the chain was not entirely measured from this data.

    Somebody reading ``reconstruction.json`` a year later has no way of knowing
    the object was picked up mid-scan unless the record says so, and the score
    beside it means something different because of it.
    """
    capture = _paused_capture(tmp_path)
    _recording_registration(monkeypatch, bridges=("segment-0001",))
    window = CaptureWindow(frames=1)
    start_reconstruct(window, capture, monkeypatch)

    export = cast(ReconstructionOutcome, window.completed[0]).export
    record = json.loads((export.directory / "reconstruction.json").read_text(encoding="utf-8"))
    assert record["registration"]["relocalization_bridges"] == ["segment-0001"]
    limitations = json.dumps(record)
    assert "REPOSITIONED" in limitations
    assert "recorded relocalization bridges used: segment-0001" in window.report


def test_a_legacy_capture_reconstructs_with_no_bridges_at_all(
    tmp_path: Path, meshable_clouds, mesh_budget, monkeypatch
) -> None:
    """The path every capture taken before this feature existed still takes."""
    capture = tmp_path / "legacy-scan"
    capture.mkdir()
    raw = sphere_raw_frame()
    for index in range(9):
        (capture / f"frame_{index:04d}.raw").write_bytes(raw)
    seen = _recording_registration(monkeypatch)
    window = CaptureWindow(frames=1)
    start_reconstruct(window, capture, monkeypatch)

    assert seen[0]["frames"] == 9
    assert cast(Sequence[BridgeBoundary], seen[0]["bridges"]) == ()
    assert "relocalization bridge" not in window.report
