"""Whether live feedback stays bounded, and describes what is actually on screen.

The owner's 2026-08-26 report was that the "scanned so far" model updated two or
three times and then sat still for far longer than the interval it is refreshed
on. Offline replay of that exact capture proved the geometry was fine: all 55
frames aligned, and the model payloads at 3, 15, 27, 39 and 51 aligned frames
all built successfully, in 0.008 s to 0.200 s. So nothing downstream of the
scanner failed, and nothing upstream of the window was slow.

That localizes the problem downstream of geometry production, and what sits
there is the boundary between the worker and the window. It was unbounded in
two ways:

- Every one of those payloads was built from EVERY frame aligned so far, so it
  cost more and carried more the longer the capture ran, and crossed the view's
  own 60,000-point ceiling on a full-length sweep - after being assembled in
  full, copied through a queued signal, and retained.
- A Qt queued connection is a queue. Depth frames, the saved-frame heartbeat and
  the model all went into it with no backpressure at all, so a window that fell
  behind did not skip - it accumulated, and then drew a backlog of pictures that
  had each been true several seconds earlier.

Neither is established as the sole cause of what the owner saw, and nothing here
claims it: no trace of that run's event loop was taken, and one cannot be
reconstructed after the fact. Both are real, both are cheap to bound, and both
would look like a freeze from outside the process - so they are corrected, and
the produced/applied/superseded evidence proved below is what makes the NEXT
physical report answerable instead of arguable.

This drives the real capture path - the real ``_start_capture``, its real
``on_frame``, the real dequeue loop against the scripted fake driver, and the
real GUI slots - and expresses "the window cannot keep up" the only way an
offline test honestly can: by not answering the wake-ups until it chooses to.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.cloud_view import MAX_RENDER_POINTS
from scanner3d.geometry import Rigid
from scanner3d.gui import (
    COVERAGE_EVERY,
    LIVE_MODEL_CLOUDS,
    LIVE_MODEL_POINTS,
    MIN_COVERAGE_CLOUDS,
    ModelSnapshot,
    _live_snapshot,
    _sample_span,
)
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


def _snapshots(window: CaptureWindow) -> list[ModelSnapshot]:
    return [cast(ModelSnapshot, snapshot) for (snapshot,) in window.worker.coverage.emitted]


# -- one slot, not a queue -------------------------------------------------


def test_a_flood_of_updates_costs_the_window_one_wake_up_and_one_draw() -> None:
    """The channel on its own, with nobody answering it.

    A hundred payloads offered while the GUI is busy leave one wake-up posted
    and one payload waiting - the hundredth. The other ninety-nine are counted
    as superseded rather than silently forgotten, because "the model was never
    produced" and "the model was produced and could not be drawn in time" have
    entirely different causes and the log has to be able to tell them apart.
    """
    wakes: list[gui.LatestOnly] = []
    drawn: list[int] = []
    channel = gui.LatestOnly("test", wakes.append)
    channel.connect(drawn.append)

    for value in range(100):
        channel.emit(value)

    assert len(wakes) == 1
    assert drawn == []
    channel.deliver()

    assert drawn == [99]
    assert (channel.offered, channel.applied, channel.superseded) == (100, 1, 99)


def test_a_channel_the_window_keeps_up_with_drops_nothing() -> None:
    """The same channel, answered every time: every payload is drawn, in order.

    Coalescing is a consequence of the consumer being slow, never a policy the
    producer applies unconditionally. A window that keeps up sees everything.
    """
    channel = gui.LatestOnly("test", lambda woken: woken.deliver())
    drawn: list[int] = []
    channel.connect(drawn.append)

    for value in range(5):
        channel.emit(value)

    assert drawn == [0, 1, 2, 3, 4]
    assert (channel.offered, channel.applied, channel.superseded) == (5, 5, 0)


def test_an_empty_slot_is_delivered_without_complaint() -> None:
    """Two wake-ups either side of a drain is normal, not an error."""
    channel = gui.LatestOnly("test", lambda _woken: None)
    drawn: list[int] = []
    channel.connect(drawn.append)

    channel.emit(1)
    channel.deliver()
    channel.deliver()

    assert drawn == [1]
    assert channel.applied == 1


def test_a_discarded_payload_is_never_drawn_by_the_wake_up_it_left_behind() -> None:
    """The slot emptied while its wake-up is still outstanding.

    The wake-up is deliberately NOT withdrawn - a queued event is not something
    a producer can take back - so the guarantee has to come from the slot being
    empty by the time it is answered. Drawing nothing is the assertion.
    """
    wakes: list[gui.LatestOnly] = []
    drawn: list[int] = []
    channel = gui.LatestOnly("test", wakes.append)
    channel.connect(drawn.append)

    channel.emit(7)
    assert channel.discard() is True
    assert len(wakes) == 1
    wakes[0].deliver()

    assert drawn == []
    assert (channel.offered, channel.applied, channel.discarded) == (1, 0, 1)


def test_discarding_an_empty_slot_takes_nothing_back() -> None:
    """A payload already applied stays applied, and nothing is counted twice.

    ``discard`` runs on every teardown, including the overwhelming majority
    where the window kept up perfectly. It must be a no-op there, and it must
    not describe a picture the operator really did see as one that was dropped.
    """
    channel = gui.LatestOnly("test", lambda woken: woken.deliver())
    drawn: list[int] = []
    channel.connect(drawn.append)

    channel.emit(3)
    assert channel.discard() is False
    assert channel.discard() is False

    assert drawn == [3]
    assert (channel.offered, channel.applied, channel.discarded) == (1, 1, 0)


def test_a_full_capture_into_a_busy_window_never_queues_more_than_one_each(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Sixty frames produced against a window that draws none of them.

    The assertion that matters is the SECOND one: whatever the capture produces,
    what is outstanding for the GUI is at most one wake-up per channel. Before
    this, sixty depth frames were sixty queued payloads, each holding a copy of
    half a megabyte, and the operator watched them arrive minutes later.
    """
    total = 60
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total, deferred=True)
    live()

    start_capture(window)

    assert len(window.frames_on_disk()) == total
    assert len(window.worker.woken) <= len(window.worker.channels())
    assert window.worker.saved.offered == total
    assert window.worker.saved.applied == 0

    window.worker.drain()

    # One draw each, and the newest value in every one of them.
    assert window.worker.saved.applied == 1
    assert window._saved_frames == total
    assert window.worker.preview.applied == 1
    assert window.worker.saved.superseded == total - 1


def test_the_heartbeat_is_never_behind_the_frames_on_disk(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models
) -> None:
    """The saved count the operator reads is the number of files that exist.

    The window is given a turn on every frame, in the middle of the live-model
    work, which is where a real one gets its turns. At each of those moments the
    heartbeat must already be showing the frame that has just been written -
    that is what makes it a heartbeat rather than another thing that lags.
    """
    total = 20
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total, deferred=True)
    alignments(set())
    models(set())
    seen: list[tuple[int, int]] = []

    def load(
        path: Path, _target_points: int, _depth_band: object = None
    ) -> tuple[tuple[float, float, float], ...]:
        # The GUI gets a slice of time, exactly as it would between frames.
        window.worker.drain(window.worker.saved)
        seen.append((len(window.frames_on_disk()), window._saved_frames))
        base = float(int(path.stem.split("_")[1]))
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)

    start_capture(window)

    assert len(seen) == total
    assert [written for written, _shown in seen] == [shown for _written, shown in seen]


def test_the_panel_describes_the_model_it_is_showing_not_the_frame_that_just_landed(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models
) -> None:
    """The 2026-08-26 counter, reattached to the picture it belongs to.

    The model built after three aligned frames is not drawn until the last frame
    of the sweep - which is what happens whenever the window is busy. The panel
    must then say "rebuilt after 3", because 3 is the model on the screen.
    Reading the live saved count instead made it claim a rebuild that had not
    happened, over a picture several frames old.
    """
    total = MIN_COVERAGE_CLOUDS + COVERAGE_EVERY
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total, deferred=True)
    alignments(set())
    models(set())

    #: The panel, read at the instant the twelve-frame-old model is drawn.
    drawn: list[str] = []

    def load(
        path: Path, _target_points: int, _depth_band: object = None
    ) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        # The heartbeat is cheap and always gets through; the model is the
        # expensive one and is not drawn until the last frame of the sweep.
        window.worker.drain(window.worker.saved)
        if index == total - 1:
            window.worker.drain(window.worker.coverage)
            drawn.append(window._capture_progress.text)
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)

    start_capture(window)

    assert window._model_refreshed_at == MIN_COVERAGE_CLOUDS
    assert drawn == [
        f"Capturing: {total} frames saved. The model below was last rebuilt after "
        f"{MIN_COVERAGE_CLOUDS} of them (1 rebuild)."
    ]
    assert window._coverage_summary.text.startswith(
        f"preview after {MIN_COVERAGE_CLOUDS} aligned frames of {MIN_COVERAGE_CLOUDS} saved -"
    )


# -- the model itself is bounded -------------------------------------------


def test_the_sampled_frames_span_the_whole_sweep() -> None:
    """Not the first forty, not the last forty: forty spread across all of it.

    Both cheaper bounds answer the model panel's question wrongly. Keeping the
    beginning shows an object that stops halfway; keeping the end forgets the
    half already scanned and sends the operator back over it.
    """
    poses = [Rigid.identity() for _ in range(500)]
    clouds = [((float(index), 0.0, 100.0),) for index in range(500)]

    sampled, sampled_poses = _sample_span(clouds, poses, LIVE_MODEL_CLOUDS)

    picked = [int(cloud[0][0]) for cloud in sampled]
    assert len(sampled) == LIVE_MODEL_CLOUDS
    assert len(sampled_poses) == LIVE_MODEL_CLOUDS
    assert picked[0] == 0
    assert picked[-1] == 499
    assert picked == sorted(set(picked))
    # Every fifth of the sweep is represented, which is what "spans it" means.
    for fifth in range(5):
        low, high = fifth * 100, (fifth + 1) * 100
        assert any(low <= index < high for index in picked), f"nothing from frames {low}-{high}"


def test_a_sweep_shorter_than_the_bound_is_not_sampled_at_all() -> None:
    """Below the ceiling nothing is thrown away, and the order is preserved."""
    poses = [Rigid.identity() for _ in range(4)]
    clouds = [((float(index), 0.0, 100.0),) for index in range(4)]

    sampled, sampled_poses = _sample_span(clouds, poses, LIVE_MODEL_CLOUDS)

    assert sampled == clouds
    assert len(sampled_poses) == 4


def test_the_live_model_payload_never_exceeds_what_the_view_can_draw(monkeypatch) -> None:
    """Bounded where it is BUILT, below the view's own ceiling.

    The view decimating a payload it has already been handed is too late: by
    then the full array has been assembled, copied across a thread boundary and
    retained. Bounding it here is what stops it existing at all.
    """
    points = np.arange(3 * 200_000, dtype=np.float32).reshape(-1, 3)
    thin = np.zeros(len(points), dtype=bool)
    thin[::2] = True
    newest = np.zeros(len(points), dtype=bool)
    newest[-len(points) // 4 :] = True
    monkeypatch.setattr(
        gui, "_coloured_cloud", lambda _clouds, _poses: (points, thin, 0.25, newest)
    )

    snapshot = _live_snapshot([((0.0, 0.0, 1.0),)], [Rigid.identity()], saved=9, aligned=7)

    assert len(snapshot.points) <= LIVE_MODEL_POINTS
    assert len(snapshot.points) <= MAX_RENDER_POINTS
    # Both flag arrays stay parallel to the points, or the colouring is
    # meaningless - and the newest frame survives the decimation rather than
    # being thinned out of existence by it.
    assert len(snapshot.thin) == len(snapshot.points)
    assert len(snapshot.newest) == len(snapshot.points)
    assert snapshot.newest.any()
    assert snapshot.aligned == 7
    assert snapshot.saved == 9


def test_the_model_build_stops_growing_with_the_capture(
    stream_kernel, capture_dir: Path, clouds: list[int], alignments, monkeypatch
) -> None:
    """A rebuild two hundred frames in costs what a rebuild forty frames in cost.

    The measured 0.008 s -> 0.200 s growth over 51 frames is not a constant
    factor problem, it is the shape of the work: rebuilding from EVERY frame so
    far means the refresh falls further behind the capture the longer the
    capture runs. Counting the frames handed to the build is the deterministic
    form of that measurement - no clock, and no machine-speed dependence.
    """
    total = 200
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    alignments(set())
    sizes: list[int] = []
    spans: list[tuple[float, float]] = []

    def build(built_from: list[object], poses: list[object]) -> object:
        sizes.append(len(built_from))
        first = cast(tuple[tuple[float, float, float], ...], built_from[0])
        last = cast(tuple[tuple[float, float, float], ...], built_from[-1])
        spans.append((first[0][0], last[0][0]))
        count = len(built_from) * gui.PREVIEW_POINTS
        return (
            np.zeros((count, 3), dtype=np.float32),
            np.zeros(count, dtype=bool),
            0.5,
            np.zeros(count, dtype=bool),
        )

    monkeypatch.setattr(gui, "_coloured_cloud", build)

    start_capture(window)

    assert len(window.frames_on_disk()) == total
    # Several rebuilds, spread over the whole capture rather than only its start.
    rebuilt_at = [MIN_COVERAGE_CLOUDS + step * COVERAGE_EVERY for step in range(len(sizes))]
    assert rebuilt_at[-1] > total - COVERAGE_EVERY, "the rebuilds stopped before the sweep did"
    assert max(sizes) == LIVE_MODEL_CLOUDS
    # And every one of them still reached from the first aligned frame to the
    # newest one it had, however far apart those had grown.
    assert [first for first, _last in spans] == [0.0] * len(spans)
    assert [last for _first, last in spans] == [float(aligned - 1) for aligned in rebuilt_at]
    assert all(len(snapshot.points) <= LIVE_MODEL_POINTS for snapshot in _snapshots(window))


def test_reconstruction_still_builds_from_every_frame_and_every_point() -> None:
    """The bound is on the preview only. Reconstruct is the number to trust.

    ``_coloured_cloud`` is the function the Reconstruct path calls directly, and
    it is deliberately not the one that samples or decimates. A hundred frames
    in gives a hundred frames' worth of points out.
    """
    total = LIVE_MODEL_CLOUDS * 3
    clouds = [
        ((float(index) * 0.1, 0.0, 100.0), (0.0, float(index) * 0.1, 100.0))
        for index in range(total)
    ]
    poses = [Rigid.identity() for _ in range(total)]

    points, thin, _covered, newest = gui._coloured_cloud(clouds, poses)

    assert len(points) == total * 2
    assert len(thin) == total * 2
    # The build flags the last frame it was given whatever the caller intends to
    # do with that; what a REGISTERED model does with it is
    # ``test_gui_model_view.py``'s subject, and the answer there is "discard it".
    assert len(newest) == total * 2


# -- the evidence a physical report needs ----------------------------------


def test_produced_applied_and_superseded_are_three_separable_log_lines(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models, caplog
) -> None:
    """The three explanations of "it froze", told apart in the log.

    A model that was never PRODUCED, one that was produced and APPLIED, and one
    produced and then SUPERSEDED before the window could draw it are three
    different faults with three different fixes, and the previous log recorded
    none of them. Nothing here is per-frame: the model lines happen once every
    ``COVERAGE_EVERY`` alignments and the counters once per capture.
    """
    total = MIN_COVERAGE_CLOUDS + 3 * COVERAGE_EVERY
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total, deferred=True)
    alignments(set())
    models(set())

    def load(
        path: Path, _target_points: int, _depth_band: object = None
    ) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        window.worker.drain(window.worker.saved)
        # Drawn once, early. Every later snapshot is superseded by the next.
        if index == MIN_COVERAGE_CLOUDS:
            window.worker.drain(window.worker.coverage)
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)

    with caplog.at_level(logging.INFO, logger="scanner3d.gui"):
        start_capture(window)

    messages = [record.getMessage() for record in caplog.records]
    produced = [message for message in messages if message.startswith("live model produced")]
    applied = [message for message in messages if message.startswith("live model applied")]
    summary = [message for message in messages if message.startswith("live feedback - live model")]
    assert len(produced) == 4
    assert len(applied) == 1
    assert summary == [
        "live feedback - live model: 4 produced, 1 applied, "
        "2 superseded before they could be drawn, 0 dropped when the stream ended"
    ]
    # The timing is in the produced line, which is what makes a report about a
    # slow rebuild answerable without asking for another capture.
    assert " in 0." in produced[0]
    # Nothing here fires per frame: that is the whole reason it is at INFO.
    assert len(messages) < total
