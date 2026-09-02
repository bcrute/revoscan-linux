"""Headless stand-ins for the window, shared by every capture- and preview-path test.

Why a stand-in rather than a window
-----------------------------------
The questions these tests ask - what reaches disk, which outcome a job ends in,
what the model panel says - are all answered by :class:`ScannerWindow`'s own
logic, and a Qt event loop contributes nothing to answering them while making
the tests unable to run on a machine with no display server. So every method
under test is called unbound against the object below, which carries exactly the
attributes that path touches and no others: a capture that reaches for something
it should not have raises ``AttributeError`` instead of passing quietly.

Why ONE of them
---------------
For the reason there is one :mod:`tests.fake_v4l2`: a second stand-in written
for a second question would be a second set of assumptions about what the real
window does, and the two would drift. The capture path is now exercised by
short-frame tests, stop tests and live-model tests, and all three need the same
object.

:class:`PreviewWindow` is the second face of the same idea and lives here for the
same reason: preview and capture share the device wait, the colour resolution and
the Stop that lands before either opens a stream, so a stand-in kept next to a
test module would drift away from the one the capture tests use.

What is faithful here, and what is not
--------------------------------------
:meth:`CaptureWindow._run` is Qt's worker thread collapsed to a call, and it
reproduces the one thing about :class:`~scanner3d.gui.Worker` that the outcome
depends on: a job that raises :class:`~scanner3d.gui.Stopped` goes to
``_was_stopped``, and anything else it returns goes to the completion callback.
Any OTHER exception is deliberately left to escape, because "an exception got
out of the job" is itself a failure these tests exist to catch.

The channels are the REAL :class:`~scanner3d.gui.LatestOnly` ones, subclassed
only to remember what the producer offered. That matters: the coalescing they do
is the behaviour several of these tests are about, and a stand-in that merely
recorded emissions would prove nothing about it. What the stand-in supplies is
the other half - the wake-up. By default it delivers immediately, which is a GUI
that always keeps up and is what every test written before coalescing existed
assumes. ``deferred=True`` is the opposite: wake-ups pile up unanswered until
:meth:`FakeWorker.drain` is called, which is a GUI too busy to draw, and is how a
flood is expressed without a Qt event loop or a clock.

``saved`` and ``coverage`` are wired to the real slots, which is what makes the
model panel's own text testable. ``preview`` and ``travel`` are recorded but not
wired, because their slots build a ``QImage`` and set styled widget text - screen
work, and not what any of this is asking about.
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import numpy as np
import pytest

from scanner3d import gui, icp_plane, pipeline, reconstruction_export, tracking
from scanner3d.capture_sequence import read_capture_sequence
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_plausibility import StaticRegionTracker
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.gui import ScannerWindow
from scanner3d.pipeline import SweepReconstruction
from scanner3d.preparation import PreparedCloud
from scanner3d.preview_provenance import ModelPicture, SurfaceEvidence
from scanner3d.reconstruction_readiness import (
    DEFAULT_POLICY,
    CoverageEvidence,
    PoseEvidence,
    ReconstructionReadiness,
    SurfaceAgreement,
    TrackingEvidence,
)
from scanner3d.relocalize import (
    LIMITATIONS,
    MIN_TENTATIVE_FRAMES,
    CandidateEvidence,
    LocalFit,
    Relocalization,
)
from scanner3d.return_health import ReturnHealthTracker
from scanner3d.simulate import FACTORY_INTRINSICS, Sphere, render

__all__ = [
    "CaptureWindow",
    "FakeSignal",
    "FakeWorker",
    "HeldExecutor",
    "PreviewWindow",
    "PreviewWorker",
    "RecordingChannel",
    "accepting",
    "alignments",
    "capture_dir",
    "clouds",
    "decisions",
    "deprojected",
    "live",
    "local_fit",
    "mesh_budget",
    "meshable_clouds",
    "models",
    "previewing",
    "refusing",
    "registration",
    "relocalization_candidate",
    "searches",
    "shell_cloud",
    "sphere_raw_frame",
    "stand_in_surface",
    "start_capture",
    "start_reconstruct",
    "stub_readiness",
    "tentative_clouds",
]

#: What a stubbed :func:`~scanner3d.gui._coloured_cloud` returns: points, the
#: per-point "needs more scanning" flags, the well-covered fraction, and the
#: per-point "came from the newest frame" flags. Four values because the build
#: returns four; a stub that returned three would pass every test here and fail
#: the moment the real one ran.
MODEL_PAYLOAD = (
    np.zeros((2, 3), dtype=np.float32),
    np.array([True, False]),
    0.5,
    np.array([False, True]),
)


class HeldExecutor(Executor):
    """The relocalization worker thread, replaced by one the test drives.

    A resume's global search runs off the stream thread, which is the whole
    point of it - but "off the thread" and "at some unpredictable later moment"
    are the same thing to a test, and a suite that raced the search would assert
    whatever the machine happened to do that run.

    So this stands in for it. By default a submitted search runs INLINE, on the
    calling thread, the instant it is submitted: every existing test then sees
    the decision it scripted arrive at the frame it scripted it for, and what is
    being tested is the state machine rather than the scheduler.

    Set :attr:`immediate` false and the search is HELD instead - submitted,
    never started - which is what lets a test freeze a relocalization in flight
    and ask what the capture did while it was: whether the buffer went back to
    the driver, whether previews advanced, whether anything was written, and
    whether Stop still worked. :meth:`run` then finishes what is waiting.
    """

    def __init__(self, *, immediate: bool = True) -> None:
        self.immediate = immediate
        #: ``[future, call, started]`` per submitted search. ``started`` is what
        #: makes :meth:`begin` meaningful: a search the worker has picked up can
        #: no longer be cancelled, which is the state an abandoned-mid-flight
        #: test needs and the one a plain pending future cannot reproduce.
        self._waiting: list[list[object]] = []
        #: How many searches have ever been submitted.
        self.submitted = 0
        #: Called at each submit, and its answer recorded, so a test can say
        #: what the driver's buffer state was at the instant the search started.
        self.watch: Callable[[], object] | None = None
        self.watched: list[object] = []
        #: Every ``shutdown`` this was asked for, with its arguments.
        self.shutdowns: list[dict[str, object]] = []

    def submit(
        self,
        fn: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> Future:
        future: Future = Future()
        self.submitted += 1
        if self.watch is not None:
            self.watched.append(self.watch())
        self._waiting.append([future, lambda: fn(*args, **kwargs), False])
        if self.immediate:
            self.run()
        return future

    @property
    def in_flight(self) -> int:
        """Searches submitted and not yet finished."""
        return len(self._waiting)

    def begin(self) -> int:
        """Mark every waiting search as RUNNING, without finishing it.

        A real worker that has already picked a search up cannot be cancelled,
        and that is the case worth testing: the capture abandons it instead, and
        the answer that arrives anyway has to reach nobody.
        """
        started = 0
        for item in self._waiting:
            future = cast(Future, item[0])
            if not item[2] and future.set_running_or_notify_cancel():
                item[2] = True
                started += 1
        return started

    def run(self) -> int:
        """Finish every waiting search, on the CALLING thread. Returns how many."""
        ran = 0
        while self._waiting:
            future, call, started = self._waiting.pop(0)
            future = cast(Future, future)
            if not started and not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(cast(Callable[[], object], call)())
            except BaseException as exc:  # a task that raises is a result, not a crash
                future.set_exception(exc)
            ran += 1
        return ran

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})
        if not cancel_futures:
            return
        # Exactly what a real executor cancels: what has not started. A search
        # already running is left alone, which is why a test that called
        # :meth:`begin` still finds it here afterwards.
        kept: list[list[object]] = []
        for item in self._waiting:
            if item[2] or not cast(Future, item[0]).cancel():
                kept.append(item)
        self._waiting = kept


class FakeSignal:
    """One Qt signal: records every emission, and calls what is connected."""

    def __init__(self) -> None:
        self.emitted: list[tuple[object, ...]] = []
        self._slots: list[Callable[..., object]] = []

    def connect(self, slot: Callable[..., object]) -> None:  # Qt's name
        self._slots.append(slot)

    def emit(self, *args: object) -> None:  # Qt's name
        self.emitted.append(args)
        for slot in self._slots:
            slot(*args)


class RecordingChannel(gui.LatestOnly):
    """A real coalescing channel that also remembers what was OFFERED to it.

    ``emitted`` is the producer's side, so it stays the same list it always was
    whether or not the consumer kept up. What the consumer actually applied is
    read from the slots wired to it, or from the channel's own ``applied``
    counter - and the gap between the two is the thing worth asserting.
    """

    def __init__(self, name: str, notify: Callable[[gui.LatestOnly], None]) -> None:
        super().__init__(name, notify)
        self.emitted: list[tuple[object, ...]] = []

    def emit(self, *payload: object) -> None:  # Qt's name
        self.emitted.append(payload)
        super().emit(*payload)


class FakeWorker:
    """The worker a job is handed, with no thread behind it.

    ``deferred`` decides what the GUI is doing while the job produces. False is
    a window that draws every wake-up the instant it is posted; True is one that
    is busy, so wake-ups queue in :attr:`woken` until :meth:`drain` answers them
    - which is the only way an offline test can be slower than its producer
    without sleeping.
    """

    def __init__(self, *, deferred: bool = False) -> None:
        self.cancelled = False
        self.running = False
        self.cancels = 0
        self.deferred = deferred
        #: Channels woken and not yet drained. At most one entry per channel,
        #: which is the coalescing this exists to make visible.
        self.woken: list[gui.LatestOnly] = []
        self.preview = RecordingChannel("depth preview", self._wake)
        self.colour = RecordingChannel("colour preview", self._wake)
        self.coverage = RecordingChannel("live model", self._wake)
        self.saved = RecordingChannel("saved-frame count", self._wake)
        self.travel = RecordingChannel("rotation speed", self._wake)
        self.pause_state = RecordingChannel("pause state", self._wake)
        self.scan_status = RecordingChannel("scan status", self._wake)
        self.progress = FakeSignal()

    def channels(self) -> tuple[gui.LatestOnly, ...]:
        return (
            self.preview,
            self.colour,
            self.coverage,
            self.saved,
            self.travel,
            self.pause_state,
            self.scan_status,
        )

    def _wake(self, channel: gui.LatestOnly) -> None:
        if self.deferred:
            self.woken.append(channel)
            return
        channel.deliver()

    def drain(self, *only: gui.LatestOnly) -> None:
        """The GUI finally getting a turn. Applies one payload per wake-up.

        Naming channels drains just those and leaves the rest waiting, which is
        the realistic shape of the problem: the cheap heartbeat gets drawn while
        the expensive model is still being rendered.
        """
        taken = [channel for channel in self.woken if not only or channel in only]
        self.woken = [channel for channel in self.woken if channel not in taken]
        for channel in taken:
            channel.deliver()

    def cancel(self) -> None:
        self.cancels += 1
        self.cancelled = True

    def check(self) -> None:
        """The real :meth:`~scanner3d.gui.Worker.check`, raise and all.

        Faithful because the raising is the behaviour under test: a job calls
        this only where unwinding is safe, and a stand-in that merely returned
        would let a capture the operator stopped carry on into the stream.
        """
        if self.cancelled:
            raise gui.Stopped

    def isRunning(self) -> bool:  # Qt's name
        return self.running


class FakeSpin:
    def __init__(self, value: int) -> None:
        self._value = value

    def value(self) -> int:  # Qt's name
        return self._value


class FakeCheck:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:  # Qt's name
        return self._checked

    def setChecked(self, checked: bool) -> None:  # Qt's name
        self._checked = checked


class FakeCombo:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:  # Qt's name
        return self._text

    def setCurrentText(self, text: str) -> None:  # Qt's name
        self._text = text


class FakeAction:
    #: Whether signals are currently blocked. The window blocks them across a
    #: programmatic change of the Pause toggle - a pause the capture REFUSED has
    #: to put the button back up without that looking like the operator pressing
    #: it - and a stand-in that silently accepted ``blockSignals`` would let that
    #: guarantee rot without a test noticing.
    blocked: bool

    #: What the action is currently CALLED. Held because the Pause control is
    #: one checkable action whose two states need two different words, and a
    #: stand-in with no title at all could not answer the one question the
    #: 2026-08-28 report was about: whether a paused capture's button says
    #: "Resume".
    text: str

    def __init__(self, checked: bool = False, enabled: bool = True, text: str = "") -> None:
        self.checked = checked
        self.enabled = enabled
        self.blocked = False
        self.text = text
        self.checks: list[bool] = []
        #: Every title the window has set, in order, so a test can ask what the
        #: control said DURING a sequence and not merely where it ended up.
        self.titles: list[str] = []
        #: What the control says when hovered. Held for the same reason the
        #: title is: after a tracking loss the tooltip is where the operator is
        #: told that Pause cannot repair it and Stop is what they want.
        self.tooltip = ""

    def isChecked(self) -> bool:  # Qt's name
        return self.checked

    def setChecked(self, on: bool) -> None:  # Qt's name
        self.checks.append(on)
        self.checked = on

    def setText(self, text: str) -> None:  # Qt's name
        self.titles.append(text)
        self.text = text

    def blockSignals(self, on: bool) -> bool:  # Qt's name
        was, self.blocked = self.blocked, on
        return was

    def setEnabled(self, on: bool) -> None:  # Qt's name
        self.enabled = on

    def setToolTip(self, text: str) -> None:  # Qt's name
        self.tooltip = text


class FakeLabel:
    #: Whether the window has shown or hidden this label. The pause line is
    #: present only while a pause is in play, and a stand-in that ignored
    #: ``setVisible`` would let a permanent "not paused" line creep back in.
    visible: bool

    #: The last stylesheet set on this label. The depth panel's return-health
    #: line says how serious it is with colour as well as words, and a stand-in
    #: that dropped ``setStyleSheet`` would let a warning go quiet unnoticed.
    style: str

    def __init__(self) -> None:
        self.text = ""
        self.visible = False
        self.style = ""
        self.tooltip = ""

    def setVisible(self, on: bool) -> None:  # Qt's name
        self.visible = on

    def setText(self, text: str) -> None:  # Qt's name
        self.text = text

    def setStyleSheet(self, style: str) -> None:  # Qt's name
        self.style = style

    def setToolTip(self, text: str) -> None:  # Qt's name
        self.tooltip = text


class FakeCloud:
    """The 3D view, reduced to what it was last handed.

    Keeps the framing request alongside the payload, because "did this update
    move the camera" is a question about the window rather than about the
    widget, and the widget's own answer to it is
    :mod:`tests.test_gui_model_view`'s subject.
    """

    def __init__(self) -> None:
        self.clouds: list[tuple[np.ndarray, np.ndarray]] = []
        #: The newest-frame flags each update carried, in order.
        self.newest: list[np.ndarray | None] = []
        #: Whether each update asked the view to re-frame itself, in order.
        self.fits: list[bool] = []
        self.thin_fraction = 0.0
        #: Every surface this was handed, in order, and every time it was told
        #: there is not one. Two lists rather than one with ``None`` holes,
        #: because "a surface arrived" and "the surface was withdrawn" are
        #: different events and a test asserting one should not have to filter
        #: the other out of a single list.
        self.surfaces: list[tuple[np.ndarray, np.ndarray]] = []
        self.surface_clears = 0
        #: Which picture the window last asked for, or None if it never has.
        self.show_surface: bool | None = None
        #: Whether a surface is currently HELD, as the real viewer's
        #: ``has_surface`` answers it: set by a surface arriving and unset by a
        #: withdrawal. Held rather than derived from the two lists above,
        #: because those record the whole history and this is the state.
        self.holds_surface = False
        #: Whether any measured points are held, likewise.
        self.holds_cloud = False

    def set_cloud(
        self,
        points: np.ndarray,
        thin: np.ndarray,
        newest: np.ndarray | None = None,
        *,
        fit: bool = False,
    ) -> None:
        self.clouds.append((points, thin))
        self.newest.append(newest)
        self.fits.append(fit)
        self.thin_fraction = float(np.count_nonzero(thin)) / max(1, len(thin))
        self.holds_cloud = bool(len(points))

    def set_surface(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.surfaces.append((vertices, faces))
        # An empty face list is how the real view is told there is no surface,
        # and it answers ``has_surface`` false afterwards - see
        # :meth:`scanner3d.cloud_view.CloudCamera.set_surface`.
        self.holds_surface = bool(len(faces))

    def clear_surface(self) -> None:
        self.surface_clears += 1
        self.holds_surface = False

    def set_show_surface(self, showing: bool) -> None:
        self.show_surface = showing

    @property
    def has_surface(self) -> bool:
        return self.holds_surface

    @property
    def has_cloud(self) -> bool:
        return self.holds_cloud

    @property
    def showing_surface(self) -> bool:
        """What the REAL viewer answers: asked for AND there to be drawn.

        The window's provenance line is derived from this rather than from the
        checkbox, so a stand-in that answered the request instead of the state
        would let the very confusion this seam prevents pass unnoticed.
        """
        return bool(self.show_surface) and self.holds_surface


class FakeReport:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:  # Qt's name
        self.clears += 1


class FakeStatusBar:
    """The status bar, reduced to what it was told to say.

    Wired up because "where did my scan go" is answered in two places at once -
    the report and the status bar - and a window that names the folder in only
    one of them is a window that names it where the operator was not looking.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str, _timeout: int = 0) -> None:  # Qt's name
        self.messages.append(message)


class CaptureWindow:
    """The parts of :class:`ScannerWindow` a capture actually touches."""

    def __init__(
        self,
        frames: int,
        decimate: int = 1,
        *,
        previewing: bool = False,
        deferred: bool = False,
        serial: str | None = None,
        workers: int = 1,
        hold_cadence: bool = False,
    ) -> None:
        self.logged: list[str] = []
        self.busy: list[bool] = []
        self.worker = FakeWorker(deferred=deferred)
        #: What :meth:`_scanner_serial` answers. ``None`` by default, and that
        #: default is load-bearing: it is what stops every capture test in this
        #: suite from starting a colour pump against whatever scanner happens to
        #: be plugged into the machine running them.
        self.serial = serial
        #: Colour frames the window was handed, in order.
        self.colours: list[bytes] = []
        self.result: object = None
        #: Every value handed to the COMPLETION callback. Empty is the assertion
        #: for a stopped capture: completion must not have been reached at all.
        self.completed: list[object] = []
        self.stopped = 0
        self._preview_action = FakeAction(checked=previewing)
        self._pause_action = FakeAction(text=gui.PAUSE_TITLE)
        self._stop_action = FakeAction()
        self._pause_label = FakeLabel()
        self._scan_statuses: dict[str, gui.ScanStatus] = {
            gui.IDLE_SCAN_STATUS.source: gui.IDLE_SCAN_STATUS
        }
        self._scan_status_title = FakeLabel()
        self._scan_status_summary = FakeLabel()
        self._scan_status_action = FakeLabel()
        self._scan_metrics = FakeLabel()
        # The REAL return-health window and the real labels' stand-ins: a
        # capture that failed to start its own history would inherit the last
        # one's collapse, which is what these make answerable.
        self._return_health = ReturnHealthTracker()
        self._return_verdict: str | None = None
        self._return_label = FakeLabel()
        self._return_advice = FakeLabel()
        #: The value half of frame health, beside the presence half. Both
        #: windows carry it because both draw depth frames, and
        #: ``_show_plausibility`` runs on the same slot that draws them.
        self._static_regions = StaticRegionTracker()
        self._said_plausibility = ""
        self._plausibility_label = FakeLabel()
        #: The real run identity, so a capture that ends without moving it on
        #: - and therefore leaves the last run's frames able to reach this
        #: one's panel - fails here rather than passing quietly.
        self._depth_run = 0
        #: Every depth frame that actually reached the panel, in order. A stale
        #: delivery that was dropped is an absence in this list.
        self.depth_frames: list[bytes] = []
        #: The model panel's Surface/Points toggle. A FakeAction because the
        #: only things the window does to it are enable, disable and tick it as
        #: a surface appears and goes away, which is exactly that stand-in's
        #: subject - blocked signals included, since the window moves this one
        #: without meaning the operator moved it.
        #:
        #: Down and disabled to begin with, as the real window builds it: at
        #: construction there is no surface for it to be describing.
        self._surface_toggle = FakeAction(checked=False, enabled=False)
        #: The operator's own last choice of picture, which the window keeps
        #: separately so that a surface going away does not silently reset it.
        #: Taken from the window's own constant rather than repeated, so this
        #: stand-in cannot go on describing a default the product has changed.
        self._surface_wanted = gui.SURFACE_SELECTED_BY_DEFAULT
        #: The model panel's provenance line and the surface numbers it may use.
        #: Real labels are not needed to ask what the window SAYS, but the state
        #: is: the sentence has to be derivable from the mounted picture at every
        #: point in a capture, not only at the end of one.
        self._model_provenance = FakeLabel()
        self._surface_evidence: SurfaceEvidence | None = None
        #: The running capture's pause request, made by the real
        #: :meth:`ScannerWindow._start_capture` and dropped when it ends.
        self._pause: gui.PauseControl | None = None
        #: What the capture said when it lost tracking, or ``None``. The window
        #: owns this and several of its slots read it, so a stand-in without it
        #: would fail on the first frame rather than answer the question.
        self._tracking_loss: str | None = None
        #: What the capture said when it lost tracking and could NOT record it.
        #: A separate field for a separate claim; see
        #: :data:`~scanner3d.gui.TRACKING_LOSS_UNRECORDED_STATE`.
        self._tracking_loss_unrecorded: str | None = None
        self._aligned_frames = 0
        #: Every pause state the capture published, in order. The panel's own
        #: words, which is what a test about a refusal or a wait has to read.
        self.pause_states: list[gui.PauseState] = []
        self._frames_spin = FakeSpin(frames)
        self._decimate_spin = FakeSpin(decimate)
        #: Whether the capture is allowed to move the spacing above while it
        #: runs. Off by default, exactly as the window's own control is, so
        #: every existing capture test still describes a fixed cadence.
        self._hold_cadence = FakeCheck(hold_cadence)
        #: Reconstruct's worker count. Here rather than in a stand-in of its
        #: own because Reconstruct shares this window with a capture - the same
        #: report, the same busy state and the same model slot - and a second
        #: stand-in would be a second set of assumptions about all three.
        self._workers = FakeSpin(workers)
        self._backend = FakeCombo("python")
        self._report = FakeReport()
        self._status = FakeStatusBar()
        self._capture_progress = FakeLabel()
        self._coverage_summary = FakeLabel()
        self._cloud = FakeCloud()
        self._capture_dir: Path | None = None
        self._resume_preview_after = False
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at: int | None = None
        self._coverage_well: float | None = None
        self._coverage_needs: float | None = None
        self._capturing = False
        self._worker: FakeWorker | None = None
        self.awaited = 0
        #: Every serial the device wait was asked for, in order. A capture that
        #: waited for a scanner other than the one colour is aimed at is the
        #: two-scanner bench failing quietly, so it has to be answerable.
        self.awaited_serials: list[str | None] = []

    # -- what the window does to itself ------------------------------------
    def _log(self, message: str) -> None:
        self.logged.append(message)

    def _busy(self, on: bool) -> None:
        """Recorded, and then the one thing the real ``_busy`` does that matters.

        The action-enabling half is Qt bookkeeping and is left out, but the
        Pause reset is not: it is what stops a Pause left checked by the capture
        that just ended from being inherited by the next one, and a stand-in
        that skipped it would let that regress silently.
        """
        self.busy.append(on)
        if not on:
            self._end_pause()

    def _stop_preview(self) -> None:
        return None

    def _scanner_serial(self) -> str | None:
        """Answered from the stand-in, never from the machine's own sysfs."""
        return self.serial

    def _reset_return_health(self) -> None:
        """The REAL one: a capture that skipped it would inherit the last run's history."""
        ScannerWindow._reset_return_health(cast(ScannerWindow, self))

    def _begin_depth_run(self) -> None:
        """The REAL one, for the same reason, run identity included."""
        ScannerWindow._begin_depth_run(cast(ScannerWindow, self))

    def _end_depth_run(self) -> None:
        """The REAL one: a capture that ended without clearing its warning would pass."""
        ScannerWindow._end_depth_run(cast(ScannerWindow, self))

    def _frame_slot(self, run: int) -> Callable[[bytes], None]:
        """The REAL one, so a stale delivery is dropped here exactly as it is there."""
        return ScannerWindow._frame_slot(cast(ScannerWindow, self), run)

    def _show_frame(self, raw: bytes) -> None:
        """The depth slot, reduced to what reached it."""
        self.depth_frames.append(raw)

    def statusBar(self) -> FakeStatusBar:  # Qt's name
        return self._status

    def _show_colour(self, jpeg: bytes) -> None:
        """The colour slot, reduced to what it was handed."""
        self.colours.append(jpeg)

    def _await_device(self, serial: str | None) -> None:
        self.awaited += 1
        self.awaited_serials.append(serial)

    def _run(self, job: Callable[[object], object], done: Callable[[object], None]) -> None:
        """Qt's thread, collapsed to a call, keeping only its outcome routing.

        Deliberately catches nothing but :class:`~scanner3d.gui.Stopped`: an
        exception escaping a job is a failure these tests are looking for, and
        swallowing it here would turn a regression into a silent pass.
        """
        self._worker = self.worker
        # Connected for THIS run, exactly as the real ``_run`` does it, so a
        # frame belonging to the preview this capture replaced is dropped here
        # rather than recorded.
        self.worker.preview.connect(self._frame_slot(self._depth_run))
        self.worker.saved.connect(self._show_saved)
        self.worker.coverage.connect(self._show_coverage)
        self.worker.colour.connect(self._show_colour)
        self.worker.pause_state.connect(self._show_pause_state)
        self.worker.scan_status.connect(self._show_scan_status)
        # The real ``_run`` wires progress to the report panel, and the pause
        # path is the first one whose important sentences - the refusals, the
        # bounded give-up - travel that way rather than through a coalescing
        # channel. A stand-in that dropped them would make those sentences
        # untestable and, worse, would make them look untested.
        self.worker.progress.connect(self._log)
        self.worker.running = True
        try:
            self.result = job(self.worker)
        except gui.Stopped:
            self.stopped += 1
            self._was_stopped()
            return
        finally:
            self.worker.running = False
        self.completed.append(self.result)
        done(self.result)

    # -- delegated to the real ones ----------------------------------------
    def _capture_name(self) -> str | None:
        return ScannerWindow._capture_name(cast(ScannerWindow, self))

    def _capture_target(self) -> Path | None:
        return ScannerWindow._capture_target(cast(ScannerWindow, self))

    def _capture_done(self, result: object) -> None:
        ScannerWindow._capture_done(cast(ScannerWindow, self), result)

    def _reconstruct_done(self, result: object) -> None:
        ScannerWindow._reconstruct_done(cast(ScannerWindow, self), result)

    def _was_stopped(self) -> None:
        ScannerWindow._was_stopped(cast(ScannerWindow, self))

    def _fail(self, text: str) -> None:
        ScannerWindow._fail(cast(ScannerWindow, self), text)

    def _stop(self) -> None:
        ScannerWindow._stop(cast(ScannerWindow, self))

    def _consume_resume_preview(self) -> bool:
        return ScannerWindow._consume_resume_preview(cast(ScannerWindow, self))

    def _refresh_capture_progress(self) -> None:
        ScannerWindow._refresh_capture_progress(cast(ScannerWindow, self))

    def _refresh_scan_metrics(self) -> None:
        ScannerWindow._refresh_scan_metrics(cast(ScannerWindow, self))

    def _replace_scan_statuses(self, *statuses: gui.ScanStatus) -> None:
        ScannerWindow._replace_scan_statuses(cast(ScannerWindow, self), *statuses)

    def _show_scan_status(self, status: gui.ScanStatus) -> None:
        ScannerWindow._show_scan_status(cast(ScannerWindow, self), status)

    def _clear_scan_status(self, source: str) -> None:
        ScannerWindow._clear_scan_status(cast(ScannerWindow, self), source)

    def _render_scan_status(self) -> None:
        ScannerWindow._render_scan_status(cast(ScannerWindow, self))

    def _finish_progress(self) -> None:
        ScannerWindow._finish_progress(cast(ScannerWindow, self))

    def _show_saved(self, saved: int) -> None:
        ScannerWindow._show_saved(cast(ScannerWindow, self), saved)

    def _show_coverage(self, snapshot: object) -> None:
        ScannerWindow._show_coverage(cast(ScannerWindow, self), snapshot)  # type: ignore[arg-type]

    def _show_surface_control(self, has_surface: bool) -> None:
        ScannerWindow._show_surface_control(cast(ScannerWindow, self), has_surface)

    def _toggle_surface(self, on: bool) -> None:
        ScannerWindow._toggle_surface(cast(ScannerWindow, self), on)

    def _mounted_picture(self) -> ModelPicture:
        """The REAL one: what the panel is showing is answered by the viewer."""
        return ScannerWindow._mounted_picture(cast(ScannerWindow, self))

    def _refresh_model_provenance(self) -> None:
        """The REAL one, so the sentence under test is the window's own."""
        ScannerWindow._refresh_model_provenance(cast(ScannerWindow, self))

    def _end_pause(self) -> None:
        ScannerWindow._end_pause(cast(ScannerWindow, self))

    def _set_pause_toggle(self, paused: bool) -> None:
        ScannerWindow._set_pause_toggle(cast(ScannerWindow, self), paused)

    def _toggle_pause(self, on: bool) -> None:
        ScannerWindow._toggle_pause(cast(ScannerWindow, self), on)

    def _show_pause_state(self, state: object) -> None:
        self.pause_states.append(cast(gui.PauseState, state))
        ScannerWindow._show_pause_state(cast(ScannerWindow, self), cast(gui.PauseState, state))

    def _show_tracking_lost(self, detail: str) -> None:
        ScannerWindow._show_tracking_lost(cast(ScannerWindow, self), detail)

    def _show_tracking_loss_unrecorded(self, detail: str) -> None:
        ScannerWindow._show_tracking_loss_unrecorded(cast(ScannerWindow, self), detail)

    def _show_recovering_tracking(self, detail: str) -> None:
        ScannerWindow._show_recovering_tracking(cast(ScannerWindow, self), detail)

    def _show_tracking_recovered(self, detail: str) -> None:
        ScannerWindow._show_tracking_recovered(cast(ScannerWindow, self), detail)

    @property
    def _lost_tracking(self) -> bool:
        return ScannerWindow._lost_tracking.fget(cast(ScannerWindow, self))  # type: ignore[attr-defined]

    def _deny_pause_after_loss(self) -> None:
        ScannerWindow._deny_pause_after_loss(cast(ScannerWindow, self))

    def _log_feedback_load(self) -> None:
        ScannerWindow._log_feedback_load(cast(ScannerWindow, self))

    # -- what a test reads -------------------------------------------------
    @property
    def report(self) -> str:
        return "\n".join(self.logged)

    def frames_on_disk(self) -> list[Path]:
        """Every raw frame this capture wrote, wherever a pause put it.

        Through the authoritative reader rather than a root glob, so a test that
        pauses gets the frames the capture actually holds - including the ones a
        published boundary wrote into its own directory - and a test that does
        not pause gets the same sorted list it always did.
        """
        assert self._capture_dir is not None
        return list(read_capture_sequence(self._capture_dir).paths)

    def raw_frames_on_disk(self) -> list[Path]:
        """Every raw frame file, found WITHOUT the authoritative reader.

        For the one case the reader is entitled to refuse outright: a capture
        that recorded a tracking loss. The question then is what actually
        reached the disk, and asking the reader would only get the refusal back
        again - which is a different test's subject.
        """
        assert self._capture_dir is not None
        return sorted(self._capture_dir.rglob("frame_*.raw"))

    def pause(self) -> None:
        """Press Pause, exactly as the toolbar does."""
        self._pause_action.setChecked(True)
        self._toggle_pause(True)

    def resume(self) -> None:
        """Press Resume, exactly as the toolbar does."""
        self._pause_action.setChecked(False)
        self._toggle_pause(False)


def start_capture(window: CaptureWindow) -> None:
    """Run the real :meth:`ScannerWindow._start_capture` against the stand-in."""
    ScannerWindow._start_capture(cast(ScannerWindow, window))


def start_reconstruct(window: CaptureWindow, frames_dir: Path, monkeypatch) -> None:
    """Run the real :meth:`ScannerWindow._start_reconstruct` against a folder.

    The folder dialog is answered rather than opened, which is the only thing
    standing between this path and a headless test: everything after it - which
    frames are read, what is registered, and what the model panel is handed - is
    the real method.
    """

    class FakeFileDialog:
        @staticmethod
        def getExistingDirectory(*_a: object, **_k: object) -> str:  # Qt's name
            return str(frames_dir)

    monkeypatch.setattr(gui, "QFileDialog", FakeFileDialog)
    ScannerWindow._start_reconstruct(cast(ScannerWindow, window))


# -- the preview path, which shares the device wait with a capture -----------


class PreviewWorker(FakeWorker):
    """:class:`~scanner3d.gui.Worker` collapsed to a synchronous call.

    Keeps the one thing about it the preview path depends on: what the job
    returns goes to ``finished_ok`` and what it raises goes to ``failed``. That
    routing is not decoration - it is how a preview that cannot open the device
    ends up saying so instead of looking like one that is running.
    """

    def __init__(self, job) -> None:
        super().__init__()
        self._job = job
        self.failed = FakeSignal()
        self.finished_ok = FakeSignal()

    def start(self) -> None:  # Qt's name
        self.running = True
        try:
            result = self._job(self)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(result)
        finally:
            self.running = False


class PreviewWindow:
    """The parts of :class:`ScannerWindow` that ``_start_preview`` touches.

    The two terminal slots are the REAL ones, so what a failed or finished
    preview does to the Preview button is answerable here rather than assumed:
    a preview that stopped while the toolbar still shows it ticked is exactly
    the "it did nothing" shape this window has produced before.
    """

    def __init__(self, *, real_wait: bool = False) -> None:
        self._preview_worker: object = None
        self._preview_action = FakeAction(checked=True)
        # As in CaptureWindow: the real history, so "a preview starts its own"
        # is a question this stand-in can answer.
        self._return_health = ReturnHealthTracker()
        self._return_verdict: str | None = None
        self._return_label = FakeLabel()
        self._return_advice = FakeLabel()
        #: The value half of frame health, beside the presence half. Both
        #: windows carry it because both draw depth frames, and
        #: ``_show_plausibility`` runs on the same slot that draws them.
        self._static_regions = StaticRegionTracker()
        self._said_plausibility = ""
        self._plausibility_label = FakeLabel()
        #: The real run identity. See :class:`CaptureWindow`.
        self._depth_run = 0
        #: What a real window would have to wait for. ``None`` is "no previous
        #: stream", which is every case except a preview started after one.
        self._previous_stream: object = None
        self.real_wait = real_wait
        self.logged: list[str] = []
        self.frames: list[bytes] = []
        self.colours: list[bytes] = []
        self.failures: list[str] = []
        self.ended: list[object] = []
        self.awaited = 0
        #: Every serial the device wait was asked for, in order.
        self.awaited_serials: list[str | None] = []

    def _scanner_serial(self) -> str | None:
        """The REAL one: which scanner colour is aimed at is what is on trial."""
        return ScannerWindow._scanner_serial(cast(ScannerWindow, self))

    def _reset_return_health(self) -> None:
        """The REAL one, for the same reason: a run that skipped it would pass."""
        ScannerWindow._reset_return_health(cast(ScannerWindow, self))

    def _begin_depth_run(self) -> None:
        """The REAL one, run identity included."""
        ScannerWindow._begin_depth_run(cast(ScannerWindow, self))

    def _end_depth_run(self) -> None:
        """The REAL one: a preview that ended without clearing its warning would pass."""
        ScannerWindow._end_depth_run(cast(ScannerWindow, self))

    def _frame_slot(self, run: int) -> Callable[[bytes], None]:
        """The REAL one, so a frame from a stopped preview is dropped here too."""
        return ScannerWindow._frame_slot(cast(ScannerWindow, self), run)

    def _preview_end_slot[T](self, run: int, end: Callable[[T], None]) -> Callable[[T], None]:
        """The REAL one: a stale completion or failure must be dropped here too."""
        return ScannerWindow._preview_end_slot(cast(ScannerWindow, self), run, end)

    def _await_device(self, serial: str | None) -> None:
        """Counted with the serial it was asked for, and by default not performed.

        The wait itself reaches for discovery, so a test about which scanner
        colour is aimed at must not run it - it would depend on whatever is
        plugged into the machine running the suite. ``real_wait=True`` is for
        the tests whose subject IS the wait, which supply their own readiness.
        """
        self.awaited += 1
        self.awaited_serials.append(serial)
        if self.real_wait:
            ScannerWindow._await_device(cast(ScannerWindow, self), serial)

    def _log(self, message: str) -> None:
        self.logged.append(message)

    def _show_frame(self, raw: bytes) -> None:
        self.frames.append(raw)

    def _show_colour(self, jpeg: bytes) -> None:
        self.colours.append(jpeg)

    def _preview_failed(self, text: str) -> None:
        self.failures.append(text)
        ScannerWindow._preview_failed(cast(ScannerWindow, self), text)

    def _preview_ended(self, result: object) -> None:
        self.ended.append(result)
        ScannerWindow._preview_ended(cast(ScannerWindow, self), result)

    def statusBar(self) -> object:  # Qt's name
        return SimpleNamespace(showMessage=lambda *_a, **_k: None)

    @property
    def report(self) -> str:
        return "\n".join(self.logged)


@pytest.fixture
def previewing(monkeypatch):
    """Run ``_start_preview`` with its worker and its stream both stood in for.

    The stream is recorded rather than run: what a preview does BEFORE opening
    one - which scanner it aims colour at, and whether it opens one at all - is
    the question, and :func:`~scanner3d.gui.stream_with_colour`'s own behaviour
    is the capture tests' subject. An empty call list is therefore a real
    assertion: no stream was opened.
    """
    calls: list[SimpleNamespace] = []

    def stream(worker, serial, on_frame, *, frames, decimate, duration_s, on_delivery=None):
        calls.append(
            SimpleNamespace(
                serial=serial, frames=frames, decimate=decimate, on_delivery=on_delivery
            )
        )
        on_frame(b"a depth frame", None)
        return "the stream ended"

    monkeypatch.setattr(gui, "stream_with_colour", stream)
    monkeypatch.setattr(gui, "Worker", PreviewWorker)

    def start(window: PreviewWindow) -> list[SimpleNamespace]:
        ScannerWindow._start_preview(cast(ScannerWindow, window))
        return calls

    return start


@pytest.fixture
def deprojected(monkeypatch) -> list[int]:
    """Record the size of every frame handed to deprojection, and skip it.

    The GUI's live coverage reads each frame back off disk the moment it is
    written. Recording the sizes is what lets a test say a frame never got that
    far, rather than merely that the run survived.

    Kept on :func:`~scanner3d.gui._load_cloud` rather than
    :func:`~scanner3d.gui._cloud_from_bytes`: a test file that also drives a
    pause holds its OWN stub on ``_cloud_from_bytes``
    (:mod:`tests.test_gui_pause`'s ``tentative_clouds``, for frames that are
    never written to disk) and the two must not collide by both patching one
    name for two different purposes. This fixture's question is answered on the
    live-capture path alone, where ``_load_cloud`` is what is called for every
    committed frame. The reconstruct job patches neither: it needs the
    preparation RECORD as well as the cloud, so it calls
    :func:`~scanner3d.gui._prepare_frame`, which both of these wrap.
    """
    sizes: list[int] = []

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        sizes.append(path.stat().st_size)
        # A cloud, not an empty tuple. A frame that deprojects to NOTHING is now
        # a tracking failure in its own right - three in a row end the capture -
        # and this fixture's subject is which frames reach deprojection at all,
        # not what a capture does about frames it cannot place. Returning
        # nothing here would have every test using it lose tracking on its third
        # selected frame.
        index = int(path.stem.split("_")[1])
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    def align(*_args: object, **_kwargs: object) -> object:
        return local_fit()

    # Stubbed alongside, for the same reason: this fixture skips the real
    # deprojection to keep its tests about cadence, and a real alignment over
    # three synthetic points would answer a question none of them asked.
    monkeypatch.setattr(icp_plane, "icp_point_to_plane", align)
    monkeypatch.setattr(tracking, "icp_point_to_plane", align)
    monkeypatch.setattr(gui, "_load_cloud", load)
    return sizes


@pytest.fixture
def clouds(monkeypatch) -> list[int]:
    """Deprojection, reduced to a distinct three-point cloud per frame.

    Patches :func:`~scanner3d.gui._load_cloud`, not
    :func:`~scanner3d.gui._cloud_from_bytes` - see :func:`deprojected` for why:
    this fixture is used alongside ``tentative_clouds`` in the pause tests,
    which owns ``_cloud_from_bytes`` for the frames that never reach disk, and
    the two stand-ins would otherwise fight over the same name. Neither is what
    the reconstruct job calls; that is :func:`~scanner3d.gui._prepare_frame`.
    """
    seen: list[int] = []

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        seen.append(index)
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)
    return seen


def shell_cloud(
    count: int = 500, radius: float = 20.0, standoff: float = 120.0, spin: float = 0.0
) -> tuple[Point, ...]:
    """A dense spherical shell of points, in the DEPTH CAMERA's optical frame.

    Deterministic - a Fibonacci lattice, no randomness - so a test that hashes an
    exported file gets the same answer on every machine. Dense enough that a
    voxel grid built over it is solid rather than a scatter of isolated cells,
    which is what a reconstruction needs to produce a surface at all.

    ``standoff`` puts it in front of the camera at a plausible working distance,
    and the +y half of it is the part the scanner saw BELOW the centre of its
    image: the optical frame counts y downwards, which is the whole reason an
    export has an upright frame to convert into.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index + spin
        points.append(
            (
                radius * ring * math.cos(angle),
                radius * y,
                standoff + radius * ring * math.sin(angle),
            )
        )
    return tuple(points)


#: The sphere :func:`shell_cloud` describes, in the world - not the optical -
#: frame: centred at the standoff every ``registration`` fixture's poses rotate
#: about (``through=(0.0, 0.0, 120.0)``), radius 20. Kept as one shape so the
#: identity :func:`sphere_raw_frame` proves and the cloud :func:`shell_cloud`
#: hands out are provably the same object, not two numbers kept in sync by
#: hand.
_SPHERE = Sphere(centre=(0.0, 0.0, 120.0), radius=20.0)


def sphere_raw_frame() -> bytes:
    """One real, 512000-byte ``Y16 `` frame of :data:`_SPHERE`.

    The export's TSDF fusion reads raw bytes now, not a decimated cloud, and
    validates their length up front - an empty or short file the old occupancy
    path tolerated is refused outright. This is what a capture of the object
    :func:`shell_cloud` describes would actually have written to
    ``frame_NNNN.raw``, rendered through the same :data:`FACTORY_INTRINSICS`
    and :data:`~scanner3d.cli.DEPTH_SCALE_METRES` the window deprojects with,
    so the fused field sits where the stubbed clouds do.

    Rendered once, and written to EVERY frame file a test needs: the
    ``registration`` fixture's poses rotate the whole scene about
    ``(0.0, 0.0, 120.0)``, which is :data:`_SPHERE`'s own centre, so every pose
    places an identical sphere in front of an unmoved camera. That is asserted
    below rather than assumed - a future change to either the poses or the
    sphere would be caught here, at the one place that relies on it, instead of
    by every test that writes this frame's bytes four times over believing
    they differ.
    """
    base = render([_SPHERE], intrinsics=FACTORY_INTRINSICS, depth_scale=DEPTH_SCALE_METRES)
    moved = render(
        [_SPHERE],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        pose=rotation_about_axis((0.0, 1.0, 0.0), 0.15, through=(0.0, 0.0, 120.0)),
    )
    assert moved.raw == base.raw, "a pose about the sphere's own centre must render identically"
    return base.raw


@pytest.fixture
def meshable_clouds(monkeypatch) -> list[int]:
    """Deprojection reduced to one dense shell per frame, and the frames it read.

    :func:`clouds` returns three points per frame, which is enough to prove
    which frames were read and nothing at all about a surface: a voxel grid over
    three points finds no boundary and there is no mesh to export. This is the
    fixture for the export path, and a test can rebuild any frame's cloud for
    itself with :func:`shell_cloud`.

    Frames are told apart by CALL ORDER - see :func:`clouds` for why bytes
    carry no filename to parse any more - which still lines up with the pose
    a test recomputes for the same frame, because both walk the same
    frame-by-frame order the reconstruct job reads and registers in.

    Patches :func:`~scanner3d.gui._prepare_frame`, which is what the reconstruct
    job calls: it needs the preparation RECORD as well as the cloud, so that the
    export can report what the filters did.
    :meth:`~scanner3d.preparation.PreparedCloud.of_points` is the record for a
    cloud that arrived already prepared, and claims no filtering of its own.
    """
    read: list[int] = []

    def load(_raw: bytes, _target_points: int) -> PreparedCloud:
        index = len(read)
        read.append(index)
        return PreparedCloud.of_points(shell_cloud(spin=0.01 * index))

    monkeypatch.setattr(gui, "_prepare_frame", load)
    return read


def stand_in_surface() -> SurfaceAgreement:
    """A surface agreement that passes every gate and gates nothing.

    Deliberately CONSISTENT rather than empty. The strict reader validates a
    record's decision against its own evidence, so a stand-in claiming nothing
    was measured beside a ready verdict is a record the reader is right to
    refuse - and one of these ends up in an export manifest a test may re-read.
    A stand-in has to be a shape a real evaluation could have produced.
    """
    return SurfaceAgreement(
        measured=True,
        voxel_size=1.0,
        truncation=1.0,
        tolerance=1.0,
        sample_cell=1.0,
        surface_points=1,
        evidence_points=1,
        source_frames_offered=1,
        source_frames_sampled=1,
        source_samples_examined=1,
        source_points=1,
        surface_supported_fraction=1.0,
        evidence_explained_fraction=1.0,
        surface_distance_median=0.0,
        surface_distance_p95=0.0,
        evidence_distance_median=0.0,
        evidence_distance_p95=0.0,
        distance_clamp=1.0,
        detail="stood in for by tests.gui_harness.stub_readiness",
    )


def stub_readiness(monkeypatch) -> list[tuple[object, ...]]:
    """Let a mechanics-only test past the reconstruction readiness gates.

    :mod:`scanner3d.reconstruction_readiness` refuses a reconstruction whose
    poses do not describe a scan, which is correct and is what several tests in
    this repository exist to prove. It is also fatal to the tests that are not
    about that at all: a support-plane masking test deliberately registers four
    identical frames at the IDENTITY, precisely so the plane the filter recorded
    can be measured against the exported surface without a pose in the way. That
    is a manufactured pose sequence by construction, the gate is right to refuse
    it, and refusing it says nothing about the masking the test is asking about.

    So this replaces the evaluation with a ready verdict, and returns the list
    of calls it intercepted so a test can still assert the gate was CONSULTED.
    It is deliberately a function a test opts into by name rather than an autouse
    fixture: the real path must stay the default, and the gates reaching the
    window and the command line are proved in
    :mod:`tests.test_reconstruction_readiness`,
    :mod:`tests.test_gui_readiness_refusal` and :mod:`tests.test_cli_readiness`
    against fixtures built to be judged.
    """
    calls: list[tuple[object, ...]] = []

    def evaluate(frames_dir, clouds, poses, **kwargs):
        calls.append(("evaluate_readiness", frames_dir, len(clouds), len(poses)))
        return ready_readiness()

    def surface(surface_points, evidence, **kwargs):
        calls.append(("surface_evidence", len(surface_points), len(evidence.points)))
        return stand_in_surface(), ()

    monkeypatch.setattr(reconstruction_export, "evaluate_readiness", evaluate)
    monkeypatch.setattr(reconstruction_export, "surface_evidence", surface)
    return calls


def ready_readiness() -> ReconstructionReadiness:
    """A readiness result with no failures, and therefore ready by derivation.

    Every number here is one a real evaluation could have produced, for the
    reason :func:`stand_in_surface` gives: the strict reader derives the
    required reason set from the evidence, so a stand-in whose evidence would
    have obliged a refusal is not a ready record however empty its failure list
    is.
    """
    return ReconstructionReadiness(
        policy=DEFAULT_POLICY,
        tracking=TrackingEvidence(0, 0, unresolved=False, detail=""),
        poses=PoseEvidence(
            poses=4,
            adjacent_steps=3,
            moving_steps=3,
            stationary_steps=0,
            travel_median=10.0 * DEFAULT_POLICY.noise_floor,
            travel_worst=10.0 * DEFAULT_POLICY.noise_floor,
            distinct_viewpoints=4,
            viewpoints_capped=False,
            excursion=20.0 * DEFAULT_POLICY.noise_floor,
            probe_points=DEFAULT_POLICY.probe_points,
        ),
        coverage=CoverageEvidence(
            measured=True,
            frames_offered=4,
            frames_used=4,
            points_used=400,
            cell_size=1.0,
            total_cells=10,
            supported_cells=10,
            supported_fraction=1.0,
            detail="stood in for by tests.gui_harness.stub_readiness",
        ),
        surface=None,
        failures=(),
    )


#: The angle between one synthetic registration pose and the next. Chosen so
#: that the three- and four-frame captures these tests build span well past the
#: 20 degrees of view direction
#: :data:`scanner3d.reconstruction_readiness.ReadinessPolicy.coverage_min_spread`
#: asks for, rather than sitting just inside or just outside it - a fixture on a
#: threshold tests the threshold instead of what it was written for.
REGISTRATION_STEP_RADIANS: Final = 0.30


@pytest.fixture
def registration(monkeypatch) -> Callable[..., SimpleNamespace]:
    """Registration reduced to one fixed, NON-identity pose per frame.

    Non-identity on purpose. Every pose being the identity would let an export
    that ignored the poses entirely, or composed them the wrong way round,
    produce a cloud that still looked like the object - which is the one mistake
    in this path that geometry alone cannot catch.

    Returns what the registration was CALLED with: ``clouds`` is one entry per
    call holding the cloud sequence it was handed, so a test can assert that
    Reconstruct passed on every frame it read, and ``settings`` is the keyword
    arguments beside them, so a test can assert that the window's own backend
    and worker controls reached the thing they are there to steer.

    The step is :data:`REGISTRATION_STEP_RADIANS`, an arc rather than a nudge,
    because the export now refuses a reconstruction whose poses do not describe
    a scan (:mod:`scanner3d.reconstruction_readiness`). At the 0.05 rad step
    this fixture used before, three or four frames spanned about nine degrees of
    view direction - genuinely too little for the coverage gate, and correctly
    refused by it. Widening the step does not change what these tests measure,
    and deliberately does not change the placed extent either: the poses turn
    about the sphere's own centre, so the object stays exactly where it is and
    the empirically chosen ``mesh_budget`` still covers it.
    """

    def install(
        *, limitations: Sequence[str] = ("a synthetic registration, fitted to nothing",)
    ) -> SimpleNamespace:
        seen: list[Sequence[Sequence[Point]]] = []
        settings: list[dict[str, object]] = []

        def run(clouds: Sequence[Sequence[Point]], **kwargs: object) -> SweepReconstruction:
            seen.append(clouds)
            settings.append(dict(kwargs))
            return SweepReconstruction(
                poses=tuple(
                    rotation_about_axis(
                        (0.0, 1.0, 0.0),
                        REGISTRATION_STEP_RADIANS * index,
                        through=(0.0, 0.0, 120.0),
                    )
                    for index in range(len(clouds))
                ),
                frames=len(clouds),
                revisits_found=0,
                chained_score=None,
                optimised_score=None,
                optimised=False,
                converged=False,
                limitations=tuple(limitations),
            )

        monkeypatch.setattr(pipeline, "reconstruct_sweep", run)
        return SimpleNamespace(clouds=seen, settings=settings)

    return install


@pytest.fixture
def mesh_budget(monkeypatch) -> int:
    """Shrink the window's TSDF voxel budget to something a test can wait for.

    The window asks for :data:`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET` (128**3,
    an eighth of :data:`scanner3d.tsdf.MAX_VOXELS`) - which is what this fixture
    replaces, and the reason it patches that name rather than the module
    ceiling. A real capture's extent needs a budget of that size and every
    synthetic capture here does not. 8000 was chosen EMPIRICALLY against this
    file's own
    :data:`_SPHERE` (radius 20, spanning about 40 mm in every axis once
    :func:`shell_cloud` is placed by the ``registration`` fixture's poses),
    by calling :func:`~scanner3d.reconstruction_export.export_reconstruction`
    over it at a range of budgets: below about 1500 voxels
    ``choose_tsdf_resolution`` refuses outright (the span cannot be covered
    even after its own coarsening steps), between roughly 1500 and 2000 the
    field builds but marches to no zero crossing, and at 2000 a small mesh - 8
    vertices, 4 faces - first appears. 8000 keeps a comfortable margin above
    that floor (over a hundred vertices, over a hundred faces, checked by hand
    against both a 4-frame and a 50-frame capture of the same sphere) while
    every export in this suite still fuses and marches in well under a tenth
    of a second - the number a test can wait for. What the budget does NOT
    change is any of the behaviour under test, since the resolution is derived
    from the extent and the budget together whatever either one is.
    """
    budget = 8000
    monkeypatch.setattr(gui, "RECONSTRUCT_VOXEL_BUDGET", budget)
    return budget


def local_fit(**overrides: object) -> LocalFit:
    """A stand-in adjacent ICP result that clears every local-chain gate.

    Every field :func:`~scanner3d.relocalize.judge_local_edge` measures, not
    just the transform. The tentative chain judges each link before it is
    chained, so a stub that carried only a transform would be reporting a NaN
    residual and no correspondences to that judgement - which is a refusal, and
    would make every stubbed test assert the failure path by accident.

    ``overrides`` is how a test asks for the failure path ON PURPOSE:
    ``local_fit(converged=False)``, ``local_fit(rmse=99.0)``, or a different
    ``transform``.
    """
    fit: dict[str, object] = {
        "transform": Rigid.identity(),
        "rmse": 0.2,
        "converged": True,
        "iterations": 9,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
    }
    fit.update(overrides)
    return cast(LocalFit, SimpleNamespace(**fit))


def frame_of(cloud: object) -> int | None:
    """Which frame a stand-in cloud came from, or ``None`` if it is not one.

    :func:`clouds` builds each frame's cloud as
    ``((n, 0, 100), (0, n, 101), (n, n, 102))`` for frame *n*, so the frame
    number is recoverable from the cloud itself. That is what lets the alignment
    stand-in below fail a FRAME rather than an attempt.

    Why that distinction now matters: a frame is no longer placed by one ICP
    solve. :class:`~scanner3d.tracking.IncrementalTracker` forms up to three
    candidates for it - the previous frame, a keyframe, a bounded model
    reference - so "the third call fails" no longer means "the third frame
    fails", and a test that named an attempt would be naming something with no
    stable relationship to the capture it is describing.
    """
    try:
        first = next(iter(cloud))  # type: ignore[call-overload]
        return int(first[0])
    except TypeError, IndexError, ValueError, StopIteration:
        return None


@pytest.fixture
def alignments(monkeypatch) -> Callable[[set[int]], list[int]]:
    """Alignment, reduced to a success unless a test names the FRAME to fail.

    Frames are numbered as :func:`clouds` numbers them, so
    ``failed_alignments={1, 2, 3}`` means frames 1, 2 and 3 cannot be placed by
    any reference - which is what a tracking loss is made of. A frame this
    fixture fails fails for every candidate the tracker tries, because a frame
    only one reference could not place is not a frame that failed.

    Patched in BOTH modules deliberately. The capture no longer calls
    :func:`~scanner3d.icp_plane.icp_point_to_plane` itself - the tracker does -
    but the relocalization path still resolves it through
    :mod:`scanner3d.icp_plane`, and a stand-in installed in only one of the two
    would leave the other running real ICP over three synthetic points.

    The returned list records the FRAMES alignment was reached on, in order and
    once each, which is what it always effectively meant when one frame was one
    solve. Counting raw calls would now count candidates, and a test asserting
    "one alignment per frame after the first" would be asserting a property of
    the candidate policy instead.
    """
    seen: list[int] = []

    def install(failing: set[int]) -> list[int]:
        def align(*args: object, **_kwargs: object) -> object:
            frame = frame_of(args[0]) if args else None
            if frame is not None and (not seen or seen[-1] != frame):
                seen.append(frame)
            if frame is not None and frame in failing:
                raise ValueError(f"synthetic alignment failure on frame {frame}")
            return local_fit()

        monkeypatch.setattr(icp_plane, "icp_point_to_plane", align)
        monkeypatch.setattr(tracking, "icp_point_to_plane", align)
        return seen

    return install


@pytest.fixture
def models(monkeypatch) -> Callable[[set[int]], list[int]]:
    """The coverage build, reduced to a counter with optional failures.

    Returns the list of BUILD ATTEMPTS, which is not the same list as the
    emissions the window received: an attempt that raises produces no emission,
    and proving the next one is still made is the point of one of the tests.
    """
    attempts: list[int] = []

    def install(failing: set[int]) -> list[int]:
        def build(clouds: object, poses: object) -> object:
            attempts.append(len(attempts) + 1)
            if attempts[-1] in failing:
                raise ValueError(f"synthetic coverage failure on attempt {attempts[-1]}")
            return MODEL_PAYLOAD

        monkeypatch.setattr(gui, "_coloured_cloud", build)
        return attempts

    return install


@pytest.fixture
def live(clouds, alignments, models) -> Callable[..., tuple[list[int], list[int]]]:
    """Install the three stubs together and hand back their attempt logs."""

    def install(
        *, failed_alignments: set[int] | None = None, failed_models: set[int] | None = None
    ) -> tuple[list[int], list[int]]:
        return (
            alignments(failed_alignments or set()),
            models(failed_models or set()),
        )

    return install


@pytest.fixture
def capture_dir(tmp_path: Path, monkeypatch) -> Path:
    """Answer the name prompt and the folder prompt, and return where frames go."""
    root = tmp_path / "captures"
    root.mkdir()

    class FakeInputDialog:
        @staticmethod
        def getText(*_a: object, **_k: object) -> tuple[str, bool]:  # Qt's name
            return ("", True)

    monkeypatch.setattr(gui, "QInputDialog", FakeInputDialog)
    monkeypatch.setattr(gui, "configured_captures_dir", lambda: root)
    return root


# -- the relocalization decision, for both paths that hold frames back -------
#
# A resume and an automatic tracking recovery fill the SAME bounded tentative
# buffer, chain it with the same judged local edges and hand it to the same
# global relocalization. So the stand-ins for that decision live here rather
# than beside either set of tests: two copies would be two sets of assumptions
# about what an accepted relocalization looks like, and the one that drifted
# would be the one proving the other path safe.


def relocalization_candidate(*, rmse: float = 0.2, overlap: float = 0.9) -> CandidateEvidence:
    """One refined candidate, carrying every field the evidence records."""
    return CandidateEvidence(
        transform=Rigid.identity(),
        origin="principal",
        coarse_score=0.1,
        coarse_matched=400,
        rmse=rmse,
        converged=True,
        iterations=12,
        matched_fraction=0.9,
        used_fraction=0.8,
        overlap=overlap,
        resolved_backend="python",
    )


def accepting(bridge: Rigid, frames_held: int = MIN_TENTATIVE_FRAMES) -> Relocalization:
    """A decision that cleared every gate, carrying a known bridge."""
    return Relocalization(
        accepted=True,
        bridge=bridge,
        refusals=(),
        candidates=(relocalization_candidate(),),
        frame_overlaps=tuple(0.8 for _ in range(frames_held)),
        runner_up_margin=math.inf,
        coarse_unambiguous=True,
        segment_points=600,
        reference_points=2000,
        frames=frames_held,
        limitations=LIMITATIONS,
    )


def refusing(reason: str) -> Relocalization:
    """A decision that did not clear a gate, and says which."""
    return Relocalization(
        accepted=False,
        bridge=None,
        refusals=(reason,),
        candidates=(relocalization_candidate(rmse=9.0, overlap=0.05),),
        frame_overlaps=(0.02, 0.03, 0.01),
        runner_up_margin=1.01,
        coarse_unambiguous=False,
        segment_points=600,
        reference_points=2000,
        frames=MIN_TENTATIVE_FRAMES,
        limitations=LIMITATIONS,
    )


@pytest.fixture
def decisions(monkeypatch) -> Callable[..., list[int]]:
    """Script what the relocalization decides, and record what it was asked.

    Returns the list of tentative FRAME COUNTS each attempt was made with, which
    is what proves both that a decision is never attempted below the minimum and
    that a retry only happens once a new frame has added evidence.
    """

    def install(outcomes: Sequence[Relocalization]) -> list[int]:
        asked: list[int] = []

        def decide(
            segment_clouds: Sequence[Sequence[Point]],
            _segment_poses: Sequence[Rigid],
            _reference: Sequence[Point],
            **_kwargs: object,
        ) -> Relocalization:
            asked.append(len(segment_clouds))
            return outcomes[min(len(asked) - 1, len(outcomes) - 1)]

        monkeypatch.setattr(gui, "relocalize", decide)
        return asked

    return install


@pytest.fixture
def tentative_clouds(monkeypatch) -> list[int]:
    """Deprojection of a frame that is not on disk, keyed by its fill byte.

    The tentative path cannot read its frames back from a file - that is the
    whole point of it - so it deprojects the bytes directly. Recorded here so a
    test can prove a held frame was examined without ever having been written.
    """
    seen: list[int] = []

    def load(raw: bytes, _target_points: int) -> tuple[Point, ...]:
        fill = raw[0]
        seen.append(fill)
        return (
            (float(fill), 0.0, 100.0),
            (0.0, float(fill), 101.0),
            (float(fill), float(fill), 102.0),
        )

    monkeypatch.setattr(gui, "_cloud_from_bytes", load)
    return seen


@pytest.fixture
def searches(monkeypatch) -> HeldExecutor:
    """The relocalization worker thread, replaced by one the test drives.

    A global search runs OFF the stream thread, and a test that let a real
    thread decide when the answer arrived would assert whatever the machine did
    that run. :class:`HeldExecutor` runs it inline by default, so a test sees
    the decision it scripted at the frame it scripted it for.
    """
    executor = HeldExecutor()
    monkeypatch.setattr(gui, "_relocalization_executor", lambda: executor)
    return executor
