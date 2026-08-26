"""A window you can point at an object and get a scan out of.

Why this exists
---------------
Everything this project can do was, until now, reachable only by typing a
command with the right arguments in the right order. The person holding the
scanner is standing at a turntable deciding where to point it next, and asking
them to alt-tab to a terminal to find out how the capture is going is asking
them to do the one thing the situation does not allow.

Three views, because there are three different questions
--------------------------------------------------------
COLOUR answers "is the object where I think it is" - the framing question, and
the reason the vendor's own software shows this camera rather than the depth
one while you position. It runs DURING a capture as well as before one, because
the four minutes of a sweep is precisely when the question keeps being asked.
The RGB camera returns nothing unless a depth stream is already running, so it
is started by the first delivered depth frame and stopped with that stream.

DEPTH answers "is the scanner actually seeing it" - an object can be perfectly
framed and still return almost nothing, if it is too close, too far, too shiny
or too dark.

MODEL answers "what have I actually got" - the accumulated cloud, turnable,
with the under-scanned parts in soft red. This is the one that decides whether
a capture was worth the four minutes, so it gets the space; the other two are
glanced at rather than studied.

A third question, "is my rotation speed sensible", is answered by the travel
readout. It reports how far the surface moves between frames rather than an
angle, because a small object turning fast moves its surface less than a large
one turning slowly, and it is the millimetres that alignment cares about.

What this does NOT do
---------------------
It does not mesh. The model view shows points, not a surface, and nothing here
exports one.

The live model shown DURING a capture is a preview built from chained
alignment only. It has no loop closure, so it drifts, and on a real sweep that
drift reached 65 degrees over one revolution while every internal metric
looked healthy. Treat it as "have I pointed the scanner at that yet", never as
a measure of quality. The number that can be trusted comes from Reconstruct,
which scores against evidence it was not allowed to fit.

It is also deliberately INCOMPLETE. It is rebuilt from a fixed number of frames
sampled across the whole sweep rather than from every frame aligned so far, so
a rebuild two hundred frames in costs what one forty frames in cost. Every frame
is still on disk at full length, and Reconstruct still reads all of them.

Nothing frequent is queued to the window
----------------------------------------
Depth frames, colour frames, the saved-frame count and the model all cross to
the GUI through a one-slot channel that replaces its contents rather than
queueing them. A window that falls behind therefore skips to the newest of each
instead of working through a backlog of pictures that were true several seconds
ago.

That backlog is a risk this corrects, not a diagnosis it proves. The owner's
2026-08-26 report - the model updating two or three times and then sitting
still - is consistent with it, and offline replay of that capture ruled out the
scanner, the frames on disk and the geometry; but no trace of that run's event
loop was ever taken, so the queue is not established as its sole cause. What
IS established is that an unbounded producer wins against any consumer
eventually. The produced/applied/superseded lines in the log are what a second
physical report can be answered with.

Long work runs on worker threads, and every long job can be stopped. A window
that freezes during a four-minute capture looks crashed; a capture that cannot
be abandoned means noticing a mistake in the first ten seconds and having to
watch the remaining four minutes anyway.

Stop stops, and stays stopped
-----------------------------
Stopping a capture is a request the job honours at the next safe boundary -
after the current buffer has gone back to the driver, because a callback that
raises skips that hand-back and this hardware answers a buffer held at teardown
by re-enumerating. So Stop is never instant, and it is never a lie either: the
capture ends, the frames already written stay written, and the live view is NOT
put back up. A capture that answered Stop by restarting the preview stream is
what the owner saw on 2026-08-26, and from outside the process it is
indistinguishable from the button doing nothing at all.

A capture that reaches its own end still resumes the preview it interrupted,
because that is the view you want next when a scan finishes on its own terms.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from scanner3d.cloud_view import CloudView
from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    GUI_CAPTURE_COMMAND,
    CaptureDirUnavailable,
    ConfigFileUnreadable,
    ConfigWriteRefused,
    allocate_capture_dir,
    captures_dir_source,
    configured_captures_dir,
    resolve_serial,
    set_captures_dir,
    validate_capture_name,
)
from scanner3d.coverage import coverage_map
from scanner3d.discovery import find_scanners
from scanner3d.geometry import Rigid
from scanner3d.logs import configure as configure_logging
from scanner3d.logs import get_logger, timed
from scanner3d.pipeline import frame_travel, travel_verdict
from scanner3d.v4l2_stream import (
    CALLER_STOP_REASON,
    MAX_DURATION_S,
    MAX_FRAMES,
    FrameRecord,
    StreamResult,
    wait_for_scanner,
)
from scanner3d.viewer import depth_to_pixels, frame_stats

#: The only frame geometry this scanner's depth mode produces. Named here
#: rather than imported so the window does not depend on another module's
#: private surface for a number the hardware fixes.
FRAME_WIDTH: Final = 640
FRAME_HEIGHT: Final = 400

#: Frames delivered per kept frame. Three gives roughly two degrees between
#: kept frames on a turntable turning once in two minutes, the density measured
#: to register well. Large steps make alignment fail silently.
DEFAULT_DECIMATE: Final = 3

#: Delivered-frame budget for one capture: a little over four minutes at the
#: measured 4.6 fps, enough for two full revolutions.
DEFAULT_FRAMES: Final = 1250

#: SUCCESSFUL ALIGNMENTS between live-coverage refreshes. The refresh rebuilds
#: from every frame aligned so far, so doing it per frame would fall
#: progressively further behind the capture it is meant to be describing.
#:
#: Counted in alignments rather than in saved frames because the refresh is
#: built out of aligned clouds and nothing else. Keying it to saved frame
#: numbers meant a frame that failed to align at frame 12 or 24 skipped that
#: refresh entirely and the model sat still for the next two dozen frames with
#: no explanation - which is what the owner saw on 2026-08-26.
COVERAGE_EVERY: Final = 12

#: Aligned clouds needed before the model is worth drawing at all. Below this
#: the coverage grid has too few observations for its "needs more scanning"
#: judgement to mean anything, and drawing one or two frames as a model invites
#: exactly the wrong conclusion about a scan that has barely started.
MIN_COVERAGE_CLOUDS: Final = 3

#: How many failures of one live-preview stage are reported individually before
#: the rest are summarised. A capture delivers frames at about 4.6 fps, so a
#: stage that fails on every frame would write thousands of identical lines into
#: a log whose whole purpose is being readable after the fact. The first few
#: carry the reason; the summaries prove it never recovered.
FAILURES_REPORTED: Final = 3

#: How often to restate a live-preview failure once it has stopped being news.
FAILURE_SUMMARY_EVERY: Final = 25

#: Points kept per cloud for the live preview. Low on purpose: this estimate is
#: thrown away and recomputed properly by Reconstruct.
PREVIEW_POINTS: Final = 900

#: Aligned frames the live model is built from, however many have been aligned.
#: A rebuild that uses EVERY frame so far costs more the longer the capture runs
#: - measured at 0.008 s over 3 frames and 0.200 s over 51, on a sweep that
#: keeps going for hundreds - and produces a payload that grows with it. Both
#: are wrong for a picture that is thrown away and rebuilt a dozen frames later.
#: Sampling a fixed number of them spread across the whole sweep makes the cost
#: and the size flat, and :func:`_sample_span` is what keeps "spread across the
#: whole sweep" from degenerating into "the beginning" or "the end".
#:
#: This bounds the PREVIEW only. Reconstruct reads every frame off disk and is
#: not affected by anything here.
LIVE_MODEL_CLOUDS: Final = 40

#: Ceiling on the points in one live model, applied where the model is BUILT.
#: Below :data:`~scanner3d.cloud_view.MAX_RENDER_POINTS`, so the view never has
#: to throw any of it away: a payload the producer bounds is one that is never
#: assembled, copied across a thread boundary and held in full just to be
#: decimated at the far end.
LIVE_MODEL_POINTS: Final = 45_000

#: How long a stopping stream waits for the colour grabber to notice. A grab
#: that has already begun holds the RGB node for up to its own 8 s deadline, and
#: the wait is bounded rather than absent so a normal stop does not race the
#: next run for that node. Nothing depends on it succeeding: a pump that is
#: still running has already been forbidden to touch the window, and its thread
#: is a daemon, so it can neither confuse the next capture nor hold the
#: application open.
COLOUR_STOP_TIMEOUT_S: Final = 10.0

#: The namespace this window claims under the captures folder. Inside it, each
#: capture gets a freshly created directory - named by whoever started the scan,
#: or dated when they left the name blank: this window writes frame_0000.raw
#: upward, so a second capture sharing a directory would overwrite the first
#: frame by frame. Defined in :mod:`scanner3d.config` so that ``scanner3d
#: config`` reports the same name this uses.
CAPTURE_PREFIX: Final = GUI_CAPTURE_COMMAND

LOG = get_logger("gui")


class Stopped(Exception):
    """Raised inside a job when the user pressed Stop.

    Carried as an exception rather than a return value so it unwinds through
    :func:`~scanner3d.v4l2_stream.stream_frames`, whose ``finally`` performs
    STREAMOFF, unmap, release and close. Stopping a capture must leave the
    device as tidy as finishing one.
    """


class LatestOnly:
    """One slot holding the newest payload, in place of a queue that can grow.

    A Qt queued connection is a QUEUE. A worker that emits faster than the GUI
    thread can draw does not slow down - it accumulates, and what the operator
    then watches is a backlog of pictures that were true several seconds ago,
    arriving in order, long after the moment they described. Seen from outside
    the process that is one of the things "the model updates two or three times
    and then freezes" can look like - which is a reason to bound the boundary,
    not a finding about any particular run - and no amount of making the drawing
    faster removes it: a producer with no backpressure will always win
    eventually.

    A depth frame, a colour frame, a saved-frame count and an accumulated model
    are all things where an older payload is not partial information, it is
    WRONG information - the newest one supersedes it completely. So this keeps
    exactly one slot, replaces it in place, and wakes the GUI at most once per
    slot. Fifty frames produced while the GUI is busy cost one redraw of the
    fiftieth, not fifty redraws ending at the fiftieth.

    Deliberately shaped like a Qt signal - :meth:`connect` and :meth:`emit` -
    so that a job producing these payloads reads the same either way, and the
    decision to coalesce lives in ONE place instead of at every call site.

    NOT for anything whose older values still matter. ``progress`` stays an
    ordinary signal: it appends lines to a report, and a coalesced log is a log
    with holes in it.

    :meth:`emit` runs on the producing thread and :meth:`deliver` on the GUI
    one; the slot and the counters are guarded by one lock held only for the
    handful of instructions between them.
    """

    def __init__(self, name: str, notify: Callable[[LatestOnly], None]) -> None:
        self.name = name
        self._notify = notify
        self._lock = threading.Lock()
        self._pending: tuple[object, ...] | None = None
        #: Whether the GUI has already been woken for the current slot and has
        #: not drained it yet. This, not the payload, is what makes N emissions
        #: cost one wake-up.
        self._waiting = False
        self._slots: list[Callable[..., object]] = []
        #: Offered by the producer, applied to the window, and replaced while
        #: still waiting to be drawn. Counted rather than silently dropped
        #: because a physical report has to be able to tell "the model was never
        #: produced" from "it was produced and superseded before it was drawn",
        #: and those two have entirely different causes.
        self.offered = 0
        self.applied = 0
        self.superseded = 0
        #: Payloads dropped by :meth:`discard` because whatever produced them
        #: ended before the window drew them. Separate from ``superseded``: one
        #: is "a newer one replaced it", the other is "it stopped being true".
        self.discarded = 0

    def connect(self, slot: Callable[..., object]) -> None:  # Qt's name
        self._slots.append(slot)

    def emit(self, *payload: object) -> None:  # Qt's name
        """Offer the newest payload. PRODUCER THREAD; never waits for the GUI."""
        with self._lock:
            self.offered += 1
            stale = self._pending is not None
            self._pending = payload
            waiting, self._waiting = self._waiting, True
            if stale:
                self.superseded += 1
        if stale:
            # DEBUG because at capture rates this is the common case and the
            # readable form of it is the one-line summary at the end of the run.
            LOG.debug("%s: an update was superseded before it could be drawn", self.name)
        if not waiting:
            self._notify(self)

    def deliver(self) -> None:
        """Apply whatever is in the slot right now. GUI THREAD.

        Empty is normal and is not an error: two wake-ups can be outstanding
        across a producer that emitted either side of a drain.
        """
        with self._lock:
            payload, self._pending = self._pending, None
            self._waiting = False
        if payload is None:
            return
        self.applied += 1
        for slot in self._slots:
            slot(*payload)

    def discard(self) -> bool:
        """Empty the slot so a queued wake-up applies nothing. ANY THREAD.

        Refusing to produce more is not enough to stop a producer that has
        already ended from reaching the window. One payload can be sitting in
        the slot with its wake-up already posted, and that wake-up is answered
        whenever the GUI next gets a turn - which can be after the stream that
        produced it has been torn down, and after the NEXT one has put its own
        picture on the screen. What the operator then sees is the previous
        scan's frame, arriving on top of the current one, with nothing to
        distinguish it.

        So teardown says so explicitly rather than relying on the emission
        guard. The wake-up itself is left outstanding on purpose: an empty
        delivery is already a normal outcome, and cancelling it is not
        something a producer thread can do to a queued Qt event anyway.

        Returns whether anything was actually dropped, and counts it: a payload
        that was produced and never drawn must not be indistinguishable from
        one that was never produced.
        """
        with self._lock:
            payload, self._pending = self._pending, None
            if payload is None:
                return False
            self.discarded += 1
        LOG.debug("%s: an update was dropped because its stream ended first", self.name)
        return True

    def summary(self) -> str:
        """One line naming what this channel produced, drew, and did not draw."""
        return (
            f"{self.name}: {self.offered} produced, {self.applied} applied, "
            f"{self.superseded} superseded before they could be drawn, "
            f"{self.discarded} dropped when the stream ended"
        )


class Worker(QThread):
    """Runs one long job off the UI thread and reports back by signal."""

    progress = Signal(str)
    finished_ok = Signal(object)
    stopped = Signal()
    failed = Signal(str)
    #: Wakes the GUI for ONE :class:`LatestOnly` channel. It carries the channel
    #: and not the payload, which is the whole difference: by the time this
    #: arrives, what should be drawn is whatever is in that channel's slot then,
    #: not whatever was in it when the wake-up was posted.
    _ready = Signal(object)

    def __init__(self, job: Callable[[Worker], object]) -> None:
        super().__init__()
        self._job = job
        self.cancelled = False
        # A QThread OBJECT lives in the thread that created it - the GUI one -
        # so this is a queued connection from run() and _deliver really does
        # happen on the GUI thread.
        self._ready.connect(self._deliver)
        #: Every kept depth frame. Latest-only: a frame the operator never saw
        #: is not worth catching up on.
        self.preview = LatestOnly("depth preview", self._wake)
        self.colour = LatestOnly("colour preview", self._wake)
        self.coverage = LatestOnly("live model", self._wake)
        #: One saved frame. Deliberately the cheapest payload here - an integer,
        #: no geometry - because it is the only thing that fires on EVERY kept
        #: frame and it is what tells the operator the capture is still alive
        #: between the much rarer model refreshes. Coalescing it loses nothing:
        #: the count only ever climbs, so the newest one is the true one.
        self.saved = LatestOnly("saved-frame count", self._wake)
        self.travel = LatestOnly("rotation speed", self._wake)

    def channels(self) -> tuple[LatestOnly, ...]:
        """Every coalescing channel, for reporting what the run cost the GUI."""
        return (self.preview, self.colour, self.coverage, self.saved, self.travel)

    def _wake(self, channel: LatestOnly) -> None:
        """PRODUCER THREAD. One posted event, whatever is waiting in the slot."""
        self._ready.emit(channel)

    def _deliver(self, channel: object) -> None:
        """GUI THREAD, by queued connection from :meth:`_wake`."""
        if isinstance(channel, LatestOnly):
            channel.deliver()

    def cancel(self) -> None:
        """Ask the job to stop. Checked by the job at its own safe points."""
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise Stopped

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._job(self))
        except Stopped:
            self.stopped.emit()
        # A GUI must show the failure, not die: an unhandled worker exception
        # would leave the window alive but permanently busy with no way to
        # find out why.
        except Exception:
            self.failed.emit(traceback.format_exc())


def read_firmware_version(serial: str) -> str:
    """The scanner's own firmware string, or a reason it could not be read.

    This briefly unbinds ``uvcvideo`` from interface 0, performs one read-only
    60-byte control transfer, and hands the interface back. Measured on this
    hardware the whole sequence takes about 16 ms and streaming works normally
    afterwards - verified by capturing frames immediately after.

    It is worth being explicit that this is the same CLASS of operation that
    wedged this scanner earlier in development, when repeated unbounded control
    traffic left the endpoint returning EPROTO. The difference is that this is
    one bounded transfer with an explicit timeout, and the interface is restored
    in every path. It is nonetheless the most invasive thing this window does,
    which is why it reports what happened rather than failing silently.

    Never raises. A version string is a nice thing to know and never a reason to
    stop the application from opening.
    """
    try:
        from scanner3d.ownership import UsbfsOwnershipBackend
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"
    try:
        outcome = UsbfsOwnershipBackend().run(serial)
    except Exception as exc:
        return f"not read ({type(exc).__name__})"
    response = outcome.as_dict().get("response")
    if isinstance(response, dict):
        version = response.get("firmware_version")
        if isinstance(version, str) and version:
            algorithm = response.get("algorithm_version")
            if isinstance(algorithm, str) and algorithm:
                return f"{version}  (algorithm {algorithm})"
            return version
    return "not read"


def _load_cloud(path: Path, target_points: int) -> tuple[tuple[float, float, float], ...]:
    from scanner3d.cli import DEPTH_SCALE_METRES
    from scanner3d.depth_metric import DepthUnitAssumption, deproject_depth_frame
    from scanner3d.simulate import FACTORY_INTRINSICS

    conversion = deproject_depth_frame(
        path.read_bytes(),
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        unit=DepthUnitAssumption(
            label="mm",
            stated_by="caliper measurement",
            basis="a 58.0 mm can measured 57.705 mm at about 122.7 mm standoff",
        ),
    )
    points = [(p.x, p.y, p.z) for p in conversion.points if 60.0 < p.z < 300.0]
    step = max(1, len(points) // target_points)
    return tuple(points[::step])


def _coloured_cloud(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Place every frame in world space and flag the points that need more scanning.

    Coverage is decided per CELL by :mod:`scanner3d.coverage`, then applied back
    to the points that fall in each cell. Colouring the points rather than the
    cells is what makes the picture look like the object instead of like a
    voxel grid, while the judgement stays where it was measured.
    """
    report = coverage_map(list(clouds), list(poses), cell=6.0, min_observations=3)
    thin_cells = {cell.cell_index for cell in report.thin_cells()}
    size = report.cell_size
    points: list[tuple[float, float, float]] = []
    thin: list[bool] = []
    for cloud, pose in zip(clouds, poses, strict=True):
        for point in pose.apply_all(cloud):
            points.append(point)
            key = (
                math.floor(point[0] / size),
                math.floor(point[1] / size),
                math.floor(point[2] / size),
            )
            thin.append(key in thin_cells)
    return (
        np.array(points, dtype=np.float32).reshape(-1, 3),
        np.array(thin, dtype=bool),
        report.well_covered_fraction,
    )


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """One accumulated model, carrying the counts it was BUILT from.

    The counts travel WITH the picture instead of being read off the window at
    the moment it is drawn, and that is the whole reason this type exists. A
    snapshot can be superseded, and a snapshot can be drawn a moment after
    several more frames have landed on disk; a panel that says "rebuilt after 41
    frames" while showing the model built at frame 27 is describing something
    that was never on the screen, and it is worse than saying nothing because it
    is believable.
    """

    points: np.ndarray
    thin: np.ndarray
    well_covered: float
    #: Aligned clouds and saved frames AT BUILD TIME, never at draw time.
    aligned: int
    saved: int
    #: How many of those aligned clouds actually contributed points, which is
    #: :data:`LIVE_MODEL_CLOUDS` once a sweep is longer than that.
    sampled: int
    note: str


def _sample_span(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
    limit: int,
) -> tuple[list[Sequence[tuple[float, float, float]]], list[Rigid]]:
    """At most ``limit`` frames, spread evenly across the WHOLE sweep.

    Not the first ``limit`` and not the last ``limit``, both of which are a
    cheaper way to bound this and both of which answer the model panel's one
    question wrongly. "Have I pointed the scanner at that yet" is about the
    object, so a model that quietly stops including anything after frame 40
    reports an object that stops there, and one that keeps only the last 40
    forgets the half already scanned and sends the operator back over it.

    The first and last frames are always kept: the last is the one that has just
    arrived, and the first is where the sweep started from.
    """
    total = len(clouds)
    if total <= limit:
        return list(clouds), list(poses)
    picks = [round(index * (total - 1) / (limit - 1)) for index in range(limit)]
    return [clouds[index] for index in picks], [poses[index] for index in picks]


def _live_snapshot(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
    *,
    saved: int,
    aligned: int,
) -> ModelSnapshot:
    """The disposable "scanned so far" model, bounded where it is BUILT.

    Two bounds, both applied here rather than in the view. Which frames
    contribute is bounded by :data:`LIVE_MODEL_CLOUDS`, which is what keeps the
    cost of a rebuild flat as the sweep grows; how many points cross to the GUI
    is bounded by :data:`LIVE_MODEL_POINTS`, which is what keeps a payload from
    being assembled in full, copied across a thread boundary and held, only for
    the view to throw most of it away at the far end.

    Neither bound touches the capture. Every frame is on disk at full length and
    Reconstruct reads all of them; what is sampled here is only the picture.
    """
    sampled_clouds, sampled_poses = _sample_span(clouds, poses, LIVE_MODEL_CLOUDS)
    points, thin, well_covered = _coloured_cloud(sampled_clouds, sampled_poses)
    if len(points) > LIVE_MODEL_POINTS:
        step = len(points) // LIVE_MODEL_POINTS + 1
        points, thin = points[::step], thin[::step]
    return ModelSnapshot(
        points=points,
        thin=thin,
        well_covered=well_covered,
        aligned=aligned,
        saved=saved,
        sampled=len(sampled_clouds),
        note=f"preview after {aligned} aligned frames of {saved} saved -",
    )


class ColourPump:
    """Pulls colour frames for as long as a depth stream is up, and no longer.

    MEASURED: the RGB camera delivers nothing on its own - a lone capture times
    out after 8 s with no buffer - but returns frames reliably while a depth
    stream is running. The vendor's own depth-first ordering is therefore a
    requirement, not a habit, and it is why :meth:`start` is called from the
    first delivered depth frame rather than before the stream opens.

    It is specifically STREAMING that is required, not merely the depth node
    being open: holding the node open with no STREAMON still times out at 8 s.
    So a colour preview cannot be had with the projector dark. This project has
    separately failed to darken the emitter by any means - registers 0x922,
    0xb04, 0xb01, 0xb00 and the device's own actuator ioctl were all accepted
    while the owner watched it stay lit - so the blue light while colour is
    running is not a bug and is not currently avoidable.

    ONE thread per stream. Not one per frame, and never one that outlives the
    stream that started it: a grabber still running after a capture ends holds
    the RGB node the NEXT preview or capture wants, and keeps pushing pictures
    into a window that has moved on to a different scan. :meth:`stop` therefore
    runs from the stream's ``finally`` - on completion, on Stop and on failure
    alike - and it both ends the loop and closes the door on any picture already
    in flight.
    """

    def __init__(
        self,
        serial: str,
        emit: Callable[[bytes], None],
        *,
        should_stop: Callable[[], bool],
    ) -> None:
        self._serial = serial
        self._emit = emit
        self._should_stop = should_stop
        # Reentrant because :meth:`_publish` holds it ACROSS the emission - that
        # is what makes the "stopped means stopped" guarantee airtight - and a
        # slot on the far end is entitled to answer a picture by stopping the
        # pump that sent it.
        self._lock = threading.RLock()
        self._stopped = False
        self._thread: threading.Thread | None = None
        #: Pictures handed to the window, and grabs that produced none.
        self.frames = 0
        self.failures = 0

    def start(self) -> None:
        """Begin pumping, at most once. Called once DEPTH has delivered."""
        with self._lock:
            if self._stopped or self._thread is not None or self._should_stop():
                return
            self._thread = threading.Thread(target=self._run, name="colour-pump", daemon=True)
            self._thread.start()
        LOG.info("colour pump started, after the first depth frame")

    def stop(self) -> None:
        """End the pump and forbid any further emission. Bounded, idempotent."""
        with self._lock:
            already, self._stopped = self._stopped, True
            thread = self._thread
            # A stop reached from the pump's OWN thread - a colour slot deciding
            # it has seen enough - cannot join itself, and has no need to: the
            # flag is set, so nothing further is published and the loop ends at
            # its next turn.
            joinable = thread is not None and thread is not threading.current_thread()
        if already or thread is None:
            return
        if joinable:
            thread.join(COLOUR_STOP_TIMEOUT_S)
            if thread.is_alive():
                LOG.warning(
                    "the colour grabber has not finished within %.0fs; it can no longer reach "
                    "the window and it is a daemon thread, so it cannot outlast the application",
                    COLOUR_STOP_TIMEOUT_S,
                )
        LOG.info(
            "colour pump ended: %d pictures shown, %d grabs without one",
            self.frames,
            self.failures,
        )

    def _done(self) -> bool:
        with self._lock:
            if self._stopped:
                return True
        return self._should_stop()

    def _publish(self, frame: bytes) -> None:
        """Hand one picture to the window, unless the pump has been stopped.

        Under the same lock :meth:`stop` sets its flag with, so a picture
        grabbed in the instant before Stop cannot arrive in a window that has
        already moved on. Cheap to hold: the far end is a one-slot channel that
        does not wait for the GUI.
        """
        with self._lock:
            if self._stopped:
                return
            self.frames += 1
            self._emit(frame)

    def _run(self) -> None:
        """The pump itself. COLOUR THREAD. Never raises into the depth stream."""
        import tempfile

        from scanner3d.rgb_frame import V4l2RgbFrameBackend

        statuses = 0
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            frame = target / "rgb-frame.jpg"
            backend = V4l2RgbFrameBackend()
            while not self._done():
                try:
                    outcome = backend.run(self._serial, target)
                except Exception as exc:
                    # Colour is a convenience for framing. Losing it must never
                    # take the depth stream down with it - but it must not
                    # vanish silently either, or a blank colour panel is
                    # unexplainable. Reported once and then given up on: a
                    # grabber that raised on the device is not going to succeed
                    # on the next turn of this loop, and retrying it twice a
                    # second for a four-minute capture is a thousand identical
                    # lines in a log whose only purpose is being read afterwards.
                    self.failures += 1
                    LOG.warning("colour grab failed, giving up: %s", exc)
                    return
                status = outcome.as_dict().get("status")
                if status != "ok":
                    # Not a failure of the run - the depth stream is unaffected
                    # and the next grab may well succeed - so it is reported on
                    # the same "first few, then summaries" budget the live model
                    # failures use rather than once per grab.
                    statuses += 1
                    self.failures += 1
                    if statuses <= FAILURES_REPORTED:
                        LOG.info("colour grab status %s (%d so far)", status, statuses)
                    elif statuses % FAILURE_SUMMARY_EVERY == 0:
                        LOG.info("colour grab status %s, %d times now", status, statuses)
                if frame.exists():
                    payload = frame.read_bytes()
                    frame.unlink(missing_ok=True)
                    self._publish(payload)


def stream_with_colour(
    worker: Worker,
    serial: str | None,
    on_frame: Callable[[bytes, FrameRecord], None],
    *,
    frames: int,
    decimate: int,
    duration_s: float,
) -> StreamResult:
    """Run the depth stream with the colour camera pumped alongside it.

    Shared by preview and capture because the ORDER is a device requirement
    rather than a preference. Launching the grabber first meant it sat timing
    out while depth was still starting and the first picture took 22 s to appear
    instead of about one; and starting no grabber at all during a capture - which
    is what happened until now - meant the colour panel froze for the entire
    sweep, at exactly the moment the operator is deciding where to point next.

    ``serial`` may be ``None``. A scanner whose serial could not be read is
    still perfectly capturable, and colour is the part that is given up.
    """
    from scanner3d.v4l2_stream import stream_frames

    pump = (
        ColourPump(serial, worker.colour.emit, should_stop=lambda: worker.cancelled)
        if serial
        else None
    )

    started = False

    def deliver(buf: bytes, record: FrameRecord) -> None:
        # Depth first, always. The frame is what the run is FOR, and starting
        # the pump is the last thing done with the FIRST one - once, here, and
        # not once per frame. :meth:`ColourPump.start` refuses a second thread
        # by itself as well, but a callback that asks per frame is a callback
        # one edit away from meaning it.
        nonlocal started
        on_frame(buf, record)
        if pump is not None and not started:
            started = True
            pump.start()

    try:
        return stream_frames(
            frames=frames,
            decimate=decimate,
            duration_s=duration_s,
            on_frame=deliver,
            # Stopping through the flag rather than by raising: the stream
            # re-queues the buffer first, and every frame already written stays
            # written.
            should_stop=lambda: worker.cancelled,
        )
    finally:
        # Every path out: the stream's own end, the operator's Stop, and an
        # exception on the way through. A pump that survives any of them keeps
        # the RGB node away from the next run and keeps drawing into a window
        # that is no longer showing this capture.
        if pump is not None:
            pump.stop()
        # AFTER the stop, and never before it: the stop is what guarantees no
        # further picture can be published, so this is the last one there will
        # ever be. Stopping the pump closes the door on future emissions but
        # says nothing about a picture already in the slot with its wake-up
        # posted - and that one, drawn whenever the GUI next gets a turn, would
        # land in whatever run is on screen by then. Frames already APPLIED are
        # untouched: this empties one slot, it does not undraw anything.
        if worker.colour.discard():
            LOG.info("colour: a picture offered before teardown was dropped rather than shown")


def capture_progress_text(
    saved: int, refreshed_at: int | None, refreshes: int, *, running: bool = True
) -> str:
    """What the model panel says about the capture it is showing.

    Two different facts, kept apart on purpose. ``saved`` advances on every
    frame written and is the heartbeat: while that number climbs, the capture is
    alive, whatever the picture below it is doing. ``refreshed_at`` is the saved
    frame count the accumulated model was last rebuilt at, and it moves far less
    often - a rebuild registers every aligned frame so far, so it is deliberately
    occasional.

    Conflating them is what made a working capture look dead: the 3D view
    changes subtly between refreshes on a slowly turning object, and with
    nothing else moving there was no way to tell "still scanning" from "stopped
    responding".

    ``running`` false is the same two counts after the capture has ended, by any
    route. The tense matters: a label still saying "Capturing" over a scan that
    stopped a minute ago is the same lie in the opposite direction.

    A pure function of its arguments so the sentence can be tested without a
    display; the caller owns the label.
    """
    if saved <= 0:
        return "No frames saved yet." if running else "No frames were saved."
    frames = "1 frame" if saved == 1 else f"{saved} frames"
    lead = f"Capturing: {frames} saved." if running else f"Capture ended: {frames} saved."
    if refreshed_at is None:
        tail = (
            "The model below appears once enough of them have aligned."
            if running
            else "Too few of them aligned to build a model here; Reconstruct still can."
        )
    else:
        rebuilds = "1 rebuild" if refreshes == 1 else f"{refreshes} rebuilds"
        tail = f"The model below was last rebuilt after {refreshed_at} of them ({rebuilds})."
    return f"{lead} {tail}"


class ScannerWindow(QMainWindow):
    """Capture and reconstruct, showing both what the scanner sees and what it has."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Scanner")
        self.resize(1400, 820)
        self._worker: Worker | None = None
        self._firmware_worker: Worker | None = None
        self._preview_worker: Worker | None = None
        self._previous_stream: Worker | None = None
        self._resume_preview_after = False
        self._capture_dir: Path | None = None
        self._firmware = "reading..."
        # The heartbeat's counters. All live on the GUI thread and are written
        # only by its slots; the worker sends numbers, not state.
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at: int | None = None
        self._capturing = False

        self._build_toolbar()

        # Three views, sized by what each is FOR. Colour and depth are things
        # you glance at to check the scanner is seeing what you think it is;
        # the model is the thing you turn around and interrogate, so it gets
        # the room. Splitters throughout, because which one matters most
        # changes depending on whether you are framing or reviewing.
        monitors = QSplitter(Qt.Orientation.Vertical)
        monitors.addWidget(self._build_colour_panel())
        monitors.addWidget(self._build_depth_panel())
        monitors.setSizes([340, 340])

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(monitors)
        split.addWidget(self._build_model_panel())
        split.setSizes([460, 940])

        centre = QWidget()
        layout = QVBoxLayout(centre)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split, stretch=3)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText("Results appear here.")
        self._report.setMaximumHeight(190)
        layout.addWidget(self._report, stretch=1)
        self.setCentralWidget(centre)

        self.setStatusBar(QStatusBar())
        self._refresh_device()
        self._start_firmware_read()

    # -- construction ------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("Controls")
        bar.setMovable(False)
        self.addToolBar(bar)

        self._preview_action = QAction("Preview", self)
        self._preview_action.setCheckable(True)
        self._preview_action.setToolTip(
            "Show what the scanner sees without recording anything.\n"
            "Use it to frame and position the object before committing to a capture."
        )
        self._preview_action.toggled.connect(self._toggle_preview)
        bar.addAction(self._preview_action)

        self._capture_action = QAction("Start capture", self)
        self._capture_action.setToolTip(
            "Record a scan. You are asked for a name first - give it one you will\n"
            "recognise, or leave it blank and today's date and time are used.\n\n"
            "Every scan gets a brand-new folder of its own under the captures\n"
            "folder, so a scan can never be written on top of an earlier one. If\n"
            "the name is already taken, -2, -3 and so on are added."
        )
        self._capture_action.triggered.connect(self._start_capture)
        bar.addAction(self._capture_action)

        self._stop_action = QAction("Stop", self)
        self._stop_action.triggered.connect(self._stop)
        self._stop_action.setEnabled(False)
        bar.addAction(self._stop_action)

        self._recon_action = QAction("Reconstruct...", self)
        self._recon_action.triggered.connect(self._start_reconstruct)
        bar.addAction(self._recon_action)
        bar.addSeparator()

        self._captures_action = QAction("Captures folder...", self)
        self._captures_action.triggered.connect(self._choose_captures_dir)
        bar.addAction(self._captures_action)
        self._refresh_captures_action()
        bar.addSeparator()

        bar.addWidget(QLabel(" frames "))
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(30, 5000)
        self._frames_spin.setValue(DEFAULT_FRAMES)
        bar.addWidget(self._frames_spin)

        bar.addWidget(QLabel(" use 1 frame in every "))
        self._decimate_spin = QSpinBox()
        self._decimate_spin.setRange(1, 10)
        self._decimate_spin.setValue(DEFAULT_DECIMATE)
        self._decimate_spin.setToolTip(
            "Keeps every Nth frame as it arrives and discards the rest unexamined.\n"
            "It does NOT pick the best frame out of each group - nothing is compared.\n\n"
            "The point is spacing, not quality: frames arrive faster than the object\n"
            "turns, and near-identical frames add capture time without adding\n"
            "information. At the measured cadence, 3 gives roughly two degrees\n"
            "between kept frames on a turntable turning once in two minutes."
        )
        bar.addWidget(self._decimate_spin)
        bar.addSeparator()

        bar.addWidget(QLabel(" compute "))
        self._backend = QComboBox()
        self._backend.addItems(["auto", "scipy", "numpy", "python"])
        bar.addWidget(self._backend)

        bar.addWidget(QLabel(" workers "))
        self._workers = QSpinBox()
        self._workers.setRange(1, 32)
        self._workers.setValue(1)
        bar.addWidget(self._workers)
        bar.addSeparator()

        refresh = QPushButton("Refresh device")
        refresh.clicked.connect(self._refresh_device)
        bar.addWidget(refresh)

    def _build_colour_panel(self) -> QWidget:
        """The colour camera: for framing, never recorded into the model."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Colour camera - for positioning")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._colour_view = QLabel("preview off")
        self._colour_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._colour_view.setMinimumSize(300, 190)
        self._colour_view.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self._colour_view, stretch=1)
        self._colour_label = QLabel("-")
        self._colour_label.setWordWrap(True)
        layout.addWidget(self._colour_label)
        return panel

    def _build_depth_panel(self) -> QWidget:
        """The depth camera: what actually becomes the model."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Depth camera - what is captured")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._view = QLabel("no frames yet")
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumSize(300, 190)
        self._view.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self._view, stretch=1)
        self._frame_label = QLabel("-")
        self._frame_label.setWordWrap(True)
        layout.addWidget(self._frame_label)

        # The speed readout sits under DEPTH because it is measured from depth
        # alignment - it describes the data being captured, not the colour view.
        self._speed_label = QLabel("rotation speed: waiting for motion")
        self._speed_label.setStyleSheet("font-size:13pt; font-weight:bold; padding:2px;")
        self._speed_label.setWordWrap(True)
        layout.addWidget(self._speed_label)
        self._speed_advice = QLabel("")
        self._speed_advice.setWordWrap(True)
        layout.addWidget(self._speed_advice)
        return panel

    def _build_model_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("What has been scanned so far")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._cloud = CloudView()
        layout.addWidget(self._cloud, stretch=1)
        key = QLabel(
            "left-drag to turn   middle-drag to move   wheel to zoom        "
            "RED = still needs scanning"
        )
        key.setWordWrap(True)
        layout.addWidget(key)
        # Above the coverage summary and separate from it, because they answer
        # different questions at different rates: this one says the capture is
        # still running, the one below says what the model currently contains.
        self._capture_progress = QLabel(capture_progress_text(0, None, 0))
        self._capture_progress.setWordWrap(True)
        self._capture_progress.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self._capture_progress)
        self._coverage_summary = QLabel("No coverage yet.")
        self._coverage_summary.setWordWrap(True)
        layout.addWidget(self._coverage_summary)
        return panel

    # -- device ------------------------------------------------------------
    def _start_firmware_read(self) -> None:
        """Read the firmware version in the background, once, at startup.

        On a worker thread because it touches the device: a read that hangs
        must not stop the window from appearing. It also runs BEFORE any
        capture, which is the safest moment to borrow interface 0 - doing it
        mid-stream would interrupt a sweep the user cannot recapture.
        """
        try:
            devices = find_scanners()
        except Exception:
            self._firmware = "no device"
            return
        if not devices:
            self._firmware = "no device"
            return
        serial = devices[0].serial
        if not serial:
            # The read mutates kernel driver binding, so it must never be aimed
            # at "whatever is plugged in".
            self._firmware = "no serial"
            return

        worker = Worker(lambda _worker: read_firmware_version(serial))
        self._firmware_worker = worker
        worker.finished_ok.connect(self._firmware_read)
        worker.failed.connect(lambda _text: self._firmware_read("not read"))
        worker.start()

    def _firmware_read(self, version: object) -> None:
        self._firmware = str(version)
        self._refresh_device()
        # Only now: the firmware read borrows interface 0, and starting a
        # stream while it holds it would collide. Previewing on open matches
        # what the object is actually for - you look before you scan.
        if self._capture_action.isEnabled():
            self._preview_action.setChecked(True)

    def _refresh_device(self) -> None:
        try:
            devices = find_scanners()
        except Exception as exc:
            self.statusBar().showMessage(f"could not enumerate: {exc}")
            return
        if not devices:
            self.statusBar().showMessage("No scanner found. Check it is plugged in.")
            self._capture_action.setEnabled(False)
            return
        device = devices[0]
        self.statusBar().showMessage(
            f"Serial: {device.serial}    "
            f"USB ID: {device.vendor_id:04x}:{device.product_id:04x}    "
            f"Port: {device.usb_path}    "
            f"Link: {device.usb_speed_label}    "
            f"Firmware: {self._firmware}"
        )
        self._capture_action.setEnabled(True)

    # -- preview -----------------------------------------------------------
    def _toggle_preview(self, on: bool) -> None:
        if on:
            self._start_preview()
        else:
            self._stop_preview()

    def _show_colour(self, jpeg: bytes) -> None:
        """Show the colour camera. Used for framing, never recorded."""
        image = QImage()
        if not image.loadFromData(jpeg):
            return
        self._colour_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._colour_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._colour_label.setText(f"{image.width()}x{image.height()}, about 2 frames a second")

    def _scanner_serial(self) -> str | None:
        """The serial the COLOUR camera will be aimed at, or ``None``.

        Resolved by :func:`~scanner3d.config.resolve_serial`, which is the same
        authority :func:`~scanner3d.v4l2_stream.stream_frames` resolves the
        depth node through: configuration first, and the single attached device
        only when nothing is configured. Asking the enumeration directly for
        whichever scanner it happened to list first was wrong on the one bench
        where it matters - two units plugged in and a serial written in the
        config file - because then colour and depth are two different scanners
        and the framing view shows an object the capture is not recording.

        ``None`` means "not resolvable", and that covers both unavailability
        (nothing attached, enumeration failed, the config file unreadable) and
        AMBIGUITY (several attached, none configured). Refusing the ambiguous
        case is the point: this feeds an optional convenience view, and the
        cost of guessing wrong is a silently misleading picture.

        It changes nothing about the depth capture either way. Every caller
        treats ``None`` as a reason to do less - a capture without a resolvable
        serial is still a capture, just without colour - and the depth stream
        resolves its own node, so an ambiguity that stops colour still reaches
        the operator as the stream's own explicit refusal rather than as this
        quietly doing nothing.

        Never raises. GUI thread, before the job starts.
        """
        try:
            return resolve_serial()
        except Exception as exc:
            # Once per run, not per frame. A blank colour panel with nothing in
            # the log is unexplainable, and "which scanner" is exactly the
            # question a two-scanner bench needs answered in words.
            LOG.info("no colour camera this run: %s", exc)
            return None

    def _start_preview(self) -> None:
        """Stream frames to the screen and write nothing.

        Framing an object is the first thing anyone does, and doing it by
        starting a real capture means either committing to four minutes or
        leaving junk on disk. This records nothing at all.
        """
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        # Only colour depends on this, exactly as in a capture. Framing that
        # cannot be resolved to one scanner must not decide whether the DEPTH
        # view runs: the stream resolves its own node and says so out loud when
        # it cannot, and that message is worth more to the operator than a
        # Preview button that silently does nothing.
        serial = self._scanner_serial()

        def job(worker: Worker) -> object:
            self._await_device()
            # The module's own ceilings rather than a number of my choosing.
            # They exist as runaway guards, and a preview that runs until the
            # user stops it is exactly the shape of thing they guard against -
            # so it ends on its own if the stop signal is ever missed. At the
            # measured rate MAX_FRAMES is about twenty minutes of framing.
            #
            # Colour is pumped by the shared helper, which starts it on the
            # first delivered depth frame and stops it on every way out. It used
            # to be a thread started here that ended only when this worker was
            # CANCELLED - so a preview that hit its own ceiling left a grabber
            # running against a stream that had closed.
            return stream_with_colour(
                worker,
                serial,
                lambda buf, _record: worker.preview.emit(buf),
                frames=MAX_FRAMES,
                decimate=1,
                duration_s=MAX_DURATION_S,
            )

        worker = Worker(job)
        self._preview_worker = worker
        worker.preview.connect(self._show_frame)
        worker.colour.connect(self._show_colour)
        worker.failed.connect(self._preview_failed)
        worker.finished_ok.connect(self._preview_ended)
        worker.start()
        self.statusBar().showMessage("previewing - nothing is being recorded", 4000)

    def _stop_preview(self) -> None:
        """Ask the preview to stop. Does NOT wait for it.

        Waiting here would block the GUI thread for as long as it takes the
        stream to notice - and then again for the device to re-enumerate, which
        it does on every stream close. That was measured at up to eleven
        seconds of unresponsive window, which is indistinguishable from a crash
        to the person looking at it.

        The waiting still has to happen; it happens on the WORKER thread that
        needs the device next, via :meth:`_await_device`. Nothing here blocks.
        """
        worker = self._preview_worker
        if worker is None:
            return
        LOG.info("preview stop requested")
        worker.cancel()
        self._previous_stream = worker
        self._preview_worker = None

    def _await_device(self) -> None:
        """Wait for the previous stream to release the device. WORKER THREAD ONLY.

        Two openers of one V4L2 node fails in a way that reads as the scanner
        being broken, so this genuinely must wait - just not where the user can
        feel it.
        """
        previous = self._previous_stream
        if previous is not None:
            with timed(LOG, "waiting for the previous stream to end", warn_after=2.0):
                previous.wait(8000)
            self._previous_stream = None
        with timed(LOG, "waiting for the device to re-enumerate", warn_after=2.0):
            wait_for_scanner()
        # This device re-enumerates on the USB bus every time a stream closes,
        # vanishing for about 0.4 s. Opening the next stream without waiting
        # fails with "no such device", which reads as the scanner having been
        # unplugged. See v4l2_stream.REENUMERATION_TIMEOUT_S.
        wait_for_scanner()

    def _preview_ended(self, _result: object) -> None:
        """The stream hit its own ceiling. Untick, so it does not look frozen."""
        LOG.info("preview stream ended on its own")
        self._preview_action.setChecked(False)
        self.statusBar().showMessage("preview ended - press Preview to resume", 6000)

    def _preview_failed(self, text: str) -> None:
        LOG.error("preview failed: %s", text.strip().splitlines()[-1] if text.strip() else text)
        self._preview_action.setChecked(False)
        self._log("\nPreview failed:\n" + text)

    # -- plumbing ----------------------------------------------------------
    def _busy(self, busy: bool) -> None:
        self._capture_action.setEnabled(not busy)
        self._recon_action.setEnabled(not busy)
        self._preview_action.setEnabled(not busy)
        self._stop_action.setEnabled(busy)

    def _stop(self) -> None:
        """Ask the running job to stop. Once, and only the once.

        Cancellation is a request the job honours at its own safe boundary - for
        a capture, after the current buffer has been handed back to the driver.
        Asking twice cannot make that boundary arrive sooner, so a second press
        is dropped rather than logged again.
        """
        worker = self._worker
        if worker is None or not worker.isRunning() or worker.cancelled:
            return
        LOG.info("stop requested by the operator")
        worker.cancel()
        self._stop_action.setEnabled(False)
        self._log("\nStopping at the next frame...")

    def _log(self, message: str) -> None:
        self._report.appendPlainText(message)

    def _consume_resume_preview(self) -> bool:
        """Whether preview should come back, clearing the intent as it answers.

        Read exactly once per job, on whichever path that job ended by. Left set,
        it would outlive the capture it belongs to and hand a later operation -
        a reconstruction, or a capture the user stopped - a preview it never
        asked for.
        """
        resume, self._resume_preview_after = self._resume_preview_after, False
        return resume

    def _was_stopped(self) -> None:
        """The operator's Stop, arrived at after the device has been let go.

        Deliberately NOT a completion. Preview is not restarted here even when
        it was running before the capture: the person who pressed Stop asked for
        the scanner to stop, and answering that by putting the live stream back
        up - projector lit, frames flowing - is indistinguishable from the button
        having done nothing. That is exactly how it read to the owner on
        2026-08-26.
        """
        self._busy(False)
        resume = self._consume_resume_preview()
        self._finish_progress()
        kept = len(list(self._capture_dir.glob("frame_*.raw"))) if self._capture_dir else 0
        LOG.info("capture stopped by the operator with %d frames written", kept)
        self._log(f"Stopped. {kept} frames were already written and are safe to reconstruct.")
        if resume:
            self._log("Preview was NOT restarted. Press Preview when you want the live view back.")

    def _fail(self, text: str) -> None:
        self._busy(False)
        # Consumed and dropped: a failed capture is not the moment to reopen the
        # device, and leaving the intent set would spring it on the next job.
        self._consume_resume_preview()
        self._finish_progress()
        self._log("\nFAILED:\n" + text)
        QMessageBox.critical(self, "3D Scanner", text.strip().splitlines()[-1])

    def _show_frame(self, raw: bytes) -> None:
        rows = depth_to_pixels(raw)
        image = QImage(
            b"".join(rows), FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH * 3, QImage.Format.Format_RGB888
        )
        self._view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        stats = frame_stats(raw)
        self._frame_label.setText(
            f"returns {stats.returned} of {stats.samples} "
            f"({100.0 * stats.yield_fraction:.1f}% of the frame)   "
            f"depth codes {stats.min_code} to {stats.max_code}, median {stats.median_code}"
        )

    def _show_travel(self, travel: float) -> None:
        verdict, advice = travel_verdict(travel)
        colour = {"good": "#1a7f37", "too slow": "#9a6700", "too fast": "#9a6700"}.get(
            verdict, "#666"
        )
        self._speed_label.setText(f"rotation speed: {verdict.upper()}  ({travel:.1f} per frame)")
        self._speed_label.setStyleSheet(
            f"font-size:15pt; font-weight:bold; padding:4px; color:{colour};"
        )
        self._speed_advice.setText(advice)

    def _refresh_capture_progress(self) -> None:
        """Restate the heartbeat from the counters. GUI THREAD."""
        self._capture_progress.setText(
            capture_progress_text(
                self._saved_frames,
                self._model_refreshed_at,
                self._model_refreshes,
                running=self._capturing,
            )
        )

    def _show_saved(self, saved: int) -> None:
        """One frame reached the disk. GUI THREAD.

        The cheapest thing this window does per frame, and the only one that is
        guaranteed to happen on every one of them: a label with a number in it.
        """
        self._saved_frames = saved
        self._refresh_capture_progress()

    def _finish_progress(self) -> None:
        """Put the heartbeat into the past tense, once, when a capture ends.

        A no-op for a job that was not a capture, so a stopped or failed
        reconstruction leaves the last capture's summary standing rather than
        overwriting it with a claim about frames it never wrote.
        """
        if not self._capturing:
            return
        self._capturing = False
        self._refresh_capture_progress()
        self._log_feedback_load()

    def _log_feedback_load(self) -> None:
        """What the run cost the GUI, once, at the end of a capture.

        The readable form of what the per-event DEBUG lines record. It is the
        difference between the three explanations a physical "it froze" report
        cannot otherwise be told apart by: a model that was never PRODUCED (the
        alignment or the build failed), one that was produced and APPLIED (the
        picture really did change and was too subtle to see), and one produced
        and then SUPERSEDED before it could be drawn (the window could not keep
        up, which is the failure this bounding exists to prevent, and which
        nothing in the previous run's evidence could have told apart from the
        other two).
        """
        worker = self._worker
        if worker is None:
            return
        for channel in worker.channels():
            LOG.info("live feedback - %s", channel.summary())

    def _show_coverage(self, snapshot: ModelSnapshot) -> None:
        """Draw one accumulated model, described by ITS OWN counts. GUI THREAD.

        ``snapshot.saved`` rather than ``self._saved_frames``: the frame count
        that belongs to this picture is the one it was built at, and by the time
        it is drawn the live count may have moved on. Reading the live one made
        the panel claim a rebuild that had not happened yet.
        """
        self._cloud.set_cloud(snapshot.points, snapshot.thin)
        self._model_refreshes += 1
        self._model_refreshed_at = snapshot.saved
        self._coverage_summary.setText(
            f"{snapshot.note}   {100.0 * snapshot.well_covered:.0f}% of the surface is covered, "
            f"{100.0 * self._cloud.thin_fraction:.0f}% of what you can see still needs scanning"
        )
        LOG.info(
            "live model applied: %d points from %d aligned frames of %d saved",
            len(snapshot.points),
            snapshot.aligned,
            snapshot.saved,
        )
        # Restated here so the heartbeat can say when the picture below it last
        # changed - which is the difference between a model that is holding
        # still because nothing is happening and one that is between rebuilds.
        if self._capturing:
            self._refresh_capture_progress()

    def _run(self, job: Callable[[Worker], object], done: Callable[[object], None]) -> None:
        worker = Worker(job)
        self._worker = worker
        worker.progress.connect(self._log)
        worker.preview.connect(self._show_frame)
        worker.colour.connect(self._show_colour)
        worker.saved.connect(self._show_saved)
        worker.coverage.connect(self._show_coverage)
        worker.travel.connect(self._show_travel)
        worker.finished_ok.connect(done)
        worker.stopped.connect(self._was_stopped)
        worker.failed.connect(self._fail)
        worker.start()

    # -- where captures go -------------------------------------------------
    def _effective_captures_dir(self) -> tuple[Path | None, str | None]:
        """The captures root that will actually be used, and what states it.

        Both, always together. The file is not the only thing that answers this
        question, and reporting a value without its source is how someone ends
        up looking for their frames under the folder they just chose while an
        environment variable quietly sends them somewhere else.
        """
        try:
            return configured_captures_dir(), captures_dir_source()
        except ConfigFileUnreadable:
            return None, None

    def _refresh_captures_action(self) -> None:
        """Keep the toolbar honest about where the next capture will land."""
        try:
            configured = configured_captures_dir()
            source = captures_dir_source()
        except ConfigFileUnreadable as exc:
            self._captures_action.setToolTip(f"Captures folder unreadable:\n{exc}")
            return
        if configured is None:
            current = "Not set - you are asked for a folder every time."
        else:
            current = f"Currently: {configured}\nStated by: {source}"
        self._captures_action.setToolTip(
            "Where captures go. Set it once and every capture lands in a fresh\n"
            f"folder of its own under {CAPTURE_PREFIX}/, without being asked where.\n"
            "Each scan is named when you start it, or dated if you leave the\n"
            "name blank.\n\n" + current
        )

    def _choose_captures_dir(self) -> None:
        """Record a captures folder, so captures stop asking where to go."""
        try:
            start = configured_captures_dir()
        except ConfigFileUnreadable:
            start = None
        directory = QFileDialog.getExistingDirectory(
            self, "Where should captures go?", str(start or Path.home())
        )
        if not directory:
            return
        try:
            target = set_captures_dir(Path(directory)).path
        except (ConfigFileUnreadable, ConfigWriteRefused) as exc:
            LOG.error("captures folder not recorded: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return
        self._refresh_captures_action()
        # Reported from the setting read back, never from the folder that was
        # chosen: an environment variable outranks the file, and saying
        # "captures will go under <what you picked>" when they will not is
        # exactly the surprise this whole setting exists to remove.
        effective, source = self._effective_captures_dir()
        if source == CAPTURES_DIR_ENV_VAR:
            message = (
                f"Recorded {target}, but {CAPTURES_DIR_ENV_VAR} is set in this session and "
                f"wins: captures go under {effective}/{CAPTURE_PREFIX}, in a new folder each "
                "time, named when you start the scan."
            )
        else:
            message = (
                f"Captures will go under {effective}/{CAPTURE_PREFIX}, in a new folder each "
                "time, named when you start the scan."
            )
        self._log(message)
        self.statusBar().showMessage(f"captures folder: {effective}", 6000)

    def _capture_name(self) -> str | None:
        """What to call this scan: a validated leaf name, "" for dated, or None.

        None means "do not capture" and covers both ways of saying it - the
        dialog was cancelled, or the name it returned cannot be a folder name.
        Both are the same instruction to the caller, and both happen before
        anything is created, so there is nothing for the caller to undo.

        Asked before the preview is released and long before a device is
        opened, because a cancelled prompt must cost nothing: no directory, no
        interrupted preview, no scanner touched.
        """
        typed, accepted = QInputDialog.getText(
            self,
            "Start capture",
            "Name this scan (optional).\n"
            "Leave it blank and the folder is named with today's date and time.\n"
            "One folder name, please: no '/' or '\\', and no tabs or line breaks.",
        )
        if not accepted:
            return None
        try:
            return validate_capture_name(typed)
        except CaptureDirUnavailable as exc:
            # CaptureNameInvalid, in practice. Caught by the base class so that
            # a name refused for a reason added later still stops the capture
            # here rather than escaping into a traceback.
            LOG.warning("scan name refused: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return None

    def _capture_target(self) -> Path | None:
        """The directory for the capture about to start, or None if there is none.

        Always a directory created here and now, one level down inside the
        captures root's ``scan`` namespace, that no earlier capture can already
        have written into. Never the captures root itself and never a folder
        somebody merely pointed at: this window writes ``frame_0000.raw``
        upward, so a second capture sharing a directory would overwrite the
        first frame by frame.

        A recorded captures folder is the whole point of having one: it answers
        where without a dialog. Without one, the question still has to be
        asked - and what the answer names is the captures ROOT, with the fresh
        per-scan folder allocated underneath it exactly as if it had been
        recorded.

        A folder that is recorded but cannot be allocated is refused rather
        than quietly demoted to the dialog: the answer to "I cannot write where
        you told me to" is to say so, not to put the frames somewhere else.
        """
        name = self._capture_name()
        if name is None:
            return None
        try:
            root = configured_captures_dir()
        except ConfigFileUnreadable as exc:
            LOG.warning("captures folder unreadable, asking instead: %s", exc)
            root = None
        if root is None:
            directory = QFileDialog.getExistingDirectory(
                self, "Which folder should captures go under?"
            )
            if not directory:
                return None
            root = Path(directory)
        try:
            return allocate_capture_dir(CAPTURE_PREFIX, root=root, name=name)
        except CaptureDirUnavailable as exc:
            LOG.error("no capture folder could be created: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return None

    # -- capture -----------------------------------------------------------
    def _start_capture(self) -> None:
        target = self._capture_target()
        if target is None:
            return
        # Preview holds the same V4L2 node a capture needs. Released here, and
        # waited for, before anything tries to open it again.
        was_previewing = self._preview_action.isChecked()
        self._preview_action.setChecked(False)
        self._stop_preview()
        self._resume_preview_after = was_previewing
        self._capture_dir = target
        # The heartbeat describes THIS capture. Left where the last one finished
        # it would open by claiming frames that are not in this folder.
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at = None
        self._capturing = True
        self._refresh_capture_progress()
        frames = self._frames_spin.value()
        decimate = self._decimate_spin.value()
        backend = self._backend.currentText()
        self._report.clear()
        self._log(f"Capturing to {target}")
        self._log("Move all the way around, and come BACK to where you started.")
        self._busy(True)
        # Read here rather than on the worker: it decides only whether the
        # colour camera can be pumped alongside the depth stream, and a scanner
        # that will not say who it is is still perfectly capturable.
        serial = self._scanner_serial()

        def job(worker: Worker) -> object:
            from scanner3d.geometry import Rigid
            from scanner3d.icp_plane import icp_point_to_plane

            self._await_device()
            # Stop pressed while that wait was running - and it can be a long
            # wait: the previous stream has to end and the device has to come
            # back on the bus. Without this the capture would go on to open the
            # node, take a buffer and write one frame for a sweep the operator
            # has already abandoned. Raising is safe HERE precisely because
            # nothing has been opened yet: no V4L2 buffer is owned, so there is
            # no VIDIOC_QBUF to skip, and no artifact has been written.
            worker.check()

            target.mkdir(parents=True, exist_ok=True)
            kept = 0
            clouds: list[tuple[tuple[float, float, float], ...]] = []
            poses: list[Rigid] = []
            #: Aligned clouds at the last SUCCESSFUL model refresh; 0 for never.
            #: A count rather than a flag, so a refresh that failed to build
            #: leaves the next frame due for another attempt instead of moving
            #: the goalposts past it.
            refreshed_at = 0
            failures = {"alignment": 0, "model": 0}

            def note_failure(stage: str, exc: BaseException) -> None:
                """Say why a live-preview stage failed, without flooding the log.

                These failures were previously swallowed whole. That is defensible
                for the CAPTURE - a frame that will not align is already safely on
                disk and Reconstruct will do the job properly later - but it left
                a stalled-looking model view with no recorded reason anywhere,
                which is precisely the state the owner had to guess about.
                """
                failures[stage] += 1
                count = failures[stage]
                if count <= FAILURES_REPORTED:
                    LOG.warning(
                        "live %s failed on frame %d (%d so far): %s", stage, kept, count, exc
                    )
                elif count % FAILURE_SUMMARY_EVERY == 0:
                    LOG.info("live %s has now failed %d times; latest: %s", stage, count, exc)

            def on_frame(buf: bytes, _record: object) -> None:
                """Save the frame, then keep the model view honest about it.

                Never raises, and specifically never raises
                :class:`Stopped`: this runs between the driver handing a buffer
                over and ``stream_frames`` handing it back, and an exception here
                would skip that ``VIDIOC_QBUF`` and leave userspace owning a
                buffer at teardown - which this hardware answers by
                re-enumerating. Cancellation is read, not thrown.
                """
                nonlocal kept, refreshed_at
                path = target / f"frame_{kept:04d}.raw"
                path.write_bytes(buf)
                kept += 1
                # Before anything that can be skipped, and never skipped itself:
                # the frame IS on disk, so the count must say so.
                worker.saved.emit(kept)
                worker.preview.emit(buf)

                # Everything below is the live model, and all of it is expensive
                # enough to be worth not doing for a capture that is ending.
                # Checked at each stage rather than once, because a Stop pressed
                # during deprojection should not still pay for the alignment
                # after it.
                if worker.cancelled:
                    return

                # Live coverage is chained alignment only - no loop closure, so
                # it drifts. It answers "have I pointed at that yet", nothing
                # more, and Reconstruct recomputes it honestly afterwards.
                cloud = _load_cloud(path, PREVIEW_POINTS)
                if not cloud or worker.cancelled:
                    return
                if not poses:
                    poses.append(Rigid.identity())
                    clouds.append(cloud)
                else:
                    try:
                        fit = icp_point_to_plane(
                            cloud,
                            clouds[-1],
                            max_correspondence_distance=12.0,
                            voxel_size=1.5,
                            keep_fraction=0.8,
                            backend=backend,
                        )
                    except Exception as exc:
                        # A frame that will not align is not a reason to stop
                        # capturing. It is dropped from the PREVIEW only; the
                        # raw frame is already safely on disk. The next frame
                        # that DOES align is still due a refresh, because the
                        # schedule below counts alignments rather than frames.
                        note_failure("alignment", exc)
                        return
                    poses.append(poses[-1].compose(fit.transform))
                    clouds.append(cloud)
                    # Measured from the alignment that just succeeded, so it
                    # reflects the motion the registration actually sees.
                    worker.travel.emit(frame_travel(cloud, fit.transform))

                aligned = len(clouds)
                if aligned < MIN_COVERAGE_CLOUDS:
                    return
                # The first useful model as soon as there is one, then on a
                # fixed alignment interval. ``>=`` rather than equality so a
                # refresh that was skipped or failed is caught up on rather than
                # lost until the next multiple comes round.
                due = refreshed_at == 0 or aligned - refreshed_at >= COVERAGE_EVERY
                if not due or worker.cancelled:
                    return
                started = time.monotonic()
                try:
                    snapshot = _live_snapshot(clouds, poses, saved=kept, aligned=aligned)
                except Exception as exc:
                    # ``refreshed_at`` deliberately unchanged: the next aligned
                    # frame tries again. A model that fails to build once must
                    # not be a model that never builds.
                    note_failure("model", exc)
                    return
                refreshed_at = aligned
                # "Produced" is the half of the story this side of the boundary
                # can vouch for. Whether it was ever DRAWN is the window's to
                # report, and the two being separate lines is what lets a
                # physical report distinguish a model that failed to build from
                # one the GUI could not keep up with. Once every
                # COVERAGE_EVERY alignments, so this is not per-frame spam.
                LOG.info(
                    "live model produced: %d points from %d of %d aligned frames "
                    "in %.3fs, at %d saved",
                    len(snapshot.points),
                    snapshot.sampled,
                    aligned,
                    time.monotonic() - started,
                    kept,
                )
                worker.coverage.emit(snapshot)

            result = stream_with_colour(
                worker,
                serial,
                on_frame,
                frames=frames,
                decimate=decimate,
                duration_s=900.0,
            )
            # Only now, with the stream returned and its finally block done -
            # STREAMOFF, unmap, buffers released, node closed. Raising this any
            # earlier would unwind through the callback and skip the re-queue
            # that keeps the device healthy; raising it here simply routes the
            # finished job to the stopped outcome instead of the completed one.
            #
            # The stream's own result is the authority, and deliberately the
            # ONLY thing consulted. ``worker.cancelled`` is not: a Stop pressed
            # in the moment between a normal return and this line would relabel
            # a capture that ran to its full frame count - or out its duration -
            # as stopped, printing "frames were already written" about a sweep
            # that in fact finished. The stream polls ``should_stop`` after
            # every re-queue, so a stop it could act on is one it reports.
            if result.stopped_early and result.stop_reason == CALLER_STOP_REASON:
                raise Stopped
            return result

        self._run(job, self._capture_done)

    def _capture_done(self, result: object) -> None:
        """A capture that ran to its own end. Never the operator's Stop.

        Stop arrives at :meth:`_was_stopped` instead, which is what makes
        resuming the preview here safe: this is the path where the capture
        finished on its own terms and the live view is the natural next thing to
        see.
        """
        self._busy(False)
        if self._consume_resume_preview():
            self._preview_action.setChecked(True)
        self._finish_progress()
        kept = len(list(self._capture_dir.glob("frame_*.raw"))) if self._capture_dir else 0
        self._log(f"\nCaptured {kept} frames to {self._capture_dir}")
        # Said out loud only when it happened. The stream drops a buffer the
        # device delivered short of a whole frame, which is the right thing to
        # do with it and is also invisible from here: the folder simply holds
        # fewer files than were asked for. Without this line that gap has no
        # explanation, and the obvious guess - that the scan failed - is wrong.
        short = getattr(result, "short_frames", 0)
        if short:
            self._log(
                f"{short} incomplete frame(s) arrived from the scanner and were skipped, "
                "so there are fewer here than you asked for. Every file above is a whole "
                "frame; a partial one is not usable and is never written."
            )
        self._log("Press Reconstruct to register them and get a score you can trust.")

    # -- reconstruct -------------------------------------------------------
    def _start_reconstruct(self) -> None:
        # This capture first, then the folder they all live in: both beat
        # opening the file dialog wherever the process happens to have started.
        try:
            configured = configured_captures_dir()
        except ConfigFileUnreadable:
            configured = None
        start = str(self._capture_dir or configured or "")
        directory = QFileDialog.getExistingDirectory(self, "Which capture?", start)
        if not directory:
            return
        frames_dir = Path(directory)
        backend = self._backend.currentText()
        workers = self._workers.value()
        self._report.clear()
        self._log(f"Reconstructing {frames_dir}\n")
        self._busy(True)

        def job(worker: Worker) -> object:
            from scanner3d.pipeline import reconstruct_sweep

            paths = sorted(frames_dir.glob("frame_*.raw"))
            clouds = []
            for index, path in enumerate(paths):
                worker.check()
                clouds.append(_load_cloud(path, 2200))
                if index % 25 == 0:
                    worker.progress.emit(f"  read {index}/{len(paths)} frames")
            worker.progress.emit(f"  {len(clouds)} frames loaded")

            def report_progress(message: str) -> None:
                worker.check()
                worker.progress.emit(message)

            result = reconstruct_sweep(
                clouds, backend=backend, workers=workers, on_progress=report_progress
            )
            # Every frame, every point: this is the one the operator is being
            # asked to trust, it is built once at the end rather than a dozen
            # times a minute, and nothing about the live preview's bounds
            # applies to it.
            points, thin, well_covered = _coloured_cloud(clouds, list(result.poses))
            worker.coverage.emit(
                ModelSnapshot(
                    points=points,
                    thin=thin,
                    well_covered=well_covered,
                    aligned=len(clouds),
                    saved=len(clouds),
                    sampled=len(clouds),
                    note="registered -",
                )
            )
            return result

        self._run(job, self._reconstruct_done)

    def _reconstruct_done(self, result: object) -> None:
        self._busy(False)
        optimised = getattr(result, "optimised_score", None)
        chained = getattr(result, "chained_score", None)
        self._log("")
        self._log(f"frames {getattr(result, 'frames', '?')}")
        self._log(f"revisited viewpoints found: {getattr(result, 'revisits_found', 0)}")
        if optimised is None or chained is None:
            self._log("\nNOT SCORED, and not corrected.")
            self._log(
                "Too few revisited viewpoints. Without somewhere the scanner returned to, a "
                "drifting scan cannot be caught, so no number here would mean anything. "
                "Capture a path that comes back to where it started."
            )
        else:
            self._log("\nHELD-OUT CLOSURE ERROR (scored against constraints never fitted):")
            self._log(
                f"  before  median {math.degrees(chained.median_error_radians):7.3f} deg"
                f"   worst {math.degrees(chained.worst_error_radians):7.3f}"
            )
            self._log(
                f"  after   median {math.degrees(optimised.median_error_radians):7.3f} deg"
                f"   worst {math.degrees(optimised.worst_error_radians):7.3f}"
            )
            self._log(
                f"  {optimised.used_for_fitting} constraints fitted, "
                f"{optimised.held_out} withheld and used only to score"
            )
        for limitation in getattr(result, "limitations", ()):
            self._log(f"\n  - {limitation}")

    def closeEvent(self, event: object) -> None:
        """Stop every worker before the window goes away.

        Qt destroys a running QThread noisily and the process can die on the
        way out - which, for someone who has just finished a capture, looks
        like the application crashing rather than closing. The waits here are
        bounded: a worker that will not stop must not hold the window open
        forever either.
        """
        LOG.info("window closing, stopping workers")
        for worker in (self._preview_worker, self._worker, self._previous_stream):
            if worker is not None and worker.isRunning():
                worker.cancel()
        for worker in (self._preview_worker, self._worker, self._previous_stream):
            if worker is not None and worker.isRunning():
                worker.wait(4000)
        LOG.info("workers stopped")
        super().closeEvent(event)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    path = configure_logging()
    LOG.info("scanner3d-gui starting; log at %s", path)
    app = QApplication(argv if argv is not None else sys.argv)
    window = ScannerWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
