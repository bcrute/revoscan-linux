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
one while you position.

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

Long work runs on worker threads, and every long job can be stopped. A window
that freezes during a four-minute capture looks crashed; a capture that cannot
be abandoned means noticing a mistake in the first ten seconds and having to
watch the remaining four minutes anyway.
"""

from __future__ import annotations

import math
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
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
    ConfigFileUnreadable,
    ConfigWriteRefused,
    captures_dir_for,
    configured_captures_dir,
    dated_capture_name,
    set_captures_dir,
)
from scanner3d.coverage import coverage_map
from scanner3d.discovery import find_scanners
from scanner3d.geometry import Rigid
from scanner3d.logs import configure as configure_logging
from scanner3d.logs import get_logger, timed
from scanner3d.pipeline import frame_travel, travel_verdict
from scanner3d.v4l2_stream import wait_for_scanner
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

#: Kept frames between live-coverage refreshes. The refresh registers every
#: frame captured so far, so doing it per frame would fall progressively
#: further behind the capture it is meant to be describing.
COVERAGE_EVERY: Final = 12

#: Points kept per cloud for the live preview. Low on purpose: this estimate is
#: thrown away and recomputed properly by Reconstruct.
PREVIEW_POINTS: Final = 900

#: What a capture directory is called under the captures folder. Dated, and one
#: per capture: this window writes frame_0000.raw upward, so a second capture
#: sharing a directory would overwrite the first frame by frame.
CAPTURE_PREFIX: Final = "scan"

LOG = get_logger("gui")


class Stopped(Exception):
    """Raised inside a job when the user pressed Stop.

    Carried as an exception rather than a return value so it unwinds through
    :func:`~scanner3d.v4l2_stream.stream_frames`, whose ``finally`` performs
    STREAMOFF, unmap, release and close. Stopping a capture must leave the
    device as tidy as finishing one.
    """


class Worker(QThread):
    """Runs one long job off the UI thread and reports back by signal."""

    progress = Signal(str)
    preview = Signal(bytes)
    colour = Signal(bytes)
    coverage = Signal(object, str)
    travel = Signal(float)
    finished_ok = Signal(object)
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, job: Callable[[Worker], object]) -> None:
        super().__init__()
        self._job = job
        self.cancelled = False

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

    def _start_preview(self) -> None:
        """Stream frames to the screen and write nothing.

        Framing an object is the first thing anyone does, and doing it by
        starting a real capture means either committing to four minutes or
        leaving junk on disk. This records nothing at all.
        """
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        try:
            devices = find_scanners()
        except Exception:
            return
        if not devices or not devices[0].serial:
            return
        serial = devices[0].serial

        def job(worker: Worker) -> object:
            import tempfile
            import threading

            from scanner3d.rgb_frame import V4l2RgbFrameBackend
            from scanner3d.v4l2_stream import MAX_DURATION_S, MAX_FRAMES, stream_frames

            self._await_device()

            def grab_colour() -> None:
                """Pull colour frames while the depth stream is up.

                MEASURED: the RGB camera delivers nothing on its own - a lone
                capture times out after 8 s with no buffer - but returns frames
                reliably while a depth stream is running. The vendor's own
                depth-first ordering is therefore a requirement, not a habit.
                About 0.43 s per frame, and depth loses none.

                It is specifically STREAMING that is required, not merely the
                depth node being open: holding the node open with no STREAMON
                still times out at 8 s. So a colour preview cannot be had with
                the projector dark. This project has separately failed to
                darken the emitter by any means - registers 0x922, 0xb04,
                0xb01, 0xb00 and the device's own actuator ioctl were all
                accepted while the owner watched it stay lit - so the blue
                light during preview is not a bug and is not currently
                avoidable.
                """
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory)
                    backend = V4l2RgbFrameBackend()
                    while not worker.cancelled:
                        try:
                            outcome = backend.run(serial, target)
                        except Exception as exc:
                            # Colour is a convenience for framing. Losing it
                            # must never take the depth preview down with it -
                            # but it must not vanish silently either, or a
                            # blank colour panel is unexplainable.
                            LOG.warning("colour grab failed, giving up: %s", exc)
                            return
                        status = outcome.as_dict().get("status")
                        if status != "ok":
                            LOG.info("colour grab status %s", status)
                        frame = target / "rgb-frame.jpg"
                        if frame.exists():
                            worker.colour.emit(frame.read_bytes())
                            frame.unlink(missing_ok=True)

            # Started only once DEPTH is actually delivering. The RGB camera
            # returns nothing while depth is idle - a lone capture blocks for
            # its full 8 s timeout - so launching this first meant the grabber
            # sat timing out while depth was still starting, and the first
            # picture took 22 s to appear instead of about one.
            #
            # This is the vendor's own depth-first ordering, arrived at the
            # hard way: paired_capture.py already modelled a one-second gap
            # between the two STREAMONs, copied from observing the SDK.
            started_colour = False

            def on_frame(buf: bytes, _record: object) -> None:
                nonlocal started_colour
                if not started_colour:
                    started_colour = True
                    threading.Thread(target=grab_colour, daemon=True).start()
                worker.preview.emit(buf)

            # The module's own ceilings rather than a number of my choosing.
            # They exist as runaway guards, and a preview that runs until the
            # user stops it is exactly the shape of thing they guard against -
            # so it ends on its own if the stop signal is ever missed. At the
            # measured rate MAX_FRAMES is about twenty minutes of framing.
            return stream_frames(
                frames=MAX_FRAMES,
                decimate=1,
                duration_s=MAX_DURATION_S,
                on_frame=on_frame,
                should_stop=lambda: worker.cancelled,
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
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._stop_action.setEnabled(False)
            self._log("\nStopping at the next frame...")

    def _log(self, message: str) -> None:
        self._report.appendPlainText(message)

    def _was_stopped(self) -> None:
        self._busy(False)
        if self._resume_preview_after:
            self._preview_action.setChecked(True)
        kept = len(list(self._capture_dir.glob("frame_*.raw"))) if self._capture_dir else 0
        self._log(f"Stopped. {kept} frames were already written and are safe to reconstruct.")

    def _fail(self, text: str) -> None:
        self._busy(False)
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

    def _show_coverage(self, payload: tuple[np.ndarray, np.ndarray, float], note: str) -> None:
        points, thin, well_covered = payload
        self._cloud.set_cloud(points, thin)
        self._coverage_summary.setText(
            f"{note}   {100.0 * well_covered:.0f}% of the surface is covered, "
            f"{100.0 * self._cloud.thin_fraction:.0f}% of what you can see still needs scanning"
        )

    def _run(self, job: Callable[[Worker], object], done: Callable[[object], None]) -> None:
        worker = Worker(job)
        self._worker = worker
        worker.progress.connect(self._log)
        worker.preview.connect(self._show_frame)
        worker.coverage.connect(self._show_coverage)
        worker.travel.connect(self._show_travel)
        worker.finished_ok.connect(done)
        worker.stopped.connect(self._was_stopped)
        worker.failed.connect(self._fail)
        worker.start()

    # -- where captures go -------------------------------------------------
    def _refresh_captures_action(self) -> None:
        """Keep the toolbar honest about where the next capture will land."""
        try:
            configured = configured_captures_dir()
        except ConfigFileUnreadable as exc:
            self._captures_action.setToolTip(f"Captures folder unreadable:\n{exc}")
            return
        self._captures_action.setToolTip(
            "Where captures go. Set it once and every capture lands in a dated\n"
            "folder of its own underneath, without being asked.\n\n"
            + (
                f"Currently: {configured}"
                if configured is not None
                else "Not set - you are asked for a folder every time."
            )
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
            target = set_captures_dir(Path(directory))
        except (ConfigFileUnreadable, ConfigWriteRefused) as exc:
            LOG.error("captures folder not recorded: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return
        self._refresh_captures_action()
        self._log(f"Captures will go under {target}, in a dated folder each time.")
        self.statusBar().showMessage(f"captures folder: {target}", 6000)

    def _capture_target(self) -> Path | None:
        """The directory for the capture about to start, or None if cancelled.

        A recorded captures folder is the whole point of having one: it answers
        this without a dialog. Without one, the question still has to be asked.
        """
        try:
            configured = captures_dir_for(dated_capture_name(CAPTURE_PREFIX))
        except ConfigFileUnreadable as exc:
            LOG.warning("captures folder unreadable, asking instead: %s", exc)
            configured = None
        if configured is not None:
            return configured
        directory = QFileDialog.getExistingDirectory(self, "Where should the frames go?")
        return Path(directory) if directory else None

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
        frames = self._frames_spin.value()
        decimate = self._decimate_spin.value()
        backend = self._backend.currentText()
        self._report.clear()
        self._log(f"Capturing to {target}")
        self._log("Move all the way around, and come BACK to where you started.")
        self._busy(True)

        def job(worker: Worker) -> object:
            from scanner3d.geometry import Rigid
            from scanner3d.icp_plane import icp_point_to_plane
            from scanner3d.v4l2_stream import stream_frames

            self._await_device()

            target.mkdir(parents=True, exist_ok=True)
            kept = 0
            clouds: list[tuple[tuple[float, float, float], ...]] = []
            poses: list[Rigid] = []

            def on_frame(buf: bytes, _record: object) -> None:
                nonlocal kept
                path = target / f"frame_{kept:04d}.raw"
                path.write_bytes(buf)
                kept += 1
                worker.preview.emit(buf)

                # Live coverage is chained alignment only - no loop closure, so
                # it drifts. It answers "have I pointed at that yet", nothing
                # more, and Reconstruct recomputes it honestly afterwards.
                cloud = _load_cloud(path, PREVIEW_POINTS)
                if not cloud:
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
                    except Exception:
                        # A frame that will not align is not a reason to stop
                        # capturing. It is dropped from the PREVIEW only; the
                        # raw frame is already safely on disk.
                        return
                    poses.append(poses[-1].compose(fit.transform))
                    clouds.append(cloud)
                    # Measured from the alignment that just succeeded, so it
                    # reflects the motion the registration actually sees.
                    worker.travel.emit(frame_travel(cloud, fit.transform))
                if kept % COVERAGE_EVERY == 0 and len(clouds) >= 3:
                    try:
                        payload = _coloured_cloud(clouds, poses)
                    except Exception:
                        return
                    worker.coverage.emit(payload, f"preview after {kept} frames -")

            return stream_frames(
                frames=frames,
                decimate=decimate,
                duration_s=900.0,
                on_frame=on_frame,
                # Stopping through the flag rather than by raising: the stream
                # re-queues the buffer first, and every frame already written
                # stays written.
                should_stop=lambda: worker.cancelled,
            )

        self._run(job, self._capture_done)

    def _capture_done(self, _result: object) -> None:
        self._busy(False)
        if self._resume_preview_after:
            self._preview_action.setChecked(True)
        kept = len(list(self._capture_dir.glob("frame_*.raw"))) if self._capture_dir else 0
        self._log(f"\nCaptured {kept} frames to {self._capture_dir}")
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
            worker.coverage.emit(_coloured_cloud(clouds, list(result.poses)), "registered -")
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
