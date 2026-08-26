"""One headless stand-in for the window, shared by every capture-path test.

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

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scanner3d import gui, icp_plane
from scanner3d.geometry import Rigid
from scanner3d.gui import ScannerWindow

__all__ = [
    "CaptureWindow",
    "FakeSignal",
    "FakeWorker",
    "RecordingChannel",
    "alignments",
    "capture_dir",
    "clouds",
    "deprojected",
    "live",
    "models",
    "start_capture",
]

#: A payload the real :meth:`~scanner3d.gui.ScannerWindow._show_coverage` can
#: unpack and hand to the view: points, the per-point "needs more scanning"
#: flags, and the well-covered fraction.
MODEL_PAYLOAD = (
    np.zeros((2, 3), dtype=np.float32),
    np.array([True, False]),
    0.5,
)


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
        self.progress = FakeSignal()

    def channels(self) -> tuple[gui.LatestOnly, ...]:
        return (self.preview, self.colour, self.coverage, self.saved, self.travel)

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


class FakeCombo:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:  # Qt's name
        return self._text


class FakeAction:
    def __init__(self, checked: bool = False, enabled: bool = True) -> None:
        self.checked = checked
        self.enabled = enabled
        self.checks: list[bool] = []

    def isChecked(self) -> bool:  # Qt's name
        return self.checked

    def setChecked(self, on: bool) -> None:  # Qt's name
        self.checks.append(on)
        self.checked = on

    def setEnabled(self, on: bool) -> None:  # Qt's name
        self.enabled = on


class FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:  # Qt's name
        self.text = text


class FakeCloud:
    """The 3D view, reduced to what it was last handed."""

    def __init__(self) -> None:
        self.clouds: list[tuple[np.ndarray, np.ndarray]] = []
        self.thin_fraction = 0.0

    def set_cloud(self, points: np.ndarray, thin: np.ndarray) -> None:
        self.clouds.append((points, thin))
        self.thin_fraction = float(np.count_nonzero(thin)) / max(1, len(thin))


class FakeReport:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:  # Qt's name
        self.clears += 1


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
        self._stop_action = FakeAction()
        self._frames_spin = FakeSpin(frames)
        self._decimate_spin = FakeSpin(decimate)
        self._backend = FakeCombo("python")
        self._report = FakeReport()
        self._capture_progress = FakeLabel()
        self._coverage_summary = FakeLabel()
        self._cloud = FakeCloud()
        self._capture_dir: Path | None = None
        self._resume_preview_after = False
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at: int | None = None
        self._capturing = False
        self._worker: FakeWorker | None = None
        self.awaited = 0

    # -- what the window does to itself ------------------------------------
    def _log(self, message: str) -> None:
        self.logged.append(message)

    def _busy(self, on: bool) -> None:
        self.busy.append(on)

    def _stop_preview(self) -> None:
        return None

    def _scanner_serial(self) -> str | None:
        """Answered from the stand-in, never from the machine's own sysfs."""
        return self.serial

    def _show_colour(self, jpeg: bytes) -> None:
        """The colour slot, reduced to what it was handed."""
        self.colours.append(jpeg)

    def _await_device(self) -> None:
        self.awaited += 1

    def _run(self, job: Callable[[object], object], done: Callable[[object], None]) -> None:
        """Qt's thread, collapsed to a call, keeping only its outcome routing.

        Deliberately catches nothing but :class:`~scanner3d.gui.Stopped`: an
        exception escaping a job is a failure these tests are looking for, and
        swallowing it here would turn a regression into a silent pass.
        """
        self._worker = self.worker
        self.worker.saved.connect(self._show_saved)
        self.worker.coverage.connect(self._show_coverage)
        self.worker.colour.connect(self._show_colour)
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

    def _finish_progress(self) -> None:
        ScannerWindow._finish_progress(cast(ScannerWindow, self))

    def _show_saved(self, saved: int) -> None:
        ScannerWindow._show_saved(cast(ScannerWindow, self), saved)

    def _show_coverage(self, snapshot: object) -> None:
        ScannerWindow._show_coverage(cast(ScannerWindow, self), snapshot)  # type: ignore[arg-type]

    def _log_feedback_load(self) -> None:
        ScannerWindow._log_feedback_load(cast(ScannerWindow, self))

    # -- what a test reads -------------------------------------------------
    @property
    def report(self) -> str:
        return "\n".join(self.logged)

    def frames_on_disk(self) -> list[Path]:
        assert self._capture_dir is not None
        return sorted(self._capture_dir.glob("frame_*.raw"))


def start_capture(window: CaptureWindow) -> None:
    """Run the real :meth:`ScannerWindow._start_capture` against the stand-in."""
    ScannerWindow._start_capture(cast(ScannerWindow, window))


@pytest.fixture
def deprojected(monkeypatch) -> list[int]:
    """Record the size of every frame handed to deprojection, and skip it.

    The GUI's live coverage reads each frame back off disk the moment it is
    written. Recording the sizes is what lets a test say a frame never got that
    far, rather than merely that the run survived.
    """
    sizes: list[int] = []

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        sizes.append(path.stat().st_size)
        return ()

    monkeypatch.setattr(gui, "_load_cloud", load)
    return sizes


@pytest.fixture
def clouds(monkeypatch) -> list[int]:
    """Deprojection, reduced to a distinct three-point cloud per frame."""
    seen: list[int] = []

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        seen.append(index)
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)
    return seen


@pytest.fixture
def alignments(monkeypatch) -> Callable[[set[int]], list[int]]:
    """Alignment, reduced to a success unless a test names the attempt to fail.

    Attempts are numbered from 1 in the order ICP is actually called, which is
    once per frame after the first - the first frame is the reference and is
    never aligned to anything.
    """
    attempts: list[int] = []

    def install(failing: set[int]) -> list[int]:
        def align(*_args: object, **_kwargs: object) -> object:
            attempts.append(len(attempts) + 1)
            if attempts[-1] in failing:
                raise ValueError(f"synthetic alignment failure on attempt {attempts[-1]}")
            return SimpleNamespace(transform=Rigid.identity())

        monkeypatch.setattr(icp_plane, "icp_point_to_plane", align)
        return attempts

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
