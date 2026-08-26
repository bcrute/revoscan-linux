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

The signals are real enough to connect to: :class:`FakeSignal` records what was
emitted AND calls whatever was connected to it, so the real slots run on the
real payloads. ``saved`` and ``coverage`` are wired to the real slots, which is
what makes the model panel's own text testable. ``preview`` and ``travel`` are
recorded but not wired, because their slots build a ``QImage`` and set styled
widget text - screen work, and not what any of this is asking about.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scanner3d import gui
from scanner3d.gui import ScannerWindow

__all__ = [
    "CaptureWindow",
    "FakeSignal",
    "FakeWorker",
    "capture_dir",
    "deprojected",
    "start_capture",
]


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


class FakeWorker:
    """The worker a job is handed, with no thread behind it."""

    def __init__(self) -> None:
        self.cancelled = False
        self.running = False
        self.cancels = 0
        self.preview = FakeSignal()
        self.colour = FakeSignal()
        self.coverage = FakeSignal()
        self.saved = FakeSignal()
        self.travel = FakeSignal()
        self.progress = FakeSignal()

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

    def __init__(self, frames: int, decimate: int = 1, *, previewing: bool = False) -> None:
        self.logged: list[str] = []
        self.busy: list[bool] = []
        self.worker = FakeWorker()
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

    def _show_coverage(self, payload: object, note: str) -> None:
        ScannerWindow._show_coverage(cast(ScannerWindow, self), payload, note)  # type: ignore[arg-type]

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
