"""What a capture writes when the scanner delivers a partial frame.

This is the owner's 2026-08-26 failure, driven end to end without a screen: a
real :meth:`~scanner3d.gui.ScannerWindow._start_capture`, its real ``on_frame``
closure, the real :func:`~scanner3d.v4l2_stream.stream_frames` dequeue loop
against a scripted fake driver, and the real
:meth:`~scanner3d.gui.ScannerWindow._capture_done`. Nothing here constructs a
widget or opens a device.

Testing the closure through ``_start_capture`` rather than reimplementing it is
the point. The bug was not in ``stream_frames`` alone and it was not in the GUI
alone - it was in what the GUI does with everything the stream hands it, and a
test that stubbed either half would have proved nothing about the pair. The
window stand-in below therefore carries exactly the attributes the capture path
touches and no others, and its ``_run`` calls the job on THIS thread so the
whole sequence is one deterministic call.

Two things are stubbed inside the job, both named:

``_await_device``
    Would wait for a USB re-enumeration that is not going to happen offline.

``_load_cloud``
    Deprojection. Replaced by a recorder rather than removed, because "the
    short frame never reached deprojection" is one of the claims under test and
    the recorder is what can state the sizes it actually saw.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.gui import ScannerWindow
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.v4l2_stream import StreamResult
from tests.fake_v4l2 import (
    OWNER_SHORT_FRAME_BYTES,
    short_frame,
    stream_kernel,
    stream_node,
    whole_frame,
)

__all__ = ["stream_kernel", "stream_node"]


class FakeSignal:
    """One Qt signal, recording what was emitted through it."""

    def __init__(self) -> None:
        self.emitted: list[tuple[object, ...]] = []

    def emit(self, *args: object) -> None:  # Qt's name
        self.emitted.append(args)


class FakeWorker:
    """The worker the capture job is handed, with no thread behind it."""

    def __init__(self) -> None:
        self.cancelled = False
        self.preview = FakeSignal()
        self.coverage = FakeSignal()
        self.travel = FakeSignal()
        self.progress = FakeSignal()


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
    def __init__(self, checked: bool = False) -> None:
        self.checked = checked

    def isChecked(self) -> bool:  # Qt's name
        return self.checked

    def setChecked(self, on: bool) -> None:  # Qt's name
        self.checked = on


class FakeReport:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:  # Qt's name
        self.clears += 1


class Window:
    """The parts of :class:`ScannerWindow` a capture actually touches."""

    def __init__(self, frames: int, decimate: int = 1) -> None:
        self.logged: list[str] = []
        self.busy: list[bool] = []
        self.worker = FakeWorker()
        self.result: object = None
        self._preview_action = FakeAction()
        self._frames_spin = FakeSpin(frames)
        self._decimate_spin = FakeSpin(decimate)
        self._backend = FakeCombo("python")
        self._report = FakeReport()
        self._capture_dir: Path | None = None
        self._resume_preview_after = False
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

    def _run(self, job, done) -> None:
        """Qt's thread, collapsed to a call.

        Deliberately NOT wrapped in a try: the pre-change failure mode was an
        exception escaping this job, and swallowing it here would turn the
        regression into a silent pass.
        """
        self.result = job(self.worker)
        done(self.result)

    # -- delegated to the real ones ----------------------------------------
    def _capture_name(self) -> str | None:
        return ScannerWindow._capture_name(cast(ScannerWindow, self))

    def _capture_target(self) -> Path | None:
        return ScannerWindow._capture_target(cast(ScannerWindow, self))

    def _capture_done(self, result: object) -> None:
        ScannerWindow._capture_done(cast(ScannerWindow, self), result)


def _start(window: Window) -> None:
    ScannerWindow._start_capture(cast(ScannerWindow, window))


@pytest.fixture
def deprojected(monkeypatch) -> list[int]:
    """Record the size of every frame handed to deprojection, and skip it.

    The GUI's live coverage reads each frame back off disk the moment it is
    written. Recording the sizes is what lets a test say the short delivery
    never got that far, rather than merely that the run survived.
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


def _frames(window: Window) -> list[Path]:
    assert window._capture_dir is not None
    return sorted(window._capture_dir.glob("frame_*.raw"))


def test_a_short_startup_delivery_writes_no_raw_file_and_the_scan_continues(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The reproduction, end to end.

    The scanner delivers 358402 bytes and then a whole frame. Before this change
    the first one was written to ``frame_0000.raw`` and immediately deprojected,
    ``DepthFrameSizeMismatch`` came back out of the job, and the capture ended at
    startup with a partial file left behind. Now the partial delivery never
    becomes an artifact and the second frame is captured normally.
    """
    stream_kernel((short_frame(), whole_frame(0xC3, buffer_index=1)))
    window = Window(frames=2)

    _start(window)

    written = _frames(window)
    assert [path.name for path in written] == ["frame_0000.raw"]
    assert [path.stat().st_size for path in written] == [EXPECTED_FRAME_BYTES]
    assert written[0].read_bytes() == bytes([0xC3]) * EXPECTED_FRAME_BYTES

    # Neither the preview nor deprojection was ever handed the short bytes.
    previewed = [payload for (payload,) in window.worker.preview.emitted]
    assert [len(cast(bytes, payload)) for payload in previewed] == [EXPECTED_FRAME_BYTES]
    assert deprojected == [EXPECTED_FRAME_BYTES]

    assert isinstance(window.result, StreamResult)
    assert window.result.short_frames == 1
    assert window.result.kept == 1
    assert window.result.delivered == 2


def test_the_completion_message_explains_the_missing_frames(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """A folder holding fewer files than were asked for needs a reason.

    Three frames requested, two delivered short, one file on disk. Without the
    count the obvious reading of that folder is that the capture broke, and the
    obvious next move is to run it again rather than to check the scanner.
    """
    stream_kernel(
        (
            short_frame(),
            short_frame(fill=0x22, buffer_index=1),
            whole_frame(0xD4, buffer_index=2),
        )
    )
    window = Window(frames=3)

    _start(window)

    assert len(_frames(window)) == 1
    report = "\n".join(window.logged)
    assert "Captured 1 frames" in report
    assert "2 incomplete frame(s)" in report
    assert "fewer here than you asked for" in report
    # And it is still a finished capture, not a failure: the window is no longer
    # busy and the next step is still offered.
    assert window.busy == [True, False]
    assert "Press Reconstruct" in report


def test_a_clean_capture_says_nothing_about_short_frames(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The normal case stays as short as it was.

    An explanation printed on every run is an explanation nobody reads, and this
    one is only ever true of a run that lost frames.
    """
    stream_kernel((whole_frame(0xE1), whole_frame(0xE2, buffer_index=1)))
    window = Window(frames=2)

    _start(window)

    assert [path.name for path in _frames(window)] == ["frame_0000.raw", "frame_0001.raw"]
    report = "\n".join(window.logged)
    assert "Captured 2 frames" in report
    assert "incomplete" not in report
    assert deprojected == [EXPECTED_FRAME_BYTES, EXPECTED_FRAME_BYTES]


def test_frame_files_are_numbered_by_what_was_written_not_by_what_arrived(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """No gap in the filenames where a short delivery was.

    Reconstruction globs ``frame_*.raw`` and registers them in sorted order, so
    the numbering only has to be dense and ordered. Leaving a hole would be a
    second way to encode the same fact the completion message already states,
    and one that every downstream reader would have to know about.
    """
    stream_kernel(
        (
            whole_frame(0xF1),
            short_frame(buffer_index=1),
            whole_frame(0xF3, buffer_index=2),
        )
    )
    window = Window(frames=3)

    _start(window)

    written = _frames(window)
    assert [path.name for path in written] == ["frame_0000.raw", "frame_0001.raw"]
    assert [path.read_bytes()[0] for path in written] == [0xF1, 0xF3]


def test_every_delivery_being_short_leaves_an_empty_but_explained_capture(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """A scanner delivering nothing whole produces nothing, and says so.

    Writing a zero-frame folder is the honest outcome; the alternative the bug
    produced was a folder full of files that no later step could read.
    """
    stream_kernel((short_frame(), short_frame(fill=0x22, buffer_index=1)))
    window = Window(frames=2)

    _start(window)

    assert _frames(window) == []
    assert window.worker.preview.emitted == []
    assert deprojected == []
    report = "\n".join(window.logged)
    assert "Captured 0 frames" in report
    assert "2 incomplete frame(s)" in report
    assert isinstance(window.result, StreamResult)
    assert window.result.short_frames == 2
    assert [record.bytes_used for record in window.result.records] == [
        OWNER_SHORT_FRAME_BYTES,
        OWNER_SHORT_FRAME_BYTES,
    ]
