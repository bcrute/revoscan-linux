"""What Preview and Capture do about a depth node that is not usable yet.

The owner's 2026-08-26 report was that Preview fails the moment the window
opens. :mod:`tests.test_capture_readiness` is about the wait that answers it;
this module is about the two buttons that have to use that wait, use it the same
way, and use it exactly once.

Four things are on trial here:

- The GUI called an enumeration-only wait TWICE and then opened. Two calls that
  both return in milliseconds are not more patient than one; they are the same
  wrong question asked again.
- Both buttons must wait for the SAME scanner they then stream and pump colour
  from. The identity is resolved once, at the top, and handed down.
- A readiness wait that times out must stop the job. Streaming anyway is how a
  permanent permission problem turned into ``PermissionError`` from inside the
  worker with the Preview button still ticked.
- A Stop that lands during the wait must still be a Stop. That boundary is
  already accepted behaviour for a wait that succeeds, and a wait that can now
  also fail must not quietly convert the operator's Stop into a dialog about
  permissions they never asked about.

Why the node was not yet openable on 2026-08-26 is an inference (see
:mod:`tests.test_capture_readiness`); nothing here depends on it. These tests
supply the readiness answer directly, because what the window DOES with each
answer is the subject.

No test here opens a node, enumerates the host, or runs the real wait: the
readiness answer is supplied, because what the window DOES with each answer is
the whole subject.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui, v4l2_stream
from scanner3d.discovery import AccessCheck
from scanner3d.errors import DoctorStatus
from scanner3d.gui import ScannerWindow
from scanner3d.v4l2_stream import CaptureReadiness, StreamDeviceNotReady
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    PreviewWindow,
    capture_dir,
    deprojected,
    previewing,
    start_capture,
)

__all__ = ["capture_dir", "deprojected", "previewing", "stream_kernel", "stream_node"]

NODE = "/dev/video7"

USABLE = CaptureReadiness(
    ready=True,
    reason=f"{NODE} is readable and writable by this process",
    serial="EXAMPLESERIAL0001",
    node_path=NODE,
    polls=3,
    waited_s=0.11,
)

NEVER_USABLE = CaptureReadiness(
    ready=False,
    reason=f"{NODE} exists but is not readable and writable by this process",
    serial="EXAMPLESERIAL0001",
    node_path=NODE,
    access=AccessCheck(path=NODE, exists=True, readable=False, writable=False, mode="0o660"),
    timed_out=True,
    polls=120,
    waited_s=6.0,
)


@pytest.fixture
def readiness(monkeypatch) -> Callable[..., list[str | None]]:
    """Supply the wait's answer, and record WHICH scanner each ask was for.

    The recorded list is both counters at once: its length is how many times the
    GUI asked, and its contents are the identities it asked about. A run that
    waits for a scanner other than the one it then aims colour at is the
    two-scanner bench failing silently, so the serial is not incidental.
    """
    calls: list[str | None] = []

    def install(
        state: CaptureReadiness, *, on_call: Callable[[], None] | None = None
    ) -> list[str | None]:
        def wait(serial: str | None = None, **_kwargs: object) -> CaptureReadiness:
            calls.append(serial)
            if on_call is not None:
                on_call()
            return state

        monkeypatch.setattr(gui, "wait_for_capture_access", wait)
        return calls

    return install


def capture_window(monkeypatch, **kwargs: object) -> CaptureWindow:
    """A capture stand-in whose device wait is the REAL one.

    The harness stubs ``_await_device`` for every other capture test, and rightly
    so - the wait is not their subject. It is this module's, so it is put back.
    """
    window = CaptureWindow(**kwargs)  # type: ignore[arg-type]
    window._previous_stream = None  # type: ignore[attr-defined]
    monkeypatch.setattr(
        window,
        "_await_device",
        lambda serial: ScannerWindow._await_device(cast(ScannerWindow, window), serial),
    )
    return window


# -- capture -----------------------------------------------------------------


def test_a_capture_asks_for_access_once_and_then_streams(
    monkeypatch, readiness, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Once, and for the serial this run resolved.

    The pair of enumeration-only waits it replaced is gone, and the identity the
    capture will stream and pump colour from is the identity it waited for.
    """
    calls = readiness(USABLE)
    stream_kernel((whole_frame(0xA1), whole_frame(0xA2, buffer_index=1)))
    window = capture_window(monkeypatch, frames=2, serial="EXAMPLESERIAL0001")

    start_capture(window)

    assert calls == ["EXAMPLESERIAL0001"]
    assert window.completed == [window.result]
    assert [path.name for path in window.frames_on_disk()] == [
        "frame_0000.raw",
        "frame_0001.raw",
    ]


def test_a_capture_never_opens_the_stream_after_a_readiness_timeout(
    monkeypatch, readiness, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """A node this process may not use is a stated failure, not a capture.

    Before the readiness wait existed the job went straight on to open it, and
    what the operator got was a traceback out of the worker for a device that
    was, from every other point of view, plugged in and working.
    """
    calls = readiness(NEVER_USABLE)
    kernel = stream_kernel((whole_frame(0xA1),))
    window = capture_window(monkeypatch, frames=1)

    with pytest.raises(StreamDeviceNotReady) as raised:
        start_capture(window)

    assert calls == [None]
    assert raised.value.status is DoctorStatus.PERMISSION_DENIED
    assert NODE in str(raised.value)
    assert "permission problem, not a scanner fault" in str(raised.value)
    # Nothing was opened, negotiated, mapped or taken.
    assert kernel.streaming is False
    assert kernel.map_calls == []
    assert kernel.dequeued == []
    assert window.frames_on_disk() == []
    assert window.completed == []


def test_a_stop_during_a_readiness_timeout_is_still_reported_as_a_stop(
    monkeypatch, readiness, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The operator stopped waiting, so the answer is theirs and not the ACL's.

    Both things really did happen - Stop was pressed, and the node never became
    usable - and only one of them is worth a dialog. Reporting the readiness
    failure here would explain permissions to somebody who had already given up
    on this capture.
    """
    kernel = stream_kernel((whole_frame(0xA1),))
    window = capture_window(monkeypatch, frames=1)
    readiness(NEVER_USABLE, on_call=window._stop)

    start_capture(window)

    assert window.stopped == 1
    assert window.completed == []
    assert kernel.streaming is False
    assert kernel.dequeued == []
    assert "Stopped. 0 frames were already written" in window.report


def test_a_stop_during_a_successful_wait_keeps_its_accepted_behaviour(
    monkeypatch, readiness, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The boundary that already existed, re-proved through the real wait.

    ``tests/test_gui_stop.py`` asserts this against a stubbed wait. Here the
    wait is the real method and the readiness it gets back is a success, which
    is the case that used to be the only one there was.
    """
    kernel = stream_kernel((whole_frame(0xA1),))
    window = capture_window(monkeypatch, frames=1)
    readiness(USABLE, on_call=window._stop)

    start_capture(window)

    assert window.stopped == 1
    assert window.completed == []
    assert kernel.streaming is False
    assert kernel.dequeued == []
    assert window.frames_on_disk() == []


# -- preview -----------------------------------------------------------------


def test_preview_asks_for_access_once_and_then_streams(monkeypatch, readiness, previewing) -> None:
    """The same wait, the same number of times, for the same device.

    "The same device" is the load-bearing part: the serial handed to the stream
    for colour is the one the readiness wait was given, so a Preview cannot
    watch one scanner become usable and then frame through another.
    """
    monkeypatch.setattr(gui, "resolve_serial", lambda: "EXAMPLESERIAL0001")
    calls = readiness(USABLE)
    window = PreviewWindow(real_wait=True)

    streams = previewing(window)

    assert calls == ["EXAMPLESERIAL0001"]
    assert streams[0].serial == "EXAMPLESERIAL0001"
    assert window.awaited_serials == ["EXAMPLESERIAL0001"]
    assert window.awaited == 1
    assert len(streams) == 1
    assert window.frames == [b"a depth frame"]
    assert window.failures == []


def test_preview_never_streams_after_a_readiness_timeout_and_says_so(readiness, previewing) -> None:
    """The reported symptom's honest ending.

    A Preview that cannot open the node must not leave the button ticked: from
    outside the process, a lit toolbar over a dead view is indistinguishable
    from a scanner that has stopped producing frames, which is what sent the
    owner looking at the hardware.
    """
    readiness(NEVER_USABLE)
    window = PreviewWindow(real_wait=True)

    streams = previewing(window)

    assert streams == []
    assert window.awaited == 1
    assert len(window.failures) == 1
    assert "permission problem, not a scanner fault" in window.failures[0]
    assert window._preview_action.checked is False
    assert "Preview failed" in window.report


def test_the_wait_for_access_follows_the_previous_stream_ending(readiness, previewing) -> None:
    """Order, not merely presence.

    Checking access while the last stream still owns the node answers about a
    device that is about to re-enumerate and take its node with it, so the
    answer is stale before it is used.
    """
    order: list[str] = []
    window = PreviewWindow(real_wait=True)

    class PreviousStream:
        def wait(self, msec: int) -> bool:  # Qt's name
            order.append(f"previous stream ended ({msec} ms budget)")
            return True

    window._previous_stream = PreviousStream()
    readiness(USABLE, on_call=lambda: order.append("access checked"))

    previewing(window)

    assert order == ["previous stream ended (8000 ms budget)", "access checked"]
    assert window._previous_stream is None


def test_preview_without_a_resolvable_serial_still_waits_for_any_scanner(
    monkeypatch, readiness, previewing
) -> None:
    """``None`` carries through unchanged rather than becoming a refusal.

    A serial that will not resolve - nothing configured and two units attached,
    or a config file that will not parse - already means "no colour, capture
    anyway". It must mean the same to the wait: any scanner, with the stream's
    own resolver left as the authority on the ambiguity. Turning it into a
    readiness failure here would stop a Preview that used to run.
    """
    monkeypatch.setattr(gui, "resolve_serial", lambda: (_ for _ in ()).throw(RuntimeError("two")))
    calls = readiness(USABLE)
    window = PreviewWindow(real_wait=True)

    streams = previewing(window)

    assert calls == [None]
    assert len(streams) == 1
    assert streams[0].serial is None
    assert window.failures == []


def test_the_gui_does_not_reach_for_the_enumeration_only_wait(
    monkeypatch, readiness, previewing
) -> None:
    """It still exists as API; the GUI must not be the thing that calls it.

    It answers "has a scanner come back?", which is true in the same millisecond
    the node is registered and before this process can necessarily open it -
    which is the defect. Deleting it would have broken an importable API for a
    caller that only wants the bus question, so instead the assertion is about
    who calls it: not this module, and specifically not on the Preview path.
    """
    assert hasattr(v4l2_stream, "wait_for_scanner")
    assert not hasattr(gui, "wait_for_scanner")

    called: list[object] = []

    def enumeration_only(*args: object, **_kwargs: object) -> bool:
        called.append(args)
        return True

    monkeypatch.setattr(v4l2_stream, "wait_for_scanner", enumeration_only)
    readiness(USABLE)

    previewing(PreviewWindow(real_wait=True))

    assert called == []
