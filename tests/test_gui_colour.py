"""Whether the colour camera keeps working while a capture owns the depth stream.

The owner reported on 2026-08-26 that the colour view stops updating the moment
a capture begins, and it did, by construction: colour was pumped by a thread
that only ``_start_preview`` ever created. Starting a capture stopped the
preview worker, and the capture worker started no replacement - so for the whole
four minutes during which the operator is deciding where to point next, the one
view that answers "is the object where I think it is" was frozen on the last
frame from before they pressed the button.

Two things make this harder than moving the code across:

- The RGB camera returns NOTHING while depth is idle. A lone grab blocks for its
  full 8 s deadline and comes back with no buffer; frames arrive reliably only
  while a depth stream is actually running. So the pump cannot be started when
  the job starts - it has to wait for depth to deliver, which is the vendor's
  own ordering, arrived at here the hard way.
- A grabber that outlives its stream is worse than no grabber. It holds the RGB
  node the next preview or capture wants, and it keeps pushing pictures into a
  window that has moved on to a different scan. So stopping has to be part of
  every way out, including the ones nobody plans for.

No test here opens a real node: :class:`FakeRgbBackend` stands in for the
capture engine, whose own correctness is ``test_v4l2_rgb_capture.py``'s subject.
The capture-path tests drive the real ``_start_capture`` against the scripted
fake driver, exactly as the stop and live-model tests do.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import discovery, gui, rgb_frame, v4l2_stream
from scanner3d.gui import ColourPump, ScannerWindow
from tests.conftest import REAL_SERIAL
from tests.fake_v4l2 import short_frame, stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    PreviewWindow,
    alignments,
    capture_dir,
    clouds,
    live,
    models,
    previewing,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "live",
    "models",
    "previewing",
    "stream_kernel",
    "stream_node",
]

#: Something JPEG-shaped. Nothing in the pump parses it - the structural check
#: belongs to :mod:`scanner3d.jpeg_inspect` and the writing to
#: :mod:`scanner3d.rgb_frame` - so what matters here is only that these exact
#: bytes are what comes out of the window's colour slot.
JPEG = b"\xff\xd8\xff\xe0" + b"framing" + b"\xff\xd9"


def frames(count: int) -> tuple[object, ...]:
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


class FakeRgbBackend:
    """The RGB capture engine, reduced to one picture per call.

    ``pause`` is not decoration: without it the pump's loop spins as fast as the
    interpreter allows, which turns a test that stops it into a test of how much
    work can be done in a scheduling quantum. A real grab takes about 0.43 s.
    """

    def __init__(
        self,
        *,
        status: str = "ok",
        error: Exception | None = None,
        writes: bool = True,
        pause: float = 0.005,
    ) -> None:
        self.status = status
        self.error = error
        self.writes = writes
        self.pause = pause
        self.calls = 0
        self.serials: list[str] = []
        #: Set the moment the engine is first entered, so a test can wait for
        #: the pump to have genuinely done something instead of guessing.
        self.entered = threading.Event()

    def run(self, serial: str, target: Path) -> object:
        self.calls += 1
        self.serials.append(serial)
        self.entered.set()
        if self.error is not None:
            raise self.error
        if self.writes:
            (target / "rgb-frame.jpg").write_bytes(JPEG)
        time.sleep(self.pause)
        return SimpleNamespace(as_dict=lambda: {"status": self.status})


@pytest.fixture
def rgb_backend(monkeypatch):
    """Install a stand-in RGB engine and hand it back. NEVER a real node."""

    def install(**kwargs: object) -> FakeRgbBackend:
        backend = FakeRgbBackend(**kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(rgb_frame, "V4l2RgbFrameBackend", lambda: backend)
        return backend

    return install


class RecordingPump:
    """A stand-in pump that records WHEN it was started and stopped.

    The real one runs a thread against a device. What the capture path owes is
    smaller and exactly testable: create one, start it only once depth has
    delivered, start it only once, and stop it on the way out however the run
    ends.
    """

    def __init__(self, serial: str, emit, *, should_stop, saved, previewed) -> None:
        self.serial = serial
        self.emit = emit
        self.should_stop = should_stop
        #: The window's saved-frame emissions, read at the moment start() runs.
        self.saved = saved
        #: Every depth frame offered to the preview, which is every COMPLETE
        #: delivery. What the pump waits for is a depth stream that is running,
        #: and a delivery is the proof of that; a keep is a decision this run
        #: made about which frames to write, and the RGB camera knows nothing
        #: about it.
        self.previewed = previewed
        self.starts = 0
        self.stops = 0
        #: How many frames had already been saved each time start() was called.
        self.started_after_saved: list[int] = []
        #: How many had been delivered and previewed each time it was called.
        self.started_after_previewed: list[int] = []

    def start(self) -> None:
        self.starts += 1
        self.started_after_saved.append(self.saved[-1] if self.saved else 0)
        self.started_after_previewed.append(len(self.previewed))
        self.emit(JPEG)

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def recording_pump(monkeypatch):
    """Replace the pump with a recorder wired to this window's saved count."""

    def install(window: CaptureWindow) -> list[RecordingPump]:
        made: list[RecordingPump] = []
        seen: list[int] = []
        shown: list[bytes] = []
        window.worker.saved.connect(seen.append)
        window.worker.preview.connect(shown.append)

        def make(serial: str, emit, *, should_stop) -> RecordingPump:
            pump = RecordingPump(serial, emit, should_stop=should_stop, saved=seen, previewed=shown)
            made.append(pump)
            return pump

        monkeypatch.setattr(gui, "ColourPump", make)
        return made

    return install


# -- the pump on its own ---------------------------------------------------


def test_the_pump_shows_colour_frames_while_a_stream_is_up(rgb_backend) -> None:
    """The picture the operator frames with, reaching the window.

    The pump stops itself from the slot that receives the first one, which is
    both the tightest way to end the loop deterministically and a real case: a
    window is entitled to answer a picture by deciding it wants no more.
    """
    backend = rgb_backend()
    shown: list[bytes] = []
    arrived = threading.Event()

    def received(jpeg: bytes) -> None:
        shown.append(jpeg)
        arrived.set()

    pump = ColourPump(REAL_SERIAL, received, should_stop=lambda: False)
    pump.start()
    assert arrived.wait(10.0), "the pump never produced a colour frame"
    pump.stop()

    assert shown[0] == JPEG
    assert pump.frames >= 1
    # Aimed at the authorized serial, which is how the RGB node is resolved at
    # all - the number in /dev/videoN is never assumed by anything here.
    assert set(backend.serials) == {REAL_SERIAL}


def test_nothing_more_is_shown_once_the_pump_has_been_stopped(rgb_backend) -> None:
    """Stopped means stopped, however many grabs the engine still had in it.

    The backend keeps producing a picture on every call and the loop is never
    told to stop by ``should_stop`` - only by the pump's own flag. Exactly one
    picture crossing to the window is the assertion: a grabber that kept
    emitting after this point would be writing into the next capture's window.
    """
    backend = rgb_backend()
    shown: list[bytes] = []
    ended = threading.Event()
    pump: ColourPump | None = None

    def received(jpeg: bytes) -> None:
        shown.append(jpeg)
        assert pump is not None
        # The window, answering the first picture by wanting no more of them.
        pump.stop()
        ended.set()

    pump = ColourPump(REAL_SERIAL, received, should_stop=lambda: False)
    pump.start()
    assert ended.wait(10.0), "the pump never produced a colour frame"
    pump.stop()

    assert len(shown) == 1
    assert backend.calls >= 1


def test_a_pump_already_stopped_never_starts(rgb_backend) -> None:
    """The order Stop can arrive in: before the first depth frame ever lands."""
    backend = rgb_backend()
    shown: list[bytes] = []
    pump = ColourPump(REAL_SERIAL, shown.append, should_stop=lambda: False)

    pump.stop()
    pump.start()
    pump.stop()

    assert backend.calls == 0
    assert shown == []


def test_a_pump_whose_stream_is_already_cancelled_never_starts(rgb_backend) -> None:
    """``should_stop`` true at the first depth frame is the same answer."""
    backend = rgb_backend()
    pump = ColourPump(REAL_SERIAL, lambda _jpeg: None, should_stop=lambda: True)

    pump.start()
    pump.stop()

    assert backend.calls == 0


def test_a_grab_that_raises_is_given_up_on_and_said_so_once(rgb_backend, caplog) -> None:
    """Non-fatal, and not silent, and not repeated.

    Colour is a convenience for framing; losing it must never take the depth
    stream down. But a blank colour panel with nothing in the log is its own
    small mystery, and a warning per grab at twice a second for four minutes is
    a log nobody can read afterwards. One line, and then it gives up.
    """
    backend = rgb_backend(error=RuntimeError("no RGB node"))
    shown: list[bytes] = []
    pump = ColourPump(REAL_SERIAL, shown.append, should_stop=lambda: False)

    with caplog.at_level(logging.WARNING, logger="scanner3d.gui"):
        pump.start()
        assert backend.entered.wait(10.0), "the pump never reached the engine"
        # Joins, so by the time this returns the loop has finished and logged.
        pump.stop()

    assert shown == []
    assert backend.calls == 1
    assert pump.failures == 1
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "colour grab failed" in record.getMessage()
    ]
    assert warnings == ["colour grab failed, giving up: no RGB node"]


def test_a_grab_that_returns_a_bad_status_is_summarised_rather_than_repeated(
    rgb_backend, caplog
) -> None:
    """A run of unhappy statuses costs a handful of lines, not one per grab."""
    backend = rgb_backend(status="timed_out", writes=False, pause=0.001)
    pump = ColourPump(REAL_SERIAL, lambda _jpeg: None, should_stop=lambda: False)

    with caplog.at_level(logging.INFO, logger="scanner3d.gui"):
        pump.start()
        deadline = time.monotonic() + 10.0
        while backend.calls <= gui.FAILURES_REPORTED and time.monotonic() < deadline:
            time.sleep(0.005)
        pump.stop()

    statuses = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("colour grab status")
    ]
    assert backend.calls > gui.FAILURES_REPORTED, "the pump did not keep trying"
    assert statuses[0].startswith("colour grab status timed_out (1 so far)")
    # The first few carry the reason; after that it is one line per summary
    # interval, never one per grab.
    assert len(statuses) < backend.calls
    assert len(statuses) <= gui.FAILURES_REPORTED + backend.calls // gui.FAILURE_SUMMARY_EVERY


# -- the capture path --------------------------------------------------------


def test_a_capture_starts_colour_only_after_the_first_depth_frame(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """Depth first, once, and never before there is a depth stream to ride on.

    ``started_after_previewed == [1]`` carries both halves: the pump was started
    with exactly one complete delivery behind it - so not before the stream
    delivered - and it was started exactly once over the whole capture, so a
    five-hundred frame sweep is one grabber and not five hundred.

    What it waits for is the DELIVERY and not the write. The measured
    requirement is that a depth stream be running; a delivered buffer is proof
    of that, and the writing that follows it is this run's own business.
    """
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()

    start_capture(window)

    assert len(made) == 1
    assert made[0].serial == REAL_SERIAL
    assert made[0].started_after_previewed == [1]
    assert made[0].started_after_saved == [0]
    assert len(window.frames_on_disk()) == 6


def test_a_capture_that_keeps_one_frame_in_three_still_starts_colour_at_once(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """Decimation is about what is WRITTEN, and colour is not written.

    At ``decimate=3`` the first kept frame is the third delivery. Waiting for it
    would leave the colour view - the one that answers "is the object where I
    think it is" - dark for two extra deliveries at the exact moment the
    operator is aiming the scanner, for no reason the RGB camera knows about.
    """
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6, decimate=3, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()

    start_capture(window)

    assert [pump.started_after_previewed for pump in made] == [[1]]
    assert [pump.started_after_saved for pump in made] == [[0]]
    # And the capture itself stayed decimated: two of the six deliveries kept.
    assert len(window.frames_on_disk()) == 2


def test_a_short_first_delivery_does_not_start_colour(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """An incomplete buffer is a transport event, not a frame.

    It is not previewed, it is not written, and it must not be read as evidence
    that the depth stream is delivering either - which is exactly what the
    owner's 2026-08-26 scanner opened with.
    """
    stream_kernel((short_frame(), *frames(3)[:3]))
    window = CaptureWindow(frames=4, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()

    start_capture(window)

    # Started by the FIRST COMPLETE delivery - the second buffer - and not by
    # the short one before it.
    assert [pump.started_after_previewed for pump in made] == [[1]]
    assert len(window.frames_on_disk()) == 3


def test_a_capture_shows_colour_while_it_is_recording(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models, rgb_backend
) -> None:
    """The reported symptom, asserted against the real pump.

    The picture has to arrive DURING the capture, not after it, so the capture
    is made to wait at the frame after the pump starts until one has. Waiting on
    an event rather than sleeping is what keeps it a proof rather than a race:
    if colour never came, this fails on the timeout instead of passing quietly.
    """
    rgb_backend()
    stream_kernel(frames(8))
    window = CaptureWindow(frames=8, serial=REAL_SERIAL)
    alignments(set())
    models(set())
    arrived = threading.Event()
    window.worker.colour.connect(lambda _jpeg: arrived.set())
    during: list[int] = []

    def load(
        path: Path, _target_points: int, _depth_band: object = None
    ) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        if index >= 1 and not during:
            assert arrived.wait(10.0), "no colour frame arrived during the capture"
            during.append(index)
        base = float(index)
        return ((base, 0.0, 100.0), (0.0, base, 101.0), (base, base, 102.0))

    monkeypatch.setattr(gui, "_load_cloud", load)

    start_capture(window)

    assert window.colours[:1] == [JPEG]
    # Recorded while frames were still being written, not after the run.
    assert during and during[0] < 7
    assert len(window.frames_on_disk()) == 8


def test_colour_failing_does_not_cost_the_capture_a_single_frame(
    stream_kernel, capture_dir: Path, live, rgb_backend
) -> None:
    """The whole point of "non-fatal": the scan is what must survive."""
    backend = rgb_backend(error=RuntimeError("no RGB node"))
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6, serial=REAL_SERIAL)
    live()

    start_capture(window)

    assert len(window.frames_on_disk()) == 6
    assert window.completed != []
    assert window.colours == []
    assert backend.calls >= 1


def test_a_capture_without_a_serial_simply_has_no_colour(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """A scanner that will not say who it is is still perfectly capturable.

    The RGB node is resolved by serial, so there is nothing to aim a grabber at.
    Refusing the capture over it would trade the thing being recorded for a
    convenience view.
    """
    stream_kernel(frames(4))
    window = CaptureWindow(frames=4, serial=None)
    made = recording_pump(window)
    live()

    start_capture(window)

    assert made == []
    assert len(window.frames_on_disk()) == 4
    assert window.completed != []


def test_the_pump_is_stopped_when_a_capture_finishes_on_its_own(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """A completed capture leaves nothing running against the RGB node."""
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()

    start_capture(window)

    assert [pump.stops for pump in made] == [1]
    assert window.completed != []


def test_the_pump_is_stopped_when_the_operator_presses_stop(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """Stop stops the colour camera too, at the same boundary as the depth one."""
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()
    window.worker.saved.connect(lambda saved: window._stop() if saved == 2 else None)

    start_capture(window)

    assert window.stopped == 1
    assert [pump.stops for pump in made] == [1]


def test_the_pump_is_stopped_when_the_stream_itself_fails(
    stream_kernel, capture_dir: Path, monkeypatch, live, recording_pump
) -> None:
    """The path nobody plans for, which is exactly where a leak survives.

    An exception on the way out of the stream is still a way out, and a pump
    left running by it would hold the RGB node against the next attempt - which
    is the attempt made immediately after something went wrong.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5, serial=REAL_SERIAL)
    made = recording_pump(window)
    live()
    real = v4l2_stream.stream_frames

    def fails(**kwargs: object) -> object:
        real(**kwargs)  # type: ignore[arg-type]
        raise OSError("the node went away")

    monkeypatch.setattr(v4l2_stream, "stream_frames", fails)

    with pytest.raises(OSError, match="the node went away"):
        start_capture(window)

    assert [pump.stops for pump in made] == [1]
    assert made[0].starts == 1


# -- which scanner colour is aimed at ---------------------------------------


def scanners(monkeypatch, *serials: str) -> None:
    """What enumeration reports, in the order it happens to report it.

    Order is the whole point of these tests: the first entry is deliberately
    NOT the one that should be chosen in either case below.
    """
    devices = tuple(SimpleNamespace(serial=serial) for serial in serials)
    monkeypatch.setattr(discovery, "find_scanners", lambda *_a, **_k: devices)


def resolved() -> str | None:
    """The real ``_scanner_serial``, against nothing but the resolver."""
    return ScannerWindow._scanner_serial(cast(ScannerWindow, SimpleNamespace()))


def test_colour_is_aimed_at_the_configured_scanner_not_the_first_one_found(
    monkeypatch,
) -> None:
    """Two units on the bench, one of them chosen in the configuration.

    The depth stream resolves its node through
    :func:`~scanner3d.config.resolve_serial`, so aiming colour at whatever
    enumeration listed first means the framing view can show a different
    scanner from the one being recorded - and both pictures look perfectly
    plausible on their own, which is what makes it worth a test.
    """
    scanners(monkeypatch, "OTHERSCANNER0002", REAL_SERIAL, "OTHERSCANNER0003")
    monkeypatch.setenv("SCANNER3D_SERIAL", REAL_SERIAL)

    assert resolved() == REAL_SERIAL


def test_several_scanners_and_no_configuration_resolves_to_no_colour(monkeypatch) -> None:
    """Ambiguity refuses rather than picking the first one.

    Colour is a convenience, so the cost of guessing is not a crash - it is a
    framing view of the wrong object, silently, for as long as the operator
    keeps scanning. ``None`` costs them the colour panel and nothing else: the
    depth stream still resolves its own node, and still refuses in its own
    words if it cannot.
    """
    scanners(monkeypatch, "OTHERSCANNER0002", REAL_SERIAL)

    assert resolved() is None


def test_no_scanner_at_all_resolves_to_no_colour(monkeypatch) -> None:
    """Unavailability answers the same way ambiguity does, and never raises."""
    scanners(monkeypatch)

    assert resolved() is None


def test_enumeration_blowing_up_resolves_to_no_colour(monkeypatch) -> None:
    """The GUI thread is the caller: this may not raise, whatever sysfs did."""

    def explode(*_a: object, **_k: object) -> object:
        raise OSError("sysfs is not readable")

    monkeypatch.setattr(discovery, "find_scanners", explode)

    assert resolved() is None


def test_one_attached_scanner_and_no_configuration_still_gets_colour(monkeypatch) -> None:
    """The ordinary bench. Refusing ambiguity must not refuse the normal case."""
    scanners(monkeypatch, REAL_SERIAL)

    assert resolved() == REAL_SERIAL


# -- a picture that outlived its stream -------------------------------------

#: What a LATER run would put on screen. Distinct bytes from :data:`JPEG` so a
#: test can say which run's picture the window is showing.
NEXT_JPEG = b"\xff\xd8\xff\xe0" + b"the next scan" + b"\xff\xd9"


def test_a_picture_still_waiting_when_the_stream_ends_is_never_shown(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """Stopping the pump is not enough; the slot has to be emptied too.

    ``stop()`` closes the door on FUTURE emissions. A picture already offered,
    with its wake-up already posted, is on the far side of that door: the GUI
    answers it whenever it next gets a turn, which can be after this capture
    has been torn down and after the next one has drawn its own first frame.
    What the operator would then see is the previous scan, arriving on top of
    the current one, with nothing about it to say so.

    The window here never answers a wake-up until the capture is over, which is
    exactly the ordering that makes the stale picture possible.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5, serial=REAL_SERIAL, deferred=True)
    made = recording_pump(window)
    live()

    start_capture(window)

    # Offered, and still undrawn when the stream ended.
    assert made[0].starts == 1
    assert window.worker.colour.offered == 1
    assert window.colours == []
    # The next run, drawing its own picture into the same panel.
    window._show_colour(NEXT_JPEG)
    # And now the old wake-up is finally answered.
    window.worker.drain()

    assert window.colours == [NEXT_JPEG]
    assert (window.worker.colour.applied, window.worker.colour.discarded) == (0, 1)
    # None of which cost the capture anything.
    assert len(window.frames_on_disk()) == 5
    assert window.completed != []


def test_a_picture_waiting_when_the_operator_stops_is_never_shown_either(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models, recording_pump
) -> None:
    """The same guarantee on the path the operator takes deliberately.

    Stop is where a stale colour frame is most misleading: the next thing that
    happens is usually another capture, and a picture from the abandoned one
    landing in it looks exactly like the new scan working.
    """
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6, serial=REAL_SERIAL, deferred=True)
    made = recording_pump(window)
    alignments(set())
    models(set())

    def load(
        path: Path, _target_points: int, _depth_band: object = None
    ) -> tuple[tuple[float, float, float], ...]:
        # From inside the capture, after the pump has offered its picture and
        # while the window is still too busy to have drawn it.
        if int(path.stem.split("_")[1]) == 2:
            window._stop()
        return ()

    monkeypatch.setattr(gui, "_load_cloud", load)

    start_capture(window)

    assert window.stopped == 1
    assert [pump.stops for pump in made] == [1]
    window.worker.drain()

    assert window.colours == []
    assert (window.worker.colour.applied, window.worker.colour.discarded) == (0, 1)


def test_a_picture_the_window_already_drew_is_not_taken_back(
    stream_kernel, capture_dir: Path, live, recording_pump
) -> None:
    """Teardown empties the slot; it does not undraw anything.

    The overwhelming majority of captures are this one - a window that keeps up
    - and the discard has to be a no-op there. Counting an applied picture as
    dropped would make the end-of-run summary describe a colour panel the
    operator was actually watching as one that never worked.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5, serial=REAL_SERIAL)
    recording_pump(window)
    live()

    start_capture(window)

    assert window.colours == [JPEG]
    assert (window.worker.colour.applied, window.worker.colour.discarded) == (1, 0)


# -- preview, which resolves colour the same way ----------------------------


def test_preview_aims_colour_at_the_configured_scanner(monkeypatch, previewing) -> None:
    """Preview and capture ask the same question of the same authority."""
    scanners(monkeypatch, "OTHERSCANNER0002", REAL_SERIAL)
    monkeypatch.setenv("SCANNER3D_SERIAL", REAL_SERIAL)
    window = PreviewWindow()

    calls = previewing(window)

    assert [call.serial for call in calls] == [REAL_SERIAL]
    assert window.frames == [b"a depth frame"]


def test_preview_still_shows_depth_when_colour_cannot_be_resolved(monkeypatch, previewing) -> None:
    """Ambiguity costs the colour panel, and must not cost the depth view.

    Two scanners and no configured serial is the case that made this worth
    stating: refusing to aim colour is right, and refusing to PREVIEW over it
    would be a Preview button that does nothing at all and explains nothing.
    The stream is still opened, still resolves its own node, and still reports
    in its own words if it cannot.
    """
    scanners(monkeypatch, "OTHERSCANNER0002", REAL_SERIAL)
    window = PreviewWindow()

    calls = previewing(window)

    assert [call.serial for call in calls] == [None]
    assert window.frames == [b"a depth frame"]
    assert window.colours == []
    assert window.failures == []
