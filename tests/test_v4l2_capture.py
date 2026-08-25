"""Drive the whole bounded V4L2 capture lifecycle against a fake kernel.

``tests/test_v4l2_uapi.py`` proves the *transcription* is right: the ioctl
numbers, struct sizes and field offsets match what the host's own
``<linux/videodev2.h>`` declares. This file proves the *lifecycle* is right,
which is a different question and the one the hardware can only be asked once.

Every syscall :class:`V4l2DepthFrameBackend` makes is injected, so a test can
put the kernel in any state the real one could reach - a short REQBUFS, a
buffer past a ceiling, a poll that only ever times out, an ``EINTR`` on
``STREAMON``, a ``STREAMOFF`` that fails during cleanup - and then assert on
what was issued, in what order, and what was released. The central claims are
the ones no amount of reading the module can settle:

* the fixed bounds hold (four buffers, one poll, one DQBUF, no retry);
* every obligation created is discharged exactly once on *every* exit path,
  including the ones an exception or a signal unwinds through;
* a failure to clean up overrides a frame that otherwise succeeded.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from scanner3d.errors import V4l2ArtifactIncomplete
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
)
from scanner3d.ownership import SignalReceived
from scanner3d.v4l2_capture import (
    CAPTURE_DEADLINE_S,
    EXPECTED_FRAME_BYTES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_BUFFER_BYTES,
    MAX_RAW_ARTIFACT_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    MIN_BYTES_PER_LINE,
    REQUESTBUFFERS_FORMAT,
    REQUESTED_BUFFERS,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_MEMORY_MMAP,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    CaptureStatus,
    Step,
    V4l2DepthFrameBackend,
    V4l2FrameBackend,
    _write_new_file,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.fake_v4l2 import (
    POLLERR,
    POLLIN,
    FakeKernel,
    FakeMapping,
    SteppedClock,
    make_backend,
    noop_guard,
    noop_shield,
    sysfs,
)

__all__ = ["POLLERR", "POLLIN", "FakeMapping", "SteppedClock", "noop_guard", "noop_shield", "sysfs"]

#: Page-aligned, and exactly the ``dwMaxVideoFrameBufferSize`` the recorded
#: descriptor advertises for ``Y16 `` 640x400 on this unit.
BUFFER_LENGTH = EXPECTED_FRAME_BYTES


# --- the fake kernel ----------------------------------------------------------

# One implementation, in tests/fake_v4l2.py, shared with every other profile's
# lifecycle tests. Re-exported here by name so the assertions below read as they
# did before the engine was split into profiles.


def backend(
    kernel: FakeKernel, sysfs: SysfsBuilder, clock: Callable[[], float] | None = None
) -> V4l2FrameBackend:
    """The shared engine bound to the depth profile, driven by ``kernel``."""
    return make_backend(kernel, sysfs, clock, V4l2DepthFrameBackend)


def run(kernel: FakeKernel, sysfs: SysfsBuilder, tmp_path: Path, clock=None):
    return backend(kernel, sysfs, clock).run(REAL_SERIAL, output_dir=tmp_path / "out")


def steps_of(outcome, step: Step) -> list:
    return [s for s in outcome.steps if s.step is step]


def kinds(outcome) -> list[str]:
    """Which evidence files the run produced, in order."""
    return [f.kind for f in outcome.artifacts.files]


def artifact(outcome, kind: str):
    """The one artifact of ``kind``; fails the test if it is not there."""
    matches = [f for f in outcome.artifacts.files if f.kind == kind]
    assert len(matches) == 1, f"expected exactly one {kind!r} artifact, got {kinds(outcome)}"
    return matches[0]


# --- the happy path -----------------------------------------------------------


def test_complete_frame_issues_each_ioctl_exactly_once_in_order(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert kernel.requests == [
        VIDIOC_QUERYCAP,
        VIDIOC_G_FMT,
        VIDIOC_S_FMT,
        VIDIOC_REQBUFS,
        # QUERYBUF and mmap for every buffer first, then QBUF for every buffer:
        # nothing is queued until all four are known to be mappable.
        *([VIDIOC_QUERYBUF] * REQUESTED_BUFFERS),
        *([VIDIOC_QBUF] * REQUESTED_BUFFERS),
        VIDIOC_STREAMON,
        VIDIOC_DQBUF,
        VIDIOC_STREAMOFF,
        VIDIOC_REQBUFS,
    ]


def test_single_frame_identity_refusal_happens_before_querycap(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    base = make_backend(kernel, sysfs)
    backend = replace(
        base,
        identity_check=lambda _fd, _target: "synthetic post-open identity drift",
    )
    outcome = backend.run(REAL_SERIAL, output_dir=tmp_path / "out")
    assert outcome.status is CaptureStatus.DEVICE_REPLACED
    assert kernel.requests == []
    assert kernel.closed


def test_single_frame_shield_entry_signal_retries_cleanup_once(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    calls = 0

    def shield():
        @contextmanager
        def context():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SignalReceived(15, "synthetic shield entry")
            yield []

        return context()

    base = make_backend(kernel, sysfs)
    outcome = replace(base, termination_shield=shield).run(REAL_SERIAL, output_dir=tmp_path / "out")
    assert outcome.status is CaptureStatus.COMPLETE
    assert calls == 3
    assert [step.step for step in outcome.cleanup.steps].count(Step.STREAMOFF) == 1
    assert [step.step for step in outcome.cleanup.steps].count(Step.CLOSE) == 1
    assert "SIGTERM" in outcome.cleanup.note


def test_single_finalizer_entry_fallback_stays_shielded_and_runs_once(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    active = False
    shield_calls = 0
    factory_calls = 0
    run_calls = 0
    received: list[int] = []

    @contextmanager
    def shield() -> Iterator[list[int]]:
        nonlocal active, shield_calls
        shield_calls += 1
        if shield_calls == 2:
            raise SignalReceived(15, "synthetic finalizer shield entry")
        active = True
        try:
            yield received
        finally:
            active = False

    class ActiveFinalizer:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            nonlocal run_calls
            assert active
            run_calls += 1
            received.append(15)
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.COMMAND_ACCEPTED,
                "synthetic fixed off accepted",
                transfer=ShutdownTransferRecord(60, True, "synthetic", 0.0),
            )

    def factory(_sysfs, _dev):
        nonlocal factory_calls
        assert active
        factory_calls += 1
        return ActiveFinalizer()

    kernel = FakeKernel()
    outcome = replace(
        backend(kernel, sysfs),
        termination_shield=shield,
        illumination_backend_factory=factory,
    ).run(REAL_SERIAL, output_dir=tmp_path / "out")
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.illumination_shutdown.succeeded
    assert outcome.illumination_shutdown.deferred_signals == (15,)
    assert factory_calls == 1
    assert run_calls == 1
    assert shield_calls == 3


def test_single_frame_guard_overlaps_first_release_shield_takeover(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    events: list[tuple[str, bool]] = []
    guarded = False

    @contextmanager
    def guard() -> Iterator[None]:
        nonlocal guarded
        events.append(("guard-enter", guarded))
        guarded = True
        try:
            yield
        finally:
            guarded = False
            events.append(("guard-exit", guarded))

    @contextmanager
    def shield() -> Iterator[list[int]]:
        events.append(("shield-enter", guarded))
        yield []
        events.append(("shield-exit", guarded))

    base = make_backend(kernel, sysfs)
    outcome = replace(base, signal_guard=guard, termination_shield=shield).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert events == [
        ("guard-enter", False),
        ("shield-enter", True),
        ("shield-exit", True),
        ("guard-exit", False),
        ("shield-enter", False),
        ("shield-exit", False),
    ]


def test_complete_frame_opens_the_role_resolved_node_not_video0(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert kernel.opened == [str(sysfs.dev / "video5")]
    assert outcome.target is not None
    assert outcome.target.node_name == "video5"
    assert outcome.target.role == "depth"
    assert outcome.target.v4l_index == 0


def test_complete_frame_maps_every_granted_buffer_once_within_the_ceilings(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    run(kernel, sysfs, tmp_path)
    assert len(kernel.map_calls) == REQUESTED_BUFFERS
    assert [offset for _, _, offset in kernel.map_calls] == [
        i * BUFFER_LENGTH for i in range(REQUESTED_BUFFERS)
    ]
    assert all(length == BUFFER_LENGTH for _, length, _ in kernel.map_calls)
    assert all(length <= MAX_BUFFER_BYTES for _, length, _ in kernel.map_calls)


def test_complete_frame_polls_exactly_once_for_what_is_left_of_the_deadline(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert len(kernel.polls) == 1
    # The budget is the deadline minus the one 1ms clock read between the
    # STREAMON that anchored it and the poll: never more than the deadline,
    # and never restarted at the full value.
    assert kernel.polls[0][1] == pytest.approx(CAPTURE_DEADLINE_S - 0.001)
    assert kernel.polls[0][1] <= CAPTURE_DEADLINE_S
    assert outcome.poll is not None
    assert outcome.poll.deadline_s == CAPTURE_DEADLINE_S
    assert outcome.poll.timeout_s == kernel.polls[0][1]


def test_complete_frame_writes_a_raw_artifact_of_exactly_the_frame_size(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE
    raw = artifact(outcome, "raw")
    assert raw.ok and raw.path is not None
    assert raw.byte_count == EXPECTED_FRAME_BYTES
    payload = Path(raw.path).read_bytes()
    assert len(payload) == EXPECTED_FRAME_BYTES
    assert payload == bytes([kernel.frame_byte]) * EXPECTED_FRAME_BYTES
    assert raw.sha256 == hashlib.sha256(payload).hexdigest()


def test_a_tight_frame_writes_no_second_file_and_says_the_hash_is_the_raw_one(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """Nothing to pack when the rows are already tight; a byte-identical second
    file would invite its hash being read as independent confirmation."""
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    packed = artifact(outcome, "packed")
    assert packed.ok is True
    assert packed.path is None
    assert packed.sha256 == artifact(outcome, "raw").sha256
    assert "already tightly packed" in packed.detail
    written = sorted(q.name for q in (tmp_path / "out").iterdir())
    assert len(written) == 1


def test_complete_frame_preserves_the_kernel_buffer_metadata(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(dequeued_index=2)
    outcome = run(kernel, sysfs, tmp_path)
    dequeued = outcome.dequeued
    assert dequeued is not None
    assert dequeued.index == 2
    assert dequeued.bytesused == EXPECTED_FRAME_BYTES
    assert dequeued.sequence == 42
    assert dequeued.timestamp_sec == 1700000000
    assert dequeued.timestamp_usec == 123456
    assert dequeued.length == BUFFER_LENGTH
    assert dequeued.memory == V4L2_MEMORY_MMAP


def test_complete_frame_preserves_the_negotiated_and_initial_formats(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.negotiated_format is not None
    assert outcome.negotiated_format.width == FRAME_WIDTH
    assert outcome.negotiated_format.height == FRAME_HEIGHT
    assert outcome.negotiated_format.fourcc == "Y16 "
    assert outcome.negotiated_format.bytesperline == MIN_BYTES_PER_LINE
    # G_FMT ran before S_FMT, so the handle's prior format is on record.
    assert outcome.initial_format is not None


def test_complete_frame_preserves_the_capability_fields(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.capability is not None
    assert outcome.capability.driver == "uvcvideo"
    assert outcome.capability.bus_info == "usb-0000:00:14.0-1.1"
    assert outcome.capability.version_text == "6.0.8"
    assert outcome.capability.missing_required == ()


def test_success_requires_both_a_complete_frame_and_clean_cleanup(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is True
    assert outcome.succeeded is True


# --- the padded frame ---------------------------------------------------------

#: A padded row whose whole-frame size stays a multiple of the 4 KiB page,
#: so the per-buffer mmap offsets remain page-aligned the way the kernel's
#: own offsets are. 1536 * 400 == 614400 == 150 pages.
PADDED_BPL = MIN_BYTES_PER_LINE + 256


def test_padded_frame_preserves_the_raw_bytes_and_packs_a_tight_copy(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    padded_size = PADDED_BPL * FRAME_HEIGHT
    kernel = FakeKernel(bytesperline=PADDED_BPL, buffer_length=padded_size)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    raw = artifact(outcome, "raw")
    packed = artifact(outcome, "packed")
    assert raw.byte_count == padded_size
    assert packed.byte_count == EXPECTED_FRAME_BYTES
    assert packed.path is not None
    assert Path(packed.path).stat().st_size == EXPECTED_FRAME_BYTES
    # Separately named hashes: the padded evidence and the tight copy are not
    # the same bytes and must never be quoted as one number.
    assert raw.sha256 != packed.sha256
    assert packed.sha256 == hashlib.sha256(Path(packed.path).read_bytes()).hexdigest()


def test_padded_frame_refuses_to_pack_when_a_row_would_run_past_the_bytes(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """``bytesused`` covers fewer rows than the height claims: no tight copy."""
    padded_size = PADDED_BPL * FRAME_HEIGHT
    kernel = FakeKernel(
        bytesperline=PADDED_BPL,
        buffer_length=padded_size,
        sizeimage=padded_size,
        bytesused=padded_size - PADDED_BPL,
    )
    outcome = run(kernel, sysfs, tmp_path)
    # Short of what the negotiated geometry needs, so it is not called a frame.
    assert outcome.status is CaptureStatus.SHORT_FRAME
    assert outcome.succeeded is False


def test_a_bytesperline_below_a_tight_row_is_refused_before_streaming(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(bytesperline=MIN_BYTES_PER_LINE - 2)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.FORMAT_MISMATCH
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.count(VIDIOC_REQBUFS) == 0


# --- format and capability refusals -------------------------------------------


@pytest.mark.parametrize(
    "make_kernel",
    [
        pytest.param(lambda: FakeKernel(width=1280), id="width"),
        pytest.param(lambda: FakeKernel(height=800), id="height"),
        pytest.param(lambda: FakeKernel(pixelformat=0x32595559), id="pixelformat-yuy2"),
        pytest.param(lambda: FakeKernel(buf_type=2), id="buffer-type"),
    ],
)
def test_a_format_the_kernel_changed_is_refused_before_any_buffer_exists(
    sysfs: SysfsBuilder, tmp_path: Path, make_kernel: Callable[[], FakeKernel]
) -> None:
    """S_FMT may legally return something else; that ends the run, it is not retried."""
    kernel = make_kernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.FORMAT_MISMATCH
    assert kernel.count(VIDIOC_REQBUFS) == 0
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.count(VIDIOC_S_FMT) == 1  # no second try at another mode
    assert kernel.mappings == []
    assert outcome.cleanup.ok is True
    assert kernel.closed  # the descriptor still went back


@pytest.mark.parametrize(
    "device_caps",
    [
        pytest.param(V4L2_CAP_STREAMING, id="no-video-capture"),
        pytest.param(V4L2_CAP_VIDEO_CAPTURE, id="no-streaming"),
        pytest.param(0, id="neither"),
    ],
)
def test_missing_capabilities_stop_the_run_at_querycap(
    sysfs: SysfsBuilder, tmp_path: Path, device_caps: int
) -> None:
    kernel = FakeKernel(device_caps=device_caps)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.CAPABILITY_MISMATCH
    assert kernel.count(VIDIOC_S_FMT) == 0
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.closed


# --- buffer-count bounds ------------------------------------------------------


@pytest.mark.parametrize("granted", [0, 1])
def test_too_few_granted_buffers_are_refused_and_released(
    sysfs: SysfsBuilder, tmp_path: Path, granted: int
) -> None:
    kernel = FakeKernel(granted_buffers=granted)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.BUFFER_COUNT_MISMATCH
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.closed


@pytest.mark.parametrize("granted", [2, 3, 4])
def test_a_granted_count_inside_the_accepted_range_streams(
    sysfs: SysfsBuilder, tmp_path: Path, granted: int
) -> None:
    kernel = FakeKernel(granted_buffers=granted)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert len(kernel.map_calls) == granted
    assert kernel.count(VIDIOC_QBUF) == granted


def test_more_buffers_than_requested_are_refused(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    """The kernel must never be trusted to have honoured the cap on its own."""
    kernel = FakeKernel(granted_buffers=REQUESTED_BUFFERS + 1)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.BUFFER_COUNT_MISMATCH
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0


# --- mapping bounds -----------------------------------------------------------


def test_a_buffer_past_the_per_buffer_ceiling_is_never_mapped(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(buffer_length=MAX_BUFFER_BYTES + 4096)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.MAPPING_ERROR
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0


def test_the_two_ceilings_agree_so_the_largest_legal_run_is_exactly_the_total(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The most this module can ever map is four buffers at the per-buffer
    ceiling, and that lands exactly on the total ceiling rather than over it.

    This is the reason the total bound can only ever be a belt-and-braces one:
    with the count capped at four and each buffer capped at 1 MiB there is no
    granted-count path that exceeds 4 MiB. ``_mapping_refusal`` is unit-tested
    against an over-ceiling running total directly in ``test_v4l2_uapi.py``;
    what matters here is that the largest reachable run is still accepted and
    still releases everything.
    """
    assert MAX_BUFFER_BYTES * REQUESTED_BUFFERS == MAX_TOTAL_MAPPED_BYTES
    kernel = FakeKernel(buffer_length=MAX_BUFFER_BYTES, buffer_offset_step=MAX_BUFFER_BYTES)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert sum(length for _, length, _ in kernel.map_calls) == MAX_TOTAL_MAPPED_BYTES
    assert all(m.closes == 1 for m in kernel.mappings)


def test_a_mapping_that_raises_is_recorded_and_earlier_ones_are_released(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    calls: list[int] = []
    real_mmap = kernel.mmap

    def failing_mmap(fd: int, length: int, offset: int) -> FakeMapping:
        calls.append(offset)
        if len(calls) == 3:
            raise OSError(errno.ENOMEM, "Cannot allocate memory")
        return real_mmap(fd, length, offset)

    outcome = replace(backend(kernel, sysfs), mmap_fn=failing_mmap).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.MAPPING_ERROR
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert [m.closes for m in kernel.mappings] == [1, 1]
    assert outcome.cleanup.ok is True


# --- the deadline and the single poll -----------------------------------------


def test_a_poll_timeout_is_a_bounded_zero_frame_result_not_a_retry(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(poll_revents=0)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.POLL_TIMEOUT
    assert len(kernel.polls) == 1
    assert kernel.polls[0][1] == pytest.approx(CAPTURE_DEADLINE_S - 0.001)
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert kernel.count(VIDIOC_STREAMON) == 1  # never started a second time
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert outcome.cleanup.ok is True
    assert outcome.succeeded is False
    # A timeout still leaves nothing mapped and nothing held.
    assert all(m.closes == 1 for m in kernel.mappings)
    assert kernel.closed


@pytest.mark.parametrize("revents", [POLLERR, 16, 32])
def test_a_poll_error_event_is_not_treated_as_readiness(
    sysfs: SysfsBuilder, tmp_path: Path, revents: int
) -> None:
    kernel = FakeKernel(poll_revents=revents)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.POLL_ERROR
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert outcome.cleanup.ok is True


def test_the_poll_records_the_events_it_saw(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.poll is not None
    assert "POLLIN" in outcome.poll.event_names
    assert outcome.poll.deadline_s == CAPTURE_DEADLINE_S


def test_a_slow_streamon_is_not_charged_against_the_one_poll(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The deadline starts when the stream does, not when the ioctl was issued.

    ``VIDIOC_STREAMON`` can take real time on this device. Anchoring the
    budget before it would silently shorten the only wait the run gets, so a
    three-second STREAMON must leave the whole deadline intact.
    """
    kernel = FakeKernel()
    clock = SteppedClock()
    real_ioctl = kernel.ioctl

    def slow_streamon(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        result = real_ioctl(fd, request, packed, mutate)
        if request == VIDIOC_STREAMON:
            clock.advance(3.0)
        return result

    outcome = replace(backend(kernel, sysfs, clock), ioctl_fn=slow_streamon).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert len(kernel.polls) == 1
    assert kernel.polls[0][1] == pytest.approx(CAPTURE_DEADLINE_S - clock.tick)
    assert kernel.polls[0][1] > CAPTURE_DEADLINE_S - 3.0
    streamon_steps = steps_of(outcome, Step.STREAMON)
    assert len(streamon_steps) == 1
    assert streamon_steps[0].duration_s is not None
    assert streamon_steps[0].duration_s >= 3.0


def test_a_budget_already_spent_polls_once_without_blocking_and_never_retries(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """Zero remaining time is still exactly one poll, given a zero timeout."""
    kernel = FakeKernel(poll_revents=0)
    clock = SteppedClock()
    real_ioctl = kernel.ioctl

    def then_time_flies(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        result = real_ioctl(fd, request, packed, mutate)
        if request == VIDIOC_STREAMON:
            # Every clock read after the stream starts costs more than the
            # whole deadline, so nothing is left by the time the poll is made.
            clock.tick = CAPTURE_DEADLINE_S + 1.0
        return result

    outcome = replace(backend(kernel, sysfs, clock), ioctl_fn=then_time_flies).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.POLL_TIMEOUT
    assert len(kernel.polls) == 1
    assert kernel.polls[0][1] == 0.0
    assert outcome.poll is not None
    assert outcome.poll.timeout_s == 0.0
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert kernel.count(VIDIOC_STREAMON) == 1
    assert outcome.cleanup.ok is True


@pytest.mark.parametrize(
    ("raised", "expected_errno"),
    [
        (OSError(errno.EIO, os.strerror(errno.EIO)), errno.EIO),
        (OSError(errno.ENODEV, os.strerror(errno.ENODEV)), errno.ENODEV),
        (RuntimeError("something no one predicted inside poll"), None),
        (ValueError("a descriptor the poller would not take"), None),
    ],
)
def test_a_poll_that_raises_is_one_failed_poll_step_and_a_poll_error(
    sysfs: SysfsBuilder, tmp_path: Path, raised: Exception, expected_errno: int | None
) -> None:
    """A raised poll is a *poll* failure, not an anonymous capture-window bug.

    Without this, an ``OSError`` out of ``poll`` unwound to the generic window
    handler and was reported as an ioctl error with no ``Step.POLL`` recorded
    at all - the run looked like it had never waited.
    """
    kernel = FakeKernel()

    def raising_poll(fd: int, timeout_s: float) -> int:
        kernel.polls.append((fd, timeout_s))
        raise raised

    outcome = replace(backend(kernel, sysfs), poll_fn=raising_poll).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.POLL_ERROR
    assert outcome.errno == expected_errno
    poll_steps = steps_of(outcome, Step.POLL)
    assert len(poll_steps) == 1
    assert poll_steps[0].ok is False
    assert poll_steps[0].errno == expected_errno
    assert type(raised).__name__ in poll_steps[0].detail
    assert outcome.poll is not None
    assert outcome.poll.failed is True
    assert outcome.poll.timed_out is False
    # One poll, no dequeue, and every obligation still discharged.
    assert len(kernel.polls) == 1
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert kernel.count(VIDIOC_STREAMON) == 1
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert outcome.cleanup.ok is True
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1
    assert outcome.succeeded is False


def test_exactly_one_dqbuf_is_issued_and_the_frame_is_never_requeued(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    run(kernel, sysfs, tmp_path)
    assert kernel.count(VIDIOC_DQBUF) == 1
    # Four QBUFs, one per buffer, all before STREAMON: none after the dequeue.
    qbuf_positions = [i for i, r in enumerate(kernel.requests) if r == VIDIOC_QBUF]
    streamon_at = kernel.requests.index(VIDIOC_STREAMON)
    assert all(pos < streamon_at for pos in qbuf_positions)


# --- frame size bounds --------------------------------------------------------


def test_a_short_frame_is_preserved_as_evidence_but_is_not_called_a_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(bytesused=EXPECTED_FRAME_BYTES - 2)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.SHORT_FRAME
    assert outcome.succeeded is False
    raw = artifact(outcome, "raw")
    assert raw.byte_count == EXPECTED_FRAME_BYTES - 2
    assert raw.path is not None


def test_a_frame_larger_than_the_tight_geometry_is_refused_not_called_complete(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The rows are tight, so 512000 is the whole frame; more is unexplained."""
    kernel = FakeKernel(buffer_length=MAX_BUFFER_BYTES, bytesused=EXPECTED_FRAME_BYTES + 2)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.OVERSIZE_FRAME
    assert outcome.succeeded is False


def test_bytesused_past_the_mapped_length_copies_nothing(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(bytesused=BUFFER_LENGTH + 4096)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.OVERSIZE_FRAME
    assert outcome.frame_bytes is None
    assert "nothing was copied" in outcome.detail


def test_a_sizeimage_past_the_artifact_ceiling_is_refused_before_streaming(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The earliest bound wins, and it fires before anything is allocated.

    The artifact ceiling and the per-buffer mapping ceiling are both 1 MiB, and
    the format check already refuses a ``sizeimage`` past it - so a frame that
    large can never reach the copy. That makes the ceiling inside ``_take_frame``
    a belt-and-braces bound rather than the first line of defence, which is what
    this test pins: no REQBUFS, no mapping, no STREAMON and no file.
    """
    kernel = FakeKernel(
        bytesperline=PADDED_BPL,
        sizeimage=MAX_RAW_ARTIFACT_BYTES + 4096,
    )
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.FORMAT_MISMATCH
    assert kernel.count(VIDIOC_REQBUFS) == 0
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.map_calls == []
    assert outcome.frame_bytes is None
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_a_buffer_the_kernel_flagged_with_an_error_is_not_a_trustworthy_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(buffer_flags=0x0040)  # V4L2_BUF_FLAG_ERROR
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.SHORT_FRAME
    assert outcome.succeeded is False
    # The bytes are still preserved: they are evidence even when untrustworthy.
    assert artifact(outcome, "raw").byte_count == EXPECTED_FRAME_BYTES


def test_a_dequeued_index_that_was_never_mapped_is_a_dequeue_error(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(granted_buffers=2, dequeued_index=3)
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.DEQUEUE_ERROR
    assert outcome.frame_bytes is None


# --- ioctl failures at every stage --------------------------------------------


@pytest.mark.parametrize(
    ("request_", "expected"),
    [
        # QUERYCAP and S_FMT have no stage-specific failure status: the
        # *_MISMATCH ones mean "the ioctl succeeded and the answer is wrong",
        # which is a different fact from "the ioctl failed", and flattening the
        # two would throw away the errno. Both are covered separately below.
        pytest.param(VIDIOC_QUERYCAP, CaptureStatus.IOCTL_ERROR, id="querycap"),
        pytest.param(VIDIOC_S_FMT, CaptureStatus.IOCTL_ERROR, id="s_fmt"),
        pytest.param(VIDIOC_REQBUFS, CaptureStatus.BUFFER_COUNT_MISMATCH, id="reqbufs"),
        pytest.param(VIDIOC_QUERYBUF, CaptureStatus.MAPPING_ERROR, id="querybuf"),
        pytest.param(VIDIOC_QBUF, CaptureStatus.QUEUE_ERROR, id="qbuf"),
        pytest.param(VIDIOC_STREAMON, CaptureStatus.STREAMON_ERROR, id="streamon"),
        pytest.param(VIDIOC_DQBUF, CaptureStatus.DEQUEUE_ERROR, id="dqbuf"),
    ],
)
def test_each_stage_failure_gets_its_own_status_and_still_releases_everything(
    sysfs: SysfsBuilder, tmp_path: Path, request_: int, expected: CaptureStatus
) -> None:
    kernel = FakeKernel(errors={request_: errno.EINVAL})
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is expected, outcome.detail
    assert outcome.succeeded is False
    assert outcome.errno == errno.EINVAL
    # Whatever was created before the failure is released, exactly once each.
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1
    # The failing ioctl is issued once and never tried again.
    assert kernel.count(request_) == 1


@pytest.mark.parametrize(
    "request_", [VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_STREAMON, VIDIOC_DQBUF]
)
def test_no_stage_failure_is_ever_retried(
    sysfs: SysfsBuilder, tmp_path: Path, request_: int
) -> None:
    kernel = FakeKernel(errors={request_: errno.EBUSY})
    run(kernel, sysfs, tmp_path)
    assert kernel.count(request_) == 1
    assert kernel.count(VIDIOC_STREAMON) <= 1
    assert len(kernel.polls) <= 1


def test_a_failed_streamon_does_not_poll_or_dequeue(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    kernel = FakeKernel(errors={VIDIOC_STREAMON: errno.EINVAL})
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.STREAMON_ERROR
    assert kernel.polls == []
    assert kernel.count(VIDIOC_DQBUF) == 0


def test_an_open_failure_is_reported_and_nothing_else_is_attempted(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()

    def refuse(path: str) -> int:
        raise OSError(errno.EACCES, "Permission denied")

    outcome = replace(backend(kernel, sysfs), opener=refuse).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status in {CaptureStatus.OPEN_ERROR, CaptureStatus.PERMISSION_DENIED}
    assert kernel.calls == []
    assert kernel.closed == []
    assert outcome.cleanup.attempted is False


# --- cleanup combinations -----------------------------------------------------


def test_a_streamoff_failure_overrides_an_otherwise_complete_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The frame arrived, but the queue may still be running: not a success."""
    kernel = FakeKernel(errors={VIDIOC_STREAMOFF: errno.EIO})
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    # The other obligations are still discharged rather than skipped.
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_a_release_buffers_failure_overrides_an_otherwise_complete_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    calls: list[int] = []
    kernel = FakeKernel()
    real_ioctl = kernel.ioctl

    def failing(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if request == VIDIOC_REQBUFS:
            calls.append(struct.unpack(REQUESTBUFFERS_FORMAT, bytes(packed))[0])
            if calls[-1] == 0:  # the release, not the allocation
                raise OSError(errno.EBUSY, "Device or resource busy")
        return real_ioctl(fd, request, packed, mutate)

    outcome = replace(backend(kernel, sysfs), ioctl_fn=failing).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    assert len(kernel.closed) == 1


def test_a_close_failure_overrides_an_otherwise_complete_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()

    def failing_close(fd: int) -> None:
        kernel.closed.append(fd)
        raise OSError(errno.EIO, "Input/output error")

    outcome = replace(backend(kernel, sysfs), closer=failing_close).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False


def test_every_cleanup_step_is_attempted_even_when_the_first_one_fails(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """A STREAMOFF failure must not skip the unmap, the release or the close."""
    kernel = FakeKernel(errors={VIDIOC_STREAMOFF: errno.EIO})
    outcome = run(kernel, sysfs, tmp_path)
    recorded = {s.step for s in outcome.steps}
    assert Step.STREAMOFF in recorded
    assert Step.UNMAP in recorded
    assert Step.RELEASE_BUFFERS in recorded
    assert Step.CLOSE in recorded
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_cleanup_runs_once_even_though_the_release_is_reachable_twice(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    run(kernel, sysfs, tmp_path)
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert len([r for r in kernel.requests if r == VIDIOC_REQBUFS]) == 2  # allocate + release
    assert len(kernel.closed) == 1
    assert all(m.closes == 1 for m in kernel.mappings)


# --- signal and exception boundaries ------------------------------------------


@pytest.mark.parametrize(
    "request_",
    [VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_QBUF, VIDIOC_STREAMON, VIDIOC_DQBUF],
)
def test_a_signal_raised_out_of_any_ioctl_still_releases_everything(
    sysfs: SysfsBuilder, tmp_path: Path, request_: int
) -> None:
    """``KeyboardInterrupt`` is not an ``OSError``: it unwinds past every
    ``except OSError`` in the module, so only the ``finally`` can save it."""
    kernel = FakeKernel(raises={request_: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(kernel, sysfs, tmp_path)
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_a_signal_during_streamon_still_attempts_streamoff(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """STREAMON is owed from *before* the ioctl: an interrupted one may have landed."""
    kernel = FakeKernel(raises={VIDIOC_STREAMON: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(kernel, sysfs, tmp_path)
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert len(kernel.closed) == 1


def test_a_signal_during_the_poll_still_releases_everything(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()

    def interrupted_poll(fd: int, timeout_s: float) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        replace(backend(kernel, sysfs), poll_fn=interrupted_poll).run(
            REAL_SERIAL, output_dir=tmp_path / "out"
        )
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_an_unexpected_exception_becomes_a_reported_outcome_not_a_traceback(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """An ``Exception`` no one predicted is still an *inspectable* result.

    The contrast with the signal tests above is deliberate and is the whole
    point of the split: ``KeyboardInterrupt`` is a ``BaseException`` and
    propagates, because a caller asking to stop must be obeyed, while an
    ordinary bug is caught, named and reported with everything released - a
    traceback out of the capture window would tell the owner nothing about
    whether the stream was left running.
    """
    kernel = FakeKernel()
    seen: list[int] = []
    real_mmap = kernel.mmap

    def exploding(fd: int, length: int, offset: int) -> FakeMapping:
        seen.append(offset)
        if len(seen) == 2:
            raise RuntimeError("something no one predicted")
        return real_mmap(fd, length, offset)

    outcome = replace(backend(kernel, sysfs), mmap_fn=exploding).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.succeeded is False
    assert "RuntimeError" in outcome.detail
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert [m.closes for m in kernel.mappings] == [1]
    assert len(kernel.closed) == 1


def test_the_shield_absorbs_signals_arriving_during_cleanup(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """Cleanup must finish; a signal that arrives inside it is recorded, not obeyed."""
    import signal as signal_mod

    @contextmanager
    def shield_that_caught_sigterm() -> Iterator[list[int]]:
        caught = [int(signal_mod.SIGTERM)]
        yield caught

    kernel = FakeKernel()
    outcome = replace(backend(kernel, sysfs), termination_shield=shield_that_caught_sigterm).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert "SIGTERM" in outcome.cleanup.note
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


# --- artifact writes ----------------------------------------------------------


def _patch_artifact_writes(
    monkeypatch: pytest.MonkeyPatch, *, chunk: int | None = None, fail_after: int | None = None
) -> list[int]:
    """Make ``os.write`` behave the way a real short write can, and count them.

    Only descriptors this module creates exclusively are affected; everything
    else - pytest's own captured output included - keeps writing normally.
    ``chunk`` caps how much each call accepts, ``fail_after`` raises an
    ``OSError`` once that many calls have been made, and ``chunk=0`` returns
    zero bytes written, which is the case a loop that trusts progress spins on
    forever.
    """
    real_open, real_write, real_close = os.open, os.write, os.close
    artifact_fds: set[int] = set()
    calls: list[int] = []

    def spy_open(path, flags, mode=0o777, **kwargs):
        fd = real_open(path, flags, mode, **kwargs)
        if flags & os.O_EXCL:
            artifact_fds.add(fd)
        return fd

    def spy_close(fd: int) -> None:
        artifact_fds.discard(fd)
        real_close(fd)

    def short_write(fd: int, data) -> int:
        if fd not in artifact_fds:
            return real_write(fd, data)
        calls.append(len(data))
        if fail_after is not None and len(calls) > fail_after:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        if chunk == 0:
            return 0
        return real_write(fd, data if chunk is None else data[:chunk])

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(os, "write", short_write)
    return calls


def test_a_short_write_is_finished_rather_than_called_a_whole_file(
    sysfs: SysfsBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.write`` may accept less than it was given; the file is still whole."""
    calls = _patch_artifact_writes(monkeypatch, chunk=64 * 1024)
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert len(calls) > 1, "a chunked write should have taken more than one call"
    raw = artifact(outcome, "raw")
    assert raw.ok is True and raw.path is not None
    written = Path(raw.path).read_bytes()
    assert len(written) == EXPECTED_FRAME_BYTES
    assert hashlib.sha256(written).hexdigest() == raw.sha256
    assert outcome.artifacts.ok is True
    assert outcome.succeeded is True


def test_a_write_that_makes_no_progress_produces_no_artifact_and_no_file(
    sysfs: SysfsBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-byte write ends the attempt; it never spins and never half-writes."""
    calls = _patch_artifact_writes(monkeypatch, chunk=0)
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert len(calls) == 1, "zero progress must stop after the call that made none"
    raw = artifact(outcome, "raw")
    assert raw.ok is False
    assert raw.path is None
    assert "no progress" in raw.detail
    assert outcome.artifacts.ok is False
    # The frame itself did arrive; only its evidence did not survive.
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.succeeded is False
    assert list((tmp_path / "out").iterdir()) == []


def test_a_write_that_fails_partway_leaves_no_partial_file_behind(
    sysfs: SysfsBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with some of a frame in it must not survive under the frame's name."""
    calls = _patch_artifact_writes(monkeypatch, chunk=1024, fail_after=1)
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    assert len(calls) == 2
    raw = artifact(outcome, "raw")
    assert raw.ok is False
    assert raw.path is None
    assert outcome.artifacts.ok is False
    assert outcome.succeeded is False
    assert list((tmp_path / "out").iterdir()) == []


def test_writing_more_than_the_artifact_ceiling_is_refused_before_the_file_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "too-big.bin"
    with pytest.raises(V4l2ArtifactIncomplete) as caught:
        _write_new_file(path, b"\x00" * (MAX_RAW_ARTIFACT_BYTES + 1))
    assert caught.value.errno == errno.EFBIG
    assert not path.exists()


def test_a_complete_write_of_the_largest_allowed_payload_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "at-the-ceiling.bin"
    _write_new_file(path, b"\xa5" * MAX_RAW_ARTIFACT_BYTES)
    assert path.stat().st_size == MAX_RAW_ARTIFACT_BYTES


# --- the dry run and the serial gate ------------------------------------------


def test_the_dry_run_opens_nothing_and_issues_no_ioctl(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    kernel = FakeKernel()
    outcome = backend(kernel, sysfs).preflight(REAL_SERIAL, output_dir=tmp_path / "out")
    assert outcome.status is CaptureStatus.NOT_ATTEMPTED
    assert kernel.opened == []
    assert kernel.calls == []
    assert kernel.map_calls == []
    assert kernel.polls == []
    assert outcome.cleanup.attempted is False


def test_the_dry_run_still_resolves_the_node_by_role(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    kernel = FakeKernel()
    outcome = backend(kernel, sysfs).preflight(REAL_SERIAL, output_dir=tmp_path / "out")
    assert outcome.target is not None
    assert outcome.target.node_name == "video5"
    assert outcome.target.role == "depth"


def test_any_serial_but_the_authorized_one_is_refused_before_resolution(
    sysfs: SysfsBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guard refuses a device that does not match the CONFIGURED one, so
    # this names one first; with nothing configured there is nothing to
    # refuse against and the call proceeds to fail later for another reason.
    monkeypatch.setenv("SCANNER3D_SERIAL", REAL_SERIAL)
    kernel = FakeKernel()
    outcome = backend(kernel, sysfs).run("SOMEONEELSES", output_dir=tmp_path / "out")
    assert outcome.status is CaptureStatus.REFUSED
    assert kernel.opened == []
    assert kernel.calls == []


# --- the JSON report ----------------------------------------------------------


def test_the_outcome_serialises_to_stable_json_for_a_complete_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    outcome = run(kernel, sysfs, tmp_path)
    blob = json.dumps(outcome.as_dict(), sort_keys=True)
    payload = json.loads(blob)
    assert payload["status"] == "complete"
    assert payload["succeeded"] is True
    assert payload["plan"]["fourcc"] == "Y16 "
    assert payload["plan"]["width"] == FRAME_WIDTH
    assert payload["plan"]["height"] == FRAME_HEIGHT
    assert payload["plan"]["requested_buffers"] == REQUESTED_BUFFERS
    assert payload["plan"]["capture_deadline_s"] == CAPTURE_DEADLINE_S
    assert payload["plan"]["max_dequeues"] == 1
    assert payload["plan"]["retries"] == 0
    assert payload["cleanup"]["ok"] is True


def test_the_outcome_serialises_to_stable_json_for_a_timeout(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(poll_revents=0)
    payload = json.loads(json.dumps(run(kernel, sysfs, tmp_path).as_dict(), sort_keys=True))
    assert payload["status"] == "poll_timeout"
    assert payload["succeeded"] is False
    assert payload["cleanup"]["ok"] is True
    assert payload["dequeued_buffer"] is None


def test_every_documented_outcome_status_has_a_distinct_json_value() -> None:
    values = [status.value for status in CaptureStatus]
    assert len(values) == len(set(values))
    required = {
        "not_attempted",
        "complete",
        "poll_timeout",
        "poll_error",
        "open_error",
        "capability_mismatch",
        "format_mismatch",
        "buffer_count_mismatch",
        "mapping_error",
        "queue_error",
        "streamon_error",
        "dequeue_error",
        "short_frame",
        "oversize_frame",
        "interrupted",
    }
    assert required <= set(values)
