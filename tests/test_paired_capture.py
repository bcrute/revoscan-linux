"""Fake-backend coverage for the offline paired lifecycle."""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from scanner3d.errors import V4l2CaptureNotAllowed
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
    not_required,
)
from scanner3d.ownership import SignalReceived
from scanner3d.paired_capture import (
    AUTHORIZED_SERIAL,
    DEPTH_SPEC,
    EXECUTE_OPT_IN,
    CleanupReport,
    PairedBackendRefusal,
    PairedCapture,
    PairedOutcome,
    PairedStatus,
    PairedV4l2Backend,
)
from scanner3d.rgb_frame import RGB_PROFILE
from scanner3d.v4l2_capture import (
    DEPTH_PROFILE,
    EXPECTED_FRAME_BYTES,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
)
from tests.fake_v4l2 import FakeKernel, SteppedClock
from tests.test_jpeg_inspect import baseline_jpeg


class _FakeIlluminationBackend:
    def run(self, _serial: str) -> IlluminationShutdownOutcome:
        return IlluminationShutdownOutcome(
            True,
            True,
            ShutdownStatus.COMMAND_ACCEPTED,
            "synthetic fixed off accepted",
            transfer=ShutdownTransferRecord(60, True, "synthetic", 0.0),
        )


def _fake_illumination_factory(_sysfs, _dev):
    return _FakeIlluminationBackend()


def test_concrete_backend_reuses_fake_v4l2_engine_for_both_roles(healthy, tmp_path: Path) -> None:
    depth_kernel = FakeKernel(
        profile=DEPTH_PROFILE,
        bytesused=EXPECTED_FRAME_BYTES,
        frame_byte=0x12,
    )
    rgb_kernel = FakeKernel(
        profile=RGB_PROFILE,
        frame_payload=baseline_jpeg(1280, 800),
    )
    kernels: dict[int, FakeKernel] = {}

    def opener(path: str) -> int:
        kernel = depth_kernel if path.endswith("video0") else rgb_kernel
        fd = kernel.opener(path)
        kernels[fd] = kernel
        return fd

    def close(fd: int) -> None:
        kernels[fd].closer(fd)

    backend = PairedV4l2Backend(
        output_dir=tmp_path / "paired",
        opener=opener,
        ioctl_fn=lambda fd, request, packed, mutate: kernels[fd].ioctl(fd, request, packed, mutate),
        mmap_fn=lambda fd, length, offset: kernels[fd].mmap(fd, length, offset),
        poll_fn=lambda fd, timeout: kernels[fd].poll(fd, timeout),
        closer=close,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
        illumination_backend_factory=_fake_illumination_factory,
    )
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )

    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.depth_result is not None and outcome.depth_result.succeeded
    assert outcome.rgb_result is not None and outcome.rgb_result.succeeded
    assert depth_kernel.count(VIDIOC_DQBUF) == 1
    assert rgb_kernel.count(VIDIOC_DQBUF) == 1
    expected = [
        VIDIOC_QUERYCAP,
        VIDIOC_G_FMT,
        VIDIOC_S_FMT,
        VIDIOC_REQBUFS,
        *([VIDIOC_QUERYBUF] * 4),
        *([VIDIOC_QBUF] * 4),
        VIDIOC_STREAMON,
        VIDIOC_DQBUF,
        VIDIOC_STREAMOFF,
        VIDIOC_REQBUFS,
    ]
    assert depth_kernel.requests == expected
    assert rgb_kernel.requests == expected
    assert depth_kernel.closed and rgb_kernel.closed


def _make_concrete_pair(healthy, tmp_path: Path, *, depth=None, rgb=None, clock=None):
    depth_kernel = FakeKernel(
        profile=DEPTH_PROFILE,
        bytesused=EXPECTED_FRAME_BYTES,
        frame_byte=0x12,
        **(depth or {}),
    )
    rgb_kernel = FakeKernel(
        profile=RGB_PROFILE,
        frame_payload=baseline_jpeg(1280, 800),
        **(rgb or {}),
    )
    kernels: dict[int, FakeKernel] = {}

    def opener(path: str) -> int:
        kernel = depth_kernel if path.endswith("video0") else rgb_kernel
        fd = kernel.opener(path)
        kernels[fd] = kernel
        return fd

    def close(fd: int) -> None:
        kernels[fd].closer(fd)

    backend = PairedV4l2Backend(
        output_dir=tmp_path / "paired",
        opener=opener,
        ioctl_fn=lambda fd, request, packed, mutate: kernels[fd].ioctl(fd, request, packed, mutate),
        mmap_fn=lambda fd, length, offset: kernels[fd].mmap(fd, length, offset),
        poll_fn=lambda fd, timeout: kernels[fd].poll(fd, timeout),
        closer=close,
        clock=clock or __import__("time").monotonic,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
        illumination_backend_factory=_fake_illumination_factory,
    )
    return backend, depth_kernel, rgb_kernel


def test_concrete_pair_reports_depth_only_and_rgb_only_delivery(healthy, tmp_path: Path) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(
        healthy, tmp_path, depth={"poll_revents": 0}
    )
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.TIMEOUT
    assert outcome.depth_result is not None and outcome.depth_result.poll is not None
    assert outcome.depth_frame is None
    assert outcome.rgb_frame is not None
    assert depth_kernel.count(VIDIOC_DQBUF) == 0
    assert rgb_kernel.count(VIDIOC_DQBUF) == 1

    backend, depth_kernel, rgb_kernel = _make_concrete_pair(
        healthy, tmp_path / "rgb-only", rgb={"poll_revents": 0}
    )
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.TIMEOUT
    assert outcome.depth_frame is not None
    assert outcome.rgb_frame is None
    assert depth_kernel.count(VIDIOC_DQBUF) == 1
    assert rgb_kernel.count(VIDIOC_DQBUF) == 0


def test_concrete_pair_consumes_depth_deadline_while_rgb_is_prepared(
    healthy, tmp_path: Path
) -> None:
    clock = SteppedClock(tick=0.001)
    backend, _depth, _rgb = _make_concrete_pair(healthy, tmp_path, clock=clock)
    outcome = PairedCapture(backend, sleep=lambda seconds: clock.advance(seconds)).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.depth_result is not None and outcome.depth_result.poll is not None
    assert outcome.rgb_result is not None and outcome.rgb_result.poll is not None
    assert outcome.depth_result.poll.timeout_s < outcome.rgb_result.poll.timeout_s


def test_concrete_pair_rejects_identity_drift_after_fd_ownership(healthy, tmp_path: Path) -> None:
    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path)

    def changed_fstat(fd: int):
        return type("Stat", (), {"st_rdev": 0, "st_ino": 999999})()

    backend.fstat_fn = changed_fstat
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.REFUSED
    assert depth_kernel.closed
    assert outcome.cleanup.errors == ()


@pytest.mark.parametrize(
    "identity_exception",
    [
        SignalReceived(signal.SIGTERM, "synthetic fstat signal"),
        RuntimeError("synthetic fstat exception"),
    ],
)
def test_concrete_pair_identity_exception_keeps_owned_fd_cleanup(
    healthy, tmp_path: Path, identity_exception: BaseException
) -> None:
    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path)

    def failing_fstat(_fd: int):
        raise identity_exception

    backend.fstat_fn = failing_fstat
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    expected = (
        PairedStatus.CANCELLED
        if isinstance(identity_exception, SignalReceived)
        else PairedStatus.FAILED
    )
    assert outcome.status is expected
    assert depth_kernel.closed
    assert any("close" in action for action in outcome.cleanup.actions)


def test_concrete_pair_continues_after_multiple_unmap_failures(healthy, tmp_path: Path) -> None:
    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path)

    def failing_mmap(fd: int, length: int, offset: int):
        mapping = depth_kernel.mmap(fd, length, offset)
        if fd == depth_kernel.fd:

            def fail_close() -> None:
                raise OSError("synthetic unmap failure")

            mapping.close = fail_close
        return mapping

    backend.mmap_fn = failing_mmap
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.exit_code == 9
    assert len([error for error in outcome.cleanup.errors if "unmap" in error]) >= 2
    assert depth_kernel.closed


def test_concrete_pair_exhausts_mixed_cleanup_failures_and_other_role(
    healthy, tmp_path: Path
) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    original_mmap = backend.mmap_fn
    original_ioctl = backend.ioctl_fn
    original_close = backend.closer
    release_calls: dict[int, int] = {}

    def mixed_mmap(fd: int, length: int, offset: int):
        mapping = original_mmap(fd, length, offset)
        if fd == depth_kernel.fd:
            index = len(depth_kernel.mappings)
            if index == 1:
                mapping.close = lambda: (_ for _ in ()).throw(RuntimeError("unmap runtime"))
            elif index == 2:
                mapping.close = lambda: (_ for _ in ()).throw(
                    SignalReceived(signal.SIGTERM, "synthetic unmap signal")
                )
        return mapping

    def failing_release(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if request == VIDIOC_REQBUFS and fd == depth_kernel.fd:
            release_calls[fd] = release_calls.get(fd, 0) + 1
            if release_calls[fd] == 2:
                raise RuntimeError("queue release runtime")
        return original_ioctl(fd, request, packed, mutate)

    def failing_close(fd: int) -> None:
        if fd == depth_kernel.fd:
            raise RuntimeError("close runtime")
        original_close(fd)

    backend.mmap_fn = mixed_mmap
    backend.ioctl_fn = failing_release
    backend.closer = failing_close
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.exit_code == 9
    assert any("unmap" in error for error in outcome.cleanup.errors)
    assert any("release_buffers" in error for error in outcome.cleanup.errors)
    assert any("close" in error for error in outcome.cleanup.errors)
    assert rgb_kernel.closed


def test_concrete_pair_uses_abi_checked_preflight_before_any_open(
    healthy, tmp_path, monkeypatch
) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    import scanner3d.v4l2_capture as v4l2

    def refuse_abi() -> None:
        raise V4l2CaptureNotAllowed("synthetic ABI mismatch")

    monkeypatch.setattr(v4l2, "ensure_layout_intact", refuse_abi)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.REFUSED
    assert depth_kernel.opened == []
    assert rgb_kernel.opened == []


def test_concrete_pair_setup_failure_is_not_retried(healthy, tmp_path) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(
        healthy, tmp_path, depth={"errors": {VIDIOC_QUERYCAP: 5}}
    )
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    assert depth_kernel.count(VIDIOC_QUERYCAP) == 1
    assert rgb_kernel.opened == []
    assert depth_kernel.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
@pytest.mark.parametrize(
    "ioctl_request",
    [VIDIOC_REQBUFS, VIDIOC_QUERYBUF, VIDIOC_QBUF, VIDIOC_STREAMON],
)
def test_concrete_pair_reports_queue_and_stream_ioctl_failure_for_each_role(
    healthy, tmp_path, role, ioctl_request
) -> None:
    kwargs = {role: {"errors": {ioctl_request: 5}}}
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path, **kwargs)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    failed = depth_kernel if role == "depth" else rgb_kernel
    assert failed.count(ioctl_request) == 1
    assert failed.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_reports_poll_exception_for_each_role(healthy, tmp_path, role) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_poll = backend.poll_fn

    def failing_poll(fd: int, timeout_s: float) -> int:
        if fd == target.fd:
            raise OSError(4, "synthetic poll exception")
        return original_poll(fd, timeout_s)

    backend.poll_fn = failing_poll
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    assert target.closed


def test_concrete_pair_rejects_fixed_spec_and_deadline_changes(healthy, tmp_path) -> None:
    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path)
    node = backend.resolve_node(AUTHORIZED_SERIAL, "depth")
    handle = backend.open(node, "depth")
    with pytest.raises(PairedBackendRefusal, match="fixed specification"):
        backend.configure(handle, replace(DEPTH_SPEC, width=DEPTH_SPEC.width + 1))
    backend.close(handle)
    assert depth_kernel.closed

    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path / "deadline")
    node = backend.resolve_node(AUTHORIZED_SERIAL, "depth")
    handle = backend.open(node, "depth")
    backend.configure(handle, DEPTH_SPEC)
    backend.map_queue(handle, DEPTH_SPEC)
    backend.stream_on(handle)
    with pytest.raises(PairedBackendRefusal, match="one dequeue"):
        backend.wait_frame(handle, DEPTH_SPEC.deadline_s + 1.0, 1)
    backend.stream_off(handle)
    backend.unmap(handle)
    backend.free_queue(handle)
    backend.close(handle)
    assert depth_kernel.closed


def test_concrete_pair_artifact_failure_is_not_success(healthy, tmp_path) -> None:
    output = tmp_path / "output-file"
    output.write_bytes(b"not a directory")
    backend, _depth, _rgb = _make_concrete_pair(healthy, output)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    assert outcome.exit_code == 5
    assert outcome.depth_result is not None and not outcome.depth_result.artifacts.ok


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_rejects_profile_and_driver_refusal_before_open(
    healthy, tmp_path, role
) -> None:
    interface = healthy.devices / "1-4.1" / f"1-4.1:1.{0 if role == 'depth' else 2}"
    (interface / "driver").unlink()
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.REFUSED
    assert depth_kernel.opened == []
    assert rgb_kernel.opened == []


def test_concrete_pair_rejects_rdev_inode_alias_before_open(healthy, tmp_path) -> None:
    (healthy.dev / "video2").unlink()
    os.link(healthy.dev / "video0", healthy.dev / "video2")
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.REFUSED
    assert depth_kernel.opened == []
    assert rgb_kernel.opened == []


@pytest.mark.parametrize("role", ["depth", "rgb"])
@pytest.mark.parametrize("ioctl_request", [VIDIOC_QUERYCAP, VIDIOC_G_FMT, VIDIOC_S_FMT])
def test_concrete_pair_reports_setup_ioctl_failure_once(
    healthy, tmp_path, role, ioctl_request
) -> None:
    kwargs = {role: {"errors": {ioctl_request: 5}}}
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path, **kwargs)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    failed = depth_kernel if role == "depth" else rgb_kernel
    assert failed.count(ioctl_request) == 1
    assert failed.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
@pytest.mark.parametrize(
    "ioctl_request",
    [
        VIDIOC_QUERYCAP,
        VIDIOC_G_FMT,
        VIDIOC_S_FMT,
        VIDIOC_REQBUFS,
        VIDIOC_QUERYBUF,
        VIDIOC_QBUF,
        VIDIOC_STREAMON,
    ],
)
def test_concrete_pair_signal_at_each_ioctl_boundary_is_cancelled_once(
    healthy, tmp_path, role, ioctl_request
) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_ioctl = backend.ioctl_fn
    attempts = 0

    def interrupted_ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal attempts
        if fd == target.fd and request == ioctl_request:
            attempts += 1
            if attempts == 1:
                raise SignalReceived(signal.SIGTERM, f"synthetic {role} {request} signal")
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = interrupted_ioctl
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED, outcome.detail
    assert outcome.exit_code == 13
    assert target.closed
    # REQBUFS is also the mandated cleanup ioctl (count=0); that second
    # invocation is teardown, not a retry of the interrupted setup call.
    assert attempts == (2 if ioctl_request == VIDIOC_REQBUFS else 1)
    assert target.count(ioctl_request) == (1 if ioctl_request == VIDIOC_REQBUFS else 0)
    if role == "rgb":
        assert depth_kernel.closed
    assert any(f"{role} close" == action for action in outcome.cleanup.actions)


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_signal_at_poll_boundary_is_cancelled_once(healthy, tmp_path, role) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_poll = backend.poll_fn
    attempts = 0

    def interrupted_poll(fd: int, timeout_s: float) -> int:
        nonlocal attempts
        if fd == target.fd:
            attempts += 1
            if attempts == 1:
                raise SignalReceived(signal.SIGTERM, f"synthetic {role} poll signal")
        return original_poll(fd, timeout_s)

    backend.poll_fn = interrupted_poll
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED
    assert outcome.exit_code == 13
    assert target.closed
    assert attempts == 1
    assert any(f"{role} close" == action for action in outcome.cleanup.actions)


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_signal_at_dequeue_boundary_is_cancelled_once(
    healthy, tmp_path, role
) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_ioctl = backend.ioctl_fn
    attempts = 0

    def interrupted_dequeue(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal attempts
        if fd == target.fd and request == VIDIOC_DQBUF:
            attempts += 1
            if attempts == 1:
                raise SignalReceived(signal.SIGTERM, f"synthetic {role} dequeue signal")
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = interrupted_dequeue
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED
    assert outcome.exit_code == 13
    assert target.closed
    assert attempts == 1
    assert any(f"{role} close" == action for action in outcome.cleanup.actions)


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_reports_mmap_failure_once(healthy, tmp_path, role) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_mmap = backend.mmap_fn

    def failing_mmap(fd: int, length: int, offset: int):
        if fd == target.fd:
            raise OSError(12, "synthetic mmap failure")
        return original_mmap(fd, length, offset)

    backend.mmap_fn = failing_mmap
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    assert target.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_signal_during_mmap_remains_cancelled(healthy, tmp_path, role) -> None:
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path)
    target = depth_kernel if role == "depth" else rgb_kernel
    original_mmap = backend.mmap_fn

    def interrupted_mmap(fd: int, length: int, offset: int):
        if fd == target.fd:
            raise SignalReceived(signal.SIGTERM, f"synthetic {role} mmap signal")
        return original_mmap(fd, length, offset)

    backend.mmap_fn = interrupted_mmap
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED
    assert outcome.exit_code == 13
    assert target.closed


def test_concrete_pair_signal_during_interval_remains_cancelled(healthy, tmp_path) -> None:
    backend, depth_kernel, _rgb = _make_concrete_pair(healthy, tmp_path)

    def interrupted_sleep(_seconds: float) -> None:
        raise SignalReceived(signal.SIGTERM, "synthetic interval signal")

    outcome = PairedCapture(backend, sleep=interrupted_sleep).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED
    assert outcome.exit_code == 13
    assert depth_kernel.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_reports_poll_failure_for_each_role(healthy, tmp_path, role) -> None:
    kwargs = {role: {"poll_revents": 8}}
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path, **kwargs)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    failed = depth_kernel if role == "depth" else rgb_kernel
    assert len(failed.polls) == 1
    assert failed.closed


@pytest.mark.parametrize("role", ["depth", "rgb"])
def test_concrete_pair_reports_dq_failure_for_each_role(healthy, tmp_path, role) -> None:
    kwargs = {role: {"errors": {VIDIOC_DQBUF: 5}}}
    backend, depth_kernel, rgb_kernel = _make_concrete_pair(healthy, tmp_path, **kwargs)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    failed = depth_kernel if role == "depth" else rgb_kernel
    assert failed.count(VIDIOC_DQBUF) == 1
    assert failed.closed


def test_concrete_pair_reports_artifact_failure_for_both_roles(healthy, tmp_path) -> None:
    output = tmp_path / "output-file"
    output.write_bytes(b"not a directory")
    backend, _depth, _rgb = _make_concrete_pair(healthy, output)
    outcome = PairedCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.FAILED
    assert outcome.depth_result is not None and not outcome.depth_result.artifacts.ok
    assert outcome.rgb_result is not None and not outcome.rgb_result.artifacts.ok


class _EntrySignalGuard:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __enter__(self):
        self.events.append("enter")
        raise SignalReceived(signal.SIGINT, "synthetic guard entry")

    def __exit__(self, _exc_type, _exc, _tb):
        self.events.append("exit")
        return False


def test_signal_guard_entry_boundary_restores_injected_guard() -> None:
    guard = _EntrySignalGuard()
    fake = FakeBackend()
    outcome = PairedCapture(fake, signal_guard=lambda: guard).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.CANCELLED
    assert guard.events == ["enter", "exit"]
    assert fake.events == []


@contextmanager
def _deferred_signal_shield():
    yield [signal.SIGTERM]


def test_deferred_cleanup_signals_are_retained_in_paired_evidence() -> None:
    outcome = PairedCapture(FakeBackend(), termination_shield=_deferred_signal_shield).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.cleanup.deferred_signals == (signal.SIGTERM,)
    assert outcome.cleanup.guidance is not None


@dataclass
class FakeBackend:
    fail: str | None = None
    same_node: bool = False
    depth_frame: bytes | None = b"frame"
    rgb_frame: bytes | None = b"jpeg"
    interrupt: str | None = None
    signal_at: str | None = None
    events: list[str] = field(default_factory=list)
    journal: list[str] = field(default_factory=list)

    def _check(self, name: str) -> None:
        self.events.append(name)
        self.journal.append(name)
        if self.signal_at == name:
            raise SignalReceived(signal.SIGTERM, f"synthetic {name}")
        if self.interrupt == name:
            raise KeyboardInterrupt
        if self.fail == name:
            raise RuntimeError(name)

    def resolve_node(self, serial: str, role: str) -> str:
        self._check(f"resolve:{role}")
        return "/dev/video0" if role == "depth" or self.same_node else "/dev/video2"

    def open(self, node: str, role: str) -> object:
        self._check(f"open:{role}")
        return role

    def configure(self, handle: object, spec) -> None:
        self._check(f"configure:{handle}")

    def map_queue(self, handle: object, spec) -> None:
        self._check(f"map:{handle}")

    def stream_on(self, handle: object) -> None:
        self._check(f"on:{handle}")

    def wait_frame(self, handle: object, deadline_s: float, max_frames: int) -> bytes | None:
        self._check(f"wait:{handle}")
        return self.depth_frame if handle == "depth" else self.rgb_frame

    def stream_off(self, handle: object) -> None:
        self._check(f"off:{handle}")

    def unmap(self, handle: object) -> None:
        self._check(f"unmap:{handle}")

    def free_queue(self, handle: object) -> None:
        self._check(f"free:{handle}")

    def close(self, handle: object) -> None:
        self._check(f"close:{handle}")

    def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
        # The fake owns the same finalizer contract as the concrete backend,
        # without adding an event to the legacy V4L2 ordering assertions.
        self.journal.append("illumination-off:depth")
        return _FakeIlluminationBackend().run(AUTHORIZED_SERIAL)


def test_dry_run_never_calls_backend() -> None:
    fake = FakeBackend()
    outcome = PairedCapture(fake).dry_run()
    assert outcome.status is PairedStatus.NOT_ATTEMPTED
    assert fake.events == []
    assert outcome.plan.as_dict()["start_interval_s"] == 1.0


def test_normal_order_and_reverse_cleanup() -> None:
    fake = FakeBackend()
    outcome = PairedCapture(fake, sleep=lambda seconds: fake.events.append(f"sleep:{seconds}")).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.COMPLETE
    assert fake.events[:10] == [
        "resolve:depth",
        "resolve:rgb",
        "open:depth",
        "configure:depth",
        "map:depth",
        "on:depth",
        "sleep:1.0",
        "open:rgb",
        "configure:rgb",
        "map:rgb",
    ]
    assert fake.events.index("on:depth") < fake.events.index("on:rgb")
    assert fake.events[-8:] == [
        "off:rgb",
        "unmap:rgb",
        "free:rgb",
        "close:rgb",
        "off:depth",
        "unmap:depth",
        "free:depth",
        "close:depth",
    ]


def test_finalizer_signal_is_retained_without_rewriting_primary_status() -> None:
    class SignalledFinalizer(FakeBackend):
        def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
            raise SignalReceived(signal.SIGTERM, "synthetic finalizer signal")

    outcome = PairedCapture(SignalledFinalizer()).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.detail == "paired run completed"
    assert not outcome.succeeded
    assert outcome.illumination_shutdown.status is ShutdownStatus.INTERRUPTED
    assert outcome.illumination_shutdown.deferred_signals == (signal.SIGTERM,)
    assert outcome.illumination_shutdown.guidance is not None


def test_paired_journal_has_one_off_after_both_role_closes() -> None:
    fake = FakeBackend()
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.illumination_shutdown.succeeded
    assert fake.journal.count("illumination-off:depth") == 1
    assert fake.journal.index("close:rgb") < fake.journal.index("illumination-off:depth")
    assert fake.journal.index("close:depth") < fake.journal.index("illumination-off:depth")


@pytest.mark.parametrize("mode", ["missing", "not_required", "failed", "raising"])
def test_paired_finalizer_matrix_blocks_success_without_rewriting_primary(mode: str) -> None:
    class FinalizerVariant(FakeBackend):
        def __getattribute__(self, name: str) -> Any:
            if mode == "missing" and name == "shutdown_depth":
                return None
            return super().__getattribute__(name)

        def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
            self.journal.append("illumination-off:depth")
            if mode == "not_required":
                return not_required("synthetic contradiction")
            if mode == "failed":
                return IlluminationShutdownOutcome(
                    True, True, ShutdownStatus.IOCTL_ERROR, "synthetic finalizer failure"
                )
            if mode == "raising":
                raise RuntimeError("synthetic finalizer exception")
            return super().shutdown_depth(_handle)

    fake = FinalizerVariant()
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.detail == "paired run completed"
    assert not outcome.succeeded
    assert outcome.illumination_shutdown.status in {
        ShutdownStatus.REFUSED,
        ShutdownStatus.NOT_REQUIRED,
        ShutdownStatus.IOCTL_ERROR,
        ShutdownStatus.UNEXPECTED_EXCEPTION,
    }
    assert fake.journal.count("illumination-off:depth") == (0 if mode == "missing" else 1)


def test_paired_outer_shield_signal_retains_one_finalizer_and_primary_truth() -> None:
    active = False
    received: list[int] = []

    class ActiveFinalizer(FakeBackend):
        def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
            assert active
            received.append(signal.SIGTERM)
            self.journal.append("illumination-off:depth")
            return _FakeIlluminationBackend().run(AUTHORIZED_SERIAL)

    fake = ActiveFinalizer()

    @contextmanager
    def shield():
        nonlocal active
        active = True
        try:
            yield received
        finally:
            active = False

    outcome = PairedCapture(fake, termination_shield=shield).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.COMPLETE
    assert outcome.illumination_shutdown.succeeded
    assert fake.journal.count("illumination-off:depth") == 1
    assert outcome.cleanup.deferred_signals == (signal.SIGTERM,)


def test_paired_shield_fallback_retains_role_failure_and_outer_evidence() -> None:
    fake = FakeBackend(fail="close:depth")
    calls = 0

    def shield():
        @contextmanager
        def context():
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise SignalReceived(signal.SIGTERM, f"synthetic shield entry {calls}")
            yield []

        return context()

    outcome = PairedCapture(fake, termination_shield=shield).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.exit_code == 9
    assert outcome.cleanup.deferred_signals == (signal.SIGTERM, signal.SIGTERM)
    assert any("depth close" in error for error in outcome.cleanup.errors)
    assert "rgb close" in outcome.cleanup.actions
    assert "depth unmap" in outcome.cleanup.actions


def test_paired_shield_entry_signal_retries_cleanup_once() -> None:
    fake = FakeBackend()
    calls = 0

    def shield():
        @contextmanager
        def context():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SignalReceived(signal.SIGTERM, "synthetic shield entry")
            yield []

        return context()

    outcome = PairedCapture(fake, termination_shield=shield).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert outcome.status is PairedStatus.COMPLETE
    assert calls == 2
    assert fake.events[-8:] == [
        "off:rgb",
        "unmap:rgb",
        "free:rgb",
        "close:rgb",
        "off:depth",
        "unmap:depth",
        "free:depth",
        "close:depth",
    ]
    assert outcome.cleanup.deferred_signals == (signal.SIGTERM,)


def test_depth_failure_means_rgb_never_starts_and_cleanup_is_attempted() -> None:
    fake = FakeBackend(fail="on:depth")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert "open:rgb" not in fake.events
    assert fake.events[-4:] == ["unmap:depth", "free:depth", "close:depth"] or fake.events[-3:] == [
        "unmap:depth",
        "free:depth",
        "close:depth",
    ]


def test_map_exception_is_treated_as_an_owed_queue() -> None:
    fake = FakeBackend(fail="map:depth")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert fake.events[-3:] == ["unmap:depth", "free:depth", "close:depth"]
    assert "open:rgb" not in fake.events


def test_map_interrupt_on_rgb_owes_rgb_and_depth_cleanup() -> None:
    fake = FakeBackend(interrupt="map:rgb")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.CANCELLED
    assert fake.events[-7:] == [
        "unmap:rgb",
        "free:rgb",
        "close:rgb",
        "off:depth",
        "unmap:depth",
        "free:depth",
        "close:depth",
    ]


def test_streamon_exception_still_attempts_streamoff() -> None:
    fake = FakeBackend(fail="on:rgb")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert "off:rgb" in fake.events
    assert "wait:depth" not in fake.events


def test_streamon_interrupt_on_depth_still_attempts_streamoff() -> None:
    fake = FakeBackend(interrupt="on:depth")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.CANCELLED
    assert "off:depth" in fake.events
    assert "open:rgb" not in fake.events


def test_rgb_setup_failure_does_not_retry() -> None:
    fake = FakeBackend(fail="configure:rgb")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert fake.events.count("open:rgb") == 1
    assert "on:rgb" not in fake.events
    assert fake.journal.count("illumination-off:depth") == 1
    assert fake.journal.index("close:depth") < fake.journal.index("illumination-off:depth")


def test_timeout_is_per_role_and_still_tears_down_both() -> None:
    fake = FakeBackend(rgb_frame=None)
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.TIMEOUT
    assert outcome.rgb_frame is None
    assert "RGB deadline" in outcome.detail
    assert fake.events.count("wait:rgb") == 1


def test_cancellation_cleans_up_without_retry() -> None:
    fake = FakeBackend()
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN, cancel=lambda: True)
    assert outcome.status is PairedStatus.CANCELLED
    assert fake.events.count("on:depth") == 1
    assert fake.events.count("on:rgb") == 1
    assert "wait:depth" not in fake.events


def test_keyboard_interrupt_is_cancelled_after_cleanup() -> None:
    fake = FakeBackend(interrupt="wait:depth")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.CANCELLED
    assert fake.events[-8:] == [
        "off:rgb",
        "unmap:rgb",
        "free:rgb",
        "close:rgb",
        "off:depth",
        "unmap:depth",
        "free:depth",
        "close:depth",
    ]


@pytest.mark.parametrize(
    "boundary",
    [
        "resolve:depth",
        "resolve:rgb",
        "open:depth",
        "configure:depth",
        "map:depth",
        "on:depth",
        "open:rgb",
        "configure:rgb",
        "map:rgb",
        "on:rgb",
        "wait:depth",
        "wait:rgb",
    ],
)
def test_signal_received_at_every_role_boundary_is_cancelled_and_cleaned(boundary) -> None:
    fake = FakeBackend(signal_at=boundary)
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.CANCELLED


def test_signal_received_during_start_interval_is_cancelled_and_cleaned() -> None:
    fake = FakeBackend()

    def signal_sleep(_seconds: float) -> None:
        raise SignalReceived(signal.SIGTERM, "synthetic interval")

    outcome = PairedCapture(fake, sleep=signal_sleep).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.CANCELLED
    assert "close:depth" in fake.events


def test_cleanup_failure_is_not_hidden() -> None:
    fake = FakeBackend(fail="off:rgb")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert not outcome.cleanup.ok
    assert any("off:rgb" in error for error in outcome.cleanup.errors)
    assert outcome.exit_code == 9
    assert fake.journal.count("illumination-off:depth") == 1
    assert fake.journal.index("close:depth") < fake.journal.index("illumination-off:depth")


def test_cleanup_continues_after_one_teardown_failure() -> None:
    fake = FakeBackend(fail="unmap:rgb")
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.status is PairedStatus.FAILED
    assert fake.events.index("free:rgb") > fake.events.index("unmap:rgb")
    assert fake.events.index("close:rgb") > fake.events.index("free:rgb")
    assert "close:depth" in fake.events


def test_same_node_and_wrong_opt_in_are_refused() -> None:
    same = FakeBackend(same_node=True)
    assert (
        PairedCapture(same).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN).status
        is PairedStatus.REFUSED
    )
    assert "open:depth" not in same.events
    wrong = FakeBackend()
    assert (
        PairedCapture(wrong).run(AUTHORIZED_SERIAL, opt_in="--start").status is PairedStatus.REFUSED
    )
    assert wrong.events == []


def test_stable_exit_codes_distinguish_dry_run_refusal_and_timeout() -> None:
    assert PairedCapture().dry_run().exit_code == 0
    assert PairedCapture().run("wrong", opt_in=EXECUTE_OPT_IN).exit_code == 10
    timeout = PairedCapture(FakeBackend(depth_frame=None)).run(
        AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN
    )
    assert timeout.status is PairedStatus.TIMEOUT
    assert timeout.exit_code == 4


def test_every_paired_status_has_repository_vocabulary_code() -> None:
    expected = {
        PairedStatus.NOT_ATTEMPTED: 0,
        PairedStatus.COMPLETE: 0,
        PairedStatus.REFUSED: 10,
        PairedStatus.FAILED: 5,
        PairedStatus.CANCELLED: 13,
        PairedStatus.TIMEOUT: 4,
    }
    for status, code in expected.items():
        assert PairedOutcome(status, "test").exit_code == code


def test_paired_success_requires_both_present_role_outcomes() -> None:
    assert not PairedOutcome(PairedStatus.COMPLETE, "test").succeeded
    fake = FakeBackend()
    outcome = PairedCapture(fake).run(AUTHORIZED_SERIAL, opt_in=EXECUTE_OPT_IN)
    assert outcome.succeeded is False


def test_cleanup_failure_overrides_every_primary_status() -> None:
    failed_cleanup = CleanupReport(errors=("teardown failed",))
    for status in PairedStatus:
        assert PairedOutcome(status, "test", cleanup=failed_cleanup).exit_code == 9
