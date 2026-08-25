"""Offline fake-kernel coverage for the fixed RGB observation series."""

from __future__ import annotations

import errno
import json
import os
import signal
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from scanner3d import cli, paired_capture
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
)
from scanner3d.ownership import SignalReceived
from scanner3d.paired_capture import (
    AUTHORIZED_SERIAL,
    RGB_ROLE,
    RGB_SERIES_COUNT,
    RGB_SERIES_EXECUTE_OPT_IN,
    RGB_SERIES_MAX_TOTAL_BYTES,
    RGB_SPEC,
    PairedBackendRefusal,
    PairedV4l2Backend,
    RgbSeriesCapture,
    RgbSeriesResult,
)
from scanner3d.rgb_frame import RGB_MAX_BUFFER_BYTES, RGB_PROFILE
from scanner3d.v4l2_capture import (
    DEPTH_PROFILE,
    EXPECTED_FRAME_BYTES,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
)
from scanner3d.v4l2_controls import (
    CONTROL_FORMAT,
    RGB_SERIES_CONTROL_WHITELIST,
    VIDIOC_G_CTRL,
)
from tests.fake_v4l2 import FakeKernel
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


class _FailedIlluminationBackend:
    def run(self, _serial: str) -> IlluminationShutdownOutcome:
        return IlluminationShutdownOutcome(
            True, True, ShutdownStatus.IOCTL_ERROR, "synthetic series off failure"
        )


def _fake_illumination_factory(_sysfs, _dev):
    return _FakeIlluminationBackend()


def _backend(
    healthy,
    tmp_path: Path,
    *,
    rgb_kwargs: dict | None = None,
    illumination_factory=None,
    illumination_journal: list[str] | None = None,
):
    depth_kernel = FakeKernel(
        profile=DEPTH_PROFILE,
        bytesused=EXPECTED_FRAME_BYTES,
        frame_byte=0x12,
    )
    rgb_options = {"frame_payload": baseline_jpeg(1280, 800)}
    rgb_options.update(rgb_kwargs or {})
    rgb_kernel = FakeKernel(profile=RGB_PROFILE, **cast(Any, rgb_options))
    kernels: dict[int, FakeKernel] = {}

    def opener(path: str) -> int:
        kernel = depth_kernel if path.endswith("video0") else rgb_kernel
        fd = kernel.opener(path)
        kernels[fd] = kernel
        return fd

    def close(fd: int) -> None:
        kernels[fd].closer(fd)

    if illumination_factory is None:
        illumination_factory = _fake_illumination_factory
    if illumination_journal is not None:
        base_factory = illumination_factory

        def counted_factory(sysfs, dev):
            illumination_journal.append("illumination-off:depth")
            return base_factory(sysfs, dev)

        illumination_factory = counted_factory

    backend = PairedV4l2Backend(
        output_dir=tmp_path / "paired",
        opener=opener,
        ioctl_fn=lambda fd, request, packed, mutate: kernels[fd].ioctl(fd, request, packed, mutate),
        mmap_fn=lambda fd, length, offset: kernels[fd].mmap(fd, length, offset),
        poll_fn=lambda fd, timeout: kernels[fd].poll(fd, timeout),
        closer=close,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
        illumination_backend_factory=illumination_factory,
    )
    return backend, depth_kernel, rgb_kernel


def test_series_reuses_pair_order_and_requeues_only_between_observations(healthy, tmp_path):
    backend, depth_kernel, rgb_kernel = _backend(healthy, tmp_path)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )

    assert outcome.status.value == "complete"
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == RGB_SERIES_COUNT
    assert outcome.rgb_series.strict_success
    assert len(outcome.rgb_series.polls) == RGB_SERIES_COUNT
    assert rgb_kernel.count(VIDIOC_DQBUF) == RGB_SERIES_COUNT
    assert rgb_kernel.count(VIDIOC_QBUF) == 4 + RGB_SERIES_COUNT - 1
    assert rgb_kernel.count(VIDIOC_STREAMON) == 1
    assert len(list((tmp_path / "paired" / "rgb" / "rgb-series").glob("*.jpg"))) == 3
    assert outcome.rgb_series.poll_count == RGB_SERIES_COUNT
    assert outcome.rgb_series.dequeue_count == RGB_SERIES_COUNT
    assert outcome.rgb_series.requeue_count == RGB_SERIES_COUNT - 1
    assert depth_kernel.closed and rgb_kernel.closed


def test_series_finalizer_is_counted_and_failure_preserves_frames_and_telemetry(healthy, tmp_path):
    journal: list[str] = []
    backend, _depth, _rgb = _backend(
        healthy,
        tmp_path,
        illumination_factory=lambda _sysfs, _dev: _FailedIlluminationBackend(),
        illumination_journal=journal,
    )
    original_ioctl = backend.ioctl_fn

    def telemetry_ioctl(fd, request, packed, mutate):
        if request == VIDIOC_G_CTRL:
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            struct.pack_into(CONTROL_FORMAT, packed, 0, requested, requested + 1)
            return 0
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = telemetry_ioctl
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.status.value == "complete"
    assert not outcome.succeeded
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == RGB_SERIES_COUNT
    assert outcome.rgb_series.telemetry_enabled
    assert len(outcome.rgb_series.observations) == RGB_SERIES_COUNT
    assert outcome.rgb_series.artifact_record.ok
    assert len(outcome.rgb_series.artifact_record.files) == RGB_SERIES_COUNT
    assert len(outcome.rgb_series.telemetry_samples) == RGB_SERIES_COUNT + 1
    assert outcome.rgb_series.retained_bytes > 0
    assert journal == ["illumination-off:depth"]
    assert outcome.illumination_shutdown.status is ShutdownStatus.IOCTL_ERROR


def test_series_outer_shield_signal_is_retained_without_duplicate_finalizer(healthy, tmp_path):
    journal: list[str] = []
    backend, _depth, _rgb = _backend(healthy, tmp_path, illumination_journal=journal)

    @contextmanager
    def shield():
        yield [signal.SIGTERM]

    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None, termination_shield=shield).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "complete"
    assert outcome.succeeded
    assert journal == ["illumination-off:depth"]
    assert outcome.cleanup.deferred_signals == (signal.SIGTERM,)


def test_series_human_shutdown_failure_reports_recovery(healthy, tmp_path, monkeypatch, capsys):
    def factory(output_dir: Path):
        return _backend(
            healthy,
            output_dir,
            illumination_factory=lambda _sysfs, _dev: _FailedIlluminationBackend(),
        )[0]

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    result = cli.main(
        [
            "paired-rgb-series",
            RGB_SERIES_EXECUTE_OPT_IN,
            "--serial",
            AUTHORIZED_SERIAL,
            "--output-dir",
            str(tmp_path / "series-human"),
        ]
    )
    captured = capsys.readouterr()
    assert result != 0
    assert "Blue-light off command (transfer only, NOT an optical result):" in captured.out
    assert "Overall result: FAILED" in captured.out
    assert "blue-light off command was not carried through" in captured.err


def test_series_rgb_control_telemetry_is_same_fd_fixed_order_and_four_points(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original = backend.ioctl_fn
    events: list[tuple[str, int | None]] = []

    def ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if fd == rgb.fd and request == VIDIOC_G_CTRL:
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            events.append(("g_ctrl", requested))
            struct.pack_into(CONTROL_FORMAT, packed, 0, requested, requested + 100)
            return 0
        return original(fd, request, packed, mutate)

    backend.ioctl_fn = ioctl
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.rgb_series is not None
    telemetry = outcome.rgb_series.telemetry_samples
    assert outcome.rgb_series.telemetry_enabled
    assert len(telemetry) == RGB_SERIES_COUNT + 1
    assert all(sample.complete for sample in telemetry)
    assert [record.control.name for record in telemetry[0].records] == [
        spec.name for spec in RGB_SERIES_CONTROL_WHITELIST
    ]
    assert len(events) == 12
    assert all(record.status.value == "value" for sample in telemetry for record in sample.records)
    assert telemetry[0].observation_index is None
    assert [sample.observation_index for sample in telemetry[1:]] == [0, 1, 2]


def test_series_rgb_control_telemetry_pins_rgb_ioctl_order(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_ioctl = backend.ioctl_fn
    original_poll = backend.poll_fn
    events: list[str] = []

    def ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if fd == rgb.fd:
            labels = {
                VIDIOC_STREAMON: "streamon",
                VIDIOC_DQBUF: "dqbuf",
                VIDIOC_QBUF: "qbuf",
                VIDIOC_STREAMOFF: "streamoff",
            }
            if request == VIDIOC_G_CTRL:
                requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
                events.append(f"g:{requested:08x}")
                struct.pack_into(CONTROL_FORMAT, packed, 0, requested, 3)
                return 0
            if request in labels:
                events.append(labels[request])
        return original_ioctl(fd, request, packed, mutate)

    def poll(fd: int, timeout_s: float) -> int:
        if fd == rgb.fd:
            events.append("poll")
        return original_poll(fd, timeout_s)

    backend.ioctl_fn = ioctl
    backend.poll_fn = poll
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.status.value == "complete"
    streamon_index = events.index("streamon")
    assert events[:streamon_index].count("qbuf") == 4
    rgb_events = events[streamon_index:]
    controls = [f"g:{spec.control_id:08x}" for spec in RGB_SERIES_CONTROL_WHITELIST]
    assert rgb_events == [
        "streamon",
        *controls,
        "poll",
        "dqbuf",
        *controls,
        "qbuf",
        "poll",
        "dqbuf",
        *controls,
        "qbuf",
        "poll",
        "dqbuf",
        *controls,
        "streamoff",
    ]


def test_series_rgb_control_telemetry_short_series_reaches_only_actual_points(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_poll = backend.poll_fn
    polls = 0

    def one_then_timeout(fd: int, timeout_s: float) -> int:
        nonlocal polls
        if fd == rgb.fd:
            polls += 1
            return 1 if polls == 1 else 0
        return original_poll(fd, timeout_s)

    backend.poll_fn = one_then_timeout
    original_ioctl = backend.ioctl_fn

    def controls(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if fd == rgb.fd and request == VIDIOC_G_CTRL:
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            struct.pack_into(CONTROL_FORMAT, packed, 0, requested, 5)
            return 0
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = controls
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.status.value == "timeout"
    assert outcome.rgb_series is not None
    assert len(outcome.rgb_series.telemetry_samples) == 2
    assert sum(len(sample.records) for sample in outcome.rgb_series.telemetry_samples) == 6
    telemetry = cast(dict[str, object], outcome.rgb_series.as_dict()["telemetry"])
    assert telemetry["status"] == "partial"


def test_series_rgb_control_telemetry_mismatch_is_typed_error(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original = backend.ioctl_fn
    mismatch_id = RGB_SERIES_CONTROL_WHITELIST[1].control_id + 1

    def mismatch(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if fd == rgb.fd and request == VIDIOC_G_CTRL:
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            returned = (
                mismatch_id
                if requested == RGB_SERIES_CONTROL_WHITELIST[1].control_id
                else requested
            )
            struct.pack_into(CONTROL_FORMAT, packed, 0, returned, 9)
            return 0
        return original(fd, request, packed, mutate)

    backend.ioctl_fn = mismatch
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.status.value == "complete"
    assert outcome.rgb_series is not None
    record = outcome.rgb_series.telemetry_samples[0].records[1]
    assert record.status.value == "returned_id_mismatch"
    assert record.returned_id == mismatch_id
    telemetry = cast(dict[str, object], outcome.rgb_series.as_dict()["telemetry"])
    assert telemetry["status"] == "errors"


@pytest.mark.parametrize("failure", ["signal", "exception"])
def test_series_rgb_control_telemetry_abort_retains_frame_and_cleanup(healthy, tmp_path, failure):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original = backend.ioctl_fn
    calls = 0

    def interrupted(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal calls
        if fd == rgb.fd and request == VIDIOC_G_CTRL:
            calls += 1
            if calls == len(RGB_SERIES_CONTROL_WHITELIST) + 1:
                if failure == "signal":
                    raise SignalReceived(signal.SIGTERM, "telemetry signal after preserved frame")
                raise RuntimeError("telemetry exception after preserved frame")
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            struct.pack_into(CONTROL_FORMAT, packed, 0, requested, 5)
            return 0
        return original(fd, request, packed, mutate)

    backend.ioctl_fn = interrupted
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.status.value == ("cancelled" if failure == "signal" else "failed")
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 1
    assert len(outcome.rgb_series.telemetry_samples) == 2
    assert outcome.rgb_series.telemetry_samples[1].records[0].status.value == (
        "interrupted" if failure == "signal" else "exception"
    )
    assert (tmp_path / "paired" / "rgb" / "rgb-series" / "rgb-series-000.jpg").exists()
    assert "rgb streamoff" in outcome.cleanup.actions
    assert "rgb close" in outcome.cleanup.actions


def test_series_rgb_control_telemetry_partial_point_and_ordinary_error_continue(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original = backend.ioctl_fn
    attempts = 0

    def ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal attempts
        if fd == rgb.fd and request == VIDIOC_G_CTRL:
            attempts += 1
            requested, _ = struct.unpack(CONTROL_FORMAT, bytes(packed))
            if attempts == 2:
                raise OSError(errno.EIO, os.strerror(errno.EIO))
            struct.pack_into(CONTROL_FORMAT, packed, 0, requested, 7)
            return 0
        return original(fd, request, packed, mutate)

    backend.ioctl_fn = ioctl
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL,
        opt_in=RGB_SERIES_EXECUTE_OPT_IN,
        read_rgb_controls_during_series=True,
    )
    assert outcome.rgb_series is not None
    assert attempts == 12
    assert len(outcome.rgb_series.telemetry_samples) == 4
    assert outcome.rgb_series.telemetry_samples[0].records[1].status.value == "ioctl_error"
    assert outcome.rgb_series.telemetry_samples[0].records[1].errno == errno.EIO
    assert outcome.rgb_series.delivered_count == 3


def test_series_without_telemetry_performs_no_g_ctrl(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.rgb_series is not None
    assert not outcome.rgb_series.telemetry_enabled
    assert outcome.rgb_series.telemetry_samples == ()
    assert rgb.count(VIDIOC_G_CTRL) == 0


def test_series_json_exposes_strict_jpeg_and_artifact_metadata(healthy, tmp_path):
    backend, _depth, _rgb = _backend(healthy, tmp_path)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    payload = outcome.as_dict()
    plan = cast(dict[str, object], payload["plan"])
    execution = cast(dict[str, object], plan["execution"])
    roles = cast(dict[str, object], plan["roles"])
    rgb_plan = cast(dict[str, object], roles[RGB_ROLE])
    assert execution["required_opt_in"] == RGB_SERIES_EXECUTE_OPT_IN
    assert execution["series_mode"] is True
    assert rgb_plan["max_frames"] == RGB_SERIES_COUNT
    series_plan = cast(dict[str, object], plan["series"])
    assert series_plan["count"] == RGB_SERIES_COUNT
    series = cast(dict[str, object], payload["rgb_series"])
    assert series["delivered_count"] == RGB_SERIES_COUNT
    assert series["poll_count"] == RGB_SERIES_COUNT
    assert series["dequeue_count"] == RGB_SERIES_COUNT
    assert series["requeue_count"] == RGB_SERIES_COUNT - 1
    assert series["max_total_bytes"] == RGB_SERIES_MAX_TOTAL_BYTES
    observations = cast(list[dict[str, object]], series["observations"])
    assert [item["index"] for item in observations] == [0, 1, 2]
    assert all(
        cast(dict[str, object], item["jpeg"])["status"] == "well_formed" for item in observations
    )
    assert all(item["eoi_prefix_bytes"] for item in observations)
    assert all(item["trailer_bytes"] == 0 for item in observations)
    json.dumps(payload)


def test_trailing_data_is_preserved_for_every_observation_but_not_success(healthy, tmp_path):
    backend, _depth, _rgb = _backend(healthy, tmp_path)
    # Rebuild only the RGB fake's mapped payload through the profile fixture
    # seam; the four bytes are deliberately opaque and never trimmed.
    _rgb.frame_payload = baseline_jpeg(1280, 800) + b"\x95\x81\x7d\x01"
    _rgb.bytesused = len(_rgb.frame_payload)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )

    assert outcome.status.value == "failed"
    assert outcome.rgb_series is not None
    assert not outcome.rgb_series.strict_success
    assert [item.status.value for item in outcome.rgb_series.observations] == [
        "malformed_frame",
        "malformed_frame",
        "malformed_frame",
    ]
    assert all(item.trailer_bytes == 4 for item in outcome.rgb_series.observations)
    assert all(item.trailer_hex == "95817d01" for item in outcome.rgb_series.observations)
    assert all(
        item.eoi_prefix_bytes == len(baseline_jpeg(1280, 800))
        for item in outcome.rgb_series.observations
    )


def test_mixed_jpeg_outcomes_are_each_retained_and_strictly_classified(healthy, tmp_path):
    valid = baseline_jpeg(1280, 800)
    trailing = valid + b"\x01\x02\x03\x04"
    backend, _depth, _rgb = _backend(
        healthy,
        tmp_path,
        rgb_kwargs={"frame_payloads": (trailing, valid, b"not-jpeg")},
    )
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "failed"
    assert outcome.rgb_series is not None
    assert [item.jpeg["status"] for item in outcome.rgb_series.observations] == [
        "trailing_data",
        "well_formed",
        "missing_soi",
    ]
    assert len(outcome.rgb_series.artifact_record.files) == RGB_SERIES_COUNT


def test_partial_timeout_does_not_requeue_after_deadline(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_poll = backend.poll_fn
    calls = 0

    def one_then_timeout(fd: int, timeout_s: float) -> int:
        nonlocal calls
        if fd == rgb.fd:
            calls += 1
            return 1 if calls == 1 else 0
        return original_poll(fd, timeout_s)

    backend.poll_fn = one_then_timeout
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "timeout"
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 1
    assert len(outcome.rgb_series.polls) == 2
    assert rgb.count(VIDIOC_DQBUF) == 1
    assert rgb.count(VIDIOC_QBUF) == 5


def test_requeue_failure_stops_without_a_second_dequeue(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_ioctl = backend.ioctl_fn
    rgb_qbufs = 0

    def fail_first_series_requeue(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal rgb_qbufs
        if fd == rgb.fd and request == VIDIOC_QBUF:
            rgb_qbufs += 1
            if rgb_qbufs == 5:
                raise OSError(5, os.strerror(5))
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = fail_first_series_requeue
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "failed"
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 1
    assert rgb.count(VIDIOC_DQBUF) == 1
    assert rgb_qbufs == 5


def test_dqbuf_errno_is_retained_as_terminal_series_evidence(healthy, tmp_path):
    backend, _depth, rgb = _backend(
        healthy, tmp_path, rgb_kwargs={"errors": {VIDIOC_DQBUF: errno.EIO}}
    )
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "failed"
    assert outcome.rgb_series is not None
    terminal = outcome.rgb_series.terminal_verdict
    assert terminal is not None
    assert terminal.status.value == "dequeue_error"
    assert terminal.errno == errno.EIO
    assert terminal.errno_name == "EIO"
    assert "Input/output error" in terminal.detail
    assert outcome.rgb_result is not None
    assert outcome.rgb_result.errno == errno.EIO
    assert outcome.rgb_series.delivered_count == 0
    assert outcome.rgb_series.dequeue_count == 1
    assert outcome.rgb_series.requeue_count == 0
    assert rgb.count(VIDIOC_QBUF) == 4


def test_per_payload_ceiling_stops_without_empty_jpeg_or_requeue(healthy, tmp_path):
    backend, _depth, rgb = _backend(
        healthy,
        tmp_path,
        rgb_kwargs={
            "frame_payload": b"x" * (RGB_MAX_BUFFER_BYTES + 1),
        },
    )
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 0
    assert outcome.rgb_series.terminal_verdict is not None
    assert outcome.rgb_series.terminal_verdict.status.value == "oversize_frame"
    assert outcome.rgb_series.observations == ()
    assert outcome.rgb_series.requeue_count == 0
    assert rgb.count(VIDIOC_DQBUF) == 1
    assert rgb.count(VIDIOC_QBUF) == 4


def test_total_retained_ceiling_is_explicit_and_bounded(healthy, tmp_path, monkeypatch):
    monkeypatch.setattr("scanner3d.paired_capture.RGB_SERIES_MAX_TOTAL_BYTES", 1)
    backend, _depth, rgb = _backend(healthy, tmp_path)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 0
    assert outcome.rgb_series.retained_bytes == 0
    assert outcome.rgb_series.max_total_bytes == 1
    assert outcome.rgb_series.terminal_verdict is not None
    assert outcome.rgb_series.terminal_verdict.status.value == "oversize_frame"
    assert rgb.count(VIDIOC_DQBUF) == 1
    assert rgb.count(VIDIOC_QBUF) == 4


def test_existing_indexed_file_is_refused_without_overwrite(healthy, tmp_path):
    target = tmp_path / "paired" / "rgb" / "rgb-series"
    target.mkdir(parents=True)
    existing = target / "rgb-series-000.jpg"
    existing.write_bytes(b"sentinel")
    backend, _depth, _rgb = _backend(healthy, tmp_path)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "failed"
    assert existing.read_bytes() == b"sentinel"
    assert outcome.rgb_series is not None
    assert not outcome.rgb_series.artifact_record.ok


def test_series_wait_guard_refuses_series_and_single_reentry(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    node = backend.resolve_node(AUTHORIZED_SERIAL, RGB_ROLE)
    handle = backend.open(node, RGB_ROLE)
    backend.configure(handle, RGB_SPEC)
    backend.map_queue(handle, RGB_SPEC)
    backend.stream_on(handle)
    first = backend.wait_rgb_series(
        handle,
        8.0,
        RGB_SERIES_COUNT,
        RGB_SERIES_MAX_TOTAL_BYTES,
    )
    assert first.poll_count == RGB_SERIES_COUNT
    counts = (rgb.count(VIDIOC_QUERYCAP), rgb.count(VIDIOC_DQBUF), rgb.count(VIDIOC_QBUF))
    with pytest.raises(PairedBackendRefusal, match="already attempted"):
        backend.wait_rgb_series(handle, 8.0, RGB_SERIES_COUNT, RGB_SERIES_MAX_TOTAL_BYTES)
    with pytest.raises(PairedBackendRefusal, match="already attempted"):
        backend.wait_frame(handle, 8.0, 1)
    assert counts == (
        rgb.count(VIDIOC_QUERYCAP),
        rgb.count(VIDIOC_DQBUF),
        rgb.count(VIDIOC_QBUF),
    )
    backend.stream_off(handle)
    backend.unmap(handle)
    backend.free_queue(handle)
    backend.close(handle)


def test_series_post_open_identity_drift_is_refused_before_ioctl(healthy, tmp_path):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    node = backend.resolve_node(AUTHORIZED_SERIAL, RGB_ROLE)
    original = backend.fstat_fn

    def drift(_fd: int):
        stat = original(_fd)
        return os.stat_result(
            (
                stat.st_mode,
                stat.st_ino + 1,
                stat.st_dev,
                stat.st_nlink,
                stat.st_uid,
                stat.st_gid,
                stat.st_size,
                stat.st_atime,
                stat.st_mtime,
                stat.st_ctime,
            )
        )

    backend.fstat_fn = drift
    handle = backend.open(node, RGB_ROLE)
    assert handle.identity_error is not None
    assert rgb.count(VIDIOC_QUERYCAP) == 0
    backend.close(handle)


def test_series_cleanup_failure_overrides_status_with_exit_nine(healthy, tmp_path):
    backend, _depth, _rgb = _backend(
        healthy, tmp_path, rgb_kwargs={"errors": {VIDIOC_STREAMOFF: errno.EIO}}
    )
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.exit_code == 9
    assert outcome.status.value == "failed"
    assert not outcome.cleanup.ok
    assert outcome.rgb_series is not None


def test_series_live_cli_json_out_and_shutdown_failure_recovery(
    healthy, tmp_path, monkeypatch, capsys
):
    def factory(output_dir: Path):
        backend, _depth, _rgb = _backend(
            healthy,
            output_dir,
            illumination_factory=lambda _sysfs, _dev: _FailedIlluminationBackend(),
        )
        return backend

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    output_path = tmp_path / "series.json"
    args = [
        "paired-rgb-series",
        RGB_SERIES_EXECUTE_OPT_IN,
        "--serial",
        AUTHORIZED_SERIAL,
        "--output-dir",
        str(tmp_path / "series-out"),
        "--json",
        "--json-out",
        str(output_path),
    ]
    result = cli.main(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result != 0
    assert payload["status"] == "complete"
    assert payload["succeeded"] is False
    assert payload["rgb_series"]["delivered_count"] == RGB_SERIES_COUNT
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert "blue-light off command was not carried through" in captured.err


def test_series_cli_gates_do_not_construct_factory_until_all_pass(monkeypatch, tmp_path, capsys):
    calls: list[Path] = []

    def factory(path: Path):
        calls.append(path)
        raise AssertionError("factory reached only for the final gate test")

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    assert cli.main(["paired-rgb-series", RGB_SERIES_EXECUTE_OPT_IN, "--serial", "wrong"]) == 10
    assert (
        cli.main(["paired-rgb-series", RGB_SERIES_EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
        == 10
    )
    assert not calls
    with pytest.raises(AssertionError, match="factory reached"):
        cli.main(
            [
                "paired-rgb-series",
                RGB_SERIES_EXECUTE_OPT_IN,
                "--serial",
                AUTHORIZED_SERIAL,
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert calls == [tmp_path]
    capsys.readouterr()


@pytest.mark.parametrize("boundary", ["dq", "poll"])
def test_series_signal_is_cancelled_and_cleanup_remains_exhaustive(healthy, tmp_path, boundary):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_ioctl = backend.ioctl_fn
    original_poll = backend.poll_fn

    def interrupted_ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if boundary == "dq" and fd == rgb.fd and request == VIDIOC_DQBUF:
            raise SignalReceived(signal.SIGTERM, "synthetic RGB series DQBUF signal")
        return original_ioctl(fd, request, packed, mutate)

    def interrupted_poll(fd: int, timeout_s: float) -> int:
        if boundary == "poll" and fd == rgb.fd:
            raise SignalReceived(signal.SIGTERM, "synthetic RGB series poll signal")
        return original_poll(fd, timeout_s)

    backend.ioctl_fn = interrupted_ioctl
    backend.poll_fn = interrupted_poll
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "cancelled"
    assert rgb.closed
    assert "rgb close" in outcome.cleanup.actions


@pytest.mark.parametrize("boundary", ["poll", "dq", "qbuf"])
def test_series_signal_after_progress_retains_partial_series_and_namespace(
    healthy, tmp_path, boundary
):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_ioctl = backend.ioctl_fn
    original_poll = backend.poll_fn
    poll_calls = 0
    dq_calls = 0
    qbuf_calls = 0

    def interrupted_ioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal dq_calls, qbuf_calls
        if fd == rgb.fd and request == VIDIOC_DQBUF:
            dq_calls += 1
            if boundary == "dq" and dq_calls == 2:
                raise SignalReceived(signal.SIGTERM, "series DQBUF after observation 0")
        if fd == rgb.fd and request == VIDIOC_QBUF:
            qbuf_calls += 1
            if boundary == "qbuf" and qbuf_calls == 5:
                raise SignalReceived(signal.SIGTERM, "series QBUF after observation 0")
        return original_ioctl(fd, request, packed, mutate)

    def interrupted_poll(fd: int, timeout_s: float) -> int:
        nonlocal poll_calls
        if fd == rgb.fd:
            poll_calls += 1
            if boundary == "poll" and poll_calls == 2:
                raise SignalReceived(signal.SIGTERM, "series poll after observation 0")
        return original_poll(fd, timeout_s)

    backend.ioctl_fn = interrupted_ioctl
    backend.poll_fn = interrupted_poll
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "cancelled"
    assert outcome.exit_code == 13
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 1
    assert outcome.rgb_series.terminal_verdict is not None
    assert outcome.rgb_series.terminal_verdict.status.value == "interrupted"
    assert outcome.rgb_series.artifact_record.ok
    assert (tmp_path / "paired" / "rgb" / "rgb-series" / "rgb-series-000.jpg").exists()
    assert not (tmp_path / "paired" / "rgb" / "rgb-frame.jpg").exists()
    assert outcome.rgb_result is not None
    assert all(
        artifact.path is None or "rgb-series" in artifact.path
        for artifact in outcome.rgb_result.artifacts.files
    )
    assert "rgb close" in outcome.cleanup.actions
    payload = outcome.as_dict()
    assert payload["rgb_frame_bytes"] == outcome.rgb_series.observations[0].artifact.byte_count
    assert cast(dict[str, object], payload["delivery"])["rgb"] is True


def test_series_exception_after_progress_retains_partial_series_without_legacy_artifact(
    healthy, tmp_path
):
    backend, _depth, rgb = _backend(healthy, tmp_path)
    original_ioctl = backend.ioctl_fn
    calls = 0

    def fail_first_series_qbuf(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        nonlocal calls
        if fd == rgb.fd and request == VIDIOC_QBUF:
            calls += 1
            if calls == 5:
                raise RuntimeError("synthetic series exception after observation 0")
        return original_ioctl(fd, request, packed, mutate)

    backend.ioctl_fn = fail_first_series_qbuf
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == "failed"
    assert outcome.rgb_series is not None
    assert outcome.rgb_series.delivered_count == 1
    assert outcome.rgb_series.terminal_verdict is not None
    assert outcome.rgb_series.terminal_verdict.status.value == "ioctl_error"
    assert not (tmp_path / "paired" / "rgb" / "rgb-frame.jpg").exists()
    payload = outcome.as_dict()
    assert payload["rgb_frame_bytes"] == outcome.rgb_series.observations[0].artifact.byte_count
    assert cast(dict[str, object], payload["delivery"])["rgb"] is True


@pytest.mark.parametrize("failure", ["signal", "exception"])
def test_series_preservation_failure_retains_copied_observation_and_delivery(
    healthy, tmp_path, monkeypatch, failure
):
    backend, _depth, _rgb = _backend(healthy, tmp_path)

    def fail_write(_path: Path, _payload: bytes, *, ceiling: int) -> None:
        if failure == "signal":
            raise SignalReceived(signal.SIGTERM, "synthetic preservation signal")
        raise RuntimeError("synthetic preservation exception")

    monkeypatch.setattr(paired_capture, "_write_new_file", fail_write)
    outcome = RgbSeriesCapture(backend, sleep=lambda _seconds: None).run(
        AUTHORIZED_SERIAL, opt_in=RGB_SERIES_EXECUTE_OPT_IN
    )
    assert outcome.status.value == ("cancelled" if failure == "signal" else "failed")
    assert outcome.rgb_series is not None
    series = outcome.rgb_series
    assert series.delivered_count == 1
    assert series.dequeue_count == 1
    assert series.retained_bytes == series.observations[0].artifact.byte_count
    assert series.artifact_record.attempted
    assert not series.artifact_record.ok
    observation = series.observations[0]
    assert not observation.artifact.ok
    assert observation.artifact.path is None
    assert observation.artifact.byte_count == series.retained_bytes
    assert observation.artifact.sha256
    assert series.terminal_verdict is not None
    assert series.terminal_verdict.status.value == (
        "interrupted" if failure == "signal" else "ioctl_error"
    )
    assert not (tmp_path / "paired" / "rgb" / "rgb-series-000.jpg").exists()
    assert not (tmp_path / "paired" / "rgb" / "rgb-frame.jpg").exists()
    payload = outcome.as_dict()
    assert payload["rgb_frame_bytes"] == series.retained_bytes
    assert cast(dict[str, object], payload["delivery"])["rgb"] is True
    assert "rgb close" in outcome.cleanup.actions


def test_series_dry_run_and_exact_gates_are_inert(capsys):
    assert cli.main(["paired-rgb-series"]) == 0
    dry_run = capsys.readouterr().out
    assert "Status: not_attempted" in dry_run
    # Refused either way; the REASON now depends on whether a device is
    # configured. With none configured the serial guard has nothing to compare
    # against and the refusal comes from the missing live backend instead.
    assert cli.main(["paired-rgb-series", RGB_SERIES_EXECUTE_OPT_IN]) == 10
    assert "Status: refused" in capsys.readouterr().out


def test_series_telemetry_flag_is_explicit_but_dry_without_exact_opt_in(capsys):
    assert cli.main(["paired-rgb-series", "--read-rgb-controls-during-series", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_attempted"
    assert payload["rgb_control_telemetry"] == {"requested": True, "enabled": False}
    assert payload["rgb_series"] is None


def test_series_telemetry_flag_alone_never_constructs_live_backend(monkeypatch, tmp_path):
    calls: list[Path] = []

    def factory(path: Path):
        calls.append(path)
        raise AssertionError("telemetry flag alone must not activate the backend")

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    assert (
        cli.main(
            [
                "paired-rgb-series",
                "--read-rgb-controls-during-series",
                "--serial",
                AUTHORIZED_SERIAL,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == []


def test_series_missing_output_refusal_preserves_telemetry_request(capsys):
    assert (
        cli.main(
            [
                "paired-rgb-series",
                RGB_SERIES_EXECUTE_OPT_IN,
                "--serial",
                AUTHORIZED_SERIAL,
                "--read-rgb-controls-during-series",
                "--json",
            ]
        )
        == 10
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["rgb_control_telemetry"] == {"requested": True, "enabled": False}


def test_series_refusal_keeps_series_plan_shape():
    outcome = RgbSeriesCapture(None).run("wrong", opt_in=RGB_SERIES_EXECUTE_OPT_IN)
    assert outcome.series_mode
    plan = outcome.as_dict()["plan"]
    assert cast(dict[str, object], plan)["execution"]
    assert cast(dict[str, object], cast(dict[str, object], plan)["roles"])[RGB_ROLE]


def test_series_result_reports_sequence_gaps_and_intervals():
    # The calculation is intentionally independent of JPEG interpretation.
    from scanner3d.v4l2_capture import BufferInfo, CaptureStatus, FrameArtifact

    def info(sequence: int, timestamp_usec: int) -> BufferInfo:
        return BufferInfo(
            index=0,
            buf_type=1,
            bytesused=2,
            flags=0,
            flag_names=(),
            field=1,
            field_name="none",
            timestamp_sec=10,
            timestamp_usec=timestamp_usec,
            timestamp_type="monotonic",
            timestamp_source="unknown",
            timecode_type=0,
            timecode_flags=0,
            timecode_frames=0,
            timecode_seconds=0,
            timecode_minutes=0,
            timecode_hours=0,
            timecode_userbits="",
            sequence=sequence,
            memory=1,
            offset=0,
            length=2,
        )

    artifact = FrameArtifact("mjpeg-series", "/tmp/x", 2, "a" * 64, True, "ok")
    from scanner3d.paired_capture import RgbSeriesObservation

    observations = tuple(
        RgbSeriesObservation(
            i,
            info(sequence, i * 1000),
            CaptureStatus.COMPLETE,
            "ok",
            artifact,
            {},
            2,
            "b" * 64,
            0,
            "",
        )
        for i, sequence in enumerate((10, 11, 14))
    )
    result = RgbSeriesResult(CaptureStatus.COMPLETE, "ok", observations=observations)
    assert result.sequence_gaps == (2,)
    assert result.intervals_ns == (1_000_000, 1_000_000)
