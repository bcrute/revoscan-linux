"""Offline diagnostic tests: no scanner opens or device commands escape fakes."""

from __future__ import annotations

import ctypes
import importlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def timing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_timing_probe")


def test_xu_abi_matches_linux_header(timing, monkeypatch):
    def ioctl(fd, request, packed, mutate):
        assert fd == 17 and request == 0xC0107521 and mutate
        assert len(packed) == 16
        unit, selector, query, size, address = struct.unpack("@BBBHP", packed)
        assert (unit, selector, query, size) == (4, 7, 1, 60)
        assert ctypes.string_at(address, 60) == b"test".ljust(60, b"\0")
        return 0

    monkeypatch.setattr(timing.fcntl, "ioctl", ioctl)
    assert timing.xu(17, 7, 1, b"test".ljust(60, b"\0"))[:4] == b"test"


@pytest.mark.parametrize("selector,query,size", [(3, 1, 60), (1, 0x81, 60), (7, 1, 4)])
def test_xu_refuses_unknown_shapes(timing, monkeypatch, selector, query, size):
    monkeypatch.setattr(timing.fcntl, "ioctl", lambda *a: pytest.fail("ioctl escaped guard"))
    with pytest.raises(ValueError):
        timing.xu(17, selector, query, bytes(size))


def test_set_timing_orders_exposure_and_frame_window(timing, monkeypatch):
    monkeypatch.setattr(timing, "state", lambda fd: ({"frame_time": 20000}, ""))
    commands = []
    monkeypatch.setattr(timing, "command", lambda fd, name, value: commands.append((name, value)))
    timing.set_timing(17, 10000, 10000)
    timing.set_timing(17, 20000, 20000)
    assert commands == [
        ("exposure_time", 10000),
        ("frame_time", 10000),
        ("frame_time", 20000),
        ("exposure_time", 20000),
    ]


@pytest.mark.parametrize("frame,exposure", [(0, 10000), (10000, 20000), (10000, 1)])
def test_invalid_timing_pair_refused_before_any_write(timing, monkeypatch, frame, exposure):
    monkeypatch.setattr(timing, "command", lambda *a: pytest.fail("setting write escaped guard"))
    with pytest.raises(ValueError):
        timing.set_timing(17, frame, exposure)


def test_only_fixed_mailbox_commands_can_be_formatted(timing, monkeypatch):
    commands = []
    monkeypatch.setattr(timing, "xu", lambda fd, sel, qry, p: commands.append((sel, qry, p)))
    timing.command(17, "frame_time", 15000)
    timing.command(17, "exposure_time")
    assert commands == [
        (7, 1, b"echo s 0x910 15000 >/dev/rk_preisp".ljust(60, b"\0")),
        (7, 1, b"echo s 0x9b1 0 >/dev/rk_preisp".ljust(60, b"\0")),
    ]
    with pytest.raises(ValueError):
        timing.command(17, "frame_time", 0)
    with pytest.raises(KeyError):
        timing.command(17, "calibrate", 10000)
    assert len(commands) == 2


def test_state_rejects_stale_getters(timing, monkeypatch):
    text = "\n".join(
        f"[10.000] DSP: id_msg_cam_get_{key} [{value}]"
        for key, value in [("frame_time", 20000), ("exposure_time", 20000), ("auto_expo_mode", 0)]
    )
    monkeypatch.setattr(timing, "kernel_log", lambda fd: text)
    monkeypatch.setattr(timing, "command", lambda *a: None)
    with pytest.raises(RuntimeError, match="No fresh"):
        timing.state(17)
    logs = iter([text, text.replace("10.000", "11.000")])
    monkeypatch.setattr(timing, "kernel_log", lambda fd: next(logs))
    assert timing.state(17)[0] == dict(frame_time=20000, exposure_time=20000, auto_expo_mode=0)


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_timing_main_restores_on_capture_success_exception_and_interrupt(
    timing, monkeypatch, tmp_path, failure
):
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["probe", "--execute", "--timings", "10000", "--frames", "16", "--output", str(output)],
    )
    device = dict(frame_time=20000, exposure_time=20000, auto_expo_mode=0)
    closed, writes = [], []
    monkeypatch.setattr(timing, "resolve_serial", lambda: "test-scanner")
    monkeypatch.setattr(
        timing, "resolve_capture_node", lambda _: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(timing, "wait_for_capture_access", lambda **k: SimpleNamespace(ready=True))
    descriptors = iter((17, 18, 19))
    monkeypatch.setattr(timing.os, "open", lambda *a: next(descriptors))
    monkeypatch.setattr(timing.os, "close", closed.append)
    monkeypatch.setattr(timing, "state", lambda fd: (dict(device), "test log"))
    monkeypatch.setattr(timing, "temperatures", lambda fd: dict(cpu_c=60.0, npu_c=60.0))

    def set_timing(fd, frame, exposure):
        writes.append((frame, exposure))
        device.update(frame_time=frame, exposure_time=exposure)

    def capture(*a, **kwargs):
        kwargs["monitor"](17)
        if failure:
            raise failure("synthetic capture failure")
        return {"summary": {"delivered": 16}}

    monkeypatch.setattr(timing, "set_timing", set_timing)
    monkeypatch.setattr(timing, "capture", capture)
    if failure:
        with pytest.raises(failure, match="synthetic"):
            timing.main()
    else:
        timing.main()
    report = json.loads((output / "report.json").read_text())
    assert writes == [(10000, 10000), (20000, 20000)]
    assert closed == ([17, 18] if failure else [17, 18, 19])
    assert report["restoration_verified"]
    assert report["restored"] == report["original"]
    assert report["status"] == ("failed" if failure else "completed")


def test_missing_temperature_aborts_and_clears_protocol(timing, monkeypatch):
    cleaned = []
    monkeypatch.setattr(timing.ObservationOperation, "read_file", lambda *a: b"")
    monkeypatch.setattr(
        timing.ObservationOperation, "protocol_cleanup", lambda *a: cleaned.append(1)
    )
    with pytest.raises(ValueError):
        timing.temperatures(17)
    assert cleaned == [1]


@pytest.mark.parametrize("plan", ["kernel", "controller-log"])
def test_short_observation_commands_fit_wire_payload(timing, monkeypatch, plan):
    module = importlib.import_module("scanner_internal_probe")
    op = module.ObservationOperation(None, plan)
    commands = []
    monkeypatch.setattr(op, "issue", lambda fd, request, payload: commands.append(payload))
    monkeypatch.setattr(op, "read_file", lambda *a: b"log")
    op.perform(17, [])
    assert len(commands) == 1 and len(commands[0]) == 60
    assert commands[0][-1] == 0


def test_file_read_cleanup_is_owed_even_when_path_write_fails(timing, monkeypatch):
    module = importlib.import_module("scanner_internal_probe")
    op = module.ObservationOperation(None, "inventory")
    attempts = []

    def issue(fd, request, payload=None):
        attempts.append(payload)
        if len(attempts) == 1:
            raise OSError("synthetic interrupted path write")
        return bytes(60)

    monkeypatch.setattr(op, "issue", issue)
    with pytest.raises(OSError):
        op.read_file(17, "/proc/uptime", 512)
    assert op.clear_owed
    assert op.protocol_cleanup(17).ok
    assert attempts[-1] == bytes(60)
    assert not op.clear_owed


@pytest.mark.parametrize(
    "reply", [struct.pack("<I", 57) + bytes(56), struct.pack("<I", 10) + bytes(56)]
)
def test_file_read_rejects_malformed_or_over_capacity_reply(timing, monkeypatch, reply):
    module = importlib.import_module("scanner_internal_probe")
    op = module.ObservationOperation(None, "inventory")
    monkeypatch.setattr(op, "issue", lambda *a: reply)
    with pytest.raises(RuntimeError, match="over-capacity"):
        op.read_file(17, "/proc/uptime", 4)
    assert op.clear_owed


def test_acquisition_hashes_complete_frames_after_stream_closes(timing, monkeypatch):
    probe = importlib.import_module("scanner_fps_probe")
    record = probe.FrameRecord(0, 0, 0, 8, 1.0, None, True)
    second = probe.FrameRecord(1, 1, 1, 8, 1.1, 100.0, True)
    closed = []
    raw = struct.pack("<4H", 0, 42, 1234, 65535)
    real_hash = probe.hashlib.sha256

    def hashed(value):
        assert closed, "payload analysis must not throttle the stream"
        return real_hash(value)

    def stream(**kwargs):
        kwargs["on_frame"](raw, record)
        kwargs["on_frame"](raw, second)
        closed.append(True)
        return SimpleNamespace(
            records=[record, second], observed_fps=10.0, as_dict=lambda: {"delivered": 2}
        )

    monkeypatch.setattr(
        probe.v4l2_stream, "wait_for_capture_access", lambda **k: SimpleNamespace(ready=True)
    )
    monkeypatch.setattr(probe.v4l2_stream, "stream_frames", stream)
    monkeypatch.setattr(probe.hashlib, "sha256", hashed)
    result = probe.run("test-scanner", 2, 4, 0.001)
    assert result["summary"]["distinct_payloads"] == 1
    assert result["summary"]["hashed_complete_frames"] == 2
    assert result["summary"]["mean_valid_code_fraction"] == 0.5
    assert result["consecutive_frame_changes"][0]["changed_fraction"] == 0


@pytest.mark.parametrize(
    "frames,buffers,poll", [(1, 4, 0.001), (257, 4, 0.001), (16, 1, 0.001), (16, 4, 0)]
)
def test_acquisition_limits_refuse_before_device_access(timing, monkeypatch, frames, buffers, poll):
    probe = importlib.import_module("scanner_fps_probe")
    monkeypatch.setattr(
        probe.v4l2_stream,
        "wait_for_capture_access",
        lambda **k: pytest.fail("device access escaped bounds"),
    )
    with pytest.raises(ValueError):
        probe.run("test-scanner", frames, buffers, poll)


@pytest.mark.parametrize("frames", [2, 319, 321, 1201, 2401])
def test_motion_acquisition_budget_is_explicit_and_closed(timing, monkeypatch, frames):
    probe = importlib.import_module("scanner_fps_probe")
    monkeypatch.setattr(
        probe.v4l2_stream, "wait_for_capture_access", lambda **k: pytest.fail("hardware escaped")
    )
    with pytest.raises(ValueError):
        probe.run("test-scanner", frames, 8, 0.001, motion=True)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("frames,duration", [(320, 180), (640, 180), (1200, 180), (2400, 360)])
def test_motion_preserves_payloads_and_monotonic_times_after_stop_even_on_failure(
    timing, monkeypatch, tmp_path, failure, frames, duration
):
    probe = importlib.import_module("scanner_fps_probe")
    closed = []
    raw = struct.pack("<4H", 0, 42, 1234, 65535)
    records = [
        probe.FrameRecord(i, i, 0, 8, 10 + i * 0.2, None if i == 0 else 200, True) for i in range(2)
    ]
    real_hash = probe.hashlib.sha256

    def hashed(value):
        assert closed
        return real_hash(value)

    def stream(**kwargs):
        assert kwargs["frames"] == frames and kwargs["duration_s"] == duration
        for record in records:
            kwargs["on_frame"](raw, record)
        closed.append(True)
        if failure:
            raise RuntimeError("synthetic stream failure")
        return SimpleNamespace(records=records, observed_fps=5.0, as_dict=lambda: {"delivered": 2})

    monkeypatch.setattr(
        probe.v4l2_stream, "wait_for_capture_access", lambda **k: SimpleNamespace(ready=True)
    )
    monkeypatch.setattr(probe.v4l2_stream, "stream_frames", stream)
    monkeypatch.setattr(probe.hashlib, "sha256", hashed)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic stream"):
            probe.run("test", frames, 8, 0.001, motion=True, raw_output=tmp_path / "raw")
    else:
        probe.run("test", frames, 8, 0.001, motion=True, raw_output=tmp_path / "raw")
    metadata = json.loads((tmp_path / "raw/capture-progress.json").read_text())
    assert bool(metadata["capture_error"]) is failure
    assert [r["monotonic_s"] for r in metadata["frames"]] == [10.0, 10.2]
    for record in metadata["frames"]:
        assert (tmp_path / "raw" / record["path"]).read_bytes() == raw
        assert record["sha256"] == real_hash(raw).hexdigest()
