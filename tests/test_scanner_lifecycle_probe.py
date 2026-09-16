"""Synthetic lifecycle sequencing and first-fault tracing; no physical USB access."""

import ctypes
import errno
import importlib
import io
import json
import os
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_lifecycle_probe")


@pytest.mark.parametrize("code", [errno.EPROTO, errno.ENODEV])
def test_first_transport_fault_is_preserved_and_blocks_more_uvc(probe, code):
    wire = []

    def ioctl(*args):
        wire.append(args)
        raise OSError(code, "synthetic USB fault")

    output = io.StringIO()
    trace = probe.WireTrace(ioctl, output)
    trace.phase = "final_gain"
    payload = ctypes.create_string_buffer(60)
    request = bytearray(struct.pack("@BBBHP", 4, 1, 1, 60, ctypes.addressof(payload)))
    with pytest.raises(OSError):
        trace(920, probe.UVC_QUERY, request)
    with pytest.raises(RuntimeError, match="No more UVC"):
        trace(920, probe.UVC_QUERY, request)
    assert len(wire) == 1
    assert trace.first_fault["errno"] == code
    assert trace.first_fault["phase"] == "final_gain"
    assert json.loads(output.getvalue())["set_payload_hex"] == "00" * 60


@pytest.mark.parametrize(
    "mode,quiet,motion_timing,motion_frames",
    [
        ("baseline", False, None, 320),
        ("post", False, None, 320),
        ("live", False, None, 320),
        ("inspect", False, None, 320),
        ("settled-post", False, None, 320),
        ("baseline", True, None, 320),
        ("live", True, None, 320),
        ("settled-post", False, 20000, 320),
        ("settled-post", False, 14000, 640),
        ("settled-post", False, 10000, 1200),
        ("settled-post", False, 14000, 2400),
    ],
)
@pytest.mark.parametrize("failure", [False, True])
def test_lifecycle_sequence_and_bounded_restoration(
    probe, monkeypatch, tmp_path, mode, failure, quiet, motion_timing, motion_frames
):

    class Witness:
        def __init__(self, *, hold):
            self.report = {"hold_requested": hold, "topology_stable": True}

        def acquire(self):
            pass

        def require_same(self, phase):
            pass

        def finish(self):
            assert closed

        def close(self):
            pass

    monkeypatch.setattr(probe, "UsbIdleWitness", Witness)
    monkeypatch.setattr(
        probe,
        "quiet_stop_observation",
        lambda witness: witness.report.update(settled_for_readback=True),
    )
    monkeypatch.setattr(probe, "resolve_serial", lambda: probe.SERIAL)
    monkeypatch.setattr(probe, "wait_for_capture_access", lambda **kw: SimpleNamespace(ready=True))
    monkeypatch.setattr(
        probe, "resolve_capture_node", lambda serial: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(probe.os, "open", lambda *args: 920)
    closed = []
    monkeypatch.setattr(probe.os, "close", closed.append)
    monkeypatch.setattr(probe.fcntl, "ioctl", lambda *args: 0)
    monkeypatch.setattr(probe, "gain", lambda fd: (24, "synthetic gain log"))
    monkeypatch.setattr(probe, "temperatures", lambda fd: dict(cpu_c=60, npu_c=59))
    monkeypatch.setattr(probe.ObservationOperation, "read_file", lambda *args: b"synthetic memory")
    monkeypatch.setattr(
        probe.ObservationOperation,
        "perform",
        lambda observer, *args: observer.blobs.update({observer.plan: b"synthetic log"}),
    )
    failure_expected = failure and mode != "inspect"
    settings = dict(probe.BASELINE)
    writes = []

    def set_timing(fd, frame, exposure):
        settings.update(frame_time=frame, exposure_time=exposure)
        writes.append((frame, exposure))

    monkeypatch.setattr(probe, "set_timing", set_timing)
    monkeypatch.setattr(probe, "state", lambda fd: (dict(settings), "synthetic timing log"))

    def capture(serial, frames, buffers, poll, **kwargs):
        assert mode != "inspect"
        wanted_frames = motion_frames if motion_timing is not None else 64
        assert (serial, frames, buffers, poll) == (probe.SERIAL, wanted_frames, 8, 0.001)
        assert kwargs.get("motion", False) is (motion_timing is not None)
        for index in range(wanted_frames):
            probe.fcntl.ioctl(920, probe.VIDIOC_DQBUF, b"", True)
            if index in (0, 35):
                kwargs["monitor"](920)
        if failure:
            raise RuntimeError("synthetic capture failure")
        return dict(summary=dict(delivered=wanted_frames, hashed_complete_frames=wanted_frames))

    monkeypatch.setattr(probe, "capture", capture)
    output = tmp_path / "probe"
    if failure_expected:
        with pytest.raises(RuntimeError, match="synthetic capture"):
            probe.run(
                output,
                mode,
                quiet_stop=quiet,
                motion_timing_us=motion_timing,
                motion_frames=motion_frames,
            )
    else:
        probe.run(
            output,
            mode,
            quiet_stop=quiet,
            motion_timing_us=motion_timing,
            motion_frames=motion_frames,
        )
    result = json.loads((output / "report.json").read_text())
    assert result["status"] == ("failed" if failure_expected else "completed")
    assert settings == probe.BASELINE
    requested = motion_timing or 14000
    assert writes == (
        [] if mode in ("baseline", "inspect") else [(requested, requested), (20000, 20000)]
    )
    if not failure_expected and not (quiet and mode == "baseline"):
        assert result["restoration_verified"]
    if mode == "inspect":
        assert "capture" not in result
    steps = {s["name"]: s for s in result["steps"]}
    if quiet:
        assert "after_stop_timing" not in steps
        assert "after_stop_gain" not in steps
        assert "final_memory" not in steps
    if mode == "live":
        assert steps["restore_live"]["dequeued"] == 36
    if mode in ("post", "settled-post") and not failure:
        assert steps["restore_post_stop"]["dequeued"] == (
            motion_frames if motion_timing is not None else 64
        )
    if mode == "settled-post":
        assert result["usb_before_readback"]["settled_for_readback"]
    assert closed


def test_lifecycle_refuses_unknown_mode_before_hardware_or_files(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "resolve_serial", lambda: pytest.fail("hardware escaped"))
    with pytest.raises(ValueError, match="Closed modes"):
        probe.run(tmp_path / "out", "reset")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "mode,timing,frames",
    [
        ("post", 14000, 320),
        ("live", 10000, 320),
        ("inspect", 20000, 320),
        ("settled-post", 7000, 320),
        ("settled-post", 14000, 321),
        ("settled-post", 14000, 2401),
        ("settled-post", None, 640),
    ],
)
def test_motion_requires_closed_preset_budget_and_quiet_boundary(
    probe, monkeypatch, tmp_path, mode, timing, frames
):
    monkeypatch.setattr(probe, "resolve_serial", lambda: pytest.fail("hardware escaped"))
    with pytest.raises(ValueError, match="Closed modes"):
        probe.run(tmp_path / "out", mode, motion_timing_us=timing, motion_frames=frames)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["post", "inspect"])
def test_quiet_stop_refuses_modes_needing_post_stop_restore(probe, monkeypatch, tmp_path, mode):
    monkeypatch.setattr(probe, "resolve_serial", lambda: pytest.fail("hardware escaped"))
    with pytest.raises(ValueError, match="Closed modes"):
        probe.run(tmp_path / "out", mode, quiet_stop=True)
    assert not (tmp_path / "out").exists()


def test_streamoff_event_has_request_boundary_and_dequeue_count(probe):
    output = io.StringIO()
    trace = probe.WireTrace(lambda *args: 0, output)
    trace.dequeued = 64
    trace(920, probe.VIDIOC_STREAMOFF, b"", True)
    event = json.loads(output.getvalue())
    assert event["kind"] == "v4l2"
    assert event["request"] == probe.VIDIOC_STREAMOFF
    assert event["dequeued"] == 64
    assert event["duration_s"] >= 0
    assert trace.count == 0


@pytest.mark.parametrize("code_name", ["VIDIOC_DQBUF", "VIDIOC_STREAMOFF"])
def test_v4l2_transport_fault_also_blocks_uvc_but_allows_local_cleanup(probe, code_name):
    calls = []

    def ioctl(fd, code, *args):
        calls.append(code)
        raise OSError(errno.ENODEV, "synthetic detached transport")

    trace = probe.WireTrace(ioctl, io.StringIO())
    with pytest.raises(OSError):
        trace(920, getattr(probe, code_name), b"", True)
    with pytest.raises(RuntimeError, match="No more UVC"):
        trace(920, probe.UVC_QUERY, b"", True)
    assert trace.first_fault["kind"] == "v4l2"
    assert len(calls) == 1
    with pytest.raises(OSError):
        trace(920, probe.VIDIOC_STREAMOFF, b"", True)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "case", ["stable", "returned", "absent", "flapping", "wrong_serial", "short"]
)
def test_quiet_boundary_requires_two_seconds_of_stable_exact_identity(probe, case):
    identity = dict(
        present=True,
        idVendor="2207",
        idProduct="110c",
        serial=probe.SERIAL,
        busnum="1",
        devnum="91",
        dev="189:90",
    )
    rows = []
    for index in range(121):
        snapshot = dict(identity)
        if case == "returned" and index < 40:
            snapshot.update(present=False)
        if case == "absent":
            snapshot.update(present=False)
        if case == "flapping" and index == 100:
            snapshot.update(devnum="92", dev="189:91")
        if case == "wrong_serial":
            snapshot.update(serial="wrong")
        if case != "short" or index > 110:
            rows.append(dict(monotonic_s=index * 0.05, snapshot=snapshot))
    observed = []
    witness = SimpleNamespace(
        report=dict(observations=rows),
        observe=lambda *args: observed.append(args),
    )
    if case in ("stable", "returned"):
        probe.quiet_stop_observation(witness)
        assert witness.report["settled_for_readback"]
    else:
        with pytest.raises(RuntimeError, match="no control retry"):
            probe.quiet_stop_observation(witness)
    assert observed == [("quiet_stop_boundary", 6.0)]


@pytest.fixture
def usb_tree(probe, monkeypatch, tmp_path):
    root = tmp_path / "usb"
    node = root / "1-7"
    (node / "power").mkdir(parents=True)
    values = {
        "idVendor": "2207",
        "idProduct": "110c",
        "serial": probe.SERIAL,
        "busnum": "1",
        "devnum": "89",
        "dev": "189:88",
        "speed": "480",
        "power/control": "auto",
        "power/autosuspend_delay_ms": "2000",
        "power/runtime_status": "suspended",
        "power/runtime_usage": "0",
        "power/runtime_active_time": "1000",
        "power/runtime_suspended_time": "2000",
    }
    for name, value in values.items():
        (node / name).write_text(value)
    monkeypatch.setattr(
        probe, "Path", lambda arg: root if arg == "/sys/bus/usb/devices" else Path(arg)
    )
    monkeypatch.setattr(probe.fcntl, "ioctl", lambda *args: pytest.fail("USB ioctl escaped"))
    return node


def test_usb_lease_exact_read_only_unclaimed_and_released_after_v4l2(probe, monkeypatch, usb_tree):
    opened, closed, phases = [], [], []
    monkeypatch.setattr(probe.os, "open", lambda *args: opened.append(args) or 930)
    monkeypatch.setattr(probe.os, "close", closed.append)
    monkeypatch.setattr(
        probe.os,
        "fstat",
        lambda fd: SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(189, 88)),
    )
    witness = probe.UsbIdleWitness(hold=True)
    witness.acquire()

    def observe(phase, seconds):
        phases.append((phase, seconds, witness.fd))
        witness.record(phase)

    monkeypatch.setattr(witness, "observe", observe)
    witness.finish()
    witness.close()
    assert opened == [("/dev/bus/usb/001/089", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)]
    assert closed == [930]
    assert phases == [("held_after_v4l2_close", 4.0, 930), ("released_idle", 6.0, -1)]
    assert witness.report["topology_stable"]
    assert (usb_tree / "power/control").read_text() == "auto"


@pytest.mark.parametrize("case", ["wrong_serial", "duplicate"])
def test_usb_lease_refuses_ambiguous_or_wrong_identity(probe, monkeypatch, usb_tree, case):
    monkeypatch.setattr(probe.os, "open", lambda *args: pytest.fail("device opened"))
    if case == "wrong_serial":
        (usb_tree / "serial").write_text("not-the-authorized-scanner")
    else:
        duplicate = usb_tree.parent / "1-8"
        duplicate.mkdir()
        for key in ("idVendor", "idProduct", "serial"):
            (duplicate / key).write_text((usb_tree / key).read_text())
    with pytest.raises(RuntimeError, match="One exact"):
        probe.UsbIdleWitness(hold=True)


@pytest.mark.parametrize("failure", ["identity_change", "wrong_rdev", "regular_file", "observe"])
def test_usb_lease_closes_on_failure(probe, monkeypatch, usb_tree, failure):
    closed = []
    monkeypatch.setattr(probe.os, "open", lambda *args: 930)
    monkeypatch.setattr(probe.os, "close", closed.append)

    def fstat(fd):
        if failure == "identity_change":
            (usb_tree / "devnum").write_text("90")
        return SimpleNamespace(
            st_mode=stat.S_IFREG if failure == "regular_file" else stat.S_IFCHR,
            st_rdev=os.makedev(189, 87 if failure == "wrong_rdev" else 88),
        )

    monkeypatch.setattr(probe.os, "fstat", fstat)
    witness = probe.UsbIdleWitness(hold=True)
    if failure == "observe":
        witness.acquire()

        def fail(*args):
            raise RuntimeError("synthetic observer failure")

        monkeypatch.setattr(witness, "observe", fail)
        with pytest.raises(RuntimeError):
            witness.finish()
    else:
        with pytest.raises(RuntimeError):
            witness.acquire()
    assert witness.fd == -1
    assert closed == [930]


def test_reenumeration_is_sticky_even_if_same_serial_returns(probe, monkeypatch, usb_tree):
    witness = probe.UsbIdleWitness(hold=False)
    witness.acquire()
    (usb_tree / "devnum").write_text("90")
    with pytest.raises(RuntimeError, match="no more device"):
        witness.require_same("after_stop")
    (usb_tree / "devnum").write_text("89")
    with pytest.raises(RuntimeError, match="no more device"):
        witness.require_same("later")
    monkeypatch.setattr(witness, "observe", lambda phase, seconds: witness.record(phase))
    witness.finish()
    assert not witness.report["topology_stable"]
    assert witness.report["first_topology_fault"]["phase"] == "after_stop"
