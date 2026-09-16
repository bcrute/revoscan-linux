"""No hardware: validate the probe's standard ioctl packing and restoration."""

from __future__ import annotations

import errno
import json
import struct
from types import SimpleNamespace

import pytest
from tools import camera_cadence_probe as probe


def test_interval_ioctls_pack_type_and_timeperframe(monkeypatch):
    requests = []

    def ioctl(fd, request, value, mutate):
        assert fd == 7 and mutate
        assert len(value) == 204
        assert struct.unpack_from("=I", value)[0] == 1
        requests.append(request)
        if request == probe.G_PARM:
            struct.pack_into("=I", value, 4, 0x1000)
            struct.pack_into("=II", value, 12, 1, 15)
        else:
            assert request == probe.S_PARM
            assert struct.unpack_from("=II", value, 12) == (1, 10)
        return 0

    monkeypatch.setattr(probe.fcntl, "ioctl", ioctl)
    assert probe.get_interval(7) == (1, 15)
    assert probe.set_interval(7, (1, 10)) == (1, 10)
    assert requests == [probe.G_PARM, probe.S_PARM]


def test_unsupported_timeperframe_is_refused(monkeypatch):
    monkeypatch.setattr(probe.fcntl, "ioctl", lambda *args: 0)
    with pytest.raises(RuntimeError, match="does not support"):
        probe.get_interval(7)


@pytest.mark.parametrize("failed", [False, True])
def test_original_interval_restored_after_success_or_capture_failure(monkeypatch, capsys, failed):
    state = {"interval": (1, 15), "closed": 0}
    monkeypatch.setattr("sys.argv", ["probe", "--rates", "10", "--frames", "2"])
    monkeypatch.setattr(probe, "resolve_serial", lambda _: "test-unit")
    monkeypatch.setattr(
        probe, "resolve_capture_node", lambda _: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(probe, "wait_for_capture_access", lambda **_: SimpleNamespace(ready=True))
    monkeypatch.setattr(probe.os, "open", lambda *args: 7)
    monkeypatch.setattr(probe.os, "close", lambda _: state.update(closed=state["closed"] + 1))

    def ioctl(fd, request, value, mutate):
        if request == probe.G_PARM:
            struct.pack_into("=I", value, 4, 0x1000)
            struct.pack_into("=II", value, 12, *state["interval"])
        elif request == probe.S_PARM:
            state["interval"] = struct.unpack_from("=II", value, 12)
        elif request == probe.ENUM_INTERVALS:
            if struct.unpack_from("=I", value)[0] > 0:
                raise OSError(errno.EINVAL, "done")
            struct.pack_into("=III", value, 16, 1, 1, 10)
        return 0

    monkeypatch.setattr(probe.fcntl, "ioctl", ioctl)

    def stream(**kwargs):
        probe.fcntl.ioctl(7, probe.VIDIOC_S_FMT, bytearray(204), True)
        assert state["interval"] == (1, 10)
        if failed:
            raise RuntimeError("capture failed")
        return SimpleNamespace(as_dict=lambda: {"delivered": 2})

    monkeypatch.setattr(probe, "stream_frames", stream)
    if failed:
        with pytest.raises(RuntimeError, match="capture failed"):
            probe.main()
    else:
        probe.main()
    assert state["interval"] == (1, 15)
    assert state["closed"] == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["restored"] == [1, 15]
