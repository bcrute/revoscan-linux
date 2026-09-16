"""Bounded scanner cadence experiment; restores the original frame interval.

Run from the repository with `.venv/bin/python tools/camera_cadence_probe.py`.
Changes only the standard V4L2 time-per-frame request, after the application's
normal format selection. Does not change exposure, vendor properties or files
in a capture. Raw frames are not saved. All observations print as JSON lines.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import struct
from typing import Any
from unittest.mock import patch

from scanner3d.config import resolve_serial
from scanner3d.v4l2_capture import (
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VIDIOC_S_FMT,
    _ioc,
    resolve_capture_node,
)
from scanner3d.v4l2_stream import (
    MAX_DURATION_S,
    ensure_request,
    stream_frames,
    wait_for_capture_access,
)

G_PARM = _ioc(3, 21, 204)
S_PARM = _ioc(3, 22, 204)
ENUM_INTERVALS = _ioc(3, 75, 52)


def get_interval(fd: int) -> tuple[int, int]:
    value = bytearray(204)
    struct.pack_into("=I", value, 0, 1)
    fcntl.ioctl(fd, G_PARM, value, True)
    if not struct.unpack_from("=I", value, 4)[0] & 0x1000:
        raise RuntimeError("Driver does not support changing time-per-frame")
    return struct.unpack_from("=II", value, 12)


def set_interval(fd: int, value: tuple[int, int]) -> tuple[int, int]:
    packed = bytearray(204)
    struct.pack_into("=I", packed, 0, 1)
    struct.pack_into("=II", packed, 12, *value)
    fcntl.ioctl(fd, S_PARM, packed, True)
    return struct.unpack_from("=II", packed, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--rates", type=int, nargs="+", default=[15, 10, 5, 15])
    args = parser.parse_args()
    ensure_request(args.frames, 1, MAX_DURATION_S)
    serial = resolve_serial(None)
    node, _ = resolve_capture_node(serial)
    fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    intervals = []
    try:
        original = get_interval(fd)
        for index in range(32):
            packed = bytearray(52)
            struct.pack_into(
                "=4I",
                packed,
                0,
                index,
                int.from_bytes(FOURCC.encode(), "little"),
                FRAME_WIDTH,
                FRAME_HEIGHT,
            )
            try:
                fcntl.ioctl(fd, ENUM_INTERVALS, packed, True)
            except OSError as exc:
                if exc.errno == errno.EINVAL:
                    break
                raise
            if struct.unpack_from("=I", packed, 16)[0] != 1:
                raise RuntimeError("Only advertised discrete intervals may be tested")
            intervals.append(struct.unpack_from("=II", packed, 20))
    finally:
        os.close(fd)
    print(json.dumps({"serial": serial, "original": original, "advertised": intervals}), flush=True)
    if any((1, rate) not in intervals for rate in args.rates):
        raise ValueError("Requested rate is not advertised; refusing")
    real_ioctl = fcntl.ioctl
    try:
        for rate in args.rates:
            if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
                raise RuntimeError("Scanner did not become ready")
            observations: list[dict[str, object]] = []

            def configured_ioctl(
                fd: int,
                request: int,
                value: Any = 0,
                mutate: bool = True,
                *,
                _rate: int = rate,
                _observations: list[dict[str, object]] = observations,
            ) -> Any:
                result = real_ioctl(fd, request, value, mutate)
                if request == VIDIOC_S_FMT:
                    negotiated = set_interval(fd, (1, _rate))
                    _observations.append({"requested": [1, _rate], "negotiated": negotiated})
                return result

            with patch.object(fcntl, "ioctl", configured_ioctl):
                result = stream_frames(serial=serial, frames=args.frames, decimate=1)
            summary = {
                k: v for k, v in result.as_dict().items() if k not in ("records", "limitations")
            }
            print(json.dumps({"configuration": observations, "stream": summary}), flush=True)
    finally:
        if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
            raise RuntimeError("Scanner unavailable for interval restoration")
        node, _ = resolve_capture_node(serial)
        fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            current = get_interval(fd)
            if current != original:
                set_interval(fd, original)
            restored = get_interval(fd)
            print(json.dumps({"restored": restored, "original": original}), flush=True)
            if restored != original:
                raise RuntimeError("Camera did not restore its original time-per-frame")
        finally:
            os.close(fd)


if __name__ == "__main__":
    main()
