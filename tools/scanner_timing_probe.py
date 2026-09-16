"""Opt-in A/B/A test of firmware-verified, volatile MINI timing controls.

Only this unit's recovered frame-time (0x910), exposure (0x911), and known
getters are used. No firmware/file upload, calibration, reset, or persistent
settings writes. Settings are restored under a termination shield and verified
from fresh DSP log replies. Hardware transfer success alone is not acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any

from scanner_fps_probe import run as capture
from scanner_internal_probe import ObservationOperation

from scanner3d.config import resolve_serial
from scanner3d.control import USBDEVFS_CONTROL
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import resolve_capture_node
from scanner3d.v4l2_stream import wait_for_capture_access

UVC_QUERY = 0xC0107521
GETTERS = {"frame_time": 0x9B0, "exposure_time": 0x9B1, "auto_expo_mode": 0x9B2}
SETTERS = {"frame_time": 0x910, "exposure_time": 0x911}
# Millisecond steps interpolate within the recovered 7--20 ms operating range.
# They are candidates, not newly claimed firmware presets. Each experiment
# requires fresh device-side acceptance/readback and baseline restoration.
TIMINGS = tuple(range(7000, 20001, 1000))


def xu(fd: int, selector: int, query: int, payload: bytes) -> bytes:
    if (selector, query) not in ((1, 1), (2, 0x81), (7, 1)) or len(payload) != 60:
        raise ValueError("Outside fixed diagnostic XU shapes")
    buffer = ctypes.create_string_buffer(60)
    buffer.raw = payload
    packed = bytearray(struct.pack("@BBBHP", 4, selector, query, 60, ctypes.addressof(buffer)))
    fcntl.ioctl(fd, UVC_QUERY, packed, True)
    return buffer.raw


def uvcioctl(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
    # Adapt the bounded observation operation; no USBFS ioctl or driver detach.
    kind, query, value, index, size, timeout, address = struct.unpack("@BBHHHIP", packed)
    if (
        request != USBDEVFS_CONTROL
        or not mutate
        or index != 0x400
        or size != 60
        or timeout != 1000
        or (kind, query, value) not in ((0x21, 1, 0x100), (0xA1, 0x81, 0x200), (0x21, 1, 0x700))
    ):
        raise ValueError("Unexpected observation transfer")
    result = xu(fd, value >> 8, query, ctypes.string_at(address, 60))
    ctypes.memmove(address, result, 60)
    return 60


def kernel_log(fd: int) -> str:
    observation = ObservationOperation(uvcioctl, "kernel")
    try:
        observation.perform(fd, [])
        return observation.blobs["kernel"].decode("utf-8", errors="replace")
    finally:
        with defer_termination():
            observation.protocol_cleanup(fd)


def temperatures(fd: int) -> dict[str, float]:
    observation = ObservationOperation(uvcioctl, "inventory")
    result = {}
    try:
        for zone, name in [(0, "cpu_c"), (1, "npu_c")]:
            raw = observation.read_file(fd, f"/sys/class/thermal/thermal_zone{zone}/temp", 512)
            value = int(raw.strip()) / 1000
            if not -10 <= value <= 130:
                raise RuntimeError("Implausible or unavailable thermal sensor")
            result[name] = value
        return result
    finally:
        with defer_termination():
            observation.protocol_cleanup(fd)


def command(fd: int, name: str, value: int | None = None) -> None:
    if value is None:
        command_id = GETTERS[name]
        value = 0
    else:
        command_id = SETTERS[name]
        if value not in TIMINGS:
            raise ValueError("Only bounded firmware timing candidates may be tested")
    payload = f"echo s 0x{command_id:x} {value} >/dev/rk_preisp".encode()
    xu(fd, 7, 1, payload.ljust(60, b"\x00"))


def state(fd: int) -> tuple[dict[str, int], str]:
    before = kernel_log(fd)
    stamps = re.findall(r"\[\s*(\d+\.\d+)\]", before)
    cutoff = max(map(float, stamps), default=-1)
    for name in GETTERS:
        command(fd, name)
    after = kernel_log(fd)
    values = {}
    for name in GETTERS:
        matches = re.findall(rf"\[\s*(\d+\.\d+)\][^\n]*id_msg_cam_get_{name}\s+\[(\d+)\]", after)
        fresh = [int(value) for stamp, value in matches if float(stamp) > cutoff]
        if not fresh:
            raise RuntimeError(f"No fresh device-side reply for {name}")
        values[name] = fresh[-1]
    return values, after


def set_timing(fd: int, frame_time: int, exposure_time: int) -> None:
    if frame_time not in TIMINGS or exposure_time not in TIMINGS or exposure_time > frame_time:
        raise ValueError("Invalid timing pair; no command issued")
    # Increase the frame window before a longer exposure; decrease exposure
    # before shortening the window. Baseline is 20000/20000 on this exact unit.
    current_frame = state(fd)[0]["frame_time"] if frame_time != 20000 else 20000
    if frame_time >= current_frame:
        command(fd, "frame_time", frame_time)
        command(fd, "exposure_time", exposure_time)
    else:
        command(fd, "exposure_time", exposure_time)
        command(fd, "frame_time", frame_time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=int, nargs="+", default=[20000, 15000, 20000])
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--save-raw", action="store_true", help="Save bounded raw sets after capture"
    )
    parser.add_argument(
        "--quality", action="store_true", help="Replay tracking after each raw capture"
    )
    args = parser.parse_args()
    if not args.execute or any(t not in TIMINGS for t in args.timings):
        parser.error("Explicit --execute and recovered presets required")
    if not 16 <= args.frames <= 256 or len(args.timings) > 8:
        parser.error("Unbounded test request")
    args.output.mkdir(exist_ok=False)
    serial = resolve_serial()
    node, _ = resolve_capture_node(serial)
    fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        original, log = state(fd)
        initial_temperatures = temperatures(fd)
        (args.output / "original-kernel.txt").write_text(log)
        print(json.dumps({"original": original, "temperatures": initial_temperatures}), flush=True)
        if original != {"frame_time": 20000, "exposure_time": 20000, "auto_expo_mode": 0}:
            raise RuntimeError("Unexpected baseline; refusing settings writes")
    except BaseException:
        os.close(fd)
        raise
    # Test policy, not a manufacturer safe-temperature rating. This unit's
    # observed CPU passive trip points are 75/85 C; critical is 115 C.
    cutoff_c = 80.0
    report: dict[str, Any] = {
        "serial": serial,
        "original": original,
        "runs": [],
        "temperature_samples": [],
        "initial_temperatures": initial_temperatures,
        "test_temperature_cutoff_c": cutoff_c,
        "temperature_scope": "RV1109 CPU/NPU sensors, not projector or case temperature",
        "status": "running",
    }

    def monitor(capture_fd: int) -> None:
        values = temperatures(capture_fd)
        report["temperature_samples"].append({"monotonic_s": time.monotonic(), **values})
        print(json.dumps({"temperatures": values}), flush=True)
        if max(values.values()) >= cutoff_c:
            raise RuntimeError(f"Thermal test cutoff reached: {values}; stopping capture")

    try:
        with cleanup_on_termination():
            for index, timing in enumerate(args.timings):
                monitor(fd)
                set_timing(fd, timing, timing)
                verified, log = state(fd)
                (args.output / f"{index}-configured-kernel.txt").write_text(log)
                print(json.dumps({"requested": timing, "verified": verified}), flush=True)
                if verified != {**original, "frame_time": timing, "exposure_time": timing}:
                    raise RuntimeError("Timing write did not earn device-side verification")
                os.close(fd)
                fd = -1
                result = capture(
                    serial,
                    args.frames,
                    4,
                    0.001,
                    quality=args.quality,
                    monitor=monitor,
                    **(
                        {"raw_output": args.output / f"{index}-{timing}-raw"}
                        if args.save_raw
                        else {}
                    ),
                )
                if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
                    raise RuntimeError("Scanner unavailable after capture")
                node, _ = resolve_capture_node(serial)
                fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
                after, log = state(fd)
                (args.output / f"{index}-after-kernel.txt").write_text(log)
                result["timing_before_capture"] = verified
                result["timing_after_capture"] = after
                report["runs"].append(result)
                print(
                    json.dumps({"requested": timing, "after": after, "summary": result["summary"]}),
                    flush=True,
                )
                if args.quality:
                    print(
                        json.dumps(
                            {
                                "quality": {
                                    k: v
                                    for k, v in result["quality"].items()
                                    if k not in ("preparation", "decisions")
                                }
                            }
                        ),
                        flush=True,
                    )
            report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                # Stream closure/settings changes can re-enumerate this unit.
                # A positive descriptor may still refer to a departed node;
                # always resolve the serial again for restoration.
                if fd >= 0:
                    os.close(fd)
                    fd = -1
                if fd < 0:
                    if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
                        raise RuntimeError("Cannot restore: scanner unavailable")
                    node, _ = resolve_capture_node(serial)
                    fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
                set_timing(fd, original["frame_time"], original["exposure_time"])
                restored, log = state(fd)
                (args.output / "restored-kernel.txt").write_text(log)
                report["restored"] = restored
                report["final_temperatures"] = temperatures(fd)
                print(json.dumps({"restored": restored}), flush=True)
                if restored != original:
                    raise RuntimeError("Firmware timing restoration did not verify")
                report["restoration_verified"] = True
            except BaseException as error:
                report["restoration_verified"] = False
                report["restoration_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if fd >= 0:
                    os.close(fd)
                with (args.output / "report.json").open("x") as stream:
                    json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
