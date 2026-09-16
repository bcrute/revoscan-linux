"""P0: bounded identity/telemetry reads and baseline-only acquisition, no timing writes."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scanner_fps_probe import run as capture
from scanner_internal_probe import ObservationOperation
from scanner_timing_probe import UVC_QUERY, state, temperatures, uvcioctl

from scanner3d.config import resolve_serial
from scanner3d.control import parse_version_payload
from scanner3d.discovery import find_scanners
from scanner3d.ownership import defer_termination
from scanner3d.v4l2_capture import resolve_capture_node
from scanner3d.v4l2_stream import wait_for_capture_access

SERIAL = "A22704522A6F00A64"
BASELINE = dict(frame_time=20000, exposure_time=20000, auto_expo_mode=0)


def version(fd: int) -> dict[str, object]:
    """One known read tuple, through the bound driver rather than USBFS detach."""
    buffer = ctypes.create_string_buffer(60)
    query = bytearray(struct.pack("@BBBHP", 3, 6, 0x81, 60, ctypes.addressof(buffer)))
    fcntl.ioctl(fd, UVC_QUERY, query, True)
    return parse_version_payload(buffer.raw).as_dict()


def run(output: Path) -> None:
    serial = resolve_serial()
    if serial != SERIAL:
        raise RuntimeError("This experiment is bound to one physical MINI")
    output.mkdir(exist_ok=False, parents=True)
    report: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "serial": serial,
        "host_loadavg": os.getloadavg(),
        "usb": [d.as_dict() for d in find_scanners() if d.serial == serial],
        "temperatures": [],
        "settings_written": False,
    }
    fd = -1

    def opened() -> int:
        if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
            raise RuntimeError("Scanner unavailable")
        node, _ = resolve_capture_node(serial)
        return os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)

    def monitor(active_fd: int) -> None:
        values = temperatures(active_fd)
        report["temperatures"].append({"utc": datetime.now(UTC).isoformat(), **values})
        print("THERMAL", values, flush=True)
        if max(values.values()) >= 80:
            raise RuntimeError("80 C cutoff reached")

    try:
        fd = opened()
        report["initial_state"], log = state(fd)
        if report["initial_state"] != BASELINE:
            raise RuntimeError("Unexpected timing baseline; no writes authorized by this probe")
        with (output / "initial-kernel.txt").open("x") as stream:
            stream.write(log)
        monitor(fd)
        try:
            report["version"] = version(fd)
        except OSError as error:
            report["version_read_error"] = f"{type(error).__name__}: {error}"
        observation = ObservationOperation(uvcioctl, "controller")
        try:
            observation.perform(fd, [])
            controller = observation.blobs["controller"]
            report["controller_sha256"] = hashlib.sha256(controller).hexdigest()
            with (output / "controller.bin").open("xb") as binary_stream:
                binary_stream.write(controller)
        finally:
            with defer_termination():
                observation.protocol_cleanup(fd)
        os.close(fd)
        fd = -1
        report["capture"] = capture(
            serial, 128, 8, 0.001, monitor=monitor, raw_output=output / "raw"
        )
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                if fd >= 0:
                    os.close(fd)
                    fd = -1
                fd = opened()
                report["final_state"], _ = state(fd)
                monitor(fd)
                report["baseline_verified"] = report["final_state"] == BASELINE
                if not report["baseline_verified"]:
                    raise RuntimeError("Final baseline differs despite no settings writes")
            except BaseException as error:
                report["baseline_verified"] = False
                report["final_check_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if fd >= 0:
                    os.close(fd)
                report["ended_utc"] = datetime.now(UTC).isoformat()
                with (output / "report.json").open("x") as report_stream:
                    json.dump(report, report_stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("capture", "usb")}), flush=True)
    print("CAPTURE", json.dumps(report["capture"]["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required")
    run(args.output)
