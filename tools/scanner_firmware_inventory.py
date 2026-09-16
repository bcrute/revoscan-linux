"""Closed, bounded firmware/configuration reads through the attached V4L2 driver.

Never uploads or executes a firmware package; does not read credentials or
write persistent device files. Missing files may be returned as empty by this
protocol, so an empty reply is not proof of an empty existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scanner_internal_probe import ObservationOperation
from scanner_timing_probe import kernel_log, state, temperatures, uvcioctl, xu
from scanner_validation_baseline import BASELINE, SERIAL

from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import resolve_capture_node
from scanner3d.v4l2_stream import wait_for_capture_access

FILES = {
    "firmware-data": ("/data/firmware_version", 1024),
    "firmware-userdata": ("/userdata/firmware_version", 1024),
    "model": ("/factory/model.ini", 8192),
    "cif-params": ("/data/cif_params.conf", 8192),
    "mtd": ("/proc/mtd", 4096),
    "partitions": ("/proc/partitions", 4096),
    "mounts": ("/proc/mounts", 8192),
    "cmdline": ("/proc/cmdline", 4096),
    "depth-intrinsics": ("/data/camparam/Pl.bin", 128),
}


def gain(fd: int) -> tuple[int, str]:
    before = kernel_log(fd)
    cutoff = max(map(float, re.findall(r"\[\s*(\d+\.\d+)\]", before)), default=-1)
    xu(fd, 7, 1, b"echo s 0x9a3 0 >/dev/rk_preisp".ljust(60, b"\x00"))
    after = kernel_log(fd)
    matches = re.findall(r"\[\s*(\d+\.\d+)\][^\n]*id_msg_cam_get_gain_reg\s+\[(\d+)\]", after)
    fresh = [int(value) for stamp, value in matches if float(stamp) > cutoff]
    if not fresh:
        raise RuntimeError("No fresh, named gain getter reply")
    return fresh[-1], after


def run(output: Path) -> None:
    serial = resolve_serial()
    if serial != SERIAL or not wait_for_capture_access(serial=serial, timeout_s=10).ready:
        raise RuntimeError("Bound physical MINI unavailable")
    output.mkdir(exist_ok=False, parents=True)
    node, _ = resolve_capture_node(serial)
    fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    observation = ObservationOperation(uvcioctl, "inventory")
    report: dict[str, Any] = dict(
        started_utc=datetime.now(UTC).isoformat(),
        serial=serial,
        status="running",
        files={},
        settings_written=False,
    )
    try:
        with cleanup_on_termination():
            report["initial_state"], _ = state(fd)
            if report["initial_state"] != BASELINE:
                raise RuntimeError("Unexpected state; no setting writes in this tool")
            report["initial_temperatures"] = temperatures(fd)
            if max(report["initial_temperatures"].values()) >= 80:
                raise RuntimeError("80 C cutoff")
            report["gain_register"], log = gain(fd)
            with (output / "gain-getter-kernel.txt").open("x") as stream:
                stream.write(log)
            for name, (path, cap) in FILES.items():
                blob = observation.read_file(fd, path, cap)
                with (output / f"{name}.bin").open("xb") as binary:
                    binary.write(blob)
                report["files"][name] = dict(
                    device_path=path, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()
                )
                print("READ", name, len(blob), flush=True)
            report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                observation.protocol_cleanup(fd)
                report["final_state"], _ = state(fd)
                report["final_temperatures"] = temperatures(fd)
                report["baseline_verified"] = report["final_state"] == BASELINE
                if not report["baseline_verified"]:
                    raise RuntimeError("Baseline changed despite read-only inventory")
            finally:
                os.close(fd)
                report["ended_utc"] = datetime.now(UTC).isoformat()
                with (output / "report.json").open("x") as stream:
                    json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required; no other scanner owner permitted")
    run(args.output)
