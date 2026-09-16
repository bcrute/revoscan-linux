"""32 baseline frames for framing/view identification; no setting writes or tracking."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scanner_firmware_inventory import gain
from scanner_fps_probe import run as capture
from scanner_timing_probe import state, temperatures
from scanner_validation_baseline import BASELINE, SERIAL

from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import resolve_capture_node
from scanner3d.v4l2_stream import wait_for_capture_access


def run(output: Path) -> None:
    if resolve_serial() != SERIAL:
        raise RuntimeError("View check is bound to the physical MINI")
    output.mkdir(exist_ok=False, parents=True)
    report: dict[str, Any] = dict(
        started_utc=datetime.now(UTC).isoformat(),
        serial=SERIAL,
        settings_written=False,
        status="running",
        scope="View/framing check only; not a settled quality comparison",
        temperatures=[],
    )
    fd = -1

    def opened() -> int:
        if not wait_for_capture_access(serial=SERIAL, timeout_s=10).ready:
            raise RuntimeError("Scanner unavailable")
        node, _ = resolve_capture_node(SERIAL)
        return os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)

    def monitor(active_fd: int) -> None:
        values = temperatures(active_fd)
        report["temperatures"].append(dict(utc=datetime.now(UTC).isoformat(), **values))
        if max(values.values()) >= 80:
            raise RuntimeError("80 C view-check cutoff")

    try:
        with cleanup_on_termination():
            fd = opened()
            report["initial_state"], _ = state(fd)
            report["initial_gain"], _ = gain(fd)
            if report["initial_state"] != BASELINE or report["initial_gain"] != 24:
                raise RuntimeError("Unexpected baseline; no setting writes in this tool")
            monitor(fd)
            os.close(fd)
            fd = -1
            report["capture"] = capture(
                SERIAL, 32, 8, 0.001, monitor=monitor, raw_output=output / "raw"
            )
            if report["capture"]["summary"]["hashed_complete_frames"] < 24:
                raise RuntimeError("Insufficient complete frames for view identification")
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
                report["final_gain"], _ = gain(fd)
                monitor(fd)
                report["baseline_verified"] = (
                    report["final_state"] == BASELINE and report["final_gain"] == 24
                )
                if not report["baseline_verified"]:
                    raise RuntimeError("Baseline changed despite read-only view check")
            except BaseException as error:
                report["status"] = "failed"
                report["final_check_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if fd >= 0:
                    os.close(fd)
                report["ended_utc"] = datetime.now(UTC).isoformat()
                with (output / "report.json").open("x") as stream:
                    json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "capture"}))
    print(json.dumps(report["capture"]["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required; leave object stationary")
    run(args.output)
