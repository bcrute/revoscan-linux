"""Bounded actuator-order readback test; no acquisition or persistent writes."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scanner_firmware_inventory import gain
from scanner_gain_probe import set_gain
from scanner_timing_probe import set_timing, state, temperatures
from scanner_validation_baseline import BASELINE, SERIAL

from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import resolve_capture_node
from scanner3d.v4l2_stream import wait_for_capture_access


def run(output: Path) -> None:
    if output.exists():
        raise ValueError("Refusing overwrite")
    if resolve_serial() != SERIAL or not wait_for_capture_access(serial=SERIAL, timeout_s=10).ready:
        raise RuntimeError("Bound MINI unavailable")
    node, _ = resolve_capture_node(SERIAL)
    fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    report: dict[str, Any] = {"status": "running", "observations": []}
    owed = False

    def observe(label: str) -> tuple[dict[str, int], int]:
        timing, _ = state(fd)
        value, _ = gain(fd)
        thermal = temperatures(fd)
        if max(thermal.values()) >= 80:
            raise RuntimeError("80 C cutoff")
        report["observations"].append(
            dict(label=label, timing=timing, gain=value, temperatures=thermal)
        )
        print(label, timing, value, flush=True)
        return timing, value

    try:
        with cleanup_on_termination():
            if observe("initial") != (BASELINE, 24):
                raise RuntimeError("Unexpected baseline; no writes")
            owed = True
            set_gain(fd, 16)
            set_timing(fd, 20000, 20000)
            observe("gain16 then timing20")
            set_gain(fd, 32)
            observe("gain32 before timing12")
            set_timing(fd, 12000, 12000)
            observe("gain32 then timing12")
            set_gain(fd, 32)
            observe("timing12 then gain32")
            set_timing(fd, 20000, 20000)
            observe("timing20 after gain32")
            report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                if owed:
                    set_timing(fd, 20000, 20000)
                    set_gain(fd, 24)
                    report["restoration_verified"] = observe("restored") == (BASELINE, 24)
                    if not report["restoration_verified"]:
                        raise RuntimeError("Restoration failed")
            finally:
                os.close(fd)
                with output.open("x") as stream:
                    json.dump(report, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required")
    run(args.output)
