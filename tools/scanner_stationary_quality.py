"""Bounded, thermally monitored exposure A/B/A in one uninterrupted stream.

The object MUST stay stationary. Known volatile controls only, no firmware
flash/reset. Discard a full second and 12 frames after every timing transition;
retain 40 full-resolution frames per setting for matched-pixel analysis.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner_firmware_inventory import gain
from scanner_gain_probe import GAINS, set_gain
from scanner_timing_probe import TIMINGS, set_timing, state, temperatures

from scanner3d import v4l2_stream
from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import VIDIOC_DQBUF, resolve_capture_node


def run(
    output: Path,
    timings: tuple[int, ...] = (20000, 10000, 9000, 8000, 7000, 20000),
    exposures: tuple[int, ...] | None = None,
    gains: tuple[int, ...] | None = None,
) -> None:
    exposures = timings if exposures is None else exposures
    if (
        not 3 <= len(timings) <= 10
        or len(exposures) != len(timings)
        or timings[0] != 20000
        or timings[-1] != 20000
        or exposures[0] != 20000
        or exposures[-1] != 20000
        or any(
            f not in TIMINGS or e not in TIMINGS or e > f
            for f, e in zip(timings, exposures, strict=True)
        )
        or (
            gains is not None
            and (
                len(gains) != len(timings)
                or gains[0] != 24
                or gains[-1] != 24
                or any(g not in GAINS for g in gains)
            )
        )
    ):
        raise ValueError("Bounded timing pairs and bracketing 20/20 ms baselines required")
    output.mkdir(exist_ok=False)
    serial = resolve_serial()
    node, _ = resolve_capture_node(serial)
    control_fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        original, _ = state(control_fd)
        if original != dict(frame_time=20000, exposure_time=20000, auto_expo_mode=0):
            raise RuntimeError("Unexpected baseline; no settings changed")
        if gains is not None and gain(control_fd)[0] != 24:
            raise RuntimeError("Unexpected gain baseline; no settings changed")
    except BaseException:
        os.close(control_fd)
        raise
    payloads: list[list[bytes]] = [[] for _ in timings]
    stamps: list[list[float]] = [[] for _ in timings]
    report: dict[str, Any] = {
        "serial": serial,
        "original": original,
        "timings": timings,
        "exposures": exposures,
        "gains": gains,
        "settled_frames_per_setting": 40,
        "operator_stationary_confirmation": True,
        "temperatures": [],
        "transitions": [],
        "status": "running",
    }
    active_fd = -1
    stage = -1
    discard = 0
    settled_at = 0.0
    next_thermal = 0.0
    complete = False
    restore_owed = False
    original_ioctl = fcntl.ioctl

    def observe_ioctl(fd: int, request: int, value: Any = 0, mutate: bool = True) -> Any:
        nonlocal active_fd
        result = original_ioctl(fd, request, value, mutate)
        if request == VIDIOC_DQBUF:
            active_fd = fd
        return result

    def frame(raw: bytes, record: object) -> None:
        nonlocal stage, discard, settled_at, next_thermal, complete, restore_owed
        now = time.monotonic()
        if now >= next_thermal:
            thermal = temperatures(active_fd)
            report["temperatures"].append({"time": now, **thermal})
            print("THERMAL", thermal, flush=True)
            if max(thermal.values()) >= 80:
                raise RuntimeError("80 C test cutoff reached")
            next_thermal = now + 5
        if stage < 0 or len(payloads[stage]) == 40:
            if stage >= 0 and gains is not None:
                ending_gain, _ = gain(active_fd)
                report["transitions"][stage]["ending_gain"] = ending_gain
                if ending_gain != gains[stage]:
                    raise RuntimeError("Gain drifted during settled acquisition")
            stage += 1
            if stage == len(timings):
                complete = True
                return
            timing = timings[stage]
            restore_owed = True
            exposure = exposures[stage]
            if gains is not None:
                set_gain(active_fd, gains[stage])
                measured_gain, _ = gain(active_fd)
                if measured_gain != gains[stage]:
                    raise RuntimeError("Gain readback disagrees")
            set_timing(active_fd, timing, exposure)
            verified, _ = state(active_fd)
            if gains is not None and gain(active_fd)[0] != gains[stage]:
                raise RuntimeError("Gain changed after timing writes")
            if verified != {**original, "frame_time": timing, "exposure_time": exposure}:
                raise RuntimeError("Timing readback disagrees")
            restore_owed = (timing, exposure) != (20000, 20000) or (
                gains is not None and gains[stage] != 24
            )
            report["transitions"].append(
                {
                    "timing": timing,
                    "exposure": exposure,
                    "verified": verified,
                    "time": now,
                    "gain": gains[stage] if gains is not None else None,
                }
            )
            print("SETTING", timing, verified, flush=True)
            discard = 12
            settled_at = time.monotonic() + 1.0
            return
        if discard > 0:
            discard -= 1
            return
        if now < settled_at:
            return
        payloads[stage].append(raw)
        stamps[stage].append(now)

    try:
        with cleanup_on_termination(), patch.object(fcntl, "ioctl", observe_ioctl):
            result = v4l2_stream.stream_frames(
                serial=serial,
                frames=800,
                duration_s=150,
                buffers=8,
                on_frame=frame,
                should_stop=lambda: complete,
            )
            report["stream"] = result.as_dict()
            report["deliveries"] = [r.as_dict() for r in result.records]
            report["status"] = "completed" if complete else "incomplete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                os.close(control_fd)
                control_fd = -1
                if restore_owed:
                    if not v4l2_stream.wait_for_capture_access(serial=serial, timeout_s=10).ready:
                        raise RuntimeError("Restoration unavailable: scanner absent")
                    node, _ = resolve_capture_node(serial)
                    control_fd = os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
                    if gains is not None:
                        set_gain(control_fd, 24)
                        if gain(control_fd)[0] != 24:
                            raise RuntimeError("Gain restoration did not verify")
                    set_timing(control_fd, 20000, 20000)
                    restored, _ = state(control_fd)
                    if restored != original:
                        raise RuntimeError("Restoration did not verify")
                report["restoration_verified"] = True
            except BaseException as error:
                report["restoration_verified"] = False
                report["restoration_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if control_fd >= 0:
                    os.close(control_fd)
                report["sample_times"] = stamps
                for index, frames in enumerate(payloads):
                    directory = output / f"{index}-{timings[index]}"
                    directory.mkdir()
                    for number, raw in enumerate(frames):
                        with (directory / f"frame_{number:04d}.raw").open("xb") as stream:
                            stream.write(raw)
                with (output / "report.json").open("x") as report_stream:
                    json.dump(report, report_stream, indent=2)
                print(
                    "RESULT",
                    report["status"],
                    "restoration",
                    report["restoration_verified"],
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--timings", type=int, nargs="+", default=[20000, 10000, 9000, 8000, 7000, 20000]
    )
    parser.add_argument("--exposures", type=int, nargs="+")
    parser.add_argument("--gains", type=int, nargs="+")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required; object must be stationary")
    run(
        args.output,
        tuple(args.timings),
        None if args.exposures is None else tuple(args.exposures),
        None if args.gains is None else tuple(args.gains),
    )
