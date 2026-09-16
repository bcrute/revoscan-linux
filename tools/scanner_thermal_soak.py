"""Staged acquisition-only soak at 7, 14 or baseline 20 ms.

No tracking/rendering, gain changes, clock changes or persistent firmware
writes. CPU/NPU temperatures are read every five seconds. Stop at 80 C (below
the observed second passive trip at 85 C), on missing telemetry, or at the
explicit time limit. Longer stages require a successful shorter-stage report.
Preserve the original 20 ms timing under signal shielding. Optional excerpts
retain ten complete raw frames per minute, never a full unbounded recording.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner_firmware_inventory import gain
from scanner_lifecycle_probe import UsbIdleWitness, WireTrace, quiet_stop_observation
from scanner_timing_probe import set_timing, state, temperatures
from scanner_validation_baseline import BASELINE, SERIAL

from scanner3d import v4l2_stream
from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import VIDIOC_DQBUF, resolve_capture_node
from scanner3d.v4l2_stream import FrameRecord, stream_frames, wait_for_capture_access


def validate_stage(duration: int, previous: dict[str, Any] | None, timing_us: int = 7000) -> None:
    if timing_us not in (7000, 14000, 20000):
        raise ValueError("Only bounded 7000/14000/20000 us soak presets")
    if duration not in (300, 900, 1800):
        raise ValueError("Only staged 300/900/1800-second durations")
    if duration == 300:
        return
    required = 300 if duration == 900 else 900
    # Earlier five-minute report predates explicit status/restoration booleans.
    # Accept it only from the equivalent recorded device/stream evidence.
    legacy_verified = previous is not None and (
        "status" not in previous
        and "error" not in previous
        and previous.get("original") == BASELINE
        and previous.get("restored") == BASELINE
        and previous.get("state_after_soak") == previous.get("verified")
        and previous.get("stream", {}).get("stop_reason") == "duration budget reached"
    )
    if previous is None or not (
        (previous.get("status") == "completed" or legacy_verified)
        and (previous.get("restoration_verified") is True or legacy_verified)
        and previous.get("serial") == SERIAL
        and previous.get("requested_timing_us") == timing_us
        and previous.get("duration_limit_s") == required
        and previous.get("thermal_samples")
        and max(t["elapsed_s"] for t in previous["thermal_samples"]) >= required - 10
        and max(max(t["cpu_c"], t["npu_c"]) for t in previous["thermal_samples"]) < 75
    ):
        raise ValueError("Longer soak requires a successful, restored, below-75 C predecessor")
    if timing_us == 20000:
        samples = previous["thermal_samples"]
        times = [t["elapsed_s"] for t in samples]
        stream = previous.get("stream", {})
        if not (
            previous.get("quiet_after_stop") is True
            and previous.get("first_transport_fault", "missing") is None
            and previous.get("usb_stop_boundary", {}).get("settled_for_readback") is True
            and previous.get("post_restore_usb", {}).get("topology_stable") is True
            and previous.get("original") == previous.get("restored") == BASELINE
            and previous.get("original_gain_register") == 24
            and previous.get("restored_gain_register") == 24
            and stream.get("stop_reason") == "duration budget reached"
            and stream.get("short_frames") == stream.get("dropped_by_sequence") == 0
            and previous.get("distinct_payloads") == previous.get("hashed_complete_frames")
            and previous.get("hashed_complete_frames", 0) > 0
            and all(math.isfinite(t[k]) for t in samples for k in ("cpu_c", "npu_c", "elapsed_s"))
            and 0 <= times[0] <= 10
            and all(0 < b - a <= 15 for a, b in pairwise(times))
        ):
            raise ValueError(
                "Baseline predecessor lacks continuous telemetry/clean quiet lifecycle"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--timing-us", type=int, choices=(7000, 14000, 20000), default=7000)
    parser.add_argument("--previous-report", type=Path)
    parser.add_argument("--raw-excerpts", type=Path)
    parser.add_argument("--quiet-after-stop", action="store_true")
    args = parser.parse_args()
    if not args.execute or args.output.exists():
        parser.error("Explicit --execute and a new report path required")
    previous = json.loads(args.previous_report.read_text()) if args.previous_report else None
    validate_stage(args.duration, previous, args.timing_us)
    if args.timing_us == 20000:
        # Baseline continuation uses the successful settled-post lifecycle;
        # there is no option to bypass its quiet boundary in this new branch.
        args.quiet_after_stop = True
    elif (args.quiet_after_stop and args.duration != 300) or (
        args.duration > 300 and previous is not None and previous.get("quiet_after_stop")
    ):
        parser.error("Quiet-boundary variant is limited to the 300-second shakedown")
    if args.raw_excerpts is not None:
        args.raw_excerpts.mkdir(exist_ok=False)
    if args.quiet_after_stop:
        trace_path = args.output.with_suffix(".uvc-trace.jsonl")
        with trace_path.open("x") as stream:
            trace = WireTrace(fcntl.ioctl, stream)
            with patch.object(fcntl, "ioctl", trace):
                report = execute(args, trace)
    else:
        report = execute(args, None)
    if report["status"] != "completed":
        raise RuntimeError("Soak did not complete its acquisition/restoration/idle checks")


def execute(args: argparse.Namespace, trace: WireTrace | None) -> dict[str, Any]:
    import numpy as np

    serial = resolve_serial()
    if serial != SERIAL:
        raise RuntimeError("Experiment bound to exact physical MINI")

    quiet_owed = False
    quiet_failed = False
    boundary: UsbIdleWitness | None = None
    final_idle: UsbIdleWitness | None = None

    def opened() -> int:
        nonlocal quiet_owed, quiet_failed, final_idle
        if quiet_failed or (trace is not None and trace.first_fault is not None):
            raise RuntimeError("No more controls after failed quiet boundary or USB transport")
        if quiet_owed:
            quiet_owed = False
            assert boundary is not None
            print("QUIET_STOP: six seconds without UVC; require two seconds stable", flush=True)
            try:
                quiet_stop_observation(boundary)
                final_idle = UsbIdleWitness(hold=False)
                report["post_restore_usb"] = final_idle.report
            except BaseException:
                quiet_failed = True
                raise
        if not wait_for_capture_access(serial=serial, timeout_s=10).ready:
            raise RuntimeError("Scanner unavailable")
        node, _ = resolve_capture_node(serial)
        return os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)

    fd = opened()
    try:
        original, _ = state(fd)
        original_gain, _ = gain(fd)
        initial = temperatures(fd)
        if not all(math.isfinite(initial[k]) for k in ("cpu_c", "npu_c")):
            raise RuntimeError("Nonfinite initial thermal telemetry")
        if original != dict(frame_time=20000, exposure_time=20000, auto_expo_mode=0):
            raise RuntimeError("Unexpected baseline; refusing timing writes")
        if max(initial.values()) >= 80:
            raise RuntimeError("Too hot to begin")
        if original_gain != 24:
            raise RuntimeError("Unexpected gain baseline; refusing timing writes")
    except BaseException:
        os.close(fd)
        raise
    report: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "serial": serial,
        "original": original,
        "original_gain_register": original_gain,
        "initial_temperatures": initial,
        "requested_timing_us": args.timing_us,
        "duration_limit_s": args.duration,
        "quiet_after_stop": args.quiet_after_stop,
        "previous_report": str(args.previous_report) if args.previous_report else None,
        "raw_excerpts": [],
        "scope": (
            "Baseline 20 ms acquisition/thermal diagnostic; no tracking/GUI qualification"
            if args.timing_us == 20000
            else (
                "Acquisition endurance only; 7 ms depth is not quality-qualified"
                if args.timing_us == 7000
                else "Provisional 14 ms stationary endurance; no tracking/GUI qualification"
            )
        ),
        "host_loadavg_start": os.getloadavg(),
        "temperature_cutoff_c": 80,
        "temperature_scope": "CPU/NPU on RV1109; not projector/case or optical accuracy",
        "thermal_samples": [],
        "density_samples": [],
        "status": "running",
    }
    hashes: set[str] = set()
    complete = 0
    active_fd = -1
    next_monitor = 0.0
    excerpt_counts: dict[int, int] = {}
    real_ioctl = fcntl.ioctl
    started = time.monotonic()
    last_temperature_at = started

    def stale_telemetry() -> bool:
        # Polled after every delivery, including shorts that never enter the
        # temperature callback. A stalled stream independently times out after
        # 10 seconds (15 for its first frame) in stream_frames.
        stale = time.monotonic() - last_temperature_at > 15
        if stale:
            report["telemetry_stale"] = True
        return stale

    def observed_ioctl(device_fd: int, request: int, value: Any = 0, mutate: bool = True) -> Any:
        nonlocal active_fd
        result = real_ioctl(device_fd, request, value, mutate)
        if request == VIDIOC_DQBUF:
            active_fd = device_fd
        return result

    def delivered(raw: bytes, record: FrameRecord) -> None:
        nonlocal next_monitor, complete, last_temperature_at
        complete += 1
        hashes.add(hashlib.sha256(raw).hexdigest())
        now = time.monotonic()
        if now >= next_monitor:
            thermal = {"elapsed_s": now - started, "complete": complete, **temperatures(active_fd)}
            report["thermal_samples"].append(thermal)
            print(json.dumps(thermal), flush=True)
            if not all(math.isfinite(thermal[k]) for k in ("cpu_c", "npu_c")):
                raise RuntimeError("Nonfinite thermal telemetry")
            if max(thermal["cpu_c"], thermal["npu_c"]) >= 80:
                raise RuntimeError("80 C thermal stop reached")
            last_temperature_at = time.monotonic()
            next_monitor = time.monotonic() + 5
        minute = int((now - started) // 60)
        count = excerpt_counts.get(minute, 0)
        if args.raw_excerpts is not None and count < 10 and minute < args.duration // 60:
            path = args.raw_excerpts / f"minute-{minute:02d}-frame-{count:02d}.raw"
            with path.open("xb") as excerpt:
                excerpt.write(raw)
            excerpt_counts[minute] = count + 1
            report["raw_excerpts"].append(
                dict(
                    path=str(path),
                    elapsed_s=now - started,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    delivery_index=record.index,
                )
            )
        if record.index % 13 == 0:
            codes = np.frombuffer(raw, dtype="<u2")
            report["density_samples"].append(
                {
                    "elapsed_s": now - started,
                    "delivery_index": record.index,
                    "valid_code_fraction": float(np.mean((codes > 0) & (codes < 65535))),
                }
            )

    try:
        with cleanup_on_termination():
            set_timing(fd, args.timing_us, args.timing_us)
            verified, _ = state(fd)
            report["verified"] = verified
            if verified != {
                **original,
                "frame_time": args.timing_us,
                "exposure_time": args.timing_us,
            }:
                raise RuntimeError("Timing did not verify")
            report["verified_gain_register"], _ = gain(fd)
            if report["verified_gain_register"] != original_gain:
                raise RuntimeError("Gain changed during timing configuration")
            os.close(fd)
            fd = -1
            started = time.monotonic()
            last_temperature_at = started
            if args.quiet_after_stop:
                boundary = UsbIdleWitness(hold=False)
                report["usb_stop_boundary"] = boundary.report
                quiet_owed = True
            # Only this diagnostic process raises the 6000-frame runaway guard.
            # No production default changes, frame skipping or scanner mode change.
            with (
                patch.object(fcntl, "ioctl", observed_ioctl),
                patch.object(v4l2_stream, "MAX_FRAMES", args.duration * 20),
            ):
                stream = stream_frames(
                    serial,
                    frames=args.duration * 20,
                    duration_s=args.duration,
                    on_frame=delivered,
                    should_stop=stale_telemetry,
                )
            report["elapsed_s"] = time.monotonic() - started
            report["stream"] = stream.as_dict()
            report["deliveries"] = [
                dict(**r.as_dict(), monotonic_s=r.monotonic) for r in stream.records
            ]
            report["hashed_complete_frames"] = complete
            report["distinct_payloads"] = len(hashes)
            if report.get("telemetry_stale"):
                raise RuntimeError("Thermal telemetry stale for more than 15 seconds")
            fd = opened()
            report["state_after_soak"], _ = state(fd)
            report["gain_register_after_soak"], _ = gain(fd)
            if (
                report["state_after_soak"] != verified
                or report["gain_register_after_soak"] != original_gain
            ):
                raise RuntimeError("Timing/gain drifted during endurance capture")
            if report["elapsed_s"] < args.duration - 2:
                raise RuntimeError("Acquisition ended before requested endurance duration")
            report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with defer_termination():
            try:
                if quiet_failed or (trace is not None and trace.first_fault is not None):
                    report.update(
                        status="failed",
                        restoration_verified=False,
                        restoration_skipped="No controls after failed quiet boundary/transport",
                    )
                else:
                    if fd < 0:
                        fd = opened()
                    set_timing(fd, original["frame_time"], original["exposure_time"])
                    report["restored"], _ = state(fd)
                    report["restored_gain_register"], _ = gain(fd)
                    report["final_temperatures"] = temperatures(fd)
                    if (
                        report["restored"] != original
                        or report["restored_gain_register"] != original_gain
                    ):
                        raise RuntimeError("Restoration did not verify")
                    report["restoration_verified"] = True
            except BaseException as error:
                report["status_before_restoration_error"] = report["status"]
                report["status"] = "failed"
                report["restoration_verified"] = False
                report["restoration_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if fd >= 0:
                    os.close(fd)
                if final_idle is not None:
                    try:
                        final_idle.observe("after_restoration_close", 2.0)
                        stable = all(
                            row["same_device"] for row in final_idle.report["observations"]
                        )
                        final_idle.report["topology_stable"] = stable
                        if not stable:
                            report.update(status="failed", final_idle_error="USB topology changed")
                    except BaseException as error:
                        report.update(status="failed", final_idle_error=str(error))
                if trace is not None:
                    report["first_transport_fault"] = trace.first_fault
                    report["uvc_requests"] = trace.count
                    report["stream_events"] = trace.stream_events
                report["host_loadavg_end"] = os.getloadavg()
                report["ended_utc"] = datetime.now(UTC).isoformat()
                with args.output.open("x") as output:
                    json.dump(report, output, indent=2)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "deliveries",
                        "density_samples",
                        "thermal_samples",
                        "usb_stop_boundary",
                        "post_restore_usb",
                        "raw_excerpts",
                    )
                }
            ),
            flush=True,
        )
    return report


if __name__ == "__main__":
    main()
