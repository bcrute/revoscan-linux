"""Bounded acquisition-only tests; no tracking or rendering during acquisition.

Only buffer count and the host's dequeue polling interval vary. Records the
kernel's buffer metadata as well as delivery times. Payload analysis happens
after stream closure so it cannot throttle acquisition. Reports are created
exclusively, never overwritten. Optional raw saves happen after closure,
including partial complete payloads on failure. Standard probes keep their
2..256-frame/90-second bounds; opt-in motion budgets are 320/640/1200 deliveries
with a 180-second ceiling, or 2400 deliveries with a 360-second ceiling.
Larger budgets do not imply the requested count was
actually delivered or that a complete rotation was observed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner3d import v4l2_stream
from scanner3d.config import resolve_serial
from scanner3d.v4l2_capture import VIDIOC_DQBUF, parse_buffer
from scanner3d.v4l2_stream import FrameRecord

MOTION_FRAME_LIMITS = {320: 180, 640: 180, 1200: 180, 2400: 360}


def run(
    serial: str,
    frames: int,
    buffers: int,
    poll_s: float,
    *,
    quality: bool = False,
    monitor: Callable[[int], None] | None = None,
    raw_output: Path | None = None,
    motion: bool = False,
) -> dict[str, Any]:
    import numpy as np

    if not 2 <= buffers <= 16 or not 0.0005 <= poll_s <= 0.02:
        raise ValueError("Probe settings outside bounded test range")
    if (motion and frames not in MOTION_FRAME_LIMITS) or (not motion and not 2 <= frames <= 256):
        raise ValueError(
            "Probe needs 2..256 frames, or an explicit 320/640/1200/2400-frame motion budget"
        )
    if raw_output is not None:
        raw_output.mkdir(exist_ok=False)
    if not v4l2_stream.wait_for_capture_access(serial=serial, timeout_s=10).ready:
        raise RuntimeError("Scanner is not ready")
    payloads: list[tuple[int, bytes]] = []
    payload_times: list[float] = []
    kernel: list[dict[str, Any]] = []
    real_ioctl = fcntl.ioctl
    cpu_started = time.process_time()
    active_fd = -1
    next_monitor = 0.0

    def observed_ioctl(fd: int, request: int, value: Any = 0, mutate: bool = True) -> Any:
        nonlocal active_fd
        result = real_ioctl(fd, request, value, mutate)
        if request == VIDIOC_DQBUF:
            active_fd = fd
            decoded = parse_buffer(bytes(value))
            kernel.append({**decoded.as_dict(), "dequeued_monotonic_ns": time.monotonic_ns()})
        return result

    def delivered(raw: bytes, record: FrameRecord) -> None:
        nonlocal next_monitor
        payloads.append((record.index, raw))
        payload_times.append(record.monotonic)
        if monitor is not None and time.monotonic() >= next_monitor:
            monitor(active_fd)
            next_monitor = time.monotonic() + 5.0

    capture_error = None
    try:
        with (
            patch.object(fcntl, "ioctl", observed_ioctl),
            patch.object(v4l2_stream, "POLL_SLEEP_S", poll_s),
        ):
            stream = v4l2_stream.stream_frames(
                serial=serial,
                frames=frames,
                buffers=buffers,
                duration_s=MOTION_FRAME_LIMITS[frames] if motion else 90,
                on_frame=delivered,
            )
    except BaseException as error:
        capture_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        capture_cpu_s = time.process_time() - cpu_started
        # This runs only after stream_frames has closed its descriptors. Keep
        # partial complete payloads even on capture failure; no new device I/O.
        if raw_output is not None:
            raw_records = []
            for (index, raw), stamp in zip(payloads, payload_times, strict=True):
                name = f"frame_{index:04d}.raw"
                with (raw_output / name).open("xb") as output:
                    output.write(raw)
                raw_records.append(
                    dict(
                        path=name,
                        delivery_index=index,
                        monotonic_s=stamp,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
            with (raw_output / "capture-progress.json").open("x") as metadata_output:
                json.dump(
                    dict(
                        scope="Raw delivery evidence, not accepted tracking frames",
                        capture_error=capture_error,
                        frames=raw_records,
                        kernel_buffers=kernel,
                    ),
                    metadata_output,
                    indent=2,
                )
    hashes = [hashlib.sha256(raw).hexdigest() for _, raw in payloads]
    images = [np.frombuffer(raw, dtype="<u2") for _, raw in payloads]
    changes = []
    for (index, _), a, b in zip(payloads[1:], images, images[1:], strict=False):
        valid = (a > 0) & (a < 65535) & (b > 0) & (b < 65535)
        differences = np.abs(a.astype(np.int32) - b.astype(np.int32))
        changes.append(
            {
                "delivery_index": index,
                "changed_fraction": float(np.mean(a != b)),
                "mutual_valid_fraction": float(np.mean(valid)),
                "median_mutual_change_raw_codes": float(np.median(differences[valid]))
                if np.any(valid)
                else None,
            }
        )
    stamps = [row["timestamp_ns"] for row in kernel]
    kernel_gaps = [(b - a) / 1e6 for a, b in pairwise(stamps)]
    gaps = [r.gap_ms for r in stream.records if r.gap_ms is not None]
    summary = {
        **stream.as_dict(),
        "requested_buffers": buffers,
        "host_poll_s": poll_s,
        "capture_cpu_s": capture_cpu_s,
        "distinct_payloads": len(set(hashes)),
        "hashed_complete_frames": len(hashes),
        "kernel_timestamp_gap_median_ms": statistics.median(kernel_gaps) if kernel_gaps else None,
        "delivery_gap_percentiles_ms": np.percentile(gaps, [0, 25, 50, 75, 100]).tolist()
        if gaps
        else [],
        "payload_bytes_per_second": stream.observed_fps * 512000 if stream.observed_fps else None,
        "mean_valid_code_fraction": float(
            np.mean([np.mean((raw > 0) & (raw < 65535)) for raw in images])
        )
        if images
        else None,
    }
    quality_report: dict[str, Any] = {}
    if quality:
        from scanner3d.gui import PREVIEW_POINTS
        from scanner3d.preparation import prepare_frame
        from scanner3d.tracking import IncrementalTracker

        tracker = IncrementalTracker(backend="scipy")
        prepared, decisions = [], []
        for _, raw in payloads:
            cloud = prepare_frame(raw, target_points=PREVIEW_POINTS)
            prepared.append(
                {
                    "deprojected_points": cloud.deprojected_points,
                    "clipped_points": cloud.clipped_points,
                    "prepared_points": len(cloud.points),
                }
            )
            started = time.perf_counter()
            decision = tracker.offer(cloud.points)
            decisions.append({**decision.as_dict(), "tracking_s": time.perf_counter() - started})
        quality_report = {
            "scope": "offline tracking after raw capture; not live throughput or metrology",
            "preparation": prepared,
            "decisions": decisions,
            "accepted": sum(d["accepted"] for d in decisions),
            "offered": len(decisions),
        }
    return {
        "summary": summary,
        "kernel_buffers": kernel,
        "deliveries": [{**r.as_dict(), "monotonic_s": r.monotonic} for r in stream.records],
        "payload_sha256": hashes,
        "consecutive_frame_changes": changes,
        "quality": quality_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report already exists")
    serial = resolve_serial()
    results = []
    for buffers, poll_s in [(4, 0.001), (2, 0.001), (8, 0.001), (4, 0.01)]:
        result = run(serial, args.frames, buffers, poll_s)
        results.append(result)
        print(json.dumps(result["summary"]), flush=True)
    with args.output.open("x") as output:
        json.dump({"serial": serial, "runs": results}, output, indent=2)


if __name__ == "__main__":
    main()
