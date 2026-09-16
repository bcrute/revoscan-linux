"""One bounded stop/readback experiment, with first-fault UVC tracing and no reset."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import struct
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner_firmware_inventory import gain
from scanner_fps_probe import MOTION_FRAME_LIMITS
from scanner_fps_probe import run as capture
from scanner_internal_probe import ObservationOperation
from scanner_timing_probe import UVC_QUERY, set_timing, state, temperatures, uvcioctl
from scanner_validation_baseline import BASELINE, SERIAL

from scanner3d.config import resolve_serial
from scanner3d.ownership import cleanup_on_termination, defer_termination
from scanner3d.v4l2_capture import (
    VIDIOC_DQBUF,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    resolve_capture_node,
)
from scanner3d.v4l2_stream import wait_for_capture_access

TRANSPORT_ERRORS = (errno.ENODEV, errno.EPROTO, errno.ETIMEDOUT, errno.EIO, errno.EPIPE)


class UsbIdleWitness:
    """Cached sysfs observations and an optional read-only, unclaimed usbfs FD.

    No USB ioctl, interface claim, reset, or power-policy write. An open usbfs
    descriptor prevents runtime autosuspend for its lifetime. Bind by exact
    serial, USB IDs and kernel device number, not a remembered bus address.
    """

    def __init__(self, *, hold: bool) -> None:
        matches = []
        for path in Path("/sys/bus/usb/devices").iterdir():
            try:
                identity = tuple(
                    (path / field).read_text().strip()
                    for field in ("idVendor", "idProduct", "serial")
                )
            except OSError:
                continue
            if identity == ("2207", "110c", SERIAL):
                matches.append(path)
        if len(matches) != 1:
            raise RuntimeError("One exact physical MINI USB identity required")
        self.path = matches[0]
        self.fd = -1
        self.report: dict[str, Any] = dict(hold_requested=hold, observations=[])
        self.initial = self.snapshot()
        if not self.initial["present"] or tuple(
            self.initial.get(key) for key in ("idVendor", "idProduct", "serial")
        ) != ("2207", "110c", SERIAL):
            raise RuntimeError("USB identity disappeared before observation")
        self.record("before_lease")

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = dict(sysfs_path=str(self.path), present=False)
        try:
            for field in ("idVendor", "idProduct", "serial", "busnum", "devnum", "dev", "speed"):
                result[field] = (self.path / field).read_text().strip()
            result["present"] = True
            result["power"] = {}
            for field in (
                "control",
                "autosuspend_delay_ms",
                "runtime_status",
                "runtime_usage",
                "runtime_active_time",
                "runtime_suspended_time",
            ):
                result["power"][field] = (self.path / "power" / field).read_text().strip()
        except OSError as error:
            result.update(present=False, error=str(error))
        return result

    def same_device(self, snapshot: dict[str, Any]) -> bool:
        return bool(snapshot["present"]) and all(
            snapshot.get(key) == self.initial.get(key)
            for key in ("idVendor", "idProduct", "serial", "busnum", "devnum", "dev")
        )

    def record(self, phase: str) -> dict[str, Any]:
        observation: dict[str, Any] = dict(
            utc=datetime.now(UTC).isoformat(),
            monotonic_s=time.monotonic(),
            phase=phase,
            lease_open=self.fd >= 0,
            snapshot=self.snapshot(),
        )
        observation["same_device"] = self.same_device(observation["snapshot"])
        self.report["observations"].append(observation)
        if not observation["same_device"]:
            self.report.setdefault("first_topology_fault", observation)
        return observation

    def require_same(self, phase: str) -> None:
        if not self.record(phase)["same_device"] or "first_topology_fault" in self.report:
            raise RuntimeError("USB topology changed; no more device operations")

    def acquire(self) -> None:
        if not self.report["hold_requested"]:
            return
        bus, device = (int(self.initial[key]) for key in ("busnum", "devnum"))
        if not 1 <= bus <= 999 or not 1 <= device <= 127:
            raise RuntimeError("Invalid cached USB bus/device number")
        path = f"/dev/bus/usb/{bus:03d}/{device:03d}"
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(self.fd)
            major, minor = (int(part) for part in self.initial["dev"].split(":"))
            if not stat.S_ISCHR(opened.st_mode) or opened.st_rdev != os.makedev(major, minor):
                raise RuntimeError("usbfs descriptor does not match the cached device")
            if not self.record("lease_acquired")["same_device"]:
                raise RuntimeError("USB identity changed while opening the lease")
            self.report["usbfs_path"] = path
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.fd >= 0:
            fd, self.fd = self.fd, -1
            os.close(fd)

    def observe(self, phase: str, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self.record(phase)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))

    def finish(self) -> None:
        try:
            self.record("all_v4l2_closed")
            if self.fd >= 0:
                self.observe("held_after_v4l2_close", 4.0)
        finally:
            self.close()
        self.record("lease_released")
        self.observe("released_idle", 6.0)
        self.report["topology_stable"] = all(
            row["same_device"] for row in self.report["observations"]
        )


def quiet_stop_observation(witness: UsbIdleWitness) -> None:
    """No device I/O for six seconds, then require two seconds of stable identity.

    This is a specifically declared stop/re-enumeration boundary, not retrying
    a failed control request. Retain all topology changes in the old epoch.
    The caller must resolve a fresh serial-bound descriptor afterward.
    """
    witness.observe("quiet_stop_boundary", 6.0)
    rows = witness.report["observations"]
    last = rows[-1]
    recent = [row for row in rows if row["monotonic_s"] >= last["monotonic_s"] - 2.1]
    keys = ("idVendor", "idProduct", "serial", "busnum", "devnum", "dev")
    snapshot = last["snapshot"]
    stable = (
        len(recent) >= 2
        and last["monotonic_s"] - recent[0]["monotonic_s"] >= 2.0
        and (snapshot.get("idVendor"), snapshot.get("idProduct"), snapshot.get("serial"))
        == ("2207", "110c", SERIAL)
        and all(
            row["snapshot"]["present"]
            and all(row["snapshot"].get(key) == snapshot.get(key) for key in keys)
            for row in recent
        )
    )
    witness.report["settled_for_readback"] = stable
    if not stable:
        raise RuntimeError("Scanner did not settle after quiet shutdown; no control retry")


class WireTrace:
    def __init__(self, original: Any, stream: Any) -> None:
        self.original = original
        self.stream = stream
        self.phase = "initial"
        self.first_fault: dict[str, Any] | None = None
        self.dequeued = 0
        self.count = 0
        self.stream_events: list[dict[str, Any]] = []

    def __call__(self, fd: int, request: int, value: Any = 0, mutate: bool = True) -> Any:
        if request != UVC_QUERY:
            event: dict[str, Any] | None = (
                dict(
                    kind="v4l2",
                    request=request,
                    phase=self.phase,
                    started_utc=datetime.now(UTC).isoformat(),
                    monotonic_s=time.monotonic(),
                    dequeued=self.dequeued,
                )
                if request in (VIDIOC_STREAMON, VIDIOC_STREAMOFF)
                else None
            )
            try:
                result = self.original(fd, request, value, mutate)
            except OSError as error:
                if event is not None:
                    event.update(error=str(error), errno=error.errno)
                if self.first_fault is None and error.errno in TRANSPORT_ERRORS:
                    self.first_fault = dict(
                        kind="v4l2",
                        request=request,
                        phase=self.phase,
                        errno=error.errno,
                        error=str(error),
                        monotonic_s=time.monotonic(),
                        dequeued=self.dequeued,
                    )
                raise
            finally:
                if event is not None:
                    event.update(
                        ended_utc=datetime.now(UTC).isoformat(),
                        duration_s=time.monotonic() - event["monotonic_s"],
                    )
                    self.stream_events.append(event)
                    self.stream.write(json.dumps(event) + "\n")
                    self.stream.flush()
            if request == VIDIOC_DQBUF:
                self.dequeued += 1
            return result
        if self.first_fault is not None:
            raise RuntimeError("No more UVC requests after the first transport fault")
        unit, selector, query, size, address = struct.unpack("@BBBHP", value)
        self.count += 1
        record: dict[str, Any] = dict(
            number=self.count,
            phase=self.phase,
            monotonic_s=time.monotonic(),
            unit=unit,
            selector=selector,
            query=query,
            size=size,
        )
        if query == 1:
            payload = ctypes.string_at(address, size)
            record["set_payload_hex"] = payload.hex()
        try:
            result = self.original(fd, request, value, mutate)
        except OSError as error:
            record.update(error=str(error), errno=error.errno)
            if error.errno in TRANSPORT_ERRORS:
                self.first_fault = dict(record)
            raise
        else:
            if query == 0x81:
                payload = ctypes.string_at(address, size)
                record["reply_sha256"] = hashlib.sha256(payload).hexdigest()
                if selector == 2 and size >= 4:
                    record["file_reply_count"] = struct.unpack_from("<I", payload)[0]
            return result
        finally:
            record["duration_s"] = time.monotonic() - record["monotonic_s"]
            self.stream.write(json.dumps(record) + "\n")
            self.stream.flush()


def run(
    output: Path,
    mode: str,
    *,
    hold_usb_open: bool = False,
    quiet_stop: bool = False,
    motion_timing_us: int | None = None,
    motion_frames: int = 320,
) -> dict[str, Any]:
    if (
        mode not in ("baseline", "post", "live", "inspect", "settled-post")
        or (quiet_stop and mode not in ("baseline", "live"))
        or (
            motion_timing_us is not None
            and (
                mode != "settled-post"
                or motion_timing_us not in (20000, 14000, 10000)
                or motion_frames not in MOTION_FRAME_LIMITS
            )
        )
        or (motion_timing_us is None and motion_frames != 320)
        or resolve_serial() != SERIAL
    ):
        raise ValueError("Closed modes and exact physical MINI required")
    output.mkdir(exist_ok=False, parents=True)
    report: dict[str, Any] = dict(
        serial=SERIAL,
        mode=mode,
        quiet_stop=quiet_stop,
        motion_timing_us=motion_timing_us,
        capture_frame_budget=motion_frames if motion_timing_us is not None else 64,
        capture_duration_budget_s=(
            MOTION_FRAME_LIMITS[motion_frames] if motion_timing_us is not None else 90
        ),
        started_utc=datetime.now(UTC).isoformat(),
        status="running",
        restoration_verified=False,
        steps=[],
        temperatures=[],
        scope=(
            "Bounded settings, temperature, uptime and log inspection; no stream or timing writes"
            if mode == "inspect"
            else (
                "Bounded raw moving acquisition; no live tracking, GUI or quality qualification"
                if motion_timing_us is not None
                else "Instrumented 64-frame lifecycle test, not throughput/quality qualification"
            )
        ),
    )
    fd = -1
    restore_owed = False
    live_restored = False
    control_blocked = False
    quiet_boundary_owed = False
    witness: UsbIdleWitness | None = None

    def opened() -> int:
        if witness is not None:
            witness.require_same("before_v4l2_open")
        if not wait_for_capture_access(serial=SERIAL, timeout_s=10).ready:
            raise RuntimeError("Bound scanner unavailable")
        node, _ = resolve_capture_node(SERIAL)
        if witness is not None:
            witness.require_same("resolved_v4l2_node")
        report.setdefault("opened_nodes", []).append(node.node_path)
        return os.open(node.node_path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)

    with (output / "uvc-trace.jsonl").open("x") as trace_stream:
        trace = WireTrace(fcntl.ioctl, trace_stream)

        def settled_epoch() -> None:
            nonlocal witness, control_blocked, quiet_boundary_owed
            assert witness is not None
            quiet_boundary_owed = False
            try:
                quiet_stop_observation(witness)
                report["usb_before_readback"] = witness.report
                witness.close()
                witness = UsbIdleWitness(hold=hold_usb_open)
                report["usb_idle"] = witness.report
                witness.acquire()
            except BaseException:
                control_blocked = True
                raise

        def step(name: str, function: Any, *args: Any) -> Any:
            prior = trace.phase
            trace.phase = name
            record: dict[str, Any] = dict(
                name=name,
                started_utc=datetime.now(UTC).isoformat(),
                dequeued=trace.dequeued,
                first_request=trace.count + 1,
            )
            report["steps"].append(record)
            print("STEP", name, flush=True)
            try:
                if witness is not None:
                    witness.require_same(name)
                result = function(*args)
                record["result"] = result
                record["status"] = "completed"
                return result
            except BaseException as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                record.update(ended_utc=datetime.now(UTC).isoformat(), last_request=trace.count)
                trace.phase = prior

        def verify(name: str, active_fd: int, timing: int) -> None:
            measured, _ = step(name + "_timing", state, active_fd)
            if measured != {**BASELINE, "frame_time": timing, "exposure_time": timing}:
                raise RuntimeError("Unexpected timing state")
            measured_gain, _ = step(name + "_gain", gain, active_fd)
            if measured_gain != 24:
                raise RuntimeError("Unexpected gain")

        def monitor(active_fd: int) -> None:
            nonlocal restore_owed, live_restored
            thermal = step(f"temperature_{len(report['temperatures'])}", temperatures, active_fd)
            report["temperatures"].append(dict(utc=datetime.now(UTC).isoformat(), **thermal))
            print("THERMAL", thermal, flush=True)
            if max(thermal.values()) >= 80:
                raise RuntimeError("80 C lifecycle cutoff")
            if mode == "live" and trace.dequeued >= 32 and not live_restored:
                step("restore_live", set_timing, active_fd, 20000, 20000)
                verify("verified_live_restore", active_fd, 20000)
                live_restored = True
                restore_owed = False

        def memory(active_fd: int) -> dict[str, str]:
            observer = ObservationOperation(uvcioctl, "inventory")
            try:
                return {
                    name: observer.read_file(active_fd, path, cap).decode("ascii")
                    for name, path, cap in (
                        ("memory", "/proc/meminfo", 4096),
                        ("uptime", "/proc/uptime", 512),
                    )
                }
            finally:
                observer.protocol_cleanup(active_fd)

        def logs(active_fd: int) -> dict[str, str]:
            results = {}
            for plan in ("controller-log", "kernel"):
                observer = ObservationOperation(uvcioctl, plan)
                try:
                    observer.perform(active_fd, [])
                    results[plan] = observer.blobs[plan].decode("utf-8", errors="replace")
                finally:
                    observer.protocol_cleanup(active_fd)
            return results

        with patch.object(fcntl, "ioctl", trace):
            try:
                with cleanup_on_termination():
                    witness = UsbIdleWitness(hold=hold_usb_open)
                    report["usb_idle"] = witness.report
                    witness.acquire()
                    fd = opened()
                    if mode == "inspect":
                        # Snapshot stop messages before our verbose getter/file
                        # traffic can evict them from the bounded log tail.
                        monitor(fd)
                        step("device_logs", logs, fd)
                    verify("initial", fd, 20000)
                    monitor(fd)
                    step("initial_memory", memory, fd)
                    requested_timing = motion_timing_us or 14000
                    if mode in ("post", "live", "settled-post"):
                        restore_owed = True
                        step(
                            f"configure_{requested_timing // 1000}ms",
                            set_timing,
                            fd,
                            requested_timing,
                            requested_timing,
                        )
                        verify("configured", fd, requested_timing)
                    if mode != "inspect":
                        os.close(fd)
                        fd = -1
                        trace.phase = "capture"
                        report["capture_started_utc"] = datetime.now(UTC).isoformat()
                        quiet_boundary_owed = mode == "settled-post"
                        report["capture"] = capture(
                            SERIAL,
                            motion_frames if motion_timing_us is not None else 64,
                            8,
                            0.001,
                            monitor=monitor,
                            raw_output=output / "raw",
                            **({"motion": True} if motion_timing_us is not None else {}),
                        )
                        report["capture_closed_utc"] = datetime.now(UTC).isoformat()
                        if mode == "settled-post":
                            settled_epoch()
                        witness.require_same("capture_returned")
                        if mode == "live" and not live_restored:
                            raise RuntimeError("Live restoration trigger was not reached")
                        if not quiet_stop:
                            fd = opened()
                            verify(
                                "after_stop",
                                fd,
                                requested_timing if mode in ("post", "settled-post") else 20000,
                            )
                    if mode in ("post", "settled-post"):
                        step("restore_post_stop", set_timing, fd, 20000, 20000)
                        verify("verified_post_stop_restore", fd, 20000)
                        restore_owed = False
                    if not quiet_stop:
                        monitor(fd)
                        step("final_memory", memory, fd)
                        report["restoration_verified"] = True
                    else:
                        report["restoration_verified"] = live_restored
                        report["post_stop_settings_verified"] = False
                        report["quiet_stop_note"] = (
                            "No UVC after capture; fresh post-return inspection is separate. "
                            "Baseline has no writes; live restoration, if any, verified in-stream."
                        )
                    report["status"] = "completed"
            except BaseException as error:
                report.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                with defer_termination():
                    try:
                        if restore_owed and trace.first_fault is None and not control_blocked:
                            if quiet_boundary_owed:
                                settled_epoch()
                            if fd < 0:
                                fd = opened()
                            step("failure_restore", set_timing, fd, 20000, 20000)
                            verify("failure_restore_verified", fd, 20000)
                            report["restoration_verified"] = True
                    except BaseException as error:
                        report["restoration_error"] = f"{type(error).__name__}: {error}"
                    finally:
                        if fd >= 0:
                            try:
                                os.close(fd)
                            except OSError as error:
                                report.update(status="failed", close_error=str(error))
                            fd = -1
                        report["controls_closed_utc"] = datetime.now(UTC).isoformat()
                        if witness is not None:
                            try:
                                witness.finish()
                                if not witness.report["topology_stable"]:
                                    report.update(
                                        status="failed",
                                        topology_error="USB disappeared or re-enumerated",
                                    )
                            except BaseException as error:
                                report.update(
                                    status="failed",
                                    usb_observation_error=f"{type(error).__name__}: {error}",
                                )
                            finally:
                                witness.close()
                        if trace.first_fault is not None:
                            report["status"] = "failed"
                        report.update(
                            first_transport_fault=trace.first_fault,
                            stream_events=trace.stream_events,
                            uvc_requests=trace.count,
                            ended_utc=datetime.now(UTC).isoformat(),
                        )
                        with (output / "report.json").open("x") as stream:
                            json.dump(report, stream, indent=2)
                        print(
                            "RESULT", report["status"], report["restoration_verified"], flush=True
                        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("baseline", "post", "live", "inspect", "settled-post"),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hold-usb-open", action="store_true")
    parser.add_argument("--quiet-stop", action="store_true")
    parser.add_argument("--motion-timing-us", type=int, choices=(20000, 14000, 10000))
    parser.add_argument(
        "--motion-frames", type=int, choices=tuple(MOTION_FRAME_LIMITS), default=320
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required; confirm the intended physical scene")
    result = run(
        args.output,
        args.mode,
        hold_usb_open=args.hold_usb_open,
        quiet_stop=args.quiet_stop,
        motion_timing_us=args.motion_timing_us,
        motion_frames=args.motion_frames,
    )
    if result["status"] != "completed":
        raise SystemExit(1)
