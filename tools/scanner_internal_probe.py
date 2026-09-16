"""Bounded, opt-in firmware observations for this scanner's FPS investigation.

Uses the previously recovered vendor file-read and short-command protocols.
No arbitrary command/path arguments, settings writes, firmware changes, reset,
or calibration. The only device file created is a uniquely named RAM scratch
file containing a bounded log excerpt. Interface handback uses the existing
signal-safe ownership lifecycle. Do not run alongside a camera stream.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scanner3d.config import resolve_serial
from scanner3d.control import USBDEVFS_CONTROL, ControlRequest, TargetSnapshot, pack_ctrltransfer
from scanner3d.ownership import (
    InterfaceZeroLifecycle,
    OwnedOperation,
    OwnershipStatus,
    PhaseResult,
    ProtocolCleanupRecord,
    _Primary,
)

PATH_REQUEST = ControlRequest("diagnostic-read-path", 0x21, 1, 0x100, 0x400, 60, 1000)
READ_REQUEST = ControlRequest("diagnostic-read-chunk", 0xA1, 0x81, 0x200, 0x400, 60, 1000)
COMMAND_REQUEST = ControlRequest("diagnostic-short-observation", 0x21, 1, 0x700, 0x400, 60, 1000)
FILES = {
    "controller": ("/tmp/zxmain", 200_000),
    "cpu": ("/proc/cpuinfo", 4096),
    "memory": ("/proc/meminfo", 4096),
    "load": ("/proc/loadavg", 512),
    "uptime": ("/proc/uptime", 512),
}


class ObservationOperation:
    def __init__(self, ioctl_fn: Any, plan: str) -> None:
        if plan not in ("inventory", "controller", "kernel", "controller-log"):
            raise ValueError("Unknown closed observation plan")
        self.ioctl_fn = ioctl_fn
        self.plan = plan
        self.clear_owed = False
        self.blobs: dict[str, bytes] = {}
        self.events: list[dict[str, object]] = []

    def issue(self, fd: int, request: ControlRequest, payload: bytes | None = None) -> bytes:
        if request not in (PATH_REQUEST, READ_REQUEST, COMMAND_REQUEST):
            raise ValueError("Request outside diagnostic whitelist")
        if payload is not None and len(payload) != 60:
            raise ValueError("Incorrect transfer length")
        buffer = ctypes.create_string_buffer(60)
        if payload is not None:
            buffer.raw = payload
        transferred = self.ioctl_fn(
            fd, USBDEVFS_CONTROL, pack_ctrltransfer(request, ctypes.addressof(buffer)), True
        )
        if transferred != 60:
            raise RuntimeError(f"{request.name}: short transfer {transferred}/60")
        return buffer.raw

    def read_file(self, fd: int, path: str, cap: int) -> bytes:
        encoded = b"\x01" + path.encode("ascii") + b"\x00"
        if len(encoded) > 60:
            raise ValueError("Path too long")
        self.clear_owed = True
        self.issue(fd, PATH_REQUEST, encoded.ljust(60, b"\x00"))
        started = time.monotonic()
        chunks = bytearray()
        for _ in range(cap // 56 + 2):
            if time.monotonic() - started > 45:
                raise TimeoutError("Bounded firmware read exceeded 45 seconds")
            reply = self.issue(fd, READ_REQUEST)
            count = struct.unpack_from("<I", reply)[0]
            if count > 56 or len(chunks) + count > cap:
                raise RuntimeError("Malformed or over-capacity firmware reply")
            if count == 0:
                self.issue(fd, PATH_REQUEST, bytes(60))
                self.clear_owed = False
                self.events.append({"path": path, "bytes": len(chunks), "complete": True})
                return bytes(chunks)
            chunks.extend(reply[4 : 4 + count])
        raise RuntimeError("Read transfer cap reached without EOF")

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        if self.plan in ("kernel", "controller-log"):
            # The suffix is generated locally, numeric, and cannot inject shell syntax.
            scratch = f"/tmp/f{time.time_ns():x}"
            command = (
                f"dmesg|tail -c 16000 >{scratch}"
                if self.plan == "kernel"
                else f"tail -c16000 /tmp/webroot/zxmain.log>{scratch}"
            )
            if len(command.encode()) >= 60:
                raise ValueError("Bounded observation exceeds command payload")
            self.issue(fd, COMMAND_REQUEST, command.encode().ljust(60, b"\x00"))
            self.events.append({"ram_observation_command": command})
            self.blobs[self.plan] = self.read_file(fd, scratch, 16000)
        else:
            keys = (
                ["controller"] if self.plan == "controller" else ["cpu", "memory", "load", "uptime"]
            )
            for key in keys:
                path, cap = FILES[key]
                self.blobs[key] = self.read_file(fd, path, cap)
        return _Primary(OwnershipStatus.OK, "Bounded observation reads completed")

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        if not self.clear_owed:
            return ProtocolCleanupRecord.not_needed("Each completed file read was cleared")
        self.issue(fd, PATH_REQUEST, bytes(60))
        self.clear_owed = False
        return ProtocolCleanupRecord(True, True, "Cleared outstanding file selector")


@dataclass(frozen=True)
class ObservationBackend(InterfaceZeroLifecycle):
    plan: str = "inventory"

    def _snapshot_request(self) -> ControlRequest:
        return PATH_REQUEST

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return ObservationOperation(self.ioctl_fn, self.plan)

    def run(self, serial: str, output: Path) -> dict[str, object]:
        preflight = self._preflight(serial, "Bounded firmware observation")
        if preflight.status != OwnershipStatus.NOT_ATTEMPTED or preflight.target is None:
            raise RuntimeError(preflight.detail)
        phases = list(preflight.phases)
        primary, window = self._perform_lifecycle(preflight.target, phases)
        operation = window.operation
        assert isinstance(operation, ObservationOperation)
        report: dict[str, Any] = {
            "serial": serial,
            "plan": self.plan,
            "status": str(primary.status),
            "detail": primary.detail,
            "driver_cleanup": window.cleanup.as_dict(),
            "protocol_cleanup": window.protocol.as_dict(),
            "events": operation.events,
            "files": {},
        }
        for key, blob in operation.blobs.items():
            destination = output / (key + (".bin" if key == "controller" else ".txt"))
            with destination.open("xb") as stream:
                stream.write(blob)
            report["files"][key] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        if primary.status != OwnershipStatus.OK or not window.cleanup.ok or not window.protocol.ok:
            raise RuntimeError(json.dumps(report))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", choices=["inventory", "controller", "kernel", "controller-log"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required; camera streams must be closed")
    args.output.mkdir(exist_ok=False)
    print(
        json.dumps(ObservationBackend(plan=args.plan).run(resolve_serial(), args.output)),
        flush=True,
    )


if __name__ == "__main__":
    main()
