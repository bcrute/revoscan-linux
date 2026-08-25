"""Assembles the ``doctor`` diagnostic report.

The report is deliberately *measured only*: every field either comes from a
kernel-exported fact or from the bounded reference probe. Nothing is inferred
about capabilities that were not observed, and any check that did not run is
recorded as ``not run`` rather than omitted.
"""

from __future__ import annotations

import getpass
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scanner3d import __version__
from scanner3d.discovery import (
    DEFAULT_DEV_ROOT,
    DEFAULT_SYSFS_ROOT,
    EXPECTED_MANUFACTURER,
    EXPECTED_PRODUCT_ID,
    EXPECTED_VENDOR_ID,
    ScannerDevice,
    find_scanners,
)
from scanner3d.errors import DoctorStatus, worst
from scanner3d.reference import ProbeResult

#: Roles the scanner must expose for later depth+colour work.
REQUIRED_ROLES: tuple[str, ...] = ("depth", "rgb")

#: Formats each role must advertise for the planned native pipeline.
REQUIRED_FORMATS: dict[str, tuple[str, ...]] = {
    "depth": ("Y16", "YUY2"),
    "rgb": ("MJPG",),
}


@dataclass(frozen=True, slots=True)
class Check:
    """One named diagnostic outcome."""

    name: str
    status: DoctorStatus
    detail: str
    ran: bool = True
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": str(self.status),
            "ran": self.ran,
            "detail": self.detail,
            "data": dict(self.data),
        }


def not_run(name: str, reason: str) -> Check:
    """A check that was deliberately skipped; never reported as passing."""
    return Check(name=name, status=DoctorStatus.OK, detail=f"not run: {reason}", ran=False)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The complete machine-readable diagnostic."""

    status: DoctorStatus
    checks: tuple[Check, ...]
    devices: tuple[ScannerDevice, ...]
    environment: dict[str, object]
    reference: ProbeResult | None
    reference_skipped_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "tool": {"name": "scanner3d", "version": __version__},
            "status": str(self.status),
            "expected_device": {
                "vendor_id": f"{EXPECTED_VENDOR_ID:04x}",
                "product_id": f"{EXPECTED_PRODUCT_ID:04x}",
                "manufacturer": EXPECTED_MANUFACTURER,
            },
            "environment": dict(self.environment),
            "checks": [c.as_dict() for c in self.checks],
            "devices": [d.as_dict() for d in self.devices],
            "reference_probe": (
                self.reference.as_dict()
                if self.reference is not None
                else {"ran": False, "reason": self.reference_skipped_reason}
            ),
        }


def _environment() -> dict[str, object]:
    try:
        user = getpass.getuser()
    except KeyError, OSError:
        user = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "user": user,
        "uid": os.getuid(),
        "groups": sorted(os.getgroups()),
    }


def _check_presence(devices: tuple[ScannerDevice, ...]) -> Check:
    if not devices:
        return Check(
            name="device_present",
            status=DoctorStatus.DEVICE_ABSENT,
            detail=(
                f"no USB device {EXPECTED_VENDOR_ID:04x}:{EXPECTED_PRODUCT_ID:04x} found; "
                "attach the scanner and re-run"
            ),
        )
    paths = [d.usb_path for d in devices]
    return Check(
        name="device_present",
        status=DoctorStatus.OK,
        detail=f"found {len(devices)} matching device(s) at {', '.join(paths)}",
        data={"count": len(devices), "usb_paths": paths},
    )


def _check_identity(device: ScannerDevice) -> Check:
    problems: list[str] = []
    if not device.identity_matches:
        problems.append(
            f"USB id {device.vendor_id:04x}:{device.product_id:04x} is not the expected "
            f"{EXPECTED_VENDOR_ID:04x}:{EXPECTED_PRODUCT_ID:04x}"
        )
    if device.manufacturer != EXPECTED_MANUFACTURER:
        problems.append(f"manufacturer {device.manufacturer!r} is not {EXPECTED_MANUFACTURER!r}")
    if not device.serial:
        problems.append("device reports no USB serial")
    data: dict[str, Any] = {
        "serial": device.serial,
        "manufacturer": device.manufacturer,
        "product": device.product,
        "bcd_device": device.bcd_device,
    }
    if problems:
        return Check(
            name="device_identity",
            status=DoctorStatus.UNEXPECTED_IDENTITY,
            detail="; ".join(problems),
            data=data,
        )
    return Check(
        name="device_identity",
        status=DoctorStatus.OK,
        detail=f"serial {device.serial}, {device.manufacturer} {device.product}",
        data=data,
    )


def _check_topology(device: ScannerDevice) -> Check:
    speed = device.usb_speed_label
    data = {
        "bus_number": device.bus_number,
        "device_number": device.device_number,
        "port_path": device.port_path,
        "usb_path": device.usb_path,
        "speed_mbps": device.speed_mbps,
        "usb_speed": speed,
        "usb_version": device.usb_version,
    }
    if device.speed_mbps is None:
        return Check(
            name="usb_topology",
            status=DoctorStatus.UNEXPECTED_IDENTITY,
            detail="kernel did not report a link speed for this device",
            data=data,
        )
    return Check(
        name="usb_topology",
        status=DoctorStatus.OK,
        detail=f"port {device.usb_path} negotiated {speed}",
        data=data,
    )


def _check_roles(device: ScannerDevice) -> Check:
    roles = {fn.role: fn for fn in device.functions}
    missing = [role for role in REQUIRED_ROLES if role not in roles]
    data: dict[str, Any] = {
        "roles": {
            role: {
                "interface_number": fn.interface_number,
                "interface_string": fn.interface_string,
                "driver": fn.driver,
                "capture_nodes": [n.name for n in fn.capture_nodes],
                "all_nodes": [n.name for n in fn.video_nodes],
            }
            for role, fn in roles.items()
        },
        "role_source": "USB interface string descriptor (not /dev/videoN ordering)",
    }
    if missing:
        return Check(
            name="camera_roles",
            status=DoctorStatus.UNEXPECTED_IDENTITY,
            detail=f"missing camera role(s): {', '.join(missing)}",
            data=data,
        )
    summary = ", ".join(
        f"{role}=interface {roles[role].interface_number}"
        f" ({','.join(n.name for n in roles[role].capture_nodes) or 'no capture node'})"
        for role in REQUIRED_ROLES
    )
    nodeless = [role for role in REQUIRED_ROLES if not roles[role].capture_nodes]
    if nodeless:
        return Check(
            name="camera_roles",
            status=DoctorStatus.UNEXPECTED_IDENTITY,
            detail=f"role(s) present but with no capture node: {', '.join(nodeless)}",
            data=data,
        )
    return Check(name="camera_roles", status=DoctorStatus.OK, detail=summary, data=data)


def _check_modes(device: ScannerDevice) -> Check:
    if device.descriptors is None:
        return Check(
            name="advertised_modes",
            status=DoctorStatus.UNSUPPORTED_MODE,
            detail=f"could not parse USB descriptors: {device.descriptor_error}",
        )
    data: dict[str, Any] = {}
    missing: list[str] = []
    for role, wanted in REQUIRED_FORMATS.items():
        fn = device.function(role)
        advertised = [f.fourcc for f in fn.formats] if fn is not None else []
        data[role] = {
            "advertised": [f.as_dict() for f in (fn.formats if fn is not None else ())],
        }
        if not any(w in advertised for w in wanted):
            missing.append(
                f"{role} advertises {advertised or 'nothing'}, expected one of {list(wanted)}"
            )
    if missing:
        return Check(
            name="advertised_modes",
            status=DoctorStatus.UNSUPPORTED_MODE,
            detail="; ".join(missing),
            data=data,
        )
    parts: list[str] = []
    for role in REQUIRED_FORMATS:
        fn = device.function(role)
        formats = fn.formats if fn is not None else ()
        parts.append(
            f"{role}: "
            + ", ".join(
                f"{f.fourcc} "
                + "/".join(f"{m.width}x{m.height}@{m.default_frame_rate_hz}" for m in f.modes)
                for f in formats
            )
        )
    summary = "; ".join(parts)
    return Check(name="advertised_modes", status=DoctorStatus.OK, detail=summary, data=data)


def _check_permissions(device: ScannerDevice) -> Check:
    blocked: list[str] = []
    data: dict[str, Any] = {"usbfs": device.usbfs.as_dict(), "nodes": [], "hid": []}
    if not device.usbfs.usable:
        blocked.append(
            f"{device.usbfs.path} is not read-write for uid {os.getuid()} "
            "(the vendor SDK and any libusb transport need it)"
        )
    for fn in device.functions:
        for node in fn.capture_nodes:
            data["nodes"].append({"role": fn.role, **node.access.as_dict()})
            if not node.access.usable:
                blocked.append(f"{node.device_path} ({fn.role}) is not read-write")
    for hid in device.hid_nodes:
        data["hid"].append(hid.as_dict())
    hid_usable = [h for h in device.hid_nodes if h.access.usable]
    data["hid_usable_count"] = len(hid_usable)
    if blocked:
        return Check(
            name="permissions",
            status=DoctorStatus.PERMISSION_DENIED,
            detail="; ".join(blocked),
            data=data,
        )
    detail = f"usbfs and {len(data['nodes'])} capture node(s) are read-write"
    if device.hid_nodes and not hid_usable:
        detail += (
            f"; {len(device.hid_nodes)} hidraw node(s) present but not accessible "
            "(not required: the vendor SDK reaches HID through usbfs)"
        )
    return Check(name="permissions", status=DoctorStatus.OK, detail=detail, data=data)


def _check_frames(probe: ProbeResult | None, skipped_reason: str | None) -> Check:
    if probe is None:
        return not_run("frame_delivery", skipped_reason or "reference probe not requested")
    if probe.timed_out:
        return Check(
            name="frame_delivery",
            status=DoctorStatus.NO_FRAMES,
            detail=f"reference probe timed out after {probe.elapsed_s:.1f}s and was killed",
        )
    data = probe.data or {}
    frames = data.get("frames") if isinstance(data.get("frames"), dict) else {}
    counts = {
        role: (frames.get(role) or {}).get("count", 0) if isinstance(frames, dict) else 0
        for role in REQUIRED_ROLES
    }
    missing = [role for role, count in counts.items() if not count]
    if missing or probe.error:
        return Check(
            name="frame_delivery",
            status=DoctorStatus.NO_FRAMES,
            detail=(probe.error or f"reference probe delivered no {', '.join(missing)} frame(s)"),
            data={"frame_counts": counts, "probe_errors": data.get("errors")},
        )
    return Check(
        name="frame_delivery",
        status=DoctorStatus.OK,
        detail=", ".join(f"{role}: {count} frame(s)" for role, count in counts.items()),
        data={"frame_counts": counts},
    )


def _check_calibration(probe: ProbeResult | None, skipped_reason: str | None) -> Check:
    if probe is None:
        return not_run("calibration_metadata", skipped_reason or "reference probe not requested")
    data = probe.data or {}
    have = {
        "depth_intrinsics": bool((data.get("intrinsics") or {}).get("depth")),
        "rgb_intrinsics": bool((data.get("intrinsics") or {}).get("rgb")),
        "extrinsics": bool(data.get("extrinsics")),
        "depth_scale": data.get("depth_scale") is not None,
    }
    missing = [k for k, v in have.items() if not v]
    if missing:
        return Check(
            name="calibration_metadata",
            status=DoctorStatus.NO_FRAMES,
            detail=f"reference probe did not return: {', '.join(missing)}",
            data=have,
        )
    return Check(
        name="calibration_metadata",
        status=DoctorStatus.OK,
        detail="depth+RGB intrinsics, depth->RGB extrinsics and depth scale retrieved",
        data=have,
    )


def run_doctor(
    *,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
    probe: ProbeResult | None = None,
    probe_skipped_reason: str | None = None,
) -> DoctorReport:
    """Collect every diagnostic and fold it into a single report."""
    devices = find_scanners(sysfs_root=sysfs_root, dev_root=dev_root)
    checks: list[Check] = [_check_presence(devices)]

    if not devices:
        reason = probe_skipped_reason or "no matching device attached"
        for name in (
            "device_identity",
            "usb_topology",
            "camera_roles",
            "advertised_modes",
            "permissions",
        ):
            checks.append(not_run(name, "no matching device attached"))
        checks.append(not_run("frame_delivery", reason))
        checks.append(not_run("calibration_metadata", reason))
        return DoctorReport(
            status=DoctorStatus.DEVICE_ABSENT,
            checks=tuple(checks),
            devices=(),
            environment=_environment(),
            reference=None,
            reference_skipped_reason=reason,
        )

    device = devices[0]
    checks += [
        _check_identity(device),
        _check_topology(device),
        _check_roles(device),
        _check_modes(device),
        _check_permissions(device),
        _check_frames(probe, probe_skipped_reason),
        _check_calibration(probe, probe_skipped_reason),
    ]
    status = worst(c.status for c in checks if c.ran)
    return DoctorReport(
        status=status,
        checks=tuple(checks),
        devices=devices,
        environment=_environment(),
        reference=probe,
        reference_skipped_reason=probe_skipped_reason,
    )
