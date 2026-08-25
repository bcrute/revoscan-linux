"""Stable Linux discovery of the Revopoint MINI over sysfs.

Everything here is rooted at injectable ``sysfs_root`` / ``dev_root`` paths so
the whole layer can be driven from recorded fixtures in tests. Nothing in this
module opens a character device or talks to the scanner; discovery is a pure
read of kernel-exported metadata.

Camera roles are resolved from the USB interface *string descriptor*
(``DepthCam<serial>`` / ``RGBCam<serial>``), never from the ``/dev/videoN``
number, because those numbers are assigned in probe order and change across
replug, reboot, and the presence of other UVC devices.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.usb_descriptors import ParsedDescriptors, StreamFormat, parse

#: The USB identity this project is built against.
EXPECTED_VENDOR_ID: Final = 0x2207
EXPECTED_PRODUCT_ID: Final = 0x110C
EXPECTED_MANUFACTURER: Final = "Chishine3d"

#: Prefix of the interface string descriptor -> logical camera role.
_ROLE_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("DepthCam", "depth"),
    ("RGBCam", "rgb"),
)

DEFAULT_SYSFS_ROOT: Final = Path("/sys")
DEFAULT_DEV_ROOT: Final = Path("/dev")


def _read_text(path: Path) -> str | None:
    """Read a sysfs attribute, returning ``None`` for anything unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError, ValueError:
        return None


def _read_int(path: Path, base: int = 10) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw, base)
    except ValueError:
        return None


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class AccessCheck:
    """Whether a device node exists and is usable by the current process."""

    path: str
    exists: bool
    readable: bool
    writable: bool
    owner_uid: int | None = None
    group_gid: int | None = None
    mode: str | None = None

    @property
    def usable(self) -> bool:
        return self.exists and self.readable and self.writable

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "readable": self.readable,
            "writable": self.writable,
            "owner_uid": self.owner_uid,
            "group_gid": self.group_gid,
            "mode": self.mode,
        }


def check_access(path: Path) -> AccessCheck:
    """Probe ``path`` for existence and R/W access without opening it."""
    try:
        st = path.stat()
    except OSError:
        return AccessCheck(path=str(path), exists=False, readable=False, writable=False)
    return AccessCheck(
        path=str(path),
        exists=True,
        readable=os.access(path, os.R_OK),
        writable=os.access(path, os.W_OK),
        owner_uid=st.st_uid,
        group_gid=st.st_gid,
        mode=oct(st.st_mode & 0o7777),
    )


@dataclass(frozen=True, slots=True)
class VideoNode:
    """One ``video4linux`` node belonging to a camera function."""

    name: str
    v4l_name: str | None
    index: int | None
    device_path: str
    access: AccessCheck

    @property
    def kind(self) -> str:
        """``capture`` for index 0, ``metadata`` for the UVC metadata node."""
        if self.index is None:
            return "unknown"
        return "capture" if self.index == 0 else "metadata"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "v4l_name": self.v4l_name,
            "index": self.index,
            "kind": self.kind,
            "device_path": self.device_path,
            "access": self.access.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class HidNode:
    """A ``hidraw`` node exposed by the scanner's HID control interface."""

    name: str
    device_path: str
    access: AccessCheck

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device_path": self.device_path,
            "access": self.access.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class UsbInterface:
    """A single USB interface of the scanner."""

    number: int
    interface_class: int | None
    interface_subclass: int | None
    interface_string: str | None
    driver: str | None
    video_nodes: tuple[VideoNode, ...] = ()
    hid_nodes: tuple[HidNode, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "class": self.interface_class,
            "subclass": self.interface_subclass,
            "interface_string": self.interface_string,
            "driver": self.driver,
            "video_nodes": [n.as_dict() for n in self.video_nodes],
            "hid_nodes": [n.as_dict() for n in self.hid_nodes],
        }


@dataclass(frozen=True, slots=True)
class CameraFunction:
    """A logical camera (depth or RGB) grouped by its interface association."""

    role: str
    interface_number: int
    interface_string: str | None
    driver: str | None
    video_nodes: tuple[VideoNode, ...]
    formats: tuple[StreamFormat, ...]

    @property
    def capture_nodes(self) -> tuple[VideoNode, ...]:
        return tuple(n for n in self.video_nodes if n.kind == "capture")

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "interface_number": self.interface_number,
            "interface_string": self.interface_string,
            "driver": self.driver,
            "video_nodes": [n.as_dict() for n in self.video_nodes],
            "advertised_formats": [f.as_dict() for f in self.formats],
        }


@dataclass(frozen=True, slots=True)
class ScannerDevice:
    """One attached scanner, as described purely by sysfs."""

    sysfs_path: str
    usb_path: str
    bus_number: int | None
    device_number: int | None
    port_path: str | None
    vendor_id: int | None
    product_id: int | None
    serial: str | None
    manufacturer: str | None
    product: str | None
    bcd_device: str | None
    usb_version: str | None
    speed_mbps: int | None
    usbfs: AccessCheck
    interfaces: tuple[UsbInterface, ...]
    functions: tuple[CameraFunction, ...]
    descriptors: ParsedDescriptors | None
    descriptor_error: str | None

    @property
    def identity_matches(self) -> bool:
        return self.vendor_id == EXPECTED_VENDOR_ID and self.product_id == EXPECTED_PRODUCT_ID

    def function(self, role: str) -> CameraFunction | None:
        for fn in self.functions:
            if fn.role == role:
                return fn
        return None

    @property
    def hid_nodes(self) -> tuple[HidNode, ...]:
        return tuple(node for iface in self.interfaces for node in iface.hid_nodes)

    @property
    def usb_speed_label(self) -> str | None:
        """Human label for the negotiated link speed."""
        match self.speed_mbps:
            case None:
                return None
            case 1 | 12:
                return f"{self.speed_mbps} Mb/s (USB 1.x)"
            case 480:
                return "480 Mb/s (USB 2.0 high-speed)"
            case 5000:
                return "5 Gb/s (USB 3.0 SuperSpeed)"
            case 10000:
                return "10 Gb/s (USB 3.1 SuperSpeed+)"
            case _:
                return f"{self.speed_mbps} Mb/s"

    def as_dict(self) -> dict[str, object]:
        return {
            "sysfs_path": self.sysfs_path,
            "usb_path": self.usb_path,
            "bus_number": self.bus_number,
            "device_number": self.device_number,
            "port_path": self.port_path,
            "vendor_id": f"{self.vendor_id:04x}" if self.vendor_id is not None else None,
            "product_id": f"{self.product_id:04x}" if self.product_id is not None else None,
            "serial": self.serial,
            "manufacturer": self.manufacturer,
            "product": self.product,
            "bcd_device": self.bcd_device,
            "usb_version": self.usb_version,
            "speed_mbps": self.speed_mbps,
            "usb_speed": self.usb_speed_label,
            "identity_matches_expected": self.identity_matches,
            "usbfs": self.usbfs.as_dict(),
            "interfaces": [i.as_dict() for i in self.interfaces],
            "camera_functions": [f.as_dict() for f in self.functions],
            "hid_nodes": [h.as_dict() for h in self.hid_nodes],
            "descriptors_parsed": self.descriptors is not None,
            "descriptors_truncated": (
                self.descriptors.truncated if self.descriptors is not None else None
            ),
            "descriptor_error": self.descriptor_error,
        }


def _role_for(interface_string: str | None) -> str | None:
    if not interface_string:
        return None
    for prefix, role in _ROLE_PREFIXES:
        if interface_string.startswith(prefix):
            return role
    return None


def _collect_video_nodes(iface_dir: Path, dev_root: Path) -> tuple[VideoNode, ...]:
    v4l_dir = iface_dir / "video4linux"
    if not v4l_dir.is_dir():
        return ()
    nodes: list[VideoNode] = []
    try:
        entries = sorted(v4l_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return ()
    for entry in entries:
        device_path = dev_root / entry.name
        nodes.append(
            VideoNode(
                name=entry.name,
                v4l_name=_read_text(entry / "name"),
                index=_read_int(entry / "index"),
                device_path=str(device_path),
                access=check_access(device_path),
            )
        )
    return tuple(nodes)


def _collect_hid_nodes(iface_dir: Path, dev_root: Path) -> tuple[HidNode, ...]:
    nodes: list[HidNode] = []
    for hid_dir in sorted(iface_dir.glob("*/hidraw")):
        try:
            entries = sorted(hid_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            device_path = dev_root / entry.name
            nodes.append(
                HidNode(
                    name=entry.name,
                    device_path=str(device_path),
                    access=check_access(device_path),
                )
            )
    return tuple(nodes)


def _collect_interfaces(device_dir: Path, dev_root: Path) -> tuple[UsbInterface, ...]:
    interfaces: list[UsbInterface] = []
    prefix = f"{device_dir.name}:"
    try:
        candidates = sorted(device_dir.glob(f"{prefix}*"), key=lambda p: p.name)
    except OSError:
        return ()
    for iface_dir in candidates:
        if not iface_dir.is_dir():
            continue
        number = _read_int(iface_dir / "bInterfaceNumber", base=16)
        if number is None:
            continue
        driver_link = iface_dir / "driver"
        driver = driver_link.readlink().name if driver_link.is_symlink() else None
        interfaces.append(
            UsbInterface(
                number=number,
                interface_class=_read_int(iface_dir / "bInterfaceClass", base=16),
                interface_subclass=_read_int(iface_dir / "bInterfaceSubClass", base=16),
                interface_string=_read_text(iface_dir / "interface"),
                driver=driver,
                video_nodes=_collect_video_nodes(iface_dir, dev_root),
                hid_nodes=_collect_hid_nodes(iface_dir, dev_root),
            )
        )
    return tuple(interfaces)


def _build_functions(
    interfaces: tuple[UsbInterface, ...],
    descriptors: ParsedDescriptors | None,
) -> tuple[CameraFunction, ...]:
    """Group interfaces into logical cameras keyed by their role string.

    An interface association groups a VideoControl interface with its
    VideoStreaming sibling; the role name and the ``/dev/video*`` nodes hang off
    the control interface, while the advertised formats live on the streaming
    one. Both are reachable from the association, so no ``videoN`` number is
    consulted.
    """
    functions: list[CameraFunction] = []
    for iface in interfaces:
        role = _role_for(iface.interface_string)
        if role is None:
            continue
        formats = descriptors.formats_for_function(iface.number) if descriptors is not None else ()
        functions.append(
            CameraFunction(
                role=role,
                interface_number=iface.number,
                interface_string=iface.interface_string,
                driver=iface.driver,
                video_nodes=iface.video_nodes,
                formats=formats,
            )
        )
    return tuple(functions)


def _iter_usb_device_dirs(sysfs_root: Path) -> Iterator[Path]:
    devices_dir = sysfs_root / "bus" / "usb" / "devices"
    try:
        entries = sorted(devices_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    for entry in entries:
        # Interface directories contain ':'; roots are named 'usbN'.
        if ":" in entry.name or entry.name.startswith("usb"):
            continue
        if (entry / "idVendor").exists():
            yield entry


def _usbfs_path(dev_root: Path, bus: int | None, device: int | None) -> Path:
    if bus is None or device is None:
        return dev_root / "bus" / "usb" / "unknown"
    return dev_root / "bus" / "usb" / f"{bus:03d}" / f"{device:03d}"


def _read_device(device_dir: Path, dev_root: Path) -> ScannerDevice:
    bus = _read_int(device_dir / "busnum")
    devnum = _read_int(device_dir / "devnum")
    usbfs = _usbfs_path(dev_root, bus, devnum)

    descriptors: ParsedDescriptors | None = None
    descriptor_error: str | None = None
    blob = _read_bytes(device_dir / "descriptors")
    if blob is None:
        descriptor_error = f"could not read {device_dir / 'descriptors'}"
    else:
        try:
            descriptors = parse(blob)
        except Exception as exc:  # defensive: a corrupt blob must not abort doctor
            descriptor_error = f"{type(exc).__name__}: {exc}"

    interfaces = _collect_interfaces(device_dir, dev_root)
    return ScannerDevice(
        sysfs_path=str(device_dir),
        usb_path=device_dir.name,
        bus_number=bus,
        device_number=devnum,
        port_path=_read_text(device_dir / "devpath"),
        vendor_id=_read_int(device_dir / "idVendor", base=16),
        product_id=_read_int(device_dir / "idProduct", base=16),
        serial=_read_text(device_dir / "serial"),
        manufacturer=_read_text(device_dir / "manufacturer"),
        product=_read_text(device_dir / "product"),
        bcd_device=_read_text(device_dir / "bcdDevice"),
        usb_version=_read_text(device_dir / "version"),
        speed_mbps=_read_int(device_dir / "speed"),
        usbfs=check_access(usbfs),
        interfaces=interfaces,
        functions=_build_functions(interfaces, descriptors),
        descriptors=descriptors,
        descriptor_error=descriptor_error,
    )


def find_scanners(
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
    vendor_id: int = EXPECTED_VENDOR_ID,
    product_id: int = EXPECTED_PRODUCT_ID,
) -> tuple[ScannerDevice, ...]:
    """Return every attached device matching ``vendor_id``/``product_id``."""
    found: list[ScannerDevice] = []
    for device_dir in _iter_usb_device_dirs(sysfs_root):
        vid = _read_int(device_dir / "idVendor", base=16)
        pid = _read_int(device_dir / "idProduct", base=16)
        if vid == vendor_id and pid == product_id:
            found.append(_read_device(device_dir, dev_root))
    return tuple(found)
