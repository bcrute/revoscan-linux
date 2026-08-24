#!/usr/bin/env python3
"""Offline reconstruction of the UVC/V4L2 state the scanner-011 RGB attempt reached.

`scanner-011` negotiated MJPG 1280x800 on the RGB node, mapped four buffers,
started the stream and then saw **no buffer become ready** in 8.007 s. This
helper reconstructs, from committed evidence only, exactly what standard state
that run reached and what it never negotiated, so the silence can be reasoned
about without touching the scanner.

It is a discovery/verification aid, **not** part of the `scanner3d` package: it
is never imported by the CLI and adds no runtime dependency. Like
[`xu_trace.py`](README.md) it is offline by construction — it reads two kinds of
committed file and nothing else:

- `tests/fixtures/mini-config-descriptor.bin`, the byte-for-byte configuration
  descriptor recorded from the physical MINI, and
- the frozen `scanner-009` / `scanner-010` / `scanner-011` run JSON.

It opens no device node, issues no USB transfer, runs no subprocess and does no
network I/O.

Why a second descriptor decoder exists alongside `scanner3d.usb_descriptors`:
the product parser deliberately answers one question — *which modes does this
unit advertise* — and ignores endpoints, alternate settings, `bcdUVC`, the VS
input header and per-frame indices. Those are exactly the fields the streaming
question turns on, and `docs/architecture.md` lists them as the unrecorded
prerequisite blocking direct-transport work. Decoding them here keeps the
product parser's contract (and its 951-test surface) untouched while making the
numbers reproducible and tested rather than hand-quoted in prose.

Invocation:

    python reference/analysis/uvc_stream_state.py \
        --descriptor tests/fixtures/mini-config-descriptor.bin \
        --rgb-attempt tests/fixtures/scanner-011-rgb-attempt.json \
        --depth-capture tests/fixtures/scanner-009-depth-capture.json

Add `--json` for the machine-readable form.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Descriptor constants. Standard USB 2.0 / UVC 1.x values.
# ---------------------------------------------------------------------------

DT_DEVICE = 0x01
DT_CONFIG = 0x02
DT_INTERFACE = 0x04
DT_ENDPOINT = 0x05
DT_INTERFACE_ASSOCIATION = 0x0B
DT_CS_INTERFACE = 0x24
DT_CS_ENDPOINT = 0x25

VC_HEADER = 0x01
VS_INPUT_HEADER = 0x01
VS_FORMAT_UNCOMPRESSED = 0x04
VS_FRAME_UNCOMPRESSED = 0x05
VS_FORMAT_MJPEG = 0x06
VS_FRAME_MJPEG = 0x07

USB_CLASS_VIDEO = 0x0E
SUBCLASS_VIDEO_CONTROL = 0x01
SUBCLASS_VIDEO_STREAMING = 0x02

_TRANSFER_TYPES = {0: "control", 1: "isochronous", 2: "bulk", 3: "interrupt"}

# UVC frame intervals are expressed in 100 ns ticks.
TICKS_PER_SECOND = 10_000_000

# The trailing 12 bytes shared by every "FourCC as a GUID" UVC format id.
_UVC_GUID_SUFFIX = bytes.fromhex("00001000800000aa00389b71")

# --- USB 2.0 high-speed bulk ceiling -------------------------------------
#
# USB 2.0 (spec 5.8.4) allows at most 13 bulk transactions of 512 bytes per
# 125 us microframe on a single high-speed endpoint. That is the protocol
# ceiling before any host-controller, hub or shared-bus effect, so a
# requirement below it does not *prove* the transfer fits -- but a requirement
# far below it removes bandwidth as an explanation.
HS_MICROFRAMES_PER_SECOND = 8_000
HS_BULK_TRANSACTIONS_PER_MICROFRAME = 13
HS_BULK_MAX_PACKET_BYTES = 512
HS_BULK_CEILING_BYTES_PER_SECOND = (
    HS_MICROFRAMES_PER_SECOND * HS_BULK_TRANSACTIONS_PER_MICROFRAME * HS_BULK_MAX_PACKET_BYTES
)

# UVC 1.1 grew the VS Probe/Commit block from 26 to 34 bytes; 1.5 grew it to
# 48. The vendor SDK selects between the first two with `bcdUVC > 0x010f`
# (lib3DCamera.so @0x25fb74/0x25fb7e), so the same threshold is applied here.
PROBE_COMMIT_BYTES_UVC10 = 26
PROBE_COMMIT_BYTES_UVC11 = 34


def probe_commit_block_bytes(bcd_uvc: int) -> int:
    """Length of the VS Probe/Commit control block for a given ``bcdUVC``."""
    if bcd_uvc > 0x010F:
        return PROBE_COMMIT_BYTES_UVC11
    return PROBE_COMMIT_BYTES_UVC10


# ---------------------------------------------------------------------------
# Decoded records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One USB endpoint descriptor."""

    address: int
    direction: str
    number: int
    transfer_type: str
    max_packet_size: int
    additional_transactions: int
    interval: int

    @property
    def bytes_per_microframe(self) -> int:
        """High-speed payload per microframe (``wMaxPacketSize`` x transactions)."""
        return self.max_packet_size * (self.additional_transactions + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": f"0x{self.address:02x}",
            "direction": self.direction,
            "number": self.number,
            "transfer_type": self.transfer_type,
            "max_packet_size": self.max_packet_size,
            "additional_transactions": self.additional_transactions,
            "interval": self.interval,
            "bytes_per_microframe": self.bytes_per_microframe,
        }


@dataclass(frozen=True, slots=True)
class Frame:
    """One VS_FRAME_UNCOMPRESSED / VS_FRAME_MJPEG descriptor.

    Both subtypes share the layout for every field read here.
    """

    frame_index: int
    width: int
    height: int
    min_bit_rate: int
    max_bit_rate: int
    max_video_frame_buffer_bytes: int
    default_frame_interval: int
    frame_intervals: tuple[int, ...]
    interval_type: int

    @property
    def default_frame_rate_hz(self) -> float | None:
        return interval_to_hz(self.default_frame_interval)

    @property
    def frame_rates_hz(self) -> tuple[float, ...]:
        rates = [interval_to_hz(t) for t in self.frame_intervals]
        return tuple(r for r in rates if r is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "min_bit_rate": self.min_bit_rate,
            "max_bit_rate": self.max_bit_rate,
            "max_video_frame_buffer_bytes": self.max_video_frame_buffer_bytes,
            "default_frame_interval": self.default_frame_interval,
            "default_frame_rate_hz": self.default_frame_rate_hz,
            "frame_intervals": list(self.frame_intervals),
            "frame_rates_hz": list(self.frame_rates_hz),
            "interval_type": self.interval_type,
        }


@dataclass(frozen=True, slots=True)
class Format:
    """One VS format descriptor and the frames that follow it."""

    format_index: int
    fourcc: str
    compressed: bool
    bits_per_pixel: int | None
    default_frame_index: int
    frames: tuple[Frame, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_index": self.format_index,
            "fourcc": self.fourcc,
            "compressed": self.compressed,
            "bits_per_pixel": self.bits_per_pixel,
            "default_frame_index": self.default_frame_index,
            "frames": [f.as_dict() for f in self.frames],
        }


@dataclass(frozen=True, slots=True)
class VsInputHeader:
    """The class-specific VS_INPUT_HEADER of a VideoStreaming interface."""

    num_formats: int
    total_length: int
    endpoint_address: int
    info: int
    terminal_link: int
    still_capture_method: int
    trigger_support: int
    trigger_usage: int
    controls: tuple[int, ...]

    @property
    def dynamic_format_change_supported(self) -> bool:
        """``bmInfo`` D0: whether the device accepts a format change mid-stream."""
        return bool(self.info & 0x01)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_formats": self.num_formats,
            "total_length": self.total_length,
            "endpoint_address": f"0x{self.endpoint_address:02x}",
            "info": self.info,
            "dynamic_format_change_supported": self.dynamic_format_change_supported,
            "terminal_link": self.terminal_link,
            "still_capture_method": self.still_capture_method,
            "trigger_support": self.trigger_support,
            "trigger_usage": self.trigger_usage,
            "controls": [f"0x{c:02x}" for c in self.controls],
        }


@dataclass(frozen=True, slots=True)
class VcHeader:
    """The class-specific VC_HEADER of a VideoControl interface."""

    bcd_uvc: int
    total_length: int
    clock_frequency_hz: int
    streaming_interfaces: tuple[int, ...]

    @property
    def uvc_version(self) -> str:
        return f"{self.bcd_uvc >> 8}.{(self.bcd_uvc >> 4) & 0xF}{self.bcd_uvc & 0xF}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "bcd_uvc": f"0x{self.bcd_uvc:04x}",
            "uvc_version": self.uvc_version,
            "probe_commit_block_bytes": probe_commit_block_bytes(self.bcd_uvc),
            "total_length": self.total_length,
            "clock_frequency_hz": self.clock_frequency_hz,
            "streaming_interfaces": list(self.streaming_interfaces),
        }


@dataclass(slots=True)
class Interface:
    """One interface *alternate setting*, with its endpoints and class data."""

    number: int
    alternate_setting: int
    interface_class: int
    interface_subclass: int
    interface_protocol: int
    string_index: int
    declared_endpoints: int
    endpoints: list[Endpoint] = field(default_factory=list)
    formats: list[Format] = field(default_factory=list)
    vs_input_header: VsInputHeader | None = None
    vc_header: VcHeader | None = None
    association_first_interface: int | None = None

    @property
    def is_video_streaming(self) -> bool:
        return (
            self.interface_class == USB_CLASS_VIDEO
            and self.interface_subclass == SUBCLASS_VIDEO_STREAMING
        )

    @property
    def is_video_control(self) -> bool:
        return (
            self.interface_class == USB_CLASS_VIDEO
            and self.interface_subclass == SUBCLASS_VIDEO_CONTROL
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "alternate_setting": self.alternate_setting,
            "interface_class": self.interface_class,
            "interface_subclass": self.interface_subclass,
            "interface_protocol": self.interface_protocol,
            "string_index": self.string_index,
            "declared_endpoints": self.declared_endpoints,
            "endpoints": [e.as_dict() for e in self.endpoints],
            "formats": [f.as_dict() for f in self.formats],
            "vs_input_header": self.vs_input_header.as_dict() if self.vs_input_header else None,
            "vc_header": self.vc_header.as_dict() if self.vc_header else None,
            "association_first_interface": self.association_first_interface,
        }


@dataclass(frozen=True, slots=True)
class Association:
    """An Interface Association Descriptor grouping VC + VS into one function."""

    first_interface: int
    interface_count: int
    function_class: int
    function_subclass: int
    string_index: int

    @property
    def interfaces(self) -> range:
        return range(self.first_interface, self.first_interface + self.interface_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "first_interface": self.first_interface,
            "interface_count": self.interface_count,
            "function_class": self.function_class,
            "function_subclass": self.function_subclass,
            "string_index": self.string_index,
        }


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """The 18-byte device descriptor that precedes the configuration."""

    bcd_usb: int
    device_class: int
    device_subclass: int
    device_protocol: int
    max_packet_size_0: int
    vendor_id: int
    product_id: int
    bcd_device: int
    num_configurations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "bcd_usb": f"0x{self.bcd_usb:04x}",
            "device_class": self.device_class,
            "device_subclass": self.device_subclass,
            "device_protocol": self.device_protocol,
            "max_packet_size_0": self.max_packet_size_0,
            "vendor_id": f"0x{self.vendor_id:04x}",
            "product_id": f"0x{self.product_id:04x}",
            "usb_id": f"{self.vendor_id:04x}:{self.product_id:04x}",
            "bcd_device": f"0x{self.bcd_device:04x}",
            "num_configurations": self.num_configurations,
        }


@dataclass(frozen=True, slots=True)
class Configuration:
    """A fully decoded configuration descriptor."""

    device: DeviceIdentity | None
    total_length: int
    num_interfaces: int
    configuration_value: int
    attributes: int
    max_power_ma: int
    interfaces: tuple[Interface, ...]
    associations: tuple[Association, ...]
    trailing_bytes: int

    def interface(self, number: int, alternate_setting: int = 0) -> Interface | None:
        for iface in self.interfaces:
            if iface.number == number and iface.alternate_setting == alternate_setting:
                return iface
        return None

    def alternate_settings(self, number: int) -> tuple[Interface, ...]:
        return tuple(i for i in self.interfaces if i.number == number)

    def association_for(self, number: int) -> Association | None:
        for assoc in self.associations:
            if number in assoc.interfaces:
                return assoc
        return None

    def duplicate_endpoint_addresses(self) -> dict[int, tuple[int, ...]]:
        """Endpoint addresses declared by more than one alt-0 interface.

        A configuration's simultaneously-active interfaces are supposed to
        declare disjoint endpoint addresses. Reporting collisions is an
        observation about the descriptor, not a diagnosis.
        """
        owners: dict[int, list[int]] = {}
        for iface in self.interfaces:
            if iface.alternate_setting != 0:
                continue
            for endpoint in iface.endpoints:
                owners.setdefault(endpoint.address, []).append(iface.number)
        return {address: tuple(numbers) for address, numbers in owners.items() if len(numbers) > 1}

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.as_dict() if self.device else None,
            "total_length": self.total_length,
            "num_interfaces": self.num_interfaces,
            "configuration_value": self.configuration_value,
            "attributes": self.attributes,
            "max_power_ma": self.max_power_ma,
            "associations": [a.as_dict() for a in self.associations],
            "interfaces": [i.as_dict() for i in self.interfaces],
            "trailing_bytes": self.trailing_bytes,
            "duplicate_endpoint_addresses": {
                f"0x{addr:02x}": list(owners)
                for addr, owners in self.duplicate_endpoint_addresses().items()
            },
        }


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def interval_to_hz(ticks: int) -> float | None:
    """Convert a UVC 100 ns frame interval to Hz, or ``None`` if not positive."""
    if ticks <= 0:
        return None
    return round(TICKS_PER_SECOND / ticks, 3)


def iter_records(blob: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(offset, descriptor_type, bytes)`` for each descriptor record.

    Stops cleanly at the first zero-length or over-long record rather than
    raising, so a truncated blob yields fewer records.
    """
    offset = 0
    while offset + 2 <= len(blob):
        length = blob[offset]
        if length < 2 or offset + length > len(blob):
            return
        yield offset, blob[offset + 1], blob[offset : offset + length]
        offset += length


def _guid_to_fourcc(guid: bytes) -> str:
    if len(guid) == 16 and guid[4:] == _UVC_GUID_SUFFIX:
        fourcc = guid[:4].decode("ascii", errors="replace").rstrip("\x00").rstrip()
        if fourcc:
            return fourcc
    return guid.hex()


def _decode_endpoint(data: bytes) -> Endpoint | None:
    if len(data) < 7:
        return None
    address = data[2]
    attributes = data[3]
    raw_max_packet = struct.unpack_from("<H", data, 4)[0]
    return Endpoint(
        address=address,
        direction="in" if address & 0x80 else "out",
        number=address & 0x0F,
        transfer_type=_TRANSFER_TYPES.get(attributes & 0x03, "unknown"),
        max_packet_size=raw_max_packet & 0x07FF,
        additional_transactions=(raw_max_packet >> 11) & 0x03,
        interval=data[6],
    )


def _decode_frame(data: bytes) -> Frame | None:
    """Decode a VS_FRAME_* descriptor.

    Continuous frame-interval descriptors (``bFrameIntervalType == 0``) carry a
    min/max/step triple instead of a discrete list; both endpoints of that range
    are reported, which is what the discrete list would have contained for a
    two-rate device.
    """
    if len(data) < 26:
        return None
    frame_index = data[3]
    width, height = struct.unpack_from("<HH", data, 5)
    min_bit_rate, max_bit_rate, max_buffer = struct.unpack_from("<III", data, 9)
    default_interval = struct.unpack_from("<I", data, 21)[0]
    interval_type = data[25]
    intervals: list[int] = []
    if interval_type == 0:
        if len(data) >= 38:
            min_i, max_i, _step = struct.unpack_from("<III", data, 26)
            intervals = [min_i, max_i]
    else:
        for index in range(interval_type):
            start = 26 + index * 4
            if start + 4 > len(data):
                break
            intervals.append(struct.unpack_from("<I", data, start)[0])
    return Frame(
        frame_index=frame_index,
        width=width,
        height=height,
        min_bit_rate=min_bit_rate,
        max_bit_rate=max_bit_rate,
        max_video_frame_buffer_bytes=max_buffer,
        default_frame_interval=default_interval,
        frame_intervals=tuple(intervals),
        interval_type=interval_type,
    )


def _decode_vs_input_header(data: bytes) -> VsInputHeader | None:
    if len(data) < 13:
        return None
    num_formats = data[3]
    total_length = struct.unpack_from("<H", data, 4)[0]
    control_size = data[12]
    controls: list[int] = []
    if control_size:
        for index in range(num_formats):
            start = 13 + index * control_size
            if start + control_size > len(data):
                break
            controls.append(int.from_bytes(data[start : start + control_size], "little"))
    return VsInputHeader(
        num_formats=num_formats,
        total_length=total_length,
        endpoint_address=data[6],
        info=data[7],
        terminal_link=data[8],
        still_capture_method=data[9],
        trigger_support=data[10],
        trigger_usage=data[11],
        controls=tuple(controls),
    )


def _decode_vc_header(data: bytes) -> VcHeader | None:
    if len(data) < 12:
        return None
    bcd_uvc, total_length = struct.unpack_from("<HH", data, 3)
    clock = struct.unpack_from("<I", data, 7)[0]
    in_collection = data[11]
    members: list[int] = []
    for index in range(in_collection):
        position = 12 + index
        if position >= len(data):
            break
        members.append(data[position])
    return VcHeader(
        bcd_uvc=bcd_uvc,
        total_length=total_length,
        clock_frequency_hz=clock,
        streaming_interfaces=tuple(members),
    )


def decode_configuration(blob: bytes) -> Configuration:
    """Decode a raw sysfs ``descriptors`` blob into a full configuration record.

    The blob sysfs exposes is the 18-byte device descriptor followed by each
    configuration descriptor. Only the first configuration is decoded; this
    unit declares exactly one.
    """
    device: DeviceIdentity | None = None
    total_length = 0
    num_interfaces = 0
    configuration_value = 0
    attributes = 0
    max_power_ma = 0
    interfaces: list[Interface] = []
    associations: list[Association] = []
    consumed = 0

    current: Interface | None = None
    pending_format: dict[str, Any] | None = None
    pending_frames: list[Frame] = []

    def flush_format() -> None:
        nonlocal pending_format, pending_frames
        if pending_format is not None and current is not None:
            current.formats.append(
                Format(
                    format_index=int(pending_format["format_index"]),
                    fourcc=str(pending_format["fourcc"]),
                    compressed=bool(pending_format["compressed"]),
                    bits_per_pixel=pending_format["bits_per_pixel"],
                    default_frame_index=int(pending_format["default_frame_index"]),
                    frames=tuple(pending_frames),
                )
            )
        pending_format = None
        pending_frames = []

    for offset, dtype, data in iter_records(blob):
        consumed = offset + len(data)
        if dtype == DT_DEVICE and len(data) >= 18:
            device = DeviceIdentity(
                bcd_usb=struct.unpack_from("<H", data, 2)[0],
                device_class=data[4],
                device_subclass=data[5],
                device_protocol=data[6],
                max_packet_size_0=data[7],
                vendor_id=struct.unpack_from("<H", data, 8)[0],
                product_id=struct.unpack_from("<H", data, 10)[0],
                bcd_device=struct.unpack_from("<H", data, 12)[0],
                num_configurations=data[17],
            )
        elif dtype == DT_CONFIG and len(data) >= 9:
            total_length = struct.unpack_from("<H", data, 2)[0]
            num_interfaces = data[4]
            configuration_value = data[5]
            attributes = data[7]
            max_power_ma = data[8] * 2
        elif dtype == DT_INTERFACE_ASSOCIATION and len(data) >= 8:
            associations.append(
                Association(
                    first_interface=data[2],
                    interface_count=data[3],
                    function_class=data[4],
                    function_subclass=data[5],
                    string_index=data[7],
                )
            )
        elif dtype == DT_INTERFACE and len(data) >= 9:
            flush_format()
            current = Interface(
                number=data[2],
                alternate_setting=data[3],
                interface_class=data[5],
                interface_subclass=data[6],
                interface_protocol=data[7],
                string_index=data[8],
                declared_endpoints=data[4],
            )
            for assoc in associations:
                if current.number in assoc.interfaces:
                    current.association_first_interface = assoc.first_interface
                    break
            interfaces.append(current)
        elif dtype == DT_ENDPOINT and current is not None:
            endpoint = _decode_endpoint(data)
            if endpoint is not None:
                current.endpoints.append(endpoint)
        elif dtype == DT_CS_INTERFACE and current is not None and len(data) >= 3:
            subtype = data[2]
            if current.is_video_streaming:
                if subtype == VS_INPUT_HEADER and current.vs_input_header is None:
                    current.vs_input_header = _decode_vs_input_header(data)
                elif subtype == VS_FORMAT_UNCOMPRESSED and len(data) >= 27:
                    flush_format()
                    pending_format = {
                        "format_index": data[3],
                        "fourcc": _guid_to_fourcc(data[5:21]),
                        "compressed": False,
                        "bits_per_pixel": data[21],
                        "default_frame_index": data[22],
                    }
                elif subtype == VS_FORMAT_MJPEG and len(data) >= 11:
                    flush_format()
                    pending_format = {
                        "format_index": data[3],
                        "fourcc": "MJPG",
                        "compressed": True,
                        "bits_per_pixel": None,
                        "default_frame_index": data[6],
                    }
                elif subtype in (VS_FRAME_UNCOMPRESSED, VS_FRAME_MJPEG):
                    frame = _decode_frame(data)
                    if frame is not None:
                        pending_frames.append(frame)
            elif current.is_video_control and subtype == VC_HEADER and current.vc_header is None:
                current.vc_header = _decode_vc_header(data)
        elif dtype == DT_CS_ENDPOINT:
            continue

    flush_format()
    return Configuration(
        device=device,
        total_length=total_length,
        num_interfaces=num_interfaces,
        configuration_value=configuration_value,
        attributes=attributes,
        max_power_ma=max_power_ma,
        interfaces=tuple(interfaces),
        associations=tuple(associations),
        trailing_bytes=max(0, len(blob) - consumed),
    )


# ---------------------------------------------------------------------------
# Stream-state reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamingPlan:
    """What the descriptor alone fixes about streaming one VS interface.

    Every field here is descriptor evidence. Nothing in this record is a
    runtime measurement, and nothing asserts that the device will honour it.
    """

    control_interface: int
    streaming_interface: int
    uvc_version: str
    probe_commit_block_bytes: int
    alternate_settings: tuple[int, ...]
    endpoint_address: int
    endpoint_transfer_type: str
    endpoint_max_packet_size: int
    header_endpoint_address: int
    terminal_link: int
    format_index: int
    frame_index: int
    fourcc: str
    compressed: bool
    width: int
    height: int
    max_video_frame_buffer_bytes: int
    default_frame_interval: int
    frame_intervals: tuple[int, ...]
    max_bit_rate: int

    @property
    def default_frame_rate_hz(self) -> float | None:
        return interval_to_hz(self.default_frame_interval)

    @property
    def requires_alt_setting_selection(self) -> bool:
        """True only when the interface offers a choice of alternate setting.

        A VideoStreaming interface with a single alternate setting has no
        bandwidth to negotiate: there is nothing for the host to choose and no
        isochronous admission control to fail.
        """
        return len(self.alternate_settings) > 1

    @property
    def max_bit_rate_bytes_per_second(self) -> float:
        return self.max_bit_rate / 8

    @property
    def bulk_ceiling_utilisation(self) -> float | None:
        """Fraction of the single-endpoint high-speed bulk ceiling required."""
        if self.endpoint_transfer_type != "bulk":
            return None
        return round(self.max_bit_rate_bytes_per_second / HS_BULK_CEILING_BYTES_PER_SECOND, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_interface": self.control_interface,
            "streaming_interface": self.streaming_interface,
            "uvc_version": self.uvc_version,
            "probe_commit_block_bytes": self.probe_commit_block_bytes,
            "alternate_settings": list(self.alternate_settings),
            "requires_alt_setting_selection": self.requires_alt_setting_selection,
            "endpoint_address": f"0x{self.endpoint_address:02x}",
            "endpoint_transfer_type": self.endpoint_transfer_type,
            "endpoint_max_packet_size": self.endpoint_max_packet_size,
            "header_endpoint_address": f"0x{self.header_endpoint_address:02x}",
            "terminal_link": self.terminal_link,
            "format_index": self.format_index,
            "frame_index": self.frame_index,
            "fourcc": self.fourcc,
            "compressed": self.compressed,
            "width": self.width,
            "height": self.height,
            "max_video_frame_buffer_bytes": self.max_video_frame_buffer_bytes,
            "default_frame_interval": self.default_frame_interval,
            "default_frame_rate_hz": self.default_frame_rate_hz,
            "frame_intervals": list(self.frame_intervals),
            "max_bit_rate": self.max_bit_rate,
            "max_bit_rate_bytes_per_second": self.max_bit_rate_bytes_per_second,
            "bulk_ceiling_bytes_per_second": HS_BULK_CEILING_BYTES_PER_SECOND,
            "bulk_ceiling_utilisation": self.bulk_ceiling_utilisation,
        }


class StreamingPlanError(ValueError):
    """The descriptor does not describe the requested streaming mode."""


def streaming_plan(
    config: Configuration,
    streaming_interface: int,
    fourcc: str,
    width: int,
    height: int,
) -> StreamingPlan:
    """Derive the descriptor-fixed streaming parameters for one mode.

    Raises `StreamingPlanError` when the interface, its endpoint, the format or
    the frame is absent, rather than inventing a default.
    """
    alt_settings = config.alternate_settings(streaming_interface)
    if not alt_settings:
        raise StreamingPlanError(f"no interface {streaming_interface} in the configuration")
    base = config.interface(streaming_interface, 0)
    if base is None or not base.is_video_streaming:
        raise StreamingPlanError(f"interface {streaming_interface} is not VideoStreaming at alt 0")
    if base.vs_input_header is None:
        raise StreamingPlanError(f"interface {streaming_interface} has no VS_INPUT_HEADER")
    if not base.endpoints:
        raise StreamingPlanError(f"interface {streaming_interface} alt 0 declares no endpoint")

    assoc = config.association_for(streaming_interface)
    control_interface = assoc.first_interface if assoc else streaming_interface
    control = config.interface(control_interface, 0)
    if control is None or control.vc_header is None:
        raise StreamingPlanError(f"no VideoControl header for interface {control_interface}")

    matched: tuple[Format, Frame] | None = None
    for fmt in base.formats:
        for frame in fmt.frames:
            if fmt.fourcc == fourcc and frame.width == width and frame.height == height:
                matched = (fmt, frame)
                break
        if matched is not None:
            break
    if matched is None:
        raise StreamingPlanError(
            f"interface {streaming_interface} does not advertise {fourcc} {width}x{height}"
        )
    fmt, frame = matched
    endpoint = base.endpoints[0]

    return StreamingPlan(
        control_interface=control_interface,
        streaming_interface=streaming_interface,
        uvc_version=control.vc_header.uvc_version,
        probe_commit_block_bytes=probe_commit_block_bytes(control.vc_header.bcd_uvc),
        alternate_settings=tuple(i.alternate_setting for i in alt_settings),
        endpoint_address=endpoint.address,
        endpoint_transfer_type=endpoint.transfer_type,
        endpoint_max_packet_size=endpoint.max_packet_size,
        header_endpoint_address=base.vs_input_header.endpoint_address,
        terminal_link=base.vs_input_header.terminal_link,
        format_index=fmt.format_index,
        frame_index=frame.frame_index,
        fourcc=fmt.fourcc,
        compressed=fmt.compressed,
        width=frame.width,
        height=frame.height,
        max_video_frame_buffer_bytes=frame.max_video_frame_buffer_bytes,
        default_frame_interval=frame.default_frame_interval,
        frame_intervals=frame.frame_intervals,
        max_bit_rate=frame.max_bit_rate,
    )


# ---------------------------------------------------------------------------
# Comparison against the frozen run JSON
# ---------------------------------------------------------------------------

# V4L2 operations a capture path can perform. The scanner-010/011 RGB runs and
# the scanner-009 depth run all drove the same engine, whose whole ioctl
# vocabulary is published in the run JSON under `plan.ioctls`. Anything absent
# from that list was never issued by owned code -- which is a statement about
# the tool, not about what uvcvideo did internally on its behalf.
NEGOTIATION_SURFACE: tuple[tuple[str, str], ...] = (
    ("VIDIOC_QUERYCAP", "capability check"),
    ("VIDIOC_G_FMT", "read the handle's current format"),
    ("VIDIOC_S_FMT", "set pixel format and resolution (drives UVC PROBE/COMMIT)"),
    ("VIDIOC_REQBUFS", "allocate the MMAP buffer queue"),
    ("VIDIOC_QUERYBUF", "query buffer offset/length"),
    ("VIDIOC_QBUF", "enqueue a buffer"),
    ("VIDIOC_DQBUF", "dequeue a filled buffer"),
    ("VIDIOC_STREAMON", "start the stream"),
    ("VIDIOC_STREAMOFF", "stop the stream"),
    ("VIDIOC_ENUM_FMT", "enumerate advertised pixel formats"),
    ("VIDIOC_ENUM_FRAMESIZES", "enumerate advertised frame sizes"),
    ("VIDIOC_ENUM_FRAMEINTERVALS", "enumerate advertised frame intervals"),
    ("VIDIOC_G_PARM", "read the negotiated frame interval"),
    ("VIDIOC_S_PARM", "request a frame interval"),
    ("VIDIOC_QUERYCTRL", "enumerate UVC controls"),
    ("VIDIOC_G_CTRL", "read a UVC control"),
    ("VIDIOC_S_CTRL", "set a UVC control"),
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The subset of a frozen run JSON this analysis reasons about."""

    mode: str
    status: str
    succeeded: bool
    exit_code: int
    role: str
    node_path: str
    interface_number: int
    fourcc: str
    width: int
    height: int
    driver_sizeimage: int
    descriptor_frame_rate_hz: float | None
    issued_ioctls: tuple[str, ...]
    steps: tuple[tuple[str, bool, float | None], ...]
    stream_duration_s: float | None

    @property
    def reached_streamon(self) -> bool:
        return any(name == "streamon" and ok for name, ok, _ in self.steps)

    @property
    def dequeued_bytes(self) -> bool:
        return any(name == "dqbuf" and ok for name, ok, _ in self.steps)

    @property
    def poll_duration_s(self) -> float | None:
        for name, _ok, duration in self.steps:
            if name == "poll":
                return duration
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "role": self.role,
            "node_path": self.node_path,
            "interface_number": self.interface_number,
            "fourcc": self.fourcc,
            "width": self.width,
            "height": self.height,
            "driver_sizeimage": self.driver_sizeimage,
            "descriptor_frame_rate_hz": self.descriptor_frame_rate_hz,
            "issued_ioctls": list(self.issued_ioctls),
            "reached_streamon": self.reached_streamon,
            "dequeued_bytes": self.dequeued_bytes,
            "poll_duration_s": self.poll_duration_s,
            "stream_duration_s": self.stream_duration_s,
        }


def summarize_run(document: dict[str, Any]) -> RunSummary:
    """Extract the comparison-relevant fields of a frozen run JSON."""
    plan = document.get("plan", {})
    target = document.get("target", {})
    negotiated = document.get("negotiated_format", {})
    steps = tuple(
        (str(s.get("step", "")), bool(s.get("ok")), s.get("duration_s"))
        for s in document.get("steps", [])
    )
    return RunSummary(
        mode=str(document.get("mode", "")),
        status=str(document.get("status", "")),
        succeeded=bool(document.get("succeeded")),
        exit_code=int(document.get("exit_code", -1)),
        role=str(plan.get("role", "")),
        node_path=str(target.get("node_path", "")),
        interface_number=int(target.get("interface_number", -1)),
        fourcc=str(negotiated.get("fourcc", plan.get("fourcc", ""))).strip(),
        width=int(negotiated.get("width", plan.get("width", 0))),
        height=int(negotiated.get("height", plan.get("height", 0))),
        driver_sizeimage=int(negotiated.get("sizeimage", 0)),
        descriptor_frame_rate_hz=plan.get("descriptor_frame_rate_hz"),
        issued_ioctls=tuple(sorted(plan.get("ioctls", {}))),
        steps=steps,
        stream_duration_s=document.get("stream_duration_s"),
    )


def negotiation_matrix(run: RunSummary) -> list[dict[str, Any]]:
    """Classify each V4L2 operation as issued or never issued by owned code."""
    issued = set(run.issued_ioctls)
    return [
        {"ioctl": name, "purpose": purpose, "issued_by_owned_code": name in issued}
        for name, purpose in NEGOTIATION_SURFACE
    ]


@dataclass(frozen=True, slots=True)
class SizeimageComparison:
    """How the driver's `sizeimage` relates to the descriptor's buffer size.

    For an uncompressed format the two are expected to agree. A compressed
    format's real frame size is not knowable from the descriptor, so a
    divergence here is a fact about the *negotiated* stream, not a defect.
    """

    role: str
    fourcc: str
    compressed: bool
    descriptor_bytes: int
    driver_bytes: int

    @property
    def ratio(self) -> float | None:
        if self.descriptor_bytes <= 0:
            return None
        return round(self.driver_bytes / self.descriptor_bytes, 4)

    @property
    def agrees(self) -> bool:
        return self.driver_bytes == self.descriptor_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fourcc": self.fourcc,
            "compressed": self.compressed,
            "descriptor_bytes": self.descriptor_bytes,
            "driver_bytes": self.driver_bytes,
            "ratio": self.ratio,
            "agrees": self.agrees,
        }


def compare_sizeimage(plan: StreamingPlan, run: RunSummary) -> SizeimageComparison:
    return SizeimageComparison(
        role=run.role,
        fourcc=plan.fourcc,
        compressed=plan.compressed,
        descriptor_bytes=plan.max_video_frame_buffer_bytes,
        driver_bytes=run.driver_sizeimage,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_endpoint(endpoint: Endpoint) -> str:
    return (
        f"0x{endpoint.address:02x} {endpoint.direction:3s} {endpoint.transfer_type:12s} "
        f"wMaxPacketSize={endpoint.max_packet_size:5d} bInterval={endpoint.interval}"
    )


def render_report(
    config: Configuration,
    rgb_plan: StreamingPlan,
    rgb_run: RunSummary,
    depth_plan: StreamingPlan | None = None,
    depth_run: RunSummary | None = None,
) -> str:
    """Render the human-readable reconstruction."""
    out: list[str] = []
    add = out.append

    device = config.device
    add("## Device")
    if device is not None:
        add(f"  USB id                 {device.vendor_id:04x}:{device.product_id:04x}")
        add(f"  bcdUSB                 0x{device.bcd_usb:04x}")
        add(f"  bcdDevice              0x{device.bcd_device:04x}")
    add(
        f"  wTotalLength           {config.total_length} "
        f"(undecoded trailing bytes: {config.trailing_bytes})"
    )
    add(f"  bNumInterfaces         {config.num_interfaces}")
    add(f"  bMaxPower              {config.max_power_ma} mA")
    duplicates = config.duplicate_endpoint_addresses()
    if duplicates:
        for address, owners in sorted(duplicates.items()):
            add(
                f"  NOTE duplicate endpoint 0x{address:02x} declared by "
                f"alt-0 interfaces {', '.join(str(o) for o in owners)}"
            )
    add("")

    add("## Interfaces (alternate settings, endpoints)")
    for iface in config.interfaces:
        kind = (
            "VideoControl"
            if iface.is_video_control
            else "VideoStreaming"
            if iface.is_video_streaming
            else f"class 0x{iface.interface_class:02x}"
        )
        add(
            f"  iface {iface.number} alt {iface.alternate_setting}  {kind:14s} "
            f"bNumEndpoints={iface.declared_endpoints} iInterface={iface.string_index}"
        )
        for endpoint in iface.endpoints:
            add(f"      ep {_format_endpoint(endpoint)}")
        if iface.vc_header is not None:
            header = iface.vc_header
            add(
                f"      VC_HEADER bcdUVC=0x{header.bcd_uvc:04x} (UVC {header.uvc_version}) "
                f"dwClockFrequency={header.clock_frequency_hz} "
                f"streaming={list(header.streaming_interfaces)}"
            )
        if iface.vs_input_header is not None:
            vs = iface.vs_input_header
            add(
                f"      VS_INPUT_HEADER bNumFormats={vs.num_formats} "
                f"bEndpointAddress=0x{vs.endpoint_address:02x} bmInfo=0x{vs.info:02x} "
                f"bTerminalLink={vs.terminal_link} "
                f"bStillCaptureMethod={vs.still_capture_method} "
                f"bTriggerSupport={vs.trigger_support} "
                f"bmaControls={[f'0x{c:02x}' for c in vs.controls]}"
            )
        for fmt in iface.formats:
            add(
                f"      format {fmt.format_index}: {fmt.fourcc} "
                f"({'compressed' if fmt.compressed else 'uncompressed'}"
                f"{'' if fmt.bits_per_pixel is None else f', {fmt.bits_per_pixel} bpp'}) "
                f"bDefaultFrameIndex={fmt.default_frame_index}"
            )
            for frame in fmt.frames:
                rates = ", ".join(f"{r:g}" for r in frame.frame_rates_hz)
                add(
                    f"          frame {frame.frame_index}: {frame.width}x{frame.height} "
                    f"dwMaxVideoFrameBufferSize={frame.max_video_frame_buffer_bytes} "
                    f"dwMaxBitRate={frame.max_bit_rate} "
                    f"default={frame.default_frame_rate_hz} Hz rates=[{rates}]"
                )
    add("")

    add("## Descriptor-fixed streaming plan (RGB)")
    for key, value in rgb_plan.as_dict().items():
        add(f"  {key:34s} {value}")
    add("")

    if depth_plan is not None:
        add("## Descriptor-fixed streaming plan (depth, the control case)")
        for key, value in depth_plan.as_dict().items():
            add(f"  {key:34s} {value}")
        add("")

    add("## Measured runs")
    rows: list[tuple[str, RunSummary]] = [("rgb", rgb_run)]
    if depth_run is not None:
        rows.append(("depth", depth_run))
    for label, run in rows:
        add(
            f"  {label:6s} {run.status:16s} exit={run.exit_code} "
            f"node={run.node_path} iface={run.interface_number} "
            f"{run.fourcc} {run.width}x{run.height} sizeimage={run.driver_sizeimage} "
            f"streamon={run.reached_streamon} dqbuf={run.dequeued_bytes} "
            f"poll={run.poll_duration_s}"
        )
    add("")

    add("## sizeimage vs descriptor")
    comparisons = [compare_sizeimage(rgb_plan, rgb_run)]
    if depth_plan is not None and depth_run is not None:
        comparisons.append(compare_sizeimage(depth_plan, depth_run))
    for comparison in comparisons:
        add(
            f"  {comparison.role:6s} {comparison.fourcc:5s} "
            f"{'compressed  ' if comparison.compressed else 'uncompressed'} "
            f"descriptor={comparison.descriptor_bytes} driver={comparison.driver_bytes} "
            f"ratio={comparison.ratio} agrees={comparison.agrees}"
        )
    add("")

    add("## V4L2 operations issued by owned code during the RGB attempt")
    for row in negotiation_matrix(rgb_run):
        mark = "issued" if row["issued_by_owned_code"] else "NEVER ISSUED"
        add(f"  {mark:12s} {row['ioctl']:28s} {row['purpose']}")
    add("")

    add("## Bandwidth")
    add(
        f"  single-endpoint high-speed bulk ceiling  "
        f"{HS_BULK_CEILING_BYTES_PER_SECOND} B/s "
        f"({HS_BULK_TRANSACTIONS_PER_MICROFRAME} x {HS_BULK_MAX_PACKET_BYTES} B "
        f"per 125 us microframe)"
    )
    add(
        f"  RGB dwMaxBitRate                         {rgb_plan.max_bit_rate} bit/s "
        f"= {rgb_plan.max_bit_rate_bytes_per_second:.0f} B/s "
        f"({rgb_plan.bulk_ceiling_utilisation} of the ceiling)"
    )
    add(
        f"  RGB alternate settings                   {list(rgb_plan.alternate_settings)} "
        f"-> alt-setting selection required: {rgb_plan.requires_alt_setting_selection}"
    )
    return "\n".join(out)


def build_json(
    config: Configuration,
    rgb_plan: StreamingPlan,
    rgb_run: RunSummary,
    depth_plan: StreamingPlan | None,
    depth_run: RunSummary | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "configuration": config.as_dict(),
        "rgb_plan": rgb_plan.as_dict(),
        "rgb_run": rgb_run.as_dict(),
        "rgb_sizeimage": compare_sizeimage(rgb_plan, rgb_run).as_dict(),
        "rgb_negotiation_matrix": negotiation_matrix(rgb_run),
    }
    if depth_plan is not None:
        payload["depth_plan"] = depth_plan.as_dict()
    if depth_run is not None:
        payload["depth_run"] = depth_run.as_dict()
        if depth_plan is not None:
            payload["depth_sizeimage"] = compare_sizeimage(depth_plan, depth_run).as_dict()
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    fixtures = repo_root / "tests" / "fixtures"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=fixtures / "mini-config-descriptor.bin",
        help="raw configuration-descriptor blob",
    )
    parser.add_argument(
        "--rgb-attempt",
        type=Path,
        default=fixtures / "scanner-011-rgb-attempt.json",
        help="frozen RGB attempt JSON",
    )
    parser.add_argument(
        "--depth-capture",
        type=Path,
        default=fixtures / "scanner-009-depth-capture.json",
        help="frozen depth capture JSON (the control case)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    config = decode_configuration(args.descriptor.read_bytes())
    rgb_run = summarize_run(_load_json(args.rgb_attempt))
    rgb_plan = streaming_plan(config, 3, rgb_run.fourcc, rgb_run.width, rgb_run.height)

    depth_run: RunSummary | None = None
    depth_plan: StreamingPlan | None = None
    if args.depth_capture.exists():
        depth_run = summarize_run(_load_json(args.depth_capture))
        depth_plan = streaming_plan(config, 1, depth_run.fourcc, depth_run.width, depth_run.height)

    if args.json:
        json.dump(
            build_json(config, rgb_plan, rgb_run, depth_plan, depth_run),
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_report(config, rgb_plan, rgb_run, depth_plan, depth_run) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
