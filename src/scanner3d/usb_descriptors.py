"""Parser for the raw USB configuration descriptor exposed by sysfs.

Reading ``/sys/bus/usb/devices/<X>/descriptors`` lets us report the modes the
scanner *advertises* without opening a ``/dev/videoN`` node, without root, and
without disturbing the ``uvcvideo`` driver. Every parser here is deliberately
tolerant: a truncated or corrupt blob yields fewer records, never an exception.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field

# USB / UVC descriptor type constants.
DT_INTERFACE = 0x04
DT_INTERFACE_ASSOCIATION = 0x0B
CS_INTERFACE = 0x24

# UVC video-streaming descriptor subtypes.
VS_FORMAT_UNCOMPRESSED = 0x04
VS_FRAME_UNCOMPRESSED = 0x05
VS_FORMAT_MJPEG = 0x06
VS_FRAME_MJPEG = 0x07

# UVC video-control descriptor subtypes. These numerically collide with the
# VS_* subtypes above (e.g. 0x06 is both VS_FORMAT_MJPEG and
# VC_EXTENSION_UNIT); which table applies depends on the enclosing
# interface's subclass, never on the subtype value alone.
VC_HEADER = 0x01
VC_INPUT_TERMINAL = 0x02
VC_OUTPUT_TERMINAL = 0x03
VC_SELECTOR_UNIT = 0x04
VC_PROCESSING_UNIT = 0x05
VC_EXTENSION_UNIT = 0x06

_VC_SUBTYPE_NAMES = {
    VC_HEADER: "header",
    VC_INPUT_TERMINAL: "input_terminal",
    VC_OUTPUT_TERMINAL: "output_terminal",
    VC_SELECTOR_UNIT: "selector_unit",
    VC_PROCESSING_UNIT: "processing_unit",
    VC_EXTENSION_UNIT: "extension_unit",
}

USB_CLASS_VIDEO = 0x0E
SUBCLASS_VIDEO_CONTROL = 0x01
SUBCLASS_VIDEO_STREAMING = 0x02

# UVC frame intervals are expressed in 100 ns ticks.
_TICKS_PER_SECOND = 10_000_000

# The trailing 12 bytes shared by every "FourCC as a GUID" UVC format identifier
# (``XXXX-0000-0010-8000-00AA00389B71``); the first four bytes are the FourCC.
_UVC_GUID_SUFFIX = bytes.fromhex("00001000800000aa00389b71")


@dataclass(frozen=True, slots=True)
class FrameMode:
    """A single advertised ``width x height`` mode and its frame rates."""

    width: int
    height: int
    frame_rates_hz: tuple[float, ...]
    default_frame_rate_hz: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_rates_hz": list(self.frame_rates_hz),
            "default_frame_rate_hz": self.default_frame_rate_hz,
        }


@dataclass(frozen=True, slots=True)
class StreamFormat:
    """An advertised UVC format (a FourCC or ``MJPG``) and its frame modes."""

    fourcc: str
    compression: str
    bits_per_pixel: int | None
    modes: tuple[FrameMode, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "fourcc": self.fourcc,
            "compression": self.compression,
            "bits_per_pixel": self.bits_per_pixel,
            "modes": [m.as_dict() for m in self.modes],
        }


@dataclass(frozen=True, slots=True)
class VideoControlEntity:
    """One class-specific VideoControl descriptor (terminal, unit, or header).

    Every field is best-effort: a record too short to hold a field this
    parser wants leaves that field at its "absent" value and sets
    ``truncated``, rather than raising.
    """

    interface_number: int
    association_first_interface: int | None
    entity_id: int
    descriptor_subtype: int
    subtype_name: str
    guid: str | None
    terminal_type: int | None
    source_ids: tuple[int, ...]
    num_controls: int | None
    control_bitmap: str | None
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "interface_number": self.interface_number,
            "association_first_interface": self.association_first_interface,
            "entity_id": self.entity_id,
            "descriptor_subtype": self.descriptor_subtype,
            "subtype_name": self.subtype_name,
            "guid": self.guid,
            "terminal_type": self.terminal_type,
            "source_ids": list(self.source_ids),
            "num_controls": self.num_controls,
            "control_bitmap": self.control_bitmap,
            "truncated": self.truncated,
        }

    @property
    def is_extension_unit(self) -> bool:
        return self.descriptor_subtype == VC_EXTENSION_UNIT


@dataclass(frozen=True, slots=True)
class InterfaceDescriptor:
    """One USB interface, with any UVC formats found in its descriptor run."""

    number: int
    interface_class: int
    interface_subclass: int
    formats: tuple[StreamFormat, ...] = field(default=())
    entities: tuple[VideoControlEntity, ...] = field(default=())

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


@dataclass(frozen=True, slots=True)
class FunctionAssociation:
    """An Interface Association Descriptor grouping control + streaming."""

    first_interface: int
    interface_count: int
    function_class: int

    @property
    def interfaces(self) -> range:
        return range(self.first_interface, self.first_interface + self.interface_count)


@dataclass(frozen=True, slots=True)
class ParsedDescriptors:
    """The subset of the configuration descriptor this project cares about."""

    interfaces: tuple[InterfaceDescriptor, ...]
    associations: tuple[FunctionAssociation, ...]
    truncated: bool

    @property
    def entities(self) -> tuple[VideoControlEntity, ...]:
        """Every VideoControl entity across all interfaces, in descriptor order."""
        flattened: list[VideoControlEntity] = []
        for iface in self.interfaces:
            flattened.extend(iface.entities)
        return tuple(flattened)

    @property
    def extension_units(self) -> tuple[VideoControlEntity, ...]:
        return tuple(e for e in self.entities if e.is_extension_unit)

    def entity(self, interface_number: int, entity_id: int) -> VideoControlEntity | None:
        for e in self.entities:
            if e.interface_number == interface_number and e.entity_id == entity_id:
                return e
        return None

    def association_for(self, interface_number: int) -> FunctionAssociation | None:
        for assoc in self.associations:
            if interface_number in assoc.interfaces:
                return assoc
        return None

    def formats_for_function(self, first_interface: int) -> tuple[StreamFormat, ...]:
        """Collect formats from every streaming interface inside a function."""
        assoc = self.association_for(first_interface)
        members = assoc.interfaces if assoc else range(first_interface, first_interface + 1)
        formats: list[StreamFormat] = []
        for iface in self.interfaces:
            if iface.number in members and iface.is_video_streaming:
                formats.extend(iface.formats)
        return tuple(formats)


def iter_descriptors(blob: bytes) -> list[tuple[int, bytes]]:
    """Split ``blob`` into ``(descriptor_type, bytes)`` records.

    Stops cleanly at the first zero-length or truncated record instead of
    raising, so malformed sysfs data degrades to partial results.
    """
    out: list[tuple[int, bytes]] = []
    offset = 0
    while offset + 2 <= len(blob):
        length = blob[offset]
        if length < 2 or offset + length > len(blob):
            break
        out.append((blob[offset + 1], blob[offset : offset + length]))
        offset += length
    return out


def _guid_to_fourcc(guid: bytes) -> str:
    """Map a UVC format GUID back to its FourCC, or hex if it is not one."""
    if len(guid) != 16:
        return guid.hex()
    if guid[4:] == _UVC_GUID_SUFFIX:
        fourcc = guid[:4].decode("ascii", errors="replace").strip("\x00").strip()
        if fourcc:
            return fourcc
    return str(uuid.UUID(bytes_le=guid)).upper()


def _interval_to_hz(ticks: int) -> float | None:
    if ticks <= 0:
        return None
    return round(_TICKS_PER_SECOND / ticks, 3)


def _parse_frame(data: bytes) -> FrameMode | None:
    """Parse a VS_FRAME_UNCOMPRESSED / VS_FRAME_MJPEG descriptor.

    Both share an identical layout for the fields we read.
    """
    if len(data) < 26:
        return None
    width, height = struct.unpack_from("<HH", data, 5)
    default_interval = struct.unpack_from("<I", data, 21)[0]
    interval_type = data[25]
    rates: list[float] = []
    if interval_type == 0:
        # Continuous: min / max / step triple.
        if len(data) >= 38:
            min_i, max_i, step = struct.unpack_from("<III", data, 26)
            for ticks in (max_i, min_i):  # max interval == slowest rate
                hz = _interval_to_hz(ticks)
                if hz is not None and hz not in rates:
                    rates.append(hz)
            del step
    else:
        for index in range(interval_type):
            start = 26 + index * 4
            if start + 4 > len(data):
                break
            hz = _interval_to_hz(struct.unpack_from("<I", data, start)[0])
            if hz is not None and hz not in rates:
                rates.append(hz)
    return FrameMode(
        width=width,
        height=height,
        frame_rates_hz=tuple(sorted(rates)),
        default_frame_rate_hz=_interval_to_hz(default_interval),
    )


def _parse_vc_entity(
    subtype: int,
    data: bytes,
    interface_number: int,
    association_first_interface: int | None,
) -> VideoControlEntity:
    """Parse one class-specific VideoControl descriptor.

    Tolerant of short/malformed records: any field this parser wanted but
    couldn't reach is left at its "absent" value and ``truncated`` is set,
    never an exception.
    """
    length = len(data)
    truncated = False
    entity_id = 0
    guid: str | None = None
    terminal_type: int | None = None
    source_ids: tuple[int, ...] = ()
    num_controls: int | None = None
    control_bitmap: str | None = None
    subtype_name = _VC_SUBTYPE_NAMES.get(subtype, f"unknown_0x{subtype:02x}")

    if subtype == VC_HEADER:
        pass
    elif subtype == VC_INPUT_TERMINAL:
        if length >= 4:
            entity_id = data[3]
        else:
            truncated = True
        if length >= 6:
            terminal_type = struct.unpack_from("<H", data, 4)[0]
        else:
            truncated = True
    elif subtype == VC_OUTPUT_TERMINAL:
        if length >= 4:
            entity_id = data[3]
        else:
            truncated = True
        if length >= 6:
            terminal_type = struct.unpack_from("<H", data, 4)[0]
        else:
            truncated = True
        if length >= 8:
            source_ids = (data[7],)
        else:
            truncated = True
    elif subtype == VC_SELECTOR_UNIT:
        if length >= 4:
            entity_id = data[3]
        else:
            truncated = True
        if length >= 5:
            num_pins = data[4]
            pins = data[5 : 5 + num_pins]
            source_ids = tuple(pins)
            if len(pins) < num_pins:
                truncated = True
        else:
            truncated = True
    elif subtype == VC_PROCESSING_UNIT:
        if length >= 4:
            entity_id = data[3]
        else:
            truncated = True
        if length >= 5:
            source_ids = (data[4],)
        else:
            truncated = True
        if length >= 8:
            control_size = data[7]
            bitmap = data[8 : 8 + control_size]
            control_bitmap = bitmap.hex()
            if len(bitmap) < control_size:
                truncated = True
        else:
            truncated = True
    elif subtype == VC_EXTENSION_UNIT:
        if length >= 4:
            entity_id = data[3]
        else:
            truncated = True
        if length >= 20:
            guid = str(uuid.UUID(bytes_le=data[4:20])).upper()
        else:
            truncated = True
        if length >= 21:
            num_controls = data[20]
        else:
            truncated = True
        if length >= 22:
            num_pins = data[21]
            pins = data[22 : 22 + num_pins]
            source_ids = tuple(pins)
            if len(pins) < num_pins:
                truncated = True
            else:
                control_size_offset = 22 + num_pins
                if length > control_size_offset:
                    control_size = data[control_size_offset]
                    bitmap_start = control_size_offset + 1
                    bitmap = data[bitmap_start : bitmap_start + control_size]
                    control_bitmap = bitmap.hex()
                    if len(bitmap) < control_size:
                        truncated = True
                else:
                    truncated = True
        else:
            truncated = True
    else:
        if length >= 4:
            entity_id = data[3]

    return VideoControlEntity(
        interface_number=interface_number,
        association_first_interface=association_first_interface,
        entity_id=entity_id,
        descriptor_subtype=subtype,
        subtype_name=subtype_name,
        guid=guid,
        terminal_type=terminal_type,
        source_ids=source_ids,
        num_controls=num_controls,
        control_bitmap=control_bitmap,
        truncated=truncated,
    )


def parse(blob: bytes) -> ParsedDescriptors:
    """Parse a raw configuration-descriptor blob into interfaces and formats."""
    records = iter_descriptors(blob)
    truncated = bool(blob) and sum(len(r) for _, r in records) < len(blob.rstrip(b"\x00"))

    interfaces: list[InterfaceDescriptor] = []
    associations: list[FunctionAssociation] = []

    current: dict[str, int] | None = None
    current_assoc_first: int | None = None
    formats: list[StreamFormat] = []
    entities: list[VideoControlEntity] = []
    pending: dict[str, object] | None = None
    modes: list[FrameMode] = []

    def flush_format() -> None:
        nonlocal pending, modes
        if pending is not None:
            formats.append(
                StreamFormat(
                    fourcc=str(pending["fourcc"]),
                    compression=str(pending["compression"]),
                    bits_per_pixel=pending["bits_per_pixel"],  # type: ignore[arg-type]
                    modes=tuple(modes),
                )
            )
        pending = None
        modes = []

    def flush_interface() -> None:
        nonlocal current, current_assoc_first, formats, entities
        flush_format()
        if current is not None:
            interfaces.append(
                InterfaceDescriptor(
                    number=current["number"],
                    interface_class=current["cls"],
                    interface_subclass=current["subcls"],
                    formats=tuple(formats),
                    entities=tuple(entities),
                )
            )
        current = None
        current_assoc_first = None
        formats = []
        entities = []

    for dtype, data in records:
        if dtype == DT_INTERFACE_ASSOCIATION and len(data) >= 8:
            associations.append(
                FunctionAssociation(
                    first_interface=data[2],
                    interface_count=data[3],
                    function_class=data[4],
                )
            )
        elif dtype == DT_INTERFACE and len(data) >= 9:
            flush_interface()
            number = data[2]
            current = {"number": number, "cls": data[5], "subcls": data[6]}
            for assoc in associations:
                if number in assoc.interfaces:
                    current_assoc_first = assoc.first_interface
                    break
        elif dtype == CS_INTERFACE and len(data) >= 3 and current is not None:
            is_video_streaming = (
                current["cls"] == USB_CLASS_VIDEO and current["subcls"] == SUBCLASS_VIDEO_STREAMING
            )
            is_video_control = (
                current["cls"] == USB_CLASS_VIDEO and current["subcls"] == SUBCLASS_VIDEO_CONTROL
            )
            subtype = data[2]
            if is_video_streaming:
                if subtype == VS_FORMAT_UNCOMPRESSED and len(data) >= 26:
                    flush_format()
                    pending = {
                        "fourcc": _guid_to_fourcc(data[5:21]),
                        "compression": "uncompressed",
                        "bits_per_pixel": data[21],
                    }
                elif subtype == VS_FORMAT_MJPEG and len(data) >= 11:
                    flush_format()
                    pending = {"fourcc": "MJPG", "compression": "mjpeg", "bits_per_pixel": None}
                elif subtype in (VS_FRAME_UNCOMPRESSED, VS_FRAME_MJPEG) and pending is not None:
                    frame = _parse_frame(data)
                    if frame is not None:
                        modes.append(frame)
            elif is_video_control:
                # VideoControl reuses the VS_* subtype numbers for unrelated
                # descriptors (0x06 is VC_EXTENSION_UNIT, not a MJPEG format).
                entities.append(
                    _parse_vc_entity(subtype, data, current["number"], current_assoc_first)
                )

    flush_interface()
    return ParsedDescriptors(
        interfaces=tuple(interfaces),
        associations=tuple(associations),
        truncated=truncated,
    )
