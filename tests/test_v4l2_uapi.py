"""Tests for the V4L2 UAPI transcription and the pure packing/parsing layer.

``scanner3d.v4l2_capture`` hand-transcribes ``<linux/videodev2.h>`` into
``struct`` formats and ``ioctl`` request numbers, and the module's own
docstring is explicit that this transcription "does not fail loudly" when
wrong: a wrong struct size or offset produces a *different, valid-looking*
ioctl number or scribbles into the wrong bytes of a real kernel buffer. The
point of this file is to make that failure loud in CI instead, by diffing the
transcription against ``tests/uapi/v4l2_layout.c`` - a small C program compiled
against whatever real kernel UAPI headers are installed on the host - field by
field, rather than trusting either side by inspection.

The lifecycle that issues these ioctls in order (open, negotiate, stream,
dequeue, clean up) is exercised elsewhere; nothing here opens a video node, a
sysfs file outside a synthesized tree, or touches real hardware.
"""

from __future__ import annotations

import ast
import json
import shutil
import struct
import subprocess
import tempfile
import tokenize
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import scanner3d.v4l2_capture as v4l2_capture_module
from scanner3d.discovery import find_scanners
from scanner3d.errors import DeviceAbsentError, V4l2CaptureNotAllowed
from scanner3d.v4l2_capture import (
    ALLOWED_IOCTLS,
    BUFFER_BYTES,
    BUFFER_FORMAT,
    BUFTYPE_BYTES,
    BUFTYPE_FORMAT,
    CAPABILITY_BYTES,
    CAPABILITY_FORMAT,
    EXPECTED_FRAME_BYTES,
    EXPECTED_STRUCT_BYTES,
    FORMAT_BYTES,
    FORMAT_FORMAT,
    FORMAT_UNION_BYTES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_BUFFER_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    MIN_BYTES_PER_LINE,
    PIX_FORMAT_BYTES,
    PIX_FORMAT_FORMAT,
    REQUESTBUFFERS_BYTES,
    REQUESTBUFFERS_FORMAT,
    REQUESTED_BUFFERS,
    V4L2_BUF_FLAG_DONE,
    V4L2_BUF_FLAG_ERROR,
    V4L2_BUF_FLAG_LAST,
    V4L2_BUF_FLAG_MAPPED,
    V4L2_BUF_FLAG_QUEUED,
    V4L2_BUF_FLAG_TIMESTAMP_COPY,
    V4L2_BUF_FLAG_TIMESTAMP_MASK,
    V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
    V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN,
    V4L2_BUF_FLAG_TSTAMP_SRC_EOF,
    V4L2_BUF_FLAG_TSTAMP_SRC_MASK,
    V4L2_BUF_FLAG_TSTAMP_SRC_SOE,
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_DEVICE_CAPS,
    V4L2_CAP_META_CAPTURE,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_FIELD_NONE,
    V4L2_MEMORY_MMAP,
    V4L2_PIX_FMT_Y16,
    BufferInfo,
    PixelFormat,
    _format_refusal,
    _mapping_refusal,
    ensure_layout_intact,
    fourcc_to_int,
    int_to_fourcc,
    pack_buf_type,
    pack_buffer,
    pack_capability,
    pack_format,
    pack_requestbuffers,
    parse_buffer,
    parse_capability,
    parse_format,
    parse_requestbuffers,
    refusal_reason,
    resolve_capture_node,
    unpack_rows,
)
from scanner3d.v4l2_controls import (
    ALLOWED_CONTROL_IOCTLS,
    CONTROL_FORMAT,
    EXPECTED_CONTROL_STRUCT_BYTES,
    QUERYCTRL_FORMAT,
    V4L2_CID_AUTO_WHITE_BALANCE,
    V4L2_CID_CAMERA_CLASS_BASE,
    V4L2_CID_EXPOSURE_ABSOLUTE,
    V4L2_CID_EXPOSURE_AUTO,
    V4L2_CID_EXPOSURE_AUTO_PRIORITY,
    V4L2_CID_GAIN,
    V4L2_CID_WHITE_BALANCE_TEMPERATURE,
    V4L2_CTRL_FLAG_DISABLED,
    V4L2_CTRL_FLAG_GRABBED,
    V4L2_CTRL_FLAG_INACTIVE,
    V4L2_CTRL_FLAG_READ_ONLY,
    V4L2_CTRL_FLAG_SLIDER,
    V4L2_CTRL_FLAG_UPDATE,
    V4L2_CTRL_FLAG_VOLATILE,
    V4L2_CTRL_FLAG_WRITE_ONLY,
    V4L2_CTRL_TYPE_BOOLEAN,
    V4L2_CTRL_TYPE_INTEGER,
    V4L2_CTRL_TYPE_INTEGER_MENU,
    V4L2_CTRL_TYPE_MENU,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder

_C_SOURCE = Path(__file__).parent / "uapi" / "v4l2_layout.c"

# --- the C ground truth ---------------------------------------------------


@pytest.fixture(scope="session")
def c_layout() -> dict[str, Any]:
    """Compile and run the C ground-truth program, parsing its JSON report.

    Skipped rather than failed when no compiler is available: the rest of this
    file's tests (packing, parsing, refusal logic) do not depend on it and
    still catch regressions in the pure-Python layer on a machine with no
    ``cc``.
    """
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available to cross-check the V4L2 UAPI transcription")
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "v4l2_layout"
        compiled = subprocess.run(
            [cc, "-O0", "-Wall", "-Wextra", "-o", str(binary), str(_C_SOURCE)],
            capture_output=True,
            text=True,
        )
        assert compiled.returncode == 0, f"compiling {_C_SOURCE} failed:\n{compiled.stderr}"
        run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    result: dict[str, Any] = json.loads(run.stdout)
    return result


def _iter_field_offsets(fmt: str) -> Iterator[tuple[int, int]]:
    """Yield ``(field_index, byte_offset)`` for every field ``fmt`` packs.

    A "field" is one value ``struct.unpack`` would return: a counted string
    (``16s``) is a single field, an ``x`` pad byte is not a field at all, and a
    count in front of any other code (``4B``, ``12I``) means that many
    one-value fields back to back, not one wide one - which is exactly the
    ambiguity a hand-transcribed offset gets wrong first.
    """
    assert fmt[0] in "=<>!", f"format {fmt!r} must fix byte order explicitly"
    offset = 0
    index = 0
    count_digits = ""
    for ch in fmt[1:]:
        if ch.isdigit():
            count_digits += ch
            continue
        count = int(count_digits) if count_digits else 1
        count_digits = ""
        if ch == "x":
            offset += count
        elif ch == "s":
            yield index, offset
            index += 1
            offset += count
        else:
            size = struct.calcsize(ch)
            for _ in range(count):
                yield index, offset
                index += 1
                offset += size


def _field_offset_list(fmt: str) -> list[int]:
    return [offset for _, offset in _iter_field_offsets(fmt)]


def test_field_offset_helper_handles_padding_counts_and_strings() -> None:
    """The helper the offset tests below depend on, checked against hand-worked cases."""
    assert list(_iter_field_offsets("=I4xI")) == [(0, 0), (1, 8)]
    assert list(_iter_field_offsets("=16s3I")) == [(0, 0), (1, 16), (2, 20), (3, 24)]
    assert _field_offset_list(BUFTYPE_FORMAT) == [0]


@pytest.mark.parametrize(
    ("python_name", "fmt", "c_key"),
    [
        ("v4l2_capability", CAPABILITY_FORMAT, "struct v4l2_capability"),
        ("v4l2_pix_format", PIX_FORMAT_FORMAT, "struct v4l2_pix_format"),
        ("v4l2_format", FORMAT_FORMAT, "struct v4l2_format"),
        ("v4l2_requestbuffers", REQUESTBUFFERS_FORMAT, "struct v4l2_requestbuffers"),
        ("v4l2_buffer", BUFFER_FORMAT, "struct v4l2_buffer"),
    ],
)
def test_struct_sizes_match_c_sizeof(
    c_layout: dict[str, Any], python_name: str, fmt: str, c_key: str
) -> None:
    real = c_layout["sizeof"][c_key]
    actual = struct.calcsize(fmt)
    assert actual == real, (
        f"struct.calcsize for {python_name} is {actual} bytes; the host's real "
        f"sizeof({c_key}) is {real}"
    )
    assert EXPECTED_STRUCT_BYTES[python_name] == real, (
        f"EXPECTED_STRUCT_BYTES[{python_name!r}] is {EXPECTED_STRUCT_BYTES[python_name]}; "
        f"the host's real sizeof({c_key}) is {real}"
    )


def test_buf_type_size_is_the_bare_int_ioctl_size() -> None:
    """No struct backs this one; it is cross-checked transitively via the ioctl numbers."""
    assert struct.calcsize(BUFTYPE_FORMAT) == 4
    assert EXPECTED_STRUCT_BYTES["buf_type"] == 4


@pytest.mark.parametrize(
    ("python_name", "fmt", "c_key"),
    [
        ("v4l2_queryctrl", QUERYCTRL_FORMAT, "struct v4l2_queryctrl"),
        ("v4l2_control", CONTROL_FORMAT, "struct v4l2_control"),
    ],
)
def test_control_struct_sizes_match_c(
    c_layout: dict[str, Any], python_name: str, fmt: str, c_key: str
) -> None:
    real = c_layout["sizeof"][c_key]
    assert struct.calcsize(fmt) == real
    assert EXPECTED_CONTROL_STRUCT_BYTES[python_name] == real


def test_control_struct_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    offsets = c_layout["offsetof"]
    assert _field_offset_list(QUERYCTRL_FORMAT) == [
        offsets["v4l2_queryctrl.id"],
        offsets["v4l2_queryctrl.type"],
        offsets["v4l2_queryctrl.name"],
        offsets["v4l2_queryctrl.minimum"],
        offsets["v4l2_queryctrl.maximum"],
        offsets["v4l2_queryctrl.step"],
        offsets["v4l2_queryctrl.default_value"],
        offsets["v4l2_queryctrl.flags"],
        offsets["v4l2_queryctrl.reserved"],
    ]
    assert _field_offset_list(CONTROL_FORMAT) == [
        offsets["v4l2_control.id"],
        offsets["v4l2_control.value"],
    ]


def test_control_ioctls_and_constants_match_c(c_layout: dict[str, Any]) -> None:
    for name, value in ALLOWED_CONTROL_IOCTLS.items():
        assert value == c_layout["ioctl"][name]["dec"]
    checks = {
        "V4L2_CID_GAIN": V4L2_CID_GAIN,
        "V4L2_CID_CAMERA_CLASS_BASE": V4L2_CID_CAMERA_CLASS_BASE,
        "V4L2_CID_EXPOSURE_AUTO": V4L2_CID_EXPOSURE_AUTO,
        "V4L2_CID_EXPOSURE_ABSOLUTE": V4L2_CID_EXPOSURE_ABSOLUTE,
        "V4L2_CID_EXPOSURE_AUTO_PRIORITY": V4L2_CID_EXPOSURE_AUTO_PRIORITY,
        "V4L2_CID_AUTO_WHITE_BALANCE": V4L2_CID_AUTO_WHITE_BALANCE,
        "V4L2_CID_WHITE_BALANCE_TEMPERATURE": V4L2_CID_WHITE_BALANCE_TEMPERATURE,
        "V4L2_CTRL_TYPE_INTEGER": V4L2_CTRL_TYPE_INTEGER,
        "V4L2_CTRL_TYPE_BOOLEAN": V4L2_CTRL_TYPE_BOOLEAN,
        "V4L2_CTRL_TYPE_MENU": V4L2_CTRL_TYPE_MENU,
        "V4L2_CTRL_TYPE_INTEGER_MENU": V4L2_CTRL_TYPE_INTEGER_MENU,
        "V4L2_CTRL_FLAG_DISABLED": V4L2_CTRL_FLAG_DISABLED,
        "V4L2_CTRL_FLAG_GRABBED": V4L2_CTRL_FLAG_GRABBED,
        "V4L2_CTRL_FLAG_READ_ONLY": V4L2_CTRL_FLAG_READ_ONLY,
        "V4L2_CTRL_FLAG_UPDATE": V4L2_CTRL_FLAG_UPDATE,
        "V4L2_CTRL_FLAG_INACTIVE": V4L2_CTRL_FLAG_INACTIVE,
        "V4L2_CTRL_FLAG_SLIDER": V4L2_CTRL_FLAG_SLIDER,
        "V4L2_CTRL_FLAG_WRITE_ONLY": V4L2_CTRL_FLAG_WRITE_ONLY,
        "V4L2_CTRL_FLAG_VOLATILE": V4L2_CTRL_FLAG_VOLATILE,
    }
    for name, value in checks.items():
        assert value == c_layout["constants"][name]["dec"]


def test_v4l2_capability_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    offsets = c_layout["offsetof"]
    field_order = [
        "v4l2_capability.driver",
        "v4l2_capability.card",
        "v4l2_capability.bus_info",
        "v4l2_capability.version",
        "v4l2_capability.capabilities",
        "v4l2_capability.device_caps",
        "v4l2_capability.reserved",
    ]
    py_offsets = _field_offset_list(CAPABILITY_FORMAT)
    assert len(py_offsets) == len(field_order)
    for name, py_offset in zip(field_order, py_offsets, strict=True):
        assert py_offset == offsets[name], (
            f"{name}: the module packs it at offset {py_offset}, the C header at {offsets[name]}"
        )


def test_v4l2_pix_format_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    offsets = c_layout["offsetof"]
    field_order = [
        "v4l2_pix_format.width",
        "v4l2_pix_format.height",
        "v4l2_pix_format.pixelformat",
        "v4l2_pix_format.field",
        "v4l2_pix_format.bytesperline",
        "v4l2_pix_format.sizeimage",
        "v4l2_pix_format.colorspace",
        "v4l2_pix_format.priv",
        "v4l2_pix_format.flags",
        "v4l2_pix_format.ycbcr_enc",
        "v4l2_pix_format.quantization",
        "v4l2_pix_format.xfer_func",
    ]
    py_offsets = _field_offset_list(PIX_FORMAT_FORMAT)
    assert len(py_offsets) == len(field_order)
    for name, py_offset in zip(field_order, py_offsets, strict=True):
        assert py_offset == offsets[name], (
            f"{name}: the module packs it at offset {py_offset}, the C header at {offsets[name]}"
        )


def test_v4l2_format_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    offsets = c_layout["offsetof"]
    py_offsets = _field_offset_list(FORMAT_FORMAT)
    assert len(py_offsets) == 2
    type_offset, fmt_offset = py_offsets
    assert type_offset == offsets["v4l2_format.type"]
    assert fmt_offset == offsets["v4l2_format.fmt"]
    # pack_format/parse_format treat the union as starting with the pix_format
    # struct at relative offset 0; that is only correct because the real
    # union's "pix" member starts at the same offset as the union itself.
    assert offsets["v4l2_format.fmt.pix"] == offsets["v4l2_format.fmt"]


def test_v4l2_requestbuffers_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    offsets = c_layout["offsetof"]
    field_order = [
        "v4l2_requestbuffers.count",
        "v4l2_requestbuffers.type",
        "v4l2_requestbuffers.memory",
        "v4l2_requestbuffers.capabilities",
        "v4l2_requestbuffers.flags",
        "v4l2_requestbuffers.reserved",
    ]
    py_offsets = _field_offset_list(REQUESTBUFFERS_FORMAT)
    assert len(py_offsets) == len(field_order)
    for name, py_offset in zip(field_order, py_offsets, strict=True):
        assert name in offsets, f"the host's kernel headers do not report offsetof({name})"
        assert py_offset == offsets[name], (
            f"{name}: the module packs it at offset {py_offset}, the C header at {offsets[name]}"
        )


def test_v4l2_buffer_field_offsets_match_c(c_layout: dict[str, Any]) -> None:
    """The big one: every scalar field, the timestamp union and the timecode struct.

    ``timestamp`` and ``timecode`` are nested C structs the Python format packs
    as raw field sequences (``qq`` and ``4B4s``), so their absolute offsets are
    reconstructed as ``offsetof(v4l2_buffer, <struct>) + offsetof(<struct>, <field>)``
    rather than looked up directly.
    """
    offsets = c_layout["offsetof"]
    timeval_bytes = c_layout["sizeof"]["struct timeval"]
    assert timeval_bytes == 16, (
        "this transcription assumes an 8-byte time_t and an 8-byte suseconds_t (LP64); "
        f"the host's struct timeval is {timeval_bytes} bytes"
    )
    py_offsets = _field_offset_list(BUFFER_FORMAT)
    assert len(py_offsets) == 20

    (
        index_off,
        type_off,
        bytesused_off,
        flags_off,
        field_off,
        ts_sec_off,
        ts_usec_off,
        tc_type_off,
        tc_flags_off,
        tc_frames_off,
        tc_seconds_off,
        tc_minutes_off,
        tc_hours_off,
        tc_userbits_off,
        sequence_off,
        memory_off,
        m_off,
        length_off,
        reserved2_off,
        request_fd_off,
    ) = py_offsets

    assert index_off == offsets["v4l2_buffer.index"]
    assert type_off == offsets["v4l2_buffer.type"]
    assert bytesused_off == offsets["v4l2_buffer.bytesused"]
    assert flags_off == offsets["v4l2_buffer.flags"]
    assert field_off == offsets["v4l2_buffer.field"]

    assert ts_sec_off == offsets["v4l2_buffer.timestamp"]
    assert ts_usec_off == offsets["v4l2_buffer.timestamp"] + 8

    timecode_base = offsets["v4l2_buffer.timecode"]
    assert tc_type_off == timecode_base + offsets["v4l2_timecode.type"]
    assert tc_flags_off == timecode_base + offsets["v4l2_timecode.flags"]
    assert tc_frames_off == timecode_base + offsets["v4l2_timecode.frames"]
    assert tc_seconds_off == timecode_base + offsets["v4l2_timecode.seconds"]
    assert tc_minutes_off == timecode_base + offsets["v4l2_timecode.minutes"]
    assert tc_hours_off == timecode_base + offsets["v4l2_timecode.hours"]
    assert tc_userbits_off == timecode_base + offsets["v4l2_timecode.userbits"]

    assert sequence_off == offsets["v4l2_buffer.sequence"]
    assert memory_off == offsets["v4l2_buffer.memory"]
    assert m_off == offsets["v4l2_buffer.m"]
    assert m_off == offsets["v4l2_buffer.m.offset"]  # offset is the union's first member
    assert length_off == offsets["v4l2_buffer.length"]
    assert reserved2_off == offsets["v4l2_buffer.reserved2"]
    assert request_fd_off == offsets["v4l2_buffer.request_fd"]


def test_allowed_ioctls_match_c(c_layout: dict[str, Any]) -> None:
    ioctl = c_layout["ioctl"]
    for name, value in ALLOWED_IOCTLS.items():
        assert value == ioctl[name]["dec"], (
            f"{name}=0x{value:08x} does not match the host's real {ioctl[name]['hex']}"
        )


def test_ioc_encoding_matches_c(c_layout: dict[str, Any]) -> None:
    enc = c_layout["ioc_encoding"]
    assert enc["_IOC_NRBITS"] == v4l2_capture_module._IOC_NRBITS
    assert enc["_IOC_TYPEBITS"] == v4l2_capture_module._IOC_TYPEBITS
    assert enc["_IOC_SIZEBITS"] == v4l2_capture_module._IOC_SIZEBITS
    assert enc["_IOC_WRITE"] == v4l2_capture_module._IOC_WRITE
    assert enc["_IOC_READ"] == v4l2_capture_module._IOC_READ
    # the module hard-codes these as bit-shift expressions rather than named
    # constants; re-derive them the same way _ioc() does and compare.
    nrbits = v4l2_capture_module._IOC_NRBITS
    typebits = v4l2_capture_module._IOC_TYPEBITS
    sizebits = v4l2_capture_module._IOC_SIZEBITS
    assert enc["_IOC_NRSHIFT"] == 0
    assert enc["_IOC_TYPESHIFT"] == nrbits
    assert enc["_IOC_SIZESHIFT"] == nrbits + typebits
    assert enc["_IOC_DIRSHIFT"] == nrbits + typebits + sizebits


def test_constants_match_c(c_layout: dict[str, Any]) -> None:
    constants = c_layout["constants"]
    checks = {
        "V4L2_BUF_TYPE_VIDEO_CAPTURE": V4L2_BUF_TYPE_VIDEO_CAPTURE,
        "V4L2_MEMORY_MMAP": V4L2_MEMORY_MMAP,
        "V4L2_FIELD_NONE": V4L2_FIELD_NONE,
        "V4L2_PIX_FMT_Y16": V4L2_PIX_FMT_Y16,
        "V4L2_CAP_VIDEO_CAPTURE": V4L2_CAP_VIDEO_CAPTURE,
        "V4L2_CAP_STREAMING": V4L2_CAP_STREAMING,
        "V4L2_CAP_DEVICE_CAPS": V4L2_CAP_DEVICE_CAPS,
        "V4L2_CAP_META_CAPTURE": V4L2_CAP_META_CAPTURE,
        "V4L2_BUF_FLAG_MAPPED": V4L2_BUF_FLAG_MAPPED,
        "V4L2_BUF_FLAG_QUEUED": V4L2_BUF_FLAG_QUEUED,
        "V4L2_BUF_FLAG_DONE": V4L2_BUF_FLAG_DONE,
        "V4L2_BUF_FLAG_ERROR": V4L2_BUF_FLAG_ERROR,
        "V4L2_BUF_FLAG_TIMESTAMP_MASK": V4L2_BUF_FLAG_TIMESTAMP_MASK,
        "V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC": V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
        "V4L2_BUF_FLAG_TIMESTAMP_COPY": V4L2_BUF_FLAG_TIMESTAMP_COPY,
        "V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN": V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN,
        # The masked timestamp-*source* field, which is not the timestamp-type
        # field above and does not share a single bit with it.
        "V4L2_BUF_FLAG_TSTAMP_SRC_MASK": V4L2_BUF_FLAG_TSTAMP_SRC_MASK,
        "V4L2_BUF_FLAG_TSTAMP_SRC_EOF": V4L2_BUF_FLAG_TSTAMP_SRC_EOF,
        "V4L2_BUF_FLAG_TSTAMP_SRC_SOE": V4L2_BUF_FLAG_TSTAMP_SRC_SOE,
        "V4L2_BUF_FLAG_LAST": V4L2_BUF_FLAG_LAST,
    }
    for name, value in checks.items():
        assert value == constants[name]["dec"], (
            f"{name}={value} does not match the host's real {constants[name]['hex']}"
        )


def test_fourcc_round_trip_matches_c(c_layout: dict[str, Any]) -> None:
    assert int_to_fourcc(V4L2_PIX_FMT_Y16) == "Y16 "
    assert int_to_fourcc(V4L2_PIX_FMT_Y16) == c_layout["V4L2_PIX_FMT_Y16_fourcc_text"]
    assert c_layout["constants"]["V4L2_PIX_FMT_Y16"]["dec"] == V4L2_PIX_FMT_Y16
    assert fourcc_to_int("Y16 ") == V4L2_PIX_FMT_Y16


# --- packing ---------------------------------------------------------------


def test_pack_capability_is_zeroed_and_sized() -> None:
    packed = pack_capability()
    assert len(packed) == CAPABILITY_BYTES
    assert bytes(packed) == b"\x00" * CAPABILITY_BYTES


def test_pack_format_places_the_four_fields_and_zeroes_the_rest() -> None:
    packed = pack_format()
    assert len(packed) == FORMAT_BYTES
    buf_type, union = struct.unpack(FORMAT_FORMAT, bytes(packed))
    assert buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE
    fields = struct.unpack(PIX_FORMAT_FORMAT, union[:PIX_FORMAT_BYTES])
    width, height, pixelformat, field_value, *rest = fields
    assert (width, height, pixelformat, field_value) == (
        FRAME_WIDTH,
        FRAME_HEIGHT,
        V4L2_PIX_FMT_Y16,
        V4L2_FIELD_NONE,
    )
    assert rest == [0] * len(rest)
    assert union[PIX_FORMAT_BYTES:] == b"\x00" * (FORMAT_UNION_BYTES - PIX_FORMAT_BYTES)


def test_pack_format_accepts_explicit_overrides() -> None:
    packed = pack_format(width=320, height=200, pixelformat=fourcc_to_int("YUY2"))
    _buf_type, union = struct.unpack(FORMAT_FORMAT, bytes(packed))
    width, height, pixelformat, *_rest = struct.unpack(PIX_FORMAT_FORMAT, union[:PIX_FORMAT_BYTES])
    assert (width, height, pixelformat) == (320, 200, fourcc_to_int("YUY2"))


def test_pack_requestbuffers_default_fields() -> None:
    packed = pack_requestbuffers()
    assert len(packed) == REQUESTBUFFERS_BYTES
    count, buf_type, memory, capabilities, flags, reserved = struct.unpack(
        REQUESTBUFFERS_FORMAT, bytes(packed)
    )
    assert (count, buf_type, memory, capabilities, flags, reserved) == (
        REQUESTED_BUFFERS,
        V4L2_BUF_TYPE_VIDEO_CAPTURE,
        V4L2_MEMORY_MMAP,
        0,
        0,
        b"\x00" * 3,
    )


def test_pack_requestbuffers_accepts_zero() -> None:
    (count, *_rest) = struct.unpack(REQUESTBUFFERS_FORMAT, bytes(pack_requestbuffers(0)))
    assert count == 0


def test_pack_requestbuffers_refuses_count_above_the_requested_ceiling() -> None:
    with pytest.raises(V4l2CaptureNotAllowed):
        pack_requestbuffers(REQUESTED_BUFFERS + 1)


def test_pack_requestbuffers_refuses_negative_count() -> None:
    with pytest.raises(V4l2CaptureNotAllowed):
        pack_requestbuffers(-1)


def test_pack_buffer_names_the_index_sets_mmap_and_zeroes_the_m_union() -> None:
    packed = pack_buffer(2)
    assert len(packed) == BUFFER_BYTES
    info = parse_buffer(bytes(packed))
    assert info.index == 2
    assert info.memory == V4L2_MEMORY_MMAP
    assert info.offset == 0  # the first 4 bytes of the zeroed 8-byte m union


@pytest.mark.parametrize("bad_index", [-1, REQUESTED_BUFFERS])
def test_pack_buffer_refuses_index_outside_the_fixed_range(bad_index: int) -> None:
    with pytest.raises(V4l2CaptureNotAllowed):
        pack_buffer(bad_index)


def test_pack_buf_type() -> None:
    packed = pack_buf_type()
    assert len(packed) == BUFTYPE_BYTES
    (value,) = struct.unpack(BUFTYPE_FORMAT, bytes(packed))
    assert value == V4L2_BUF_TYPE_VIDEO_CAPTURE


# --- parsing -----------------------------------------------------------------


def _pack_negotiated_format(
    *,
    buf_type: int = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    pixelformat: int = V4L2_PIX_FMT_Y16,
    field: int = V4L2_FIELD_NONE,
    bytesperline: int = MIN_BYTES_PER_LINE,
    sizeimage: int | None = None,
) -> tuple[int, PixelFormat]:
    """Build and parse a ``v4l2_format`` as if the kernel had answered S_FMT/G_FMT.

    ``pack_format`` always zeroes ``bytesperline``/``sizeimage`` because that is
    what a *request* looks like; a kernel *answer* fills them in, so the format
    and refusal-logic tests below need to build that answer by hand rather than
    reuse ``pack_format`` unmodified.
    """
    if sizeimage is None:
        sizeimage = bytesperline * height
    pix = struct.pack(
        PIX_FORMAT_FORMAT,
        width,
        height,
        pixelformat,
        field,
        bytesperline,
        sizeimage,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    raw = struct.pack(FORMAT_FORMAT, buf_type, pix.ljust(FORMAT_UNION_BYTES, b"\x00"))
    return parse_format(raw)


def test_parse_format_round_trip() -> None:
    buf_type, pf = _pack_negotiated_format()
    assert buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE
    assert pf.fourcc == "Y16 "
    assert pf.field_name == "V4L2_FIELD_NONE"
    assert pf.padded is False
    assert pf.tight_bytes == EXPECTED_FRAME_BYTES


def test_parse_format_padded_property_and_unknown_field_name() -> None:
    _buf_type, tight = _pack_negotiated_format(bytesperline=MIN_BYTES_PER_LINE)
    assert tight.padded is False
    _buf_type, padded = _pack_negotiated_format(bytesperline=MIN_BYTES_PER_LINE + 64)
    assert padded.padded is True
    _buf_type, unknown_field = _pack_negotiated_format(field=99)
    assert unknown_field.field_name == "unknown (99)"


def test_parse_capability_round_trip_and_string_padding() -> None:
    raw = struct.pack(
        CAPABILITY_FORMAT,
        b"uvcvideo",
        b"ZXProduct: video0",
        b"usb-0000:00:14.0-1.4.1",
        0x00050001,
        V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING | V4L2_CAP_DEVICE_CAPS,
        V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
        b"\x00" * 12,
    )
    cap = parse_capability(raw)
    assert cap.driver == "uvcvideo"
    assert cap.card == "ZXProduct: video0"
    assert cap.bus_info == "usb-0000:00:14.0-1.4.1"
    assert cap.version == 0x00050001
    # KERNEL_VERSION(a, b, c) == (a << 16) + (b << 8) + c, so 0x00050001 is 5.0.1.
    assert cap.version_text == "5.0.1"
    assert cap.effective_capabilities == V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING
    assert cap.missing_required == ()


def test_capability_effective_capabilities_uses_whole_word_without_device_caps_bit() -> None:
    raw = struct.pack(
        CAPABILITY_FORMAT,
        b"x",
        b"y",
        b"z",
        0,
        V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,  # no V4L2_CAP_DEVICE_CAPS bit
        0xDEADBEEF,  # device_caps: must be ignored since the bit above is unset
        b"\x00" * 12,
    )
    cap = parse_capability(raw)
    assert cap.effective_capabilities == V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING


def test_capability_missing_required_lists_absent_bits() -> None:
    raw = struct.pack(
        CAPABILITY_FORMAT, b"x", b"y", b"z", 0, V4L2_CAP_VIDEO_CAPTURE, 0, b"\x00" * 12
    )
    cap = parse_capability(raw)
    assert cap.missing_required == ("V4L2_CAP_STREAMING",)


def test_parse_requestbuffers_round_trip() -> None:
    raw = struct.pack(
        REQUESTBUFFERS_FORMAT,
        3,
        V4L2_BUF_TYPE_VIDEO_CAPTURE,
        V4L2_MEMORY_MMAP,
        0xAB,
        0,
        b"\x00" * 3,
    )
    rb = parse_requestbuffers(raw)
    assert (rb.count, rb.buf_type, rb.memory, rb.capabilities) == (
        3,
        V4L2_BUF_TYPE_VIDEO_CAPTURE,
        V4L2_MEMORY_MMAP,
        0xAB,
    )


def _pack_raw_buffer(
    *,
    index: int = 0,
    buf_type: int = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    bytesused: int = 0,
    flags: int = 0,
    field: int = V4L2_FIELD_NONE,
    ts_sec: int = 0,
    ts_usec: int = 0,
    tc_type: int = 0,
    tc_flags: int = 0,
    tc_frames: int = 0,
    tc_seconds: int = 0,
    tc_minutes: int = 0,
    tc_hours: int = 0,
    tc_userbits: bytes = b"\x00" * 4,
    sequence: int = 0,
    memory: int = V4L2_MEMORY_MMAP,
    m_union: bytes = b"\x00" * 8,
    length: int = 0,
    reserved2: int = 0,
    request_fd: int = -1,
) -> bytes:
    """Pack a ``v4l2_buffer`` field by field, in the C-verified field order."""
    return struct.pack(
        BUFFER_FORMAT,
        index,
        buf_type,
        bytesused,
        flags,
        field,
        ts_sec,
        ts_usec,
        tc_type,
        tc_flags,
        tc_frames,
        tc_seconds,
        tc_minutes,
        tc_hours,
        tc_userbits,
        sequence,
        memory,
        m_union,
        length,
        reserved2,
        request_fd,
    )


def test_parse_buffer_round_trip_of_scalar_fields() -> None:
    raw = _pack_raw_buffer(
        index=3,
        bytesused=512_000,
        flags=V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_QUEUED,
        sequence=42,
        length=512_000,
    )
    info = parse_buffer(raw)
    assert info.index == 3
    assert info.buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE
    assert info.bytesused == 512_000
    assert info.sequence == 42
    assert info.memory == V4L2_MEMORY_MMAP
    assert info.length == 512_000
    assert info.field_name == "V4L2_FIELD_NONE"


def test_buffer_flag_names_decode_every_set_bit_and_errored_property() -> None:
    raw = _pack_raw_buffer(flags=V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_DONE | V4L2_BUF_FLAG_ERROR)
    info = parse_buffer(raw)
    assert set(info.flag_names) == {
        "V4L2_BUF_FLAG_MAPPED",
        "V4L2_BUF_FLAG_DONE",
        "V4L2_BUF_FLAG_ERROR",
    }
    assert info.errored is True
    assert parse_buffer(_pack_raw_buffer(flags=0)).errored is False


@pytest.mark.parametrize(
    ("flags", "expected_type"),
    [
        (0, "V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN"),
        (V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, "V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC"),
        (V4L2_BUF_FLAG_TIMESTAMP_COPY, "V4L2_BUF_FLAG_TIMESTAMP_COPY"),
        (
            V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TIMESTAMP_COPY,
            "unknown (0x00006000)",
        ),
    ],
)
def test_buffer_timestamp_type_decoding(flags: int, expected_type: str) -> None:
    """The ``TIMESTAMP_MASK`` field says which clock the timestamp is on."""
    info = parse_buffer(_pack_raw_buffer(flags=flags))
    assert info.timestamp_type == expected_type


@pytest.mark.parametrize(
    ("flags", "expected_source"),
    [
        (0, "V4L2_BUF_FLAG_TSTAMP_SRC_EOF"),
        (V4L2_BUF_FLAG_TSTAMP_SRC_SOE, "V4L2_BUF_FLAG_TSTAMP_SRC_SOE"),
        # 0x00020000 and 0x00040000 are inside the mask but are not defined
        # values; naming them would be inventing kernel vocabulary.
        (0x00020000, "unknown (0x00020000)"),
        (V4L2_BUF_FLAG_TSTAMP_SRC_MASK, "unknown (0x00070000)"),
    ],
)
def test_buffer_timestamp_source_decoding(flags: int, expected_source: str) -> None:
    """The ``TSTAMP_SRC_MASK`` field is a separate field from the type above."""
    info = parse_buffer(_pack_raw_buffer(flags=flags))
    assert info.timestamp_source == expected_source


def test_the_timestamp_type_and_source_fields_do_not_overlap() -> None:
    assert V4L2_BUF_FLAG_TIMESTAMP_MASK & V4L2_BUF_FLAG_TSTAMP_SRC_MASK == 0
    # Every named value lies inside its own mask, and the two zero-valued ones
    # are only readable through it.
    for value in (
        V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN,
        V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
        V4L2_BUF_FLAG_TIMESTAMP_COPY,
    ):
        assert value & ~V4L2_BUF_FLAG_TIMESTAMP_MASK == 0
    for value in (V4L2_BUF_FLAG_TSTAMP_SRC_EOF, V4L2_BUF_FLAG_TSTAMP_SRC_SOE):
        assert value & ~V4L2_BUF_FLAG_TSTAMP_SRC_MASK == 0
    assert V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN == 0
    assert V4L2_BUF_FLAG_TSTAMP_SRC_EOF == 0


def test_0x00010000_is_the_soe_timestamp_source_and_not_v4l2_buf_flag_last() -> None:
    """The exact confusion the scanner-009 round-0 record made, pinned down.

    ``0x00010000`` was reported as ``V4L2_BUF_FLAG_LAST``. It is not: it is
    ``V4L2_BUF_FLAG_TSTAMP_SRC_SOE``, a value of the timestamp-source field,
    and ``V4L2_BUF_FLAG_LAST`` is ``0x00100000`` - one hex digit over, and a
    genuinely different claim about the stream.
    """
    assert V4L2_BUF_FLAG_TSTAMP_SRC_SOE == 0x00010000
    assert V4L2_BUF_FLAG_LAST == 0x00100000
    assert V4L2_BUF_FLAG_LAST & V4L2_BUF_FLAG_TSTAMP_SRC_MASK == 0
    soe_only = parse_buffer(_pack_raw_buffer(flags=V4L2_BUF_FLAG_TSTAMP_SRC_SOE))
    assert "V4L2_BUF_FLAG_LAST" not in soe_only.flag_names
    assert soe_only.timestamp_source == "V4L2_BUF_FLAG_TSTAMP_SRC_SOE"
    last_only = parse_buffer(_pack_raw_buffer(flags=V4L2_BUF_FLAG_LAST))
    assert "V4L2_BUF_FLAG_LAST" in last_only.flag_names
    assert last_only.timestamp_source == "V4L2_BUF_FLAG_TSTAMP_SRC_EOF"


@pytest.mark.parametrize(
    ("raw_flags", "mapped", "moment"),
    [
        # The two flag words the one live scanner-009 run actually recorded:
        # 0x00012000 from VIDIOC_QUERYBUF and 0x00012001 from VIDIOC_DQBUF.
        (0x00012000, False, "queried"),
        (0x00012001, True, "dequeued"),
    ],
)
def test_the_live_scanner_009_flag_words_decode_exactly(
    raw_flags: int, mapped: bool, moment: str
) -> None:
    """Regression over the exact numbers the hardware produced, once.

    The raw words are the evidence and are unchanged; only their reading was
    wrong. Both carry a monotonic timestamp taken at start of exposure, and
    neither carries ``V4L2_BUF_FLAG_LAST``, ``DONE`` or ``ERROR``.
    """
    info = parse_buffer(_pack_raw_buffer(flags=raw_flags))
    assert info.flags == raw_flags
    assert info.timestamp_type == "V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC"
    assert info.timestamp_source == "V4L2_BUF_FLAG_TSTAMP_SRC_SOE"
    expected_names = ("V4L2_BUF_FLAG_MAPPED",) if mapped else ()
    assert info.flag_names == expected_names, moment
    assert "V4L2_BUF_FLAG_LAST" not in info.flag_names
    assert "V4L2_BUF_FLAG_DONE" not in info.flag_names
    assert info.errored is False
    # Nothing outside the two masked fields and MAPPED is set in either word.
    named = V4L2_BUF_FLAG_TIMESTAMP_MASK | V4L2_BUF_FLAG_TSTAMP_SRC_MASK | V4L2_BUF_FLAG_MAPPED
    assert raw_flags & ~named == 0


def test_buffer_info_timestamp_ns() -> None:
    info = parse_buffer(_pack_raw_buffer(ts_sec=5, ts_usec=250_000))
    assert info.timestamp_ns == 5_000_000_000 + 250_000_000


def test_buffer_offset_decodes_only_the_first_four_bytes_of_the_m_union() -> None:
    """A nonzero high half must not leak into ``offset`` via a wide native load."""
    m_union = struct.pack("=I", 0x1000) + b"\xff\xff\xff\xff"
    info = parse_buffer(_pack_raw_buffer(m_union=m_union))
    assert info.offset == 0x1000


def test_buffer_field_name_decoding_known_and_unknown() -> None:
    assert parse_buffer(_pack_raw_buffer(field=V4L2_FIELD_NONE)).field_name == "V4L2_FIELD_NONE"
    assert parse_buffer(_pack_raw_buffer(field=99)).field_name == "unknown (99)"


# --- ensure_layout_intact -----------------------------------------------------


def test_ensure_layout_intact_passes_on_this_host() -> None:
    ensure_layout_intact()


def test_ensure_layout_intact_refuses_a_disagreeing_size_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = dict(EXPECTED_STRUCT_BYTES)
    bad["v4l2_buffer"] = EXPECTED_STRUCT_BYTES["v4l2_buffer"] + 1
    monkeypatch.setattr(v4l2_capture_module, "EXPECTED_STRUCT_BYTES", MappingProxyType(bad))
    with pytest.raises(V4l2CaptureNotAllowed):
        ensure_layout_intact()


# --- unpack_rows ---------------------------------------------------------------


def test_unpack_rows_extracts_correct_bytes_from_a_padded_frame() -> None:
    bytesperline = MIN_BYTES_PER_LINE + 64
    _buf_type, pf = _pack_negotiated_format(
        bytesperline=bytesperline, sizeimage=bytesperline * FRAME_HEIGHT
    )
    rows = [bytes([row & 0xFF]) * MIN_BYTES_PER_LINE + b"\xee" * 64 for row in range(FRAME_HEIGHT)]
    raw = b"".join(rows)
    packed = unpack_rows(raw, pf)
    assert len(packed) == pf.tight_bytes == EXPECTED_FRAME_BYTES
    for row in range(FRAME_HEIGHT):
        start = row * MIN_BYTES_PER_LINE
        expected_row = bytes([row & 0xFF]) * MIN_BYTES_PER_LINE
        assert packed[start : start + MIN_BYTES_PER_LINE] == expected_row


def test_unpack_rows_refuses_bytesperline_below_one_row() -> None:
    _buf_type, pf = _pack_negotiated_format(bytesperline=MIN_BYTES_PER_LINE - 1, sizeimage=1)
    with pytest.raises(V4l2CaptureNotAllowed):
        unpack_rows(b"\x00" * 10, pf)


def test_unpack_rows_refuses_a_raw_buffer_too_short_for_its_own_geometry() -> None:
    bytesperline = MIN_BYTES_PER_LINE + 16
    _buf_type, pf = _pack_negotiated_format(
        bytesperline=bytesperline, sizeimage=bytesperline * FRAME_HEIGHT
    )
    raw = b"\x00" * (bytesperline * FRAME_HEIGHT - 1)  # one byte short of what bytesperline implies
    with pytest.raises(V4l2CaptureNotAllowed):
        unpack_rows(raw, pf)


def test_unpack_rows_refuses_when_the_result_is_not_the_tight_size() -> None:
    """A width other than FRAME_WIDTH desyncs tight_bytes from the fixed row width extracted."""
    bytesperline = MIN_BYTES_PER_LINE + 32
    _buf_type, pf = _pack_negotiated_format(
        width=320, bytesperline=bytesperline, sizeimage=bytesperline * FRAME_HEIGHT
    )
    raw = b"\x00" * (bytesperline * FRAME_HEIGHT)
    with pytest.raises(V4l2CaptureNotAllowed):
        unpack_rows(raw, pf)


# --- _format_refusal -----------------------------------------------------------


def test_format_refusal_none_for_a_good_tight_format() -> None:
    buf_type, pf = _pack_negotiated_format(bytesperline=MIN_BYTES_PER_LINE)
    assert _format_refusal(buf_type, pf) is None


def test_format_refusal_none_for_a_good_padded_format() -> None:
    bytesperline = MIN_BYTES_PER_LINE + 64
    buf_type, pf = _pack_negotiated_format(
        bytesperline=bytesperline, sizeimage=bytesperline * FRAME_HEIGHT
    )
    assert _format_refusal(buf_type, pf) is None


def _format_refusal_cases() -> dict[str, tuple[int, PixelFormat]]:
    return {
        "wrong_buf_type": _pack_negotiated_format(buf_type=2),
        "wrong_fourcc": _pack_negotiated_format(pixelformat=fourcc_to_int("YUY2")),
        "wrong_width": _pack_negotiated_format(width=320),
        "wrong_height": _pack_negotiated_format(height=200),
        "bytesperline_too_small": _pack_negotiated_format(
            bytesperline=MIN_BYTES_PER_LINE - 1, sizeimage=1
        ),
        "sizeimage_smaller_than_implied": _pack_negotiated_format(
            bytesperline=MIN_BYTES_PER_LINE, sizeimage=EXPECTED_FRAME_BYTES - 1
        ),
        "sizeimage_over_ceiling": _pack_negotiated_format(
            bytesperline=MIN_BYTES_PER_LINE, sizeimage=MAX_BUFFER_BYTES + 1
        ),
    }


@pytest.mark.parametrize("case_name", sorted(_format_refusal_cases()))
def test_format_refusal_flags_each_bad_case(case_name: str) -> None:
    buf_type, pf = _format_refusal_cases()[case_name]
    assert _format_refusal(buf_type, pf) is not None, f"{case_name} should have been refused"


def test_format_refusal_reasons_are_distinct_per_cause() -> None:
    reasons = {name: _format_refusal(*case) for name, case in _format_refusal_cases().items()}
    assert len(set(reasons.values())) == len(reasons), reasons


# --- _mapping_refusal ----------------------------------------------------------


def _buffer_info(**overrides: object) -> BufferInfo:
    info = parse_buffer(_pack_raw_buffer(length=4096))
    return replace(info, **overrides)  # type: ignore[arg-type]


def test_mapping_refusal_none_for_a_good_buffer() -> None:
    info = _buffer_info(index=0, length=4096, offset=0)
    assert _mapping_refusal(info, expected_index=0, running_total=4096) is None


def test_mapping_refusal_wrong_index() -> None:
    info = _buffer_info(index=1)
    assert _mapping_refusal(info, expected_index=0, running_total=info.length) is not None


def test_mapping_refusal_non_mmap_memory() -> None:
    info = _buffer_info(memory=2)  # V4L2_MEMORY_USERPTR, not MMAP
    assert _mapping_refusal(info, expected_index=info.index, running_total=info.length) is not None


def test_mapping_refusal_zero_length() -> None:
    info = _buffer_info(length=0)
    assert _mapping_refusal(info, expected_index=info.index, running_total=0) is not None


def test_mapping_refusal_length_over_the_per_buffer_ceiling() -> None:
    info = _buffer_info(length=MAX_BUFFER_BYTES + 1)
    assert _mapping_refusal(info, expected_index=info.index, running_total=info.length) is not None


def test_mapping_refusal_running_total_over_the_total_ceiling() -> None:
    info = _buffer_info(length=4096)
    running_total = MAX_TOTAL_MAPPED_BYTES + 1
    reason = _mapping_refusal(info, expected_index=info.index, running_total=running_total)
    assert reason is not None


def test_mapping_refusal_page_unaligned_offset() -> None:
    info = _buffer_info(offset=1)
    assert _mapping_refusal(info, expected_index=info.index, running_total=info.length) is not None


# --- node resolution -----------------------------------------------------------


def test_resolve_capture_node_by_role_not_by_device_node_number(builder: SysfsBuilder) -> None:
    """The depth capture node is video7 while an unrelated video0 exists elsewhere."""
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video7", 0), ("video8", 1)),
    )
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{REAL_SERIAL}",
        video_nodes=(("video0", 0), ("video1", 1)),
    )
    snapshot, _device = resolve_capture_node(
        REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    assert snapshot.node_name == "video7"
    assert snapshot.node_path == str(builder.dev / "video7")
    assert snapshot.role == "depth"
    assert snapshot.interface_number == 0
    assert snapshot.v4l_index == 0


def test_resolve_capture_node_raises_for_no_matching_serial(empty_sysfs: SysfsBuilder) -> None:
    with pytest.raises(DeviceAbsentError):
        resolve_capture_node(REAL_SERIAL, sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev)


def test_resolve_capture_node_raises_for_a_serial_matching_two_devices(
    builder: SysfsBuilder,
) -> None:
    builder.add_healthy_mini()
    second = builder.add_usb_device("1-5", devnum="30", serial=REAL_SERIAL)
    builder.add_usbfs(device=30)
    builder.add_interface(
        second, 0, interface_string=f"DepthCam{REAL_SERIAL}", video_nodes=(("video9", 0),)
    )
    with pytest.raises(DeviceAbsentError):
        resolve_capture_node(REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev)


def test_resolve_capture_node_raises_when_device_has_no_depth_function(
    builder: SysfsBuilder,
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, interface_string="SomethingElse")
    with pytest.raises(DeviceAbsentError):
        resolve_capture_node(REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev)


def test_resolve_capture_node_raises_when_depth_function_has_no_capture_node(
    builder: SysfsBuilder,
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video1", 1),),  # metadata node only, no index-0 capture node
    )
    with pytest.raises(DeviceAbsentError):
        resolve_capture_node(REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev)


def test_resolve_capture_node_raises_when_the_dev_node_is_missing(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base, 0, interface_string=f"DepthCam{REAL_SERIAL}", video_nodes=(("video0", 0),)
    )
    (builder.dev / "video0").unlink()
    with pytest.raises(DeviceAbsentError):
        resolve_capture_node(REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev)


def test_node_snapshot_as_dict_records_role_indices_and_resolution_explanation(
    builder: SysfsBuilder,
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base, 0, interface_string=f"DepthCam{REAL_SERIAL}", video_nodes=(("video0", 0),)
    )
    snapshot, _device = resolve_capture_node(
        REAL_SERIAL, sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    payload = snapshot.as_dict()
    assert payload["role"] == "depth"
    assert payload["interface_number"] == 0
    assert payload["v4l_index"] == 0
    resolved_by = payload["resolved_by"]
    assert isinstance(resolved_by, str)
    assert "depth" in resolved_by
    assert "V4L2 index" in resolved_by


# --- refusal_reason --------------------------------------------------------------


def test_refusal_reason_none_for_the_healthy_tree(healthy: SysfsBuilder) -> None:
    (device,) = find_scanners(sysfs_root=healthy.sysfs, dev_root=healthy.dev)
    assert refusal_reason(device, REAL_SERIAL) is None


def test_refusal_reason_wrong_vendor_or_product_id(healthy: SysfsBuilder) -> None:
    (device,) = find_scanners(sysfs_root=healthy.sysfs, dev_root=healthy.dev)
    bad = replace(device, vendor_id=0x1234)
    assert refusal_reason(bad, REAL_SERIAL) is not None


def test_refusal_reason_wrong_serial(healthy: SysfsBuilder) -> None:
    (device,) = find_scanners(sysfs_root=healthy.sysfs, dev_root=healthy.dev)
    assert refusal_reason(device, "not-the-real-serial") is not None


def test_refusal_reason_wrong_manufacturer(healthy: SysfsBuilder) -> None:
    (device,) = find_scanners(sysfs_root=healthy.sysfs, dev_root=healthy.dev)
    bad = replace(device, manufacturer="NotChishine3d")
    assert refusal_reason(bad, REAL_SERIAL) is not None


def test_refusal_reason_depth_interface_bound_to_a_different_driver(
    builder: SysfsBuilder,
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        driver="usbhid",
        video_nodes=(("video0", 0),),
    )
    (device,) = find_scanners(sysfs_root=builder.sysfs, dev_root=builder.dev)
    assert refusal_reason(device, REAL_SERIAL) is not None


def test_refusal_reason_unbound_depth_interface(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        driver=None,
        video_nodes=(("video0", 0),),
    )
    (device,) = find_scanners(sysfs_root=builder.sysfs, dev_root=builder.dev)
    assert refusal_reason(device, REAL_SERIAL) is not None


def test_refusal_reason_descriptors_do_not_advertise_y16_640x400(builder: SysfsBuilder) -> None:
    """A corrupt descriptor blob parses to zero advertised formats (see test_discovery.py)."""
    builder.add_healthy_mini(descriptors=b"\xff\xff\xff")
    (device,) = find_scanners(sysfs_root=builder.sysfs, dev_root=builder.dev)
    assert refusal_reason(device, REAL_SERIAL) is not None


# --- the closed ioctl repertoire -----------------------------------------------


def _executable_source(module_path: Path) -> str:
    """The module's code with comments and string literals removed."""
    with module_path.open("rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))
    return " ".join(
        token.string for token in tokens if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def test_module_source_names_no_usbfs_command() -> None:
    """No USBDEVFS_* command number - this module talks V4L2 only, via the kernel driver."""
    code = _executable_source(Path(v4l2_capture_module.__file__))
    assert "USBDEVFS" not in code


def test_module_constructs_exactly_the_nine_allowed_ioctls() -> None:
    """Every ``_ioc(...)`` call site in the module builds one of ALLOWED_IOCTLS' values.

    Modeled on ``test_the_module_knows_exactly_six_usbfs_commands`` in
    tests/test_ownership.py: walk the module's own source rather than trust a
    comment, so a new hidden ``_ioc(...)`` call anywhere in the file - not just
    at module scope - would fail this test.
    """
    tree = ast.parse(Path(v4l2_capture_module.__file__).read_text(encoding="utf-8"))

    total_ioc_calls = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_ioc"
    )
    assert total_ioc_calls == len(ALLOWED_IOCTLS)

    ioc_assignments: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_ioc"
        ):
            ioc_assignments[node.target.id] = node.value
    assert set(ioc_assignments) == set(ALLOWED_IOCTLS), sorted(ioc_assignments)
    for name in ioc_assignments:
        assert getattr(v4l2_capture_module, name) == ALLOWED_IOCTLS[name]
