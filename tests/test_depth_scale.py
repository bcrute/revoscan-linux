"""Tests for the one bounded read of the vendor depth-scale property ``0x918``.

Nothing here touches hardware. Every syscall is injected, so these tests assert
on the exact setup bytes, the exact four outbound payload bytes, the exact
60-byte reply handling, and - most of the point - that the module is a *closed
two-transfer whitelist*: no third transfer on any branch, no retry anywhere, no
argument that can name another property id, and no run that failed or was
cleaned up badly ever offering a scale.

The setup packets are additionally cross-checked against the committed
``objdump -d`` excerpt of ``cs::libuvcReadValueExtension`` that
:mod:`tests.test_reference_analysis` already carries, so the whitelist is
pinned to the recovered disassembly rather than to a transcription of it.
"""

from __future__ import annotations

import ast
import ctypes
import dataclasses
import errno
import hashlib
import inspect
import json
import math
import signal
import struct
from pathlib import Path

import pytest

import scanner3d.control as control_module
import scanner3d.depth_scale as scale_module
import scanner3d.ownership as ownership_module
from scanner3d.control import ControlRequest, pack_ctrltransfer
from scanner3d.depth_scale import (
    AUTHORIZED_SERIAL,
    DENOMINATOR_FORMAT,
    DEPTH_SCALE_PROPERTY_ID,
    ENTITY_ID,
    EXPECTED_MARKER,
    MARKER_OFFSET,
    MAX_CONTROL_TRANSFERS,
    PROPERTY_PAYLOAD,
    PROPERTY_PAYLOAD_LENGTH,
    PROPERTY_SELECTOR,
    PROPERTY_WRITE_TRANSFER,
    RESULT_LENGTH,
    RESULT_READ_TRANSFER,
    SDK_MINI_FALLBACK_SCALE,
    TIMEOUT_MS,
    UNIT_DISCLAIMER,
    WHITELIST,
    DepthScaleOutcome,
    ScaleOutcome,
    Step,
    UsbfsDepthScaleBackend,
    decode_reply,
    ensure_transfer_allowed,
    ensure_whitelist_intact,
    float32,
    float32_bits,
    vendor_reciprocal,
)
from scanner3d.errors import DepthScaleReadNotAllowed
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONNECT,
    USBDEVFS_DISCONNECT,
    USBDEVFS_GETDRIVER,
    USBDEVFS_RELEASEINTERFACE,
    OwnershipStatus,
    Phase,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.test_control import ExplodingOpener
from tests.test_init_sequence import FORBIDDEN_IOCTLS, _executable_source
from tests.test_reference_analysis import READ_VALUE_EXTENSION_EXCERPT, xu_trace

# --- fixtures for the wire ----------------------------------------------------

#: The packed ``usbdevfs_ctrltransfer`` prefixes the recovered call sites use.
EXPECTED_WRITE_SETUP = bytes.fromhex("2101000e00040400e8030000")
EXPECTED_READ_SETUP = bytes.fromhex("a181000e00043c00e8030000")


def reply(denominator: int, *, marker: int = EXPECTED_MARKER, length: int = RESULT_LENGTH) -> bytes:
    """A synthetic 60-byte reply: int32 denominator, filler, marker last."""
    body = bytearray(RESULT_LENGTH)
    body[0:4] = struct.pack(DENOMINATOR_FORMAT, denominator)
    body[MARKER_OFFSET] = marker & 0xFF
    return bytes(body[:length])


#: A denominator whose reciprocal differs between binary32 and float64, so the
#: rounding assertions cannot pass by accident.
ROUNDING_DENOMINATOR = 3

#: 1/200 - the denominator that would reproduce the SDK's 0.005f fallback.
FALLBACK_DENOMINATOR = 200


class FakeUsbfs:
    """Records every open/ioctl/close and answers each usbfs command.

    Unlike the file-read fake, the two control transfers here have *different*
    lengths, so the outbound bytes are read using the ``wLength`` in the packed
    setup rather than a single constant.
    """

    def __init__(
        self,
        *,
        driver: str = "uvcvideo",
        replies: list[bytes] | None = None,
        open_error: OSError | None = None,
        errors: dict[int, OSError] | None = None,
        control_errors: dict[int, OSError] | None = None,
        control_returns: dict[int, int] | None = None,
        on_control: object = None,
        on_ioctl: object = None,
    ) -> None:
        self.driver = driver
        self.replies = list(replies if replies is not None else [reply(FALLBACK_DENOMINATOR)])
        self.open_error = open_error
        self.errors = errors or {}
        self.control_errors = control_errors or {}
        self.control_returns = control_returns or {}
        self.on_control = on_control
        self.on_ioctl = on_ioctl
        self.opened: list[str] = []
        self.calls: list[tuple[int, bytes]] = []
        self.controls: list[tuple[bytes, bytes]] = []
        self.closed: list[int] = []
        self.ticks = 0.0

    # -- injected syscalls ---------------------------------------------------

    def open(self, path: str) -> int:
        if self.open_error is not None:
            raise self.open_error
        self.opened.append(path)
        return 7

    def ioctl(self, fd: int, cmd: int, arg: bytearray, mutate: bool) -> int:
        self.calls.append((cmd, bytes(arg)))
        code = cmd
        if cmd == ownership_module.USBDEVFS_IOCTL:
            _, code, _ = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, bytes(arg))
        if callable(self.on_ioctl):
            self.on_ioctl(code)
        if cmd == control_module.USBDEVFS_CONTROL:
            return self._control(bytes(arg))
        if code in self.errors:
            raise self.errors[code]
        if cmd == USBDEVFS_GETDRIVER:
            arg[:] = struct.pack(
                ownership_module._GETDRIVER_FORMAT, OWNED_INTERFACE, self.driver.encode("ascii")
            )
        return 0

    def _control(self, packed: bytes) -> int:
        index = len(self.controls)
        *_, length, _timeout, address = struct.unpack(control_module._CTRLTRANSFER_FORMAT, packed)
        outbound = ctypes.string_at(address, length)
        self.controls.append((packed[:12], outbound))
        if callable(self.on_control):
            self.on_control(index)
        if index in self.control_errors:
            raise self.control_errors[index]
        if packed[0] & 0x80:
            answer = self.replies.pop(0) if self.replies else bytes(RESULT_LENGTH)
            ctypes.memmove(address, answer, min(len(answer), length))
            return self.control_returns.get(index, len(answer))
        return self.control_returns.get(index, length)

    def close(self, fd: int) -> None:
        self.closed.append(fd)

    def clock(self) -> float:
        self.ticks += 0.25
        return self.ticks

    # -- convenience ---------------------------------------------------------

    @property
    def commands(self) -> list[int]:
        return [cmd for cmd, _ in self.calls]

    @property
    def driver_codes(self) -> list[int]:
        codes = []
        for cmd, arg in self.calls:
            if cmd == ownership_module.USBDEVFS_IOCTL:
                _, code, _ = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, arg)
                codes.append(code)
        return codes

    @property
    def setups(self) -> list[bytes]:
        return [setup for setup, _ in self.controls]

    @property
    def written(self) -> list[bytes]:
        """The outbound payload of every host-to-device transfer."""
        return [data for setup, data in self.controls if not setup[0] & 0x80]


def make_backend(builder: SysfsBuilder, fake: FakeUsbfs) -> UsbfsDepthScaleBackend:
    return UsbfsDepthScaleBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def run_with(builder: SysfsBuilder, fake: FakeUsbfs) -> DepthScaleOutcome:
    return make_backend(builder, fake).run(REAL_SERIAL)


def phases_of(outcome: DepthScaleOutcome) -> list[Phase]:
    return [result.phase for result in outcome.phases]


# --- the whitelist is the recovered disassembly -------------------------------


def test_the_whitelist_matches_the_committed_objdump_of_libuvcReadValueExtension() -> None:
    """Both setup packets are pinned to the disassembly, not to a transcription."""
    recovered = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)
    assert len(recovered) == len(WHITELIST) == MAX_CONTROL_TRANSFERS
    for transfer, source in zip(WHITELIST, recovered, strict=True):
        request = transfer.request
        assert request.request_type == source.bmRequestType
        assert request.request == source.bRequest
        assert request.value == source.wValue
        assert request.index == source.wIndex
        assert request.length == source.wLength
        assert request.timeout_ms == source.timeout


def test_the_recovered_call_sites_are_the_two_this_module_issues() -> None:
    set_cur, get_cur = xu_trace.extract_control_transfers(READ_VALUE_EXTENSION_EXCERPT)
    assert set_cur.address == "262dec"
    assert get_cur.address == "262e6a"
    # The four-byte payload is the caller's argument, and getDepthScale's is 0x918.
    (store,) = set_cur.payload
    assert store.offset == 0
    assert store.width == PROPERTY_PAYLOAD_LENGTH == len(PROPERTY_PAYLOAD)
    # The GET carries no payload store, which is why there is no third transfer.
    assert get_cur.payload == []


def test_the_property_payload_is_0x918_little_endian() -> None:
    assert PROPERTY_PAYLOAD == b"\x18\x09\x00\x00"
    assert struct.pack("<I", DEPTH_SCALE_PROPERTY_ID) == PROPERTY_PAYLOAD
    assert len(PROPERTY_PAYLOAD) == PROPERTY_PAYLOAD_LENGTH == 4


def test_the_packed_setup_bytes_are_exactly_the_recovered_ones() -> None:
    """What ``pack_ctrltransfer`` would hand the kernel, byte for byte."""
    buffer = ctypes.create_string_buffer(RESULT_LENGTH)
    address = ctypes.addressof(buffer)
    write = bytes(pack_ctrltransfer(PROPERTY_WRITE_TRANSFER.request, address))
    read = bytes(pack_ctrltransfer(RESULT_READ_TRANSFER.request, address))
    assert (
        write[:12]
        == EXPECTED_WRITE_SETUP
        == bytes.fromhex("2101000e000404 00e8030000".replace(" ", ""))
    )
    assert read[:12] == EXPECTED_READ_SETUP


def test_both_transfers_name_entity_4_selector_0x0e_on_interface_0() -> None:
    for transfer in WHITELIST:
        request = transfer.request
        assert request.entity_id == ENTITY_ID == 4
        assert request.interface_number == OWNED_INTERFACE == 0
        assert request.control_selector == PROPERTY_SELECTOR == 0x0E
        assert request.value == 0x0E00
        assert request.index == 0x0400
        assert request.timeout_ms == TIMEOUT_MS == 1000


def test_the_two_transfers_are_a_write_then_a_read_of_fixed_lengths() -> None:
    write, read = WHITELIST
    assert write.step is Step.PROPERTY_WRITE
    assert not write.request.is_read
    assert write.request.length == 4
    assert write.payload == PROPERTY_PAYLOAD
    assert read.step is Step.RESULT_READ
    assert read.request.is_read
    assert read.request.length == RESULT_LENGTH == 60
    assert read.payload is None


# --- the safety boundary ------------------------------------------------------


def test_the_module_knows_exactly_one_usbfs_command() -> None:
    names = {n for n in dir(scale_module) if n.startswith("USBDEVFS_")}
    assert names == {"USBDEVFS_CONTROL"}
    code = _executable_source(Path(scale_module.__file__))
    for name, number in FORBIDDEN_IOCTLS.items():
        assert name not in code
        assert str(number) not in code
        assert f"0x{number:x}" not in code.lower()


def test_the_module_names_no_other_property_id_and_no_file_read_key() -> None:
    """0x918 is the only 0x9xx literal, and no path or licence string is here."""
    source = Path(scale_module.__file__).read_text(encoding="utf-8")
    code = _executable_source(Path(scale_module.__file__))
    for forbidden in ("/data/", "camparam", "licen", "Pl.bin", "Prgb", "LC_RT", "Distort"):
        assert forbidden.lower() not in source.lower()
    hex_literals = {
        literal.lower()
        for literal in __import__("re").findall(r"0[xX][0-9a-fA-F]+", code)
        if literal.lower().startswith("0x9")
    }
    assert hex_literals == {"0x918"}


def test_no_public_callable_takes_a_transfer_shaping_argument() -> None:
    """No id, selector, entity, interface, length, timeout, count or retry."""
    forbidden = {
        "property_id",
        "command_id",
        "selector",
        "entity",
        "entity_id",
        "interface",
        "interface_number",
        "request",
        "request_type",
        "length",
        "wlength",
        "timeout",
        "timeout_ms",
        "count",
        "retries",
        "retry",
        "attempts",
        "capacity",
        "path",
        "key",
    }
    for name, obj in vars(scale_module).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != scale_module.__name__:
            continue  # imported helper; its own module owns its surface
        parameters = set(inspect.signature(obj).parameters)
        assert parameters.isdisjoint(forbidden), f"{name} takes {parameters & forbidden}"
    for method in ("preflight", "run"):
        parameters = set(inspect.signature(getattr(UsbfsDepthScaleBackend, method)).parameters)
        assert parameters == {"self", "serial"}


def test_module_imports_nothing_that_could_stream_or_load_the_sdk() -> None:
    tree = ast.parse(Path(scale_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert imported <= {
        "__future__",
        "ctypes",
        "enum",
        "hashlib",
        "math",
        "struct",
        "collections.abc",
        "dataclasses",
        "types",
        "typing",
        "scanner3d.control",
        "scanner3d.config",
        "scanner3d.errors",
        "scanner3d.ownership",
    }


def test_the_whitelist_cannot_be_reordered_extended_or_truncated() -> None:
    ensure_whitelist_intact()
    for bad in (
        (RESULT_READ_TRANSFER, PROPERTY_WRITE_TRANSFER),
        (PROPERTY_WRITE_TRANSFER,),
        (RESULT_READ_TRANSFER,),
        (*WHITELIST, PROPERTY_WRITE_TRANSFER),
        (),
    ):
        with pytest.raises(DepthScaleReadNotAllowed):
            ensure_whitelist_intact(bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_type", 0x41),
        ("request", 0xFF),
        ("value", 0x0100),
        ("value", 0x0200),
        ("index", 0x0300),
        ("index", 0x0401),
        ("length", 60),
        ("timeout_ms", 0),
        ("timeout_ms", 5000),
    ],
)
def test_a_transfer_with_any_altered_setup_field_is_refused(field: str, value: int) -> None:
    original = PROPERTY_WRITE_TRANSFER.request
    fields = {
        "name": original.name,
        "request_type": original.request_type,
        "request": original.request,
        "value": original.value,
        "index": original.index,
        "length": original.length,
        "timeout_ms": original.timeout_ms,
    }
    fields[field] = value
    altered = dataclasses.replace(PROPERTY_WRITE_TRANSFER, request=ControlRequest(**fields))  # type: ignore[arg-type]
    with pytest.raises(DepthScaleReadNotAllowed):
        ensure_transfer_allowed(altered)


@pytest.mark.parametrize(
    "payload",
    [
        struct.pack("<I", 0x919),
        struct.pack("<I", 0x900),
        struct.pack("<I", 0x000),
        b"\x18\x09\x00",
        b"\x18\x09\x00\x00\x00",
    ],
)
def test_a_payload_that_is_not_the_0x918_id_is_refused(payload: bytes) -> None:
    with pytest.raises(DepthScaleReadNotAllowed):
        ensure_transfer_allowed(dataclasses.replace(PROPERTY_WRITE_TRANSFER, payload=payload))


# --- the decode ---------------------------------------------------------------


def test_a_valid_reply_decodes_to_the_vendor_reciprocal() -> None:
    reading = decode_reply(reply(FALLBACK_DENOMINATOR))
    assert reading is not None
    assert reading.outcome is ScaleOutcome.COMPLETE
    assert reading.valid
    assert reading.marker_uint8 == 4
    assert reading.marker_int8 == 4
    assert reading.denominator == FALLBACK_DENOMINATOR
    assert reading.scale == float32(1.0 / 200.0)
    assert reading.scale_bits == float32_bits(reading.scale)


def test_the_marker_is_reported_as_both_uint8_and_int8() -> None:
    """0xff is 255 unsigned and -1 signed; the vendor reads it as int8."""
    reading = decode_reply(reply(FALLBACK_DENOMINATOR, marker=0xFF))
    assert reading is not None
    assert reading.marker_uint8 == 255
    assert reading.marker_int8 == -1
    assert reading.outcome is ScaleOutcome.UNEXPECTED_MARKER
    assert not reading.valid
    assert reading.scale is None


@pytest.mark.parametrize("marker", [0, 1, 3, 5, 127, 128, 200, 255])
def test_only_marker_4_is_accepted(marker: int) -> None:
    reading = decode_reply(reply(FALLBACK_DENOMINATOR, marker=marker))
    assert reading is not None
    if marker == EXPECTED_MARKER:  # pragma: no cover - 4 is not in the list
        pytest.fail("the parametrisation must not include the accepted marker")
    assert reading.outcome is ScaleOutcome.UNEXPECTED_MARKER
    assert reading.scale is None
    assert reading.reference is None


def test_a_zero_denominator_is_typed_and_never_divided() -> None:
    reading = decode_reply(reply(0))
    assert reading is not None
    assert reading.outcome is ScaleOutcome.ZERO_DENOMINATOR
    assert reading.denominator == 0
    assert reading.scale is None
    assert not reading.valid


@pytest.mark.parametrize("denominator", [-1, -200, -2147483648])
def test_a_negative_denominator_is_typed_and_refused(denominator: int) -> None:
    reading = decode_reply(reply(denominator))
    assert reading is not None
    assert reading.outcome is ScaleOutcome.NEGATIVE_DENOMINATOR
    assert reading.denominator == denominator
    assert reading.scale is None
    assert not reading.valid


def test_a_non_finite_reciprocal_is_typed_rather_than_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is defensive - no int32 reaches it - so it is driven directly.

    Every nonzero positive ``int32`` has a finite, nonzero binary32 reciprocal,
    so this branch cannot be reached through the marker/zero/negative gates.
    It exists so that loosening one of those gates cannot silently produce a
    non-finite scale, and that is exactly what is checked here.
    """
    monkeypatch.setattr(scale_module, "vendor_reciprocal", lambda denominator: math.inf)
    reading = decode_reply(reply(FALLBACK_DENOMINATOR))
    assert reading is not None
    assert reading.outcome is ScaleOutcome.NON_FINITE
    assert reading.scale is None
    assert not reading.valid


def test_no_positive_int32_denominator_can_produce_a_non_finite_scale() -> None:
    for denominator in (1, 2, 3, 200, 1000, 65535, 2**30, 2**31 - 1):
        value = vendor_reciprocal(denominator)
        assert math.isfinite(value)
        assert value > 0.0


@pytest.mark.parametrize("length", [0, 1, 4, 59, 61])
def test_a_reply_that_is_not_exactly_60_bytes_is_never_decoded(length: int) -> None:
    payload = reply(FALLBACK_DENOMINATOR, length=length)
    if length > RESULT_LENGTH:  # pragma: no cover - reply() cannot over-produce
        payload = payload + bytes(length - len(payload))
    assert decode_reply(payload) is None


# --- the vendor's float32 rounding --------------------------------------------


def test_the_scale_is_float32_not_python_float64() -> None:
    """1/3 differs between binary32 and float64; the vendor computes binary32."""
    reading = decode_reply(reply(ROUNDING_DENOMINATOR))
    assert reading is not None and reading.scale is not None
    assert reading.scale == float32(1.0 / 3.0)
    assert reading.scale != 1.0 / 3.0
    assert reading.scale_bits == 0x3EAAAAAB


def test_the_reciprocal_rounds_the_int_conversion_as_cvtsi2ss_does() -> None:
    """An int32 with more than 24 significant bits is rounded on conversion."""
    denominator = 2**24 + 1  # not representable in binary32
    assert float32(float(denominator)) == float(2**24)
    assert vendor_reciprocal(denominator) == float32(1.0 / float(2**24))


def test_the_reported_bit_pattern_round_trips_to_the_reported_value() -> None:
    for denominator in (1, 3, 7, 200, 1000, 2**31 - 1):
        value = vendor_reciprocal(denominator)
        bits = float32_bits(value)
        assert struct.unpack("<f", struct.pack("<I", bits))[0] == value


def test_float32_is_idempotent_on_values_it_produces() -> None:
    for denominator in (1, 3, 200, 2**31 - 1):
        value = vendor_reciprocal(denominator)
        assert float32(value) == value


# --- the SDK fallback is reference context, never a result --------------------


def test_the_sdk_fallback_is_compared_and_never_substituted() -> None:
    reading = decode_reply(reply(FALLBACK_DENOMINATOR))
    assert reading is not None and reading.reference is not None
    assert reading.reference.fallback == SDK_MINI_FALLBACK_SCALE == float32(0.005)
    assert reading.reference.matches is True
    assert reading.reference.ratio == pytest.approx(1.0)
    assert reading.scale == SDK_MINI_FALLBACK_SCALE


def test_a_reading_that_differs_from_the_fallback_is_still_reported_as_read() -> None:
    reading = decode_reply(reply(1000))
    assert reading is not None and reading.reference is not None
    assert reading.reference.matches is False
    assert reading.scale == float32(0.001)
    assert reading.scale != SDK_MINI_FALLBACK_SCALE


@pytest.mark.parametrize("denominator", [0, -5])
def test_an_invalid_reply_gets_no_reference_comparison_at_all(denominator: int) -> None:
    """The fallback is never shown as a stand-in for a value that was not read."""
    reading = decode_reply(reply(denominator))
    assert reading is not None
    assert reading.reference is None
    assert reading.scale is None
    assert reading.as_dict()["vendor_depth_scale"] is None
    assert reading.as_dict()["reference_comparison"] is None


def test_no_report_claims_a_physical_unit(healthy: SysfsBuilder) -> None:
    """Unit words may appear only inside the sentences that DENY the claim."""
    outcome = run_with(healthy, FakeUsbfs())
    payload = outcome.as_dict()
    assert payload["unit"] == UNIT_DISCLAIMER
    assert "unknown from this evidence" in UNIT_DISCLAIMER

    blob = json.dumps(payload).lower()
    assert UNIT_DISCLAIMER.lower() in blob
    # Remove every disclaiming sentence, then no unit vocabulary may survive.
    for disclaimer in (UNIT_DISCLAIMER, scale_module.NO_PROTOCOL_CLEANUP_NOTE):
        blob = blob.replace(disclaimer.lower(), "")
    reading = payload["reading"]
    assert isinstance(reading, dict)
    reference = reading["reference_comparison"]
    assert isinstance(reference, dict)
    blob = blob.replace(str(reference["note"]).lower(), "")
    for claim in ("millimetre", "millimeter", "metre", "meter", " mm", "accuracy"):
        assert claim not in blob, f"the report uses {claim!r} outside a disclaimer"


# --- the happy path -----------------------------------------------------------


def test_a_complete_run_sends_exactly_the_two_whitelisted_transfers(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(replies=[reply(FALLBACK_DENOMINATOR)])
    outcome = run_with(healthy, fake)

    assert outcome.status is OwnershipStatus.OK
    assert outcome.scale_outcome is ScaleOutcome.COMPLETE
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.accepted_scale == float32(0.005)

    assert len(fake.controls) == MAX_CONTROL_TRANSFERS == 2
    assert fake.setups == [EXPECTED_WRITE_SETUP, EXPECTED_READ_SETUP]
    assert fake.written == [PROPERTY_PAYLOAD]
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


def test_the_complete_run_walks_the_lifecycle_in_order(healthy: SysfsBuilder) -> None:
    outcome = run_with(healthy, FakeUsbfs())
    assert phases_of(outcome) == [
        Phase.CONFIRM_DRIVER,
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.CLAIM,
        Phase.OWNED_SEQUENCE,
        Phase.RELEASE,
        Phase.RECONNECT,
    ]
    assert all(result.ok for result in outcome.phases)


def test_the_raw_reply_is_preserved_in_full_with_its_digest(healthy: SysfsBuilder) -> None:
    raw = reply(FALLBACK_DENOMINATOR)
    outcome = run_with(healthy, FakeUsbfs(replies=[raw]))
    assert outcome.reply.length == RESULT_LENGTH
    assert outcome.reply.hex == raw.hex()
    assert outcome.reply.sha256 == hashlib.sha256(raw).hexdigest()
    assert outcome.reply.exact_length
    assert outcome.reply.complete_answer


def test_both_transfers_record_their_timing_and_status(healthy: SysfsBuilder) -> None:
    outcome = run_with(healthy, FakeUsbfs())
    write = outcome.transfer(Step.PROPERTY_WRITE)
    read = outcome.transfer(Step.RESULT_READ)
    assert write is not None and read is not None
    for record in (write, read):
        assert record.ok
        assert record.duration_s is not None and record.duration_s > 0
        assert record.errno is None
    assert write.transferred == PROPERTY_PAYLOAD_LENGTH
    assert write.sent_hex == PROPERTY_PAYLOAD.hex()
    assert write.sent_sha256 == hashlib.sha256(PROPERTY_PAYLOAD).hexdigest()
    assert read.transferred == RESULT_LENGTH
    assert read.response_hex is not None and len(read.response_hex) == RESULT_LENGTH * 2


# --- there is no protocol cleanup, and none is invented -----------------------


def test_the_run_reports_protocol_cleanup_as_not_applicable(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = run_with(healthy, fake)
    assert outcome.protocol_cleanup.attempted is False
    assert outcome.protocol_cleanup.ok is True
    assert "no protocol cleanup" in outcome.protocol_cleanup.note
    assert "NOT borrowed" in outcome.protocol_cleanup.note
    assert Phase.PROTOCOL_CLEANUP not in phases_of(outcome)
    # No 60-zero clearing write, on any branch.
    assert bytes(RESULT_LENGTH) not in fake.written


def test_the_module_contains_no_zero_clearing_payload() -> None:
    assert not hasattr(scale_module, "ZERO_PAYLOAD")
    code = _executable_source(Path(scale_module.__file__))
    assert "ZERO_PAYLOAD" not in code


# --- both transfer boundaries -------------------------------------------------


def test_a_failing_property_write_stops_before_the_read(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={0: OSError(errno.EPIPE, "broken pipe")})
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.TRANSFER_ERROR
    assert not outcome.succeeded
    assert outcome.accepted_scale is None
    assert len(fake.controls) == 1
    assert outcome.reading is None
    assert outcome.reply.length == 0
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


def test_a_short_property_write_stops_before_the_read(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={0: 2})
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.SHORT_TRANSFER
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    assert len(fake.controls) == 1
    assert not outcome.succeeded


def test_a_negative_return_from_the_write_is_a_transfer_error(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={0: -1})
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.TRANSFER_ERROR
    assert len(fake.controls) == 1
    assert not outcome.succeeded


def test_a_failing_result_read_is_typed_and_never_retried(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={1: OSError(errno.ETIMEDOUT, "timed out")})
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.TRANSFER_ERROR
    assert outcome.errno == errno.ETIMEDOUT
    assert len(fake.controls) == MAX_CONTROL_TRANSFERS
    assert outcome.reading is None
    assert outcome.accepted_scale is None
    assert outcome.cleanup.ok


def test_a_short_result_read_is_never_decoded(healthy: SysfsBuilder) -> None:
    short = reply(FALLBACK_DENOMINATOR)[:32]
    fake = FakeUsbfs(replies=[short])
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.SHORT_TRANSFER
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    assert outcome.reading is None
    assert outcome.accepted_scale is None
    # The bytes that did arrive are still preserved as evidence.
    assert outcome.reply.length == 32
    assert not outcome.reply.exact_length
    assert outcome.reply.hex == short.hex()


def test_a_negative_return_from_the_read_is_a_transfer_error(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={1: -1})
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is ScaleOutcome.TRANSFER_ERROR
    assert len(fake.controls) == MAX_CONTROL_TRANSFERS
    assert not outcome.succeeded


# --- every decode outcome, end to end -----------------------------------------


@pytest.mark.parametrize(
    ("payload", "outcome_name", "exit_code"),
    [
        pytest.param(reply(200), ScaleOutcome.COMPLETE, 0, id="complete"),
        pytest.param(reply(200, marker=0), ScaleOutcome.UNEXPECTED_MARKER, 19, id="marker_0"),
        pytest.param(reply(200, marker=5), ScaleOutcome.UNEXPECTED_MARKER, 19, id="marker_5"),
        pytest.param(reply(0), ScaleOutcome.ZERO_DENOMINATOR, 20, id="zero"),
        pytest.param(reply(-7), ScaleOutcome.NEGATIVE_DENOMINATOR, 21, id="negative"),
    ],
)
def test_each_decode_outcome_is_typed_end_to_end(
    healthy: SysfsBuilder, payload: bytes, outcome_name: ScaleOutcome, exit_code: int
) -> None:
    fake = FakeUsbfs(replies=[payload])
    outcome = run_with(healthy, fake)
    assert outcome.scale_outcome is outcome_name
    assert outcome.exit_code == exit_code
    assert len(fake.controls) == MAX_CONTROL_TRANSFERS
    assert outcome.cleanup.ok
    if outcome_name is ScaleOutcome.COMPLETE:
        assert outcome.succeeded and outcome.accepted_scale is not None
    else:
        assert not outcome.succeeded
        assert outcome.accepted_scale is None
        assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
        # The raw bytes are still the primary evidence.
        assert outcome.reply.hex == payload.hex()
        assert outcome.reading is not None


def test_every_scale_outcome_has_a_status_or_is_decided_elsewhere() -> None:
    """No member of the enum is unreachable or unmapped."""
    decided_by_the_lifecycle = {ScaleOutcome.TRANSFER_ERROR, ScaleOutcome.NOT_ATTEMPTED}
    for member in ScaleOutcome:
        if member in decided_by_the_lifecycle:
            continue
        assert member in scale_module._OUTCOME_STATUS


# --- no retry, no third transfer ----------------------------------------------


@pytest.mark.parametrize(
    "fake_kwargs",
    [
        pytest.param({"replies": [reply(200)]}, id="complete"),
        pytest.param({"replies": [reply(0)]}, id="zero_denominator"),
        pytest.param({"replies": [reply(200, marker=9)]}, id="bad_marker"),
        pytest.param({"replies": [reply(200)[:10]]}, id="short_reply"),
        pytest.param({"control_returns": {0: 1}}, id="short_write"),
        pytest.param({"control_errors": {1: OSError(errno.EIO, "io")}}, id="read_error"),
        pytest.param({"control_errors": {0: OSError(errno.EIO, "io")}}, id="write_error"),
    ],
)
def test_no_branch_issues_a_third_transfer_or_repeats_one(
    healthy: SysfsBuilder, fake_kwargs: dict[str, object]
) -> None:
    fake = FakeUsbfs(**fake_kwargs)  # type: ignore[arg-type]
    run_with(healthy, fake)
    assert len(fake.controls) <= MAX_CONTROL_TRANSFERS
    # Each setup packet appears at most once: nothing is ever re-issued.
    for setup in fake.setups:
        assert fake.setups.count(setup) == 1
    assert fake.setups == [EXPECTED_WRITE_SETUP, EXPECTED_READ_SETUP][: len(fake.setups)]


def test_the_source_has_no_loop_around_the_transfers() -> None:
    """The run is straight-line: a loop is how a bound quietly becomes a retry."""
    source = inspect.getsource(scale_module.DepthScaleReadOperation._run)
    tree = ast.parse(source.lstrip())
    for node in ast.walk(tree):
        assert not isinstance(node, ast.For | ast.While)


# --- termination and exceptions at every boundary -----------------------------


@pytest.mark.parametrize("index", [0, 1])
def test_a_signal_at_either_transfer_still_reaches_the_driver_handback(
    healthy: SysfsBuilder, index: int
) -> None:
    def interrupt(control_index: int) -> None:
        if control_index == index:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake = FakeUsbfs(on_control=interrupt)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.scale_outcome is ScaleOutcome.INTERRUPTED
    assert "SIGTERM" in outcome.detail
    assert outcome.reading is None
    assert outcome.accepted_scale is None
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert outcome.protocol_cleanup.ok and not outcome.protocol_cleanup.attempted
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_RELEASEINTERFACE in fake.commands
    assert fake.closed == [7]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(USBDEVFS_DISCONNECT, id="during_the_detach"),
        pytest.param(USBDEVFS_CLAIMINTERFACE, id="during_the_claim"),
    ],
)
def test_a_signal_before_the_write_leaves_nothing_on_the_wire(
    healthy: SysfsBuilder, code: int
) -> None:
    def interrupt(seen: int) -> None:
        if seen == code:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(on_ioctl=interrupt)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert fake.controls == []
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert USBDEVFS_CONNECT in fake.driver_codes


def test_a_signal_during_the_driver_check_leaves_the_device_untouched(
    healthy: SysfsBuilder,
) -> None:
    def interrupt(seen: int) -> None:
        if seen == USBDEVFS_GETDRIVER:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(on_ioctl=interrupt)
    with pytest.raises(ownership_module.SignalReceived):
        run_with(healthy, fake)
    assert fake.controls == []
    assert fake.driver_codes == []
    assert fake.closed == [7]


@pytest.mark.parametrize("index", [0, 1])
def test_an_unexpected_exception_at_either_transfer_still_reconnects(
    healthy: SysfsBuilder, index: int
) -> None:
    """A non-OSError is typed by the lifecycle, and the handback still runs."""

    def explode(control_index: int) -> None:
        if control_index == index:
            raise RuntimeError("something else entirely")

    fake = FakeUsbfs(on_control=explode)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert outcome.scale_outcome is ScaleOutcome.TRANSFER_ERROR
    assert "RuntimeError" in outcome.detail
    assert outcome.reading is None
    assert outcome.accepted_scale is None
    assert not outcome.succeeded
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_RELEASEINTERFACE in fake.commands
    assert fake.closed == [7]
    assert len(fake.controls) == index + 1


# --- driver restoration outranks a perfect reply ------------------------------


@pytest.mark.parametrize(
    "failing",
    [
        pytest.param(USBDEVFS_RELEASEINTERFACE, id="release"),
        pytest.param(USBDEVFS_CONNECT, id="reconnect"),
    ],
)
def test_a_cleanup_failure_overrides_a_valid_reply(healthy: SysfsBuilder, failing: int) -> None:
    fake = FakeUsbfs(
        replies=[reply(FALLBACK_DENOMINATOR)],
        errors={failing: OSError(errno.EIO, "cleanup failed")},
    )
    outcome = run_with(healthy, fake)
    # The reply itself was fine...
    assert outcome.scale_outcome is ScaleOutcome.COMPLETE
    assert outcome.reading is not None and outcome.reading.valid
    # ...and it is still not an accepted scale.
    assert not outcome.cleanup.ok
    assert not outcome.succeeded
    assert outcome.accepted_scale is None
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert outcome.cleanup.guidance is not None
    assert not outcome.reply.complete_answer


# --- dry mode and the serial gate ---------------------------------------------


def test_dry_mode_opens_nothing(healthy: SysfsBuilder) -> None:
    backend = UsbfsDepthScaleBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.preflight(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.NOT_ATTEMPTED
    assert outcome.scale_outcome is ScaleOutcome.NOT_ATTEMPTED
    assert outcome.transfers == ()
    assert outcome.reading is None
    assert outcome.reply.length == 0
    assert outcome.accepted_scale is None
    assert "--execute-depth-scale-read" in outcome.detail


def test_dry_mode_still_reports_the_whole_whitelist(healthy: SysfsBuilder) -> None:
    backend = UsbfsDepthScaleBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    payload = backend.preflight(REAL_SERIAL).as_dict()
    whitelist = payload["whitelist"]
    assert isinstance(whitelist, list) and len(whitelist) == 2
    assert whitelist[0]["payload_hex"] == "18090000"
    assert whitelist[1]["payload_hex"] is None


@pytest.mark.parametrize("serial", ["", "OTHER-SERIAL", REAL_SERIAL.lower(), REAL_SERIAL + "X"])
def test_the_live_path_refuses_any_other_serial(
    healthy: SysfsBuilder, serial: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal only applies when the operator NAMED a device. Configure one,
    # or there is nothing for the guard to refuse against.
    monkeypatch.setenv("SCANNER3D_SERIAL", AUTHORIZED_SERIAL)
    # The refusal only applies when the operator NAMED a device. Configure
    # one, or there is nothing for the guard to refuse against.
    monkeypatch.setenv("SCANNER3D_SERIAL", AUTHORIZED_SERIAL)
    backend = UsbfsDepthScaleBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.run(serial)
    assert outcome.status is OwnershipStatus.REFUSED
    assert outcome.exit_code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    assert AUTHORIZED_SERIAL in outcome.detail
    assert outcome.transfers == ()
    assert outcome.accepted_scale is None


def test_the_authorized_serial_is_the_one_the_owner_named() -> None:
    assert AUTHORIZED_SERIAL == REAL_SERIAL == "EXAMPLESERIAL0001"


# --- the evidence is stable ---------------------------------------------------


def test_the_json_payload_is_serialisable_and_carries_the_whole_record(
    healthy: SysfsBuilder,
) -> None:
    raw = reply(FALLBACK_DENOMINATOR)
    outcome = run_with(healthy, FakeUsbfs(replies=[raw]))
    payload = json.loads(json.dumps(outcome.as_dict()))

    assert payload["mode"] == "vendor-depth-scale-property-read"
    assert payload["property_id"] == "0x918"
    assert payload["control_selector"] == "0x0e"
    assert payload["entity_id"] == 4
    assert payload["owned_interface"] == 0
    assert payload["bounds"] == {
        "max_control_transfers": 2,
        "property_payload_bytes": 4,
        "result_bytes": 60,
        "timeout_ms": 1000,
        "retries": 0,
    }
    assert payload["reply"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["reply"]["hex"] == raw.hex()
    assert payload["reply"]["length"] == 60
    assert payload["reading"]["marker_uint8"] == 4
    assert payload["reading"]["marker_int8"] == 4
    assert payload["reading"]["denominator"] == 200
    assert payload["reading"]["vendor_depth_scale_float32_bits"] == "0x3ba3d70a"
    assert payload["reading"]["unit"] == UNIT_DISCLAIMER
    assert payload["accepted_scale"] == float32(0.005)
    assert payload["succeeded"] is True
    assert len(payload["transfers"]) == 2
    assert payload["protocol_cleanup"]["attempted"] is False


def test_the_json_payload_of_a_refused_run_offers_no_scale(healthy: SysfsBuilder) -> None:
    backend = UsbfsDepthScaleBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    payload = json.loads(json.dumps(backend.run("NOPE").as_dict()))
    assert payload["succeeded"] is False
    assert payload["accepted_scale"] is None
    assert payload["reading"] is None
    assert payload["reply"]["length"] == 0
    assert payload["unit"] == UNIT_DISCLAIMER


def test_a_missing_device_is_reported_without_opening_anything(
    empty_sysfs: SysfsBuilder,
) -> None:
    backend = UsbfsDepthScaleBackend(
        opener=ExplodingOpener(), sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert outcome.transfers == ()
    assert outcome.accepted_scale is None
