"""Tests for the one bounded read of ``/data/camparam/Pl.bin``.

Nothing here touches hardware. Every syscall is injected, so these tests assert
on the exact setup bytes, the exact outbound payload, the assembly of the
length-prefixed chunks, and - most of the point, because this is the second
thing in the project that writes - that every one of the fixed bounds holds,
that no transfer is ever retried, that the clearing write and the driver
handback both happen after *every* way the middle can fail, and that a run
which did not earn it never offers the bytes as calibration.

The module under test is also checked as a *closed whitelist*: it can name one
key, three transfers and one usbfs command, and no argument anywhere can widen
that.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import inspect
import json
import math
import signal
import struct
from dataclasses import replace
from pathlib import Path

import pytest

import scanner3d.control as control_module
import scanner3d.intrinsics as intr
import scanner3d.ownership as ownership_module
from scanner3d.control import pack_ctrltransfer
from scanner3d.errors import IntrinsicsReadNotAllowed
from scanner3d.intrinsics import (
    AUTHORIZED_SERIAL,
    CHUNK_READ_TRANSFER,
    CLEAR_WRITE_TRANSFER,
    ENTITY_ID,
    INTRINSICS_FIELDS,
    INTRINSICS_FORMAT,
    INTRINSICS_KEY,
    INTRINSICS_LENGTH,
    MAX_CHUNK_LENGTH,
    MAX_CONTROL_TRANSFERS,
    MAX_GET_TRANSFERS,
    OUTPUT_CAPACITY,
    PATH_PAYLOAD,
    PATH_WRITE_TRANSFER,
    PAYLOAD_LENGTH,
    TIMEOUT_MS,
    WHITELIST,
    ZERO_PAYLOAD,
    IntrinsicsOutcome,
    ReadOutcome,
    Step,
    UsbfsIntrinsicsBackend,
    advertised_depth_modes,
    ensure_transfer_allowed,
    ensure_whitelist_intact,
    parse_intrinsics,
    read_depth_modes,
)
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    PROTOCOL_CLEANUP_FAILED_EXIT,
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONNECT,
    USBDEVFS_DISCONNECT,
    USBDEVFS_GETDRIVER,
    OwnershipStatus,
    Phase,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.test_control import ExplodingOpener
from tests.test_init_sequence import FORBIDDEN_IOCTLS, FakeUsbfs, _executable_source

#: The depth modes the recorded descriptor of this exact unit advertises.
FIXTURE_DEPTH_MODES = ((640, 400), (1280, 800), (640, 800))

#: The lifecycle a fully successful bounded read walks, in order.
FULL_PHASES = [
    Phase.CONFIRM_DRIVER,  # from sysfs, in preflight
    Phase.CONFIRM_DRIVER,  # from USBDEVFS_GETDRIVER, on the open descriptor
    Phase.DISCONNECT,
    Phase.CLAIM,
    Phase.OWNED_SEQUENCE,
    Phase.PROTOCOL_CLEANUP,
    Phase.RELEASE,
    Phase.RECONNECT,
]


# --- fixtures for the wire ----------------------------------------------------


def chunk_reply(data: bytes, declared: int | None = None) -> bytes:
    """One 60-byte result reply: a ``uint32_le`` length word then the data."""
    length = len(data) if declared is None else declared
    body = length.to_bytes(4, "little") + data
    assert len(body) <= PAYLOAD_LENGTH
    return body + bytes(PAYLOAD_LENGTH - len(body))


#: The reply that ends the data: a zero length word.
TERMINATOR = chunk_reply(b"")


def record_bytes(
    width: int = 640,
    height: int = 400,
    *,
    fx: float = 560.5,
    fy: float = 560.5,
    cx: float = 320.25,
    cy: float = 200.75,
    zero01: float = 0.0,
    zero10: float = 0.0,
    zero20: float = 0.0,
    zero21: float = 0.0,
    one22: float = 1.0,
) -> bytes:
    """A 40-byte vendor ``Intrinsics`` record, by default a well-formed one."""
    return struct.pack(
        INTRINSICS_FORMAT, width, height, fx, zero01, cx, zero10, fy, cy, zero20, zero21, one22
    )


#: A complete, structurally exact record for a mode this unit advertises.
GOOD_RECORD = record_bytes()


def make_backend(builder: SysfsBuilder, fake: FakeUsbfs) -> UsbfsIntrinsicsBackend:
    return UsbfsIntrinsicsBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def run_with(builder: SysfsBuilder, fake: FakeUsbfs) -> IntrinsicsOutcome:
    return make_backend(builder, fake).run(REAL_SERIAL)


def setups_of(fake: FakeUsbfs) -> list[str]:
    return [setup.hex() for setup, _ in fake.controls]


def phases_of(outcome: IntrinsicsOutcome) -> list[Phase]:
    return [p.phase for p in outcome.phases]


def steps_of(outcome: IntrinsicsOutcome) -> list[Step]:
    return [t.step for t in outcome.transfers]


# --- the closed whitelist -----------------------------------------------------


def test_the_module_knows_exactly_one_usbfs_command() -> None:
    """Every driver-management ioctl belongs to the lifecycle, not to this module."""
    names = {n for n in dir(intr) if n.startswith("USBDEVFS_")}
    assert names == {"USBDEVFS_CONTROL"}
    code = _executable_source(Path(intr.__file__))
    for name, number in FORBIDDEN_IOCTLS.items():
        assert name not in code
        assert str(number) not in code
        assert f"0x{number:x}" not in code.lower()


def test_module_imports_nothing_that_could_stream_or_load_the_sdk() -> None:
    tree = ast.parse(Path(intr.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert imported <= {
        "__future__",
        "collections.abc",
        "ctypes",
        "dataclasses",
        "enum",
        "hashlib",
        "math",
        "pathlib",
        "struct",
        "types",
        "typing",
        "scanner3d.control",
        "scanner3d.discovery",
        "scanner3d.errors",
        "scanner3d.ownership",
    }, sorted(imported)


def test_the_only_device_file_key_in_the_module_is_the_depth_intrinsics_one() -> None:
    """The sibling calibration keys and the licence paths are out of reach.

    Checked against the executable source, so the module docstring may name
    them while naming them as things it cannot send.
    """
    code = _executable_source(Path(intr.__file__))
    assert INTRINSICS_KEY == "/data/camparam/Pl.bin"
    for forbidden in ("Prgb", "LC_RT", "Distort", "persist", "sdk-license", "/tmp/inited"):
        assert forbidden not in code
    assert PATH_PAYLOAD.count(b"/") == 3
    assert PATH_PAYLOAD.startswith(b"\x01" + INTRINSICS_KEY.encode("ascii") + b"\x00")


def test_the_depth_scale_channel_is_not_reachable_from_here() -> None:
    """``wValue=0x0e00`` is a different experiment and is not in this module."""
    code = _executable_source(Path(intr.__file__))
    assert "0x0e00" not in code.lower()
    assert "0x0E00" not in code
    for transfer in WHITELIST:
        assert transfer.request.value in (intr.PATH_SELECTOR << 8, intr.RESULT_SELECTOR << 8)


def test_the_whitelist_is_exactly_three_transfers_in_the_only_order() -> None:
    assert [t.step for t in WHITELIST] == [Step.PATH_WRITE, Step.CHUNK_READ, Step.CLEAR_WRITE]
    assert WHITELIST[0].payload == PATH_PAYLOAD
    assert WHITELIST[1].payload is None
    assert WHITELIST[2].payload == ZERO_PAYLOAD
    assert bytes(PAYLOAD_LENGTH) == ZERO_PAYLOAD


def test_the_bounds_are_the_fixed_ones() -> None:
    assert OUTPUT_CAPACITY == 256
    assert MAX_GET_TRANSFERS == 8
    assert MAX_CONTROL_TRANSFERS == 10
    assert MAX_CHUNK_LENGTH == 56
    assert PAYLOAD_LENGTH == 60
    assert TIMEOUT_MS == 1000
    assert INTRINSICS_LENGTH == 40
    assert ENTITY_ID == 4


def test_packed_setup_bytes_are_exact() -> None:
    """The first twelve bytes of the ioctl argument are the setup packet."""
    expected = {
        Step.PATH_WRITE: "2101000100043c00e8030000",
        Step.CHUNK_READ: "a181000200043c00e8030000",
        Step.CLEAR_WRITE: "2101000100043c00e8030000",
    }
    for transfer in WHITELIST:
        packed = pack_ctrltransfer(transfer.request, 0)
        assert packed[:12].hex() == expected[transfer.step], transfer.step


def test_the_path_payload_is_the_key_nul_terminated_and_zero_padded() -> None:
    assert len(PATH_PAYLOAD) == PAYLOAD_LENGTH
    assert PATH_PAYLOAD.hex() == (
        "01" + INTRINSICS_KEY.encode("ascii").hex() + "00" + "00" * (PAYLOAD_LENGTH - 23)
    )
    assert PATH_PAYLOAD[0] == intr.READ_FILE_OPCODE == 0x01


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"value": 0x0E00}, id="depth_scale_selector"),
        pytest.param({"value": 0x0300}, id="another_selector"),
        pytest.param({"index": (5 << 8) | OWNED_INTERFACE}, id="another_entity"),
        pytest.param({"index": (ENTITY_ID << 8) | 1}, id="another_interface"),
        pytest.param({"request": 0x02}, id="another_brequest"),
        pytest.param({"request_type": 0xA1}, id="another_direction"),
        pytest.param({"length": 64}, id="another_length"),
        pytest.param({"timeout_ms": 0}, id="zero_timeout"),
        pytest.param({"timeout_ms": control_module.MAX_TIMEOUT_MS + 1}, id="over_long_timeout"),
    ],
)
def test_a_varied_setup_packet_is_rejected(mutation: dict[str, int]) -> None:
    tampered = replace(
        PATH_WRITE_TRANSFER,
        request=replace(PATH_WRITE_TRANSFER.request, **mutation),  # type: ignore[arg-type]
    )
    with pytest.raises(IntrinsicsReadNotAllowed):
        ensure_transfer_allowed(tampered)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            (b"\x01/data/camparam/Prgb.bin\x00").ljust(PAYLOAD_LENGTH, b"\x00"),
            id="a_sibling_calibration_key",
        ),
        pytest.param(
            (b"\x02" + INTRINSICS_KEY.encode("ascii") + b"\x00").ljust(PAYLOAD_LENGTH, b"\x00"),
            id="the_write_file_opcode",
        ),
        pytest.param(
            (b"\x01/persist/sdk-license\x00").ljust(PAYLOAD_LENGTH, b"\x00"),
            id="a_licence_path",
        ),
        pytest.param(PATH_PAYLOAD[:-1], id="a_short_payload"),
        pytest.param(PATH_PAYLOAD + b"\x00", id="an_over_long_payload"),
    ],
)
def test_a_varied_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(IntrinsicsReadNotAllowed):
        ensure_transfer_allowed(replace(PATH_WRITE_TRANSFER, payload=payload))


def test_a_read_that_carries_a_payload_is_rejected() -> None:
    with pytest.raises(IntrinsicsReadNotAllowed):
        ensure_transfer_allowed(replace(CHUNK_READ_TRANSFER, payload=PATH_PAYLOAD))


@pytest.mark.parametrize(
    "whitelist",
    [
        pytest.param(
            (CLEAR_WRITE_TRANSFER, CHUNK_READ_TRANSFER, PATH_WRITE_TRANSFER), id="reordered"
        ),
        pytest.param((PATH_WRITE_TRANSFER, CHUNK_READ_TRANSFER), id="truncated"),
        pytest.param((*WHITELIST, CHUNK_READ_TRANSFER), id="extended"),
        pytest.param((), id="empty"),
    ],
)
def test_a_reshaped_whitelist_is_rejected(whitelist: tuple[object, ...]) -> None:
    with pytest.raises(IntrinsicsReadNotAllowed):
        ensure_whitelist_intact(whitelist)  # type: ignore[arg-type]


def test_the_intact_whitelist_is_accepted() -> None:
    ensure_whitelist_intact()
    for transfer in WHITELIST:
        assert ensure_transfer_allowed(transfer) is transfer


def test_transfers_are_immutable() -> None:
    with pytest.raises((AttributeError, TypeError)):
        PATH_WRITE_TRANSFER.payload = ZERO_PAYLOAD  # type: ignore[misc]


def test_no_public_function_takes_a_path_or_a_bound() -> None:
    """Nothing a caller can pass names a key, an address or a limit."""
    forbidden = {
        "path",
        "key",
        "file",
        "selector",
        "entity",
        "entity_id",
        "interface",
        "timeout",
        "timeout_ms",
        "cap",
        "capacity",
        "count",
        "repeat",
        "retry",
        "retries",
        "request",
        "request_type",
        "value",
        "index",
        "length",
        "payload",
        "opcode",
    }
    for name, obj in vars(intr).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != intr.__name__:
            continue
        assert forbidden.isdisjoint(inspect.signature(obj).parameters), name


def test_the_read_is_not_a_parameter_of_anything() -> None:
    for method in (UsbfsIntrinsicsBackend.run, UsbfsIntrinsicsBackend.preflight):
        assert list(inspect.signature(method).parameters) == ["self", "serial"]


def test_a_tampered_whitelist_stops_the_run_before_anything_is_opened(
    healthy: SysfsBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intr, "WHITELIST", (PATH_WRITE_TRANSFER,))
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    with pytest.raises(IntrinsicsReadNotAllowed):
        backend.run(REAL_SERIAL)
    with pytest.raises(IntrinsicsReadNotAllowed):
        backend.preflight(REAL_SERIAL)


# --- the record ---------------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 4, 39, 41, 56, 256])
def test_only_a_forty_byte_blob_parses(length: int) -> None:
    assert parse_intrinsics(bytes(length)) is None


def test_a_well_formed_record_parses_exactly_as_it_arrived() -> None:
    record = parse_intrinsics(GOOD_RECORD, FIXTURE_DEPTH_MODES)
    assert record is not None
    assert (record.width, record.height) == (640, 400)
    assert record.values["fx"] == 560.5
    assert record.values["fy"] == 560.5
    assert record.values["cx"] == 320.25
    assert record.values["cy"] == 200.75
    assert list(record.values) == list(INTRINSICS_FIELDS)
    assert record.matrix == (
        (560.5, 0.0, 320.25),
        (0.0, 560.5, 200.75),
        (0.0, 0.0, 1.0),
    )
    assert record.all_finite
    assert record.structurally_expected
    assert record.plausible


def test_the_parsed_values_are_the_unpacked_bytes_and_nothing_else() -> None:
    record = parse_intrinsics(GOOD_RECORD)
    assert record is not None
    width, height, *floats = struct.unpack(INTRINSICS_FORMAT, GOOD_RECORD)
    assert (record.width, record.height) == (width, height)
    assert list(record.values.values()) == floats


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        pytest.param("zero01", 3.5, "zero01_is_0", id="skew_is_not_zero"),
        pytest.param("zero10", 1.0, "zero10_is_0", id="zero10_diverges"),
        pytest.param("zero20", 0.5, "zero20_is_0", id="zero20_diverges"),
        pytest.param("zero21", 0.5, "zero21_is_0", id="zero21_diverges"),
        pytest.param("one22", 0.0, "one22_is_1", id="bottom_right_is_not_one"),
    ],
)
def test_a_structurally_divergent_record_is_evidence_not_a_crash(
    field: str, value: float, check: str
) -> None:
    record = parse_intrinsics(
        record_bytes(**{field: value}),  # type: ignore[arg-type]
        FIXTURE_DEPTH_MODES,
    )
    assert record is not None
    assert not record.structurally_expected
    assert record.values[field] == value
    failed = [c.name for c in record.structure if not c.ok]
    assert failed == [check]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_float_is_reported_not_repaired(value: float) -> None:
    record = parse_intrinsics(record_bytes(fx=value), FIXTURE_DEPTH_MODES)
    assert record is not None
    assert not record.all_finite
    assert not record.structurally_expected
    assert not record.plausible
    finite_check = next(c for c in record.structure if c.name == "all_floats_finite")
    assert not finite_check.ok
    if math.isnan(value):
        assert math.isnan(record.values["fx"])
    else:
        assert record.values["fx"] == value


def test_dimensions_are_compared_only_to_the_advertised_depth_modes() -> None:
    record = parse_intrinsics(record_bytes(1024, 768), FIXTURE_DEPTH_MODES)
    assert record is not None
    assert record.structurally_expected  # the divergence is not structural
    check = next(
        c for c in record.plausibility if c.name == "dimensions_match_an_advertised_depth_mode"
    )
    assert not check.ok
    assert "640x400" in check.detail
    assert not record.plausible


def test_a_principal_point_outside_the_frame_is_flagged_but_kept() -> None:
    record = parse_intrinsics(record_bytes(cx=9000.0), FIXTURE_DEPTH_MODES)
    assert record is not None
    assert record.values["cx"] == 9000.0
    assert record.structurally_expected
    assert not record.plausible
    check = next(c for c in record.plausibility if c.name == "principal_point_inside_frame")
    assert not check.ok


def test_the_record_dictionary_labels_the_units_and_keeps_the_checks_apart() -> None:
    record = parse_intrinsics(GOOD_RECORD, FIXTURE_DEPTH_MODES)
    assert record is not None
    payload = record.as_dict()
    assert "pixels" in str(payload["units"])
    assert payload["matrix"] == [[560.5, 0.0, 320.25], [0.0, 560.5, 200.75], [0.0, 0.0, 1.0]]
    structure = payload["structure_checks"]
    plausibility = payload["plausibility_checks"]
    assert isinstance(structure, list)
    assert isinstance(plausibility, list)
    assert {c["name"] for c in structure} == {
        "all_floats_finite",
        "zero01_is_0",
        "zero10_is_0",
        "zero20_is_0",
        "zero21_is_0",
        "one22_is_1",
    }
    assert {c["name"] for c in plausibility} == {
        "positive_dimensions",
        "positive_focal_lengths",
        "principal_point_inside_frame",
        "principal_point_near_frame_centre",
        "dimensions_match_an_advertised_depth_mode",
    }


def test_depth_modes_come_from_the_devices_own_descriptors(healthy: SysfsBuilder) -> None:
    assert read_depth_modes(REAL_SERIAL, sysfs_root=healthy.sysfs, dev_root=healthy.dev) == (
        FIXTURE_DEPTH_MODES
    )


def test_depth_modes_are_empty_when_no_device_is_attached(empty_sysfs: SysfsBuilder) -> None:
    assert (
        read_depth_modes(REAL_SERIAL, sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev) == ()
    )


def test_advertised_depth_modes_of_a_device_without_a_depth_function(
    builder: SysfsBuilder,
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, interface_string="something else")
    from scanner3d.discovery import find_scanners

    device = find_scanners(sysfs_root=builder.sysfs, dev_root=builder.dev)[0]
    if device.function("depth") is None:
        assert advertised_depth_modes(device) == ()
    else:  # the recorded descriptor still declares the modes
        assert advertised_depth_modes(device) == FIXTURE_DEPTH_MODES


# --- the bounded read: the successful shape -----------------------------------


def test_a_complete_read_yields_the_record_and_both_undos(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR])
    outcome = run_with(healthy, fake)

    assert outcome.status is OwnershipStatus.OK
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.get_transfers == 2
    assert steps_of(outcome) == [
        Step.PATH_WRITE,
        Step.CHUNK_READ,
        Step.CHUNK_READ,
        Step.CLEAR_WRITE,
    ]
    assert phases_of(outcome) == FULL_PHASES

    assert outcome.blob.length == INTRINSICS_LENGTH
    assert outcome.blob.sha256 == hashlib.sha256(GOOD_RECORD).hexdigest()
    assert outcome.blob.hex == GOOD_RECORD.hex()
    assert outcome.blob.terminated
    assert outcome.blob.complete_answer

    assert outcome.record is not None
    assert outcome.accepted_calibration is outcome.record
    assert outcome.record.matrix[0][0] == 560.5
    assert outcome.depth_modes == FIXTURE_DEPTH_MODES

    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert setups_of(fake) == [
        "2101000100043c00e8030000",
        "a181000200043c00e8030000",
        "a181000200043c00e8030000",
        "2101000100043c00e8030000",
    ]
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


def test_chunks_are_assembled_in_the_order_they_arrive(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD[:20]), chunk_reply(GOOD_RECORD[20:]), TERMINATOR]
    )
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.blob.hex == GOOD_RECORD.hex()
    assert outcome.get_transfers == 3
    reads = [t for t in outcome.transfers if t.step is Step.CHUNK_READ]
    assert [t.declared_length for t in reads] == [20, 20, 0]
    assert [t.copied_bytes for t in reads] == [20, 20, None]
    assert reads[0].chunk_sha256 == hashlib.sha256(GOOD_RECORD[:20]).hexdigest()
    assert len(fake.controls) == 5


def test_a_record_that_is_only_implausible_is_still_a_complete_read(
    healthy: SysfsBuilder,
) -> None:
    """Plausibility is a separate axis: it never turns a read into a failure."""
    blob = record_bytes(1024, 768)
    fake = FakeUsbfs(replies=[chunk_reply(blob), TERMINATOR])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.succeeded
    assert outcome.record is not None
    assert outcome.record.structurally_expected
    assert not outcome.record.plausible
    assert "plausibility" in outcome.detail


# --- every stop condition -----------------------------------------------------


def test_a_zero_first_chunk_is_no_content_and_says_the_cause_is_ambiguous(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(replies=[TERMINATOR])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.NO_CONTENT
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == 15
    assert not outcome.succeeded
    assert outcome.blob.length == 0
    assert not outcome.blob.terminated
    assert outcome.record is None
    assert outcome.accepted_calibration is None
    assert "indistinguishable" in outcome.detail
    assert outcome.protocol_cleanup.ok and outcome.cleanup.ok
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_the_read_cap_stops_a_device_that_never_terminates(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(b"\x11" * 8) for _ in range(MAX_GET_TRANSFERS)])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.CAP_EXHAUSTED
    assert outcome.exit_code == 16
    assert outcome.get_transfers == MAX_GET_TRANSFERS
    assert outcome.blob.length == 8 * MAX_GET_TRANSFERS
    assert not outcome.blob.terminated
    assert outcome.record is None
    assert "not widened" in outcome.detail
    assert len(fake.controls) == MAX_CONTROL_TRANSFERS
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_the_capacity_stops_before_a_chunk_that_would_pass_it(healthy: SysfsBuilder) -> None:
    """The chunk that would overflow is not copied, and not truncated either."""
    full = bytes(range(MAX_CHUNK_LENGTH))
    fake = FakeUsbfs(replies=[chunk_reply(full) for _ in range(5)])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.CAPACITY_EXHAUSTED
    assert outcome.exit_code == 17
    assert outcome.blob.length == 4 * MAX_CHUNK_LENGTH == 224
    assert outcome.blob.length <= OUTPUT_CAPACITY
    assert outcome.get_transfers == 5
    assert "not copied" in outcome.detail
    reads = [t for t in outcome.transfers if t.step is Step.CHUNK_READ]
    assert [t.copied_bytes for t in reads] == [56, 56, 56, 56, None]
    assert outcome.record is None


def test_a_declared_length_over_the_maximum_is_rejected_before_any_copy(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(b"", declared=MAX_CHUNK_LENGTH + 1)])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.MALFORMED_LENGTH
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == 18
    assert outcome.blob.length == 0
    assert outcome.record is None
    read = next(t for t in outcome.transfers if t.step is Step.CHUNK_READ)
    assert read.declared_length == MAX_CHUNK_LENGTH + 1
    assert read.chunk_hex is None
    assert read.copied_bytes is None
    assert read.response_hex is not None
    assert read.response_sha256 is not None


def test_a_malformed_length_keeps_the_bytes_already_collected_as_evidence(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD[:20]), chunk_reply(b"", declared=1000)],
    )
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.MALFORMED_LENGTH
    assert outcome.blob.hex == GOOD_RECORD[:20].hex()
    assert not outcome.blob.terminated
    assert not outcome.blob.complete_answer
    assert outcome.record is None
    assert outcome.accepted_calibration is None


def test_a_short_result_transfer_stops_the_read(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR], control_returns={1: 32})
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.SHORT_TRANSFER
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    assert outcome.exit_code == OWNERSHIP_EXIT_CODES[OwnershipStatus.SHORT_RESPONSE]
    assert "32 of 60" in outcome.detail
    assert outcome.record is None
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_a_short_key_write_stops_the_read_and_still_clears(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR], control_returns={0: 30})
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.SHORT_TRANSFER
    assert outcome.get_transfers == 0
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert steps_of(outcome) == [Step.PATH_WRITE, Step.CLEAR_WRITE]


@pytest.mark.parametrize(
    ("number", "status", "code"),
    [
        pytest.param(errno.EIO, OwnershipStatus.IOCTL_ERROR, 5, id="eio"),
        pytest.param(errno.ETIMEDOUT, OwnershipStatus.TIMEOUT, 4, id="etimedout"),
        pytest.param(errno.EACCES, OwnershipStatus.PERMISSION_DENIED, 3, id="eacces"),
        pytest.param(errno.ENODEV, OwnershipStatus.DEVICE_ABSENT, 2, id="enodev"),
    ],
)
def test_a_failed_result_read_is_typed_by_its_errno(
    healthy: SysfsBuilder, number: int, status: OwnershipStatus, code: int
) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD), TERMINATOR],
        control_errors={1: OSError(number, "injected")},
    )
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.TRANSFER_ERROR
    assert outcome.status is status
    assert outcome.exit_code == code
    assert outcome.errno == number
    assert outcome.record is None
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok


def test_a_failed_key_write_still_owes_the_clearing_write(healthy: SysfsBuilder) -> None:
    """The write may have landed even though the ioctl reported failure."""
    fake = FakeUsbfs(replies=[TERMINATOR], control_errors={0: OSError(errno.EIO, "injected")})
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.TRANSFER_ERROR
    assert outcome.get_transfers == 0
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_a_terminated_blob_of_the_wrong_length_is_evidence_not_calibration(
    healthy: SysfsBuilder,
) -> None:
    payload = b"\x01" * 20
    fake = FakeUsbfs(replies=[chunk_reply(payload), TERMINATOR])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.UNEXPECTED_CONTENT
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == OWNERSHIP_EXIT_CODES[OwnershipStatus.UNEXPECTED_CONTENT]
    assert outcome.blob.terminated
    assert outcome.blob.hex == payload.hex()
    assert outcome.record is None
    assert outcome.accepted_calibration is None
    assert not outcome.succeeded
    assert "not accepted calibration" in outcome.detail


def test_a_structurally_divergent_record_is_reported_but_never_accepted(
    healthy: SysfsBuilder,
) -> None:
    blob = record_bytes(one22=0.0)
    fake = FakeUsbfs(replies=[chunk_reply(blob), TERMINATOR])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.UNEXPECTED_CONTENT
    assert not outcome.succeeded
    assert outcome.record is not None
    assert outcome.record.values["one22"] == 0.0
    assert not outcome.record.structurally_expected
    assert outcome.accepted_calibration is None
    assert outcome.blob.hex == blob.hex()
    assert "one22_is_1" in outcome.detail


# --- no retry, ever -----------------------------------------------------------


def test_a_failed_transfer_is_never_repeated(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD), TERMINATOR],
        control_errors={1: OSError(errno.EIO, "injected")},
    )
    outcome = run_with(healthy, fake)
    assert len(fake.controls) == 3
    assert setups_of(fake).count("a181000200043c00e8030000") == 1
    assert steps_of(outcome) == [Step.PATH_WRITE, Step.CHUNK_READ, Step.CLEAR_WRITE]


def test_no_run_can_issue_more_than_the_fixed_number_of_transfers(
    healthy: SysfsBuilder,
) -> None:
    for replies in (
        [TERMINATOR],
        [chunk_reply(GOOD_RECORD), TERMINATOR],
        [chunk_reply(b"\x11" * 8) for _ in range(MAX_GET_TRANSFERS)],
        [chunk_reply(bytes(MAX_CHUNK_LENGTH)) for _ in range(5)],
    ):
        fake = FakeUsbfs(replies=list(replies))
        run_with(healthy, fake)
        assert len(fake.controls) <= MAX_CONTROL_TRANSFERS
        assert fake.written.count(ZERO_PAYLOAD) == 1
        assert fake.written.count(PATH_PAYLOAD) == 1


def test_the_operation_refuses_to_clear_twice(healthy: SysfsBuilder) -> None:
    """The non-retry guard, exercised directly rather than through the lifecycle."""
    fake = FakeUsbfs(replies=[])
    operation = intr.IntrinsicsReadOperation(
        ioctl_fn=fake.ioctl, clock=fake.clock, usb_path="1-4.1"
    )
    operation.key_write_attempted = True
    first = operation.protocol_cleanup(7)
    assert first.ok
    second = operation.protocol_cleanup(7)
    assert not second.ok
    assert "never repeated" in second.note
    assert fake.written == [ZERO_PAYLOAD]


def test_the_clearing_write_is_not_owed_when_the_key_write_never_happened() -> None:
    fake = FakeUsbfs(replies=[])
    operation = intr.IntrinsicsReadOperation(
        ioctl_fn=fake.ioctl, clock=fake.clock, usb_path="1-4.1"
    )
    record = operation.protocol_cleanup(7)
    assert not record.attempted
    assert record.ok
    assert fake.controls == []


# --- both cleanups, in every combination --------------------------------------


def test_a_failed_clearing_write_outranks_a_perfect_read(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD), TERMINATOR],
        control_errors={3: OSError(errno.EIO, "injected")},
    )
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.status is OwnershipStatus.OK
    assert not outcome.succeeded
    assert outcome.exit_code == PROTOCOL_CLEANUP_FAILED_EXIT
    assert not outcome.protocol_cleanup.ok
    assert outcome.protocol_cleanup.guidance is not None
    assert INTRINSICS_KEY in outcome.protocol_cleanup.guidance
    assert outcome.cleanup.ok
    assert outcome.record is not None  # kept as unverified evidence
    assert outcome.accepted_calibration is None
    assert not outcome.blob.complete_answer
    assert "not calibration" in outcome.blob.note


def test_a_failed_driver_handback_outranks_everything(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD), TERMINATOR],
        errors={USBDEVFS_CONNECT: OSError(errno.EBUSY, "injected")},
    )
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.protocol_cleanup.ok
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert not outcome.succeeded
    assert outcome.accepted_calibration is None
    assert outcome.cleanup.guidance is not None


def test_when_both_undos_fail_the_driver_one_is_reported_first(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(
        replies=[chunk_reply(GOOD_RECORD), TERMINATOR],
        control_errors={3: OSError(errno.EIO, "injected")},
        errors={USBDEVFS_CONNECT: OSError(errno.EBUSY, "injected")},
    )
    outcome = run_with(healthy, fake)
    assert not outcome.protocol_cleanup.ok
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert not outcome.succeeded
    assert outcome.accepted_calibration is None


# --- termination at every boundary --------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_signal_during_any_transfer_still_clears_and_reconnects(
    healthy: SysfsBuilder, index: int
) -> None:
    def interrupt(control_index: int) -> None:
        if control_index == index:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR], on_control=interrupt)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.read_outcome is ReadOutcome.INTERRUPTED
    assert outcome.exit_code == 13
    assert "SIGTERM" in outcome.detail
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    assert outcome.record is None
    assert outcome.accepted_calibration is None
    assert fake.written[-1] == ZERO_PAYLOAD
    assert fake.written.count(ZERO_PAYLOAD) == 1
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(USBDEVFS_DISCONNECT, id="during_the_detach"),
        pytest.param(USBDEVFS_CLAIMINTERFACE, id="during_the_claim"),
    ],
)
def test_a_signal_before_the_key_write_leaves_nothing_to_clear(
    healthy: SysfsBuilder, code: int
) -> None:
    def interrupt(seen: int) -> None:
        if seen == code:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(replies=[TERMINATOR], on_ioctl=interrupt)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert fake.controls == []
    assert not outcome.protocol_cleanup.attempted
    assert outcome.protocol_cleanup.ok
    assert Phase.PROTOCOL_CLEANUP not in phases_of(outcome)
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert USBDEVFS_CONNECT in fake.driver_codes


def test_a_signal_during_the_driver_check_leaves_the_device_untouched(
    healthy: SysfsBuilder,
) -> None:
    def interrupt(seen: int) -> None:
        if seen == USBDEVFS_GETDRIVER:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(replies=[TERMINATOR], on_ioctl=interrupt)
    with pytest.raises(ownership_module.SignalReceived):
        run_with(healthy, fake)
    assert fake.controls == []
    assert fake.driver_codes == []
    assert fake.closed == [7]


def test_an_unexpected_python_exception_still_clears_and_reconnects(
    healthy: SysfsBuilder,
) -> None:
    def explode(index: int) -> None:
        if index == 1:
            raise RuntimeError("something else entirely")

    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR], on_control=explode)
    outcome = run_with(healthy, fake)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert outcome.read_outcome is ReadOutcome.TRANSFER_ERROR
    assert "RuntimeError" in outcome.detail
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    assert outcome.record is None
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert fake.closed == [7]


# --- the dry path -------------------------------------------------------------


def test_a_dry_run_opens_nothing(healthy: SysfsBuilder) -> None:
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.preflight(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.NOT_ATTEMPTED
    assert outcome.read_outcome is ReadOutcome.NOT_ATTEMPTED
    assert outcome.exit_code == 0
    assert not outcome.succeeded
    assert "--execute-calibration-read" in outcome.detail
    assert outcome.transfers == ()
    assert outcome.blob.length == 0
    assert outcome.blob.note == "nothing was read"
    assert not outcome.protocol_cleanup.attempted
    assert not outcome.cleanup.attempted


def test_a_dry_run_without_a_device_is_device_absent(empty_sysfs: SysfsBuilder) -> None:
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev
    )
    outcome = backend.preflight(None)
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert outcome.exit_code == 2


def test_the_live_path_refuses_any_serial_but_the_authorized_one(
    healthy: SysfsBuilder,
) -> None:
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.run("NOT-THE-AUTHORIZED-UNIT")
    assert outcome.status is OwnershipStatus.REFUSED
    assert outcome.exit_code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    assert AUTHORIZED_SERIAL in outcome.detail
    assert outcome.transfers == ()
    assert outcome.target is None
    assert not outcome.protocol_cleanup.attempted


def test_the_authorized_serial_is_this_units_serial() -> None:
    assert AUTHORIZED_SERIAL == REAL_SERIAL


# --- serialisation ------------------------------------------------------------


TOP_LEVEL_KEYS = {
    "mode",
    "status",
    "read_outcome",
    "succeeded",
    "exit_code",
    "detail",
    "key",
    "owned_interface",
    "entity_id",
    "bounds",
    "whitelist",
    "target",
    "phases",
    "transfers",
    "get_transfers",
    "blob",
    "record",
    "accepted_calibration",
    "advertised_depth_modes",
    "protocol_cleanup",
    "cleanup",
    "read_duration_s",
    "total_duration_s",
    "errno",
    "errno_name",
}


def test_the_dry_payload_has_the_stable_shape(healthy: SysfsBuilder) -> None:
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    payload = backend.preflight(REAL_SERIAL).as_dict()
    assert set(payload) == TOP_LEVEL_KEYS
    assert payload["mode"] == "depth-intrinsics-file-read"
    assert payload["key"] == INTRINSICS_KEY
    assert payload["record"] is None
    assert payload["accepted_calibration"] is False
    assert json.loads(json.dumps(payload)) == payload


def test_the_live_payload_has_the_same_shape_and_carries_the_evidence(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(replies=[chunk_reply(GOOD_RECORD), TERMINATOR])
    payload = run_with(healthy, fake).as_dict()
    assert set(payload) == TOP_LEVEL_KEYS
    assert payload["read_outcome"] == "complete"
    assert payload["succeeded"] is True
    assert payload["accepted_calibration"] is True
    blob = payload["blob"]
    assert isinstance(blob, dict)
    assert blob["hex"] == GOOD_RECORD.hex()
    assert blob["sha256"] == hashlib.sha256(GOOD_RECORD).hexdigest()
    assert blob["length"] == INTRINSICS_LENGTH
    record = payload["record"]
    assert isinstance(record, dict)
    assert record["width"] == 640
    assert record["matrix"] == [[560.5, 0.0, 320.25], [0.0, 560.5, 200.75], [0.0, 0.0, 1.0]]
    assert payload["advertised_depth_modes"] == [
        {"width": 640, "height": 400},
        {"width": 1280, "height": 800},
        {"width": 640, "height": 800},
    ]
    assert json.loads(json.dumps(payload)) == payload


def test_the_payload_states_the_bounds_and_the_whole_whitelist(healthy: SysfsBuilder) -> None:
    backend = UsbfsIntrinsicsBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    payload = backend.preflight(REAL_SERIAL).as_dict()
    assert payload["bounds"] == {
        "output_capacity_bytes": OUTPUT_CAPACITY,
        "max_get_transfers": MAX_GET_TRANSFERS,
        "max_control_transfers": MAX_CONTROL_TRANSFERS,
        "max_chunk_length_bytes": MAX_CHUNK_LENGTH,
        "timeout_ms": TIMEOUT_MS,
    }
    whitelist = payload["whitelist"]
    assert isinstance(whitelist, list)
    assert [entry["step"] for entry in whitelist] == ["path_write", "chunk_read", "clear_write"]
    assert whitelist[0]["payload_hex"] == PATH_PAYLOAD.hex()
    assert whitelist[1]["payload_hex"] is None
    assert whitelist[2]["payload_hex"] == ZERO_PAYLOAD.hex()


# --- the record this exact unit actually returned -----------------------------

#: The 40 bytes the attached MINI ``A22704522A6F00A64`` returned for
#: ``/data/camparam/Pl.bin`` on 2026-08-21, preserved verbatim from the one
#: live run (``/tmp/3d-scanner-intrinsics/live.json``). Pinned here so the
#: parse of the real factory record is exercised offline, forever.
MEASURED_BLOB = bytes.fromhex(
    "80029001b6ab604400000000f9bf294200000000b6ab60442bd5404300000000000000000000803f"
)

MEASURED_BLOB_SHA256 = "fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf"


def test_the_measured_factory_record_parses_as_the_vendor_layout() -> None:
    assert len(MEASURED_BLOB) == INTRINSICS_LENGTH
    assert hashlib.sha256(MEASURED_BLOB).hexdigest() == MEASURED_BLOB_SHA256
    record = parse_intrinsics(MEASURED_BLOB, FIXTURE_DEPTH_MODES)
    assert record is not None
    assert (record.width, record.height) == (640, 400)
    assert record.values["fx"] == pytest.approx(898.6829833984375)
    assert record.values["fy"] == pytest.approx(898.6829833984375)
    assert record.values["cx"] == pytest.approx(42.43747329711914)
    assert record.values["cy"] == pytest.approx(192.8326873779297)
    assert record.all_finite
    assert record.structurally_expected


def test_the_measured_records_off_centre_principal_point_is_reported() -> None:
    """cx sits 277 px left of the 640x400 centre. That is measured, not repaired."""
    record = parse_intrinsics(MEASURED_BLOB, FIXTURE_DEPTH_MODES)
    assert record is not None
    centre = next(c for c in record.plausibility if c.name == "principal_point_near_frame_centre")
    assert not centre.ok
    assert not record.plausible
    # Every other plausibility observation holds, so the flag is specific.
    assert [c.name for c in record.plausibility if not c.ok] == [
        "principal_point_near_frame_centre"
    ]
    # And it is not a structural divergence: the fixed entries are exact.
    assert record.structurally_expected


def test_a_centred_principal_point_passes_the_centre_observation() -> None:
    record = parse_intrinsics(record_bytes(cx=320.0, cy=200.0), FIXTURE_DEPTH_MODES)
    assert record is not None
    centre = next(c for c in record.plausibility if c.name == "principal_point_near_frame_centre")
    assert centre.ok
    assert record.plausible


def test_the_centre_observation_never_changes_the_read_outcome(
    healthy: SysfsBuilder,
) -> None:
    """A flagged principal point is an observation; the read still completed."""
    fake = FakeUsbfs(replies=[chunk_reply(MEASURED_BLOB), TERMINATOR])
    outcome = run_with(healthy, fake)
    assert outcome.read_outcome is ReadOutcome.COMPLETE
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.record is not None
    assert not outcome.record.plausible
    assert outcome.accepted_calibration is outcome.record
    assert outcome.blob.sha256 == MEASURED_BLOB_SHA256
