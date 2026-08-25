"""Tests for the single owned, read-only control request.

Nothing here touches hardware: every syscall the backend makes is injected, so
these tests can assert on the exact bytes that *would* reach the kernel and on
the exact set of ioctl commands the backend is capable of issuing.
"""

from __future__ import annotations

import ast
import ctypes
import errno
import hashlib
import struct
import tokenize
from dataclasses import replace
from pathlib import Path

import pytest

import scanner3d.control as control_module
from scanner3d.control import (
    CONTROL_EXIT_CODES,
    EXPECTED_ALGORITHM_VERSION,
    EXPECTED_FIRMWARE_VERSION,
    EXPECTED_PAYLOAD_TEXT,
    FIRMWARE_ALGORITHM_VERSION_READ,
    MAX_TIMEOUT_MS,
    USBDEVFS_CONTROL,
    ControlRequest,
    ControlStatus,
    UsbfsControlBackend,
    ensure_allowed,
    pack_ctrltransfer,
    parse_version_payload,
    resolve_target,
)
from scanner3d.errors import ControlRequestNotAllowed, DeviceAbsentError
from tests.conftest import REAL_SERIAL, SysfsBuilder

# usbfs commands this module must never be able to issue. Computed here rather
# than imported, precisely so the production module never names them.
_IOC = control_module._ioc
_U = ord("U")
FORBIDDEN_IOCTLS = {
    "USBDEVFS_BULK": _IOC(3, _U, 2, struct.calcsize("@IIIIP")),
    "USBDEVFS_RESETEP": _IOC(1, _U, 3, 4),
    "USBDEVFS_SETINTERFACE": _IOC(1, _U, 4, 8),
    "USBDEVFS_SETCONFIGURATION": _IOC(1, _U, 5, 4),
    "USBDEVFS_SUBMITURB": _IOC(1, _U, 10, struct.calcsize("@iiBBhiPiiiP")),
    "USBDEVFS_DISCARDURB": _IOC(0, _U, 11, 0),
    "USBDEVFS_CLAIMINTERFACE": _IOC(2, _U, 15, 4),
    "USBDEVFS_RELEASEINTERFACE": _IOC(2, _U, 16, 4),
    "USBDEVFS_IOCTL": _IOC(3, _U, 18, struct.calcsize("@iiP")),
    "USBDEVFS_RESET": _IOC(0, _U, 20, 0),
    "USBDEVFS_CLEAR_HALT": _IOC(2, _U, 21, 4),
    "USBDEVFS_DISCONNECT": _IOC(0, _U, 22, 0),
}

PADDED_OK = EXPECTED_PAYLOAD_TEXT.encode("ascii").ljust(
    FIRMWARE_ALGORITHM_VERSION_READ.length, b"\x00"
)


class FakeSyscalls:
    """Records every open/ioctl/close and writes a canned reply."""

    def __init__(
        self,
        *,
        reply: bytes = PADDED_OK,
        transferred: int | None = None,
        open_error: OSError | None = None,
        ioctl_error: OSError | None = None,
    ) -> None:
        self.reply = reply
        self.transferred = len(reply) if transferred is None else transferred
        self.open_error = open_error
        self.ioctl_error = ioctl_error
        self.opened: list[tuple[str, int]] = []
        self.ioctls: list[tuple[int, int, bytes]] = []
        self.closed: list[int] = []
        self.ticks = iter([1.0, 1.25])

    def open(self, path: str) -> int:
        if self.open_error is not None:
            raise self.open_error
        self.opened.append((path, 7))
        return 7

    def ioctl(self, fd: int, cmd: int, arg: bytearray, mutate: bool) -> int:
        self.ioctls.append((fd, cmd, bytes(arg)))
        if self.ioctl_error is not None:
            raise self.ioctl_error
        *_, address = struct.unpack(control_module._CTRLTRANSFER_FORMAT, bytes(arg))
        ctypes.memmove(address, self.reply, len(self.reply))
        return self.transferred

    def close(self, fd: int) -> None:
        self.closed.append(fd)

    def clock(self) -> float:
        return next(self.ticks, 1.25)


def make_backend(healthy: SysfsBuilder, fake: FakeSyscalls) -> UsbfsControlBackend:
    return UsbfsControlBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )


class ExplodingOpener:
    """An opener that fails the test if it is ever called."""

    def __call__(self, path: str) -> int:
        raise AssertionError(f"a device node was opened: {path}")


# --- the whitelist ------------------------------------------------------------


def test_only_one_request_is_whitelisted() -> None:
    assert list(control_module.ALLOWED_REQUESTS) == ["firmware-algorithm-version"]
    assert FIRMWARE_ALGORITHM_VERSION_READ.wire_tuple == (0xA1, 0x81, 0x0600, 0x0300, 60, 3000)
    assert FIRMWARE_ALGORITHM_VERSION_READ.is_read
    assert FIRMWARE_ALGORITHM_VERSION_READ.timeout_ms <= MAX_TIMEOUT_MS


def test_whitelisted_request_decodes_to_interface_0_entity_3() -> None:
    request = FIRMWARE_ALGORITHM_VERSION_READ
    assert (request.interface_number, request.entity_id) == (0, 3)
    assert request.control_selector == 0x06


def _varied(
    *,
    request_type: int = FIRMWARE_ALGORITHM_VERSION_READ.request_type,
    request: int = FIRMWARE_ALGORITHM_VERSION_READ.request,
    value: int = FIRMWARE_ALGORITHM_VERSION_READ.value,
    index: int = FIRMWARE_ALGORITHM_VERSION_READ.index,
    length: int = FIRMWARE_ALGORITHM_VERSION_READ.length,
    timeout_ms: int = FIRMWARE_ALGORITHM_VERSION_READ.timeout_ms,
) -> ControlRequest:
    """The whitelisted request with exactly one field changed."""
    return ControlRequest(
        name="varied",
        request_type=request_type,
        request=request,
        value=value,
        index=index,
        length=length,
        timeout_ms=timeout_ms,
    )


@pytest.mark.parametrize(
    "varied",
    [
        pytest.param(_varied(request_type=0x21), id="set_cur_write"),
        pytest.param(_varied(request_type=0xA0), id="recipient_device"),
        pytest.param(_varied(request_type=0x21, request=0x01), id="set_cur_pair"),
        pytest.param(_varied(request=0x01), id="request_set_cur"),
        pytest.param(_varied(request=0x82), id="request_get_min"),
        pytest.param(_varied(value=0x0200), id="other_selector"),
        pytest.param(_varied(index=0x0400), id="other_entity"),
        pytest.param(_varied(index=0x0301), id="other_interface"),
        pytest.param(_varied(length=61), id="longer"),
        pytest.param(_varied(length=4), id="shorter"),
        pytest.param(_varied(timeout_ms=0), id="unbounded_timeout"),
        pytest.param(_varied(timeout_ms=MAX_TIMEOUT_MS + 1), id="over_budget"),
        pytest.param(_varied(timeout_ms=60_000), id="way_over_budget"),
    ],
)
def test_varied_requests_are_rejected(varied: ControlRequest) -> None:
    with pytest.raises(ControlRequestNotAllowed):
        ensure_allowed(varied)


def test_rejection_happens_before_any_device_is_opened(healthy: SysfsBuilder) -> None:
    backend = UsbfsControlBackend(
        opener=ExplodingOpener(),
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )
    write = _varied(request_type=0x21, request=0x01, value=0x0100, index=0x0400)
    with pytest.raises(ControlRequestNotAllowed):
        backend.read(REAL_SERIAL, write)
    with pytest.raises(ControlRequestNotAllowed):
        backend.preflight(REAL_SERIAL, write)


def test_a_renamed_request_with_the_same_bytes_is_still_allowed() -> None:
    """The whitelist keys on the wire tuple, not on the label."""
    renamed = replace(FIRMWARE_ALGORITHM_VERSION_READ, name="whatever")
    assert ensure_allowed(renamed) is renamed


# --- byte packing -------------------------------------------------------------


def test_packed_setup_bytes_are_exact() -> None:
    packed = pack_ctrltransfer(FIRMWARE_ALGORITHM_VERSION_READ, 0)
    assert bytes(packed[:12]) == bytes.fromhex("a181000600033c00b80b0000")
    assert len(packed) == struct.calcsize(control_module._CTRLTRANSFER_FORMAT)


def test_packed_struct_carries_the_buffer_address() -> None:
    buffer = ctypes.create_string_buffer(60)
    packed = pack_ctrltransfer(FIRMWARE_ALGORITHM_VERSION_READ, ctypes.addressof(buffer))
    *_, address = struct.unpack(control_module._CTRLTRANSFER_FORMAT, bytes(packed))
    assert address == ctypes.addressof(buffer)


def test_usbdevfs_control_number_matches_the_kernel_encoding() -> None:
    assert USBDEVFS_CONTROL == 0xC0185500


def _executable_source(module_path: Path) -> str:
    """The module's code with comments and string literals removed.

    Prose may (and does) discuss the commands this module refuses to issue;
    what must not appear is a *use* of one.
    """
    with module_path.open("rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))
    return " ".join(
        token.string for token in tokens if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def test_control_is_the_only_usbfs_command_the_module_knows() -> None:
    names = {n for n in dir(control_module) if n.startswith("USBDEVFS_")}
    assert names == {"USBDEVFS_CONTROL"}
    code = _executable_source(Path(control_module.__file__))
    for name in FORBIDDEN_IOCTLS:
        assert name not in code
    for number in FORBIDDEN_IOCTLS.values():
        assert str(number) not in code
        assert f"0x{number:x}" not in code.lower()


def test_module_imports_nothing_that_could_stream_or_load_the_sdk() -> None:
    """An AST check, so 'no SDK, no V4L2, no subprocess' is structural."""
    tree = ast.parse(Path(control_module.__file__).read_text(encoding="utf-8"))
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
        "errno",
        "fcntl",
        "hashlib",
        "os",
        "struct",
        "time",
        "collections.abc",
        "dataclasses",
        "pathlib",
        "types",
        "typing",
        "scanner3d.discovery",
        "scanner3d.config",
        "scanner3d.errors",
        "scanner3d.usb_descriptors",
    }, sorted(imported)


# --- response parsing ---------------------------------------------------------


def test_expected_payload_parses_and_validates() -> None:
    payload = parse_version_payload(PADDED_OK)
    assert payload.firmware_version == EXPECTED_FIRMWARE_VERSION
    assert payload.algorithm_version == EXPECTED_ALGORITHM_VERSION
    assert payload.nul_padded
    assert payload.matches_expected
    assert payload.sha256 == hashlib.sha256(PADDED_OK).hexdigest()
    assert payload.raw_length == 60


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        bytes(60),  # all zero, the SDK's observed failure shape
        b"v2.8.7.20220610".ljust(60, b"\x00"),  # no separator
        b"v0.0.0|v0.0.0".ljust(60, b"\x00"),  # wrong versions
        EXPECTED_PAYLOAD_TEXT.encode("ascii").ljust(60, b"\xff"),  # not NUL padded
        b"\xff\xfe" + bytes(58),  # not ASCII
        EXPECTED_PAYLOAD_TEXT.encode("ascii"),  # right text, wrong length
    ],
)
def test_wrong_payloads_do_not_validate(raw: bytes) -> None:
    payload = parse_version_payload(raw)
    assert not payload.matches_expected
    assert payload.sha256 == hashlib.sha256(raw).hexdigest()


# --- the happy path -----------------------------------------------------------


def test_successful_read_issues_exactly_one_control_ioctl(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls()
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)

    assert outcome.status is ControlStatus.OK
    assert outcome.exit_code == 0
    assert outcome.payload is not None
    assert outcome.payload.firmware_version == EXPECTED_FIRMWARE_VERSION
    assert outcome.payload.algorithm_version == EXPECTED_ALGORITHM_VERSION
    assert outcome.duration_s == pytest.approx(0.25)

    assert len(fake.opened) == 1
    assert [cmd for _, cmd, _ in fake.ioctls] == [USBDEVFS_CONTROL]
    assert fake.closed == [7]

    (_, _, packed) = fake.ioctls[0]
    assert packed[:12] == bytes.fromhex("a181000600033c00b80b0000")


def test_no_forbidden_ioctl_is_ever_issued(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls()
    make_backend(healthy, fake).read(REAL_SERIAL)
    issued = {cmd for _, cmd, _ in fake.ioctls}
    assert issued.isdisjoint(FORBIDDEN_IOCTLS.values())


def test_read_targets_the_serial_resolved_node(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls()
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.target is not None
    assert outcome.target.serial == REAL_SERIAL
    assert outcome.target.node_path == str(healthy.dev / "bus" / "usb" / "001" / "024")
    assert fake.opened[0][0] == outcome.target.node_path
    assert outcome.target.interface_number == 0
    assert outcome.target.interface_driver == "uvcvideo"


def test_read_reports_the_entity_the_wire_index_names(healthy: SysfsBuilder) -> None:
    """wIndex 0x0300 names entity 3, which the device declares as a terminal."""
    outcome = make_backend(healthy, FakeSyscalls()).read(REAL_SERIAL)
    entity = outcome.addressed_entity
    assert entity is not None
    assert entity.entity_id == 3
    assert entity.subtype_name == "output_terminal"
    assert not entity.is_extension_unit


# --- the dry run --------------------------------------------------------------


def test_preflight_opens_nothing(healthy: SysfsBuilder) -> None:
    backend = UsbfsControlBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.preflight(REAL_SERIAL)
    assert outcome.status is ControlStatus.NOT_ATTEMPTED
    assert outcome.exit_code == 0
    assert outcome.payload is None
    assert outcome.duration_s is None
    assert outcome.target is not None


# --- typed failures -----------------------------------------------------------


def test_absent_device_is_reported_not_raised(empty_sysfs: SysfsBuilder) -> None:
    backend = UsbfsControlBackend(
        opener=ExplodingOpener(), sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev
    )
    outcome = backend.read(REAL_SERIAL)
    assert outcome.status is ControlStatus.DEVICE_ABSENT
    assert outcome.exit_code == 2
    assert outcome.target is None


def test_a_different_serial_is_not_targeted(healthy: SysfsBuilder) -> None:
    backend = UsbfsControlBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.read("NOT-THIS-ONE")
    assert outcome.status is ControlStatus.DEVICE_ABSENT
    assert "NOT-THIS-ONE" in outcome.detail


def test_two_matching_devices_without_a_serial_is_refused(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini()
    builder.add_usb_device(usb_path="1-5", serial="OTHER", busnum="1", devnum="25")
    with pytest.raises(DeviceAbsentError, match="pass --serial"):
        resolve_target(
            None,
            FIRMWARE_ALGORITHM_VERSION_READ,
            sysfs_root=builder.sysfs,
            dev_root=builder.dev,
        )


def test_permission_failure_is_distinct(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(open_error=PermissionError(errno.EACCES, "Permission denied"))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.PERMISSION_DENIED
    assert outcome.exit_code == 3
    assert outcome.errno_name == "EACCES"
    assert fake.ioctls == []
    assert fake.closed == []


def test_disconnect_during_open_is_reported_as_absent(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(open_error=OSError(errno.ENODEV, "No such device"))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.DEVICE_ABSENT
    assert outcome.errno_name == "ENODEV"


def test_timeout_is_distinct_from_other_ioctl_errors(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(ioctl_error=OSError(errno.ETIMEDOUT, "Connection timed out"))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.TIMEOUT
    assert outcome.exit_code == 4
    assert outcome.duration_s is not None
    assert fake.closed == [7]


def test_busy_interface_is_an_ioctl_error(healthy: SysfsBuilder) -> None:
    """What a kernel driver holding the interface actually produces."""
    fake = FakeSyscalls(ioctl_error=OSError(errno.EBUSY, "Device or resource busy"))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.IOCTL_ERROR
    assert outcome.exit_code == 5
    assert outcome.errno_name == "EBUSY"
    assert fake.closed == [7]


def test_short_response_is_distinct(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(reply=b"v2.8.7" + bytes(6), transferred=12)
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.SHORT_RESPONSE
    assert outcome.exit_code == 7
    assert outcome.payload is not None
    assert outcome.payload.raw_length == 12


def test_unexpected_content_is_distinct(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(reply=bytes(60))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == 6
    assert outcome.payload is not None
    assert not outcome.payload.matches_expected


def test_malformed_non_ascii_response_is_unexpected_content(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(reply=bytes(range(60)))
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.UNEXPECTED_CONTENT
    assert outcome.payload is not None
    assert outcome.payload.decode_error is not None


def test_reenumeration_between_resolve_and_transfer_is_caught(healthy: SysfsBuilder) -> None:
    """The device is replaced under the same node while we hold it open."""
    fake = FakeSyscalls()
    backend = make_backend(healthy, fake)
    sysfs_device = healthy.devices / "1-4.1"

    real_open = fake.open

    def open_then_replug(path: str) -> int:
        fd = real_open(path)
        (sysfs_device / "serial").write_text("A-DIFFERENT-UNIT\n", encoding="utf-8")
        return fd

    outcome = replace(backend, opener=open_then_replug).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.DEVICE_REPLACED
    assert outcome.exit_code == 8
    assert fake.ioctls == []
    assert fake.closed == [7]


def test_devnum_change_between_resolve_and_transfer_is_caught(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls()
    backend = make_backend(healthy, fake)
    sysfs_device = healthy.devices / "1-4.1"

    real_open = fake.open

    def open_then_renumber(path: str) -> int:
        fd = real_open(path)
        (sysfs_device / "devnum").write_text("42\n", encoding="utf-8")
        return fd

    outcome = replace(backend, opener=open_then_renumber).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.DEVICE_REPLACED
    assert "devnum" in outcome.detail
    assert fake.ioctls == []
    assert fake.closed == [7]


def test_missing_usbfs_node_is_absent(builder: SysfsBuilder) -> None:
    builder.add_usb_device()  # no add_usbfs()
    backend = UsbfsControlBackend(
        opener=ExplodingOpener(), sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    outcome = backend.read(REAL_SERIAL)
    assert outcome.status is ControlStatus.DEVICE_ABSENT
    assert "cannot stat" in outcome.detail


def test_fd_is_closed_even_when_parsing_the_reply_fails(healthy: SysfsBuilder) -> None:
    fake = FakeSyscalls(reply=bytes(60), transferred=-1)
    outcome = make_backend(healthy, fake).read(REAL_SERIAL)
    assert outcome.status is ControlStatus.IOCTL_ERROR
    assert fake.closed == [7]


# --- outcome shape ------------------------------------------------------------


def test_every_status_has_an_exit_code() -> None:
    assert set(CONTROL_EXIT_CODES) == set(ControlStatus)
    assert len(CONTROL_EXIT_CODES) == len(ControlStatus)


def test_status_mappings_declare_no_duplicate_keys_in_source() -> None:
    """A repeated dict key is silent at runtime, so assert it at the source level."""
    tree = ast.parse(Path(control_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [ast.unparse(key) for key in node.keys if key is not None]
        assert len(keys) == len(set(keys)), f"duplicate dict key at line {node.lineno}: {keys}"


def test_outcome_as_dict_is_json_shaped(healthy: SysfsBuilder) -> None:
    outcome = make_backend(healthy, FakeSyscalls()).read(REAL_SERIAL)
    payload = outcome.as_dict()
    assert payload["status"] == "ok"
    request = payload["request"]
    assert isinstance(request, dict)
    assert request["bmRequestType"] == "0xa1"
    assert request["wIndex"] == "0x0300"
    assert request["direction"] == "in"


def test_control_request_is_immutable() -> None:
    with pytest.raises((AttributeError, TypeError)):
        FIRMWARE_ALGORITHM_VERSION_READ.value = 0x0200  # type: ignore[misc]


def test_allowed_requests_mapping_cannot_be_extended() -> None:
    smuggled = ControlRequest(
        name="smuggled",
        request_type=0x21,
        request=0x01,
        value=0x0100,
        index=0x0400,
        length=60,
        timeout_ms=1000,
    )
    with pytest.raises(TypeError):
        control_module.ALLOWED_REQUESTS["smuggled"] = smuggled  # type: ignore[index]
