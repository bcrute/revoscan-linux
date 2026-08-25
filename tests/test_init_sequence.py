"""Tests for the one bounded reproduction of the SDK's /tmp/inited exchange.

Nothing here touches hardware. Every syscall is injected, so these tests assert
on the exact setup bytes, the exact outbound payloads, the order of the four
transfers, and - most of the point, because this is the first thing in the
project that writes - that the clearing write and the driver handback both
happen after *every* way the middle can fail, are never repeated, and are never
reported as success when they do not.
"""

from __future__ import annotations

import ast
import ctypes
import errno
import inspect
import signal
import struct
import tokenize
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import scanner3d.control as control_module
import scanner3d.init_sequence as init_module
import scanner3d.ownership as ownership_module
from scanner3d.control import USBDEVFS_CONTROL
from scanner3d.errors import InitSequenceNotAllowed
from scanner3d.init_sequence import (
    CLEARING_TRANSFER,
    INITED_PATH,
    INITED_SEQUENCE,
    OBSERVED_ALGORITHM_RECORD,
    OBSERVED_ZERO_RESPONSE,
    PATH_PAYLOAD,
    ZERO_PAYLOAD,
    InitTransfer,
    Step,
    UsbfsInitSequenceBackend,
    ensure_sequence_allowed,
    ensure_transfer_allowed,
    parse_response,
)
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    EXPECTED_INTERFACE_DRIVER,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    PROTOCOL_CLEANUP_FAILED_EXIT,
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONNECT,
    USBDEVFS_DISCONNECT,
    USBDEVFS_GETDRIVER,
    USBDEVFS_IOCTL,
    USBDEVFS_RELEASEINTERFACE,
    OwnershipStatus,
    Phase,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder

_IOC = control_module._ioc
_U = ord("U")

#: usbfs commands this module must never be able to issue.
FORBIDDEN_IOCTLS = {
    "USBDEVFS_BULK": _IOC(3, _U, 2, struct.calcsize("@IIIIP")),
    "USBDEVFS_RESETEP": _IOC(1, _U, 3, 4),
    "USBDEVFS_SETINTERFACE": _IOC(1, _U, 4, 8),
    "USBDEVFS_SETCONFIGURATION": _IOC(1, _U, 5, 4),
    "USBDEVFS_SUBMITURB": _IOC(1, _U, 10, struct.calcsize("@iiBBhiPiiiP")),
    "USBDEVFS_DISCARDURB": _IOC(0, _U, 11, 0),
    "USBDEVFS_RESET": _IOC(0, _U, 20, 0),
    "USBDEVFS_CLEAR_HALT": _IOC(2, _U, 21, 4),
    "USBDEVFS_CLAIM_PORT": _IOC(2, _U, 24, 4),
    "USBDEVFS_DISCONNECT_CLAIM": _IOC(1, _U, 27, struct.calcsize("@II256s")),
}

#: The whole lifecycle of a write-class run, in the only order it may occur.
FULL_ORDER = [
    Phase.CONFIRM_DRIVER,  # from sysfs, in preflight
    Phase.CONFIRM_DRIVER,  # from USBDEVFS_GETDRIVER, on the open descriptor
    Phase.DISCONNECT,
    Phase.CLAIM,
    Phase.OWNED_SEQUENCE,
    Phase.PROTOCOL_CLEANUP,
    Phase.RELEASE,
    Phase.RECONNECT,
]

#: The usbfs commands a fully successful run issues, in order.
FULL_COMMANDS = [
    USBDEVFS_GETDRIVER,
    USBDEVFS_IOCTL,  # disconnect
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONTROL,  # path write
    USBDEVFS_CONTROL,  # first read
    USBDEVFS_CONTROL,  # second read
    USBDEVFS_CONTROL,  # clearing write
    USBDEVFS_RELEASEINTERFACE,
    USBDEVFS_IOCTL,  # connect
]


class FakeUsbfs:
    """Records every open/ioctl/close and answers each usbfs command.

    Control transfers are recorded as ``(setup prefix, outbound bytes)`` so a
    test can assert exactly what would have reached the wire, and are keyed by
    their index in the run so a single transfer can be made to fail.
    """

    def __init__(
        self,
        *,
        driver: str = EXPECTED_INTERFACE_DRIVER,
        replies: list[bytes] | None = None,
        open_error: OSError | None = None,
        errors: dict[int, OSError] | None = None,
        control_errors: dict[int, OSError] | None = None,
        control_returns: dict[int, int] | None = None,
        on_control: object = None,
        on_ioctl: object = None,
    ) -> None:
        self.driver = driver
        self.replies = list(
            replies if replies is not None else [OBSERVED_ALGORITHM_RECORD, OBSERVED_ZERO_RESPONSE]
        )
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
        if cmd == USBDEVFS_IOCTL:
            _, code, _ = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, bytes(arg))
        if callable(self.on_ioctl):
            self.on_ioctl(code)
        if cmd == USBDEVFS_CONTROL:
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
        *_, address = struct.unpack(control_module._CTRLTRANSFER_FORMAT, packed)
        outbound = ctypes.string_at(address, init_module.PAYLOAD_LENGTH)
        self.controls.append((packed[:12], outbound))
        if callable(self.on_control):
            self.on_control(index)
        if index in self.control_errors:
            raise self.control_errors[index]
        if packed[0] & 0x80:
            reply = self.replies.pop(0) if self.replies else bytes(init_module.PAYLOAD_LENGTH)
            ctypes.memmove(address, reply, len(reply))
            return self.control_returns.get(index, len(reply))
        return self.control_returns.get(index, init_module.PAYLOAD_LENGTH)

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
            if cmd == USBDEVFS_IOCTL:
                _, code, _ = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, arg)
                codes.append(code)
        return codes

    @property
    def written(self) -> list[bytes]:
        """The outbound payload of every host-to-device transfer."""
        return [data for setup, data in self.controls if not setup[0] & 0x80]


class ExplodingOpener:
    """An opener that fails the test if it is ever called."""

    def __call__(self, path: str) -> int:
        raise AssertionError(f"a device node was opened: {path}")


def make_backend(builder: SysfsBuilder, fake: FakeUsbfs) -> UsbfsInitSequenceBackend:
    return UsbfsInitSequenceBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def phases_of(outcome: init_module.InitSequenceOutcome) -> list[Phase]:
    return [p.phase for p in outcome.phases]


def steps_of(outcome: init_module.InitSequenceOutcome) -> list[Step]:
    return [t.step for t in outcome.transfers]


def _executable_source(path: Path) -> str:
    """Module source with comments and string literals removed."""
    with path.open("rb") as handle:
        return "".join(
            token.string
            for token in tokenize.tokenize(handle.readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )


# --- the safety boundary ------------------------------------------------------


def test_the_module_knows_exactly_one_usbfs_command() -> None:
    """Every driver-management ioctl belongs to the lifecycle, not to this module."""
    names = {n for n in dir(init_module) if n.startswith("USBDEVFS_")}
    assert names == {"USBDEVFS_CONTROL"}
    code = _executable_source(Path(init_module.__file__))
    for name, number in FORBIDDEN_IOCTLS.items():
        assert name not in code
        assert str(number) not in code
        assert f"0x{number:x}" not in code.lower()


def test_module_imports_nothing_that_could_stream_or_load_the_sdk() -> None:
    tree = ast.parse(Path(init_module.__file__).read_text(encoding="utf-8"))
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
        "struct",
        "dataclasses",
        "typing",
        "scanner3d.control",
        "scanner3d.config",
        "scanner3d.errors",
        "scanner3d.ownership",
    }, sorted(imported)


def test_the_only_path_in_the_module_is_tmp_inited() -> None:
    """The other two paths in the same capture are out of reach from here.

    Checked against the executable source, so the docstring may name them while
    naming them as things this module cannot send.
    """
    code = _executable_source(Path(init_module.__file__))
    assert INITED_PATH == "/tmp/inited"
    assert "sdk-license" not in code
    assert "persist" not in code
    assert PATH_PAYLOAD.count(b"/") == 2


def test_the_sequence_is_exactly_four_transfers_in_the_observed_order() -> None:
    assert [t.step for t in INITED_SEQUENCE] == [
        Step.PATH_WRITE,
        Step.RESULT_READ,
        Step.SECOND_READ,
        Step.CLEAR_WRITE,
    ]
    assert len(INITED_SEQUENCE) == len(set(INITED_SEQUENCE))
    assert CLEARING_TRANSFER is INITED_SEQUENCE[3]
    ensure_sequence_allowed()


def test_every_transfer_addresses_entity_4_on_interface_0() -> None:
    for transfer in INITED_SEQUENCE:
        assert transfer.request.entity_id == init_module.ENTITY_ID == 4
        assert transfer.request.interface_number == OWNED_INTERFACE == 0
        assert transfer.request.length == init_module.PAYLOAD_LENGTH == 60
        assert 0 < transfer.request.timeout_ms <= control_module.MAX_TIMEOUT_MS == 3000


def test_packed_setup_bytes_are_exact() -> None:
    """The three distinct setup packets, byte for byte, as the kernel sees them."""
    packed = {
        t.step: bytes(control_module.pack_ctrltransfer(t.request, 0)[:12]) for t in INITED_SEQUENCE
    }
    # 0x21 0x01, wValue 0x0100 -> 00 01, wIndex 0x0400 -> 00 04, wLength 60 -> 3c 00,
    # timeout 3000 ms -> b8 0b 00 00
    assert packed[Step.PATH_WRITE] == bytes.fromhex("2101000100043c00b80b0000")
    assert packed[Step.CLEAR_WRITE] == packed[Step.PATH_WRITE]
    # 0xa1 0x81, wValue 0x0200 -> 00 02, wIndex 0x0400 -> 00 04
    assert packed[Step.RESULT_READ] == bytes.fromhex("a181000200043c00b80b0000")
    assert packed[Step.SECOND_READ] == packed[Step.RESULT_READ]


def test_outbound_payload_bytes_are_exact() -> None:
    opcode_then_path = b"\x01" + b"/tmp/inited" + b"\x00" + bytes(47)
    sixty_zeros = bytes(60)
    assert opcode_then_path == PATH_PAYLOAD
    assert len(PATH_PAYLOAD) == 60
    assert sixty_zeros == ZERO_PAYLOAD
    assert INITED_SEQUENCE[0].payload == PATH_PAYLOAD
    assert INITED_SEQUENCE[3].payload == ZERO_PAYLOAD
    assert INITED_SEQUENCE[1].payload is None
    assert INITED_SEQUENCE[2].payload is None


def test_the_observed_responses_are_exactly_what_the_capture_recorded() -> None:
    length_then_version = bytes.fromhex("0c000000") + b"v7.5.7.0426" + b"\x00" + bytes(44)
    sixty_zeros = bytes(60)
    assert length_then_version == OBSERVED_ALGORITHM_RECORD
    assert len(OBSERVED_ALGORITHM_RECORD) == 60
    assert sixty_zeros == OBSERVED_ZERO_RESPONSE


def _varied(base: InitTransfer, **changes: Any) -> InitTransfer:
    """One transfer with one field altered, for the rejection table below."""
    request_changes: dict[str, Any] = {k: v for k, v in changes.items() if k != "payload"}
    request = replace(base.request, **request_changes) if request_changes else base.request
    payload: Any = changes.get("payload", base.payload)
    return replace(base, request=request, payload=payload)


@pytest.mark.parametrize(
    "transfer",
    [
        pytest.param(_varied(INITED_SEQUENCE[0], value=0x0300), id="other_selector"),
        pytest.param(_varied(INITED_SEQUENCE[0], index=0x0300), id="other_entity"),
        pytest.param(_varied(INITED_SEQUENCE[0], index=0x0404), id="interface_4_hid"),
        pytest.param(_varied(INITED_SEQUENCE[0], request_type=0xA1), id="direction_flipped"),
        pytest.param(_varied(INITED_SEQUENCE[0], request=0x82), id="get_min"),
        pytest.param(_varied(INITED_SEQUENCE[0], length=61), id="longer"),
        pytest.param(_varied(INITED_SEQUENCE[0], length=4), id="shorter"),
        pytest.param(_varied(INITED_SEQUENCE[0], timeout_ms=0), id="unbounded_timeout"),
        pytest.param(_varied(INITED_SEQUENCE[0], timeout_ms=3001), id="over_budget"),
        pytest.param(
            _varied(INITED_SEQUENCE[0], payload=b"\x01/persist/sdk-license\x00" + bytes(38)),
            id="other_path",
        ),
        pytest.param(
            _varied(INITED_SEQUENCE[0], payload=b"\x02" + b"/tmp/inited\x00" + bytes(47)),
            id="other_opcode",
        ),
        pytest.param(_varied(INITED_SEQUENCE[0], payload=bytes(59)), id="short_payload"),
        pytest.param(_varied(INITED_SEQUENCE[1], payload=ZERO_PAYLOAD), id="read_with_a_payload"),
        pytest.param(_varied(INITED_SEQUENCE[3], payload=b"\x01" * 60), id="clear_carrying_junk"),
    ],
)
def test_varied_transfers_are_rejected(transfer: InitTransfer) -> None:
    with pytest.raises(InitSequenceNotAllowed):
        ensure_transfer_allowed(transfer)


@pytest.mark.parametrize(
    "sequence",
    [
        pytest.param(INITED_SEQUENCE[:3], id="clearing_write_dropped"),
        pytest.param(INITED_SEQUENCE[:2] + INITED_SEQUENCE[3:], id="second_read_dropped"),
        pytest.param(tuple(reversed(INITED_SEQUENCE)), id="reversed"),
        pytest.param((*INITED_SEQUENCE, CLEARING_TRANSFER), id="repeated"),
        pytest.param((), id="empty"),
    ],
)
def test_a_reordered_or_resized_sequence_is_rejected(sequence: tuple[InitTransfer, ...]) -> None:
    with pytest.raises(InitSequenceNotAllowed):
        ensure_sequence_allowed(sequence)


def test_the_whitelist_keys_on_bytes_not_on_labels() -> None:
    """A ``clear_write`` carrying the path payload *is* the path write, byte for byte.

    It is allowed for that reason, and it is not a way to send anything new:
    the operation issues :data:`_ATTEMPTED_IN_ORDER` and
    :data:`CLEARING_TRANSFER`, which are module constants, so a relabelled
    transfer has nothing to reach the wire through.
    """
    relabelled = replace(INITED_SEQUENCE[3], payload=PATH_PAYLOAD)
    assert ensure_transfer_allowed(relabelled) is relabelled
    assert relabelled.identity == INITED_SEQUENCE[0].identity


def test_a_renamed_transfer_with_the_same_bytes_is_still_allowed() -> None:
    """The whitelist keys on the setup packet and the payload, not on the label."""
    renamed = replace(
        INITED_SEQUENCE[0], request=replace(INITED_SEQUENCE[0].request, name="whatever")
    )
    assert ensure_transfer_allowed(renamed) is renamed


def test_the_sequence_is_checked_before_any_device_is_opened(
    healthy: SysfsBuilder, monkeypatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise InitSequenceNotAllowed("nope")

    backend = UsbfsInitSequenceBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    monkeypatch.setattr(init_module, "ensure_sequence_allowed", refuse)
    with pytest.raises(InitSequenceNotAllowed):
        backend.run(REAL_SERIAL)
    with pytest.raises(InitSequenceNotAllowed):
        backend.preflight(REAL_SERIAL)


def test_the_sequence_is_not_a_parameter_of_anything() -> None:
    for name in ("run", "preflight"):
        parameters = list(inspect.signature(getattr(UsbfsInitSequenceBackend, name)).parameters)
        assert parameters == ["self", "serial"], (name, parameters)


def test_transfers_are_immutable() -> None:
    with pytest.raises((AttributeError, TypeError)):
        INITED_SEQUENCE[0].payload = ZERO_PAYLOAD  # type: ignore[misc]


def test_exit_codes_cover_every_status_and_neither_cleanup_code_is_among_them() -> None:
    assert set(OWNERSHIP_EXIT_CODES) == set(OwnershipStatus)
    values = set(OWNERSHIP_EXIT_CODES.values())
    assert CLEANUP_FAILED_EXIT not in values
    assert PROTOCOL_CLEANUP_FAILED_EXIT not in values
    # distinct, so the owner can tell a stranded driver from an uncleared path
    assert {CLEANUP_FAILED_EXIT, PROTOCOL_CLEANUP_FAILED_EXIT} == {9, 14}


# --- response parsing ---------------------------------------------------------


def test_the_observed_algorithm_record_parses_as_a_length_prefixed_string() -> None:
    parsed = parse_response(OBSERVED_ALGORITHM_RECORD, OBSERVED_ALGORITHM_RECORD)
    assert parsed.matches_observed
    assert parsed.declared_length == 12
    assert parsed.text == "v7.5.7.0426"
    assert parsed.nul_padded
    assert not parsed.all_zero
    assert parsed.decode_error is None
    assert parsed.raw_length == 60
    assert parsed.hex_prefix == OBSERVED_ALGORITHM_RECORD[:16].hex()


def test_an_all_zero_reply_is_recognised_without_being_called_an_error() -> None:
    parsed = parse_response(bytes(60), OBSERVED_ZERO_RESPONSE)
    assert parsed.all_zero
    assert parsed.matches_observed
    assert parsed.declared_length == 0
    assert parsed.text == ""
    assert parsed.decode_error is None


def test_a_reply_that_is_neither_shape_is_reported_as_a_divergence() -> None:
    raw = bytes.fromhex("ff000000") + b"nonsense" + bytes(48)
    parsed = parse_response(raw, OBSERVED_ALGORITHM_RECORD)
    assert not parsed.matches_observed
    assert parsed.declared_length == 255
    assert parsed.decode_error is not None
    assert "255" in parsed.decode_error


def test_a_truncated_reply_never_raises() -> None:
    parsed = parse_response(b"\x0c\x00", OBSERVED_ALGORITHM_RECORD)
    assert parsed.raw_length == 2
    assert parsed.declared_length is None
    assert not parsed.matches_observed
    assert parsed.decode_error is not None


def test_a_non_ascii_reply_never_raises() -> None:
    raw = bytes.fromhex("0c000000") + b"\xff\xfe\xfd" + bytes(53)
    parsed = parse_response(raw, OBSERVED_ALGORITHM_RECORD)
    assert parsed.text is None
    assert parsed.decode_error is not None
    assert not parsed.matches_observed


# --- preconditions ------------------------------------------------------------


def test_preflight_opens_nothing(healthy: SysfsBuilder) -> None:
    backend = UsbfsInitSequenceBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.preflight(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.NOT_ATTEMPTED
    assert outcome.exit_code == 0
    assert outcome.transfers == ()
    assert not outcome.protocol_cleanup.attempted
    assert not outcome.cleanup.attempted
    assert outcome.protocol_cleanup.ok and outcome.cleanup.ok
    assert phases_of(outcome) == [Phase.CONFIRM_DRIVER]


def test_absent_device_is_refused_before_anything_opens(empty_sysfs: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(empty_sysfs, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert outcome.exit_code == 2
    assert fake.opened == []
    assert outcome.transfers == ()


def test_a_different_serial_is_not_targeted(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run("NOT-THIS-ONE")
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert fake.controls == []


@pytest.mark.parametrize("driver", [None, "usbfs", "snd-usb-audio"])
def test_interface_0_must_be_held_by_uvcvideo(builder: SysfsBuilder, driver: str | None) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, driver=driver)
    fake = FakeUsbfs()
    outcome = make_backend(builder, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert outcome.exit_code == 10
    assert fake.opened == []
    assert not outcome.protocol_cleanup.attempted


def test_permission_failure_on_open_writes_nothing(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(open_error=PermissionError(errno.EACCES, "Permission denied"))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.PERMISSION_DENIED
    assert outcome.exit_code == 3
    assert fake.calls == []
    assert not outcome.protocol_cleanup.attempted
    assert not outcome.cleanup.attempted


def test_reenumeration_between_resolve_and_detach_writes_nothing(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    backend = make_backend(healthy, fake)

    def replug(path: str) -> int:
        (healthy.devices / "1-4.1" / "serial").write_text("SOMETHINGELSE\n", encoding="utf-8")
        return fake.open(path)

    outcome = UsbfsInitSequenceBackend(
        opener=replug,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    ).run(REAL_SERIAL)
    assert backend is not None
    assert outcome.status is OwnershipStatus.DEVICE_REPLACED
    assert outcome.exit_code == 8
    assert fake.calls == []
    assert fake.closed == [7]


# --- the happy path -----------------------------------------------------------


def test_a_successful_run_issues_exactly_the_four_transfers_in_order(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.OK
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert fake.commands == FULL_COMMANDS
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.opened == [str(healthy.dev / "bus" / "usb" / "001" / "024")]
    assert fake.closed == [7]
    assert steps_of(outcome) == [
        Step.PATH_WRITE,
        Step.RESULT_READ,
        Step.SECOND_READ,
        Step.CLEAR_WRITE,
    ]
    assert phases_of(outcome) == FULL_ORDER


def test_a_successful_run_puts_exactly_the_observed_bytes_on_the_wire(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs()
    make_backend(healthy, fake).run(REAL_SERIAL)
    setups = [setup for setup, _ in fake.controls]
    assert setups == [
        bytes.fromhex("2101000100043c00b80b0000"),
        bytes.fromhex("a181000200043c00b80b0000"),
        bytes.fromhex("a181000200043c00b80b0000"),
        bytes.fromhex("2101000100043c00b80b0000"),
    ]
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_a_successful_run_reports_both_replies(healthy: SysfsBuilder) -> None:
    import hashlib

    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    first = outcome.transfer(Step.RESULT_READ)
    second = outcome.transfer(Step.SECOND_READ)
    assert first is not None and second is not None
    assert first.transferred == 60
    assert first.matched
    assert first.response is not None
    assert first.response.sha256 == hashlib.sha256(OBSERVED_ALGORITHM_RECORD).hexdigest()
    assert first.response.declared_length == 12
    assert first.response.text == "v7.5.7.0426"
    assert second.matched
    assert second.response is not None
    assert second.response.all_zero
    write = outcome.transfer(Step.PATH_WRITE)
    assert write is not None
    assert write.sent_sha256 == hashlib.sha256(PATH_PAYLOAD).hexdigest()
    assert write.response is None
    assert outcome.sequence_duration_s is not None
    assert outcome.total_duration_s is not None


def test_no_forbidden_ioctl_is_ever_issued(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    make_backend(healthy, fake).run(REAL_SERIAL)
    issued = set(fake.commands) | set(fake.driver_codes)
    assert issued.isdisjoint(set(FORBIDDEN_IOCTLS.values()))


def test_every_transfer_shares_the_one_descriptor(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    make_backend(healthy, fake).run(REAL_SERIAL)
    assert len(fake.opened) == 1
    assert fake.closed == [7]


# --- divergence ---------------------------------------------------------------


def test_a_divergent_first_read_is_reported_and_the_sequence_still_finishes(
    healthy: SysfsBuilder,
) -> None:
    """Content that is not what the SDK saw is data, not a transport failure."""
    odd = bytes.fromhex("ff000000") + b"surprise" + bytes(48)
    fake = FakeUsbfs(replies=[odd, OBSERVED_ZERO_RESPONSE])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == 6
    assert not outcome.succeeded
    assert fake.commands == FULL_COMMANDS
    assert outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    first = outcome.transfer(Step.RESULT_READ)
    assert first is not None and not first.matched


def test_a_divergent_second_read_is_reported(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[OBSERVED_ALGORITHM_RECORD, OBSERVED_ALGORITHM_RECORD])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    second = outcome.transfer(Step.SECOND_READ)
    assert second is not None and not second.matched
    assert outcome.cleanup.ok


def test_a_short_read_is_reported_as_a_short_response(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[OBSERVED_ALGORITHM_RECORD[:12], OBSERVED_ZERO_RESPONSE])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    assert outcome.exit_code == 7
    first = outcome.transfer(Step.RESULT_READ)
    assert first is not None
    assert first.transferred == 12
    assert first.response is not None and first.response.raw_length == 12
    assert fake.commands == FULL_COMMANDS


def test_a_short_path_write_is_reported_and_still_cleared(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={0: 12})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    write = outcome.transfer(Step.PATH_WRITE)
    assert write is not None and write.transferred == 12 and not write.matched
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok


def test_a_negative_return_is_a_transport_failure(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_returns={1: -1})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert steps_of(outcome) == [Step.PATH_WRITE, Step.RESULT_READ, Step.CLEAR_WRITE]
    assert outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok


# --- transport failure at every boundary --------------------------------------


def test_a_failed_detach_writes_nothing_and_needs_no_cleanup(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_DISCONNECT: OSError(errno.ENODATA, "No data available")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DETACH_FAILED
    assert outcome.exit_code == 11
    assert fake.controls == []
    assert not outcome.protocol_cleanup.attempted
    assert not outcome.cleanup.attempted
    assert USBDEVFS_CONNECT not in fake.driver_codes


def test_a_failed_claim_writes_nothing_and_still_reconnects(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_CLAIMINTERFACE: OSError(errno.EBUSY, "busy")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.CLAIM_FAILED
    assert outcome.exit_code == 12
    assert fake.controls == []
    assert not outcome.protocol_cleanup.attempted
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert USBDEVFS_RELEASEINTERFACE not in fake.commands
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        pytest.param(OSError(errno.ETIMEDOUT, "timed out"), OwnershipStatus.TIMEOUT, 4, id="to"),
        pytest.param(OSError(errno.EBUSY, "busy"), OwnershipStatus.IOCTL_ERROR, 5, id="busy"),
        pytest.param(OSError(errno.EPIPE, "stall"), OwnershipStatus.IOCTL_ERROR, 5, id="stall"),
        pytest.param(OSError(errno.ENODEV, "gone"), OwnershipStatus.DEVICE_ABSENT, 2, id="gone"),
    ],
)
def test_a_failed_path_write_still_clears_and_reconnects(
    healthy: SysfsBuilder, error: OSError, status: OwnershipStatus, code: int
) -> None:
    """The write may have reached the device even when the ioctl reported failure."""
    fake = FakeUsbfs(control_errors={0: error})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is status
    assert outcome.exit_code == code
    assert steps_of(outcome) == [Step.PATH_WRITE, Step.CLEAR_WRITE]
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


@pytest.mark.parametrize("index", [1, 2])
def test_a_failed_read_stops_the_sequence_but_still_clears(
    healthy: SysfsBuilder, index: int
) -> None:
    fake = FakeUsbfs(control_errors={index: OSError(errno.ETIMEDOUT, "timed out")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.TIMEOUT
    assert outcome.exit_code == 4
    assert steps_of(outcome)[-1] is Step.CLEAR_WRITE
    assert len(steps_of(outcome)) == index + 2
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok


def test_no_transfer_is_ever_retried(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={1: OSError(errno.ETIMEDOUT, "timed out")})
    make_backend(healthy, fake).run(REAL_SERIAL)
    setups = [setup for setup, _ in fake.controls]
    assert setups.count(bytes.fromhex("a181000200043c00b80b0000")) == 1
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


# --- the clearing write is cleanup, and cannot be reported as success ---------


def test_a_failed_clearing_write_is_never_reported_as_success(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={3: OSError(errno.EPIPE, "stall")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.OK
    assert not outcome.succeeded
    assert outcome.exit_code == PROTOCOL_CLEANUP_FAILED_EXIT == 14
    assert outcome.protocol_cleanup.attempted
    assert not outcome.protocol_cleanup.ok
    assert outcome.protocol_cleanup.guidance is not None
    assert INITED_PATH in outcome.protocol_cleanup.guidance
    # the driver is still given back, and that half is reported separately
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    clear = outcome.transfer(Step.CLEAR_WRITE)
    assert clear is not None and not clear.ok and clear.errno == errno.EPIPE


def test_a_failed_clearing_write_and_a_failed_reconnect_both_report(
    healthy: SysfsBuilder,
) -> None:
    """The two undos are independent; the driver failure is the graver one."""
    fake = FakeUsbfs(
        control_errors={3: OSError(errno.EPIPE, "stall")},
        errors={USBDEVFS_CONNECT: OSError(errno.EBUSY, "busy")},
    )
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert not outcome.succeeded
    assert outcome.exit_code == CLEANUP_FAILED_EXIT == 9
    assert not outcome.protocol_cleanup.ok
    assert not outcome.cleanup.ok
    assert outcome.protocol_cleanup.guidance is not None
    assert outcome.cleanup.guidance is not None
    assert "replug" in outcome.cleanup.guidance
    assert USBDEVFS_RELEASEINTERFACE in fake.commands


def test_a_failed_release_skips_neither_the_clear_nor_the_reconnect(
    healthy: SysfsBuilder,
) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_RELEASEINTERFACE: OSError(errno.EINVAL, "invalid")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert [p.phase for p in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]
    assert USBDEVFS_CONNECT in fake.driver_codes
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]


def test_a_clearing_write_that_raises_outright_does_not_cost_the_driver(
    healthy: SysfsBuilder,
) -> None:
    """Even a BaseException out of the undo must not skip the driver handback."""

    def explode(index: int) -> None:
        if index == 3:
            raise KeyboardInterrupt("during the clearing write")

    fake = FakeUsbfs(on_control=explode)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert not outcome.protocol_cleanup.ok
    assert "KeyboardInterrupt" in outcome.protocol_cleanup.note
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert outcome.exit_code == PROTOCOL_CLEANUP_FAILED_EXIT


# --- termination at every boundary --------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_signal_during_any_transfer_still_clears_and_reconnects(
    healthy: SysfsBuilder, index: int
) -> None:
    def interrupt(control_index: int) -> None:
        if control_index == index:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake = FakeUsbfs(on_control=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.exit_code == 13
    assert "SIGTERM" in outcome.detail
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    assert fake.written[-1] == ZERO_PAYLOAD
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(USBDEVFS_DISCONNECT, id="during_the_detach"),
        pytest.param(USBDEVFS_CLAIMINTERFACE, id="during_the_claim"),
    ],
)
def test_a_signal_before_the_path_write_leaves_nothing_to_clear(
    healthy: SysfsBuilder, code: int
) -> None:
    def interrupt(seen: int) -> None:
        if seen == code:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(on_ioctl=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
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
    """Before the detach the guard is not installed yet, and nothing is owed.

    The signal unwinds out of ``run`` rather than being converted, which is the
    accepted behaviour of the lifecycle: at that point no driver has been taken
    and nothing has been written, so there is nothing for a ``finally`` to undo.
    """

    def interrupt(seen: int) -> None:
        if seen == USBDEVFS_GETDRIVER:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(on_ioctl=interrupt)
    with pytest.raises(ownership_module.SignalReceived):
        make_backend(healthy, fake).run(REAL_SERIAL)
    assert fake.controls == []
    assert fake.driver_codes == []
    assert fake.closed == [7]


def test_an_interruption_after_the_path_write_returns_still_clears(
    healthy: SysfsBuilder,
) -> None:
    """The gap between the write's return and the first read is covered too."""
    seen: list[int] = []

    def interrupt(index: int) -> None:
        seen.append(index)
        if index == 1:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake = FakeUsbfs(on_control=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert outcome.protocol_cleanup.ok


def test_the_clearing_write_happens_at_most_once(healthy: SysfsBuilder) -> None:
    """Both ``finally`` blocks reach cleanup; only one of them may write."""

    def interrupt(index: int) -> None:
        if index == 0:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake = FakeUsbfs(on_control=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert fake.written.count(ZERO_PAYLOAD) == 1
    assert fake.driver_codes.count(USBDEVFS_CONNECT) == 1
    assert [p.phase for p in outcome.phases].count(Phase.PROTOCOL_CLEANUP) == 1


def test_an_unexpected_python_exception_still_clears_and_reconnects(
    healthy: SysfsBuilder,
) -> None:
    def explode(index: int) -> None:
        if index == 1:
            raise RuntimeError("something else entirely")

    fake = FakeUsbfs(on_control=explode)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert "RuntimeError" in outcome.detail
    assert outcome.protocol_cleanup.attempted and outcome.protocol_cleanup.ok
    assert outcome.cleanup.ok
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    assert fake.closed == [7]


def test_the_operation_refuses_to_clear_twice(healthy: SysfsBuilder) -> None:
    """The non-retry guard, exercised directly rather than through the lifecycle."""
    fake = FakeUsbfs()
    operation = init_module.InitedSequenceOperation(
        ioctl_fn=fake.ioctl, clock=fake.clock, usb_path="1-4.1"
    )
    operation.path_write_attempted = True
    first = operation.protocol_cleanup(7)
    assert first.ok
    second = operation.protocol_cleanup(7)
    assert not second.ok
    assert "never repeated" in second.note
    assert fake.written == [ZERO_PAYLOAD]


# --- serialisation ------------------------------------------------------------


def test_outcome_as_dict_is_json_shaped(healthy: SysfsBuilder) -> None:
    payload = make_backend(healthy, FakeUsbfs()).run(REAL_SERIAL).as_dict()
    assert payload["mode"] == "inited-path-sequence"
    assert payload["status"] == "ok"
    assert payload["succeeded"] is True
    assert payload["exit_code"] == 0
    assert payload["owned_interface"] == 0
    assert payload["entity_id"] == 4
    assert payload["path"] == INITED_PATH
    sequence = payload["sequence"]
    assert isinstance(sequence, list)
    assert [t["step"] for t in sequence] == [str(s) for s in Step]
    transfers = payload["transfers"]
    assert isinstance(transfers, list)
    assert [t["step"] for t in transfers] == [str(s) for s in Step]
    assert [t["direction"] for t in transfers] == ["out", "in", "in", "out"]
    phases = payload["phases"]
    assert isinstance(phases, list)
    assert [p["phase"] for p in phases] == [str(p) for p in FULL_ORDER]
    protocol = payload["protocol_cleanup"]
    assert isinstance(protocol, dict)
    assert protocol["attempted"] is True
    assert protocol["ok"] is True
    assert protocol["recovery_guidance"] is None
    cleanup = payload["cleanup"]
    assert isinstance(cleanup, dict)
    assert cleanup["ok"] is True


def test_a_protocol_undo_that_escapes_wholesale_still_reconnects(
    healthy: SysfsBuilder, monkeypatch
) -> None:
    """The handback is owed even if the undo step fails in a way it cannot report."""

    def escapes(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the undo step itself came apart")

    monkeypatch.setattr(ownership_module.InterfaceZeroLifecycle, "_protocol_cleanup", escapes)
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.cleanup.attempted and outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_RELEASEINTERFACE in fake.commands
    assert fake.closed == [7]
