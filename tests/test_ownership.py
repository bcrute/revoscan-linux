"""Tests for the one bounded interface-0 ownership lifecycle.

Nothing here touches hardware. Every syscall the backend makes is injected, so
these tests assert on the exact ioctl numbers, the exact struct bytes, the exact
order of the phases, and - most of the point - that the release and reconnect
happen after *every* way the middle of the lifecycle can fail.
"""

from __future__ import annotations

import ast
import contextlib
import ctypes
import errno
import signal
import struct
import tokenize
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import pytest

import scanner3d.control as control_module
import scanner3d.ownership as ownership_module
from scanner3d.control import FIRMWARE_ALGORITHM_VERSION_READ, USBDEVFS_CONTROL
from scanner3d.errors import InterfaceOwnershipNotAllowed
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    EXPECTED_INTERFACE_DRIVER,
    LIFECYCLE,
    OWNED_INTERFACE,
    OWNED_REQUEST,
    OWNERSHIP_EXIT_CODES,
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONNECT,
    USBDEVFS_DISCONNECT,
    USBDEVFS_GETDRIVER,
    USBDEVFS_IOCTL,
    USBDEVFS_RELEASEINTERFACE,
    OwnershipStatus,
    Phase,
    UsbfsOwnershipBackend,
    cleanup_on_termination,
    defer_termination,
    pack_driver_ioctl,
    pack_getdriver,
    pack_interface,
    refusal_reason,
    unpack_getdriver,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder

_IOC = control_module._ioc
_U = ord("U")

#: usbfs commands the ownership module must never be able to issue. Computed
#: here rather than imported, precisely so the module never names them.
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

PADDED_OK = control_module.EXPECTED_PAYLOAD_TEXT.encode("ascii").ljust(
    FIRMWARE_ALGORITHM_VERSION_READ.length, b"\x00"
)

#: The whole lifecycle, in the only order it may occur.
FULL_ORDER = [
    Phase.CONFIRM_DRIVER,  # from sysfs, in preflight
    Phase.CONFIRM_DRIVER,  # from USBDEVFS_GETDRIVER, on the open descriptor
    Phase.DISCONNECT,
    Phase.CLAIM,
    Phase.CONTROL_READ,
    Phase.RELEASE,
    Phase.RECONNECT,
]


class FakeUsbfs:
    """Records every open/ioctl/close and answers each usbfs command."""

    def __init__(
        self,
        *,
        driver: str = EXPECTED_INTERFACE_DRIVER,
        reply: bytes = PADDED_OK,
        transferred: int | None = None,
        open_error: OSError | None = None,
        errors: dict[int, OSError] | None = None,
        on_ioctl: object = None,
    ) -> None:
        self.driver = driver
        self.reply = reply
        self.transferred = len(reply) if transferred is None else transferred
        self.open_error = open_error
        # keyed by the ioctl command number, or by the driver ioctl_code for
        # the USBDEVFS_IOCTL envelope
        self.errors = errors or {}
        self.on_ioctl = on_ioctl
        self.opened: list[str] = []
        self.calls: list[tuple[int, bytes]] = []
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
        if code in self.errors:
            raise self.errors[code]
        if cmd == USBDEVFS_GETDRIVER:
            arg[:] = struct.pack(
                ownership_module._GETDRIVER_FORMAT, OWNED_INTERFACE, self.driver.encode("ascii")
            )
            return 0
        if cmd == USBDEVFS_CONTROL:
            *_, address = struct.unpack(control_module._CTRLTRANSFER_FORMAT, bytes(arg))
            ctypes.memmove(address, self.reply, len(self.reply))
            return self.transferred
        return 0

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


class ExplodingOpener:
    """An opener that fails the test if it is ever called."""

    def __call__(self, path: str) -> int:
        raise AssertionError(f"a device node was opened: {path}")


def make_backend(healthy: SysfsBuilder, fake: FakeUsbfs) -> UsbfsOwnershipBackend:
    return UsbfsOwnershipBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )


def phases_of(outcome: ownership_module.OwnershipOutcome) -> list[Phase]:
    return [p.phase for p in outcome.phases]


# --- the ABI ------------------------------------------------------------------


def test_ioctl_numbers_match_the_kernel_encoding() -> None:
    """Hand-computed from <linux/usbdevice_fs.h> on a 64-bit kernel."""
    assert USBDEVFS_GETDRIVER == 0x41045508
    assert USBDEVFS_CLAIMINTERFACE == 0x8004550F
    assert USBDEVFS_RELEASEINTERFACE == 0x80045510
    assert USBDEVFS_IOCTL == 0xC0105512
    assert USBDEVFS_DISCONNECT == 0x5516
    assert USBDEVFS_CONNECT == 0x5517
    assert USBDEVFS_CONTROL == 0xC0185500


def test_struct_sizes_match_the_kernel_definitions() -> None:
    assert ownership_module._USBDEVFS_IOCTL_SIZE == 16  # int, int, void *
    assert ownership_module._GETDRIVER_SIZE == 4 + 256  # unsigned int, char[256]


def test_claim_and_release_argument_is_the_bare_interface_number() -> None:
    assert bytes(pack_interface(OWNED_INTERFACE)) == struct.pack("@I", 0)


def test_driver_ioctl_struct_names_interface_0_and_a_null_payload() -> None:
    packed = bytes(pack_driver_ioctl(OWNED_INTERFACE, USBDEVFS_DISCONNECT))
    ifno, code, data = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, packed)
    assert (ifno, code, data) == (0, USBDEVFS_DISCONNECT, 0)

    packed = bytes(pack_driver_ioctl(OWNED_INTERFACE, USBDEVFS_CONNECT))
    ifno, code, data = struct.unpack(ownership_module._USBDEVFS_IOCTL_FORMAT, packed)
    assert (ifno, code, data) == (0, USBDEVFS_CONNECT, 0)


def test_getdriver_round_trips_a_driver_name() -> None:
    packed = pack_getdriver(OWNED_INTERFACE)
    assert len(packed) == ownership_module._GETDRIVER_SIZE
    filled = struct.pack(ownership_module._GETDRIVER_FORMAT, 0, b"uvcvideo")
    assert unpack_getdriver(filled) == "uvcvideo"


# --- the safety boundary ------------------------------------------------------


@pytest.mark.parametrize("packer", [pack_interface, pack_getdriver])
@pytest.mark.parametrize("number", [1, 2, 3, 4, -1, 255])
def test_no_interface_but_0_can_be_encoded(packer: object, number: int) -> None:
    with pytest.raises(InterfaceOwnershipNotAllowed):
        packer(number)  # type: ignore[operator]


@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_no_interface_but_0_can_be_detached_or_reconnected(number: int) -> None:
    with pytest.raises(InterfaceOwnershipNotAllowed):
        pack_driver_ioctl(number, USBDEVFS_DISCONNECT)
    with pytest.raises(InterfaceOwnershipNotAllowed):
        pack_driver_ioctl(number, USBDEVFS_CONNECT)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(_IOC(0, _U, 20, 0), id="usbdevfs_reset"),
        pytest.param(_IOC(1, _U, 5, 4), id="set_configuration"),
        pytest.param(_IOC(1, _U, 4, 8), id="set_interface"),
        pytest.param(0, id="zero"),
    ],
)
def test_only_disconnect_and_connect_can_ride_the_driver_envelope(code: int) -> None:
    with pytest.raises(InterfaceOwnershipNotAllowed):
        pack_driver_ioctl(OWNED_INTERFACE, code)


def _executable_source(module_path: Path) -> str:
    """The module's code with comments and string literals removed."""
    with module_path.open("rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))
    return " ".join(
        token.string for token in tokens if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def test_the_module_knows_exactly_six_usbfs_commands() -> None:
    names = {n for n in dir(ownership_module) if n.startswith("USBDEVFS_")}
    assert names == {
        "USBDEVFS_CONTROL",
        "USBDEVFS_GETDRIVER",
        "USBDEVFS_CLAIMINTERFACE",
        "USBDEVFS_RELEASEINTERFACE",
        "USBDEVFS_IOCTL",
        "USBDEVFS_DISCONNECT",
        "USBDEVFS_CONNECT",
    }
    code = _executable_source(Path(ownership_module.__file__))
    for name, number in FORBIDDEN_IOCTLS.items():
        assert name not in code
        assert str(number) not in code
        assert f"0x{number:x}" not in code.lower()


def test_the_control_module_is_still_the_narrow_one() -> None:
    """Adding ownership must not have widened the direct-read path."""
    assert {n for n in dir(control_module) if n.startswith("USBDEVFS_")} == {"USBDEVFS_CONTROL"}


def test_module_imports_nothing_that_could_stream_or_load_the_sdk() -> None:
    tree = ast.parse(Path(ownership_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert imported <= {
        "__future__",
        "contextlib",
        "ctypes",
        "enum",
        "errno",
        "fcntl",
        "os",
        "signal",
        "struct",
        "time",
        "collections.abc",
        "dataclasses",
        "pathlib",
        "types",
        "typing",
        "scanner3d.control",
        "scanner3d.discovery",
        "scanner3d.errors",
    }, sorted(imported)


def test_the_request_is_not_a_parameter_of_anything() -> None:
    """No caller can substitute a different setup packet."""
    assert OWNED_REQUEST is FIRMWARE_ALGORITHM_VERSION_READ
    for name in ("run", "preflight"):
        method = getattr(UsbfsOwnershipBackend, name)
        assert list(method.__code__.co_varnames[: method.__code__.co_argcount]) == [
            "self",
            "serial",
        ]


def test_the_owned_interface_is_the_one_the_request_addresses() -> None:
    assert OWNED_INTERFACE == 0
    assert OWNED_REQUEST.interface_number == OWNED_INTERFACE
    assert EXPECTED_INTERFACE_DRIVER == "uvcvideo"


def test_every_status_has_an_exit_code_and_none_of_them_is_the_cleanup_code() -> None:
    assert set(OWNERSHIP_EXIT_CODES) == set(OwnershipStatus)
    assert CLEANUP_FAILED_EXIT not in set(OWNERSHIP_EXIT_CODES.values())


def test_status_mappings_declare_no_duplicate_keys_in_source() -> None:
    tree = ast.parse(Path(ownership_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [ast.unparse(key) for key in node.keys if key is not None]
        assert len(keys) == len(set(keys)), f"duplicate dict key at line {node.lineno}: {keys}"


# --- preconditions ------------------------------------------------------------


def test_preflight_opens_nothing(healthy: SysfsBuilder) -> None:
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.preflight(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.NOT_ATTEMPTED
    assert outcome.exit_code == 0
    assert phases_of(outcome) == [Phase.CONFIRM_DRIVER]
    assert not outcome.cleanup.attempted
    assert outcome.cleanup.ok


def test_absent_device_is_refused_before_anything_opens(empty_sysfs: SysfsBuilder) -> None:
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=empty_sysfs.sysfs, dev_root=empty_sysfs.dev
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert outcome.exit_code == 2
    assert not outcome.cleanup.attempted


def test_a_different_serial_is_not_targeted(healthy: SysfsBuilder) -> None:
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=healthy.sysfs, dev_root=healthy.dev
    )
    outcome = backend.run("NOT-THIS-ONE")
    assert outcome.status is OwnershipStatus.DEVICE_ABSENT
    assert "NOT-THIS-ONE" in outcome.detail


@pytest.mark.parametrize(
    ("driver", "expected_fragment"),
    [
        pytest.param(None, "already unbound", id="already_unbound"),
        pytest.param("usbfs", "owned by 'usbfs'", id="another_driver"),
        pytest.param("snd-usb-audio", "owned by 'snd-usb-audio'", id="unknown_driver"),
    ],
)
def test_interface_0_must_be_held_by_uvcvideo(
    builder: SysfsBuilder, driver: str | None, expected_fragment: str
) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, interface_string=f"DepthCam{REAL_SERIAL}", driver=driver)
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert outcome.exit_code == 10
    assert expected_fragment in outcome.detail
    assert not outcome.cleanup.attempted


def test_a_device_that_is_not_the_expected_mini_is_refused(builder: SysfsBuilder) -> None:
    """Same vendor/product, wrong manufacturer string: still refused."""
    base = builder.add_usb_device(manufacturer="Somebody Else")
    builder.add_usbfs()
    builder.add_interface(base, 0, interface_string=f"DepthCam{REAL_SERIAL}")
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert "manufacturer" in outcome.detail


def test_missing_interface_0_is_refused(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 2, interface_string=f"RGBCam{REAL_SERIAL}")
    backend = UsbfsOwnershipBackend(
        opener=ExplodingOpener(), sysfs_root=builder.sysfs, dev_root=builder.dev
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert "no interface 0" in outcome.detail


def test_refusal_reason_accepts_only_the_expected_state(healthy: SysfsBuilder) -> None:
    from scanner3d.discovery import find_scanners

    device = find_scanners(sysfs_root=healthy.sysfs, dev_root=healthy.dev)[0]
    assert refusal_reason(device, REAL_SERIAL) is None
    assert refusal_reason(device, "SOME-OTHER-SERIAL") is not None


def test_a_driver_change_between_sysfs_and_the_open_descriptor_is_refused(
    healthy: SysfsBuilder,
) -> None:
    """sysfs said uvcvideo; the kernel, asked directly, says otherwise."""
    fake = FakeUsbfs(driver="usbfs")
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert "'usbfs'" in outcome.detail
    assert fake.driver_codes == []
    assert USBDEVFS_CLAIMINTERFACE not in fake.commands
    assert fake.closed == [7]


def test_reenumeration_between_resolve_and_detach_is_caught(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    backend = make_backend(healthy, fake)
    sysfs_device = healthy.devices / "1-4.1"
    real_open = fake.open

    def open_then_replug(path: str) -> int:
        fd = real_open(path)
        (sysfs_device / "serial").write_text("A-DIFFERENT-UNIT\n", encoding="utf-8")
        return fd

    from dataclasses import replace

    outcome = replace(backend, opener=open_then_replug).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DEVICE_REPLACED
    assert outcome.exit_code == 8
    assert fake.calls == []
    assert fake.closed == [7]


def test_permission_failure_on_open_is_distinct(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(open_error=PermissionError(errno.EACCES, "Permission denied"))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.PERMISSION_DENIED
    assert outcome.exit_code == 3
    assert fake.calls == []
    assert not outcome.cleanup.attempted


# --- the happy path -----------------------------------------------------------


def test_successful_lifecycle_runs_exactly_six_ioctls_in_order(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is OwnershipStatus.OK
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert phases_of(outcome) == FULL_ORDER
    assert fake.commands == [
        USBDEVFS_GETDRIVER,
        USBDEVFS_IOCTL,  # disconnect
        USBDEVFS_CLAIMINTERFACE,
        USBDEVFS_CONTROL,
        USBDEVFS_RELEASEINTERFACE,
        USBDEVFS_IOCTL,  # connect
    ]
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert len(fake.opened) == 1
    assert fake.closed == [7]


def test_the_read_uses_the_unchanged_whitelisted_setup_packet(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    make_backend(healthy, fake).run(REAL_SERIAL)
    control = next(arg for cmd, arg in fake.calls if cmd == USBDEVFS_CONTROL)
    assert control[:12] == bytes.fromhex("a181000600033c00b80b0000")


def test_successful_lifecycle_parses_the_known_payload(healthy: SysfsBuilder) -> None:
    outcome = make_backend(healthy, FakeUsbfs()).run(REAL_SERIAL)
    assert outcome.payload is not None
    assert outcome.payload.matches_expected
    assert outcome.payload.firmware_version == control_module.EXPECTED_FIRMWARE_VERSION
    assert outcome.payload.algorithm_version == control_module.EXPECTED_ALGORITHM_VERSION
    assert outcome.transfer_duration_s is not None
    assert outcome.total_duration_s is not None


def test_no_forbidden_ioctl_is_ever_issued(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    make_backend(healthy, fake).run(REAL_SERIAL)
    assert set(fake.commands).isdisjoint(FORBIDDEN_IOCTLS.values())
    assert set(fake.driver_codes).isdisjoint(FORBIDDEN_IOCTLS.values())


def test_the_claim_and_the_read_share_one_descriptor(healthy: SysfsBuilder) -> None:
    """A claim only counts on the file descriptor that made it."""
    fake = FakeUsbfs()
    seen: list[int] = []
    original = fake.ioctl

    def recording(fd: int, cmd: int, arg: bytearray, mutate: bool) -> int:
        seen.append(fd)
        return original(fd, cmd, arg, mutate)

    from dataclasses import replace

    replace(make_backend(healthy, fake), ioctl_fn=recording).run(REAL_SERIAL)
    assert len(set(seen)) == 1
    assert len(fake.opened) == 1


# --- failure inside the detached window ---------------------------------------


def test_a_failed_detach_needs_no_cleanup(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_DISCONNECT: OSError(errno.ENODATA, "No data available")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DETACH_FAILED
    assert outcome.exit_code == 11
    assert not outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert phases_of(outcome) == [Phase.CONFIRM_DRIVER, Phase.CONFIRM_DRIVER, Phase.DISCONNECT]
    assert USBDEVFS_CLAIMINTERFACE not in fake.commands
    assert USBDEVFS_CONNECT not in fake.driver_codes


def test_a_failed_claim_still_reconnects_and_does_not_release(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_CLAIMINTERFACE: OSError(errno.EBUSY, "busy")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.CLAIM_FAILED
    assert outcome.exit_code == 12
    assert outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert phases_of(outcome) == [
        Phase.CONFIRM_DRIVER,
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.CLAIM,
        Phase.RECONNECT,
    ]
    assert USBDEVFS_RELEASEINTERFACE not in fake.commands
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_CONTROL not in fake.commands


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        pytest.param(
            OSError(errno.ETIMEDOUT, "timed out"), OwnershipStatus.TIMEOUT, 4, id="timeout"
        ),
        pytest.param(OSError(errno.EBUSY, "busy"), OwnershipStatus.IOCTL_ERROR, 5, id="busy"),
        pytest.param(
            OSError(errno.EPIPE, "broken pipe"), OwnershipStatus.IOCTL_ERROR, 5, id="stall"
        ),
        pytest.param(
            OSError(errno.ENODEV, "no device"), OwnershipStatus.DEVICE_ABSENT, 2, id="unplugged"
        ),
    ],
)
def test_a_failed_transfer_still_releases_and_reconnects(
    healthy: SysfsBuilder, error: OSError, status: OwnershipStatus, code: int
) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_CONTROL: error})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is status
    assert outcome.exit_code == code
    assert outcome.cleanup.ok
    assert phases_of(outcome) == FULL_ORDER
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


def test_a_short_response_still_releases_and_reconnects(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(reply=b"v2.8.7" + bytes(6), transferred=12)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.SHORT_RESPONSE
    assert outcome.exit_code == 7
    assert outcome.payload is not None
    assert outcome.payload.raw_length == 12
    assert phases_of(outcome) == FULL_ORDER
    assert outcome.cleanup.ok


def test_an_unexpected_payload_still_releases_and_reconnects(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(reply=bytes(60))
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert outcome.exit_code == 6
    assert not outcome.succeeded
    assert phases_of(outcome) == FULL_ORDER
    assert outcome.cleanup.ok


def test_an_unexpected_python_exception_still_reconnects(healthy: SysfsBuilder) -> None:
    """Not an OSError: a bug in our own code must not strand the driver."""
    fake = FakeUsbfs()

    def explode(code: int) -> None:
        if code == USBDEVFS_CONTROL:
            raise RuntimeError("boom")

    fake.on_ioctl = explode
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert "RuntimeError" in outcome.detail
    assert outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


def test_a_signal_inside_the_detached_window_still_reconnects(healthy: SysfsBuilder) -> None:
    """What a Ctrl-C or an external SIGTERM does mid-transfer."""
    fake = FakeUsbfs()

    def interrupt(code: int) -> None:
        if code == USBDEVFS_CONTROL:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake.on_ioctl = interrupt
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.exit_code == 13
    assert "SIGTERM" in outcome.detail
    assert outcome.cleanup.ok
    # the claim was held, so it is released before the driver is given back
    assert phases_of(outcome) == [
        Phase.CONFIRM_DRIVER,
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.CLAIM,
        Phase.RELEASE,
        Phase.RECONNECT,
    ]
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


# --- the boundary around the disconnect ioctl ---------------------------------
#
# From the attempt to disconnect onwards, nothing may escape without one
# reconnect attempt: a handler runs between bytecodes, so an exception
# surfacing from that ioctl leaves it unknown whether the kernel completed it.


@pytest.mark.parametrize(
    ("raised", "status", "code", "marker"),
    [
        pytest.param(
            ownership_module.SignalReceived(signal.SIGINT),
            OwnershipStatus.INTERRUPTED,
            13,
            "SIGINT",
            id="signal",
        ),
        pytest.param(
            RuntimeError("boom"), OwnershipStatus.IOCTL_ERROR, 5, "RuntimeError", id="exception"
        ),
    ],
)
def test_an_interruption_of_the_disconnect_ioctl_still_reconnects(
    healthy: SysfsBuilder,
    raised: BaseException,
    status: OwnershipStatus,
    code: int,
    marker: str,
) -> None:
    fake = FakeUsbfs()

    def interrupt(ioctl_code: int) -> None:
        if ioctl_code == USBDEVFS_DISCONNECT:
            raise raised

    fake.on_ioctl = interrupt
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is status
    assert marker in outcome.detail
    assert outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert outcome.exit_code == code
    # the detach never returned, so no claim is assumed and none is released
    assert phases_of(outcome) == [
        Phase.CONFIRM_DRIVER,
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.RECONNECT,
    ]
    ambiguous = outcome.phase(Phase.DISCONNECT)
    assert ambiguous is not None
    assert not ambiguous.ok
    assert "may or may not have completed it" in ambiguous.detail
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_CLAIMINTERFACE not in fake.commands
    assert USBDEVFS_RELEASEINTERFACE not in fake.commands
    assert USBDEVFS_CONTROL not in fake.commands
    assert fake.closed == [7]


def test_an_interruption_after_the_disconnect_returns_still_reconnects(
    healthy: SysfsBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap between a successful disconnect and the claim is covered too."""
    fake = FakeUsbfs()

    def never_gets_there(*args: object, **kwargs: object) -> ownership_module._Primary:
        raise ownership_module.SignalReceived(signal.SIGTERM)

    monkeypatch.setattr(UsbfsOwnershipBackend, "_claim_and_read", never_gets_there)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.exit_code == 13
    assert outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_CLAIMINTERFACE not in fake.commands


def test_an_interrupted_claim_is_released_before_the_reconnect(healthy: SysfsBuilder) -> None:
    """The claim is ambiguous the same way; usbfs refuses CONNECT while it is held."""
    fake = FakeUsbfs()

    def interrupt(code: int) -> None:
        if code == USBDEVFS_CLAIMINTERFACE:
            raise ownership_module.SignalReceived(signal.SIGINT)

    fake.on_ioctl = interrupt
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.cleanup.ok
    assert [p.phase for p in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert USBDEVFS_CONTROL not in fake.commands


def test_a_baseexception_the_guard_cannot_convert_still_reconnects(
    healthy: SysfsBuilder,
) -> None:
    """Off the main thread no handler can be installed; the ``finally`` is all there is."""
    fake = FakeUsbfs()

    def interrupt(code: int) -> None:
        if code == USBDEVFS_DISCONNECT:
            raise KeyboardInterrupt

    fake.on_ioctl = interrupt
    backend = replace(make_backend(healthy, fake), signal_guard=contextlib.nullcontext)
    with pytest.raises(KeyboardInterrupt):
        backend.run(REAL_SERIAL)
    # the exception is not swallowed, but the driver is still handed back first
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert fake.closed == [7]


@pytest.mark.parametrize(
    "failing",
    [
        pytest.param(USBDEVFS_DISCONNECT, id="during_disconnect"),
        pytest.param(USBDEVFS_CLAIMINTERFACE, id="during_claim"),
        pytest.param(USBDEVFS_CONTROL, id="during_transfer"),
    ],
)
def test_an_interruption_cleans_up_exactly_once(healthy: SysfsBuilder, failing: int) -> None:
    """Two nested ``finally`` blocks guard the window; neither may retry."""
    fake = FakeUsbfs()

    def interrupt(code: int) -> None:
        if code == failing:
            raise ownership_module.SignalReceived(signal.SIGTERM)

    fake.on_ioctl = interrupt
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert fake.driver_codes.count(USBDEVFS_CONNECT) == 1
    assert fake.commands.count(USBDEVFS_RELEASEINTERFACE) <= 1
    assert [p.phase for p in outcome.phases].count(Phase.RECONNECT) == 1


# --- the boundary at cleanup entry --------------------------------------------
#
# Cleanup is the last thing between a detached interface 0 and a process that
# exits without giving it back, so the first termination is injected at the
# entry to cleanup itself - not inside a release or reconnect that is already
# guarded. Either it beats the shield into place, in which case nothing has
# happened yet and the outer ``finally`` must still make the one attempt, or it
# arrives after, in which case it must not abort the ioctls.


class ShieldTimeline:
    """A stand-in for ``defer_termination`` that fires a signal at a chosen moment.

    ``beats_the_shield`` raises on the first entry, exactly as a real handler
    would if SIGINT landed while the takeover was still being installed.
    ``arrives_inside`` is handed to the caller as an absorbed signal instead.
    """

    def __init__(self, *, beats_the_shield: int | None = None, arrives_inside: int | None = None):
        self.beats_the_shield = beats_the_shield
        self.arrives_inside = arrives_inside
        self.entries = 0

    def __call__(self) -> AbstractContextManager[list[int]]:
        return self._window()

    @contextlib.contextmanager
    def _window(self) -> Iterator[list[int]]:
        self.entries += 1
        if self.beats_the_shield is not None and self.entries == 1:
            raise ownership_module.SignalReceived(self.beats_the_shield)
        yield [] if self.arrives_inside is None else [self.arrives_inside]


def test_a_termination_at_cleanup_entry_still_cleans_up_once(healthy: SysfsBuilder) -> None:
    """The gap the pre-marking version left: nothing was done, so it must be redone."""
    fake = FakeUsbfs()
    shield = ShieldTimeline(beats_the_shield=signal.SIGINT)
    backend = replace(make_backend(healthy, fake), termination_shield=shield)

    outcome = backend.run(REAL_SERIAL)

    # the read did happen; the signal landed on the way into cleanup
    assert USBDEVFS_CONTROL in fake.commands
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert "SIGINT" in outcome.detail
    # entry 1 (the inner ``finally``) was beaten by the signal and issued
    # nothing; entry 2 (the outer ``finally``) did the work exactly once
    assert shield.entries == 2
    assert outcome.cleanup.attempted
    assert outcome.cleanup.ok
    assert [p.phase for p in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]
    assert fake.commands.count(USBDEVFS_RELEASEINTERFACE) == 1
    assert fake.driver_codes.count(USBDEVFS_CONNECT) == 1
    assert phases_of(outcome).count(Phase.RECONNECT) == 1
    assert fake.closed == [7]


def test_a_termination_after_cleanup_starts_is_absorbed_and_reported(
    healthy: SysfsBuilder,
) -> None:
    """Release and reconnect finish; the signal shows up in the note, not the ioctls."""
    fake = FakeUsbfs()
    shield = ShieldTimeline(arrives_inside=signal.SIGTERM)
    backend = replace(make_backend(healthy, fake), termination_shield=shield)

    outcome = backend.run(REAL_SERIAL)

    # entry 1 did the work uninterrupted; entry 2 is the outer ``finally``
    # finding it already done and touching nothing
    assert shield.entries == 2
    assert outcome.status is OwnershipStatus.OK  # the primary result is preserved
    assert outcome.succeeded
    assert outcome.exit_code == 0
    assert outcome.cleanup.ok
    assert "SIGTERM" in outcome.cleanup.note
    assert "ignored" in outcome.cleanup.note
    assert phases_of(outcome) == FULL_ORDER
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


def test_cleanup_is_only_marked_done_once_it_has_been_attempted(
    healthy: SysfsBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``_cleanup`` that escapes wholesale must not leave a clean-looking record."""
    fake = FakeUsbfs()
    attempts: list[bool] = []

    def escapes(self: object, fd: int, snapshot: object, *, claimed: bool) -> None:
        attempts.append(claimed)
        raise RuntimeError("cleanup bookkeeping blew up")

    monkeypatch.setattr(UsbfsOwnershipBackend, "_cleanup", escapes)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert attempts == [True]  # attempted once, and the mark prevents a retry
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert "RuntimeError" in outcome.detail
    assert outcome.cleanup.attempted
    assert not outcome.cleanup.ok
    assert "did not finish" in outcome.cleanup.note
    assert outcome.cleanup.guidance is not None
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert fake.closed == [7]


# --- cleanup failure ----------------------------------------------------------


def test_a_failed_reconnect_cannot_be_reported_as_success(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_CONNECT: OSError(errno.EBUSY, "busy")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)

    assert outcome.status is OwnershipStatus.OK  # the read itself did work
    assert outcome.payload is not None
    assert outcome.payload.matches_expected
    assert not outcome.succeeded
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert outcome.exit_code != 0
    assert outcome.cleanup.guidance is not None
    assert "replug" in outcome.cleanup.guidance
    assert "/sys/bus/usb/drivers/uvcvideo/bind" in outcome.cleanup.guidance
    assert outcome.as_dict()["succeeded"] is False


def test_a_failed_release_does_not_skip_the_reconnect(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_RELEASEINTERFACE: OSError(errno.ENOENT, "no such")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert [p.phase for p in outcome.cleanup.phases] == [Phase.RELEASE, Phase.RECONNECT]
    assert USBDEVFS_CONNECT in fake.driver_codes


def test_a_cleanup_that_raises_is_recorded_not_propagated(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()

    def explode(code: int) -> None:
        if code == USBDEVFS_CONNECT:
            raise KeyboardInterrupt

    fake.on_ioctl = explode
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert not outcome.cleanup.ok
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    reconnect = outcome.phase(Phase.RECONNECT)
    assert reconnect is not None
    assert "KeyboardInterrupt" in reconnect.detail
    assert fake.closed == [7]


def test_cleanup_failure_outranks_every_primary_status(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        reply=bytes(60),
        errors={USBDEVFS_CONNECT: OSError(errno.EBUSY, "busy")},
    )
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.UNEXPECTED_CONTENT
    assert OWNERSHIP_EXIT_CODES[outcome.status] == 6
    assert outcome.exit_code == CLEANUP_FAILED_EXIT


# --- no retries ---------------------------------------------------------------


@pytest.mark.parametrize(
    "failing",
    [USBDEVFS_GETDRIVER, USBDEVFS_DISCONNECT, USBDEVFS_CLAIMINTERFACE, USBDEVFS_CONTROL],
)
def test_no_phase_is_ever_retried(healthy: SysfsBuilder, failing: int) -> None:
    fake = FakeUsbfs(errors={failing: OSError(errno.EBUSY, "busy")})
    make_backend(healthy, fake).run(REAL_SERIAL)
    attempts = [c for c in fake.commands if c == failing] + [
        c for c in fake.driver_codes if c == failing
    ]
    assert len(attempts) == 1


def test_the_lifecycle_constant_matches_what_a_run_performs(healthy: SysfsBuilder) -> None:
    outcome = make_backend(healthy, FakeUsbfs()).run(REAL_SERIAL)
    assert list(dict.fromkeys(phases_of(outcome))) == list(LIFECYCLE)


# --- signal guard -------------------------------------------------------------


def test_the_signal_guard_restores_the_previous_handlers() -> None:
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with cleanup_on_termination():
        during = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    after = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert during != before
    assert after == before


def test_the_signal_guard_raises_once_and_then_stops_raising() -> None:
    """A second Ctrl-C must not abort a cleanup that is already running."""
    with pytest.raises(ownership_module.SignalReceived) as first, cleanup_on_termination():
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        try:
            handler(signal.SIGINT, None)
        finally:
            # a second delivery, while "cleanup" runs, is dropped
            assert handler(signal.SIGTERM, None) is None
    assert first.value.signum == signal.SIGINT


def test_the_cleanup_shield_absorbs_terminations_and_restores_the_handlers() -> None:
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with defer_termination() as received:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handler = signal.getsignal(signum)
            assert callable(handler)
            assert handler(signum, None) is None  # absorbed, not raised
        assert received == [signal.SIGINT, signal.SIGTERM]
    assert {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)} == before


def test_the_cleanup_shield_outranks_the_raising_guard() -> None:
    """Inside cleanup the guard must not be able to turn a signal into an unwind."""
    with cleanup_on_termination():
        with defer_termination() as received:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            assert handler(signal.SIGINT, None) is None
        assert received == [signal.SIGINT]
        # and the guard is back in charge once cleanup is over
        after = signal.getsignal(signal.SIGTERM)
        assert callable(after)
        with pytest.raises(ownership_module.SignalReceived):
            after(signal.SIGTERM, None)


# --- serialization ------------------------------------------------------------


def test_outcome_as_dict_is_json_shaped(healthy: SysfsBuilder) -> None:
    payload = make_backend(healthy, FakeUsbfs()).run(REAL_SERIAL).as_dict()
    assert payload["mode"] == "temporary-interface-ownership"
    assert payload["status"] == "ok"
    assert payload["succeeded"] is True
    assert payload["owned_interface"] == 0
    assert payload["expected_interface_driver"] == "uvcvideo"
    request = payload["request"]
    assert isinstance(request, dict)
    assert request["wIndex"] == "0x0300"
    phases = payload["phases"]
    assert isinstance(phases, list)
    assert [p["phase"] for p in phases] == [str(p) for p in FULL_ORDER]
    cleanup = payload["cleanup"]
    assert isinstance(cleanup, dict)
    assert cleanup["ok"] is True
    assert cleanup["recovery_guidance"] is None
