"""Offline proof for the immutable RGB exposure GET_INFO boundary."""

from __future__ import annotations

import ast
import ctypes
import dataclasses
import errno
import hashlib
import json
from pathlib import Path

import pytest

import scanner3d.cli as cli
import scanner3d.rgb_exposure_info as info
from scanner3d.control import pack_ctrltransfer
from scanner3d.errors import RgbExposureInfoNotAllowed
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    USBDEVFS_CLAIMINTERFACE,
    USBDEVFS_CONNECT,
    USBDEVFS_DISCONNECT,
    USBDEVFS_RELEASEINTERFACE,
    OwnershipStatus,
    Phase,
    SignalReceived,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.test_depth_scale import FakeUsbfs
from tests.test_init_sequence import _executable_source

EXPECTED_SETUP = bytes.fromhex("a186001200040100e8030000")


def make_backend(builder: SysfsBuilder, fake: FakeUsbfs) -> info.UsbfsRgbExposureInfoBackend:
    return info.UsbfsRgbExposureInfoBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def test_fixed_get_info_setup_and_whitelist_are_immutable() -> None:
    request = info.READ_TRANSFER.request
    buffer = ctypes.create_string_buffer(1)
    assert bytes(pack_ctrltransfer(request, ctypes.addressof(buffer)))[:12] == EXPECTED_SETUP
    assert request.wire_tuple == (0xA1, 0x86, 0x1200, 0x0400, 1, 1000)
    assert info.MAX_CONTROL_TRANSFERS == 1
    assert info.WHITELIST == (info.READ_TRANSFER,)
    assert info.READ_TRANSFER.request.request_type == 0xA1
    assert info.READ_TRANSFER.request.request == 0x86


def test_setup_drift_and_executable_surface_are_rejected() -> None:
    request = info.READ_TRANSFER.request
    for altered in (
        dataclasses.replace(request, request_type=0x21),
        dataclasses.replace(request, request=0x81),
        dataclasses.replace(request, value=0x1300),
        dataclasses.replace(request, index=0x0401),
        dataclasses.replace(request, index=0x0500),
        dataclasses.replace(request, length=3),
        dataclasses.replace(request, timeout_ms=999),
    ):
        with pytest.raises(RgbExposureInfoNotAllowed):
            info.ensure_transfer_allowed(info.InfoTransfer(info.Step.INFO_READ, altered, "drift"))
    with pytest.raises(RgbExposureInfoNotAllowed):
        info.ensure_transfer_allowed(
            info.InfoTransfer(info.Step.INFO_READ, request, "second transfer")
        )

    assert {name for name in dir(info) if name.startswith("USBDEVFS_")} == {"USBDEVFS_CONTROL"}
    tree = ast.parse(Path(info.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    imported_names: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
            imported_names.setdefault(node.module, set()).update(alias.name for alias in node.names)
    assert imported_modules <= {
        "__future__",
        "ctypes",
        "enum",
        "hashlib",
        "signal",
        "dataclasses",
        "types",
        "typing",
        "scanner3d.control",
        "scanner3d.errors",
        "scanner3d.ownership",
    }
    assert imported_names["scanner3d.control"] == {
        "USBDEVFS_CONTROL",
        "Clock",
        "ControlRequest",
        "IoctlFn",
        "TargetSnapshot",
        "pack_ctrltransfer",
    }
    assert imported_names["scanner3d.errors"] == {"RgbExposureInfoNotAllowed"}
    assert imported_names["scanner3d.ownership"] == {
        "_ERRNO_STATUS",
        "CLEANUP_FAILED_EXIT",
        "OWNED_INTERFACE",
        "OWNERSHIP_EXIT_CODES",
        "CleanupRecord",
        "InterfaceZeroLifecycle",
        "OwnedOperation",
        "OwnershipStatus",
        "Phase",
        "PhaseResult",
        "Preflight",
        "ProtocolCleanupRecord",
        "SignalReceived",
        "_errno_of",
        "_Primary",
    }
    forbidden_surface = (
        "SET_CUR",
        "GET_CUR",
        "GET_LEN",
        "GET_MIN",
        "GET_MAX",
        "GET_RES",
        "GET_DEF",
        "V4L2",
        "STREAM",
        "HID",
        "SDK",
        "illumination",
        "blue",
        "brightness",
        "RESET",
        "FIRMWARE",
        "arbitrary",
        "vendor",
    )
    source = _executable_source(Path(info.__file__))
    assert "0x86" in source
    assert "0x85" not in source
    source_casefold = source.casefold()
    assert all(token.casefold() not in source_casefold for token in forbidden_surface)
    assert "0x81" not in source
    assert "0x21" not in source
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("raw", "value"),
    [(b"\x00", 0), (b"\x01", 1), (b"\x10", 0x10), (b"\xff", 0xFF)],
)
def test_decode_exact_one_byte_as_raw_info(raw: bytes, value: int) -> None:
    reading = info.decode_reply(raw)
    assert reading is not None
    assert reading.raw_info == value
    assert info.decode_reply(b"") is None
    assert info.decode_reply(raw + b"\x00") is None


def test_every_defined_bit_and_reserved_bits_are_reported_syntactically() -> None:
    reading = info.decode_reply(b"\x1f")
    assert reading is not None
    assert reading.bit_view == {
        "supports_get": True,
        "supports_set": True,
        "disabled": True,
        "autoupdate": True,
        "asynchronous": True,
        "reserved_high_bits": 0,
        "interpretation": "syntactic Linux-UAPI capability-bit view only",
    }
    reserved = info.decode_reply(b"\xe0")
    assert reserved is not None
    assert reserved.bit_view["reserved_high_bits"] == 0xE0


@pytest.mark.parametrize(
    ("mask", "name"),
    [
        (info.INFO_GET, "supports_get"),
        (info.INFO_SET, "supports_set"),
        (info.INFO_DISABLED, "disabled"),
        (info.INFO_AUTOUPDATE, "autoupdate"),
        (info.INFO_ASYNCHRONOUS, "asynchronous"),
    ],
)
def test_each_get_info_bit_has_its_own_one_hot_view(mask: int, name: str) -> None:
    reading = info.decode_reply(bytes([mask]))
    assert reading is not None
    assert reading.bit_view[name] is True
    for other in {"supports_get", "supports_set", "disabled", "autoupdate", "asynchronous"} - {
        name
    }:
        assert reading.bit_view[other] is False


def test_dry_run_is_inert_and_does_not_construct_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry run constructed a live backend")

    monkeypatch.setattr(cli, "rgb_exposure_info_backend_factory", fail)
    assert cli.main(["rgb-exposure-info", "--serial", REAL_SERIAL]) == 0


def test_wrong_serial_cli_is_rejected_before_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("wrong serial constructed a live backend")

    monkeypatch.setattr(cli, "rgb_exposure_info_backend_factory", fail)
    assert (
        cli.main(
            [
                "rgb-exposure-info",
                "--execute-rgb-exposure-info-read",
                "--serial",
                "not-authorized",
            ]
        )
        == 10
    )


def test_wrong_serial_opens_nothing(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run("not-authorized")
    assert outcome.status is OwnershipStatus.REFUSED
    assert fake.opened == []
    assert fake.controls == []
    assert outcome.transfers == ()


def test_missing_serial_requires_exact_opt_in() -> None:
    assert cli.main(["rgb-exposure-info", "--execute-rgb-exposure-info-read"]) == 10


def test_truncated_opt_in_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["rgb-exposure-info", "--execute-rgb-exposure-info"])


def test_refused_preflight_does_not_open_or_transfer(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, driver=None)
    fake = FakeUsbfs()
    outcome = make_backend(builder, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.REFUSED
    assert fake.opened == []
    assert fake.controls == []


def test_human_and_help_rendering(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["rgb-exposure-info"]) == 0
    human = capsys.readouterr().out
    assert "bRequest=0x86" in human
    assert "GET_INFO only" in human
    assert "Status: not_attempted" in human
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["rgb-exposure-info", "--help"])
    assert "--execute-rgb-exposure-info-read" in capsys.readouterr().out


def test_exact_zero_reply_is_complete_raw_zero_not_semantic(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[b"\x00"])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.succeeded
    assert outcome.reading is not None
    assert outcome.reading.raw_info == 0
    assert outcome.reply.raw_info == 0
    assert outcome.reply.raw_hex == "00"
    assert outcome.reply.sha256 == hashlib.sha256(b"\x00").hexdigest()
    assert fake.setups == [EXPECTED_SETUP]
    assert len(fake.controls) == 1
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (b"", info.InfoOutcome.SHORT_RESPONSE),
        (b"\x01\x02\x03", info.InfoOutcome.OVERLONG_RESPONSE),
    ],
)
def test_short_and_overlong_are_typed_without_decode_or_retry(
    healthy: SysfsBuilder, reply: bytes, expected: info.InfoOutcome
) -> None:
    fake = FakeUsbfs(replies=[reply])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.info_outcome is expected
    assert outcome.reading is None
    assert not outcome.succeeded
    assert outcome.reply.actual_length == len(reply)
    assert len(fake.controls) == 1
    assert outcome.exit_code != 0


def test_errno_and_cleanup_failure_are_typed(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={0: OSError(errno.EIO, "read")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert outcome.info_outcome is info.InfoOutcome.TRANSFER_ERROR
    assert outcome.errno == errno.EIO
    cleanup = FakeUsbfs(
        replies=[b"\x01"], errors={USBDEVFS_CONNECT: OSError(errno.EIO, "reconnect")}
    )
    failed = make_backend(healthy, cleanup).run(REAL_SERIAL)
    assert failed.exit_code == CLEANUP_FAILED_EXIT
    assert not failed.succeeded
    assert cleanup.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


@pytest.mark.parametrize("code", [USBDEVFS_DISCONNECT, USBDEVFS_CLAIMINTERFACE])
def test_ownership_failures_do_not_transfer(healthy: SysfsBuilder, code: int) -> None:
    fake = FakeUsbfs(errors={code: OSError(errno.EBUSY, "ownership")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.transfers == ()
    assert fake.controls == []


def test_release_failure_still_reconnects(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(
        replies=[b"\x02"], errors={USBDEVFS_RELEASEINTERFACE: OSError(errno.EIO, "release")}
    )
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert len(fake.controls) == 1


class _CloseFailureFake(FakeUsbfs):
    def close(self, fd: int) -> None:
        self.closed.append(fd)
        raise OSError(errno.EIO, "close failed")


def test_close_failure_preserves_exact_reply_and_blocks_success(healthy: SysfsBuilder) -> None:
    fake = _CloseFailureFake(replies=[b"\x1f"])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.info_outcome is info.InfoOutcome.COMPLETE
    assert outcome.reply.raw_hex == "1f"
    assert outcome.cleanup.ok is False
    assert outcome.cleanup.phases[-1].phase is Phase.CLOSE
    assert not outcome.succeeded
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert len(fake.controls) == 1


def test_signal_and_exception_preserve_one_attempt_and_cleanup(healthy: SysfsBuilder) -> None:
    def signal_read(index: int) -> None:
        raise SignalReceived(15)

    fake = FakeUsbfs(on_control=signal_read)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.info_outcome is info.InfoOutcome.INTERRUPTED
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert len(outcome.transfers) == 1
    assert outcome.transfers[0].exception_type == "SignalReceived"
    assert outcome.cleanup.ok
    assert [p.phase for p in outcome.phases].count(Phase.CONTROL_READ) == 1

    def explode(index: int) -> None:
        raise RuntimeError("boom")

    failed = make_backend(healthy, FakeUsbfs(on_control=explode)).run(REAL_SERIAL)
    assert failed.info_outcome is info.InfoOutcome.EXCEPTION
    assert failed.cleanup.ok


@pytest.mark.parametrize("signum", [2, 15])
def test_both_termination_signals_are_typed_and_cleanup(healthy: SysfsBuilder, signum: int) -> None:
    def signal_read(index: int) -> None:
        raise SignalReceived(signum)

    outcome = make_backend(healthy, FakeUsbfs(on_control=signal_read)).run(REAL_SERIAL)
    assert outcome.info_outcome is info.InfoOutcome.INTERRUPTED
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.transfers[0].exception_type == "SignalReceived"
    assert outcome.cleanup.ok


def test_keyboard_interrupt_and_non_exception_base_exception_are_typed(
    healthy: SysfsBuilder,
) -> None:
    outcome = make_backend(
        healthy, FakeUsbfs(on_control=lambda _index: (_ for _ in ()).throw(KeyboardInterrupt()))
    ).run(REAL_SERIAL)
    assert outcome.info_outcome is info.InfoOutcome.INTERRUPTED
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.cleanup.ok

    class CustomBase(BaseException):
        pass

    failed = make_backend(
        healthy, FakeUsbfs(on_control=lambda _index: (_ for _ in ()).throw(CustomBase("boom")))
    ).run(REAL_SERIAL)
    assert failed.info_outcome is info.InfoOutcome.EXCEPTION
    assert failed.status is OwnershipStatus.IOCTL_ERROR
    assert failed.transfers[0].exception_type == "CustomBase"
    assert failed.cleanup.ok


def test_identity_drift_refuses_before_transfer(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()

    def mutate_after_open(path: str) -> int:
        fd = fake.open(path)
        (healthy.devices / "1-4.1" / "serial").write_text("replacement\n", encoding="utf-8")
        return fd

    backend = info.UsbfsRgbExposureInfoBackend(
        opener=mutate_after_open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )
    outcome = backend.run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DEVICE_REPLACED
    assert outcome.transfers == ()
    assert fake.controls == []


def test_cli_json_stdout_and_json_out_match(
    healthy: SysfsBuilder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUsbfs(replies=[b"\x1b"])
    monkeypatch.setattr(
        cli, "rgb_exposure_info_backend_factory", lambda _s, _d: make_backend(healthy, fake)
    )
    output = tmp_path / "info.json"
    assert (
        cli.main(
            [
                "rgb-exposure-info",
                "--execute-rgb-exposure-info-read",
                "--serial",
                REAL_SERIAL,
                "--json",
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(output.read_text(encoding="utf-8"))
    assert stdout["reply"]["requested_length"] == 1
    assert stdout["reply"]["actual_length"] == 1
    assert stdout["reading"]["raw_info"] == 0x1B
    assert stdout["bit_view"]["supports_get"] is True
    assert stdout["bit_view"]["supports_set"] is True
    assert len(fake.controls) == 1


def test_cli_live_human_reports_limitations(
    healthy: SysfsBuilder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeUsbfs(replies=[b"\x1f"])
    monkeypatch.setattr(
        cli, "rgb_exposure_info_backend_factory", lambda _s, _d: make_backend(healthy, fake)
    )
    assert (
        cli.main(["rgb-exposure-info", "--execute-rgb-exposure-info-read", "--serial", REAL_SERIAL])
        == 0
    )
    human = capsys.readouterr().out
    assert "raw_info: 31" in human
    assert "supports_get: True" in human
    assert "syntactic Linux-UAPI" in human
