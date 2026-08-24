"""Offline proof for the fixed MINI RGB exposure-extension GET_CUR.

The fake usbfs object is the same syscall-level fixture used by the accepted
depth-scale tests.  No test in this module opens a host device node: the
backend receives every open, ioctl and close operation through injection.
"""

from __future__ import annotations

import ast
import ctypes
import dataclasses
import errno
import hashlib
import json
import struct
from pathlib import Path

import pytest

import scanner3d.cli as cli
import scanner3d.rgb_exposure_extension as extension
from scanner3d.control import USBDEVFS_CONTROL, pack_ctrltransfer
from scanner3d.errors import RgbExposureExtensionNotAllowed
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

EXPECTED_SETUP = bytes.fromhex("a181001200043c00e8030000")


def extension_reply(*words: int) -> bytes:
    assert len(words) == 4
    return struct.pack("<IIII", *words) + bytes(extension.RESULT_LENGTH - 16)


def make_backend(
    builder: SysfsBuilder, fake: FakeUsbfs
) -> extension.UsbfsRgbExposureExtensionBackend:
    return extension.UsbfsRgbExposureExtensionBackend(
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=builder.sysfs,
        dev_root=builder.dev,
    )


def test_fixed_wire_setup_and_whitelist_are_immutable() -> None:
    request = extension.READ_TRANSFER.request
    buffer = ctypes.create_string_buffer(extension.RESULT_LENGTH)
    assert bytes(pack_ctrltransfer(request, ctypes.addressof(buffer)))[:12] == EXPECTED_SETUP
    assert request.wire_tuple == (0xA1, 0x81, 0x1200, 0x0400, 60, 1000)
    assert extension.MAX_CONTROL_TRANSFERS == 1
    assert len(extension.WHITELIST) == 1
    assert extension.WHITELIST[0] == extension.READ_TRANSFER
    assert not any(
        transfer.request for transfer in extension.WHITELIST if not transfer.request.is_read
    )


def test_setup_drift_and_arbitrary_surface_are_rejected() -> None:
    request = extension.READ_TRANSFER.request
    for altered in (
        dataclasses.replace(request, request=0x01),
        dataclasses.replace(request, value=0x1300),
        dataclasses.replace(request, index=0x0500),
        dataclasses.replace(request, length=59),
        dataclasses.replace(request, timeout_ms=999),
    ):
        with pytest.raises(RgbExposureExtensionNotAllowed):
            extension.ensure_transfer_allowed(
                extension.ExtensionTransfer(extension.Step.RESULT_READ, altered, "drift")
            )
    source = Path(extension.__file__).read_text(encoding="utf-8")
    assert "SET_CUR" not in source
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
        for node in ast.walk(tree)
    )


def test_decode_fixture_reports_raw_words_sdk_mapping_and_warnings() -> None:
    raw = extension_reply(17, 9, 3, 7)
    reading = extension.decode_reply(raw)
    assert reading is not None
    assert reading.words == (17, 9, 3, 7)
    assert reading.mapping.current == 17_000
    assert reading.mapping.minimum == 9_000
    assert reading.mapping.maximum == 3_000
    assert reading.mapping.step == 1000
    assert reading.as_dict()["word3_logged_only"] is True
    assert {warning.code for warning in reading.warnings} == {
        "minimum_greater_than_maximum",
        "current_outside_range",
    }
    assert extension.decode_reply(raw[:-1]) is None


def test_dry_run_is_inert_and_does_not_construct_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry run constructed a live backend")

    monkeypatch.setattr(cli, "rgb_exposure_extension_backend_factory", fail)
    assert cli.main(["rgb-exposure-extension", "--serial", REAL_SERIAL]) == 0


def test_cli_json_stdout_and_json_out_are_equivalent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result.json"
    assert (
        cli.main(
            [
                "rgb-exposure-extension",
                "--json",
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == written
    assert stdout["bounds"] == {
        "max_control_transfers": 1,
        "reply_bytes": 60,
        "timeout_ms": 1000,
        "retries": 0,
    }


def test_cli_requires_full_execute_flag_and_serial() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["rgb-exposure-extension", "--execute-rgb-exposure-extension-re"]
        )
    assert cli.main(["rgb-exposure-extension", "--execute-rgb-exposure-extension-read"]) == 10


def test_wrong_serial_opens_nothing(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()
    outcome = make_backend(healthy, fake).run("not-authorized")
    assert outcome.status is OwnershipStatus.REFUSED
    assert fake.opened == []
    assert fake.commands == []
    assert outcome.transfers == ()


def test_human_rendering_and_exact_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["rgb-exposure-extension"]) == 0
    human = capsys.readouterr().out
    assert "Fixed transfer" in human
    assert "GET_CUR only" in human
    assert "Status: not_attempted" in human
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["rgb-exposure-extension", "--help"])
    help_text = capsys.readouterr().out
    assert "--execute-rgb-exposure-extension-read" in help_text
    assert "--json-out" in help_text


def test_exact_single_read_decodes_and_cleans_up(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(replies=[extension_reply(17, 1, 60_000, 8)])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.succeeded
    assert outcome.reading is not None
    assert outcome.reading.words == (17, 1, 60_000, 8)
    assert outcome.reading.mapping.current == 17_000
    assert outcome.reading.mapping.minimum == 1_000
    assert outcome.reading.mapping.maximum == 60_000_000
    assert outcome.reading.mapping.step == 1000
    assert len(outcome.transfers) == 1
    assert len(fake.controls) == 1
    assert fake.setups == [EXPECTED_SETUP]
    assert outcome.reply.raw_length == 60
    assert outcome.reply.sha256 == hashlib.sha256(extension_reply(17, 1, 60_000, 8)).hexdigest()
    assert [phase.phase for phase in outcome.phases] == [
        Phase.CONFIRM_DRIVER,
        Phase.CONFIRM_DRIVER,
        Phase.DISCONNECT,
        Phase.CLAIM,
        Phase.CONTROL_READ,
        Phase.RELEASE,
        Phase.RECONNECT,
    ]
    assert fake.commands.count(USBDEVFS_CONTROL) == 1
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (extension_reply(1, 2, 3, 4)[:59], extension.ExtensionOutcome.SHORT_RESPONSE),
        (extension_reply(1, 2, 3, 4) + b"x", extension.ExtensionOutcome.OVERLONG_RESPONSE),
    ],
)
def test_short_and_overlong_are_typed_without_decode_or_retry(
    healthy: SysfsBuilder, reply: bytes, expected: extension.ExtensionOutcome
) -> None:
    fake = FakeUsbfs(replies=[reply])
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.extension_outcome is expected
    assert outcome.reading is None
    assert not outcome.succeeded
    assert len(outcome.transfers) == 1
    assert len(fake.controls) == 1
    assert outcome.reply.actual_length == len(reply)
    assert outcome.reply.raw_length == min(len(reply), 60)


def test_ioctl_errno_and_cleanup_failures_remain_typed(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(control_errors={0: OSError(errno.EIO, "synthetic read error")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert outcome.extension_outcome is extension.ExtensionOutcome.TRANSFER_ERROR
    assert outcome.errno == errno.EIO
    assert len(fake.controls) == 1
    assert [phase.phase for phase in outcome.phases][-2:] == [Phase.RELEASE, Phase.RECONNECT]

    cleanup_fake = FakeUsbfs(errors={USBDEVFS_CONNECT: OSError(errno.EIO, "reconnect")})
    cleanup = make_backend(healthy, cleanup_fake).run(REAL_SERIAL)
    assert cleanup.exit_code == CLEANUP_FAILED_EXIT
    assert not cleanup.succeeded
    assert len(cleanup_fake.controls) == 1


def test_detach_errno_does_not_issue_a_transfer(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_DISCONNECT: OSError(errno.EBUSY, "detach")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.status is OwnershipStatus.DETACH_FAILED
    assert fake.driver_codes == [USBDEVFS_DISCONNECT]
    assert fake.controls == []
    assert outcome.transfers == ()


def test_release_failure_still_attempts_reconnect(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(errors={USBDEVFS_RELEASEINTERFACE: OSError(errno.EIO, "release")})
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert len(fake.controls) == 1
    assert outcome.transfers[0].ok


def test_open_absent_ambiguous_and_unbound_stop_before_transfer(builder: SysfsBuilder) -> None:
    assert (
        make_backend(builder, FakeUsbfs()).run(REAL_SERIAL).status is OwnershipStatus.DEVICE_ABSENT
    )
    builder.add_healthy_mini()
    builder.add_healthy_mini(
        usb_path="1-4.2", serial="other", busnum="2", devnum="25", devpath="4.2"
    )
    builder.add_usbfs(bus=2, device=25)
    ambiguous = make_backend(builder, FakeUsbfs()).preflight(None)
    assert ambiguous.status is OwnershipStatus.DEVICE_ABSENT
    assert "pass --serial" in ambiguous.detail

    unbound = SysfsBuilder(builder.root / "unbound")
    base = unbound.add_usb_device()
    unbound.add_usbfs()
    unbound.add_interface(base, 0, driver=None)
    result = make_backend(unbound, FakeUsbfs()).run(REAL_SERIAL)
    assert result.status is OwnershipStatus.REFUSED


def test_open_permission_and_driver_claim_failures_do_not_retry(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs(open_error=PermissionError(errno.EACCES, "denied"))
    result = make_backend(healthy, fake).run(REAL_SERIAL)
    assert result.status is OwnershipStatus.PERMISSION_DENIED
    assert not fake.controls

    claim = FakeUsbfs(errors={USBDEVFS_CLAIMINTERFACE: OSError(errno.EBUSY, "busy")})
    result = make_backend(healthy, claim).run(REAL_SERIAL)
    assert result.status is OwnershipStatus.CLAIM_FAILED
    assert not claim.controls
    assert claim.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]


def test_signal_or_exception_in_read_is_preserved_and_cleanup_runs(healthy: SysfsBuilder) -> None:
    def signal_read(index: int) -> None:
        raise SignalReceived(15)

    fake = FakeUsbfs(on_control=signal_read)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.extension_outcome is extension.ExtensionOutcome.INTERRUPTED
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert len(outcome.transfers) == 1
    assert outcome.transfers[0].exception_type == "SignalReceived"
    assert len(fake.controls) == 1
    assert outcome.cleanup.ok
    control_phases = [phase for phase in outcome.phases if phase.phase is Phase.CONTROL_READ]
    assert len(control_phases) == 1
    assert not control_phases[0].ok
    assert control_phases[0].duration_s is not None


def test_unexpected_runtime_error_is_exception_with_cleanup(healthy: SysfsBuilder) -> None:
    def explode(index: int) -> None:
        raise RuntimeError("synthetic ioctl callback")

    fake = FakeUsbfs(on_control=explode)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.extension_outcome is extension.ExtensionOutcome.EXCEPTION
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert len(outcome.transfers) == 1
    assert outcome.transfers[0].exception_type == "RuntimeError"
    control_phases = [phase for phase in outcome.phases if phase.phase is Phase.CONTROL_READ]
    assert len(control_phases) == 1
    assert not control_phases[0].ok
    assert outcome.cleanup.ok
    assert fake.driver_codes == [USBDEVFS_DISCONNECT, USBDEVFS_CONNECT]
    assert len(fake.controls) == 1


def test_unexpected_base_exception_is_not_mislabeled_signal(healthy: SysfsBuilder) -> None:
    class UnexpectedBase(BaseException):
        pass

    def explode(index: int) -> None:
        raise UnexpectedBase("synthetic base exception")

    fake = FakeUsbfs(on_control=explode)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.extension_outcome is extension.ExtensionOutcome.EXCEPTION
    assert outcome.status is OwnershipStatus.IOCTL_ERROR
    assert outcome.transfers[0].exception_type == "UnexpectedBase"
    assert [phase.phase for phase in outcome.phases].count(Phase.CONTROL_READ) == 1
    assert outcome.cleanup.ok
    assert len(fake.controls) == 1


def test_keyboard_interrupt_is_typed_as_interrupted(healthy: SysfsBuilder) -> None:
    def interrupt(index: int) -> None:
        raise KeyboardInterrupt()

    fake = FakeUsbfs(on_control=interrupt)
    outcome = make_backend(healthy, fake).run(REAL_SERIAL)
    assert outcome.extension_outcome is extension.ExtensionOutcome.INTERRUPTED
    assert outcome.status is OwnershipStatus.INTERRUPTED
    assert outcome.transfers[0].exception_type == "KeyboardInterrupt"
    assert outcome.cleanup.ok
    assert len(fake.controls) == 1


def test_changed_identity_after_open_is_refused(healthy: SysfsBuilder) -> None:
    fake = FakeUsbfs()

    def mutate_after_open(path: str) -> int:
        fd = fake.open(path)
        (healthy.devices / "1-4.1" / "serial").write_text("replacement\n", encoding="utf-8")
        return fd

    backend = extension.UsbfsRgbExposureExtensionBackend(
        opener=mutate_after_open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
        sysfs_root=healthy.sysfs,
        dev_root=healthy.dev,
    )
    result = backend.run(REAL_SERIAL)
    assert result.status is OwnershipStatus.DEVICE_REPLACED
    assert result.transfers == ()
    assert fake.closed == [7]
    assert fake.driver_codes == []
    assert fake.controls == []


def test_live_cli_json_stdout_and_json_out_equivalence(
    healthy: SysfsBuilder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUsbfs(replies=[extension_reply(17, 1, 60_000, 8)])
    monkeypatch.setattr(
        cli,
        "rgb_exposure_extension_backend_factory",
        lambda _sysfs, _dev: make_backend(healthy, fake),
    )
    output = tmp_path / "live.json"
    assert (
        cli.main(
            [
                "rgb-exposure-extension",
                "--execute-rgb-exposure-extension-read",
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
    written = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == written
    assert stdout["succeeded"] is True
    assert len(stdout["transfers"]) == 1
    assert len(fake.controls) == 1
