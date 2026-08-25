"""End-to-end CLI tests over fixture trees (no hardware, no SDK)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scanner3d.cli as cli_module
from scanner3d.cli import main
from scanner3d.control import UsbfsControlBackend
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
)
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    EXPECTED_INTERFACE_DRIVER,
    OWNED_INTERFACE,
    OWNED_REQUEST,
    OWNERSHIP_EXIT_CODES,
    CleanupRecord,
    OwnershipOutcome,
    OwnershipStatus,
    UsbfsOwnershipBackend,
)
from scanner3d.v4l2_controls import (
    InventoryOutcome,
    InventoryStatus,
    inventory_plan,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.fake_v4l2 import FakeKernel
from tests.test_control import PADDED_OK, ExplodingOpener, FakeSyscalls


def _roots(b: SysfsBuilder) -> list[str]:
    return ["--sysfs-root", str(b.sysfs), "--dev-root", str(b.dev)]


class _RgbControlsSpy:
    def __init__(self) -> None:
        self.dry_calls: list[str | None] = []
        self.run_calls: list[str] = []

    @staticmethod
    def _outcome(
        detail: str, status: InventoryStatus = InventoryStatus.NOT_ATTEMPTED
    ) -> InventoryOutcome:
        return InventoryOutcome(status, detail, inventory_plan())

    def dry_run(self, serial: str | None = None) -> InventoryOutcome:
        self.dry_calls.append(serial)
        return self._outcome("spy dry run")

    def run(self, serial: str) -> InventoryOutcome:
        self.run_calls.append(serial)
        return self._outcome("spy refusal", InventoryStatus.REFUSED)


def test_doctor_human_output_on_healthy_device(healthy: SysfsBuilder, capsys) -> None:
    code = main(["doctor", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert REAL_SERIAL in out
    assert "480 Mb/s (USB 2.0 high-speed)" in out
    assert "Status: ok" in out
    # Skipped hardware checks must be visibly skipped, not passed.
    assert "[SKIP] frame_delivery" in out
    assert "[SKIP] calibration_metadata" in out


def test_doctor_json_output_is_machine_readable(healthy: SysfsBuilder, capsys) -> None:
    code = main(["doctor", "--json", *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    device = payload["devices"][0]
    assert device["serial"] == REAL_SERIAL
    assert device["speed_mbps"] == 480
    roles = {f["role"]: f for f in device["camera_functions"]}
    assert set(roles) == {"depth", "rgb"}
    assert roles["depth"]["interface_number"] == 0
    assert roles["rgb"]["interface_number"] == 2
    assert payload["reference_probe"]["ran"] is False


def test_doctor_absent_device_exits_cleanly(empty_sysfs: SysfsBuilder, capsys) -> None:
    code = main(["doctor", "--json", *_roots(empty_sysfs)])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_CODES[DoctorStatus.DEVICE_ABSENT] == 2
    assert payload["status"] == "device_absent"
    assert payload["devices"] == []


def test_doctor_json_out_writes_the_same_report(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "report.json"
    main(["doctor", "--json", "--json-out", str(out), *_roots(healthy)])
    stdout_payload = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text()) == stdout_payload


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda b: (b.dev / "video0").chmod(0o000), DoctorStatus.PERMISSION_DENIED),
        (lambda b: (b.dev / "bus/usb/001/024").chmod(0o000), DoctorStatus.PERMISSION_DENIED),
    ],
)
def test_doctor_exit_code_matches_status(builder: SysfsBuilder, capsys, mutate, status) -> None:
    builder.add_healthy_mini()
    mutate(builder)
    code = main(["doctor", "--json", *_roots(builder)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == str(status)
    assert code == EXIT_CODES[status]


def test_list_reports_the_device(healthy: SysfsBuilder, capsys) -> None:
    code = main(["list", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert REAL_SERIAL in out and "2207:110c" in out


def test_list_with_no_device_exits_two(empty_sysfs: SysfsBuilder, capsys) -> None:
    code = main(["list", *_roots(empty_sysfs)])
    assert code == 2
    assert "no matching scanner found" in capsys.readouterr().out


def test_reference_probe_without_sdk_reports_and_does_not_hang(
    healthy: SysfsBuilder, tmp_path: Path, capsys, monkeypatch
) -> None:
    """The headline no-SDK case: doctor must degrade, not crash or block."""
    monkeypatch.delenv("REVOPOINT_SDK_ROOT", raising=False)
    monkeypatch.delenv("SCANNER3D_REFERENCE_PROBE", raising=False)
    code = main(
        [
            "doctor",
            "--json",
            "--reference-probe",
            "--sdk-manifest",
            str(tmp_path / "absent.json"),
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0  # discovery still passed; hardware checks are 'not run'
    assert payload["reference_probe"]["ran"] is False
    assert "REVOPOINT_SDK_ROOT" in payload["reference_probe"]["reason"]
    frame_check = next(c for c in payload["checks"] if c["name"] == "frame_delivery")
    assert frame_check["ran"] is False
    assert frame_check["detail"].startswith("not run:")
    assert "reference probe unavailable" in captured.err


def test_reference_probe_runs_a_stub_binary(healthy: SysfsBuilder, tmp_path: Path, capsys) -> None:
    from tests.test_reference import _fake_sdk, _manifest_for, _script

    root = _fake_sdk(tmp_path)
    binary = _script(
        tmp_path,
        "import json;print(json.dumps({"
        '"frames": {"depth": {"count": 2}, "rgb": {"count": 2}},'
        '"intrinsics": {"depth": {"fx": 1}, "rgb": {"fx": 2}},'
        '"extrinsics": {"rotation": [1,0,0,0,1,0,0,0,1], "translation": [0,0,0]},'
        '"depth_scale": 0.1}))',
    )
    code = main(
        [
            "doctor",
            "--json",
            "--reference-probe",
            "--sdk-root",
            str(root),
            "--probe-binary",
            str(binary),
            "--sdk-manifest",
            str(_manifest_for(root, tmp_path)),
            "--probe-timeout",
            "10",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["reference_probe"]["sdk"]["verified"] is True
    frame_check = next(c for c in payload["checks"] if c["name"] == "frame_delivery")
    assert frame_check["ran"] and frame_check["status"] == "ok"


def test_reference_probe_timeout_is_bounded_and_reported(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    from tests.test_reference import _fake_sdk, _manifest_for, _script

    root = _fake_sdk(tmp_path)
    binary = _script(tmp_path, "import time;time.sleep(60)")
    code = main(
        [
            "doctor",
            "--json",
            "--reference-probe",
            "--sdk-root",
            str(root),
            "--probe-binary",
            str(binary),
            "--sdk-manifest",
            str(_manifest_for(root, tmp_path)),
            "--probe-timeout",
            "1",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_CODES[DoctorStatus.NO_FRAMES]
    assert payload["reference_probe"]["timed_out"] is True
    assert payload["status"] == "no_frames"


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_no_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


# --- firmware-version: the one owned control read ----------------------------


def _patch_backend(monkeypatch, **injected) -> None:
    """Make the CLI build a backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> UsbfsControlBackend:
        return UsbfsControlBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "UsbfsControlBackend", factory)


def _patch_fake(monkeypatch, fake: FakeSyscalls) -> None:
    _patch_backend(
        monkeypatch,
        opener=fake.open,
        ioctl_fn=fake.ioctl,
        closer=fake.close,
        clock=fake.clock,
    )


def test_firmware_version_defaults_to_a_dry_run(healthy: SysfsBuilder, monkeypatch, capsys) -> None:
    """No --execute: nothing is opened, and the real syscalls are never reached."""
    _patch_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["firmware-version", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out
    assert "--execute" in out
    assert "bmRequestType=0xa1 bRequest=0x81 wValue=0x0600 wIndex=0x0300 wLength=60" in out
    assert "timeout=3000 ms" in out
    assert "device -> host (read-only)" in out


def test_firmware_version_dry_run_names_the_declared_entity(healthy: SysfsBuilder, capsys) -> None:
    main(["firmware-version", *_roots(healthy)])
    out = capsys.readouterr().out
    assert "entity     id 3 subtype output_terminal" in out
    assert "interface  0 kernel driver uvcvideo" in out


def test_firmware_version_execute_reports_the_validated_payload(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    fake = FakeSyscalls()
    _patch_fake(monkeypatch, fake)
    code = main(["firmware-version", "--execute", "--serial", REAL_SERIAL, *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: ok" in out
    assert "v2.8.7.20220610" in out
    assert "v7.5.7.0426" in out
    assert hashlib.sha256(PADDED_OK).hexdigest() in out
    assert "bytes      60" in out
    assert "validated  True" in out
    assert "Duration:" in out
    assert len(fake.ioctls) == 1


def test_firmware_version_execute_json_shape(healthy: SysfsBuilder, monkeypatch, capsys) -> None:
    _patch_fake(monkeypatch, FakeSyscalls())
    code = main(["firmware-version", "--execute", "--json", *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["request"]["wIndex"] == "0x0300"
    assert payload["response"]["firmware_version"] == "v2.8.7.20220610"
    assert payload["response"]["algorithm_version"] == "v7.5.7.0426"
    assert payload["target"]["serial"] == REAL_SERIAL
    assert payload["addressed_entity"]["subtype_name"] == "output_terminal"


def test_firmware_version_absent_device_exits_two(empty_sysfs: SysfsBuilder, capsys) -> None:
    code = main(["firmware-version", "--json", *_roots(empty_sysfs)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "device_absent"
    assert payload["target"] is None


def test_firmware_version_json_out_matches_stdout(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "versions.json"
    main(["firmware-version", "--json", "--json-out", str(out), *_roots(healthy)])
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_firmware_version_output_is_stable_across_runs(healthy: SysfsBuilder, capsys) -> None:
    main(["firmware-version", "--json", *_roots(healthy)])
    first = capsys.readouterr().out
    main(["firmware-version", "--json", *_roots(healthy)])
    assert capsys.readouterr().out == first


def test_firmware_version_exposes_no_arbitrary_request_surface(capsys) -> None:
    """There is no flag that can change any field of the setup packet."""
    with pytest.raises(SystemExit):
        main(["firmware-version", "--help"])
    out = capsys.readouterr().out
    for flag in ("--request", "--value", "--index", "--length", "--timeout", "--request-type"):
        assert flag not in out
    assert "--execute" in out
    assert "--serial" in out


# --- firmware-version --detach-uvcvideo-from-interface-0 (opt-in ownership) ---


def _patch_ownership_backend(monkeypatch, **injected) -> None:
    """Make the CLI build an ownership backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> UsbfsOwnershipBackend:
        return UsbfsOwnershipBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "UsbfsOwnershipBackend", factory)


def test_firmware_version_detach_flag_defaults_to_false_and_dry_run_is_unchanged(
    healthy: SysfsBuilder, capsys
) -> None:
    """Without the new flag, nothing about the existing dry-run path changes."""
    parser = cli_module.build_parser()
    args = parser.parse_args(["firmware-version", *_roots(healthy)])
    assert args.detach_uvcvideo_from_interface_0 is False

    code = main(["firmware-version", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out


def test_ownership_flag_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The mutating mode refuses to guess a target; it must not open anything."""
    _patch_ownership_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["firmware-version", "--detach-uvcvideo-from-interface-0", *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err
    assert "--detach-uvcvideo-from-interface-0" in err


def test_ownership_dry_run_with_serial_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """Without --execute the fixed lifecycle is only described, never performed."""
    _patch_ownership_backend(monkeypatch, opener=ExplodingOpener())
    code = main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--serial",
            REAL_SERIAL,
            *_roots(healthy),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out
    assert (
        f"Lifecycle (fixed): confirm {EXPECTED_INTERFACE_DRIVER} -> disconnect -> claim -> "
        f"one read -> release -> reconnect, all on interface {OWNED_INTERFACE}"
    ) in out


def test_ownership_json_shape(healthy: SysfsBuilder, capsys) -> None:
    code = main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["mode"] == "temporary-interface-ownership"
    assert payload["owned_interface"] == 0
    assert payload["expected_interface_driver"] == "uvcvideo"
    assert payload["request"]["wIndex"] == "0x0300"
    assert payload["request"]["bmRequestType"] == "0xa1"


def test_ownership_json_out_matches_stdout(healthy: SysfsBuilder, tmp_path: Path, capsys) -> None:
    out = tmp_path / "ownership.json"
    main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--serial",
            REAL_SERIAL,
            "--json",
            "--json-out",
            str(out),
            *_roots(healthy),
        ]
    )
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_ownership_refuses_when_interface0_is_not_uvcvideo(builder: SysfsBuilder, capsys) -> None:
    """A fixture-driven refusal: interface 0 is bound to a driver we do not expect."""
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, driver="usbfs")
    code = main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(builder),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert payload["status"] == "refused"
    assert "interface 0" in payload["detail"]


def test_ownership_device_absent_exits_two(empty_sysfs: SysfsBuilder, capsys) -> None:
    code = main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(empty_sysfs),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "device_absent"


def test_firmware_version_help_mentions_detach_flag(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["firmware-version", "--help"])
    out = capsys.readouterr().out
    assert "--detach-uvcvideo-from-interface-0" in out
    assert "TEMPORARILY DETACH" in out


def test_ownership_cleanup_failure_forces_nonzero_exit_and_prints_recovery(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """A failed cleanup must never be reported as success, whatever the read did."""
    guidance = "as root, rebind interface 0 to uvcvideo, or physically replug the scanner"
    outcome = OwnershipOutcome(
        status=OwnershipStatus.OK,
        request=OWNED_REQUEST,
        target=None,
        phases=(),
        cleanup=CleanupRecord(
            attempted=True,
            phases=(),
            ok=False,
            note="interface 0 was NOT fully restored",
            guidance=guidance,
        ),
        payload=None,
        transfer_duration_s=0.001,
        total_duration_s=0.002,
        detail="read succeeded but cleanup did not restore the driver",
    )

    class _StubBackend:
        def run(self, serial: str) -> OwnershipOutcome:
            return outcome

    monkeypatch.setattr(cli_module, "UsbfsOwnershipBackend", lambda **kwargs: _StubBackend())

    code = main(
        [
            "firmware-version",
            "--detach-uvcvideo-from-interface-0",
            "--execute",
            "--serial",
            REAL_SERIAL,
            *_roots(healthy),
        ]
    )
    err = capsys.readouterr().err
    assert code == CLEANUP_FAILED_EXIT == 9
    assert outcome.succeeded is False
    assert guidance in err


# --- inited-sequence: the four-transfer /tmp/inited write sequence -----------

import hashlib as _hashlib  # noqa: E402  (grouped with the section it serves)

from scanner3d.init_sequence import (  # noqa: E402
    ENTITY_ID,
    INITED_PATH,
    OBSERVED_ALGORITHM_RECORD,
    UsbfsInitSequenceBackend,
)
from scanner3d.ownership import PROTOCOL_CLEANUP_FAILED_EXIT, USBDEVFS_CONNECT  # noqa: E402
from tests.test_init_sequence import FakeUsbfs  # noqa: E402


def _patch_init_backend(monkeypatch, **injected) -> None:
    """Make the CLI build an init-sequence backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> UsbfsInitSequenceBackend:
        return UsbfsInitSequenceBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "UsbfsInitSequenceBackend", factory)


def test_inited_sequence_dry_run_opens_nothing_and_describes_the_fixed_sequence(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """No --execute-write-sequence: nothing is opened, only the four fixed transfers."""
    _patch_init_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["inited-sequence", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out
    assert out.count("wLength=60") == 4
    assert ("bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60") in out
    assert ("bmRequestType=0xa1 bRequest=0x81 wValue=0x0200 wIndex=0x0400 wLength=60") in out
    assert "012f746d702f696e69746564" in out


def test_inited_sequence_execute_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The only write-capable mode in this project refuses to guess a target."""
    _patch_init_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["inited-sequence", "--execute-write-sequence", *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err
    assert "--execute-write-sequence" in err


def test_inited_sequence_execute_reports_the_validated_sequence(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    fake = FakeUsbfs()
    _patch_init_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        ["inited-sequence", "--execute-write-sequence", "--serial", REAL_SERIAL, *_roots(healthy)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: ok" in out
    assert "v7.5.7.0426" in out
    assert _hashlib.sha256(OBSERVED_ALGORITHM_RECORD).hexdigest() in out
    assert "matched observed  True" in out
    assert "cleared    True" in out
    assert "restored   True" in out


def test_inited_sequence_execute_json_shape(healthy: SysfsBuilder, monkeypatch, capsys) -> None:
    fake = FakeUsbfs()
    _patch_init_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "inited-sequence",
            "--execute-write-sequence",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["mode"] == "inited-path-sequence"
    assert payload["entity_id"] == ENTITY_ID == 4
    assert payload["path"] == INITED_PATH == "/tmp/inited"
    assert len(payload["transfers"]) == 4
    assert payload["protocol_cleanup"]["ok"] is True
    assert payload["cleanup"]["ok"] is True


def test_inited_sequence_json_out_matches_stdout(
    healthy: SysfsBuilder, tmp_path: Path, capsys, monkeypatch
) -> None:
    fake = FakeUsbfs()
    _patch_init_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    out = tmp_path / "inited.json"
    main(
        [
            "inited-sequence",
            "--execute-write-sequence",
            "--serial",
            REAL_SERIAL,
            "--json",
            "--json-out",
            str(out),
            *_roots(healthy),
        ]
    )
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_inited_sequence_failed_clearing_write_is_never_reported_as_success(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    import errno as errno_mod

    fake = FakeUsbfs(control_errors={3: OSError(errno_mod.EPIPE, "stall")})
    _patch_init_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "inited-sequence",
            "--execute-write-sequence",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == PROTOCOL_CLEANUP_FAILED_EXIT == 14
    assert payload["succeeded"] is False
    assert "the clearing write did not complete" in captured.err
    assert INITED_PATH in captured.err


def test_inited_sequence_failed_clearing_write_and_failed_reconnect_both_report(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    import errno as errno_mod

    fake = FakeUsbfs(
        control_errors={3: OSError(errno_mod.EPIPE, "stall")},
        errors={USBDEVFS_CONNECT: OSError(errno_mod.EBUSY, "busy")},
    )
    _patch_init_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "inited-sequence",
            "--execute-write-sequence",
            "--serial",
            REAL_SERIAL,
            *_roots(healthy),
        ]
    )
    err = capsys.readouterr().err
    assert code == CLEANUP_FAILED_EXIT == 9
    assert "the clearing write did not complete" in err
    assert "cleanup did not fully restore the device" in err


def test_inited_sequence_exposes_no_arbitrary_transfer_surface(capsys) -> None:
    """There is no flag that can vary the path, entity, selector, length or timeout."""
    with pytest.raises(SystemExit):
        main(["inited-sequence", "--help"])
    out = capsys.readouterr().out
    for flag in (
        "--path",
        "--selector",
        "--entity",
        "--interface",
        "--value",
        "--index",
        "--length",
        "--timeout",
        "--request",
        "--repeat",
        "--count",
    ):
        assert flag not in out
    assert "--execute-write-sequence" in out
    assert "--serial" in out


def test_inited_sequence_device_absent_exits_two(empty_sysfs: SysfsBuilder, capsys) -> None:
    code = main(
        [
            "inited-sequence",
            "--execute-write-sequence",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(empty_sysfs),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "device_absent"


# --- depth-intrinsics: the three-transfer /data/camparam/Pl.bin key read -----

import errno as errno_mod  # noqa: E402  (grouped with the section it serves)
import struct  # noqa: E402

from scanner3d.intrinsics import (  # noqa: E402
    AUTHORIZED_SERIAL,
    INTRINSICS_KEY,
    INTRINSICS_LENGTH,
    MAX_GET_TRANSFERS,
    PATH_PAYLOAD,
    PAYLOAD_LENGTH,
    ZERO_PAYLOAD,
    UsbfsIntrinsicsBackend,
)

#: The setup prefix (everything but the data pointer) of each whitelisted
#: transfer, as it must appear on the wire. Computed once from the fixed
#: request fields so a test failure here means the request itself changed.
_INTRINSICS_PATH_WRITE_SETUP = bytes.fromhex("2101000100043c00e8030000")
_INTRINSICS_CHUNK_READ_SETUP = bytes.fromhex("a181000200043c00e8030000")


def _patch_intrinsics_backend(monkeypatch, **injected) -> None:
    """Make the CLI build an intrinsics backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> UsbfsIntrinsicsBackend:
        return UsbfsIntrinsicsBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "UsbfsIntrinsicsBackend", factory)


def _intrinsics_reply(data: bytes = b"") -> bytes:
    """One 60-byte device->host reply: a uint32_le length word, data, zero padding."""
    header = struct.pack("<I", len(data))
    reply = header + data
    assert len(reply) <= PAYLOAD_LENGTH
    return reply + bytes(PAYLOAD_LENGTH - len(reply))


def _intrinsics_record_bytes() -> bytes:
    """A complete 40-byte vendor Intrinsics record with exactly representable floats."""
    record = struct.pack("<hh9f", 640, 400, 615.5, 0.0, 320.0, 0.0, 616.25, 200.0, 0.0, 0.0, 1.0)
    assert len(record) == INTRINSICS_LENGTH == 40
    return record


def test_depth_intrinsics_dry_run_opens_nothing_and_describes_the_fixed_whitelist(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """No --execute-calibration-read: nothing is opened, only the fixed whitelist."""
    _patch_intrinsics_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-intrinsics", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out
    assert "bmRequestType=0x21 bRequest=0x01 wValue=0x0100 wIndex=0x0400 wLength=60" in out
    assert "bmRequestType=0xa1 bRequest=0x81 wValue=0x0200" in out
    assert INTRINSICS_KEY in out
    assert "output capacity   256 bytes" in out
    assert f"result reads      at most {MAX_GET_TRANSFERS}" in out
    assert MAX_GET_TRANSFERS == 8


def test_depth_intrinsics_execute_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The write-capable mode refuses to guess a target, just like inited-sequence."""
    _patch_intrinsics_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-intrinsics", "--execute-calibration-read", *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err
    assert "--execute-calibration-read" in err


def test_depth_intrinsics_execute_reports_the_parsed_record(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    record_bytes = _intrinsics_record_bytes()
    first, second = record_bytes[:20], record_bytes[20:]
    fake = FakeUsbfs(
        replies=[_intrinsics_reply(first), _intrinsics_reply(second), _intrinsics_reply()]
    )
    _patch_intrinsics_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "depth-intrinsics",
            "--execute-calibration-read",
            "--serial",
            REAL_SERIAL,
            *_roots(healthy),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: ok" in out
    assert "  length     40 bytes" in out
    assert f"  sha256     {hashlib.sha256(record_bytes).hexdigest()}" in out
    assert f"  hex        {record_bytes.hex()}" in out
    assert "  frame      640x400" in out
    assert "  fx 615.5  fy 616.25  cx 320.0  cy 200.0" in out
    assert "    [615.5  0.0  320.0]" in out
    assert "    [0.0  616.25  200.0]" in out
    assert "    [0.0  0.0  1.0]" in out
    assert fake.written == [PATH_PAYLOAD, ZERO_PAYLOAD]
    setups = [setup for setup, _ in fake.controls]
    assert setups == [
        _INTRINSICS_PATH_WRITE_SETUP,
        _INTRINSICS_CHUNK_READ_SETUP,
        _INTRINSICS_CHUNK_READ_SETUP,
        _INTRINSICS_CHUNK_READ_SETUP,
        _INTRINSICS_PATH_WRITE_SETUP,
    ]


def test_depth_intrinsics_json_output_is_machine_readable_and_suppresses_human_render(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    record_bytes = _intrinsics_record_bytes()
    fake = FakeUsbfs(replies=[_intrinsics_reply(record_bytes), _intrinsics_reply()])
    _patch_intrinsics_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "depth-intrinsics",
            "--execute-calibration-read",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert {"read_outcome", "blob", "record", "protocol_cleanup", "cleanup", "exit_code"} <= set(
        payload
    )
    assert payload["record"] is not None
    assert "scanner3d depth-intrinsics" not in captured.out


def test_depth_intrinsics_json_out_matches_stdout(
    healthy: SysfsBuilder, tmp_path: Path, monkeypatch, capsys
) -> None:
    record_bytes = _intrinsics_record_bytes()
    fake = FakeUsbfs(replies=[_intrinsics_reply(record_bytes), _intrinsics_reply()])
    _patch_intrinsics_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    out = tmp_path / "intrinsics.json"
    main(
        [
            "depth-intrinsics",
            "--execute-calibration-read",
            "--serial",
            REAL_SERIAL,
            "--json",
            "--json-out",
            str(out),
            *_roots(healthy),
        ]
    )
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_depth_intrinsics_wrong_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The live path is authorized for one named unit; every other serial is refused."""
    assert REAL_SERIAL == AUTHORIZED_SERIAL
    monkeypatch.setenv("SCANNER3D_SERIAL", AUTHORIZED_SERIAL)
    _patch_intrinsics_backend(monkeypatch, opener=ExplodingOpener())
    code = main(
        [
            "depth-intrinsics",
            "--execute-calibration-read",
            "--serial",
            "SOMEOTHERSERIAL0000",
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert payload["status"] == "refused"


def test_depth_intrinsics_exposes_no_arbitrary_transfer_surface(capsys) -> None:
    """There is no flag that can vary the key, entity, selector, length, timeout or cap."""
    with pytest.raises(SystemExit):
        main(["depth-intrinsics", "--help"])
    out = capsys.readouterr().out
    for flag in (
        "--path",
        "--key",
        "--selector",
        "--entity",
        "--interface",
        "--value",
        "--index",
        "--length",
        "--timeout",
        "--request",
        "--repeat",
        "--count",
        "--capacity",
        "--max-reads",
        "--retry",
    ):
        assert flag not in out
    assert "--execute-calibration-read" in out
    assert "--serial" in out


def test_depth_intrinsics_failed_clearing_write_is_never_reported_as_success(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    record_bytes = _intrinsics_record_bytes()
    fake = FakeUsbfs(
        replies=[_intrinsics_reply(record_bytes), _intrinsics_reply()],
        control_errors={3: OSError(errno_mod.EPIPE, "stall")},
    )
    _patch_intrinsics_backend(
        monkeypatch, opener=fake.open, ioctl_fn=fake.ioctl, closer=fake.close, clock=fake.clock
    )
    code = main(
        [
            "depth-intrinsics",
            "--execute-calibration-read",
            "--serial",
            REAL_SERIAL,
            "--json",
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == PROTOCOL_CLEANUP_FAILED_EXIT
    assert payload["succeeded"] is False
    assert "the clearing write did not complete" in captured.err
    assert INTRINSICS_KEY in captured.err


# --- depth-scale: the two-transfer 0x918 property read -----------------------

import argparse  # noqa: E402  (grouped with the section it serves)

from scanner3d.depth_scale import AUTHORIZED_SERIAL as DEPTH_SCALE_AUTHORIZED_SERIAL  # noqa: E402
from scanner3d.depth_scale import (  # noqa: E402
    UNIT_DISCLAIMER,
    UsbfsDepthScaleBackend,
)

#: The exact, closed set of option strings the depth-scale subparser may ever
#: expose. No property id, selector, entity, interface, request code, length,
#: timeout, count or retry flag belongs here - or anywhere else in this CLI.
_DEPTH_SCALE_EXPECTED_OPTIONS = frozenset(
    {
        "-h",
        "--help",
        "--sysfs-root",
        "--dev-root",
        "--json",
        "--serial",
        "--execute-depth-scale-read",
        "--json-out",
    }
)


def _patch_depth_scale_backend(monkeypatch, **injected) -> None:
    """Make the CLI build a depth-scale backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> UsbfsDepthScaleBackend:
        return UsbfsDepthScaleBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "UsbfsDepthScaleBackend", factory)


def _depth_scale_subparser() -> argparse.ArgumentParser:
    """The live ``depth-scale`` subparser, from the real ``build_parser()``."""
    parser = cli_module.build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    result = subparsers_action.choices["depth-scale"]
    assert isinstance(result, argparse.ArgumentParser)
    return result


def test_depth_scale_dry_run_opens_nothing_and_describes_the_fixed_whitelist(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """No --execute-depth-scale-read: nothing is opened, only the fixed whitelist."""
    _patch_depth_scale_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-scale", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: not_attempted" in out


def test_depth_scale_dry_run_never_calls_backend_run(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """Belt and braces: the dry path must never reach ``UsbfsDepthScaleBackend.run``."""

    def _exploding_run(self, serial: str) -> None:
        raise AssertionError("UsbfsDepthScaleBackend.run was called during a dry run")

    monkeypatch.setattr(UsbfsDepthScaleBackend, "run", _exploding_run)
    code = main(["depth-scale", *_roots(healthy)])
    assert code == 0


def test_depth_scale_execute_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The write-capable mode refuses to guess a target and never touches a backend."""

    def _exploding_factory(**kwargs):
        raise AssertionError("a depth-scale backend was constructed without --serial")

    monkeypatch.setattr(cli_module, "UsbfsDepthScaleBackend", _exploding_factory)
    code = main(["depth-scale", "--execute-depth-scale-read", *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err
    assert "--execute-depth-scale-read" in err


def test_depth_scale_execute_flag_is_its_own_unmistakable_flag(healthy: SysfsBuilder) -> None:
    """The flag must be typed in full, and no sibling command's flag works here.

    The ``depth-scale`` subparser sets ``allow_abbrev=False`` precisely so that
    a bare ``--execute`` is a parse error rather than an unambiguous prefix of
    ``--execute-depth-scale-read``. A flag that arms a write to the scanner
    should not be reachable by a prefix that means something else on a sibling
    command.
    """
    for other_flag in (
        "--execute",
        "--execute-depth-scale",
        "--execute-calibration-read",
        "--execute-write-sequence",
        "--detach-uvcvideo-from-interface-0",
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(["depth-scale", other_flag, *_roots(healthy)])
        assert excinfo.value.code == 2

    parser = cli_module.build_parser()
    args = parser.parse_args(["depth-scale", "--execute-depth-scale-read", *_roots(healthy)])
    assert args.execute_depth_scale_read is True


def test_depth_scale_closed_cli_surface_has_no_property_knobs(healthy: SysfsBuilder) -> None:
    """The whole option surface is the fixed set - nothing that could vary the protocol."""
    scale_parser = _depth_scale_subparser()
    option_strings: set[str] = set()
    for action in scale_parser._actions:
        option_strings.update(action.option_strings)
    assert option_strings == set(_DEPTH_SCALE_EXPECTED_OPTIONS)


def test_depth_scale_dry_run_human_output_is_stable_and_truthful(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    _patch_depth_scale_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-scale", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "bmRequestType=0x21 bRequest=0x01 wValue=0x0e00 wIndex=0x0400 wLength=4" in out
    assert "bmRequestType=0xa1 bRequest=0x81 wValue=0x0e00 wIndex=0x0400 wLength=60" in out
    assert "18090000" in out
    assert UNIT_DISCLAIMER in out
    assert "not applicable" in out
    assert "protocol cleanup" in out.lower()


def test_depth_scale_json_output_and_json_out_agree(
    healthy: SysfsBuilder, tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_depth_scale_backend(monkeypatch, opener=ExplodingOpener())
    out_path = tmp_path / "depth_scale.json"
    code = main(["depth-scale", "--json", "--json-out", str(out_path), *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["mode"] == "vendor-depth-scale-property-read"
    assert json.loads(out_path.read_text()) == payload


def test_depth_scale_wrong_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The live path is authorized for one named unit; every other serial is refused."""
    monkeypatch.setenv("SCANNER3D_SERIAL", DEPTH_SCALE_AUTHORIZED_SERIAL)
    assert REAL_SERIAL == DEPTH_SCALE_AUTHORIZED_SERIAL
    _patch_depth_scale_backend(monkeypatch, opener=ExplodingOpener())
    code = main(
        [
            "depth-scale",
            "--execute-depth-scale-read",
            "--serial",
            "SOMEOTHERSERIAL0000",
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert payload["status"] == "refused"


# --- depth-frame: one bounded V4L2/MMAP raw depth frame -----------------------

from scanner3d.v4l2_capture import AUTHORIZED_SERIAL as DEPTH_FRAME_AUTHORIZED_SERIAL  # noqa: E402
from scanner3d.v4l2_capture import (  # noqa: E402
    CAPTURE_DEADLINE_S,
    DEFAULT_OUTPUT_DIR,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    V4l2DepthFrameBackend,
    plan_as_dict,
)

#: The hardware-shaping surface this command must never expose. The mode, the
#: buffer count, the deadline and the wait/retry policy are fixed at import
#: time in scanner3d.v4l2_capture; none of them is a flag.
_DEPTH_FRAME_FORBIDDEN_FLAGS = (
    "--format",
    "--width",
    "--height",
    "--fps",
    "--frame-rate",
    "--buffers",
    "--buffer-count",
    "--timeout",
    "--retries",
    "--retry",
    "--control",
    "--node",
    "--device",
)


def _patch_depth_frame_backend(monkeypatch, **injected) -> None:
    """Make the CLI build a depth-frame backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> V4l2DepthFrameBackend:
        return V4l2DepthFrameBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "V4l2DepthFrameBackend", factory)


def test_depth_frame_dry_run_opens_nothing_and_is_not_attempted(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """No --execute-depth-frame-capture: the node is resolved from sysfs, never opened."""
    _patch_depth_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-frame", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Result: not_attempted" in out
    assert "--execute-depth-frame-capture" in out


def test_depth_frame_dry_run_never_calls_backend_run(healthy: SysfsBuilder, monkeypatch) -> None:
    """Belt and braces: the dry path must never reach ``V4l2DepthFrameBackend.run``."""

    def _exploding_run(self, serial: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        raise AssertionError("V4l2DepthFrameBackend.run was called during a dry run")

    monkeypatch.setattr(V4l2DepthFrameBackend, "run", _exploding_run)
    code = main(["depth-frame", *_roots(healthy)])
    assert code == 0


def test_depth_frame_dry_run_resolves_node_by_role_not_by_number(
    builder: SysfsBuilder, capsys
) -> None:
    """The depth node must be found by role/interface-string, never by the lowest videoN."""
    base = builder.add_usb_device()
    builder.add_usbfs()
    # The depth capture node is video5 here, and an unrelated video0 sits on the
    # RGB interface - a number-based guess would pick video0 and be wrong.
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video5", 0), ("video6", 1)),
    )
    builder.add_interface(base, 1, iface_subclass="02", interface_string="Video Streaming")
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{REAL_SERIAL}",
        video_nodes=(("video0", 0), ("video1", 1)),
    )
    builder.add_interface(base, 3, iface_subclass="02", interface_string="Video Streaming")

    code = main(["depth-frame", "--json", *_roots(builder)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "not_attempted"
    assert payload["target"]["node_name"] == "video5"
    assert payload["target"]["node_path"].endswith("video5")


def test_depth_frame_execute_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The stream-starting mode refuses to guess a target, just like the write-capable ones."""
    _patch_depth_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["depth-frame", "--execute-depth-frame-capture", *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err


def test_depth_frame_execute_wrong_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    # The refusal only applies when the operator NAMED a device. Configure
    # one, or there is nothing for the guard to refuse against.
    monkeypatch.setenv("SCANNER3D_SERIAL", AUTHORIZED_SERIAL)
    """The live path is authorized for one physical unit; every other serial is refused."""
    assert REAL_SERIAL == DEPTH_FRAME_AUTHORIZED_SERIAL
    _patch_depth_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(
        [
            "depth-frame",
            "--execute-depth-frame-capture",
            "--serial",
            "SOMEOTHERSERIAL0000",
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert payload["status"] == "refused"
    assert DEPTH_FRAME_AUTHORIZED_SERIAL in payload["detail"]


@pytest.mark.parametrize("flag", _DEPTH_FRAME_FORBIDDEN_FLAGS)
def test_depth_frame_parser_exposes_no_hardware_shaping_flags(
    healthy: SysfsBuilder, flag: str
) -> None:
    """The mode, buffer count, timeout and retry policy are fixed; none is a flag."""
    with pytest.raises(SystemExit) as excinfo:
        main(["depth-frame", flag, *_roots(healthy)])
    assert excinfo.value.code == 2


def test_depth_frame_json_dry_run_reports_the_fixed_plan(healthy: SysfsBuilder, capsys) -> None:
    code = main(["depth-frame", "--json", *_roots(healthy)])
    raw = capsys.readouterr().out
    payload = json.loads(raw)  # must round-trip through json.loads
    assert code == 0
    assert payload["status"] == "not_attempted"
    plan = payload["plan"]
    assert plan == plan_as_dict()
    assert plan["fourcc"] == FOURCC == "Y16 "
    assert plan["width"] == FRAME_WIDTH == 640
    assert plan["height"] == FRAME_HEIGHT == 400
    assert plan["requested_buffers"] == 4
    assert plan["capture_deadline_s"] == CAPTURE_DEADLINE_S == 8.0
    assert plan["max_dequeues"] == 1
    assert plan["retries"] == 0


def test_depth_frame_json_out_matches_stdout(healthy: SysfsBuilder, tmp_path: Path, capsys) -> None:
    out = tmp_path / "depth_frame.json"
    main(["depth-frame", "--json", "--json-out", str(out), *_roots(healthy)])
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_live_depth_cli_json_out_and_shutdown_failure_recovery(
    healthy: SysfsBuilder, tmp_path: Path, monkeypatch, capsys
) -> None:
    kernel = FakeKernel()
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )

    class FailedFinalizer:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True, True, ShutdownStatus.IOCTL_ERROR, "synthetic off failure"
            )

    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: FailedFinalizer()
    )
    output_path = tmp_path / "depth-failure.json"
    args = [
        "depth-frame",
        "--json",
        "--json-out",
        str(output_path),
        "--execute-depth-frame-capture",
        "--serial",
        REAL_SERIAL,
        *_roots(healthy),
    ]
    result = main(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result != 0
    assert payload["status"] == "complete"
    assert payload["succeeded"] is False
    assert payload["illumination_shutdown"]["status"] == "ioctl_error"
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert "blue-light off command was not carried through" in captured.err


def test_live_depth_cli_human_success_mentions_shutdown(
    healthy: SysfsBuilder, monkeypatch, capsys, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )

    class AcceptedFinalizer:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.COMMAND_ACCEPTED,
                "synthetic off accepted",
                transfer=ShutdownTransferRecord(60, True, "accepted", 0.0),
            )

    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: AcceptedFinalizer()
    )
    result = main(
        [
            "depth-frame",
            "--execute-depth-frame-capture",
            "--serial",
            REAL_SERIAL,
            "--output-dir",
            str(tmp_path / "depth"),
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Blue-light off command (transfer only, NOT an optical result):" in captured.out


def test_live_depth_cli_human_shutdown_failure_is_unmistakable(
    healthy: SysfsBuilder, monkeypatch, capsys, tmp_path: Path
) -> None:
    kernel = FakeKernel()
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )

    class FailedFinalizer:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True, True, ShutdownStatus.IOCTL_ERROR, "synthetic off failure"
            )

    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: FailedFinalizer()
    )
    result = main(
        [
            "depth-frame",
            "--execute-depth-frame-capture",
            "--serial",
            REAL_SERIAL,
            "--output-dir",
            str(tmp_path / "depth-failure"),
            *_roots(healthy),
        ]
    )
    captured = capsys.readouterr()
    assert result != 0
    assert "Blue-light off command (transfer only, NOT an optical result):" in captured.out
    assert "blue-light off command was not carried through" in captured.err
    assert "DO NOT CLAIM THE BLUE LIGHT IS OFF" in captured.err


def test_depth_frame_output_dir_is_accepted_and_reported(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    """--output-dir shapes only local evidence, never the hardware, and is echoed back."""
    custom = tmp_path / "custom-depth-frame-out"
    code = main(["depth-frame", "--json", "--output-dir", str(custom), *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["artifacts"]["directory"] == str(custom)


def test_depth_frame_default_output_dir_is_reported_without_output_dir_flag(
    healthy: SysfsBuilder, capsys
) -> None:
    code = main(["depth-frame", "--json", *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert str(DEFAULT_OUTPUT_DIR) == "/tmp/scanner3d-depth-frame"
    assert payload["artifacts"]["directory"] == str(DEFAULT_OUTPUT_DIR)


def test_depth_frame_human_output_describes_the_fixed_dry_run_truthfully(
    healthy: SysfsBuilder, capsys
) -> None:
    code = main(["depth-frame", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert f"{FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT}" in out
    assert f"at most {CAPTURE_DEADLINE_S}s" in out
    assert f"{EXPECTED_INTERFACE_DRIVER} stays bound" in out


# --- rgb-frame: one bounded V4L2/MMAP compressed RGB frame (MJPG) ------------
#
# Same engine as depth-frame - one CaptureProfile is the whole difference - so
# this block mirrors the depth-frame block above: dry by default, double-gated
# live mode, a closed and disjoint flag surface, and the same "opens nothing"
# proof via an opener that fails the test if it is ever reached.

from scanner3d.jpeg_inspect import MAX_INSPECT_BYTES  # noqa: E402
from scanner3d.rgb_frame import (  # noqa: E402
    RGB_ARTIFACT_NAME,
    RGB_CAPTURE_DEADLINE_S,
    RGB_DEFAULT_OUTPUT_DIR,
    RGB_DESCRIPTOR_FRAME_RATE_HZ,
    RGB_EXECUTE_FLAG,
    RGB_FOURCC,
    RGB_FRAME_HEIGHT,
    RGB_FRAME_WIDTH,
    RGB_MAX_BUFFER_BYTES,
    RGB_MAX_RAW_ARTIFACT_BYTES,
    RGB_MAX_TOTAL_MAPPED_BYTES,
    RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES,
    RGB_PROFILE,
    RGB_REQUESTED_BUFFERS,
    V4l2RgbFrameBackend,
)

#: The exact, closed set of option strings the rgb-frame subparser may ever
#: expose. No format, fourcc, size, fps, buffer, timeout, retry, device/node
#: or quality/control flag belongs here: the fixed profile in
#: scanner3d.rgb_frame is the only thing that may ever shape this capture.
_RGB_FRAME_EXPECTED_OPTIONS = frozenset(
    {
        "-h",
        "--help",
        "--sysfs-root",
        "--dev-root",
        "--json",
        "--serial",
        RGB_EXECUTE_FLAG,
        "--output-dir",
        "--json-out",
        "--session-dir",
    }
)

#: The hardware-shaping surface this command must never expose - the format,
#: the size, the frame rate, the buffer count, the deadline/retry policy, the
#: device/node and any encoder knob are all fixed at import time in
#: scanner3d.rgb_frame; none of them is a flag.
_RGB_FRAME_FORBIDDEN_FLAGS = (
    "--format",
    "--fourcc",
    "--width",
    "--height",
    "--size",
    "--fps",
    "--frame-rate",
    "--buffers",
    "--buffer-count",
    "--timeout",
    "--deadline",
    "--retries",
    "--retry",
    "--device",
    "--node",
    "--video-node",
    "--jpeg-quality",
    "--quality",
    "--control",
)


def _named_subparser(name: str) -> argparse.ArgumentParser:
    """One live subparser, pulled from the real ``build_parser()``."""
    parser = cli_module.build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    result = subparsers_action.choices[name]
    assert isinstance(result, argparse.ArgumentParser)
    return result


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    strings: set[str] = set()
    for action in parser._actions:
        strings.update(action.option_strings)
    return strings


def _patch_rgb_frame_backend(monkeypatch, **injected) -> None:
    """Make the CLI build an rgb-frame backend whose syscalls are the injected ones."""

    def factory(**kwargs) -> V4l2RgbFrameBackend:
        return V4l2RgbFrameBackend(**injected, **kwargs)

    monkeypatch.setattr(cli_module, "V4l2RgbFrameBackend", factory)


def test_rgb_frame_dry_run_opens_nothing_and_is_not_attempted(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """No --execute-rgb-frame-capture: the node is resolved from sysfs, never opened."""
    _patch_rgb_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["rgb-frame", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Result: not_attempted" in out
    assert RGB_EXECUTE_FLAG in out


def test_rgb_frame_dry_run_never_calls_backend_run(healthy: SysfsBuilder, monkeypatch) -> None:
    """Belt and braces: the dry path must never reach ``V4l2RgbFrameBackend.run``."""

    def _exploding_run(self, serial: str, output_dir: Path = RGB_DEFAULT_OUTPUT_DIR) -> None:
        raise AssertionError("V4l2RgbFrameBackend.run was called during a dry run")

    monkeypatch.setattr(V4l2RgbFrameBackend, "run", _exploding_run)
    code = main(["rgb-frame", *_roots(healthy)])
    assert code == 0


def test_rgb_frame_execute_without_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    """The stream-starting mode refuses to guess a target, just like depth-frame."""
    _patch_rgb_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(["rgb-frame", RGB_EXECUTE_FLAG, *_roots(healthy)])
    err = capsys.readouterr().err
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert "--serial" in err


def test_rgb_frame_execute_wrong_serial_is_refused_and_opens_nothing(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    # The refusal only applies when the operator NAMED a device. Configure
    # one, or there is nothing for the guard to refuse against.
    monkeypatch.setenv("SCANNER3D_SERIAL", AUTHORIZED_SERIAL)
    """The live path is authorized for one physical unit - the same one depth-frame is,
    since both profiles share the one engine-wide ``AUTHORIZED_SERIAL``."""
    assert REAL_SERIAL == DEPTH_FRAME_AUTHORIZED_SERIAL
    _patch_rgb_frame_backend(monkeypatch, opener=ExplodingOpener())
    code = main(
        [
            "rgb-frame",
            RGB_EXECUTE_FLAG,
            "--serial",
            "SOMEOTHERSERIAL0000",
            "--json",
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED] == 10
    assert payload["status"] == "refused"
    assert DEPTH_FRAME_AUTHORIZED_SERIAL in payload["detail"]


@pytest.mark.parametrize("flag", _RGB_FRAME_FORBIDDEN_FLAGS)
def test_rgb_frame_parser_rejects_hardware_shaping_flags(
    healthy: SysfsBuilder, flag: str, capsys
) -> None:
    """None of these is a flag: the mode, ceilings, deadline, retries, node resolution and
    JPEG handling are all fixed at import time in scanner3d.rgb_frame."""
    with pytest.raises(SystemExit) as excinfo:
        main(["rgb-frame", flag, *_roots(healthy)])
    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err


def test_rgb_frame_closed_cli_surface_has_no_hardware_shaping_knobs(
    healthy: SysfsBuilder,
) -> None:
    """Belt and braces on the parametrized rejection above: the *entire* option surface
    is the fixed set, not just the specific forbidden names."""
    assert _option_strings(_named_subparser("rgb-frame")) == set(_RGB_FRAME_EXPECTED_OPTIONS)


def test_depth_frame_rejects_the_rgb_execute_flag(healthy: SysfsBuilder, capsys) -> None:
    """No abbreviation or accident may arm depth-frame's live path with rgb-frame's opt-in."""
    with pytest.raises(SystemExit) as excinfo:
        main(["depth-frame", RGB_EXECUTE_FLAG, *_roots(healthy)])
    assert excinfo.value.code == 2
    assert RGB_EXECUTE_FLAG in capsys.readouterr().err


def test_rgb_frame_rejects_the_depth_execute_flag(healthy: SysfsBuilder, capsys) -> None:
    """Nor may the reverse happen: rgb-frame must not accept depth-frame's opt-in."""
    with pytest.raises(SystemExit) as excinfo:
        main(["rgb-frame", "--execute-depth-frame-capture", *_roots(healthy)])
    assert excinfo.value.code == 2
    assert "--execute-depth-frame-capture" in capsys.readouterr().err


def test_execute_flags_are_disjoint_between_depth_frame_and_rgb_frame() -> None:
    """Each command's option surface must contain only its own opt-in, never the other's."""
    depth_options = _option_strings(_named_subparser("depth-frame"))
    rgb_options = _option_strings(_named_subparser("rgb-frame"))
    assert "--execute-depth-frame-capture" in depth_options
    assert RGB_EXECUTE_FLAG in rgb_options
    assert RGB_EXECUTE_FLAG not in depth_options
    assert "--execute-depth-frame-capture" not in rgb_options


def test_rgb_frame_json_dry_run_reports_the_fixed_plan(healthy: SysfsBuilder, capsys) -> None:
    code = main(["rgb-frame", "--json", *_roots(healthy)])
    raw = capsys.readouterr().out
    payload = json.loads(raw)  # must round-trip through json.loads
    assert code == 0
    assert payload["status"] == "not_attempted"
    plan = payload["plan"]
    assert plan == RGB_PROFILE.plan_as_dict()
    assert plan["fourcc"] == RGB_FOURCC == "MJPG"
    assert plan["width"] == RGB_FRAME_WIDTH == 1280
    assert plan["height"] == RGB_FRAME_HEIGHT == 800
    assert plan["descriptor_frame_rate_hz"] == RGB_DESCRIPTOR_FRAME_RATE_HZ == 30.0
    assert plan["frame_rate_measured"] is False
    assert plan["requested_buffers"] == RGB_REQUESTED_BUFFERS == 4
    assert plan["max_buffer_bytes"] == RGB_MAX_BUFFER_BYTES == 8192000
    assert plan["max_total_mapped_bytes"] == RGB_MAX_TOTAL_MAPPED_BYTES == 32768000
    assert plan["max_raw_artifact_bytes"] == RGB_MAX_RAW_ARTIFACT_BYTES == 8192000
    # The plan quotes both numbers and never conflates them: what the device
    # advertises, and what the driver was measured to allocate.
    assert plan["descriptor_max_video_frame_buffer_bytes"] == 2048000
    assert plan["measured_driver_sizeimage_bytes"] == RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES == 8192000
    # The inspector is handed this profile's artifact ceiling, not the module
    # default, and the plan says so rather than quoting the default as if it
    # were the ceiling in force.
    assert plan["jpeg_inspect_ceiling_bytes"] == RGB_MAX_RAW_ARTIFACT_BYTES == 8192000
    assert plan["jpeg_inspect_module_default_ceiling_bytes"] == MAX_INSPECT_BYTES == 2097152
    assert plan["capture_deadline_s"] == RGB_CAPTURE_DEADLINE_S == 8.0
    assert plan["artifact_name"] == RGB_ARTIFACT_NAME == "rgb-frame.jpg"
    assert plan["max_dequeues"] == 1
    assert plan["retries"] == 0


def test_rgb_frame_json_out_matches_stdout(healthy: SysfsBuilder, tmp_path: Path, capsys) -> None:
    out = tmp_path / "rgb_frame.json"
    main(["rgb-frame", "--json", "--json-out", str(out), *_roots(healthy)])
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_rgb_frame_output_dir_is_accepted_and_reported(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    """--output-dir shapes only local evidence, never the hardware, and is echoed back."""
    custom = tmp_path / "custom-rgb-frame-out"
    code = main(["rgb-frame", "--json", "--output-dir", str(custom), *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["artifacts"]["directory"] == str(custom)


def test_rgb_frame_default_output_dir_is_reported_without_output_dir_flag(
    healthy: SysfsBuilder, capsys
) -> None:
    code = main(["rgb-frame", "--json", *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert str(RGB_DEFAULT_OUTPUT_DIR) == "/tmp/scanner3d-rgb-frame"
    assert payload["artifacts"]["directory"] == str(RGB_DEFAULT_OUTPUT_DIR)


def test_rgb_frame_output_dir_and_json_out_do_not_open_a_node(
    healthy: SysfsBuilder, tmp_path: Path, monkeypatch, capsys
) -> None:
    """--output-dir and --json-out only shape local evidence; in dry mode neither
    reaches the device."""
    _patch_rgb_frame_backend(monkeypatch, opener=ExplodingOpener())
    out = tmp_path / "rgb_frame.json"
    custom = tmp_path / "custom-rgb-frame-out"
    code = main(
        [
            "rgb-frame",
            "--json",
            "--output-dir",
            str(custom),
            "--json-out",
            str(out),
            *_roots(healthy),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text())["artifacts"]["directory"] == str(custom)


def test_rgb_frame_human_output_describes_the_fixed_dry_run_truthfully(
    healthy: SysfsBuilder, capsys
) -> None:
    code = main(["rgb-frame", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert f"{RGB_FOURCC!r} {RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}" in out
    assert f"at most {RGB_CAPTURE_DEADLINE_S}s" in out
    assert f"{RGB_MAX_BUFFER_BYTES} B per buffer, {RGB_MAX_TOTAL_MAPPED_BYTES} B total" in out
    assert (
        f"{RGB_DESCRIPTOR_FRAME_RATE_HZ} fps default "
        "(descriptor value; no S_PARM was issued and no interval was timed)"
    ) in out


def test_rgb_frame_help_names_the_fixed_profile_honestly(capsys) -> None:
    """The static --help text is part of the "dry" surface too: it must name the fixed
    mode, ceilings, artifact and opt-in flag truthfully, and must not promise a knob this
    profile does not have."""
    with pytest.raises(SystemExit) as excinfo:
        main(["rgb-frame", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # argparse wraps the description to the terminal width, which under pytest's
    # non-tty capture can fall inside a phrase this test checks for; collapsing
    # whitespace makes the assertions robust to exactly where that happens.
    normalized = " ".join(out.split())
    assert f"{RGB_FOURCC!r} {RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}" in normalized
    assert RGB_ARTIFACT_NAME in normalized
    assert RGB_EXECUTE_FLAG in normalized
    assert "no VIDIOC_S_PARM is issued" in normalized
    assert "never claimed as measured" in normalized
    for flag in _RGB_FRAME_FORBIDDEN_FLAGS:
        assert flag not in normalized


def test_rgb_frame_disclaimer_makes_no_pixel_or_colour_claims(
    healthy: SysfsBuilder, capsys
) -> None:
    """The report's closing disclaimer, in the profile's own words, must survive verbatim
    into the human output and must not be softened."""
    code = main(["rgb-frame", *_roots(healthy)])
    out = capsys.readouterr().out
    assert code == 0
    assert RGB_PROFILE.disclaimer in out
    assert "no colour accuracy was claimed" in out
    assert "no decode" in out
    assert "white balance" in out
    assert "calibration applied" in out
    assert "no synchronisation with the depth camera was established or attempted" in out


def test_rgb_controls_dry_run_never_reaches_live_backend(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    spy = _RgbControlsSpy()
    factory_calls: list[tuple[Path, Path]] = []

    def factory(sysfs_root: Path, dev_root: Path) -> _RgbControlsSpy:
        factory_calls.append((sysfs_root, dev_root))
        return spy

    monkeypatch.setattr(cli_module, "rgb_controls_backend_factory", factory)
    code = main(["rgb-controls", "--json", *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert factory_calls == [(healthy.sysfs, healthy.dev)]
    assert spy.dry_calls == [None]
    assert spy.run_calls == []
    assert payload["status"] == "not_attempted"


@pytest.mark.parametrize("serial", ["wrong-serial", ""])
def test_rgb_controls_serial_gate_refuses_before_open_or_ioctl(
    healthy: SysfsBuilder, serial: str, monkeypatch, capsys
) -> None:
    spy = _RgbControlsSpy()
    monkeypatch.setattr(cli_module, "rgb_controls_backend_factory", lambda *_roots: spy)
    argv = [
        "rgb-controls",
        "--json",
        "--read-rgb-controls",
        *([] if serial == "" else ["--serial", serial]),
        *_roots(healthy),
    ]
    code = main(argv)
    payload = json.loads(capsys.readouterr().out)
    assert code == 10
    assert payload["status"] == "refused"
    assert spy.run_calls == [serial]
    assert spy.dry_calls == []


def test_rgb_controls_authorized_serial_without_exact_opt_in_stays_dry(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    spy = _RgbControlsSpy()
    monkeypatch.setattr(cli_module, "rgb_controls_backend_factory", lambda *_roots: spy)
    code = main(["rgb-controls", "--json", "--serial", REAL_SERIAL, *_roots(healthy)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "not_attempted"
    assert spy.dry_calls == [REAL_SERIAL]
    assert spy.run_calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["rgb-controls", "--read-rgb"],
        ["rgb-controls", "--control-id", "123"],
        ["rgb-controls", "--node", "/dev/video2"],
        ["rgb-controls", "--format", "MJPG"],
        ["rgb-controls", "--write"],
        ["rgb-controls", "--set", "gain=1"],
    ],
)
def test_rgb_controls_parser_rejects_abbreviation_and_hardware_knobs(
    argv: list[str], monkeypatch
) -> None:
    monkeypatch.setattr(
        cli_module,
        "rgb_controls_backend_factory",
        lambda *_roots: pytest.fail("factory must not be reached by parser refusal"),
    )
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_rgb_controls_json_and_human_render_the_same_typed_result(
    healthy: SysfsBuilder, monkeypatch, capsys
) -> None:
    spy = _RgbControlsSpy()
    monkeypatch.setattr(cli_module, "rgb_controls_backend_factory", lambda *_roots: spy)
    json_code = main(["rgb-controls", "--json", *_roots(healthy)])
    json_payload = json.loads(capsys.readouterr().out)
    human_code = main(["rgb-controls", *_roots(healthy)])
    human_output = capsys.readouterr().out
    assert json_code == human_code == 0
    assert json_payload["status"] == "not_attempted"
    assert json_payload["detail"] == "spy dry run"
    assert "Status: not_attempted - spy dry run" in human_output
    for control in json_payload["plan"]["controls"]:
        assert control["name"] in human_output
    assert spy.dry_calls == [None, None]
    assert spy.run_calls == []


# --- --session-dir: wiring the session manifest into the capture path --------
#
# A capture command that opts in with --session-dir must create or continue a
# session, write its raw bytes through the atomic artifact store, and record
# the stage's success or failure - never inferring success from the absence
# of an exception. These tests exercise that wiring directly through the CLI,
# entirely offline, using the same FakeKernel every other depth-frame/rgb-frame
# lifecycle test in this file uses.

from scanner3d.session import (  # noqa: E402
    MANIFEST_NAME,
    SessionState,
    SessionStore,
    StageState,
    read_manifest,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, RAW_ARTIFACT_NAME  # noqa: E402
from tests.test_jpeg_inspect import baseline_jpeg  # noqa: E402


class _AcceptedFinalizer:
    """A blue-light finalizer that always reports its transfer accepted.

    Session-recording tests capture a depth frame, which always raises the
    finalizer obligation; this keeps that obligation out of the way so each
    test's assertions are about session recording, not illumination.
    """

    def run(self, _serial: str) -> IlluminationShutdownOutcome:
        return IlluminationShutdownOutcome(
            True,
            True,
            ShutdownStatus.COMMAND_ACCEPTED,
            "synthetic off accepted",
            transfer=ShutdownTransferRecord(60, True, "accepted", 0.0),
        )


def _execute_depth_frame_args(
    healthy: SysfsBuilder, *, output_dir: Path, session_dir: Path, extra: list[str] | None = None
) -> list[str]:
    return [
        "depth-frame",
        "--json",
        "--execute-depth-frame-capture",
        "--serial",
        REAL_SERIAL,
        "--output-dir",
        str(output_dir),
        "--session-dir",
        str(session_dir),
        *(extra or []),
        *_roots(healthy),
    ]


def test_depth_frame_execute_records_a_verified_session_stage(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path, capsys
) -> None:
    """A successful capture creates the session and records a complete stage
    whose artifact is the exact bytes the capture wrote, re-hashed from disk."""
    kernel = FakeKernel()
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )
    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: _AcceptedFinalizer()
    )
    output_dir = tmp_path / "depth-out"
    session_dir = tmp_path / "session"
    code = main(_execute_depth_frame_args(healthy, output_dir=output_dir, session_dir=session_dir))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "complete"
    assert payload["succeeded"] is True

    manifest = read_manifest(session_dir / MANIFEST_NAME)
    assert manifest.state is SessionState.RUNNING
    assert manifest.device is not None
    assert manifest.device.serial == REAL_SERIAL
    stage = manifest.stage("depth_frame")
    assert stage is not None
    assert stage.state is StageState.COMPLETE
    assert stage.error == ""
    assert len(stage.artifacts) == 1
    artifact = stage.artifacts[0]
    assert artifact.path.startswith("raw/depth_frame-")
    assert artifact.immutable is True
    assert stage.counters["delivered"] == 1
    assert stage.counters["bytes"] == artifact.bytes

    on_disk = (session_dir / artifact.path).read_bytes()
    captured_raw = (output_dir / RAW_ARTIFACT_NAME).read_bytes()
    assert on_disk == captured_raw
    assert artifact.sha256 == hashlib.sha256(captured_raw).hexdigest()
    assert artifact.bytes == len(captured_raw)


def test_depth_frame_dry_run_with_session_dir_touches_no_session(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path
) -> None:
    """--session-dir only takes effect on the execute path; a dry run creates
    nothing, since a dry run writes no frame there is nothing to record."""
    _patch_depth_frame_backend(monkeypatch, opener=ExplodingOpener())
    session_dir = tmp_path / "session"
    code = main(["depth-frame", "--session-dir", str(session_dir), *_roots(healthy)])
    assert code == 0
    assert not session_dir.exists()


def test_depth_frame_execute_second_capture_restarts_and_accumulates_the_stage(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path
) -> None:
    """Two captures into the same --session-dir append to the same stage,
    each preserved under its own content-addressed artifact name."""
    session_dir = tmp_path / "session"

    kernel_one = FakeKernel(frame_byte=0xAB)
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel_one.opener,
        ioctl_fn=kernel_one.ioctl,
        mmap_fn=kernel_one.mmap,
        poll_fn=kernel_one.poll,
        closer=kernel_one.closer,
    )
    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: _AcceptedFinalizer()
    )
    first_out = tmp_path / "depth-out-1"
    code_one = main(
        _execute_depth_frame_args(healthy, output_dir=first_out, session_dir=session_dir)
    )
    assert code_one == 0

    kernel_two = FakeKernel(frame_byte=0xCD)
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel_two.opener,
        ioctl_fn=kernel_two.ioctl,
        mmap_fn=kernel_two.mmap,
        poll_fn=kernel_two.poll,
        closer=kernel_two.closer,
    )
    second_out = tmp_path / "depth-out-2"
    code_two = main(
        _execute_depth_frame_args(healthy, output_dir=second_out, session_dir=session_dir)
    )
    assert code_two == 0

    manifest = read_manifest(session_dir / MANIFEST_NAME)
    stage = manifest.stage("depth_frame")
    assert stage is not None
    assert stage.state is StageState.COMPLETE
    assert len(stage.artifacts) == 2
    assert stage.artifacts[0].sha256 != stage.artifacts[1].sha256
    assert stage.artifacts[0].path != stage.artifacts[1].path


def test_depth_frame_execute_records_a_failed_stage_on_a_short_frame(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path, capsys
) -> None:
    """A capture that does not succeed is recorded as a failed stage with no
    artifact - never inferred complete because nothing raised."""
    kernel = FakeKernel(bytesused=EXPECTED_FRAME_BYTES - 2)
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )
    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: _AcceptedFinalizer()
    )
    output_dir = tmp_path / "depth-out"
    session_dir = tmp_path / "session"
    code = main(_execute_depth_frame_args(healthy, output_dir=output_dir, session_dir=session_dir))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code != 0
    assert payload["status"] == "short_frame"
    # The underlying capture failure is not also reported as a session
    # problem: it was faithfully recorded, which is a working session, not a
    # broken one.
    assert "session recording did not complete" not in captured.err

    manifest = read_manifest(session_dir / MANIFEST_NAME)
    stage = manifest.stage("depth_frame")
    assert stage is not None
    assert stage.state is StageState.FAILED
    assert stage.error != ""
    assert stage.artifacts == ()


def test_depth_frame_execute_reports_a_session_problem_for_a_terminal_session(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path, capsys
) -> None:
    """A --session-dir that already names a finished session cannot silently
    swallow a real capture: the CLI must say recording failed and must not
    report success."""
    session_dir = tmp_path / "session"
    SessionStore.create(tmp_path, "session", software_version="scanner3d-test").finish(
        SessionState.COMPLETE
    )
    assert session_dir.is_dir()

    kernel = FakeKernel()
    _patch_depth_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )
    monkeypatch.setattr(
        cli_module, "illumination_backend_factory", lambda _sysfs, _dev: _AcceptedFinalizer()
    )
    output_dir = tmp_path / "depth-out"
    code = main(_execute_depth_frame_args(healthy, output_dir=output_dir, session_dir=session_dir))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    # The capture itself still succeeded and is still reported as such...
    assert payload["status"] == "complete"
    assert payload["succeeded"] is True
    # ...but the CLI's own exit code must not claim the run is clean when its
    # evidence could not be filed into the session the caller asked for.
    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "session recording did not complete" in captured.err

    # The pre-existing completed session must not have been silently mutated.
    manifest = read_manifest(session_dir / MANIFEST_NAME)
    assert manifest.state is SessionState.COMPLETE
    assert manifest.stages == ()


def test_rgb_frame_execute_records_a_verified_session_stage(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path, capsys
) -> None:
    """rgb-frame shares the same --session-dir wiring through the same engine."""
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    _patch_rgb_frame_backend(
        monkeypatch,
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
    )
    output_dir = tmp_path / "rgb-out"
    session_dir = tmp_path / "session"
    code = main(
        [
            "rgb-frame",
            "--json",
            RGB_EXECUTE_FLAG,
            "--serial",
            REAL_SERIAL,
            "--output-dir",
            str(output_dir),
            "--session-dir",
            str(session_dir),
            *_roots(healthy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "complete"
    assert payload["succeeded"] is True

    manifest = read_manifest(session_dir / MANIFEST_NAME)
    stage = manifest.stage("rgb_frame")
    assert stage is not None
    assert stage.state is StageState.COMPLETE
    assert len(stage.artifacts) == 1
    artifact = stage.artifacts[0]
    assert artifact.path.startswith("raw/rgb_frame-")
    assert artifact.path.endswith(".jpg")
    on_disk = (session_dir / artifact.path).read_bytes()
    assert on_disk == (output_dir / RGB_ARTIFACT_NAME).read_bytes()
