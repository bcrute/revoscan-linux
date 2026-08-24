"""Doctor report tests: status classification and honest 'not run' handling."""

from __future__ import annotations

import json

from scanner3d.doctor import run_doctor
from scanner3d.errors import DoctorStatus
from scanner3d.reference import ProbeResult
from tests.conftest import REAL_SERIAL, SysfsBuilder


def _run(b: SysfsBuilder, **kw: object):
    return run_doctor(sysfs_root=b.sysfs, dev_root=b.dev, **kw)  # type: ignore[arg-type]


def _check(report, name: str):
    return next(c for c in report.checks if c.name == name)


def _probe(data: dict | None, **kw: object) -> ProbeResult:
    defaults = dict(
        ran=True,
        command=["probe"],
        exit_code=0,
        timed_out=False,
        elapsed_s=1.0,
        stdout="",
        stderr="",
        data=data,
        error=None,
        sdk=None,
    )
    defaults.update(kw)
    return ProbeResult(**defaults)  # type: ignore[arg-type]


FULL_PROBE_DATA = {
    "frames": {
        "depth": {"count": 3, "width": 640, "height": 400},
        "rgb": {"count": 2, "width": 1280, "height": 800},
    },
    "intrinsics": {"depth": {"fx": 1.0}, "rgb": {"fx": 2.0}},
    "extrinsics": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [0, 0, 0]},
    "depth_scale": 0.1,
}


def test_absent_device_reports_device_absent(empty_sysfs: SysfsBuilder) -> None:
    report = _run(empty_sysfs)
    assert report.status is DoctorStatus.DEVICE_ABSENT
    assert _check(report, "device_present").status is DoctorStatus.DEVICE_ABSENT
    # Downstream checks must be marked not-run, never passing.
    for name in ("device_identity", "camera_roles", "permissions", "frame_delivery"):
        check = _check(report, name)
        assert not check.ran
        assert check.detail.startswith("not run:")


def test_healthy_device_without_probe_is_ok_with_skipped_hardware_checks(
    healthy: SysfsBuilder,
) -> None:
    report = _run(healthy)
    assert report.status is DoctorStatus.OK
    assert _check(report, "device_identity").status is DoctorStatus.OK
    assert _check(report, "usb_topology").data["speed_mbps"] == 480
    assert not _check(report, "frame_delivery").ran
    assert not _check(report, "calibration_metadata").ran


def test_permission_failure_on_usbfs(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini()
    (builder.dev / "bus" / "usb" / "001" / "024").chmod(0o000)
    report = _run(builder)
    assert report.status is DoctorStatus.PERMISSION_DENIED
    assert "not read-write" in _check(report, "permissions").detail


def test_permission_failure_on_capture_node(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini()
    (builder.dev / "video0").chmod(0o000)
    report = _run(builder)
    assert report.status is DoctorStatus.PERMISSION_DENIED
    assert "/dev/video0" in _check(report, "permissions").detail


def test_inaccessible_hidraw_alone_does_not_fail(builder: SysfsBuilder) -> None:
    """hidraw is 0600 root:root on a stock host; that is reported, not fatal."""
    builder.add_healthy_mini()
    (builder.dev / "hidraw10").chmod(0o000)  # stand in for root:root ownership
    report = _run(builder)
    check = _check(report, "permissions")
    assert check.status is DoctorStatus.OK
    assert check.data["hid_usable_count"] == 0
    assert "hidraw" in check.detail


def test_unexpected_manufacturer_is_flagged(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini(manufacturer="Definitely Not Chishine3d")
    report = _run(builder)
    assert report.status is DoctorStatus.UNEXPECTED_IDENTITY
    assert "manufacturer" in _check(report, "device_identity").detail


def test_missing_rgb_role_is_flagged(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base, 0, interface_string=f"DepthCam{REAL_SERIAL}", video_nodes=(("video0", 0),)
    )
    report = _run(builder)
    assert report.status is DoctorStatus.UNEXPECTED_IDENTITY
    assert "rgb" in _check(report, "camera_roles").detail


def test_role_present_but_no_capture_node_is_flagged(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(base, 0, interface_string=f"DepthCam{REAL_SERIAL}")
    builder.add_interface(base, 2, interface_string=f"RGBCam{REAL_SERIAL}")
    report = _run(builder)
    assert report.status is DoctorStatus.UNEXPECTED_IDENTITY
    assert "no capture node" in _check(report, "camera_roles").detail


def test_unsupported_mode_when_descriptors_lack_required_formats(
    builder: SysfsBuilder,
) -> None:
    builder.add_healthy_mini(descriptors=b"")
    report = _run(builder)
    assert report.status is DoctorStatus.UNSUPPORTED_MODE
    assert _check(report, "advertised_modes").status is DoctorStatus.UNSUPPORTED_MODE


def test_probe_success_marks_frames_and_calibration_ok(healthy: SysfsBuilder) -> None:
    report = _run(healthy, probe=_probe(FULL_PROBE_DATA))
    assert report.status is DoctorStatus.OK
    assert _check(report, "frame_delivery").status is DoctorStatus.OK
    assert _check(report, "calibration_metadata").status is DoctorStatus.OK


def test_probe_timeout_reports_no_frames(healthy: SysfsBuilder) -> None:
    report = _run(healthy, probe=_probe(None, timed_out=True, exit_code=-9, elapsed_s=12.0))
    assert report.status is DoctorStatus.NO_FRAMES
    assert "timed out" in _check(report, "frame_delivery").detail


def test_probe_connect_failure_reports_no_frames_with_error_detail(
    healthy: SysfsBuilder,
) -> None:
    data = {
        "connect": False,
        "frames": {"depth": {"count": 0}, "rgb": {"count": 0}},
        "errors": [{"stage": "connect", "code": 8, "code_name": "ERROR_NOT_SUPPORT"}],
    }
    report = _run(healthy, probe=_probe(data, exit_code=3, error="probe exited 3"))
    assert report.status is DoctorStatus.NO_FRAMES
    check = _check(report, "frame_delivery")
    assert check.detail == "probe exited 3"
    assert check.data["probe_errors"][0]["code_name"] == "ERROR_NOT_SUPPORT"


def test_depth_frames_without_rgb_is_still_no_frames(healthy: SysfsBuilder) -> None:
    data = dict(FULL_PROBE_DATA, frames={"depth": {"count": 1}, "rgb": {"count": 0}})
    report = _run(healthy, probe=_probe(data))
    assert _check(report, "frame_delivery").status is DoctorStatus.NO_FRAMES


def test_partial_calibration_is_reported_as_missing(healthy: SysfsBuilder) -> None:
    data = dict(FULL_PROBE_DATA, depth_scale=None, extrinsics=None)
    report = _run(healthy, probe=_probe(data))
    check = _check(report, "calibration_metadata")
    assert check.status is DoctorStatus.NO_FRAMES
    assert "extrinsics" in check.detail and "depth_scale" in check.detail


def test_worst_status_wins_over_ok_checks(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini(manufacturer="wrong")
    (builder.dev / "video0").chmod(0o000)
    report = _run(builder)
    # unexpected_identity outranks permission_denied.
    assert report.status is DoctorStatus.UNEXPECTED_IDENTITY


def test_report_round_trips_through_json(healthy: SysfsBuilder) -> None:
    report = _run(healthy, probe=_probe(FULL_PROBE_DATA))
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["status"] == "ok"
    assert payload["schema_version"] == 1
    assert payload["devices"][0]["serial"] == REAL_SERIAL
    assert payload["reference_probe"]["ran"] is True
