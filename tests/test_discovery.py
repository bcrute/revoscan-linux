"""Discovery tests driven entirely by synthesized sysfs trees."""

from __future__ import annotations

from pathlib import Path

from scanner3d.discovery import check_access, find_scanners
from tests.conftest import REAL_SERIAL, SysfsBuilder


def _find(b: SysfsBuilder) -> tuple:
    return find_scanners(sysfs_root=b.sysfs, dev_root=b.dev)


def test_absent_device_returns_empty(empty_sysfs: SysfsBuilder) -> None:
    assert _find(empty_sysfs) == ()


def test_missing_sysfs_tree_returns_empty(tmp_path: Path) -> None:
    assert find_scanners(sysfs_root=tmp_path / "nope", dev_root=tmp_path / "dev") == ()


def test_other_vendor_device_is_ignored(builder: SysfsBuilder) -> None:
    builder.add_usb_device("1-2", vendor="1d6b", product="0002")
    assert _find(builder) == ()


def test_healthy_device_identity(healthy: SysfsBuilder) -> None:
    (device,) = _find(healthy)
    assert device.vendor_id == 0x2207
    assert device.product_id == 0x110C
    assert device.serial == REAL_SERIAL
    assert device.manufacturer == "Chishine3d"
    assert device.identity_matches
    assert device.speed_mbps == 480
    assert device.usb_speed_label == "480 Mb/s (USB 2.0 high-speed)"
    assert device.usb_path == "1-4.1"
    assert device.port_path == "4.1"


def test_roles_come_from_interface_strings_not_video_numbers(builder: SysfsBuilder) -> None:
    """The same roles must resolve when /dev/videoN numbering is shuffled."""
    base = builder.add_usb_device()
    builder.add_usbfs()
    # Deliberately give RGB the *lower* node numbers and depth the higher ones.
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
        video_nodes=(("video3", 0), ("video4", 1)),
    )
    (device,) = _find(builder)
    depth = device.function("depth")
    rgb = device.function("rgb")
    assert depth is not None and rgb is not None
    assert [n.name for n in depth.capture_nodes] == ["video7"]
    assert [n.name for n in rgb.capture_nodes] == ["video3"]
    assert depth.interface_number == 0
    assert rgb.interface_number == 2


def test_node_index_separates_capture_from_metadata(healthy: SysfsBuilder) -> None:
    (device,) = _find(healthy)
    depth = device.function("depth")
    assert depth is not None
    kinds = {n.name: n.kind for n in depth.video_nodes}
    assert kinds == {"video0": "capture", "video1": "metadata"}


def test_formats_are_attached_to_the_right_role(healthy: SysfsBuilder) -> None:
    (device,) = _find(healthy)
    depth = device.function("depth")
    rgb = device.function("rgb")
    assert depth is not None and rgb is not None
    assert {f.fourcc for f in depth.formats} == {"Y16", "YUY2"}
    assert [f.fourcc for f in rgb.formats] == ["MJPG"]


def test_hid_interface_is_reported(healthy: SysfsBuilder) -> None:
    (device,) = _find(healthy)
    assert [h.name for h in device.hid_nodes] == ["hidraw10"]


def test_missing_usbfs_node_is_reported_not_raised(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini()
    (builder.dev / "bus" / "usb" / "001" / "024").unlink()
    (device,) = _find(builder)
    assert device.usbfs.exists is False
    assert device.usbfs.usable is False


def test_unreadable_device_node_is_reported(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video0", 0),),
        node_mode=0o000,
    )
    (device,) = _find(builder)
    depth = device.function("depth")
    assert depth is not None
    assert depth.capture_nodes[0].access.exists
    assert not depth.capture_nodes[0].access.usable


def test_malformed_sysfs_attributes_degrade_gracefully(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device(speed="not-a-number", bcd_device="")
    (base / "devnum").write_text("banana\n")
    builder.add_interface(base, 0, interface_string=f"DepthCam{REAL_SERIAL}")
    (device,) = _find(builder)
    assert device.speed_mbps is None
    assert device.usb_speed_label is None
    assert device.device_number is None
    # usbfs path degrades to a sentinel rather than crashing.
    assert device.usbfs.exists is False


def test_missing_serial_and_manufacturer_are_none(builder: SysfsBuilder) -> None:
    builder.add_usb_device(serial=None, manufacturer=None)
    (device,) = _find(builder)
    assert device.serial is None
    assert device.manufacturer is None


def test_corrupt_descriptor_blob_is_recorded(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini(descriptors=b"\xff\xff\xff")
    (device,) = _find(builder)
    assert device.descriptors is not None
    assert device.function("depth") is not None
    assert device.function("depth").formats == ()


def test_interface_without_number_is_skipped(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    iface = builder.add_interface(base, 0, interface_string=f"DepthCam{REAL_SERIAL}")
    (iface / "bInterfaceNumber").unlink()
    (device,) = _find(builder)
    assert device.interfaces == ()
    assert device.functions == ()


def test_unknown_interface_string_yields_no_role(builder: SysfsBuilder) -> None:
    base = builder.add_usb_device()
    builder.add_interface(base, 0, interface_string="SomethingElse")
    (device,) = _find(builder)
    assert device.functions == ()


def test_multiple_devices_are_all_returned(builder: SysfsBuilder) -> None:
    builder.add_healthy_mini()
    second = builder.add_usb_device("1-5", devnum="30", serial="OTHERSERIAL")
    builder.add_interface(second, 0, interface_string="DepthCamOTHERSERIAL")
    devices = _find(builder)
    assert {d.serial for d in devices} == {REAL_SERIAL, "OTHERSERIAL"}


def test_check_access_on_missing_path(tmp_path: Path) -> None:
    result = check_access(tmp_path / "absent")
    assert not result.exists and not result.usable
    assert result.owner_uid is None


def test_as_dict_is_json_serializable(healthy: SysfsBuilder) -> None:
    import json

    (device,) = _find(healthy)
    payload = json.loads(json.dumps(device.as_dict()))
    assert payload["serial"] == REAL_SERIAL
    assert payload["vendor_id"] == "2207"
