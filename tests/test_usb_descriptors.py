"""Descriptor-parser tests, including malformed and truncated blobs."""

from __future__ import annotations

import pytest

from scanner3d.usb_descriptors import VideoControlEntity, iter_descriptors, parse
from tests.conftest import descriptor_blob

# The GUID carried by the recorded MINI's extension units, decoded by hand
# from the fixture bytes ``6d1a21c2 7e78 4a3c ac06 7e434b676f9a`` (bytes_le).
_EXTENSION_UNIT_GUID = "C2211A6D-787E-3C4A-AC06-7E434B676F9A"


def test_parses_recorded_mini_descriptor() -> None:
    parsed = parse(descriptor_blob())
    assert not parsed.truncated
    assert [i.number for i in parsed.interfaces] == [0, 1, 2, 3, 4]
    assert [(a.first_interface, a.interface_count) for a in parsed.associations] == [(0, 2), (2, 2)]


def test_depth_function_advertises_y16_and_yuy2() -> None:
    formats = parse(descriptor_blob()).formats_for_function(0)
    by_fourcc = {f.fourcc: f for f in formats}
    assert set(by_fourcc) == {"YUY2", "Y16"}
    y16_modes = {(m.width, m.height) for m in by_fourcc["Y16"].modes}
    assert y16_modes == {(640, 400), (1280, 800), (640, 800)}
    assert by_fourcc["Y16"].bits_per_pixel == 16
    for mode in by_fourcc["Y16"].modes:
        assert mode.frame_rates_hz == (5.0, 10.0, 15.0)
        assert mode.default_frame_rate_hz == 15.0


def test_rgb_function_advertises_mjpeg_only() -> None:
    formats = parse(descriptor_blob()).formats_for_function(2)
    assert [f.fourcc for f in formats] == ["MJPG"]
    (mode,) = formats[0].modes
    assert (mode.width, mode.height) == (1280, 800)
    assert mode.frame_rates_hz == (10.0, 15.0, 20.0, 30.0)
    assert mode.default_frame_rate_hz == 30.0


def test_videocontrol_subtypes_are_not_mistaken_for_formats() -> None:
    """VC_EXTENSION_UNIT (0x06) collides numerically with VS_FORMAT_MJPEG."""
    parsed = parse(descriptor_blob())
    control_interfaces = [i for i in parsed.interfaces if i.number in (0, 2)]
    assert control_interfaces
    for iface in control_interfaces:
        assert iface.formats == ()


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x00",
        b"\x01\x04",  # length < 2 header
        b"\xff\x04\x00",  # length runs past the end
        bytes(range(64)),  # arbitrary noise
    ],
)
def test_malformed_blobs_do_not_raise(blob: bytes) -> None:
    parsed = parse(blob)
    assert isinstance(parsed.interfaces, tuple)


def test_truncated_blob_is_flagged_and_yields_partial_results() -> None:
    full = descriptor_blob()
    parsed = parse(full[:120])
    assert parsed.truncated
    # The first interfaces still parse; the later ones are simply absent.
    assert [i.number for i in parsed.interfaces] == [0]


def test_iter_descriptors_stops_at_first_bad_record() -> None:
    good = b"\x04\x04\x00\x00"
    records = iter_descriptors(good + b"\x00\x00\x00")
    assert records == [(0x04, good)]


def test_frame_descriptor_shorter_than_header_is_skipped() -> None:
    """A format with a truncated frame descriptor yields a format, no modes."""
    iad = bytes([8, 0x0B, 0, 2, 0x0E, 3, 0, 0])
    vs_iface = bytes([9, 0x04, 0, 0, 1, 0x0E, 0x02, 0, 0])
    guid = bytes.fromhex("59313620") + bytes.fromhex("00001000800000aa00389b71")
    fmt = bytes([27, 0x24, 0x04, 1, 1]) + guid + bytes([16, 0, 0, 0, 0, 0])
    short_frame = bytes([10, 0x24, 0x05, 1, 0, 0x80, 0x02, 0x90, 0x01, 0])
    parsed = parse(iad + vs_iface + fmt + short_frame)
    (iface,) = parsed.interfaces
    assert [f.fourcc for f in iface.formats] == ["Y16"]
    assert iface.formats[0].modes == ()


def test_unknown_guid_falls_back_to_uuid_string() -> None:
    vs_iface = bytes([9, 0x04, 0, 0, 1, 0x0E, 0x02, 0, 0])
    guid = bytes(range(16))
    fmt = bytes([27, 0x24, 0x04, 1, 0]) + guid + bytes([16, 0, 0, 0, 0, 0])
    parsed = parse(vs_iface + fmt)
    assert parsed.interfaces[0].formats[0].fourcc.count("-") == 4


def _entity_table(entities: tuple[VideoControlEntity, ...]) -> list[tuple[object, ...]]:
    return [
        (
            e.entity_id,
            e.subtype_name,
            e.descriptor_subtype,
            e.guid,
            e.terminal_type,
            e.source_ids,
            e.num_controls,
            e.control_bitmap,
            e.truncated,
            e.association_first_interface,
        )
        for e in entities
    ]


_EXPECTED_VC_ENTITIES = [
    (0, "header", 0x01, None, None, (), None, None, False),
    (1, "input_terminal", 0x02, None, 0x0201, (), None, None, False),
    (2, "processing_unit", 0x05, None, None, (1,), None, "5f1f", False),
    (3, "output_terminal", 0x03, None, 0x0101, (2,), None, None, False),
    (4, "extension_unit", 0x06, _EXTENSION_UNIT_GUID, None, (2,), 9, "ffa201", False),
]


def test_interface_zero_entity_table_matches_hand_decoded_fixture() -> None:
    """Full entity table for VideoControl interface 0 of the real MINI fixture.

    Ground truth decoded by hand from ``tests/fixtures/mini-config-descriptor.bin``:
    header, input terminal (id 1, wTerminalType 0x0201), processing unit
    (id 2, source 1), output terminal (id 3, wTerminalType 0x0101, source 2,
    *not* an extension unit), extension unit (id 4, the GUID above,
    bNumControls 9, source 2, bmControls ``ffa201``).
    """
    parsed = parse(descriptor_blob())
    (iface0,) = [i for i in parsed.interfaces if i.number == 0]
    table = _entity_table(iface0.entities)
    assert [row[:-1] for row in table] == _EXPECTED_VC_ENTITIES
    assert all(row[-1] == 0 for row in table)  # association_first_interface == 0


def test_interface_two_carries_the_same_four_entities() -> None:
    parsed = parse(descriptor_blob())
    (iface2,) = [i for i in parsed.interfaces if i.number == 2]
    table = _entity_table(iface2.entities)
    assert [row[:-1] for row in table] == _EXPECTED_VC_ENTITIES
    assert all(row[-1] == 2 for row in table)  # association_first_interface == 2


def test_extension_units_returns_exactly_the_two_id_four_units() -> None:
    parsed = parse(descriptor_blob())
    units = parsed.extension_units
    assert [(u.interface_number, u.entity_id) for u in units] == [(0, 4), (2, 4)]
    assert all(u.guid == _EXTENSION_UNIT_GUID for u in units)
    assert all(u.is_extension_unit for u in units)


def test_entity_lookup_distinguishes_output_terminal_from_extension_unit() -> None:
    """Entity 3 is an output terminal, not an extension unit -- the whole point."""
    parsed = parse(descriptor_blob())
    output_terminal = parsed.entity(0, 3)
    assert output_terminal is not None
    assert output_terminal.subtype_name == "output_terminal"
    assert output_terminal.terminal_type == 0x0101
    assert output_terminal.source_ids == (2,)
    assert not output_terminal.is_extension_unit

    extension_unit = parsed.entity(0, 4)
    assert extension_unit is not None
    assert extension_unit.subtype_name == "extension_unit"
    assert extension_unit.is_extension_unit
    assert extension_unit.guid == _EXTENSION_UNIT_GUID

    assert parsed.entity(0, 99) is None


def test_entities_property_is_flattened_in_descriptor_order() -> None:
    parsed = parse(descriptor_blob())
    assert [(e.interface_number, e.entity_id) for e in parsed.entities] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]


def test_video_streaming_interfaces_have_no_entities() -> None:
    parsed = parse(descriptor_blob())
    streaming = [i for i in parsed.interfaces if i.number in (1, 3)]
    assert streaming
    for iface in streaming:
        assert iface.entities == ()


def _vc_iface(number: int = 0) -> bytes:
    return bytes([9, 0x04, number, 0, 0, 0x0E, 0x01, 0, 0])


def test_hand_built_short_extension_unit_is_truncated_not_raised() -> None:
    """bLength itself is short: only the unit ID and a partial GUID are present."""
    short_ext = bytes([8, 0x24, 0x06, 4, 0xAA, 0xBB, 0xCC, 0xDD])
    parsed = parse(_vc_iface() + short_ext)
    (iface,) = parsed.interfaces
    (entity,) = iface.entities
    assert entity.entity_id == 4
    assert entity.subtype_name == "extension_unit"
    assert entity.guid is None
    assert entity.num_controls is None
    assert entity.source_ids == ()
    assert entity.control_bitmap is None
    assert entity.truncated is True


def test_extension_unit_declared_source_pins_run_past_record_end() -> None:
    """bNrInPins claims 5 pins but bLength only leaves room for 1."""
    guid = bytes.fromhex("6d1a21c27e784a3cac067e434b676f9a")
    data = bytes([0, 0x24, 0x06, 4]) + guid + bytes([9, 5, 2])
    data = bytes([len(data)]) + data[1:]
    parsed = parse(_vc_iface() + data)
    (iface,) = parsed.interfaces
    (entity,) = iface.entities
    assert entity.entity_id == 4
    assert entity.guid == _EXTENSION_UNIT_GUID
    assert entity.num_controls == 9
    assert entity.source_ids == (2,)
    assert entity.control_bitmap is None
    assert entity.truncated is True


def test_extension_unit_control_bitmap_runs_past_record_end() -> None:
    """bNrInPins is fully present but the declared bmControls is cut short."""
    guid = bytes.fromhex("6d1a21c27e784a3cac067e434b676f9a")
    data = bytes([0, 0x24, 0x06, 4]) + guid + bytes([9, 1, 2, 5, 0xFF, 0xA2])
    data = bytes([len(data)]) + data[1:]
    parsed = parse(_vc_iface() + data)
    (iface,) = parsed.interfaces
    (entity,) = iface.entities
    assert entity.entity_id == 4
    assert entity.guid == _EXTENSION_UNIT_GUID
    assert entity.num_controls == 9
    assert entity.source_ids == (2,)
    assert entity.control_bitmap == "ffa2"
    assert entity.truncated is True


def test_videocontrol_entity_as_dict_round_trips_all_fields() -> None:
    parsed = parse(descriptor_blob())
    entity = parsed.entity(0, 4)
    assert entity is not None
    as_dict = entity.as_dict()
    assert as_dict["entity_id"] == 4
    assert as_dict["guid"] == _EXTENSION_UNIT_GUID
    assert as_dict["source_ids"] == [2]
    assert as_dict["num_controls"] == 9
    assert as_dict["control_bitmap"] == "ffa201"
    assert as_dict["truncated"] is False
