"""Offline tests for the scanner-012 ``reference/analysis/uvc_stream_state`` helper.

These run against the committed fixtures only: the byte-for-byte MINI
configuration descriptor and the frozen scanner-009/010/011 run JSON. No SDK,
no binary, no device, no network -- so they are part of the default gate.

The expected values below were decoded by hand from
``tests/fixtures/mini-config-descriptor.bin`` before the decoder was written,
and are quoted in ``docs/architecture.md``. Their whole purpose is to make the
numbers that section relies on fail loudly if the decoder drifts.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parent.parent / "reference" / "analysis" / "uvc_stream_state.py"
_spec = importlib.util.spec_from_file_location("uvc_stream_state", _HELPER)
assert _spec is not None
uvc_stream_state = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("uvc_stream_state", uvc_stream_state)
_spec.loader.exec_module(uvc_stream_state)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DESCRIPTOR = FIXTURES / "mini-config-descriptor.bin"
RGB_ATTEMPT = FIXTURES / "scanner-011-rgb-attempt.json"
RGB_FIRST_ATTEMPT = FIXTURES / "scanner-010-rgb-attempt.json"
DEPTH_CAPTURE = FIXTURES / "scanner-009-depth-capture.json"


@pytest.fixture(scope="module")
def config():
    return uvc_stream_state.decode_configuration(DESCRIPTOR.read_bytes())


@pytest.fixture(scope="module")
def rgb_run():
    return uvc_stream_state.summarize_run(json.loads(RGB_ATTEMPT.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def depth_run():
    return uvc_stream_state.summarize_run(json.loads(DEPTH_CAPTURE.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def rgb_plan(config):
    return uvc_stream_state.streaming_plan(config, 3, "MJPG", 1280, 800)


@pytest.fixture(scope="module")
def depth_plan(config):
    return uvc_stream_state.streaming_plan(config, 1, "Y16", 640, 400)


# --------------------------------------------------------------------------
# The blob itself
# --------------------------------------------------------------------------


def test_blob_is_a_complete_single_configuration(config):
    """645 bytes = 18-byte device descriptor + a 627-byte configuration.

    If this ever fails the fixture is truncated and every number below is
    suspect, so it is asserted first and separately.
    """
    assert len(DESCRIPTOR.read_bytes()) == 645
    assert config.total_length == 627
    assert config.trailing_bytes == 0
    assert config.num_interfaces == 5
    assert len(config.interfaces) == 5


def test_device_identity_matches_the_recorded_unit(config):
    device = config.device
    assert device is not None
    assert (device.vendor_id, device.product_id) == (0x2207, 0x110C)
    assert device.bcd_usb == 0x0200
    assert device.bcd_device == 0x0419
    assert device.num_configurations == 1


def test_configuration_is_bus_powered_at_500_ma(config):
    assert config.max_power_ma == 500
    assert config.configuration_value == 1


def test_two_video_function_associations(config):
    assert [(a.first_interface, a.interface_count) for a in config.associations] == [(0, 2), (2, 2)]
    assert all(a.function_class == 0x0E for a in config.associations)


# --------------------------------------------------------------------------
# Alternate settings and endpoints -- the six values docs/architecture.md
# listed as unrecorded prerequisites for direct streaming.
# --------------------------------------------------------------------------


def test_every_interface_declares_exactly_one_alternate_setting(config):
    """No interface offers a choice of alt setting, so there is none to pick.

    This is what removes isochronous bandwidth negotiation from the RGB
    silence hypothesis space: there is no alternate setting to select and no
    isochronous admission control to fail.
    """
    for number in range(5):
        assert [i.alternate_setting for i in config.alternate_settings(number)] == [0]


def test_endpoint_map_is_exact(config):
    seen = {
        (i.number, e.address): (e.transfer_type, e.max_packet_size, e.direction, e.interval)
        for i in config.interfaces
        for e in i.endpoints
    }
    assert seen == {
        (0, 0x81): ("interrupt", 16, "in", 8),
        (1, 0x82): ("bulk", 512, "in", 0),
        (2, 0x83): ("interrupt", 16, "in", 8),
        (3, 0x84): ("bulk", 512, "in", 0),
        (4, 0x85): ("interrupt", 1024, "in", 4),
        (4, 0x01): ("interrupt", 1024, "out", 4),
    }


def test_no_endpoint_address_is_declared_by_two_interfaces(config):
    """Simultaneously-active interfaces must declare disjoint endpoints."""
    assert config.duplicate_endpoint_addresses() == {}


def test_both_streaming_endpoints_are_bulk_not_isochronous(config):
    for number in (1, 3):
        iface = config.interface(number, 0)
        assert iface is not None
        assert iface.is_video_streaming
        assert [e.transfer_type for e in iface.endpoints] == ["bulk"]


# --------------------------------------------------------------------------
# Class-specific headers
# --------------------------------------------------------------------------


def test_both_functions_declare_uvc_1_00(config):
    """bcdUVC 0x0100 fixes the VS Probe/Commit block at 26 bytes, not 34.

    The kernel log's "Found UVC 1.00 device ZXProduct (2207:110c)" (emitted
    once per function) is the independent confirmation of this descriptor
    field.
    """
    for number in (0, 2):
        iface = config.interface(number, 0)
        assert iface is not None
        header = iface.vc_header
        assert header is not None
        assert header.bcd_uvc == 0x0100
        assert header.uvc_version == "1.00"
        assert header.clock_frequency_hz == 48_000_000
        assert uvc_stream_state.probe_commit_block_bytes(header.bcd_uvc) == 26


def test_video_control_headers_name_their_streaming_interface(config):
    depth_control = config.interface(0, 0)
    rgb_control = config.interface(2, 0)
    assert depth_control is not None and depth_control.vc_header is not None
    assert rgb_control is not None and rgb_control.vc_header is not None
    assert depth_control.vc_header.streaming_interfaces == (1,)
    assert rgb_control.vc_header.streaming_interfaces == (3,)


def test_probe_commit_block_length_threshold():
    """The 26/34 split follows the SDK's own `bcdUVC > 0x010f` test."""
    assert uvc_stream_state.probe_commit_block_bytes(0x0100) == 26
    assert uvc_stream_state.probe_commit_block_bytes(0x010F) == 26
    assert uvc_stream_state.probe_commit_block_bytes(0x0110) == 34
    assert uvc_stream_state.probe_commit_block_bytes(0x0150) == 34


def test_rgb_vs_input_header(config):
    iface = config.interface(3, 0)
    assert iface is not None
    header = iface.vs_input_header
    assert header is not None
    assert header.num_formats == 1
    assert header.total_length == 73
    assert header.endpoint_address == 0x84
    assert header.terminal_link == 3
    assert header.still_capture_method == 0
    assert header.trigger_support == 0
    assert header.controls == (0x00,)
    assert header.dynamic_format_change_supported is False


def test_depth_vs_input_header(config):
    iface = config.interface(1, 0)
    assert iface is not None
    header = iface.vs_input_header
    assert header is not None
    assert header.num_formats == 2
    assert header.total_length == 265
    assert header.endpoint_address == 0x82
    assert header.terminal_link == 3
    assert header.controls == (0x00, 0x00)


def test_vs_input_header_endpoint_matches_the_declared_endpoint(config):
    """The class header and the standard endpoint descriptor must agree."""
    for number in (1, 3):
        iface = config.interface(number, 0)
        assert iface is not None
        assert iface.vs_input_header is not None
        assert iface.vs_input_header.endpoint_address == iface.endpoints[0].address


# --------------------------------------------------------------------------
# Formats and frames
# --------------------------------------------------------------------------


def test_rgb_advertises_exactly_one_format_and_one_frame(config):
    """bFormatIndex=1 / bFrameIndex=1 is forced -- no wrong mode is selectable."""
    iface = config.interface(3, 0)
    assert iface is not None
    assert len(iface.formats) == 1
    fmt = iface.formats[0]
    assert (fmt.format_index, fmt.fourcc, fmt.compressed) == (1, "MJPG", True)
    assert fmt.default_frame_index == 1
    assert len(fmt.frames) == 1
    frame = fmt.frames[0]
    assert (frame.frame_index, frame.width, frame.height) == (1, 1280, 800)
    assert frame.max_video_frame_buffer_bytes == 2_048_000
    assert frame.min_bit_rate == 81_920_000
    assert frame.max_bit_rate == 163_840_000
    assert frame.default_frame_interval == 333_333
    assert frame.default_frame_rate_hz == 30.0
    assert frame.interval_type == 4
    assert frame.frame_intervals == (333_333, 500_000, 666_666, 1_000_000)
    assert frame.frame_rates_hz == (30.0, 20.0, 15.0, 10.0)


def test_depth_formats_and_frames(config):
    iface = config.interface(1, 0)
    assert iface is not None
    assert [(f.format_index, f.fourcc, f.bits_per_pixel) for f in iface.formats] == [
        (1, "YUY2", 16),
        (2, "Y16", 16),
    ]
    assert all(not f.compressed for f in iface.formats)
    yuy2, y16 = iface.formats
    assert [(f.frame_index, f.width, f.height) for f in yuy2.frames] == [
        (1, 640, 400),
        (2, 1280, 800),
    ]
    assert [(f.frame_index, f.width, f.height) for f in y16.frames] == [
        (1, 640, 400),
        (2, 1280, 800),
        (3, 640, 800),
    ]
    assert [f.max_video_frame_buffer_bytes for f in y16.frames] == [512_000, 2_048_000, 1_024_000]
    assert all(f.frame_rates_hz == (15.0, 10.0, 5.0) for f in y16.frames)


def test_uncompressed_buffer_sizes_are_exactly_width_times_height_times_bpp(config):
    """Every uncompressed frame's declared buffer equals its tight pixel size.

    That consistency is what makes the RGB MJPG mismatch (see
    ``test_rgb_driver_sizeimage_is_four_times_the_descriptor``) stand out as a
    property of the compressed path rather than a vendor typo.
    """
    iface = config.interface(1, 0)
    assert iface is not None
    for fmt in iface.formats:
        assert fmt.bits_per_pixel is not None
        for frame in fmt.frames:
            expected = frame.width * frame.height * fmt.bits_per_pixel // 8
            assert frame.max_video_frame_buffer_bytes == expected


def test_mjpg_declared_buffer_equals_the_uncompressed_16bpp_size(config):
    """2048000 == 1280*800*2: the vendor filled the MJPG field generically.

    A compressed format's ``dwMaxVideoFrameBufferSize`` carries no information
    about real JPEG sizes, which is why the driver is free to report something
    else after probing.
    """
    iface = config.interface(3, 0)
    assert iface is not None
    frame = iface.formats[0].frames[0]
    assert frame.max_video_frame_buffer_bytes == 1280 * 800 * 2


# --------------------------------------------------------------------------
# The derived streaming plan
# --------------------------------------------------------------------------


def test_rgb_streaming_plan(rgb_plan):
    assert rgb_plan.control_interface == 2
    assert rgb_plan.streaming_interface == 3
    assert rgb_plan.uvc_version == "1.00"
    assert rgb_plan.probe_commit_block_bytes == 26
    assert rgb_plan.alternate_settings == (0,)
    assert rgb_plan.requires_alt_setting_selection is False
    assert rgb_plan.endpoint_address == 0x84
    assert rgb_plan.endpoint_transfer_type == "bulk"
    assert rgb_plan.endpoint_max_packet_size == 512
    assert (rgb_plan.format_index, rgb_plan.frame_index) == (1, 1)
    assert rgb_plan.compressed is True
    assert rgb_plan.default_frame_interval == 333_333
    assert rgb_plan.default_frame_rate_hz == 30.0


def test_depth_streaming_plan(depth_plan):
    assert depth_plan.control_interface == 0
    assert depth_plan.streaming_interface == 1
    assert depth_plan.alternate_settings == (0,)
    assert depth_plan.endpoint_address == 0x82
    assert depth_plan.endpoint_transfer_type == "bulk"
    assert (depth_plan.format_index, depth_plan.frame_index) == (2, 1)
    assert depth_plan.compressed is False
    assert depth_plan.default_frame_rate_hz == 15.0


def test_streaming_plan_refuses_an_unadvertised_mode(config):
    with pytest.raises(uvc_stream_state.StreamingPlanError):
        uvc_stream_state.streaming_plan(config, 3, "YUY2", 1280, 800)
    with pytest.raises(uvc_stream_state.StreamingPlanError):
        uvc_stream_state.streaming_plan(config, 3, "MJPG", 640, 480)
    with pytest.raises(uvc_stream_state.StreamingPlanError):
        uvc_stream_state.streaming_plan(config, 9, "MJPG", 1280, 800)


def test_streaming_plan_refuses_a_video_control_interface(config):
    """Interface 2 is the RGB VideoControl interface, not its streaming half."""
    with pytest.raises(uvc_stream_state.StreamingPlanError):
        uvc_stream_state.streaming_plan(config, 2, "MJPG", 1280, 800)


# --------------------------------------------------------------------------
# Bandwidth
# --------------------------------------------------------------------------


def test_high_speed_bulk_ceiling_is_the_usb2_protocol_limit():
    assert uvc_stream_state.HS_BULK_CEILING_BYTES_PER_SECOND == 8_000 * 13 * 512
    assert uvc_stream_state.HS_BULK_CEILING_BYTES_PER_SECOND == 53_248_000


def test_rgb_peak_rate_is_well_under_the_bulk_ceiling(rgb_plan):
    """20.48 MB/s of a 53.248 MB/s single-endpoint ceiling.

    Bandwidth cannot explain the silence: the peak advertised rate needs about
    38% of what one high-speed bulk endpoint can carry, and bulk transfers
    have no reserved-bandwidth admission control to refuse at STREAMON.
    """
    assert rgb_plan.max_bit_rate_bytes_per_second == 20_480_000
    assert rgb_plan.bulk_ceiling_utilisation == 0.3846


def test_depth_peak_rate_is_far_under_the_ceiling(depth_plan):
    assert depth_plan.max_bit_rate_bytes_per_second == 5_120_000
    assert depth_plan.bulk_ceiling_utilisation == 0.0962


# --------------------------------------------------------------------------
# The frozen runs
# --------------------------------------------------------------------------


def test_rgb_run_reached_streamon_and_dequeued_nothing(rgb_run):
    assert rgb_run.status == "poll_timeout"
    assert rgb_run.succeeded is False
    assert rgb_run.exit_code == 4
    assert rgb_run.role == "rgb"
    assert rgb_run.node_path == "/dev/video2"
    assert rgb_run.interface_number == 2
    assert (rgb_run.fourcc, rgb_run.width, rgb_run.height) == ("MJPG", 1280, 800)
    assert rgb_run.reached_streamon is True
    assert rgb_run.dequeued_bytes is False
    assert rgb_run.poll_duration_s is not None
    assert rgb_run.poll_duration_s > 8.0


def test_depth_run_is_the_positive_control(depth_run):
    assert depth_run.status == "complete"
    assert depth_run.succeeded is True
    assert depth_run.role == "depth"
    assert depth_run.node_path == "/dev/video0"
    assert (depth_run.fourcc, depth_run.width, depth_run.height) == ("Y16", 640, 400)
    assert depth_run.reached_streamon is True
    assert depth_run.dequeued_bytes is True
    assert depth_run.poll_duration_s is not None
    assert depth_run.poll_duration_s < 1.1


def test_the_two_runs_differ_in_function_not_in_kernel_path(rgb_run, depth_run):
    """Same engine, same driver, same host -- different USB function."""
    assert rgb_run.mode == "v4l2-mmap-rgb-frame"
    assert depth_run.mode == "v4l2-mmap-depth-frame"
    assert rgb_run.issued_ioctls == depth_run.issued_ioctls


def test_scanner_010_never_reached_streamon():
    """The first RGB attempt refused itself at the ceiling, before streaming.

    It is therefore evidence about `sizeimage` only, and says nothing about
    whether the stream would have produced a frame.
    """
    run = uvc_stream_state.summarize_run(json.loads(RGB_FIRST_ATTEMPT.read_text(encoding="utf-8")))
    assert run.status == "format_mismatch"
    assert run.reached_streamon is False
    assert run.dequeued_bytes is False
    assert run.driver_sizeimage == 8_192_000


def test_rgb_driver_sizeimage_is_four_times_the_descriptor(rgb_plan, rgb_run):
    comparison = uvc_stream_state.compare_sizeimage(rgb_plan, rgb_run)
    assert comparison.descriptor_bytes == 2_048_000
    assert comparison.driver_bytes == 8_192_000
    assert comparison.ratio == 4.0
    assert comparison.agrees is False
    assert comparison.compressed is True


def test_depth_driver_sizeimage_matches_the_descriptor_exactly(depth_plan, depth_run):
    comparison = uvc_stream_state.compare_sizeimage(depth_plan, depth_run)
    assert comparison.descriptor_bytes == 512_000
    assert comparison.driver_bytes == 512_000
    assert comparison.ratio == 1.0
    assert comparison.agrees is True
    assert comparison.compressed is False


# --------------------------------------------------------------------------
# What was and was not negotiated
# --------------------------------------------------------------------------


def test_negotiation_matrix_marks_the_interval_and_control_ioctls_as_never_issued(rgb_run):
    matrix = {
        row["ioctl"]: row["issued_by_owned_code"]
        for row in uvc_stream_state.negotiation_matrix(rgb_run)
    }
    assert matrix["VIDIOC_S_FMT"] is True
    assert matrix["VIDIOC_STREAMON"] is True
    assert matrix["VIDIOC_DQBUF"] is True
    # Never issued by owned code during the attempt:
    assert matrix["VIDIOC_S_PARM"] is False
    assert matrix["VIDIOC_G_PARM"] is False
    assert matrix["VIDIOC_ENUM_FRAMEINTERVALS"] is False
    assert matrix["VIDIOC_QUERYCTRL"] is False
    assert matrix["VIDIOC_S_CTRL"] is False


def test_the_depth_run_negotiated_no_more_than_the_rgb_run(rgb_run, depth_run):
    """The successful run used a strict subset of the same operations.

    Whatever the RGB path is missing, it is not a V4L2 call the working depth
    capture made and the RGB attempt skipped.
    """
    rgb = {
        r["ioctl"]
        for r in uvc_stream_state.negotiation_matrix(rgb_run)
        if r["issued_by_owned_code"]
    }
    depth = {
        r["ioctl"]
        for r in uvc_stream_state.negotiation_matrix(depth_run)
        if r["issued_by_owned_code"]
    }
    assert depth <= rgb


# --------------------------------------------------------------------------
# Tolerance
# --------------------------------------------------------------------------


def test_truncated_blob_degrades_instead_of_raising():
    blob = DESCRIPTOR.read_bytes()[:100]
    config = uvc_stream_state.decode_configuration(blob)
    assert config.device is not None
    assert len(config.interfaces) < 5


def test_empty_blob_yields_an_empty_configuration():
    config = uvc_stream_state.decode_configuration(b"")
    assert config.device is None
    assert config.interfaces == ()
    assert config.trailing_bytes == 0


def test_zero_length_record_stops_the_walk():
    blob = DESCRIPTOR.read_bytes()[:35] + b"\x00\x04" + DESCRIPTOR.read_bytes()[37:]
    config = uvc_stream_state.decode_configuration(blob)
    assert config.trailing_bytes > 0


# --------------------------------------------------------------------------
# The rendered report and JSON
# --------------------------------------------------------------------------


def test_report_renders_and_names_the_decisive_numbers(
    config, rgb_plan, rgb_run, depth_plan, depth_run
):
    text = uvc_stream_state.render_report(config, rgb_plan, rgb_run, depth_plan, depth_run)
    assert "2207:110c" in text
    assert "0x84" in text
    assert "bulk" in text
    assert "NEVER ISSUED" in text
    assert "VIDIOC_S_PARM" in text


def test_json_payload_round_trips(config, rgb_plan, rgb_run, depth_plan, depth_run):
    payload = uvc_stream_state.build_json(config, rgb_plan, rgb_run, depth_plan, depth_run)
    restored = json.loads(json.dumps(payload))
    assert restored["rgb_plan"]["endpoint_transfer_type"] == "bulk"
    assert restored["rgb_sizeimage"]["ratio"] == 4.0
    assert restored["depth_sizeimage"]["agrees"] is True


def test_cli_runs_offline_against_the_committed_fixtures(capsys):
    assert uvc_stream_state.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"]["device"]["usb_id"] == "2207:110c"
    assert payload["rgb_plan"]["frame_index"] == 1
