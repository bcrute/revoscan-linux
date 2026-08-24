"""Drive the RGB (MJPEG) V4L2/MMAP capture lifecycle against a fake kernel.

``tests/test_v4l2_capture.py`` already proves the shared engine's lifecycle -
the fixed bounds, the exactly-once obligations, the shielded cleanup - against
the depth profile. Nothing here duplicates that proof. What was still
untested before this file is the other half of the pair
:mod:`scanner3d.v4l2_capture` was refactored around:
:data:`~scanner3d.rgb_frame.RGB_PROFILE` and
:class:`~scanner3d.rgb_frame.V4l2RgbFrameBackend`, which had zero lifecycle
coverage of their own.

The central new claim is *isolation*: two profiles now share one engine, and
this file exists to prove that sharing an engine did not smuggle in any way
for one profile to reach the other's camera, ceilings, or notion of what a
complete frame is. Everything else - the ceilings, the single-attempt bounds,
the cleanup guarantees, the JPEG-shaped outcomes - is asked of the RGB profile
specifically, because a profile that merely inherited the depth numbers by
accident would pass a suite that never checked the numbers themselves.
"""

from __future__ import annotations

import errno
import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from scanner3d.ownership import CLEANUP_FAILED_EXIT
from scanner3d.rgb_frame import (
    RGB_ARTIFACT_NAME,
    RGB_CAPTURE_DEADLINE_S,
    RGB_DESCRIPTOR_FRAME_BUFFER_BYTES,
    RGB_DESCRIPTOR_FRAME_RATE_HZ,
    RGB_EXECUTE_FLAG,
    RGB_FOURCC,
    RGB_FRAME_HEIGHT,
    RGB_FRAME_WIDTH,
    RGB_MAX_BUFFER_BYTES,
    RGB_MAX_RAW_ARTIFACT_BYTES,
    RGB_MAX_TOTAL_MAPPED_BYTES,
    RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES,
    RGB_MIN_ACCEPTED_BUFFERS,
    RGB_PROFILE,
    RGB_REQUESTED_BUFFERS,
    RGB_ROLE,
    V4l2RgbFrameBackend,
    preserve_rgb_frame,
)
from scanner3d.v4l2_capture import (
    DEPTH_PROFILE,
    EXPECTED_FRAME_BYTES,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_BUFFER_BYTES,
    MAX_RAW_ARTIFACT_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    MIN_ACCEPTED_BUFFERS,
    REQUESTBUFFERS_FORMAT,
    REQUESTED_BUFFERS,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    BufferInfo,
    CaptureOutcome,
    CaptureStatus,
    FrameArtifact,
    Step,
    V4l2DepthFrameBackend,
    V4l2FrameBackend,
    _mapping_refusal,
    pack_buffer,
    parse_buffer,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.fake_v4l2 import FakeKernel, FakeMapping, SteppedClock, make_backend, sysfs
from tests.test_jpeg_inspect import baseline_jpeg, progressive_jpeg

# ``sysfs`` is a pytest fixture re-exported by name from tests.fake_v4l2, used
# as a parameter throughout this file; ruff cannot see that use statically.
__all__ = ["FakeMapping", "SteppedClock", "sysfs"]


# --- shared helpers -------------------------------------------------------------


def rgb_backend(
    kernel: FakeKernel, sysfs: SysfsBuilder, clock: Callable[[], float] | None = None
) -> V4l2FrameBackend:
    """The shared engine bound to the RGB profile, driven by ``kernel``."""
    return make_backend(kernel, sysfs, clock, V4l2RgbFrameBackend)


def depth_backend(
    kernel: FakeKernel, sysfs: SysfsBuilder, clock: Callable[[], float] | None = None
) -> V4l2FrameBackend:
    """The shared engine bound to the depth profile - the isolation contrast."""
    return make_backend(kernel, sysfs, clock, V4l2DepthFrameBackend)


def run_rgb(
    kernel: FakeKernel,
    sysfs: SysfsBuilder,
    tmp_path: Path,
    clock: Callable[[], float] | None = None,
) -> CaptureOutcome:
    return rgb_backend(kernel, sysfs, clock).run(REAL_SERIAL, output_dir=tmp_path / "out")


def kinds(outcome: CaptureOutcome) -> list[str]:
    """Which evidence files the run produced, in order."""
    return [f.kind for f in outcome.artifacts.files]


def artifact(outcome: CaptureOutcome, kind: str = "mjpeg") -> FrameArtifact:
    """The one artifact of ``kind``; fails the test if it is not there."""
    matches = [f for f in outcome.artifacts.files if f.kind == kind]
    assert len(matches) == 1, f"expected exactly one {kind!r} artifact, got {kinds(outcome)}"
    return matches[0]


def _reqbufs_requested_counts(kernel: FakeKernel) -> list[int]:
    """Every ``count`` this run ever asked ``VIDIOC_REQBUFS`` for, in order."""
    return [
        struct.unpack(REQUESTBUFFERS_FORMAT, payload)[0]
        for request, payload in kernel.calls
        if request == VIDIOC_REQBUFS
    ]


def _buffer_info(**overrides: object) -> BufferInfo:
    """A ``BufferInfo`` built from a real ``pack_buffer``/``parse_buffer`` round trip,
    with a couple of fields overridden - mirrors ``tests/test_v4l2_uapi.py``'s own
    ``_mapping_refusal`` helper rather than inventing a second one."""
    info = parse_buffer(bytes(pack_buffer(0)))
    fields_: dict[str, object] = {"length": 4096, "offset": 0}
    fields_.update(overrides)
    return replace(info, **fields_)  # type: ignore[arg-type]


class _UnmapExplodes:
    """A mapping whose ``close()`` always fails, wrapping a real ``FakeMapping``.

    Used only for the cleanup-failure tests in section D: the wrapped mapping's
    own double-unmap guard is irrelevant there because the point is that the
    *first* unmap attempt itself fails.
    """

    def __init__(self, inner: FakeMapping) -> None:
        self._inner = inner

    def close(self) -> None:
        raise OSError(errno.EIO, "cannot unmap")

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: slice, /) -> bytes:
        return self._inner[index]


# =================================================================================
# A. Profile isolation
# =================================================================================


def test_the_rgb_backend_opens_only_the_rgbcam_video0_node(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The RGB backend must resolve the ``RGBCam`` interface's index-0 node and
    nothing else - never the ``DepthCam`` node, and never a fixed ``/dev/videoN``."""
    kernel = FakeKernel(profile=RGB_PROFILE)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert kernel.opened == [str(sysfs.dev / "video0")]
    assert outcome.target is not None
    assert outcome.target.node_name == "video0"
    assert outcome.target.role == "rgb"
    assert outcome.target.v4l_index == 0


def test_the_depth_backend_still_opens_only_the_depthcam_video5_node(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The contrasting half of the isolation claim: sharing one engine between
    two profiles must not let the depth backend drift onto the RGB node either."""
    kernel = FakeKernel(profile=DEPTH_PROFILE)
    outcome = depth_backend(kernel, sysfs).run(REAL_SERIAL, output_dir=tmp_path / "out")
    assert kernel.opened == [str(sysfs.dev / "video5")]
    assert outcome.target is not None
    assert outcome.target.node_name == "video5"
    assert outcome.target.role == "depth"
    assert outcome.target.v4l_index == 0


def test_node_resolution_does_not_depend_on_any_fixed_video_number(tmp_path: Path) -> None:
    """Node numbers are assigned in USB probe order and move under replug or
    reboot; resolution must key on role plus V4L2 index, never on a number.

    Builds a tree where the RGB camera's index-0 node is ``video9`` (not
    ``video0``) and the depth camera's is ``video3`` (not ``video5``), and
    confirms each backend still finds its own camera.
    """
    builder = SysfsBuilder(tmp_path)
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video3", 0), ("video4", 1)),
    )
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{REAL_SERIAL}",
        video_nodes=(("video9", 0), ("video10", 1)),
    )

    rgb_outcome = rgb_backend(FakeKernel(profile=RGB_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-rgb"
    )
    assert rgb_outcome.target is not None
    assert rgb_outcome.target.node_name == "video9"
    assert rgb_outcome.target.role == "rgb"

    depth_outcome = depth_backend(FakeKernel(profile=DEPTH_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-depth"
    )
    assert depth_outcome.target is not None
    assert depth_outcome.target.node_name == "video3"
    assert depth_outcome.target.role == "depth"


def test_rgb_profile_constants_match_every_literal_in_the_assignment() -> None:
    """Every RGB ceiling, bound and mode value, pinned by literal rather than by
    reference to the module constant that also defines it - so a constant and
    its only user cannot silently drift together."""
    assert RGB_PROFILE.role == "rgb"
    assert RGB_PROFILE.fourcc == "MJPG"
    assert RGB_PROFILE.width == 1280
    assert RGB_PROFILE.height == 800
    assert RGB_PROFILE.compressed is True
    assert RGB_PROFILE.requested_buffers == 4
    assert RGB_PROFILE.min_accepted_buffers == 2
    assert RGB_PROFILE.max_buffer_bytes == 8192000
    assert RGB_PROFILE.max_total_mapped_bytes == 32768000
    assert RGB_PROFILE.capture_deadline_s == 8.0
    assert RGB_PROFILE.max_raw_artifact_bytes == 8192000
    assert RGB_PROFILE.descriptor_frame_rate_hz == 30.0
    assert RGB_PROFILE.execute_flag == "--execute-rgb-frame-capture"
    assert RGB_PROFILE.name == "rgb-frame"
    # The named constants agree with the literals above, so there is exactly
    # one place - not two - where "8192000" etc. could ever be decided.
    assert RGB_ROLE == "rgb"
    assert RGB_FOURCC == "MJPG"
    assert RGB_FRAME_WIDTH == 1280
    assert RGB_FRAME_HEIGHT == 800
    assert RGB_REQUESTED_BUFFERS == 4
    assert RGB_MIN_ACCEPTED_BUFFERS == 2
    assert RGB_MAX_BUFFER_BYTES == 8192000
    assert RGB_MAX_TOTAL_MAPPED_BYTES == 32768000
    assert RGB_CAPTURE_DEADLINE_S == 8.0
    assert RGB_MAX_RAW_ARTIFACT_BYTES == 8192000
    assert RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES == 8192000
    assert RGB_DESCRIPTOR_FRAME_BUFFER_BYTES == 2048000
    assert RGB_DESCRIPTOR_FRAME_RATE_HZ == 30.0
    assert RGB_EXECUTE_FLAG == "--execute-rgb-frame-capture"
    assert RGB_ARTIFACT_NAME == "rgb-frame.jpg"


def test_depth_profile_constants_are_unchanged_and_distinct_from_rgbs() -> None:
    """The depth profile's own literals, so a change made for RGB's sake cannot
    have quietly moved depth's numbers too."""
    assert DEPTH_PROFILE.role == "depth"
    assert DEPTH_PROFILE.fourcc == "Y16 "
    assert DEPTH_PROFILE.width == 640
    assert DEPTH_PROFILE.height == 400
    assert DEPTH_PROFILE.compressed is False
    assert DEPTH_PROFILE.requested_buffers == 4
    assert DEPTH_PROFILE.min_accepted_buffers == 2
    assert DEPTH_PROFILE.max_buffer_bytes == 1024 * 1024
    assert DEPTH_PROFILE.max_total_mapped_bytes == 4 * 1024 * 1024
    assert DEPTH_PROFILE.capture_deadline_s == 8.0
    assert DEPTH_PROFILE.max_raw_artifact_bytes == 1024 * 1024
    assert DEPTH_PROFILE.descriptor_frame_rate_hz == 15.0
    assert DEPTH_PROFILE.execute_flag == "--execute-depth-frame-capture"
    assert FOURCC == "Y16 "
    assert FRAME_WIDTH == 640
    assert FRAME_HEIGHT == 400
    assert EXPECTED_FRAME_BYTES == 640 * 400 * 2
    assert REQUESTED_BUFFERS == 4
    assert MIN_ACCEPTED_BUFFERS == 2
    assert MAX_BUFFER_BYTES == 1024 * 1024
    assert MAX_TOTAL_MAPPED_BYTES == 4 * 1024 * 1024
    assert MAX_RAW_ARTIFACT_BYTES == 1024 * 1024

    # And the two profiles disagree on exactly the fields that make them two
    # different cameras, not one profile wearing a second name.
    assert DEPTH_PROFILE.role != RGB_PROFILE.role
    assert DEPTH_PROFILE.fourcc != RGB_PROFILE.fourcc
    assert (DEPTH_PROFILE.width, DEPTH_PROFILE.height) != (RGB_PROFILE.width, RGB_PROFILE.height)
    assert DEPTH_PROFILE.compressed != RGB_PROFILE.compressed
    assert DEPTH_PROFILE.max_buffer_bytes != RGB_PROFILE.max_buffer_bytes
    assert DEPTH_PROFILE.max_total_mapped_bytes != RGB_PROFILE.max_total_mapped_bytes
    assert DEPTH_PROFILE.max_raw_artifact_bytes != RGB_PROFILE.max_raw_artifact_bytes
    assert DEPTH_PROFILE.execute_flag != RGB_PROFILE.execute_flag
    assert DEPTH_PROFILE.descriptor_frame_rate_hz != RGB_PROFILE.descriptor_frame_rate_hz


@pytest.mark.parametrize(
    "attribute",
    ["width", "height", "fourcc", "max_buffer_bytes", "requested_buffers", "compressed"],
)
def test_neither_profile_can_be_mutated_or_widened_by_a_caller(attribute: str) -> None:
    """``CaptureProfile`` is frozen: a caller cannot reach in and widen a ceiling
    or substitute a mode field on either the shared RGB or depth instance."""
    with pytest.raises(FrozenInstanceError):
        setattr(RGB_PROFILE, attribute, getattr(RGB_PROFILE, attribute))
    with pytest.raises(FrozenInstanceError):
        setattr(DEPTH_PROFILE, attribute, getattr(DEPTH_PROFILE, attribute))


def test_v4l2_rgb_backend_adds_no_field_beyond_the_profile_default() -> None:
    """``V4l2RgbFrameBackend`` must be the shared engine plus a different default
    for the one field that selects a profile - not a second dataclass shape."""
    base_names = {f.name for f in fields(V4l2FrameBackend)}
    rgb_names = {f.name for f in fields(V4l2RgbFrameBackend)}
    assert rgb_names == base_names
    profile_field = next(f for f in fields(V4l2RgbFrameBackend) if f.name == "profile")
    assert profile_field.default is RGB_PROFILE
    default_profile_field = next(f for f in fields(V4l2FrameBackend) if f.name == "profile")
    assert default_profile_field.default is DEPTH_PROFILE


def test_a_tree_with_only_the_depth_interface_leaves_rgb_unresolved(tmp_path: Path) -> None:
    """A device tree exposing no RGB interface must never let the RGB backend
    fall back to guessing a node, while the depth backend is unaffected.

    The docstring of :class:`~scanner3d.rgb_frame.V4l2RgbFrameBackend` and this
    assignment both describe this as ``NODE_UNRESOLVED``, and
    ``CaptureStatus.NODE_UNRESOLVED`` does exist (with its own exit code) - but
    no code path in :mod:`scanner3d.v4l2_capture` ever assigns it:
    ``resolve_capture_node`` raises ``DeviceAbsentError`` both when the device
    itself is missing and when it exposes no function of the requested role,
    and ``V4l2FrameBackend._preconditions`` maps every ``DeviceAbsentError`` to
    ``CaptureStatus.DEVICE_ABSENT``. This asserts the status the engine
    actually returns rather than the one that was never wired up; see this
    suite's final report for the same observation.
    """
    builder = SysfsBuilder(tmp_path)
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video5", 0), ("video6", 1)),
    )

    rgb_outcome = rgb_backend(FakeKernel(profile=RGB_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-rgb"
    )
    assert rgb_outcome.status is CaptureStatus.DEVICE_ABSENT
    assert rgb_outcome.target is None
    assert rgb_outcome.cleanup.attempted is False

    depth_outcome = depth_backend(FakeKernel(profile=DEPTH_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-depth"
    )
    assert depth_outcome.status is not CaptureStatus.DEVICE_ABSENT
    assert depth_outcome.target is not None
    assert depth_outcome.target.node_name == "video5"


def test_a_tree_with_only_the_rgb_interface_leaves_depth_unresolved(tmp_path: Path) -> None:
    """The mirror image of the previous test: no depth interface must not stop
    the RGB backend from resolving its own camera."""
    builder = SysfsBuilder(tmp_path)
    base = builder.add_usb_device()
    builder.add_usbfs()
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{REAL_SERIAL}",
        video_nodes=(("video0", 0), ("video1", 1)),
    )

    depth_outcome = depth_backend(FakeKernel(profile=DEPTH_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-depth"
    )
    assert depth_outcome.status is CaptureStatus.DEVICE_ABSENT
    assert depth_outcome.target is None

    rgb_outcome = rgb_backend(FakeKernel(profile=RGB_PROFILE), builder).run(
        REAL_SERIAL, output_dir=tmp_path / "out-rgb"
    )
    assert rgb_outcome.status is not CaptureStatus.DEVICE_ABSENT
    assert rgb_outcome.target is not None
    assert rgb_outcome.target.node_name == "video0"


# =================================================================================
# B. Exact RGB ceilings
# =================================================================================


def test_a_buffer_length_of_exactly_the_ceiling_is_accepted(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, buffer_length=RGB_MAX_BUFFER_BYTES)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is not CaptureStatus.MAPPING_ERROR, outcome.detail
    assert len(kernel.map_calls) == RGB_REQUESTED_BUFFERS
    assert all(length == RGB_MAX_BUFFER_BYTES for _, length, _ in kernel.map_calls)
    assert kernel.count(VIDIOC_STREAMON) == 1


def test_a_buffer_length_one_over_the_ceiling_is_refused_before_mmap(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, buffer_length=RGB_MAX_BUFFER_BYTES + 1)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.MAPPING_ERROR
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_total_mapped_bytes_past_the_rgb_ceiling_is_refused() -> None:
    """``RGB_MAX_BUFFER_BYTES * RGB_REQUESTED_BUFFERS == RGB_MAX_TOTAL_MAPPED_BYTES``
    (8192000 * 4 == 32768000), so - exactly as for the depth profile - the largest
    run four legally-sized buffers can ever produce lands exactly on the total
    ceiling and never past it; there is no way to drive the *running total*
    over its ceiling through the full engine without a buffer that also trips
    the *per-buffer* ceiling first. The arithmetic is asserted as well as the
    two edges, so raising one ceiling without the other cannot go unnoticed.
    ``_mapping_refusal`` is exercised directly with ``profile=RGB_PROFILE``, the
    same way ``tests/test_v4l2_uapi.py`` already does for the depth profile's
    own total ceiling.
    """
    assert RGB_MAX_BUFFER_BYTES * RGB_REQUESTED_BUFFERS == RGB_MAX_TOTAL_MAPPED_BYTES == 32768000
    info = _buffer_info(length=4096)
    assert (
        _mapping_refusal(
            info,
            expected_index=info.index,
            running_total=RGB_MAX_TOTAL_MAPPED_BYTES,
            profile=RGB_PROFILE,
        )
        is None
    )
    over_total = RGB_MAX_TOTAL_MAPPED_BYTES + 1
    reason = _mapping_refusal(
        info, expected_index=info.index, running_total=over_total, profile=RGB_PROFILE
    )
    assert reason is not None
    assert str(RGB_MAX_TOTAL_MAPPED_BYTES) in reason


def test_a_negotiated_sizeimage_past_the_rgb_ceiling_is_refused_before_reqbufs(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, sizeimage=RGB_MAX_BUFFER_BYTES + 4096)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.FORMAT_MISMATCH
    assert kernel.count(VIDIOC_REQBUFS) == 0
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.map_calls == []


def test_a_negotiated_sizeimage_one_over_the_rgb_ceiling_is_refused_before_reqbufs(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The exact boundary, one byte past, not merely a comfortably oversize
    value: an off-by-one in the comparison would let a buffer the profile never
    agreed to map reach ``VIDIOC_REQBUFS``."""
    kernel = FakeKernel(profile=RGB_PROFILE, sizeimage=RGB_MAX_BUFFER_BYTES + 1)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.FORMAT_MISMATCH
    assert str(RGB_MAX_BUFFER_BYTES) in outcome.detail
    assert kernel.count(VIDIOC_REQBUFS) == 0
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert kernel.map_calls == []


def test_the_measured_driver_sizeimage_is_accepted_and_maps_exactly_the_total_ceiling(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The whole point of this task, driven end to end.

    ``uvcvideo`` was measured answering ``sizeimage`` 8192000 for this mode on
    this unit, four times the mode's own descriptor
    ``dwMaxVideoFrameBufferSize``. That exact value must now be *accepted* -
    the previous descriptor-derived 2 MiB ceiling refused it before
    ``VIDIOC_REQBUFS`` and no frame could ever be dequeued - and four such
    buffers must land exactly on the total ceiling rather than past it.
    """
    kernel = FakeKernel(
        profile=RGB_PROFILE,
        sizeimage=RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES,
        frame_payload=baseline_jpeg(1280, 800),
    )
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert outcome.negotiated_format is not None
    assert outcome.negotiated_format.sizeimage == RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES
    assert kernel.count(VIDIOC_STREAMON) == 1
    assert len(kernel.map_calls) == RGB_REQUESTED_BUFFERS
    assert all(length == RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES for _, length, _ in kernel.map_calls)
    # Exactly on the total ceiling, never over it.
    assert sum(length for _, length, _ in kernel.map_calls) == RGB_MAX_TOTAL_MAPPED_BYTES


def test_the_measured_sizeimage_is_four_times_the_descriptor_and_both_are_kept(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """Raising the ceiling did not overwrite the descriptor's own number.

    Both values are evidence and they say different things: 2048000 is what
    the device advertises, 8192000 is what the Linux driver was observed to
    allocate. The plan must quote both, and must not present the measured one
    as a prediction of how many bytes a compressed frame will occupy.
    """
    assert RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES == 4 * RGB_DESCRIPTOR_FRAME_BUFFER_BYTES
    plan = RGB_PROFILE.plan_as_dict()
    assert plan["descriptor_max_video_frame_buffer_bytes"] == RGB_DESCRIPTOR_FRAME_BUFFER_BYTES
    assert plan["measured_driver_sizeimage_bytes"] == RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES
    assert plan["max_buffer_bytes"] == RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES
    # A real frame is nowhere near the ceiling, and the run reports what
    # arrived rather than what could have.
    payload = baseline_jpeg(1280, 800)
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert outcome.frame_bytes == len(payload) < RGB_MAX_RAW_ARTIFACT_BYTES


# =================================================================================
# B2. The raw-artifact ceiling, exactly at and one over
# =================================================================================


def test_a_dequeued_payload_of_exactly_the_artifact_ceiling_is_preserved_whole(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """``bytesused`` exactly on the ceiling is copied and written in full.

    Not a realistic MJPEG frame - it is mapping filler, and the run is
    correctly called ``malformed_frame`` - but it is the largest payload the
    engine may ever copy, and every byte of it must reach the file with a hash
    over the same bytes. An off-by-one here would either truncate the biggest
    legal frame or refuse it.
    """
    kernel = FakeKernel(
        profile=RGB_PROFILE,
        buffer_length=RGB_MAX_RAW_ARTIFACT_BYTES,
        bytesused=RGB_MAX_RAW_ARTIFACT_BYTES,
    )
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.MALFORMED_FRAME, outcome.detail
    assert outcome.frame_bytes == RGB_MAX_RAW_ARTIFACT_BYTES
    record = artifact(outcome)
    assert record.ok is True
    assert record.byte_count == RGB_MAX_RAW_ARTIFACT_BYTES
    assert record.path is not None
    written = Path(record.path).read_bytes()
    assert len(written) == RGB_MAX_RAW_ARTIFACT_BYTES
    assert hashlib.sha256(written).hexdigest() == record.sha256


def test_a_bytesused_one_over_the_mapping_copies_nothing_and_writes_no_file(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """One byte past what is mapped is refused before a single byte is copied.

    The refusal is the *mapping* bound, and that is the point of the next test:
    with the per-buffer and artifact ceilings deliberately equal, this is the
    only way an oversize ``bytesused`` can present itself through the engine.
    """
    kernel = FakeKernel(
        profile=RGB_PROFILE,
        buffer_length=RGB_MAX_RAW_ARTIFACT_BYTES,
        bytesused=RGB_MAX_RAW_ARTIFACT_BYTES + 1,
    )
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.OVERSIZE_FRAME
    # ``None``, not zero: no byte count was ever taken because nothing was copied.
    assert outcome.frame_bytes is None
    assert kinds(outcome) == []
    out = tmp_path / "out"
    assert not out.exists() or list(out.iterdir()) == []
    # Refusing the frame is not refusing the cleanup.
    assert kernel.count(VIDIOC_STREAMOFF) == 1


def test_the_artifact_ceiling_cannot_be_reached_without_the_mapping_bound_firing_first() -> None:
    """Why the engine's own artifact-ceiling branch stays unreachable.

    ``max_raw_artifact_bytes`` and ``max_buffer_bytes`` are the same number on
    purpose, and ``_mapping_refusal`` has already rejected any buffer longer
    than that, so ``bytesused`` can never be both inside the mapping and past
    the artifact ceiling. The identity is asserted rather than assumed, because
    if the two ceilings were ever set apart the branch would become live and
    this comment would become wrong.
    """
    assert RGB_PROFILE.max_raw_artifact_bytes == RGB_PROFILE.max_buffer_bytes
    over = _buffer_info(length=RGB_MAX_BUFFER_BYTES + 1)
    assert (
        _mapping_refusal(over, expected_index=over.index, running_total=0, profile=RGB_PROFILE)
        is not None
    )
    at = _buffer_info(length=RGB_MAX_BUFFER_BYTES)
    assert (
        _mapping_refusal(at, expected_index=at.index, running_total=0, profile=RGB_PROFILE) is None
    )


def test_preserve_rgb_frame_writes_exactly_at_the_ceiling_and_refuses_one_over(
    tmp_path: Path,
) -> None:
    """The preserver's own bound, exercised directly at both edges.

    ``preserve_rgb_frame`` is reachable with any byte string, so its ceiling is
    checked independently of whatever the engine would have allowed through.
    One byte over must leave no file at all: a truncated file under a name that
    promises a frame is worse evidence than none.
    """
    at_dir = tmp_path / "at"
    at = preserve_rgb_frame(b"\x00" * RGB_MAX_RAW_ARTIFACT_BYTES, at_dir)
    assert at.ok is True
    assert at.ceiling_bytes == RGB_MAX_RAW_ARTIFACT_BYTES
    assert (at_dir / RGB_ARTIFACT_NAME).stat().st_size == RGB_MAX_RAW_ARTIFACT_BYTES

    over_dir = tmp_path / "over"
    over = preserve_rgb_frame(b"\x00" * (RGB_MAX_RAW_ARTIFACT_BYTES + 1), over_dir)
    assert over.attempted is True
    assert over.ok is False
    assert len(over.files) == 1
    assert over.files[0].ok is False
    assert over.files[0].path is None
    assert str(RGB_MAX_RAW_ARTIFACT_BYTES) in over.files[0].detail
    assert not (over_dir / RGB_ARTIFACT_NAME).exists()


def test_a_granted_count_of_one_is_refused(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, granted_buffers=1)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.BUFFER_COUNT_MISMATCH
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0


@pytest.mark.parametrize("granted", [2, 3, 4])
def test_a_granted_count_inside_the_accepted_range_streams(
    sysfs: SysfsBuilder, tmp_path: Path, granted: int
) -> None:
    kernel = FakeKernel(
        profile=RGB_PROFILE, granted_buffers=granted, frame_payload=baseline_jpeg(1280, 800)
    )
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert len(kernel.map_calls) == granted
    assert kernel.count(VIDIOC_QBUF) == granted


def test_more_buffers_than_requested_are_refused_and_never_over_requested(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """The kernel must never be trusted to have honoured the cap on its own,
    and the request itself must never ask for more than four."""
    kernel = FakeKernel(profile=RGB_PROFILE, granted_buffers=RGB_REQUESTED_BUFFERS + 1)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.BUFFER_COUNT_MISMATCH
    assert kernel.map_calls == []
    assert kernel.count(VIDIOC_STREAMON) == 0
    requested_counts = _reqbufs_requested_counts(kernel)
    assert requested_counts, "REQBUFS was never issued"
    assert requested_counts[0] == RGB_REQUESTED_BUFFERS


def test_a_zero_bytesperline_is_accepted_for_the_compressed_rgb_mode(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    """``uvcvideo`` reports ``bytesperline`` 0 for MJPEG because a bitstream has
    no row stride. Unlike the depth profile, this must not be refused."""
    kernel = FakeKernel(profile=RGB_PROFILE)  # default bytesperline for RGB is already 0
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.negotiated_format is not None
    assert outcome.negotiated_format.bytesperline == 0
    assert outcome.status is not CaptureStatus.FORMAT_MISMATCH
    # REQBUFS is issued twice - the allocation and the cleanup release - and
    # the allocation is the one that would have been skipped by a stride
    # refusal, so it is what is checked by name here.
    assert _reqbufs_requested_counts(kernel)[0] == RGB_REQUESTED_BUFFERS


# =================================================================================
# C. Single-attempt lifecycle bounds
# =================================================================================


def test_a_healthy_rgb_run_issues_each_ioctl_exactly_once_with_no_requeue(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert kernel.requests == [
        VIDIOC_QUERYCAP,
        VIDIOC_G_FMT,
        VIDIOC_S_FMT,
        VIDIOC_REQBUFS,
        *([VIDIOC_QUERYBUF] * RGB_REQUESTED_BUFFERS),
        *([VIDIOC_QBUF] * RGB_REQUESTED_BUFFERS),
        VIDIOC_STREAMON,
        VIDIOC_DQBUF,
        VIDIOC_STREAMOFF,
        VIDIOC_REQBUFS,
    ]
    assert len(kernel.map_calls) <= RGB_REQUESTED_BUFFERS
    # No requeue: every QBUF happened before STREAMON, none after DQBUF.
    qbuf_positions = [i for i, r in enumerate(kernel.requests) if r == VIDIOC_QBUF]
    streamon_at = kernel.requests.index(VIDIOC_STREAMON)
    assert all(pos < streamon_at for pos in qbuf_positions)
    assert kernel.count(VIDIOC_QBUF) == RGB_REQUESTED_BUFFERS


def test_a_poll_timeout_is_bounded_by_the_deadline_and_is_not_a_retry(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, poll_revents=0)
    clock = SteppedClock()
    outcome = run_rgb(kernel, sysfs, tmp_path, clock=clock)
    assert outcome.status is CaptureStatus.POLL_TIMEOUT
    # The live scanner-011 attempt is reported as "exit 4"; pin that mapping
    # here rather than leaving it to the status enum alone.
    assert outcome.exit_code == 4
    assert len(kernel.polls) == 1
    assert kernel.polls[0][1] == pytest.approx(RGB_CAPTURE_DEADLINE_S - clock.tick)
    assert kernel.polls[0][1] <= RGB_CAPTURE_DEADLINE_S
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert kernel.count(VIDIOC_STREAMON) == 1
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert outcome.cleanup.ok is True
    assert outcome.succeeded is False
    assert all(m.closes == 1 for m in kernel.mappings)


@pytest.mark.parametrize(
    ("request_", "expected"),
    [
        pytest.param(VIDIOC_QUERYCAP, CaptureStatus.IOCTL_ERROR, id="querycap"),
        pytest.param(VIDIOC_G_FMT, CaptureStatus.IOCTL_ERROR, id="g_fmt"),
        pytest.param(VIDIOC_S_FMT, CaptureStatus.IOCTL_ERROR, id="s_fmt"),
        pytest.param(VIDIOC_REQBUFS, CaptureStatus.BUFFER_COUNT_MISMATCH, id="reqbufs"),
        pytest.param(VIDIOC_QUERYBUF, CaptureStatus.MAPPING_ERROR, id="querybuf"),
        pytest.param(VIDIOC_QBUF, CaptureStatus.QUEUE_ERROR, id="qbuf"),
        pytest.param(VIDIOC_STREAMON, CaptureStatus.STREAMON_ERROR, id="streamon"),
        pytest.param(VIDIOC_DQBUF, CaptureStatus.DEQUEUE_ERROR, id="dqbuf"),
    ],
)
def test_each_ioctl_failure_gets_its_own_status_and_is_never_retried(
    sysfs: SysfsBuilder, tmp_path: Path, request_: int, expected: CaptureStatus
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, errors={request_: errno.EINVAL})
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is expected, outcome.detail
    assert outcome.succeeded is False
    assert outcome.errno == errno.EINVAL
    assert kernel.count(request_) == 1
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_a_mapping_failure_is_reported_once_and_earlier_buffers_are_released(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE)
    real_mmap = kernel.mmap
    calls: list[int] = []

    def failing_mmap(fd: int, length: int, offset: int) -> FakeMapping:
        calls.append(offset)
        if len(calls) == 2:
            raise OSError(errno.ENOMEM, "Cannot allocate memory")
        return real_mmap(fd, length, offset)

    outcome = replace(rgb_backend(kernel, sysfs), mmap_fn=failing_mmap).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.MAPPING_ERROR
    assert kernel.count(VIDIOC_STREAMON) == 0
    assert [m.closes for m in kernel.mappings] == [1]
    assert outcome.cleanup.ok is True


def test_a_poll_that_raises_is_a_poll_error_never_retried(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE)

    def raising_poll(fd: int, timeout_s: float) -> int:
        kernel.polls.append((fd, timeout_s))
        raise OSError(errno.EIO, "Input/output error")

    outcome = replace(rgb_backend(kernel, sysfs), poll_fn=raising_poll).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.POLL_ERROR
    assert outcome.errno == errno.EIO
    assert len(kernel.polls) == 1
    assert kernel.count(VIDIOC_DQBUF) == 0
    assert kernel.count(VIDIOC_STREAMON) == 1
    assert outcome.cleanup.ok is True


# =================================================================================
# D. Cleanup on every failure/signal boundary
# =================================================================================


def test_a_streamoff_failure_overrides_an_otherwise_complete_rgb_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(
        profile=RGB_PROFILE,
        errors={VIDIOC_STREAMOFF: errno.EIO},
        frame_payload=baseline_jpeg(1280, 800),
    )
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1
    # Cleanup failing must not throw away the frame evidence that did arrive.
    assert outcome.frame_bytes == len(baseline_jpeg(1280, 800))
    assert outcome.artifacts.ok is True


def test_a_release_buffers_failure_overrides_an_otherwise_complete_rgb_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    real_ioctl = kernel.ioctl

    def failing(fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        if request == VIDIOC_REQBUFS:
            count = struct.unpack(REQUESTBUFFERS_FORMAT, bytes(packed))[0]
            if count == 0:  # the release, not the allocation
                raise OSError(errno.EBUSY, "Device or resource busy")
        return real_ioctl(fd, request, packed, mutate)

    outcome = replace(rgb_backend(kernel, sysfs), ioctl_fn=failing).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    assert len(kernel.closed) == 1


def test_a_close_failure_overrides_an_otherwise_complete_rgb_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))

    def failing_close(fd: int) -> None:
        kernel.closed.append(fd)
        raise OSError(errno.EIO, "Input/output error")

    outcome = replace(rgb_backend(kernel, sysfs), closer=failing_close).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    assert outcome.exit_code == CLEANUP_FAILED_EXIT


def test_an_unmap_that_raises_overrides_an_otherwise_complete_rgb_frame(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    real_mmap = kernel.mmap

    def exploding_mmap(fd: int, length: int, offset: int) -> _UnmapExplodes:
        return _UnmapExplodes(real_mmap(fd, length, offset))

    outcome = replace(rgb_backend(kernel, sysfs), mmap_fn=exploding_mmap).run(
        REAL_SERIAL, output_dir=tmp_path / "out"
    )
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.cleanup.ok is False
    assert outcome.succeeded is False
    assert outcome.exit_code == CLEANUP_FAILED_EXIT
    unmap_steps = [s for s in outcome.steps if s.step is Step.UNMAP]
    assert unmap_steps and all(s.ok is False for s in unmap_steps)
    # The frame itself still arrived and was still preserved.
    assert outcome.frame_bytes == len(baseline_jpeg(1280, 800))
    assert outcome.artifacts.ok is True


def test_every_cleanup_step_runs_once_on_a_healthy_rgb_run(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    outcome = run_rgb(kernel, sysfs, tmp_path)
    recorded = {s.step for s in outcome.steps}
    assert Step.STREAMOFF in recorded
    assert Step.UNMAP in recorded
    assert Step.RELEASE_BUFFERS in recorded
    assert Step.CLOSE in recorded
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert len([r for r in kernel.requests if r == VIDIOC_REQBUFS]) == 2  # allocate + release
    assert len(kernel.closed) == 1
    assert all(m.closes == 1 for m in kernel.mappings)


@pytest.mark.parametrize(
    "request_",
    [VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_QBUF, VIDIOC_STREAMON, VIDIOC_DQBUF],
)
def test_an_exception_raised_out_of_any_rgb_ioctl_still_releases_everything(
    sysfs: SysfsBuilder, tmp_path: Path, request_: int
) -> None:
    """``KeyboardInterrupt`` is a ``BaseException``, not an ``OSError``: it must
    unwind past the module and still leave nothing mapped or open."""
    kernel = FakeKernel(profile=RGB_PROFILE, raises={request_: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run_rgb(kernel, sysfs, tmp_path)
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


def test_a_signal_during_the_rgb_poll_still_releases_everything(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE)

    def interrupted_poll(fd: int, timeout_s: float) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        replace(rgb_backend(kernel, sysfs), poll_fn=interrupted_poll).run(
            REAL_SERIAL, output_dir=tmp_path / "out"
        )
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert all(m.closes == 1 for m in kernel.mappings)
    assert len(kernel.closed) == 1


# =================================================================================
# E. JPEG outcomes through the engine
# =================================================================================


def test_a_baseline_jpeg_gives_complete_and_writes_exactly_one_named_file(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    payload = baseline_jpeg(1280, 800)
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    written = list((tmp_path / "out").iterdir())
    assert [p.name for p in written] == [RGB_ARTIFACT_NAME]
    on_disk = written[0].read_bytes()
    assert on_disk == payload
    record = artifact(outcome)
    assert record.ok is True
    assert record.path == str(written[0])
    assert record.byte_count == len(payload)
    assert record.sha256 == hashlib.sha256(on_disk).hexdigest()


def test_bytesused_zero_gives_empty_frame_and_writes_no_file(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, bytesused=0)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.EMPTY_FRAME
    assert outcome.succeeded is False
    assert outcome.artifacts.files == ()
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_malformed_bytes_give_malformed_frame_and_still_preserve_them(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    payload = b"\xff\xd8not a jpeg"
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.MALFORMED_FRAME
    assert outcome.succeeded is False
    record = artifact(outcome)
    assert record.ok is True
    assert record.byte_count == len(payload)
    assert record.path is not None
    on_disk = Path(record.path).read_bytes()
    assert on_disk == payload
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_a_baseline_jpeg_of_the_wrong_size_gives_dimension_mismatch_and_still_preserves_it(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    payload = baseline_jpeg(640, 480)
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.DIMENSION_MISMATCH
    assert outcome.succeeded is False
    record = artifact(outcome)
    assert record.ok is True
    assert record.byte_count == len(payload)
    assert record.path is not None
    on_disk = Path(record.path).read_bytes()
    assert on_disk == payload
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_a_progressive_jpeg_is_also_complete(sysfs: SysfsBuilder, tmp_path: Path) -> None:
    payload = progressive_jpeg(1280, 800)
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    record = artifact(outcome)
    assert record.byte_count == len(payload)


def test_only_bytesused_bytes_are_preserved_never_the_whole_mapping(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    payload = baseline_jpeg(1280, 800)
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=payload)
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert outcome.frame_bytes == len(payload)
    assert len(payload) < kernel.buffer_length  # the mapping holds far more than the payload
    record = artifact(outcome)
    assert record.byte_count == len(payload)
    assert record.path is not None
    assert Path(record.path).stat().st_size == len(payload)


def test_the_evidence_never_claims_a_pixel_was_touched_and_the_json_round_trips(
    sysfs: SysfsBuilder, tmp_path: Path
) -> None:
    kernel = FakeKernel(profile=RGB_PROFILE, frame_payload=baseline_jpeg(1280, 800))
    outcome = run_rgb(kernel, sysfs, tmp_path)
    assert outcome.status is CaptureStatus.COMPLETE, outcome.detail
    assert outcome.frame_evidence is not None
    assert outcome.frame_evidence["pixels_decoded"] is False
    assert outcome.frame_evidence["colour_corrected"] is False
    assert outcome.frame_evidence["resized"] is False
    assert outcome.frame_evidence["calibration_applied"] is False

    blob = json.dumps(outcome.as_dict(), sort_keys=True)
    payload = json.loads(blob)
    assert payload["status"] == "complete"
    assert payload["succeeded"] is True
    assert payload["frame_evidence"]["pixels_decoded"] is False
    assert payload["plan"]["fourcc"] == "MJPG"
    assert payload["plan"]["width"] == RGB_FRAME_WIDTH
    assert payload["plan"]["height"] == RGB_FRAME_HEIGHT
    assert payload["plan"]["compressed"] is True
    assert payload["plan"]["max_dequeues"] == 1
    assert payload["plan"]["retries"] == 0
