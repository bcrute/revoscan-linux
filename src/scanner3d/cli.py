"""Command-line entry point for ``scanner3d``."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import struct
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, TextIO, cast

from scanner3d import __version__, paired_capture
from scanner3d.cloud_export import CloudExport, export_depth_frame
from scanner3d.control import (
    FIRMWARE_ALGORITHM_VERSION_READ,
    ControlOutcome,
    ControlRequest,
    TargetSnapshot,
    UsbfsControlBackend,
    VersionPayload,
)
from scanner3d.depth_metric import (
    DEPROJECTION_EXPRESSION,
    UNVALIDATED_UNIT,
    DepthIntrinsicsUnusable,
    DepthUnitAssumption,
)
from scanner3d.depth_scale import AUTHORIZED_SERIAL as DEPTH_SCALE_AUTHORIZED_SERIAL
from scanner3d.depth_scale import (
    DEPTH_SCALE_PROPERTY_ID,
    PROPERTY_PAYLOAD,
    PROPERTY_SELECTOR,
    RESULT_LENGTH,
    UNIT_DISCLAIMER,
    DepthScaleOutcome,
    UsbfsDepthScaleBackend,
)
from scanner3d.depth_scale import MAX_CONTROL_TRANSFERS as DEPTH_SCALE_MAX_TRANSFERS
from scanner3d.depth_scale import WHITELIST as DEPTH_SCALE_WHITELIST
from scanner3d.discovery import (
    DEFAULT_DEV_ROOT,
    DEFAULT_SYSFS_ROOT,
    EXPECTED_PRODUCT_ID,
    EXPECTED_VENDOR_ID,
    find_scanners,
)
from scanner3d.doctor import Check, DoctorReport, run_doctor
from scanner3d.errors import (
    EXIT_CODES,
    ControlRequestNotAllowed,
    DepthScaleReadNotAllowed,
    DoctorStatus,
    InitSequenceNotAllowed,
    InterfaceOwnershipNotAllowed,
    IntrinsicsReadNotAllowed,
    ReferenceProbeError,
    RgbExposureExtensionNotAllowed,
    RgbExposureInfoNotAllowed,
    RgbExposureLengthNotAllowed,
    ScannerError,
    TraceFormatError,
    V4l2CaptureNotAllowed,
)
from scanner3d.illumination_shutdown import (
    EMITTER_STATE_LINE,
    IlluminationShutdownOutcome,
    UsbfsIlluminationShutdownBackend,
)
from scanner3d.init_sequence import (
    ENTITY_ID,
    INITED_PATH,
    INITED_SEQUENCE,
    InitSequenceOutcome,
    UsbfsInitSequenceBackend,
)
from scanner3d.intrinsics import (
    AUTHORIZED_SERIAL,
    INTRINSICS_FORMAT,
    INTRINSICS_KEY,
    INTRINSICS_LENGTH,
    MAX_CHUNK_LENGTH,
    MAX_CONTROL_TRANSFERS,
    MAX_GET_TRANSFERS,
    OUTPUT_CAPACITY,
    WHITELIST,
    IntrinsicsOutcome,
    IntrinsicsRecord,
    UsbfsIntrinsicsBackend,
    parse_intrinsics,
)
from scanner3d.ownership import (
    EXPECTED_INTERFACE_DRIVER,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    OwnershipOutcome,
    OwnershipStatus,
    UsbfsOwnershipBackend,
    defer_termination,
)
from scanner3d.pointcloud import PlyFormat, PointCloudWriteIncomplete
from scanner3d.reference import (
    DEFAULT_MANIFEST,
    DEFAULT_PROBE_TIMEOUT_S,
    MAX_PROBE_TIMEOUT_S,
    ProbeResult,
    run_probe,
)
from scanner3d.rgb_exposure_extension import (
    AUTHORIZED_SERIAL as RGB_EXPOSURE_EXTENSION_AUTHORIZED_SERIAL,
)
from scanner3d.rgb_exposure_extension import (
    MAX_CONTROL_TRANSFERS as RGB_EXPOSURE_EXTENSION_MAX_TRANSFERS,
)
from scanner3d.rgb_exposure_extension import (
    READ_TRANSFER as RGB_EXPOSURE_EXTENSION_READ_TRANSFER,
)
from scanner3d.rgb_exposure_extension import (
    RgbExposureExtensionOutcome,
    UsbfsRgbExposureExtensionBackend,
)
from scanner3d.rgb_exposure_extension import (
    dry_run as rgb_exposure_extension_dry_run,
)
from scanner3d.rgb_exposure_info import (
    AUTHORIZED_SERIAL as RGB_EXPOSURE_INFO_AUTHORIZED_SERIAL,
)
from scanner3d.rgb_exposure_info import (
    MAX_CONTROL_TRANSFERS as RGB_EXPOSURE_INFO_MAX_TRANSFERS,
)
from scanner3d.rgb_exposure_info import (
    READ_TRANSFER as RGB_EXPOSURE_INFO_READ_TRANSFER,
)
from scanner3d.rgb_exposure_info import (
    RgbExposureInfoOutcome,
    UsbfsRgbExposureInfoBackend,
)
from scanner3d.rgb_exposure_info import (
    dry_run as rgb_exposure_info_dry_run,
)
from scanner3d.rgb_exposure_length import (
    AUTHORIZED_SERIAL as RGB_EXPOSURE_LENGTH_AUTHORIZED_SERIAL,
)
from scanner3d.rgb_exposure_length import (
    MAX_CONTROL_TRANSFERS as RGB_EXPOSURE_LENGTH_MAX_TRANSFERS,
)
from scanner3d.rgb_exposure_length import (
    READ_TRANSFER as RGB_EXPOSURE_LENGTH_READ_TRANSFER,
)
from scanner3d.rgb_exposure_length import (
    RgbExposureLengthOutcome,
    UsbfsRgbExposureLengthBackend,
)
from scanner3d.rgb_exposure_length import (
    dry_run as rgb_exposure_length_dry_run,
)
from scanner3d.rgb_frame import (
    RGB_ARTIFACT_NAME,
    RGB_CAPTURE_DEADLINE_S,
    RGB_DEFAULT_OUTPUT_DIR,
    RGB_DESCRIPTOR_FRAME_BUFFER_BYTES,
    RGB_EXECUTE_FLAG,
    RGB_FOURCC,
    RGB_FRAME_HEIGHT,
    RGB_FRAME_WIDTH,
    RGB_MAX_TOTAL_MAPPED_BYTES,
    RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES,
    RGB_PROFILE,
    RGB_REQUESTED_BUFFERS,
    RGB_ROLE,
    V4l2RgbFrameBackend,
)
from scanner3d.session import (
    MANIFEST_NAME,
    DeviceIdentity,
    SessionError,
    SessionState,
    SessionStore,
    StageState,
    read_manifest,
)
from scanner3d.usbfs_trace import load_trace, render_human
from scanner3d.v4l2_capture import (
    ALLOWED_IOCTLS,
    CAPTURE_DEADLINE_S,
    CAPTURE_ROLE,
    DEFAULT_OUTPUT_DIR,
    DEPTH_PROFILE,
    EXPECTED_FRAME_BYTES,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    REQUESTED_BUFFERS,
    CaptureOutcome,
    CaptureProfile,
    NodeSnapshot,
    V4l2DepthFrameBackend,
    V4l2FrameBackend,
)
from scanner3d.v4l2_capture import AUTHORIZED_SERIAL as DEPTH_FRAME_AUTHORIZED_SERIAL
from scanner3d.v4l2_controls import (
    AUTHORIZED_SERIAL as RGB_CONTROLS_AUTHORIZED_SERIAL,
)
from scanner3d.v4l2_controls import (
    RGB_CONTROLS_EXECUTE_OPT_IN,
    V4l2RgbControlBackend,
)
from scanner3d.v4l2_controls import (
    render_human as render_rgb_controls_human,
)


def _symbol(check: Check) -> str:
    if not check.ran:
        return "SKIP"
    return "PASS" if check.status is DoctorStatus.OK else "FAIL"


def _render_human(report: DoctorReport, stream: TextIO) -> None:
    env = report.environment
    print(f"scanner3d doctor {__version__}", file=stream)
    print(f"  python {env['python_version']} on {env['kernel']} as uid {env['uid']}", file=stream)
    print(file=stream)

    for device in report.devices:
        print(
            f"Device {device.vendor_id:04x}:{device.product_id:04x} "
            f"serial {device.serial} at {device.usb_path}",
            file=stream,
        )
        print(f"  link      {device.usb_speed_label}", file=stream)
        print(f"  usbfs     {device.usbfs.path}", file=stream)
        for fn in device.functions:
            nodes = ", ".join(f"{n.name}[{n.kind}]" for n in fn.video_nodes) or "none"
            print(
                f"  {fn.role:<9} interface {fn.interface_number} "
                f"({fn.interface_string}) -> {nodes}",
                file=stream,
            )
            for fmt in fn.formats:
                modes = ", ".join(
                    f"{m.width}x{m.height}@{m.default_frame_rate_hz}" for m in fmt.modes
                )
                print(f"    {fmt.fourcc:<6} {modes}", file=stream)
        print(file=stream)

    print("Checks", file=stream)
    for check in report.checks:
        print(f"  [{_symbol(check):<4}] {check.name:<22} {check.detail}", file=stream)
    print(file=stream)
    print(f"Status: {report.status}", file=stream)


def _maybe_run_probe(
    args: argparse.Namespace, stream: TextIO
) -> tuple[ProbeResult | None, str | None]:
    """Run the reference probe if opted in; return (result, skip reason)."""
    if not args.reference_probe:
        return None, "not requested (pass --reference-probe to opt in)"
    serial = args.serial
    if serial is None:
        devices = find_scanners(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
        if not devices or not devices[0].serial:
            return None, "no attached device serial to target"
        serial = devices[0].serial
    try:
        return (
            run_probe(
                serial,
                sdk_root=args.sdk_root,
                probe_binary=args.probe_binary,
                timeout_s=args.probe_timeout,
                manifest_path=args.sdk_manifest,
                require_verified_sdk=not args.allow_unverified_sdk,
                json_out=args.probe_json_out,
            ),
            None,
        )
    except ReferenceProbeError as exc:
        print(f"reference probe unavailable: {exc}", file=stream)
        return None, str(exc)


def _cmd_doctor(args: argparse.Namespace) -> int:
    probe, skip_reason = _maybe_run_probe(args, sys.stderr)
    report = run_doctor(
        sysfs_root=args.sysfs_root,
        dev_root=args.dev_root,
        probe=probe,
        probe_skipped_reason=skip_reason,
    )
    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    else:
        _render_human(report, sys.stdout)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return EXIT_CODES[report.status]


def _cmd_list(args: argparse.Namespace) -> int:
    devices = find_scanners(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    if args.json:
        json.dump([d.as_dict() for d in devices], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if not devices:
            print("no matching scanner found", file=sys.stdout)
        for device in devices:
            print(
                f"{device.usb_path}  {device.vendor_id:04x}:{device.product_id:04x}  "
                f"{device.serial}  {device.usb_speed_label}",
                file=sys.stdout,
            )
    return 0 if devices else EXIT_CODES[DoctorStatus.DEVICE_ABSENT]


def _cmd_trace(args: argparse.Namespace) -> int:
    """Decode a usbfs trace offline. Never touches hardware or the SDK."""
    try:
        trace = load_trace(args.path)
    except TraceFormatError as exc:
        print(f"not a usable trace: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    except OSError as exc:
        print(f"cannot read trace: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = trace.as_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_human(trace))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _render_request(request: ControlRequest, stream: TextIO) -> None:
    """The setup packet block, shared by both firmware-version modes."""
    print("Request (the only one this command can send)", file=stream)
    print(
        f"  bmRequestType=0x{request.request_type:02x} bRequest=0x{request.request:02x} "
        f"wValue=0x{request.value:04x} wIndex=0x{request.index:04x} "
        f"wLength={request.length} timeout={request.timeout_ms} ms",
        file=stream,
    )
    print(
        f"  direction  {'device -> host (read-only)' if request.is_read else 'host -> device'}",
        file=stream,
    )
    print(
        f"  addresses  interface {request.interface_number}, entity "
        f"{request.entity_id}, control selector 0x{request.control_selector:02x}",
        file=stream,
    )
    print(file=stream)


def _render_target(target: TargetSnapshot, stream: TextIO) -> None:
    """The resolved usbfs node block, shared by both firmware-version modes."""
    print(f"Target {target.serial} at {target.usb_path}", file=stream)
    print(f"  node       {target.node_path}", file=stream)
    print(
        f"  rdev       {target.rdev_major}:{target.rdev_minor} "
        f"(inode {target.inode}, mode {target.mode})",
        file=stream,
    )
    print(
        f"  interface  {target.interface_number} kernel driver {target.interface_driver or 'none'}",
        file=stream,
    )


def _render_payload(payload: VersionPayload, stream: TextIO) -> None:
    """The decoded response block, shared by both firmware-version modes."""
    print("Response", file=stream)
    print(f"  bytes      {payload.raw_length}", file=stream)
    print(f"  sha256     {payload.sha256}", file=stream)
    print(f"  text       {payload.text!r}", file=stream)
    print(
        f"  firmware   {payload.firmware_version} "
        f"(expected {payload.as_dict()['expected_firmware_version']})",
        file=stream,
    )
    print(
        f"  algorithm  {payload.algorithm_version} "
        f"(expected {payload.as_dict()['expected_algorithm_version']})",
        file=stream,
    )
    print(f"  nul padded {payload.nul_padded}", file=stream)
    print(f"  validated  {payload.matches_expected}", file=stream)
    print(file=stream)


def _render_control(outcome: ControlOutcome, stream: TextIO) -> None:
    """Human rendering of the one owned control read."""
    print(f"scanner3d firmware-version {__version__}", file=stream)
    print(file=stream)
    _render_request(outcome.request, stream)

    target = outcome.target
    if target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_target(target, stream)
    entity = outcome.addressed_entity
    if entity is None:
        print(f"  entity     unresolved - {outcome.entity_lookup_note}", file=stream)
    else:
        sources = ",".join(str(s) for s in entity.source_ids) or "none"
        print(
            f"  entity     id {entity.entity_id} subtype {entity.subtype_name} "
            f"(guid {entity.guid or 'n/a'}, sources {sources})",
            file=stream,
        )
    print(file=stream)

    if outcome.payload is not None:
        _render_payload(outcome.payload, stream)
    if outcome.duration_s is not None:
        print(f"Duration: {outcome.duration_s * 1000:.3f} ms", file=stream)
    if outcome.errno is not None:
        print(f"Errno: {outcome.errno} ({outcome.errno_name})", file=stream)
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)


def _render_ownership(outcome: OwnershipOutcome, stream: TextIO) -> None:
    """Human rendering of the temporary interface-ownership lifecycle."""
    print(f"scanner3d firmware-version {__version__} (temporary interface ownership)", file=stream)
    print(file=stream)
    print(
        f"Lifecycle (fixed): confirm {EXPECTED_INTERFACE_DRIVER} -> disconnect -> claim -> "
        f"one read -> release -> reconnect, all on interface {OWNED_INTERFACE}",
        file=stream,
    )
    print(file=stream)
    _render_request(outcome.request, stream)

    if outcome.target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_target(outcome.target, stream)
    print(file=stream)

    print("Phases", file=stream)
    if not outcome.phases:
        print("  none attempted", file=stream)
    for phase in outcome.phases:
        mark = "ok  " if phase.ok else "FAIL"
        timing = "" if phase.duration_s is None else f" [{phase.duration_s * 1000:.3f} ms]"
        errno = "" if phase.errno is None else f" ({phase.errno_name})"
        print(f"  [{mark}] {phase.phase!s:<14} {phase.detail}{errno}{timing}", file=stream)
    print(file=stream)

    if outcome.payload is not None:
        _render_payload(outcome.payload, stream)

    cleanup = outcome.cleanup
    print("Cleanup", file=stream)
    print(f"  attempted  {cleanup.attempted}", file=stream)
    print(f"  restored   {cleanup.ok}", file=stream)
    print(f"  note       {cleanup.note}", file=stream)
    if cleanup.guidance is not None:
        print(f"  RECOVERY   {cleanup.guidance}", file=stream)
    print(file=stream)

    if outcome.transfer_duration_s is not None:
        print(f"Transfer duration: {outcome.transfer_duration_s * 1000:.3f} ms", file=stream)
    if outcome.total_duration_s is not None:
        print(f"Total duration: {outcome.total_duration_s * 1000:.3f} ms", file=stream)
    if outcome.errno is not None:
        print(f"Errno: {outcome.errno} ({outcome.errno_name})", file=stream)
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)


def _render_init_sequence(outcome: InitSequenceOutcome, stream: TextIO) -> None:
    """Human rendering of the four-transfer /tmp/inited sequence."""
    print(f"scanner3d inited-sequence {__version__}", file=stream)
    print(file=stream)
    print(
        f"Sequence (fixed): the four transfers the vendor SDK was observed making to "
        f"entity {ENTITY_ID} on interface {OWNED_INTERFACE} for {INITED_PATH!r}",
        file=stream,
    )
    for index, transfer in enumerate(INITED_SEQUENCE, start=1):
        request = transfer.request
        direction = "device -> host" if request.is_read else "host -> device"
        print(
            f"  {index}. {transfer.step!s:<12} bmRequestType=0x{request.request_type:02x} "
            f"bRequest=0x{request.request:02x} wValue=0x{request.value:04x} "
            f"wIndex=0x{request.index:04x} wLength={request.length} "
            f"timeout={request.timeout_ms} ms",
            file=stream,
        )
        payload = "none (read)" if transfer.payload is None else transfer.payload[:16].hex()
        print(f"     {direction:<14} payload {payload}  {transfer.note}", file=stream)
    print(file=stream)

    if outcome.target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_target(outcome.target, stream)
    print(file=stream)

    print("Phases", file=stream)
    if not outcome.phases:
        print("  none attempted", file=stream)
    for phase in outcome.phases:
        mark = "ok  " if phase.ok else "FAIL"
        timing = "" if phase.duration_s is None else f" [{phase.duration_s * 1000:.3f} ms]"
        errno = "" if phase.errno is None else f" ({phase.errno_name})"
        print(f"  [{mark}] {phase.phase!s:<18} {phase.detail}{errno}{timing}", file=stream)
    print(file=stream)

    print("Transfers", file=stream)
    if not outcome.transfers:
        print("  none attempted", file=stream)
    for result in outcome.transfers:
        mark = "ok  " if result.ok else "FAIL"
        timing = "" if result.duration_s is None else f" [{result.duration_s * 1000:.3f} ms]"
        print(f"  [{mark}] {result.step!s:<12} {result.detail}{timing}", file=stream)
        print(f"       matched observed  {result.matched}", file=stream)
        if result.sent_sha256 is not None:
            print(f"       sent sha256       {result.sent_sha256}", file=stream)
        if result.errno is not None:
            print(f"       errno             {result.errno} ({result.errno_name})", file=stream)
        response = result.response
        if response is not None:
            print(f"       bytes             {response.raw_length}", file=stream)
            print(f"       sha256            {response.sha256}", file=stream)
            print(f"       leading hex       {response.hex_prefix}", file=stream)
            print(f"       all zero          {response.all_zero}", file=stream)
            print(f"       declared length   {response.declared_length}", file=stream)
            print(f"       text              {response.text!r}", file=stream)
            print(f"       nul padded        {response.nul_padded}", file=stream)
            if response.decode_error is not None:
                print(f"       decode note       {response.decode_error}", file=stream)
    print(file=stream)

    protocol = outcome.protocol_cleanup
    print("Protocol cleanup (the clearing SET_CUR)", file=stream)
    print(f"  attempted  {protocol.attempted}", file=stream)
    print(f"  cleared    {protocol.ok}", file=stream)
    print(f"  note       {protocol.note}", file=stream)
    if protocol.guidance is not None:
        print(f"  RECOVERY   {protocol.guidance}", file=stream)
    print(file=stream)

    cleanup = outcome.cleanup
    print("Driver cleanup (release and reconnect)", file=stream)
    print(f"  attempted  {cleanup.attempted}", file=stream)
    print(f"  restored   {cleanup.ok}", file=stream)
    print(f"  note       {cleanup.note}", file=stream)
    if cleanup.guidance is not None:
        print(f"  RECOVERY   {cleanup.guidance}", file=stream)
    print(file=stream)

    if outcome.sequence_duration_s is not None:
        print(f"Sequence duration: {outcome.sequence_duration_s * 1000:.3f} ms", file=stream)
    if outcome.total_duration_s is not None:
        print(f"Total duration: {outcome.total_duration_s * 1000:.3f} ms", file=stream)
    if outcome.errno is not None:
        print(f"Errno: {outcome.errno} ({outcome.errno_name})", file=stream)
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)


def _render_intrinsics(outcome: IntrinsicsOutcome, stream: TextIO) -> None:
    """Human rendering of the bounded depth-intrinsics key read."""
    print(f"scanner3d depth-intrinsics {__version__}", file=stream)
    print(file=stream)
    print(
        "Whitelist (fixed): the three transfers of the recovered libuvcReadFile "
        f"protocol, for {INTRINSICS_KEY!r} and no other key",
        file=stream,
    )
    for index, transfer in enumerate(WHITELIST, start=1):
        request = transfer.request
        direction = "device -> host" if request.is_read else "host -> device"
        print(
            f"  {index}. {transfer.step!s:<11} bmRequestType=0x{request.request_type:02x} "
            f"bRequest=0x{request.request:02x} wValue=0x{request.value:04x} "
            f"wIndex=0x{request.index:04x} wLength={request.length} "
            f"timeout={request.timeout_ms} ms",
            file=stream,
        )
        payload = "none (read)" if transfer.payload is None else transfer.payload[:16].hex()
        print(f"     {direction:<14} payload {payload}  {transfer.note}", file=stream)
    print(file=stream)

    print("Bounds (fixed at import; never widened at runtime)", file=stream)
    print(f"  output capacity   {OUTPUT_CAPACITY} bytes", file=stream)
    print(f"  result reads      at most {MAX_GET_TRANSFERS}", file=stream)
    print(f"  control transfers at most {MAX_CONTROL_TRANSFERS}", file=stream)
    print(f"  declared chunk    at most {MAX_CHUNK_LENGTH} bytes", file=stream)
    print("  retries           none; every transfer is attempted at most once", file=stream)
    print(file=stream)

    if outcome.target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_target(outcome.target, stream)
    print(file=stream)

    print("Phases", file=stream)
    if not outcome.phases:
        print("  none attempted", file=stream)
    for phase in outcome.phases:
        mark = "ok  " if phase.ok else "FAIL"
        timing = "" if phase.duration_s is None else f" [{phase.duration_s * 1000:.3f} ms]"
        errno = "" if phase.errno is None else f" ({phase.errno_name})"
        print(f"  [{mark}] {phase.phase!s:<18} {phase.detail}{errno}{timing}", file=stream)
    print(file=stream)

    print(
        f"Transfers ({outcome.get_transfers} of at most {MAX_GET_TRANSFERS} result reads)",
        file=stream,
    )
    if not outcome.transfers:
        print("  none attempted", file=stream)
    for result in outcome.transfers:
        mark = "ok  " if result.ok else "FAIL"
        timing = "" if result.duration_s is None else f" [{result.duration_s * 1000:.3f} ms]"
        print(f"  [{mark}] {result.step!s:<11} {result.detail}{timing}", file=stream)
        if result.sent_sha256 is not None:
            print(f"       sent sha256       {result.sent_sha256}", file=stream)
        if result.transferred is not None:
            print(f"       transferred       {result.transferred} bytes", file=stream)
        if result.response_sha256 is not None:
            print(f"       reply sha256      {result.response_sha256}", file=stream)
            print(f"       reply hex         {result.response_hex}", file=stream)
        if result.declared_length is not None:
            print(f"       declared length   {result.declared_length}", file=stream)
        if result.chunk_sha256 is not None:
            print(f"       chunk sha256      {result.chunk_sha256}", file=stream)
        if result.copied_bytes is not None:
            print(f"       copied            {result.copied_bytes} bytes", file=stream)
        if result.errno is not None:
            print(f"       errno             {result.errno} ({result.errno_name})", file=stream)
    print(file=stream)

    blob = outcome.blob
    print("Raw blob (the primary evidence, whatever the parse says)", file=stream)
    print(f"  length     {blob.length} bytes", file=stream)
    print(f"  sha256     {blob.sha256}", file=stream)
    print(f"  hex        {blob.hex or '(empty)'}", file=stream)
    print(f"  terminated {blob.terminated}", file=stream)
    print(f"  complete   {blob.complete_answer}", file=stream)
    print(f"  note       {blob.note}", file=stream)
    print(file=stream)

    print("Advertised depth modes (from this device's own descriptors)", file=stream)
    if not outcome.depth_modes:
        print("  none read", file=stream)
    for width, height in outcome.depth_modes:
        print(f"  {width}x{height}", file=stream)
    print(file=stream)

    record = outcome.record
    print(f"Record (vendor Types.hpp Intrinsics, {INTRINSICS_LENGTH} bytes)", file=stream)
    if record is None:
        print("  not parsed: no complete 40-byte blob was read", file=stream)
    else:
        values = record.values
        print(f"  frame      {record.width}x{record.height}", file=stream)
        print(
            f"  fx {values['fx']!r}  fy {values['fy']!r}  cx {values['cx']!r}  cy {values['cy']!r}",
            file=stream,
        )
        print("  units      fx, fy, cx and cy are in pixels of the frame above", file=stream)
        print("  matrix", file=stream)
        for row in record.matrix:
            print("    [" + "  ".join(f"{value!r}" for value in row) + "]", file=stream)
        print("  structure", file=stream)
        for check in record.structure:
            print(
                f"    [{'ok  ' if check.ok else 'FAIL'}] {check.name}: {check.detail}", file=stream
            )
        print("  plausibility (shape only; claims no metric accuracy)", file=stream)
        for check in record.plausibility:
            print(
                f"    [{'ok  ' if check.ok else 'note'}] {check.name}: {check.detail}", file=stream
            )
    print(f"  accepted calibration  {outcome.accepted_calibration is not None}", file=stream)
    print(file=stream)

    protocol = outcome.protocol_cleanup
    print("Protocol cleanup (the clearing SET_CUR)", file=stream)
    print(f"  attempted  {protocol.attempted}", file=stream)
    print(f"  cleared    {protocol.ok}", file=stream)
    print(f"  note       {protocol.note}", file=stream)
    if protocol.guidance is not None:
        print(f"  RECOVERY   {protocol.guidance}", file=stream)
    print(file=stream)

    cleanup = outcome.cleanup
    print("Driver cleanup (release and reconnect)", file=stream)
    print(f"  attempted  {cleanup.attempted}", file=stream)
    print(f"  restored   {cleanup.ok}", file=stream)
    print(f"  note       {cleanup.note}", file=stream)
    if cleanup.guidance is not None:
        print(f"  RECOVERY   {cleanup.guidance}", file=stream)
    print(file=stream)

    if outcome.read_duration_s is not None:
        print(f"Read duration: {outcome.read_duration_s * 1000:.3f} ms", file=stream)
    if outcome.total_duration_s is not None:
        print(f"Total duration: {outcome.total_duration_s * 1000:.3f} ms", file=stream)
    if outcome.errno is not None:
        print(f"Errno: {outcome.errno} ({outcome.errno_name})", file=stream)
    print(f"Read outcome: {outcome.read_outcome}", file=stream)
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)


def _render_depth_scale(outcome: DepthScaleOutcome, stream: TextIO) -> None:
    """Human rendering of the bounded vendor depth-scale property read."""
    print(f"scanner3d depth-scale {__version__}", file=stream)
    print(file=stream)
    print(
        "Whitelist (fixed): the two transfers of the recovered libuvcReadValueExtension "
        f"pair, for property 0x{DEPTH_SCALE_PROPERTY_ID:03x} and no other",
        file=stream,
    )
    for index, transfer in enumerate(DEPTH_SCALE_WHITELIST, start=1):
        request = transfer.request
        direction = "device -> host" if request.is_read else "host -> device"
        print(
            f"  {index}. {transfer.step!s:<14} bmRequestType=0x{request.request_type:02x} "
            f"bRequest=0x{request.request:02x} wValue=0x{request.value:04x} "
            f"wIndex=0x{request.index:04x} wLength={request.length} "
            f"timeout={request.timeout_ms} ms",
            file=stream,
        )
        payload = "none (read)" if transfer.payload is None else transfer.payload.hex()
        print(f"     {direction:<14} payload {payload}  {transfer.note}", file=stream)
    print(file=stream)

    print("Bounds (fixed at import; never widened at runtime)", file=stream)
    print(f"  control transfers exactly {DEPTH_SCALE_MAX_TRANSFERS}", file=stream)
    print(f"  property payload  {len(PROPERTY_PAYLOAD)} bytes", file=stream)
    print(f"  reply             exactly {RESULT_LENGTH} bytes", file=stream)
    print("  retries           none; every transfer is attempted at most once", file=stream)
    print(file=stream)

    if outcome.target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_target(outcome.target, stream)
    print(file=stream)

    print("Phases", file=stream)
    if not outcome.phases:
        print("  none attempted", file=stream)
    for phase in outcome.phases:
        mark = "ok  " if phase.ok else "FAIL"
        timing = "" if phase.duration_s is None else f" [{phase.duration_s * 1000:.3f} ms]"
        errno = "" if phase.errno is None else f" ({phase.errno_name})"
        print(f"  [{mark}] {phase.phase!s:<18} {phase.detail}{errno}{timing}", file=stream)
    print(file=stream)

    print(
        f"Transfers ({len(outcome.transfers)} of at most {DEPTH_SCALE_MAX_TRANSFERS})", file=stream
    )
    if not outcome.transfers:
        print("  none attempted", file=stream)
    for result in outcome.transfers:
        mark = "ok  " if result.ok else "FAIL"
        timing = "" if result.duration_s is None else f" [{result.duration_s * 1000:.3f} ms]"
        print(f"  [{mark}] {result.step!s:<14} {result.detail}{timing}", file=stream)
        if result.sent_hex is not None:
            print(f"       sent hex          {result.sent_hex}", file=stream)
            print(f"       sent sha256       {result.sent_sha256}", file=stream)
        if result.transferred is not None:
            print(f"       transferred       {result.transferred} bytes", file=stream)
        if result.response_sha256 is not None:
            print(f"       reply sha256      {result.response_sha256}", file=stream)
            print(f"       reply hex         {result.response_hex}", file=stream)
        if result.errno is not None:
            print(f"       errno             {result.errno} ({result.errno_name})", file=stream)
    print(file=stream)

    reply = outcome.reply
    print("Raw reply (the primary evidence, whatever the decode says)", file=stream)
    print(
        f"  length     {reply.length} bytes (exact {RESULT_LENGTH} required: {reply.exact_length})",
        file=stream,
    )
    print(f"  sha256     {reply.sha256}", file=stream)
    print(f"  hex        {reply.hex or '(empty)'}", file=stream)
    print(f"  complete   {reply.complete_answer}", file=stream)
    print(f"  note       {reply.note}", file=stream)
    print(file=stream)

    reading = outcome.reading
    print("Decode", file=stream)
    if reading is None:
        print(f"  not decoded: no exact {RESULT_LENGTH}-byte reply was read", file=stream)
    else:
        marker_mark = "ok  " if reading.marker_uint8 == 4 else "FAIL"
        print(
            f"  [{marker_mark}] marker byte 59    uint8 {reading.marker_uint8}, "
            f"int8 {reading.marker_int8} (accepted only when 4)",
            file=stream,
        )
        print(
            f"       denominator       {reading.denominator} "
            "(bytes 0:4, signed little-endian int32, the cvtsi2ss operand)",
            file=stream,
        )
        if reading.scale is None:
            print("       vendor scale      none: this reply yields no scale", file=stream)
        else:
            print(f"       vendor scale      {reading.scale!r}", file=stream)
            print(
                f"       float32 bits      0x{reading.scale_bits:08x}  "
                "= float32(1.0f / float32(denominator))",
                file=stream,
            )
        print(f"       outcome           {reading.outcome}", file=stream)
        print(f"       detail            {reading.detail}", file=stream)
        reference = reading.reference
        if reference is not None:
            print(
                f"       SDK fallback      {reference.fallback!r} "
                f"(0x{reference.fallback_bits:08x}); matches exactly: {reference.matches}",
                file=stream,
            )
            if reference.ratio is not None:
                print(f"       read / fallback   {reference.ratio!r}", file=stream)
            print(f"       reference note    {reference.note}", file=stream)
    print(f"  UNIT       {UNIT_DISCLAIMER}", file=stream)
    print(f"  accepted scale        {outcome.accepted_scale!r}", file=stream)
    print(file=stream)

    protocol = outcome.protocol_cleanup
    print("Protocol cleanup", file=stream)
    print(f"  attempted  {protocol.attempted}", file=stream)
    print(f"  ok         {protocol.ok}", file=stream)
    print(f"  note       {protocol.note}", file=stream)
    print(file=stream)

    cleanup = outcome.cleanup
    print("Driver cleanup (release and reconnect)", file=stream)
    print(f"  attempted  {cleanup.attempted}", file=stream)
    print(f"  restored   {cleanup.ok}", file=stream)
    print(f"  note       {cleanup.note}", file=stream)
    if cleanup.guidance is not None:
        print(f"  RECOVERY   {cleanup.guidance}", file=stream)
    print(file=stream)

    if outcome.read_duration_s is not None:
        print(f"Read duration: {outcome.read_duration_s * 1000:.3f} ms", file=stream)
    if outcome.total_duration_s is not None:
        print(f"Total duration: {outcome.total_duration_s * 1000:.3f} ms", file=stream)
    if outcome.errno is not None:
        print(f"Errno: {outcome.errno} ({outcome.errno_name})", file=stream)
    print(f"Scale outcome: {outcome.scale_outcome}", file=stream)
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)


def _render_rgb_exposure_extension(outcome: RgbExposureExtensionOutcome, stream: TextIO) -> None:
    """Human rendering of the fixed, read-only RGB extension plan/result."""
    print(f"scanner3d rgb-exposure-extension {__version__}", file=stream)
    print("Fixed transfer (the only expressible operation)", file=stream)
    request = RGB_EXPOSURE_EXTENSION_READ_TRANSFER.request
    print(
        f"  bmRequestType=0x{request.request_type:02x} bRequest=0x{request.request:02x} "
        f"wValue=0x{request.value:04x} wIndex=0x{request.index:04x} "
        f"wLength={request.length} timeout={request.timeout_ms} ms",
        file=stream,
    )
    print(
        f"  transfers at most {RGB_EXPOSURE_EXTENSION_MAX_TRANSFERS}; retries none; "
        "GET_CUR only (no SET or caller-controlled fields)",
        file=stream,
    )
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)
    print(f"Extension outcome: {outcome.extension_outcome}", file=stream)
    print(f"Exit: {outcome.exit_code}", file=stream)
    if outcome.reply.actual_length is not None or outcome.reply.raw_length:
        print(
            f"Reply: requested={outcome.reply.requested_length} actual="
            f"{outcome.reply.actual_length} retained={outcome.reply.raw_length} "
            f"sha256={outcome.reply.sha256}",
            file=stream,
        )
        print(f"  raw hex: {outcome.reply.raw_hex}", file=stream)
    if outcome.reading is None:
        print("Decode: not available until an exact 60-byte reply is received", file=stream)
    else:
        reading = outcome.reading
        print(f"Raw words (LE u32 at offsets 0,4,8,12): {list(reading.words)}", file=stream)
        mapping = reading.mapping
        print(
            f"SDK mapping: current={mapping.current} minimum={mapping.minimum} "
            f"maximum={mapping.maximum} step={mapping.step} (unit unknown)",
            file=stream,
        )
        print("word3: logged by SDK; not returned as step", file=stream)
        for warning in reading.warnings:
            print(f"WARNING {warning.code}: {warning.detail}", file=stream)
    print(f"Transfers recorded: {len(outcome.transfers)}", file=stream)
    for transfer in outcome.transfers:
        print(
            f"  {transfer.step}: {'ok' if transfer.ok else 'FAIL'} "
            f"transferred={transfer.transferred} {transfer.detail}",
            file=stream,
        )
    print(f"Cleanup: {outcome.cleanup.note} (ok={outcome.cleanup.ok})", file=stream)


def _cmd_rgb_exposure_extension(args: argparse.Namespace) -> int:
    """Read the fixed RGB extension state once, offline unless explicitly opted in."""
    if args.execute_rgb_exposure_extension_read and not args.serial:
        print(
            "--execute-rgb-exposure-extension-read requires an explicit --serial: "
            "this mode temporarily owns interface 0 and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    if not args.execute_rgb_exposure_extension_read:
        outcome = rgb_exposure_extension_dry_run(args.serial)
    else:
        backend = rgb_exposure_extension_backend_factory(args.sysfs_root, args.dev_root)
        try:
            outcome = backend.run(args.serial)
        except (RgbExposureExtensionNotAllowed, InterfaceOwnershipNotAllowed) as exc:
            print(f"refusing that transfer: {exc}", file=sys.stderr)
            return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_rgb_exposure_extension(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}",
            file=sys.stderr,
        )
    return outcome.exit_code


def _default_rgb_exposure_extension_backend(
    sysfs_root: Path, dev_root: Path
) -> UsbfsRgbExposureExtensionBackend:
    return UsbfsRgbExposureExtensionBackend(sysfs_root=sysfs_root, dev_root=dev_root)


rgb_exposure_extension_backend_factory: Callable[[Path, Path], UsbfsRgbExposureExtensionBackend] = (
    _default_rgb_exposure_extension_backend
)


def _render_rgb_exposure_length(outcome: RgbExposureLengthOutcome, stream: TextIO) -> None:
    """Render the fixed GET_LEN plan/result without assigning semantics."""
    print(f"scanner3d rgb-exposure-length {__version__}", file=stream)
    print("Fixed transfer (the only expressible operation)", file=stream)
    request = RGB_EXPOSURE_LENGTH_READ_TRANSFER.request
    print(
        f"  bmRequestType=0x{request.request_type:02x} bRequest=0x{request.request:02x} "
        f"wValue=0x{request.value:04x} wIndex=0x{request.index:04x} "
        f"wLength={request.length} timeout={request.timeout_ms} ms",
        file=stream,
    )
    print(
        f"  transfers at most {RGB_EXPOSURE_LENGTH_MAX_TRANSFERS}; retries none; "
        "GET_LEN only (no exposure or other setting change)",
        file=stream,
    )
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)
    print(f"Length outcome: {outcome.length_outcome}", file=stream)
    print(f"Exit: {outcome.exit_code}", file=stream)
    reply = outcome.reply
    print(
        f"Reply: requested={reply.requested_length} actual={reply.actual_length} "
        f"retained={reply.raw_length} sha256={reply.sha256}",
        file=stream,
    )
    print(
        f"  raw hex: {reply.raw_hex or '(empty)'}, decoded raw value: {reply.decoded_raw_value}",
        file=stream,
    )
    if outcome.reading is None:
        print("Decode: unavailable until an exact two-byte reply is received", file=stream)
    else:
        print(
            f"Decoded GET_LEN value: {outcome.reading.reported_control_length} "
            "(raw only; not support, exposure, units, success or failure)",
            file=stream,
        )
    print(
        "Limitation: GET_LEN reports only a raw control length; it is not exposure evidence",
        file=stream,
    )
    print(f"Transfers recorded: {len(outcome.transfers)}", file=stream)
    for transfer in outcome.transfers:
        print(
            f"  {transfer.step}: {'ok' if transfer.ok else 'FAIL'} "
            f"transferred={transfer.transferred} {transfer.detail}",
            file=stream,
        )
    print(f"Cleanup: {outcome.cleanup.note} (ok={outcome.cleanup.ok})", file=stream)
    if outcome.cleanup.guidance is not None:
        print(f"RECOVERY: {outcome.cleanup.guidance}", file=stream)


def _cmd_rgb_exposure_length(args: argparse.Namespace) -> int:
    """Read the fixed GET_LEN once, offline unless explicitly opted in."""
    if args.execute_rgb_exposure_length_read and not args.serial:
        print(
            "--execute-rgb-exposure-length-read requires an explicit --serial: "
            "this mode temporarily owns interface 0 and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    if (
        args.execute_rgb_exposure_length_read
        and args.serial != RGB_EXPOSURE_LENGTH_AUTHORIZED_SERIAL
    ):
        print(
            "refusing that transfer: this read is authorized for serial "
            f"{RGB_EXPOSURE_LENGTH_AUTHORIZED_SERIAL!r} only; refusing {args.serial!r}",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    if not args.execute_rgb_exposure_length_read:
        outcome = rgb_exposure_length_dry_run(args.serial)
    else:
        backend = rgb_exposure_length_backend_factory(args.sysfs_root, args.dev_root)
        try:
            outcome = backend.run(args.serial)
        except (RgbExposureLengthNotAllowed, InterfaceOwnershipNotAllowed) as exc:
            print(f"refusing that transfer: {exc}", file=sys.stderr)
            return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    payload = outcome.as_dict()
    if not args.json:
        _render_rgb_exposure_length(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}",
            file=sys.stderr,
        )
    return outcome.exit_code


def _default_rgb_exposure_length_backend(
    sysfs_root: Path, dev_root: Path
) -> UsbfsRgbExposureLengthBackend:
    return UsbfsRgbExposureLengthBackend(sysfs_root=sysfs_root, dev_root=dev_root)


rgb_exposure_length_backend_factory: Callable[[Path, Path], UsbfsRgbExposureLengthBackend] = (
    _default_rgb_exposure_length_backend
)


def _render_rgb_exposure_info(outcome: RgbExposureInfoOutcome, stream: TextIO) -> None:
    """Render the fixed GET_INFO plan/result with its syntactic bit view."""
    print(f"scanner3d rgb-exposure-info {__version__}", file=stream)
    print("Fixed transfer (the only expressible operation)", file=stream)
    request = RGB_EXPOSURE_INFO_READ_TRANSFER.request
    print(
        f"  bmRequestType=0x{request.request_type:02x} bRequest=0x{request.request:02x} "
        f"wValue=0x{request.value:04x} wIndex=0x{request.index:04x} "
        f"wLength={request.length} timeout={request.timeout_ms} ms",
        file=stream,
    )
    print(
        f"  transfers at most {RGB_EXPOSURE_INFO_MAX_TRANSFERS}; retries none; "
        "GET_INFO only (no setting or other request)",
        file=stream,
    )
    print(f"Status: {outcome.status} - {outcome.detail}", file=stream)
    print(f"Info outcome: {outcome.info_outcome}", file=stream)
    print(f"Exit: {outcome.exit_code}", file=stream)
    reply = outcome.reply
    print(
        f"Reply: requested={reply.requested_length} actual={reply.actual_length} "
        f"retained={reply.raw_length} sha256={reply.sha256}",
        file=stream,
    )
    print(f"  raw hex: {reply.raw_hex or '(empty)'}, raw_info: {reply.raw_info}", file=stream)
    if outcome.reading is None:
        print("Decode: unavailable until an exact one-byte reply is received", file=stream)
    else:
        print("Bit view (syntactic Linux-UAPI interpretation only):", file=stream)
        for name, value in outcome.reading.bit_view.items():
            print(f"  {name}: {value}", file=stream)
    print(
        "Limitation: these bits do not prove selector advertisement, exposure semantics, "
        "SET acceptance, units or frame behavior",
        file=stream,
    )
    print(f"Transfers recorded: {len(outcome.transfers)}", file=stream)
    for transfer in outcome.transfers:
        print(
            f"  {transfer.step}: {'ok' if transfer.ok else 'FAIL'} "
            f"transferred={transfer.transferred} {transfer.detail}",
            file=stream,
        )
    print(f"Cleanup: {outcome.cleanup.note} (ok={outcome.cleanup.ok})", file=stream)
    if outcome.cleanup.guidance is not None:
        print(f"RECOVERY: {outcome.cleanup.guidance}", file=stream)


def _cmd_rgb_exposure_info(args: argparse.Namespace) -> int:
    """Read the fixed GET_INFO once, offline unless explicitly opted in."""
    if args.execute_rgb_exposure_info_read and not args.serial:
        print(
            "--execute-rgb-exposure-info-read requires an explicit --serial: "
            "this mode temporarily owns interface 0 and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    if args.execute_rgb_exposure_info_read and args.serial != RGB_EXPOSURE_INFO_AUTHORIZED_SERIAL:
        print(
            "refusing that transfer: this read is authorized for serial "
            f"{RGB_EXPOSURE_INFO_AUTHORIZED_SERIAL!r} only; refusing {args.serial!r}",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]
    if not args.execute_rgb_exposure_info_read:
        outcome = rgb_exposure_info_dry_run(args.serial)
    else:
        backend = rgb_exposure_info_backend_factory(args.sysfs_root, args.dev_root)
        try:
            outcome = backend.run(args.serial)
        except (RgbExposureInfoNotAllowed, InterfaceOwnershipNotAllowed) as exc:
            print(f"refusing that transfer: {exc}", file=sys.stderr)
            return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    payload = outcome.as_dict()
    if not args.json:
        _render_rgb_exposure_info(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}",
            file=sys.stderr,
        )
    return outcome.exit_code


def _default_rgb_exposure_info_backend(
    sysfs_root: Path, dev_root: Path
) -> UsbfsRgbExposureInfoBackend:
    return UsbfsRgbExposureInfoBackend(sysfs_root=sysfs_root, dev_root=dev_root)


rgb_exposure_info_backend_factory: Callable[[Path, Path], UsbfsRgbExposureInfoBackend] = (
    _default_rgb_exposure_info_backend
)


def _render_capture_node(target: NodeSnapshot, stream: TextIO) -> None:
    """The resolved capture node, and how it was resolved rather than guessed."""
    print(f"Target {target.serial} at {target.usb_path}", file=stream)
    print(f"  node       {target.node_path} ({target.node_name})", file=stream)
    print(
        f"  resolved   role {target.role!r} on interface {target.interface_number} "
        f"({target.interface_string!r}), V4L2 index {target.v4l_index}",
        file=stream,
    )
    print(f"  v4l name   {target.v4l_name!r}", file=stream)
    print(
        f"  rdev       {target.rdev_major}:{target.rdev_minor} "
        f"(inode {target.inode}, mode {target.mode})",
        file=stream,
    )
    print(f"  driver     {target.interface_driver or 'none'} (kept bound throughout)", file=stream)


def _render_illumination(illumination: IlluminationShutdownOutcome, stream: TextIO) -> None:
    """The off-command block, worded so it cannot be read as a safety outcome.

    Three deliberate choices, all of them corrections of output that shipped
    reading ``Blue-light shutdown ... status=complete``: the header names a
    *command* and a *write*, never a shutdown; the flags are labelled for what
    they measure (``command_accepted`` is about bytes, ``device_restored`` is
    about the kernel driver); and the emitter line sits immediately under the
    status, because that pair is what a hurried reader takes away.
    """
    print("Blue-light off command (one fixed write; NOT an optical result)", file=stream)
    print(
        f"  required={illumination.required} attempted={illumination.attempted} "
        f"command_accepted={illumination.accepted} "
        f"device_restored={illumination.device_restored} "
        f"finalizer_ok={illumination.succeeded}",
        file=stream,
    )
    print(f"  status={illumination.status} - {illumination.detail}", file=stream)
    print(f"  {EMITTER_STATE_LINE}", file=stream)
    if illumination.guidance is not None:
        # A discharged finalizer still carries the standing warning, so the
        # label has to distinguish "there is nothing to repair, and you still
        # may not conclude darkness" from "something needs repairing".
        label = "WARNING" if illumination.succeeded else "RECOVERY"
        print(f"  {label:<10} {illumination.guidance}", file=stream)


def _print_illumination_summary(illumination: IlluminationShutdownOutcome) -> None:
    """The same content, unindented, for the paired and series renderers."""
    print(
        "Blue-light off command (transfer only, NOT an optical result): "
        f"required={illumination.required} attempted={illumination.attempted} "
        f"command_accepted={illumination.accepted} "
        f"device_restored={illumination.device_restored} "
        f"finalizer_ok={illumination.succeeded} status={illumination.status}"
    )
    print(f"Emitter: {EMITTER_STATE_LINE}")
    if illumination.guidance is not None:
        label = "Blue-light warning" if illumination.succeeded else "Blue-light recovery"
        print(f"{label}: {illumination.guidance}")


def _warn_illumination(illumination: IlluminationShutdownOutcome) -> None:
    """The stderr side: warn on every run that streamed depth, success included.

    An accepted transfer is not a reason to stop warning, because this command
    does not darken this unit's emitter; the notice is therefore unconditional
    once depth actually streamed and only gains recovery wording when the
    finalizer did not discharge or the device was left unrestored.
    """
    if not illumination.required:
        return
    if not illumination.succeeded:
        print(
            "blue-light off command was not carried through, or the USB interface was "
            "not fully restored; do not claim the light is off",
            file=sys.stderr,
        )
    elif not illumination.accepted:
        print(
            "blue-light off command was not accepted by the device; do not claim the light is off",
            file=sys.stderr,
        )
    if illumination.guidance:
        print(illumination.guidance, file=sys.stderr)


def _render_frame(outcome: CaptureOutcome, stream: TextIO) -> None:
    """Human rendering of one bounded V4L2/MMAP capture attempt.

    One renderer for every profile, for the same reason there is one engine:
    the plan, the bounds and the disclaimer all come off
    :attr:`CaptureOutcome.profile`, so a second capture command cannot quietly
    print a different set of promises about the same lifecycle.
    """
    profile = outcome.profile
    print(f"scanner3d {profile.name} {__version__}", file=stream)
    print(file=stream)

    print("Plan (fixed at import; no flag widens any of it)", file=stream)
    print(
        f"  role/kind         {profile.role} {profile.node_kind} node, resolved from sysfs",
        file=stream,
    )
    print(f"  mode              {profile.mode_text}", file=stream)
    print(
        f"  descriptor rate   {profile.descriptor_frame_rate_hz} fps default "
        "(descriptor value; no S_PARM was issued and no interval was timed)",
        file=stream,
    )
    print(f"  complete frame    {profile.completeness_text}", file=stream)
    print(
        f"  buffers           request {profile.requested_buffers} MMAP, accept "
        f"{profile.min_accepted_buffers}-{profile.requested_buffers}",
        file=stream,
    )
    print(
        f"  mapping ceilings  {profile.max_buffer_bytes} B per buffer, "
        f"{profile.max_total_mapped_bytes} B total",
        file=stream,
    )
    print(
        f"  wait              one poll of at most {profile.capture_deadline_s}s; no restart, "
        "no retry, at most one DQBUF",
        file=stream,
    )
    print(f"  artifact ceiling  {profile.max_raw_artifact_bytes} B", file=stream)
    print(
        f"  driver            {EXPECTED_INTERFACE_DRIVER} stays bound; no usbfs, no HID",
        file=stream,
    )
    print("  ioctls            " + ", ".join(ALLOWED_IOCTLS), file=stream)
    print(file=stream)

    if outcome.target is None:
        print("Target: none resolved", file=stream)
    else:
        _render_capture_node(outcome.target, stream)
    print(file=stream)

    if outcome.capability is not None:
        cap = outcome.capability
        print("Capability (VIDIOC_QUERYCAP)", file=stream)
        print(f"  driver     {cap.driver!r} version {cap.version_text}", file=stream)
        print(f"  card       {cap.card!r}", file=stream)
        print(f"  bus        {cap.bus_info!r}", file=stream)
        print(f"  effective  0x{cap.effective_capabilities:08x}", file=stream)
        print(
            f"  named      {', '.join(cap.device_capability_names or cap.capability_names)}",
            file=stream,
        )
        print(file=stream)

    for label, fmt in (
        ("Initial format (VIDIOC_G_FMT)", outcome.initial_format),
        ("Negotiated format (VIDIOC_S_FMT)", outcome.negotiated_format),
    ):
        if fmt is None:
            continue
        print(label, file=stream)
        print(
            f"  {fmt.fourcc!r} {fmt.width}x{fmt.height} field {fmt.field_name}",
            file=stream,
        )
        if profile.compressed:
            layout = "(a compressed bitstream; bytesperline carries no row stride)"
        else:
            layout = "(rows padded)" if fmt.padded else "(rows tightly packed)"
        print(
            f"  bytesperline {fmt.bytesperline} sizeimage {fmt.sizeimage} " + layout,
            file=stream,
        )
        print(file=stream)

    if outcome.buffers is not None:
        print(
            f"Buffers: {outcome.buffers.count} MMAP allocated of "
            f"{profile.requested_buffers} requested",
            file=stream,
        )
        for mapping in outcome.mappings:
            print(
                f"  index {mapping['index']} offset {mapping['offset']} "
                f"mapped {mapping['mapped_length']} B",
                file=stream,
            )
        print(file=stream)

    if outcome.poll is not None:
        poll = outcome.poll
        verdict = "timed out" if poll.timed_out else ", ".join(poll.event_names)
        print(
            f"Poll: waited {poll.waited_s:.3f}s of a {poll.deadline_s}s deadline "
            f"in {poll.as_dict()['calls']} call -> {verdict}",
            file=stream,
        )
        print(file=stream)

    if outcome.dequeued is not None:
        buf = outcome.dequeued
        print("Dequeued buffer (VIDIOC_DQBUF, once)", file=stream)
        print(
            f"  index      {buf.index} sequence {buf.sequence} field {buf.field_name}", file=stream
        )
        print(f"  bytesused  {buf.bytesused} of {buf.length} B mapped", file=stream)
        print(f"  flags      0x{buf.flags:08x} {', '.join(buf.flag_names)}", file=stream)
        print(
            f"  timestamp  {buf.timestamp_sec}.{buf.timestamp_usec:06d} "
            f"(type {buf.timestamp_type}, source {buf.timestamp_source})",
            file=stream,
        )
        print(file=stream)

    _render_frame_evidence(outcome, stream)

    print("Steps", file=stream)
    if not outcome.steps:
        print("  none attempted", file=stream)
    for step in outcome.steps:
        mark = "ok  " if step.ok else "FAIL"
        timing = "" if step.duration_s is None else f" [{step.duration_s * 1000:.3f} ms]"
        errno = "" if step.errno is None else f" ({step.errno_name})"
        print(f"  [{mark}] {step.step!s:<16} {step.detail}{errno}{timing}", file=stream)
    print(file=stream)

    artifacts = outcome.artifacts
    print(f"Artifacts (directory {artifacts.directory})", file=stream)
    if not artifacts.files:
        print(f"  {artifacts.note}", file=stream)
    for artifact in artifacts.files:
        mark = "ok  " if artifact.ok else "FAIL"
        where = artifact.path or "(no separate file)"
        print(f"  [{mark}] {artifact.kind:<7} {where}", file=stream)
        print(f"          {artifact.byte_count} B sha256 {artifact.sha256 or 'n/a'}", file=stream)
        print(f"          {artifact.detail}", file=stream)
    print(file=stream)

    cleanup = outcome.cleanup
    print(f"Cleanup: {'ok' if cleanup.ok else 'INCOMPLETE'} - {cleanup.note}", file=stream)
    for step in cleanup.steps:
        mark = "ok  " if step.ok else "FAIL"
        print(f"  [{mark}] {step.step!s:<16} {step.detail}", file=stream)
    if cleanup.guidance is not None:
        print(f"  recovery: {cleanup.guidance}", file=stream)
    print(file=stream)

    _render_illumination(outcome.illumination_shutdown, stream)
    print(file=stream)

    print(f"Result: {outcome.status} - {outcome.detail}", file=stream)
    if outcome.total_duration_s is not None:
        print(f"  total {outcome.total_duration_s * 1000:.3f} ms", file=stream)
    print(f"  {profile.disclaimer}", file=stream)


def _render_frame_evidence(outcome: CaptureOutcome, stream: TextIO) -> None:
    """Whatever the profile learned about the payload beyond the shared bounds.

    For the MJPEG profile that is the structural JPEG report: markers found,
    the frame header's own declared size, and the counts that show the scan was
    walked rather than guessed at. Nothing here decoded a pixel.
    """
    evidence = outcome.frame_evidence
    if evidence is None:
        return
    jpeg = evidence.get("jpeg")
    if not isinstance(jpeg, dict):  # pragma: no cover - only the MJPEG profile sets this
        return
    print("JPEG structure (markers only; no pixel was decoded)", file=stream)
    print(f"  status     {jpeg.get('status')} - {jpeg.get('detail')}", file=stream)
    print(
        f"  bounds     SOI {jpeg.get('has_soi')}, EOI {jpeg.get('has_eoi')}, "
        f"terminal EOI {jpeg.get('eoi_terminal')}, {jpeg.get('trailing_bytes')} trailing B",
        file=stream,
    )
    frame = jpeg.get("frame")
    if isinstance(frame, dict):
        print(
            f"  frame      {frame.get('marker_name')} {frame.get('width')}x"
            f"{frame.get('height')}, {frame.get('components')} component(s), "
            f"{frame.get('precision')}-bit, progressive {frame.get('progressive')}",
            file=stream,
        )
    else:
        print("  frame      no supported SOF marker was found", file=stream)
    print(
        f"  expected   {evidence.get('expected_width')}x{evidence.get('expected_height')} "
        f"-> dimensions match: {evidence.get('dimensions_match')}",
        file=stream,
    )
    print(
        f"  scan       {jpeg.get('scan_count')} SOS, "
        f"{jpeg.get('restart_marker_count')} restart marker(s), "
        f"{jpeg.get('stuffed_byte_count')} stuffed ff00 byte(s)",
        file=stream,
    )
    segments = jpeg.get("segments")
    if isinstance(segments, list):
        names = ", ".join(str(seg.get("name")) for seg in segments if isinstance(seg, dict))
        print(f"  segments   {names}", file=stream)
    print(file=stream)


def _emit(payload: dict[str, object], args: argparse.Namespace) -> None:
    """Write the result as JSON to stdout and/or --json-out."""
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cmd_firmware_version(args: argparse.Namespace) -> int:
    """Read the firmware/algorithm version string. Offline unless --execute."""
    if args.detach_uvcvideo_from_interface_0:
        return _run_ownership(args)
    backend = UsbfsControlBackend(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = backend.read(args.serial) if args.execute else backend.preflight(args.serial)
    except ControlRequestNotAllowed as exc:
        print(f"refusing to send that request: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_control(outcome, sys.stdout)
    _emit(payload, args)
    return outcome.exit_code


def _run_ownership(args: argparse.Namespace) -> int:
    """The opt-in path: temporarily take interface 0 from ``uvcvideo``.

    Refuses without an explicit ``--serial``: this mutates kernel driver
    binding, so the target may not be inferred from "whatever is plugged in".
    """
    if not args.serial:
        print(
            "--detach-uvcvideo-from-interface-0 requires an explicit --serial: "
            "this mode detaches a kernel driver and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    backend = UsbfsOwnershipBackend(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = backend.run(args.serial) if args.execute else backend.preflight(args.serial)
    except (ControlRequestNotAllowed, InterfaceOwnershipNotAllowed) as exc:
        print(f"refusing that usbfs operation: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_ownership(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}", file=sys.stderr
        )
    return outcome.exit_code


def _cmd_inited_sequence(args: argparse.Namespace) -> int:
    """Reproduce the observed /tmp/inited exchange. Offline unless opted in.

    Refuses to write without an explicit ``--serial``: this is the only command
    that sends bytes *to* the scanner, so the target may not be inferred from
    "whatever is plugged in".
    """
    if args.execute_write_sequence and not args.serial:
        print(
            "--execute-write-sequence requires an explicit --serial: this mode writes to "
            "the scanner and detaches a kernel driver, and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    backend = UsbfsInitSequenceBackend(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = (
            backend.run(args.serial)
            if args.execute_write_sequence
            else backend.preflight(args.serial)
        )
    except (InitSequenceNotAllowed, InterfaceOwnershipNotAllowed) as exc:
        print(f"refusing that transfer: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_init_sequence(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.protocol_cleanup.ok and outcome.protocol_cleanup.guidance is not None:
        print(
            f"the clearing write did not complete: {outcome.protocol_cleanup.guidance}",
            file=sys.stderr,
        )
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}", file=sys.stderr
        )
    return outcome.exit_code


def _cmd_depth_intrinsics(args: argparse.Namespace) -> int:
    """Read the factory depth intrinsics key once. Offline unless opted in.

    Refuses to act without an explicit ``--serial``, and the operation itself
    refuses any serial but the one the owner authorized: this mode writes a key
    to the scanner, and the target may not be inferred from "whatever is
    plugged in".
    """
    if args.execute_calibration_read and not args.serial:
        print(
            "--execute-calibration-read requires an explicit --serial: this mode writes to "
            "the scanner and detaches a kernel driver, and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    backend = UsbfsIntrinsicsBackend(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = (
            backend.run(args.serial)
            if args.execute_calibration_read
            else backend.preflight(args.serial)
        )
    except (IntrinsicsReadNotAllowed, InterfaceOwnershipNotAllowed) as exc:
        print(f"refusing that transfer: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_intrinsics(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.protocol_cleanup.ok and outcome.protocol_cleanup.guidance is not None:
        print(
            f"the clearing write did not complete: {outcome.protocol_cleanup.guidance}",
            file=sys.stderr,
        )
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}", file=sys.stderr
        )
    return outcome.exit_code


def _cmd_depth_scale(args: argparse.Namespace) -> int:
    """Read the vendor depth-scale property once. Offline unless opted in.

    Refuses to act without an explicit ``--serial``, and the operation itself
    refuses any serial but the one the owner authorized: this mode writes a
    property id to the scanner, and the target may not be inferred from
    "whatever is plugged in".
    """
    if args.execute_depth_scale_read and not args.serial:
        print(
            "--execute-depth-scale-read requires an explicit --serial: this mode writes to "
            "the scanner and detaches a kernel driver, and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    backend = UsbfsDepthScaleBackend(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = (
            backend.run(args.serial)
            if args.execute_depth_scale_read
            else backend.preflight(args.serial)
        )
    except (DepthScaleReadNotAllowed, InterfaceOwnershipNotAllowed) as exc:
        print(f"refusing that transfer: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_depth_scale(outcome, sys.stdout)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully restore the device: {outcome.cleanup.guidance}", file=sys.stderr
        )
    return outcome.exit_code


def _run_frame_capture(
    args: argparse.Namespace,
    profile: CaptureProfile,
    backend_factory: Callable[..., V4l2FrameBackend],
    execute: bool,
) -> int:
    """Run one fixed capture, dry unless the command's own execute flag is set.

    Shared by every capture command for the same reason the engine is: the
    refusal to guess a target, the ordering of stdout and JSON, and the two
    stderr warnings that outrank a frame are policy, and having one copy of
    them means a new profile cannot arrive with a weaker version.
    """
    if execute and not args.serial:
        print(
            f"{profile.execute_flag} requires an explicit --serial: this mode starts "
            "a capture stream on the scanner and will not guess its target",
            file=sys.stderr,
        )
        return OWNERSHIP_EXIT_CODES[OwnershipStatus.REFUSED]

    output_dir = args.output_dir if args.output_dir is not None else profile.default_output_dir
    backend = backend_factory(sysfs_root=args.sysfs_root, dev_root=args.dev_root)
    try:
        outcome = (
            backend.run(args.serial, output_dir)
            if execute
            else backend.preflight(args.serial or DEPTH_FRAME_AUTHORIZED_SERIAL, output_dir)
        )
    except V4l2CaptureNotAllowed as exc:
        print(f"refusing that capture: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload = outcome.as_dict()
    if not args.json:
        _render_frame(outcome, sys.stdout)
        # Keep the final-state result prominent even if a profile-specific
        # renderer is substituted by an injected CLI test seam.
        _print_illumination_summary(outcome.illumination_shutdown)
    _emit(payload, args)
    if not outcome.cleanup.ok and outcome.cleanup.guidance is not None:
        print(
            f"cleanup did not fully release the device: {outcome.cleanup.guidance}",
            file=sys.stderr,
        )
    if not outcome.artifacts.ok:
        print(
            f"the frame was not fully preserved: {outcome.artifacts.note}",
            file=sys.stderr,
        )
    _warn_illumination(outcome.illumination_shutdown)
    exit_code = outcome.exit_code
    session_dir = getattr(args, "session_dir", None)
    if execute and session_dir is not None:
        problem = _record_capture_in_session(session_dir, profile, outcome)
        if problem is not None:
            print(f"session recording did not complete: {problem}", file=sys.stderr)
            if exit_code == 0:
                exit_code = EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    return exit_code


def _open_or_create_session(
    session_dir: Path, *, software_version: str, device: DeviceIdentity | None
) -> SessionStore:
    """Create the session at ``session_dir`` if it is new, else continue it.

    A directory that already holds a manifest is the same session resuming -
    :meth:`SessionStore.open` honestly relabels one an earlier process left
    ``running`` rather than silently promoting it. A directory with no
    manifest is a brand new session, created exclusively so this can never
    adopt a directory some unrelated writer already populated.
    """
    if (session_dir / MANIFEST_NAME).is_file():
        return SessionStore.open(session_dir)
    return SessionStore.create(
        session_dir.parent,
        session_dir.name,
        software_version=software_version,
        device=device,
    )


def _record_capture_stage(
    store: SessionStore, stage: str, outcome: CaptureOutcome, *, parameters: dict[str, object]
) -> str | None:
    """Record one capture attempt as a session stage.

    Returns ``None`` once the stage is recorded as a verified success, or a
    problem description otherwise. Nothing here is inferred from
    ``outcome.succeeded`` alone: the stage is only marked complete after its
    raw bytes have been written through the atomic artifact store and
    re-hashed back off disk, so a stage this function marks complete is
    complete because the store proved it, not because no exception was
    raised.
    """
    store.start()
    store.begin_stage(stage, parameters=parameters)
    # Every profile's preserver appends the verbatim dequeued bytes as the
    # first file - "raw" for depth, the only file ("mjpeg") for RGB - so the
    # first entry is always that one, regardless of a profile-specific kind
    # name or of a second, derived file (depth's tightly packed copy) after it.
    raw_file = outcome.artifacts.files[0] if outcome.artifacts.files else None
    if not outcome.succeeded:
        # A capture that did not succeed is not a session-recording problem -
        # it is faithfully recorded as a failed stage, and the caller already
        # surfaces the capture failure itself through the outcome's own exit
        # code and rendering. Surfacing it a second time here as a "problem"
        # would misreport working recording machinery as broken.
        store.finish_stage(
            stage, StageState.FAILED, error=outcome.detail or "the capture did not succeed"
        )
        return None
    if raw_file is None or not raw_file.ok or raw_file.path is None:
        # This is the one truly unexpected case: the capture reported success
        # yet left no verified raw artifact to record. That is an internal
        # inconsistency worth surfacing, not an ordinary capture failure.
        error = "the capture reported success but left no verified raw artifact to record"
        store.finish_stage(stage, StageState.FAILED, error=error)
        return error
    try:
        payload = Path(raw_file.path).read_bytes()
    except OSError as exc:
        error = f"cannot read back the raw artifact at {raw_file.path}: {exc}"
        store.finish_stage(stage, StageState.FAILED, error=error)
        return error
    if (
        len(payload) != raw_file.byte_count
        or hashlib.sha256(payload).hexdigest() != raw_file.sha256
    ):
        error = f"the raw artifact at {raw_file.path} no longer matches what was captured"
        store.finish_stage(stage, StageState.FAILED, error=error)
        return error
    artifact_name = f"{stage}-{raw_file.sha256[:16]}{Path(raw_file.path).suffix}"
    try:
        artifact = store.write_artifact(stage, f"raw/{artifact_name}", payload, kind=f"raw_{stage}")
    except (SessionError, OSError) as exc:
        error = f"cannot record the artifact into the session: {exc}"
        store.finish_stage(stage, StageState.FAILED, error=error)
        return error
    problems = store.verify_artifacts()
    if artifact.path in problems:
        error = f"artifact verification failed right after recording: {problems[artifact.path]}"
        store.finish_stage(stage, StageState.FAILED, error=error)
        return error
    store.finish_stage(
        stage, StageState.COMPLETE, counters={"delivered": 1, "bytes": artifact.bytes}
    )
    return None


def _record_capture_in_session(
    session_dir: Path, profile: CaptureProfile, outcome: CaptureOutcome
) -> str | None:
    """Create-or-open the session at ``session_dir`` and record this capture.

    Returns ``None`` on a verified success, or a problem description the
    caller should surface. Failure here - a terminal session, a filesystem
    error, a corrupt manifest - never raises past this function: it is
    additional evidence-keeping around a capture that has already happened,
    and a broken session must be reported, not let crash the command that
    just captured a real frame.
    """
    target = outcome.target
    device = (
        DeviceIdentity(
            vendor_id=EXPECTED_VENDOR_ID,
            product_id=EXPECTED_PRODUCT_ID,
            serial=target.serial,
            usb_path=target.usb_path,
            bus_number=target.bus_number,
            device_number=target.device_number,
        )
        if target is not None
        else None
    )
    try:
        store = _open_or_create_session(
            session_dir, software_version=f"scanner3d-{__version__}", device=device
        )
        stage = profile.name.replace("-", "_")
        return _record_capture_stage(
            store,
            stage,
            outcome,
            parameters={
                "mode": profile.mode_label,
                "width": profile.width,
                "height": profile.height,
            },
        )
    except (SessionError, OSError) as exc:
        return f"cannot use session at {session_dir}: {exc}"


def _default_illumination_backend(
    sysfs_root: Path, dev_root: Path
) -> UsbfsIlluminationShutdownBackend:
    return UsbfsIlluminationShutdownBackend(sysfs_root=sysfs_root, dev_root=dev_root)


illumination_backend_factory: Callable[[Path, Path], UsbfsIlluminationShutdownBackend] = (
    _default_illumination_backend
)


def _depth_frame_backend_factory(**kwargs: Path) -> V4l2FrameBackend:
    """Construct depth with the mandatory post-STREAMON finalizer seam."""
    return V4l2DepthFrameBackend(
        **cast(Any, kwargs),
        illumination_backend_factory=illumination_backend_factory,
    )


def _cmd_depth_frame(args: argparse.Namespace) -> int:
    """Attempt one bounded V4L2 depth frame. Offline unless opted in.

    Refuses to act without an explicit ``--serial``, and the backend itself
    refuses any serial but the one the owner authorized: this mode starts a
    stream on the scanner, and the target may not be inferred from "whatever is
    plugged in".
    """
    # ``V4l2DepthFrameBackend`` is resolved from this module's globals at call
    # time, which is what lets a test substitute an injected-syscall factory.
    return _run_frame_capture(
        args,
        DEPTH_PROFILE,
        _depth_frame_backend_factory,
        args.execute_depth_frame_capture,
    )


def _cmd_rgb_frame(args: argparse.Namespace) -> int:
    """Attempt one bounded V4L2 RGB frame. Offline unless opted in.

    The same refusals as the depth command, for the same reasons, through the
    same engine - only the profile differs.
    """
    return _run_frame_capture(
        args, RGB_PROFILE, V4l2RgbFrameBackend, args.execute_rgb_frame_capture
    )


def _cmd_rgb_controls(args: argparse.Namespace) -> int:
    """Inventory fixed standard RGB controls, offline unless explicitly opted in."""
    backend = rgb_controls_backend_factory(args.sysfs_root, args.dev_root)
    outcome = (
        backend.run(args.serial or "") if args.read_rgb_controls else backend.dry_run(args.serial)
    )
    payload = outcome.as_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_rgb_controls_human(outcome))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return outcome.exit_code


def _default_rgb_controls_backend(sysfs_root: Path, dev_root: Path) -> V4l2RgbControlBackend:
    return V4l2RgbControlBackend(sysfs_root=sysfs_root, dev_root=dev_root)


rgb_controls_backend_factory: Callable[[Path, Path], V4l2RgbControlBackend] = (
    _default_rgb_controls_backend
)


def _default_paired_backend(output_dir: Path) -> paired_capture.StreamBackend:
    return paired_capture.PairedV4l2Backend(output_dir=output_dir)


# A narrow factory seam lets offline CLI tests inject the already-synthetic
# backend without constructing or reaching the host default backend.
paired_backend_factory: Callable[[Path], paired_capture.StreamBackend] = _default_paired_backend
paired_termination_shield = defer_termination


def _cmd_paired_capture(args: argparse.Namespace) -> int:
    """Run the fixed paired backend only after the exact opt-in."""
    if args.start_depth_first_rgb_second:
        if args.serial != paired_capture.AUTHORIZED_SERIAL:
            runner = paired_capture.PairedCapture(
                None, termination_shield=paired_termination_shield
            )
            outcome = runner.run(args.serial or "", opt_in=paired_capture.EXECUTE_OPT_IN)
        elif args.output_dir is None:
            outcome = paired_capture.PairedOutcome(
                paired_capture.PairedStatus.REFUSED,
                "required local --output-dir is missing",
                plan=paired_capture.PairedPlan(),
            )
        else:
            backend = paired_backend_factory(args.output_dir)
            runner = paired_capture.PairedCapture(
                backend, termination_shield=paired_termination_shield
            )
            outcome = runner.run(args.serial, opt_in=paired_capture.EXECUTE_OPT_IN)
    else:
        runner = paired_capture.PairedCapture(None, termination_shield=paired_termination_shield)
        outcome = runner.dry_run()
    payload = outcome.as_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("scanner3d paired-capture (depth first, RGB second)")
        print(f"Status: {outcome.status} - {outcome.detail}")
        print(f"Exit: {outcome.exit_code}")
        print("Roles: depth Y16 640x400; rgb MJPG 1280x800")
        print("Ordering: resolve both -> depth STREAMON -> 1.0 s -> RGB STREAMON -> one wait each")
        print("Cleanup: RGB reverse teardown, then depth reverse teardown")
        print(
            f"Execution requires serial {paired_capture.AUTHORIZED_SERIAL} and "
            f"{paired_capture.EXECUTE_OPT_IN}"
        )
        for role, result in (("depth", outcome.depth_result), ("rgb", outcome.rgb_result)):
            if result is None:
                print(f"{role}: not attempted")
                continue
            print(
                f"{role}: {result.status} bytes={result.frame_bytes} "
                f"stream_duration={result.stream_duration_s} "
                f"total_duration={result.total_duration_s}"
            )
            if result.dequeued is not None:
                print(
                    f"  sequence={result.dequeued.sequence} "
                    f"timestamp_ns={result.dequeued.timestamp_ns}"
                )
            print(f"  artifact: {result.artifacts.note}")
            print(f"  cleanup: {result.cleanup.note}")
            if result.cleanup.guidance:
                print(f"  recovery: {result.cleanup.guidance}")
        timing = cast(dict[str, object], outcome.as_dict()["timing"])
        print(
            f"Timing: depth={timing['depth_timestamp_ns']} "
            f"rgb={timing['rgb_timestamp_ns']} "
            f"sequences={timing['depth_sequence']}/{timing['rgb_sequence']} "
            f"skew={timing['observed_skew_ns']} ns; "
            "hardware synchronization not established"
        )
        if outcome.cleanup.deferred_signals:
            print(f"Cleanup deferred signals: {list(outcome.cleanup.deferred_signals)}")
        if outcome.cleanup.actions:
            print(f"Cleanup actions: {list(outcome.cleanup.actions)}")
        if outcome.cleanup.errors:
            print(f"Cleanup errors: {list(outcome.cleanup.errors)}")
        if outcome.cleanup.guidance:
            print(f"Cleanup recovery: {outcome.cleanup.guidance}")
        illumination = outcome.illumination_shutdown
        _print_illumination_summary(illumination)
        if not outcome.succeeded:
            # Name the actual reason. The old wording blamed blue-light
            # finalization for every failed run in which the finalizer was
            # merely required, including runs it had nothing to do with.
            print(
                "Overall result: FAILED "
                + (
                    "(blue-light off command was not discharged or the device was not restored)"
                    if illumination.required and not illumination.succeeded
                    else "(see the status above)"
                )
            )
    _warn_illumination(outcome.illumination_shutdown)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return outcome.exit_code


def _cmd_paired_rgb_series(args: argparse.Namespace) -> int:
    """Run the fixed depth-first paired RGB-series diagnostic."""
    telemetry_requested = bool(args.read_rgb_controls_during_series)
    if args.start_depth_first_rgb_series:
        if args.serial != paired_capture.AUTHORIZED_SERIAL:
            runner = paired_capture.RgbSeriesCapture(
                None, termination_shield=paired_termination_shield
            )
            outcome = runner.run(
                args.serial or "",
                opt_in=paired_capture.RGB_SERIES_EXECUTE_OPT_IN,
                read_rgb_controls_during_series=telemetry_requested,
            )
        elif args.output_dir is None:
            outcome = paired_capture.PairedOutcome(
                paired_capture.PairedStatus.REFUSED,
                "required local --output-dir is missing",
                plan=paired_capture.PairedPlan(),
                series_mode=True,
                rgb_control_telemetry_requested=telemetry_requested,
            )
        else:
            backend = paired_backend_factory(args.output_dir)
            runner = paired_capture.RgbSeriesCapture(
                backend, termination_shield=paired_termination_shield
            )
            outcome = runner.run(
                args.serial,
                opt_in=paired_capture.RGB_SERIES_EXECUTE_OPT_IN,
                read_rgb_controls_during_series=telemetry_requested,
            )
    else:
        outcome = paired_capture.RgbSeriesCapture(
            None, termination_shield=paired_termination_shield
        ).dry_run(read_rgb_controls_during_series=telemetry_requested)
    payload = outcome.as_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("scanner3d paired-rgb-series (depth first, RGB series)")
        print(f"Status: {outcome.status} - {outcome.detail}")
        print(f"Exit: {outcome.exit_code}")
        print(
            f"Series: {paired_capture.RGB_SERIES_COUNT} RGB observations, "
            f"one {paired_capture.RGB_SERIES_DEADLINE_S}s absolute deadline"
        )
        print("Ordering: depth STREAMON -> 1.0 s -> RGB STREAMON -> one depth wait -> RGB series")
        print(
            "Strict JPEG validity is unchanged; EOI prefixes and opaque trailers are evidence only"
        )
        print(
            f"Execution requires serial {paired_capture.AUTHORIZED_SERIAL} and "
            f"{paired_capture.RGB_SERIES_EXECUTE_OPT_IN}"
        )
        print(
            "RGB control telemetry: "
            + (
                "requested (only active with the exact series opt-in)"
                if telemetry_requested
                else "disabled"
            )
        )
        series = outcome.rgb_series
        if series is None:
            print("RGB series: not attempted")
        else:
            print(
                f"RGB series: {series.status} delivered={series.delivered_count}/"
                f"{paired_capture.RGB_SERIES_COUNT} strict_success={series.strict_success}"
            )
            print(
                f"  retained={series.retained_bytes}/{series.max_total_bytes} B "
                f"polls={series.poll_count} dequeues={series.dequeue_count} "
                f"requeues={series.requeue_count}"
            )
            print(
                f"  telemetry: {'enabled' if series.telemetry_enabled else 'disabled'} "
                f"points={len(series.telemetry_samples)}/{paired_capture.RGB_SERIES_CONTROL_POINT_LIMIT}"
            )
            for sample in series.telemetry_samples:
                values = ", ".join(
                    f"{record.control.name}="
                    f"{record.value if record.value is not None else record.status}"
                    for record in sample.records
                )
                print(f"    point {sample.point}: {values}")
            for observation in series.observations:
                print(
                    f"  [{observation.index}] {observation.status} "
                    f"sequence={observation.buffer.sequence} "
                    f"timestamp_ns={observation.buffer.timestamp_ns} "
                    f"bytes={observation.artifact.byte_count} "
                    f"sha256={observation.artifact.sha256}"
                )
                print(
                    f"      EOI-prefix={observation.eoi_prefix_bytes} "
                    f"trailer={observation.trailer_bytes} bytes {observation.trailer_hex}"
                )
                print(f"      artifact={observation.artifact.path}")
            if series.terminal_verdict is not None:
                verdict = series.terminal_verdict
                errno = "" if verdict.errno is None else f" ({verdict.errno_name})"
                print(f"  terminal: {verdict.status}{errno} - {verdict.detail}")
            print(f"  sequence gaps: {list(series.sequence_gaps)}")
            print(f"  intervals_ns: {list(series.intervals_ns)}")
        if outcome.cleanup.actions:
            print(f"Cleanup actions: {list(outcome.cleanup.actions)}")
        if outcome.cleanup.errors:
            print(f"Cleanup errors: {list(outcome.cleanup.errors)}")
        if outcome.cleanup.deferred_signals:
            print(f"Cleanup deferred signals: {list(outcome.cleanup.deferred_signals)}")
        if outcome.cleanup.guidance:
            print(f"Cleanup recovery: {outcome.cleanup.guidance}")
        illumination = outcome.illumination_shutdown
        _print_illumination_summary(illumination)
        if not outcome.succeeded:
            # Name the actual reason. The old wording blamed blue-light
            # finalization for every failed run in which the finalizer was
            # merely required, including runs it had nothing to do with.
            print(
                "Overall result: FAILED "
                + (
                    "(blue-light off command was not discharged or the device was not restored)"
                    if illumination.required and not illumination.succeeded
                    else "(see the status above)"
                )
            )
    _warn_illumination(outcome.illumination_shutdown)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return outcome.exit_code


SWEEP_COMMAND = "sweep"
RECONSTRUCT_COMMAND = "reconstruct"

#: Metres per depth code, as the factory record decodes it. Validated against a
#: caliper: a 58.0 mm can measured 57.705 mm, an error of -0.51%. The unit is
#: millimetres; the ACCURACY across the working range is not established, and
#: ranges from -0.49% to -2.26% with no correction model.
DEPTH_SCALE_METRES: Final = 0.004999999888241291


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Stream frames to disk, writing each as it arrives rather than buffering."""
    from scanner3d.v4l2_stream import stream_frames

    args.output.mkdir(parents=True, exist_ok=True)
    written = 0

    def on_frame(buf: bytes, _record: object) -> None:
        nonlocal written
        (args.output / f"frame_{written:04d}.raw").write_bytes(buf)
        written += 1
        if written % 25 == 0 and not args.json:
            print(f"  {written} frames kept", file=sys.stderr, flush=True)

    kwargs: dict[str, object] = {
        "frames": args.frames,
        "decimate": args.decimate,
        "duration_s": args.duration,
        "on_frame": on_frame,
    }
    if args.serial:
        kwargs["serial"] = args.serial
    try:
        result = stream_frames(**kwargs)  # type: ignore[arg-type]
    except ScannerError as exc:
        print(f"capture refused: {exc}", file=sys.stderr)
        return EXIT_CODES[getattr(exc, "status", DoctorStatus.INTERNAL_ERROR)]
    except OSError as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload: dict[str, object] = {
        "command": {"name": SWEEP_COMMAND, "output": str(args.output)},
        "kept": written,
        "delivered": result.delivered,
        "dropped_by_sequence": result.dropped_by_sequence,
        "short_frames": result.short_frames,
        "first_frame_ms": result.first_frame_ms,
        "stopped_early": result.stopped_early,
        "stop_reason": result.stop_reason,
    }
    if not args.json:
        print(f"kept {written} of {result.delivered} delivered frames to {args.output}")
        print(f"  dropped_by_sequence {result.dropped_by_sequence}  short {result.short_frames}")
        print(f"  stopped_early {result.stopped_early}: {result.stop_reason}")
        if result.dropped_by_sequence:
            print(
                "  NOTE: dropped frames leave gaps in the sweep. Registration tolerates "
                "some, but a path with many gaps may not revisit its own viewpoints.",
            )
    _emit(payload, args)
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    """Register a captured sweep offline and report what the evidence says."""
    from scanner3d.depth_metric import DepthUnitAssumption, deproject_depth_frame
    from scanner3d.pipeline import reconstruct_sweep
    from scanner3d.simulate import FACTORY_INTRINSICS

    paths = sorted(args.frames.glob("frame_*.raw"))[:: max(1, args.stride)]
    if len(paths) < 3:
        print(
            f"{args.frames} holds {len(paths)} usable frames at stride {args.stride}; "
            "a sweep needs at least three",
            file=sys.stderr,
        )
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    unit = DepthUnitAssumption(
        label="mm",
        stated_by="caliper measurement recorded in docs/metric-validation.md",
        basis="a 58.0 mm can measured 57.705 mm at about 122.7 mm standoff",
    )
    clouds: list[tuple[tuple[float, float, float], ...]] = []
    for path in paths:
        conversion = deproject_depth_frame(
            path.read_bytes(),
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=DEPTH_SCALE_METRES,
            unit=unit,
        )
        points = [(p.x, p.y, p.z) for p in conversion.points if 60.0 < p.z < 300.0]
        step = max(1, len(points) // 2200)
        clouds.append(tuple(points[::step]))

    if not args.json:
        print(f"{len(clouds)} frames, about {len(clouds[0])} points each", flush=True)

    def progress(message: str) -> None:
        if not args.json:
            print(message, file=sys.stderr, flush=True)

    try:
        result = reconstruct_sweep(
            clouds,
            backend=args.backend,
            workers=args.workers,
            point_to_plane=not args.point_to_point,
            robust=not args.no_robust,
            on_progress=progress,
        )
    except ScannerError as exc:
        print(f"reconstruction refused: {exc}", file=sys.stderr)
        return EXIT_CODES[getattr(exc, "status", DoctorStatus.INTERNAL_ERROR)]

    if args.poses_out is not None:
        args.poses_out.write_text(
            json.dumps([pose.as_dict() for pose in result.poses], indent=2) + "\n",
            encoding="utf-8",
        )

    payload: dict[str, object] = {
        **result.as_dict(),
        "command": {
            "name": RECONSTRUCT_COMMAND,
            "frames_directory": str(args.frames),
            "stride": args.stride,
            "backend": args.backend,
            "workers": args.workers,
            "offline": (
                "this command resolved no scanner and opened no device node; every "
                "input came from files on disk"
            ),
            "dimensional_accuracy_claimed": False,
        },
    }
    if not args.json:
        _render_reconstruction(result, sys.stdout)
    _emit(payload, args)
    return 0


def _render_reconstruction(result: object, stream: TextIO) -> None:
    """Print the scores, leading with the one that cannot be gamed."""
    chained = getattr(result, "chained_score", None)
    optimised = getattr(result, "optimised_score", None)
    print(f"\nframes {getattr(result, 'frames', '?')}", file=stream)
    print(f"revisited viewpoints found: {getattr(result, 'revisits_found', 0)}", file=stream)

    if optimised is None or chained is None:
        print("\nNOT SCORED, and not corrected.", file=stream)
        print(
            "Too few revisited viewpoints to withhold any. Without a viewpoint the "
            "scanner returned to, there is nothing a drifting trajectory can be "
            "caught against, so no number here would mean anything.",
            file=stream,
        )
    else:
        print("\nHELD-OUT CLOSURE ERROR (measured against constraints never fitted):", file=stream)
        print(
            f"  before optimisation  median {math.degrees(chained.median_error_radians):7.3f} deg"
            f"   worst {math.degrees(chained.worst_error_radians):7.3f}",
            file=stream,
        )
        print(
            f"  after  optimisation  median {math.degrees(optimised.median_error_radians):7.3f} deg"
            f"   worst {math.degrees(optimised.worst_error_radians):7.3f}",
            file=stream,
        )
        print(
            f"  {optimised.used_for_fitting} constraints used for fitting, "
            f"{optimised.held_out} withheld and used only to score",
            file=stream,
        )
        factor = getattr(result, "improvement_factor", None)
        if factor is not None and factor < 1.0:
            print(
                f"\n  WARNING: optimisation made this WORSE ({1.0 / factor:.1f}x). That usually "
                "means a false revisit - two views a failed alignment claimed were the "
                "same. Re-run with --no-robust to compare, and inspect the sweep.",
                file=stream,
            )

    for limitation in getattr(result, "limitations", ()):
        print(f"\n  - {limitation}", file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanner3d",
        description="Linux diagnostics for the Revopoint MINI structured-light scanner.",
    )
    parser.add_argument("--version", action="version", version=f"scanner3d {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_roots(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--sysfs-root",
            type=Path,
            default=DEFAULT_SYSFS_ROOT,
            help="sysfs mount point to inspect (override for fixture replay)",
        )
        p.add_argument(
            "--dev-root",
            type=Path,
            default=DEFAULT_DEV_ROOT,
            help="device-node root to inspect (override for fixture replay)",
        )
        p.add_argument("--json", action="store_true", help="emit JSON instead of text")

    doctor = sub.add_parser("doctor", help="diagnose the attached scanner")
    add_roots(doctor)
    doctor.add_argument("--json-out", type=Path, help="also write the JSON report to this path")
    doctor.add_argument(
        "--reference-probe",
        action="store_true",
        help="opt in to running the external vendor-SDK reference probe (touches hardware)",
    )
    doctor.add_argument("--serial", help="target this USB serial (default: first device found)")
    doctor.add_argument("--sdk-root", type=Path, help="path to the external SDK's 3DCamera dir")
    doctor.add_argument("--probe-binary", type=Path, help="path to the built reference probe")
    doctor.add_argument(
        "--probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_S,
        help=f"hard wall-clock budget for the probe, max {MAX_PROBE_TIMEOUT_S}s",
    )
    doctor.add_argument("--probe-json-out", type=Path, help="write raw probe JSON here")
    doctor.add_argument("--sdk-manifest", type=Path, default=DEFAULT_MANIFEST)
    doctor.add_argument(
        "--allow-unverified-sdk",
        action="store_true",
        help="run the probe even if the SDK hashes do not match the pinned manifest",
    )
    doctor.set_defaults(func=_cmd_doctor)

    listing = sub.add_parser("list", help="list attached matching devices")
    add_roots(listing)
    listing.set_defaults(func=_cmd_list)

    trace = sub.add_parser(
        "trace",
        help="decode a recorded usbfs trace (offline; no device or SDK needed)",
        description=(
            "Validate and summarize a trace written by reference/tracer. This is a "
            "pure file read: it never opens a device, never runs the vendor SDK, and "
            "never sends anything to the scanner."
        ),
    )
    trace.add_argument("path", type=Path, help="path to the recorded trace file")
    trace.add_argument("--json", action="store_true", help="emit JSON instead of text")
    trace.add_argument("--json-out", type=Path, help="also write the JSON report to this path")
    trace.set_defaults(func=_cmd_trace)

    firmware = sub.add_parser(
        "firmware-version",
        help="read the scanner's firmware/algorithm version over usbfs (opt-in)",
        description=(
            "Issue the single owned, read-only USB control request "
            f"(bmRequestType=0x{FIRMWARE_ALGORITHM_VERSION_READ.request_type:02x}, "
            f"bRequest=0x{FIRMWARE_ALGORITHM_VERSION_READ.request:02x}, "
            f"wValue=0x{FIRMWARE_ALGORITHM_VERSION_READ.value:04x}, "
            f"wIndex=0x{FIRMWARE_ALGORITHM_VERSION_READ.index:04x}, "
            f"wLength={FIRMWARE_ALGORITHM_VERSION_READ.length}) and validate the reply "
            "against the versions the vendor SDK already reported for this unit. "
            "This is the only request the command can send: there is no way to vary "
            "the setup packet, no write, no interface claim or detach, and no reset. "
            "Without --execute nothing is opened and the command only describes what "
            "it would do."
        ),
    )
    add_roots(firmware)
    firmware.add_argument(
        "--serial",
        help="target this exact USB serial (default: the sole attached scanner)",
    )
    firmware.add_argument(
        "--execute",
        action="store_true",
        help="opt in to actually opening the usbfs node and performing the one read",
    )
    firmware.add_argument(
        "--detach-uvcvideo-from-interface-0",
        action="store_true",
        help=(
            "TEMPORARILY DETACH the uvcvideo kernel driver from interface 0, claim it, "
            "perform the same one read, then release and reconnect the driver. Requires "
            "an explicit --serial. This is the only mode that mutates kernel driver "
            "binding; without --execute it only checks the preconditions."
        ),
    )
    firmware.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    firmware.set_defaults(func=_cmd_firmware_version)

    inited = sub.add_parser(
        "inited-sequence",
        help=(
            f"reproduce the SDK-observed {INITED_PATH} exchange on entity {ENTITY_ID} "
            "(opt-in; WRITES to the scanner)"
        ),
        description=(
            "Issue, exactly once and in order, the four 60-byte transfers the vendor SDK "
            f"was observed performing for {INITED_PATH}: a SET_CUR carrying opcode 0x01 and "
            f"that path, two GET_CUR reads, and a clearing SET_CUR of 60 zero bytes - all to "
            f"entity {ENTITY_ID} on interface {OWNED_INTERFACE}, inside the same fixed "
            "detach/claim/release/reconnect lifecycle as firmware-version. This is the only "
            "command in this project that writes to the scanner, and these four transfers are "
            "the only thing it can write: there is no flag for a path, a selector, an entity, "
            "an interface, a length, a timeout or a repetition. The clearing write is protocol "
            "cleanup and is attempted once even if a read fails or the process is interrupted. "
            "Without --execute-write-sequence nothing is opened and nothing is sent."
        ),
    )
    add_roots(inited)
    inited.add_argument(
        "--serial",
        help="target this exact USB serial (required by --execute-write-sequence)",
    )
    inited.add_argument(
        "--execute-write-sequence",
        action="store_true",
        help=(
            "opt in to actually WRITING to the scanner: detach uvcvideo from interface 0, "
            f"claim it, send the {INITED_PATH} SET_CUR, perform the two reads, send the "
            "clearing SET_CUR, then release and reconnect the driver. Requires --serial."
        ),
    )
    inited.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    inited.set_defaults(func=_cmd_inited_sequence)

    intrinsics = sub.add_parser(
        "depth-intrinsics",
        help=(
            f"read the factory depth intrinsics key {INTRINSICS_KEY} once "
            "(opt-in; WRITES the key to the scanner)"
        ),
        description=(
            "Perform, exactly once, the bounded three-transfer libuvcReadFile exchange the "
            f"vendor firmware serves for {INTRINSICS_KEY}: a SET_CUR carrying opcode 0x01 and "
            f"that key, at most {MAX_GET_TRANSFERS} length-prefixed GET_CUR result reads, and a "
            f"clearing SET_CUR of 60 zero bytes - all to entity {ENTITY_ID} on interface "
            f"{OWNED_INTERFACE}, inside the same fixed detach/claim/release/reconnect lifecycle "
            "as inited-sequence. This is not a file reader: the key is a compile-time constant, "
            "and there is no flag for a path, a selector, an entity, an interface, a length, a "
            "timeout, a cap, a capacity or a repetition. The assembled bytes are reported in "
            "full as the primary evidence; only a complete 40-byte blob is additionally parsed "
            "as the vendor Intrinsics record, and only a run whose clearing write and driver "
            "handback both succeeded offers it as calibration. Without "
            "--execute-calibration-read nothing is opened and nothing is sent."
        ),
    )
    add_roots(intrinsics)
    intrinsics.add_argument(
        "--serial",
        help="target this exact USB serial (required by --execute-calibration-read)",
    )
    intrinsics.add_argument(
        "--execute-calibration-read",
        action="store_true",
        help=(
            "opt in to actually WRITING to the scanner: detach uvcvideo from interface 0, "
            f"claim it, send the {INTRINSICS_KEY} SET_CUR, perform the bounded result reads, "
            f"send the clearing SET_CUR, then release and reconnect the driver. Requires "
            f"--serial, and refuses any serial but {AUTHORIZED_SERIAL}."
        ),
    )
    intrinsics.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    intrinsics.set_defaults(func=_cmd_depth_intrinsics)

    rgb_exposure_extension = sub.add_parser(
        "rgb-exposure-extension",
        allow_abbrev=False,
        help="describe or read the fixed RGB exposure extension state (read-only opt-in)",
        description=(
            "Describe the MINI SDK's separate RGB exposure extension state. The default is "
            "a completely inert offline plan: no backend is constructed, no device is "
            "resolved or opened, and no ioctl is sent. The exact opt-in performs one and "
            "only one fixed UVC GET_CUR: bmRequestType=0xa1, bRequest=0x81, wValue=0x1200, "
            "wIndex=0x0400, wLength=60, timeout 1000 ms, through the shared interface-0 "
            "detach/claim/release/reconnect lifecycle. There is no SET, retry, second read, "
            "V4L2 path or caller-controlled setup field. Physical execution is separately "
            "authorized only for the exact serial and is not claimed by this offline task."
        ),
    )
    add_roots(rgb_exposure_extension)
    rgb_exposure_extension.add_argument(
        "--serial",
        help=("target this exact USB serial (required by --execute-rgb-exposure-extension-read)"),
    )
    rgb_exposure_extension.add_argument(
        "--execute-rgb-exposure-extension-read",
        action="store_true",
        help=(
            "opt in to one fixed read for serial "
            f"{RGB_EXPOSURE_EXTENSION_AUTHORIZED_SERIAL}; no SET or retry"
        ),
    )
    rgb_exposure_extension.add_argument(
        "--json-out", type=Path, help="also write the JSON result to this path"
    )
    rgb_exposure_extension.set_defaults(func=_cmd_rgb_exposure_extension)

    rgb_exposure_length = sub.add_parser(
        "rgb-exposure-length",
        allow_abbrev=False,
        help="describe or read the fixed RGB extension GET_LEN (read-only opt-in)",
        description=(
            "Describe the MINI's fixed UVC GET_LEN boundary. The default is completely "
            "inert: no backend is constructed, no device is resolved or opened, and no "
            "ioctl is sent. The exact opt-in performs one and only one fixed request: "
            "bmRequestType=0xa1, bRequest=0x85, wValue=0x1200, wIndex=0x0400, "
            "wLength=2, timeout 1000 ms, through the shared interface-0 lifecycle. "
            "There is no exposure or other setting change, retry, second transfer, "
            "V4L2 path or caller-controlled setup field. Scanner-030 already ran "
            "this boundary once and completed its separate immutable blue-light-off "
            "finalizer; every new physical invocation must perform its own finalizer "
            "and clean restoration."
        ),
    )
    add_roots(rgb_exposure_length)
    rgb_exposure_length.add_argument(
        "--serial",
        help=("target this exact USB serial (required by --execute-rgb-exposure-length-read)"),
    )
    rgb_exposure_length.add_argument(
        "--execute-rgb-exposure-length-read",
        action="store_true",
        help=(
            "opt in to one fixed GET_LEN for serial "
            f"{RGB_EXPOSURE_LENGTH_AUTHORIZED_SERIAL}; no exposure/other setting or retry"
        ),
    )
    rgb_exposure_length.add_argument(
        "--json-out", type=Path, help="also write the JSON result to this path"
    )
    rgb_exposure_length.set_defaults(func=_cmd_rgb_exposure_length)

    rgb_exposure_info = sub.add_parser(
        "rgb-exposure-info",
        allow_abbrev=False,
        help="describe or read the fixed RGB extension GET_INFO (read-only opt-in)",
        description=(
            "Describe the MINI's fixed UVC GET_INFO boundary. The default is completely "
            "inert: no backend is constructed, no device is resolved or opened, and no "
            "ioctl is sent. The exact opt-in performs one and only one fixed request: "
            "bmRequestType=0xa1, bRequest=0x86, wValue=0x1200, wIndex=0x0400, "
            "wLength=1, timeout 1000 ms, through the shared interface-0 lifecycle. "
            "There is no setting change, retry, second transfer, V4L2 path or "
            "caller-controlled setup field. A later physical task must perform its "
            "separate immutable blue-light-off finalizer."
        ),
    )
    add_roots(rgb_exposure_info)
    rgb_exposure_info.add_argument(
        "--serial",
        help=("target this exact USB serial (required by --execute-rgb-exposure-info-read)"),
    )
    rgb_exposure_info.add_argument(
        "--execute-rgb-exposure-info-read",
        action="store_true",
        help=(
            "opt in to one fixed GET_INFO for serial "
            f"{RGB_EXPOSURE_INFO_AUTHORIZED_SERIAL}; no setting or retry"
        ),
    )
    rgb_exposure_info.add_argument(
        "--json-out", type=Path, help="also write the JSON result to this path"
    )
    rgb_exposure_info.set_defaults(func=_cmd_rgb_exposure_info)

    scale = sub.add_parser(
        "depth-scale",
        # No prefix matching: the execute flag must be typed in full. argparse
        # would otherwise accept a bare "--execute" as an abbreviation of
        # --execute-depth-scale-read, and a flag that arms a write to the
        # scanner should not be reachable by a prefix that means something else
        # on a sibling command.
        allow_abbrev=False,
        help=(
            f"read the vendor depth-scale property 0x{DEPTH_SCALE_PROPERTY_ID:03x} once "
            "(opt-in; WRITES the property id to the scanner)"
        ),
        description=(
            "Perform, exactly once, the bounded two-transfer libuvcReadValueExtension exchange "
            f"the vendor firmware serves for property 0x{DEPTH_SCALE_PROPERTY_ID:03x}: a 4-byte "
            f"SET_CUR carrying {PROPERTY_PAYLOAD.hex()} and one {RESULT_LENGTH}-byte GET_CUR "
            f"reply, both to entity {ENTITY_ID} selector 0x{PROPERTY_SELECTOR:02x} on interface "
            f"{OWNED_INTERFACE}, inside the same fixed detach/claim/release/reconnect lifecycle "
            "as depth-intrinsics. This is not a property reader: the id is a compile-time "
            "constant, and there is no flag for an id, a selector, an entity, an interface, a "
            "length, a timeout or a repetition. There is no protocol-clear transfer, because "
            "the vendor primitive has none. The reply is reported in full as the primary "
            f"evidence; only an exact {RESULT_LENGTH}-byte reply carrying marker 4 and a "
            "positive int32 denominator yields a scale, and only a run whose driver handback "
            "succeeded offers it. The scalar's physical unit is unknown from this evidence. "
            "Without --execute-depth-scale-read nothing is opened and nothing is sent."
        ),
    )
    add_roots(scale)
    scale.add_argument(
        "--serial",
        help="target this exact USB serial (required by --execute-depth-scale-read)",
    )
    scale.add_argument(
        "--execute-depth-scale-read",
        action="store_true",
        help=(
            "opt in to actually WRITING to the scanner: detach uvcvideo from interface 0, "
            f"claim it, send the 4-byte 0x{DEPTH_SCALE_PROPERTY_ID:03x} SET_CUR, read the one "
            f"{RESULT_LENGTH}-byte reply, then release and reconnect the driver. Requires "
            f"--serial, and refuses any serial but {DEPTH_SCALE_AUTHORIZED_SERIAL}."
        ),
    )
    scale.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    scale.set_defaults(func=_cmd_depth_scale)

    frame = sub.add_parser(
        "depth-frame",
        # No prefix matching: the execute flag must be typed in full, for the
        # same reason depth-scale disables it - a bare "--execute" means
        # something else on a sibling command and must not arm this one.
        allow_abbrev=False,
        help=(
            f"capture one raw {FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT} depth frame through "
            "uvcvideo (opt-in; STARTS a stream on the scanner)"
        ),
        description=(
            "Attempt, exactly once, one bounded V4L2/MMAP capture of a single raw depth "
            f"frame from the {CAPTURE_ROLE} capture node, which is resolved from the USB "
            "interface string descriptor and the V4L2 node index - never from a /dev/videoN "
            f"number and never from a caller-supplied path. The mode is fixed at {FOURCC!r} "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}; there is no flag for a format, a size, a frame "
            "rate, a buffer count, a timeout, a retry or a control. The lifecycle is "
            "QUERYCAP, G_FMT, one S_FMT, one REQBUFS for "
            f"{REQUESTED_BUFFERS} MMAP buffers, QUERYBUF/mmap/QBUF for each, one STREAMON, "
            f"one poll of at most {CAPTURE_DEADLINE_S}s and at most one DQBUF - then "
            "STREAMOFF, unmap, REQBUFS(0) and close, each exactly once, under a shield "
            f"neither SIGINT nor SIGTERM can abort. {EXPECTED_INTERFACE_DRIVER} stays bound "
            "throughout: no usbfs claim, no interface detach, no vendor or HID transfer. A "
            "dequeued frame is written to a local directory with its SHA-256; a zero-frame "
            "timeout is reported in full as a bounded result rather than retried. No "
            "calibration, scale or metric unit is applied to the bytes. Without "
            "--execute-depth-frame-capture nothing is opened."
        ),
    )
    add_roots(frame)
    frame.add_argument(
        "--serial",
        help="target this exact USB serial (required by --execute-depth-frame-capture)",
    )
    frame.add_argument(
        "--execute-depth-frame-capture",
        action="store_true",
        help=(
            "opt in to actually STREAMING from the scanner: open the resolved depth capture "
            f"node, negotiate {FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT}, map and queue "
            f"{REQUESTED_BUFFERS} buffers, start the stream and wait once for at most "
            f"{CAPTURE_DEADLINE_S}s for one frame. Requires --serial, and refuses any "
            f"serial but {DEPTH_FRAME_AUTHORIZED_SERIAL}."
        ),
    )
    frame.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "where to write the raw frame and its tightly packed copy. Shapes only local "
            f"evidence, never the hardware. Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    frame.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    frame.add_argument(
        "--session-dir",
        type=Path,
        help=(
            "record this capture into a scan session manifest at this directory "
            "(created if absent, continued if present). Only takes effect with "
            "--execute-depth-frame-capture, and only once a whole frame has been "
            "preserved and re-hashed off disk; a failed or short capture is still "
            "recorded, as a failed stage."
        ),
    )
    frame.set_defaults(func=_cmd_depth_frame)

    rgb = sub.add_parser(
        "rgb-frame",
        # No prefix matching, for the same reason depth-frame disables it: the
        # execute flag must be typed in full, and no abbreviation of one
        # command's opt-in may ever arm another's.
        allow_abbrev=False,
        help=(
            f"capture one {RGB_FOURCC!r} {RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT} RGB frame "
            "through uvcvideo (opt-in; STARTS a stream on the scanner)"
        ),
        description=(
            "Attempt, exactly once, one bounded V4L2/MMAP capture of a single compressed "
            f"frame from the {RGB_ROLE} capture node, which is resolved from the USB "
            "interface string descriptor and the V4L2 node index - never from a /dev/videoN "
            "number and never from a caller-supplied path. This is the same engine, the "
            "same nine ioctls and the same shielded cleanup the depth-frame command uses; "
            f"only the fixed profile differs. The mode is {RGB_FOURCC!r} "
            f"{RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}, the only mode this camera advertises, "
            "at the descriptor's own default interval - no VIDIOC_S_PARM is issued, so the "
            "frame rate is quoted from the descriptor and never claimed as measured. There "
            "is no flag for a format, a size, a frame rate, a buffer count, a timeout, a "
            "retry, a JPEG quality or a control. The dequeued payload is inspected "
            "structurally - SOI, marker segments, byte stuffing, restart markers, the SOF's "
            "own declared size, a terminal EOI - without decoding a single pixel, and is "
            f"written verbatim to {RGB_ARTIFACT_NAME} with its SHA-256 whether or not it "
            "turns out to be well-formed. Nothing is decoded, resized or colour-corrected, "
            "and no colour accuracy, white balance or synchronisation with the depth camera "
            f"is claimed. Without {RGB_EXECUTE_FLAG} nothing is opened."
        ),
    )
    add_roots(rgb)
    rgb.add_argument(
        "--serial",
        help=f"target this exact USB serial (required by {RGB_EXECUTE_FLAG})",
    )
    rgb.add_argument(
        RGB_EXECUTE_FLAG,
        action="store_true",
        help=(
            "opt in to actually STREAMING from the scanner: open the resolved RGB capture "
            f"node, negotiate {RGB_FOURCC!r} {RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}, map and "
            f"queue {RGB_REQUESTED_BUFFERS} buffers (the descriptor asks for "
            f"{RGB_DESCRIPTOR_FRAME_BUFFER_BYTES} B each; uvcvideo was measured allocating "
            f"{RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES} B each, which is the fixed per-buffer "
            f"ceiling and at most {RGB_MAX_TOTAL_MAPPED_BYTES} B mapped in total), start "
            "the stream and wait once "
            f"for at most {RGB_CAPTURE_DEADLINE_S}s for one frame. Requires --serial, and "
            f"refuses any serial but {DEPTH_FRAME_AUTHORIZED_SERIAL}."
        ),
    )
    rgb.add_argument(
        "--output-dir",
        type=Path,
        help=(
            f"where to write {RGB_ARTIFACT_NAME}. Shapes only local evidence, never the "
            f"hardware. Default: {RGB_DEFAULT_OUTPUT_DIR}"
        ),
    )
    rgb.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    rgb.add_argument(
        "--session-dir",
        type=Path,
        help=(
            "record this capture into a scan session manifest at this directory "
            f"(created if absent, continued if present). Only takes effect with "
            f"{RGB_EXECUTE_FLAG}, and only once a whole frame has been preserved and "
            "re-hashed off disk; a failed or short capture is still recorded, as a "
            "failed stage."
        ),
    )
    rgb.set_defaults(func=_cmd_rgb_frame)

    rgb_controls = sub.add_parser(
        "rgb-controls",
        allow_abbrev=False,
        help="describe or read the fixed standard RGB V4L2 controls (read-only opt-in)",
        description=(
            "Inventory the fixed standard exposure, gain and white-balance controls exposed "
            "by the RGB uvcvideo node. The default is inert and performs no discovery, open "
            "or ioctl. With the exact opt-in and authorized serial, it resolves the RGB node "
            "by stable identity, opens it read-only, issues one QUERYCTRL per fixed control "
            "and at most one G_CTRL for each readable scalar/menu control, then closes once. "
            "It never changes a control, enumerates menus, negotiates a format or starts a "
            "stream; the vendor RGB exposure extension is outside this command."
        ),
    )
    add_roots(rgb_controls)
    rgb_controls.add_argument(
        "--serial",
        help=f"target this exact USB serial (required by {RGB_CONTROLS_EXECUTE_OPT_IN})",
    )
    rgb_controls.add_argument(
        RGB_CONTROLS_EXECUTE_OPT_IN,
        dest="read_rgb_controls",
        action="store_true",
        help=(
            "opt in to one read-only RGB control inventory; requires --serial "
            f"{RGB_CONTROLS_AUTHORIZED_SERIAL}"
        ),
    )
    rgb_controls.add_argument(
        "--json-out", type=Path, help="also write the JSON result to this path"
    )
    rgb_controls.set_defaults(func=_cmd_rgb_controls)

    paired = sub.add_parser(
        "paired-capture",
        allow_abbrev=False,
        help="dry-run the bounded depth-first/RGB-second paired capture plan",
        description=(
            "Run the exact bounded experiment derived from the pinned 3DViewer ordering. "
            "The default is offline and opens nothing; execution requires the exact opt-in "
            "and authorized serial."
        ),
    )
    paired.add_argument("--json", action="store_true", help="emit JSON instead of text")
    paired.add_argument("--json-out", type=Path, help="also write the plan to this path")
    paired.add_argument(
        "--output-dir",
        type=Path,
        help="local evidence directory for an opted-in paired run",
    )
    paired.add_argument("--serial", help="required exact scanner serial for execution")
    paired.add_argument(
        paired_capture.EXECUTE_OPT_IN,
        dest="start_depth_first_rgb_second",
        action="store_true",
        help=(
            "explicitly opt in to starting both streams; requires --serial "
            f"{paired_capture.AUTHORIZED_SERIAL}"
        ),
    )
    paired.set_defaults(func=_cmd_paired_capture)

    series = sub.add_parser(
        "paired-rgb-series",
        allow_abbrev=False,
        help="dry-run the bounded multi-frame depth-first/RGB-series diagnostic",
        description=(
            "Run the fixed depth-first paired RGB-series diagnostic. The default is offline "
            "and opens nothing; execution requires the exact opt-in, authorized serial and "
            "an explicit output directory."
        ),
    )
    series.add_argument("--json", action="store_true", help="emit JSON instead of text")
    series.add_argument("--json-out", type=Path, help="also write the result to this path")
    series.add_argument("--output-dir", type=Path, help="local evidence directory for execution")
    series.add_argument("--serial", help="required exact scanner serial for execution")
    series.add_argument(
        "--read-rgb-controls-during-series",
        action="store_true",
        dest="read_rgb_controls_during_series",
        help=(
            "read gain, exposure-auto and exposure-absolute once at each bounded series point; "
            "requires the exact series opt-in"
        ),
    )
    series.add_argument(
        paired_capture.RGB_SERIES_EXECUTE_OPT_IN,
        dest="start_depth_first_rgb_series",
        action="store_true",
        help=(
            "opt in to one depth-first stream and the fixed finite RGB series; requires --serial "
            f"{paired_capture.AUTHORIZED_SERIAL}"
        ),
    )
    series.set_defaults(func=_cmd_paired_rgb_series)
    session = sub.add_parser(
        "session",
        allow_abbrev=False,
        help="inspect a scan session manifest offline",
        description=(
            "Read a scan session's versioned manifest and report its state, stages, "
            "counters, artifacts and limitations. Entirely offline: it opens no device, "
            "runs no SDK probe and writes nothing. --verify re-hashes every recorded "
            "artifact, so a session that lost or gained bytes cannot read as intact."
        ),
    )
    session.add_argument("path", type=Path, help="a session directory, or a manifest.json path")
    session.add_argument("--json", action="store_true", help="emit the manifest as JSON")
    session.add_argument("--verify", action="store_true", help="re-hash every recorded artifact")
    session.add_argument(
        "--redact",
        action="store_true",
        help="replace identifying values, for a shareable support report",
    )
    session.set_defaults(func=_cmd_session)

    to_ply = sub.add_parser(
        DEPTH_FRAME_TO_PLY_COMMAND,
        allow_abbrev=False,
        help="convert a captured raw depth frame on disk into a PLY point cloud (offline)",
        description=(
            f"Deproject one previously captured {EXPECTED_FRAME_BYTES}-byte {FOURCC!r} "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} depth frame into camera-frame points and write "
            "them as a PLY. This is a file-to-file transform and nothing else: it resolves "
            "no scanner, opens no device node and runs no vendor-SDK probe, so it can be "
            "run with the hardware unplugged and on a frame captured months ago. Every "
            "calibration input is required on the command line, because this command "
            "carries no unit's fx, fy, cx, cy or depth scale as a default: applying numbers "
            "the caller never stated is exactly how an unvalidated cloud comes to look "
            "calibrated. NOTHING it writes is metrically validated. No physical measurement "
            "in this repository has related a converted distance to a millimetre, a metre "
            "or any other length, and the factory depth scalar's own physical unit is "
            "unproven, so the coordinates are unitless numbers. That refusal to claim a "
            "unit, and every other limitation of the conversion, is written into the PLY "
            "header, which is the only place a consumer of the file will ever find it."
        ),
    )
    to_ply.add_argument(
        "path", type=Path, help="the captured raw depth frame to convert (read, never written)"
    )
    to_ply.add_argument(
        "output", type=Path, help="the PLY to create; an existing name is refused, never replaced"
    )
    to_ply.add_argument(
        "--fx",
        type=float,
        required=True,
        help=(
            "focal length in x, in pixels of the intrinsics record's own frame. Required: "
            "this command has no default for it and will not read one from a device"
        ),
    )
    to_ply.add_argument(
        "--fy", type=float, required=True, help="focal length in y, in the same pixels (required)"
    )
    to_ply.add_argument(
        "--cx",
        type=float,
        required=True,
        help="principal point column, in the same pixels; x = 0 lands on it (required)",
    )
    to_ply.add_argument(
        "--cy",
        type=float,
        required=True,
        help="principal point row, in the same pixels; y = 0 lands on it (required)",
    )
    to_ply.add_argument(
        "--depth-scale",
        type=float,
        required=True,
        help=(
            "the scalar each raw uint16 sample is multiplied by. Required, and its physical "
            f"unit is not established by anything here: {UNIT_DISCLAIMER}"
        ),
    )
    to_ply.add_argument(
        "--calibration-source",
        required=True,
        help=(
            "where the numbers above came from, in your own words (a device read, a stored "
            "blob, a guess). Required: the PLY header records it, and a blank source would "
            "read as 'no calibration was applied' rather than as 'nobody said'"
        ),
    )
    to_ply.add_argument(
        "--intrinsics-width",
        type=int,
        default=FRAME_WIDTH,
        help=(
            "the raster fx, fy, cx and cy are in pixels of; defaults to the frame's own "
            f"{FRAME_WIDTH}. A record for another raster is converted as asked and the "
            "mismatch is reported, never rescaled"
        ),
    )
    to_ply.add_argument(
        "--intrinsics-height",
        type=int,
        default=FRAME_HEIGHT,
        help=f"the rows those values are in pixels of; defaults to the frame's own {FRAME_HEIGHT}",
    )
    to_ply.add_argument(
        "--intrinsics-sha256",
        help=(
            "a SHA-256 hex digest of the calibration blob these values were read from, for "
            "the header. Omitted means 'none recorded'; a string that is not a digest is refused"
        ),
    )
    to_ply.add_argument(
        "--assume-unit",
        help=(
            "state what raw * depth-scale is ASSUMED to measure. Omitted, the file says a "
            "unit is declined rather than naming one, which is the only honest default: "
            "nothing here has validated any distance, whatever you pass"
        ),
    )
    to_ply.add_argument(
        "--ply-format",
        type=PlyFormat,
        choices=list(PlyFormat),
        default=PlyFormat.BINARY_LITTLE_ENDIAN,
        help="PLY encoding to write (default: binary_little_endian)",
    )
    to_ply.add_argument("--json", action="store_true", help="emit JSON instead of text")
    to_ply.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    to_ply.set_defaults(func=_cmd_depth_frame_to_ply)

    sweep = sub.add_parser(
        SWEEP_COMMAND,
        help="capture a continuous sweep of depth frames to a directory "
        "(opt-in; STARTS a stream on the scanner)",
        description=(
            "Stream depth frames while the object or the scanner moves. Continuous "
            "capture is what makes registration work: measured on this hardware, "
            "large angular steps make alignment fail silently, while a streamed "
            "sweep at about 2 degrees per frame does not. Come back around to "
            "where you started - a path that revisits its own viewpoints is the "
            "only thing that lets drift be detected and corrected afterwards."
        ),
    )
    sweep.add_argument("output", type=Path, help="directory to write raw frames into")
    sweep.add_argument(
        "--frames", type=int, default=1250, help="frames to DELIVER before stopping (default 1250)"
    )
    sweep.add_argument(
        "--decimate",
        type=int,
        default=3,
        help="keep one frame in N (default 3, giving about 2 degrees per kept frame "
        "on a turntable running one revolution in two minutes)",
    )
    sweep.add_argument("--serial", help="scanner serial to target")
    sweep.add_argument(
        "--duration", type=float, default=400.0, help="hard stop after this many seconds"
    )
    sweep.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sweep.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    sweep.set_defaults(func=_cmd_sweep)

    recon = sub.add_parser(
        RECONSTRUCT_COMMAND,
        help="register a captured sweep and report held-out closure error (offline)",
        description=(
            "Register a directory of captured frames and score the result against "
            "evidence the optimiser was never given. Half the revisited viewpoints "
            "are withheld and used only to measure the answer, because a pipeline "
            "scored on constraints it was fitted to will report success on a scan "
            "that is badly wrong - measured on real data, a trajectory 65 degrees "
            "out reported healthy alignment rmse, full frame overlap and good "
            "self-consistency. This command resolves no scanner and opens no "
            "device node."
        ),
    )
    recon.add_argument("frames", type=Path, help="directory of raw depth frames")
    recon.add_argument("--stride", type=int, default=1, help="use one frame in N (default 1)")
    recon.add_argument(
        "--backend",
        default="auto",
        help="compute backend: auto, python, numpy or scipy (default auto)",
    )
    recon.add_argument(
        "--workers", type=int, default=1, help="processes for pairwise alignment (default 1)"
    )
    recon.add_argument(
        "--point-to-point",
        action="store_true",
        help="use point-to-point matching instead of point-to-plane. Measured on a "
        "real capture, point-to-point drifted 14.4 degrees per lap where "
        "point-to-plane drifted 2.6",
    )
    recon.add_argument(
        "--no-robust",
        action="store_true",
        help="do not reject closure constraints that disagree with the consensus",
    )
    recon.add_argument("--poses-out", type=Path, help="write the registered poses here as JSON")
    recon.add_argument("--json", action="store_true", help="emit JSON instead of text")
    recon.add_argument("--json-out", type=Path, help="also write the JSON result to this path")
    recon.set_defaults(func=_cmd_reconstruct)

    return parser


def _cmd_session(args: argparse.Namespace) -> int:
    """Inspect a scan session offline. Never opens a device or the SDK.

    Reports what the manifest actually says rather than what the directory
    looks like, and re-hashes every recorded artifact so a session that has
    lost or gained bytes since it was written cannot read as intact.
    """
    manifest_path = args.path if args.path.is_file() else args.path / MANIFEST_NAME
    try:
        manifest = read_manifest(manifest_path)
    except SessionError as exc:
        print(f"cannot read session: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    problems: dict[str, str] = {}
    if args.verify:
        root = manifest_path.parent
        for artifact in manifest.artifacts:
            target = root / artifact.path
            if not target.is_file():
                problems[artifact.path] = "missing"
                continue
            payload = target.read_bytes()
            if len(payload) != artifact.bytes:
                problems[artifact.path] = f"expected {artifact.bytes} bytes, found {len(payload)}"
            elif hashlib.sha256(payload).hexdigest() != artifact.sha256:
                problems[artifact.path] = "checksum mismatch"

    payload = manifest.redacted() if args.redact else manifest.as_dict()
    if args.verify:
        payload = {**payload, "artifact_verification": problems or "all recorded artifacts match"}

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"session      {manifest.session_id}")
        print(f"state        {manifest.state}")
        print(f"complete     {manifest.is_complete}")
        print(f"schema       {manifest.schema_version}")
        print(f"software     {manifest.software_version}")
        if manifest.device is not None:
            serial = "[redacted]" if args.redact else manifest.device.serial
            print(
                f"device       {manifest.device.vendor_id:04x}:"
                f"{manifest.device.product_id:04x}  {serial}"
            )
        for stage in manifest.stages:
            counters = " ".join(f"{k}={v}" for k, v in sorted(stage.counters.items()))
            print(f"  stage      {stage.name}: {stage.state}{'  ' + counters if counters else ''}")
            if stage.error:
                print(f"    error    {stage.error}")
            for artifact in stage.artifacts:
                print(f"    artifact {artifact.path}  {artifact.bytes} B  {artifact.kind}")
        if manifest.state is SessionState.RUNNING:
            # Reading a manifest cannot tell whether a process still owns the
            # session, so say so rather than implying it is live.
            print(
                "note         recorded as running; if no process still owns this "
                "session it was interrupted, and opening it will relabel it incomplete"
            )
        for limitation in manifest.limitations:
            print(f"  limitation {limitation}")
        if manifest.error:
            print(f"error        {manifest.error}")
        if args.verify:
            if problems:
                for path, detail in sorted(problems.items()):
                    print(f"  MISMATCH   {path}: {detail}")
            else:
                print("verification all recorded artifacts match")

    if problems:
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    # An unfinished session is not an error to inspect; it reports its own state.
    return 0


#: The subcommand name, written once so the parser, the report and the strings
#: that name the caller of this command cannot drift apart.
DEPTH_FRAME_TO_PLY_COMMAND = "depth-frame-to-ply"

#: The first thing the human rendering prints and the last thing it repeats.
#: It is stated before any number appears on screen, because a reader who stops
#: after the point count must already have been told the count is not evidence
#: of a measured shape.
NOT_VALIDATED_BANNER = (
    "THESE COORDINATES ARE NOT METRICALLY VALIDATED. Nothing in this repository has "
    "compared a distance it produces against a ruler, a reference target or the vendor "
    "SDK's own output, and the depth scalar's physical unit is unproven, so every "
    "coordinate here is a unitless quantity and not a length. The LIMITATIONS block is "
    "part of this result, not a disclaimer to skip."
)


def _injected_intrinsics(args: argparse.Namespace) -> IntrinsicsRecord:
    """The caller's fx, fy, cx and cy as a record, through the real parser.

    Packed into the vendor's own 40-byte layout and decoded by
    :func:`~scanner3d.intrinsics.parse_intrinsics` rather than assembled field
    by field, so a record built from the command line is checked by exactly the
    code that checks one read off the device, and carries the same structural
    and plausibility observations into the report.
    """
    try:
        blob = struct.pack(
            INTRINSICS_FORMAT,
            args.intrinsics_width,
            args.intrinsics_height,
            args.fx,
            0.0,
            args.cx,
            0.0,
            args.fy,
            args.cy,
            0.0,
            0.0,
            1.0,
        )
    except (struct.error, OverflowError) as exc:
        raise DepthIntrinsicsUnusable(
            f"the stated calibration does not fit the vendor {INTRINSICS_FORMAT!r} record this "
            f"project decodes ({INTRINSICS_LENGTH} bytes): {exc}. Nothing was converted"
        ) from exc
    record = parse_intrinsics(blob)
    if record is None:  # pragma: no cover - struct.pack always yields INTRINSICS_LENGTH bytes
        raise DepthIntrinsicsUnusable(
            f"the stated calibration did not decode as a {INTRINSICS_LENGTH}-byte vendor record"
        )
    return record


def _stated_unit(args: argparse.Namespace) -> DepthUnitAssumption:
    """The caller's unit assertion, or the standing refusal to name one.

    ``--assume-unit`` is carried with the caller named as its author and with a
    basis that says out loud that this command measured nothing: a label a user
    typed is an assumption reaching the header, never evidence. Omitting the
    flag yields :data:`~scanner3d.depth_metric.UNVALIDATED_UNIT`, which declines
    the question in the report instead of skipping it.
    """
    if args.assume_unit is None:
        return UNVALIDATED_UNIT
    return DepthUnitAssumption(
        label=args.assume_unit,
        stated_by=f"the caller of 'scanner3d {DEPTH_FRAME_TO_PLY_COMMAND}', via --assume-unit",
        basis=(
            "asserted on the command line and nowhere else. This command performed no "
            "measurement that could support it, and no physical measurement in this "
            "repository has related a converted distance to any length"
        ),
    )


def _render_cloud_export(export: CloudExport, source_path: Path, stream: TextIO) -> None:
    """Human rendering of one offline frame-to-PLY conversion.

    Ordered so the caveats cannot be missed: the unvalidated unit is stated
    before the first number, the assumed unit is labelled as an assumption where
    it appears, and the complete limitation block - the same one written into
    the PLY header - is the last thing on screen, because everything above it is
    unvalidated by everything in this repository.
    """
    conversion = export.conversion
    cloud = export.cloud
    provenance = cloud.provenance
    calibration = provenance.calibration
    samples = conversion.width * conversion.height

    print(f"scanner3d {DEPTH_FRAME_TO_PLY_COMMAND} {__version__}", file=stream)
    print(NOT_VALIDATED_BANNER, file=stream)
    print(file=stream)

    print("Source frame (read from disk; no device was opened, no SDK probe was run)", file=stream)
    print(f"  path       {source_path}", file=stream)
    print(
        f"  bytes      {conversion.frame_bytes} (a {FOURCC!r} {conversion.width}x"
        f"{conversion.height} frame is exactly {EXPECTED_FRAME_BYTES})",
        file=stream,
    )
    print(f"  sha256     {conversion.frame_sha256}", file=stream)
    print(file=stream)

    values = conversion.intrinsics.values
    print("Calibration (yours: this command reads none from any device)", file=stream)
    print(
        f"  fx {values['fx']!r}  fy {values['fy']!r}  cx {values['cx']!r}  cy {values['cy']!r}",
        file=stream,
    )
    print(
        f"  pixels of  the {conversion.intrinsics.width}x{conversion.intrinsics.height} frame "
        "the stated record describes",
        file=stream,
    )
    print(f"  scale      {conversion.depth_scale!r}", file=stream)
    print(f"  applied as {DEPROJECTION_EXPRESSION}", file=stream)
    print(f"  SCALE UNIT {calibration.depth_scale_unit}", file=stream)
    print(f"  source     {calibration.source}", file=stream)
    print(f"  intrinsics sha256 {calibration.intrinsics_sha256 or 'none recorded'}", file=stream)
    print(file=stream)

    unit = conversion.unit
    print("Assumed unit (an assertion, never a measurement)", file=stream)
    print(f"  label      {unit.label}", file=stream)
    print(f"  stated by  {unit.stated_by}", file=stream)
    print(f"  basis      {unit.basis}", file=stream)
    print(file=stream)

    counts = ", ".join(f"{reason}={count}" for reason, count in conversion.counts_by_reason.items())
    print("Conversion", file=stream)
    print(f"  points     {conversion.included} of {samples} samples became points", file=stream)
    print(f"  excluded   {conversion.excluded} produced none ({counts})", file=stream)
    print(
        f"  accounted  {conversion.accounted} of {samples} "
        f"(every sample accounted for: {conversion.accounting_holds})",
        file=stream,
    )
    print(f"  truncated  {conversion.truncated}", file=stream)
    extent = conversion.extent
    if extent is None:
        print("  extent     none: no sample in this frame produced a point", file=stream)
    else:
        print(
            f"  extent     x [{extent.min_x!r}, {extent.max_x!r}]  "
            f"y [{extent.min_y!r}, {extent.max_y!r}]  z [{extent.min_z!r}, {extent.max_z!r}]",
            file=stream,
        )
        print(
            f"             raw codes [{extent.min_raw}, {extent.max_raw}]; the x, y and z spans "
            "are in the assumed unit above and are not measured lengths",
            file=stream,
        )
    print("  observations (made and deliberately not acted on)", file=stream)
    for check in conversion.checks:
        print(f"    [{'ok  ' if check.ok else 'note'}] {check.name}: {check.detail}", file=stream)
    print(file=stream)

    print("Point cloud", file=stream)
    print(f"  path       {cloud.path}", file=stream)
    print(f"  format     {cloud.ply_format}", file=stream)
    print(
        f"  vertices   {cloud.point_count} "
        f"(every converted point written: {export.every_point_written})",
        file=stream,
    )
    print(f"  bytes      {cloud.byte_count}", file=stream)
    print(f"  sha256     {cloud.sha256}", file=stream)
    print(f"  convention {provenance.convention.summary}", file=stream)
    print(f"  units      {provenance.units}", file=stream)
    if cloud.point_count == 0:
        print(
            "  EMPTY      this file has zero vertices: the conversion ran and found no "
            "sample with depth. That is a record of a run, not of an empty space",
            file=stream,
        )
    print(file=stream)

    print("LIMITATIONS (every one of these is also written into the PLY header)", file=stream)
    for limitation in export.limitations:
        print(f"  - {limitation}", file=stream)
    print(file=stream)
    print(NOT_VALIDATED_BANNER, file=stream)
    print(f"Status: wrote {cloud.point_count} vertices to {cloud.path}", file=stream)


def _cmd_depth_frame_to_ply(args: argparse.Namespace) -> int:
    """Convert a captured raw depth frame on disk into a PLY. Entirely offline.

    A file-to-file transform and nothing else: it resolves no scanner, opens no
    device node and runs no SDK probe, so every calibration input is the
    caller's and is required rather than defaulted to this unit's numbers.

    The conversion and the writing are :mod:`scanner3d.cloud_export`'s, so the
    caveats that reach the header are the ones that module carries across and
    not a set this command restates. What is decided here is only what a command
    line has to decide: which refusal a user sees, and with which exit code.
    """
    if not args.path.is_file():
        print(
            f"cannot read depth frame: {args.path} is not a readable regular file",
            file=sys.stderr,
        )
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    try:
        size = args.path.stat().st_size
    except OSError as exc:
        print(f"cannot read depth frame: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    if size != EXPECTED_FRAME_BYTES:
        # Checked before the read, so a file of any other size is never loaded,
        # reshaped or padded into something that looks like a frame.
        print(
            f"not one {FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT} frame: {args.path} holds {size} "
            f"bytes and a frame of this mode is exactly {EXPECTED_FRAME_BYTES}. Nothing was "
            "read: padding or truncating a buffer of another length would invent pixels the "
            "device never sent",
            file=sys.stderr,
        )
        return EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    if args.output.exists() or args.output.is_symlink():
        # Ahead of the conversion rather than at the open, so a run that cannot
        # produce a file does not spend 256000 samples finding that out. The
        # writer refuses an existing name again underneath, which is what makes
        # the race safe rather than this check.
        print(
            f"refusing to overwrite {args.output}: this command never replaces an existing "
            "file. A cloud is derived evidence, and the run that wrote the one already there "
            "is not this run. Choose another name, or remove that file yourself",
            file=sys.stderr,
        )
        return EXIT_CODES[DoctorStatus.PERMISSION_DENIED]
    try:
        raw = args.path.read_bytes()
    except OSError as exc:
        print(f"cannot read depth frame: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    try:
        export = export_depth_frame(
            args.output,
            raw,
            intrinsics=_injected_intrinsics(args),
            depth_scale=args.depth_scale,
            unit=_stated_unit(args),
            calibration_source=args.calibration_source,
            intrinsics_sha256=args.intrinsics_sha256,
            ply_format=args.ply_format,
        )
    except ScannerError as exc:
        # Every refusal of the three modules involved arrives here, before a
        # path was opened, and each carries the status that earns its exit code.
        print(f"refusing that conversion: {exc}", file=sys.stderr)
        return EXIT_CODES[exc.status]
    except PointCloudWriteIncomplete as exc:
        print(f"the point cloud was not written: {exc}", file=sys.stderr)
        status = (
            DoctorStatus.PERMISSION_DENIED
            if exc.errno == errno.EEXIST
            else DoctorStatus.INTERNAL_ERROR
        )
        return EXIT_CODES[status]
    except OSError as exc:
        print(f"cannot write point cloud: {exc}", file=sys.stderr)
        return EXIT_CODES[DoctorStatus.INTERNAL_ERROR]

    payload: dict[str, object] = {
        **export.as_dict(),
        "command": {
            "name": DEPTH_FRAME_TO_PLY_COMMAND,
            "source_frame_path": str(args.path),
            "output_path": str(args.output),
            "offline": (
                "this command resolved no scanner, opened no device node and ran no vendor-SDK "
                "probe; every calibration input was supplied on the command line"
            ),
            "dimensional_accuracy_claimed": False,
        },
    }
    if not args.json:
        _render_cloud_export(export, args.path, sys.stdout)
    _emit(payload, args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
