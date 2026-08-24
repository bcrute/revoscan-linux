"""Bounded lifecycle for the depth/RGB paired experiment.

The lifecycle keeps a small injectable :class:`StreamBackend` seam for fake
verification and ships a concrete :class:`PairedV4l2Backend` assembled from
the shared fixed-profile V4L2 engine.  Planning remains harmless: the CLI
constructs that backend only for an explicit execution request with a local
output directory.  The ordering is the ordering observed in the pinned
3DViewer source: depth is completely started, the one-second SDK interval is
preserved, and RGB is then started.
"""

from __future__ import annotations

import enum
import errno as errno_mod
import fcntl
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Final, Protocol, cast

from scanner3d.control import Closer, IoctlFn, Opener
from scanner3d.errors import DeviceAbsentError, V4l2CaptureNotAllowed
from scanner3d.illumination_shutdown import (
    AUTHORIZED_SERIAL as ILLUMINATION_AUTHORIZED_SERIAL,
)
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    UsbfsIlluminationShutdownBackend,
)
from scanner3d.illumination_shutdown import (
    not_required as illumination_not_required,
)
from scanner3d.jpeg_inspect import inspect_jpeg
from scanner3d.ownership import (
    SignalGuard,
    SignalReceived,
    TerminationShield,
    cleanup_on_termination,
    defer_termination,
)
from scanner3d.rgb_frame import (
    RGB_CAPTURE_DEADLINE_S,
    RGB_FOURCC,
    RGB_FRAME_HEIGHT,
    RGB_FRAME_WIDTH,
    RGB_MAX_BUFFER_BYTES,
    RGB_MAX_RAW_ARTIFACT_BYTES,
    RGB_MAX_TOTAL_MAPPED_BYTES,
    RGB_PROFILE,
    RGB_REQUESTED_BUFFERS,
)
from scanner3d.v4l2_capture import (
    CAPTURE_DEADLINE_S,
    DEPTH_PROFILE,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_BUFFER_BYTES,
    MAX_RAW_ARTIFACT_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    REQUESTED_BUFFERS,
    ArtifactRecord,
    BufferInfo,
    CaptureOutcome,
    CaptureProfile,
    CaptureStatus,
    CleanupRecord,
    FrameArtifact,
    FrameVerdict,
    MmapFn,
    NodeSnapshot,
    Poller,
    PollRecord,
    PreparedV4l2Session,
    SeriesObservation,
    SeriesRun,
    Step,
    StepResult,
    V4l2FrameBackend,
    V4l2SessionError,
    _default_mmap,
    _default_opener,
    _default_poll,
    _write_new_file,
    preflight_capture_node,
    sha256_of,
)
from scanner3d.v4l2_controls import (
    RGB_SERIES_CONTROL_WHITELIST,
    VIDIOC_G_CTRL,
    ControlSpec,
    pack_control,
    parse_control,
)

AUTHORIZED_SERIAL: Final = "A22704522A6F00A64"
DEPTH_ROLE: Final = "depth"
RGB_ROLE: Final = "rgb"
EXECUTE_OPT_IN: Final = "--start-depth-first-rgb-second"
START_INTERVAL_S: Final = 1.0
MAX_DEPTH_FRAMES: Final = 1
MAX_RGB_FRAMES: Final = 1
RGB_SERIES_COUNT: Final = 3
RGB_SERIES_DEADLINE_S: Final = RGB_CAPTURE_DEADLINE_S
RGB_SERIES_MAX_TOTAL_BYTES: Final = RGB_MAX_RAW_ARTIFACT_BYTES * RGB_SERIES_COUNT
RGB_SERIES_EXECUTE_OPT_IN: Final = "--start-depth-first-rgb-series"
RGB_SERIES_ARTIFACT_PREFIX: Final = "rgb-series-"
RGB_SERIES_ARTIFACT_SUFFIX: Final = ".jpg"
RGB_SERIES_MAX_TRAILER_HEX_BYTES: Final = 256
RGB_SERIES_CONTROL_POINT_LIMIT: Final = RGB_SERIES_COUNT + 1


class PairedStatus(enum.StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


PAIRED_EXIT_CODES: Final[dict[PairedStatus, int]] = {
    PairedStatus.NOT_ATTEMPTED: 0,
    PairedStatus.COMPLETE: 0,
    PairedStatus.REFUSED: 10,
    PairedStatus.FAILED: 5,
    PairedStatus.CANCELLED: 13,
    PairedStatus.TIMEOUT: 4,
}
PAIRED_CLEANUP_FAILED_EXIT: Final = 9
PAIR_SIGNAL_GUARD: Final[SignalGuard] = partial(cleanup_on_termination, "paired V4L2 capture")


@dataclass(frozen=True, slots=True)
class StreamSpec:
    role: str
    fourcc: str
    width: int
    height: int
    requested_buffers: int
    max_buffer_bytes: int
    max_total_mapped_bytes: int
    deadline_s: float
    max_artifact_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "fourcc": self.fourcc,
            "width": self.width,
            "height": self.height,
            "size": f"{self.width}x{self.height}",
            "requested_buffers": self.requested_buffers,
            "max_buffer_bytes": self.max_buffer_bytes,
            "max_total_mapped_bytes": self.max_total_mapped_bytes,
            "deadline_s": self.deadline_s,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_frames": 1,
        }


@dataclass(frozen=True, slots=True)
class RgbSeriesObservation:
    """One raw RGB dequeue plus strict and prefix/trailer evidence."""

    index: int
    buffer: BufferInfo
    status: CaptureStatus
    detail: str
    artifact: FrameArtifact
    jpeg: Mapping[str, object]
    eoi_prefix_bytes: int | None
    eoi_prefix_sha256: str | None
    trailer_bytes: int
    trailer_hex: str
    trailer_hex_truncated: bool = False
    errno: int | None = None
    errno_name: str | None = None
    evidence: Mapping[str, object] | None = None
    preservation_error: BaseException | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "buffer": self.buffer.as_dict(),
            "status": str(self.status),
            "detail": self.detail,
            "artifact": self.artifact.as_dict(),
            "jpeg": dict(self.jpeg),
            "eoi_prefix_bytes": self.eoi_prefix_bytes,
            "eoi_prefix_sha256": self.eoi_prefix_sha256,
            "trailer_bytes": self.trailer_bytes,
            "trailer_hex": self.trailer_hex,
            "trailer_hex_truncated": self.trailer_hex_truncated,
            "errno": self.errno,
            "errno_name": self.errno_name,
            "evidence": None if self.evidence is None else dict(self.evidence),
        }


class RgbTelemetryRecordStatus(enum.StrEnum):
    VALUE = "value"
    IOCTL_ERROR = "ioctl_error"
    RETURNED_ID_MISMATCH = "returned_id_mismatch"
    INTERRUPTED = "interrupted"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class RgbTelemetryRecord:
    """One bounded G_CTRL attempt, including failures and returned identity."""

    point: int
    attempt: int
    control: ControlSpec
    requested_id: int
    returned_id: int | None
    value: int | None
    status: RgbTelemetryRecordStatus
    errno: int | None = None
    errno_name: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "point": self.point,
            "attempt": self.attempt,
            "control": self.control.as_dict(),
            "requested_id": f"0x{self.requested_id:08x}",
            "returned_id": (None if self.returned_id is None else f"0x{self.returned_id:08x}"),
            "value": self.value,
            "status": str(self.status),
            "errno": self.errno,
            "errno_name": self.errno_name,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RgbTelemetrySample:
    """A point-in-time fixed-order telemetry sample."""

    point: int
    observation_index: int | None
    buffer_index: int | None
    sequence: int | None
    timestamp_ns: int | None
    records: tuple[RgbTelemetryRecord, ...] = ()

    @property
    def complete(self) -> bool:
        return len(self.records) == len(RGB_SERIES_CONTROL_WHITELIST) and all(
            record.status is RgbTelemetryRecordStatus.VALUE for record in self.records
        )

    @property
    def error_count(self) -> int:
        return sum(record.status is not RgbTelemetryRecordStatus.VALUE for record in self.records)

    def as_dict(self) -> dict[str, object]:
        return {
            "point": self.point,
            "observation_index": self.observation_index,
            "buffer_index": self.buffer_index,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "complete": self.complete,
            "attempt_count": len(self.records),
            "records": [record.as_dict() for record in self.records],
        }


@dataclass(slots=True)
class RgbSeriesResult:
    """Cumulative RGB-series evidence, including partial delivery."""

    status: CaptureStatus
    detail: str
    observations: tuple[RgbSeriesObservation, ...] = ()
    polls: tuple[PollRecord, ...] = ()
    artifact_record: ArtifactRecord = field(
        default_factory=lambda: ArtifactRecord.not_needed("series not attempted", None)
    )
    deadline_s: float = RGB_SERIES_DEADLINE_S
    retained_bytes: int = 0
    max_total_bytes: int = RGB_SERIES_MAX_TOTAL_BYTES
    poll_count: int = 0
    dequeue_count: int = 0
    requeue_count: int = 0
    terminal_verdict: FrameVerdict | None = field(default=None, repr=False)
    last_payload: bytes | None = field(default=None, repr=False)
    telemetry_enabled: bool = False
    telemetry_samples: tuple[RgbTelemetrySample, ...] = ()

    @property
    def delivered_count(self) -> int:
        return len(self.observations)

    @property
    def sequence_gaps(self) -> tuple[int, ...]:
        sequences = [observation.buffer.sequence for observation in self.observations]
        return tuple(
            current - previous - 1
            for previous, current in pairwise(sequences)
            if current > previous + 1
        )

    @property
    def intervals_ns(self) -> tuple[int, ...]:
        timestamps = [observation.buffer.timestamp_ns for observation in self.observations]
        return tuple(current - previous for previous, current in pairwise(timestamps))

    @property
    def strict_success(self) -> bool:
        return (
            self.status is CaptureStatus.COMPLETE
            and self.delivered_count == RGB_SERIES_COUNT
            and all(
                observation.status is CaptureStatus.COMPLETE for observation in self.observations
            )
            and self.artifact_record.ok
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": str(self.status),
            "detail": self.detail,
            "deadline_s": self.deadline_s,
            "requested_count": RGB_SERIES_COUNT,
            "delivered_count": self.delivered_count,
            "retained_bytes": self.retained_bytes,
            "max_total_bytes": self.max_total_bytes,
            "poll_count": self.poll_count,
            "dequeue_count": self.dequeue_count,
            "requeue_count": self.requeue_count,
            "sequence_gaps": list(self.sequence_gaps),
            "sequence_gap_count": sum(self.sequence_gaps),
            "intervals_ns": list(self.intervals_ns),
            "polls": [poll.as_dict() for poll in self.polls],
            "observations": [observation.as_dict() for observation in self.observations],
            "artifacts": self.artifact_record.as_dict(),
            "terminal_verdict": (
                None
                if self.terminal_verdict is None
                else {
                    "status": str(self.terminal_verdict.status),
                    "detail": self.terminal_verdict.detail,
                    "errno": self.terminal_verdict.errno,
                    "errno_name": self.terminal_verdict.errno_name,
                    "evidence": (
                        None
                        if self.terminal_verdict.evidence is None
                        else dict(self.terminal_verdict.evidence)
                    ),
                }
            ),
            "strict_success": self.strict_success,
            "telemetry": {
                "enabled": self.telemetry_enabled,
                "status": (
                    "disabled"
                    if not self.telemetry_enabled
                    else (
                        "errors"
                        if any(sample.error_count for sample in self.telemetry_samples)
                        else (
                            "partial"
                            if len(self.telemetry_samples) < RGB_SERIES_CONTROL_POINT_LIMIT
                            else "complete"
                        )
                    )
                ),
                "point_count": len(self.telemetry_samples),
                "max_points": RGB_SERIES_CONTROL_POINT_LIMIT,
                "control_count": len(RGB_SERIES_CONTROL_WHITELIST),
                "complete_points": sum(sample.complete for sample in self.telemetry_samples),
                "error_count": sum(sample.error_count for sample in self.telemetry_samples),
                "samples": [sample.as_dict() for sample in self.telemetry_samples],
            },
        }


DEPTH_SPEC: Final = StreamSpec(
    DEPTH_ROLE,
    FOURCC,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    REQUESTED_BUFFERS,
    MAX_BUFFER_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    CAPTURE_DEADLINE_S,
    MAX_RAW_ARTIFACT_BYTES,
)
RGB_SPEC: Final = StreamSpec(
    RGB_ROLE,
    RGB_FOURCC,
    RGB_FRAME_WIDTH,
    RGB_FRAME_HEIGHT,
    RGB_REQUESTED_BUFFERS,
    RGB_MAX_BUFFER_BYTES,
    RGB_MAX_TOTAL_MAPPED_BYTES,
    RGB_CAPTURE_DEADLINE_S,
    RGB_MAX_RAW_ARTIFACT_BYTES,
)


class StreamBackend(Protocol):
    """The deliberately small seam for a future V4L2 implementation."""

    def resolve_node(self, serial: str, role: str) -> str: ...

    def open(self, node: str, role: str) -> object: ...

    def configure(self, handle: object, spec: StreamSpec) -> None: ...

    def map_queue(self, handle: object, spec: StreamSpec) -> None: ...

    def stream_on(self, handle: object) -> None: ...

    def wait_frame(self, handle: object, deadline_s: float, max_frames: int) -> bytes | None: ...

    def stream_off(self, handle: object) -> None: ...

    def unmap(self, handle: object) -> None: ...

    def free_queue(self, handle: object) -> None: ...

    def close(self, handle: object) -> None: ...


class PairedBackendRefusal(RuntimeError):
    """A paired target was unsafe to use before any stream was opened."""


def _preserve_rgb_series_observation(
    observation: SeriesObservation, directory: Path
) -> RgbSeriesObservation:
    """Write one raw dequeue before the backend is allowed to requeue it."""
    payload = observation.payload
    report = inspect_jpeg(payload, ceiling=RGB_MAX_RAW_ARTIFACT_BYTES)
    jpeg = report.as_dict()
    prefix_bytes: int | None
    prefix_hash: str | None
    trailer = b""
    if report.has_eoi:
        prefix_bytes = report.byte_count - report.trailing_bytes
        prefix_hash = sha256_of(payload[:prefix_bytes])
        trailer = payload[prefix_bytes:]
    else:
        prefix_bytes = None
        prefix_hash = None
    shown_trailer = trailer[:RGB_SERIES_MAX_TRAILER_HEX_BYTES]
    path = directory / (
        f"{RGB_SERIES_ARTIFACT_PREFIX}{observation.index:03d}{RGB_SERIES_ARTIFACT_SUFFIX}"
    )
    digest = sha256_of(payload) if observation.payload_copied else ""
    try:
        if not observation.payload_copied:
            raise OSError(
                "the kernel reported a payload past the copy ceiling; raw bytes were not retained"
            )
        directory.mkdir(parents=True, exist_ok=True)
        _write_new_file(path, payload, ceiling=RGB_MAX_RAW_ARTIFACT_BYTES)
    except BaseException as exc:
        artifact = FrameArtifact(
            kind="mjpeg-series",
            path=None,
            byte_count=len(payload) if observation.payload_copied else observation.buffer.bytesused,
            sha256=digest,
            ok=False,
            detail=f"could not preserve observation {observation.index} at {path}: {exc}",
        )
        preservation_error = None if isinstance(exc, OSError) else exc
    else:
        artifact = FrameArtifact(
            kind="mjpeg-series",
            path=str(path),
            byte_count=len(payload),
            sha256=digest,
            ok=True,
            detail="dequeued bytes preserved byte-for-byte before any requeue",
        )
        preservation_error = None
    return RgbSeriesObservation(
        index=observation.index,
        buffer=observation.buffer,
        status=observation.verdict.status,
        detail=observation.verdict.detail,
        artifact=artifact,
        jpeg=jpeg,
        eoi_prefix_bytes=prefix_bytes,
        eoi_prefix_sha256=prefix_hash,
        trailer_bytes=len(trailer),
        trailer_hex=shown_trailer.hex(),
        trailer_hex_truncated=len(trailer) > len(shown_trailer),
        errno=observation.verdict.errno,
        errno_name=observation.verdict.errno_name,
        evidence=observation.verdict.evidence,
        preservation_error=preservation_error,
    )


@dataclass(slots=True)
class _PairedHandle:
    role: str
    prepared: PreparedV4l2Session
    target: NodeSnapshot
    fd: int
    steps: list[StepResult] = field(default_factory=list)
    primary: FrameVerdict | None = None
    cleanup_steps: list[StepResult] = field(default_factory=list)
    artifact: ArtifactRecord | None = None
    series: RgbSeriesResult | None = None
    identity_error: PairedBackendRefusal | None = None
    identity_exception: BaseException | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    cleanup_signals: list[int] = field(default_factory=list)
    closed: bool = False
    configured: bool = False
    queued: bool = False
    streamed: bool = False
    waited: bool = False
    series_attempted: bool = False
    telemetry_enabled: bool = False
    telemetry_samples: list[RgbTelemetrySample] = field(default_factory=list)


@dataclass(slots=True)
class PairedV4l2Backend:
    """Concrete paired backend assembled from the shared V4L2 engine.

    The backend intentionally exposes the small :class:`StreamBackend` seam
    used by :class:`PairedCapture`; it does not transcribe a second ioctl
    stack.  ``discovery`` and every syscall/filesystem boundary are injectable
    so all verification can use synthetic sysfs and fake kernel calls.
    """

    output_dir: Path = Path("/tmp/scanner3d-paired-capture")
    opener: Opener = _default_opener
    ioctl_fn: IoctlFn = fcntl.ioctl
    mmap_fn: MmapFn = _default_mmap
    poll_fn: Poller = _default_poll
    closer: Closer = os.close
    clock: Callable[[], float] = time.monotonic
    sysfs_root: Path = Path("/sys")
    dev_root: Path = Path("/dev")
    discovery: Callable[..., tuple[NodeSnapshot, object]] = preflight_capture_node
    fstat_fn: Callable[[int], os.stat_result] = os.fstat
    read_text_fn: Callable[[Path], str] = lambda path: path.read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    illumination_backend_factory: Callable[[Path, Path], UsbfsIlluminationShutdownBackend] = (
        lambda sysfs_root, dev_root: UsbfsIlluminationShutdownBackend(
            sysfs_root=sysfs_root, dev_root=dev_root
        )
    )
    _targets: dict[str, NodeSnapshot] = field(init=False, default_factory=dict)

    def _profile(self, role: str) -> CaptureProfile:
        if role == DEPTH_ROLE:
            return DEPTH_PROFILE
        if role == RGB_ROLE:
            return RGB_PROFILE
        raise PairedBackendRefusal(f"unknown paired role {role!r}")

    def _engine(self, role: str) -> V4l2FrameBackend:
        return V4l2FrameBackend(
            opener=self.opener,
            ioctl_fn=self.ioctl_fn,
            mmap_fn=self.mmap_fn,
            poll_fn=self.poll_fn,
            closer=self.closer,
            clock=self.clock,
            identity_check=self._identity_reason,
            sysfs_root=self.sysfs_root,
            dev_root=self.dev_root,
            profile=self._profile(role),
            illumination_backend_factory=self.illumination_backend_factory,
        )

    @staticmethod
    def _identity_key(target: NodeSnapshot) -> tuple[object, ...]:
        return (
            target.serial,
            target.sysfs_path,
            target.interface_number,
            target.interface_string,
            target.interface_driver,
            target.v4l_index,
            target.node_name,
            target.rdev,
            target.inode,
        )

    def resolve_node(self, serial: str, role: str) -> str:
        profile = self._profile(role)
        target, _device = self.discovery(
            serial, profile=profile, sysfs_root=self.sysfs_root, dev_root=self.dev_root
        )
        prior = self._targets.get(role)
        if prior is not None and self._identity_key(prior) != self._identity_key(target):
            raise PairedBackendRefusal(f"{role} identity drifted while resolving the pair")
        self._targets[role] = target
        if role == RGB_ROLE and DEPTH_ROLE in self._targets:
            depth = self._targets[DEPTH_ROLE]
            if self._identity_key(depth) == self._identity_key(target):
                raise PairedBackendRefusal("depth and RGB resolve to the same underlying node")
            if (depth.rdev, depth.inode) == (target.rdev, target.inode):
                raise PairedBackendRefusal(
                    "depth and RGB aliases share rdev/inode; refusing to open either node"
                )
        return target.node_path

    def _identity_reason(self, fd: int, target: NodeSnapshot) -> str | None:
        try:
            stat = self.fstat_fn(fd)
        except OSError as exc:
            return f"cannot fstat {target.node_path}: {exc}"
        if (stat.st_rdev, stat.st_ino) != (target.rdev, target.inode):
            return (
                f"{target.node_path} changed from rdev/inode "
                f"{target.rdev}/{target.inode} to {stat.st_rdev}/{stat.st_ino}"
            )
        sysfs = Path(target.sysfs_path)
        for name, expected in (("serial", target.serial), ("devnum", str(target.device_number))):
            try:
                current = self.read_text_fn(sysfs / name)
            except OSError as exc:
                return f"{sysfs / name} vanished: {exc}"
            if current != expected:
                return f"{name} changed from {expected!r} to {current!r} before the first ioctl"
        return None

    def open(self, node: str, role: str) -> object:
        target = self._targets.get(role)
        if target is None or target.node_path != node:
            raise PairedBackendRefusal(f"{role} was not resolved from the current snapshot")
        engine = self._engine(role)
        started = self.clock()
        prepared = PreparedV4l2Session.open(
            engine,
            target,
            self.output_dir / role,
            started,
        )
        handle = _PairedHandle(
            role=role,
            prepared=prepared,
            target=target,
            fd=prepared.fd,
        )
        handle.steps = prepared.steps
        try:
            prepared.validate_identity()
        except V4l2SessionError as exc:
            refusal = PairedBackendRefusal(exc.verdict.detail)
            handle.identity_error = refusal
            handle.primary = exc.verdict
        except BaseException as exc:
            handle.identity_exception = exc
            if isinstance(exc, (SignalReceived, KeyboardInterrupt)):
                handle.primary = FrameVerdict(CaptureStatus.INTERRUPTED, str(exc))
            else:
                handle.primary = FrameVerdict(
                    CaptureStatus.IOCTL_ERROR,
                    f"unexpected post-open identity error: {type(exc).__name__}: {exc}",
                )
        return handle

    def configure(self, handle: object, spec: StreamSpec) -> None:
        item = self._as_handle(handle)
        if item.identity_exception is not None:
            raise item.identity_exception
        if item.identity_error is not None:
            raise item.identity_error
        if spec.role != item.role:
            raise PairedBackendRefusal(f"profile role {spec.role!r} mismatches {item.role!r}")
        expected = DEPTH_SPEC if item.role == DEPTH_ROLE else RGB_SPEC
        if spec != expected:
            raise PairedBackendRefusal(
                f"{item.role} profile does not match its fixed specification"
            )
        item.prepared.configure()
        item.configured = True

    def map_queue(self, handle: object, spec: StreamSpec) -> None:
        item = self._as_handle(handle)
        if not item.configured:
            raise PairedBackendRefusal(f"{item.role} queue requested before configuration")
        item.prepared.queue()
        item.queued = True

    def stream_on(self, handle: object) -> None:
        item = self._as_handle(handle)
        if not item.queued or item.streamed:
            raise PairedBackendRefusal(
                f"{item.role} stream start was not exactly once after queueing"
            )
        item.prepared.stream_on()
        item.streamed = True

    def wait_frame(self, handle: object, deadline_s: float, max_frames: int) -> bytes | None:
        item = self._as_handle(handle)
        expected = DEPTH_SPEC if item.role == DEPTH_ROLE else RGB_SPEC
        if deadline_s != expected.deadline_s or max_frames != 1:
            raise PairedBackendRefusal("paired backend accepts one dequeue per role")
        if item.waited:
            raise PairedBackendRefusal(f"{item.role} dequeue was already attempted")
        if not item.streamed:
            raise PairedBackendRefusal(f"{item.role} wait requested before STREAMON")
        item.waited = True
        item.prepared.wait_once()
        item.primary = item.prepared.primary
        return item.prepared.state.frame

    def wait_rgb_series(
        self,
        handle: object,
        deadline_s: float,
        max_frames: int,
        max_total_bytes: int,
        read_rgb_controls: bool = False,
    ) -> RgbSeriesResult:
        """Collect and preserve the fixed RGB observation series."""
        item = self._as_handle(handle)
        if item.role != RGB_ROLE:
            raise PairedBackendRefusal("RGB series can only run on the RGB role")
        if deadline_s != RGB_SERIES_DEADLINE_S or max_frames != RGB_SERIES_COUNT:
            raise PairedBackendRefusal("RGB series accepts only its fixed count and deadline")
        if max_total_bytes != RGB_SERIES_MAX_TOTAL_BYTES:
            raise PairedBackendRefusal("RGB series accepts only its fixed retained-byte ceiling")
        if not item.streamed:
            raise PairedBackendRefusal("RGB series wait requested before STREAMON")
        if item.waited:
            raise PairedBackendRefusal("rgb dequeue was already attempted")
        # Consume the one-wait guard before entering the fallible series
        # operation. A partial/error result is still a consumed attempt.
        item.waited = True
        item.series_attempted = True
        item.telemetry_enabled = read_rgb_controls

        records: list[RgbSeriesObservation] = []
        artifact_files: list[FrameArtifact] = []
        output_dir = item.prepared.output_dir / "rgb-series"
        latest_run: SeriesRun | None = None

        def preserve(observation: SeriesObservation) -> None:
            record = _preserve_rgb_series_observation(observation, output_dir)
            records.append(record)
            artifact_files.append(record.artifact)
            if record.preservation_error is not None:
                raise record.preservation_error

        def progress(run: SeriesRun) -> None:
            nonlocal latest_run
            latest_run = run

        def make_result(run: SeriesRun) -> RgbSeriesResult:
            artifact_record = (
                ArtifactRecord(
                    attempted=True,
                    ok=bool(artifact_files) and all(artifact.ok for artifact in artifact_files),
                    note=(
                        "every dequeued RGB payload was preserved"
                        if artifact_files and all(artifact.ok for artifact in artifact_files)
                        else "one or more dequeued RGB payloads were not preserved"
                    ),
                    directory=str(output_dir),
                    files=tuple(artifact_files),
                    ceiling_bytes=RGB_MAX_RAW_ARTIFACT_BYTES,
                )
                if artifact_files
                else ArtifactRecord.not_needed(
                    "no RGB payload was dequeued, so no series files were written",
                    str(output_dir),
                    RGB_MAX_RAW_ARTIFACT_BYTES,
                )
            )
            return RgbSeriesResult(
                status=run.status,
                detail=run.detail,
                observations=tuple(records),
                polls=run.polls,
                retained_bytes=run.retained_bytes,
                max_total_bytes=run.max_total_bytes,
                poll_count=run.poll_count,
                dequeue_count=run.dequeue_count,
                requeue_count=run.requeue_count,
                terminal_verdict=run.terminal_verdict,
                artifact_record=artifact_record,
                last_payload=(
                    run.observations[-1].payload
                    if run.observations and run.observations[-1].payload_copied
                    else None
                ),
                telemetry_enabled=item.telemetry_enabled,
                telemetry_samples=tuple(item.telemetry_samples),
            )

        def read_controls(point: int, observation: SeriesObservation | None) -> None:
            """Read the fixed controls on this already-open RGB descriptor."""
            records: list[RgbTelemetryRecord] = []
            try:
                for attempt, control in enumerate(RGB_SERIES_CONTROL_WHITELIST):
                    requested = control.control_id
                    packed = pack_control(requested)
                    try:
                        self.ioctl_fn(item.fd, VIDIOC_G_CTRL, packed, True)
                        returned, value = parse_control(bytes(packed))
                    except OSError as exc:
                        number = exc.errno
                        records.append(
                            RgbTelemetryRecord(
                                point,
                                attempt,
                                control,
                                requested,
                                None,
                                None,
                                RgbTelemetryRecordStatus.IOCTL_ERROR,
                                number,
                                None if number is None else errno_mod.errorcode.get(number),
                                str(exc),
                            )
                        )
                        continue
                    except Exception as exc:
                        records.append(
                            RgbTelemetryRecord(
                                point,
                                attempt,
                                control,
                                requested,
                                None,
                                None,
                                RgbTelemetryRecordStatus.EXCEPTION,
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        raise
                    except BaseException as exc:
                        number = exc.errno if isinstance(exc, OSError) else None
                        status = (
                            RgbTelemetryRecordStatus.INTERRUPTED
                            if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                            else RgbTelemetryRecordStatus.EXCEPTION
                        )
                        records.append(
                            RgbTelemetryRecord(
                                point,
                                attempt,
                                control,
                                requested,
                                None,
                                None,
                                status,
                                number,
                                None if number is None else errno_mod.errorcode.get(number),
                                f"{type(exc).__name__}: {exc}",
                            )
                        )
                        raise
                    if returned != requested:
                        records.append(
                            RgbTelemetryRecord(
                                point,
                                attempt,
                                control,
                                requested,
                                returned,
                                value,
                                RgbTelemetryRecordStatus.RETURNED_ID_MISMATCH,
                                detail=(
                                    f"G_CTRL returned 0x{returned:08x}; expected 0x{requested:08x}"
                                ),
                            )
                        )
                    else:
                        records.append(
                            RgbTelemetryRecord(
                                point,
                                attempt,
                                control,
                                requested,
                                returned,
                                value,
                                RgbTelemetryRecordStatus.VALUE,
                            )
                        )
            finally:
                sample = RgbTelemetrySample(
                    point,
                    None if observation is None else observation.index,
                    None if observation is None else observation.buffer.index,
                    None if observation is None else observation.buffer.sequence,
                    None if observation is None else observation.buffer.timestamp_ns,
                    tuple(records),
                )
                item.telemetry_samples.append(sample)

        try:
            run = item.prepared.wait_series(
                max_frames=max_frames,
                max_total_bytes=max_total_bytes,
                on_observation=preserve,
                on_progress=progress,
                on_telemetry=read_controls if read_rgb_controls else None,
            )
        except BaseException as exc:
            if latest_run is None:
                status = (
                    CaptureStatus.INTERRUPTED
                    if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                    else CaptureStatus.IOCTL_ERROR
                )
                terminal = FrameVerdict(
                    status,
                    f"RGB series raised {type(exc).__name__}: {exc}",
                )
                latest_run = SeriesRun(
                    status,
                    terminal.detail,
                    max_total_bytes=max_total_bytes,
                    terminal_verdict=terminal,
                )
            run = latest_run
            item.prepared.state.frame = None
            item.primary = run.terminal_verdict or FrameVerdict(run.status, run.detail)
            item.series = make_result(run)
            item.artifact = item.series.artifact_record
            raise
        item.primary = item.prepared.primary
        item.series = make_result(run)
        item.artifact = item.series.artifact_record
        return item.series

    def _cleanup_result(self, item: _PairedHandle, result: StepResult) -> None:
        item.cleanup_steps.append(result)
        if not result.ok:
            item.cleanup_errors.append(result.detail)

    def _guard_cleanup(
        self, item: _PairedHandle, step: Step, action: Callable[[], StepResult]
    ) -> None:
        try:
            result = action()
        except BaseException as exc:
            result = StepResult(
                step=step,
                ok=False,
                detail=f"{step} raised {type(exc).__name__}: {exc}",
            )
        self._cleanup_result(item, result)

    def stream_off(self, handle: object) -> None:
        item = self._as_handle(handle)
        if item.prepared.session.streaming:
            self._guard_cleanup(item, Step.STREAMOFF, item.prepared.stream_off)

    def unmap(self, handle: object) -> None:
        item = self._as_handle(handle)
        for result in item.prepared.unmap_all():
            self._cleanup_result(item, result)

    def free_queue(self, handle: object) -> None:
        item = self._as_handle(handle)
        if item.prepared.session.buffers_allocated:
            self._guard_cleanup(item, Step.RELEASE_BUFFERS, item.prepared.free_queue)

    def close(self, handle: object) -> None:
        item = self._as_handle(handle)
        if item.closed:
            return
        try:
            self._guard_cleanup(item, Step.CLOSE, item.prepared.close)
        finally:
            item.closed = True

    def shutdown_depth(self, handle: object) -> IlluminationShutdownOutcome:
        """Run the one off command after both paired role descriptors close."""
        item = self._as_handle(handle)
        if item.role != DEPTH_ROLE or not item.prepared.session.illumination_required:
            return illumination_not_required("depth STREAMON was never attempted")
        backend = self.illumination_backend_factory(self.sysfs_root, self.dev_root)
        return backend.run(ILLUMINATION_AUTHORIZED_SERIAL)

    def role_outcome(self, handle: object) -> CaptureOutcome:
        item = self._as_handle(handle)
        primary = item.primary or FrameVerdict(
            CaptureStatus.IOCTL_ERROR, "role did not reach a result"
        )
        if item.artifact is None and not item.series_attempted:
            item.artifact = item.prepared.preserve(primary)
        elif item.artifact is None:
            item.artifact = ArtifactRecord.not_needed(
                "RGB series owns preservation under its indexed artifact namespace",
                str(item.prepared.output_dir / "rgb-series"),
                RGB_MAX_RAW_ARTIFACT_BYTES,
            )
        cleanup = CleanupRecord(
            attempted=True,
            steps=tuple(item.cleanup_steps),
            ok=bool(item.cleanup_steps) and not item.cleanup_errors,
            note=(
                "role cleanup completed"
                if item.cleanup_steps and not item.cleanup_errors
                else "role cleanup failed"
            ),
            guidance=(
                None
                if not item.cleanup_errors
                else f"inspect {item.target.node_path}; cleanup errors require recovery"
            ),
        )
        return CaptureOutcome(
            status=primary.status,
            detail=primary.detail,
            target=item.target,
            steps=tuple(item.steps),
            cleanup=cleanup,
            artifacts=item.artifact,
            capability=item.prepared.state.capability,
            initial_format=item.prepared.state.initial_format,
            negotiated_format=item.prepared.state.negotiated_format,
            buffers=item.prepared.state.buffers,
            queried_buffers=tuple(item.prepared.state.queried),
            mappings=tuple(item.prepared.state.mappings),
            poll=item.prepared.state.poll,
            dequeued=item.prepared.state.dequeued,
            frame_bytes=(
                None if item.prepared.state.frame is None else len(item.prepared.state.frame)
            ),
            stream_duration_s=(
                None
                if item.prepared.state.stream_started is None
                else item.prepared.engine.clock() - item.prepared.state.stream_started
            ),
            total_duration_s=item.prepared.engine.clock() - item.prepared.started,
            errno=primary.errno,
            errno_name=primary.errno_name,
            profile=item.prepared.engine.profile,
            frame_evidence=primary.evidence,
        )

    @staticmethod
    def _as_handle(handle: object) -> _PairedHandle:
        if not isinstance(handle, _PairedHandle):
            raise TypeError("paired V4L2 backend received an unknown handle")
        return handle


# Keep the conventional engine-first spelling available to callers that name
# the transport before the lifecycle, without creating a second implementation.
V4l2PairedBackend = PairedV4l2Backend


@dataclass(frozen=True, slots=True)
class PairedPlan:
    """The complete dry-run contract, including safety and cleanup boundaries."""

    depth: StreamSpec = DEPTH_SPEC
    rgb: StreamSpec = RGB_SPEC

    def as_dict(self) -> dict[str, object]:
        return {
            "roles": {"depth": self.depth.as_dict(), "rgb": self.rgb.as_dict()},
            "resolve": (
                "find the exact USB serial and role-labelled capture nodes; never guess videoN"
            ),
            "ordering": [
                "resolve depth and rgb; reject missing roles or one node for both",
                "open/configure/map/queue depth",
                "STREAMON depth",
                "sleep exactly one bounded one-second interval",
                "open/configure/map/queue rgb",
                "STREAMON rgb",
                "wait once for depth then once for rgb, preserving one depth and at most one RGB "
                "frame",
            ],
            "start_interval_s": START_INTERVAL_S,
            "deadlines_s": {"depth": self.depth.deadline_s, "rgb": self.rgb.deadline_s},
            "frame_caps": {"depth": MAX_DEPTH_FRAMES, "rgb": MAX_RGB_FRAMES},
            "artifacts": {"depth": "one raw Y16 frame", "rgb": "at most one MJPG bitstream"},
            "cleanup_order": [
                "rgb STREAMOFF, unmap, free queue, close",
                "depth STREAMOFF, unmap, free queue, close",
            ],
            "permissions": (
                "ordinary V4L2 access only; no usbfs, HID, XU, module, trace, reset, or SDK access"
            ),
            "execution": {
                "default": "dry-run; opens nothing",
                "required_serial": AUTHORIZED_SERIAL,
                "required_opt_in": EXECUTE_OPT_IN,
                "retry": False,
                "alternate_mode": False,
            },
            "source": {
                "repository": "Revopoint/3DViewer",
                "commit": "516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7",
                "ordering_location": "src/cscamera/cscamera.cpp:275-310",
                "depth_start": "src/cscamera/cscamera.cpp:376-399",
                "rgb_start": "src/cscamera/cscamera.cpp:401-417",
                "vendor_library_sha256": (
                    "49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4"
                ),
            },
        }

    def as_series_dict(self) -> dict[str, object]:
        """The same fixed pair plan with the bounded RGB-series wait."""
        plan = self.as_dict()
        ordering = list(cast(list[str], plan["ordering"]))
        ordering[-1] = (
            "wait once for depth then poll/dequeue/requeue at most "
            f"{RGB_SERIES_COUNT} RGB observations under one absolute deadline"
        )
        plan["ordering"] = ordering
        roles = cast(dict[str, dict[str, object]], plan["roles"])
        roles[RGB_ROLE] = dict(roles[RGB_ROLE])
        roles[RGB_ROLE]["max_frames"] = RGB_SERIES_COUNT
        plan["roles"] = roles
        deadlines = dict(cast(dict[str, float], plan["deadlines_s"]))
        deadlines["rgb"] = RGB_SERIES_DEADLINE_S
        plan["deadlines_s"] = deadlines
        frame_caps = dict(cast(dict[str, int], plan["frame_caps"]))
        frame_caps["rgb"] = RGB_SERIES_COUNT
        plan["frame_caps"] = frame_caps
        plan["artifacts"] = {
            "depth": "one raw Y16 frame",
            "rgb": f"{RGB_SERIES_COUNT} indexed raw MJPG payloads",
        }
        plan["series"] = {
            "count": RGB_SERIES_COUNT,
            "deadline_s": RGB_SERIES_DEADLINE_S,
            "max_total_bytes": RGB_SERIES_MAX_TOTAL_BYTES,
            "artifact_prefix": RGB_SERIES_ARTIFACT_PREFIX,
            "strict_jpeg": "trailing bytes remain invalid; no trimming",
            "rgb_control_telemetry": {
                "flag": "--read-rgb-controls-during-series",
                "enabled_by_default": False,
                "controls": [spec.as_dict() for spec in RGB_SERIES_CONTROL_WHITELIST],
                "points": RGB_SERIES_CONTROL_POINT_LIMIT,
                "max_attempts": RGB_SERIES_CONTROL_POINT_LIMIT * len(RGB_SERIES_CONTROL_WHITELIST),
                "same_fd": True,
                "read_only": True,
            },
        }
        execution = dict(cast(dict[str, object], plan["execution"]))
        execution["required_opt_in"] = RGB_SERIES_EXECUTE_OPT_IN
        execution["series_mode"] = True
        plan["execution"] = execution
        return plan


class _Phase(enum.StrEnum):
    DEPTH_OPEN = "depth_open"
    DEPTH_CONFIGURE = "depth_configure"
    DEPTH_MAP_QUEUE = "depth_map_queue"
    DEPTH_STREAMON = "depth_streamon"
    INTERVAL = "one_second_interval"
    RGB_OPEN = "rgb_open"
    RGB_CONFIGURE = "rgb_configure"
    RGB_MAP_QUEUE = "rgb_map_queue"
    RGB_STREAMON = "rgb_streamon"
    DEPTH_WAIT = "depth_wait"
    RGB_WAIT = "rgb_wait"


@dataclass(frozen=True, slots=True)
class PhaseResult:
    phase: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CleanupReport:
    actions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    deferred_signals: tuple[int, ...] = ()
    guidance: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    status: PairedStatus
    detail: str
    depth_node: str | None = None
    rgb_node: str | None = None
    depth_frame: bytes | None = None
    rgb_frame: bytes | None = None
    phases: tuple[PhaseResult, ...] = ()
    cleanup: CleanupReport = field(default_factory=CleanupReport)
    plan: PairedPlan = field(default_factory=PairedPlan)
    depth_result: CaptureOutcome | None = None
    rgb_result: CaptureOutcome | None = None
    rgb_series: RgbSeriesResult | None = None
    series_mode: bool = False
    rgb_control_telemetry_requested: bool = False
    illumination_shutdown: IlluminationShutdownOutcome = field(
        default_factory=illumination_not_required
    )

    @property
    def exit_code(self) -> int:
        if not self.cleanup.ok:
            return PAIRED_CLEANUP_FAILED_EXIT
        if not self.illumination_shutdown.succeeded:
            return self.illumination_shutdown.exit_code
        return PAIRED_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        return (
            self.status is PairedStatus.COMPLETE
            and self.cleanup.ok
            and self.depth_result is not None
            and self.rgb_result is not None
            and self.depth_result.succeeded
            and self.rgb_result.succeeded
            and self.illumination_shutdown.succeeded
        )

    def as_dict(self) -> dict[str, object]:
        depth_timestamp = (
            None
            if self.depth_result is None or self.depth_result.dequeued is None
            else self.depth_result.dequeued.timestamp_ns
        )
        rgb_timestamp = (
            None
            if self.rgb_result is None or self.rgb_result.dequeued is None
            else self.rgb_result.dequeued.timestamp_ns
        )
        depth_sequence = (
            None
            if self.depth_result is None or self.depth_result.dequeued is None
            else self.depth_result.dequeued.sequence
        )
        rgb_sequence = (
            None
            if self.rgb_result is None or self.rgb_result.dequeued is None
            else self.rgb_result.dequeued.sequence
        )
        return {
            "status": str(self.status),
            "exit_code": self.exit_code,
            "succeeded": self.succeeded,
            "detail": self.detail,
            "depth_node": self.depth_node,
            "rgb_node": self.rgb_node,
            "depth_frame_bytes": None if self.depth_frame is None else len(self.depth_frame),
            "rgb_frame_bytes": None if self.rgb_frame is None else len(self.rgb_frame),
            "roles": {
                "depth": None if self.depth_result is None else self.depth_result.as_dict(),
                "rgb": None if self.rgb_result is None else self.rgb_result.as_dict(),
            },
            "rgb_series": None if self.rgb_series is None else self.rgb_series.as_dict(),
            "rgb_control_telemetry": {
                "requested": self.rgb_control_telemetry_requested,
                "enabled": (self.rgb_series is not None and self.rgb_series.telemetry_enabled),
            },
            "delivery": {
                "depth": self.depth_result is not None and self.depth_frame is not None,
                "rgb": self.rgb_result is not None and self.rgb_frame is not None,
            },
            "timing": {
                "depth_timestamp_ns": depth_timestamp,
                "rgb_timestamp_ns": rgb_timestamp,
                "depth_sequence": depth_sequence,
                "rgb_sequence": rgb_sequence,
                "observed_skew_ns": (
                    None
                    if depth_timestamp is None or rgb_timestamp is None
                    else abs(rgb_timestamp - depth_timestamp)
                ),
                "synchronized": False,
                "synchronization_limit": (
                    "V4L2 timestamps are reported for evidence only; hardware synchronization "
                    "and pairing accuracy were not established"
                ),
            },
            "phases": [{"phase": p.phase, "ok": p.ok, "detail": p.detail} for p in self.phases],
            "cleanup": {
                "actions": list(self.cleanup.actions),
                "errors": list(self.cleanup.errors),
                "deferred_signals": list(self.cleanup.deferred_signals),
                "recovery_guidance": self.cleanup.guidance,
            },
            "illumination_shutdown": self.illumination_shutdown.as_dict(),
            "plan": self.plan.as_series_dict() if self.series_mode else self.plan.as_dict(),
        }


class PairedCapture:
    """One non-retrying depth-first/RGB-second attempt over an injected backend."""

    def __init__(
        self,
        backend: StreamBackend | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        plan: PairedPlan | None = None,
        termination_shield: TerminationShield = defer_termination,
        signal_guard: SignalGuard = PAIR_SIGNAL_GUARD,
    ) -> None:
        self.backend = backend
        self.sleep = sleep
        self.plan = PairedPlan() if plan is None else plan
        self.termination_shield = termination_shield
        self.signal_guard = signal_guard

    def dry_run(self, *, read_rgb_controls_during_series: bool = False) -> PairedOutcome:
        return PairedOutcome(
            PairedStatus.NOT_ATTEMPTED,
            "dry-run: no node was resolved or opened",
            plan=self.plan,
            rgb_control_telemetry_requested=read_rgb_controls_during_series,
        )

    def run(
        self,
        serial: str,
        *,
        opt_in: str,
        cancel: Callable[[], bool] | None = None,
        rgb_series: bool = False,
        read_rgb_controls_during_series: bool = False,
    ) -> PairedOutcome:
        if read_rgb_controls_during_series and not rgb_series:
            return PairedOutcome(
                PairedStatus.REFUSED,
                "RGB control telemetry requires paired-rgb-series mode",
                plan=self.plan,
                rgb_control_telemetry_requested=True,
            )
        if serial != AUTHORIZED_SERIAL:
            return PairedOutcome(
                PairedStatus.REFUSED,
                f"serial must be {AUTHORIZED_SERIAL}",
                plan=self.plan,
                series_mode=rgb_series,
                rgb_control_telemetry_requested=read_rgb_controls_during_series,
            )
        expected_opt_in = RGB_SERIES_EXECUTE_OPT_IN if rgb_series else EXECUTE_OPT_IN
        if opt_in != expected_opt_in:
            return PairedOutcome(
                PairedStatus.REFUSED,
                f"missing exact opt-in {expected_opt_in}",
                plan=self.plan,
                series_mode=rgb_series,
                rgb_control_telemetry_requested=read_rgb_controls_during_series,
            )
        if self.backend is None:
            return PairedOutcome(
                PairedStatus.REFUSED,
                "no live backend is installed; use the dry-run plan",
                plan=self.plan,
                series_mode=rgb_series,
                rgb_control_telemetry_requested=read_rgb_controls_during_series,
            )

        phases: list[PhaseResult] = []
        handles: dict[str, object] = {}
        nodes: dict[str, str] = {}
        frames: dict[str, bytes | None] = {DEPTH_ROLE: None, RGB_ROLE: None}
        cleanup_actions: list[str] = []
        cleanup_errors: list[str] = []
        role_results: dict[str, CaptureOutcome] = {}
        rgb_series_result: RgbSeriesResult | None = None
        illumination_shutdown = illumination_not_required()
        # These sets are obligations, not success flags.  They are raised
        # before the risky backend call so a signal/exception after the kernel
        # creates state cannot skip the corresponding cleanup.
        streamed_owed: set[str] = set()
        mapped_owed: set[str] = set()
        deferred_signals: list[int] = []
        cleanup_started = False
        status = PairedStatus.FAILED
        detail = "paired run did not reach a result"
        window = self.signal_guard()
        window_entered = False

        def restore_window() -> None:
            """Restore a guard even if its entry raised part-way through."""
            try:
                window.__exit__(None, None, None)
            except RuntimeError:
                # ``contextlib.contextmanager`` already runs its generator's
                # finally when __enter__ raises before yielding.  A custom
                # injected guard may instead need the explicit __exit__ call.
                if window_entered:
                    raise

        try:
            # Keep entry inside the same outer try/finally as cleanup.  This
            # matters for a signal or exception raised while a synthetic (or
            # future platform) guard is installing its handlers.
            window.__enter__()
            window_entered = True
            nodes[DEPTH_ROLE] = self.backend.resolve_node(serial, DEPTH_ROLE)
            nodes[RGB_ROLE] = self.backend.resolve_node(serial, RGB_ROLE)
            if nodes[DEPTH_ROLE] == nodes[RGB_ROLE]:
                return PairedOutcome(
                    PairedStatus.REFUSED,
                    "depth and RGB resolved to the same node",
                    phases=tuple(phases),
                    plan=self.plan,
                    series_mode=rgb_series,
                )
            for role, spec in ((DEPTH_ROLE, self.plan.depth), (RGB_ROLE, self.plan.rgb)):
                handle = self.backend.open(nodes[role], role)
                handles[role] = handle
                phases.append(PhaseResult(f"{role}_open", True, nodes[role]))
                self.backend.configure(handle, spec)
                phases.append(PhaseResult(f"{role}_configure", True, spec.fourcc))
                mapped_owed.add(role)
                self.backend.map_queue(handle, spec)
                phases.append(
                    PhaseResult(f"{role}_map_queue", True, f"{spec.requested_buffers} buffers")
                )
                if role == RGB_ROLE:
                    break
                streamed_owed.add(role)
                self.backend.stream_on(handle)
                phases.append(PhaseResult(f"{role}_streamon", True, "stream started"))
                self.sleep(START_INTERVAL_S)
                phases.append(PhaseResult(_Phase.INTERVAL, True, "one second preserved"))
            handle = handles[RGB_ROLE]
            streamed_owed.add(RGB_ROLE)
            self.backend.stream_on(handle)
            phases.append(PhaseResult(f"{RGB_ROLE}_streamon", True, "stream started"))
            if cancel is not None and cancel():
                raise _Cancelled("cancelled before waits")
            frames[DEPTH_ROLE] = self.backend.wait_frame(
                handles[DEPTH_ROLE], self.plan.depth.deadline_s, MAX_DEPTH_FRAMES
            )
            phases.append(PhaseResult(_Phase.DEPTH_WAIT, True, "one frame maximum"))
            if rgb_series:
                wait_series = getattr(self.backend, "wait_rgb_series", None)
                if not callable(wait_series):
                    raise PairedBackendRefusal("backend does not provide the RGB-series operation")
                series_args = (
                    handles[RGB_ROLE],
                    RGB_SERIES_DEADLINE_S,
                    RGB_SERIES_COUNT,
                    RGB_SERIES_MAX_TOTAL_BYTES,
                )
                if read_rgb_controls_during_series:
                    rgb_series_result = wait_series(*series_args, read_rgb_controls=True)
                else:
                    # Keep the legacy backend seam's call shape unchanged
                    # when the optional mode is disabled.
                    rgb_series_result = wait_series(*series_args)
                if rgb_series_result.observations:
                    frames[RGB_ROLE] = rgb_series_result.last_payload
                phases.append(
                    PhaseResult(
                        "rgb_series_wait",
                        True,
                        f"{rgb_series_result.delivered_count} of {RGB_SERIES_COUNT} observations",
                    )
                )
            else:
                frames[RGB_ROLE] = self.backend.wait_frame(
                    handles[RGB_ROLE], self.plan.rgb.deadline_s, MAX_RGB_FRAMES
                )
                phases.append(PhaseResult(_Phase.RGB_WAIT, True, "one frame maximum"))
            if frames[DEPTH_ROLE] is None:
                status, detail = PairedStatus.TIMEOUT, "depth deadline elapsed without a frame"
            elif frames[RGB_ROLE] is None:
                status, detail = PairedStatus.TIMEOUT, "RGB deadline elapsed without a frame"
            else:
                status, detail = PairedStatus.COMPLETE, "paired run completed"
        except SignalReceived as exc:
            status, detail = PairedStatus.CANCELLED, str(exc)
        except (DeviceAbsentError, V4l2CaptureNotAllowed) as exc:
            status, detail = PairedStatus.REFUSED, str(exc)
        except PairedBackendRefusal as exc:
            status, detail = PairedStatus.REFUSED, str(exc)
        except _Cancelled as exc:
            status, detail = PairedStatus.CANCELLED, str(exc)
        except KeyboardInterrupt:
            status, detail = PairedStatus.CANCELLED, "interrupted; cleanup completed"
        except TimeoutError as exc:
            status, detail = PairedStatus.TIMEOUT, str(exc) or "stream deadline elapsed"
        except BaseException as exc:
            status, detail = PairedStatus.FAILED, f"{type(exc).__name__}: {exc}"
        finally:
            if status is PairedStatus.CANCELLED:
                for handle in handles.values():
                    if isinstance(handle, _PairedHandle) and handle.primary is None:
                        handle.primary = FrameVerdict(CaptureStatus.INTERRUPTED, detail)

            backend = self.backend
            assert backend is not None

            def cleanup_body() -> None:
                nonlocal cleanup_started, illumination_shutdown, deferred_signals
                cleanup_started = True
                for role in (RGB_ROLE, DEPTH_ROLE):
                    handle = handles.get(role)
                    if handle is None:
                        continue
                    for action, method in (
                        (
                            "streamoff",
                            backend.stream_off if role in streamed_owed else None,
                        ),
                        ("unmap", backend.unmap if role in mapped_owed else None),
                        (
                            "free_queue",
                            backend.free_queue if role in mapped_owed else None,
                        ),
                        ("close", backend.close),
                    ):
                        if method is None:
                            continue
                        before_cleanup_errors = len(getattr(handle, "cleanup_errors", ()))
                        try:
                            method(handle)
                            cleanup_actions.append(f"{role} {action}")
                        except BaseException as exc:
                            cleanup_errors.append(f"{role} {action}: {type(exc).__name__}: {exc}")
                        else:
                            backend_errors = getattr(handle, "cleanup_errors", ())
                            cleanup_errors.extend(backend_errors[before_cleanup_errors:])
                    role_result = getattr(backend, "role_outcome", None)
                    if role_result is not None:
                        try:
                            role_results[role] = role_result(handle)
                        except BaseException as exc:
                            cleanup_errors.append(f"{role} result: {type(exc).__name__}: {exc}")
                if DEPTH_ROLE in streamed_owed:
                    shutdown = getattr(backend, "shutdown_depth", None)
                    depth_handle = handles.get(DEPTH_ROLE)
                    if not callable(shutdown):
                        illumination_shutdown = IlluminationShutdownOutcome(
                            True,
                            False,
                            ShutdownStatus.REFUSED,
                            (
                                "depth STREAMON occurred but backend exposes no "
                                "shutdown_depth finalizer"
                            ),
                        )
                        cleanup_actions.append("illumination shutdown unavailable")
                    else:
                        try:
                            illumination_shutdown = shutdown(depth_handle)
                            if not illumination_shutdown.required:
                                illumination_shutdown = IlluminationShutdownOutcome(
                                    True,
                                    False,
                                    ShutdownStatus.REFUSED,
                                    "depth STREAMON occurred but finalizer reported not_required",
                                )
                            cleanup_actions.append("illumination shutdown")
                        except SignalReceived as exc:
                            deferred_signals.append(exc.signum)
                            illumination_shutdown = IlluminationShutdownOutcome(
                                True,
                                True,
                                ShutdownStatus.INTERRUPTED,
                                str(exc),
                                deferred_signals=(exc.signum,),
                            )
                            cleanup_actions.append("illumination shutdown (signal)")
                        except BaseException as exc:
                            illumination_shutdown = IlluminationShutdownOutcome(
                                True,
                                True,
                                ShutdownStatus.UNEXPECTED_EXCEPTION,
                                f"shutdown finalizer raised {type(exc).__name__}: {exc}",
                            )
                            cleanup_actions.append("illumination shutdown (raised)")

            def cleanup_roles() -> list[int]:
                with self.termination_shield() as deferred:
                    cleanup_body()
                    return list(deferred)

            try:
                deferred_signals.extend(cleanup_roles())
            except SignalReceived as exc:
                deferred_signals.append(exc.signum)
                if cleanup_started:
                    raise
                try:
                    deferred_signals.extend(cleanup_roles())
                except SignalReceived as retry_exc:
                    deferred_signals.append(retry_exc.signum)
                    if cleanup_started:
                        raise
                    cleanup_body()
                except BaseException:
                    if cleanup_started:
                        raise
                    cleanup_body()
            except BaseException:
                if cleanup_started:
                    raise
                try:
                    deferred_signals.extend(cleanup_roles())
                except SignalReceived as retry_exc:
                    deferred_signals.append(retry_exc.signum)
                    if cleanup_started:
                        raise
                    cleanup_body()
                except BaseException:
                    if cleanup_started:
                        raise
                    # The shield itself can be interrupted before it yields;
                    # retain the plain finally guarantee as the final backstop.
                    cleanup_body()
            finally:
                restore_window()
        cleanup = CleanupReport(
            tuple(cleanup_actions),
            tuple(cleanup_errors),
            tuple(deferred_signals),
            (
                "inspect both role nodes and recover any unreleased V4L2 queue or mapping"
                if cleanup_errors or deferred_signals
                else None
            ),
        )
        depth_result = role_results.get(DEPTH_ROLE)
        rgb_result = role_results.get(RGB_ROLE)
        if rgb_series_result is None and rgb_series:
            rgb_handle = handles.get(RGB_ROLE)
            if isinstance(rgb_handle, _PairedHandle):
                rgb_series_result = rgb_handle.series
        if rgb_series_result is not None and rgb_series_result.last_payload is not None:
            # Series delivery is byte delivery, independent of strict JPEG
            # validity, artifact success, or the final paired status. Keep the
            # payload in the top-level evidence without invoking legacy
            # single-frame preservation.
            frames[RGB_ROLE] = rgb_series_result.last_payload
        if (
            depth_result is not None or rgb_result is not None
        ) and status is not PairedStatus.CANCELLED:
            role_statuses = [
                result.status for result in (depth_result, rgb_result) if result is not None
            ]
            if any(
                result.status is CaptureStatus.POLL_TIMEOUT
                for result in (depth_result, rgb_result)
                if result is not None
            ):
                status = PairedStatus.TIMEOUT
                detail = "one or more role deadlines elapsed without a complete frame"
            elif any(
                not result.artifacts.ok
                for result in (depth_result, rgb_result)
                if result is not None and result.artifacts.attempted
            ):
                status = PairedStatus.FAILED
                detail = "a delivered role frame could not be preserved as bounded evidence"
            elif any(
                result.status
                in (
                    CaptureStatus.REFUSED,
                    CaptureStatus.DEVICE_ABSENT,
                    CaptureStatus.DEVICE_REPLACED,
                )
                for result in (depth_result, rgb_result)
                if result is not None
            ):
                status = PairedStatus.REFUSED
                detail = "a role target was refused after its descriptor was opened"
            elif any(
                result.status not in (CaptureStatus.COMPLETE, CaptureStatus.POLL_TIMEOUT)
                for result in (depth_result, rgb_result)
                if result is not None
            ):
                status = PairedStatus.FAILED
                detail = "one or more role setup, poll, dequeue, or profile steps failed"
            elif role_statuses and all(
                result is CaptureStatus.COMPLETE for result in role_statuses
            ):
                status = PairedStatus.COMPLETE
                detail = "paired run completed with one complete frame per role"
            elif status is PairedStatus.COMPLETE:
                status = PairedStatus.FAILED
                detail = "one or more delivered role payloads failed profile validation"
        if rgb_series_result is not None and status is not PairedStatus.CANCELLED:
            if rgb_series_result.status is CaptureStatus.POLL_TIMEOUT:
                status = PairedStatus.TIMEOUT
                detail = rgb_series_result.detail
            elif not rgb_series_result.strict_success:
                status = PairedStatus.FAILED
                detail = rgb_series_result.detail
        if not cleanup.ok:
            status = PairedStatus.FAILED
            detail = f"{detail}; cleanup failed"
        return PairedOutcome(
            status,
            detail,
            nodes.get(DEPTH_ROLE),
            nodes.get(RGB_ROLE),
            frames[DEPTH_ROLE],
            frames[RGB_ROLE],
            tuple(phases),
            cleanup,
            self.plan,
            depth_result,
            rgb_result,
            rgb_series_result,
            rgb_series,
            read_rgb_controls_during_series,
            illumination_shutdown,
        )


class RgbSeriesCapture(PairedCapture):
    """Dry-by-default depth-first pair with the fixed RGB observation series."""

    def dry_run(self, *, read_rgb_controls_during_series: bool = False) -> PairedOutcome:
        return PairedOutcome(
            PairedStatus.NOT_ATTEMPTED,
            "dry-run: no node was resolved or opened",
            plan=self.plan,
            series_mode=True,
            rgb_control_telemetry_requested=read_rgb_controls_during_series,
        )

    def run(
        self,
        serial: str,
        *,
        opt_in: str,
        cancel: Callable[[], bool] | None = None,
        rgb_series: bool = True,
        read_rgb_controls_during_series: bool = False,
    ) -> PairedOutcome:
        return super().run(
            serial,
            opt_in=opt_in,
            cancel=cancel,
            rgb_series=True,
            read_rgb_controls_during_series=read_rgb_controls_during_series,
        )


class _Cancelled(Exception):
    pass


def dry_run_plan() -> dict[str, object]:
    return PairedPlan().as_dict()
