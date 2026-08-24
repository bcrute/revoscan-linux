"""Versioned scan-session manifest and atomic artifact store.

This is the durable boundary that capture, processing, the CLI, the future
local service, and the mapper handoff all read and write. It lands before
continuous capture so evidence stops living only in per-command JSON blobs
under ``/tmp``.

Three rules shape everything here:

*Success is recorded, never inferred.* A stage is complete only because
something wrote :attr:`StageState.COMPLETE`. A directory full of plausible
artifacts means nothing on its own, and a session interrupted mid-write stays
readable and honestly labelled :attr:`SessionState.INCOMPLETE` rather than
being silently promoted or silently discarded.

*Raw evidence is immutable.* Anything under ``raw/`` is created exclusively and
never overwritten, so a second write under a name that already exists is an
error rather than a quiet replacement of the only copy of an observation.
Derived artifacts live under ``derived/`` and follow the same rule; replacing a
derived result means writing a new name, which keeps the older one inspectable.

*A partial write leaves no artifact.* A truncated file under a name that
promises a whole frame is worse evidence than no file at all, so a failed write
is discarded and reported through the manifest.
"""

from __future__ import annotations

import contextlib
import errno as errno_mod
import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

from scanner3d.errors import ScannerError

#: Manifest schema this module writes. A reader refuses anything newer, because
#: a future version may add a field whose absence changes what the record means.
SCHEMA_VERSION: Final = 1

#: Oldest schema this module can still read.
MIN_READABLE_SCHEMA_VERSION: Final = 1

#: Refuse to load a manifest larger than this. A session manifest is metadata;
#: anything at this size is a malformed or hostile file, not a record.
MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024

#: Ceiling for a single artifact written through the store.
MAX_ARTIFACT_BYTES: Final = 512 * 1024 * 1024

#: Top-level buckets a session directory may contain. ``raw`` holds immutable
#: observations, ``derived`` holds everything computed from them, ``logs`` holds
#: text records.
RAW_BUCKET: Final = "raw"
DERIVED_BUCKET: Final = "derived"
LOGS_BUCKET: Final = "logs"
ARTIFACT_BUCKETS: Final = (RAW_BUCKET, DERIVED_BUCKET, LOGS_BUCKET)

MANIFEST_NAME: Final = "manifest.json"

#: A session id is used as a directory name and appears in artifact paths, so it
#: is restricted to characters that cannot escape a path or confuse a shell.
_SESSION_ID_RE: Final = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

#: One path segment of an artifact's relative path.
_SEGMENT_RE: Final = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

#: Manifest keys whose values are redacted out of a support bundle.
_SENSITIVE_KEYS: Final = frozenset({"serial", "wifi_psk", "psk", "ssid", "user_data"})

_REDACTED: Final = "[redacted]"


class SessionError(ScannerError):
    """Base class for every session-store failure."""


class SessionSchemaUnsupported(SessionError):
    """A manifest declares a schema version this build cannot read.

    Raised for both directions. A version below
    :data:`MIN_READABLE_SCHEMA_VERSION` is too old to interpret, and one above
    :data:`SCHEMA_VERSION` may rely on fields this build would silently drop,
    which would turn an upgrade into evidence loss.
    """


class SessionManifestInvalid(SessionError):
    """A manifest is unreadable, malformed, truncated, or internally
    inconsistent.

    Deliberately distinct from :class:`SessionSchemaUnsupported`: a version this
    build does not know is a compatibility question, while a manifest whose
    stage list is not a list is a damaged record.
    """


class SessionPathNotAllowed(SessionError):
    """An artifact path escaped the session directory or its allowed buckets.

    Raised before any file is created, so a traversing, absolute, symlinked or
    otherwise unsafe path can never be opened, let alone written.
    """


class ArtifactWriteIncomplete(OSError):
    """An artifact could not be written in full, so none was kept.

    An ``OSError`` for the same reason :class:`~scanner3d.errors.
    V4l2ArtifactIncomplete` is one: it is raised on the same path as a real
    write failure and carries an ``errno``, so a short write, a zero-byte write
    and a kernel error stay describable in one field.
    """


class SessionState(StrEnum):
    """Lifecycle of a whole session.

    ``INCOMPLETE`` is the state a session is *found* in when a previous process
    died between writes. It is never a state a caller sets deliberately, and it
    is not a synonym for failure: the evidence collected so far is still valid,
    it simply does not amount to a finished session.
    """

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class StageState(StrEnum):
    """Lifecycle of one stage within a session."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


#: States after which no further work happens on a session or stage.
TERMINAL_SESSION_STATES: Final = frozenset(
    {SessionState.COMPLETE, SessionState.FAILED, SessionState.CANCELLED}
)
TERMINAL_STAGE_STATES: Final = frozenset(
    {StageState.COMPLETE, StageState.FAILED, StageState.CANCELLED}
)


class Clock(Protocol):
    """Wall and monotonic time, injectable so fixtures round-trip byte-exactly."""

    def wall(self) -> str:
        """An ISO-8601 UTC timestamp."""

    def monotonic(self) -> float:
        """A monotonic reading, for durations that survive a clock adjustment."""


class SystemClock:
    """The real clock."""

    __slots__ = ()

    def wall(self) -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds")

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True, slots=True)
class FixedClock:
    """A clock that does not move, so a fixture serialises identically twice."""

    stamp: str = "2026-01-01T00:00:00.000000+00:00"
    tick: float = 0.0

    def wall(self) -> str:
        return self.stamp

    def monotonic(self) -> float:
        return self.tick


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Named frame plus the conventions needed to interpret coordinates in it.

    Every field is required because each one has silently differed between two
    plausible readings of this scanner's calibration record at some point. A
    geometry artifact without units, axis order and handedness is not
    dimensionally explicit, whatever its file format claims.
    """

    name: str
    units: str
    axes: str
    handedness: str
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "units": self.units,
            "axes": self.axes,
            "handedness": self.handedness,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CoordinateFrame:
        return cls(
            name=_require_str(raw, "name"),
            units=_require_str(raw, "units"),
            axes=_require_str(raw, "axes"),
            handedness=_require_str(raw, "handedness"),
            note=_optional_str(raw, "note", ""),
        )


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Which physical unit produced a session, and how it was attached.

    ``topology`` is a free-form snapshot rather than a fixed shape because the
    interesting details have changed between investigations - bound drivers one
    week, hub port numbering the next - and losing an unmodelled detail is worse
    than carrying a dict.
    """

    vendor_id: int
    product_id: int
    serial: str
    usb_path: str = ""
    bus_number: int | None = None
    device_number: int | None = None
    firmware_version: str = ""
    topology: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "usb_identity": f"{self.vendor_id:04x}:{self.product_id:04x}",
            "serial": self.serial,
            "usb_path": self.usb_path,
            "bus_number": self.bus_number,
            "device_number": self.device_number,
            "firmware_version": self.firmware_version,
            "topology": dict(self.topology),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DeviceIdentity:
        return cls(
            vendor_id=_require_int(raw, "vendor_id"),
            product_id=_require_int(raw, "product_id"),
            serial=_require_str(raw, "serial"),
            usb_path=_optional_str(raw, "usb_path", ""),
            bus_number=_optional_int(raw, "bus_number"),
            device_number=_optional_int(raw, "device_number"),
            firmware_version=_optional_str(raw, "firmware_version", ""),
            topology=_optional_mapping(raw, "topology"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One file the store wrote, with the checksum that identifies it.

    ``path`` is always relative to the session directory and always POSIX-style,
    so a manifest stays meaningful when the session is copied to another host.
    """

    path: str
    bytes: int
    sha256: str
    kind: str
    immutable: bool
    created_wall: str
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "kind": self.kind,
            "immutable": self.immutable,
            "created_wall": self.created_wall,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ArtifactRecord:
        return cls(
            path=_require_str(raw, "path"),
            bytes=_require_int(raw, "bytes"),
            sha256=_require_str(raw, "sha256"),
            kind=_require_str(raw, "kind"),
            immutable=_require_bool(raw, "immutable"),
            created_wall=_require_str(raw, "created_wall"),
            note=_optional_str(raw, "note", ""),
        )


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One stage of a session and exactly how it ended.

    ``counters`` carries the per-stage evidence the roadmap requires of an
    acquisition stage - delivered frames, drops, late and short frames, missing
    sequence numbers - without fixing a schema that processing stages would then
    have to pretend to fill in.
    """

    name: str
    state: StageState
    started_wall: str = ""
    ended_wall: str = ""
    duration_s: float | None = None
    error: str = ""
    counters: Mapping[str, int] = field(default_factory=dict)
    parameters: Mapping[str, object] = field(default_factory=dict)
    artifacts: Sequence[ArtifactRecord] = ()
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": str(self.state),
            "started_wall": self.started_wall,
            "ended_wall": self.ended_wall,
            "duration_s": self.duration_s,
            "error": self.error,
            "counters": dict(self.counters),
            "parameters": dict(self.parameters),
            "artifacts": [a.as_dict() for a in self.artifacts],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> StageRecord:
        artifacts_raw = raw.get("artifacts", [])
        if not isinstance(artifacts_raw, list):
            raise SessionManifestInvalid("stage 'artifacts' must be a list")
        return cls(
            name=_require_str(raw, "name"),
            state=_require_enum(raw, "state", StageState),
            started_wall=_optional_str(raw, "started_wall", ""),
            ended_wall=_optional_str(raw, "ended_wall", ""),
            duration_s=_optional_float(raw, "duration_s"),
            error=_optional_str(raw, "error", ""),
            counters=_optional_int_mapping(raw, "counters"),
            parameters=_optional_mapping(raw, "parameters"),
            artifacts=tuple(
                ArtifactRecord.from_dict(_as_mapping(a, "artifact")) for a in artifacts_raw
            ),
            note=_optional_str(raw, "note", ""),
        )


@dataclass(frozen=True, slots=True)
class SessionManifest:
    """The durable truth about one scan session.

    Everything a later reader needs to interpret the artifacts is here:
    which unit produced them, what software version wrote them, what frames and
    conventions the geometry is expressed in, which calibration inputs were in
    force, and what actually happened at each stage.
    """

    schema_version: int
    session_id: str
    state: SessionState
    created_wall: str
    updated_wall: str
    software_version: str
    device: DeviceIdentity | None = None
    frames: Sequence[CoordinateFrame] = ()
    calibration: Mapping[str, object] = field(default_factory=dict)
    capture_profile: Mapping[str, object] = field(default_factory=dict)
    stages: Sequence[StageRecord] = ()
    limitations: Sequence[str] = ()
    error: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "state": str(self.state),
            "created_wall": self.created_wall,
            "updated_wall": self.updated_wall,
            "software_version": self.software_version,
            "device": self.device.as_dict() if self.device is not None else None,
            "frames": [f.as_dict() for f in self.frames],
            "calibration": dict(self.calibration),
            "capture_profile": dict(self.capture_profile),
            "stages": [s.as_dict() for s in self.stages],
            "limitations": list(self.limitations),
            "error": self.error,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SessionManifest:
        version = _require_int(raw, "schema_version")
        if version < MIN_READABLE_SCHEMA_VERSION or version > SCHEMA_VERSION:
            raise SessionSchemaUnsupported(
                f"manifest schema version {version} is outside the readable range "
                f"{MIN_READABLE_SCHEMA_VERSION}..{SCHEMA_VERSION}"
            )
        stages_raw = raw.get("stages", [])
        if not isinstance(stages_raw, list):
            raise SessionManifestInvalid("'stages' must be a list")
        frames_raw = raw.get("frames", [])
        if not isinstance(frames_raw, list):
            raise SessionManifestInvalid("'frames' must be a list")
        limitations_raw = raw.get("limitations", [])
        if not isinstance(limitations_raw, list) or not all(
            isinstance(item, str) for item in limitations_raw
        ):
            raise SessionManifestInvalid("'limitations' must be a list of strings")
        device_raw = raw.get("device")
        if device_raw is not None and not isinstance(device_raw, dict):
            raise SessionManifestInvalid("'device' must be an object or null")
        session_id = _require_str(raw, "session_id")
        if not _SESSION_ID_RE.match(session_id):
            raise SessionManifestInvalid(f"session_id {session_id!r} is not a safe identifier")
        return cls(
            schema_version=version,
            session_id=session_id,
            state=_require_enum(raw, "state", SessionState),
            created_wall=_require_str(raw, "created_wall"),
            updated_wall=_require_str(raw, "updated_wall"),
            software_version=_optional_str(raw, "software_version", ""),
            device=DeviceIdentity.from_dict(device_raw) if device_raw is not None else None,
            frames=tuple(CoordinateFrame.from_dict(_as_mapping(f, "frame")) for f in frames_raw),
            calibration=_optional_mapping(raw, "calibration"),
            capture_profile=_optional_mapping(raw, "capture_profile"),
            stages=tuple(StageRecord.from_dict(_as_mapping(s, "stage")) for s in stages_raw),
            limitations=tuple(limitations_raw),
            error=_optional_str(raw, "error", ""),
            note=_optional_str(raw, "note", ""),
        )

    def stage(self, name: str) -> StageRecord | None:
        """The named stage, or ``None``. Names are unique within a session."""
        for item in self.stages:
            if item.name == name:
                return item
        return None

    @property
    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        """Every artifact across every stage, in stage order."""
        return tuple(a for s in self.stages for a in s.artifacts)

    @property
    def is_complete(self) -> bool:
        """True only when the session says so *and* no stage contradicts it.

        A session cannot be complete while a stage is still running or was left
        incomplete: that combination means a writer crashed between updating a
        stage and updating the session, and reporting it as finished would be
        exactly the success-only status this project refuses to produce.
        """
        if self.state is not SessionState.COMPLETE:
            return False
        return all(s.state in TERMINAL_STAGE_STATES for s in self.stages)

    def redacted(self) -> dict[str, object]:
        """The manifest with identifying values removed, for a support bundle.

        The serial is the identifier that ties evidence to one physical unit, so
        it is replaced rather than dropped: a reader still sees that a serial was
        recorded. Structure, states, counters and checksums are untouched,
        because those are the parts a support report exists to show.
        """
        return _redact_mapping(self.as_dict())


def _redact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (_REDACTED if key in _SENSITIVE_KEYS and item else _redact_value(item))
        for key, item in value.items()
    }


def _redact_value(value: object) -> object:
    if isinstance(value, dict):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


class SessionStore:
    """Owns one session directory: safe paths, atomic writes, durable manifest.

    The store is the only thing that writes inside a session directory. Callers
    hand it bytes and a relative path; it decides whether that path is
    admissible, writes the file completely or not at all, and records the result
    in the manifest. There is deliberately no method that returns an open file
    handle, because a caller holding one could produce a half-written artifact
    the manifest would then describe as whole.
    """

    __slots__ = ("_clock", "_manifest", "_root")

    def __init__(
        self, root: Path, manifest: SessionManifest, *, clock: Clock | None = None
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._clock: Clock = clock if clock is not None else SystemClock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> SessionManifest:
        """The in-memory manifest. Always equal to what was last flushed."""
        return self._manifest

    @classmethod
    def create(
        cls,
        parent: Path,
        session_id: str,
        *,
        software_version: str,
        device: DeviceIdentity | None = None,
        frames: Iterable[CoordinateFrame] = (),
        calibration: Mapping[str, object] | None = None,
        capture_profile: Mapping[str, object] | None = None,
        clock: Clock | None = None,
    ) -> SessionStore:
        """Create a new session directory and write its planned manifest.

        The directory is created exclusively. An existing session id is an
        error, never a resumption: silently adopting a directory whose contents
        were written by an unknown earlier run would attach this run's manifest
        to someone else's evidence.
        """
        if not _SESSION_ID_RE.match(session_id):
            raise SessionPathNotAllowed(f"session id {session_id!r} is not a safe identifier")
        used_clock: Clock = clock if clock is not None else SystemClock()
        root = parent / session_id
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        for bucket in ARTIFACT_BUCKETS:
            (root / bucket).mkdir(mode=0o700, exist_ok=False)
        now = used_clock.wall()
        manifest = SessionManifest(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            state=SessionState.PLANNED,
            created_wall=now,
            updated_wall=now,
            software_version=software_version,
            device=device,
            frames=tuple(frames),
            calibration=dict(calibration or {}),
            capture_profile=dict(capture_profile or {}),
        )
        store = cls(root, manifest, clock=used_clock)
        store._flush()
        return store

    @classmethod
    def open(cls, root: Path, *, clock: Clock | None = None) -> SessionStore:
        """Open an existing session, marking an interrupted one honestly.

        A session found in :attr:`SessionState.RUNNING` was not shut down by the
        process that owned it. It is reported as :attr:`SessionState.INCOMPLETE`
        so it stays fully inspectable without ever being mistaken for finished.
        The change is written back, so the honest label survives the next
        reader too.
        """
        manifest = read_manifest(root / MANIFEST_NAME)
        store = cls(root, manifest, clock=clock)
        if manifest.state is SessionState.RUNNING:
            store._manifest = replace(
                manifest,
                state=SessionState.INCOMPLETE,
                stages=tuple(
                    s
                    if s.state is not StageState.RUNNING
                    else replace(s, state=StageState.INCOMPLETE)
                    for s in manifest.stages
                ),
                updated_wall=store._clock.wall(),
                note=_join_note(manifest.note, "reopened after an interrupted run"),
            )
            store._flush()
        return store

    def start(self) -> None:
        """Move a planned session to running."""
        self._require_not_terminal()
        self._replace_manifest(state=SessionState.RUNNING)

    def begin_stage(self, name: str, *, parameters: Mapping[str, object] | None = None) -> None:
        """Add or restart a stage in the running state."""
        self._require_not_terminal()
        if not _SEGMENT_RE.match(name):
            raise SessionPathNotAllowed(f"stage name {name!r} is not a safe identifier")
        existing = self._manifest.stage(name)
        record = StageRecord(
            name=name,
            state=StageState.RUNNING,
            started_wall=self._clock.wall(),
            parameters=dict(parameters or {}),
            artifacts=existing.artifacts if existing is not None else (),
        )
        self._put_stage(record)

    def finish_stage(
        self,
        name: str,
        state: StageState,
        *,
        counters: Mapping[str, int] | None = None,
        error: str = "",
        note: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Close a stage in an explicit terminal state.

        ``state`` must be terminal. There is no ``finish`` that defaults to
        success, because the whole point of this layer is that nothing becomes
        complete by omission.
        """
        if state not in TERMINAL_STAGE_STATES:
            raise SessionError(f"{state} is not a terminal stage state")
        current = self._manifest.stage(name)
        if current is None:
            raise SessionError(f"stage {name!r} was never begun")
        self._put_stage(
            replace(
                current,
                state=state,
                ended_wall=self._clock.wall(),
                duration_s=duration_s,
                counters=dict(counters or current.counters),
                error=error,
                note=_join_note(current.note, note),
            )
        )

    def finish(self, state: SessionState, *, error: str = "", note: str = "") -> None:
        """Close the session in an explicit terminal state."""
        if state not in TERMINAL_SESSION_STATES:
            raise SessionError(f"{state} is not a terminal session state")
        self._replace_manifest(state=state, error=error, note=_join_note(self._manifest.note, note))

    def add_limitation(self, text: str) -> None:
        """Record something this session cannot support a claim about."""
        self._replace_manifest(limitations=(*self._manifest.limitations, text))

    def write_artifact(
        self,
        stage: str,
        relative_path: str,
        payload: bytes,
        *,
        kind: str,
        note: str = "",
        ceiling: int = MAX_ARTIFACT_BYTES,
    ) -> ArtifactRecord:
        """Write ``payload`` completely, or leave no file and raise.

        ``relative_path`` must begin with one of :data:`ARTIFACT_BUCKETS` and is
        validated segment by segment before anything is opened.
        """
        target = self._resolve_artifact_path(relative_path)
        if len(payload) > ceiling:
            raise ArtifactWriteIncomplete(
                errno_mod.EFBIG,
                f"refusing to write {len(payload)} bytes to {relative_path}, "
                f"past the {ceiling} B artifact ceiling",
            )
        record = self._manifest.stage(stage)
        if record is None:
            raise SessionError(f"stage {stage!r} was never begun")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_exclusive(target, payload)
        artifact = ArtifactRecord(
            path=relative_path,
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            kind=kind,
            immutable=relative_path.startswith(f"{RAW_BUCKET}/"),
            created_wall=self._clock.wall(),
            note=note,
        )
        self._put_stage(replace(record, artifacts=(*record.artifacts, artifact)))
        return artifact

    def verify_artifacts(self) -> dict[str, str]:
        """Re-hash every recorded artifact and report anything that disagrees.

        The returned mapping is empty when the session is intact. A present key
        names an artifact path and describes how it failed: missing, a different
        size, or a different checksum.
        """
        problems: dict[str, str] = {}
        for artifact in self._manifest.artifacts:
            path = self._root / PurePosixPath(artifact.path)
            if not path.is_file():
                problems[artifact.path] = "missing"
                continue
            payload = path.read_bytes()
            if len(payload) != artifact.bytes:
                problems[artifact.path] = f"expected {artifact.bytes} bytes, found {len(payload)}"
            elif hashlib.sha256(payload).hexdigest() != artifact.sha256:
                problems[artifact.path] = "checksum mismatch"
        return problems

    def _resolve_artifact_path(self, relative_path: str) -> Path:
        """Validate a relative artifact path, or refuse it before any open."""
        if not relative_path or relative_path != relative_path.strip():
            raise SessionPathNotAllowed(f"artifact path {relative_path!r} is not usable")
        if relative_path.startswith("/") or "\\" in relative_path:
            raise SessionPathNotAllowed(f"artifact path {relative_path!r} must be relative POSIX")
        parts = PurePosixPath(relative_path).parts
        # The manifest records this string verbatim, so a non-canonical spelling
        # would describe an artifact at a path that is not literally where it
        # sits. Insist the caller hand over the canonical form.
        if str(PurePosixPath(relative_path)) != relative_path:
            raise SessionPathNotAllowed(f"artifact path {relative_path!r} is not in canonical form")
        if len(parts) < 2:
            raise SessionPathNotAllowed(
                f"artifact path {relative_path!r} needs a bucket and a name"
            )
        if parts[0] not in ARTIFACT_BUCKETS:
            raise SessionPathNotAllowed(
                f"artifact path {relative_path!r} must start with one of {ARTIFACT_BUCKETS}"
            )
        for segment in parts:
            if not _SEGMENT_RE.match(segment):
                raise SessionPathNotAllowed(f"artifact path segment {segment!r} is not a safe name")
        target = self._root / PurePosixPath(relative_path)
        # Guard against a symlinked parent redirecting the write outside the
        # session even though every segment looked ordinary.
        anchor = self._root.resolve(strict=True)
        existing = target.parent
        while not existing.exists():
            existing = existing.parent
        if not existing.resolve(strict=True).is_relative_to(anchor):
            raise SessionPathNotAllowed(
                f"artifact path {relative_path!r} resolves outside the session directory"
            )
        return target

    def _put_stage(self, record: StageRecord) -> None:
        stages = list(self._manifest.stages)
        for index, item in enumerate(stages):
            if item.name == record.name:
                stages[index] = record
                break
        else:
            stages.append(record)
        self._replace_manifest(stages=tuple(stages))

    def _replace_manifest(self, **changes: object) -> None:
        self._manifest = replace(
            self._manifest,
            updated_wall=self._clock.wall(),
            **changes,  # type: ignore[arg-type]
        )
        self._flush()

    def _require_not_terminal(self) -> None:
        if self._manifest.state in TERMINAL_SESSION_STATES:
            raise SessionError(f"session is already {self._manifest.state}")

    def _flush(self) -> None:
        """Replace the manifest atomically, so a reader never sees half of it."""
        rendered = json.dumps(
            self._manifest.as_dict(), sort_keys=True, indent=2, ensure_ascii=False
        )
        payload = (rendered + "\n").encode("utf-8")
        _write_atomic_replace(self._root / MANIFEST_NAME, payload)


def read_manifest(path: Path) -> SessionManifest:
    """Load and validate a manifest, refusing anything this build cannot trust."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SessionManifestInvalid(f"cannot stat manifest {path}: {exc}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise SessionManifestInvalid(
            f"manifest {path} is {size} bytes, past the {MAX_MANIFEST_BYTES} B ceiling"
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SessionManifestInvalid(f"cannot read manifest {path}: {exc}") from exc
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # A truncated write lands here, which is why a partial manifest is a
        # refusal with a reason rather than a crash deep in a field accessor.
        raise SessionManifestInvalid(f"manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SessionManifestInvalid(f"manifest {path} is not a JSON object")
    return SessionManifest.from_dict(decoded)


def build_fixture_manifest(
    *,
    session_id: str = "fixture-session",
    state: SessionState = SessionState.COMPLETE,
    clock: Clock | None = None,
) -> SessionManifest:
    """A deterministic manifest for tests and for exercising readers offline.

    Deterministic on purpose: with the default :class:`FixedClock` this
    serialises to identical bytes every time, so a round-trip test compares
    exact JSON rather than a normalised subset.
    """
    used: Clock = clock if clock is not None else FixedClock()
    now = used.wall()
    return SessionManifest(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        state=state,
        created_wall=now,
        updated_wall=now,
        software_version="scanner3d-fixture",
        device=DeviceIdentity(
            vendor_id=0x2207,
            product_id=0x110C,
            serial="FIXTURESERIAL0001",
            usb_path="1-1.1",
            bus_number=1,
            device_number=50,
            topology={"0": None, "1": None, "2": "uvcvideo", "3": "uvcvideo", "4": "usbhid"},
        ),
        frames=(
            CoordinateFrame(
                name="depth_camera",
                units="millimetre",
                axes="x_right,y_down,z_forward",
                handedness="right",
                note="units unverified against a physical target",
            ),
        ),
        calibration={"depth_scale_raw": "3d23d70a", "depth_scale_note": "unit unproven"},
        capture_profile={"pixel_format": "Y16 ", "width": 640, "height": 400},
        stages=(
            StageRecord(
                name="depth_capture",
                state=StageState.COMPLETE,
                started_wall=now,
                ended_wall=now,
                duration_s=0.0,
                counters={"delivered": 1, "dropped": 0, "short": 0, "errors": 0},
                parameters={"frames_requested": 1},
                artifacts=(
                    ArtifactRecord(
                        path="raw/depth_000000.y16",
                        bytes=4,
                        sha256=hashlib.sha256(b"\x00\x01\x02\x03").hexdigest(),
                        kind="raw_depth_frame",
                        immutable=True,
                        created_wall=now,
                    ),
                ),
            ),
        ),
        limitations=("fixture data; no physical measurement supports any dimension",),
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create ``path`` exclusively and write all of it, or leave nothing behind."""
    try:
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
    except FileExistsError as exc:
        raise ArtifactWriteIncomplete(
            errno_mod.EEXIST, f"refusing to overwrite existing artifact {path}"
        ) from exc
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise ArtifactWriteIncomplete(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
        os.fsync(fd)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    finally:
        os.close(fd)


def _write_atomic_replace(path: Path, payload: bytes) -> None:
    """Write ``path`` through a temporary sibling, then rename it into place.

    The rename is what makes this atomic: a reader either sees the previous
    manifest or the new one, never a partially written file. The temporary is a
    sibling so the rename stays within one filesystem, and it is removed if
    anything fails.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    with contextlib.suppress(OSError):
        temporary.unlink()
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise ArtifactWriteIncomplete(
                    errno_mod.EIO,
                    f"write to {temporary} made no progress after {written} of "
                    f"{len(payload)} bytes",
                )
            written += just_written
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    else:
        os.close(fd)
    temporary.replace(path)
    # Durably record the rename itself, so a crash cannot leave a directory
    # entry that points at a file the filesystem has not committed.
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _join_note(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing}; {addition}"


def _as_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SessionManifestInvalid(f"{what} must be an object")
    return value


def _require_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise SessionManifestInvalid(f"'{key}' must be a string")
    return value


def _optional_str(raw: Mapping[str, object], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise SessionManifestInvalid(f"'{key}' must be a string when present")
    return value


def _require_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    # bool is an int subclass; a boolean here is a malformed record, not a 0/1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise SessionManifestInvalid(f"'{key}' must be an integer")
    return value


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise SessionManifestInvalid(f"'{key}' must be an integer or null")
    return value


def _optional_float(raw: Mapping[str, object], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionManifestInvalid(f"'{key}' must be a number or null")
    return float(value)


def _require_bool(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise SessionManifestInvalid(f"'{key}' must be a boolean")
    return value


def _optional_mapping(raw: Mapping[str, object], key: str) -> dict[str, object]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise SessionManifestInvalid(f"'{key}' must be an object")
    if not all(isinstance(name, str) for name in value):
        raise SessionManifestInvalid(f"'{key}' must have string keys")
    return dict(value)


def _optional_int_mapping(raw: Mapping[str, object], key: str) -> dict[str, int]:
    value = _optional_mapping(raw, key)
    result: dict[str, int] = {}
    for name, item in value.items():
        if not isinstance(item, int) or isinstance(item, bool):
            raise SessionManifestInvalid(f"'{key}.{name}' must be an integer")
        result[name] = item
    return result


def _require_enum[E: StrEnum](raw: Mapping[str, object], key: str, kind: type[E]) -> E:
    value = _require_str(raw, key)
    try:
        return kind(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in kind))
        raise SessionManifestInvalid(f"'{key}' must be one of {allowed}") from exc
