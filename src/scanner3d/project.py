"""One file per scan that says what the scan is, so a person picks one file.

Why this exists
---------------
A capture is a folder of ``frame_*.raw`` files, plus boundary directories for
a repositioned capture and, after a reconstruction, one output folder per run.
Asked to reconstruct one, the window used to open a FOLDER chooser over a tree
of dated folders full of identically named frames, and an operator who picked
the wrong level was told nothing about why the result was empty. The settings
the capture ran under - the depth band, the frame spacing, whether it stopped
early - lived nowhere at all.

So every capture writes ``scan.scanner3d`` beside its frames: a small JSON
record of how the frames were taken, which reconstructions were run on them
and where each one's files are. The reconstruct action asks for THAT file.
An older capture that has none is adopted on first use, which writes the same
file with what can still be read off the folder and says the settings were not
recorded.

Rules
-----
* The file describes; it never contains. Frames and outputs stay where they
  are, and the file names them by paths relative to itself, so a scan folder
  can be moved or copied whole.
* It is written atomically and re-read strictly: a truncated or foreign JSON
  file is refused rather than partly believed, because a project file that is
  half right sends a reconstruction at the wrong frames.
* Nothing here opens a device or reads a frame.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from scanner3d.capture_sequence import frame_times
from scanner3d.errors import DoctorStatus, ScannerError

#: Schema this module writes. A reader refuses anything newer.
SCHEMA_VERSION: Final = 1

#: What the file says it is, so any other JSON file is refused by name.
KIND: Final = "scanner3d-scan"

#: The suffix a file chooser filters on, and the fixed name inside a capture.
PROJECT_SUFFIX: Final = ".scanner3d"
PROJECT_NAME: Final = "scan" + PROJECT_SUFFIX

#: The frames a capture folder is recognised by, at any depth (a repositioned
#: capture keeps later frames in boundary directories).
FRAME_GLOB: Final = "frame_*.raw"

#: A project file is a few kilobytes of metadata; anything this large is not one.
MAX_PROJECT_BYTES: Final = 4 * 1024 * 1024


class ProjectError(ScannerError):
    """A project file could not be read, written or trusted."""

    status = DoctorStatus.UNSUPPORTED_MODE


class ProjectUnreadable(ProjectError):
    """The path names no project file, or one that cannot be read."""


class ProjectInvalid(ProjectError):
    """The file exists but does not say what a project file has to say."""


class NotACapture(ProjectError):
    """A folder offered for adoption holds no frames."""


@dataclass(frozen=True, slots=True)
class ScanProject:
    """A project file, read and checked."""

    path: Path
    frames_directory: Path
    created: str
    capture: dict[str, object]
    reconstructions: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "created": self.created,
            "frames_directory": _relative(self.frames_directory, self.path.parent),
            "capture": dict(self.capture),
            "reconstructions": [dict(entry) for entry in self.reconstructions],
        }


def project_path_for(directory: Path) -> Path:
    """Where a capture folder's project file lives."""
    return Path(directory) / PROJECT_NAME


def _timestamp(now: datetime | None) -> str:
    moment = now if now is not None else datetime.now(UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _relative(target: Path, base: Path) -> str:
    try:
        rel = Path(target).resolve().relative_to(Path(base).resolve())
    except ValueError:
        return str(Path(target).resolve())
    return "." if str(rel) == "." else rel.as_posix()


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write ``payload`` as JSON so that the file is either whole or absent."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_project(
    directory: Path,
    *,
    source: str,
    capture: Mapping[str, object],
    now: datetime | None = None,
) -> Path:
    """Create ``scan.scanner3d`` in ``directory`` describing a capture taken there.

    ``source`` names what took the capture; ``capture`` is whatever that
    source knows about it, as plain JSON-safe values. Replaces an existing file
    atomically - a capture folder has one project file, and the capture that
    is writing into the folder is the authority on it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ProjectUnreadable(f"{directory} is not a directory; no project file was written")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created": _timestamp(now),
        "frames_directory": ".",
        "capture": {"source": source, **dict(capture)},
        "reconstructions": [],
    }
    path = project_path_for(directory)
    _write_atomic(path, payload)
    return path


def read_project(path: Path) -> ScanProject:
    """Read and check a project file, or the one inside a capture folder.

    Every refusal names the file. A directory is looked in for its project
    file; anything else has to be the file itself.
    """
    path = Path(path)
    if path.is_dir():
        path = project_path_for(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ProjectUnreadable(f"{path} does not exist") from exc
    except OSError as exc:
        raise ProjectUnreadable(f"{path} could not be read: {exc}") from exc
    if len(raw) > MAX_PROJECT_BYTES:
        raise ProjectInvalid(f"{path} is {len(raw)} bytes, which is not a project file")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectInvalid(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectInvalid(f"{path} does not hold a JSON object")
    if data.get("kind") != KIND:
        raise ProjectInvalid(f"{path} is not a {KIND} file (kind={data.get('kind')!r})")
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProjectInvalid(f"{path} states no usable schema_version ({version!r})")
    if version > SCHEMA_VERSION:
        raise ProjectInvalid(
            f"{path} is schema version {version}, newer than the {SCHEMA_VERSION} this reads; "
            "a newer field may change what the record means"
        )
    frames = data.get("frames_directory")
    if not isinstance(frames, str) or not frames:
        raise ProjectInvalid(f"{path} names no frames_directory")
    frames_directory = (path.parent / frames).resolve()
    if not frames_directory.is_dir():
        raise ProjectInvalid(f"{path} points at {frames_directory}, which is not a directory")
    capture = data.get("capture")
    if not isinstance(capture, dict):
        raise ProjectInvalid(f"{path} holds no capture record")
    reconstructions = data.get("reconstructions", [])
    if not isinstance(reconstructions, list) or not all(
        isinstance(entry, dict) for entry in reconstructions
    ):
        raise ProjectInvalid(f"{path} holds a malformed reconstructions list")
    created = data.get("created")
    return ScanProject(
        path=path,
        frames_directory=frames_directory,
        created=created if isinstance(created, str) else "",
        capture=dict(capture),
        reconstructions=tuple(dict(entry) for entry in reconstructions),
    )


def adopt_capture(directory: Path, *, now: datetime | None = None) -> Path:
    """Give an older capture folder a project file, from what the folder still shows.

    Returns the existing file untouched when the folder already has one: an
    adoption must never overwrite a record a capture wrote. Refuses a folder
    with no frames, because a project file over an empty folder is a promise
    of frames that do not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotACapture(f"{directory} is not a directory")
    existing = project_path_for(directory)
    if existing.is_file():
        return existing
    frames = sorted(directory.rglob(FRAME_GLOB))
    if not frames:
        raise NotACapture(f"{directory} holds no {FRAME_GLOB}, so there is nothing to adopt")
    return write_project(
        directory,
        source="adopted",
        capture={
            "frames_written": len(frames),
            "note": (
                "adopted from a capture folder that predates project files; the settings "
                "it was captured under were not recorded"
            ),
        },
        now=now,
    )


def resolve_capture_input(path: Path) -> tuple[Path, Path | None]:
    """The frames directory and the project file, from whatever the operator picked.

    Accepts a capture folder, a project file, or any file inside a capture
    folder - the last so that an older capture can be chosen from a file
    dialog that cannot select folders. The project file comes back as None
    when the folder has none yet; callers adopt it or not as they see fit.
    """
    path = Path(path)
    if path.is_dir():
        project = project_path_for(path)
        return path, (project if project.is_file() else None)
    if path.suffix == PROJECT_SUFFIX:
        loaded = read_project(path)
        return loaded.frames_directory, path
    if path.is_file():
        project = project_path_for(path.parent)
        return path.parent, (project if project.is_file() else None)
    raise ProjectUnreadable(
        f"{path} is neither a capture folder, a {PROJECT_NAME} file, nor a file inside a capture"
    )


#: The capture record's list of each frame's delivery time, seconds from the first.
FRAME_TIMES_KEY: Final = "frame_times_s"


def capture_frame_times(paths: Sequence[Path], project_path: Path | None) -> tuple[float, ...]:
    """Each frame's delivery time in seconds from the first.

    From the project file's record when it has one for exactly these frames,
    and from the files' own modification times otherwise - a capture read at
    a stride, or one whose project file predates the record, still has the
    times its files carry. The record exists because copying a capture can
    lose those.
    """
    if project_path is not None:
        try:
            recorded = read_project(project_path).capture.get(FRAME_TIMES_KEY)
        except ProjectError:
            recorded = None
        if (
            isinstance(recorded, list)
            and len(recorded) == len(paths)
            and all(isinstance(value, (int, float)) for value in recorded)
        ):
            first = float(recorded[0]) if recorded else 0.0
            return tuple(float(value) - first for value in recorded)
    return frame_times(paths)


def update_capture(path: Path, **fields: object) -> None:
    """Add or replace fields of the capture record, keeping everything else."""
    project = read_project(path)
    payload = project.as_dict()
    capture = dict(project.capture)
    capture.update(fields)
    payload["capture"] = capture
    _write_atomic(project.path, payload)


def record_reconstruction(
    path: Path,
    *,
    directory: Path,
    record_name: str,
    details: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> None:
    """Append one reconstruction's output folder and record to the project file."""
    project = read_project(path)
    payload = project.as_dict()
    folder = _relative(directory, project.path.parent)
    entry: dict[str, object] = {
        "directory": folder,
        "record": f"{folder}/{record_name}",
        "when": _timestamp(now),
        **dict(details or {}),
    }
    payload["reconstructions"] = [*payload["reconstructions"], entry]  # type: ignore[misc]
    _write_atomic(project.path, payload)
