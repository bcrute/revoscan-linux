"""PLY export for depth-derived point clouds, and the caveats that travel with them.

A PLY file is a list of numbers behind a header that names their *types* and
nothing else. The format has no field for a unit, no field for which way the
axes point, and no field for where the numbers came from. A consumer handed a
bare PLY therefore guesses, and for this scanner every guess on offer is wrong:
the factory depth scalar recovered in :mod:`scanner3d.depth_scale` is
``float32(1/200)`` whose **physical unit is not established**, the factory
principal point recovered in :mod:`scanner3d.intrinsics` sits noticeably off the
frame centre, and no physical measurement has validated either one. A file that
let a reader quietly assume millimetres would be worse evidence than no file.

So this module writes two things - the geometry and what is *not* known about
it - and neither half is optional. The units label and the axis convention are
required arguments that end up in the PLY header as comments, because comments
are the only place a PLY can carry them; :data:`INHERENT_LIMITATIONS` is
prepended to whatever the caller states, so no call site can produce a cloud
from this repository that omits the uncertainty it inherits.

What is fixed here, and why:

* **no dependency on whatever produces the points.** The boundary is a plain
  iterable of three-float sequences plus a :class:`CloudProvenance` record. The
  module that unprojects depth samples into coordinates owns the geometry and
  the calibration; this module owns the file format. Neither imports the other,
  and they compose through data alone.
* **complete or absent.** :func:`write_ply` creates its file with
  ``O_EXCL | O_NOFOLLOW``, resumes short writes from the offset the kernel
  actually reached, treats a zero-progress write as an error rather than
  spinning, and unlinks whatever it made if any of that fails. A truncated file
  under a name promising a whole cloud is the one artifact worse than none.
* **bounded twice.** :data:`MAX_POINTS` caps how many points may be encoded and
  :data:`MAX_OUTPUT_BYTES` caps the file. A caller may lower either bound and
  may never raise it, so no argument widens a limit at run time. The point cap
  is enforced while the iterable is being consumed, so an unbounded generator
  is refused rather than exhausted.
* **nothing non-finite reaches a file.** NaN and both infinities are refused,
  and so is a value that is finite as a float64 but overflows binary32. The
  refusal happens while the points are being collected - before anything is
  opened, created or written.
* **binary32 in both encodings.** The binary body packs three little-endian
  ``float`` (binary32) values per point, and the ASCII body writes the
  ``repr`` of *those same rounded values*. Rendering fewer digits would make
  the two encodings of one cloud disagree in the last bits, which would turn a
  choice of file format into a change in the data.

Nothing in this module computes a coordinate, and nothing in it claims
dimensional accuracy.
"""

from __future__ import annotations

import contextlib
import enum
import errno as errno_mod
import hashlib
import math
import os
import struct
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from scanner3d import __version__
from scanner3d.errors import ScannerError

#: Companion-metadata schema. Bumped only when a reader would misread the old
#: shape, never for an added field.
SCHEMA_VERSION: Final = 1

#: One point as it crosses the boundary into this module: three finite floats,
#: ``x`` then ``y`` then ``z``. A plain sequence on purpose - the producer of the
#: geometry must not have to import a record type from the file writer.
type Point = Sequence[float]

#: The most points one file may carry. A depth frame of this unit is 640x400 =
#: 256000 samples, so this is eight full frames' worth of surviving points: room
#: to merge a handful of captures, and still a hard stop long before a runaway
#: producer fills a disk. Never widened at run time.
DEPTH_FRAME_SAMPLES: Final = 640 * 400
MAX_POINTS: Final = 8 * DEPTH_FRAME_SAMPLES

#: The most bytes one file may occupy. :data:`MAX_POINTS` in the binary encoding
#: is about 24 MiB and in the ASCII encoding roughly three times that, so this
#: ceiling binds the ASCII case and nothing else. Never widened at run time.
MAX_OUTPUT_BYTES: Final = 128 * 1024 * 1024

#: The header is comments plus a dozen fixed lines. This bounds what a caller's
#: provenance strings can inflate it to; a header past it is refused, never
#: truncated, because a half-stated caveat reads as a complete one.
#:
#: 64 KiB rather than the 8 KiB it was: a real turntable registration on
#: 2026-09-04 carried five limitations (1200 characters, three of them only
#: present when closures were rejected) and a 1645-character preparation
#: statement, and its header measured 8965 bytes - so the first export of the
#: best capture this project had produced was refused for its caveats being
#: too complete. The bound is against runaway provenance, not against prose.
MAX_HEADER_BYTES: Final = 64 * 1024

#: Where comment text is wrapped. Well inside the 79 columns that make a header
#: readable in a terminal, with room for the ``comment `` prefix.
COMMENT_WRAP_WIDTH: Final = 70

PLY_MAGIC: Final = "ply"
PLY_VERSION: Final = "1.0"

#: One vertex in the binary encoding: x, y, z as little-endian binary32.
VERTEX_STRUCT: Final = struct.Struct("<fff")
VERTEX_BYTES: Final = VERTEX_STRUCT.size

#: One binary32, used to round a coordinate to the width both encodings carry.
_BINARY32: Final = struct.Struct("<f")

#: The units label for coordinates produced with the factory depth scalar
#: applied. Restated here rather than imported from :mod:`scanner3d.depth_scale`
#: so that a file-format module does not drag in the USB control stack; the
#: uncertainty it describes is the same one that module preserves.
SCALED_UNIT_LABEL: Final = (
    "device depth sample units multiplied by the factory depth scalar; the physical "
    "unit of that scalar is NOT established by any evidence in this repository, so "
    "these coordinates are unitless numbers and are not millimetres, metres, or any "
    "other length"
)

#: The units label for coordinates carrying raw depth samples, unscaled.
RAW_UNIT_LABEL: Final = (
    "raw device depth sample units with no scalar applied; unitless numbers, not lengths"
)

#: Caveats every cloud this repository writes inherits, whatever the caller
#: says. Prepended to the caller's own limitations by
#: :attr:`CloudProvenance.all_limitations`, so a call site cannot drop one by
#: passing a shorter list.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "no physical measurement has validated these coordinates; nothing here establishes "
    "dimensional accuracy, and no dimension read off this file is a measured length",
    "the factory depth scalar's physical unit is unknown, so the numeric scale of these "
    "coordinates is unknown with it",
    "the factory principal point recorded by this unit sits off the frame centre and has "
    "not been independently verified; any coordinate computed through it inherits that",
    "a depth sample of zero means 'no depth returned', not zero distance; such samples "
    "belong nowhere in this file, and a point at the origin is not a measurement",
)


# --- refusals -----------------------------------------------------------------


class PointCloudNotAllowed(ScannerError):
    """A cloud was offered that this writer refuses to encode.

    Raised before any path is opened, created or written, so a refused cloud
    never leaves a file - not even an empty one - behind it.
    """


class NonFiniteCoordinate(PointCloudNotAllowed):
    """A coordinate is NaN, an infinity, or outside the binary32 range.

    All three are the same defect from a consumer's point of view: a vertex
    that no viewer can place. Writing one would also silently destroy the
    bounding box of every other point in the file.
    """


class PointCountExceeded(PointCloudNotAllowed):
    """More points were offered than :data:`MAX_POINTS`, or than the caller's cap.

    Raised on the first point past the cap, so an unbounded producer is stopped
    rather than drained.
    """


class OutputSizeExceeded(PointCloudNotAllowed):
    """The encoded file would pass :data:`MAX_OUTPUT_BYTES`, or the caller's cap."""


class PointCloudWriteIncomplete(OSError):
    """The file could not be written in full, so none was kept.

    An ``OSError`` for the same reason
    :class:`scanner3d.errors.V4l2ArtifactIncomplete` and
    :class:`scanner3d.session.ArtifactWriteIncomplete` are: it is raised on the
    same path as a real write failure and carries an ``errno``, so a short
    write, a zero-progress write, an existing name and a kernel error all stay
    describable in one field.
    """


# --- what the header has to say -----------------------------------------------


class PlyFormat(enum.StrEnum):
    """The two encodings this module writes.

    The values are the exact tokens the PLY ``format`` line uses, so the header
    is built from the enum rather than from a parallel table that could drift.
    """

    ASCII = "ascii"
    BINARY_LITTLE_ENDIAN = "binary_little_endian"


@dataclass(frozen=True, slots=True)
class CoordinateConvention:
    """Which way the axes point, stated because a PLY cannot state it.

    This record is a *label*: nothing here rotates, flips or checks a
    coordinate. It describes the frame the producer says its points are already
    in, and carries that description into the file and the metadata so a
    consumer is not left inferring handedness from the sign of ``z``.
    """

    name: str
    x_axis: str
    y_axis: str
    z_axis: str
    origin: str
    handedness: str

    @property
    def summary(self) -> str:
        """One line, for the header comment."""
        return (
            f"{self.name} ({self.handedness}); +x {self.x_axis}; +y {self.y_axis}; "
            f"+z {self.z_axis}; origin at {self.origin}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "z_axis": self.z_axis,
            "origin": self.origin,
            "handedness": self.handedness,
        }


#: The frame a pinhole unprojection of a depth image lands in: the optical frame
#: of the depth camera, with the image axes kept as they are read. Offered as a
#: named constant so producers agree on one description instead of inventing a
#: phrasing each; it is still only a label, and a producer that emits some other
#: frame must say so rather than reuse this one.
CAMERA_OPTICAL_FRAME: Final = CoordinateConvention(
    name="depth_camera_optical_frame",
    x_axis="right along a sensor row, in the direction of increasing column",
    y_axis="down a sensor column, in the direction of increasing row",
    z_axis="away from the camera along the optical axis",
    origin="the depth camera's projection centre as the factory intrinsics place it",
    handedness="right-handed",
)


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Where the numbers that turned samples into coordinates came from.

    Every field is a statement about evidence, not about accuracy: naming the
    device read that produced an intrinsics record says the bytes came off this
    unit, and says nothing at all about whether the record describes the world
    correctly.
    """

    source: str
    intrinsics_sha256: str | None = None
    depth_scale: float | None = None
    depth_scale_unit: str = "unknown; this repository has established no physical unit for it"
    caveats: tuple[str, ...] = ()

    @classmethod
    def none_applied(cls, detail: str) -> CalibrationProvenance:
        """Provenance for a cloud built without any calibration at all."""
        return cls(source=f"none applied: {detail}")

    @property
    def summary(self) -> str:
        """One line, for the header comment."""
        scale = "none applied" if self.depth_scale is None else repr(self.depth_scale)
        digest = self.intrinsics_sha256 or "none recorded"
        return (
            f"source: {self.source}; intrinsics sha256: {digest}; depth scale: {scale} "
            f"(unit {self.depth_scale_unit})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "intrinsics_sha256": self.intrinsics_sha256,
            "depth_scale": self.depth_scale,
            "depth_scale_unit": self.depth_scale_unit,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True, slots=True)
class CloudProvenance:
    """Everything the geometry cannot say about itself.

    Required at every call: :func:`encode_ply` takes one of these rather than a
    pile of optional keywords, so a cloud with no stated units or no stated
    convention is not a shape this API can express.
    """

    units: str
    convention: CoordinateConvention
    calibration: CalibrationProvenance
    source_frame_sha256: str | None = None
    producer: str = f"scanner3d {__version__}"
    limitations: tuple[str, ...] = ()

    @property
    def all_limitations(self) -> tuple[str, ...]:
        """:data:`INHERENT_LIMITATIONS` first, then the caller's, de-duplicated.

        The inherent ones are not the caller's to omit, and de-duplication keeps
        a caller that restates one from making the header say it twice.
        """
        seen: list[str] = []
        for item in (*INHERENT_LIMITATIONS, *self.limitations, *self.calibration.caveats):
            if item not in seen:
                seen.append(item)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "units": {
                "label": self.units,
                # Stated as a field rather than left to prose: a machine reading
                # this file must be able to see that no accuracy is claimed.
                "dimensional_accuracy_claimed": False,
            },
            "coordinate_convention": self.convention.as_dict(),
            "calibration": self.calibration.as_dict(),
            "source_frame_sha256": self.source_frame_sha256,
            "limitations": list(self.all_limitations),
        }


# --- the encoded file ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodedCloud:
    """A complete PLY held in memory, plus the metadata that belongs beside it.

    ``path`` is ``None`` until :func:`write_ply` has actually written the bytes,
    so a record naming a path is a record of a file that exists in full.
    :meth:`as_dict` is the companion metadata a caller serialises next to the
    PLY; it deliberately excludes :attr:`payload`, because the metadata sits
    beside the geometry and never in place of it.
    """

    ply_format: PlyFormat
    point_count: int
    payload: bytes
    header: str
    sha256: str
    provenance: CloudProvenance
    max_points: int
    max_output_bytes: int
    path: str | None = None

    @property
    def byte_count(self) -> int:
        return len(self.payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": {
                "path": self.path,
                "written": self.path is not None,
                "format": str(self.ply_format),
                "byte_count": self.byte_count,
                "sha256": self.sha256,
            },
            "point_count": self.point_count,
            "vertex_properties": ["x", "y", "z"],
            "vertex_property_type": "float32 (PLY 'float'), little-endian in the binary encoding",
            "provenance": self.provenance.as_dict(),
            "bounds": {
                "max_points": self.max_points,
                "max_output_bytes": self.max_output_bytes,
                "module_max_points": MAX_POINTS,
                "module_max_output_bytes": MAX_OUTPUT_BYTES,
            },
            # The exact header, so a reader of the metadata alone can see what
            # the file itself declares without parsing it.
            "ply_header": self.header,
        }


# --- encoding -----------------------------------------------------------------


def sha256_of(payload: bytes) -> str:
    """The digest every record here quotes. One implementation, so the metadata
    and any caller-side check cannot drift into hashing the same bytes twice."""
    return hashlib.sha256(payload).hexdigest()


def binary32(value: float) -> float:
    """Round ``value`` to IEEE-754 binary32, the width a PLY ``float`` carries.

    Raises :class:`NonFiniteCoordinate` for NaN, either infinity, a magnitude
    outside the binary32 range, and an integer too large to be a float at all.
    All four end the same way - a vertex no consumer can place - so they raise
    one type, with a message that says which happened.
    """
    try:
        as_float = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise NonFiniteCoordinate(f"coordinate {value!r} is not a usable float: {exc}") from exc
    if not math.isfinite(as_float):
        raise NonFiniteCoordinate(
            f"coordinate {as_float!r} is not finite; NaN and infinities are refused before "
            "anything is opened, because one of them destroys the bounding box of the "
            "whole cloud"
        )
    try:
        packed = _BINARY32.pack(as_float)
    except OverflowError as exc:
        # struct's range check is the binary32 gate: a float64 as ordinary as
        # 1e39 is finite here and an infinity in the file.
        raise NonFiniteCoordinate(
            f"coordinate {as_float!r} is finite as a float64 but overflows the binary32 "
            "range a PLY 'float' property carries, so it would be written as an infinity"
        ) from exc
    rounded: float = _BINARY32.unpack(packed)[0]
    return rounded


def _collect(points: Iterable[Point], max_points: int) -> tuple[tuple[float, float, float], ...]:
    """Consume ``points`` into validated, binary32-rounded triples.

    Stops at the first point past ``max_points`` rather than after the iterable
    is drained, so an unbounded producer is refused instead of exhausting
    memory, and no coordinate is trusted before it has been through
    :func:`binary32`.
    """
    collected: list[tuple[float, float, float]] = []
    for index, point in enumerate(points):
        if len(collected) >= max_points:
            raise PointCountExceeded(
                f"refusing to encode more than {max_points} points; the producer offered at "
                f"least {index + 1}, and the cap is never widened at run time"
            )
        values = tuple(point)
        if len(values) != 3:
            raise PointCloudNotAllowed(
                f"point {index} carries {len(values)} coordinates, not the 3 an (x, y, z) "
                "triple has"
            )
        collected.append((binary32(values[0]), binary32(values[1]), binary32(values[2])))
    return tuple(collected)


def _one_line(text: str) -> str:
    """Flatten ``text`` to a single line of printable ASCII.

    This is the header's injection guard as much as a tidying step. Provenance
    strings come from a caller, a PLY header is line-structured, and a units
    label containing a newline and ``end_header`` would otherwise end the header
    early and turn the rest of the file into garbage. Everything outside
    printable ASCII - newlines included - becomes a space, so no caller-supplied
    string can produce a header line of its own.
    """
    printable = "".join(ch if " " <= ch <= "~" else " " for ch in text)
    return " ".join(printable.split())


def _comment(label: str, text: str) -> list[str]:
    """``comment`` lines carrying ``label: text``, wrapped rather than truncated."""
    body = _one_line(text) or "(not stated)"
    wrapped = textwrap.wrap(f"{label}: {body}", width=COMMENT_WRAP_WIDTH, subsequent_indent="  ")
    return [f"comment {line}" for line in wrapped]


def build_header(ply_format: PlyFormat, point_count: int, provenance: CloudProvenance) -> str:
    """The complete PLY header, comments and all, ending in ``end_header``.

    The comments are the point: they are the only place the format lets this
    file record its units, its axes and its provenance, and a consumer that
    reads the vertices without them is guessing.
    """
    lines = [PLY_MAGIC, f"format {ply_format} {PLY_VERSION}"]
    lines += _comment("produced by", provenance.producer)
    lines += _comment(
        "why these comments exist",
        "a PLY header records types and nothing else, so units, axes and provenance can "
        "only be stated here; they are not decoration and this file makes no sense without "
        "them",
    )
    lines += _comment("units", provenance.units)
    lines += _comment("dimensional accuracy", "none is claimed; see the limitations below")
    lines += _comment("coordinate convention", provenance.convention.summary)
    lines += _comment("source frame sha256", provenance.source_frame_sha256 or "none recorded")
    lines += _comment("calibration", provenance.calibration.summary)
    for limitation in provenance.all_limitations:
        lines += _comment("limitation", limitation)
    lines += [
        f"element vertex {point_count}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    # "\n" and never "\r\n": the PLY header is defined line by line and a
    # trailing carriage return ends up inside the last token of readers that
    # split on "\n" alone, which is exactly the kind of breakage a binary body
    # makes hard to diagnose.
    header = "".join(f"{line}\n" for line in lines)
    encoded = header.encode("ascii")
    if len(encoded) > MAX_HEADER_BYTES:
        raise PointCloudNotAllowed(
            f"the header is {len(encoded)} bytes, past the {MAX_HEADER_BYTES}-byte ceiling; "
            "it is refused rather than truncated, because half a stated caveat reads like a "
            "whole one"
        )
    return header


def _ascii_body(points: Sequence[tuple[float, float, float]]) -> bytes:
    """One ``x y z`` line per point, in the binary encoding's own precision.

    ``repr`` of an already binary32-rounded value is the shortest decimal that
    reads back as that exact value, so the ASCII and binary files decode to
    identical coordinates. A prettier fixed-width rendering would quietly make
    the choice of format a choice of data.
    """
    return "".join(f"{x!r} {y!r} {z!r}\n" for x, y, z in points).encode("ascii")


def _binary_body(points: Sequence[tuple[float, float, float]]) -> bytes:
    """Three little-endian binary32 values per point, tightly packed."""
    return b"".join(VERTEX_STRUCT.pack(x, y, z) for x, y, z in points)


def encode_ply(
    points: Iterable[Point],
    *,
    provenance: CloudProvenance,
    ply_format: PlyFormat = PlyFormat.BINARY_LITTLE_ENDIAN,
    max_points: int = MAX_POINTS,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> EncodedCloud:
    """Encode a whole PLY in memory, or refuse. Opens nothing and writes nothing.

    Binary is the default because a full frame's worth of points is a quarter
    of a million vertices; ASCII is the opt-in for reading a small cloud by eye.

    ``max_points`` and ``max_output_bytes`` may be *lowered* by a caller and
    never raised: the module's ceilings are the ceilings, and an argument that
    tried to widen one would make the bound a suggestion.
    """
    if not 0 <= max_points <= MAX_POINTS:
        raise PointCloudNotAllowed(
            f"max_points {max_points} is outside [0, {MAX_POINTS}]; the module ceiling is "
            "never widened at run time"
        )
    if not 0 < max_output_bytes <= MAX_OUTPUT_BYTES:
        raise PointCloudNotAllowed(
            f"max_output_bytes {max_output_bytes} is outside (0, {MAX_OUTPUT_BYTES}]; the "
            "module ceiling is never widened at run time"
        )
    collected = _collect(points, max_points)
    header = build_header(ply_format, len(collected), provenance)
    body = _ascii_body(collected) if ply_format is PlyFormat.ASCII else _binary_body(collected)
    payload = header.encode("ascii") + body
    if len(payload) > max_output_bytes:
        raise OutputSizeExceeded(
            f"the encoded {ply_format} cloud is {len(payload)} bytes for {len(collected)} "
            f"points, past the {max_output_bytes}-byte ceiling; nothing was written"
        )
    return EncodedCloud(
        ply_format=ply_format,
        point_count=len(collected),
        payload=payload,
        header=header,
        sha256=sha256_of(payload),
        provenance=provenance,
        max_points=max_points,
        max_output_bytes=max_output_bytes,
    )


# --- writing ------------------------------------------------------------------


def write_ply(
    path: Path,
    points: Iterable[Point],
    *,
    provenance: CloudProvenance,
    ply_format: PlyFormat = PlyFormat.BINARY_LITTLE_ENDIAN,
    max_points: int = MAX_POINTS,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> EncodedCloud:
    """Encode the cloud, then write it whole or leave nothing behind.

    Everything that can refuse the cloud - a non-finite coordinate, a point
    past the cap, a file past the ceiling - has already refused it by the time
    ``path`` is touched, so a rejected cloud never creates a file.

    Returns the record of what was written, whose :meth:`EncodedCloud.as_dict`
    is the companion metadata to serialise beside the PLY.
    """
    cloud = encode_ply(
        points,
        provenance=provenance,
        ply_format=ply_format,
        max_points=max_points,
        max_output_bytes=max_output_bytes,
    )
    _write_new_file(path, cloud.payload, ceiling=max_output_bytes)
    return replace(cloud, path=str(path))


def _write_new_file(path: Path, payload: bytes, *, ceiling: int = MAX_OUTPUT_BYTES) -> None:
    """Create ``path`` exclusively and write all of ``payload``, or leave nothing.

    This is the same discipline as ``scanner3d.v4l2_capture._write_new_file``
    and ``scanner3d.session._write_exclusive``, reimplemented rather than
    imported. Sharing it would mean this module - a file format with no
    hardware in it - importing the V4L2 capture stack, which pulls in the
    ownership, control and discovery layers behind it, so that a PLY writer
    could open a file. The three copies also raise their own layer's error
    type, which is what makes each one's failure reportable where it happens.
    The guarantees are identical and are the point of the duplication:

    ``O_EXCL | O_NOFOLLOW`` because an output directory is often a
    world-writable ``/tmp``: an existing name is an error rather than a licence
    to clobber, and a symlink is never followed out of the directory.

    ``os.write`` may write fewer bytes than it was given, so one call is not a
    file. This resumes from the offset the kernel actually reached, and it is
    bounded three ways: the payload is at most ``ceiling`` bytes, every
    iteration must make strictly positive progress, and a write of zero bytes
    ends the attempt instead of spinning. Anything that stops the write - a
    kernel error, a stalled write, a signal - unlinks the partial file, because
    a truncated file under a name promising a whole cloud is worse than no file
    at all.
    """
    if len(payload) > ceiling:
        raise PointCloudWriteIncomplete(
            errno_mod.EFBIG,
            f"refusing to write {len(payload)} bytes to {path}, past the {ceiling} B ceiling",
        )
    try:
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
    except FileExistsError as exc:
        raise PointCloudWriteIncomplete(
            errno_mod.EEXIST, f"refusing to overwrite existing point cloud {path}"
        ) from exc
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise PointCloudWriteIncomplete(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
        # The cloud is derived evidence a later run may be compared against, so
        # it is on the platter before this function claims it exists.
        os.fsync(fd)
    except BaseException:
        _discard_partial(path)
        raise
    finally:
        os.close(fd)


def _discard_partial(path: Path) -> None:
    """Remove a file that was created but not completely written.

    Best effort by design: the write already failed, and a failure to unlink
    must not replace the error that explains why there is no cloud.
    """
    with contextlib.suppress(OSError):
        path.unlink()
