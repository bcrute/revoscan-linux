"""The one seam between a deprojected depth frame and the PLY that carries it.

:mod:`scanner3d.depth_metric` turns a raw ``Y16 `` frame into camera-frame
points plus a large amount of carefully stated uncertainty.
:mod:`scanner3d.pointcloud` writes a PLY whose ``comment`` block is the only
place that uncertainty can survive, because the format itself has no field for a
unit, an axis convention or a provenance. The two were built to compose through
data alone and deliberately do not import each other. That leaves exactly one
thing missing between them, and this module is it.

The gap is not plumbing. Both modules define a record named
``CoordinateConvention`` and **they are not the same record**:

both carry ``origin``, ``x_axis``, ``y_axis``, ``z_axis`` and ``handedness``;
depth_metric's alone carries ``note``, and pointcloud's alone carries ``name``.
A field-by-field copy therefore silently loses one field and cannot fill the
other, which is exactly the kind of translation that looks finished.

The missing ``note`` is the whole problem. It is where
:data:`~scanner3d.depth_metric.CAMERA_FRAME` says that the convention is the one
that module *applies* - the one implied by its own arithmetic - and not one the
device ever declared. Translating the record by copying the four axis strings
and dropping the note would produce a PLY header that states an axis convention
as though the hardware had confirmed it. That is the precise failure this
repository exists to avoid, so the note is carried across, into the header, by
the route explained at :func:`convention_caveat`.

The same rule decides everything else here:

* **every limitation reaches the file.** ``DepthConversion.limitations`` is
  passed through verbatim into :attr:`~scanner3d.pointcloud.CloudProvenance.
  limitations`, which :func:`~scanner3d.pointcloud.build_header` renders as its
  own ``comment limitation:`` lines. A PLY that lost them would be worse than no
  PLY: a consumer would take the numbers at face value, and nothing in the
  geometry would contradict them.
* **the caller's assumed unit is carried as an assumption.** The three strings
  of a :class:`~scanner3d.depth_metric.DepthUnitAssumption` - the label, who
  stakes it and on what - become the units line, always with the word *assumed*
  and never as a measured fact.
* **what did not survive the conversion is stated too.** A cloud is a subset of
  a frame: most samples of every frame captured from this unit return no depth.
  The per-reason exclusion counts go into the header, because a gap in a point
  cloud is an absent measurement and reads exactly like empty space.
* **observations the conversion made and did not act on are not lost.** Where
  the principal point was placed, and any check
  :func:`~scanner3d.depth_metric.deproject_depth_frame` recorded as failing,
  become calibration caveats - including the structural-record check, which
  ``DepthConversion.limitations`` does not otherwise mention.

Nothing here computes a coordinate, opens a device or claims dimensional
accuracy. **No physical measurement has validated any distance this repository
produces**, so no assertion in this module, and no field of any record it
builds, may be read as one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.depth_metric import (
    CAMERA_FRAME,
    DEPROJECTION_EXPRESSION,
    DepthConversion,
    DepthUnitAssumption,
    deproject_depth_frame,
)
from scanner3d.depth_metric import CoordinateConvention as AppliedConvention
from scanner3d.depth_scale import UNIT_DISCLAIMER
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.pointcloud import (
    MAX_OUTPUT_BYTES,
    MAX_POINTS,
    CalibrationProvenance,
    CloudProvenance,
    EncodedCloud,
    PlyFormat,
    write_ply,
)
from scanner3d.pointcloud import CoordinateConvention as FileConvention
from scanner3d.v4l2_capture import sha256_of

#: Every top-level report this module emits stamps this.
SCHEMA_VERSION: Final = 1

#: What this module does, said once for the report.
EXPORT_MODE: Final = "depth-frame-to-ply"

#: How long a SHA-256 digest is, in lowercase hex characters.
SHA256_HEX_LENGTH: Final = 64

_HEX_DIGITS: Final = frozenset("0123456789abcdef")

#: The name given to the frame :data:`~scanner3d.depth_metric.CAMERA_FRAME`
#: describes. Deliberately the same string
#: :data:`scanner3d.pointcloud.CAMERA_OPTICAL_FRAME` uses for it, so two files
#: written by this repository in the same frame carry the same label - but the
#: axis text below is depth_metric's own wording rather than that constant's
#: shorter paraphrase, because the wording is the part a consumer acts on.
CAMERA_FRAME_NAME: Final = "depth_camera_optical_frame"

#: The name any *other* convention gets. It claims nothing: this module has no
#: way to check that some convention it has never seen is the optical frame, and
#: a label that guesses is worse than one that declines.
UNRECOGNISED_FRAME_NAME: Final = "producer_stated_frame"

#: Said in the header whenever the conversion produced no point at all.
EMPTY_CLOUD_NOTE: Final = (
    "no sample in the source frame produced a point, so this file has zero vertices. That "
    "records a conversion that ran and found nothing - a bad target, an unlit scene, a "
    "sensor that returned no depth - and is not a measurement of an empty space"
)


# --- refusals -----------------------------------------------------------------


class CloudExportNotAllowed(ScannerError):
    """The export was refused before any file was created.

    Every subclass is raised up front by the functions below, before a path is
    opened and before :func:`~scanner3d.pointcloud.write_ply` is reached, so a
    refused export leaves nothing on disk - not a partial file, not an empty
    one.

    The status is :attr:`~scanner3d.errors.DoctorStatus.UNSUPPORTED_MODE` for
    the same reason :class:`~scanner3d.depth_metric.DepthConversionNotAllowed`
    uses it: the doctor vocabulary has no class for "the inputs do not describe
    an export this module can perform", and that is nearer the truth than
    calling a caller's bad argument an internal error.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class CalibrationSourceNotStated(CloudExportNotAllowed):
    """The caller did not say where the calibration came from.

    The one fact in the header that a :class:`~scanner3d.depth_metric.
    DepthConversion` genuinely cannot supply: the record it was handed does not
    know whether it was read off this unit, parsed from a file or invented by a
    test. A blank source would put an empty ``source:`` in the calibration
    comment, which reads as "no calibration was applied" rather than as "nobody
    said", so it is refused instead of defaulted.
    """


class IntrinsicsDigestMalformed(CloudExportNotAllowed):
    """The intrinsics digest offered is not a SHA-256 hex digest.

    ``None`` is a legitimate answer and means "none recorded". A string that is
    not 64 lowercase hex characters is not: it would sit in a field named
    ``intrinsics_sha256`` looking exactly like evidence a reader could check.
    """


class SourceFrameMismatch(CloudExportNotAllowed):
    """The frame bytes offered are not the frame the conversion was built from.

    Raised only when a caller passes the bytes to be checked. The digest in the
    header is the conversion's own, so writing it beside a different frame would
    make the file's provenance point at something that never produced it.
    """


class ConversionAccountingUnusable(CloudExportNotAllowed):
    """The conversion's own pixel accounting does not add up.

    :func:`~scanner3d.depth_metric.deproject_depth_frame` always produces a
    conversion whose points plus per-reason exclusions equal the frame's sample
    count, so this can only reach here from a record assembled by hand. It is
    refused rather than exported: this module's entire job is carrying a
    conversion's account of itself into a file, and an account that contradicts
    itself cannot be carried honestly.
    """


def ensure_calibration_source_stated(source: str) -> str:
    """Return ``source`` if it says something, else raise."""
    if not source.strip():
        raise CalibrationSourceNotStated(
            "the calibration source is blank; this module will not supply one, because a "
            "DepthConversion does not know whether its intrinsics were read from this unit, "
            "parsed from a stored blob or built by a test. State it, even if the statement is "
            "that no calibration was applied"
        )
    return source


def ensure_intrinsics_digest_usable(digest: str | None) -> str | None:
    """Return ``digest`` if it is ``None`` or a SHA-256 hex digest, else raise."""
    if digest is None:
        return None
    if len(digest) != SHA256_HEX_LENGTH or not set(digest) <= _HEX_DIGITS:
        raise IntrinsicsDigestMalformed(
            f"the intrinsics digest {digest!r} is not {SHA256_HEX_LENGTH} lowercase hex "
            "characters, so it is not a SHA-256 digest a reader could check. Pass None to "
            "record that none is known rather than a string that looks like evidence"
        )
    return digest


def ensure_conversion_accounted(conversion: DepthConversion) -> DepthConversion:
    """Return ``conversion`` if every sample it describes is accounted for, else raise."""
    if not conversion.accounting_holds:
        raise ConversionAccountingUnusable(
            f"the conversion accounts for {conversion.accounted} samples "
            f"({conversion.included} points plus {conversion.excluded} exclusions) of the "
            f"{conversion.width * conversion.height} its own frame holds. A file built from it "
            "would carry an accounting a reader cannot reconcile, so nothing was written"
        )
    return conversion


def ensure_frame_matches(conversion: DepthConversion, raw: bytes) -> bytes:
    """Return ``raw`` if it hashes to the conversion's own frame digest, else raise."""
    digest = sha256_of(raw)
    if digest != conversion.frame_sha256:
        raise SourceFrameMismatch(
            f"the {len(raw)} bytes offered hash to {digest}, but the conversion was built from "
            f"{conversion.frame_sha256} ({conversion.frame_bytes} bytes). The header records the "
            "conversion's digest, so writing it beside these bytes would point the file's "
            "provenance at a frame that did not produce it"
        )
    return raw


# --- translating the convention -----------------------------------------------


def _axis_text(text: str, marker: str) -> str:
    """Drop a leading ``+x``/``+y``/``+z`` marker the target template re-adds.

    :attr:`~scanner3d.pointcloud.CoordinateConvention.summary` renders each axis
    as ``+x {x_axis}``, while depth_metric's strings already begin with their own
    ``+x``. Copying them unchanged would put ``+x +x toward ...`` in the header.
    Only the duplicated marker is removed; every word that says something about
    the axis is kept exactly as written.

    This is the right home for the translation, not a workaround, and the two
    modules should NOT be "aligned" to remove it. Each is internally correct for
    how it renders:

    * :mod:`~scanner3d.depth_metric` exposes ``x_axis`` verbatim through
      ``CoordinateConvention.as_dict()``, where the string is read on its own
      with no template around it. Strip the ``+x`` there and a consumer reading
      the JSON no longer learns which direction is being described.
    * :mod:`~scanner3d.pointcloud` supplies the marker from its ``summary``
      template. Add ``+x`` to the string there and it renders twice.

    Both contracts are satisfiable only if the marker is present in one and
    absent in the other, so something has to translate between them. A seam is
    exactly where that belongs; pushing it into either module would break that
    module's own rendering to spare this function four lines.
    """
    return text.strip().removeprefix(f"{marker} ")


def cloud_convention(convention: AppliedConvention) -> FileConvention:
    """Translate a depth_metric convention into the record the PLY writer takes.

    Faithful, not cosmetic: the axis, origin and handedness strings are
    depth_metric's own words, carried across rather than re-described. The one
    field the target record has and the source does not is ``name``, and it is
    filled by recognition rather than by assumption - anything this module has
    not seen before is named :data:`UNRECOGNISED_FRAME_NAME`, which says only
    that a producer stated a frame.

    What this function *cannot* carry is ``note``: the target record has no
    field for free text and no consumer would find it in one of the axis lines.
    See :func:`convention_caveat`, which is how it reaches the header instead;
    calling this function without also carrying that string is how a header
    ends up claiming an axis convention the device never declared.
    """
    return FileConvention(
        name=CAMERA_FRAME_NAME if convention == CAMERA_FRAME else UNRECOGNISED_FRAME_NAME,
        x_axis=_axis_text(convention.x_axis, "+x"),
        y_axis=_axis_text(convention.y_axis, "+y"),
        z_axis=_axis_text(convention.z_axis, "+z"),
        origin=convention.origin.strip(),
        handedness=convention.handedness.strip(),
    )


def convention_caveat(convention: AppliedConvention) -> str:
    """The convention's ``note``, as the limitation it actually is.

    **This is the decision the module docstring promises to justify.** The note
    on :data:`~scanner3d.depth_metric.CAMERA_FRAME` says that the frame is the
    one depth_metric *applies* and not one the device declared, and that a
    consumer expecting y-up or z-back must transform the points itself. The
    target :class:`~scanner3d.pointcloud.CoordinateConvention` has nowhere to
    put it, so it needs a home, and there are only three candidates:

    * **an axis field.** Refused. ``summary`` builds the one line a consumer
      reads for axis semantics out of those fields; a paragraph inside
      ``origin`` or ``handedness`` corrupts precisely the sentence that has to
      stay parseable.
    * **dropped.** Refused, obviously, and it is the default outcome of any
      field-by-field translation - which is why this function exists rather than
      a dict comprehension.
    * **a limitation.** Taken, and not as a fallback. The note *is* a statement
      of what is not known: that nothing measured on this unit establishes the
      handedness or frame convention of the factory record. ``limitations`` is
      the block :func:`~scanner3d.pointcloud.build_header` renders as its own
      ``comment limitation:`` lines and the block a reader is directed to for
      exactly that class of fact, so this is the semantically right place for
      it, not merely the available one.

    The prefix names the subject, because a limitation block is a list of
    unrelated sentences and this one is about the axes the file just declared.
    """
    return f"coordinate convention: {convention.note}"


# --- translating the unit and the calibration ---------------------------------


def units_label(unit: DepthUnitAssumption, depth_scale: float) -> str:
    """The units comment, built so the assumption cannot read as a measurement.

    All three strings of the assumption travel: the label alone would appear in
    a header with no owner, and the basis is where
    :data:`~scanner3d.depth_metric.UNVALIDATED_UNIT` says out loud that there is
    no basis. The word *assumed* comes first so that a reader who stops at the
    first clause has still not been told a unit was established.
    """
    return (
        f"ASSUMED, never measured: {unit.label}. Stated by: {unit.stated_by}. Basis: "
        f"{unit.basis}. Each coordinate is a raw uint16 depth sample through "
        f"{DEPROJECTION_EXPRESSION} with depth_scale={depth_scale!r}; the scalar's own "
        "physical unit is stated in the calibration comment and in the limitations below"
    )


def _calibration_caveats(conversion: DepthConversion) -> tuple[str, ...]:
    """The observations the conversion made and deliberately did not act on.

    The principal point's placement is always stated, not only when it is odd: a
    reader should not have to notice an *absent* warning to learn where ``x = 0``
    was put. Every check the conversion recorded as failing follows it -
    including ``intrinsics_structurally_expected``, which is the one such
    observation ``DepthConversion.limitations`` does not otherwise carry, so
    without this it would reach no file at all.

    De-duplicated on the detail text before the prefix is added, because the
    failing ``principal_point_near_frame_centre`` check's detail *is* the
    principal point note verbatim, and a header that says the same sentence
    twice teaches a reader to skim the block that matters most.
    """
    details: list[str] = [conversion.principal_point.note]
    for check in conversion.checks:
        if not check.ok and check.detail not in details:
            details.append(check.detail)
    return tuple(f"calibration observation: {detail}" for detail in details)


def cloud_calibration(
    conversion: DepthConversion, *, source: str, intrinsics_sha256: str | None = None
) -> CalibrationProvenance:
    """Where the numbers that turned samples into coordinates came from.

    ``source`` and ``intrinsics_sha256`` are the caller's, because they are the
    only facts here a conversion does not already hold. The scale and every
    caveat are read off the conversion itself, so they cannot be restated
    differently at a call site.
    """
    return CalibrationProvenance(
        source=source,
        intrinsics_sha256=intrinsics_sha256,
        depth_scale=conversion.depth_scale,
        # The default on CalibrationProvenance already says the unit is unknown;
        # this says it in the words scanner-008 established it in, which is the
        # sentence the rest of the repository quotes.
        depth_scale_unit=UNIT_DISCLAIMER,
        caveats=_calibration_caveats(conversion),
    )


def _survival_caveat(conversion: DepthConversion) -> str:
    """What the frame held, what the file carries, and why the gaps are not space."""
    samples = conversion.width * conversion.height
    counts = ", ".join(f"{reason}={count}" for reason, count in conversion.counts_by_reason.items())
    return (
        f"this file carries {conversion.included} points from a "
        f"{conversion.width}x{conversion.height} frame of {samples} samples; "
        f"{conversion.excluded} samples produced no point ({counts}). A PLY can only record "
        "what survived, so the gaps in this cloud are absent measurements rather than observed "
        "empty space, and no surface, hole or edge may be inferred across them"
    )


# --- the seam -----------------------------------------------------------------


def cloud_provenance_for(
    conversion: DepthConversion,
    *,
    calibration_source: str,
    intrinsics_sha256: str | None = None,
    extra_limitations: Sequence[str] = (),
) -> CloudProvenance:
    """Everything :func:`~scanner3d.pointcloud.write_ply` needs, from one conversion.

    This is the seam. A caller that builds a
    :class:`~scanner3d.pointcloud.CloudProvenance` by hand can drop a caveat
    without noticing; a caller that comes through here cannot, because every
    string the conversion carries is collected in one place and the standing
    caveats of :data:`~scanner3d.pointcloud.INHERENT_LIMITATIONS` are added by
    the record itself.

    ``extra_limitations`` is appended rather than substituted, so a caller with
    something of its own to say adds to the uncertainty instead of replacing it.

    Refuses - before returning anything, so before any caller can write a file -
    a blank ``calibration_source``, a malformed ``intrinsics_sha256`` and a
    conversion whose own accounting does not add up.
    """
    ensure_calibration_source_stated(calibration_source)
    ensure_intrinsics_digest_usable(intrinsics_sha256)
    ensure_conversion_accounted(conversion)

    limitations: list[str] = [
        # First, because it is the misreading with the largest consequence: a
        # cloud silently mirrored or rotated relative to what a reader assumed.
        convention_caveat(CAMERA_FRAME),
        _survival_caveat(conversion),
        # Verbatim and in full. Not one of these is this module's to summarise.
        *conversion.limitations,
    ]
    if conversion.extent is None:
        limitations.append(EMPTY_CLOUD_NOTE)
    limitations.extend(extra_limitations)

    return CloudProvenance(
        units=units_label(conversion.unit, conversion.depth_scale),
        convention=cloud_convention(CAMERA_FRAME),
        calibration=cloud_calibration(
            conversion, source=calibration_source, intrinsics_sha256=intrinsics_sha256
        ),
        source_frame_sha256=conversion.frame_sha256,
        limitations=tuple(limitations),
    )


def camera_points(conversion: DepthConversion) -> Iterator[tuple[float, float, float]]:
    """The conversion's points as the ``(x, y, z)`` triples the writer takes.

    In the frame's own raster order, which is the order the conversion built
    them in, so two exports of one conversion are byte-identical files.
    """
    for point in conversion.points:
        yield (point.x, point.y, point.z)


# --- the record the caller gets back ------------------------------------------


@dataclass(frozen=True, slots=True)
class CloudExport:
    """One frame, converted and written, with both halves of the evidence.

    The conversion is kept beside the file record rather than discarded, because
    the accounting that says which samples produced no point is the context the
    vertex count needs, and a caller that has to re-derive it will not.
    """

    conversion: DepthConversion
    cloud: EncodedCloud

    @property
    def path(self) -> str | None:
        """Where the file is, or ``None`` if it was never written."""
        return self.cloud.path

    @property
    def point_count(self) -> int:
        """How many vertices the file carries."""
        return self.cloud.point_count

    @property
    def header(self) -> str:
        """The exact PLY header, comments and all."""
        return self.cloud.header

    @property
    def limitations(self) -> tuple[str, ...]:
        """Every caveat in the header: inherited, converted and caller-stated."""
        return self.cloud.provenance.all_limitations

    @property
    def every_point_written(self) -> bool:
        """Whether the file carries every point the conversion built."""
        return self.cloud.point_count == self.conversion.included

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": EXPORT_MODE,
            "points_converted": self.conversion.included,
            "points_written": self.cloud.point_count,
            "every_point_written": self.every_point_written,
            "conversion": self.conversion.as_dict(),
            "cloud": self.cloud.as_dict(),
            "limitations": list(self.limitations),
        }


# --- the two ways across ------------------------------------------------------


def write_conversion_ply(
    path: Path,
    conversion: DepthConversion,
    *,
    calibration_source: str,
    intrinsics_sha256: str | None = None,
    source_frame: bytes | None = None,
    ply_format: PlyFormat = PlyFormat.BINARY_LITTLE_ENDIAN,
    max_points: int = MAX_POINTS,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    extra_limitations: Sequence[str] = (),
) -> CloudExport:
    """Write one already-converted frame to ``path``, caveats and all.

    ``source_frame`` is optional and is only ever checked: pass the bytes the
    conversion was built from and a mismatch is refused before the path is
    touched, so the digest in the header cannot end up beside a different frame.

    Every bound is :mod:`~scanner3d.pointcloud`'s, unchanged and never widened -
    ``max_points`` and ``max_output_bytes`` are forwarded, and that module
    refuses any value above its own ceiling.
    """
    # Run here as well as inside cloud_provenance_for, and deliberately: it puts
    # the caller's own mistakes ahead of the frame check, and it keeps every
    # entry point in this module independently safe rather than safe by the
    # accident of which helper it happens to call first.
    ensure_calibration_source_stated(calibration_source)
    ensure_intrinsics_digest_usable(intrinsics_sha256)
    ensure_conversion_accounted(conversion)
    if source_frame is not None:
        ensure_frame_matches(conversion, source_frame)

    provenance = cloud_provenance_for(
        conversion,
        calibration_source=calibration_source,
        intrinsics_sha256=intrinsics_sha256,
        extra_limitations=extra_limitations,
    )
    cloud = write_ply(
        path,
        camera_points(conversion),
        provenance=provenance,
        ply_format=ply_format,
        max_points=max_points,
        max_output_bytes=max_output_bytes,
    )
    return CloudExport(conversion=conversion, cloud=cloud)


def export_depth_frame(
    path: Path,
    raw: bytes,
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    unit: DepthUnitAssumption,
    calibration_source: str,
    intrinsics_sha256: str | None = None,
    ply_format: PlyFormat = PlyFormat.BINARY_LITTLE_ENDIAN,
    max_points: int = MAX_POINTS,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    extra_limitations: Sequence[str] = (),
) -> CloudExport:
    """Raw frame bytes in, a PLY on disk, the record of both out.

    The whole path in one call: deproject the frame with the caller's injected
    intrinsics, scale and stated unit, then write the points with every caveat
    the conversion produced. Nothing is defaulted that the caller alone can
    know - the intrinsics, the scale, the assumed unit and where the calibration
    came from are all required.

    The refusals of both modules apply and all of them happen before ``path`` is
    touched: this module's for a blank calibration source or a malformed digest,
    :mod:`~scanner3d.depth_metric`'s for an unstated unit, unusable intrinsics,
    an unusable scale or a buffer that is not exactly one frame, and
    :mod:`~scanner3d.pointcloud`'s for a non-finite coordinate or a file past a
    ceiling.
    """
    # Ahead of the conversion, not after it: a blank source is the caller's
    # mistake and there is no reason to deproject 256000 samples to find out.
    ensure_calibration_source_stated(calibration_source)
    ensure_intrinsics_digest_usable(intrinsics_sha256)

    conversion = deproject_depth_frame(
        raw, intrinsics=intrinsics, depth_scale=depth_scale, unit=unit
    )
    return write_conversion_ply(
        path,
        conversion,
        calibration_source=calibration_source,
        intrinsics_sha256=intrinsics_sha256,
        source_frame=raw,
        ply_format=ply_format,
        max_points=max_points,
        max_output_bytes=max_output_bytes,
        extra_limitations=extra_limitations,
    )
