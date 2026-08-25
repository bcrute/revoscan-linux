"""Deprojecting one raw ``Y16 `` depth frame into camera-frame points.

This module is the first place in the repository where the three things the
earlier tasks recovered separately are put *together*: the 512000-byte frame
:mod:`scanner3d.v4l2_capture` dequeued (scanner-009), the 40-byte factory
``Intrinsics`` record :mod:`scanner3d.intrinsics` read off the device
(scanner-007), and the ``0x918`` scalar :mod:`scanner3d.depth_scale` decoded
(scanner-008). It exists because that combination is exactly where a project
like this normally starts lying to itself - a pinhole deprojection is four
lines of arithmetic, it produces plausible-looking numbers from any inputs at
all, and nothing in the arithmetic knows that the unit of its own output was
never established.

So the arithmetic is the small part of this file. The rest is the bookkeeping
that keeps the result honest:

* **The unit is the caller's assertion, never this module's.** ``raw * scale``
  is a number; what that number *measures* is not established by anything
  measured on this unit (:data:`scanner3d.depth_scale.UNIT_DISCLAIMER`), and no
  default is offered here that would quietly decide it. A
  :class:`DepthUnitAssumption` - a label, who is staking it, and what it rests
  on - is a required argument, and :data:`UNVALIDATED_UNIT` is the one constant
  a caller can reach for that says "I decline to name one" out loud. It is
  carried verbatim into the report, always beside the word *assumed*.
* **Zero is not a distance.** A zero raw sample means the sensor returned no
  depth for that pixel. Multiplying it by the scale would put a point at the
  camera origin, and on the frames captured from this unit the majority of
  every frame is zero, so those points would swamp the cloud they were meant to
  describe. Zeros are excluded, counted, and named.
* **Every pixel is accounted for.** :data:`PIXELS_PER_FRAME` = 256000 samples
  go in; the included points plus the per-reason exclusion counts add back up
  to exactly that, and :attr:`DepthConversion.accounting_holds` says so. An
  excluded sample is never silently dropped, and every exclusion carries a
  named reason rather than a shrug.
* **The suspicious principal point is surfaced, not resolved.** This unit's
  factory ``cx`` sits ~278 px left of the 320 px horizontal centre. That is
  reported with its offsets in every result, because it is the one recovered
  number most likely to make a cloud look subtly wrong, and because nothing
  here can decide whether it describes a cropped sensor window, a different
  field order or a genuinely eccentric optical centre.
* **The coordinate convention is stated in full**, axis by axis, with its
  handedness - and stated as *the convention this module applies*, not as one
  the device declared. The frame convention of the factory record is not
  established by anything measured (``docs/architecture.md``, Domain 4).

Bounds, because a frame is 256000 samples and a conversion should not be able
to surprise its caller with memory: the frame length is exact or the conversion
is refused, the sample tuple is therefore a fixed 256000 entries, and at most
:data:`MAX_POINTS` points are ever built. Every sample past that ceiling is
excluded as :attr:`ExclusionReason.POINT_LIMIT` and counted, so hitting the
bound narrows the result rather than corrupting the accounting.

**No physical measurement has validated any of this.** No distance produced
here has been compared with a ruler, a reference target or the vendor SDK's own
output, no lens distortion is modelled or inverted, and nothing in this module
claims metric accuracy. The full list travels with every result in
:attr:`DepthConversion.limitations` so that a reader of the JSON cannot miss it.
"""

from __future__ import annotations

import enum
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from scanner3d.depth_scale import UNIT_DISCLAIMER
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.intrinsics import CENTRE_TOLERANCE, IntrinsicsRecord, RecordCheck

# Imported across the module boundary on purpose: :mod:`scanner3d.v4l2_capture`
# is where the one depth mode this repository captures is *defined*, and a
# second copy of 640, 400, 2 and 512000 here would be a set of numbers that
# could drift apart from the capture that produces the bytes.
from scanner3d.v4l2_capture import (
    BYTES_PER_PIXEL,
    EXPECTED_FRAME_BYTES,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    sha256_of,
)

#: Every top-level report this module emits stamps this.
SCHEMA_VERSION: Final = 1

#: Samples in one frame. The number every accounting in this module adds to.
PIXELS_PER_FRAME: Final = FRAME_WIDTH * FRAME_HEIGHT

#: The fixed ceiling on points built from one frame. One point per pixel is the
#: most a deprojection can produce, so this is the structural bound rather than
#: a policy: it is not a parameter, is never widened at runtime, and a run that
#: reaches it stops building points and counts the rest as excluded.
MAX_POINTS: Final = PIXELS_PER_FRAME

#: The whole frame, unpacked in one call. The exact-length gate runs first, so
#: this allocation is fixed at :data:`PIXELS_PER_FRAME` samples and cannot be
#: grown by the input.
SAMPLES_FORMAT: Final = f"<{PIXELS_PER_FRAME}H"

#: How the samples sit in the buffer, said once for the report.
SAMPLE_LAYOUT: Final = (
    f"little-endian uint16, tightly packed, {FRAME_WIDTH * BYTES_PER_PIXEL} bytes per row, "
    "row-major from the top-left pixel"
)

#: How many points :meth:`DepthConversion.as_dict` shows. A report is evidence,
#: not a point cloud: the cloud stays on :attr:`DepthConversion.points`, and the
#: dict carries a bounded preview plus the extent so it cannot grow to megabytes.
PREVIEW_POINTS: Final = 8

#: The raw sample value the sensor returns when it has no depth for a pixel.
NO_RETURN_SAMPLE: Final = 0


# --- refusals -----------------------------------------------------------------


class DepthConversionNotAllowed(ScannerError):
    """The conversion was refused before a single sample was read.

    Every subclass is raised by :func:`deproject_depth_frame` up front, so a
    refused conversion has unpacked nothing, built no point and allocated
    nothing beyond the message: a caller that catches one knows the inputs were
    never partially processed.

    The status is :attr:`~scanner3d.errors.DoctorStatus.UNSUPPORTED_MODE`
    because the doctor vocabulary has no class for "the inputs do not describe a
    conversion this module can perform", and that is nearer the truth than
    calling a caller's bad argument an internal error.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class DepthFrameSizeMismatch(DepthConversionNotAllowed):
    """The bytes are not one whole frame of the one mode this module converts.

    Raised before the buffer is unpacked. A short or long buffer is not
    silently padded, truncated or reshaped: the row stride, the sample width
    and the frame size of ``Y16 `` 640x400 are all exact, so a different length
    is a different thing and gets no interpretation here.
    """


class DepthIntrinsicsUnusable(DepthConversionNotAllowed):
    """The injected intrinsics cannot carry a pinhole deprojection.

    Raised before any sample is read. Only what the arithmetic strictly needs
    is checked - the four free entries present, finite, with positive focal
    lengths - because everything else about the record is an *observation* that
    :class:`~scanner3d.intrinsics.IntrinsicsRecord` already carries and that
    this module surfaces rather than enforces.
    """


class DepthScaleUnusable(DepthConversionNotAllowed):
    """The injected depth scale cannot turn a raw code into a distance.

    Raised before any sample is read. A non-finite or non-positive scalar would
    make every distance meaningless in a different way for every sample, so it
    is refused rather than reported per pixel.
    """


class DepthUnitNotStated(DepthConversionNotAllowed):
    """The caller did not state what the converted distances are assumed to be.

    Raised before any sample is read. This module cannot decide the unit - the
    ``0x918`` scalar's physical meaning is not established by anything measured
    on this device - so a blank label, a blank claimant or a blank basis is a
    refusal rather than an invitation for this module to guess.
    """


# --- the unit the caller stakes -----------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthUnitAssumption:
    """What ``raw * scale`` is *assumed* to measure, and on whose authority.

    Three separate strings on purpose. ``label`` alone would let a unit appear
    in a report with no owner; ``stated_by`` names who is answerable for it, and
    ``basis`` forces the reason to be written down - including, for
    :data:`UNVALIDATED_UNIT`, that there is no reason at all. None of the three
    may be blank, and none of them is ever treated as evidence: the report
    always presents them beside :data:`scanner3d.depth_scale.UNIT_DISCLAIMER`.
    """

    label: str
    stated_by: str
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "stated_by": self.stated_by,
            "basis": self.basis,
            "assumed": True,
            "established_by_measurement": False,
            "disclaimer": UNIT_DISCLAIMER,
        }


#: The one unit assumption this module ships: the honest refusal to name a unit.
#: A caller must still pass it explicitly, so declining to name a unit is a
#: decision that appears in the report rather than a default that hides there.
UNVALIDATED_UNIT: Final = DepthUnitAssumption(
    label="unvalidated depth unit",
    stated_by="no one: this module declines to name a unit",
    basis=(
        "none. The 0x918 scalar was read from the device and its reciprocal decoded exactly, "
        "which establishes what the device returned and not what it measures; no physical "
        "measurement has related a converted distance to millimetres, metres or anything else"
    ),
)


def ensure_unit_stated(unit: DepthUnitAssumption) -> DepthUnitAssumption:
    """Return ``unit`` if all three of its strings say something, else raise."""
    for field_name, value in (
        ("label", unit.label),
        ("stated_by", unit.stated_by),
        ("basis", unit.basis),
    ):
        if not value.strip():
            raise DepthUnitNotStated(
                f"the assumed depth unit has a blank {field_name}; this module will not "
                "supply one, because the physical meaning of the depth scale is not "
                f"established by anything measured on this unit. Pass {UNVALIDATED_UNIT.label!r} "
                "via UNVALIDATED_UNIT to decline explicitly"
            )
    return unit


# --- the coordinate frame -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoordinateConvention:
    """The frame deprojected points are expressed in, stated axis by axis.

    Spelled out rather than named, because "camera frame" means at least three
    different things across the graphics, robotics and computer-vision worlds,
    and a point cloud that is silently mirrored or rotated relative to what a
    reader assumed is worse than no point cloud.
    """

    origin: str
    x_axis: str
    y_axis: str
    z_axis: str
    handedness: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "z_axis": self.z_axis,
            "handedness": self.handedness,
            "note": self.note,
        }


#: The convention the deprojection below actually implements. Note what it is
#: not: a statement about the device. The handedness and frame convention of the
#: factory record are open questions (``docs/architecture.md``, Domain 4).
CAMERA_FRAME: Final = CoordinateConvention(
    origin=(
        "the camera's optical centre, i.e. the point that projects to the principal point "
        "(cx, cy); z = 0 there"
    ),
    x_axis=(
        "+x toward increasing column index u (image right), so x = 0 lies on the cx column "
        "and not on the frame's centre column"
    ),
    y_axis="+y toward increasing row index v (image down), so y = 0 lies on the cy row",
    z_axis="+z along the optical axis, away from the camera, into the scene",
    handedness="right-handed: x_hat cross y_hat = z_hat, with x right, y down and z forward",
    note=(
        "this is the convention this module applies - the one implied by "
        "x = (u - cx) * d / fx, y = (v - cy) * d / fy, z = d - and not one the device "
        "declared. Nothing measured on this unit establishes the handedness or frame "
        "convention of the factory record, so a consumer that assumes y-up or z-back must "
        "transform these points itself rather than assume this module already did"
    ),
)

#: The deprojection, written once, so the report and the code cannot disagree.
DEPROJECTION_EXPRESSION: Final = (
    "d = raw * depth_scale; x = (u - cx) * d / fx; y = (v - cy) * d / fy; z = d"
)


# --- exclusions ---------------------------------------------------------------


class ExclusionReason(enum.StrEnum):
    """Why a sample produced no point. Every reason is counted, never dropped.

    Declared in the order :func:`deproject_depth_frame` tests them. A sample
    that would qualify for two is counted once, under the first that applies,
    so the accounting stays a partition of the frame rather than a tally that
    can double-count.
    """

    NO_RETURN = "no_return"
    NON_FINITE_DISTANCE = "non_finite_distance"
    POINT_LIMIT = "point_limit"
    NON_FINITE_COORDINATE = "non_finite_coordinate"


#: What each reason means, in one place so the human and JSON renderings of a
#: run cannot drift apart.
REASON_DETAIL: Final[Mapping[ExclusionReason, str]] = MappingProxyType(
    {
        ExclusionReason.NO_RETURN: (
            f"the raw sample was {NO_RETURN_SAMPLE}, which this repository reads as 'the sensor "
            "returned no depth for this pixel'. It is NOT a measurement of zero distance and is "
            "never converted to a point at the origin. The firmware's own invalid-sample "
            "convention is undocumented, so any other sentinel it may use is not recognised here"
        ),
        ExclusionReason.NON_FINITE_DISTANCE: (
            "raw * depth_scale was not finite, so no coordinate could be computed from it. With "
            "a finite positive scale and a uint16 sample this cannot arise; it is counted rather "
            "than assumed away so that an extreme scale narrows the cloud instead of poisoning it"
        ),
        ExclusionReason.POINT_LIMIT: (
            f"the fixed {MAX_POINTS}-point ceiling had already been reached, so this sample was "
            "counted and not built. The ceiling is one point per pixel and is never widened at "
            "runtime; reaching it makes the result explicitly partial rather than unbounded"
        ),
        ExclusionReason.NON_FINITE_COORDINATE: (
            "the distance was finite but (u - cx) * d / fx or (v - cy) * d / fy overflowed, so "
            "the point has no position. A point carrying an infinity would silently ruin any "
            "extent, centroid or fit computed downstream, so it is excluded and counted instead"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Exclusion:
    """One named reason and how many of the frame's samples it accounts for.

    Every reason appears in every result, with ``count`` zero when it did not
    occur, so a caller can total the accounting without first discovering which
    reasons this particular frame happened to trigger.
    """

    reason: ExclusionReason
    count: int
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"reason": str(self.reason), "count": self.count, "detail": self.detail}


# --- the record the caller gets back ------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthPoint:
    """One deprojected sample: where it came from, and where it landed.

    The pixel and the raw code travel with the coordinates because a point
    whose provenance has been discarded cannot be checked against the frame
    again, and checking is the only thing that will ever validate this module.
    """

    u: int
    v: int
    raw: int
    x: float
    y: float
    z: float

    def as_dict(self) -> dict[str, object]:
        return {"u": self.u, "v": self.v, "raw": self.raw, "x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True, slots=True)
class PointExtent:
    """The axis-aligned span of the points that were built. Numbers, not claims.

    Reported in the caller's assumed unit for x, y and z, and in raw codes for
    the samples, so the two never get conflated in a reader's head.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    min_raw: int
    max_raw: int

    def as_dict(self) -> dict[str, object]:
        return {
            "x": [self.min_x, self.max_x],
            "y": [self.min_y, self.max_y],
            "z": [self.min_z, self.max_z],
            "raw": [self.min_raw, self.max_raw],
            "xyz_unit": "the caller's assumed unit; see the 'unit' block",
            "raw_unit": "uint16 sensor codes, unconverted",
        }


@dataclass(frozen=True, slots=True)
class PrincipalPointNote:
    """Where the injected principal point sits, and what that does to the cloud.

    Present in every result, not only when it is odd: a reader should not have
    to notice an *absent* warning to learn where x = 0 was placed.
    """

    cx: float
    cy: float
    offset_u: float
    offset_v: float
    centre_u: float
    centre_v: float
    near_centre: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "frame_centre_u": self.centre_u,
            "frame_centre_v": self.centre_v,
            "offset_from_centre_u_px": self.offset_u,
            "offset_from_centre_v_px": self.offset_v,
            "centre_tolerance_fraction": CENTRE_TOLERANCE,
            "near_frame_centre": self.near_centre,
            "note": self.note,
        }


#: The limitations that hold for every conversion this module performs,
#: whatever the inputs were. Per-run limitations - a principal point far off
#: centre, an intrinsics record describing a different raster, a truncated
#: cloud - are appended to these by :func:`deproject_depth_frame`.
STATIC_LIMITATIONS: Final[tuple[str, ...]] = (
    f"depth unit: {UNIT_DISCLAIMER}",
    "no physical measurement has validated any distance produced here: nothing in this "
    "repository has compared a converted distance against a ruler, a reference target or the "
    "vendor SDK's own output, so no metric accuracy is claimed at any magnitude",
    "no lens distortion is modelled: the pinhole deprojection assumes an ideal lens, this "
    "unit's sibling 'Distort' calibration key has never been read, and no distortion is "
    "applied or inverted here",
    "the coordinate convention reported alongside these points is the one this module "
    "applies; the handedness and frame convention of the device's own factory record are "
    "not established by anything measured on this unit",
    "the raw-to-distance step is one multiplication and nothing else: no temporal filtering, "
    "no hole filling, no smoothing, no outlier rejection and no plane fitting was applied",
    "zero samples are read as 'no depth returned' by this repository's own reading of the "
    "frames; the firmware's invalid-sample convention is undocumented and no other sentinel "
    "value is recognised",
)


@dataclass(frozen=True, slots=True)
class DepthConversion:
    """One frame, deprojected, with everything needed to disbelieve the result.

    The points are the small part. ``exclusions`` accounts for every sample that
    produced none, ``checks`` records the observations that were made and not
    acted on, and ``limitations`` carries the standing uncertainties into
    whatever the caller renders - so a consumer that reads only
    :meth:`as_dict` still cannot come away believing these are validated metric
    coordinates.
    """

    frame_sha256: str
    frame_bytes: int
    width: int
    height: int
    intrinsics: IntrinsicsRecord
    depth_scale: float
    unit: DepthUnitAssumption
    points: tuple[DepthPoint, ...]
    exclusions: tuple[Exclusion, ...]
    checks: tuple[RecordCheck, ...]
    principal_point: PrincipalPointNote
    extent: PointExtent | None
    limitations: tuple[str, ...]

    @property
    def included(self) -> int:
        """How many samples became points."""
        return len(self.points)

    @property
    def excluded(self) -> int:
        """How many samples did not, across every named reason."""
        return sum(exclusion.count for exclusion in self.exclusions)

    @property
    def accounted(self) -> int:
        """Included plus excluded: what this result says it looked at."""
        return self.included + self.excluded

    @property
    def accounting_holds(self) -> bool:
        """Whether every one of the frame's samples is accounted for exactly once."""
        return self.accounted == self.width * self.height

    @property
    def counts_by_reason(self) -> Mapping[str, int]:
        """The per-reason exclusion counts, every reason present."""
        return MappingProxyType({str(e.reason): e.count for e in self.exclusions})

    @property
    def truncated(self) -> bool:
        """Whether the fixed point ceiling cut the cloud short."""
        return self.counts_by_reason[str(ExclusionReason.POINT_LIMIT)] > 0

    def count_of(self, reason: ExclusionReason) -> int:
        """How many samples were excluded for exactly ``reason``."""
        return self.counts_by_reason[str(reason)]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "depth-frame-to-camera-points",
            "source_frame": {
                "fourcc": FOURCC,
                "width": self.width,
                "height": self.height,
                "bytes": self.frame_bytes,
                "expected_bytes": EXPECTED_FRAME_BYTES,
                "sha256": self.frame_sha256,
                "sample_layout": SAMPLE_LAYOUT,
            },
            "intrinsics": self.intrinsics.as_dict(),
            "principal_point": self.principal_point.as_dict(),
            "depth_scale": {
                "value": self.depth_scale,
                "applied_as": DEPROJECTION_EXPRESSION,
                "unit": UNIT_DISCLAIMER,
            },
            "unit": self.unit.as_dict(),
            "coordinate_convention": CAMERA_FRAME.as_dict(),
            "pixel_accounting": {
                "samples": self.width * self.height,
                "included": self.included,
                "excluded": self.excluded,
                "accounted": self.accounted,
                "accounting_holds": self.accounting_holds,
                "counts_by_reason": dict(self.counts_by_reason),
                "exclusions": [exclusion.as_dict() for exclusion in self.exclusions],
            },
            "bounds": {
                "max_points": MAX_POINTS,
                "expected_frame_bytes": EXPECTED_FRAME_BYTES,
                "truncated": self.truncated,
                "preview_points": PREVIEW_POINTS,
            },
            "checks": [check.as_dict() for check in self.checks],
            "extent": None if self.extent is None else self.extent.as_dict(),
            "points": {
                "count": self.included,
                "preview": [point.as_dict() for point in self.points[:PREVIEW_POINTS]],
                "note": (
                    f"the full cloud is on DepthConversion.points; at most {PREVIEW_POINTS} "
                    "points are shown here so a report stays a report"
                ),
            },
            "limitations": list(self.limitations),
        }


# --- the conversion -----------------------------------------------------------


def _required_value(values: Mapping[str, float], name: str) -> float:
    """One intrinsics entry the arithmetic cannot proceed without."""
    value = values.get(name)
    if value is None:
        raise DepthIntrinsicsUnusable(
            f"the injected intrinsics record has no {name!r} entry, so the pinhole "
            f"deprojection {DEPROJECTION_EXPRESSION} cannot be evaluated"
        )
    if not math.isfinite(value):
        raise DepthIntrinsicsUnusable(
            f"the injected intrinsics record has {name}={value!r}, which is not finite; "
            "no coordinate computed from it would mean anything"
        )
    return value


def ensure_intrinsics_usable(intrinsics: IntrinsicsRecord) -> tuple[float, float, float, float]:
    """Return ``(fx, fy, cx, cy)`` if the deprojection can use them, else raise.

    Only what the arithmetic strictly needs is enforced. Everything else the
    record says about itself - the fixed 0/0/0/0/1 entries, the frame it names,
    how far off centre its principal point sits - is an observation that
    :class:`~scanner3d.intrinsics.IntrinsicsRecord` already carries and that
    this module reports rather than polices: refusing a factory record for
    being surprising would be deciding a question this project has not answered.
    """
    fx = _required_value(intrinsics.values, "fx")
    fy = _required_value(intrinsics.values, "fy")
    cx = _required_value(intrinsics.values, "cx")
    cy = _required_value(intrinsics.values, "cy")
    if fx <= 0.0 or fy <= 0.0:
        raise DepthIntrinsicsUnusable(
            f"the injected intrinsics record has fx={fx!r}, fy={fy!r}; a pinhole "
            "deprojection divides by both, so neither may be zero or negative"
        )
    return fx, fy, cx, cy


def ensure_scale_usable(depth_scale: float) -> float:
    """Return ``depth_scale`` if it can turn a raw code into a distance, else raise."""
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise DepthScaleUnusable(
            f"the injected depth scale is {depth_scale!r}; a raw uint16 code multiplied by it "
            "would not be a usable distance. Only a finite, positive scalar is accepted, and "
            "this module still makes no claim about what the product measures"
        )
    return depth_scale


def ensure_frame_length(raw: bytes) -> bytes:
    """Return ``raw`` if it is exactly one frame of the one mode, else raise.

    Raised before the buffer is unpacked, so a rejected buffer is never
    partially read, reshaped or padded into something that looks like a frame.
    """
    if len(raw) != EXPECTED_FRAME_BYTES:
        raise DepthFrameSizeMismatch(
            f"a {FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT} frame is exactly "
            f"{EXPECTED_FRAME_BYTES} bytes ({SAMPLE_LAYOUT}); got {len(raw)}. Nothing was "
            "unpacked: a buffer of another length is not this mode's frame, and padding or "
            "truncating it here would invent pixels the device never sent"
        )
    return raw


def _principal_point_note(cx: float, cy: float, width: int, height: int) -> PrincipalPointNote:
    """Where x = 0 and y = 0 were placed, in the frame's own pixels."""
    centre_u = width / 2
    centre_v = height / 2
    offset_u = cx - centre_u
    offset_v = cy - centre_v
    near = abs(offset_u) <= width * CENTRE_TOLERANCE and abs(offset_v) <= height * CENTRE_TOLERANCE
    return PrincipalPointNote(
        cx=cx,
        cy=cy,
        offset_u=offset_u,
        offset_v=offset_v,
        centre_u=centre_u,
        centre_v=centre_v,
        near_centre=near,
        note=(
            f"x = 0 lies on column {cx:g} and y = 0 on row {cy:g} of the {width}x{height} "
            f"frame, i.e. {offset_u:+.2f} px and {offset_v:+.2f} px from its centre. The "
            "values are used exactly as they arrived; nothing here recentres them"
        ),
    )


def _off_centre_limitation(note: PrincipalPointNote, width: int, height: int) -> str:
    """The dynamic limitation an off-centre principal point earns."""
    return (
        f"the injected principal point sits {note.offset_u:+.2f} px horizontally and "
        f"{note.offset_v:+.2f} px vertically from the centre of the {width}x{height} frame, "
        f"outside the {CENTRE_TOLERANCE:.0%} observation tolerance. Every x and y in this "
        "result is measured from that point, so the cloud is correspondingly asymmetric. This is "
        "surfaced, not corrected: a cropped or offset sensor window, a different intended "
        "field order and a genuinely eccentric optical centre all remain open readings, and "
        "no measurement here distinguishes them"
    )


def _extent_of(points: Sequence[DepthPoint]) -> PointExtent | None:
    """The span of the built points, or ``None`` when none were built."""
    if not points:
        return None
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    raws = [point.raw for point in points]
    return PointExtent(
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
        min_z=min(zs),
        max_z=max(zs),
        min_raw=min(raws),
        max_raw=max(raws),
    )


def _conversion_checks(
    intrinsics: IntrinsicsRecord, note: PrincipalPointNote
) -> tuple[RecordCheck, ...]:
    """The observations this conversion makes and deliberately does not act on."""
    return (
        RecordCheck(
            name="intrinsics_frame_matches_depth_frame",
            ok=(intrinsics.width, intrinsics.height) == (FRAME_WIDTH, FRAME_HEIGHT),
            detail=(
                f"the injected record names a {intrinsics.width}x{intrinsics.height} frame and "
                f"the samples are {FRAME_WIDTH}x{FRAME_HEIGHT}. fx, fy, cx and cy are in pixels "
                "of the record's own frame, so a mismatch means the pixel coordinates fed to "
                "the deprojection are not the ones these values were measured in. The "
                "conversion is performed as asked and the mismatch is reported, because "
                "rescaling a factory record here would be inventing a calibration"
            ),
        ),
        RecordCheck(
            name="principal_point_near_frame_centre",
            ok=note.near_centre,
            detail=note.note,
        ),
        RecordCheck(
            name="intrinsics_structurally_expected",
            ok=intrinsics.structurally_expected,
            detail=(
                "whether the record's own structural checks - finiteness and the five entries "
                "the vendor layout fixes at 0/0/0/0/1 - all passed. A record that diverges is "
                "still converted if fx, fy, cx and cy are usable; the divergence is reported "
                "so the result is not read as calibrated"
            ),
        ),
    )


def deproject_depth_frame(
    raw: bytes,
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
    unit: DepthUnitAssumption,
) -> DepthConversion:
    """Deproject one raw ``Y16 `` 640x400 frame into camera-frame points.

    ``intrinsics`` is injected, always: this module reads no device, opens no
    node and carries no unit's numbers as a default, so the caller decides which
    calibration is being applied and the report records which one it was.
    ``unit`` is likewise the caller's own assertion about what
    ``raw * depth_scale`` measures - pass :data:`UNVALIDATED_UNIT` to decline
    explicitly rather than to skip the question.

    Every refusal is raised **before a single sample is read**, so a rejected
    call has unpacked nothing and built nothing:
    :class:`DepthUnitNotStated` for a blank unit assumption,
    :class:`DepthIntrinsicsUnusable` for intrinsics the pinhole model cannot
    use, :class:`DepthScaleUnusable` for a scale that cannot produce a
    distance, and :class:`DepthFrameSizeMismatch` for a buffer that is not
    exactly one frame of this mode.

    Zero samples are excluded as "no depth returned" and counted, never
    converted to points at the origin, and the returned
    :class:`DepthConversion` accounts for all 256000 samples by name.
    """
    ensure_unit_stated(unit)
    fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
    scale = ensure_scale_usable(depth_scale)
    ensure_frame_length(raw)

    # The length gate above fixes this allocation at exactly PIXELS_PER_FRAME
    # samples: it cannot be grown, shrunk or shaped by the input.
    samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)

    points: list[DepthPoint] = []
    no_return = 0
    non_finite_distance = 0
    non_finite_coordinate = 0
    over_limit = 0
    for index, sample in enumerate(samples):
        if sample == NO_RETURN_SAMPLE:
            no_return += 1
            continue
        distance = sample * scale
        if not math.isfinite(distance):
            non_finite_distance += 1
            continue
        if len(points) >= MAX_POINTS:
            # Counted rather than dropped, and counted for *every* remaining
            # eligible sample, so hitting the ceiling leaves the accounting
            # exact and the result explicitly partial.
            over_limit += 1
            continue
        row, column = divmod(index, FRAME_WIDTH)
        x = (column - cx) * distance / fx
        y = (row - cy) * distance / fy
        if not (math.isfinite(x) and math.isfinite(y)):
            non_finite_coordinate += 1
            continue
        points.append(DepthPoint(u=column, v=row, raw=sample, x=x, y=y, z=distance))

    note = _principal_point_note(cx, cy, FRAME_WIDTH, FRAME_HEIGHT)
    limitations = list(STATIC_LIMITATIONS)
    if not note.near_centre:
        limitations.append(_off_centre_limitation(note, FRAME_WIDTH, FRAME_HEIGHT))
    if (intrinsics.width, intrinsics.height) != (FRAME_WIDTH, FRAME_HEIGHT):
        limitations.append(
            f"the injected intrinsics name a {intrinsics.width}x{intrinsics.height} frame while "
            f"the samples are {FRAME_WIDTH}x{FRAME_HEIGHT}; the pixel coordinates deprojected "
            "here are therefore not the ones fx, fy, cx and cy were measured in, and no "
            "rescaling was applied"
        )
    if over_limit:
        limitations.append(
            f"the cloud is partial: the fixed {MAX_POINTS}-point ceiling was reached and "
            f"{over_limit} further samples were counted rather than built"
        )

    return DepthConversion(
        frame_sha256=sha256_of(raw),
        frame_bytes=len(raw),
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        intrinsics=intrinsics,
        depth_scale=scale,
        unit=unit,
        points=tuple(points),
        exclusions=(
            Exclusion(
                reason=ExclusionReason.NO_RETURN,
                count=no_return,
                detail=REASON_DETAIL[ExclusionReason.NO_RETURN],
            ),
            Exclusion(
                reason=ExclusionReason.NON_FINITE_DISTANCE,
                count=non_finite_distance,
                detail=REASON_DETAIL[ExclusionReason.NON_FINITE_DISTANCE],
            ),
            Exclusion(
                reason=ExclusionReason.POINT_LIMIT,
                count=over_limit,
                detail=REASON_DETAIL[ExclusionReason.POINT_LIMIT],
            ),
            Exclusion(
                reason=ExclusionReason.NON_FINITE_COORDINATE,
                count=non_finite_coordinate,
                detail=REASON_DETAIL[ExclusionReason.NON_FINITE_COORDINATE],
            ),
        ),
        checks=_conversion_checks(intrinsics, note),
        principal_point=note,
        extent=_extent_of(points),
        limitations=tuple(limitations),
    )
