"""The one path from a raw depth frame to the cloud that gets registered.

Why this module exists
----------------------
There were two of these. The window built its registration clouds in
``scanner3d.gui._cloud_from_bytes`` and the CLI built them again inside
``scanner3d.cli._cmd_reconstruct``, with the same four steps written out twice
and a comment in each pointing at the other asking the reader to check they
still agreed. They did agree, for a while. That is not a property anyone can
maintain, and the cost of them drifting is the worst kind: two entry points
that reconstruct the SAME captured frames into two different clouds, each
reporting its own numbers as if they described the capture rather than the
program that read it.

So the steps live here once, and both entry points call
:func:`prepare_frame`. There is no second copy to keep in step, and a record
written by either one says which policy version produced it.

The order, and why it is this order
-----------------------------------
:data:`PREPARATION_ORDER`, stated once so it can be quoted into a report:

1. **deproject** - :func:`scanner3d.depth_metric.deproject_depth_frame` turns
   the Y16 bytes into camera-frame coordinates. It is not a filter and removes
   nothing except the samples the sensor itself marked absent.
2. **depth clip** - keep :data:`DEPTH_CLIP_NEAR` to :data:`DEPTH_CLIP_FAR`.
   Outside that interval this scanner is returning the far wall and its own
   housing, not the object.
3. **deterministic thinning** - every ``step``-th point, where ``step`` makes
   the count about ``target_points``. A stride rather than a sample: no seed,
   no shuffle, and the same frame thins to the same points on every machine.
4. **speckle filtering** - :func:`scanner3d.denoise.strip_speckle`, AFTER the
   thinning and not before. Filtering the full frame first costs about fifty
   times as much on the thread that owns the device, and a point that has
   neighbours at full density can be alone once the frame is thinned - which is
   the cloud that is actually registered.
5. **support-plane filtering** - :func:`scanner3d.support_plane.suppress_support_plane`,
   last, because it fits a plane and a cloud full of strays fits a worse one.
   It removes the turntable or bed when the frame proves one is there, and
   returns the cloud untouched with a recorded reason when it does not.

Two artifacts, not one
----------------------
A reconstruction builds two things out of one capture, by two different routes.
The REGISTERED cloud comes from the decimated per-frame clouds this path
produces. The exported MESH does not read those clouds at all: it is fused from
the FULL-RESOLUTION frames at the poses registration accepted. A filter that
runs only here therefore fixes the poses and leaves the mesh alone, which for a
support plane means the bed comes out of the registration and stays in the PLY.

So :func:`mask_support_band` applies the same plane and the same band a second
time, to an in-memory copy of the frame that is about to be fused, for the
frames the filter APPLIED to. Speckle rejection is deliberately NOT carried
across: it ran on the decimated cloud and its exchange rate was measured there,
and claiming it of the fused frames would be a claim about work that did not
happen. Every count either way is kept in its own block of
:meth:`PreparationSummary.as_dict`, so neither artifact's numbers can be read
into the other's file.

Truthfully empty, truthfully withdrawn
--------------------------------------
A frame that catches nothing is a frame, not a refusal: deprojection legitimately
returns nothing when the scanner is pointed at air, and every caller here already
treats an empty cloud as "this one contributed nothing". Such a frame stops after
the thinning and carries a record that says both filters were not run - which is
a different fact from either of them having run and removed nothing.

Neither filter can silently fail. Each either acts, or withdraws with a reason
on its record, or raises its own typed error - and this module raises those
onwards rather than substituting a cloud of its own.

Points that arrive already deprojected are gated before anything classifies
them - see :class:`PointsUnusable`. A malformed or non-finite coordinate is a
typed refusal here rather than an :class:`IndexError` from inside a
comprehension or, worse, a point that silently failed the depth clip and left
no trace of having been dropped for being unmeasurable.

Nothing here writes a file, opens a device or touches a captured frame. The raw
bytes are read; the points and the masked frames derived from them are what is
filtered. The capture on disk is unchanged and complete.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from scanner3d.denoise import FilteredCloud, strip_speckle
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point
from scanner3d.support_plane import (
    POLICY_VERSION,
    REASON_APPLIED,
    SupportPlaneResult,
    suppress_support_plane,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scanner3d.depth_metric import DepthUnitAssumption
    from scanner3d.intrinsics import IntrinsicsRecord

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: The identity a record carries so two clouds can be compared for having been
#: prepared the same way. Bumped whenever the ORDER or the steps change;
#: :data:`scanner3d.support_plane.POLICY_VERSION` moves independently for the
#: plane filter's own thresholds.
PREPARATION_VERSION: Final = 1

#: The steps, in the order they run, for quoting into a report rather than
#: re-describing at each call site.
PREPARATION_ORDER: Final[tuple[str, ...]] = (
    "deproject",
    "depth clip",
    "deterministic thinning",
    "speckle filtering",
    "support-plane filtering",
)

#: The depth interval kept, in the coordinate scale deprojection produces.
#: Nearer than the first is the scanner's own housing and the projector's
#: overexposed near field; further than the second is whatever the room put
#: behind the object.
DEPTH_CLIP_NEAR: Final = 60.0
DEPTH_CLIP_FAR: Final = 300.0

#: Points kept per frame when a capture is read back off disk to be registered.
#:
#: A frame holds 256000 samples and returns tens of thousands of points; all of
#: them through pairwise ICP for a sweep of hundreds of frames is hours, so
#: every frame is thinned to about this many. Both the window's Reconstruct
#: action and ``scanner3d reconstruct`` use this one constant - they used to use
#: two matching literals - and it is stated in the exported record.
RECONSTRUCT_POINTS: Final = 2200


def preparation_unit() -> DepthUnitAssumption:
    """What ``raw * scale`` is assumed to measure, on whose authority.

    One function rather than one literal per entry point. This assumption now
    reaches the deprojection here and the provenance written into every
    exported file, and a second copy of these three strings would be a second
    claim about the same measurement, free to drift from the first without
    anything noticing.
    """
    from scanner3d.depth_metric import DepthUnitAssumption

    return DepthUnitAssumption(
        label="mm",
        stated_by="caliper measurement recorded in docs/metric-validation.md",
        basis="a 58.0 mm can measured 57.705 mm at about 122.7 mm standoff",
    )


class PreparationNotAllowed(ScannerError):
    """A preparation was requested that this module refuses to run.

    Raised before a frame is deprojected, so a refused call has read nothing and
    written nothing. The FILTERS in this path do not raise for ordinary input -
    each withdraws with a reason instead - so this is about the caller's
    request, not about the scene.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class TargetPointsUnusable(PreparationNotAllowed):
    """The thinning target is not a positive whole count of points."""


class PointsUnusable(PreparationNotAllowed):
    """An already-deprojected point offered to this module is not usable.

    Raised for a cloud that is not a sequence, an entry that is not an
    ``(x, y, z)`` triple, and a coordinate that is not finite - before the
    depth clip classifies anything.

    Checked here rather than left to the arithmetic because every one of those
    used to escape as something else, and each of the three was worse than a
    refusal. A two-element point raised :class:`IndexError` from inside a
    comprehension; a point holding a string raised :class:`TypeError` from
    :func:`float`; and a NaN coordinate SILENTLY compared false against both
    ends of the clip, so the frame simply came out shorter with nothing on the
    record to say a point had been dropped for being unmeasurable rather than
    for being out of range.
    """


def ensure_points_preparable(points: Sequence[Point]) -> list[tuple[float, float, float]]:
    """Every offered point as a finite triple, refusing the first one that is not.

    The public boundary for :func:`prepare_points` and
    :meth:`PreparedCloud.of_points`: both take coordinates the caller
    deprojected somewhere else, so neither can rely on
    :mod:`scanner3d.depth_metric` having already gated them.
    """
    # Through ``object`` deliberately: the annotation promises a sequence and
    # this check exists for the callers that do not honour it, which is a
    # runtime fact a type checker is entitled to think impossible.
    offered: object = points
    if isinstance(offered, str | bytes) or not isinstance(offered, Sequence):
        raise PointsUnusable(
            f"points must be a sequence of (x, y, z) triples, got {type(points).__name__}; "
            "a stream or a scalar cannot be clipped, thinned or counted, and consuming one to "
            "find that out would leave the caller's own iterator half read"
        )
    checked: list[tuple[float, float, float]] = []
    for index, point in enumerate(points):
        if isinstance(point, str | bytes) or not isinstance(point, Sequence) or len(point) != 3:
            raise PointsUnusable(
                f"point {index} is {point!r}, which is not an (x, y, z) triple; nothing was "
                "clipped, thinned or filtered"
            )
        try:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
        except (TypeError, ValueError) as exc:
            raise PointsUnusable(
                f"point {index} is {point!r}, whose coordinates are not numbers: {exc}"
            ) from exc
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise PointsUnusable(
                f"point {index} is {point!r}, which is not finite. Refused rather than clipped: "
                "a NaN coordinate compares false against BOTH ends of the depth clip, so it "
                "would vanish from the cloud as though it had been out of range"
            )
        checked.append((x, y, z))
    return checked


def ensure_target_points(target_points: int) -> int:
    """Refuse a thinning target that is not a count, and return it unchanged."""
    if isinstance(target_points, bool) or not isinstance(target_points, int):
        raise TargetPointsUnusable(
            f"target_points must be an int, got {target_points!r}; a fractional or boolean "
            "number of points is a caller's mistake, not a rounding decision"
        )
    if target_points < 1:
        raise TargetPointsUnusable(
            f"target_points {target_points} is not a positive count; a frame thinned to "
            "nothing is not a decimation of the frame, and an empty cloud downstream is "
            "indistinguishable from a frame that caught nothing"
        )
    return target_points


@dataclass(frozen=True, slots=True)
class PreparedCloud:
    """One registration cloud and everything that happened on the way to it.

    :attr:`points` is what a caller that only wants the cloud reads; the rest is
    the evidence a report carries. ``speckle`` and ``support`` are ``None`` only
    when the frame was already empty before either could run - which is stated
    on the record rather than left to be inferred from a count of zero.
    """

    points: tuple[Point, ...]
    target_points: int
    deprojected_points: int
    clipped_points: int
    thinned_points: int
    speckle: FilteredCloud | None
    support: SupportPlaneResult | None

    @classmethod
    def of_points(cls, points: Sequence[Point], *, target_points: int = 0) -> PreparedCloud:
        """A record for a cloud that arrived already prepared, claiming nothing.

        For a caller - a test harness standing in for deprojection, a consumer
        handed points from somewhere else - that has a cloud but no preparation
        to describe. Every count reflects the points given and both filter
        records are absent, which says "this did not run here" rather than
        "this ran and removed nothing".
        """
        held = tuple(ensure_points_preparable(points))
        return cls(
            points=held,
            target_points=target_points,
            deprojected_points=len(held),
            clipped_points=len(held),
            thinned_points=len(held),
            speckle=None,
            support=None,
        )

    @property
    def kept_points(self) -> int:
        """How many points came out."""
        return len(self.points)

    @property
    def speckle_removed(self) -> int:
        """Points dropped for having too few neighbours of their own frame."""
        return 0 if self.speckle is None else self.speckle.removed_points

    @property
    def support_applied(self) -> bool:
        """Whether a support plane was proved and removed from this frame."""
        return self.support is not None and self.support.applied

    @property
    def support_removed(self) -> int:
        """Points dropped for lying in the proved support plane's band."""
        return 0 if self.support is None else self.support.removed_points

    @property
    def support_reason(self) -> str:
        """Why the support filter acted, or which gate stopped it."""
        return "not-run" if self.support is None else self.support.reason

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "preparation_version": PREPARATION_VERSION,
            "support_plane_policy_version": POLICY_VERSION,
            "order": list(PREPARATION_ORDER),
            "target_points": self.target_points,
            "depth_clip": {"near": DEPTH_CLIP_NEAR, "far": DEPTH_CLIP_FAR},
            "counts": {
                "deprojected": self.deprojected_points,
                "after_depth_clip": self.clipped_points,
                "after_thinning": self.thinned_points,
                "kept": self.kept_points,
            },
            "speckle": None if self.speckle is None else self.speckle.as_dict(),
            "support_plane": None if self.support is None else self.support.as_dict(),
        }


def prepare_points(
    points: Sequence[Point],
    *,
    target_points: int,
    deprojected_points: int | None = None,
) -> PreparedCloud:
    """Clip, thin, denoise and support-plane filter an already deprojected cloud.

    Split from :func:`prepare_frame` so a caller holding camera-frame points -
    a test, a simulated sweep - runs exactly the arithmetic a captured frame
    runs, without a fake depth buffer standing in for the only step it is
    missing.
    """
    ensure_target_points(target_points)
    # Validated in full BEFORE the clip classifies anything, so a malformed or
    # non-finite point is a typed refusal rather than a point that quietly did
    # not survive the comparison. See :class:`PointsUnusable`.
    offered = ensure_points_preparable(points)
    clipped = [p for p in offered if DEPTH_CLIP_NEAR < p[2] < DEPTH_CLIP_FAR]
    step = max(1, len(clipped) // target_points)
    thinned = clipped[::step]
    counts = {
        "target_points": target_points,
        "deprojected_points": len(offered) if deprojected_points is None else deprojected_points,
        "clipped_points": len(clipped),
        "thinned_points": len(thinned),
    }
    if not thinned:
        # An empty frame is a frame. Both filters refuse an empty cloud rather
        # than answering one, and a refusal here would turn a scanner pointed at
        # air into a failed capture.
        return PreparedCloud(points=(), speckle=None, support=None, **counts)
    speckle = strip_speckle(thinned)
    if not speckle.points:
        return PreparedCloud(points=(), speckle=speckle, support=None, **counts)
    support = suppress_support_plane(speckle.points)
    return PreparedCloud(points=support.points, speckle=speckle, support=support, **counts)


def prepare_frame(raw: bytes, *, target_points: int) -> PreparedCloud:
    """Deproject one raw depth frame and prepare it for registration.

    Takes bytes rather than a path because the frames a resumed capture holds
    TENTATIVELY have been delivered and selected but must not be written as
    ``frame_*.raw`` until the relocalization that would justify them has been
    accepted - so for those there is no file to read back from. The arithmetic
    is the same arithmetic either way, which is the point of it being one
    function.
    """
    ensure_target_points(target_points)

    from scanner3d.cli import DEPTH_SCALE_METRES
    from scanner3d.depth_metric import deproject_depth_frame
    from scanner3d.simulate import FACTORY_INTRINSICS

    conversion = deproject_depth_frame(
        raw,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
        unit=preparation_unit(),
    )
    points = [(p.x, p.y, p.z) for p in conversion.points]
    return prepare_points(points, target_points=target_points)


# --- the same plane, at full resolution ---------------------------------------

#: Serialised shape version for every fusion-masking record below. Separate
#: from :data:`SCHEMA_VERSION` because this is a different claim about a
#: different artifact, and a consumer that only reads one of them should not
#: have to track the other's revisions.
FUSION_MASK_SCHEMA_VERSION: Final = 1


class FusionFrameUnusable(PreparationNotAllowed):
    """A raw frame or its calibration cannot be masked for fusion.

    Raised before a sample is read, so a refused call has built no derived
    frame and the caller still holds the original bytes.
    """


@dataclass(frozen=True, slots=True)
class FusionFrame:
    """One frame as it will be FUSED, and whether anything was taken out of it.

    Why this exists
    ---------------
    :func:`prepare_frame` produces the decimated cloud that gets REGISTERED.
    The exported mesh is a different artifact built a different way: it is
    fused from the full-resolution frames at the poses registration accepted,
    which means a support plane removed from the registration cloud was still
    being integrated into the surface the operator opens. The proved bed came
    out of the poses and stayed in the mesh.

    So the same plane, the same band and the same decision are applied a second
    time - to the frame's own samples, at full resolution, in memory. The
    masked samples are set to :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`,
    which is the value the sensor itself returns for a pixel it had no depth
    for and which the fusion already skips and counts. Nothing else about the
    frame is touched: no crop, no smoothing, no resampling, and the FUSION'S
    ARITHMETIC is unchanged - it is given a frame with fewer returns in it, not
    a different algorithm.

    :attr:`raw` is the ORIGINAL object, not a copy, whenever nothing was
    masked. A withdrawn frame, a frame the filter never ran on, and a frame
    whose applied band happened to catch no full-resolution sample are all
    fused byte for byte as captured.

    Nothing here writes, moves or rewrites a file. The capture on disk is
    untouched; this is a derived frame that exists for the length of one
    export.
    """

    raw: bytes
    applied: bool
    reason: str
    samples_returned: int
    samples_masked: int

    @property
    def samples_fused(self) -> int:
        """Returns left for the fusion to project."""
        return self.samples_returned - self.samples_masked

    @property
    def masked(self) -> bool:
        """Whether this frame's bytes differ from the ones that were captured."""
        return self.samples_masked > 0


def mask_support_band(
    raw: bytes,
    support: SupportPlaneResult | None,
    *,
    intrinsics: IntrinsicsRecord,
    depth_scale: float,
) -> FusionFrame:
    """The frame to fuse: the captured one, minus an APPLIED support band.

    ``support`` is the record :func:`prepare_frame` produced for THIS frame.
    When it is absent, withdrew, or carries no plane, the frame is returned
    unchanged and says so - a filter that did not act on the registration cloud
    must not act on the fusion input either, or the two halves of one export
    would disagree about what the frame contained.

    ``intrinsics`` and ``depth_scale`` must be the ones the frame will be FUSED
    under. The band is a region of the camera's own space, so a sample's
    membership is decided by deprojecting it with the same pinhole arithmetic
    :func:`scanner3d.depth_metric.deproject_depth_frame` uses - the same
    expressions in the same order, so a full-resolution sample at the position
    of a decimated point is classified the way that point was.

    Raises :class:`FusionFrameUnusable` - before a sample is read - for a frame
    of the wrong length or a calibration the pinhole model cannot use.
    """
    from scanner3d.depth_metric import (
        NO_RETURN_SAMPLE,
        SAMPLES_FORMAT,
        DepthConversionNotAllowed,
        ensure_frame_length,
        ensure_intrinsics_usable,
        ensure_scale_usable,
    )
    from scanner3d.support_plane import BAND_TOLERANCE
    from scanner3d.v4l2_capture import BYTES_PER_PIXEL, FRAME_WIDTH

    try:
        fx, fy, cx, cy = ensure_intrinsics_usable(intrinsics)
        scale = ensure_scale_usable(depth_scale)
        ensure_frame_length(raw)
    except DepthConversionNotAllowed as exc:
        raise FusionFrameUnusable(
            f"this frame cannot be masked for fusion: {exc}. Nothing was derived, and the "
            "captured bytes are untouched"
        ) from exc

    samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)
    returned = sum(1 for sample in samples if sample != NO_RETURN_SAMPLE)
    if support is None or not support.applied or support.plane is None:
        # Byte for byte the frame that was captured, and the SAME object: a
        # copy here would be a different frame with the same contents, which is
        # a distinction a digest comparison in a test would not survive.
        return FusionFrame(
            raw=raw,
            applied=False,
            reason="not-run" if support is None else support.reason,
            samples_returned=returned,
            samples_masked=0,
        )

    plane = support.plane
    nx, ny, nz = plane.normal
    offset = plane.offset
    half = (support.band / 2.0) * (1.0 + BAND_TOLERANCE)
    buffer = bytearray(raw)
    masked = 0
    for index, sample in enumerate(samples):
        if sample == NO_RETURN_SAMPLE:
            continue
        distance = sample * scale
        if not math.isfinite(distance):
            continue
        row, column = divmod(index, FRAME_WIDTH)
        # Deliberately the same expressions, in the same order, as
        # scanner3d.depth_metric.deproject_depth_frame. Algebraically equal
        # rearrangements are not equal in float64, and the point of this
        # function is that a sample lands on the same side of the band as the
        # decimated point that was fitted to it.
        x = (column - cx) * distance / fx
        y = (row - cy) * distance / fy
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if abs(x * nx + y * ny + distance * nz - offset) <= half:
            # Set to "no depth returned" rather than removed: the frame keeps
            # its shape, its length and its row stride, and the fusion already
            # skips and counts this value. A shorter buffer would be a frame of
            # a different mode.
            start = index * BYTES_PER_PIXEL
            buffer[start : start + BYTES_PER_PIXEL] = bytes(BYTES_PER_PIXEL)
            masked += 1
    if masked == 0:
        return FusionFrame(
            raw=raw,
            applied=True,
            reason=support.reason,
            samples_returned=returned,
            samples_masked=0,
        )
    return FusionFrame(
        raw=bytes(buffer),
        applied=True,
        reason=support.reason,
        samples_returned=returned,
        samples_masked=masked,
    )


@dataclass(frozen=True, slots=True)
class FusionMaskSummary:
    """What full-resolution masking did across a sweep, beside the cloud's own.

    Kept apart from :class:`PreparationSummary` on purpose. The two describe
    DIFFERENT artifacts from different inputs: one is the decimated cloud that
    was registered, the other the full-resolution frames that were fused, and a
    single set of counts covering both would let a reader carry the speckle
    filter's numbers - which never touched the fusion - into the mesh.
    """

    frames: int
    masked_frames: int
    unmasked_frames: int
    samples_returned: int
    samples_masked: int
    reasons: tuple[tuple[str, int], ...]

    @classmethod
    def of(cls, frames: Sequence[FusionFrame]) -> FusionMaskSummary:
        """Add up a sweep's per-frame fusion records."""
        reasons: dict[str, int] = {}
        for frame in frames:
            if not frame.masked:
                reasons[frame.reason] = reasons.get(frame.reason, 0) + 1
        return cls(
            frames=len(frames),
            masked_frames=sum(1 for frame in frames if frame.masked),
            unmasked_frames=sum(1 for frame in frames if not frame.masked),
            samples_returned=sum(frame.samples_returned for frame in frames),
            samples_masked=sum(frame.samples_masked for frame in frames),
            reasons=tuple(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        )

    def summary_line(self) -> str:
        """One sentence an operator can read while the export runs."""
        if not self.frames:
            return "no frames were masked for fusion"
        return (
            f"support band masked out of {self.masked_frames}/{self.frames} full-resolution "
            f"frames before fusion ({self.samples_masked} of {self.samples_returned} depth "
            f"samples); {self.unmasked_frames} frame(s) are fused exactly as captured"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": FUSION_MASK_SCHEMA_VERSION,
            "support_plane_policy_version": POLICY_VERSION,
            "ran": True,
            "frames": self.frames,
            "masked_frames": self.masked_frames,
            "unmasked_frames": self.unmasked_frames,
            "samples_returned": self.samples_returned,
            "samples_masked": self.samples_masked,
            "unmasked_reasons": [{"reason": name, "frames": count} for name, count in self.reasons],
            "means": (
                "the SAME plane and band the registration cloud lost were set to the sensor's "
                "own no-return value in an in-memory copy of each full-resolution frame, so an "
                "applied support surface is absent from the fused mesh as well as from the "
                "poses. Only a frame whose support filter APPLIED is masked; a withdrawn frame "
                "and a frame the filter never ran on are fused byte for byte as captured. The "
                "captured files on disk were not read for writing, modified or moved, and the "
                "fusion's own arithmetic is unchanged - it was handed a frame with fewer "
                "returns, not a different algorithm"
            ),
        }


#: What :meth:`FusionMaskSummary.as_dict` would have said, for a path that
#: fuses nothing. ``scanner3d reconstruct`` registers and reports without
#: building a surface, and a record that simply omitted this block would let a
#: reader assume the masking had run and found nothing.
FUSION_NOT_RUN: Final[dict[str, object]] = {
    "schema_version": FUSION_MASK_SCHEMA_VERSION,
    "support_plane_policy_version": POLICY_VERSION,
    "ran": False,
    "means": (
        "no full-resolution fusion happened on this path, so no frame was masked for one. "
        "This is not a statement that masking ran and removed nothing"
    ),
}


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    """What preparation did across a whole sweep, for a report or a progress line.

    An aggregate, not a substitute: every frame's own record is still on its
    :class:`PreparedCloud`. This exists because an operator watching a
    reconstruction and a reader opening ``reconstruction.json`` both need one
    honest sentence, and counting the frames a filter withdrew on is the only
    way either of them finds out that it did.
    """

    frames: int
    frames_prepared: int
    support_applied_frames: int
    support_withdrawn_frames: int
    support_not_run_frames: int
    speckle_removed_points: int
    support_removed_points: int
    kept_points: int
    withdrawal_reasons: tuple[tuple[str, int], ...]

    @classmethod
    def of(cls, prepared: Sequence[PreparedCloud]) -> PreparationSummary:
        """Add up a sweep's per-frame records."""
        reasons: dict[str, int] = {}
        applied = withdrawn = not_run = 0
        for cloud in prepared:
            if cloud.support is None:
                not_run += 1
            elif cloud.support.applied:
                applied += 1
            else:
                withdrawn += 1
                reasons[cloud.support.reason] = reasons.get(cloud.support.reason, 0) + 1
        return cls(
            frames=len(prepared),
            frames_prepared=sum(1 for cloud in prepared if cloud.support is not None),
            support_applied_frames=applied,
            support_withdrawn_frames=withdrawn,
            support_not_run_frames=not_run,
            speckle_removed_points=sum(cloud.speckle_removed for cloud in prepared),
            support_removed_points=sum(cloud.support_removed for cloud in prepared),
            kept_points=sum(cloud.kept_points for cloud in prepared),
            # Sorted by count and then by name, so the same sweep prints the
            # same line rather than whatever order a dict happened to fill in.
            withdrawal_reasons=tuple(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        )

    def summary_line(self) -> str:
        """One sentence an operator can read while the reconstruction runs.

        The KEPT count is in it, not only the removed one. "Removed 41000
        points" alone is unreadable as progress: an operator cannot tell a
        filter that took the bed off a healthy cloud from one that took the
        cloud, and the number that distinguishes them is what is left.
        """
        if not self.frames:
            return "no frames were prepared"
        parts = [
            f"support plane removed from {self.support_applied_frames}/{self.frames} frames "
            f"({self.support_removed_points} points), {self.kept_points} points kept for "
            "registration"
        ]
        if self.support_withdrawn_frames:
            reasons = ", ".join(f"{name} x{count}" for name, count in self.withdrawal_reasons)
            parts.append(
                f"{self.support_withdrawn_frames} frame(s) kept their whole cloud: {reasons}"
            )
        if self.support_not_run_frames:
            parts.append(f"{self.support_not_run_frames} frame(s) were empty before the filter")
        return "; ".join(parts)

    def as_dict(self, fusion: FusionMaskSummary | None = None) -> dict[str, object]:
        """The structured aggregate a report publishes.

        Two blocks, because there are two artifacts. ``registered_cloud``
        describes the decimated clouds that were REGISTERED - the only thing
        the speckle filter ever touched - and ``full_resolution_fusion``
        describes the separate in-memory masking of the frames a mesh is FUSED
        from. Rolling them together would let a reader carry either one's
        counts into the other's artifact, which is exactly the mistake that
        left a proved bed in an exported mesh while the record said it had been
        removed.

        ``fusion`` is absent on a path that fuses nothing, and the block then
        says so rather than being omitted; see :data:`FUSION_NOT_RUN`.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "preparation_version": PREPARATION_VERSION,
            "support_plane_policy_version": POLICY_VERSION,
            "order": list(PREPARATION_ORDER),
            "frames": self.frames,
            "registered_cloud": {
                "means": (
                    "the decimated per-frame clouds that were registered. Every count in this "
                    "block is about those clouds and about no exported mesh"
                ),
                "frames_prepared": self.frames_prepared,
                "kept_points": self.kept_points,
                "support_plane": {
                    "applied_frames": self.support_applied_frames,
                    "withdrawn_frames": self.support_withdrawn_frames,
                    "not_run_frames": self.support_not_run_frames,
                    "removed_points": self.support_removed_points,
                    "withdrawal_reasons": [
                        {"reason": name, "frames": count} for name, count in self.withdrawal_reasons
                    ],
                    "means": (
                        "a withdrawn frame kept every point it had: the filter ran, could not "
                        "prove the largest plane it found was the surface the object stands "
                        "on, and said so rather than removing it. A not-run frame was empty "
                        "before the filter could look"
                    ),
                },
                "speckle": {
                    "removed_points": self.speckle_removed_points,
                    "means": (
                        "scanner3d.denoise.strip_speckle, applied to the DECIMATED cloud only. "
                        "It did not touch the full-resolution frames a mesh is fused from"
                    ),
                },
            },
            "full_resolution_fusion": (
                dict(FUSION_NOT_RUN) if fusion is None else fusion.as_dict()
            ),
        }


def filter_statement(summary: PreparationSummary, fusion: FusionMaskSummary | None = None) -> str:
    """The exported record's sentence about what every frame was put through.

    Assembled from the run rather than written as a constant, because the claim
    has to be exact: the two filters below can both withdraw, and a fixed
    string saying they ran would be wrong for whichever frames they did not act
    on.

    It also has to be exact about WHICH ARTIFACT each filter reached. Speckle
    rejection ran on the decimated clouds that were registered and on nothing
    else; support-plane suppression ran on those clouds AND, for the frames it
    applied to, on an in-memory copy of the full-resolution frames a mesh is
    fused from. ``fusion`` is what the second of those did, and is absent on a
    path that fuses nothing.
    """
    speckle = (
        "Speckle rejection (scanner3d.denoise.strip_speckle, default settings) dropped "
        f"{summary.speckle_removed_points} point(s) for having fewer than 3 other points of the "
        "same frame in the block of grid cells around them; on synthetic frames that removes "
        "essentially every stray further than two spacings from real surface and about 1-3% of "
        "the real points - rims, grazing faces, the last samples before an edge - and it cannot "
        "tell the two apart. It ran on the DECIMATED registration clouds only: the "
        "full-resolution frames any fused surface was built from still contain every stray it "
        "rejected."
    )
    support = (
        "Support-plane suppression (scanner3d.support_plane.suppress_support_plane) removed "
        f"{summary.support_removed_points} point(s) lying in a proved support surface across "
        f"{summary.support_applied_frames} of {summary.frames} registration clouds, and withdrew "
        f"on {summary.support_withdrawn_frames} frame(s), which therefore still contain whatever "
        "they were resting on. It removes only a narrow band around a plane the frame's own "
        "geometry proves is a surrounding support surface - never a crop, a half-space, a "
        "background or a hand - and a large flat surface belonging to the object is deliberately "
        "preserved wherever the evidence is ambiguous."
    )
    if fusion is None:
        reach = (
            "No surface was fused on this path, so no full-resolution frame was masked for one. "
            "Every point either filter removed was removed from a registration cloud, before "
            "registration, so it contributed to no pose."
        )
    else:
        reach = (
            "For each of those applied frames the SAME plane and band were also set to the "
            "sensor's no-return value in an in-memory copy of the full-resolution frame, so the "
            "proved support surface is absent from the fused mesh as well as from the poses: "
            f"{fusion.samples_masked} of {fusion.samples_returned} depth samples across "
            f"{fusion.masked_frames} of {fusion.frames} frames. The other "
            f"{fusion.unmasked_frames} frame(s) were fused exactly as captured. No point either "
            "filter removed contributed to a pose, and no masked sample reached the fused field."
        )
    return (
        "each frame was prepared by scanner3d.preparation.prepare_frame at preparation version "
        f"{PREPARATION_VERSION} and support-plane policy version {POLICY_VERSION}, in the order "
        + " -> ".join(PREPARATION_ORDER)
        + ". "
        + speckle
        + " "
        + support
        + " "
        + reach
        + " The raw frames on disk are unchanged and complete: masking derives a new frame in "
        "memory and writes nothing"
    )


#: Every outcome name a support-plane record can carry, re-exported so a caller
#: does not have to import two modules to branch on one field.
SUPPORT_APPLIED: Final = REASON_APPLIED
