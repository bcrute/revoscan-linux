"""Pose-compensated temporal completion of raw no-return depth, across posed frames.

What this is
------------
:mod:`scanner3d.depth_completion` recovers a hole from the frame the hole is
in. That is all one frame can do, and the aperture control in the depth
completeness corpus shows exactly where it stops: a one-sample gap in a surface
and a one-sample dropout on that surface present byte-identical evidence to any
single-frame method, so the method fills both and is confidently wrong about
one of them.

The way out of that is not a better local rule. It is a second measurement,
taken from somewhere else. This module takes the *original raw samples of other
frames*, carries them through the exact accepted poses of those frames into the
target camera, projects them into the target image, and asks whether they agree
about a sample the target did not return. When enough independent posed
measurements agree, the sample is reconstructed with temporal provenance and
names the exact other-frame pixels it rests on. When they do not - because one
of them is occluded, because they disagree, because they all come from the same
viewpoint, or because there are not enough of them - the sample stays missing,
with a named reason.

What it is not
--------------
* It is not a same-pixel carry. "Use the previous frame" without applying the
  pose between the two is how a hole gets filled with the depth of whatever
  else happened to be at that pixel, and :func:`project_observation` exists so
  that the pose composition can be asserted directly rather than inferred from
  an aggregate score.
* It is not a pose estimator. Every pose it uses must already be accepted by
  :mod:`scanner3d.tracking`, finite and rigid. This module never estimates,
  repairs or substitutes one; an unaccepted pose is a refusal, not an input to
  be improved.
* It is not a smoother, and it does not touch measured samples. Raw frame bytes
  are immutable, temporal evidence may occupy only a raw target zero, and every
  measured code comes back byte-identical under :data:`STATE_MEASURED`.
* It is not wired into capture, the live preview, the tracker, the TSDF, the
  mesh or export. Nothing in :mod:`scanner3d` calls it. Tracking continues to
  consume measured geometry only. Whether a temporal reconstruction is eligible
  for fusion, and at what weight, is a decision a later stage makes with its own
  evidence; this module exists so that decision has something honest to make.
* It is not physical evidence. The synthetic corpus that scores it renders its
  own truth and hands it exact poses. A real capture has neither.

Why this is an offline post-tracking stage
------------------------------------------
Every transform here is built from poses that tracking has already accepted. So
this stage cannot run before tracking, cannot feed tracking without making
tracking's own input depend on tracking's own output, and cannot be run live on
a frame whose pose is not yet settled. That is a property of the method, not a
scheduling choice: an unconfirmed pose turns a temporal reconstruction into a
measurement of the pose error rather than of the scene, and the reconstruction
would still look like geometry.

The coordinate and pose convention
----------------------------------
Stated once, because a composition in the wrong order produces a scan that is
wrong in a way that still looks like a scan:

* A **camera frame** is the one :data:`scanner3d.depth_metric.CAMERA_FRAME`
  describes - ``+x`` toward increasing column, ``+y`` toward increasing row,
  ``+z`` along the optical axis into the scene - and a sample is deprojected by
  :data:`scanner3d.depth_metric.DEPROJECTION_EXPRESSION`, exactly.
* A **pose** carries that frame's camera coordinates into world coordinates,
  the same direction :mod:`scanner3d.tracking` publishes. So the world point of
  a source sample is ``source.pose.transform.apply(point_in_source_camera)``.
* The **temporal transform** from a source frame into the target camera is
  therefore ``target.pose.transform.inverse().compose(source.pose.transform)``,
  and :func:`temporal_transform` is the only place it is built. Composing those
  two the other way round is the single most likely error in this module, and
  it is a *silent* one on a stationary pair, so the tests assert the direction
  on a non-identity motion with an analytically known answer.

The method, and why each bound is there
---------------------------------------
Deterministic, bounded, two passes over fixed data, and conservative at every
place a wrong answer would look like geometry:

1. **Only original raw measurements of other frames are evidence.** A source
   sample is eligible when its own raw code is nonzero in the immutable frame
   this module was handed. Not a spatial reconstruction, not another temporal
   reconstruction, and never a sample of the target frame itself: a temporal
   reconstruction that rested on the target's own frame would be a spatial
   method wearing a different label, and one that rested on a reconstruction
   would let a fill walk across a capture one frame at a time.

2. **Projection is exact, and refusals are named.** Each eligible source sample
   is deprojected in its own camera, carried by :func:`temporal_transform` into
   the target camera and projected into the target image. A point behind either
   camera, outside the target rectangle, further than
   :attr:`TemporalPolicy.max_reprojection_distance` from the pixel centre it
   would vote for, or whose predicted code leaves the policy's own range, is
   refused with the reason that fired. Nothing is clamped into range.

3. **Visibility is decided by a z-buffer, not by iteration order.** Every
   surviving observation is deposited first, and only then judged. A target
   pixel's nearest evidence is the smallest depth among everything deposited
   there *and* the target's own measured depth if it returned one, so a farther
   surface cannot vote through nearer measured geometry, and which observation
   happened to be visited first cannot decide the answer.
   :attr:`TemporalPolicy.occlusion_margin_codes` is the tolerance below which
   two depths are the same surface rather than one hiding the other.

4. **Support has to be independent.** A reconstruction needs at least
   :attr:`TemporalPolicy.min_support_frames` *distinct other frames* that
   returned a raw measurement, and at least
   :attr:`TemporalPolicy.min_distinct_viewpoints` genuinely different
   viewpoints among them: two frames whose transforms into the target differ by
   less than :attr:`TemporalPolicy.min_baseline` in translation *and* less than
   :attr:`TemporalPolicy.min_rotation_degrees` in rotation are one viewpoint
   seen twice, whatever their frame indices say. A stationary camera does not
   become a second view by taking a second frame.

5. **Disagreement is detected before it is averaged.** The per-frame
   representatives at one target must agree within
   :attr:`TemporalPolicy.max_depth_disagreement_codes`; a wider spread means the
   frames are looking at different surfaces and the sample stays missing. A
   target hole whose own measured neighbours straddle a depth discontinuity
   wider than :attr:`TemporalPolicy.cluster_gap_codes` is refused too - that is
   the silhouette, the disocclusion and the grazing edge, where "which surface
   is this" is exactly the question the evidence cannot settle.

6. **Work is bounded before the loops.** The number of frames, the samples per
   frame, the target footprint, the candidates per target and the support count
   all have absolute product ceilings, and the whole call's worst case -
   ``eligible_source_samples * footprint_cells + holes * (2 +
   max_candidates_per_target)`` - is derived from the input before the first
   projection and refused with :class:`WorkTooLarge` past
   :data:`MAX_WORK_UNITS`. A finite Python integer is not a bound on work; a
   ceiling checked before the loop is.

Spatial confirmation is not temporal recovery
---------------------------------------------
An accepted :class:`scanner3d.depth_completion.CompletedPatch` may be handed in
for the target frame. Where it proposes a value at a target hole, this module
publishes what the independent posed evidence said about it, in the closed
vocabulary :data:`CONFIRMATIONS`:

``confirmed``
    Independent posed raw measurements reproduced the spatial estimate within
    :attr:`TemporalPolicy.confirmation_tolerance_codes`. The sample becomes a
    temporal reconstruction, carrying the *temporal* value, and names both the
    other-frame raw samples that support it and the spatial support it confirms.
``contradicted``
    Independent posed evidence produced a value the spatial estimate does not
    match. The two are **not** averaged - a number halfway between two
    incompatible answers is not evidence of anything. The independent value is
    published whole, so the sample becomes a temporal reconstruction resting on
    the other frames' raw samples, the overruled estimate's own support stays
    named, and the disagreement is published in the counts.

    Overruling rather than keeping the estimate is deliberate. The evidence at
    that sample is exactly the evidence a run given no spatial estimate has
    there, and such a run reconstructs it; letting a weaker same-frame guess
    veto a well-supported independent measurement would make the published
    depth depend on whether an earlier stage happened to guess. A contradiction
    is therefore separate evidence about the *estimate*, and never by itself a
    recovered sample - the recovered sample is the independent measurement that
    replaced it.
``not_observed``
    No independent evidence reached the sample. The spatial estimate stays
    exactly what it was, explicitly spatial, and is not laundered into a
    temporal one by having passed through this module.
``no_spatial_estimate``
    There was no spatial proposal here. The sample is temporally reconstructed
    if the posed evidence is sufficient, and missing otherwise.

So an unsupported spatial hypothesis is not made safe by being run through a
temporal stage, which is the specific failure this vocabulary exists to prevent.

Confidence
----------
:attr:`TemporalPatch.confidence` is a fraction of 1 derived from named evidence.
It is a **reconstruction** confidence: it says nothing about the sensor, never
turns a reconstruction into a measurement, and is capped strictly below the
``1.0`` a measured sample carries - see
:data:`MAX_RECONSTRUCTION_CONFIDENCE`. It is the geometric mean of five terms,
each in ``[0, 1]`` and each monotone in the direction its name implies; see
:func:`temporal_confidence` for the exact formula.

Backends
--------
The dependency-free ``"python"`` reference is the only implementation and is the
authority. ``"auto"`` selects it. Naming any other backend is an error rather
than a silent downgrade, on the same rule as :mod:`scanner3d.compute`.

Units
-----
Depth is in **raw codes** and in the caller's own length unit through
:attr:`CameraModel.depth_scale`; image distances are in **pixels**; baselines
are in the same length unit as the poses. This module never states what that
length is - that is the caller's provenance to carry, and
:mod:`scanner3d.depth_metric` owns the decoding rule that a raw code of zero is
no measurement at all.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d import depth_completion
from scanner3d.compute import ComputeBackendUnavailable, ComputeBackendUnknown
from scanner3d.depth_completion import MAX_CODE, MAX_PATCH_SAMPLES, CompletedPatch, DepthPatch
from scanner3d.depth_metric import NO_RETURN_SAMPLE, ensure_scale_usable
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid, ensure_finite, ensure_rotation

#: The schema of every public type in this module. A caller that persists a
#: result records this beside it; a reader that does not understand it refuses
#: rather than guessing which fields it is looking at.
SCHEMA_VERSION: Final = 1

# --- the absolute ceilings -----------------------------------------------------------
#
# Product maxima, not defaults. None of them is reachable by the shipped
# defaults, and each is checked before the loop that could spend the work it
# bounds - because a policy stating a finite number has not stated a bound.

#: The most frames one temporal completion may ever consider, the target
#: included. Sixteen posed views of one hole is already far past what any
#: bounded live capture would offer a single sample.
MAX_FRAMES: Final = 16

#: The most *distinct other frames* one reconstruction may ever rest on. Beyond
#: this the marginal frame is not evidence, it is arithmetic.
MAX_SUPPORT_FRAMES: Final = 8

#: The largest target footprint one projected observation may ever occupy, in
#: samples either side of the pixel it rounds to. Two gives a 5x5 stamp; a
#: projection that needs more than that is not landing on the sample it claims.
MAX_FOOTPRINT_RADIUS: Final = 2

#: The furthest, in pixels, a projected observation may ever be from the centre
#: of a target sample it votes for.
MAX_REPROJECTION_DISTANCE: Final = 4.0

#: The most posed observations one target sample may ever accumulate before the
#: sample is refused as ambiguous rather than resolved by a tie-break.
MAX_CANDIDATES_PER_TARGET: Final = 256

#: The most projection and comparison work one call may be permitted to spend,
#: derived from its input before it runs. Three whole 640x400 source frames of
#: entirely measured samples at a 3x3 footprint cost 6,912,000 units, and a
#: whole target frame of holes adds 256,000 * (2 + 64); this ceiling admits
#: that and refuses anything materially larger before the first projection.
MAX_WORK_UNITS: Final = 25_000_000

#: The most a reconstruction's confidence may ever be. Strictly below the
#: ``1.0`` a measured sample carries, so the two populations cannot meet even
#: when every term of the formula is at its best. A reconstruction that scored
#: perfectly on its own evidence is still not a measurement.
MAX_RECONSTRUCTION_CONFIDENCE: Final = 0.99

#: What this module says about one sample. Stored compactly as one byte each and
#: read back through :meth:`TemporalPatch.state_at`.
#:
#: The first three are exactly :mod:`scanner3d.depth_completion`'s, with the
#: same numeric values, because a result that carried a spatial estimate through
#: has not changed what that estimate is. The fourth is this module's own.
STATE_MISSING: Final = depth_completion.STATE_MISSING
STATE_MEASURED: Final = depth_completion.STATE_MEASURED
STATE_SPATIALLY_RECONSTRUCTED: Final = depth_completion.STATE_SPATIALLY_RECONSTRUCTED
STATE_TEMPORALLY_RECONSTRUCTED: Final = 3
STATE_NAMES: Final[tuple[str, ...]] = (
    "missing",
    "measured",
    "spatially_reconstructed",
    "temporally_reconstructed",
)

#: What independent posed evidence said about a spatial estimate at a target
#: hole. Closed, and always published for every sample, so "we did not check"
#: and "we checked and nothing came" are different facts in the artifact.
CONFIRMATION_NO_ESTIMATE: Final = 0
CONFIRMATION_CONFIRMED: Final = 1
CONFIRMATION_CONTRADICTED: Final = 2
CONFIRMATION_NOT_OBSERVED: Final = 3
CONFIRMATIONS: Final[tuple[str, ...]] = (
    "no_spatial_estimate",
    "confirmed",
    "contradicted",
    "not_observed",
)

#: Every reason one posed observation can be discarded before it ever becomes a
#: candidate, in the order they are tested. Closed and always fully reported.
OBSERVATION_REJECTIONS: Final[tuple[str, ...]] = (
    "source_behind_camera",
    "behind_target_camera",
    "outside_target_patch",
    "reprojection_too_far",
    "target_already_measured",
    "predicted_code_out_of_range",
    "occluded_by_nearer_evidence",
)

#: Every reason a target hole can be left without a temporal reconstruction, in
#: the order they are tested. Closed and always fully reported, at zero when the
#: reason never fired, so a run's conservatism is a stable shape.
REJECTION_REASONS: Final[tuple[str, ...]] = (
    "no_posed_evidence",
    "all_evidence_refused",
    "too_many_candidates",
    "insufficient_support_frames",
    "insufficient_viewpoint_diversity",
    "conflicting_depth_evidence",
    "edge_discontinuity",
    "prediction_out_of_range",
    "confidence_below_threshold",
)

#: The reference backend, and the only one implemented here.
PYTHON_BACKEND: Final = "python"
AUTO_BACKEND: Final = "auto"

#: Backends this module recognises the name of and does not provide.
_UNIMPLEMENTED_BACKENDS: Final[tuple[str, ...]] = ("numpy", "scipy", "gpu")


class TemporalNotAllowed(ScannerError):
    """A temporal completion that was refused before it ran, and the reason."""


class CameraUnusable(TemporalNotAllowed):
    """A camera model whose numbers could not project or deproject anything."""


class PoseUnusable(TemporalNotAllowed):
    """A pose that is not an accepted, finite, rigid transform."""


class FrameUnusable(TemporalNotAllowed):
    """A frame whose identity, geometry or raw patch is not usable as evidence."""


class PolicyUnusable(TemporalNotAllowed):
    """A temporal policy that states a bound nothing could be held to."""


class ResultUnusable(TemporalNotAllowed):
    """A temporal result whose own account of itself does not hold."""


class WorkTooLarge(TemporalNotAllowed):
    """A call whose worst-case work, derived before it runs, is past the ceiling."""


# --- small validators, type before value ---------------------------------------------


def _exact_int(value: object, name: str, error: type[TemporalNotAllowed]) -> int:
    """One integer that has to be an integer, checked before it is compared.

    ``True == 1`` and ``1.0 == 1`` in Python, so an equality or a range test
    that runs first happily accepts a boolean width and a float schema version.
    Every number here whose identity is load-bearing is type-checked before its
    value is looked at.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(
            f"{name} is {value!r}, which is not an integer. ``True`` and ``1.0`` both compare "
            "equal to 1, so the type is checked before the value and not after it"
        )
    return value


def _finite_number(value: object, name: str, error: type[TemporalNotAllowed]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error(f"{name} is {value!r}, which is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise error(f"{name} is {value!r}, which is not finite")
    return number


def _positive_int(value: object, name: str, maximum: int) -> int:
    """A count, inside the product ceiling stated for it."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyUnusable(f"a temporal policy states {name} {value!r}, which is not a count")
    if value > maximum:
        raise PolicyUnusable(
            f"a temporal policy states {name} {value!r}, past the ceiling of {maximum} this "
            "module states. A finite integer is not a bound on work; this is"
        )
    return value


def _positive_number(value: object, name: str, maximum: float) -> float:
    number = _finite_number(value, f"a temporal policy states {name}", PolicyUnusable)
    if number <= 0.0:
        raise PolicyUnusable(
            f"a temporal policy states {name} {value!r}; every bound here is finite and "
            "positive, and an unbounded one is not a bound"
        )
    if number > maximum:
        raise PolicyUnusable(
            f"a temporal policy states {name} {value!r}, past the ceiling of {maximum} this "
            "module states"
        )
    return number


def _count(value: object, name: str) -> int:
    """One count in a work record: a real non-negative integer, never a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultUnusable(f"work states {name} {value!r}, which is not a count")
    if value < 0:
        raise ResultUnusable(f"work states {name} {value!r}; a count of something is not negative")
    return value


# --- the camera ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraModel:
    """The pinhole model every frame here is deprojected and projected with.

    Not a decoder and not a claim about a device. It is the four numbers the
    deprojection :data:`scanner3d.depth_metric.DEPROJECTION_EXPRESSION` needs
    plus the scale that turns a raw code into a length, in the frame's own full
    pixel coordinates - so a patch that is a window of a frame carries its
    origin and the intrinsics stay the frame's, rather than being shifted into
    a private coordinate system nobody else can check.

    :mod:`scanner3d.depth_metric` remains the authority for what a raw code
    means and for the coordinate convention; this type states the same
    convention and refuses numbers that convention cannot be evaluated with.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            _finite_number(getattr(self, name), f"a camera states {name}", CameraUnusable)
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise CameraUnusable(
                f"a camera states fx={self.fx!r}, fy={self.fy!r}; a pinhole projection divides "
                "by both, so neither may be zero or negative"
            )
        scale = _finite_number(self.depth_scale, "a camera states depth_scale", CameraUnusable)
        try:
            ensure_scale_usable(scale)
        except ScannerError as error:  # pragma: no cover - the message is depth_metric's
            raise CameraUnusable(str(error)) from error

    def deproject(self, u: float, v: float, code: int) -> Point:
        """One raw code at one *frame* pixel, as a point in that frame's camera.

        Exactly :data:`scanner3d.depth_metric.DEPROJECTION_EXPRESSION`, written
        for a single sample rather than a whole 512000-byte frame, because a
        temporal candidate is one sample and building a quarter of a million
        points to use one of them is not a bounded operation.
        """
        depth = code * self.depth_scale
        return ((u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth)

    def project(self, point: Point) -> tuple[float, float]:
        """One point in a camera frame, as the *frame* pixel it lands on.

        The exact inverse of :meth:`deproject`, and the caller is responsible
        for having checked that ``point[2] > 0``: a point at or behind the
        optical centre does not project, and returning a number for it would be
        a coordinate rather than a refusal.
        """
        x, y, z = point
        return (self.cx + x * self.fx / z, self.cy + y * self.fy / z)

    def code_of(self, depth: float) -> float:
        """The raw code one depth corresponds to, unrounded and unclamped."""
        return depth / self.depth_scale

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "depth_scale": self.depth_scale,
        }


# --- the pose --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoseRecord:
    """One frame's accepted world pose, with the acceptance that made it usable.

    ``transform`` carries that frame's camera coordinates into world
    coordinates, the direction :mod:`scanner3d.tracking` publishes.

    ``accepted`` is carried rather than assumed because this module refuses to
    estimate, repair or substitute a pose: the tracker either placed this frame
    or it did not, and a temporal reconstruction built on a pose nobody accepted
    is a measurement of the pose error. ``status`` is the tracker's own word for
    how it was placed, and ``quality`` is a fraction of 1 that enters the
    published confidence - a pose accepted with little margin cannot produce a
    reconstruction with a lot of it.
    """

    transform: Rigid
    accepted: bool
    status: str
    quality: float = 1.0

    def __post_init__(self) -> None:
        transform = self.transform
        if not isinstance(transform, Rigid):
            raise PoseUnusable(f"a pose carries {transform!r}, which is not a Rigid transform")
        # Rigid.of validates, but a caller can build the dataclass directly and
        # a promise the type checker keeps is not a promise the data keeps.
        try:
            ensure_finite(transform.translation, "pose translation")
            ensure_rotation(transform.rotation)
        except ScannerError as error:
            raise PoseUnusable(
                f"a pose is not a finite rigid transform: {error}. Nothing here repairs one"
            ) from error
        if not isinstance(self.accepted, bool):
            raise PoseUnusable(
                f"a pose states accepted {self.accepted!r}; acceptance is a boolean the tracker "
                "decided, and a truthy value is not a decision"
            )
        if not isinstance(self.status, str) or not self.status:
            raise PoseUnusable(f"a pose states status {self.status!r}, which is not a name")
        quality = _finite_number(self.quality, "a pose states quality", PoseUnusable)
        if not 0.0 < quality <= 1.0:
            raise PoseUnusable(
                f"a pose states quality {self.quality!r}; it is a fraction of 1 above zero, "
                "because a pose worth nothing does not place a frame"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "quality": self.quality,
            "transform": self.transform.as_dict(),
        }


def temporal_transform(target: PoseRecord, source: PoseRecord) -> Rigid:
    """The transform carrying ``source`` camera coordinates into ``target``'s.

    ``target.transform.inverse().compose(source.transform)`` and nothing else.
    Both poses carry camera coordinates into the world, so the way back into the
    target camera is the target's inverse applied after the source's own pose.
    Built here, once, so no caller can compose it the other way round; the tests
    assert the direction on a motion whose answer is known analytically, because
    a reversed composition is invisible on a stationary pair and produces a
    plausible-looking wrong surface on every other one.
    """
    if not isinstance(target, PoseRecord) or not isinstance(source, PoseRecord):
        raise PoseUnusable("a temporal transform is built from two PoseRecords")
    return target.transform.inverse().compose(source.transform)


def viewpoint_separation(a: Rigid, b: Rigid) -> tuple[float, float]:
    """How far apart two transforms into the target camera are: baseline, degrees.

    The baseline is the distance between the two source optical centres as the
    target camera sees them, and the angle is the geodesic rotation between
    them. Two views that differ in neither are the same view twice, which is
    what :attr:`TemporalPolicy.min_baseline` and
    :attr:`TemporalPolicy.min_rotation_degrees` are for.
    """
    between = a.inverse().compose(b)
    baseline = math.hypot(
        math.hypot(between.translation[0], between.translation[1]), between.translation[2]
    )
    return baseline, math.degrees(between.rotation_angle)


# --- one frame's evidence ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalFrame:
    """One frame of a temporal completion: its raw patch, where it sits, its pose.

    ``patch`` is the immutable raw rectangle, ``(u0, v0)`` is where that
    rectangle starts in the full frame the camera model describes, and ``pose``
    is the accepted world pose of the camera that produced it. Every one of
    those is needed to say what a sample of this frame *is*: a code with no
    pixel is not a ray and a ray with no pose is not a place.
    """

    index: int
    patch: DepthPatch
    u0: int
    v0: int
    pose: PoseRecord

    def __post_init__(self) -> None:
        _exact_int(self.index, "a frame's index", FrameUnusable)
        if self.index < 0:
            raise FrameUnusable(f"a frame calls itself {self.index}, which is not an index")
        if not isinstance(self.patch, DepthPatch):
            raise FrameUnusable(
                f"a frame carries {self.patch!r}, which is not a DepthPatch. The raw evidence is "
                "the validated immutable type or it is a list somebody still holds"
            )
        for name in ("u0", "v0"):
            _exact_int(getattr(self, name), f"a frame's {name}", FrameUnusable)
            if getattr(self, name) < 0:
                raise FrameUnusable(
                    f"a frame starts at {name}={getattr(self, name)}, which is outside its frame"
                )
        if not isinstance(self.pose, PoseRecord):
            raise FrameUnusable(f"a frame carries pose {self.pose!r}, which is not a PoseRecord")

    def pixel_of(self, index: int) -> tuple[int, int]:
        """The full-frame ``(u, v)`` of one patch index."""
        column, row = self.patch.position_of(index)
        return self.u0 + column, self.v0 + row

    def index_of(self, u: int, v: int) -> int | None:
        """The patch index of a full-frame pixel, or ``None`` when it is outside."""
        return self.patch.index_of(u - self.u0, v - self.v0)

    def measured_indices(self) -> tuple[int, ...]:
        return self.patch.measured_indices()

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "u0": self.u0,
            "v0": self.v0,
            "width": self.patch.width,
            "height": self.patch.height,
            "measured": len(self.patch.measured_indices()),
            "pose": self.pose.as_dict(),
        }


# --- the policy --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    """Every bound :func:`complete_temporally` is held to, in codes and in pixels.

    Immutable, validated on construction and published in the result, so an
    accepted sample is reproducible from its named support and these numbers
    alone. Every field is a conservatism knob: tightening any of them refuses
    more and invents less.

    Two of them are not free choices and are called out here rather than left in
    a comment. :attr:`max_depth_disagreement_codes` has to come from the
    sensor's own noise floor plus the pose error the tracker actually leaves -
    it is the width at which two independent posed measurements of one surface
    are still one surface. And :attr:`min_baseline` is in the same length unit
    as the poses, so a caller working in metres and a caller working in
    scanner-scale units do not get the same number.
    """

    #: The fewest distinct *other* frames one reconstruction may rest on. Two:
    #: one other frame is a carry, and a carry cannot be checked against
    #: anything.
    min_support_frames: int = 2
    #: The most distinct other frames one reconstruction uses. A cap, not a
    #: target: it bounds the support a single sample can name.
    max_support_frames: int = 4
    #: The fewest genuinely different viewpoints among those frames. Two: a
    #: stationary camera does not become a second view by taking a second frame,
    #: and same-pixel carry across a stationary pair is exactly what this module
    #: exists to rule out.
    min_distinct_viewpoints: int = 2
    #: Below both of these, two transforms into the target are one viewpoint.
    #: In the poses' own length unit, and in degrees.
    min_baseline: float = 0.05
    min_rotation_degrees: float = 0.25
    #: How far, in samples either side of the pixel a projection rounds to, one
    #: observation may be deposited. One gives a 3x3 stamp, so a sample whose
    #: projection lands between two pixels can still support the one it is
    #: nearest to without being smeared across the image.
    footprint_radius: int = 1
    #: How far, in pixels, a projected observation may be from the centre of a
    #: sample it votes for. Half a pixel by default, which is the strict rule:
    #: an observation supports only the sample whose centre it is nearest to,
    #: and a point landing exactly on the boundary supports both.
    #:
    #: This is the single most consequential number in the policy and it is set
    #: conservatively on purpose. Widening it to 0.6 on this project's
    #: three-view synthetic case raises the recovered samples from 16 to 195 and
    #: the worst error against held-out truth from 1 raw code to 205, because an
    #: observation two thirds of a pixel away is a measurement of a NEARBY point
    #: on the surface rather than of this one, and on a surface that is turning
    #: away from the camera those are not the same depth. The default recovers
    #: only what another posed frame actually measured at this sample.
    max_reprojection_distance: float = 0.5
    #: The most observations one target sample may accumulate. Past it the
    #: sample is refused as ambiguous rather than settled by a tie-break.
    max_candidates_per_target: int = 64
    #: How far apart, in raw codes, two independent posed measurements of one
    #: target sample may be and still be one surface. The sensor-dependent bound.
    max_depth_disagreement_codes: float = 32.0
    #: How much nearer, in raw codes, other evidence has to be before it is
    #: treated as occluding rather than as the same surface.
    occlusion_margin_codes: float = 32.0
    #: How wide a step among a target hole's own measured neighbours makes the
    #: hole a discontinuity rather than a gap in one surface.
    cluster_gap_codes: float = 64.0
    #: How far, in raw codes, a spatial estimate may be from the independent
    #: temporal value and still be called confirmed by it.
    confirmation_tolerance_codes: float = 8.0
    #: The lowest confidence a reconstruction may be published with. Applied
    #: here, so a caller that ignores confidence still gets the conservative
    #: answer.
    min_confidence: float = 0.2
    #: The code range a prediction must land inside. One is the lowest code that
    #: is a return at all, because zero is no measurement.
    min_code: int = 1
    max_code: int = MAX_CODE

    def __post_init__(self) -> None:
        minimum = _positive_int(self.min_support_frames, "min_support_frames", MAX_SUPPORT_FRAMES)
        maximum = _positive_int(self.max_support_frames, "max_support_frames", MAX_SUPPORT_FRAMES)
        viewpoints = _positive_int(
            self.min_distinct_viewpoints, "min_distinct_viewpoints", MAX_SUPPORT_FRAMES
        )
        # Zero is a legal footprint - it means an observation votes only for the
        # sample it rounds to - so this is the one count that is not positive,
        # and it is checked for type before its value like every other.
        if isinstance(self.footprint_radius, bool) or not isinstance(self.footprint_radius, int):
            raise PolicyUnusable(
                f"a temporal policy states footprint_radius {self.footprint_radius!r}, which is "
                "not a number of samples"
            )
        if not 0 <= self.footprint_radius <= MAX_FOOTPRINT_RADIUS:
            raise PolicyUnusable(
                f"a temporal policy states footprint_radius {self.footprint_radius!r}, outside "
                f"the 0..{MAX_FOOTPRINT_RADIUS} this module states. A finite integer is not a "
                "bound on work; this is"
            )
        _positive_int(
            self.max_candidates_per_target, "max_candidates_per_target", MAX_CANDIDATES_PER_TARGET
        )
        min_code = _positive_int(self.min_code, "min_code", MAX_CODE)
        max_code = _positive_int(self.max_code, "max_code", MAX_CODE)
        _positive_number(self.min_baseline, "min_baseline", 1e9)
        _positive_number(self.min_rotation_degrees, "min_rotation_degrees", 180.0)
        _positive_number(
            self.max_reprojection_distance, "max_reprojection_distance", MAX_REPROJECTION_DISTANCE
        )
        _positive_number(
            self.max_depth_disagreement_codes, "max_depth_disagreement_codes", float(MAX_CODE)
        )
        _positive_number(self.occlusion_margin_codes, "occlusion_margin_codes", float(MAX_CODE))
        _positive_number(self.cluster_gap_codes, "cluster_gap_codes", float(MAX_CODE))
        _positive_number(
            self.confirmation_tolerance_codes, "confirmation_tolerance_codes", float(MAX_CODE)
        )
        confidence = _finite_number(
            self.min_confidence, "a temporal policy states min_confidence", PolicyUnusable
        )
        if not 0.0 < confidence <= MAX_RECONSTRUCTION_CONFIDENCE:
            raise PolicyUnusable(
                f"a temporal policy states min_confidence {self.min_confidence!r}; it is a "
                f"fraction above zero and at most {MAX_RECONSTRUCTION_CONFIDENCE}, because a "
                "reconstruction nothing supports is not published at any confidence and none of "
                "them is ever worth a measurement"
            )
        if maximum < minimum:
            raise PolicyUnusable(
                f"a temporal policy allows at most {maximum} supporting frame(s) and requires at "
                f"least {minimum}"
            )
        if viewpoints > minimum:
            raise PolicyUnusable(
                f"a temporal policy requires {viewpoints} distinct viewpoint(s) among at least "
                f"{minimum} supporting frame(s), which could never be satisfied"
            )
        if max_code <= min_code:
            raise PolicyUnusable(
                f"a temporal policy predicts into codes {min_code}..{max_code}, which is not a "
                f"range inside 1..{MAX_CODE}"
            )

    @property
    def footprint_cells(self) -> int:
        """The most target samples one projected observation can be deposited into."""
        return (2 * self.footprint_radius + 1) ** 2

    def worst_case_work(self, eligible_samples: int, holes: int) -> int:
        """The most projection and comparison work this call can cost.

        Exact rather than an estimate, and derivable from the input before the
        run: every eligible source sample is projected once and deposited into,
        and judged for visibility at, at most :attr:`footprint_cells` target
        samples; and every target hole is examined once and then costs one unit
        per candidate deposited at it. That per-hole term is
        ``2 + max_candidates_per_target`` rather than
        ``1 + max_candidates_per_target``: the deposit keeps one observation
        past the cap so an over-cap target is provably over it, and that
        retained sentinel is examined like any other candidate. Work a run does
        is work its budget covers.
        :func:`complete_temporally` refuses a call whose value here exceeds
        :data:`MAX_WORK_UNITS` before it projects anything.
        """
        for value, name in ((eligible_samples, "eligible samples"), (holes, "holes")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PolicyUnusable(f"{value!r} is not a number of {name}")
        return eligible_samples * self.footprint_cells + holes * (
            2 + self.max_candidates_per_target
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "min_support_frames": self.min_support_frames,
            "max_support_frames": self.max_support_frames,
            "min_distinct_viewpoints": self.min_distinct_viewpoints,
            "min_baseline": self.min_baseline,
            "min_rotation_degrees": self.min_rotation_degrees,
            "footprint_radius": self.footprint_radius,
            "max_reprojection_distance": self.max_reprojection_distance,
            "max_candidates_per_target": self.max_candidates_per_target,
            "max_depth_disagreement_codes": self.max_depth_disagreement_codes,
            "occlusion_margin_codes": self.occlusion_margin_codes,
            "cluster_gap_codes": self.cluster_gap_codes,
            "confirmation_tolerance_codes": self.confirmation_tolerance_codes,
            "min_confidence": self.min_confidence,
            "min_code": self.min_code,
            "max_code": self.max_code,
        }


#: The policy every caller gets unless it states otherwise.
DEFAULT_POLICY: Final = TemporalPolicy()


# --- one posed observation ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One raw measurement of one other frame, carried into the target camera.

    Everything a decision about it needs, and its full provenance: which frame
    it came from, which sample of that frame, and where the pose composition put
    it. Nothing here is a reconstruction - ``code`` is the source's own raw
    code, and ``predicted_code`` is what that measurement says the target sample
    would have returned.
    """

    frame: int
    source_index: int
    source_u: int
    source_v: int
    code: int
    target_u: float
    target_v: float
    depth: float
    predicted_code: int
    distance: float

    @property
    def support(self) -> tuple[int, int]:
        """How this observation is named as support: its frame and its sample."""
        return (self.frame, self.source_index)


def project_observation(
    camera: CameraModel,
    transform: Rigid,
    *,
    source_u: int,
    source_v: int,
    code: int,
) -> tuple[float, float, float] | None:
    """One raw sample, deprojected, carried through ``transform`` and projected.

    Returns ``(target_u, target_v, depth)`` in the target camera's full-frame
    pixel coordinates and the target's own depth unit, or ``None`` when the
    sample does not project - a code of zero, which is no measurement, or a
    point at or behind either optical centre.

    Public and pure, so the pose composition can be asserted directly. Given
    ``transform`` the identity, a sample comes back at the pixel it started
    from; given a known motion, it comes back where that motion analytically
    puts it, and a composition built the other way round does not.
    """
    if code == NO_RETURN_SAMPLE:
        return None
    in_source = camera.deproject(float(source_u), float(source_v), code)
    if in_source[2] <= 0.0:
        return None
    in_target = transform.apply(in_source)
    if in_target[2] <= 0.0 or not math.isfinite(in_target[2]):
        return None
    target_u, target_v = camera.project(in_target)
    if not (math.isfinite(target_u) and math.isfinite(target_v)):
        return None
    return target_u, target_v, in_target[2]


# --- confidence -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    """What one accepted target sample rests on, carried out of the decision.

    Every field is an input to :func:`temporal_confidence` and to the
    reproduction check, so an accepted sample can be recomputed from its named
    evidence and the policy alone.
    """

    frames: int
    viewpoints: int
    mean_distance: float
    disagreement_codes: float
    pose_quality: float


def temporal_confidence(evidence: TemporalEvidence, policy: TemporalPolicy) -> float:
    """How much this reconstruction is worth, from the evidence that produced it.

    Five terms, each a fraction of 1, combined as their geometric mean so that
    every one of them can veto and none of them can be traded away:

    ``support``
        ``(frames - min_support_frames + 1) / (max_support_frames -
        min_support_frames + 1)``, clamped. More distinct supporting frames
        raises it. The ``+ 1`` is deliberate: the least support the policy
        accepts is still support, and scoring it at exactly zero would make the
        acceptance bound and the confidence floor the same test twice.
    ``diversity``
        ``viewpoints / max_support_frames``, clamped. More genuinely different
        viewpoints among that support raises it; frames that are the same view
        twice do not.
    ``proximity``
        ``1 - mean_distance / (max_reprojection_distance + 1)``. Observations
        landing further from the sample they vote for lowers it.
    ``agreement``
        ``1 - disagreement_codes / max_depth_disagreement_codes``. Independent
        posed measurements disagreeing about the depth lowers it.
    ``pose``
        the worst ``PoseRecord.quality`` among the supporting frames. A
        reconstruction is never worth more than the pose that placed the
        evidence for it.

    Each term is monotone in the direction its name implies, so the geometric
    mean is too: adding a frame, adding a viewpoint, landing nearer, agreeing
    more closely or resting on better-placed poses can only raise the result.
    That monotonicity is what a caller thresholding on confidence depends on,
    and it is tested directly.

    The result is capped at :data:`MAX_RECONSTRUCTION_CONFIDENCE`. It is a
    property of the reconstruction's evidence and **not** a camera confidence:
    it says nothing about the sensor, and no value of it ever makes a
    reconstructed sample a measurement.
    """
    spare = float(max(policy.max_support_frames - policy.min_support_frames, 0) + 1)
    support = _clamp((evidence.frames - policy.min_support_frames + 1) / spare)
    diversity = _clamp(evidence.viewpoints / float(policy.max_support_frames))
    proximity = _clamp(1.0 - evidence.mean_distance / (policy.max_reprojection_distance + 1.0))
    agreement = _clamp(1.0 - evidence.disagreement_codes / policy.max_depth_disagreement_codes)
    pose = _clamp(evidence.pose_quality)
    terms = (support, diversity, proximity, agreement, pose)
    if any(term <= 0.0 for term in terms):
        return 0.0
    value = _clamp(math.exp(math.fsum(math.log(term) for term in terms) / len(terms)))
    return min(value, MAX_RECONSTRUCTION_CONFIDENCE)


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


# --- what a run cost ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalWork:
    """What one temporal completion actually did: its cost, and every refusal.

    Reported so conservatism is inspectable rather than inferred from the
    output. A run that recovered little because the views did not overlap and
    one that recovered little because every observation was refused as occluded
    are different facts, and without these counts they produce the same picture.
    """

    backend: str
    policy: TemporalPolicy
    source_frames: int
    eligible_samples: int
    observations: int
    observation_rejections: Mapping[str, int]
    candidates: int
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    confirmed: int
    contradicted: int
    not_observed: int
    support_samples: int
    work_units: int
    work_budget: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _exact_int(self.schema_version, "a work record's schema version", ResultUnusable)
        if self.schema_version != SCHEMA_VERSION:
            raise ResultUnusable(
                f"work states schema version {self.schema_version!r}; this module writes "
                f"{SCHEMA_VERSION} and a reader that does not know a version refuses rather "
                "than guessing which fields it is looking at"
            )
        if self.backend != PYTHON_BACKEND:
            raise ResultUnusable(
                f"work names backend {self.backend!r}; {PYTHON_BACKEND!r} is the only "
                "implementation in this module and so the only one that can have produced a run"
            )
        if not isinstance(self.policy, TemporalPolicy):
            raise ResultUnusable(
                f"work states policy {self.policy!r}, which is not a TemporalPolicy; the bounds "
                "a run was held to are the validated type or they are unchecked prose"
            )
        for name in (
            "source_frames",
            "eligible_samples",
            "observations",
            "candidates",
            "holes",
            "attempted",
            "accepted",
            "rejected",
            "confirmed",
            "contradicted",
            "not_observed",
            "support_samples",
            "work_units",
            "work_budget",
        ):
            _count(getattr(self, name), name)
        object.__setattr__(
            self,
            "observation_rejections",
            _frozen_counts(self.observation_rejections, OBSERVATION_REJECTIONS, "observation"),
        )
        object.__setattr__(
            self, "rejections", _frozen_counts(self.rejections, REJECTION_REASONS, "hole")
        )
        if self.attempted != self.accepted + self.rejected:
            raise ResultUnusable(
                f"work attempted {self.attempted} hole(s) and accounts for "
                f"{self.accepted + self.rejected}"
            )
        if self.attempted != self.holes:
            raise ResultUnusable(
                f"work attempted {self.attempted} of {self.holes} hole(s); every hole is examined "
                "once, and one that was not looked at is a refusal with no reason"
            )
        if sum(self.rejections.values()) != self.rejected:
            raise ResultUnusable(
                f"work rejected {self.rejected} hole(s) and gives reasons for "
                f"{sum(self.rejections.values())}"
            )
        ceiling = self.observations * self.policy.footprint_cells
        if self.candidates > ceiling:
            raise ResultUnusable(
                f"work deposited {self.candidates} candidate(s) from {self.observations} "
                f"projected observation(s), past the {ceiling} its own footprint of "
                f"{self.policy.footprint_radius} allows. One projection may support every "
                "target sample in its footprint - a point on an exact half-pixel boundary "
                "supports both sides of it - so these are not the same unit and the only "
                "honest relation between them is the footprint's"
            )
        if self.confirmed + self.contradicted > self.accepted:
            raise ResultUnusable(
                f"work confirmed {self.confirmed} and contradicted {self.contradicted} spatial "
                f"estimate(s) beside {self.accepted} temporal reconstruction(s); both verdicts "
                "are reached by independent evidence that settled the sample, so each of them is "
                "one of those reconstructions"
            )
        if self.work_units > self.work_budget:
            raise ResultUnusable(
                f"work spent {self.work_units} unit(s) against a budget of {self.work_budget}; "
                "the budget is derived from the input before the run and is not advisory"
            )
        bounds = (
            self.accepted * self.policy.min_support_frames,
            self.accepted * self.policy.max_support_frames,
        )
        if not bounds[0] <= self.support_samples <= bounds[1]:
            raise ResultUnusable(
                f"work names {self.support_samples} support sample(s) for {self.accepted} "
                f"reconstruction(s), and its own policy allows {bounds[0]}..{bounds[1]}"
            )

    @property
    def within_budget(self) -> bool:
        return self.work_units <= self.work_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "policy": self.policy.as_dict(),
            "source_frames": self.source_frames,
            "eligible_samples": self.eligible_samples,
            "observations": self.observations,
            "observation_rejections": {
                reason: self.observation_rejections[reason] for reason in OBSERVATION_REJECTIONS
            },
            "candidates": self.candidates,
            "holes": self.holes,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections": {reason: self.rejections[reason] for reason in REJECTION_REASONS},
            "confirmed": self.confirmed,
            "contradicted": self.contradicted,
            "not_observed": self.not_observed,
            "support_samples": self.support_samples,
            "work_units": self.work_units,
            "work_budget": self.work_budget,
            "within_budget": self.within_budget,
        }


def _frozen_counts(stated: object, vocabulary: Sequence[str], what: str) -> Mapping[str, int]:
    """One closed count map, copied out of whatever the caller still holds."""
    if not isinstance(stated, Mapping):
        raise ResultUnusable(f"work states {what} rejections {stated!r}, which is not a map")
    for reason, count in stated.items():
        if not isinstance(reason, str):
            raise ResultUnusable(f"work states a {what} rejection reason {reason!r}")
        _count(count, f"{what} rejections[{reason!r}]")
    if sorted(stated) != sorted(vocabulary):
        raise ResultUnusable(
            f"work states {what} rejection reasons that are not exactly the declared vocabulary "
            f"{sorted(vocabulary)}; a run reporting only the reasons that fired hides the shape "
            "of its own conservatism"
        )
    return MappingProxyType({str(reason): int(count) for reason, count in stated.items()})


# --- the derivation ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SpatialView:
    """The hashable part of an accepted spatial result this method actually reads.

    Only the codes, the states and the support of a
    :class:`scanner3d.depth_completion.CompletedPatch` enter the derivation, and
    a ``CompletedPatch`` carries mapping proxies that are not hashable. Reducing
    it to this makes the derivation memoisable on its inputs without giving the
    memo a value that could change under it.
    """

    codes: tuple[int, ...]
    states: bytes
    confidence: tuple[float, ...]
    support: tuple[tuple[int, tuple[int, ...]], ...]

    @classmethod
    def of(cls, spatial: CompletedPatch | None) -> _SpatialView | None:
        if spatial is None:
            return None
        return cls(
            codes=tuple(spatial.codes),
            states=bytes(spatial.state_codes),
            confidence=tuple(float(value) for value in spatial.confidence),
            support=tuple(
                (index, tuple(sources)) for index, sources in sorted(spatial.support.items())
            ),
        )

    def support_at(self, index: int) -> tuple[int, ...]:
        for at, sources in self.support:
            if at == index:
                return sources
        return ()


@dataclass(frozen=True, slots=True)
class _Temporal:
    """Everything one target, one set of sources and one policy deterministically produce."""

    codes: tuple[int, ...]
    states: bytes
    confidence: tuple[float, ...]
    confirmations: bytes
    temporal_support: Mapping[int, tuple[tuple[int, int], ...]]
    spatial_support: Mapping[int, tuple[int, ...]]
    accepted: int
    confirmed: int
    contradicted: int
    not_observed: int
    support_samples: int
    observations: int
    candidates: int
    work_units: int
    rejections: Mapping[str, int]
    observation_rejections: Mapping[str, int]


@dataclass(slots=True)
class _Tally:
    """Mutable bookkeeping for one run. Never escapes :func:`_derive`."""

    observations: int = 0
    candidates: int = 0
    work_units: int = 0
    rejections: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REJECTION_REASONS, 0))
    observation_rejections: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(OBSERVATION_REJECTIONS, 0)
    )

    def reject(self, reason: str) -> None:
        self.rejections[reason] += 1

    def discard(self, reason: str) -> None:
        self.observation_rejections[reason] += 1


def _deposit(
    target: TemporalFrame,
    sources: Sequence[TemporalFrame],
    camera: CameraModel,
    policy: TemporalPolicy,
    tally: _Tally,
) -> dict[int, list[Observation]]:
    """Every eligible source sample, projected into the target and filed by sample.

    The first of the two passes. Nothing is judged here beyond whether the
    projection exists at all and lands somewhere this call could use: occlusion
    needs every observation to have been deposited before any of them can be
    called hidden, and deciding it while depositing would let iteration order
    choose the answer.
    """
    deposited: dict[int, list[Observation]] = {}
    footprint = range(-policy.footprint_radius, policy.footprint_radius + 1)
    # One past the cap, so a target that is over it is recognisable as over it
    # without the list of what landed there being unbounded.
    ceiling = policy.max_candidates_per_target + 1
    for source in sources:
        transform = temporal_transform(target.pose, source.pose)
        for source_index in source.measured_indices():
            tally.work_units += policy.footprint_cells
            code = source.patch.codes[source_index]
            source_u, source_v = source.pixel_of(source_index)
            # The arithmetic project_observation performs, with each refusal
            # named rather than collapsed into one None: which of them fired is
            # the difference between two views that do not overlap and two views
            # whose geometry was carried the wrong way round.
            in_source = camera.deproject(float(source_u), float(source_v), code)
            if in_source[2] <= 0.0:  # pragma: no cover - a measured code is positive
                tally.discard("source_behind_camera")
                continue
            in_target = transform.apply(in_source)
            if not math.isfinite(in_target[2]) or in_target[2] <= 0.0:
                tally.discard("behind_target_camera")
                continue
            target_u, target_v = camera.project(in_target)
            if not (math.isfinite(target_u) and math.isfinite(target_v)):
                tally.discard("behind_target_camera")  # pragma: no cover - z > 0 keeps it finite
                continue
            depth = in_target[2]
            tally.observations += 1
            landed = refused_distance = False
            for step_v in footprint:
                for step_u in footprint:
                    u = round(target_u) + step_u
                    v = round(target_v) + step_v
                    index = target.index_of(u, v)
                    if index is None:
                        continue
                    distance = math.hypot(target_u - u, target_v - v)
                    if distance > policy.max_reprojection_distance:
                        refused_distance = True
                        continue
                    landed = True
                    if len(deposited.get(index, ())) >= ceiling:
                        continue
                    deposited.setdefault(index, []).append(
                        Observation(
                            frame=source.index,
                            source_index=source_index,
                            source_u=source_u,
                            source_v=source_v,
                            code=code,
                            target_u=target_u,
                            target_v=target_v,
                            depth=depth,
                            predicted_code=round(camera.code_of(depth)),
                            distance=distance,
                        )
                    )
            if not landed:
                tally.discard(
                    "reprojection_too_far" if refused_distance else "outside_target_patch"
                )
    return deposited


def _nearest_depths(
    target: TemporalFrame,
    camera: CameraModel,
    deposited: Mapping[int, Sequence[Observation]],
) -> dict[int, float]:
    """The nearest evidence at every target sample: measured first, then projected.

    The z-buffer. A target sample the sensor returned states its own depth, and
    that depth is nearer evidence than anything projected onto it; a sample it
    did not return takes the smallest depth among the observations deposited
    there. Built from the whole deposit rather than incrementally, so no
    farther surface can vote through nearer measured geometry and no ordering
    of the projection loop can decide which observation wins.
    """
    nearest: dict[int, float] = {}
    for index in range(target.patch.sample_count):
        code = target.patch.codes[index]
        if code != NO_RETURN_SAMPLE:
            nearest[index] = code * camera.depth_scale
    for index, observations in deposited.items():
        depths = [observation.depth for observation in observations]
        here = nearest.get(index)
        nearest[index] = min(depths) if here is None else min(here, min(depths))
    return nearest


def _representatives(
    observations: Sequence[Observation], policy: TemporalPolicy
) -> list[Observation]:
    """One observation per distinct source frame: the nearest, then the lowest index.

    Nearest rather than averaged, because two observations of one frame landing
    on one target sample are two surfaces that frame saw along nearly the same
    ray, and the nearer one is the one that is not behind the other. The
    tie-break is a total order on facts the observation carries, so the choice
    is a function of the input and not of the iteration.
    """
    by_frame: dict[int, Observation] = {}
    for observation in sorted(
        observations, key=lambda item: (item.frame, item.depth, item.source_index)
    ):
        by_frame.setdefault(observation.frame, observation)
    chosen = sorted(by_frame.values(), key=lambda item: item.frame)
    return chosen[: policy.max_support_frames]


def _viewpoint_groups(
    target: TemporalFrame,
    sources: Mapping[int, TemporalFrame],
    frames: Sequence[int],
    policy: TemporalPolicy,
) -> int:
    """How many genuinely different viewpoints a set of source frames is.

    Greedy, in frame order, so the grouping is deterministic: each frame joins
    the first group whose representative it is not separated from, and starts a
    new one otherwise. Two frames whose transforms into the target differ by
    less than the policy's baseline *and* less than its rotation are one
    viewpoint however many frame indices they occupy - which is what stops a
    stationary camera's second frame from being counted as a second view.
    """
    groups: list[Rigid] = []
    for frame in frames:
        transform = temporal_transform(target.pose, sources[frame].pose)
        for representative in groups:
            baseline, degrees = viewpoint_separation(representative, transform)
            if baseline < policy.min_baseline and degrees < policy.min_rotation_degrees:
                break
        else:
            groups.append(transform)
    return len(groups)


def _neighbour_step(target: TemporalFrame, index: int) -> float:
    """How wide a depth step the target's own measured neighbours straddle, in codes.

    Zero when fewer than two of the four neighbours returned. A hole whose own
    neighbours disagree by more than
    :attr:`TemporalPolicy.cluster_gap_codes` is a discontinuity - a silhouette,
    a disocclusion or a grazing edge - and which surface it belongs to is
    exactly the question the evidence around it cannot settle.
    """
    column, row = target.patch.position_of(index)
    codes: list[int] = []
    for step_column, step_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = target.patch.index_of(column + step_column, row + step_row)
        if neighbour is None:
            continue
        code = target.patch.codes[neighbour]
        if code != NO_RETURN_SAMPLE:
            codes.append(code)
    if len(codes) < 2:
        return 0.0
    return float(max(codes) - min(codes))


def _resolve(
    target: TemporalFrame,
    sources: Mapping[int, TemporalFrame],
    policy: TemporalPolicy,
    observations: Sequence[Observation],
    index: int,
    tally: _Tally,
) -> tuple[int, float, tuple[tuple[int, int], ...]] | None:
    """One target hole, taken as far as the policy allows. ``None`` stays missing.

    Every refusal is counted against its own reason before returning, so the
    reason a frame came back sparse is in the work record and not only in the
    absence of a value.
    """
    if not observations:
        tally.reject("no_posed_evidence")
        return None
    if len(observations) > policy.max_candidates_per_target:
        tally.reject("too_many_candidates")
        return None
    chosen = _representatives(observations, policy)
    frames = [observation.frame for observation in chosen]
    if len(frames) < policy.min_support_frames:
        tally.reject("insufficient_support_frames")
        return None
    viewpoints = _viewpoint_groups(target, sources, frames, policy)
    if viewpoints < policy.min_distinct_viewpoints:
        tally.reject("insufficient_viewpoint_diversity")
        return None
    codes = [float(observation.predicted_code) for observation in chosen]
    disagreement = max(codes) - min(codes)
    if disagreement > policy.max_depth_disagreement_codes:
        tally.reject("conflicting_depth_evidence")
        return None
    if _neighbour_step(target, index) > policy.cluster_gap_codes:
        tally.reject("edge_discontinuity")
        return None
    predicted = round(math.fsum(codes) / len(codes))
    if not policy.min_code <= predicted <= policy.max_code:
        tally.reject("prediction_out_of_range")
        return None
    evidence = TemporalEvidence(
        frames=len(frames),
        viewpoints=viewpoints,
        mean_distance=math.fsum(observation.distance for observation in chosen) / len(chosen),
        disagreement_codes=disagreement,
        pose_quality=min(sources[frame].pose.quality for frame in frames),
    )
    confidence = temporal_confidence(evidence, policy)
    if confidence < policy.min_confidence:
        tally.reject("confidence_below_threshold")
        return None
    return predicted, confidence, tuple(sorted(observation.support for observation in chosen))


def _derive(
    target: TemporalFrame,
    sources: tuple[TemporalFrame, ...],
    camera: CameraModel,
    policy: TemporalPolicy,
    spatial: _SpatialView | None,
) -> _Temporal:
    """The two passes, over immutable inputs. Pure.

    Reads the target patch, the source patches, the camera, the poses, the
    policy and the accepted spatial view, and nothing else - no clock, no
    randomness, no I/O and, crucially, none of the values it is producing.
    Support comes from the source frames' own raw codes, so no reconstruction is
    ever evidence for another and no sample of the target frame is ever evidence
    for a temporal claim about the target frame.
    """
    samples = target.patch.sample_count
    codes = list(target.patch.codes)
    states = bytearray(
        STATE_MEASURED if target.patch.is_measured(index) else STATE_MISSING
        for index in range(samples)
    )
    confidence = [1.0 if target.patch.is_measured(index) else 0.0 for index in range(samples)]
    confirmations = bytearray(CONFIRMATION_NO_ESTIMATE for _ in range(samples))
    temporal_support: dict[int, tuple[tuple[int, int], ...]] = {}
    spatial_support: dict[int, tuple[int, ...]] = {}
    if spatial is not None:
        for index in range(samples):
            if spatial.states[index] != depth_completion.STATE_SPATIALLY_RECONSTRUCTED:
                continue
            codes[index] = spatial.codes[index]
            states[index] = STATE_SPATIALLY_RECONSTRUCTED
            confidence[index] = spatial.confidence[index]
            spatial_support[index] = spatial.support_at(index)
    tally = _Tally()
    by_index = {frame.index: frame for frame in sources}
    deposited = _deposit(target, sources, camera, policy, tally)
    nearest = _nearest_depths(target, camera, deposited)
    visible: dict[int, list[Observation]] = {}
    for index in sorted(deposited):
        surviving: list[Observation] = []
        for observation in deposited[index]:
            if target.patch.is_measured(index):
                tally.discard("target_already_measured")
                continue
            if not policy.min_code <= observation.predicted_code <= policy.max_code:
                tally.discard("predicted_code_out_of_range")
                continue
            floor = nearest.get(index)
            if (
                floor is not None
                and camera.code_of(observation.depth - floor) > policy.occlusion_margin_codes
            ):
                tally.discard("occluded_by_nearer_evidence")
                continue
            surviving.append(observation)
        tally.candidates += len(surviving)
        if surviving:
            visible[index] = surviving
    accepted = confirmed = contradicted = not_observed = support_samples = 0
    for index in target.patch.hole_indices():
        # One unit for examining the hole, then one for every candidate that
        # was deposited at it - including the single over-cap sentinel _deposit
        # retains, which the visibility pass above examined like any other. The
        # cap here is the deposit ceiling, not max_candidates_per_target, so
        # the published units cover exactly the candidates this run processed.
        tally.work_units += 1 + min(
            len(deposited.get(index, ())), policy.max_candidates_per_target + 1
        )
        observations = visible.get(index, [])
        if not observations and deposited.get(index):
            tally.reject("all_evidence_refused")
            resolved = None
        else:
            resolved = _resolve(target, by_index, policy, observations, index, tally)
        proposed = spatial is not None and states[index] == STATE_SPATIALLY_RECONSTRUCTED
        if resolved is None:
            if proposed:
                confirmations[index] = CONFIRMATION_NOT_OBSERVED
                not_observed += 1
            continue
        code, value, support = resolved
        if proposed:
            if abs(code - codes[index]) > policy.confirmation_tolerance_codes:
                # Not averaged, on purpose. A value halfway between two
                # incompatible answers is evidence of neither, so the spatial
                # estimate is overruled rather than blended: what is published
                # is the independent posed measurement, whole, and the fact
                # that a one-frame hypothesis said something else is published
                # beside it as CONTRADICTED.
                #
                # Overruled rather than kept, because the evidence here is the
                # same evidence a run with no spatial input has at this sample,
                # and that run reconstructs it. Letting a weaker same-frame
                # guess veto a well-supported independent measurement would
                # make the stage's output depend on whether an earlier stage
                # happened to guess, which is not a property of the evidence.
                # The spatial value is not carried anywhere: the code, the
                # confidence and the support all come from the other frames,
                # and the estimate's own support stays named so the reader can
                # see exactly which hypothesis was overruled.
                confirmations[index] = CONFIRMATION_CONTRADICTED
                contradicted += 1
            else:
                confirmations[index] = CONFIRMATION_CONFIRMED
                confirmed += 1
        codes[index] = code
        states[index] = STATE_TEMPORALLY_RECONSTRUCTED
        confidence[index] = value
        temporal_support[index] = support
        accepted += 1
        support_samples += len(support)
    for index in list(spatial_support):
        if states[index] not in (STATE_SPATIALLY_RECONSTRUCTED, STATE_TEMPORALLY_RECONSTRUCTED):
            del spatial_support[index]  # pragma: no cover - no path clears a spatial estimate
    return _Temporal(
        codes=tuple(codes),
        states=bytes(states),
        confidence=tuple(confidence),
        confirmations=bytes(confirmations),
        temporal_support=MappingProxyType(temporal_support),
        spatial_support=MappingProxyType(spatial_support),
        accepted=accepted,
        confirmed=confirmed,
        contradicted=contradicted,
        not_observed=not_observed,
        support_samples=support_samples,
        observations=tally.observations,
        candidates=tally.candidates,
        work_units=tally.work_units,
        rejections=MappingProxyType(dict(tally.rejections)),
        observation_rejections=MappingProxyType(dict(tally.observation_rejections)),
    )


#: How many derivations are remembered at once. Two, deliberately: the only hit
#: that has to land is the one a result's own validation makes immediately after
#: :func:`complete_temporally` derived the same thing, so a completion costs one
#: pass rather than two.
_REMEMBERED_DERIVATIONS: Final = 2


@functools.lru_cache(maxsize=_REMEMBERED_DERIVATIONS)
def _temporal_of(
    target: TemporalFrame,
    sources: tuple[TemporalFrame, ...],
    camera: CameraModel,
    policy: TemporalPolicy,
    spatial: _SpatialView | None,
) -> _Temporal:
    """:func:`_derive`, memoised on the five immutable values it reads.

    Sound only because every argument is frozen, hashable and deeply copied at
    construction and because :func:`_derive` is pure. Callers must have checked
    :data:`MAX_WORK_UNITS` against ``policy.worst_case_work`` before arriving
    here; this function does no work bounding of its own.
    """
    return _derive(target, sources, camera, policy, spatial)


# --- the result -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalPatch:
    """One target rectangle after temporal evidence: code, state, confidence, support.

    The whole rectangle, always. Every array here is exactly ``width * height``
    long and every index in it is accounted for, so a caller can count what
    happened against a denominator it fixed rather than one the completion
    chose.

    The sources are the point
    -------------------------
    :attr:`target`, :attr:`sources`, :attr:`camera`, :attr:`spatial` and
    :attr:`work.policy` are *fields* rather than construction-time arguments
    that are thrown away, because without them a result could only be checked
    against its own labels - and a label is exactly what a forged result would
    change. Calling a sample ``"temporally_reconstructed"`` has said nothing
    about whether another frame measured anything, and accepting a support
    reference because the result says that frame returned something there is
    letting a claim vouch for its own evidence.

    So every rule below is stated against the immutable raw patches:

    * a target sample the sensor returned comes back with the same code, the
      state ``"measured"``, confidence ``1.0`` and no support - byte-for-byte
      identity, not similarity;
    * a target sample the sensor did not return is exactly missing, or the
      accepted spatial estimate for it unchanged, or a temporal reconstruction
      with a nonzero code, a confidence strictly between zero and
      :data:`MAX_RECONSTRUCTION_CONFIDENCE`, and distinct support references
      naming *other* frames at samples whose own raw codes are nonzero;
    * a confirmation state is published for every sample, and the three that
      say something about a spatial estimate exist only where the accepted
      spatial result actually made one.

    And the shape is not enough. A result whose every label and index is legal
    can still carry a reconstructed code, a confidence, a support selection, a
    confirmation or a work total that this evidence does not give. So the whole
    run is re-derived from the immutable inputs and the published policy, and
    the result is required to *be* that derivation.
    """

    width: int
    height: int
    codes: tuple[int, ...]
    state_codes: bytes
    confidence: tuple[float, ...]
    confirmation_codes: bytes
    temporal_support: Mapping[int, tuple[tuple[int, int], ...]]
    spatial_support: Mapping[int, tuple[int, ...]]
    work: TemporalWork
    #: The exact frames, camera and accepted spatial result this is derived
    #: from. Immutable, and the authority for every provenance claim made here.
    target: TemporalFrame
    sources: tuple[TemporalFrame, ...]
    camera: CameraModel
    spatial: CompletedPatch | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _exact_int(self.schema_version, "a result's schema version", ResultUnusable)
        if self.schema_version != SCHEMA_VERSION:
            raise ResultUnusable(
                f"a result states schema version {self.schema_version!r}; this module writes "
                f"{SCHEMA_VERSION}"
            )
        _exact_int(self.width, "a result's width", ResultUnusable)
        _exact_int(self.height, "a result's height", ResultUnusable)
        target = self.target
        if not isinstance(target, TemporalFrame):
            raise ResultUnusable(
                f"a result names target {target!r}, which is not a TemporalFrame. A result that "
                "carries no source can only be checked against its own labels"
            )
        if not isinstance(self.camera, CameraModel):
            raise ResultUnusable(f"a result names camera {self.camera!r}, which is not one")
        sources = self.sources
        if not isinstance(sources, tuple) or not all(
            isinstance(frame, TemporalFrame) for frame in sources
        ):
            raise ResultUnusable(
                f"a result names sources {sources!r}; they are the immutable frames the temporal "
                "evidence came from and they are checked, not described"
            )
        _ensure_frame_set(target, sources)
        samples = target.patch.sample_count
        if (self.width, self.height) != (target.patch.width, target.patch.height):
            raise ResultUnusable(
                f"a {self.width}x{self.height} result claims to complete a "
                f"{target.patch.width}x{target.patch.height} patch; the denominator is the "
                "rectangle the caller handed over and a completion does not get to move it"
            )
        object.__setattr__(self, "codes", _frozen_codes(self.codes))
        object.__setattr__(self, "state_codes", _frozen_bytes(self.state_codes, "states"))
        object.__setattr__(
            self, "confirmation_codes", _frozen_bytes(self.confirmation_codes, "confirmations")
        )
        object.__setattr__(self, "confidence", _frozen_confidence(self.confidence))
        object.__setattr__(
            self, "temporal_support", _frozen_temporal_support(self.temporal_support)
        )
        object.__setattr__(self, "spatial_support", _frozen_spatial_support(self.spatial_support))
        counts = (
            len(self.codes),
            len(self.state_codes),
            len(self.confidence),
            len(self.confirmation_codes),
        )
        if counts != (samples,) * 4:
            raise ResultUnusable(
                f"a {self.width}x{self.height} result carries {counts} value(s) for {samples} "
                "sample(s); a result that is not the whole rectangle cannot be counted against it"
            )
        for index, state in enumerate(self.state_codes):
            if state >= len(STATE_NAMES):
                raise ResultUnusable(f"a result carries state {state!r} at index {index}")
        for index, confirmation in enumerate(self.confirmation_codes):
            if confirmation >= len(CONFIRMATIONS):
                raise ResultUnusable(
                    f"a result carries confirmation {confirmation!r} at index {index}"
                )
        for named in (self.temporal_support, self.spatial_support):
            for index in named:
                if index not in range(samples):
                    raise ResultUnusable(
                        f"a result names support for index {index}, which it has not"
                    )
        self._bind_to_sources(samples)
        self._bind_to_spatial(samples)
        self._bind_to_work(samples)
        self._bind_to_replay()

    def _bind_to_sources(self, samples: int) -> None:
        """Every sample of the result, against the raw samples it came from.

        Nothing here reads one of the result's own labels as evidence for
        anything: what a sample is allowed to be follows from the immutable raw
        codes, and the labels are then required to agree with that.
        """
        target = self.target
        by_index = {frame.index: frame for frame in self.sources}
        for index in range(samples):
            raw = target.patch.codes[index]
            state = self.state_codes[index]
            code = self.codes[index]
            value = float(self.confidence[index])
            support = self.temporal_support.get(index, ())
            if raw != NO_RETURN_SAMPLE:
                if state != STATE_MEASURED:
                    raise ResultUnusable(
                        f"a result calls index {index} {self.state_at(index)!r}, and the target "
                        f"frame returned code {raw} there. A measurement stays a measurement"
                    )
                if code != raw:
                    raise ResultUnusable(
                        f"a result carries code {code} at index {index}, where the target frame "
                        f"carries {raw}. Raw measurements come back byte-for-byte or this is not "
                        "a completion"
                    )
                if value != 1.0:
                    raise ResultUnusable(
                        f"a result publishes measured index {index} at confidence {value!r}; a "
                        "sample the sensor returned is carried at 1.0 and nothing else"
                    )
                if support or index in self.spatial_support:
                    raise ResultUnusable(
                        f"a result names support for measured index {index}; a measurement rests "
                        "on the sensor and not on another frame"
                    )
                continue
            if state == STATE_MISSING:
                if code != NO_RETURN_SAMPLE or value != 0.0 or support:
                    raise ResultUnusable(
                        f"a result calls index {index} missing and carries code {code!r} at "
                        f"confidence {value!r} with {len(support)} support sample(s); a refused "
                        "hole is exactly code zero, confidence zero and no evidence"
                    )
                continue
            if code == NO_RETURN_SAMPLE:
                raise ResultUnusable(
                    f"a result reconstructs index {index} and carries code zero there, which is "
                    "no measurement rather than a depth"
                )
            if not 0.0 < value <= MAX_RECONSTRUCTION_CONFIDENCE:
                raise ResultUnusable(
                    f"a result publishes reconstructed index {index} at confidence {value!r}; a "
                    f"reconstruction is worth more than nothing and at most "
                    f"{MAX_RECONSTRUCTION_CONFIDENCE}, because the 1.0 above it is reserved for "
                    "what the sensor returned"
                )
            if state == STATE_SPATIALLY_RECONSTRUCTED:
                if support:
                    raise ResultUnusable(
                        f"a result calls index {index} spatially reconstructed and names "
                        f"{len(support)} other-frame sample(s) for it; support from another frame "
                        "is what a temporal reconstruction is"
                    )
                continue
            if state != STATE_TEMPORALLY_RECONSTRUCTED:  # pragma: no cover - states are closed
                raise ResultUnusable(f"a result carries state {state!r} at index {index}")
            if not support:
                raise ResultUnusable(
                    f"a result reconstructs index {index} temporally and names no support for "
                    "it; a value with no named evidence is invention, whatever its confidence says"
                )
            if len(set(support)) != len(support):
                raise ResultUnusable(
                    f"a result supports index {index} with the same sample twice; counting one "
                    "measurement as two is how thin evidence is made to look like enough"
                )
            named_frames = {frame for frame, _sample in support}
            if len(named_frames) < self.work.policy.min_support_frames:
                raise ResultUnusable(
                    f"a result supports index {index} from {len(named_frames)} distinct frame(s), "
                    f"and its own policy requires {self.work.policy.min_support_frames}"
                )
            for frame_index, source_index in support:
                if frame_index == self.target.index:
                    raise ResultUnusable(
                        f"a result supports temporal index {index} with frame "
                        f"{frame_index}, which is the target's own frame. A same-frame sample is "
                        "not independent temporal evidence, whatever the state says"
                    )
                frame = by_index.get(frame_index)
                if frame is None:
                    raise ResultUnusable(
                        f"a result supports index {index} with frame {frame_index}, which it does "
                        "not name as a source"
                    )
                if source_index not in range(frame.patch.sample_count):
                    raise ResultUnusable(
                        f"a result supports index {index} with sample {source_index} of frame "
                        f"{frame_index}, which that frame has not"
                    )
                if frame.patch.codes[source_index] == NO_RETURN_SAMPLE:
                    raise ResultUnusable(
                        f"a result supports index {index} with sample {source_index} of frame "
                        f"{frame_index}, where that frame returned nothing. Only an original raw "
                        "measurement is temporal evidence, and neither a reconstruction nor a "
                        "label saying otherwise is"
                    )

    def _bind_to_spatial(self, samples: int) -> None:
        """Every confirmation state, against the accepted spatial result it describes."""
        spatial = self.spatial
        if spatial is not None and not isinstance(spatial, CompletedPatch):
            raise ResultUnusable(
                f"a result names spatial estimate {spatial!r}, which is not a CompletedPatch; the "
                "accepted spatial result is the validated type or it is an unchecked claim"
            )
        if spatial is not None and spatial.source != self.target.patch:
            raise ResultUnusable(
                "a result names an accepted spatial estimate derived from a different raw patch "
                "than the target frame it completes; a confirmation is about this frame or it is "
                "about nothing"
            )
        for index in range(samples):
            confirmation = self.confirmation_codes[index]
            proposed = (
                spatial is not None
                and spatial.state_codes[index] == depth_completion.STATE_SPATIALLY_RECONSTRUCTED
            )
            if not proposed:
                if confirmation != CONFIRMATION_NO_ESTIMATE:
                    raise ResultUnusable(
                        f"a result calls index {index} {self.confirmation_at(index)!r} and no "
                        "accepted spatial estimate proposed a value there; confirming, "
                        "contradicting or failing to observe something nobody claimed is not a "
                        "state this module has"
                    )
                if index in self.spatial_support:
                    raise ResultUnusable(
                        f"a result names spatial support for index {index}, where no accepted "
                        "spatial estimate proposed a value"
                    )
                continue
            if confirmation == CONFIRMATION_NO_ESTIMATE:
                raise ResultUnusable(
                    f"a result calls index {index} 'no_spatial_estimate' beside an accepted "
                    "spatial result that estimated it; a spatial hypothesis is not made safe by "
                    "a temporal stage forgetting it existed"
                )
            assert spatial is not None
            if self.spatial_support.get(index, ()) != tuple(spatial.support_at(index)):
                raise ResultUnusable(
                    f"a result names spatial support {self.spatial_support.get(index, ())} for "
                    f"index {index}, and the accepted spatial result it carries names "
                    f"{tuple(spatial.support_at(index))}"
                )
            state = self.state_codes[index]
            if confirmation == CONFIRMATION_CONFIRMED and state != STATE_TEMPORALLY_RECONSTRUCTED:
                raise ResultUnusable(
                    f"a result confirms index {index} and calls it {self.state_at(index)!r}; a "
                    "confirmed estimate has independent other-frame support and carries the "
                    "provenance that support gives it"
                )
            if (
                confirmation == CONFIRMATION_CONTRADICTED
                and state != STATE_TEMPORALLY_RECONSTRUCTED
            ):
                raise ResultUnusable(
                    f"a result contradicts index {index} and calls it {self.state_at(index)!r}; a "
                    "contradiction is reached by independent posed measurements that settled the "
                    "sample themselves, and what is published there is theirs"
                )
            if confirmation == CONFIRMATION_NOT_OBSERVED and state != STATE_SPATIALLY_RECONSTRUCTED:
                raise ResultUnusable(
                    f"a result calls index {index} {self.confirmation_at(index)!r} and "
                    f"{self.state_at(index)!r}; an estimate no independent evidence observed "
                    "keeps its spatial provenance explicitly, and is not quietly relabelled"
                )
            if state == STATE_SPATIALLY_RECONSTRUCTED and self.codes[index] != spatial.codes[index]:
                raise ResultUnusable(
                    f"a result carries code {self.codes[index]} at unobserved index {index}, "
                    f"where the accepted spatial estimate says {spatial.codes[index]}. An "
                    "estimate nothing independent looked at is republished unchanged or it is a "
                    "different claim wearing its provenance"
                )
            if (
                confirmation == CONFIRMATION_CONTRADICTED
                and abs(self.codes[index] - spatial.codes[index])
                <= self.work.policy.confirmation_tolerance_codes
            ):
                raise ResultUnusable(
                    f"a result contradicts index {index} with code {self.codes[index]}, which is "
                    f"within its own confirmation tolerance of the spatial estimate's "
                    f"{spatial.codes[index]}. A contradiction is published, never averaged: a "
                    "value between two incompatible answers is evidence of neither"
                )
            if state == STATE_SPATIALLY_RECONSTRUCTED and float(self.confidence[index]) != float(
                spatial.confidence[index]
            ):
                raise ResultUnusable(
                    f"a result publishes unobserved index {index} at confidence "
                    f"{float(self.confidence[index])!r}, where the accepted spatial estimate says "
                    f"{float(spatial.confidence[index])!r}"
                )

    def _bind_to_work(self, samples: int) -> None:
        """The work record, against the frames, the result and the published policy."""
        work = self.work
        if not isinstance(work, TemporalWork):
            raise ResultUnusable(f"a result carries work {work!r}, which is not a TemporalWork")
        holes = sum(1 for code in self.target.patch.codes if code == NO_RETURN_SAMPLE)
        if work.holes != holes or work.attempted != holes:
            raise ResultUnusable(
                f"a result's work faced {work.holes} hole(s) and attempted {work.attempted} of "
                f"them, beside a target frame with {holes}. How many holes there were is the "
                "input's fact and not the run's"
            )
        if work.source_frames != len(self.sources):
            raise ResultUnusable(
                f"a result's work names {work.source_frames} source frame(s) beside "
                f"{len(self.sources)}"
            )
        eligible = sum(len(frame.measured_indices()) for frame in self.sources)
        if work.eligible_samples != eligible:
            raise ResultUnusable(
                f"a result's work names {work.eligible_samples} eligible sample(s) beside source "
                f"frames that returned {eligible}. Which samples are evidence is the raw frames' "
                "fact and not the run's"
            )
        reconstructed = sum(
            1 for state in self.state_codes if state == STATE_TEMPORALLY_RECONSTRUCTED
        )
        if work.accepted != reconstructed or work.rejected != holes - reconstructed:
            raise ResultUnusable(
                f"a result carries {reconstructed} temporal reconstruction(s) beside work that "
                f"accepted {work.accepted} and rejected {work.rejected} of {holes}"
            )
        named = sum(len(support) for support in self.temporal_support.values())
        if named != work.support_samples:
            raise ResultUnusable(
                f"a result names {named} support sample(s) and its own work states "
                f"{work.support_samples}"
            )
        for name, code in (
            ("confirmed", CONFIRMATION_CONFIRMED),
            ("contradicted", CONFIRMATION_CONTRADICTED),
            ("not_observed", CONFIRMATION_NOT_OBSERVED),
        ):
            counted = sum(1 for value in self.confirmation_codes if value == code)
            if getattr(work, name) != counted:
                raise ResultUnusable(
                    f"a result's work states {getattr(work, name)} {name} spatial estimate(s) and "
                    f"the result carries {counted}"
                )
        budget = work.policy.worst_case_work(eligible, holes)
        if work.work_budget != budget:
            raise ResultUnusable(
                f"a result's work states a budget of {work.work_budget} for {eligible} eligible "
                f"sample(s) and {holes} hole(s), which derives {budget}. The budget is the input "
                "and the policy, not a number the run chose"
            )
        if budget > MAX_WORK_UNITS:
            raise ResultUnusable(
                f"a result's work derives a worst case of {budget} unit(s), past the ceiling of "
                f"{MAX_WORK_UNITS} this module states"
            )
        if samples > MAX_PATCH_SAMPLES:  # pragma: no cover - DepthPatch refuses this first
            raise ResultUnusable(f"a result of {samples} sample(s) is past the patch ceiling")

    def _bind_to_replay(self) -> None:
        """Every published value, against what these exact inputs and policy derive.

        The rules above establish that a result has a *shape* the raw frames
        permit. None of that says the published number is the one this method
        produces: a result can satisfy every one of them and still carry a
        reconstructed code the evidence does not give, a confidence the formula
        does not give, a legal-but-different support selection, a confirmation
        the comparison does not reach, or a work record that spent nothing.

        So the whole run is re-derived here from the immutable frames, camera,
        accepted spatial result and published policy, and the result is required
        to *be* that derivation. Nothing here reads the result: the derivation is
        a pure function of values the result cannot change, so a forged field is
        refused by recomputation rather than by consistency with its neighbours.
        """
        derived = _temporal_of(
            self.target,
            self.sources,
            self.camera,
            self.work.policy,
            _SpatialView.of(self.spatial),
        )
        for index in range(self.target.patch.sample_count):
            if self.state_codes[index] != derived.states[index]:
                raise ResultUnusable(
                    f"a result calls index {index} {self.state_at(index)!r}, and these frames "
                    f"under the policy it publishes derive "
                    f"{STATE_NAMES[derived.states[index]]!r}. Which holes independent posed "
                    "evidence settles is a consequence of the frames and the bounds, not a "
                    "decision a result reports"
                )
            if self.codes[index] != derived.codes[index]:
                raise ResultUnusable(
                    f"a result carries code {self.codes[index]} at index {index}, where these "
                    f"frames under the policy it publishes derive {derived.codes[index]}"
                )
            if float(self.confidence[index]) != derived.confidence[index]:
                raise ResultUnusable(
                    f"a result publishes index {index} at confidence "
                    f"{float(self.confidence[index])!r}, where these frames under the policy it "
                    f"publishes derive {derived.confidence[index]!r}. Confidence is computed from "
                    "named evidence and is not a number a result may state"
                )
            if self.confirmation_codes[index] != derived.confirmations[index]:
                raise ResultUnusable(
                    f"a result calls index {index} {self.confirmation_at(index)!r}, where these "
                    f"frames under the policy it publishes derive "
                    f"{CONFIRMATIONS[derived.confirmations[index]]!r}"
                )
            named = self.temporal_support.get(index, ())
            expected = derived.temporal_support.get(index, ())
            if named != expected:
                raise ResultUnusable(
                    f"a result supports index {index} with {named}, where these frames under the "
                    f"policy it publishes select {expected}. Changed, omitted and extra support "
                    "are all refused: a result that got to choose its own evidence could name "
                    "support for a value that evidence does not give"
                )
            if self.spatial_support.get(index, ()) != derived.spatial_support.get(index, ()):
                raise ResultUnusable(
                    f"a result names spatial support {self.spatial_support.get(index, ())} at "
                    f"index {index}, where the accepted spatial result gives "
                    f"{derived.spatial_support.get(index, ())}"
                )
        work = self.work
        for name, value in (
            ("work_units", derived.work_units),
            ("observations", derived.observations),
            ("candidates", derived.candidates),
        ):
            if getattr(work, name) != value:
                raise ResultUnusable(
                    f"a result's work states {name} {getattr(work, name)}, and these frames under "
                    f"the policy it publishes spend {value}. What a run cost is what it did, not "
                    "a total that merely fits under the budget"
                )
        for reason in REJECTION_REASONS:
            if work.rejections[reason] != derived.rejections[reason]:
                raise ResultUnusable(
                    f"a result's work refuses {work.rejections[reason]} hole(s) for {reason!r}, "
                    f"and these frames under the policy it publishes refuse "
                    f"{derived.rejections[reason]}. Every refusal is counted against the reason "
                    "that actually fired; a taxonomy that only adds up to the right total says "
                    "nothing about why a frame came back sparse"
                )
        for reason in OBSERVATION_REJECTIONS:
            if work.observation_rejections[reason] != derived.observation_rejections[reason]:
                raise ResultUnusable(
                    f"a result's work discards {work.observation_rejections[reason]} "
                    f"observation(s) for {reason!r}, and these frames under the policy it "
                    f"publishes discard {derived.observation_rejections[reason]}"
                )

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    def state_at(self, index: int) -> str:
        """The provenance of one sample, by name."""
        return STATE_NAMES[self.state_codes[index]]

    def states(self) -> tuple[str, ...]:
        """Every provenance state, by name, in index order."""
        return tuple(STATE_NAMES[state] for state in self.state_codes)

    def confirmation_at(self, index: int) -> str:
        """What independent posed evidence said about one spatial estimate, by name."""
        return CONFIRMATIONS[self.confirmation_codes[index]]

    def confirmations(self) -> tuple[str, ...]:
        """Every confirmation state, by name, in index order."""
        return tuple(CONFIRMATIONS[value] for value in self.confirmation_codes)

    def support_at(self, index: int) -> tuple[tuple[int, int], ...]:
        """The other-frame raw samples one temporal reconstruction rests on."""
        return self.temporal_support.get(index, ())

    def spatial_support_at(self, index: int) -> tuple[int, ...]:
        """The same-frame measured samples the spatial estimate here rested on."""
        return self.spatial_support.get(index, ())

    def position_of(self, index: int) -> tuple[int, int]:
        row, column = divmod(index, self.width)
        return column, row

    def temporally_reconstructed_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, state in enumerate(self.state_codes)
            if state == STATE_TEMPORALLY_RECONSTRUCTED
        )

    def spatially_reconstructed_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, state in enumerate(self.state_codes)
            if state == STATE_SPATIALLY_RECONSTRUCTED
        )

    def as_dict(self) -> dict[str, object]:
        """A JSON-shaped summary. Counts and work, never a quarter-million codes."""
        states = self.states()
        confirmations = self.confirmations()
        return {
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "sample_count": self.sample_count,
            "measured": states.count(STATE_NAMES[STATE_MEASURED]),
            "spatially_reconstructed": states.count(STATE_NAMES[STATE_SPATIALLY_RECONSTRUCTED]),
            "temporally_reconstructed": states.count(STATE_NAMES[STATE_TEMPORALLY_RECONSTRUCTED]),
            "missing": states.count(STATE_NAMES[STATE_MISSING]),
            "confirmations": {name: confirmations.count(name) for name in CONFIRMATIONS},
            "camera": self.camera.as_dict(),
            "target": self.target.as_dict(),
            "sources": [frame.as_dict() for frame in self.sources],
            "work": self.work.as_dict(),
        }


def _frozen_codes(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ResultUnusable(f"a result carries {value!r}, which is not a sequence of codes")
    codes: list[int] = []
    for index, code in enumerate(value):
        if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= MAX_CODE:
            raise ResultUnusable(f"a result carries depth code {code!r} at index {index}")
        codes.append(code)
    return tuple(codes)


def _frozen_bytes(value: object, what: str) -> bytes:
    """One byte per sample, copied out of whatever buffer the caller still holds."""
    if isinstance(value, bytes):
        return value
    if not isinstance(value, bytearray | memoryview):
        raise ResultUnusable(f"a result carries {what} {value!r}, which is not one byte per sample")
    return bytes(value)


def _frozen_confidence(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ResultUnusable(f"a result carries confidence {value!r}, which is not a sequence")
    values: list[float] = []
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            raise ResultUnusable(f"a result carries confidence {entry!r} at index {index}")
        number = float(entry)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ResultUnusable(
                f"a result carries confidence {entry!r} at index {index}; confidence is a finite "
                "fraction of 1"
            )
        values.append(number)
    return tuple(values)


def _frozen_temporal_support(value: object) -> Mapping[int, tuple[tuple[int, int], ...]]:
    """Copy the support map, so a value that can change after validation cannot."""
    if not isinstance(value, Mapping):
        raise ResultUnusable(f"a result carries support {value!r}, which is not a mapping")
    frozen: dict[int, tuple[tuple[int, int], ...]] = {}
    for index, references in value.items():
        at = f"a result names temporal support for {index!r}"
        if isinstance(index, bool) or not isinstance(index, int):
            raise ResultUnusable(f"{at}, which is not a sample index")
        if not isinstance(references, Sequence) or isinstance(references, str | bytes | bytearray):
            raise ResultUnusable(f"{at} as {references!r}, which is not a list of references")
        named: list[tuple[int, int]] = []
        for reference in references:
            if (
                not isinstance(reference, tuple)
                or len(reference) != 2
                or any(isinstance(part, bool) or not isinstance(part, int) for part in reference)
            ):
                raise ResultUnusable(
                    f"{at} with {reference!r}; a temporal support reference is exactly a "
                    "(frame index, sample index) pair"
                )
            named.append((int(reference[0]), int(reference[1])))
        frozen[index] = tuple(named)
    return MappingProxyType(frozen)


def _frozen_spatial_support(value: object) -> Mapping[int, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise ResultUnusable(f"a result carries spatial support {value!r}, which is not a mapping")
    frozen: dict[int, tuple[int, ...]] = {}
    for index, references in value.items():
        at = f"a result names spatial support for {index!r}"
        if isinstance(index, bool) or not isinstance(index, int):
            raise ResultUnusable(f"{at}, which is not a sample index")
        if not isinstance(references, Sequence) or isinstance(references, str | bytes | bytearray):
            raise ResultUnusable(f"{at} as {references!r}, which is not a list of indices")
        named: list[int] = []
        for reference in references:
            if isinstance(reference, bool) or not isinstance(reference, int):
                raise ResultUnusable(f"{at} with {reference!r}, which is not an index")
            named.append(reference)
        frozen[index] = tuple(named)
    return MappingProxyType(frozen)


def _ensure_frame_set(target: TemporalFrame, sources: Sequence[TemporalFrame]) -> None:
    """The frames of one call: distinct, ordered, accepted, and the same rectangle.

    Ordered and distinct because a support reference names a frame index, and a
    frame index that appears twice names two different things. Accepted because
    this module refuses to estimate a pose. The same rectangle because the
    intrinsics are the full frame's and a source patch of a different size at a
    different origin is a different question - not one this call can answer by
    resampling it.
    """
    if not isinstance(sources, Sequence) or not sources:
        raise FrameUnusable(
            "a temporal completion needs at least one other frame; a frame completed from itself "
            "is a spatial completion, which scanner3d.depth_completion already is"
        )
    if len(sources) + 1 > MAX_FRAMES:
        raise FrameUnusable(
            f"a temporal completion was offered {len(sources) + 1} frame(s), past the ceiling of "
            f"{MAX_FRAMES} this module states"
        )
    if len(sources) > MAX_SUPPORT_FRAMES:
        raise FrameUnusable(
            f"a temporal completion was offered {len(sources)} source frame(s), past the ceiling "
            f"of {MAX_SUPPORT_FRAMES} distinct frames one reconstruction may rest on"
        )
    indices = [frame.index for frame in sources]
    if indices != sorted(set(indices)):
        raise FrameUnusable(
            f"a temporal completion names source frames {indices}; they are distinct and in "
            "index order, because a support reference names a frame index and one that appears "
            "twice names two different things"
        )
    if target.index in indices:
        raise FrameUnusable(
            f"a temporal completion names frame {target.index} as both target and source. A "
            "sample of the frame being completed is not independent temporal evidence about it"
        )
    if not target.pose.accepted:
        raise PoseUnusable(
            f"the target frame {target.index} carries a pose the tracker did not accept "
            f"({target.pose.status!r}); nothing here estimates, repairs or substitutes one"
        )
    for frame in sources:
        if not frame.pose.accepted:
            raise PoseUnusable(
                f"source frame {frame.index} carries a pose the tracker did not accept "
                f"({frame.pose.status!r}); evidence carried through an unaccepted pose measures "
                "the pose error and not the scene"
            )
        if (frame.patch.width, frame.patch.height) != (target.patch.width, target.patch.height):
            raise FrameUnusable(
                f"source frame {frame.index} is {frame.patch.width}x{frame.patch.height} beside a "
                f"target of {target.patch.width}x{target.patch.height}; every frame here is one "
                "rectangle of one camera"
            )


# --- the completion --------------------------------------------------------------------------


def select_backend(name: str = AUTO_BACKEND) -> str:
    """The backend that will actually run, or an error naming why not.

    Exact, never a downgrade, on :mod:`scanner3d.compute`'s rule: a caller who
    selected a backend and got a different one cannot interpret their own result.
    """
    if name in (AUTO_BACKEND, PYTHON_BACKEND):
        return PYTHON_BACKEND
    if name in _UNIMPLEMENTED_BACKENDS:
        raise ComputeBackendUnavailable(
            f"scanner3d.depth_temporal has no {name!r} backend. The dependency-free 'python' "
            "reference is the only implementation and is the authority this module's benchmark "
            "scores; a vectorised backend would have to be parity-tested against it before it "
            "could be selected"
        )
    raise ComputeBackendUnknown(
        f"{name!r} is not a temporal backend; this module states {[AUTO_BACKEND, PYTHON_BACKEND]}"
    )


def complete_temporally(
    target: TemporalFrame,
    sources: Sequence[TemporalFrame],
    *,
    camera: CameraModel,
    policy: TemporalPolicy = DEFAULT_POLICY,
    spatial: CompletedPatch | None = None,
    backend: str = AUTO_BACKEND,
) -> TemporalPatch:
    """Confirm or recover ``target``'s no-return samples from other posed frames.

    Every measured sample of ``target`` comes back exactly as it was given,
    labelled ``"measured"`` at confidence ``1.0``. Every no-return sample is
    examined exactly once: the ones enough independent posed raw measurements
    agree about come back ``"temporally_reconstructed"`` with a finite
    confidence and the exact other-frame samples they rest on, and every other
    one comes back at the accepted spatial estimate if there was one - still
    labelled spatial - or at raw code zero and ``"missing"``, with its refusal
    counted by reason in :attr:`TemporalPatch.work`.

    Pure: the same frames, camera, poses, spatial result and policy always give
    the same result; no clock, no randomness and no I/O. Bounded: two passes,
    at most ``eligible_samples * footprint_cells + holes * (2 +
    max_candidates_per_target)`` units of work, which
    :attr:`TemporalWork.work_budget` states before the run.
    """
    resolved = select_backend(backend)
    if not isinstance(policy, TemporalPolicy):
        raise PolicyUnusable(f"a temporal completion needs a TemporalPolicy, not {policy!r}")
    if not isinstance(target, TemporalFrame):
        raise FrameUnusable(f"a temporal completion needs a TemporalFrame target, not {target!r}")
    if not isinstance(camera, CameraModel):
        raise CameraUnusable(f"a temporal completion needs a CameraModel, not {camera!r}")
    frames = tuple(sources)
    _ensure_frame_set(target, frames)
    if spatial is not None:
        if not isinstance(spatial, CompletedPatch):
            raise ResultUnusable(
                f"a temporal completion was given spatial estimate {spatial!r}, which is not a "
                "CompletedPatch; the accepted spatial result is the validated type or it is an "
                "unchecked claim"
            )
        if spatial.source != target.patch:
            raise ResultUnusable(
                "a temporal completion was given an accepted spatial estimate derived from a "
                "different raw patch than the target frame; a confirmation is about this frame "
                "or it is about nothing"
            )
    holes = len(target.patch.hole_indices())
    eligible = sum(len(frame.measured_indices()) for frame in frames)
    # Before the first projection, and derived rather than measured.
    budget = policy.worst_case_work(eligible, holes)
    if budget > MAX_WORK_UNITS:
        raise WorkTooLarge(
            f"carrying {eligible} measured sample(s) into {holes} hole(s) at a footprint of "
            f"{policy.footprint_radius} is worth up to {budget} unit(s) of projection and "
            f"comparison work, past the ceiling of {MAX_WORK_UNITS} this module states. Offer "
            "fewer frames or complete a smaller patch; a bound that large is not one"
        )
    derived = _temporal_of(target, frames, camera, policy, _SpatialView.of(spatial))
    work = TemporalWork(
        backend=resolved,
        policy=policy,
        source_frames=len(frames),
        eligible_samples=eligible,
        observations=derived.observations,
        observation_rejections=dict(derived.observation_rejections),
        candidates=derived.candidates,
        holes=holes,
        attempted=holes,
        accepted=derived.accepted,
        rejected=holes - derived.accepted,
        rejections=dict(derived.rejections),
        confirmed=derived.confirmed,
        contradicted=derived.contradicted,
        not_observed=derived.not_observed,
        support_samples=derived.support_samples,
        work_units=derived.work_units,
        work_budget=budget,
    )
    return TemporalPatch(
        width=target.patch.width,
        height=target.patch.height,
        codes=derived.codes,
        state_codes=derived.states,
        confidence=derived.confidence,
        confirmation_codes=derived.confirmations,
        temporal_support=derived.temporal_support,
        spatial_support=derived.spatial_support,
        work=work,
        target=target,
        sources=frames,
        camera=camera,
        spatial=spatial,
    )


def temporal_samples(result: TemporalPatch) -> Iterator[tuple[int, int, float]]:
    """Every temporal reconstruction, as ``(index, code, confidence)``, in index order."""
    for index in result.temporally_reconstructed_indices():
        yield index, result.codes[index], result.confidence[index]
