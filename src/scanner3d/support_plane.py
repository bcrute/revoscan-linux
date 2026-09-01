"""Removing the surface the object is standing on, and only when the frame proves it.

The problem this answers
------------------------
The object being scanned turns. The turntable, the bed, the sheet of MDF it is
standing on does not. Every frame therefore contains a large, stationary,
strongly planar patch, and pairwise registration - which has no notion of which
points belong to the thing being reconstructed - is perfectly happy to explain
the next frame by leaving that patch where it is. The two failure modes are the
ones the operator actually sees: the object freezes, because the bed dominates
the correspondence set and the fitted pose is near identity; or the fused result
is a bed-shaped blob with a smear on top of it.

Cropping is the obvious fix and is the wrong one. A box drawn round the object
needs to know where the object is, which is what has just gone wrong; a
"keep everything above z" rule needs to know which way is up, and this scanner
has no gravity reference of any kind - the support surface is wherever the
operator put it, at whatever slant the tripod happened to give it.

What this does
--------------
It fits ONE plane per frame, from the frame's own geometry, and removes only the
narrow band of points lying in that plane - and only when six independent gates
all agree that the plane is a surrounding support surface rather than a face of
the object. Everything else about the frame is left exactly as it arrived: no
crop, no half-space discard, no hole filling, no bounding box, no estimate of
geometry that was not measured. The raw depth bytes are never touched; this
operates on a derived cloud and nothing it does can reach a captured file.

Why the gates matter more than the fit
--------------------------------------
"The largest plane in the frame" is not the bed. On the motivating object - a
scale chassis - the largest plane in a good many frames is a body panel or a
flat plate, and removing it would delete the very geometry the scan exists to
capture. So a fitted plane is only ACTED on when all of the following hold, and
a frame that fails any of them gets its whole cloud back with the reason
recorded:

* enough inliers, absolutely and as a fraction (:data:`MIN_INLIER_POINTS`,
  :data:`MIN_INLIER_FRACTION`) - a plane through a handful of points is noise;
* broad two-dimensional spread: the inliers must cover most of the frame's own
  extent in BOTH in-plane principal directions
  (:data:`MIN_SPREAD_FRACTION`) and not be a strip
  (:data:`MIN_SPREAD_ASPECT`) - a compact planar face fails here, a bed the
  object sits in the middle of does not;
* meaningful non-plane geometry left over (:data:`MIN_OBJECT_POINTS`,
  :data:`MIN_OBJECT_FRACTION`) - an almost-entirely planar frame is a frame
  pointed at a panel, and removing its one plane would empty it;
* that geometry predominantly on ONE side of the plane
  (:data:`MIN_ONE_SIDED_FRACTION`) - things rest ON a support surface. Points
  spread through both half-spaces mean the plane cuts through the object, which
  is what a body panel looks like;
* the plane reaching BEYOND the geometry standing on it
  (:data:`MAX_OBJECT_FOOTPRINT_FRACTION`) - the gate that separates a bed from
  the lid of a box seen from above. Both are the largest plane in the frame,
  both span it, and both have everything else on one side; the difference is
  that a thing standing on a surface leaves surface showing around it, while a
  lid is exactly as wide as the box it closes;
* enough left afterwards (:data:`MIN_KEPT_POINTS`, :data:`MIN_KEPT_FRACTION`)
  and bounded removal (:data:`MAX_REMOVED_FRACTION`).

Withdrawal is a success, not a failure. A conservative refusal costs a frame
that still contains its bed; a wrong application costs the object.

Determinism
-----------
There is no random seed here, and no RANSAC. Candidate normals are the three
eigenvectors of the cloud's covariance, refined by a bounded mode-and-refit
loop, over :data:`DEFLATION_ROUNDS` rounds that re-seed on whatever the previous
round did not claim - a whole-cloud covariance points at the bed only when the
bed dominates the whole cloud, and subtracting what was found is how a plane
that is 40% of the frame gets a covariance of its own to be found by. Every sum
is :func:`math.fsum`, which is correctly rounded and therefore independent of
the order the points arrived in; every tie is broken by an explicit rule. The
same points in any order give bit-identical answers, and a
rigidly transformed cloud gives the same DECISION and the same kept point set
for any plane whose inliers are not sitting on the band edge. No optional
dependency, no wall clock, no device state and no file on disk is consulted:
this module imports :mod:`math` and this package's own error and geometry
vocabulary, and nothing else.

What it does not do
-------------------
It does not calibrate gravity, remove hands, subtract a background, track a
model, or claim any improvement to a physical scan. It removes a plane it can
prove is there, or it says why it did not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: What :meth:`SupportPlaneResult.as_dict` says it did, so a consumer branching
#: on the record does not have to infer the method from the field names.
FILTER_MODE: Final = "covariance-seeded-plane-band"

#: The policy version a caller records alongside a prepared cloud. Bumped when
#: any threshold below changes, because a record that says only "support plane
#: filtering ran" cannot be compared across two runs that disagreed about what
#: that meant.
POLICY_VERSION: Final = 1

#: The most points this module will read. Far above anything the preparation
#: path produces (about 900 for the live preview, 2200 for reconstruction) and
#: far below the point at which its linear passes stop being instant. Checked
#: from ``len()`` alone, so an oversized cloud is refused untouched.
MAX_INPUT_POINTS: Final = 200_000

#: The most points used for CANDIDATE generation and scoring. Everything from
#: the covariance to the refinement loop runs over at most this many, so the
#: cost of candidates times input is bounded before any of it starts; the final
#: inlier classification is one linear pass over the whole cloud. Above this the
#: sample is taken by uniform stride over the canonically sorted cloud, which
#: keeps the choice independent of the order the points arrived in.
FIT_SAMPLE_POINTS: Final = 4_000

#: Below this many points, no plane is fitted at all. A cloud this small is a
#: glimpse of a corner of a surface, and a plane through it describes the
#: decimation rather than the scene.
MIN_POINTS_TO_FIT: Final = 64

#: Candidate normals seeded per round, in ascending order of the covariance
#: eigenvalue they came from. Three, because a 3x3 covariance has three
#: eigenvectors and there is no fourth direction with any claim to be a
#: starting guess.
CANDIDATE_SEEDS: Final = 3

#: How many times the seeding is repeated after subtracting what the previous
#: round found.
#:
#: One round is not enough, and the reason is worth stating. A whole-cloud
#: covariance points at the support plane only when the support plane dominates
#: the whole cloud; on a frame that is 40% bed and 60% object the smallest
#: eigenvector is some compromise direction belonging to neither, and every
#: refinement seeded from it converges to a slab through the object. Removing
#: the band a round found and re-seeding on what is left recovers the bed on the
#: next pass, because the thing that was drowning it is no longer in the
#: covariance. Bounded at three, so the worst case is nine refinements of a
#: sample that is itself bounded - not a search whose cost depends on the scene.
DEFLATION_ROUNDS: Final = 3

#: Refinements a single call may run, and so the ceiling on the whole search.
CANDIDATES: Final = CANDIDATE_SEEDS * DEFLATION_ROUNDS

#: Mode-and-refit passes per candidate. The loop stops early when the inlier set
#: stops changing, which on a real frame happens in two or three; the bound is
#: what makes the worst case as cheap as the ordinary one.
REFINE_PASSES: Final = 8

#: Jacobi sweeps for the 3x3 symmetric eigen decomposition, and the off-diagonal
#: sum below which it stops. Eight sweeps converges a 3x3 to machine precision
#: with room to spare; the bound exists so the loop cannot depend on the data.
JACOBI_SWEEPS: Final = 8
JACOBI_TOLERANCE: Final = 1e-30

#: The inlier band's total width, as a fraction of the cloud's bounding-box
#: diagonal, when the caller does not state one. Scale free on purpose: this
#: repository has not established the physical unit of its coordinates, so an
#: absolute default would be a length claim it cannot make.
BAND_DIAGONAL_FRACTION: Final = 0.01

#: Bounds on a band a caller does state. The floor is
#: :data:`scanner3d.fusion.MIN_VOXEL_SIZE`'s reasoning - below it, membership is
#: decided by float64 rounding rather than by where the point is. The ceiling is
#: relative, because a band that is a quarter of the cloud is not a plane.
MIN_BAND: Final = 1e-9
MAX_BAND_FRACTION: Final = 0.25

#: The relative slack on the band's own edge. A synthetic scene puts points at
#: EXACTLY the plane, and a fitted offset that lands exactly half a band away
#: from them then has their membership decided by the last bit of a subtraction.
#: Widening the closed interval by a part in a billion is not a wider band - it
#: is the statement that the edge is inclusive, made in floating point.
BAND_TOLERANCE: Final = 1e-9

#: The smallest bounding-box diagonal a plane can be fitted in. A cloud flatter
#: than this against float64 has no geometry to fit and no scale to state a band
#: in.
MIN_DIAGONAL: Final = 1e-9

#: Inliers required before the band is believed to be a surface at all.
MIN_INLIER_POINTS: Final = 32
MIN_INLIER_FRACTION: Final = 0.15

#: How much of the frame's own extent the inliers must span, in BOTH in-plane
#: principal directions, for the plane to be a surrounding surface rather than a
#: face inside the scene; and how square that footprint must be, so a strip
#: along one edge is not accepted as a bed.
MIN_SPREAD_FRACTION: Final = 0.5
MIN_SPREAD_ASPECT: Final = 0.2

#: How much of the plane's own footprint the off-plane geometry may cover before
#: the plane stops being SURROUNDING. This is the gate that separates a bed from
#: the top face of a box seen from above: both are planes with all the other
#: geometry on one side of them, and the difference is that an object sits in
#: the middle of the surface it rests on, while a box's lid is exactly as wide
#: as the box. Nothing here knows which way is up, so this is the evidence that
#: is actually available.
MAX_OBJECT_FOOTPRINT_FRACTION: Final = 0.8

#: Non-plane geometry required, so an almost-entirely planar frame withdraws
#: instead of being emptied of the one thing it measured.
MIN_OBJECT_POINTS: Final = 32
MIN_OBJECT_FRACTION: Final = 0.1

#: How one-sided that geometry must be. Things REST on a support surface; a
#: plane with substantial geometry in both half-spaces is cutting through
#: something, and this repository would rather keep a bed than slice an object.
MIN_ONE_SIDED_FRACTION: Final = 0.85

#: What must survive the removal, and the largest share of a cloud this will
#: take. A frame that is nine tenths bed is a frame that measured almost nothing
#: else, and handing registration the tenth that is left is worse than handing
#: it the whole thing with a recorded reason.
MIN_KEPT_POINTS: Final = 32
MIN_KEPT_FRACTION: Final = 0.15
MAX_REMOVED_FRACTION: Final = 0.85

#: What :attr:`SupportPlaneResult.reason` holds. One string per outcome, as
#: constants rather than literals, because these are read by tests, by the
#: aggregate a report carries and by an operator looking at a progress line.
REASON_APPLIED: Final = "applied"
REASON_SPARSE: Final = "sparse-input"
REASON_DEGENERATE: Final = "degenerate-extent"
REASON_FEW_INLIERS: Final = "too-few-inliers"
REASON_NARROW_SPREAD: Final = "spread-too-narrow"
REASON_LITTLE_OBJECT: Final = "too-little-off-plane-geometry"
REASON_TWO_SIDED: Final = "off-plane-geometry-on-both-sides"
REASON_NOT_SURROUNDING: Final = "plane-does-not-surround-the-off-plane-geometry"
REASON_LITTLE_KEPT: Final = "too-little-would-remain"
REASON_LARGE_REMOVAL: Final = "removal-too-large"

#: True of every run, whatever the input was, and a caller cannot shorten it.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "this removes a plane the frame's own geometry proves is a surrounding support surface. "
    "It is not background subtraction, hand removal or cropping: a wall, an arm, a clamp and "
    "a second object are all kept, and nothing outside the fitted band is removed for any "
    "reason",
    "a large flat surface that IS part of the object - a chassis plate, a body panel - is "
    "deliberately preserved whenever the gates cannot tell it from a support surface, so a "
    "frame whose bed was not removed is the expected conservative outcome and not a fault",
    "the plane is fitted per frame and independently of every other frame, so a sweep may "
    "apply the filter to some frames and withdraw on others; the record for each frame says "
    "which happened and why",
    "no gravity, tripod, turntable or calibration input is used, and no direction is assumed "
    "to be up; the plane's orientation comes from the points alone",
    "removal happens before registration, so a removed point contributed to no pose and is in "
    "no registered cloud. Whether it also left an exported MESH is a separate question this "
    "module cannot answer: a mesh is fused from the full-resolution frames, not from the cloud "
    "handed to this filter, and it is the caller that decides whether to mask the same band out "
    "of those - see scanner3d.preparation.mask_support_band. The raw depth frames on disk are "
    "unchanged and complete either way",
    "bands, spreads and extents are stated in the same coordinate scale as the points, whose "
    "physical unit this repository has not established; they are numbers, not lengths",
)


# --- refusals -----------------------------------------------------------------


class SupportPlaneNotAllowed(ScannerError):
    """A support-plane filter was requested that this module refuses to run.

    Every one of these is raised before any per-point work, so a refused call
    has fitted nothing and the caller still holds exactly the cloud it offered.
    Distinct from a WITHDRAWAL, which is a completed run that decided not to
    act and returns a full record saying so.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class NothingToFilter(SupportPlaneNotAllowed):
    """No points were offered.

    An empty answer would be indistinguishable from a filter that removed the
    whole cloud, and those are not the same event.
    """


class InputTooLarge(SupportPlaneNotAllowed):
    """More points were offered than :data:`MAX_INPUT_POINTS`."""


class PointUnusable(SupportPlaneNotAllowed):
    """An input point is not a finite ``(x, y, z)`` triple.

    Checked before the covariance rather than left to the arithmetic: a single
    NaN propagates into every eigenvalue, and the fit would then fail every
    comparison it made without ever raising.
    """


class BandUnusable(SupportPlaneNotAllowed):
    """The requested inlier band cannot describe a plane in this cloud."""


class ThresholdUnusable(SupportPlaneNotAllowed):
    """A gate threshold is outside the range in which it is a gate."""


# --- the gates that actually ran ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The five caller-settable gate values that decided one particular call.

    Carried on the record rather than read back off the module constants when
    the record is serialised. Those two are the same thing only for a caller
    that changed nothing, and a record that quotes the DEFAULT gate beside a
    decision the CALLER'S gate made is a record of a run that did not happen:
    a reader reconciling "withdrew on spread" against
    ``min_spread_fraction: 0.5`` cannot tell that the caller asked for 0.9.

    The gates a caller cannot set are still stated in :meth:`as_dict`, because
    they judged the frame too; they are simply the same on every run.
    """

    min_inlier_fraction: float
    min_spread_fraction: float
    min_object_fraction: float
    min_one_sided_fraction: float
    max_removed_fraction: float

    @property
    def overridden(self) -> tuple[str, ...]:
        """Which of the five the caller moved off this module's default.

        Named rather than left to be spotted by comparing numbers: "these gates
        were not the shipped policy" is the fact a reader of an unexpected
        decision needs first.
        """
        return tuple(
            name
            for name, value, default in (
                ("min_inlier_fraction", self.min_inlier_fraction, MIN_INLIER_FRACTION),
                ("min_spread_fraction", self.min_spread_fraction, MIN_SPREAD_FRACTION),
                ("min_object_fraction", self.min_object_fraction, MIN_OBJECT_FRACTION),
                ("min_one_sided_fraction", self.min_one_sided_fraction, MIN_ONE_SIDED_FRACTION),
                ("max_removed_fraction", self.max_removed_fraction, MAX_REMOVED_FRACTION),
            )
            if value != default
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_points_to_fit": MIN_POINTS_TO_FIT,
            "min_inlier_points": MIN_INLIER_POINTS,
            "min_inlier_fraction": self.min_inlier_fraction,
            "min_spread_fraction": self.min_spread_fraction,
            "min_spread_aspect": MIN_SPREAD_ASPECT,
            "max_object_footprint_fraction": MAX_OBJECT_FOOTPRINT_FRACTION,
            "min_object_points": MIN_OBJECT_POINTS,
            "min_object_fraction": self.min_object_fraction,
            "min_one_sided_fraction": self.min_one_sided_fraction,
            "min_kept_points": MIN_KEPT_POINTS,
            "min_kept_fraction": MIN_KEPT_FRACTION,
            "max_removed_fraction": self.max_removed_fraction,
            "overridden_by_caller": list(self.overridden),
            "means": (
                "the values that judged THIS frame. The five a caller may set are reported as "
                "they were supplied, not as this module's defaults; overridden_by_caller names "
                "the ones that differ from the shipped policy version"
            ),
        }


#: The gates a caller that asks for nothing gets, and the values
#: :data:`POLICY_VERSION` refers to.
DEFAULT_THRESHOLDS: Final = Thresholds(
    min_inlier_fraction=MIN_INLIER_FRACTION,
    min_spread_fraction=MIN_SPREAD_FRACTION,
    min_object_fraction=MIN_OBJECT_FRACTION,
    min_one_sided_fraction=MIN_ONE_SIDED_FRACTION,
    max_removed_fraction=MAX_REMOVED_FRACTION,
)


# --- the plane ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plane:
    """A plane as a unit normal and the offset along it, in the points' own frame.

    ``offset`` is the signed distance from the origin, so a point's signed
    distance is ``dot(point, normal) - offset``. The normal's SIGN is chosen so
    that the off-plane geometry lies on the positive side - the object is
    "above" the support surface in the only sense this module has, which is
    "on the side most of the other points are on".
    """

    normal: Point
    offset: float

    def signed_distance(self, point: Point) -> float:
        """How far ``point`` lies from this plane, positive on the object's side."""
        return (
            point[0] * self.normal[0]
            + point[1] * self.normal[1]
            + point[2] * self.normal[2]
            - self.offset
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "normal": list(self.normal),
            "offset": self.offset,
            "means": (
                "a point's signed distance is dot(point, normal) - offset, positive on the "
                "side the off-plane geometry was found on"
            ),
        }


# --- the result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportPlaneResult:
    """One cloud with its support plane removed, or not, and the whole accounting.

    ``applied`` is the field to read before believing the rest. When it is
    false, :attr:`points` is every point that was offered, in the order it was
    offered, and :attr:`reason` says which gate refused - a run that decided not
    to act, which is a different event from one that found nothing to remove.
    """

    points: tuple[Point, ...]
    removed: tuple[Point, ...]
    applied: bool
    reason: str
    plane: Plane | None
    band: float
    input_points: int
    inlier_points: int
    off_plane_points: int
    positive_points: int
    negative_points: int
    diagonal: float
    spread_major: float
    spread_minor: float
    spread_aspect: float
    object_footprint_major: float
    object_footprint_minor: float
    candidates_scored: int
    sample_points: int
    thresholds: Thresholds
    limitations: tuple[str, ...]

    @property
    def kept_points(self) -> int:
        """How many points came out."""
        return len(self.points)

    @property
    def removed_points(self) -> int:
        """How many were dropped for lying in the support plane's band."""
        return len(self.removed)

    @property
    def inlier_fraction(self) -> float:
        """What share of the input lay in the fitted band, in [0, 1]."""
        return self.inlier_points / self.input_points if self.input_points else 0.0

    @property
    def removed_fraction(self) -> float:
        """What share of the input was actually removed, in [0, 1]."""
        return self.removed_points / self.input_points if self.input_points else 0.0

    @property
    def kept_fraction(self) -> float:
        """What share of the input survived, in [0, 1]."""
        return self.kept_points / self.input_points if self.input_points else 0.0

    @property
    def one_sided_fraction(self) -> float:
        """The larger half-space's share of the off-plane geometry, in [0, 1]."""
        total = self.positive_points + self.negative_points
        return max(self.positive_points, self.negative_points) / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "mode": FILTER_MODE,
            "applied": self.applied,
            "reason": self.reason,
            "plane": None if self.plane is None else self.plane.as_dict(),
            "band": self.band,
            "band_means": (
                "the total width of the removed slab, centred on the fitted plane; a point is "
                "removed when its absolute signed distance is at most half of this"
            ),
            "accounting": {
                "input_points": self.input_points,
                "kept_points": self.kept_points,
                "removed_points": self.removed_points,
                "inlier_points": self.inlier_points,
                "inlier_fraction": self.inlier_fraction,
                "removed_fraction": self.removed_fraction,
                "kept_fraction": self.kept_fraction,
            },
            "support_evidence": {
                "diagonal": self.diagonal,
                "spread_major": self.spread_major,
                "spread_minor": self.spread_minor,
                "spread_aspect": self.spread_aspect,
                "object_footprint_major": self.object_footprint_major,
                "object_footprint_minor": self.object_footprint_minor,
                "off_plane_points": self.off_plane_points,
                "positive_points": self.positive_points,
                "negative_points": self.negative_points,
                "one_sided_fraction": self.one_sided_fraction,
                "means": (
                    "spread_major and spread_minor are the inlier footprint's extent along the "
                    "two in-plane principal directions, each as a fraction of the whole "
                    "cloud's extent along the same direction; object_footprint_* are the "
                    "off-plane geometry's extents along those directions as fractions of the "
                    "inlier footprint's own, so a plane that merely caps the object rather "
                    "than surrounding it reads as 1; one_sided_fraction is the larger "
                    "half-space's share of the points outside the band"
                ),
            },
            "search": {
                "candidates_scored": self.candidates_scored,
                "sample_points": self.sample_points,
                "refine_passes": REFINE_PASSES,
            },
            "thresholds": self.thresholds.as_dict(),
            "bounds": {
                "max_input_points": MAX_INPUT_POINTS,
                "fit_sample_points": FIT_SAMPLE_POINTS,
                "candidates": CANDIDATES,
                "min_band": MIN_BAND,
                "max_band_fraction": MAX_BAND_FRACTION,
                "min_diagonal": MIN_DIAGONAL,
            },
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


# --- boundary checks ----------------------------------------------------------


def ensure_points_usable(points: Sequence[Point]) -> int:
    """Refuse an input this module will not read, and return how many points it holds."""
    count = len(points)
    if count == 0:
        raise NothingToFilter(
            "no points were offered, so there is no plane to look for; refused rather than "
            "answered with an empty cloud, which is indistinguishable downstream from a filter "
            "that removed every point it was given"
        )
    if count > MAX_INPUT_POINTS:
        raise InputTooLarge(
            f"{count} points is past the {MAX_INPUT_POINTS}-point ceiling; nothing was read. "
            "Decimate the cloud before fitting a support plane to it"
        )
    return count


def _checked(points: Sequence[Point]) -> list[tuple[float, float, float]]:
    """Every point as a finite triple, refusing the first one that is not."""
    checked: list[tuple[float, float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 3:
            raise PointUnusable(
                f"point {index} is {point!r}, which is not an (x, y, z) triple; nothing was fitted"
            )
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise PointUnusable(
                f"point {index} is {point!r}, which is not finite; a plane fitted through it "
                "would compare false against every threshold without ever failing"
            )
        checked.append((x, y, z))
    return checked


def _ensure_fraction(name: str, value: float) -> float:
    """Refuse a gate that is not a fraction, and return it unchanged."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ThresholdUnusable(
            f"{name} {value!r} is not a finite fraction in [0, 1]; a gate outside that range "
            "either admits everything or refuses everything, and neither is a gate"
        )
    return value


def _ensure_band(band: float, diagonal: float) -> float:
    """Refuse a band that cannot describe a plane in a cloud this size."""
    if not math.isfinite(band) or band < MIN_BAND:
        raise BandUnusable(
            f"band {band!r} is not a finite width at or above the {MIN_BAND} floor; under it, "
            "whether a point is in the band is decided by float64 rounding rather than by "
            "where the point is"
        )
    if band > MAX_BAND_FRACTION * diagonal:
        raise BandUnusable(
            f"band {band!r} is more than {MAX_BAND_FRACTION} of this cloud's {diagonal} "
            "diagonal; a slab that thick is a region of the scene, not a plane"
        )
    return band


# --- linear algebra, in the small ---------------------------------------------


def _mean(values: Sequence[float]) -> float:
    """The mean, summed exactly, so the answer does not depend on the order."""
    return math.fsum(values) / len(values)


def _covariance(points: Sequence[tuple[float, float, float]]) -> list[list[float]]:
    """The 3x3 covariance about the centroid, every sum correctly rounded.

    :func:`math.fsum` rather than ``sum``: it is exact, so a permutation of the
    same points produces the identical matrix and therefore the identical
    plane. That is the whole of this module's order independence.
    """
    cx = _mean([p[0] for p in points])
    cy = _mean([p[1] for p in points])
    cz = _mean([p[2] for p in points])
    dx = [p[0] - cx for p in points]
    dy = [p[1] - cy for p in points]
    dz = [p[2] - cz for p in points]
    n = float(len(points))
    xx = math.fsum(a * a for a in dx) / n
    yy = math.fsum(a * a for a in dy) / n
    zz = math.fsum(a * a for a in dz) / n
    xy = math.fsum(a * b for a, b in zip(dx, dy, strict=True)) / n
    xz = math.fsum(a * b for a, b in zip(dx, dz, strict=True)) / n
    yz = math.fsum(a * b for a, b in zip(dy, dz, strict=True)) / n
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


def _oriented(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """One of the two unit vectors along an axis, chosen by a fixed rule.

    An eigenvector is defined up to sign, and a solver's choice of sign is an
    accident of its arithmetic. Fixing it here - first component large enough to
    have a sign, made positive - is what stops two runs on the same points
    disagreeing about a direction they agree about.
    """
    for component in vector:
        if abs(component) > 1e-12:
            if component < 0.0:
                return (-vector[0], -vector[1], -vector[2])
            return vector
    return vector


def _eigen_symmetric(
    matrix: Sequence[Sequence[float]],
) -> list[tuple[float, tuple[float, float, float]]]:
    """Eigenvalues and unit eigenvectors of a 3x3 symmetric matrix, ascending.

    Cyclic Jacobi, bounded at :data:`JACOBI_SWEEPS`. Three by three, so this is
    a fixed amount of arithmetic rather than an iterative solve whose cost
    depends on the data - and it needs no optional dependency, which
    :mod:`scanner3d.compute` explains this package will not require.
    """
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(JACOBI_SWEEPS):
        off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
        if off <= JACOBI_TOLERANCE:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) <= 1e-300:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    pairs: list[tuple[float, tuple[float, float, float]]] = []
    for i in range(3):
        raw = (v[0][i], v[1][i], v[2][i])
        length = math.sqrt(raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2)
        unit = raw if length == 0.0 else (raw[0] / length, raw[1] / length, raw[2] / length)
        pairs.append((a[i][i], _oriented(unit)))
    # Ascending eigenvalue, and the vector itself as the tie-break, so a cloud
    # with a repeated eigenvalue - a sphere, a cube of points - still orders its
    # candidates the same way on every run.
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return pairs


def _in_plane_basis(normal: tuple[float, float, float]) -> tuple[Point, Point]:
    """Two orthonormal directions spanning the plane, from a fixed construction."""
    axis = min(range(3), key=lambda i: (abs(normal[i]), i))
    seed = [0.0, 0.0, 0.0]
    seed[axis] = 1.0
    u = (
        seed[1] * normal[2] - seed[2] * normal[1],
        seed[2] * normal[0] - seed[0] * normal[2],
        seed[0] * normal[1] - seed[1] * normal[0],
    )
    length = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    u = (u[0] / length, u[1] / length, u[2] / length)
    v = (
        normal[1] * u[2] - normal[2] * u[1],
        normal[2] * u[0] - normal[0] * u[2],
        normal[0] * u[1] - normal[1] * u[0],
    )
    return u, v


def _principal_extents(
    coords: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """Extents of a 2D point set along its own principal axes, larger first.

    Along the SET'S OWN axes rather than the basis it happens to be expressed
    in, because the basis is a construction and the footprint is a fact: a
    rectangle measured along its diagonal is not a different rectangle, and a
    rigidly rotated cloud must produce the same two numbers.
    """
    cu = _mean([c[0] for c in coords])
    cv = _mean([c[1] for c in coords])
    du = [c[0] - cu for c in coords]
    dv = [c[1] - cv for c in coords]
    n = float(len(coords))
    uu = math.fsum(a * a for a in du) / n
    vv = math.fsum(a * a for a in dv) / n
    uv = math.fsum(a * b for a, b in zip(du, dv, strict=True)) / n
    # Closed-form eigenvectors of a 2x2 symmetric matrix. The angle is halved
    # because a covariance rotates at twice the rate of the frame it describes.
    angle = 0.5 * math.atan2(2.0 * uv, uu - vv)
    axes = ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
    extents: list[float] = []
    for ax in axes:
        projected = [c[0] * ax[0] + c[1] * ax[1] for c in coords]
        extents.append(max(projected) - min(projected))
    if extents[1] > extents[0]:
        return (extents[1], extents[0]), (axes[1], axes[0])
    return (extents[0], extents[1]), (axes[0], axes[1])


def _extents_along(
    coords: Sequence[tuple[float, float]],
    axes: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """How far a 2D point set reaches along two given directions."""
    spans: list[float] = []
    for axis in axes:
        projected = [c[0] * axis[0] + c[1] * axis[1] for c in coords]
        spans.append(max(projected) - min(projected))
    return spans[0], spans[1]


# --- the fit ------------------------------------------------------------------


def _sample(points: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """At most :data:`FIT_SAMPLE_POINTS` points, chosen without regard to input order.

    Sorted first, then strided. Sorting is what makes the CHOICE independent of
    the order the caller happened to hand the points over in; below the ceiling
    nothing is dropped and the question does not arise, which is the case every
    cloud this repository's preparation path produces falls into.
    """
    if len(points) <= FIT_SAMPLE_POINTS:
        return list(points)
    ordered = sorted(points)
    step = math.ceil(len(ordered) / FIT_SAMPLE_POINTS)
    return ordered[::step]


def _best_window(distances: Sequence[float], band: float) -> float:
    """The centre of the fullest band-wide slab, leftmost on a tie.

    A histogram would split a plane that happens to straddle a bin edge; this
    slides the interval instead, so the answer depends on the points rather
    than on where the bins were put.

    The centre returned is the midpoint of the points the window actually
    caught, not of the window. A plane placed at a window EDGE sits half a band
    away from the surface it just found, which for an exactly planar synthetic
    scene puts every one of those points on its own band's boundary.
    """
    ordered = sorted(distances)
    best_count = 0
    best_low = 0
    best_high = 1
    high = 0
    for low, start in enumerate(ordered):
        if high < low:
            high = low
        while high < len(ordered) and ordered[high] - start <= band:
            high += 1
        count = high - low
        if count > best_count:
            best_count = count
            best_low = low
            best_high = high
    return (ordered[best_low] + ordered[best_high - 1]) / 2.0


def _refine(
    sample: Sequence[tuple[float, float, float]],
    normal: tuple[float, float, float],
    band: float,
) -> tuple[Plane, int]:
    """Alternate "which points are in the band" and "what plane fits them".

    Seeded from a covariance eigenvector, which is a whole-cloud answer and so
    is pulled towards the bed only in proportion to how much of the cloud the
    bed is - which is exactly the case this module exists for. Each pass places
    the band where it holds the most points and refits the plane to just those,
    so the fit stops describing the object as soon as the band stops containing
    it.
    """
    current = normal
    offset = _best_window([_dot(p, current) for p in sample], band)
    previous: tuple[int, ...] = ()
    for _ in range(REFINE_PASSES):
        chosen = tuple(
            index
            for index, point in enumerate(sample)
            if _in_band(_dot(point, current), offset, band / 2.0)
        )
        if len(chosen) < 3 or chosen == previous:
            break
        previous = chosen
        inliers = [sample[index] for index in chosen]
        current = _eigen_symmetric(_covariance(inliers))[0][1]
        # Centred on the points it just fitted, not on the sliding window that
        # found them: the window only has to LOCATE the slab, and leaving the
        # plane at its edge would put the surface it describes on the boundary
        # of its own band, where a point either side of zero falls differently.
        offset = _mean([_dot(p, current) for p in inliers])
    population = sum(1 for p in sample if _in_band(_dot(p, current), offset, band / 2.0))
    return Plane(normal=current, offset=offset), population


def _dot(point: tuple[float, float, float], normal: tuple[float, float, float]) -> float:
    """The projection of ``point`` onto ``normal``."""
    return point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2]


def _in_band(distance: float, offset: float, half: float) -> bool:
    """Whether a projected point lies in the band, with the edge counted as in.

    One function so the refinement, the final classification and the removal all
    answer the question the same way. They must: a point the fit called an
    inlier and the removal did not would be counted in the evidence and left in
    the cloud.
    """
    return abs(distance - offset) <= half * (1.0 + BAND_TOLERANCE)


@dataclass(frozen=True, slots=True)
class _Measured:
    """Everything measured about the winning candidate, before any gate judged it.

    A dataclass rather than a dict because a withdrawal carries these numbers
    too - a refusal whose evidence is not visible is indistinguishable from one
    that never looked - and the two paths that build a record must not be free
    to disagree about what is in it.
    """

    plane: Plane
    inliers: int
    off_plane: int
    positive: int
    negative: int
    spread_major: float
    spread_minor: float
    aspect: float
    footprint_major: float
    footprint_minor: float
    sample: int
    #: Candidate planes actually SCORED before this one won - the deflation
    #: loop stops early when nothing is left to re-seed on, so an exactly
    #: planar frame weighs three candidates rather than the module's nine, and
    #: a record naming the ceiling would describe a search that did not run.
    candidates: int


# --- the entry point ----------------------------------------------------------


def suppress_support_plane(
    points: Sequence[Point],
    *,
    band: float | None = None,
    min_inlier_fraction: float = MIN_INLIER_FRACTION,
    min_spread_fraction: float = MIN_SPREAD_FRACTION,
    min_object_fraction: float = MIN_OBJECT_FRACTION,
    min_one_sided_fraction: float = MIN_ONE_SIDED_FRACTION,
    max_removed_fraction: float = MAX_REMOVED_FRACTION,
) -> SupportPlaneResult:
    """Remove this cloud's support plane, or say why it was left alone.

    Returns the kept points in the ORDER THEY WERE GIVEN, so a caller that
    pairs a cloud with anything else positionally is unaffected by whether the
    filter acted.

    Raises :class:`SupportPlaneNotAllowed` - before any fitting - for an empty
    or oversized cloud, a non-finite point, an unusable band or a gate that is
    not a fraction. Every other outcome is a completed run: either
    ``applied=True`` with the band removed, or ``applied=False`` with the whole
    input returned and :attr:`SupportPlaneResult.reason` naming the gate that
    refused. Withdrawal is deliberately not an exception, because a frame
    without a provable support plane is an ordinary frame and this runs on
    every one of them.
    """
    for name, value in (
        ("min_inlier_fraction", min_inlier_fraction),
        ("min_spread_fraction", min_spread_fraction),
        ("min_object_fraction", min_object_fraction),
        ("min_one_sided_fraction", min_one_sided_fraction),
    ):
        _ensure_fraction(name, value)
    if not math.isfinite(max_removed_fraction) or not 0.0 < max_removed_fraction <= 1.0:
        raise ThresholdUnusable(
            f"max_removed_fraction {max_removed_fraction!r} is outside (0, 1]; a filter that "
            "may remove no points, or more than all of them, is not a bound"
        )
    if band is not None and (not math.isfinite(band) or band < MIN_BAND):
        raise BandUnusable(
            f"band {band!r} is not a finite width at or above the {MIN_BAND} floor; under it, "
            "whether a point is in the band is decided by float64 rounding rather than by "
            "where the point is"
        )
    thresholds = Thresholds(
        min_inlier_fraction=min_inlier_fraction,
        min_spread_fraction=min_spread_fraction,
        min_object_fraction=min_object_fraction,
        min_one_sided_fraction=min_one_sided_fraction,
        max_removed_fraction=max_removed_fraction,
    )
    count = ensure_points_usable(points)
    checked = _checked(points)

    spans = [max(p[axis] for p in checked) - min(p[axis] for p in checked) for axis in range(3)]
    diagonal = math.sqrt(spans[0] ** 2 + spans[1] ** 2 + spans[2] ** 2)
    proposed = BAND_DIAGONAL_FRACTION * diagonal if band is None else band

    # The two withdrawals that come before the band is checked against the
    # cloud, because on an input this small or this flat the RATIO the check
    # tests is not a fact about the caller's request.
    if count < MIN_POINTS_TO_FIT:
        # No candidate was generated, so the count of candidates SCORED is
        # zero. It is not the module's ceiling: a record that quoted the
        # ceiling here would say nine planes were weighed against this frame.
        return _withdrawn(checked, REASON_SPARSE, proposed, count, diagonal, thresholds)
    if diagonal < MIN_DIAGONAL:
        return _withdrawn(checked, REASON_DEGENERATE, proposed, count, diagonal, thresholds)
    width = _ensure_band(proposed, diagonal)

    sample = _sample(checked)
    half = width / 2.0
    scored: list[tuple[int, Plane]] = []
    remaining = sample
    for _round in range(DEFLATION_ROUNDS):
        if len(remaining) < 3:
            break
        round_planes: list[tuple[int, Plane]] = []
        for _value, vector in _eigen_symmetric(_covariance(remaining))[:CANDIDATE_SEEDS]:
            # Fitted to what is LEFT and scored against the WHOLE sample.
            # Deflation only decides where to look: a plane found among the
            # remainder still has to beat every other candidate on the cloud it
            # is actually going to be applied to, so a later round cannot win by
            # being measured against an easier population.
            plane, _population = _refine(remaining, vector, width)
            population = sum(
                1 for point in sample if _in_band(_dot(point, plane.normal), plane.offset, half)
            )
            round_planes.append((population, plane))
        scored.extend(round_planes)
        best_of_round = max(range(len(round_planes)), key=lambda i: (round_planes[i][0], -i))
        found = round_planes[best_of_round][1]
        remaining = [
            point
            for point in remaining
            if not _in_band(_dot(point, found.normal), found.offset, half)
        ]
    # Most populated band wins; the candidate order - ascending eigenvalue, then
    # the vector itself - breaks a tie, so the choice never depends on which
    # comparison the sort happened to make first.
    best_index = max(range(len(scored)), key=lambda i: (scored[i][0], -i))
    candidate = scored[best_index][1]

    signed = [_dot(p, candidate.normal) - candidate.offset for p in checked]
    inliers = [p for p, d in zip(checked, signed, strict=True) if _in_band(d, 0.0, half)]
    positive = sum(1 for d in signed if d > 0.0 and not _in_band(d, 0.0, half))
    negative = sum(1 for d in signed if d < 0.0 and not _in_band(d, 0.0, half))
    # The object is on whichever side most of the off-plane geometry is, and the
    # normal is flipped to point that way so "positive" means "away from the
    # support surface" in every record this module writes.
    if negative > positive:
        candidate = Plane(
            normal=(-candidate.normal[0], -candidate.normal[1], -candidate.normal[2]),
            offset=-candidate.offset,
        )
        positive, negative = negative, positive
        signed = [-d for d in signed]
    off_plane = positive + negative

    spread_major = 0.0
    spread_minor = 0.0
    aspect = 0.0
    footprint_major = 1.0
    footprint_minor = 1.0
    if len(inliers) >= 3:
        u, v = _in_plane_basis(candidate.normal)
        (major, minor), axes = _principal_extents([(_dot(p, u), _dot(p, v)) for p in inliers])
        whole = [(_dot(p, u), _dot(p, v)) for p in checked]
        outside = [
            (_dot(p, u), _dot(p, v))
            for p, d in zip(checked, signed, strict=True)
            if not _in_band(d, 0.0, half)
        ]
        frame_extents = _extents_along(whole, axes)
        spread_major = major / frame_extents[0] if frame_extents[0] > 0.0 else 0.0
        spread_minor = minor / frame_extents[1] if frame_extents[1] > 0.0 else 0.0
        aspect = minor / major if major > 0.0 else 0.0
        if outside:
            # Measured along the PLANE'S OWN axes, not the object's: the
            # question is how much of this surface the thing standing on it
            # covers, which is a question about the surface's directions.
            object_extents = _extents_along(outside, axes)
            footprint_major = object_extents[0] / major if major > 0.0 else 1.0
            footprint_minor = object_extents[1] / minor if minor > 0.0 else 1.0

    measured = _Measured(
        plane=candidate,
        inliers=len(inliers),
        off_plane=off_plane,
        positive=positive,
        negative=negative,
        spread_major=spread_major,
        spread_minor=spread_minor,
        aspect=aspect,
        footprint_major=footprint_major,
        footprint_minor=footprint_minor,
        sample=len(sample),
        candidates=len(scored),
    )

    reason = _gate(
        count=count,
        inliers=len(inliers),
        off_plane=off_plane,
        positive=positive,
        negative=negative,
        spread_major=spread_major,
        spread_minor=spread_minor,
        aspect=aspect,
        footprint_major=footprint_major,
        footprint_minor=footprint_minor,
        min_inlier_fraction=min_inlier_fraction,
        min_spread_fraction=min_spread_fraction,
        min_object_fraction=min_object_fraction,
        min_one_sided_fraction=min_one_sided_fraction,
        max_removed_fraction=max_removed_fraction,
    )
    if reason != REASON_APPLIED:
        return _withdrawn(checked, reason, width, count, diagonal, thresholds, measured)

    kept = [p for p, d in zip(checked, signed, strict=True) if not _in_band(d, 0.0, half)]
    return SupportPlaneResult(
        points=tuple(kept),
        removed=tuple(inliers),
        applied=True,
        reason=REASON_APPLIED,
        plane=candidate,
        band=width,
        input_points=count,
        inlier_points=len(inliers),
        off_plane_points=off_plane,
        positive_points=positive,
        negative_points=negative,
        diagonal=diagonal,
        spread_major=spread_major,
        spread_minor=spread_minor,
        spread_aspect=aspect,
        object_footprint_major=footprint_major,
        object_footprint_minor=footprint_minor,
        candidates_scored=len(scored),
        sample_points=len(sample),
        thresholds=thresholds,
        limitations=INHERENT_LIMITATIONS,
    )


def _gate(
    *,
    count: int,
    inliers: int,
    off_plane: int,
    positive: int,
    negative: int,
    spread_major: float,
    spread_minor: float,
    aspect: float,
    footprint_major: float,
    footprint_minor: float,
    min_inlier_fraction: float,
    min_spread_fraction: float,
    min_object_fraction: float,
    min_one_sided_fraction: float,
    max_removed_fraction: float,
) -> str:
    """Which gate refuses this candidate, or :data:`REASON_APPLIED` if none does.

    Ordered from the cheapest claim to the most specific, so the reason a frame
    was left alone names the thing that was actually missing rather than the
    last check that happened to run.
    """
    if inliers < MIN_INLIER_POINTS or inliers / count < min_inlier_fraction:
        return REASON_FEW_INLIERS
    if spread_major < min_spread_fraction or spread_minor < min_spread_fraction:
        return REASON_NARROW_SPREAD
    if aspect < MIN_SPREAD_ASPECT:
        return REASON_NARROW_SPREAD
    if off_plane < MIN_OBJECT_POINTS or off_plane / count < min_object_fraction:
        return REASON_LITTLE_OBJECT
    if max(positive, negative) / off_plane < min_one_sided_fraction:
        return REASON_TWO_SIDED
    if (
        footprint_major > MAX_OBJECT_FOOTPRINT_FRACTION
        or footprint_minor > MAX_OBJECT_FOOTPRINT_FRACTION
    ):
        return REASON_NOT_SURROUNDING
    kept = count - inliers
    if kept < MIN_KEPT_POINTS or kept / count < MIN_KEPT_FRACTION:
        return REASON_LITTLE_KEPT
    if inliers / count > max_removed_fraction:
        return REASON_LARGE_REMOVAL
    return REASON_APPLIED


def _withdrawn(
    checked: Sequence[tuple[float, float, float]],
    reason: str,
    band: float,
    count: int,
    diagonal: float,
    thresholds: Thresholds,
    measured: _Measured | None = None,
) -> SupportPlaneResult:
    """The whole input back, on a record that says which gate refused and why.

    The candidate that was measured is carried even though it was not acted on:
    a withdrawal whose numbers are not visible is indistinguishable from one
    that never looked, and this repository's rule is that a filter says what it
    saw.
    """
    return SupportPlaneResult(
        points=tuple(checked),
        removed=(),
        applied=False,
        reason=reason,
        plane=None if measured is None else measured.plane,
        band=band,
        input_points=count,
        inlier_points=0 if measured is None else measured.inliers,
        off_plane_points=0 if measured is None else measured.off_plane,
        positive_points=0 if measured is None else measured.positive,
        negative_points=0 if measured is None else measured.negative,
        diagonal=diagonal,
        spread_major=0.0 if measured is None else measured.spread_major,
        spread_minor=0.0 if measured is None else measured.spread_minor,
        spread_aspect=0.0 if measured is None else measured.aspect,
        object_footprint_major=1.0 if measured is None else measured.footprint_major,
        object_footprint_minor=1.0 if measured is None else measured.footprint_minor,
        candidates_scored=0 if measured is None else measured.candidates,
        sample_points=0 if measured is None else measured.sample,
        thresholds=thresholds,
        limitations=(
            *INHERENT_LIMITATIONS,
            f"this frame was returned unchanged: {reason}",
        ),
    )
