"""Turning a caliper reading into a pass, a fail, or an honest 'cannot decide'.

This module is the machinery for the one experiment this repository has never
run: comparing a distance this scanner produced against a distance measured
independently. Everything upstream of here - the frame, the factory intrinsics,
the ``0x918`` scalar, the deprojection - is arithmetic over recovered constants,
and arithmetic produces plausible-looking numbers from any inputs at all
(``docs/depth-to-pointcloud.md``). Nothing in this repository has yet compared
one of those numbers with a physical reference. This module is what such a
comparison would have to go through, and it is built to make the comparison
hard to fake rather than easy to pass.

The trap this module is shaped around
-------------------------------------
This unit's factory ``cx`` sits 277.56 px left of the 320 px frame centre. If
the true principal point were the frame centre, the deprojection adds

    x += (320 - 42.44) / fx * z  =  0.3089 * z

to every point. That is a **shear along z, not a rigid shift**, and a shear maps
planes to planes: a flat target, fitted with a plane, produces a residual just as
small with an arbitrarily wrong principal point as with a right one. A validation
harness that supports only a flat target is therefore worse than no harness at
all, because it manufactures confidence out of an experiment that cannot fail.

Three consequences run through the whole module:

* :func:`fit_plane` produces **no length**. Its
  :attr:`PlaneFit.fitted_value` is ``None`` and validating it is a first-class
  :attr:`Verdict.INCONCLUSIVE`, with the reason spelled out. The flat-target
  experiment is not refused - a refusal is easy to route around - it is
  performed, recorded, and reported as deciding nothing.
* :func:`fit_sphere` reports a **shear indicator**. A shear maps a sphere to an
  ellipsoid, so the signed radial residual of a sphere fit acquires an ``x``-``z``
  cross term whose size names the shear coefficient directly. Structured
  residual is evidence of a systematic fault; noisy residual is not. The
  indicator separates those two and says plainly which causes it cannot tell
  apart.
* :func:`plane_separation` reports how sensitive its own geometry is to a shear.
  Two faces whose common normal has no ``x`` component are exactly blind to it -
  the separation is unchanged - so such a pair can pass cleanly while ``cx`` is
  wrong, and the result says so in its own
  :attr:`ValidationResult.does_not_establish`.

What a verdict here does and does not mean
------------------------------------------
* A tolerance must be **declared before the fit is computed**, and the check is
  enforced (:class:`ToleranceDeclaredLate`). A bound chosen after the numbers
  are in is not a test; it is a description.
* A ground truth must carry its **own uncertainty**. A comparison against a
  reference of unstated precision cannot fail honestly, so
  :class:`CaliperMeasurement` cannot be built without one.
* Uncertainties combine **in quadrature**, and the result reports the band
  ``|difference| +/- combined`` rather than a bare difference. A pass requires
  the whole band inside the tolerance; a fail requires the whole band outside;
  anything else is :attr:`Verdict.INCONCLUSIVE`, which is a real outcome and not
  a soft failure.
* **No physical unit appears anywhere in this module.** The comparison is
  arithmetic between two numbers, and it presumes only that the operator
  intended them to be on the same scale. A pass is evidence for that intention
  at the distances measured; it attaches no unit to the depth scale, and neither
  does a fail.
* A pass over one dimension at one distance establishes **scale at that
  distance, in that part of the frame, on this one unit** - and nothing about
  linearity, distortion, or the principal point. Every result carries that
  sentence in :attr:`ValidationResult.does_not_establish` so a consumer reading
  only the JSON cannot lose it.

Cost, because a real frame is up to ~89000 points: every fit here is O(n) per
pass with a fixed iteration ceiling, there is no pairwise work anywhere, and the
point count is bounded by :data:`MAX_FIT_POINTS` before anything is allocated.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, dot, norm, subtract

#: Every record this module serialises stamps this.
SCHEMA_VERSION: Final = 1

#: The largest point set any fit here will accept. Twice a full 640x400 frame,
#: so a whole frame - or a whole frame plus a second view - passes, while a
#: fused multi-frame cloud handed over by accident is refused rather than
#: quietly turned into minutes of pure-Python arithmetic. Checked against
#: ``len()`` before a single coordinate is read, which is why these functions
#: take a sequence rather than an iterable: a bound that can only be enforced
#: after consuming the input is not a bound.
MAX_FIT_POINTS: Final = 512_000

#: Fewest points a sphere fit will accept. Four is the algebraic minimum, and
#: four is exactly the count at which the residual is identically zero: a fit
#: that cannot produce a nonzero residual cannot be disbelieved, and this module
#: exists to be disbelieved. Eight leaves four degrees of freedom for the
#: residual to live in.
MIN_SPHERE_POINTS: Final = 8

#: Fewest points a plane fit will accept, for the same reason: three points
#: define a plane exactly and would report a perfect fit whatever they were.
MIN_PLANE_POINTS: Final = 6

#: Hard ceiling on the geometric refinement below. A Gauss-Newton step on four
#: parameters converges in a handful of iterations from an algebraic start; a
#: run that reaches this ceiling has not converged and says so rather than
#: iterating on the caller's time.
MAX_REFINEMENT_ITERATIONS: Final = 32

#: Refinement stops when a step moves the centre and radius by less than this
#: fraction of the radius. Well above float64 dust, so convergence is a real
#: statement about the fit rather than about rounding.
REFINEMENT_STEP_TOLERANCE: Final = 1e-13

#: A pivot smaller than this fraction of the system's largest entry means the
#: normal equations are singular to working precision: the point set does not
#: determine the shape being fitted. Scale-relative, because the systems here
#: are built from squared coordinates and an absolute floor would mean something
#: different for every input.
PIVOT_FLOOR_FRACTION: Final = 1e-12

#: How much of a sphere fit's residual must lie along the shear direction before
#: the indicator fires, as an uncentred correlation in [-1, 1]. Its square is the
#: fraction of residual energy the shear model explains, so 0.35 fires at about
#: 12% - deliberately sensitive. The failure this module exists to prevent is a
#: false pass, so the threshold is set where it costs inconclusive verdicts
#: rather than where it buys clean ones.
SHEAR_CORRELATION_THRESHOLD: Final = 0.35

#: Below this the point set has too little *detectable* spread in the ``x``-``z``
#: product for the shear indicator to see anything, and its reading means nothing
#: either way. The quantity is ``rms(dx * dz) / radius^2`` after the part of that
#: shape the fit's own centre and radius can absorb has been removed - see
#: :func:`fit_sphere` for why the removal is what makes the number mean anything.
#: For points spread evenly over a whole sphere it is ``1 / sqrt(15)`` ~ 0.258,
#: so this floor is about a fifth of full leverage. Coverage costs leverage
#: steeply and that is the point: an even hemisphere reads about 0.14, a
#: 60-degree cap about 0.055, and a 30-degree cap falls under this floor, which
#: is the honest report that such a cap cannot see a shear rather than that it
#: looked and found none.
MIN_SHEAR_LEVERAGE: Final = 0.05

#: The smallest angular cap, in radians, over which a sphere fit's centre and
#: radius are separately determined. Over a narrow cap the two trade off almost
#: freely: a slightly larger sphere pushed slightly further away fits the same
#: points, so the fitted diameter is weakly determined however small the residual
#: looks. 0.35 rad is 20 degrees.
MIN_CAP_HALF_ANGLE: Final = 0.35

#: How far the two fitted normals of a face pair may diverge, in radians, before
#: the pair no longer describes one separation. 0.05 rad is 2.9 degrees.
MAX_NORMAL_DISAGREEMENT: Final = 0.05

#: A normal whose ``x`` component is at least this large carries some sensitivity
#: to a principal-point shear. Below it the face pair is, for practical purposes,
#: blind to one - which is a fact about the experiment, reported, never corrected.
MIN_SHEAR_SENSITIVITY: Final = 0.1

#: The limitations that hold for every verdict this module reaches, whatever the
#: inputs were. Fit-specific limitations are appended to these, never substituted.
STATIC_LIMITATIONS: Final[tuple[str, ...]] = (
    "this module names no physical unit and attaches none. A verdict compares two numbers on "
    "the assumption that the operator intended them to be on the same scale; agreement is "
    "evidence for that intention at the distances measured and is not a unit",
    "a fit is least squares over exactly the points it was given. Nothing here decides which "
    "points came from the target: no segmentation, no outlier rejection and no background "
    "removal is applied, so a stray surface in the point set moves the answer silently",
    "one target at one distance constrains scale at that distance only. Linearity of the "
    "raw-to-distance mapping, lens distortion, and the principal point are each separate "
    "questions that need the target moved, not the tolerance widened",
    "no lens distortion is modelled or inverted anywhere upstream of these points, so any "
    "distortion present is inside every residual this module reports and is not separated "
    "from the fault the residual is being read for",
    "the predeclaration check compares a wall-clock timestamp the tolerance carries against "
    "one the fit stamped. It catches a tolerance written down carelessly after the fact; it "
    "cannot catch one written down dishonestly, and it is not offered as proof of order",
)


# --- refusals -----------------------------------------------------------------


class ValidationNotAllowed(ScannerError):
    """A validation was refused before any fit was computed or verdict reached.

    Every subclass is raised up front, so a refused call has fitted nothing and
    produced no partial record. A caller that catches this base has caught every
    refusal this module makes - which is why the finiteness scan below repeats
    work :func:`scanner3d.geometry.ensure_finite` would also do rather than
    letting a ``GeometryNotAllowed`` escape from underneath and split the
    hierarchy a caller has to handle.

    The status is :attr:`~scanner3d.errors.DoctorStatus.UNSUPPORTED_MODE`
    because the doctor vocabulary has no class for "these inputs do not describe
    an experiment that can decide anything", and that is nearer the truth than
    calling a caller's bad argument an internal error.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class PointSetUnusable(ValidationNotAllowed):
    """The point set cannot carry the fit that was asked of it.

    Empty, too few to leave the residual any freedom, or past
    :data:`MAX_FIT_POINTS`. The lower bounds are not the algebraic minimums: a
    fit with exactly as many points as parameters reports a residual of zero
    whatever the points were, and a residual that cannot be nonzero is not
    evidence.
    """


class NonFiniteInput(ValidationNotAllowed):
    """A point's coordinate is NaN or an infinity.

    Raised before any moment is accumulated. One non-finite coordinate poisons
    every sum it enters, and the resulting fit fails in a way that looks like a
    fit rather than like an error. A non-finite *caliper reading* or *tolerance*
    is refused separately, by :class:`MeasurementUnusable` and
    :class:`ToleranceUnusable`, so that a bad ground truth and a bad point set
    stay distinguishable to a caller handling them.
    """


class DegenerateFit(ValidationNotAllowed):
    """The points do not determine the shape being fitted.

    Coplanar points do not determine a sphere; collinear points do not determine
    a plane. In both cases the normal equations are singular to working
    precision and any answer returned would be a direction picked out of
    rounding error, reported with a small residual because the points really do
    lie on the degenerate shape.
    """


class MeasurementUnusable(ValidationNotAllowed):
    """A caliper reading cannot serve as ground truth as stated.

    A blank label, claimant or instrument; a non-positive or non-finite reading;
    or - the one this class mostly exists for - an uncertainty of zero. Zero
    uncertainty is a claim that the instrument is perfect, and a comparison
    against a perfect reference cannot fail honestly.
    """


class ToleranceUnusable(ValidationNotAllowed):
    """A tolerance cannot decide anything as stated.

    A non-positive or non-finite limit, a blank claimant or rationale, or a
    declaration time that is not a finite positive wall-clock reading.
    """


class ToleranceDeclaredLate(ValidationNotAllowed):
    """The tolerance was declared after the fit it is being applied to.

    A bound chosen once the numbers are in describes the result; it does not
    test it, and no verdict computed from it means anything. Refused rather than
    reported, because a record that carried both the verdict and the note that
    the verdict is meaningless would still be quoted for its verdict.
    """


class QuantityMismatch(ValidationNotAllowed):
    """The tolerance was declared for a different quantity than the fit produced.

    A tolerance is bound to a named quantity at declaration time precisely so it
    cannot be moved onto whichever fit happens to satisfy it.
    """


def _ensure_point_set(points: Sequence[Point], minimum: int, what: str) -> Sequence[Point]:
    """Return ``points`` if it can carry a fit, else raise.

    Order matters. All three count bounds are settled from ``len()`` alone,
    before a single coordinate is looked at, so an oversized set is refused
    rather than scanned; only then is the surviving set checked for finiteness.
    That scan is the one O(n) pass this module makes before committing to a fit,
    and it is still cheaper than the fit it protects.
    """
    count = len(points)
    if count == 0:
        raise PointSetUnusable(
            f"a {what} needs points and was given none. An empty set has no shape, and a fit "
            "over it would report a centre, a radius and a residual computed from nothing"
        )
    if count < minimum:
        raise PointSetUnusable(
            f"a {what} needs at least {minimum} points and was given {count}. Fewer than that "
            "leaves the residual no freedom to be nonzero, so the fit could not fail whatever "
            "the points were"
        )
    if count > MAX_FIT_POINTS:
        raise PointSetUnusable(
            f"a {what} was given {count} points, past the {MAX_FIT_POINTS} ceiling. The bound "
            "is checked before any coordinate is read, so nothing was fitted; downsample, or "
            "fit the views separately and compare the verdicts"
        )
    for index, point in enumerate(points):
        if not (math.isfinite(point[0]) and math.isfinite(point[1]) and math.isfinite(point[2])):
            raise NonFiniteInput(
                f"{what}: point {index} is {point!r}, which is not finite. One such coordinate "
                "poisons every sum it enters and the fit would fail looking like a fit"
            )
    return points


# --- the vocabulary a bound is declared in ------------------------------------


class FittedQuantity(enum.StrEnum):
    """The named quantities a fit here can produce and a tolerance can bind to.

    :attr:`PLANE_FLATNESS` is in the list so that the flat-target experiment can
    be run and recorded rather than merely refused - it produces no length, and
    its verdict is always :attr:`Verdict.INCONCLUSIVE`.
    """

    SPHERE_DIAMETER = "sphere_diameter"
    PLANE_SEPARATION = "plane_separation"
    PLANE_FLATNESS = "plane_flatness"


class Verdict(enum.StrEnum):
    """The three outcomes. ``INCONCLUSIVE`` is an outcome, not a soft failure."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


# --- the ground truth and the bound declared against it -----------------------


@dataclass(frozen=True, slots=True)
class CaliperMeasurement:
    """A dimension measured independently of this scanner, with its precision.

    Validated in ``__post_init__`` rather than by an ``ensure_*`` function, which
    is this module's one departure from the pattern the rest of the package
    follows. The reason is that nothing stands between a caller and this record:
    it is constructed directly at the call site, so a check that lives in a
    separate function is a check a call site can skip. ``uncertainty`` in
    particular is a required field with a refusal behind it, because a ground
    truth of unstated precision cannot support a verdict in either direction -
    it can only support the verdict its reader was hoping for.
    """

    label: str
    value: float
    uncertainty: float
    measured_by: str
    instrument: str

    def __post_init__(self) -> None:
        for name, text in (
            ("label", self.label),
            ("measured_by", self.measured_by),
            ("instrument", self.instrument),
        ):
            if not text.strip():
                raise MeasurementUnusable(
                    f"the caliper measurement has a blank {name}. A ground truth with no "
                    "owner or no named instrument cannot be traced back to anything, and a "
                    "verdict computed against it is unattributable"
                )
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise MeasurementUnusable(
                f"the caliper measurement reads {self.value!r}; a measured dimension must be "
                "a finite positive number"
            )
        if not math.isfinite(self.uncertainty) or self.uncertainty <= 0.0:
            raise MeasurementUnusable(
                f"the caliper measurement states an uncertainty of {self.uncertainty!r}. Every "
                "instrument has one, and a stated zero is a claim that this one is perfect: a "
                "comparison against a perfect reference cannot fail honestly, so it is refused"
            )

    @property
    def relative_uncertainty(self) -> float:
        """The stated uncertainty as a fraction of the reading."""
        return self.uncertainty / self.value

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "relative_uncertainty": self.relative_uncertainty,
            "measured_by": self.measured_by,
            "instrument": self.instrument,
            "note": (
                "measured independently of this scanner. The reading and its uncertainty are "
                "on whatever scale the instrument works in; this module names no unit and "
                "converts nothing"
            ),
        }


@dataclass(frozen=True, slots=True)
class Tolerance:
    """A bound on disagreement, declared before the measurement it will judge.

    ``declared_at`` is a wall-clock reading rather than a monotonic one on
    purpose: a tolerance is written down in one session and applied in another,
    possibly on another day, and a monotonic reading has no meaning across
    processes. :func:`validate` refuses when it is not strictly earlier than the
    fit's own stamp.

    That check is worth exactly what it is: it catches a bound written down
    after the numbers came in, which is the common and mostly careless way this
    goes wrong. It does not catch a forged timestamp, and it is not offered as
    proof of order - only as the difference between a project that records the
    order and one that does not.

    ``quantity`` binds the bound to one named thing at declaration time, so a
    tolerance cannot later be moved onto whichever fit happens to satisfy it.
    """

    quantity: FittedQuantity
    limit: float
    declared_by: str
    declared_at: float
    rationale: str

    def __post_init__(self) -> None:
        for name, text in (("declared_by", self.declared_by), ("rationale", self.rationale)):
            if not text.strip():
                raise ToleranceUnusable(
                    f"the tolerance has a blank {name}. A bound with no owner and no stated "
                    "reason is a number someone can always find a justification for afterwards"
                )
        if not math.isfinite(self.limit) or self.limit <= 0.0:
            raise ToleranceUnusable(
                f"the tolerance limit is {self.limit!r}; it must be a finite positive number. "
                "A zero limit can only ever be failed and decides nothing about the scanner"
            )
        if not math.isfinite(self.declared_at) or self.declared_at <= 0.0:
            raise ToleranceUnusable(
                f"the tolerance was declared at {self.declared_at!r}, which is not a usable "
                "wall-clock reading, so its order against the fit cannot be established"
            )

    @classmethod
    def declare(
        cls, *, quantity: FittedQuantity, limit: float, declared_by: str, rationale: str
    ) -> Tolerance:
        """Declare a tolerance now, stamped with the current wall clock.

        The honest path made the short one. Reaching for the constructor and
        supplying ``declared_at`` by hand is still possible - a stored
        declaration has to be replayable - but it is the longer path, and it is
        the only way to state a time this process did not observe.
        """
        return cls(
            quantity=quantity,
            limit=limit,
            declared_by=declared_by,
            declared_at=time.time(),
            rationale=rationale,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "quantity": str(self.quantity),
            "limit": self.limit,
            "declared_by": self.declared_by,
            "declared_at": self.declared_at,
            "rationale": self.rationale,
            "note": (
                "declared before the fit it judges; validate() refuses a tolerance stamped "
                "later than the fit, and states in the record's limitations what that check "
                "is and is not worth"
            ),
        }


# --- what a fit produces ------------------------------------------------------


class DimensionalFit(Protocol):
    """What :func:`validate` needs from a fit, whatever shape produced it.

    Stated as a protocol rather than a union so that the contract is written
    down in one place: a fit must name its quantity, offer a value or admit it
    has none, carry its own residual and its own conservative uncertainty, say
    what would already make it undecidable, and say what it cannot establish
    even when it decides.
    """

    @property
    def quantity(self) -> FittedQuantity: ...
    @property
    def fitted_value(self) -> float | None: ...
    @property
    def fit_uncertainty(self) -> float: ...
    @property
    def residual_rms(self) -> float: ...
    @property
    def residual_max(self) -> float: ...
    @property
    def point_count(self) -> int: ...
    @property
    def fitted_at(self) -> float: ...
    @property
    def z_min(self) -> float: ...
    @property
    def z_max(self) -> float: ...
    @property
    def inconclusive_reasons(self) -> tuple[str, ...]: ...
    @property
    def does_not_establish(self) -> tuple[str, ...]: ...
    @property
    def limitations(self) -> tuple[str, ...]: ...
    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class SphereFit:
    """A least-squares sphere, its residual, and the shear it may be hiding.

    The centre and radius are the small part. What makes this fit useful for
    validating *this* scanner is :attr:`shear_correlation`: a shear maps a sphere
    to an ellipsoid, and to first order in the shear coefficient ``k`` the signed
    radial residual of a best-fit sphere becomes ``k * dx * dz / radius``, where
    ``dx`` and ``dz`` are offsets from the fitted centre. That is a specific
    shape, not a general roughness, so a residual which lies along it is evidence
    of a systematic fault and a residual which does not is evidence of noise.

    :attr:`shear_coefficient` reads the size of that term back out as the ``k``
    that would explain it - the same dimensionless ``x`` per ``z`` that a wrong
    principal point contributes, so it is directly comparable against
    ``(centre_column - cx) / fx``.

    What the indicator cannot do, stated plainly: it detects an ``x``-``z`` cross
    term in the radial residual and identifies **no cause**. A wrong principal
    point produces one. So would a target that is genuinely an ellipsoid, a
    coupling between ``x`` and ``z`` anywhere in the depth model, or lens
    distortion across the part of the frame the target occupies. It separates
    structured from noisy; it does not tell them apart from each other, and a
    fired indicator is a reason to stop, not a diagnosis.

    A quiet indicator is only meaningful when the points have leverage to see
    with: :attr:`shear_leverage` reports how much, and below
    :data:`MIN_SHEAR_LEVERAGE` a quiet reading means nothing either way. Leverage
    here counts only the part of the shear shape this fit's own centre and radius
    could not absorb, which is why a narrow cap reports little of it however
    widely its raw ``dx * dz`` ranges.
    """

    centre: Point
    radius: float
    point_count: int
    residual_rms: float
    residual_max: float
    shear_correlation: float
    shear_coefficient: float
    shear_leverage: float
    cap_half_angle: float
    converged: bool
    iterations: int
    z_min: float
    z_max: float
    fitted_at: float

    @property
    def quantity(self) -> FittedQuantity:
        return FittedQuantity.SPHERE_DIAMETER

    @property
    def diameter(self) -> float:
        """Twice the radius - the quantity a caliper across a ball actually reads."""
        return 2.0 * self.radius

    @property
    def fitted_value(self) -> float | None:
        return self.diameter

    @property
    def fit_uncertainty(self) -> float:
        """The RMS residual, doubled because a diameter spans two surfaces.

        Deliberately not divided by the square root of the point count. That
        division is the standard way to shrink a fit's stated uncertainty, and it
        assumes the per-point errors are independent. Structured-light depth
        noise is correlated across neighbouring pixels - speckle and the pattern
        period are both larger than a pixel - so the assumption is not available
        here and the conservative figure is reported instead.
        """
        return 2.0 * self.residual_rms

    @property
    def shear_indicator_reliable(self) -> bool:
        """Whether these points spread enough in ``dx * dz`` to see a shear at all."""
        return self.shear_leverage >= MIN_SHEAR_LEVERAGE

    @property
    def shear_detected(self) -> bool:
        """Whether the residual lies along the shear direction, and could be seen to."""
        return (
            self.shear_indicator_reliable
            and abs(self.shear_correlation) >= SHEAR_CORRELATION_THRESHOLD
        )

    @property
    def well_conditioned(self) -> bool:
        """Whether the points cover enough of the sphere to fix centre and radius apart."""
        return self.cap_half_angle >= MIN_CAP_HALF_ANGLE

    @property
    def inconclusive_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.shear_detected:
            reasons.append(
                f"the sphere's radial residual lies along the x-z shear direction "
                f"(correlation {self.shear_correlation:+.3f}, at or past the "
                f"{SHEAR_CORRELATION_THRESHOLD} threshold), consistent with a shear of about "
                f"{self.shear_coefficient:+.4f} x per z. The fitted diameter is then a "
                "measurement of the target convolved with an unknown shear, not of the target"
            )
        if not self.shear_indicator_reliable:
            reasons.append(
                f"the points have a shear leverage of {self.shear_leverage:.4f}, below the "
                f"{MIN_SHEAR_LEVERAGE} floor, so the shear indicator is quiet because it "
                "cannot see rather than because there is nothing there. A whole sphere would "
                "read about 0.258 and an even hemisphere about 0.14; a cap this narrow has "
                "almost no shear signal the fitted centre and radius have not already absorbed"
            )
        if not self.well_conditioned:
            reasons.append(
                f"the points cover a cap of half-angle {self.cap_half_angle:.3f} rad, below "
                f"the {MIN_CAP_HALF_ANGLE} rad floor. Over a cap that narrow a larger sphere "
                "pushed further away fits the same points, so the diameter is weakly "
                "determined however small the residual is"
            )
        if not self.converged:
            reasons.append(
                f"the geometric refinement did not converge within {MAX_REFINEMENT_ITERATIONS} "
                "iterations; the reported centre and radius are the last iterate, and their "
                "residual is not a minimum anything was found at"
            )
        return tuple(reasons)

    @property
    def does_not_establish(self) -> tuple[str, ...]:
        return (
            "the shear indicator identifies no cause. An x-z cross term in the radial residual "
            "is what a wrong principal point produces, and also what a genuinely ellipsoidal "
            "target, an x-z coupling in the depth model, or lens distortion over this part of "
            "the frame would produce; this fit separates structured residual from noise and "
            "does not attribute it",
            "a residual shaped like z-squared rather than like x times z is a different fault "
            "again - a nonlinear raw-to-distance mapping - and this indicator does not report it",
        )

    @property
    def limitations(self) -> tuple[str, ...]:
        return (
            "the fitted centre and radius minimise the sum of squared radial residuals over "
            "exactly the points given, with no weighting: a denser patch of the target pulls "
            "the fit toward itself",
            "the reported uncertainty is twice the RMS residual and is not divided by the "
            "square root of the point count, because that division assumes independent "
            "per-point noise and structured-light speckle is correlated across neighbours",
            *self.does_not_establish,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "quantity": str(self.quantity),
            "centre": list(self.centre),
            "radius": self.radius,
            "diameter": self.diameter,
            "point_count": self.point_count,
            "residual": {
                "rms": self.residual_rms,
                "max": self.residual_max,
                "note": (
                    "signed radial residual, distance from the fitted centre minus the fitted "
                    "radius, on whatever scale the input points are on"
                ),
            },
            "shear_indicator": {
                "correlation": self.shear_correlation,
                "coefficient": self.shear_coefficient,
                "explained_fraction": self.shear_correlation**2,
                "leverage": self.shear_leverage,
                "leverage_floor": MIN_SHEAR_LEVERAGE,
                "threshold": SHEAR_CORRELATION_THRESHOLD,
                "reliable": self.shear_indicator_reliable,
                "detected": self.shear_detected,
                "model": "signed radial residual ~ coefficient * dx * dz / radius",
                "note": (
                    "the coefficient is dimensionless x per z, directly comparable against "
                    "(assumed principal column - cx) / fx. It names a size, not a cause. "
                    "Correlation and coefficient are partial: the shape is first stripped of "
                    "the part the fitted centre and radius already absorbed, without which "
                    "both read near zero over a cap however large the shear"
                ),
            },
            "conditioning": {
                "cap_half_angle": self.cap_half_angle,
                "cap_half_angle_floor": MIN_CAP_HALF_ANGLE,
                "well_conditioned": self.well_conditioned,
                "converged": self.converged,
                "iterations": self.iterations,
                "iteration_ceiling": MAX_REFINEMENT_ITERATIONS,
            },
            "z_span": [self.z_min, self.z_max],
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class PlaneFit:
    """A least-squares plane - and the reason it can decide nothing on its own.

    :attr:`fitted_value` is ``None``, and that is the substance of this class
    rather than an omission in it. Two separate facts make a plane fit unable to
    validate this scanner:

    * A plane fit produces no length. There is nothing here for a caliper
      reading to be compared against.
    * A shear maps planes to planes. The residual of this fit would be exactly as
      small with an arbitrarily wrong principal point as with a right one, so
      even a perfect flatness result is consistent with the one calibration
      error this unit is most likely to have.

    The flat-target experiment is therefore performed and recorded here rather
    than refused, and :func:`validate` reports it as
    :attr:`Verdict.INCONCLUSIVE` with both reasons attached. A refusal is easy to
    route around; a recorded verdict of "this cannot decide" is what a reader
    later finds in the file.

    The fit itself is still worth having: :func:`plane_separation` builds on two
    of them, and the residual is a real flatness observation about the target and
    the sensor together.
    """

    normal: Point
    offset: float
    centroid: Point
    point_count: int
    residual_rms: float
    residual_max: float
    eigenvalues: tuple[float, float, float]
    z_min: float
    z_max: float
    fitted_at: float

    @property
    def quantity(self) -> FittedQuantity:
        return FittedQuantity.PLANE_FLATNESS

    @property
    def fitted_value(self) -> float | None:
        """Always ``None``: a plane fit yields no length to compare against anything."""
        return None

    @property
    def fit_uncertainty(self) -> float:
        return self.residual_rms

    @property
    def shear_sensitivity(self) -> float:
        """How much of this plane's normal points along ``x``.

        Reported for the same reason as everywhere else in this module: a shear
        along ``x`` moves a plane's normal by an amount proportional to this, and
        a reader should be able to see that a plane facing the camera is exactly
        the orientation that hides one.
        """
        return abs(self.normal[0])

    @property
    def inconclusive_reasons(self) -> tuple[str, ...]:
        return (
            "a plane fit produces no length, so there is no fitted dimension for a caliper "
            "reading to be compared against",
            "a shear maps planes to planes: this residual would be just as small with an "
            "arbitrarily wrong principal point, so a flat target fitted with a plane is "
            "precisely the experiment that passes while cx is wrong",
        )

    @property
    def does_not_establish(self) -> tuple[str, ...]:
        return (
            "flatness establishes nothing about scale. A cloud scaled by any factor, or "
            "sheared by any coefficient, is exactly as flat as this one",
            "the residual mixes the target's own flatness with the sensor's noise and with "
            "any lens distortion over this part of the frame; nothing here separates them",
        )

    @property
    def limitations(self) -> tuple[str, ...]:
        return (
            "the normal is the covariance eigenvector of least variance over exactly the "
            "points given, unweighted and with no outlier rejection",
            "the normal's sign is fixed by a deterministic convention - its largest-magnitude "
            "component is made positive - so that two runs over the same points report the "
            "same normal. The convention carries no claim about which side of the plane the "
            "target is on",
            *self.does_not_establish,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "quantity": str(self.quantity),
            "normal": list(self.normal),
            "offset": self.offset,
            "centroid": list(self.centroid),
            "point_count": self.point_count,
            "residual": {
                "rms": self.residual_rms,
                "max": self.residual_max,
                "note": "signed distance from the fitted plane, on the input points' own scale",
            },
            "covariance_eigenvalues": list(self.eigenvalues),
            "shear_sensitivity": self.shear_sensitivity,
            "fitted_value": None,
            "z_span": [self.z_min, self.z_max],
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class SeparationFit:
    """The distance between two fitted parallel faces - a length a caliper can judge.

    This is the one flat-target experiment that produces a dimension, and it is
    the reason :func:`fit_plane` is worth having despite deciding nothing itself.

    Its blind spot is specific and is reported rather than corrected. Under a
    shear ``x += k * z`` a plane with unit normal ``n`` acquires the unnormalised
    normal ``(n_x, n_y, n_z - k * n_x)``, so the separation between two parallel
    faces is divided by the length of that vector. When ``n_x`` is zero - faces
    square-on to the camera, which is how a gauge block is naturally presented -
    the length is exactly one and the separation is completely unchanged. Such a
    pair validates scale along the optical axis and is exactly blind to the
    principal point. :attr:`shear_sensitivity` reports which case this pair is
    in, and the verdict carries the consequence.
    """

    separation: float
    normal: Point
    plane_a: PlaneFit
    plane_b: PlaneFit
    normal_disagreement: float
    z_min: float
    z_max: float
    fitted_at: float

    @property
    def quantity(self) -> FittedQuantity:
        return FittedQuantity.PLANE_SEPARATION

    @property
    def fitted_value(self) -> float | None:
        return self.separation

    @property
    def point_count(self) -> int:
        return self.plane_a.point_count + self.plane_b.point_count

    @property
    def residual_rms(self) -> float:
        """The two faces' flatness residuals combined in quadrature."""
        return math.sqrt(self.plane_a.residual_rms**2 + self.plane_b.residual_rms**2)

    @property
    def residual_max(self) -> float:
        return max(self.plane_a.residual_max, self.plane_b.residual_max)

    @property
    def fit_uncertainty(self) -> float:
        """The combined flatness residual, undivided by the square root of the count.

        Each face's offset along the common normal is uncertain by roughly its
        own flatness residual, and the two combine in quadrature because they are
        separate fits over separate points. The usual division by the square root
        of the point count is not applied, for the same reason as in
        :attr:`SphereFit.fit_uncertainty`: it assumes independent per-point
        noise, and structured-light speckle is correlated across neighbours.
        """
        return self.residual_rms

    @property
    def shear_sensitivity(self) -> float:
        """How much of the common normal points along ``x``, in ``[0, 1]``."""
        return abs(self.normal[0])

    @property
    def sensitive_to_shear(self) -> bool:
        return self.shear_sensitivity >= MIN_SHEAR_SENSITIVITY

    @property
    def faces_parallel(self) -> bool:
        return self.normal_disagreement <= MAX_NORMAL_DISAGREEMENT

    @property
    def inconclusive_reasons(self) -> tuple[str, ...]:
        if self.faces_parallel:
            return ()
        return (
            f"the two fitted normals disagree by {self.normal_disagreement:.4f} rad, past the "
            f"{MAX_NORMAL_DISAGREEMENT} rad limit. Non-parallel faces do not have one "
            "separation, so the number reported is a separation along an averaged direction "
            "that neither face is square to",
        )

    @property
    def does_not_establish(self) -> tuple[str, ...]:
        if self.sensitive_to_shear:
            shear_note = (
                f"the common normal has an x component of {self.shear_sensitivity:.3f}, so a "
                "shear along x would change this separation by a factor of about "
                "sqrt(1 + (k * n_x)^2). A pass therefore bounds the shear only as coarsely as "
                "that factor allows, which for a small coefficient is very coarse indeed"
            )
        else:
            shear_note = (
                f"the common normal has an x component of {self.shear_sensitivity:.3f}, below "
                f"the {MIN_SHEAR_SENSITIVITY} floor: this face pair is effectively blind to a "
                "shear along x. A shear leaves the separation between two faces square-on to "
                "the camera exactly unchanged, so this experiment can pass cleanly while the "
                "principal point is arbitrarily wrong. Turn the target, or use a sphere"
            )
        return (
            shear_note,
            "the separation is measured along the direction the two fits agree on, not along "
            "any direction the operator intended; a target rotated relative to its assumed "
            "orientation is not detected here",
        )

    @property
    def limitations(self) -> tuple[str, ...]:
        return (
            "the separation is the projection of the two face centroids' difference onto a "
            "single common normal averaged from the two fits, so a real wedge between the "
            "faces is reported as one distance and its disagreement angle, not as two",
            "the uncertainty is the two flatness residuals in quadrature and is not divided "
            "by the square root of the point count, because that would assume independent "
            "per-point noise",
            *self.does_not_establish,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "quantity": str(self.quantity),
            "separation": self.separation,
            "common_normal": list(self.normal),
            "normal_disagreement": self.normal_disagreement,
            "normal_disagreement_limit": MAX_NORMAL_DISAGREEMENT,
            "faces_parallel": self.faces_parallel,
            "shear_sensitivity": self.shear_sensitivity,
            "shear_sensitivity_floor": MIN_SHEAR_SENSITIVITY,
            "sensitive_to_shear": self.sensitive_to_shear,
            "point_count": self.point_count,
            "residual": {"rms": self.residual_rms, "max": self.residual_max},
            "plane_a": self.plane_a.as_dict(),
            "plane_b": self.plane_b.as_dict(),
            "z_span": [self.z_min, self.z_max],
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "limitations": list(self.limitations),
        }


# --- linear algebra, small and dense ------------------------------------------


def _solve(matrix: list[list[float]], rhs: list[float], *, what: str) -> list[float]:
    """Solve a small dense system by Gaussian elimination with partial pivoting.

    Refuses a singular system rather than returning the direction rounding error
    happened to leave in the smallest pivot. The floor is relative to the
    system's own largest entry because these matrices are built from squared
    coordinates and an absolute floor would mean something different for every
    input scale.
    """
    size = len(rhs)
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    floor = scale * PIVOT_FLOOR_FRACTION
    augmented = [[*row, rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = column
        for row in range(column + 1, size):
            if abs(augmented[row][column]) > abs(augmented[pivot][column]):
                pivot = row
        if abs(augmented[pivot][column]) <= floor:
            raise DegenerateFit(
                f"{what}: the normal equations are singular to working precision (pivot "
                f"{augmented[pivot][column]!r} against a system scale of {scale!r}). The "
                "points do not determine this shape, and any answer would be a direction "
                "picked out of rounding error"
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = 1.0 / augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] * inverse
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    solution = [0.0] * size
    for row in reversed(range(size)):
        known = math.fsum(augmented[row][index] * solution[index] for index in range(row + 1, size))
        solution[row] = (augmented[row][size] - known) / augmented[row][row]
    return solution


def _symmetric_eigen(matrix: list[list[float]]) -> tuple[list[float], list[Point]]:
    """Eigenvalues ascending and matching eigenvectors of a symmetric 3x3.

    Cyclic Jacobi, for the same reason :mod:`scanner3d.geometry` uses it on its
    4x4: unconditionally stable for symmetric matrices, needs no starting guess,
    and this repository has no numerical library. That implementation is not
    reused because it is private, fixed at 4x4, and returns only the principal
    eigenvector - a plane needs the *least* one, and the full spectrum is worth
    reporting so a reader can see how planar the points actually were.
    """
    size = 3
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(64):
        off_diagonal = math.fsum(a[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off_diagonal < 1e-30:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(a[p][q]) < 1e-30:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(size):
                    vkp, vkq = vectors[k][p], vectors[k][q]
                    vectors[k][p], vectors[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(size), key=lambda i: a[i][i])
    values = [a[i][i] for i in order]
    directions = [(vectors[0][i], vectors[1][i], vectors[2][i]) for i in order]
    return values, directions


def _mean(points: Sequence[Point]) -> Point:
    """The arithmetic mean, summed pairwise so a large set does not drift."""
    count = len(points)
    return (
        math.fsum(p[0] for p in points) / count,
        math.fsum(p[1] for p in points) / count,
        math.fsum(p[2] for p in points) / count,
    )


def _z_span(points: Sequence[Point]) -> tuple[float, float]:
    """The distances the fit actually saw, so a verdict can name them."""
    return min(p[2] for p in points), max(p[2] for p in points)


def _oriented(normal: Point) -> Point:
    """A deterministic sign for a plane normal.

    An eigenvector is defined up to sign, so two runs over the same points could
    otherwise report opposite normals and a reader comparing them would see a
    flipped plane where nothing changed. The largest-magnitude component is made
    positive. This is a convention for reproducibility and carries no claim
    about which side of the plane the target is on.
    """
    largest = max(range(3), key=lambda i: abs(normal[i]))
    if normal[largest] < 0.0:
        return (-normal[0], -normal[1], -normal[2])
    return normal


# --- the fits -----------------------------------------------------------------


def _algebraic_sphere(centred: Sequence[Point]) -> tuple[Point, float]:
    """Kasa's linear sphere fit over points already centred on their mean.

    Minimises the algebraic residual ``|p - c|^2 - r^2``, which is linear in the
    centre and in ``r^2 - |c|^2`` and therefore has a closed form. It is exact for
    noiseless points and biased for noisy ones over a partial sphere, which is
    why it is only the starting point for the geometric refinement below.

    Centring first is not cosmetic: the normal equations are built from fourth
    moments of the coordinates, and on points sitting hundreds of units from the
    origin - which is exactly where this scanner's clouds sit - the uncentred
    system loses most of its significant digits before it is solved.
    """
    count = len(centred)
    squared = [dot(p, p) for p in centred]
    sums = [
        math.fsum(p[0] * p[0] for p in centred),
        math.fsum(p[0] * p[1] for p in centred),
        math.fsum(p[0] * p[2] for p in centred),
        math.fsum(p[1] * p[1] for p in centred),
        math.fsum(p[1] * p[2] for p in centred),
        math.fsum(p[2] * p[2] for p in centred),
    ]
    linear = [
        math.fsum(p[0] for p in centred),
        math.fsum(p[1] for p in centred),
        math.fsum(p[2] for p in centred),
    ]
    matrix = [
        [4.0 * sums[0], 4.0 * sums[1], 4.0 * sums[2], 2.0 * linear[0]],
        [4.0 * sums[1], 4.0 * sums[3], 4.0 * sums[4], 2.0 * linear[1]],
        [4.0 * sums[2], 4.0 * sums[4], 4.0 * sums[5], 2.0 * linear[2]],
        [2.0 * linear[0], 2.0 * linear[1], 2.0 * linear[2], float(count)],
    ]
    rhs = [
        2.0 * math.fsum(p[0] * s for p, s in zip(centred, squared, strict=True)),
        2.0 * math.fsum(p[1] * s for p, s in zip(centred, squared, strict=True)),
        2.0 * math.fsum(p[2] * s for p, s in zip(centred, squared, strict=True)),
        math.fsum(squared),
    ]
    cx, cy, cz, gauge = _solve(matrix, rhs, what="sphere fit")
    centre = (cx, cy, cz)
    squared_radius = gauge + dot(centre, centre)
    if squared_radius <= 0.0:
        raise DegenerateFit(
            f"sphere fit: the algebraic solution implies a squared radius of {squared_radius!r}. "
            "The points do not lie on any sphere this fit can express - coplanar points are the "
            "usual cause, since a plane is a sphere of infinite radius"
        )
    return centre, math.sqrt(squared_radius)


def _refine_sphere(
    centred: Sequence[Point], centre: Point, radius: float
) -> tuple[Point, float, bool, int]:
    """Gauss-Newton on the geometric residual, from the algebraic start.

    The algebraic fit minimises the wrong thing: ``|p - c|^2 - r^2`` weights a
    point by its distance from the centre, which over a partial sphere pulls the
    radius systematically. That bias is invisible - it produces a clean, small
    residual around a wrong radius - and a wrong radius is precisely the failure
    this module exists to catch, so the extra iterations are not optional.

    The normal equations here are accumulated with ordinary addition rather than
    :func:`math.fsum`, unlike everything else in this module. Rounding in a
    Gauss-Newton step perturbs only the *step*: the iteration is judged by the
    size of the step it takes and the residual it lands on, and both of those are
    measured afterwards with ``fsum``. Buying exactness inside the step would
    cost a pass over the points per matrix entry and change no reported number.
    """
    for iteration in range(1, MAX_REFINEMENT_ITERATIONS + 1):
        normal_matrix = [[0.0] * 4 for _ in range(4)]
        rhs = [0.0] * 4
        for point in centred:
            dx = point[0] - centre[0]
            dy = point[1] - centre[1]
            dz = point[2] - centre[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= 0.0:
                raise DegenerateFit(
                    "sphere fit: a point coincides with the fitted centre, so it has no radial "
                    "direction and the fit has no gradient there"
                )
            ux, uy, uz = dx / distance, dy / distance, dz / distance
            residual = distance - radius
            row = (ux, uy, uz, 1.0)
            for i in range(4):
                for j in range(i, 4):
                    normal_matrix[i][j] += row[i] * row[j]
                rhs[i] += row[i] * residual
        for i in range(4):
            for j in range(i):
                normal_matrix[i][j] = normal_matrix[j][i]
        step = _solve(normal_matrix, rhs, what="sphere refinement")
        centre = (centre[0] + step[0], centre[1] + step[1], centre[2] + step[2])
        radius += step[3]
        if radius <= 0.0:
            raise DegenerateFit(
                f"sphere fit: the refinement drove the radius to {radius!r}. The points do not "
                "describe a sphere of positive radius"
            )
        if norm((step[0], step[1], step[2])) + abs(step[3]) <= REFINEMENT_STEP_TOLERANCE * radius:
            return centre, radius, True, iteration
    return centre, radius, False, MAX_REFINEMENT_ITERATIONS


def fit_sphere(points: Sequence[Point]) -> SphereFit:
    """Least-squares sphere, with its residual and its shear indicator.

    A sphere is the target that makes a principal-point error visible, which is
    why this is the fit that matters here. A shear maps a sphere to an ellipsoid,
    and the resulting residual is *structured* - it lies along ``dx * dz`` - where
    sensor noise is not. :attr:`SphereFit.shear_correlation` measures how much of
    the residual lies along that direction and
    :attr:`SphereFit.shear_coefficient` reads back the coefficient that would
    explain it.

    Two passes of arithmetic: an algebraic fit for a starting point, then
    Gauss-Newton on the true geometric residual, capped at
    :data:`MAX_REFINEMENT_ITERATIONS`. Cost is O(n) per iteration with no
    pairwise work anywhere.

    The residual, the centre and the radius are all on whatever scale the input
    points are on. Nothing here names a unit for any of them.
    """
    _ensure_point_set(points, MIN_SPHERE_POINTS, "sphere fit")
    fitted_at = time.time()

    origin = _mean(points)
    centred = [subtract(point, origin) for point in points]
    centre, radius = _algebraic_sphere(centred)
    centre, radius, converged, iterations = _refine_sphere(centred, centre, radius)

    offsets = [subtract(point, centre) for point in centred]
    distances = [norm(offset) for offset in offsets]
    residuals = [distance - radius for distance in distances]
    count = len(points)
    residual_rms = math.sqrt(math.fsum(r * r for r in residuals) / count)
    residual_max = max(abs(r) for r in residuals)

    unit_directions: list[Point] = [
        (o[0] / d, o[1] / d, o[2] / d) for o, d in zip(offsets, distances, strict=True)
    ]

    # The shear model, first order in the coefficient: a sphere sheared by
    # x += k * z has a best-fit-sphere residual of about k * dx * dz / radius.
    # Projecting the residual onto that shape is what separates a systematic
    # fault from noise - but the raw shape is the wrong thing to project onto.
    #
    # At the least-squares optimum the residual is orthogonal to this fit's own
    # Jacobian columns, (ux, uy, uz, 1): those four directions are exactly what
    # the centre and the radius were moved to absorb. Whatever part of the shear
    # shape lies in that span has already been fitted away and no residual can
    # ever lie along it. Over a full sphere that part is negligible, but over a
    # cap - which is all a single view of a ball can ever produce - dz is nearly
    # constant, so dx * dz is nearly a multiple of dx and is almost entirely
    # absorbable. Projecting onto the raw shape then divides by a norm made
    # mostly of unreachable directions and drives the correlation and the
    # coefficient toward zero, which is a silent false negative on precisely the
    # geometry this module exists to protect: a 60-degree cap carrying the full
    # factory shear reads a correlation of 0.17 and passes.
    #
    # So the shape is residualised against those four columns first, making the
    # reading a partial correlation and the coefficient a partial regression.
    # The columns are consumed as fresh generators rather than materialised,
    # because at MAX_FIT_POINTS four extra lists would cost more than the fit.
    def _column(which: int) -> Iterator[float]:
        if which == 3:
            return (1.0 for _ in unit_directions)
        return (direction[which] for direction in unit_directions)

    shape = [offset[0] * offset[2] / radius for offset in offsets]
    for which in range(4):
        column_energy = math.fsum(value * value for value in _column(which))
        if column_energy <= 0.0:
            continue
        cross_term = math.fsum(w * value for w, value in zip(shape, _column(which), strict=True))
        projection = cross_term / column_energy
        if projection == 0.0:
            continue
        for index, value in enumerate(_column(which)):
            shape[index] -= projection * value

    shape_energy = math.fsum(w * w for w in shape)
    residual_energy = math.fsum(r * r for r in residuals)
    cross = math.fsum(r * w for r, w in zip(residuals, shape, strict=True))
    if shape_energy <= 0.0 or residual_energy <= 0.0:
        correlation, coefficient = 0.0, 0.0
    else:
        correlation = cross / math.sqrt(shape_energy * residual_energy)
        coefficient = cross / shape_energy
    leverage = math.sqrt(shape_energy / count) / radius

    # How much of the sphere the points actually cover. A near-zero mean
    # direction means they surround the centre, which is full coverage; anything
    # else is a cap, and its half-angle is what conditions the fit.
    mean_direction = _mean(unit_directions)
    mean_length = norm(mean_direction)
    if mean_length <= 1e-9:
        cap_half_angle = math.pi
    else:
        axis = (
            mean_direction[0] / mean_length,
            mean_direction[1] / mean_length,
            mean_direction[2] / mean_length,
        )
        cap_half_angle = max(
            math.acos(max(-1.0, min(1.0, dot(direction, axis)))) for direction in unit_directions
        )

    z_min, z_max = _z_span(points)
    return SphereFit(
        centre=(centre[0] + origin[0], centre[1] + origin[1], centre[2] + origin[2]),
        radius=radius,
        point_count=count,
        residual_rms=residual_rms,
        residual_max=residual_max,
        shear_correlation=correlation,
        shear_coefficient=coefficient,
        shear_leverage=leverage,
        cap_half_angle=cap_half_angle,
        converged=converged,
        iterations=iterations,
        z_min=z_min,
        z_max=z_max,
        fitted_at=fitted_at,
    )


def fit_plane(points: Sequence[Point]) -> PlaneFit:
    """Least-squares plane, which validates nothing and says so.

    The fit is real and its residual is a real flatness observation. What it
    cannot do is decide anything about this scanner's scale: it produces no
    length, and a shear maps planes to planes, so its residual is insensitive to
    the one calibration error this unit is most likely to have. Both facts travel
    on :attr:`PlaneFit.inconclusive_reasons` and :func:`validate` acts on them.

    Total least squares - the eigenvector of least covariance - rather than a
    regression of one coordinate on the other two, because the latter measures
    distance along an axis rather than distance to the plane and its answer
    depends on which axis the caller happened to pick.
    """
    _ensure_point_set(points, MIN_PLANE_POINTS, "plane fit")
    fitted_at = time.time()

    origin = _mean(points)
    centred = [subtract(point, origin) for point in points]
    count = len(points)
    covariance = [
        [math.fsum(p[i] * p[j] for p in centred) / count for j in range(3)] for i in range(3)
    ]
    values, directions = _symmetric_eigen(covariance)
    if values[2] <= 0.0 or values[1] <= values[2] * PIVOT_FLOOR_FRACTION:
        raise DegenerateFit(
            f"plane fit: the point set's covariance eigenvalues are {values!r}. Two of them are "
            "indistinguishable from zero, so the points are collinear or coincident and span no "
            "plane; the normal returned would be a direction picked out of rounding error"
        )

    normal = _oriented(directions[0])
    residuals = [dot(normal, point) for point in centred]
    residual_rms = math.sqrt(math.fsum(r * r for r in residuals) / count)
    residual_max = max(abs(r) for r in residuals)
    z_min, z_max = _z_span(points)
    return PlaneFit(
        normal=normal,
        offset=dot(normal, origin),
        centroid=origin,
        point_count=count,
        residual_rms=residual_rms,
        residual_max=residual_max,
        eigenvalues=(values[0], values[1], values[2]),
        z_min=z_min,
        z_max=z_max,
        fitted_at=fitted_at,
    )


def plane_separation(points_a: Sequence[Point], points_b: Sequence[Point]) -> SeparationFit:
    """The distance between two parallel faces - a length a caliper can judge.

    Each face is fitted independently, the two normals are brought into agreement
    of sign, and the separation is the projection of the two centroids' offset
    onto their averaged common normal. The angle the two normals still disagree
    by is reported and is never corrected away: a wedge between the faces is a
    fact about the target or about the sensor, and averaging it into silence
    would hide exactly the case where the "separation" is not one number.

    This is the flat-target experiment that produces a dimension, but its
    sensitivity to a principal-point error depends entirely on how the target was
    turned. See :class:`SeparationFit` for the geometry;
    :attr:`SeparationFit.shear_sensitivity` reports which case a given pair is
    in, and the verdict carries the consequence in its own
    ``does_not_establish``.
    """
    plane_a = fit_plane(points_a)
    plane_b = fit_plane(points_b)
    fitted_at = time.time()

    normal_b = plane_b.normal
    if dot(plane_a.normal, normal_b) < 0.0:
        normal_b = (-normal_b[0], -normal_b[1], -normal_b[2])
    disagreement = math.acos(max(-1.0, min(1.0, dot(plane_a.normal, normal_b))))

    common = (
        plane_a.normal[0] + normal_b[0],
        plane_a.normal[1] + normal_b[1],
        plane_a.normal[2] + normal_b[2],
    )
    length = norm(common)
    if length <= 1e-9:
        raise DegenerateFit(
            "plane separation: the two fitted normals are opposed even after sign alignment, so "
            "the pair has no common direction and no single separation"
        )
    common = (common[0] / length, common[1] / length, common[2] / length)
    separation = abs(dot(common, subtract(plane_a.centroid, plane_b.centroid)))

    return SeparationFit(
        separation=separation,
        normal=common,
        plane_a=plane_a,
        plane_b=plane_b,
        normal_disagreement=disagreement,
        z_min=min(plane_a.z_min, plane_b.z_min),
        z_max=max(plane_a.z_max, plane_b.z_max),
        fitted_at=fitted_at,
    )


# --- the verdict --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """One comparison, its verdict, and the narrow thing that verdict is worth.

    :attr:`passed` is deliberately three-valued. ``None`` means the experiment
    could not decide, which happens whenever the uncertainty band on the
    disagreement straddles the declared tolerance - including the common case
    where the combined uncertainty is itself as large as the tolerance, so that
    neither a pass nor a fail was ever reachable. That is not a soft failure and
    it is not a near miss; it is the honest report that this run settled nothing.

    :attr:`establishes` is empty unless the run decided something. :attr:`
    does_not_establish` is never empty, because a single dimension at a single
    distance never establishes linearity, distortion or the principal point,
    whatever the verdict.
    """

    verdict: Verdict
    quantity: FittedQuantity
    fitted_value: float | None
    caliper_value: float
    difference: float | None
    difference_low: float | None
    difference_high: float | None
    ratio: float | None
    tolerance_limit: float
    fit_uncertainty: float
    caliper_uncertainty: float
    combined_uncertainty: float
    residual_rms: float
    residual_max: float
    point_count: int
    z_min: float
    z_max: float
    measurement: CaliperMeasurement
    tolerance: Tolerance
    fit: dict[str, object]
    inconclusive_reasons: tuple[str, ...]
    establishes: tuple[str, ...]
    does_not_establish: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def passed(self) -> bool | None:
        """``True``, ``False``, or ``None`` for a run that could not decide."""
        if self.verdict is Verdict.INCONCLUSIVE:
            return None
        return self.verdict is Verdict.PASS

    @property
    def combined_uncertainty_method(self) -> str:
        return (
            "in quadrature: sqrt(fit_uncertainty^2 + caliper_uncertainty^2). The two are "
            "independent - one comes from the scanner's residual, the other from the "
            "instrument - so their variances add and their magnitudes do not. Neither term "
            "is divided by the square root of a point count anywhere in this module"
        )

    @property
    def decision_rule(self) -> str:
        return (
            "pass when |difference| + combined_uncertainty <= limit, fail when |difference| - "
            "combined_uncertainty > limit, inconclusive otherwise. A band that straddles the "
            "limit decides nothing, and a combined uncertainty as large as the limit makes a "
            "pass unreachable by construction"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "fitted-dimension-against-caliper-reading",
            "verdict": str(self.verdict),
            "passed": self.passed,
            "quantity": str(self.quantity),
            "comparison": {
                "fitted_value": self.fitted_value,
                "caliper_value": self.caliper_value,
                "difference": self.difference,
                "difference_band": [self.difference_low, self.difference_high],
                "ratio": self.ratio,
                "ratio_note": (
                    "fitted divided by measured. A ratio constant across distances but not 1 "
                    "means the model is right and the scale is off by that factor; a ratio "
                    "that varies with distance means the model itself is wrong. One distance "
                    "cannot tell those apart"
                ),
                "tolerance_limit": self.tolerance_limit,
                "decision_rule": self.decision_rule,
            },
            "uncertainty": {
                "fit": self.fit_uncertainty,
                "caliper": self.caliper_uncertainty,
                "combined": self.combined_uncertainty,
                "method": self.combined_uncertainty_method,
                "resolves_tolerance": self.combined_uncertainty < self.tolerance_limit,
            },
            "residual": {"rms": self.residual_rms, "max": self.residual_max},
            "points": {"count": self.point_count, "z_span": [self.z_min, self.z_max]},
            "measurement": self.measurement.as_dict(),
            "tolerance": self.tolerance.as_dict(),
            "fit": self.fit,
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "establishes": list(self.establishes),
            "does_not_establish": list(self.does_not_establish),
            "limitations": list(self.limitations),
        }


#: What no single comparison establishes, whatever it decided. Prepended to the
#: fit's own list rather than replaced by it, so no fit type can produce a
#: verdict that omits the standing gaps.
UNIVERSAL_GAPS: Final[tuple[str, ...]] = (
    "scale at the distances this point set spans, and nothing outside them. A mapping that is "
    "correct here and wrong elsewhere agrees with this result exactly",
    "nothing about linearity of the raw-to-distance mapping: that needs the same target at "
    "several distances and a ratio compared across them, not one comparison at one distance",
    "nothing about lens distortion, which is not modelled or inverted anywhere upstream and is "
    "therefore inside this residual rather than separated from it",
    "nothing about the principal point unless the fit's own shear indicator addressed it, and "
    "nothing at all about it from a flat target or from faces square-on to the camera",
    "nothing about any other unit of this device, whose factory record differs, and nothing "
    "about this one at another temperature or after another power cycle",
    "no physical unit for anything. A pass is evidence that the caliper reading and the fitted "
    "value are on the same scale to the declared tolerance; the scale itself remains unnamed",
)


def _deduplicated(items: Sequence[str]) -> tuple[str, ...]:
    """Order-preserving unique, so a merged list reads once per distinct statement."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


def validate(
    fit: DimensionalFit, measurement: CaliperMeasurement, tolerance: Tolerance
) -> ValidationResult:
    """Compare a fitted dimension against a caliper reading, under a predeclared bound.

    Refuses, before computing anything, a tolerance declared for a different
    quantity than the fit produced and a tolerance stamped later than the fit.
    Everything else is recorded rather than raised: an experiment that cannot
    decide produces a :attr:`Verdict.INCONCLUSIVE` result a reader can hold,
    which is more use than an exception a caller can swallow.

    The arithmetic presumes only that the operator intended the two numbers to be
    on the same scale. It attaches no physical unit to either, and the result
    says so in its own ``does_not_establish``.
    """
    if tolerance.quantity is not fit.quantity:
        raise QuantityMismatch(
            f"the tolerance was declared for {str(tolerance.quantity)!r} and this fit produced "
            f"{str(fit.quantity)!r}. A bound is tied to one named quantity at declaration time "
            "so that it cannot be moved onto whichever fit happens to satisfy it"
        )
    if tolerance.declared_at >= fit.fitted_at:
        raise ToleranceDeclaredLate(
            f"the tolerance was declared at {tolerance.declared_at!r} and the fit was computed "
            f"at {fit.fitted_at!r}: the bound is not earlier than the result it judges. A limit "
            "chosen once the numbers are in describes them and tests nothing, so no verdict is "
            "computed from it"
        )

    fitted = fit.fitted_value
    caliper = measurement.value
    combined = math.hypot(fit.fit_uncertainty, measurement.uncertainty)
    reasons = list(fit.inconclusive_reasons)

    difference: float | None = None
    low: float | None = None
    high: float | None = None
    ratio: float | None = None
    establishes: tuple[str, ...] = ()
    if fitted is None:
        # A fit that produced no length - a plane - has nothing to compare, and
        # the reasons it carries already say so. Nothing is established here.
        verdict = Verdict.INCONCLUSIVE
    else:
        difference = fitted - caliper
        ratio = fitted / caliper
        magnitude = abs(difference)
        low = max(magnitude - combined, 0.0)
        high = magnitude + combined
        if combined >= tolerance.limit:
            reasons.append(
                f"the combined uncertainty ({combined!r}) is at least the declared tolerance "
                f"({tolerance.limit!r}), so a pass was unreachable before the target was ever "
                "measured. The fit's residual, the instrument, or both, must improve before "
                "this bound can be tested"
            )
        if reasons:
            verdict = Verdict.INCONCLUSIVE
        elif high <= tolerance.limit:
            verdict = Verdict.PASS
            establishes = (
                f"the fitted {fit.quantity!s} agrees with the caliper reading of "
                f"{measurement.label!r} to within the tolerance of {tolerance.limit!r} declared "
                f"by {tolerance.declared_by}, with the whole combined-uncertainty band inside "
                f"it, over points spanning z from {fit.z_min!r} to {fit.z_max!r}",
            )
        elif low > tolerance.limit:
            verdict = Verdict.FAIL
            establishes = (
                f"the fitted {fit.quantity!s} disagrees with the caliper reading of "
                f"{measurement.label!r} by {magnitude!r}, more than the declared "
                f"{tolerance.limit!r} even after the combined uncertainty is taken off. The "
                "disagreement is real; which of the scale, the model, the principal point or "
                "the target it belongs to is not decided here",
            )
        else:
            verdict = Verdict.INCONCLUSIVE
            reasons.append(
                f"the disagreement band [{low!r}, {high!r}] straddles the {tolerance.limit!r} "
                "limit: part of it is inside the tolerance and part is outside, so this run is "
                "consistent with both a pass and a fail and asserts neither"
            )

    return ValidationResult(
        verdict=verdict,
        quantity=fit.quantity,
        fitted_value=fitted,
        caliper_value=caliper,
        difference=difference,
        difference_low=low,
        difference_high=high,
        ratio=ratio,
        tolerance_limit=tolerance.limit,
        fit_uncertainty=fit.fit_uncertainty,
        caliper_uncertainty=measurement.uncertainty,
        combined_uncertainty=combined,
        residual_rms=fit.residual_rms,
        residual_max=fit.residual_max,
        point_count=fit.point_count,
        z_min=fit.z_min,
        z_max=fit.z_max,
        measurement=measurement,
        tolerance=tolerance,
        fit=fit.as_dict(),
        inconclusive_reasons=tuple(reasons),
        establishes=establishes,
        does_not_establish=_deduplicated([*UNIVERSAL_GAPS, *fit.does_not_establish]),
        limitations=_deduplicated([*STATIC_LIMITATIONS, *fit.limitations]),
    )
