"""Reconstruction quality, measured offline against geometry that is known.

Why this module exists
----------------------
Everything else this repository measures about a reconstruction is either
internal to an optimiser or judged by eye.
:func:`scanner3d.pipeline.reconstruct_sweep` reports held-out closure error,
which is a real number and a real check - but it is a check on SELF
CONSISTENCY: it says the poses agree with revisits that were withheld from the
fit. A reconstruction can be perfectly self-consistent and still be the wrong
shape, at the wrong scale, with a third of the object missing. On 2026-08-28 a
relocalization bridge was accepted at residual 0.515 with a 1.50x refined
margin - every internal number healthy - and the placement it chose was wrong;
a genuinely different candidate covered more of the model (0.598 against
0.483). That is the failure this module exists to catch, and it is not the
failure closure error can see.

So every measurement here compares a reconstruction against truth the CALLER
supplies: exact poses, a reference surface, a planar region that is known to
be planar, named dimensions someone measured with an instrument. Truth is
never inferred from the reconstruction being judged, and never inferred from a
capture manifest's prose.

Separate facts stay separate
----------------------------
There is deliberately no single quality score. Four different things go wrong
in four different ways and one number cannot say which happened:

- **pose error** (:func:`pose_sequence_error`) - the camera is in the wrong
  place. Rotation and translation are reported separately because a degree
  and a millimetre are not interchangeable and the fixes differ.
- **surface agreement** (:func:`cloud_agreement`) - the geometry is displaced,
  lumpy, or missing. Reported in BOTH directions, because one direction cannot
  see both failures: observed-to-reference is blind to a hole (every point
  that exists can sit perfectly on the reference while half the object is
  absent), and reference-to-observed is blind to spurious geometry. Because it
  compares against a reference the CALLER supplies rather than against the
  reconstruction itself, a dense reference in the same frame also sees a
  uniform scale - a 2% larger model is 2% further from truth everywhere.
- **local flatness** (:func:`plane_roughness`) - a surface that is known to be
  flat is not coming out flat. This is sensitive to noise and fusion artefacts
  that a whole-cloud distance averages into nothing, and insensitive to scale.
- **dimension and landmark truth** (:func:`dimension_errors`,
  :func:`landmark_offsets`) - the object is the right shape at the wrong size,
  or a known feature is not where it belongs. A uniform scale error leaves
  held-out closure, plane roughness and every other self-consistency check
  perfect, because all of them are computed inside the reconstruction's own
  coordinate system. Scale is only visible to truth from outside it: a dense
  reference surface if one exists, and otherwise these named lengths, which
  are the sparse physical check a caliper can actually supply.

Plus one audit that is not a geometric measurement at all:
:func:`audit_bridge_record` re-applies the CURRENT relocalization policy to a
bridge record that was written earlier, so a decision that was accepted under
an older policy is re-judged rather than merely remembered.

Gates are the caller's, not this module's
-----------------------------------------
This module owns no notion of "good". Every threshold is supplied by the
caller through :func:`at_most`, :func:`at_least` or :func:`must_be`, in the
same unit as the measurement it judges, with a sentence saying what a reader
should DO when it fails. A named limit in a test file next to the case it
governs can be argued about and changed by someone who can see the argument; a
constant buried here would be a hidden standard nobody voted on.

:class:`CaseReport` and :class:`SuiteReport` evaluate every gate and list every
failure. They never stop at the first one, because the second failure is
usually what says which of the four things above actually happened.

CPU and correctness first
-------------------------
Nothing here needs a GPU, a display server, a scanner or a network.
:func:`cloud_agreement` searches through :mod:`scanner3d.compute`'s backend
seam and records which backend resolved, so the accelerated appliance path can
be proved to agree with the pure-Python reference to a stated tolerance
(:data:`BACKEND_AGREEMENT`) rather than assumed to. "python" remains the
correctness oracle.

Timing is deliberately absent from every report. A report that carried a
duration would not be reproducible, and a duration is not a correctness
statement in any case: it is a property of the machine. Measure it beside the
harness, never inside a gate.

What this module does NOT establish
-----------------------------------
It does not know what a coordinate means. Lengths are reported in
:data:`COORDINATE_UNIT` unless the caller names a real unit, because this
repository has not established the physical unit of a depth sample. Passing
every gate says the reconstruction matched the truth that was supplied, to the
limits that were supplied - not that the truth was right, nor that the parts
of the object nobody measured are correct.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, assert_never

from scanner3d.compute import PYTHON_BACKEND, ComputeBackend, select_backend
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid, distance, dot
from scanner3d.multiview import AMBIGUITY_MARGIN
from scanner3d.pipeline import NOISE_FLOOR
from scanner3d.relocalize import (
    MIN_FRAME_OVERLAP,
    MIN_MATCHED_FRACTION,
    MIN_OVERLAP,
    MIN_USED_FRACTION,
    RESIDUAL_MULTIPLE,
    RUNNER_UP_MARGIN,
    thresholds,
)
from scanner3d.validation import fit_plane

SCHEMA_VERSION: Final = 1

#: The unit every length reported here carries unless the caller names another.
#:
#: Copied verbatim from the phrase :mod:`scanner3d.relocalize` already writes
#: into every bridge record, so a reader meeting the two side by side does not
#: have to work out whether they mean the same thing. They do: this repository
#: has established the depth scale is millimetres only for one measured target
#: at one standoff (see ``docs/metric-validation.md``), which is not the same
#: as establishing it for an arbitrary reconstruction's coordinates.
COORDINATE_UNIT: Final = "input coordinate units, whose physical meaning is not established"

#: Rotation errors are reported in degrees, never radians. A reader judging a
#: scan can picture a tenth of a degree; nobody pictures 1.7e-3 radians.
DEGREES_UNIT: Final = "degrees"

#: The unit a :attr:`Comparison.IS` gate carries. Named rather than left empty
#: so every gate answers "in what unit" with something.
BOOLEAN_UNIT: Final = "boolean"

#: The unit a fraction-of-a-whole gate carries.
FRACTION_UNIT: Final = "fraction of 1"

#: Most points either side of a cloud comparison.
#:
#: A reconstruction can hold millions of points; this harness is not the tool
#: for judging one at that size, and a caller who hands it one should be told
#: so before the allocation rather than after the machine starts swapping.
#: Downsample deliberately and record that you did.
MAX_POINTS: Final = 100_000

#: Most query-target pairs a single cloud comparison may cost.
#:
#: The search gate :func:`cloud_agreement` uses spans the whole scene on
#: purpose - see that function - which collapses the "python" backend's spatial
#: index to a full scan, so its cost really is the product of the two point
#: counts. Measured at this ceiling on the machine this was written on (a
#: 20-core i9-10900K, one core): 3.9 seconds for 5000 points against 5000, both
#: directions, on the pure-Python reference backend - linear in the product, at
#: about 6.5 million pairs a second. Slow enough that reaching it should be a
#: deliberate decision, fast enough that it is not a wall. The fast regression
#: tier runs clouds three orders of magnitude below it.
MAX_NEAREST_WORK: Final = 25_000_000

#: Most poses in one sequence comparison.
MAX_POSES: Final = 20_000

#: Most named dimensions or landmarks in one comparison.
MAX_NAMED: Final = 1_000

#: How closely an accelerated backend must reproduce the "python" reference's
#: cloud distances, AS A FRACTION OF THE COMPARED SCENE'S OWN SIZE.
#:
#: Relative to the scene rather than absolute, because that is the unit the
#: disagreement actually lives in. The "numpy" backend computes
#: ``|q-t|^2 = |q|^2 - 2 q.t + |t|^2`` for speed - see
#: :meth:`scanner3d.compute.ComputeBackend.nearest_neighbors` - and that
#: identity loses about half the mantissa to cancellation where the true
#: distance is near zero, which is exactly where a good reconstruction sits.
#: The error it leaves is proportional to the coordinates' magnitude, so an
#: absolute tolerance would silently mean something different on a 40-unit
#: object and a 400-unit one.
#:
#: Measured on the parity scene in ``tests/test_quality.py``: "scipy" agrees
#: exactly (0.0), "numpy" disagrees by at most 7.5e-9 of the scene's extent -
#: 4.8e-07 in absolute terms on a 64-unit scene, and only on the median, where
#: the true distance is zero. 1e-6 sits two orders above that and four orders
#: below the tightest displacement gate this harness uses, so a violation
#: means a backend changed the answer rather than rounded it.
BACKEND_AGREEMENT: Final = 1e-6

#: Slack added to a computed search gate so the farthest possible pair is still
#: strictly inside it. :meth:`ComputeBackend.nearest_neighbors` admits a
#: neighbour strictly inside the gate, and the scene diagonal is exactly
#: attainable, so a gate equal to it could drop the one pair that matters.
_GATE_MARGIN: Final = 1e-9

#: The gate used when every point in both clouds is coincident, where the
#: diagonal is zero and a zero gate would admit nothing.
_DEGENERATE_GATE: Final = 1e-9


class QualityNotAllowed(ScannerError):
    """Base for every refusal this module makes."""


class MeasurementInputUnusable(QualityNotAllowed):
    """The measurement was not attempted: its input could not support one.

    Empty, non-finite, mismatched in length, or larger than the bounds this
    module states. Raised BEFORE any allocation proportional to the input, so
    a caller who passes a million points against a million gets a sentence
    rather than a stalled machine.
    """


class GateUnusable(QualityNotAllowed):
    """A gate could not be evaluated as stated.

    A non-finite measurement or limit, a comparison against a limit of the
    wrong kind, a comparison this module does not define, or a gate with no
    name, unit or context. A gate whose failure a reader cannot act on is not
    a gate, so the context is required.
    """


class BridgeRecordUnusable(QualityNotAllowed):
    """A recorded bridge could not be re-judged: the record is not one.

    Distinct from "this bridge would be refused". A record missing the fields
    current policy reads has not been judged at all, and reporting it as a
    refusal would let a corrupt file masquerade as a caught regression.
    """


# --- input validation --------------------------------------------------------------


def _ensure_text(
    value: str,
    what: str,
    *,
    error: type[QualityNotAllowed] = MeasurementInputUnusable,
) -> str:
    """Refuse an unnamed field, as whichever refusal the caller is making.

    A measurement with no name and a gate with no context are the same defect -
    a report line a reader cannot act on - but they are raised by different
    callers, so the exception type travels with the call rather than being
    fixed here.
    """
    if not value or not value.strip():
        raise error(
            f"{what} must be a non-empty string; a report field nobody named is not evidence"
        )
    return value


def _ensure_finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise MeasurementInputUnusable(
            f"{what} is {value!r}, which is not finite. A non-finite measurement cannot be "
            "compared to a limit and cannot be written to strict JSON, so it is refused here "
            "rather than carried into a report that claims to round-trip"
        )
    return value


def _ensure_points(points: Sequence[Point], *, what: str, minimum: int = 1) -> Sequence[Point]:
    """Refuse a point set before anything proportional to its size is built."""
    count = len(points)
    if count < minimum:
        raise MeasurementInputUnusable(
            f"{what} carries {count} point(s), under the {minimum} this measurement needs"
        )
    if count > MAX_POINTS:
        raise MeasurementInputUnusable(
            f"{what} carries {count} points, over the {MAX_POINTS} bound this module states. "
            "Downsample deliberately and record that you did; a harness that silently judges a "
            "sample it chose itself is not a harness"
        )
    for index, point in enumerate(points):
        if len(point) != 3:
            raise MeasurementInputUnusable(
                f"{what} point {index} has {len(point)} coordinate(s), not 3"
            )
        for axis, value in enumerate(point):
            if not math.isfinite(value):
                raise MeasurementInputUnusable(
                    f"{what} point {index} coordinate {axis} is {value!r}, which is not finite"
                )
    return points


def _ensure_pose(pose: Rigid, what: str) -> Rigid:
    """Check a transform's numbers even though :meth:`Rigid.of` would have.

    :class:`~scanner3d.geometry.Rigid` is validated by ``of``, but it is a
    plain dataclass and nothing stops a caller constructing one directly from
    values that were never checked. A harness that trusted that would report a
    NaN pose error as a passing gate.
    """
    for row_index, row in enumerate(pose.rotation):
        for column, value in enumerate(row):
            if not math.isfinite(value):
                raise MeasurementInputUnusable(
                    f"{what} rotation entry ({row_index}, {column}) is {value!r}, not finite"
                )
    for axis, value in enumerate(pose.translation):
        if not math.isfinite(value):
            raise MeasurementInputUnusable(
                f"{what} translation component {axis} is {value!r}, not finite"
            )
    return pose


# --- distributions -----------------------------------------------------------------


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """The nearest-rank percentile of an already sorted sequence.

    Nearest rank rather than linear interpolation, so every quantile this
    module reports is a value that was actually observed. An interpolated p95
    is a number no point in the cloud has, which makes "which point was that?"
    unanswerable exactly when a reader wants to go and look at it.
    """
    if not ordered:
        raise MeasurementInputUnusable("a percentile of no values is not defined")
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class Distribution:
    """A set of non-negative magnitudes, summarised four ways.

    Four rather than one because they fail differently: a median moves only
    when most of the surface is wrong, an RMS is pulled by the tail, a p95 is
    the tail without the single worst outlier deciding everything, and the
    worst value is the one a person looking at the model will actually see.
    A harness that reported only a mean would pass a reconstruction with one
    catastrophic lump.
    """

    count: int
    median: float
    rms: float
    p95: float
    worst: float
    unit: str

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "median": self.median,
            "rms": self.rms,
            "p95": self.p95,
            "worst": self.worst,
            "unit": self.unit,
        }


def _distribution(values: Sequence[float], *, unit: str, what: str) -> Distribution:
    if not values:
        raise MeasurementInputUnusable(f"{what} produced no values to summarise")
    for index, value in enumerate(values):
        _ensure_finite(value, f"{what} value {index}")
    ordered = sorted(values)
    return Distribution(
        count=len(ordered),
        median=_percentile(ordered, 0.5),
        rms=math.sqrt(math.fsum(value * value for value in ordered) / len(ordered)),
        p95=_percentile(ordered, 0.95),
        worst=ordered[-1],
        unit=unit,
    )


# --- gates -------------------------------------------------------------------------


class Comparison(enum.StrEnum):
    """How a gate compares a measurement to its limit."""

    AT_MOST = "<="
    AT_LEAST = ">="
    IS = "is"


def _format(value: float | bool) -> str:
    """One deterministic rendering of a measured value or a limit."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class Gate:
    """One named threshold, the value measured against it, and the verdict.

    Every field is here so a failure line in a CI log is self-contained: what
    was measured, which way the comparison runs, the limit it ran against, the
    unit both are in, and what a reader should do about it. A gate that
    reported only "failed" would send whoever reads it back to the source to
    find out what number it even looked at.
    """

    name: str
    measured: float | bool
    comparison: Comparison
    limit: float | bool
    unit: str
    context: str

    def __post_init__(self) -> None:
        """The one place a gate is validated, however it was built.

        :func:`at_most`, :func:`at_least` and :func:`must_be` are the sanctioned
        way to build one, but this is a public dataclass and nothing stops a
        caller constructing it directly. So the semantic checks live HERE and
        the factories are thin wrappers over this constructor: every gate that
        exists has been through this, and "every report round-trips through
        strict JSON" is a claim about the gate rather than a claim about how it
        happened to be made.
        """
        _ensure_text(self.name, "gate name", error=GateUnusable)
        # Declared as a Comparison, but a caller is not obliged to be type
        # checked; widening to object keeps the runtime check honest.
        comparison: object = self.comparison
        if not isinstance(comparison, Comparison):
            raise GateUnusable(
                f"gate {self.name!r} compares with {comparison!r}, which is not a "
                f"{Comparison.__name__}. A comparison this module does not define cannot be "
                "evaluated, and letting it fall through to an ordering would report PASS for a "
                "question that was never asked"
            )
        _ensure_text(self.unit, "gate unit", error=GateUnusable)
        _ensure_text(self.context, f"gate {self.name!r} context", error=GateUnusable)
        if comparison is Comparison.IS:
            if not isinstance(self.measured, bool) or not isinstance(self.limit, bool):
                raise GateUnusable(
                    f"gate {self.name!r} compares {self.measured!r} to {self.limit!r}, which are "
                    "not both booleans. Use at_most()/at_least() for a measurement with a "
                    "magnitude"
                )
            return
        if isinstance(self.measured, bool) or isinstance(self.limit, bool):
            raise GateUnusable(
                f"gate {self.name!r} compares {self.measured!r} to {self.limit!r} with "
                f"{comparison}, but one of them is a boolean. Use must_be() for a flag; an "
                "ordering comparison on true/false is a coincidence of Python's type system, "
                "not a measurement"
            )
        if not math.isfinite(self.measured):
            raise GateUnusable(
                f"gate {self.name!r} measured {self.measured!r}, which is not finite and could "
                "not be written to strict JSON"
            )
        if not math.isfinite(self.limit):
            raise GateUnusable(
                f"gate {self.name!r} has limit {self.limit!r}, which is not finite. An unbounded "
                "limit is a gate that cannot fail, and a gate that cannot fail is not evidence"
            )

    @property
    def passed(self) -> bool:
        """The verdict, from the comparison this gate actually carries.

        Every comparison is answered by its own branch and nothing falls
        through to an ordering. :meth:`__post_init__` already refuses a
        comparison this module does not define, so the final line is that
        invariant restated where it would otherwise be silently assumed.
        """
        if self.comparison is Comparison.IS:
            return bool(self.measured) is bool(self.limit)
        if self.comparison is Comparison.AT_MOST:
            return float(self.measured) <= float(self.limit)
        if self.comparison is Comparison.AT_LEAST:
            return float(self.measured) >= float(self.limit)
        assert_never(self.comparison)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "measured": self.measured,
            "comparison": str(self.comparison),
            "limit": self.limit,
            "unit": self.unit,
            "passed": self.passed,
            "context": self.context,
        }

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} {self.name}: measured {_format(self.measured)} "
            f"{self.comparison} {_format(self.limit)} ({self.unit}) - {self.context}"
        )


def _numeric_gate(
    name: str,
    measured: float,
    limit: float,
    *,
    comparison: Comparison,
    unit: str,
    context: str,
) -> Gate:
    """The one construction site for an ordered gate; :class:`Gate` validates it."""
    return Gate(
        name=name,
        measured=measured,
        comparison=comparison,
        limit=limit,
        unit=unit,
        context=context,
    )


def at_most(name: str, measured: float, limit: float, *, unit: str, context: str) -> Gate:
    """``measured <= limit``, in ``unit``, with the action a failure implies."""
    return _numeric_gate(
        name, measured, limit, comparison=Comparison.AT_MOST, unit=unit, context=context
    )


def at_least(name: str, measured: float, limit: float, *, unit: str, context: str) -> Gate:
    """``measured >= limit``, in ``unit``, with the action a failure implies."""
    return _numeric_gate(
        name, measured, limit, comparison=Comparison.AT_LEAST, unit=unit, context=context
    )


def must_be(name: str, measured: bool, expected: bool, *, context: str) -> Gate:
    """``measured is expected``, for a flag no ordering applies to.

    The bridge audit needs this: "current policy refuses this record" is the
    whole finding, and there is no threshold to put it under.
    """
    return Gate(
        name=name,
        measured=measured,
        comparison=Comparison.IS,
        limit=expected,
        unit=BOOLEAN_UNIT,
        context=context,
    )


# --- the measurement protocol -------------------------------------------------------


class QualityMeasurement(Protocol):
    """Anything a case can carry as evidence beside its gates.

    Every measurement names itself, so a case holding two planar regions or
    two clouds says which is which, and serialises to a JSON-shaped mapping
    that names its own kind.
    """

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def as_dict(self) -> dict[str, object]: ...


# --- pose sequence error ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoseFrameError:
    """One frame's placement error, after the common gauge is removed."""

    index: int
    rotation_degrees: float
    translation_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "rotation_degrees": self.rotation_degrees,
            "translation_error": self.translation_error,
        }


#: How the shared degree of freedom between two pose sequences is removed.
#:
#: Written into every report because it is the single thing a reader has to
#: trust about a pose comparison, and because the wrong choice here is
#: invisible in the numbers it produces.
GAUGE_STATEMENT: Final = (
    "Each sequence is expressed relative to its OWN first pose "
    "(pose[0].inverse().compose(pose[i])) before comparison. That removes exactly the one "
    "rigid transform the two sequences do not share - the arbitrary choice of where the "
    "world origin sits - and nothing else. No transform is fitted to the trajectories: a "
    "least-squares alignment over all frames would absorb part of the very error being "
    "measured and would report a drifting scan as a good one."
)


@dataclass(frozen=True, slots=True)
class PoseSequenceError:
    """Estimated poses against exact truth, gauge removed, frame by frame.

    ``frames`` excludes the gauge frame. Frame 0's error is exactly zero by
    construction - it is what the gauge was removed against - and including a
    guaranteed zero in the summaries would pull the median toward passing by
    an amount that grows as the sequence gets shorter.
    """

    name: str
    pose_count: int
    gauge_frame_index: int
    gauge: str
    frames: tuple[PoseFrameError, ...]
    rotation: Distribution
    translation: Distribution

    @property
    def kind(self) -> str:
        return "pose_sequence_error"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "pose_count": self.pose_count,
            "gauge_frame_index": self.gauge_frame_index,
            "gauge": self.gauge,
            "frames": [frame.as_dict() for frame in self.frames],
            "rotation": self.rotation.as_dict(),
            "translation": self.translation.as_dict(),
        }


def pose_sequence_error(
    estimated: Sequence[Rigid],
    truth: Sequence[Rigid],
    *,
    name: str,
    unit: str = COORDINATE_UNIT,
) -> PoseSequenceError:
    """Compare a pose sequence to exact truth without aligning the error away.

    ``estimated`` and ``truth`` are parallel: entry *i* of each places the same
    frame. See :data:`GAUGE_STATEMENT` for what is removed and why nothing
    else is.

    Rotation and translation are kept apart the whole way. The rotation error
    is the geodesic angle of the residual rotation
    (:attr:`scanner3d.geometry.Rigid.rotation_angle`, which stays accurate near
    zero where a well-placed frame lives); the translation error is the
    distance between where the two sequences put the frame's origin, which is
    a length and never has an angle folded into it.
    """
    _ensure_text(name, "measurement name")
    _ensure_text(unit, "translation unit")
    if len(estimated) != len(truth):
        raise MeasurementInputUnusable(
            f"{len(estimated)} estimated poses against {len(truth)} truth poses do not pair up; "
            "a pose comparison needs one truth per estimate"
        )
    if len(estimated) < 2:
        raise MeasurementInputUnusable(
            f"{len(estimated)} pose(s): after the gauge frame there would be nothing left to "
            "measure. At least 2 are needed"
        )
    if len(estimated) > MAX_POSES:
        raise MeasurementInputUnusable(
            f"{len(estimated)} poses is over the {MAX_POSES} bound this module states"
        )
    for index, pose in enumerate(estimated):
        _ensure_pose(pose, f"estimated pose {index}")
    for index, pose in enumerate(truth):
        _ensure_pose(pose, f"truth pose {index}")

    estimated_gauge = estimated[0].inverse()
    truth_gauge = truth[0].inverse()
    frames: list[PoseFrameError] = []
    for index in range(1, len(estimated)):
        relative_estimate = estimated_gauge.compose(estimated[index])
        relative_truth = truth_gauge.compose(truth[index])
        residual = relative_truth.inverse().compose(relative_estimate)
        frames.append(
            PoseFrameError(
                index=index,
                rotation_degrees=math.degrees(residual.rotation_angle),
                translation_error=distance(
                    relative_estimate.translation, relative_truth.translation
                ),
            )
        )
    return PoseSequenceError(
        name=name,
        pose_count=len(estimated),
        gauge_frame_index=0,
        gauge=GAUGE_STATEMENT,
        frames=tuple(frames),
        rotation=_distribution(
            [frame.rotation_degrees for frame in frames],
            unit=DEGREES_UNIT,
            what=f"{name} rotation error",
        ),
        translation=_distribution(
            [frame.translation_error for frame in frames],
            unit=unit,
            what=f"{name} translation error",
        ),
    )


# --- cloud agreement ----------------------------------------------------------------


def _scene_gate(first: Sequence[Point], second: Sequence[Point]) -> float:
    """A search gate no pair of points in either cloud can fall outside.

    :meth:`scanner3d.compute.ComputeBackend.nearest_neighbors` returns nothing
    for a query with no target inside its gate, and a distance summary that
    silently dropped the worst points would report a broken reconstruction as
    a good one. So the gate is the diagonal of the box containing BOTH clouds:
    every query is guaranteed a hit, and the distances summarised are the real
    nearest-neighbour distances rather than a censored sample of them.

    The cost of that choice is stated rather than hidden: a gate this size
    collapses the reference backend's voxel index to one cell, so the search is
    a full scan and :data:`MAX_NEAREST_WORK` bounds it.
    """
    low = [min(point[axis] for point in first) for axis in range(3)]
    high = [max(point[axis] for point in first) for axis in range(3)]
    for point in second:
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
    diagonal = distance((low[0], low[1], low[2]), (high[0], high[1], high[2]))
    if diagonal <= 0.0:
        return _DEGENERATE_GATE
    return diagonal * (1.0 + _GATE_MARGIN)


@dataclass(frozen=True, slots=True)
class DirectedDistance:
    """Nearest-neighbour distances one way round, and coverage at a tolerance.

    ``coverage_fraction`` is the fraction of queries whose nearest neighbour
    sits within ``coverage_tolerance``. It answers a different question from
    the distribution beside it: the distribution says how far wrong the
    surface is where it exists, and coverage says how much of it is there at
    all.
    """

    direction: str
    query_count: int
    target_count: int
    distances: Distribution
    coverage_fraction: float
    coverage_tolerance: float
    unit: str

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "query_count": self.query_count,
            "target_count": self.target_count,
            "distances": self.distances.as_dict(),
            "coverage_fraction": self.coverage_fraction,
            "coverage_tolerance": self.coverage_tolerance,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class CloudAgreement:
    """How two clouds agree, measured in both directions.

    Both, always, and never reduced to one. ``observed_to_reference`` is blind
    to a missing half - every point that exists can sit exactly on the
    reference - and ``reference_to_observed`` is blind to geometry that should
    not be there at all. Reporting either alone passes a reconstruction that
    the other one fails.
    """

    name: str
    observed_to_reference: DirectedDistance
    reference_to_observed: DirectedDistance
    search_gate: float
    resolved_backend: str
    unit: str

    @property
    def kind(self) -> str:
        return "cloud_agreement"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "observed_to_reference": self.observed_to_reference.as_dict(),
            "reference_to_observed": self.reference_to_observed.as_dict(),
            "search_gate": self.search_gate,
            "resolved_backend": self.resolved_backend,
            "unit": self.unit,
        }


def _directed(
    queries: Sequence[Point],
    targets: Sequence[Point],
    *,
    direction: str,
    gate: float,
    coverage_tolerance: float,
    unit: str,
    backend: ComputeBackend,
) -> DirectedDistance:
    hits = backend.nearest_neighbors(queries, targets, gate=gate)
    values: list[float] = []
    for index, hit in enumerate(hits):
        if hit is None:  # pragma: no cover - the gate spans the whole scene
            raise AssertionError(
                f"unreachable: query {index} of {direction} found no neighbour inside a gate "
                f"({gate!r}) that spans both clouds"
            )
        values.append(hit[0])
    within = sum(1 for value in values if value <= coverage_tolerance)
    return DirectedDistance(
        direction=direction,
        query_count=len(queries),
        target_count=len(targets),
        distances=_distribution(values, unit=unit, what=f"{direction} distance"),
        coverage_fraction=within / len(values),
        coverage_tolerance=coverage_tolerance,
        unit=unit,
    )


def cloud_agreement(
    observed: Sequence[Point],
    reference: Sequence[Point],
    *,
    name: str,
    coverage_tolerance: float,
    unit: str = COORDINATE_UNIT,
    backend: str = PYTHON_BACKEND,
) -> CloudAgreement:
    """Symmetric nearest-neighbour agreement between a reconstruction and truth.

    ``observed`` is what the pipeline produced; ``reference`` is the surface it
    should have produced, in the same frame - this measures agreement, not
    registration, and fits nothing.

    ``coverage_tolerance`` is the caller's statement of how close a point has
    to be to count as present, in ``unit``. It is required rather than
    defaulted: what counts as covered depends on the scanner's noise floor and
    the question being asked, and a default here would become an unexamined
    standard.

    ``backend`` selects through :mod:`scanner3d.compute`. The default is the
    "python" reference, so the correctness tier does not depend on numpy or
    scipy being installed; the resolved name travels in the report so a caller
    who chose an accelerator can prove it agreed.
    """
    _ensure_text(name, "measurement name")
    _ensure_text(unit, "distance unit")
    _ensure_points(observed, what=f"{name} observed cloud")
    _ensure_points(reference, what=f"{name} reference cloud")
    _ensure_finite(coverage_tolerance, f"{name} coverage tolerance")
    if coverage_tolerance <= 0.0:
        raise MeasurementInputUnusable(
            f"{name} coverage tolerance {coverage_tolerance!r} must be positive; a tolerance of "
            "zero admits only points that coincide exactly, which no measured surface does"
        )
    work = len(observed) * len(reference)
    if work > MAX_NEAREST_WORK:
        raise MeasurementInputUnusable(
            f"{name} would cost {work} query-target pairs, over the {MAX_NEAREST_WORK} bound "
            f"this module states ({len(observed)} observed against {len(reference)} reference). "
            "Downsample deliberately"
        )

    # Resolved once and reused for both directions: selection is what reports
    # which backend actually ran, and asking twice could in principle report
    # one name while the second direction ran on another.
    selection = select_backend(backend)
    gate = _scene_gate(observed, reference)
    return CloudAgreement(
        name=name,
        observed_to_reference=_directed(
            observed,
            reference,
            direction="observed_to_reference",
            gate=gate,
            coverage_tolerance=coverage_tolerance,
            unit=unit,
            backend=selection.backend,
        ),
        reference_to_observed=_directed(
            reference,
            observed,
            direction="reference_to_observed",
            gate=gate,
            coverage_tolerance=coverage_tolerance,
            unit=unit,
            backend=selection.backend,
        ),
        search_gate=gate,
        resolved_backend=selection.resolved,
        unit=unit,
    )


# --- plane roughness ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlaneRoughness:
    """The residual distribution of a region the caller says is flat.

    The plane is fitted by :func:`scanner3d.validation.fit_plane` - total
    least squares, so the residual is distance TO the plane rather than along
    somebody's chosen axis - and this reports the spread of those residuals
    rather than one number, because a region that is flat except for one bump
    and a region that is uniformly noisy have the same RMS and different
    causes.

    A flat fit establishes nothing about scale: a shear maps planes to planes.
    That limitation belongs to the measurement and travels with it.
    """

    name: str
    region: str
    point_count: int
    normal: Point
    offset: float
    residuals: Distribution
    signed_min: float
    signed_max: float
    unit: str
    limitation: str

    @property
    def kind(self) -> str:
        return "plane_roughness"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "region": self.region,
            "point_count": self.point_count,
            "normal": list(self.normal),
            "offset": self.offset,
            "residuals": self.residuals.as_dict(),
            "signed_min": self.signed_min,
            "signed_max": self.signed_max,
            "unit": self.unit,
            "limitation": self.limitation,
        }


#: What a flatness residual cannot say, carried on every roughness measurement.
PLANE_LIMITATION: Final = (
    "A plane fit produces no length and a shear maps planes to planes, so this residual is "
    "insensitive to a scale or shear error in the reconstruction it measures. It says the "
    "selected region is flat, not that it is the right size or in the right place."
)


def plane_roughness(
    points: Sequence[Point],
    *,
    name: str,
    region: str,
    unit: str = COORDINATE_UNIT,
) -> PlaneRoughness:
    """Fit the caller's planar region and report how far its points miss it.

    ``region`` is the caller's own statement of WHICH surface this is - "the
    truck bed floor", "the fixture's reference face". The selection is the
    caller's because only they know which part of the object is genuinely
    planar; this module will happily fit a plane through a curved surface and
    report an honest residual for a meaningless question.

    A region that spans no plane at all raises
    :class:`scanner3d.validation.DegenerateFit` from the fit itself, rather
    than being reported as very rough.
    """
    _ensure_text(name, "measurement name")
    _ensure_text(region, "region description")
    _ensure_text(unit, "residual unit")
    _ensure_points(points, what=f"{name} plane region", minimum=3)

    fit = fit_plane(points)
    signed = [dot(fit.normal, point) - fit.offset for point in points]
    return PlaneRoughness(
        name=name,
        region=region,
        point_count=fit.point_count,
        normal=fit.normal,
        offset=fit.offset,
        residuals=_distribution(
            [abs(value) for value in signed], unit=unit, what=f"{name} plane residual"
        ),
        signed_min=min(signed),
        signed_max=max(signed),
        unit=unit,
        limitation=PLANE_LIMITATION,
    )


# --- named dimensions and landmarks -------------------------------------------------


def _ensure_named(
    measured: Mapping[str, object], truth: Mapping[str, object], *, what: str
) -> tuple[str, ...]:
    """Both mappings name exactly the same things, and not too many of them."""
    if not truth:
        raise MeasurementInputUnusable(
            f"{what} was given no truth. A measurement with nothing to compare against is not "
            "an error of zero, and reporting it as one is how a harness passes everything"
        )
    if len(truth) > MAX_NAMED:
        raise MeasurementInputUnusable(
            f"{what} names {len(truth)} entries, over the {MAX_NAMED} bound this module states"
        )
    missing = sorted(set(truth) - set(measured))
    extra = sorted(set(measured) - set(truth))
    if missing or extra:
        raise MeasurementInputUnusable(
            f"{what}: measured and truth name different things - "
            f"missing from measured {missing}, absent from truth {extra}. A silent intersection "
            "would let a reconstruction pass by simply not reporting the feature that is wrong"
        )
    return tuple(sorted(truth))


@dataclass(frozen=True, slots=True)
class DimensionError:
    """One named length, measured against a length someone else established."""

    name: str
    measured: float
    truth: float
    error: float
    absolute_error: float
    relative_error: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "measured": self.measured,
            "truth": self.truth,
            "error": self.error,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True, slots=True)
class DimensionErrors:
    """Named dimensions against explicitly supplied truth.

    The practical way to see a uniform scale error. Held-out closure, plane
    roughness and every other self-consistency check are computed inside the
    reconstruction's own coordinate system, and a reconstruction that is 2%
    too big is perfectly self-consistent in it, so none of them can. Scale is
    visible only to truth from outside that system. :func:`cloud_agreement`
    qualifies when the caller genuinely has a dense reference surface in the
    same frame - it would report a 2% model as 2% away everywhere - but such a
    surface rarely exists for a physical object, whereas a few lengths off a
    caliper usually do. That is what this measurement is for.
    """

    name: str
    entries: tuple[DimensionError, ...]
    absolute: Distribution
    relative: Distribution | None
    unit: str
    truth_source: str

    @property
    def kind(self) -> str:
        return "dimension_errors"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "entries": [entry.as_dict() for entry in self.entries],
            "absolute": self.absolute.as_dict(),
            "relative": self.relative.as_dict() if self.relative is not None else None,
            "unit": self.unit,
            "truth_source": self.truth_source,
        }


def dimension_errors(
    measured: Mapping[str, float],
    truth: Mapping[str, float],
    *,
    name: str,
    truth_source: str,
    unit: str = COORDINATE_UNIT,
) -> DimensionErrors:
    """Compare named lengths to truth, and say where the truth came from.

    ``truth_source`` is required and is written into the report: a dimension
    gate is only as good as the instrument behind its truth, and "roughly 110 x
    95 x 24 mm" from a capture manifest is not an instrument. State the
    calipers, the drawing, or the synthetic construction.

    ``relative_error`` is reported per entry and summarised only when every
    truth value is non-zero - a relative error against a truth of zero is not
    defined, and substituting a large number for it would invent a failure.
    """
    _ensure_text(name, "measurement name")
    _ensure_text(unit, "dimension unit")
    _ensure_text(truth_source, "truth source")
    names = _ensure_named(measured, truth, what=f"{name} dimensions")

    entries: list[DimensionError] = []
    for entry_name in names:
        observed = _ensure_finite(measured[entry_name], f"{name} measured {entry_name!r}")
        expected = _ensure_finite(truth[entry_name], f"{name} truth {entry_name!r}")
        error = observed - expected
        entries.append(
            DimensionError(
                name=entry_name,
                measured=observed,
                truth=expected,
                error=error,
                absolute_error=abs(error),
                relative_error=(error / expected) if expected != 0.0 else None,
            )
        )
    relatives = [entry.relative_error for entry in entries]
    return DimensionErrors(
        name=name,
        entries=tuple(entries),
        absolute=_distribution(
            [entry.absolute_error for entry in entries],
            unit=unit,
            what=f"{name} dimension error",
        ),
        relative=(
            _distribution(
                [abs(value) for value in relatives if value is not None],
                unit=FRACTION_UNIT,
                what=f"{name} relative dimension error",
            )
            if all(value is not None for value in relatives)
            else None
        ),
        unit=unit,
        truth_source=truth_source,
    )


@dataclass(frozen=True, slots=True)
class LandmarkOffset:
    """One named point, and how far it sits from where it belongs."""

    name: str
    measured: Point
    truth: Point
    offset: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "measured": list(self.measured),
            "truth": list(self.truth),
            "offset": self.offset,
        }


@dataclass(frozen=True, slots=True)
class LandmarkOffsets:
    """Named landmark positions against explicitly supplied truth."""

    name: str
    entries: tuple[LandmarkOffset, ...]
    offsets: Distribution
    unit: str
    truth_source: str

    @property
    def kind(self) -> str:
        return "landmark_offsets"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "entries": [entry.as_dict() for entry in self.entries],
            "offsets": self.offsets.as_dict(),
            "unit": self.unit,
            "truth_source": self.truth_source,
        }


def landmark_offsets(
    measured: Mapping[str, Point],
    truth: Mapping[str, Point],
    *,
    name: str,
    truth_source: str,
    unit: str = COORDINATE_UNIT,
) -> LandmarkOffsets:
    """How far each named landmark sits from the position truth gives it.

    Both mappings are in the SAME frame; nothing is fitted between them. A
    landmark set compared after a best-fit alignment would answer a different
    question - whether the shape is right - and this one is whether the
    features are where they should be.
    """
    _ensure_text(name, "measurement name")
    _ensure_text(unit, "offset unit")
    _ensure_text(truth_source, "truth source")
    names = _ensure_named(measured, truth, what=f"{name} landmarks")

    entries: list[LandmarkOffset] = []
    for entry_name in names:
        observed = measured[entry_name]
        expected = truth[entry_name]
        _ensure_points([observed], what=f"{name} measured {entry_name!r}")
        _ensure_points([expected], what=f"{name} truth {entry_name!r}")
        entries.append(
            LandmarkOffset(
                name=entry_name,
                measured=observed,
                truth=expected,
                offset=distance(observed, expected),
            )
        )
    return LandmarkOffsets(
        name=name,
        entries=tuple(entries),
        offsets=_distribution(
            [entry.offset for entry in entries], unit=unit, what=f"{name} landmark offset"
        ),
        unit=unit,
        truth_source=truth_source,
    )


# --- bridge policy audit ------------------------------------------------------------


def _record_mapping(record: Mapping[str, object], key: str, what: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise BridgeRecordUnusable(f"{what}: {key!r} is {type(value).__name__}, not a mapping")
    return value


def _record_bool(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise BridgeRecordUnusable(f"{what}: {key!r} is {value!r}, not a boolean")
    return value


def _record_float(record: Mapping[str, object], key: str, what: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BridgeRecordUnusable(f"{what}: {key!r} is {value!r}, not a number")
    return _ensure_finite(float(value), f"{what} {key!r}")


def _record_candidates(evidence: Mapping[str, object], what: str) -> list[Mapping[str, object]]:
    raw = evidence.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise BridgeRecordUnusable(f"{what}: 'candidates' is not a list")
    candidates: list[Mapping[str, object]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise BridgeRecordUnusable(f"{what}: candidate {index} is not a mapping")
        candidates.append(entry)
    if not candidates:
        raise BridgeRecordUnusable(
            f"{what}: the record carries no candidates, so there is no winner to re-judge. A "
            "record like this was never a decision and reporting it as a refusal would let a "
            "truncated file look like a caught regression"
        )
    return candidates


def _record_overlaps(evidence: Mapping[str, object], what: str) -> tuple[float, ...]:
    raw = evidence.get("frame_overlaps")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise BridgeRecordUnusable(f"{what}: 'frame_overlaps' is not a list")
    overlaps: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BridgeRecordUnusable(f"{what}: frame overlap {index} is {value!r}, not a number")
        overlaps.append(_ensure_finite(float(value), f"{what} frame overlap {index}"))
    return tuple(overlaps)


@dataclass(frozen=True, slots=True)
class BridgeAudit:
    """A recorded relocalization bridge, re-judged under CURRENT policy.

    A policy that only governs future decisions leaves every record written
    under the old one saying what it always said. This re-reads the numbers a
    bridge record already carries and applies today's gates to them, so a scan
    accepted under a policy that has since been tightened is reported as the
    known-bad case it is rather than remembered as an acceptance.

    What is re-derived and what is taken as recorded is stated explicitly:
    the winner is re-selected from the recorded candidates by today's rule
    (lowest residual, higher overlap breaking a tie) and every scalar gate is
    re-applied to it; the runner-up margin and the coarse-ambiguity flag are
    read as recorded, because both were decided during a search whose
    intermediate state a record does not carry.
    """

    name: str
    source: str
    source_sha256: str | None
    recorded_accepted: bool
    coarse_unambiguous: bool
    winner_residual: float
    winner_overlap: float
    winner_converged: bool
    winner_matched_fraction: float
    winner_used_fraction: float
    other_candidate_max_overlap: float | None
    a_rival_covers_more: bool
    runner_up_margin: float | None
    had_runner_up: bool
    frame_overlaps: tuple[float, ...]
    worst_frame_overlap: float | None
    refusals: tuple[str, ...]
    policy: Mapping[str, object]

    @property
    def kind(self) -> str:
        return "bridge_audit"

    @property
    def accepted_under_current_policy(self) -> bool:
        return not self.refusals

    @property
    def regressed(self) -> bool:
        """The record says accepted and current policy says no."""
        return self.recorded_accepted and not self.accepted_under_current_policy

    @property
    def refused_for_coarse_ambiguity(self) -> bool:
        """Refused specifically because the coarse search was ambiguous.

        Separate from :attr:`accepted_under_current_policy` on purpose. A test
        that only asserted "refused" would still pass if the record were
        refused for an unrelated reason - a truncated overlap, a residual typo -
        and would stop testing the policy change it was written for.
        """
        return not self.coarse_unambiguous

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "recorded_accepted": self.recorded_accepted,
            "accepted_under_current_policy": self.accepted_under_current_policy,
            "regressed": self.regressed,
            "refused_for_coarse_ambiguity": self.refused_for_coarse_ambiguity,
            "coarse_unambiguous": self.coarse_unambiguous,
            "winner_residual": self.winner_residual,
            "winner_overlap": self.winner_overlap,
            "winner_converged": self.winner_converged,
            "winner_matched_fraction": self.winner_matched_fraction,
            "winner_used_fraction": self.winner_used_fraction,
            "other_candidate_max_overlap": self.other_candidate_max_overlap,
            "a_rival_covers_more": self.a_rival_covers_more,
            "runner_up_margin": self.runner_up_margin,
            "had_runner_up": self.had_runner_up,
            "frame_overlaps": list(self.frame_overlaps),
            "worst_frame_overlap": self.worst_frame_overlap,
            "refusals": list(self.refusals),
            "policy": dict(self.policy),
        }


def audit_bridge_record(
    record: Mapping[str, object],
    *,
    name: str,
    source: str,
    source_sha256: str | None = None,
) -> BridgeAudit:
    """Re-judge a written bridge record under the policy in force today.

    ``record`` is a parsed ``bridge.json`` - either the whole file, in which
    case its ``evidence`` block is used, or that block on its own.

    Nothing is recomputed from frames: this reads the measurements the record
    already carries. That is the point. The 2026-08-28 record is byte-for-byte
    what the capture wrote, and the finding is that today's gates refuse the
    numbers it recorded, not that a re-run would produce different ones.
    """
    _ensure_text(name, "audit name")
    _ensure_text(source, "audit source")
    what = f"bridge record {name!r}"
    evidence = _record_mapping(record, "evidence", what) if "evidence" in record else record

    recorded_accepted = _record_bool(evidence, "accepted", what)
    coarse_unambiguous = _record_bool(evidence, "coarse_unambiguous", what)
    candidates = _record_candidates(evidence, what)
    frame_overlaps = _record_overlaps(evidence, what)

    # Re-select the winner by today's rule rather than trusting the recorded
    # order: lowest residual, and where two are indistinguishable the one
    # explaining MORE of the segment. Matches scanner3d.relocalize.
    ranked = sorted(
        candidates,
        key=lambda entry: (
            _record_float(entry, "rmse", what),
            -_record_float(entry, "overlap", what),
        ),
    )
    winner = ranked[0]
    winner_residual = _record_float(winner, "rmse", what)
    winner_overlap = _record_float(winner, "overlap", what)
    winner_converged = _record_bool(winner, "converged", what)
    winner_matched = _record_float(winner, "matched_fraction", what)
    winner_used = _record_float(winner, "used_fraction", what)

    others = [_record_float(entry, "overlap", what) for entry in ranked[1:]]
    other_max = max(others) if others else None

    had_runner_up = _record_bool(evidence, "had_runner_up", what)
    raw_margin = evidence.get("runner_up_margin")
    if had_runner_up:
        margin: float | None = _record_float(evidence, "runner_up_margin", what)
    elif raw_margin is not None:
        raise BridgeRecordUnusable(
            f"{what}: 'had_runner_up' is false but 'runner_up_margin' is {raw_margin!r}. The two "
            "are written together and disagreeing means the record cannot be read"
        )
    else:
        margin = None

    max_residual = RESIDUAL_MULTIPLE * NOISE_FLOOR
    refusals: list[str] = []
    # First and unconditionally, exactly as scanner3d.relocalize orders it: an
    # ambiguous coarse ranking is a statement about the SEARCH, so no number
    # measured after it can withdraw it.
    if not coarse_unambiguous:
        refusals.append(
            "the record says the coarse search did not separate its best orientation from a "
            f"genuinely different one (coarse_unambiguous=false, {AMBIGUITY_MARGIN} margin). "
            "Current policy refuses that before anything measured afterwards: every refined "
            "candidate descends from the same ambiguous ranking, so a margin between two of them "
            "says where the optimiser stopped, not which placement is right"
        )
    if not winner_converged:
        refusals.append(
            "the winning alignment never converged; it used its whole iteration budget and was "
            "still moving"
        )
    if winner_residual > max_residual:
        refusals.append(
            f"winner residual {winner_residual:.3f} is over the {max_residual:.3f} gate "
            f"({RESIDUAL_MULTIPLE:g}x the {NOISE_FLOOR} noise floor)"
        )
    if winner_matched < MIN_MATCHED_FRACTION:
        refusals.append(
            f"only {winner_matched:.2f} of the segment found a partner, under the "
            f"{MIN_MATCHED_FRACTION} gate"
        )
    if winner_used < MIN_USED_FRACTION:
        refusals.append(
            f"only {winner_used:.2f} of the segment survived the trim, under the "
            f"{MIN_USED_FRACTION} gate"
        )
    if winner_overlap < MIN_OVERLAP:
        refusals.append(
            f"the segment overlaps the accepted model by {winner_overlap:.2f}, under the "
            f"{MIN_OVERLAP} gate"
        )
    if margin is not None and margin < RUNNER_UP_MARGIN:
        refusals.append(
            f"a genuinely different placement fits almost as well ({margin:.2f}x the winner's "
            f"residual, under the {RUNNER_UP_MARGIN} margin)"
        )
    thin = [index for index, value in enumerate(frame_overlaps) if value < MIN_FRAME_OVERLAP]
    if thin:
        refusals.append(
            f"{len(thin)} of the {len(frame_overlaps)} recorded frames overlap the model by less "
            f"than {MIN_FRAME_OVERLAP} (worst {min(frame_overlaps):.2f})"
        )

    return BridgeAudit(
        name=name,
        source=source,
        source_sha256=source_sha256,
        recorded_accepted=recorded_accepted,
        coarse_unambiguous=coarse_unambiguous,
        winner_residual=winner_residual,
        winner_overlap=winner_overlap,
        winner_converged=winner_converged,
        winner_matched_fraction=winner_matched,
        winner_used_fraction=winner_used,
        other_candidate_max_overlap=other_max,
        a_rival_covers_more=other_max is not None and other_max > winner_overlap,
        runner_up_margin=margin,
        had_runner_up=had_runner_up,
        frame_overlaps=frame_overlaps,
        worst_frame_overlap=min(frame_overlaps) if frame_overlaps else None,
        refusals=tuple(refusals),
        policy=thresholds(),
    )


# --- reports -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseReport:
    """One quality case: everything measured, every gate, every failure.

    ``notes`` is where a tier says what it could NOT measure. A case that
    simply omits a landmark gate because no truth was supplied reads, to
    anyone scanning the report, exactly like a case that measured landmarks
    and passed. A note saying "landmarks: not measured, no instrument truth
    supplied" does not.
    """

    name: str
    measurements: tuple[QualityMeasurement, ...]
    gates: tuple[Gate, ...]
    notes: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[Gate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case": self.name,
            "passed": self.passed,
            # Beside "passed", always. A case that applied no gate at all
            # passes trivially, and a reader skimming JSON must be able to see
            # the difference between "nothing failed" and "nothing was asked".
            "gate_count": len(self.gates),
            "measurements": [measurement.as_dict() for measurement in self.measurements],
            "gates": [gate.as_dict() for gate in self.gates],
            "failures": [gate.as_dict() for gate in self.failures],
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Every gate, in order, with failures marked. Deterministic.

        Every gate rather than only the failures: the passing ones are what
        say which failure mode was ruled out, and a log that showed only
        failures would make a case that failed one gate look identical to a
        case that only had one.
        """
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"[{verdict}] case {self.name}: {len(self.failures)}/{len(self.gates)} failed"]
        lines.extend(f"  {gate.render()}" for gate in self.gates)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """Several cases, evaluated and reported together."""

    name: str
    cases: tuple[CaseReport, ...]

    @property
    def failures(self) -> tuple[tuple[str, Gate], ...]:
        return tuple((case.name, gate) for case in self.cases for gate in case.failures)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite": self.name,
            "passed": self.passed,
            "case_count": len(self.cases),
            "gate_count": sum(len(case.gates) for case in self.cases),
            "failure_count": len(self.failures),
            "cases": [case.as_dict() for case in self.cases],
        }

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        header = (
            f"[{verdict}] suite {self.name}: {len(self.failures)} failed gate(s) across "
            f"{len(self.cases)} case(s)"
        )
        return "\n".join([header, *(case.render() for case in self.cases)])


def to_json(report: CaseReport | SuiteReport, *, indent: int = 2) -> str:
    """Serialise a report strictly: no NaN, no infinity, no silent widening.

    ``allow_nan=False`` is the point. Python's default emits ``NaN`` and
    ``Infinity``, which are not JSON, and a report that quietly wrote them
    would be accepted by this repository's own reader and rejected by any
    other consumer - or worse, read back as something that compares false to
    everything. Every measurement here refuses a non-finite value at its own
    door; this is the check that the whole assembly held.
    """
    return json.dumps(report.as_dict(), allow_nan=False, indent=indent, sort_keys=False)
