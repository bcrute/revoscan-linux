"""Match a scan to its own flipped-over twin, when shape alone cannot tell them apart.

The problem this solves
------------------------
A shallow object - a 1/18 scale truck bed is the motivating case - is scanned
resting on one side, then physically turned over to reach the underside, and
scanned again. The two point clouds must be registered against each other, and
:mod:`~scanner3d.multiview` is the module built for exactly that job: align
whole fused clouds with no shared axis between them.

It fails on this specific object, and not because it is under-powered. A
shallow box maps almost onto itself under a 180-degree turn: measured on the
real truckbed geometry, :func:`~scanner3d.multiview.coarse_align` reports
``unambiguous=False`` with a margin near 1.0, and rescoring its candidates with
full-resolution ICP (15000 points) has the WRONG candidate win by about one
per cent - an upside-down twin of the object, not the true flip, at a lower
RMSE than the truth. More ICP iterations, more points, a finer voxel: none of
it fixes this, because the objective being optimised - point-to-point
distance - genuinely cannot separate the two. They are both good fits to the
same shape.

The fix
-------
Shape is not the only evidence available. Every scan also observes the
surface the object was resting on, and that surface's normal - "up", in each
scan's own local frame - is a fact about which way the object was sitting,
not about its shape. If the object was genuinely turned over between the two
scans, the transform that correctly carries one scan onto the other must
carry the support-plane normal measured in the source scan onto the
*negation* of the support-plane normal measured in the target scan: the face
that touched the resting surface in one scan is, by construction, facing away
from it in the other. A shape-alone twin that merely happens to fit the
geometry well has no reason to respect that constraint, and observably does
not - see :func:`align_across_flip` for the measured numbers on the exact
case that motivated this module.

This is a filter on top of :func:`~scanner3d.multiview.coarse_align`, not a
replacement for it. Shape still proposes the candidates and ICP still ranks
them; the support-plane test only removes candidates that are geometrically
plausible but physically impossible given how the object was actually
resting.

Where the support-plane normal comes from is deliberately out of scope here.
A turntable is one convenient way to observe it - the object sits on a known
flat surface throughout the whole scan - but nothing about ``source_up`` or
``target_up`` assumes one exists. A handheld sweep that happens to capture a
few points of the table, floor or shelf the object was resting on is just as
valid a source for these vectors, since the physical fact they encode - which
way the object was sitting, relative to that scan's own frame - has nothing
to do with how the scan was captured.

What this does NOT do
----------------------
It cannot help when neither scan observed any resting surface to measure a
normal from, or when an object's true flip and its shape-twin also happen to
agree on the support-plane sign - the box used in this module's own tests is
built to have a twin that gets the sign wrong specifically to demonstrate the
mechanism, but nothing guarantees every real object's ambiguity resolves that
cleanly. It trusts ``source_up`` and ``target_up`` completely: a wrong or
noisy normal estimate produces a confidently wrong rejection or a confidently
wrong acceptance, and nothing here can detect that from the point data alone.
And it is still built on :func:`~scanner3d.multiview.coarse_align`'s own
candidate set - a true alignment that coarse_align never proposes cannot be
rescued by filtering candidates that were never there.

No physical unit is implied anywhere. Distances are in whatever the depth
scale produced, and angles are radians.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, dot, normalize
from scanner3d.multiview import AlignmentCandidate, coarse_align, overlap_fraction
from scanner3d.registration import (
    DEFAULT_KEEP_FRACTION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    IcpResult,
    cloud_extent,
    icp,
)

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: How far, in radians, a candidate's rotated support normal may sit from
#: exactly matching ``expected_sign`` and still count as passing. Measured
#: on the case this module was built for, the true flip and its shape-twin
#: land within a fraction of a degree of exactly +1 and -1 - see the module
#: docstring. 45 degrees is generous by that margin on purpose: it is meant
#: to absorb real axis-estimation noise, not to be a tight discriminator. The
#: discrimination comes from the sign being decisively wrong, not from
#: precision near it.
DEFAULT_UP_TEST_TOLERANCE: Final = math.radians(45.0)

#: How much better the ICP-rescored winner must be than the runner-up, among
#: candidates that survived the support-plane test, before the result is
#: reported as trustworthy rather than merely chosen. This object family
#: genuinely produces near-ties even after filtering - a caller needs to know
#: when that happened rather than being handed a confident-looking single
#: answer.
TRUST_MARGIN: Final = 1.03

#: :func:`~scanner3d.multiview.overlap_fraction`'s tolerance, as a fraction of
#: the target cloud's own extent, when the caller does not supply one. Tight
#: on purpose: by the time this runs, ICP has already converged, so a point
#: that is still far from its nearest neighbour is evidence the fit missed,
#: not noise to be tolerant of.
DEFAULT_OVERLAP_FRACTION: Final = 1.0 / 32.0


class SupportNotAllowed(ScannerError):
    """A flip-matching request was made that this module refuses to answer."""

    status = DoctorStatus.UNSUPPORTED_MODE


class EmptyCloud(SupportNotAllowed):
    """A source or target cloud carried no points at all."""


class UpVectorUnusable(SupportNotAllowed):
    """A support-plane normal is not finite, or has no length to give it a direction."""


class ExpectedSignUnusable(SupportNotAllowed):
    """``expected_sign`` was not exactly ``+1.0`` or ``-1.0``.

    Deliberately exact rather than "any positive/negative number": the sign of
    the flip is a fact the user knows - they either turned the object over or
    they did not - and this is not a value to be inferred, tuned or defaulted.
    """


class UpTestToleranceUnusable(SupportNotAllowed):
    """The support-plane test's angular tolerance is not a finite radian value in ``(0, pi]``."""


class NoCandidateSurvivedUpTest(SupportNotAllowed):
    """Every coarse candidate placed the support-plane normal where it should not be.

    Raised rather than falling back to the best unfiltered candidate. That
    fallback would silently reintroduce the exact failure this module exists
    to prevent: a confident-looking wrong answer, chosen by shape alone.
    """


class NoCandidateAlignable(SupportNotAllowed):
    """Candidates survived the support-plane test, but ICP could not rescore any of them.

    Means the clouds did not overlap at any of the surviving candidates'
    poses - a real failure of the source data, not something this module can
    paper over.
    """


@dataclass(frozen=True, slots=True)
class FlipAlignment:
    """The chosen transform across a physical flip, and how much to trust it.

    ``candidates_proposed`` is what :func:`~scanner3d.multiview.coarse_align`
    offered; ``candidates_rejected`` is how many of those the support-plane
    test removed before ICP ever ran on them; ``candidates_kept`` is what
    survived to be rescored. ``margin`` is the runner-up's RMSE over the
    winner's, among the *kept* candidates only - a rejected candidate's score
    plays no part in it, because a candidate that failed the physical test is
    not a competitor, whatever its shape-only fit looked like.
    """

    transform: Rigid
    rmse: float
    overlap: float
    candidates_proposed: int
    candidates_rejected: int
    candidates_kept: int
    margin: float
    runner_up_rmse: float | None
    trustworthy: bool
    expected_sign: float
    up_test_tolerance: float
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transform": self.transform.as_dict(),
            "rmse": self.rmse,
            "overlap": self.overlap,
            "candidates_proposed": self.candidates_proposed,
            "candidates_rejected": self.candidates_rejected,
            "candidates_kept": self.candidates_kept,
            "margin": self.margin,
            "runner_up_rmse": self.runner_up_rmse,
            "trustworthy": self.trustworthy,
            "expected_sign": self.expected_sign,
            "up_test_tolerance_radians": self.up_test_tolerance,
            "limitations": list(self.limitations),
        }


def _ensure_up_vector(vector: Point, *, what: str) -> None:
    """Refuse a support-plane normal before it can produce a confident-looking wrong answer.

    Written as an explicit check rather than a call to
    :func:`~scanner3d.geometry.normalize` so that a bad ``source_up`` or
    ``target_up`` is reported by *this* module's own exception, naming its own
    argument - matching how :func:`~scanner3d.registration.refine_on_turntable`
    validates its axis direction rather than letting a geometry-level
    exception speak for it.
    """
    if not all(math.isfinite(component) for component in vector) or dot(vector, vector) <= 0.0:
        raise UpVectorUnusable(
            f"{what} must be a finite, non-zero vector to have a direction, got {vector!r}"
        )


def _rotated(rotation_source: Rigid, vector: Point) -> Point:
    """Apply only the rotation of a transform to a direction, never its translation.

    A support-plane normal is a direction, not a position: adding a
    translation to it would mix a location into a quantity that has none, and
    the result would depend on where the coordinate origin happened to sit
    rather than on the rotation alone. Built through
    ``Rigid.of(rotation, (0, 0, 0))`` - the same rotation-only-application
    idiom :func:`~scanner3d.multiview.coarse_align` and
    :func:`~scanner3d.multiview.principal_frame` use - rather than
    multiplying the matrix by hand, so this stays validated the same way
    every other rotation-only application in this codebase is.
    """
    return Rigid.of(rotation_source.rotation, (0.0, 0.0, 0.0)).apply(vector)


def align_across_flip(
    source: Sequence[Point],
    target: Sequence[Point],
    *,
    source_up: Point,
    target_up: Point,
    expected_sign: float,
    tolerance: float = DEFAULT_UP_TEST_TOLERANCE,
    max_candidates: int | None = None,
    overlap_tolerance: float | None = None,
    icp_max_iterations: int = DEFAULT_MAX_ITERATIONS,
    icp_tolerance: float = DEFAULT_TOLERANCE,
    icp_voxel_size: float | None = None,
    icp_max_correspondence_distance: float | None = None,
    icp_keep_fraction: float = DEFAULT_KEEP_FRACTION,
) -> FlipAlignment:
    """The transform carrying ``source`` onto ``target``, filtered by physical plausibility.

    ``source_up`` and ``target_up`` are each cloud's own measured support-
    plane normal, expressed in that cloud's own local frame - the normal of
    whatever surface the object was resting on during that capture, recovered
    from however much of that surface the scan happened to see, not from the
    object itself. ``expected_sign`` states, as a fact the caller already
    knows, whether the object was physically turned over between the two
    scans: ``-1.0`` means it was (the transform must carry ``source_up`` onto
    ``-target_up``), ``+1.0`` means it was not (onto ``+target_up``).

    The pipeline is exactly the one that rescued the case this module was
    built for: :func:`~scanner3d.multiview.coarse_align` proposes candidate
    orientations from shape alone; every candidate is scored against
    ``expected_sign`` using its rotation only, and any that lands more than
    ``tolerance`` radians from the expected direction is dropped *before* any
    expensive matching runs on it; the survivors are rescored with full
    :func:`~scanner3d.registration.icp` at the resolution the caller's clouds
    were given at; the lowest-RMSE survivor is returned.

    Measured on the real case this module exists for: coarse_align proposed
    10 candidates, the support-plane test rejected 8 of them (including the
    one that would have won on RMSE alone, by about a percent, at 179 degrees
    of true error), and the true flip - RMSE 1.7529, 5.6 degrees of true error
    - won among the two that survived.

    Raises :class:`NoCandidateSurvivedUpTest` if nothing survives the filter,
    rather than falling back to the best unfiltered candidate - that fallback
    is exactly the failure this module exists to prevent. Raises
    :class:`NoCandidateAlignable` if every survivor fails to overlap at ICP
    time, which is a real statement about the data, not about this module's
    filtering.
    """
    source_points = tuple(source)
    target_points = tuple(target)
    if not source_points:
        raise EmptyCloud("source cloud is empty; nothing to align")
    if not target_points:
        raise EmptyCloud("target cloud is empty; nothing to align")
    _ensure_up_vector(source_up, what="source_up")
    _ensure_up_vector(target_up, what="target_up")
    if expected_sign != 1.0 and expected_sign != -1.0:
        raise ExpectedSignUnusable(
            f"expected_sign must be exactly +1.0 or -1.0, got {expected_sign!r}"
        )
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= math.pi:
        raise UpTestToleranceUnusable(
            f"tolerance must be a finite radian value in (0, pi], got {tolerance!r}"
        )

    coarse = coarse_align(source_points, target_points)
    candidates: Sequence[AlignmentCandidate] = coarse.candidates
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    source_up_unit = normalize(source_up)
    target_up_unit = normalize(target_up)
    sign_threshold = math.cos(tolerance)

    survivors: list[AlignmentCandidate] = []
    for candidate in candidates:
        rotated_up = _rotated(candidate.transform, source_up_unit)
        alignment = dot(rotated_up, target_up_unit) * expected_sign
        if alignment >= sign_threshold:
            survivors.append(candidate)

    rejected = len(candidates) - len(survivors)
    if not survivors:
        degrees = math.degrees(tolerance)
        raise NoCandidateSurvivedUpTest(
            f"none of the {len(candidates)} coarse candidates carried source_up to within "
            f"{degrees:.1f} degrees of the direction expected_sign={expected_sign!r} requires. "
            "Refusing rather than returning the best shape-only candidate, which is exactly "
            "the failure this module exists to prevent."
        )

    rescored: list[IcpResult] = []
    for candidate in survivors:
        try:
            result = icp(
                source_points,
                target_points,
                initial=candidate.transform,
                max_iterations=icp_max_iterations,
                tolerance=icp_tolerance,
                voxel_size=icp_voxel_size,
                max_correspondence_distance=icp_max_correspondence_distance,
                keep_fraction=icp_keep_fraction,
            )
        except ScannerError:
            # This one survivor could not be rescored - a degenerate
            # correspondence set at its particular pose, most likely. The
            # other survivors are unaffected; only if none of them can be
            # rescored does that become a reportable failure, below.
            continue
        rescored.append(result)

    if not rescored:
        raise NoCandidateAlignable(
            f"{len(survivors)} candidate(s) survived the support-plane test, but ICP could not "
            "rescore any of them - the clouds did not overlap at any of the surviving "
            "candidates' poses"
        )

    rescored.sort(key=lambda result: result.rmse)
    winner_transform = rescored[0].transform
    winner_rmse = rescored[0].rmse
    runner_up_rmse = rescored[1].rmse if len(rescored) > 1 else None
    margin = (
        runner_up_rmse / winner_rmse
        if runner_up_rmse is not None and winner_rmse > 0.0
        else math.inf
    )
    trustworthy = margin >= TRUST_MARGIN

    resolved_overlap_tolerance = (
        overlap_tolerance
        if overlap_tolerance is not None
        else cloud_extent(target_points) * DEFAULT_OVERLAP_FRACTION
    )
    overlap = overlap_fraction(
        source_points, target_points, winner_transform, tolerance=resolved_overlap_tolerance
    )

    limitations = [
        "The support-plane test trusts source_up and target_up completely: a wrong or noisy "
        "axis estimate produces a confidently wrong rejection or acceptance, and nothing here "
        "can detect that from the point data alone.",
        "Filtering removes candidates that coarse_align proposed; it cannot supply a correct "
        "orientation coarse_align never proposed in the first place.",
        "A low RMSE among survivors is not proof of a correct alignment, only the best of what "
        "passed the physical test - see `trustworthy` and `margin` for whether the winner beat "
        "the runner-up by a real margin or by a coin flip.",
        "No physical unit is implied. rmse and overlap tolerance are in whatever the depth "
        "scale produced.",
    ]
    if not trustworthy:
        limitations.append(
            f"NOT TRUSTWORTHY: the winning candidate's RMSE ({winner_rmse:.6g}) beat the "
            f"runner-up's ({runner_up_rmse:.6g}) by only {(margin - 1.0) * 100.0:.2f} per cent, "
            f"under the {(TRUST_MARGIN - 1.0) * 100.0:.0f} per cent margin this module requires "
            "to call a result confident. This object genuinely produces near-ties even after "
            "the support-plane filter; treat the winner as a plausible candidate, not a decided "
            "answer."
        )

    return FlipAlignment(
        transform=winner_transform,
        rmse=winner_rmse,
        overlap=overlap,
        candidates_proposed=len(candidates),
        candidates_rejected=rejected,
        candidates_kept=len(survivors),
        margin=margin,
        runner_up_rmse=runner_up_rmse,
        trustworthy=trustworthy,
        expected_sign=expected_sign,
        up_test_tolerance=tolerance,
        limitations=tuple(limitations),
    )
