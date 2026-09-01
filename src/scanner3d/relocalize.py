"""Where a repositioned scanner is, decided from geometry rather than hope.

Why this exists
---------------
A capture is a chain: every frame is placed by aligning it against the one
before it, and that works precisely because two consecutive frames of a slowly
turning object are nearly the same picture. Picking the object up and putting it
down the other way round breaks that assumption completely. The next frame is
not a small step from the last one, it is a different view of the same surface
from somewhere else entirely, and adjacent ICP handed that pair does not report
a failure - it reports a confident, small residual for a transform that is
nonsense, because it optimised agreement between whatever points happened to
land near each other.

So the reposition has to be treated as what it is: a GLOBAL search for where the
new frames sit relative to the model already accepted, followed by a decision
about whether the answer found is believable. That decision is this module.

What "believable" is allowed to mean
------------------------------------
Not "the best candidate". A best candidate exists for every input, including
inputs with no overlap at all, and turning "best" into "known" is exactly the
mistake that produces a scan with two halves joined at a plausible-looking angle
that is simply wrong.

Every gate here is a named threshold with a recorded measurement beside it, and
every one of them has to pass:

- the tentative frames chained to each other locally, so the segment being
  placed is a small surface rather than one noisy view;
- the COARSE search separated its best orientation from the runner-up, so the
  shape being placed admits one placement rather than several. This one is
  decided before any refinement and cannot be argued out of afterwards: see
  below;
- ICP converged, rather than merely finishing its iteration budget;
- the residual sits within a multiple of the measured per-point noise floor
  this repository already uses to express every other alignment threshold;
- enough of the segment found a partner (matched) and enough of it survived the
  trim (used), because a tiny residual over three points is not evidence;
- the segment genuinely overlaps the old model, measured as a fraction rather
  than assumed from the residual;
- the runner-up candidate is either the SAME answer or clearly worse, so a
  symmetric object - which offers several equally good placements - refuses
  instead of picking one;
- and every individual tentative frame agrees, not just their union, because a
  union score can be carried by one flattering corner while the rest of the
  segment sits nowhere near the model.

A refusal is not a failure of this module. It is the answer, and the operator
can act on it: move back into the overlap, pause and reposition again, or stop.

Why an ambiguous coarse search ends the decision
------------------------------------------------
:func:`~scanner3d.multiview.coarse_align` reports ``unambiguous=False`` when its
best orientation is not clearly better than a genuinely different one. That is a
statement about the SHAPE - this segment sits several ways on this model and the
geometry does not say which - and refining is not an answer to it. Refinement
takes one starting point and slides it downhill; handed a shape with several
minima it returns whichever one it started nearest, at a residual that looks
exactly like the residual of a placement that was never in doubt.

This module used to record that as a limitation beside an accepted bridge and
let the refined runner-up margin speak for the decision. It cost a real scan: on
2026-08-28 a bridge was accepted at residual 0.515 with a 1.50x refined margin
whose own record said ``coarse_unambiguous: false``, and the placement it chose
was wrong - a different refined candidate covered MORE of the model (0.598
against 0.483). A margin measured between two candidates that both descend from
an ambiguous ranking cannot distinguish them; it only says the optimiser
stopped in different places.

So an ambiguous coarse search is a refusal, before and regardless of every
number measured after it. The candidates are still refined and still recorded,
because a refusal a reader cannot inspect is not evidence - but no bridge comes
out of one, and the operator is told what would fix it: show the scanner a part
of the object that does not look like the rest of it.

What this does NOT do
---------------------
It does not establish that the reposition is correct, only that the evidence
listed above cleared thresholds chosen conservatively. A repeated, self-similar
surface can satisfy all of them at the wrong placement; that is why the accepted
evidence is written down rather than reduced to a boolean, and why the
reconstruction that later consumes it states the limitation.

It does not RUN the local chaining. The caller owns the tentative segment's own
adjacent alignment - that is ordinary capture work - but it does not get to
decide whether a link is good enough: :func:`judge_local_edge` applies the same
named gates to each adjacent fit before the frame is allowed into the segment,
so a segment that never chained to itself cannot be rescued by a flattering
global match.

It knows nothing about turntables. A reposition is exactly the case where an
estimated spindle from before the pause stopped describing the object, so no
axis is assumed and no candidate is seeded at the identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.multiview import AMBIGUITY_MARGIN, coarse_align, overlap_fraction
from scanner3d.pipeline import NOISE_FLOOR

SCHEMA_VERSION: Final = 1

#: Fewest tentative frames a relocalization may be decided from.
#:
#: Three, not one. One frame is a single view whose noise is indistinguishable
#: from its geometry at this scale, and a global search over it will find
#: somewhere plausible to put it on almost any model. Three chained frames are a
#: small local surface patch, and a patch has an orientation a single view does
#: not.
MIN_TENTATIVE_FRAMES: Final = 3

#: Most tentative frames held in memory before the buffer starts sliding.
#:
#: A resume that never relocalizes must not grow without limit, and this is the
#: frame half of that bound. Past it the oldest tentative frame is dropped, so
#: the segment keeps describing where the scanner is NOW rather than where it
#: was when the operator first pressed Resume.
MAX_TENTATIVE_FRAMES: Final = 12

#: Ceiling on the raw bytes held for an unaccepted segment. One frame of this
#: scanner's depth mode is 512000 bytes, so this is roughly twenty-four of them
#: - comfortably above :data:`MAX_TENTATIVE_FRAMES` at that size, and a hard
#: stop for any future mode whose frames are larger than assumed here.
MAX_TENTATIVE_BYTES: Final = 24 * 1024 * 1024

#: How long a resume may keep collecting before it gives up and says so.
#: Bounded because an operator who has walked away should find a capture that
#: reported a refusal, not one still buffering.
MAX_TENTATIVE_SECONDS: Final = 180.0

#: Relocalization attempts one resume may make. Each attempt costs a coarse
#: ranking and several ICP refinements, and an attempt is only made when a new
#: frame has added evidence, so this bounds the CPU a hopeless resume can spend.
MAX_ATTEMPTS: Final = 8

#: Points in the bounded reference built from the accepted model. Chosen so a
#: coarse ranking and a handful of refinements stay within a second or two on
#: this host; the reference is a ranking and scoring surface, not the model.
REFERENCE_POINTS: Final = 4000

#: Points in the tentative segment's union cloud, for the same reason.
SEGMENT_POINTS: Final = 1500

#: Coarse candidates carried through to ICP refinement. The coarse stage ranks
#: twenty-five orientations; refining all of them buys nothing, because a
#: candidate ranked below this by the coarse score has never won after refining
#: on any synthetic or real case measured here, and refinement is the expensive
#: half.
CANDIDATES_REFINED: Final = 5

#: Residual ceiling, as a multiple of :data:`~scanner3d.pipeline.NOISE_FLOOR`.
#: The same vocabulary the travel guidance and the drift thresholds use, so a
#: number here is comparable with the numbers elsewhere rather than being a new
#: scale invented for this decision.
RESIDUAL_MULTIPLE: Final = 4.0

#: Distance within which a segment point counts as overlapping the reference,
#: as a multiple of the noise floor.
OVERLAP_TOLERANCE_MULTIPLE: Final = 3.0

#: Fraction of the tentative segment that must land on the old model.
#: Deliberately well above zero: the whole point of asking the operator to move
#: back into the overlap is that a bridge with no shared surface has nothing to
#: be decided from.
MIN_OVERLAP: Final = 0.35

#: The same fraction, required of EVERY tentative frame individually.
#: Lower than :data:`MIN_OVERLAP`, because a frame at the leading edge of the
#: segment legitimately sees less of the old surface than the union does - but
#: not zero, which is what "one corner carried the whole score" looks like.
MIN_FRAME_OVERLAP: Final = 0.20

#: Share of segment points that must find a partner inside the ICP gate.
MIN_MATCHED_FRACTION: Final = 0.55

#: Share of segment points the final solve must actually have used.
MIN_USED_FRACTION: Final = 0.40

#: How much worse a genuinely different runner-up must be before the winner
#: counts as clearly best. A ratio of residuals, so it carries no unit.
RUNNER_UP_MARGIN: Final = 1.25

#: Residual ceiling for ONE adjacent link inside the tentative segment, as a
#: multiple of :data:`~scanner3d.pipeline.NOISE_FLOOR`.
#:
#: Separate constants from the global gates above, and the same numbers, on
#: purpose. A tentative link is ordinary adjacent alignment - two frames a small
#: step apart - so it is judged in the same vocabulary as every other alignment
#: threshold in this repository; but it is a DIFFERENT decision from "where does
#: this segment sit in the model", and a future measurement that moves one of
#: them should not silently move the other.
LOCAL_RESIDUAL_MULTIPLE: Final = 4.0

#: Share of a tentative frame that must find a partner in the frame before it.
LOCAL_MIN_MATCHED_FRACTION: Final = 0.55

#: Share of it the final local solve must actually have used.
LOCAL_MIN_USED_FRACTION: Final = 0.40

#: When two refined candidates count as the SAME answer rather than rivals.
#: Two coarse seeds routinely fall into one minimum, and calling that pair
#: "ambiguous" would refuse every well-determined reposition there is.
SAME_POSE_DEGREES: Final = 3.0
SAME_POSE_DISTANCE_MULTIPLE: Final = 3.0


class RelocalizationNotAllowed(ScannerError):
    """A relocalization was requested this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


def thresholds() -> dict[str, object]:
    """Every named gate, as data, for the record a decision is written into.

    A record that says "accepted" and nothing else cannot be re-judged later
    against a threshold that has since moved. This is what makes the accepted
    evidence interpretable by a reader who was not here.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "noise_floor": NOISE_FLOOR,
        "min_tentative_frames": MIN_TENTATIVE_FRAMES,
        "max_tentative_frames": MAX_TENTATIVE_FRAMES,
        "max_tentative_bytes": MAX_TENTATIVE_BYTES,
        "max_tentative_seconds": MAX_TENTATIVE_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "reference_points": REFERENCE_POINTS,
        "segment_points": SEGMENT_POINTS,
        "candidates_refined": CANDIDATES_REFINED,
        # Owned by :mod:`~scanner3d.multiview` and enforced here, so it is
        # recorded here: a reader re-judging an accepted bridge has to be able
        # to see which coarse margin it was required to clear, and that number
        # lives in another module's constant where this record cannot reach it.
        "coarse_ambiguity_margin": AMBIGUITY_MARGIN,
        "residual_multiple": RESIDUAL_MULTIPLE,
        "max_residual": RESIDUAL_MULTIPLE * NOISE_FLOOR,
        "overlap_tolerance_multiple": OVERLAP_TOLERANCE_MULTIPLE,
        "overlap_tolerance": OVERLAP_TOLERANCE_MULTIPLE * NOISE_FLOOR,
        "min_overlap": MIN_OVERLAP,
        "min_frame_overlap": MIN_FRAME_OVERLAP,
        "min_matched_fraction": MIN_MATCHED_FRACTION,
        "min_used_fraction": MIN_USED_FRACTION,
        "runner_up_margin": RUNNER_UP_MARGIN,
        "same_pose_degrees": SAME_POSE_DEGREES,
        "same_pose_distance": SAME_POSE_DISTANCE_MULTIPLE * NOISE_FLOOR,
        "local_chain": local_chain_gates(),
    }


def local_chain_gates() -> dict[str, object]:
    """The gates one adjacent link inside the tentative segment has to clear.

    Named separately from :func:`thresholds` as well as embedded in it, because
    the durable record writes an edge's MEASUREMENTS beside the gate it was
    judged against, and a reader comparing the two should not have to know which
    subset of the global thresholds applied to a local link.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "noise_floor": NOISE_FLOOR,
        "local_residual_multiple": LOCAL_RESIDUAL_MULTIPLE,
        "local_max_residual": LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR,
        "local_min_matched_fraction": LOCAL_MIN_MATCHED_FRACTION,
        "local_min_used_fraction": LOCAL_MIN_USED_FRACTION,
    }


def _json_number(value: float) -> float | None:
    """``value``, or ``None`` where JSON has no word for it.

    Every number here ends up inside a durable bridge record written with
    ``allow_nan=False``, because ``Infinity`` is what Python's encoder emits and
    is not JSON - a consumer outside Python refuses it, and the record would be
    unreadable exactly where it mattered most. ``None`` is also the honest word
    rather than a substitution: the one field that genuinely reaches infinity is
    the runner-up margin when there was no runner-up, and "not measurable" is
    what that means. The same convention
    :attr:`~scanner3d.pipeline.SweepReconstruction.improvement_factor` already
    uses.
    """
    return value if math.isfinite(value) else None


def _rigid_finite(transform: Rigid) -> bool:
    """Whether every component of a transform is a real number.

    One expression, used by both the candidate evidence and the local links, so
    "finite" cannot come to mean two different things in one module.
    """
    if not all(math.isfinite(value) for value in transform.translation):
        return False
    return all(math.isfinite(value) for row in transform.rotation for value in row)


#: What a caller is told when a decision refused, and why it is a tuple of
#: strings rather than an enum: each refusal carries the MEASUREMENT that failed
#: its gate, and the measurement is the part an operator can act on.
Refusals = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One coarse orientation, refined, and every number it was judged on."""

    #: The transform carrying SEGMENT-LOCAL coordinates into the accepted
    #: model's frame, after refinement.
    transform: Rigid
    origin: str
    coarse_score: float
    coarse_matched: int
    rmse: float
    converged: bool
    iterations: int
    matched_fraction: float
    used_fraction: float
    overlap: float
    resolved_backend: str

    @property
    def finite(self) -> bool:
        """Whether every number here, and the transform, is a real quantity."""
        if not all(
            math.isfinite(value)
            for value in (self.rmse, self.matched_fraction, self.used_fraction, self.overlap)
        ):
            return False
        return _rigid_finite(self.transform)

    def as_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform.as_dict(),
            "origin": self.origin,
            "coarse_score": self.coarse_score,
            "coarse_matched": self.coarse_matched,
            "rmse": _json_number(self.rmse),
            "rmse_unit": "input coordinate units, whose physical meaning is not established",
            "converged": self.converged,
            "iterations": self.iterations,
            "matched_fraction": _json_number(self.matched_fraction),
            "used_fraction": _json_number(self.used_fraction),
            "overlap": _json_number(self.overlap),
            "resolved_backend": self.resolved_backend,
        }


@dataclass(frozen=True, slots=True)
class LocalEdge:
    """One adjacent link inside the tentative segment, and how it was judged.

    The segment is chained to ITSELF before anything global is attempted, and
    until this existed a returned ICP fit was taken as a link whatever it said:
    a solve that ran out of iterations still moving, over a handful of surviving
    correspondences, with a residual several times the noise floor, was chained
    in exactly like a good one. A global search then placed that segment, and
    the placement inherited a local chain nothing had ever looked at.

    So each link carries the same four measurements the global decision carries,
    against gates in the same vocabulary, and the accepted ones are written into
    the durable record beside the bridge. ``frame`` is the position IN THE
    SEGMENT of the newer of the two frames, which is 1 for the first link,
    because frame 0 is the anchor and has no link.
    """

    frame: int
    rmse: float
    converged: bool
    iterations: int
    matched_fraction: float
    used_fraction: float
    resolved_backend: str
    accepted: bool
    refusals: Refusals

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "rmse": _json_number(self.rmse),
            "rmse_unit": "input coordinate units, whose physical meaning is not established",
            "converged": self.converged,
            "iterations": self.iterations,
            "matched_fraction": _json_number(self.matched_fraction),
            "used_fraction": _json_number(self.used_fraction),
            "resolved_backend": self.resolved_backend,
            "accepted": self.accepted,
            "refusals": list(self.refusals),
        }


class LocalFit(Protocol):
    """What one adjacent tentative alignment has to be able to report.

    A structural type rather than :class:`~scanner3d.icp_plane.IcpPlaneResult`
    itself, because this judgement reads six numbers and a transform and cares
    about nothing else - and because a test that has to construct a whole ICP
    result to ask "is a residual of 99 refused" is testing ICP's ability to fail
    on demand instead of testing the gate. ``IcpPlaneResult`` satisfies it.
    """

    @property
    def transform(self) -> Rigid: ...
    @property
    def rmse(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def matched_fraction(self) -> float: ...
    @property
    def used_fraction(self) -> float: ...
    @property
    def resolved_backend(self) -> str: ...


def judge_local_edge(fit: LocalFit, *, frame: int) -> LocalEdge:
    """Decide whether one adjacent tentative fit is a link worth chaining.

    Never raises: an unusable fit is an ordinary outcome of pointing a scanner
    somewhere while the object is being moved, and the caller is inside a
    capture that must not end over it.

    Finiteness first, and of the TRANSFORM as well as the numbers. A solve that
    produced a NaN rotation reports a NaN residual on some backends and a
    perfectly ordinary-looking one on others; a chain built from it is not
    slightly wrong, it is meaningless, and every gate below would be comparing
    against a quantity that is not one.
    """
    finite = all(
        math.isfinite(value) for value in (fit.rmse, fit.matched_fraction, fit.used_fraction)
    ) and _rigid_finite(fit.transform)
    refusals: list[str] = []
    if not finite:
        refusals.append(
            "the local alignment of this frame against the one before it did not produce "
            "finite numbers, so there is no link here to measure."
        )
    else:
        if not fit.converged:
            refusals.append(
                f"the local alignment used all {fit.iterations} of its iterations and was "
                "still moving, so this frame never settled against the one before it."
            )
        if fit.rmse > LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR:
            refusals.append(
                f"the local residual {fit.rmse:.3f} is over the "
                f"{LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR:.3f} gate "
                f"({LOCAL_RESIDUAL_MULTIPLE:g}x the {NOISE_FLOOR} noise floor)."
            )
        if fit.matched_fraction < LOCAL_MIN_MATCHED_FRACTION:
            refusals.append(
                f"only {fit.matched_fraction:.2f} of this frame found a partner in the one "
                f"before it, under the {LOCAL_MIN_MATCHED_FRACTION} gate."
            )
        if fit.used_fraction < LOCAL_MIN_USED_FRACTION:
            refusals.append(
                f"only {fit.used_fraction:.2f} of this frame survived the trim into the local "
                f"solve, under the {LOCAL_MIN_USED_FRACTION} gate."
            )
    return LocalEdge(
        frame=frame,
        rmse=fit.rmse,
        converged=fit.converged,
        iterations=fit.iterations,
        matched_fraction=fit.matched_fraction,
        used_fraction=fit.used_fraction,
        resolved_backend=fit.resolved_backend,
        accepted=not refusals,
        refusals=tuple(refusals),
    )


@dataclass(frozen=True, slots=True)
class Relocalization:
    """Where a tentative segment belongs, or why that was not established.

    ``accepted`` is never inferred from ``bridge`` being present and ``bridge``
    is never present without ``accepted``: the two are set together, once, at
    the end of :func:`relocalize`, so no caller can read one and act on the
    other.
    """

    accepted: bool
    #: SEGMENT-LOCAL coordinates into the accepted model's frame. ``None``
    #: whenever ``accepted`` is false, including when a best candidate existed.
    bridge: Rigid | None
    refusals: Refusals
    candidates: tuple[CandidateEvidence, ...]
    #: Overlap of each tentative frame, placed through ``bridge``, against the
    #: reference. Parallel to the clouds the caller passed in.
    frame_overlaps: tuple[float, ...]
    #: Ratio of the runner-up's residual to the winner's, for a runner-up that
    #: is a genuinely DIFFERENT placement. ``inf`` when there was no rival.
    runner_up_margin: float
    coarse_unambiguous: bool
    segment_points: int
    reference_points: int
    frames: int
    limitations: tuple[str, ...]

    @property
    def best(self) -> CandidateEvidence | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "accepted": self.accepted,
            "bridge": self.bridge.as_dict() if self.bridge is not None else None,
            "refusals": list(self.refusals),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "frame_overlaps": [_json_number(overlap) for overlap in self.frame_overlaps],
            # ``None`` when no refined candidate was a genuinely DIFFERENT
            # placement, which is the ordinary shape of a confident acceptance
            # rather than an error. The flag beside it is what stops a reader
            # having to guess which of the two "null" means.
            "runner_up_margin": _json_number(self.runner_up_margin),
            "had_runner_up": math.isfinite(self.runner_up_margin),
            "coarse_unambiguous": self.coarse_unambiguous,
            "segment_points": self.segment_points,
            "reference_points": self.reference_points,
            "frames": self.frames,
            "thresholds": thresholds(),
            "limitations": list(self.limitations),
        }


#: Everything a relocalization cannot establish, stated once here and copied
#: into every result so a record carries it whether it accepted or refused.
LIMITATIONS: Final = (
    "A relocalization places a small segment against an accumulated model whose own poses "
    "came from chained alignment, so it inherits that chain's drift. It establishes where the "
    "segment sits relative to the model as the model stood, not where either sits in truth.",
    "The gates are conservative thresholds on overlap, residual, matched fraction and "
    "candidate margin. A repeated or self-similar surface can satisfy all of them at the "
    "wrong placement; nothing here detects that, which is why the evidence is recorded "
    "rather than reduced to a decision.",
    "The reference is a bounded, evenly spread sample of the accepted clouds, not every "
    "point of them. Overlap and residual are measured against that sample.",
)


def bounded_reference(
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    limit: int = REFERENCE_POINTS,
) -> tuple[Point, ...]:
    """A bounded sample of the accepted model, placed by its accepted poses.

    Spread across ALL accepted frames rather than taken from the last few, and
    that is the whole design of it. The operator repositions to reach a part of
    the object the scanner has not seen, so the surface a resumed segment
    overlaps is very often one scanned minutes earlier - and a reference built
    from the final cloud alone would refuse exactly the reposition that was
    worth making.

    Frames are sampled evenly across the sweep first and points evenly within
    the survivors second, so the result is bounded by ``limit`` however long the
    capture has run, and the cost of building it does not grow with it.
    """
    if not clouds or not poses:
        return ()
    if len(clouds) != len(poses):
        raise RelocalizationNotAllowed(
            f"{len(clouds)} accepted clouds and {len(poses)} accepted poses do not pair up; "
            "a reference cannot be placed"
        )
    if limit <= 0:
        raise RelocalizationNotAllowed(f"a reference limit must be positive, got {limit}")
    # At most one frame per handful of points asked for, so a very long capture
    # does not contribute one point per frame and call the scatter a surface.
    frame_budget = max(1, min(len(clouds), limit // 32 or 1))
    if len(clouds) <= frame_budget:
        picks = list(range(len(clouds)))
    elif frame_budget == 1:
        picks = [len(clouds) - 1]
    else:
        span = (len(clouds) - 1) / (frame_budget - 1)
        picks = sorted({round(index * span) for index in range(frame_budget)})
    per_frame = max(1, limit // len(picks))
    points: list[Point] = []
    for index in picks:
        cloud = clouds[index]
        if not cloud:
            continue
        step = max(1, len(cloud) // per_frame)
        points.extend(poses[index].apply_all(tuple(cloud)[::step][:per_frame]))
    return tuple(points[:limit])


def segment_union(
    clouds: Sequence[Sequence[Point]],
    poses: Sequence[Rigid],
    *,
    limit: int = SEGMENT_POINTS,
) -> tuple[Point, ...]:
    """The tentative segment as one cloud in its own local frame, bounded.

    Its own frame, deliberately: the segment is chained against itself and knows
    nothing yet about where it sits. The transform this module looks for is the
    one that answers that.
    """
    if len(clouds) != len(poses):
        raise RelocalizationNotAllowed(
            f"{len(clouds)} tentative clouds and {len(poses)} tentative poses do not pair up"
        )
    if limit <= 0:
        raise RelocalizationNotAllowed(f"a segment limit must be positive, got {limit}")
    per_frame = max(1, limit // max(1, len(clouds)))
    points: list[Point] = []
    for cloud, pose in zip(clouds, poses, strict=True):
        if not cloud:
            continue
        step = max(1, len(cloud) // per_frame)
        points.extend(pose.apply_all(tuple(cloud)[::step][:per_frame]))
    return tuple(points[:limit])


def _same_placement(a: Rigid, b: Rigid) -> bool:
    """Whether two refined candidates are one answer rather than two rivals."""
    relative = a.inverse().compose(b)
    return (
        relative.rotation_angle <= math.radians(SAME_POSE_DEGREES)
        and relative.translation_magnitude <= SAME_POSE_DISTANCE_MULTIPLE * NOISE_FLOOR
    )


def _refine(
    source: Sequence[Point],
    target: Sequence[Point],
    initial: Rigid,
    origin: str,
    coarse_score: float,
    coarse_matched: int,
    *,
    backend: str,
    tolerance: float,
) -> CandidateEvidence | None:
    """One coarse candidate, refined and scored. ``None`` if ICP refused it.

    ICP raising is an ordinary outcome for a candidate that put the segment
    somewhere with nothing under it - too few correspondences survive the gate -
    and it is information about that candidate, not a failure of the attempt.
    """
    try:
        fit = icp_point_to_plane(source, target, initial=initial, backend=backend)
    except ScannerError:
        return None
    except ValueError, ZeroDivisionError, OverflowError:
        return None
    return CandidateEvidence(
        transform=fit.transform,
        origin=origin,
        coarse_score=coarse_score,
        coarse_matched=coarse_matched,
        rmse=fit.rmse,
        converged=fit.converged,
        iterations=fit.iterations,
        matched_fraction=fit.matched_fraction,
        used_fraction=fit.used_fraction,
        overlap=overlap_fraction(source, target, fit.transform, tolerance=tolerance),
        resolved_backend=fit.resolved_backend,
    )


def _refused(
    refusals: Sequence[str],
    *,
    candidates: Sequence[CandidateEvidence] = (),
    frame_overlaps: Sequence[float] = (),
    margin: float = math.inf,
    unambiguous: bool = False,
    segment_points: int = 0,
    reference_points: int = 0,
    frames: int = 0,
) -> Relocalization:
    """A refusal, with whatever evidence was gathered before it was reached."""
    return Relocalization(
        accepted=False,
        bridge=None,
        refusals=tuple(refusals),
        candidates=tuple(candidates),
        frame_overlaps=tuple(frame_overlaps),
        runner_up_margin=margin,
        coarse_unambiguous=unambiguous,
        segment_points=segment_points,
        reference_points=reference_points,
        frames=frames,
        limitations=LIMITATIONS,
    )


def relocalize(
    segment_clouds: Sequence[Sequence[Point]],
    segment_poses: Sequence[Rigid],
    reference: Sequence[Point],
    *,
    backend: str = "python",
) -> Relocalization:
    """Decide where a chained tentative segment sits in the accepted model.

    ``segment_poses`` are the segment's OWN chained poses, first one the
    identity; ``reference`` is a bounded sample of the accepted model placed by
    its accepted poses, which :func:`bounded_reference` builds. The bridge
    returned carries segment-local coordinates into the model's frame, so the
    world pose of tentative frame *i* is ``bridge.compose(segment_poses[i])``.

    Never raises for ordinary data. Every way of not having an answer - too few
    frames, an empty reference, a degenerate cloud, no surviving candidate, a
    gate that did not clear - comes back as a refusal carrying the measurement,
    because a caller running inside a stream callback must not be handed an
    exception and a capture must not end over a reposition that did not work.
    """
    frames = len(segment_clouds)
    tolerance = OVERLAP_TOLERANCE_MULTIPLE * NOISE_FLOOR

    if frames < MIN_TENTATIVE_FRAMES:
        return _refused(
            [
                f"{frames} tentative frame(s) collected, under the {MIN_TENTATIVE_FRAMES} this "
                "decision is allowed to be made from. One view is not a surface."
            ],
            frames=frames,
        )
    if len(segment_poses) != frames:
        return _refused(
            [
                f"{frames} tentative clouds and {len(segment_poses)} chained poses do not pair "
                "up, so the segment did not chain locally."
            ],
            frames=frames,
        )
    try:
        source = segment_union(segment_clouds, segment_poses)
    except RelocalizationNotAllowed as exc:
        return _refused([str(exc)], frames=frames)
    target = tuple(reference)
    if len(source) < MIN_TENTATIVE_FRAMES or len(target) < MIN_TENTATIVE_FRAMES:
        return _refused(
            [
                f"the tentative segment carries {len(source)} points and the reference "
                f"{len(target)}; there is not enough geometry on one side to align at all."
            ],
            segment_points=len(source),
            reference_points=len(target),
            frames=frames,
        )

    try:
        coarse = coarse_align(source, target)
    except ScannerError as exc:
        # A cloud with too few points, or one that does not span three
        # dimensions - a flat wall is the ordinary case. Neither determines an
        # orientation, so there is nothing to rank and nothing to accept.
        return _refused(
            [f"no coarse orientation could be ranked: {exc}"],
            segment_points=len(source),
            reference_points=len(target),
            frames=frames,
        )

    refined = [
        evidence
        for candidate in coarse.candidates[:CANDIDATES_REFINED]
        if (
            evidence := _refine(
                source,
                target,
                candidate.transform,
                candidate.origin,
                candidate.score,
                candidate.matched,
                backend=backend,
                tolerance=tolerance,
            )
        )
        is not None
    ]
    finite = [evidence for evidence in refined if evidence.finite]
    if not finite:
        return _refused(
            [
                f"{len(coarse.candidates)} coarse orientation(s) were ranked and none of them "
                "refined to a finite alignment; there is nothing to judge."
            ],
            candidates=tuple(refined),
            segment_points=len(source),
            reference_points=len(target),
            frames=frames,
        )

    # Residual first, overlap as the tie-break. Both directions are stated
    # rather than left to a sort key nobody can read back: a smaller residual is
    # a better fit, and where two fits are indistinguishable the one that
    # explains MORE of the segment is the one meant.
    finite.sort(key=lambda evidence: (evidence.rmse, -evidence.overlap))
    best = finite[0]
    rival = next(
        (
            evidence
            for evidence in finite[1:]
            if not _same_placement(best.transform, evidence.transform)
        ),
        None,
    )
    # Infinite when there is no genuinely different rival at all, and when the
    # winner's residual is zero - a ratio against zero says nothing, and a
    # perfect fit is not the case this gate exists to catch.
    margin = math.inf if rival is None or best.rmse <= 0.0 else rival.rmse / best.rmse

    bridge = best.transform
    frame_overlaps = tuple(
        overlap_fraction(cloud, target, bridge.compose(pose), tolerance=tolerance) if cloud else 0.0
        for cloud, pose in zip(segment_clouds, segment_poses, strict=True)
    )

    refusals: list[str] = []
    # First, and unconditionally. This is a fact about the SEARCH rather than
    # about the winner, so no measurement taken after it can withdraw it: every
    # refined candidate below descends from the same ambiguous ranking, and a
    # margin between two of them says where the optimiser stopped, not which
    # placement is right. Stated first so :func:`evidence_summary` - which reads
    # the first refusal - leads with the one the operator can act on.
    if not coarse.unambiguous:
        refusals.append(
            f"the coarse search ranked its best orientation only {coarse.margin:.2f}x ahead of "
            f"a genuinely different one, under the {AMBIGUITY_MARGIN} margin, so this shape "
            "sits several ways on the model already scanned and the geometry does not say "
            "which. Refining an ambiguous start chooses one of those placements; it cannot "
            "tell them apart. Move back over a part of the object you have already scanned "
            "that does NOT look like the rest of it - a corner, a handle, a marking - and "
            "resume again."
        )
    if not best.converged:
        refusals.append(
            f"the winning alignment used all {best.iterations} of its iterations and was still "
            "moving, so it never converged. Finishing is not converging."
        )
    if best.rmse > RESIDUAL_MULTIPLE * NOISE_FLOOR:
        refusals.append(
            f"residual {best.rmse:.3f} is over the {RESIDUAL_MULTIPLE * NOISE_FLOOR:.3f} gate "
            f"({RESIDUAL_MULTIPLE:g}x the {NOISE_FLOOR} noise floor this scanner was measured at)."
        )
    if best.matched_fraction < MIN_MATCHED_FRACTION:
        refusals.append(
            f"only {best.matched_fraction:.2f} of the segment found a partner inside the gate, "
            f"under the {MIN_MATCHED_FRACTION} needed for a residual to describe the object "
            "rather than a corner of it."
        )
    if best.used_fraction < MIN_USED_FRACTION:
        refusals.append(
            f"only {best.used_fraction:.2f} of the segment survived the trim into the final "
            f"solve, under the {MIN_USED_FRACTION} gate."
        )
    if best.overlap < MIN_OVERLAP:
        refusals.append(
            f"the segment overlaps the accepted model by {best.overlap:.2f}, under the "
            f"{MIN_OVERLAP} needed. Move back over surface that has already been scanned."
        )
    if rival is not None and margin < RUNNER_UP_MARGIN:
        refusals.append(
            f"a genuinely different placement fits almost as well ({margin:.2f}x the winner's "
            f"residual, under the {RUNNER_UP_MARGIN} margin). The geometry does not "
            "distinguish them - usual for a symmetric object - and choosing one anyway would "
            "manufacture confidence that is not there."
        )
    thin = [index for index, overlap in enumerate(frame_overlaps) if overlap < MIN_FRAME_OVERLAP]
    if thin:
        worst = min(frame_overlaps) if frame_overlaps else 0.0
        refusals.append(
            f"{len(thin)} of the {frames} tentative frames overlap the accepted model by less "
            f"than {MIN_FRAME_OVERLAP} (worst {worst:.2f}), so the union score is being carried "
            "by part of the segment rather than agreed by all of it."
        )

    limitations = list(LIMITATIONS)
    # No entry here for an ambiguous coarse ranking. It used to be one - beside
    # an ACCEPTED bridge, saying the refinement had disambiguated the start -
    # and that claim was wrong; it is a refusal now, and a refusal belongs in
    # ``refusals`` where a caller has to act on it rather than in a list a
    # record carries for the reader's information.
    if coarse.source_principal_degenerate or coarse.target_principal_degenerate:
        limitations.append(
            "At least one of the two clouds has near-equal principal extents, so its principal "
            "orientation is not determined by its geometry and the octahedral candidate set is "
            "what did the work."
        )

    if refusals:
        return Relocalization(
            accepted=False,
            bridge=None,
            refusals=tuple(refusals),
            candidates=tuple(finite),
            frame_overlaps=frame_overlaps,
            runner_up_margin=margin,
            coarse_unambiguous=coarse.unambiguous,
            segment_points=len(source),
            reference_points=len(target),
            frames=frames,
            limitations=tuple(limitations),
        )
    return Relocalization(
        accepted=True,
        bridge=bridge,
        refusals=(),
        candidates=tuple(finite),
        frame_overlaps=frame_overlaps,
        runner_up_margin=margin,
        coarse_unambiguous=coarse.unambiguous,
        segment_points=len(source),
        reference_points=len(target),
        frames=frames,
        limitations=tuple(limitations),
    )


def evidence_summary(result: Relocalization) -> str:
    """One line an operator can read, whichever way the decision went."""
    best = result.best
    if result.accepted and best is not None:
        margin = (
            "no rival" if math.isinf(result.runner_up_margin) else f"{result.runner_up_margin:.2f}x"
        )
        return (
            f"relocalized: {best.overlap:.2f} overlap, residual {best.rmse:.3f}, "
            f"matched {best.matched_fraction:.2f}, margin {margin}"
        )
    if result.refusals:
        return f"not yet placed: {result.refusals[0]}"
    return "not yet placed"


def as_record(result: Relocalization) -> Mapping[str, object]:
    """The decision as the mapping a durable bridge record embeds."""
    return result.as_dict()
