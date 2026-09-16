"""Capture to model, in one call, with an honest quality number attached.

Why this exists
---------------
Every piece of the reconstruction was built and tested separately: point-to-plane
alignment, revisit detection, pose-graph optimisation, fusion, meshing. Wiring
them together was left to whoever knew the order. That means the person holding
the scanner cannot run their own scan, which makes the whole thing useless to
them.

The quality number is the point
-------------------------------
This module always reports HELD-OUT CLOSURE ERROR, and that choice is the most
important thing in it.

Measured on a real two-lap capture of a model truck bed: per-pair alignment rmse
0.65, frame overlap 100%, self-agreement residuals around 2-3. Every internal
metric said the scan was healthy. The trajectory was wrong by 65 degrees after
one revolution and 139 after two.

Self-consistency cannot detect that. Registration optimises agreement between
frames, so measuring agreement afterwards is training error, and a drifted
trajectory is perfectly self-consistent. What exposes it is evidence the
optimiser never got to fit: revisited viewpoints, detected geometrically, half
of them withheld from the optimisation and used only to score it.

So a number printed by this module is a number the pipeline could not have
cheated. That is worth more than a smaller number that could.

What this does NOT do
---------------------
It cannot tell you the model is dimensionally correct. Held-out closure error
measures whether the trajectory is self-consistent WITH EVIDENCE IT DID NOT FIT,
which is a much stronger claim than self-consistency alone - but a systematic
depth-scale error produces a model that closes on itself perfectly and is
uniformly the wrong size. Only a caliper catches that.

It cannot register a sweep with no revisits. If the scanner never returns to a
viewpoint it already occupied, there is nothing to correct the chain against and
this reports that plainly rather than pretending. Coming back around to where
you started is a capture decision, not something software can supply.

No physical unit is implied. Distances are in whatever the depth scale produced.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from scanner3d.capture_sequence import BridgeBoundary
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.posegraph import PoseEdge, edge_residuals, optimise_poses
from scanner3d.registration import icp
from scanner3d.revisit import detect_revisits

if TYPE_CHECKING:  # pragma: no cover - imported for types only
    # :mod:`scanner3d.tracking` and :mod:`scanner3d.relocalize` both express
    # their thresholds as multiples of this module's :data:`NOISE_FLOOR`, so
    # they import it and this module cannot import them back at module scope.
    # The one runtime use is a local import inside :func:`reconstruct_sweep`;
    # everything else here is a type.
    from scanner3d.relocalize import LocalFit

SCHEMA_VERSION: Final = 1

#: Fraction of correspondences ICP keeps. Measured on real frames: the library
#: default of 0.6 recovers only 85% of the true rotation, because points far
#: from the axis move furthest between frames and are therefore the ones most
#: likely to be discarded as "too far" - and they are also the ones carrying the
#: rotation. 0.8 recovers 94%. Going to 1.0 is worse again, at 86%, because then
#: genuine outliers are kept.
DEFAULT_KEEP_FRACTION: Final = 0.8

#: Fewest verified revisits before the closure set is worth splitting. Below
#: this, holding half back leaves too few to score against and the reported
#: error would be noise wearing a decimal point.
MIN_CLOSURES_TO_SPLIT: Final = 4

#: Per-point measurement noise, measured on this scanner: repeated frames of a
#: stationary surface disagree by about this much. Every threshold below is
#: expressed as a multiple of it, because that is the quantity that makes them
#: portable to another scanner rather than to this one alone.
NOISE_FLOOR: Final = 0.65

#: How often the pairwise alignment reports its count: every this many pairs
#: and at the last one, on both the serial and the parallel path, so that a
#: caller drawing a bar from the lines sees it move rather than sit at zero
#: for the whole of the longest stage.
PROGRESS_EVERY: Final = 10

#: How far the surface should travel between kept frames, as a multiple of the
#: noise floor.
#:
#: These bounds are CALIBRATED AGAINST OUTCOMES, not derived. An earlier version
#: of this constant was estimated from the object's radius and an assumed
#: geometry, and it was wrong in the worst way: it called the two best-measured
#: settings "too slow" and the worst one "good". What follows is the measured
#: relationship instead, where travel is the median measured by
#: :func:`frame_travel` and drift is held-out closure error per revolution:
#:
#:     travel   multiple   drift per lap
#:     1.7        2.6x        31.2 deg
#:     2.5        3.8x         0.5 deg
#:     3.8        5.8x        23.4 deg
#:     7.8       12.0x        65.2 deg
#:
#: Below about 2 the frames are so alike that each alignment measures noise more
#: than motion, and those errors accumulate over many more steps. Above about 6
#: alignment begins to fail outright, and by 12 it is badly wrong.
#:
#: HOW FAR TO TRUST THIS: four points, two captures, and the 2.5 row is from a
#: capture of the object in a different orientation, so it is not a clean
#: comparison with the rest. The band is deliberately drawn wider than the
#: points justify narrowing it to. It is guidance for turning a knob, not a
#: calibration.
TRAVEL_MIN_MULTIPLE: Final = 3.0
TRAVEL_MAX_MULTIPLE: Final = 9.0


def frame_travel(cloud: Sequence[Point], transform: Rigid) -> float:
    """Median distance the surface moved between two frames, in depth units.

    This is the quantity that decides whether a rotation speed is sensible, and
    it is deliberately NOT an angle. What matters to alignment is how far the
    surface moved relative to the sensor's noise, and a small object turning
    fast moves its surface less than a large one turning slowly at the same
    angular rate. Degrees are a property of the turntable; millimetres are a
    property of the problem.

    The median is used rather than the mean because points near the rotation
    axis barely move however fast it turns, and averaging them in would report
    a speed lower than what the geometry that carries the alignment actually
    experiences.
    """
    if not cloud:
        return 0.0
    moved = sorted(math.dist(point, transform.apply(point)) for point in cloud)
    return moved[len(moved) // 2]


def travel_verdict(travel: float) -> tuple[str, str]:
    """Turn a measured travel into a verdict and an instruction.

    Returns ``(verdict, advice)``. The advice names the physical action - turn
    the knob one way or the other - because that is what the person holding the
    scanner can actually do about it.
    """
    low = TRAVEL_MIN_MULTIPLE * NOISE_FLOOR
    high = TRAVEL_MAX_MULTIPLE * NOISE_FLOOR
    if travel <= 0.0:
        return ("no motion", "Nothing is moving. Is the turntable running?")
    if travel < low:
        return (
            "too slow",
            f"Frames are only {travel:.1f} apart, under the {low:.1f} that alignment "
            "needs to see motion rather than noise. Speed the rotation up.",
        )
    if travel > high:
        return (
            "too fast",
            f"Frames are {travel:.1f} apart, over the {high:.1f} where alignment starts "
            "to fail. Slow the rotation down.",
        )
    return ("good", f"{travel:.1f} between frames is in the band that registers well.")


class PipelineNotAllowed(ScannerError):
    """A reconstruction was requested that this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


class SweepFrameUnplaceable(PipelineNotAllowed):
    """One frame of the sweep could not be placed, so the sweep is refused.

    A missing pose cannot be quietly skipped and must not be manufactured. A
    skipped frame silently renumbers everything after it, and a manufactured one
    - the previous pose repeated, an interpolation between neighbours - produces
    a chain that looks complete and is wrong exactly where the evidence ran out.
    So the reconstruction refuses, naming the frame and carrying the complete
    placement evidence for it in :attr:`decision`.
    """

    def __init__(self, frame: int, decision: object, message: str) -> None:
        super().__init__(message)
        #: Position in the sweep of the frame that could not be placed.
        self.frame = frame
        #: The full :meth:`~scanner3d.tracking.TrackDecision.as_dict` record for
        #: it: every candidate tried, every gate, and every measurement.
        self.decision = decision


@dataclass(frozen=True, slots=True)
class ClosureScore:
    """How well a trajectory agrees with revisits it was never given.

    ``held_out`` is the count of constraints withheld from the optimisation.
    When it is zero, the errors below are training error and say very little;
    the flag exists so a caller cannot mistake one for the other.
    """

    median_error_radians: float
    worst_error_radians: float
    held_out: int
    used_for_fitting: int

    @property
    def is_held_out(self) -> bool:
        return self.held_out > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "median_error_degrees": math.degrees(self.median_error_radians),
            "worst_error_degrees": math.degrees(self.worst_error_radians),
            "held_out": self.held_out,
            "used_for_fitting": self.used_for_fitting,
            "is_held_out": self.is_held_out,
        }


@dataclass(frozen=True, slots=True)
class SweepReconstruction:
    """Poses for one sweep, and what the evidence says about them."""

    poses: tuple[Rigid, ...]
    frames: int
    revisits_found: int
    chained_score: ClosureScore | None
    optimised_score: ClosureScore | None
    optimised: bool
    converged: bool
    limitations: tuple[str, ...]
    #: Which recorded relocalization bridges were used INSTEAD of fitting an
    #: adjacent pair, named by the segment each came from. Empty for every
    #: ordinary capture, and absent from :meth:`as_dict` when it is empty, so a
    #: sweep with no reposition produces byte-for-byte the record it always did.
    bridges: tuple[str, ...] = ()
    #: What :mod:`scanner3d.tracking` decided while building the sequential
    #: chain: the schema-versioned policy it ran under, the keyframes it ended
    #: with, how many frames each kind of reference placed, and how many of
    #: those were model-only rescues. Present on every reconstruction, because
    #: the live capture and this path share that policy and a reader comparing
    #: the two has to be able to see it rather than assume it.
    tracking: Mapping[str, object] | None = None

    @property
    def improvement_factor(self) -> float | None:
        """How much optimisation reduced held-out error. None if unmeasurable."""
        before, after = self.chained_score, self.optimised_score
        if before is None or after is None or not after.is_held_out:
            return None
        if after.median_error_radians <= 0.0:
            return math.inf
        return before.median_error_radians / after.median_error_radians

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "frames": self.frames,
            "revisits_found": self.revisits_found,
            "optimised": self.optimised,
            "converged": self.converged,
            "chained": self.chained_score.as_dict() if self.chained_score else None,
            "optimised_score": self.optimised_score.as_dict() if self.optimised_score else None,
            "improvement_factor": self.improvement_factor,
            "limitations": list(self.limitations),
        }
        if self.bridges:
            payload["relocalization_bridges"] = list(self.bridges)
        if self.tracking is not None:
            payload["tracking"] = dict(self.tracking)
        return payload


@dataclass(frozen=True, slots=True)
class _PairSettings:
    """Everything a worker needs to align one pair, and nothing it does not.

    Kept separate from the clouds so the settings are one small picklable object
    rather than something reconstructed per pair.
    """

    point_to_plane: bool
    backend: str
    max_correspondence_distance: float
    voxel_size: float
    keep_fraction: float
    #: The fraction of the noise floor the residual must stop moving by before
    #: the fit counts as settled - :data:`scanner3d.tracking.CONVERGENCE_NOISE_FRACTION`,
    #: carried here so a pair pre-fitted in a worker process is solved under the
    #: identical convergence setting the tracker would have solved it under.
    #: Without it the offline path and the live path would judge the same pair's
    #: ``converged`` flag differently, which is now a hard gate.
    convergence_noise_fraction: float


def _fit_pair(job: tuple[_PairSettings, Sequence[Point], Sequence[Point]]) -> LocalFit | None:
    """Align one pair. Module-level and taking a single argument so it pickles.

    ``multiprocessing.Pool.map`` needs a top-level function and one argument;
    a closure over the settings would not survive being sent to a subprocess.

    Returns the WHOLE fit rather than its transform, because the tracker that
    consumes it judges a fit on its residual, convergence and matched and used
    fractions, and a transform alone throws all of that away. ``None`` when ICP
    refused the pair outright - a singular flat panel is the ordinary cause -
    which is information about the pair and is handled by the tracker like any
    other candidate that could not be formed.

    ``source`` and ``target`` must already be the bounded clouds the tracker
    itself would have used - :func:`scanner3d.tracking.bounded_source` and
    :func:`scanner3d.tracking.retained_cloud` - and the convergence setting is
    the tracker's own. A pre-fit solved on different points, or settled against
    a different threshold, is a fit of a different problem, and the whole reason
    to hand it in is that it is the same one.
    """
    from scanner3d.tracking import convergence_tolerance

    settings, source, target = job
    tolerance = convergence_tolerance(target, fraction=settings.convergence_noise_fraction)
    try:
        if settings.point_to_plane:
            return icp_point_to_plane(
                source,
                target,
                tolerance=tolerance,
                max_correspondence_distance=settings.max_correspondence_distance,
                voxel_size=settings.voxel_size,
                keep_fraction=settings.keep_fraction,
                backend=settings.backend,
            )
        return icp(
            source,
            target,
            tolerance=tolerance,
            max_correspondence_distance=settings.max_correspondence_distance,
            voxel_size=settings.voxel_size,
            keep_fraction=settings.keep_fraction,
            backend=settings.backend,
        )
    except ScannerError:
        return None
    except ValueError, ArithmeticError:
        return None


def _validated_bridges(
    bridges: Sequence[BridgeBoundary] | None, frames: int
) -> dict[int, BridgeBoundary]:
    """The bridges as a position lookup, or a refusal naming what was wrong.

    Validated here rather than trusted, because a bridge REPLACES a measurement:
    a position outside the sweep, a duplicate, or one at the very first frame
    would either be silently ignored or silently displace a pair that should
    have been fitted, and both are the kind of quiet wrongness this pipeline
    exists to refuse.
    """
    if not bridges:
        return {}
    placed: dict[int, BridgeBoundary] = {}
    for bridge in bridges:
        if not 1 <= bridge.position < frames:
            raise PipelineNotAllowed(
                f"{bridge.source} bridges position {bridge.position} of a {frames}-frame sweep; "
                "a boundary sits between two frames that are both in it"
            )
        if bridge.position in placed:
            raise PipelineNotAllowed(
                f"{bridge.source} and {placed[bridge.position].source} both bridge position "
                f"{bridge.position}; a pair has one relationship"
            )
        placed[bridge.position] = bridge
    return placed


def _bridge_names(boundaries: dict[int, BridgeBoundary]) -> tuple[str, ...]:
    return tuple(boundaries[position].source for position in sorted(boundaries))


def _bridge_limitations(boundaries: dict[int, BridgeBoundary]) -> tuple[str, ...]:
    """What a reader has to be told about a sweep that was repositioned.

    First in the list, ahead of every other limitation, because it changes what
    the score below it means: the sequential chain is no longer entirely
    measured from this data, and the parts of it that are not came from a
    decision made live, under thresholds recorded elsewhere.
    """
    if not boundaries:
        return ()
    named = ", ".join(f"{boundaries[p].source} at frame {p}" for p in sorted(boundaries))
    limitations = [
        f"This sweep was REPOSITIONED {len(boundaries)} time(s) mid-capture ({named}). At each "
        "of those positions the relative transform came from a recorded relocalization decided "
        "during the capture, and the pair was NOT fitted here. The chain is therefore only as "
        "good as those decisions, whose own evidence and thresholds are in each segment's "
        "bridge record.",
    ]
    for position in sorted(boundaries):
        limitations.extend(boundaries[position].limitations)
    return tuple(limitations)


def _score(poses: Sequence[Rigid], held: Sequence[PoseEdge], used: int) -> ClosureScore:
    errors = sorted(r.rotation_error_radians for r in edge_residuals(poses, held))
    return ClosureScore(
        median_error_radians=errors[len(errors) // 2] if errors else 0.0,
        worst_error_radians=errors[-1] if errors else 0.0,
        held_out=len(held),
        used_for_fitting=used,
    )


def reconstruct_sweep(
    clouds: Sequence[Sequence[Point]],
    *,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    max_correspondence_distance: float = 12.0,
    voxel_size: float = 1.5,
    point_to_plane: bool = True,
    backend: str = "python",
    workers: int = 1,
    robust: bool = True,
    min_index_gap: int | None = None,
    max_candidates: int = 60,
    iterations: int = 400,
    bridges: Sequence[BridgeBoundary] | None = None,
    on_progress: object = None,
) -> SweepReconstruction:
    """Register one sweep, and score it against evidence it was not given.

    ``point_to_plane`` selects the matcher. Measured on a real capture, the
    point-to-point default drifts 14.4 degrees per lap where point-to-plane
    drifts 2.6, because point-to-point is insensitive to motion along a surface
    and a rotation moves distant points along the surface. It is left switchable
    because that finding is one object on one scanner, not a law.

    Revisits are detected geometrically, so nothing here assumes a turntable, an
    angular step, or any particular capture path. A sweep taken by walking
    around an object and one taken by turning it are the same problem.

    ``bridges`` names the positions where this sweep was REPOSITIONED mid-capture
    and supplies the relative transform recorded for each, from
    :func:`scanner3d.capture_sequence.read_capture_sequence`. At such a position
    the pair is not fitted: the two frames are not a small step apart, and ICP
    handed them returns a confident, small residual for a transform that is
    nonsense. The recorded transform is used instead and enters the graph as a
    protected sequential edge, exactly like a fitted one. With no bridges - every
    ordinary capture, and every capture taken before pausing existed - this
    fits precisely the pairs it always did, in the same order.

    ``backend`` selects the compute path and defaults to the pure-Python one, so
    the answer is the reference answer unless a caller asks otherwise. ``workers``
    spreads the pairwise alignments over processes: frame *i* against *i-1*
    depends on no other pair's result, so they are independent even though the
    poses they compose into are not. Neither changes the answer, and both are
    reported rather than assumed - a run whose speed cannot be attributed to a
    setting is a run whose timing means nothing.
    """
    if len(clouds) < 3:
        raise PipelineNotAllowed(f"a sweep needs at least three frames, got {len(clouds)}")
    if not 0.0 < keep_fraction <= 1.0:
        raise PipelineNotAllowed(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    boundaries = _validated_bridges(bridges, len(clouds))

    def report(message: str) -> None:
        if callable(on_progress):
            on_progress(message)

    gap = min_index_gap if min_index_gap is not None else max(8, len(clouds) // 8)

    matcher = "point-to-plane" if point_to_plane else "point-to-point"
    report(f"aligning {len(clouds)} frames ({matcher}, backend={backend}, workers={workers})")
    from scanner3d.tracking import DEFAULT_POLICY, bounded_source, retained_cloud

    settings = _PairSettings(
        point_to_plane=point_to_plane,
        backend=backend,
        max_correspondence_distance=max_correspondence_distance,
        voxel_size=voxel_size,
        keep_fraction=keep_fraction,
        convergence_noise_fraction=DEFAULT_POLICY.convergence_noise_fraction,
    )
    # The positions this run FITS, which is every adjacent pair except the ones
    # a recorded bridge already answers. With no bridges this is the whole range
    # and ``pairs`` is the identical list, in the identical order, that this
    # function has always built.
    fitted = [i for i in range(1, len(clouds)) if i not in boundaries]
    # The IDENTICAL two clouds the tracker's own adjacent candidate would be
    # built from: the offered frame thinned to the candidate source budget, and
    # the previous frame thinned exactly as the tracker retains it. An offline
    # cloud is denser than a live preview frame, so without this the pre-fit
    # would be a different solve from the one being handed in.
    pairs = [
        (bounded_source(clouds[i], DEFAULT_POLICY), retained_cloud(clouds[i - 1], DEFAULT_POLICY))
        for i in fitted
    ]
    if boundaries:
        report(f"  {len(boundaries)} recorded relocalization bridge(s); those pairs are NOT fitted")

    fits: list[LocalFit | None] = []
    if workers > 1:
        # Each pair is independent, so they parallelise even though the poses
        # they compose into are strictly sequential. Composition happens after,
        # in order, so the result cannot depend on completion order.
        import multiprocessing

        try:
            with multiprocessing.Pool(processes=workers) as pool:
                # imap rather than map: the same order, but each fit is handed
                # back as it completes, so the count below can be reported.
                jobs = [(settings, s, t) for s, t in pairs]
                for position, fit in enumerate(pool.imap(_fit_pair, jobs), start=1):
                    fits.append(fit)
                    if position % PROGRESS_EVERY == 0 or position == len(pairs):
                        report(f"  aligned {position}/{len(pairs)}")
            report(f"  aligned {len(fits)} pairs across {workers} workers")
        except RuntimeError as exc:
            # Python 3.14 defaults to forkserver on Linux, which re-imports the
            # caller's __main__. A caller without an `if __name__ == "__main__"`
            # guard therefore cannot spawn workers at all. Falling back keeps
            # their scan alive - but SILENTLY falling back would make their next
            # timing meaningless, so this says so as loudly as it can.
            report(
                f"  WORKERS UNAVAILABLE, running single-process instead: {exc.__class__.__name__}. "
                'Wrap the call in `if __name__ == "__main__":` to use multiple processes.'
            )
            fits = []
    if not fits and pairs:
        for position, (source, target) in enumerate(pairs, start=1):
            fits.append(_fit_pair((settings, source, target)))
            if position % PROGRESS_EVERY == 0 or position == len(pairs):
                report(f"  aligned {position}/{len(pairs)}")

    # Back into position order. The adjacent fits above are only ONE of the
    # candidates each frame is placed from - see :mod:`scanner3d.tracking` - and
    # they are computed here, up front and across processes, because an adjacent
    # pair depends on no other pair even though the poses they compose into are
    # strictly sequential. Everything after this point is sequential, so a run
    # with workers and a run without reach the identical chain.
    fitted_at = dict(zip(fitted, fits, strict=True))
    from scanner3d.tracking import IncrementalTracker

    tracker = IncrementalTracker(
        backend=backend,
        point_to_plane=point_to_plane,
        max_correspondence_distance=max_correspondence_distance,
        voxel_size=voxel_size,
        keep_fraction=keep_fraction,
    )
    report(f"placing {len(clouds)} frames against keyframes and the accepted model")
    decisions = []
    for index in range(len(clouds)):
        if index == 0:
            decision = tracker.offer(clouds[0])
        elif index in boundaries:
            # The pair is NOT fitted. The two frames are not a small step apart,
            # and the transform recorded live is what places this one; the
            # tracker continues from there against the same accepted model.
            decision = tracker.seed(
                clouds[index], tracker.poses[-1].compose(boundaries[index].transform)
            )
        else:
            decision = tracker.offer(clouds[index], adjacent_fit=fitted_at[index])
        if not decision.accepted:
            # Refused HERE rather than after the loop, so the frame index in the
            # refusal is the sweep position: a refusal accepts nothing, so the
            # tracker's own numbering has not moved past it, and every frame
            # before it was placed.
            raise SweepFrameUnplaceable(
                index,
                decision.as_dict(),
                f"frame {index} of this {len(clouds)}-frame sweep could not be placed: "
                f"{decision.message()} A sweep with an unplaceable frame is refused rather "
                "than reconstructed with that frame skipped or its pose manufactured.",
            )
        decisions.append(decision)
        if index and index % 25 == 0:
            report(f"  placed {index}/{len(clouds) - 1}")

    poses: list[Rigid] = [decision.pose for decision in decisions if decision.pose is not None]
    edges: list[PoseEdge] = [
        PoseEdge(index - 1, index, poses[index - 1].inverse().compose(poses[index]), 1.0)
        for index in range(1, len(poses))
    ]
    origins = Counter(decision.origin for decision in decisions)
    report("  placed by " + ", ".join(f"{count} {origin}" for origin, count in origins.items()))
    tracking = {
        **tracker.record(),
        "origins": dict(origins),
        "rescues": sum(1 for decision in decisions if decision.rescue),
        # Counted apart from ``rescues``. A model-only rescue and a bounded
        # coarse recovery are different events - one is the model placing a
        # frame the previous one could not, the other is a coarse search
        # placing a frame no ordinary reference agreed on - and a reader
        # comparing a reconstruction with the capture that produced it needs to
        # see which of the two carried the sweep.
        "coarse_recoveries": sum(
            1
            for decision in decisions
            if decision.recovery is not None and decision.recovery.accepted
        ),
        # PER FRAME, not only the totals. Two runs that placed the same number
        # of frames against the same number of models did not necessarily make
        # the same decisions, and an aggregate count cannot tell the difference
        # - which is exactly what the backend-equivalence test needs to assert
        # and what round 1 could only assert around.
        "placements": [
            {
                "frame": position,
                "origin": decision.origin,
                "rescue": decision.rescue,
                "recovered": decision.recovery is not None and decision.recovery.accepted,
                "keyframes": list(decision.keyframes),
            }
            for position, decision in enumerate(decisions)
        ],
    }

    report("looking for revisited viewpoints")
    revisits = detect_revisits(clouds, min_index_gap=gap, max_candidates=max_candidates)
    closures = list(revisits.pose_edges())
    report(f"  {len(closures)} verified revisits")

    limitations: list[str] = list(_bridge_limitations(boundaries))
    if len(closures) < MIN_CLOSURES_TO_SPLIT:
        limitations.append(
            f"Only {len(closures)} revisits were found, fewer than the {MIN_CLOSURES_TO_SPLIT} "
            "needed to withhold any. The chain was NOT corrected and NOT scored: with no "
            "revisited viewpoint there is nothing a drifting trajectory can be caught against. "
            "Capture a path that returns to where it started."
        )
        return SweepReconstruction(
            poses=tuple(poses),
            frames=len(clouds),
            revisits_found=len(closures),
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=tuple(limitations),
            bridges=_bridge_names(boundaries),
            tracking=tracking,
        )

    # Half the closures are withheld. Scoring against constraints the optimiser
    # was given would be training error - the exact mistake that let a
    # 65-degree drift look healthy on every other metric.
    train = closures[0::2]
    held = closures[1::2]
    before = _score(poses, held, len(train))

    report(f"optimising against {len(train)} revisits, holding {len(held)} back")
    # Sequential edges are protected because they carry a systematic bias, not
    # random disagreement, and robust rejection can misfire on them. Closures
    # are left exposed deliberately: a false revisit looks perfect locally - on
    # real data, five of them had BETTER rmse than the true ones - so the only
    # place the lie is visible is in disagreement with the consensus.
    protected = [PoseEdge(e.i, e.j, e.transform, e.weight, protected=True) for e in edges]
    result = optimise_poses(
        poses, protected + train, anchor=0, iterations=iterations, robust=robust
    )
    after = _score(list(result.poses), held, len(train))

    rejected = getattr(result, "rejected_edges", ())
    if rejected:
        report(f"  robust fitting rejected {len(rejected)} closure(s) as inconsistent")
        limitations.append(
            f"Robust fitting rejected {len(rejected)} closure constraint(s) as inconsistent "
            "with the consensus. A rejected closure is usually a false revisit - two views "
            "that a failed alignment claimed were the same. Inspect them before trusting a "
            "scan where many were dropped."
        )
    limitations.extend(
        [
            "Held-out closure error is the strongest number here, but it is not accuracy. "
            "A systematic depth-scale error produces a model that closes on itself perfectly "
            "and is uniformly the wrong size; only a caliper catches that.",
            "Revisits are verified by aligning raw frames, so a genuinely repeated viewpoint "
            "is required. An object that changed between passes - shifted, relit, or handled - "
            "breaks that assumption silently.",
            *result.limitations[:2],
        ]
    )
    # The held-out set exists to judge the answer, so use it. If optimisation
    # made agreement with constraints it never saw WORSE, the chain was better
    # and handing back the optimised poses would be knowingly returning the
    # worse of two measured options. Measured on a real sweep: chained 2.7
    # degrees, optimised 22.8, because false revisits survived rejection.
    #
    # This is only defensible because the comparison is held-out. Judged on
    # training error the optimised poses always win, by construction, and this
    # branch would never fire on the case it exists for.
    optimisation_helped = after.median_error_radians <= before.median_error_radians
    if not optimisation_helped:
        limitations.insert(
            0,
            f"OPTIMISATION REJECTED: it made held-out agreement WORSE "
            f"({math.degrees(before.median_error_radians):.3f} -> "
            f"{math.degrees(after.median_error_radians):.3f} degrees), so the CHAINED poses "
            "were returned instead. The usual cause is a false revisit that survived "
            "rejection - two views a failed alignment claimed were the same. The chain is "
            "not thereby good; it is only the better of two measured options.",
        )
    return SweepReconstruction(
        poses=tuple(result.poses) if optimisation_helped else tuple(poses),
        frames=len(clouds),
        revisits_found=len(closures),
        chained_score=before,
        optimised_score=after,
        optimised=optimisation_helped,
        converged=result.converged,
        limitations=tuple(limitations),
        bridges=_bridge_names(boundaries),
        tracking=tracking,
    )
