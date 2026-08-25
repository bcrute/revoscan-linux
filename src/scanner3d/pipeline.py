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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.posegraph import PoseEdge, edge_residuals, optimise_poses
from scanner3d.registration import icp
from scanner3d.revisit import detect_revisits

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


class PipelineNotAllowed(ScannerError):
    """A reconstruction was requested that this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


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
        return {
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


def _fit_pair(job: tuple[_PairSettings, Sequence[Point], Sequence[Point]]) -> Rigid:
    """Align one pair. Module-level and taking a single argument so it pickles.

    ``multiprocessing.Pool.map`` needs a top-level function and one argument;
    a closure over the settings would not survive being sent to a subprocess.
    """
    settings, source, target = job
    if settings.point_to_plane:
        return icp_point_to_plane(
            source,
            target,
            max_correspondence_distance=settings.max_correspondence_distance,
            voxel_size=settings.voxel_size,
            keep_fraction=settings.keep_fraction,
            backend=settings.backend,
        ).transform
    return icp(
        source,
        target,
        max_correspondence_distance=settings.max_correspondence_distance,
        voxel_size=settings.voxel_size,
        keep_fraction=settings.keep_fraction,
    ).transform


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

    def report(message: str) -> None:
        if callable(on_progress):
            on_progress(message)

    gap = min_index_gap if min_index_gap is not None else max(8, len(clouds) // 8)

    matcher = "point-to-plane" if point_to_plane else "point-to-point"
    report(f"aligning {len(clouds)} frames ({matcher}, backend={backend}, workers={workers})")
    settings = _PairSettings(
        point_to_plane=point_to_plane,
        backend=backend,
        max_correspondence_distance=max_correspondence_distance,
        voxel_size=voxel_size,
        keep_fraction=keep_fraction,
    )
    pairs = [(clouds[i], clouds[i - 1]) for i in range(1, len(clouds))]

    transforms: list[Rigid] = []
    if workers > 1:
        # Each pair is independent, so they parallelise even though the poses
        # they compose into are strictly sequential. Composition happens after,
        # in order, so the result cannot depend on completion order.
        import multiprocessing

        try:
            with multiprocessing.Pool(processes=workers) as pool:
                transforms = pool.map(_fit_pair, [(settings, s, t) for s, t in pairs])
            report(f"  aligned {len(transforms)} pairs across {workers} workers")
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
            transforms = []
    if not transforms:
        for position, (source, target) in enumerate(pairs, start=1):
            transforms.append(_fit_pair((settings, source, target)))
            if position % 25 == 0:
                report(f"  aligned {position}/{len(pairs)}")

    poses: list[Rigid] = [Rigid.identity()]
    edges: list[PoseEdge] = []
    for index, transform in enumerate(transforms, start=1):
        edges.append(PoseEdge(index - 1, index, transform, 1.0))
        poses.append(poses[-1].compose(transform))

    report("looking for revisited viewpoints")
    revisits = detect_revisits(clouds, min_index_gap=gap, max_candidates=max_candidates)
    closures = list(revisits.pose_edges())
    report(f"  {len(closures)} verified revisits")

    limitations: list[str] = []
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
    )
