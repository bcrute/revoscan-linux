"""Turntable registration: many clouds of one object into one frame, and the
loop-closure residual that says whether they agree.

Why a turntable is a different problem
--------------------------------------
Freehand scanning asks for an unknown six-degree-of-freedom transform between
every pair of frames, and the search for it is the whole difficulty. A powered
turntable removes almost all of that. The camera does not move; the object turns
by an approximately known angle about one fixed line in space between one
capture and the next. Two consequences shape everything below:

* The line is **not** the camera origin. It is the spindle, some line in the
  camera frame that has to be estimated from the data, and getting it wrong
  spreads the object into a hollow shell rather than collapsing it onto itself.
  :func:`estimate_turntable_axis` recovers it once, from the screw axes of the
  frame-to-frame transforms, and reports how much the individual estimates
  disagreed.
* Once the line is known, each frame needs one number - an angle - not six.
  :func:`register_turntable` refines that angle in closed form against nearest
  neighbours, which is far better conditioned than a general alignment and
  cannot drift sideways off the spindle no matter how poor the overlap.

And, uniquely, the result is **independently checkable**. After a full
revolution the composition of every step must return the first frame to itself.
Nothing in the fit enforces that; the frame that closes the loop is registered
against its neighbour exactly like every other frame, and the residual against
the identity is a measurement of the whole chain rather than a restatement of
what was fitted. :class:`LoopClosure` reports it, and reports its rotation and
translation halves separately because averaging an angle with a length is
meaningless.

Shape, not size
---------------
Registration works even though the physical unit of this device's depth scale is
unknown, and the reason is worth stating plainly: every cloud carries the *same*
unknown scale, so the transform relating two of them is unaffected by it. A
registered scan is therefore correctly **shaped** and of unknown **size**. No
distance in this module - no residual, no translation, no correspondence gate -
is a length in millimetres or in anything else. Every parameter that would
otherwise need a unit is expressed as a fraction of the input cloud's own
extent, so a caller never has to name a unit this repository has not earned.

What this module does NOT establish
-----------------------------------
Every residual here is an *internal consistency* measure. Frames agreeing with
each other is not frames agreeing with the object: a wrong depth scale, an
uncorrected lens distortion or the off-centre principal point recorded by this
unit moves every frame in the same way and leaves all of these numbers small. A
loop closure near zero says the poses are self-consistent around one revolution.
It does not say the object was reconstructed correctly, and nothing in this
repository has compared any of it against a physical measurement.

Selectable compute backend
----------------------------
:func:`icp` takes an optional ``backend`` - see :mod:`scanner3d.compute` for
the full protocol and the measured reason a KD-tree backend can be
substantially faster. It routes exactly the one cost that dominates this
module's runtime - the nearest-neighbour correspondence search
(:func:`_correspond_via_backend`) - through the selected backend. It changes
NOTHING else: the trim, the Horn solve, the gate arithmetic and the
convergence test are the identical code, unconditionally, regardless of
``backend``. The default, ``"python"``, does not merely produce equivalent
output to the code that existed before this parameter did - it IS that code,
unmodified, on the unmodified :class:`_VoxelIndex` path; the backend-driven
path is only ever reached by an explicit request for a live
:class:`~scanner3d.compute.ComputeBackend` or a non-``"python"`` backend
name, never by a caller who did not ask for it. :func:`register_turntable`
and everything it chains through :func:`_fit_pair` are untouched by this -
they call the same :class:`_VoxelIndex` path exactly as before, with no
``backend`` parameter of their own to thread through.

:mod:`scanner3d.compute` is imported lazily, inside :func:`_resolve_backend`,
rather than at this module's top level: that module imports
:class:`_VoxelIndex` from THIS one at ITS OWN top level, so an eager import
here in the other direction would be a genuine circular import, not merely an
awkward one. Deferring it to call time - after both modules have finished
loading - is what breaks the cycle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import (
    MIN_VECTOR_NORM,
    Point,
    Rigid,
    alignment_rmse,
    cross,
    dot,
    norm,
    normalize,
    pose_difference,
    rigid_from_correspondences,
    rotation_about_axis,
    scaled,
    subtract,
)

if TYPE_CHECKING:
    # Only for static type-checking - see the module docstring's "Selectable
    # compute backend" section on why this is never imported at runtime here.
    from scanner3d.compute import ComputeBackend

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: One cloud as it crosses the boundary into this module: an iterable of
#: three-float sequences. Deliberately permissive at the edge and normalised to
#: bare tuples inside, so a producer of geometry need not import a record type
#: from a registration module to hand it points.
type Cloud = Iterable[Sequence[float]]

#: The identity pose, bound once. A module constant rather than a call in a
#: default argument: :class:`Rigid` is frozen, so one shared instance is safe,
#: and the signature then names the default instead of hiding it behind a call.
IDENTITY_POSE: Final = Rigid.identity()

#: The most points one cloud may carry into a match. Four full 640x400 depth
#: frames' worth of samples - room to register a merged cloud, and a hard stop
#: long before a runaway producer exhausts memory. Never widened at run time.
MAX_CLOUD_POINTS: Final = 4 * 640 * 400

#: The fewest points a cloud may carry. Three is what
#: :func:`scanner3d.geometry.rigid_from_correspondences` needs to constrain a
#: rotation at all; fewer is not a sparse cloud, it is not a cloud.
MIN_CLOUD_POINTS: Final = 3

#: The most clouds one sweep may hold: a full revolution at half-degree steps.
MAX_CLOUDS: Final = 720

#: The fewest clouds any of this can be asked about. Two gives one pair, which
#: yields an axis with no cross-check; that case is allowed and flagged, never
#: silently reported as consistent.
MIN_CLOUDS: Final = 2

#: The most cells a grid may *span* across a cloud's bounding box. Only occupied
#: cells are ever allocated, so this does not bound memory directly - it bounds
#: the absurd case, a cell so small that no two points share one and the grid
#: reduces nothing while manufacturing millions of distinct keys. Refused rather
#: than attempted, because the failure mode is a machine that stops responding.
MAX_GRID_CELLS: Final = 4_000_000

#: The most ICP iterations one call may run. A bound, not a recommendation: a
#: fit that has not settled in far fewer than this is not about to.
MAX_ICP_ITERATIONS: Final = 500

#: Iterations an ICP call runs before giving up, unless the caller says otherwise.
DEFAULT_MAX_ITERATIONS: Final = 50

#: Bootstrap sweeps :func:`estimate_turntable_axis` runs by default, and the
#: ceiling on what a caller may ask for. Two by default because the first pass
#: has nothing to start from and the second starts from what the first found,
#: which on coarse partial-overlap data is the difference between a line that is
#: roughly right and one that is exact. Past a handful the estimate is not
#: improving, it is wandering.
DEFAULT_AXIS_PASSES: Final = 2
MAX_AXIS_PASSES: Final = 8

#: Convergence tolerance, as a fraction of the target cloud's bounding-box
#: diagonal. A fraction and not a length, because this repository has no length:
#: see the module docstring. The absolute threshold the run actually used is
#: recorded in :attr:`IcpResult.convergence_threshold`.
DEFAULT_TOLERANCE: Final = 1e-6

#: Voxel side used to downsample a large cloud, as a fraction of its extent.
#: A 64-cell grid across the object keeps the shape a surface fit needs while
#: cutting a real depth frame's 89000 surviving points to a few hundred - 243 of
#: them for the ``base_4`` fixture, 505 for ``final_test``. The gap between
#: those two is the point: a frame's extent is set by its furthest stray return,
#: not by the object, so this is a bound on cost and not a target point count.
DEFAULT_VOXEL_FRACTION: Final = 1.0 / 64.0

#: The correspondence gate, as a fraction of the target cloud's extent. Nothing
#: further away than this is ever paired. Generous on purpose: it has to cover
#: the displacement of a whole turntable step when a fit is started from the
#: identity, which is what the axis bootstrap does. It is the trim, not this
#: bound, that does the discriminating once a fit is under way.
DEFAULT_CORRESPONDENCE_FRACTION: Final = 1.0 / 8.0

#: The share of matched pairs a fit solves on, closest first. The rest are
#: trimmed, and trimming them is what makes this usable on a turntable at all.
#:
#: Every turntable step turns some surface out of view and some into it, so a
#: large minority of the points in one frame have no true partner in the next.
#: Their nearest neighbour is whatever happens to be closest, which is a false
#: pair, and on a turntable the false pairs all pull the same way. Measured on
#: the partial-overlap test sweep, an untrimmed fit under-rotated every step by
#: 15.7 per cent and the revolution then missed closing by 0.99 rad. The false
#: pairs are also, reliably, the *furthest* ones - which is exactly what a
#: quantile can cut. Trimming the same data at 0.6 recovers the step to
#: floating-point precision and closes the loop to 5e-15.
#:
#: 0.6 is deliberately below the overlap a sane sweep actually has, and the
#: margin is not free to give away: the same sweep trimmed at 0.8 still leaves
#: enough false pairs to close 1.9e-3 rad short. Keeping too few pairs costs
#: precision that a few hundred survivors make back; keeping too many leaves
#: false pairs in the solve, and those do not average out. A caller who knows
#: their overlap may raise it, and 1.0 turns trimming off for data whose
#: correspondences really are complete.
DEFAULT_KEEP_FRACTION: Final = 0.6

#: A cloud at or below this many points is matched as it stands. Downsampling
#: exists to make matching affordable, and it is not free: a voxel grid is fixed
#: in the camera frame, so the downsample of a rotated cloud is not the rotated
#: downsample, and the mismatch becomes a residual of its own. A cloud already
#: small enough to match directly should not pay it.
#:
#: Where "small enough" is: one frame-to-frame fit over a pair of 2400-point
#: clouds took about a second of CPython on the machine this was written on, so
#: a nine-frame sweep at that size is around twenty seconds. That is the scale
#: at which downsampling starts being the cheaper answer, and a caller who hands
#: in a cloud already past it has chosen to pay for the density.
DOWNSAMPLE_THRESHOLD: Final = 2048

#: Correspondences below which no solve is attempted. Three is the rigid case's
#: minimum; the constrained case needs only one, but a single surviving pair is
#: an overlap failure, not a measurement, whichever solver reads it.
MIN_CORRESPONDENCES: Final = 3

#: Below this angle a frame-to-frame rotation carries no recoverable spindle.
#: The anchor is recovered through ``cot(angle / 2)``, whose magnitude grows as
#: ``2 / angle``, so at a thousandth of a radian any error in the translation is
#: amplified two-thousand-fold into the anchor. The axis would still be returned
#: with confident-looking digits, and it would point nowhere in particular.
MIN_RECOVERABLE_ANGLE: Final = 1e-3

#: Above this angle the direction is lost instead: the skew-symmetric part of a
#: rotation is ``2 sin(angle)`` times the axis, and at ``pi`` it vanishes.
MAX_RECOVERABLE_ANGLE: Final = math.pi - 1e-3

#: The largest turntable step this module will accept. Beyond a quarter turn two
#: captures of a solid object share almost no surface, and a correspondence
#: search over what remains finds neighbours rather than the same points.
MAX_STEP_RADIANS: Final = math.pi / 2.0

#: How near a whole number of revolutions a sweep must land before its loop is
#: treated as closed, in radians. Missing it is not an error - it means the
#: captures do not form a revolution, and no closure figure is invented for them.
DEFAULT_CLOSURE_TOLERANCE: Final = 1e-3

#: Below this share of surviving correspondences a fit is flagged. A residual
#: computed over a tenth of the points is a statement about that tenth.
LOW_OVERLAP_FRACTION: Final = 0.5

#: Caveats every registration from this module inherits, whatever the caller
#: states. Prepended to a caller's own by :func:`register_turntable`, so no call
#: site can produce a result that has dropped one.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "registration recovers shape, not size: every cloud carries the same unknown depth scale, "
    "so a registered scan is correctly shaped and of unknown size. No residual, translation or "
    "distance in this result is a length in millimetres or in any other unit",
    "every residual here measures whether the frames agree with each other, not whether they "
    "agree with the object; a wrong depth scale, an uncorrected lens distortion or a wrong "
    "principal point moves every frame the same way and leaves all of these numbers small",
    "the spindle was estimated from the point data alone; no mechanical reference for the "
    "turntable's axis has been measured or compared against it anywhere in this repository",
    "a loop-closure residual near zero says the poses are self-consistent around one "
    "revolution. It does not say the object was reconstructed correctly",
    "a loop that closes is not a loop that is right. Where every frame really is a rigid "
    "transform of one shape and the overlap is complete, consecutive fits telescope and the "
    "chain composes to the identity whatever those transforms were - so a small closure says "
    "the fits succeeded, not that the poses are the ones the object took. What the closure "
    "detects is a fit that could not be made: partial overlap, noise, a frame whose shape does "
    "not match its neighbours, or an alignment that settled in the wrong place",
    "point-to-point matching has no notion of surface: it minimises distance between nearest "
    "samples, so two clouds sliding along a smooth or symmetric shape can register with a "
    "small residual and the wrong pose",
)


# --- refusals -----------------------------------------------------------------


class RegistrationNotAllowed(ScannerError):
    """An input was offered that this module refuses to register.

    Raised before any matching is attempted, so a refused sweep costs nothing
    and leaves no half-built pose chain behind it. A bad pose is worse than no
    pose: it misplaces every point it touches and the result still looks like a
    scan.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class CloudUnusable(RegistrationNotAllowed):
    """A point cloud cannot take part in a match.

    Empty, too short to constrain a rotation, past the point ceiling, holding a
    coordinate that is not three finite floats, or collapsed to a single
    location so that it has no extent for any relative bound to be a fraction of.
    """


class AngleStepUnusable(RegistrationNotAllowed):
    """The stated turntable step is not one a spindle can be recovered from.

    Non-finite, non-positive, below :data:`MIN_RECOVERABLE_ANGLE`, or past
    :data:`MAX_STEP_RADIANS`.
    """


class IcpParametersUnusable(RegistrationNotAllowed):
    """An iteration bound, tolerance or grid resolution is outside its range."""


class CorrespondencesInsufficient(RegistrationNotAllowed):
    """Too few point pairs survived the correspondence gate to solve anything.

    The honest reading is that the two clouds do not overlap where the current
    estimate puts them - a step larger than the surfaces share, a bad seed, or a
    gate too tight. Solving on what is left would return a transform fitted to a
    handful of points and an encouragingly small residual.
    """


class TurntableGeometryUnusable(RegistrationNotAllowed):
    """The spindle cannot be recovered from the transforms that were measured.

    A frame-to-frame rotation too near the identity or too near a half turn to
    carry a direction, or a correspondence set whose points all lie on the axis,
    which leaves the angle about it undetermined.
    """


# --- validation ---------------------------------------------------------------


def _ensure_cloud(points: Cloud, *, what: str) -> tuple[Point, ...]:
    """Normalise one cloud to validated ``(x, y, z)`` tuples, or refuse it.

    Consumed eagerly and checked as it is consumed, so an unbounded generator is
    stopped at the ceiling rather than drained into memory first.
    """
    collected: list[Point] = []
    for index, point in enumerate(points):
        if len(collected) >= MAX_CLOUD_POINTS:
            raise CloudUnusable(
                f"{what}: refusing more than {MAX_CLOUD_POINTS} points; the producer offered "
                f"at least {index + 1}, and the ceiling is never widened at run time"
            )
        values = tuple(point)
        if len(values) != 3:
            raise CloudUnusable(
                f"{what}: point {index} carries {len(values)} coordinates, not the 3 an "
                "(x, y, z) triple has"
            )
        x, y, z = values
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise CloudUnusable(
                f"{what}: point {index} is {values!r}; NaN and infinities are refused here "
                "rather than allowed to poison a centroid and, through it, every pose"
            )
        collected.append((float(x), float(y), float(z)))
    if len(collected) < MIN_CLOUD_POINTS:
        raise CloudUnusable(
            f"{what}: {len(collected)} points cannot constrain a rigid transform; at least "
            f"{MIN_CLOUD_POINTS} are needed"
        )
    return tuple(collected)


def _ensure_clouds(clouds: Sequence[Cloud]) -> tuple[tuple[Point, ...], ...]:
    """Validate a whole sweep before a single correspondence is looked up."""
    if not MIN_CLOUDS <= len(clouds) <= MAX_CLOUDS:
        raise CloudUnusable(
            f"a turntable sweep of {len(clouds)} clouds is outside [{MIN_CLOUDS}, "
            f"{MAX_CLOUDS}]; one cloud is not a sweep and the ceiling is a full revolution "
            "at half-degree steps"
        )
    return tuple(_ensure_cloud(cloud, what=f"cloud {index}") for index, cloud in enumerate(clouds))


def ensure_angle_step(angle_step: float) -> float:
    """Refuse a turntable step no spindle can be recovered from, saying which way."""
    if not math.isfinite(angle_step):
        raise AngleStepUnusable(f"angle step {angle_step!r} is not finite")
    if angle_step <= 0.0:
        raise AngleStepUnusable(
            f"angle step {angle_step!r} is not positive; the direction of rotation is carried "
            "by the sign of the recovered axis, not by a negative step"
        )
    if angle_step < MIN_RECOVERABLE_ANGLE:
        raise AngleStepUnusable(
            f"angle step {angle_step!r} rad is below {MIN_RECOVERABLE_ANGLE}; the spindle "
            "anchor is recovered through cot(angle / 2), so a step this small amplifies any "
            "translation error into an axis that is confidently placed and wrong"
        )
    if angle_step > MAX_STEP_RADIANS:
        raise AngleStepUnusable(
            f"angle step {angle_step!r} rad is past {MAX_STEP_RADIANS}; beyond a quarter turn "
            "two captures of a solid object share almost no surface, and matching what is "
            "left finds neighbouring points rather than the same points"
        )
    return angle_step


def _ensure_bound(value: float | None, *, name: str) -> float | None:
    """Refuse a grid or gate resolution that is not a usable positive size."""
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0.0:
        raise IcpParametersUnusable(
            f"{name} is {value!r}; it must be a finite positive size, or None to be derived "
            "from the target cloud's own extent"
        )
    return value


def _ensure_icp_parameters(
    *,
    max_iterations: int,
    tolerance: float,
    voxel_size: float | None,
    max_correspondence_distance: float | None,
    keep_fraction: float,
) -> None:
    """Check every knob before any point is touched."""
    if not 1 <= max_iterations <= MAX_ICP_ITERATIONS:
        raise IcpParametersUnusable(
            f"max_iterations {max_iterations} is outside [1, {MAX_ICP_ITERATIONS}]"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise IcpParametersUnusable(
            f"tolerance {tolerance!r} is not a finite non-negative fraction of the target "
            "cloud's extent"
        )
    if not math.isfinite(keep_fraction) or not 0.0 < keep_fraction <= 1.0:
        raise IcpParametersUnusable(
            f"keep_fraction {keep_fraction!r} is outside (0, 1]; it is the share of matched "
            "pairs a solve uses, so it cannot be zero and cannot exceed all of them"
        )
    _ensure_bound(voxel_size, name="voxel_size")
    _ensure_bound(max_correspondence_distance, name="max_correspondence_distance")


# --- extents and grids --------------------------------------------------------


def _bounds(points: Sequence[Point]) -> tuple[Point, Point]:
    """The axis-aligned bounding box, as ``(minimum, maximum)`` corners."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def cloud_extent(points: Sequence[Point]) -> float:
    """The bounding-box diagonal: the one scale this module measures in.

    Every relative default - the voxel side, the correspondence gate, the
    convergence threshold - is a fraction of this. That is what lets the module
    take no parameter in a physical unit, which is the only honest option while
    the depth scale's unit is unestablished.

    A cloud whose points are all at one location has no extent, and no fraction
    of zero is a usable bound, so it is refused rather than given an arbitrary
    one.
    """
    low, high = _bounds(points)
    diagonal = math.dist(low, high)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise CloudUnusable(
            f"the cloud's bounding-box diagonal is {diagonal!r}: every point is at the same "
            "location, so it has no extent for a relative bound to be a fraction of"
        )
    return diagonal


def _ensure_grid_affordable(points: Sequence[Point], cell: float, *, what: str) -> None:
    """Refuse a cell size whose grid would span an unreasonable number of cells.

    A cell small enough that a span overflows to infinity is refused here rather
    than allowed to reach ``math.floor``, which would raise a bare
    ``OverflowError`` - an escape from this module's contract that every refused
    input leaves by a :class:`RegistrationNotAllowed` carrying a
    :class:`~scanner3d.errors.DoctorStatus`. Subnormal cells reach that point:
    ``1.0 / 1e-320`` is ``inf``, and ``1e-320`` passes a finite-and-positive
    check.
    """
    low, high = _bounds(points)
    ratios = [(high[i] - low[i]) / cell for i in range(3)]
    if not all(math.isfinite(ratio) for ratio in ratios):
        raise IcpParametersUnusable(
            f"{what}: a cell of {cell!r} is so small that the number of cells across this "
            "cloud overflows to infinity, which is past any ceiling by definition"
        )
    spans = [math.floor(ratio) + 1 for ratio in ratios]
    total = spans[0] * spans[1] * spans[2]
    if total > MAX_GRID_CELLS:
        raise IcpParametersUnusable(
            f"{what}: a cell of {cell!r} spans {total} cells across this cloud, past the "
            f"{MAX_GRID_CELLS} ceiling. A grid that fine puts every point in a cell of its "
            "own: it reduces nothing and manufactures millions of keys"
        )


def _cell_key(point: Point, cell: float) -> tuple[int, int, int]:
    return (
        math.floor(point[0] / cell),
        math.floor(point[1] / cell),
        math.floor(point[2] / cell),
    )


def voxel_downsample(points: Sequence[Point], cell: float) -> tuple[Point, ...]:
    """One point per occupied cell of a uniform grid, at the cell's centroid.

    The centroid rather than an arbitrary representative: it averages away some
    of the per-sample noise instead of picking one noisy sample to stand for its
    neighbours. Cell occupancy is small, so the sums are accumulated plainly -
    unlike :func:`scanner3d.geometry.centroid`, which sums a whole cloud and
    needs ``fsum`` to keep its low bits.

    Output order is the order cells were first seen, which is the input order,
    so downsampling one cloud twice gives the same points in the same sequence.
    """
    _ensure_grid_affordable(points, cell, what="voxel_size")
    buckets: dict[tuple[int, int, int], list[float]] = {}
    for x, y, z in points:
        key = _cell_key((x, y, z), cell)
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = [x, y, z, 1.0]
        else:
            bucket[0] += x
            bucket[1] += y
            bucket[2] += z
            bucket[3] += 1.0
    return tuple((sx / n, sy / n, sz / n) for sx, sy, sz, n in buckets.values())


#: The 27 cells a nearest neighbour can be in. See :class:`_VoxelIndex`.
_NEIGHBOUR_CELLS: Final = tuple(itertools.product((-1, 0, 1), repeat=3))


class _VoxelIndex:
    """Nearest-neighbour lookup over a uniform voxel hash, bounded to 27 cells.

    The cell side is *exactly* the widest correspondence gate the fit will use,
    and that is what makes the bound sound rather than a heuristic: if a target
    point lies within one cell width of a query, then along each axis their cell
    indices differ by at most one, so it is in the 3x3x3 block around the
    query's own cell. A point outside that block is further away than the gate
    and would be rejected anyway. No ring has to be expanded and no candidate is
    missed.

    The cost of a query is the number of target points inside that block, and
    that is a *constant fraction* of the cloud, not a constant - so a match
    remains quadratic in the point count and this index is a constant-factor
    saving, not an asymptotic one. How large a factor depends entirely on how
    the points sit inside their own bounding box, and it is much smaller than it
    looks. Measured at the default gate of an eighth of the extent: 21 per cent
    of the cloud per query on the synthetic test curve, 32 per cent on one
    occluded synthetic frame, and **91 per cent on the real ``base_4``
    capture** - which is to say that on real data this is very nearly the linear
    scan it replaces. The reason is the one
    :data:`DEFAULT_VOXEL_FRACTION` already gives: a real frame's extent is set
    by its furthest stray return, so the object occupies a small corner of the
    box and every surviving point ends up within one cell of every other.

    What keeps a real frame affordable is therefore
    :data:`DOWNSAMPLE_THRESHOLD` reducing 89000 points to a few hundred before
    the index is ever built - not the index. Lowering the gate, or downsampling
    harder, is what actually buys anything here.
    """

    __slots__ = ("_cells", "_gate")

    def __init__(self, points: Sequence[Point], *, gate: float) -> None:
        _ensure_grid_affordable(points, gate, what="max_correspondence_distance")
        self._gate = gate
        cells: dict[tuple[int, int, int], list[Point]] = {}
        for point in points:
            cells.setdefault(_cell_key(point, gate), []).append(point)
        self._cells = cells

    @property
    def gate(self) -> float:
        """The correspondence gate, which is also the cell side."""
        return self._gate

    def nearest(self, query: Point) -> Point | None:
        """The closest indexed point strictly inside the gate, or ``None``.

        ``None`` is a rejection and is counted as one by the caller: it means
        this query has no partner near enough to be believed, not that the
        match failed to run.
        """
        base = _cell_key(query, self._gate)
        qx, qy, qz = query
        best: Point | None = None
        best_squared = self._gate * self._gate
        for dx, dy, dz in _NEIGHBOUR_CELLS:
            bucket = self._cells.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if bucket is None:
                continue
            for point in bucket:
                ex = point[0] - qx
                ey = point[1] - qy
                ez = point[2] - qz
                squared = ex * ex + ey * ey + ez * ez
                if squared < best_squared:
                    best_squared = squared
                    best = point
        return best


# --- the screw axis of one transform ------------------------------------------


@dataclass(frozen=True, slots=True)
class ScrewAxis:
    """One rigid transform re-expressed as a rotation about a line in space.

    Every rigid transform is a rotation about some line plus a slide along it.
    A turntable step should have no slide, so :attr:`axial_slip` is not a
    formality: it is the diagnostic that says whether the transform that was
    measured is the kind of transform a turntable can produce.
    """

    direction: Point
    through: Point
    angle: float
    axial_slip: float

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": list(self.direction),
            "through": list(self.through),
            "angle_radians": self.angle,
            "axial_slip": self.axial_slip,
        }


def screw_axis_of(transform: Rigid) -> ScrewAxis:
    """Decompose ``transform`` into the line it turns about, and by how much.

    The direction comes from the skew-symmetric part of the rotation, which is
    ``2 sin(angle)`` times the unit axis, so its sign follows the direction of
    rotation: a turntable running the other way returns the opposite direction
    with the same positive angle, and that is the intended behaviour.

    :attr:`ScrewAxis.through` is the point of the line closest to the origin,
    which makes it a canonical anchor that two estimates of the same line can be
    compared at. It is recovered in closed form from the translation's component
    perpendicular to the axis::

        through = (perpendicular + cot(angle / 2) * (axis x perpendicular)) / 2

    Refused, rather than approximated, when the angle is too near zero or a half
    turn: near zero the ``cot`` term grows without bound and turns translation
    error into a confidently placed and entirely wrong line, and near a half
    turn the skew part vanishes and the direction itself is lost.
    """
    angle = transform.rotation_angle
    if not MIN_RECOVERABLE_ANGLE <= angle <= MAX_RECOVERABLE_ANGLE:
        raise TurntableGeometryUnusable(
            f"a rotation of {angle!r} rad is outside [{MIN_RECOVERABLE_ANGLE}, "
            f"{MAX_RECOVERABLE_ANGLE}] and carries no recoverable axis: too near the identity "
            "and the anchor is amplified noise, too near a half turn and the direction vanishes"
        )
    r = transform.rotation
    skew = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
    direction = normalize(skew)
    translation = transform.translation
    slip = dot(translation, direction)
    perpendicular = subtract(translation, scaled(direction, slip))
    half_cotangent = 1.0 / math.tan(angle / 2.0)
    turned = cross(direction, perpendicular)
    through = scaled(
        (
            perpendicular[0] + half_cotangent * turned[0],
            perpendicular[1] + half_cotangent * turned[1],
            perpendicular[2] + half_cotangent * turned[2],
        ),
        0.5,
    )
    return ScrewAxis(direction=direction, through=through, angle=angle, axial_slip=slip)


def optimal_angle_about_axis(
    source: Sequence[Point],
    target: Sequence[Point],
    *,
    direction: Point,
    through: Point,
) -> float:
    """The angle about a *fixed* line that best carries ``source`` onto ``target``.

    Closed form, not a search. Writing the residual of a Rodrigues rotation about
    a known axis and dropping every term independent of the angle leaves
    ``A cos(angle) + B sin(angle)`` to be maximised, so the least-squares
    optimum is ``atan2(B, A)`` with

        ``A = sum(v . u_perpendicular)``   ``B = sum(v . (axis x u))``

    where ``u`` and ``v`` are the source and target points measured from a point
    on the line. This is the turntable's one degree of freedom solved exactly,
    and it is what makes the refinement in :func:`register_turntable` better
    conditioned than a general alignment: no amount of noise can push the answer
    off the spindle, because the answer has nowhere off the spindle to go.

    The result is signed and lies in ``(-pi, pi]``. A negative angle means the
    correspondences say the object turned the other way about ``direction``,
    which is information, not an error, and is reported rather than folded away.
    """
    if len(source) != len(target):
        raise CorrespondencesInsufficient(
            f"{len(source)} source points against {len(target)} target points: a "
            "correspondence set must pair one to one"
        )
    if not source:
        raise CorrespondencesInsufficient("no correspondences: the angle is undetermined")
    axis = normalize(direction)
    cosine_term = 0.0
    sine_term = 0.0
    for p, q in zip(source, target, strict=True):
        u = subtract(p, through)
        v = subtract(q, through)
        perpendicular = subtract(u, scaled(axis, dot(axis, u)))
        cosine_term += dot(v, perpendicular)
        sine_term += dot(v, cross(axis, u))
    if math.hypot(cosine_term, sine_term) < MIN_VECTOR_NORM:
        raise TurntableGeometryUnusable(
            "the correspondences leave the angle about this axis undetermined - every point "
            "lies on the axis itself, where rotation moves nothing"
        )
    return math.atan2(sine_term, cosine_term)


# --- iterative closest point --------------------------------------------------


@dataclass(frozen=True, slots=True)
class IcpResult:
    """What one alignment did, and on how much of the data it did it.

    :attr:`rmse` is meaningless without :attr:`correspondences`,
    :attr:`trimmed` and :attr:`rejected` beside it: a fit that keeps one point
    in ten reports the residual of that tenth, and the number is small precisely
    because the rest were dropped. The four are one reading.

    The two ways of not being used are counted separately, because they say
    different things about the data. :attr:`rejected` is a source point that
    found no target inside the gate at all - no overlap here. :attr:`trimmed`
    is one that found a partner and was cut by the quantile as too far to
    believe relative to the rest.

    :attr:`converged` is recorded, never inferred. It is ``True`` only when the
    residual stopped improving by more than :attr:`convergence_threshold`
    *within* :attr:`iterations`; a run that used its whole iteration budget and
    was still moving reports ``False``, because finishing is not converging.

    :attr:`trim_distance` is the distance of the furthest pair the final solve
    actually used, so a reader can say exactly how near a pair had to be to
    count towards :attr:`rmse`.

    :attr:`resolved_backend` names which :mod:`scanner3d.compute` backend
    actually ran the correspondence search - ``"python"``, ``"numpy"`` or
    ``"scipy"``, never the bare request (``"auto"``), for the same reason
    :attr:`scanner3d.compute.BackendSelection.resolved` never is: a result
    that only reports what was ASKED for, not what RAN, cannot be used to
    interpret its own timing. Always ``"python"`` for a fit produced by
    :func:`register_turntable`, which has no ``backend`` parameter of its
    own - see the module docstring's "Selectable compute backend" section.
    """

    transform: Rigid
    rmse: float
    iterations: int
    converged: bool
    correspondences: int
    trimmed: int
    rejected: int
    source_points: int
    target_points: int
    voxel_size: float | None
    max_correspondence_distance: float
    keep_fraction: float
    trim_distance: float
    convergence_threshold: float
    resolved_backend: str

    @property
    def offered(self) -> int:
        """Every source point the fit put to the gate."""
        return self.correspondences + self.trimmed + self.rejected

    @property
    def matched_fraction(self) -> float:
        """The share of source points that found a partner inside the gate.

        Trimmed pairs count as matched here: they had a plausible partner, and
        this fraction is a statement about overlap, not about the trim.
        """
        return (self.correspondences + self.trimmed) / self.offered if self.offered else 0.0

    @property
    def used_fraction(self) -> float:
        """The share of source points the final solve actually used."""
        return self.correspondences / self.offered if self.offered else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transform": self.transform.as_dict(),
            "rmse": self.rmse,
            "rmse_unit": "input coordinate units, whose physical meaning is not established",
            "iterations": self.iterations,
            "converged": self.converged,
            "correspondences": self.correspondences,
            "trimmed": self.trimmed,
            "rejected": self.rejected,
            "matched_fraction": self.matched_fraction,
            "used_fraction": self.used_fraction,
            "source_points": self.source_points,
            "target_points": self.target_points,
            "voxel_size": self.voxel_size,
            "max_correspondence_distance": self.max_correspondence_distance,
            "keep_fraction": self.keep_fraction,
            "trim_distance": self.trim_distance,
            "convergence_threshold": self.convergence_threshold,
            "resolved_backend": self.resolved_backend,
        }


@dataclass(frozen=True, slots=True)
class _Prepared:
    """The sampled clouds, the index and the resolved bounds one fit runs against."""

    source: tuple[Point, ...]
    target: tuple[Point, ...]
    index: _VoxelIndex | None
    voxel_size: float | None
    gate: float
    threshold: float
    keep_fraction: float


def _prepare(
    source: tuple[Point, ...],
    target: tuple[Point, ...],
    *,
    tolerance: float,
    voxel_size: float | None,
    max_correspondence_distance: float | None,
    keep_fraction: float,
    build_index: bool = True,
) -> _Prepared:
    """Resolve every relative bound against the target's extent, then sample and index.

    The target's extent and not the source's: the gate and the threshold are
    both statements about where a partner is expected to be found, and that is
    the target's geometry.

    ``build_index`` is ``True`` by default - every existing caller
    (:func:`_fit_pair`, and :func:`icp` on its unaccelerated path) gets the
    exact :class:`_VoxelIndex` it always got. :func:`icp` passes ``False``
    only when a live :mod:`scanner3d.compute` backend was selected, because a
    backend "never grids at all" (see that module's docstring) and building
    an unused :class:`_VoxelIndex` could refuse a gate the backend would have
    handled fine - see :func:`_ensure_grid_affordable`. ``prepared.index`` is
    then ``None``, and the caller must not read it.
    """
    extent = cloud_extent(target)
    gate = (
        max_correspondence_distance
        if max_correspondence_distance is not None
        else extent * DEFAULT_CORRESPONDENCE_FRACTION
    )
    resolved_voxel = voxel_size
    if resolved_voxel is None and max(len(source), len(target)) > DOWNSAMPLE_THRESHOLD:
        resolved_voxel = extent * DEFAULT_VOXEL_FRACTION
    sampled_source = source if resolved_voxel is None else voxel_downsample(source, resolved_voxel)
    sampled_target = target if resolved_voxel is None else voxel_downsample(target, resolved_voxel)
    if len(sampled_source) < MIN_CLOUD_POINTS or len(sampled_target) < MIN_CLOUD_POINTS:
        raise CloudUnusable(
            f"downsampling at a voxel side of {resolved_voxel!r} left "
            f"{len(sampled_source)} source and {len(sampled_target)} target points, fewer "
            f"than the {MIN_CLOUD_POINTS} a rigid transform needs"
        )
    return _Prepared(
        source=tuple(sampled_source),
        target=tuple(sampled_target),
        index=_VoxelIndex(sampled_target, gate=gate) if build_index else None,
        voxel_size=resolved_voxel,
        gate=gate,
        threshold=tolerance * extent,
        keep_fraction=keep_fraction,
    )


def _correspond(
    source: Sequence[Point], transform: Rigid, index: _VoxelIndex
) -> tuple[list[tuple[float, Point, Point]], int]:
    """Pair each source point with its nearest target, dropping those past the gate.

    Returns ``(distance, original_source_point, target_point)`` triples and the
    number of source points that found nothing at all. The *original* source
    point is carried, not the transformed one, so each solver fits an absolute
    transform from the input frame rather than an increment that has to be
    composed onto the last guess. The distance rides along because the trim
    needs it and recomputing it would be the same arithmetic twice.
    """
    pairs: list[tuple[float, Point, Point]] = []
    rejected = 0
    for original, moved in zip(source, transform.apply_all(source), strict=True):
        nearest = index.nearest(moved)
        if nearest is None:
            rejected += 1
            continue
        pairs.append((math.dist(moved, nearest), original, nearest))
    return pairs, rejected


#: What :func:`_correspond` (bound to a :class:`_VoxelIndex`) and
#: :func:`_correspond_via_backend` (bound to a live
#: :class:`~scanner3d.compute.ComputeBackend`) both are, once bound: given the
#: current source cloud and transform estimate, the matched ``(distance,
#: original_source_point, target_point)`` triples and the reject count.
#: :func:`_iterate` calls one of the two through this shape without knowing or
#: caring which - see :func:`icp` and :func:`_fit_pair`. Matches
#: :mod:`scanner3d.icp_plane`'s identically-named, identically-shaped seam.
_Correspond = Callable[[Sequence[Point], Rigid], tuple[list[tuple[float, Point, Point]], int]]


def _correspond_via_backend(
    source: Sequence[Point],
    transform: Rigid,
    backend: ComputeBackend,
    target: Sequence[Point],
    *,
    gate: float,
) -> tuple[list[tuple[float, Point, Point]], int]:
    """The backend-driven equivalent of :func:`_correspond`: same inputs, same
    outputs, same gate semantics -
    :meth:`~scanner3d.compute.ComputeBackend.nearest_neighbors` uses the
    identical strict "closer than the gate" rule as
    :meth:`_VoxelIndex.nearest`, verified in :mod:`scanner3d.compute`'s own
    test suite. Matches :func:`scanner3d.icp_plane._correspond_via_backend`.
    """
    pairs: list[tuple[float, Point, Point]] = []
    rejected = 0
    moved = backend.apply_transform(transform, source)
    hits = backend.nearest_neighbors(moved, target, gate=gate)
    for original, hit in zip(source, hits, strict=True):
        if hit is None:
            rejected += 1
            continue
        distance, nearest = hit
        pairs.append((distance, original, nearest))
    return pairs, rejected


def _resolve_backend(backend: str | ComputeBackend) -> tuple[str, ComputeBackend | None]:
    """``(resolved name, live backend or None)``. ``None`` means "run the
    original, unaccelerated code path exactly as written before ``backend``
    existed as a parameter" - see the module docstring's "Selectable compute
    backend" section.

    ``None`` is returned for the string ``"python"`` (the default) and for
    ``"auto"`` when it resolves to python, so "python" means the SAME thing -
    the untouched reference implementation - regardless of how it was
    reached, rather than a second, merely-equivalent code path through
    :mod:`scanner3d.compute`. A caller who hands a live
    :class:`~scanner3d.compute.ComputeBackend` INSTANCE is opting into the
    backend-driven path deliberately, even if that instance's own name is
    ``"python"``; only the bare string is treated as "use the legacy path".
    Matches :func:`scanner3d.icp_plane._resolve_backend` exactly, duplicated
    rather than imported for the same reason every other private helper in
    this codebase is duplicated across sibling modules - see that function's
    own docstring.

    :mod:`scanner3d.compute` is imported here, not at module scope - see this
    module's own docstring for why that particular direction is a genuine
    circular import and not merely an avoidable one.
    """
    from scanner3d.compute import PYTHON_BACKEND
    from scanner3d.compute import select_backend as _select_compute_backend

    if isinstance(backend, str):
        if backend == PYTHON_BACKEND:
            return PYTHON_BACKEND, None
        selection = _select_compute_backend(backend)
        if selection.resolved == PYTHON_BACKEND:
            return PYTHON_BACKEND, None
        return selection.resolved, selection.backend
    return backend.name, backend


def kept_count(matched: int, keep_fraction: float) -> int:
    """How many of ``matched`` pairs a fit solves on, closest first.

    Rounded up, and never below :data:`MIN_CORRESPONDENCES`, so trimming can
    thin a correspondence set but can never be the thing that empties it. A set
    already too small to solve is refused by the caller instead - which is the
    honest report, because "the clouds do not overlap here" and "the trim was
    aggressive" are different failures and must not be reported as one.
    """
    return max(MIN_CORRESPONDENCES, math.ceil(matched * keep_fraction))


#: A solver maps matched pairs to an absolute transform, and to the signed angle
#: about a constrained axis when there is one. ``None`` is not "no angle
#: happened": it is "this solver had six degrees of freedom and no single angle
#: describes what it did".
_Solver = Callable[[Sequence[Point], Sequence[Point]], tuple[Rigid, float | None]]


@dataclass(frozen=True, slots=True)
class _Fit:
    """The raw outcome of the iteration loop, before it is dressed as a record."""

    transform: Rigid
    angle: float | None
    rmse: float
    iterations: int
    converged: bool
    correspondences: int
    trimmed: int
    rejected: int
    trim_distance: float


def _iterate(
    source: Sequence[Point],
    correspond: _Correspond,
    *,
    initial: Rigid,
    solve: _Solver,
    max_iterations: int,
    threshold: float,
    keep_fraction: float,
    gate: float,
) -> _Fit:
    """Match, trim, solve, repeat - and record whether it actually settled.

    ``correspond`` is either :func:`_correspond` (bound to a
    :class:`_VoxelIndex`) or :func:`_correspond_via_backend` (bound to a
    :class:`~scanner3d.compute.ComputeBackend`) - chosen once by the caller
    (:func:`icp` or :func:`_fit_pair`) before this function ever runs.
    Nothing below this line changed to add that seam: the trim, the solve
    and the convergence test are the same code that ran before ``backend``
    existed anywhere in this module. ``gate`` is carried alongside purely
    for the refusal message and the pre-first-iteration default of
    ``trim_distance`` - :func:`_correspond_via_backend` has no ``.gate``
    attribute to read it back from the way :class:`_VoxelIndex` does.

    Convergence is the residual ceasing to *improve*. The first iteration has
    nothing to compare against, so a single-iteration budget always reports
    ``converged=False``: one pass fits a transform, it does not establish that a
    second pass would agree with it.

    The trim is what makes the residual mean something on a turntable. Sorting
    the pairs by distance and keeping the closest share puts every pair whose
    true partner was never captured at the wrong end of the list, which is
    precisely where a quantile can cut them. Both counts survive into the
    result, because "found no partner at all" and "found one and it was cut"
    are different facts about the overlap.
    """
    transform = initial
    angle: float | None = None
    rmse = math.inf
    previous: float | None = None
    correspondences = 0
    trimmed = 0
    rejected = 0
    trim_distance = gate
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        pairs, rejected = correspond(source, transform)
        if len(pairs) < MIN_CORRESPONDENCES:
            raise CorrespondencesInsufficient(
                f"only {len(pairs)} of {len(pairs) + rejected} source points found a partner "
                f"inside a correspondence gate of {gate!r} at iteration {iterations}; "
                "the clouds do not overlap where this estimate puts them"
            )
        pairs.sort(key=lambda pair: pair[0])
        keep = min(len(pairs), kept_count(len(pairs), keep_fraction))
        correspondences = keep
        trimmed = len(pairs) - keep
        trim_distance = pairs[keep - 1][0]
        matched_source = [pair[1] for pair in pairs[:keep]]
        matched_target = [pair[2] for pair in pairs[:keep]]
        transform, angle = solve(matched_source, matched_target)
        rmse = alignment_rmse(matched_source, matched_target, transform)
        if previous is not None and abs(previous - rmse) <= threshold:
            converged = True
            break
        previous = rmse
    return _Fit(
        transform=transform,
        angle=angle,
        rmse=rmse,
        iterations=iterations,
        converged=converged,
        correspondences=correspondences,
        trimmed=trimmed,
        rejected=rejected,
        trim_distance=trim_distance,
    )


def _rigid_solver(source: Sequence[Point], target: Sequence[Point]) -> tuple[Rigid, float | None]:
    """Six degrees of freedom, in closed form: Horn's method on the matched pairs."""
    return rigid_from_correspondences(source, target), None


def _angle_solver(direction: Point, through: Point) -> _Solver:
    """One degree of freedom: the best angle about a line held fixed."""

    def solve(source: Sequence[Point], target: Sequence[Point]) -> tuple[Rigid, float | None]:
        angle = optimal_angle_about_axis(source, target, direction=direction, through=through)
        return rotation_about_axis(direction, angle, through=through), angle

    return solve


def _as_result(fit: _Fit, prepared: _Prepared, *, resolved_backend: str = "python") -> IcpResult:
    return IcpResult(
        transform=fit.transform,
        rmse=fit.rmse,
        iterations=fit.iterations,
        converged=fit.converged,
        correspondences=fit.correspondences,
        trimmed=fit.trimmed,
        rejected=fit.rejected,
        source_points=len(prepared.source),
        target_points=len(prepared.target),
        voxel_size=prepared.voxel_size,
        max_correspondence_distance=prepared.gate,
        keep_fraction=prepared.keep_fraction,
        trim_distance=fit.trim_distance,
        convergence_threshold=prepared.threshold,
        resolved_backend=resolved_backend,
    )


def icp(
    source: Cloud,
    target: Cloud,
    *,
    initial: Rigid = IDENTITY_POSE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    voxel_size: float | None = None,
    max_correspondence_distance: float | None = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    backend: str | ComputeBackend = "python",
) -> IcpResult:
    """Point-to-point ICP: the transform carrying ``source`` onto ``target``.

    Each iteration pairs every source point with its nearest target through a
    uniform voxel hash whose cell is the widest correspondence gate, so a query
    touches 27 cells and provably cannot miss a candidate. That bound is a
    correctness argument, not a performance one: the block still holds a
    constant fraction of the cloud, so a match is quadratic in the point count
    either way, and on the real ``base_4`` capture the fraction is 91 per cent.
    What makes a full depth frame affordable is :data:`DOWNSAMPLE_THRESHOLD`
    cutting it to a few hundred points first. See :class:`_VoxelIndex`.

    Pairs further apart than the current gate are dropped and counted, and the
    count is part of the answer - see :class:`IcpResult`. If fewer than
    :data:`MIN_CORRESPONDENCES` survive, the call raises rather than fitting a
    transform to a handful of points and reporting the flattering residual that
    results.

    Of the pairs that survive the gate, only the closest ``keep_fraction`` are
    solved on. That trim is what keeps a partial overlap - the normal case on a
    turntable, where every step turns some surface out of view and some into it
    - from biasing the answer through pairs whose true partner was never
    captured. See :data:`DEFAULT_KEEP_FRACTION` for what that costs and buys.

    ``tolerance``, ``voxel_size`` and ``max_correspondence_distance`` are all
    *fractions of the target cloud's own extent* when left to their defaults,
    and the resolved absolute values are recorded in the result. Passing an
    absolute value for the latter two is allowed and is how a caller who knows
    their data overrides the derivation; there is no unit to state either way.

    ``initial`` is where the search starts, and on a turntable it is the whole
    game: a rotation of the nominal step about an estimated spindle puts the
    source within a fraction of a step of the truth, where the fit has one
    obvious minimum instead of several plausible ones.

    ``backend`` selects what runs the correspondence search - see
    :mod:`scanner3d.compute` and the module docstring's "Selectable compute
    backend" section. The default, ``"python"``, is the unmodified code path
    this function has always run; :attr:`IcpResult.resolved_backend` names
    what actually ran, which matters most for ``"auto"``.
    """
    source_points = _ensure_cloud(source, what="ICP source")
    target_points = _ensure_cloud(target, what="ICP target")
    _ensure_icp_parameters(
        max_iterations=max_iterations,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
    )
    resolved_backend_name, compute_backend = _resolve_backend(backend)
    prepared = _prepare(
        source_points,
        target_points,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
        build_index=compute_backend is None,
    )

    correspond: _Correspond
    if compute_backend is None:
        prepared_index = prepared.index
        assert prepared_index is not None  # build_index=True above guarantees this

        def correspond(
            src: Sequence[Point], tr: Rigid
        ) -> tuple[list[tuple[float, Point, Point]], int]:
            return _correspond(src, tr, prepared_index)
    else:
        prepared_target = prepared.target
        prepared_gate = prepared.gate
        backend_object = compute_backend

        def correspond(
            src: Sequence[Point], tr: Rigid
        ) -> tuple[list[tuple[float, Point, Point]], int]:
            return _correspond_via_backend(
                src, tr, backend_object, prepared_target, gate=prepared_gate
            )

    fit = _iterate(
        prepared.source,
        correspond,
        initial=initial,
        solve=_rigid_solver,
        max_iterations=max_iterations,
        threshold=prepared.threshold,
        keep_fraction=keep_fraction,
        gate=prepared.gate,
    )
    return _as_result(fit, prepared, resolved_backend=resolved_backend_name)


# --- the spindle --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurntableAxis:
    """The estimated spindle, and every number that says how much to trust it.

    The residual fields are ``None`` on an axis that was *declared* rather than
    estimated - see :meth:`declared`. ``None`` there means "not measured", which
    is a different statement from a residual of zero, and the two are never
    allowed to look alike.

    :attr:`pair_iterations` is how much work each frame-to-frame fit needed. It
    is cost, not accuracy, and it is recorded because it is the only visible
    evidence that seeding each pair from the one before it did anything - see
    :func:`estimate_turntable_axis`.

    :attr:`through` is the point of the line closest to the camera origin. Any
    point on the line would describe the same line; pinning it to the closest
    one makes two estimates directly comparable.
    """

    direction: Point
    through: Point
    angle_step_radians: float | None
    angle_residual_radians: float | None
    direction_spread_radians: float | None
    through_spread: float | None
    axial_slip: float | None
    fit_rmse: float | None
    pair_count: int
    pair_iterations: tuple[int, ...]
    passes: int
    converged: bool | None
    limitations: tuple[str, ...] = ()

    @classmethod
    def declared(cls, direction: Point, through: Point, *, basis: str) -> TurntableAxis:
        """An axis the caller asserts, carrying no residual because none was measured.

        ``basis`` names what the assertion rests on and travels in the
        limitations, so a result registered against a declared spindle says so
        wherever it is read.
        """
        unit = normalize(direction)
        anchor = subtract(through, scaled(unit, dot(through, unit)))
        return cls(
            direction=unit,
            through=anchor,
            angle_step_radians=None,
            angle_residual_radians=None,
            direction_spread_radians=None,
            through_spread=None,
            axial_slip=None,
            fit_rmse=None,
            pair_count=0,
            pair_iterations=(),
            passes=0,
            converged=None,
            limitations=(
                f"this spindle was declared by the caller ({basis}), not estimated from the "
                "clouds; no residual in this result supports the axis itself, and an axis "
                "that is wrong bends every pose that was fitted against it",
            ),
        )

    @property
    def estimated(self) -> bool:
        """True when this axis came from data rather than from an assertion."""
        return self.pair_count > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "direction": list(self.direction),
            "through": list(self.through),
            "through_is": "the point of the spindle closest to the camera origin",
            "estimated": self.estimated,
            "pair_count": self.pair_count,
            "pair_iterations": list(self.pair_iterations),
            "passes": self.passes,
            "converged": self.converged,
            "angle_step_radians": self.angle_step_radians,
            "residuals": {
                "angle_residual_radians": self.angle_residual_radians,
                "direction_spread_radians": self.direction_spread_radians,
                "through_spread": self.through_spread,
                "axial_slip": self.axial_slip,
                "fit_rmse": self.fit_rmse,
            },
            "units": {
                "angles": "radians",
                "lengths": "input coordinate units, whose physical meaning is not established",
            },
            "limitations": list(self.limitations),
        }


def estimate_turntable_axis(
    clouds: Sequence[Cloud],
    *,
    angle_step: float,
    seed: TurntableAxis | None = None,
    passes: int = DEFAULT_AXIS_PASSES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    voxel_size: float | None = None,
    max_correspondence_distance: float | None = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
) -> TurntableAxis:
    """Recover the spindle from the sweep itself, and say how well it is pinned.

    One pass of the method, which is worth stating because its failure modes
    follow directly from it:

    1. Align each consecutive pair with unconstrained ICP. The first pair starts
       from a rotation of ``angle_step`` about the seed line if one is offered
       and from the identity otherwise; every later pair starts from the
       transform recovered for the pair before it.

       That chaining is the turntable insight applied to the bootstrap - on a
       turntable every step is the *same* transform, so the previous answer is
       an excellent guess for the next question. It buys **cost, not accuracy**,
       and the distinction is worth keeping straight: the answer it reaches is
       the same either way. What changes is how far the search has to go. On the
       partial-overlap test sweep the chained pairs needed 12, 14 and then 2
       iterations each, against 12 to 36 apiece unseeded - 38 iterations across
       the sweep rather than 164. In pure Python that is the difference between
       a bootstrap that costs a tenth of a second and one that costs four
       tenths. :attr:`TurntableAxis.pair_iterations` records it.
    2. Decompose each of those transforms into its screw axis. A turntable step
       is a rotation about the spindle, so every pair should name the same line.
    3. Take the consensus - the normalised mean direction, and the mean anchor
       projected back onto that line - and report how far the individual
       estimates strayed from it.

    ``passes`` runs that whole method more than once, each pass seeded with the
    line the pass before it found. Unlike the chaining, this one buys accuracy,
    and it is why the default is two rather than one. The first pass has to
    start from the identity, which on a coarse sweep of a partly self-occluding
    object is a poor enough guess that the pairwise fits settle in a local
    minimum; seeded with even a rough line, the second pass starts a fraction of
    a step away and lands on the answer. On the partial-overlap test sweep a
    40-degree revolution went from a direction wrong by 0.056 rad after one pass
    to floating-point exact after two, and the loop closure with it, from
    0.9 rad to 5e-15. It is not monotone, though - more passes is not reliably
    better, and the residuals below, not the pass count, are what say whether an
    estimate is any good.

    Failure modes, all of them visible in those residuals rather than hidden
    by them:

    * **The bootstrap settles in a local minimum.** The tell is a large
      ``direction_spread_radians`` or ``angle_residual_radians`` - the pairs
      disagree with each other - together with ``fit_rmse``, and often
      ``axial_slip``: a wrong fit buys residual by sliding along the axis, which
      a real turntable cannot do.
    * **Too little rotation.** A step near zero leaves the anchor recovered
      through an amplified ``cot(angle / 2)`` term;
      :data:`MIN_RECOVERABLE_ANGLE` is where that is refused outright rather
      than reported.
    * **A featureless or symmetric object.** A cylinder centred on the spindle
      matches itself at every angle. The residuals will be small and the axis
      may be anything; no internal measure can detect this, and it is stated in
      the limitations rather than tested for.
    * **One pair only.** Two clouds give one estimate, so every spread is zero
      by construction. That is flagged in the limitations, because a zero that
      means "nothing was compared" reads exactly like a zero that means
      "everything agreed".

    ``angle_step`` is *not* used to constrain the fit here - it is deliberately
    left free so that the measured :attr:`TurntableAxis.angle_step_radians` can
    be compared against what the caller claimed, which is the one check that
    catches a mis-stated step. It bounds and seeds the input instead.
    """
    ensure_angle_step(angle_step)
    if not 1 <= passes <= MAX_AXIS_PASSES:
        raise IcpParametersUnusable(f"passes {passes} is outside [1, {MAX_AXIS_PASSES}]")
    _ensure_icp_parameters(
        max_iterations=max_iterations,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
    )
    validated = _ensure_clouds(clouds)

    settings = _IcpSettings(
        max_iterations=max_iterations,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
    )
    estimate = _one_axis_pass(validated, seed, angle_step=angle_step, settings=settings, passes=1)
    for attempt in range(2, passes + 1):
        estimate = _one_axis_pass(
            validated, estimate, angle_step=angle_step, settings=settings, passes=attempt
        )
    return estimate


@dataclass(frozen=True, slots=True)
class _IcpSettings:
    """The four fit knobs, carried together so they are never partly forwarded."""

    max_iterations: int
    tolerance: float
    voxel_size: float | None
    max_correspondence_distance: float | None
    keep_fraction: float


def _one_axis_pass(
    clouds: Sequence[tuple[Point, ...]],
    seed: TurntableAxis | None,
    *,
    angle_step: float,
    settings: _IcpSettings,
    passes: int,
) -> TurntableAxis:
    """One bootstrap sweep: fit every consecutive pair, then take the consensus."""
    guess = IDENTITY_POSE
    if seed is not None:
        guess = rotation_about_axis(seed.direction, angle_step, through=seed.through)
    screws: list[ScrewAxis] = []
    fits: list[IcpResult] = []
    for index in range(len(clouds) - 1):
        fit = icp(
            clouds[index],
            clouds[index + 1],
            initial=guess,
            max_iterations=settings.max_iterations,
            tolerance=settings.tolerance,
            voxel_size=settings.voxel_size,
            max_correspondence_distance=settings.max_correspondence_distance,
            keep_fraction=settings.keep_fraction,
        )
        fits.append(fit)
        screws.append(screw_axis_of(fit.transform))
        # Every step is the same transform, so the answer to this pair is the
        # question for the next one.
        guess = fit.transform
    return _consensus_axis(screws, fits, passes=passes)


def _consensus_axis(
    screws: Sequence[ScrewAxis], fits: Sequence[IcpResult], *, passes: int
) -> TurntableAxis:
    """Combine per-pair screw axes into one line, and measure the disagreement."""
    total = (
        math.fsum(s.direction[0] for s in screws),
        math.fsum(s.direction[1] for s in screws),
        math.fsum(s.direction[2] for s in screws),
    )
    try:
        direction = normalize(total)
    except ScannerError as exc:
        raise TurntableGeometryUnusable(
            "the per-pair rotation axes cancel rather than agree, so there is no consensus "
            "spindle: the frame-to-frame fits did not find one consistent rotation"
        ) from exc

    anchor_mean = (
        math.fsum(s.through[0] for s in screws) / len(screws),
        math.fsum(s.through[1] for s in screws) / len(screws),
        math.fsum(s.through[2] for s in screws) / len(screws),
    )
    # Pin the anchor to the line's closest approach to the origin, so two
    # estimates of one line are compared at the same point on it.
    through = subtract(anchor_mean, scaled(direction, dot(anchor_mean, direction)))

    angles = [s.angle for s in screws]
    mean_angle = math.fsum(angles) / len(angles)
    spread = max(angle_between(s.direction, direction) for s in screws)
    through_spread = max(_distance_to_line(s.through, direction, through) for s in screws)

    limitations: list[str] = []
    if len(screws) == 1:
        limitations.append(
            "one frame-to-frame pair only: the direction, angle and anchor spreads compare that "
            "single estimate against itself, so they are near zero whatever the data says, and "
            "not because two estimates were found to agree"
        )
    if not all(f.converged for f in fits):
        limitations.append(
            "at least one bootstrap alignment used its whole iteration budget without the "
            "residual settling, so the axis it contributed is where the fit stopped, not "
            "where it converged"
        )
    if any(f.matched_fraction < LOW_OVERLAP_FRACTION for f in fits):
        limitations.append(
            "at least one bootstrap alignment matched under half its source points; the "
            "residual it reports describes the points that survived the gate and says nothing "
            "about the ones that did not"
        )
    return TurntableAxis(
        direction=direction,
        through=through,
        angle_step_radians=mean_angle,
        angle_residual_radians=max(abs(a - mean_angle) for a in angles),
        direction_spread_radians=spread,
        through_spread=through_spread,
        axial_slip=max(abs(s.axial_slip) for s in screws),
        fit_rmse=max(f.rmse for f in fits),
        pair_count=len(screws),
        pair_iterations=tuple(f.iterations for f in fits),
        passes=passes,
        converged=all(f.converged for f in fits),
        limitations=tuple(limitations),
    )


def angle_between(a: Point, b: Point) -> float:
    """The angle in radians between two directions, in ``[0, pi]``.

    ``atan2(|a x b|, a . b)`` rather than ``acos(a . b)``, for the same reason
    :attr:`scanner3d.geometry.Rigid.rotation_angle` does it: near zero ``acos``
    is severely ill-conditioned, and float64 rounding of the dot product puts a
    noise floor of about ``1.5e-8`` rad on the result. This function's whole
    job is to report how *little* two estimates of one spindle disagree, so a
    floor at 1.5e-8 would stop the measurement exactly where the answer is
    good - and would report a direction that is exact to the last bit as
    disagreeing by a hundredth of a microradian. ``acos`` also needs its
    argument clamped, because the dot product of two unit vectors can leave
    ``[-1, 1]`` by an ulp; ``atan2`` needs no clamp and cannot raise.
    """
    return math.atan2(norm(cross(a, b)), dot(a, b))


def _distance_to_line(point: Point, direction: Point, through: Point) -> float:
    """Perpendicular distance from ``point`` to the line, in input coordinate units."""
    offset = subtract(point, through)
    return math.dist(offset, scaled(direction, dot(offset, direction)))


# --- the sweep ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FramePose:
    """Where one frame's cloud sits in the reference frame, and what it cost to say so.

    :attr:`transform` carries this frame's points into the frame of cloud 0,
    which is the reference by definition and therefore carries the identity.

    :attr:`rmse` and :attr:`correspondences` describe the *relative* fit that
    placed this frame against its immediate predecessor - not this frame against
    the reference. Error accumulates along the chain and these numbers do not
    report the accumulation. :class:`LoopClosure` is what measures that, and it
    is the reason a closed sweep is worth capturing.

    The reference frame has no fit of its own, so its :attr:`rmse` is ``None``:
    a residual of zero would claim a perfect alignment that was never performed.
    """

    index: int
    transform: Rigid
    rmse: float | None
    correspondences: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "transform": self.transform.as_dict(),
            "rmse": self.rmse,
            "rmse_is": "the residual against the previous frame, not against the reference",
            "correspondences": self.correspondences,
        }


@dataclass(frozen=True, slots=True)
class LoopClosure:
    """How far a full revolution of composed steps misses returning to its start.

    The two residuals are reported separately and are never combined. An angle
    and a distance answer different questions, and this repository cannot even
    say what unit the distance is in, so any single "error" mixing them would be
    a number with no meaning at all.

    :attr:`note` states how the figure was produced, including - and this
    matters for reading :attr:`translation_error` - whether the steps that were
    composed had six degrees of freedom or one.

    What this cannot see is worth knowing before trusting it. If one frame is
    displaced by a rigid transform and the fits on either side of it still find
    their global optimum, those two fits absorb the displacement exactly and the
    chain closes as if nothing had happened - the frame's pose is wrong by the
    whole displacement and the residual here is at the floating-point floor.
    Closure is a check on whether the *fits* held together, not on whether the
    poses are the object's. That is why it is reported beside the per-frame
    residuals rather than in place of them.
    """

    rotation_error_radians: float
    translation_error: float
    frames: int
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation_error_radians": self.rotation_error_radians,
            "translation_error": self.translation_error,
            "translation_error_unit": (
                "input coordinate units, whose physical meaning is not established"
            ),
            "frames": self.frames,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """A registered sweep: every pose, every residual, and what none of it proves.

    :attr:`loop_closure` is ``None`` whenever the captures do not span a whole
    number of revolutions. That is not a missing feature - there is no closure
    to measure, and a number invented for an open sweep would be the most
    misleading thing this module could produce. The reason is put in
    :attr:`limitations` so a reader of the record alone can see it.
    """

    poses: tuple[FramePose, ...]
    pairwise: tuple[IcpResult, ...]
    axis: TurntableAxis
    angle_step_radians: float
    step_angles_radians: tuple[float, ...]
    loop_closure: LoopClosure | None
    limitations: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.poses)

    @property
    def converged(self) -> bool:
        """True only when every frame-to-frame fit settled inside its budget."""
        return all(fit.converged for fit in self.pairwise)

    @property
    def worst_rmse(self) -> float:
        """The largest frame-to-frame residual in the sweep."""
        return max(fit.rmse for fit in self.pairwise)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_count": self.frame_count,
            "angle_step_radians": self.angle_step_radians,
            "step_angles_radians": list(self.step_angles_radians),
            "converged": self.converged,
            "worst_rmse": self.worst_rmse,
            "axis": self.axis.as_dict(),
            "poses": [pose.as_dict() for pose in self.poses],
            "pairwise": [fit.as_dict() for fit in self.pairwise],
            "loop_closure": None if self.loop_closure is None else self.loop_closure.as_dict(),
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


def _revolutions_in(clouds: int, angle_step: float, closure_tolerance: float) -> int | None:
    """Whole revolutions the sweep spans, or ``None`` if it does not close.

    A sweep closes when its total swept angle lands within ``closure_tolerance``
    of a positive whole number of turns. Two turns close as honestly as one -
    the last frame still has to come back to the first - so the multiple is
    reported rather than assumed to be one.
    """
    swept = clouds * angle_step
    turns = round(swept / math.tau)
    if turns < 1 or abs(swept - turns * math.tau) > closure_tolerance:
        return None
    return turns


def register_turntable(
    clouds: Sequence[Cloud],
    *,
    angle_step: float,
    axis: TurntableAxis | None = None,
    constrain_to_axis: bool = True,
    seed: TurntableAxis | None = None,
    axis_passes: int = DEFAULT_AXIS_PASSES,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    voxel_size: float | None = None,
    max_correspondence_distance: float | None = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    closure_tolerance: float = DEFAULT_CLOSURE_TOLERANCE,
    limitations: Sequence[str] = (),
) -> RegistrationResult:
    """Align a turntable sweep into the first cloud's frame, and check it closes.

    Two stages, and the second is the one a turntable makes possible:

    1. The spindle is estimated once by :func:`estimate_turntable_axis`, unless
       the caller supplies one in ``axis``.
    2. Each consecutive pair is fitted starting from a rotation of ``angle_step``
       about that spindle. With ``constrain_to_axis`` left true the fit has a
       single degree of freedom - the angle, solved exactly by
       :func:`optimal_angle_about_axis` - so noise cannot push a frame off the
       spindle in a direction the mechanism does not have.

    Poses compose backwards along the chain: the transform of frame ``k`` is the
    inverse of every step that led to it, which is what carries that frame's
    points back into the reference frame's coordinates.

    **Loop closure.** When ``len(clouds) * angle_step`` lands on a whole number
    of revolutions, one extra pair is fitted - the last cloud against the first,
    with no special treatment - and the whole cycle of steps is composed. The
    product must be the identity, and
    :func:`scanner3d.geometry.pose_difference` measures how far it is not.
    Nothing in the fitting enforces this, which is exactly what makes it a
    check. When the sweep does not close, :attr:`RegistrationResult.loop_closure`
    is ``None`` and the arithmetic that decided so is recorded as a limitation.

    **Reading the translation half of that closure.** With
    ``constrain_to_axis`` true, every composed step is a rotation about one
    fixed line, so the residual is too, and its translation is exactly
    ``2 * |through| * sin(rotation_error / 2)`` - the rotation residual restated
    at the spindle's distance from the camera origin, not a second independent
    measurement. It is still reported, because it is what a consumer of the
    poses will actually see, but the note says what it is. Pass
    ``constrain_to_axis=False`` for six-degree-of-freedom steps, whose two
    residuals are independent, at the cost of a much worse conditioned fit.
    """
    ensure_angle_step(angle_step)
    _ensure_icp_parameters(
        max_iterations=max_iterations,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
    )
    if not math.isfinite(closure_tolerance) or closure_tolerance < 0.0:
        raise AngleStepUnusable(
            f"closure_tolerance {closure_tolerance!r} is not a finite non-negative angle"
        )
    validated = _ensure_clouds(clouds)

    spindle = (
        axis
        if axis is not None
        else estimate_turntable_axis(
            validated,
            angle_step=angle_step,
            seed=seed,
            passes=axis_passes,
            max_iterations=max_iterations,
            tolerance=tolerance,
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            keep_fraction=keep_fraction,
        )
    )

    turns = _revolutions_in(len(validated), angle_step, closure_tolerance)
    pairs = [(index, index + 1) for index in range(len(validated) - 1)]
    if turns is not None:
        # The pair that closes the ring, fitted exactly like every other pair.
        # Giving it any special treatment would make the check circular.
        pairs.append((len(validated) - 1, 0))

    # Every pair starts from the same place: a nominal step about the spindle.
    # That is the turntable constraint paying for itself - the fit begins a
    # fraction of a step from the answer instead of a whole step away.
    pair_seed = rotation_about_axis(spindle.direction, angle_step, through=spindle.through)
    solver = _angle_solver(spindle.direction, spindle.through) if constrain_to_axis else None

    fits: list[IcpResult] = []
    angles: list[float] = []
    for source_index, target_index in pairs:
        fit = _fit_pair(
            validated[source_index],
            validated[target_index],
            initial=pair_seed,
            solver=solver,
            max_iterations=max_iterations,
            tolerance=tolerance,
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            keep_fraction=keep_fraction,
        )
        fits.append(fit[0])
        angles.append(fit[1] if fit[1] is not None else fit[0].transform.rotation_angle)

    poses = _chain_poses(fits, len(validated))
    closure = _close_loop(fits, turns, constrain_to_axis=constrain_to_axis)
    return RegistrationResult(
        poses=poses,
        pairwise=tuple(fits),
        axis=spindle,
        angle_step_radians=angle_step,
        step_angles_radians=tuple(angles),
        loop_closure=closure,
        limitations=_all_limitations(
            fits,
            spindle,
            turns=turns,
            clouds=len(validated),
            angle_step=angle_step,
            closure_tolerance=closure_tolerance,
            constrain_to_axis=constrain_to_axis,
            extra=limitations,
        ),
    )


def _fit_pair(
    source: tuple[Point, ...],
    target: tuple[Point, ...],
    *,
    initial: Rigid,
    solver: _Solver | None,
    max_iterations: int,
    tolerance: float,
    voxel_size: float | None,
    max_correspondence_distance: float | None,
    keep_fraction: float,
) -> tuple[IcpResult, float | None]:
    """One frame-to-frame fit, constrained to the spindle or free, as asked.

    Always the unaccelerated :class:`_VoxelIndex` path - this has no
    ``backend`` parameter of its own, and :func:`register_turntable`, which
    calls this, is untouched by the "Selectable compute backend" section of
    this module's docstring.
    """
    prepared = _prepare(
        source,
        target,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
    )
    index = prepared.index
    assert index is not None  # build_index defaults to True; this path never sets it False

    def correspond(src: Sequence[Point], tr: Rigid) -> tuple[list[tuple[float, Point, Point]], int]:
        return _correspond(src, tr, index)

    fit = _iterate(
        prepared.source,
        correspond,
        initial=initial,
        solve=_rigid_solver if solver is None else solver,
        max_iterations=max_iterations,
        threshold=prepared.threshold,
        keep_fraction=keep_fraction,
        gate=prepared.gate,
    )
    return _as_result(fit, prepared), fit.angle


def _chain_poses(fits: Sequence[IcpResult], frames: int) -> tuple[FramePose, ...]:
    """Absolute poses from relative steps, walking the chain from the reference.

    The step transforms carry a frame *forward* onto its successor, so a pose -
    which has to carry a frame *back* into the reference - is the inverse chain.
    Composing the inverses in this order rather than inverting a product at the
    end keeps each intermediate pose available, and they are what a consumer
    applies to a cloud.
    """
    poses = [FramePose(index=0, transform=IDENTITY_POSE, rmse=None, correspondences=0)]
    for index in range(1, frames):
        step = fits[index - 1]
        poses.append(
            FramePose(
                index=index,
                transform=poses[index - 1].transform.compose(step.transform.inverse()),
                rmse=step.rmse,
                correspondences=step.correspondences,
            )
        )
    return tuple(poses)


def _close_loop(
    fits: Sequence[IcpResult], turns: int | None, *, constrain_to_axis: bool
) -> LoopClosure | None:
    """Compose the whole cycle and measure how far it is from having done nothing.

    ``None`` when the sweep does not span whole revolutions. There is nothing to
    close and nothing to report, and an invented figure would be worse than the
    absence of one.
    """
    if turns is None:
        return None
    accumulated = IDENTITY_POSE
    for fit in fits:
        accumulated = fit.transform.compose(accumulated)
    # The composed cycle is where the reference frame ends up after being carried
    # all the way round. It should be exactly where it started, so the difference
    # against the identity is the error of the entire chain.
    residual = pose_difference(IDENTITY_POSE, accumulated)
    if constrain_to_axis:
        note = (
            f"composed {len(fits)} frame-to-frame steps over {turns} revolution(s) and "
            "compared the product against the identity with geometry.pose_difference. Every "
            "step was constrained to one fixed spindle, so the residual is a rotation about "
            "that spindle and its translation is 2 * |through| * sin(rotation_error / 2) - the "
            "rotation error restated at the spindle's offset from the camera origin, not an "
            "independent measurement. The rotation error is the informative half"
        )
    else:
        note = (
            f"composed {len(fits)} frame-to-frame steps over {turns} revolution(s) and "
            "compared the product against the identity with geometry.pose_difference. Every "
            "step was fitted with six degrees of freedom, so the two residuals are "
            "independent: a translation error here is not explained by the rotation error"
        )
    return LoopClosure(
        rotation_error_radians=residual.rotation_angle,
        translation_error=residual.translation_magnitude,
        frames=len(fits),
        note=note,
    )


def _all_limitations(
    fits: Sequence[IcpResult],
    spindle: TurntableAxis,
    *,
    turns: int | None,
    clouds: int,
    angle_step: float,
    closure_tolerance: float,
    constrain_to_axis: bool,
    extra: Sequence[str],
) -> tuple[str, ...]:
    """Everything a consumer of this result must not be able to lose.

    :data:`INHERENT_LIMITATIONS` first, then what this particular run found, then
    the caller's own, de-duplicated. The inherent ones are not the caller's to
    omit by passing a shorter list.
    """
    found: list[str] = []
    if turns is None:
        found.append(
            f"no loop closure was measured: {clouds} clouds at {angle_step!r} rad sweep "
            f"{clouds * angle_step!r} rad, which is not within {closure_tolerance!r} of a whole "
            "number of revolutions, so these captures do not form a closed loop and no closure "
            "figure was invented for them"
        )
    if not all(fit.converged for fit in fits):
        stalled = [index for index, fit in enumerate(fits) if not fit.converged]
        found.append(
            f"frame-to-frame fits {stalled} used their whole iteration budget without the "
            "residual settling; those poses are where the fit stopped, not where it converged"
        )
    thin = [index for index, fit in enumerate(fits) if fit.matched_fraction < LOW_OVERLAP_FRACTION]
    if thin:
        found.append(
            f"frame-to-frame fits {thin} matched under half their source points; the residuals "
            "they report describe the points that survived the correspondence gate and the "
            "trim, and say nothing about the ones that did not"
        )
    if constrain_to_axis:
        found.append(
            "every step was constrained to one degree of freedom about the estimated spindle, "
            "so a spindle error is only partly visible here. A grossly wrong axis does show "
            "up, because the angles fitted about it stop summing to a whole revolution, but a "
            "composition that happens to total a whole turn is the identity about any line - "
            "so a small spindle error can leave a small closure residual. The direction and "
            "anchor spreads on the axis record are the direct evidence about the spindle"
        )
    return tuple(dict.fromkeys((*INHERENT_LIMITATIONS, *found, *spindle.limitations, *extra)))
