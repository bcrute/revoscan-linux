"""Point-to-plane ICP: align onto the target's surface, not its samples.

Why this exists
----------------
:func:`scanner3d.registration.icp` is POINT-TO-POINT, and its own inherent
limitations say so plainly: "point-to-point matching has no notion of
surface: it minimises distance between nearest samples." That is not a minor
caveat on this scanner's object. A 1/18 truck bed is mostly large flat
panels, and sliding two overlapping flat patches TANGENTIALLY - along the
panel, not off it - barely moves any point's nearest neighbour, so the
point-to-point residual barely notices. A rotation moves distant points
tangentially. The result is a fit that systematically UNDER-rotates, because
the residual it minimises is nearly blind to the one motion a rotation
mostly is.

Measured on the user's real two-lap capture, chained point-to-point ICP:

    step size 7.8 deg/frame -> 65 deg of drift per lap
    step size 3.9 deg/frame -> 23 deg of drift per lap

- each pairwise fit recovering only about 82% of the true rotation. That
under-rotation, compounded around a loop, is exactly the 124-degree error
:mod:`scanner3d.posegraph` exists to repair after the fact. This module is
the other half: fix the bias at its source instead of only redistributing it
once it has already accumulated.

The fix, and why it works
--------------------------
Point-to-plane ICP minimises distance along the TARGET SURFACE's normal
instead of to the nearest sample: for a matched pair ``(p, q)`` with target
normal ``n``, the residual is ``n . (p - q)`` rather than ``|p - q|``.
Sliding a point along its own tangent plane changes this residual by (to
first order) nothing, because the displacement is perpendicular to ``n`` -
so a candidate transform that under-rotates a flat panel is no longer
mistaken for a good fit. Genuinely constrained directions - motion that
actually pushes a point off the surface it is supposed to lie on - are
weighted properly instead of drowned out by the tangential blind spot. This
is the standard remedy for exactly this geometry (Chen & Medioni 1992, Low
2004) and it typically needs far fewer iterations to settle, for the same
reason: the objective it climbs down is shaped like the actual constraint
instead of a proxy for it.

What this does NOT do
----------------------
It is not a full generalised-ICP or a surfel matcher; it uses one normal per
target point from a single PCA pass over a fixed-size local neighbourhood,
computed once per call, not adapted per iteration or per correspondence
quality.

It solves a LINEARISED system every iteration - the rotation update is a
small-angle approximation (Rodrigues about a rotation VECTOR, not a search
over SO(3)) composed onto the current estimate - unlike point-to-point's
closed-form Horn solve, which refits from scratch every iteration and can
therefore take a large first step. Point-to-plane needs a reasonably close
``initial`` guess for the same reason any Gauss-Newton method does: a step
that is not small breaks the linearisation it is built on. It is not a
substitute for a coarse starting point - see
:mod:`scanner3d.multiview` for that - only a better refinement once one is
available.

Normals from local PCA are UNORIENTED: the eigenvector of the smallest
covariance eigenvalue names a LINE, not a direction, and its sign is an
arbitrary accident of the eigensolver, unrelated to which side of the
surface anything is on. This module never tries to make the signs
consistent, because point-to-plane's objective is ``(n . (p - q))**2`` -
already squared - and squaring a quantity is exactly invariant to negating
one of its factors: flipping every ``n`` in a pair flips that pair's row and
right-hand side in the linear system together, leaving the least-squares
minimiser unchanged. An orientation pass would spend work correcting
something the objective was already indifferent to.

A scene that does not constrain all six degrees of freedom - a single flat
panel is the textbook case, where sliding within the plane or turning about
its own normal moves no point-to-plane distance at all - is DETECTED and
REFUSED, not silently solved into a transform that is free to wander in the
unconstrained directions. See :func:`_solve_6x6` and
:exc:`IcpPlaneGeometryUnusable`.

No physical unit is implied anywhere, matching
:mod:`scanner3d.registration`: every distance-shaped parameter here is a
fraction of the target cloud's own extent unless the caller overrides it
with an absolute value, and :attr:`IcpPlaneResult.rmse` is in the input
cloud's own units. That RMSE is a POINT-TO-PLANE residual, not the
point-to-point Euclidean residual :attr:`~scanner3d.registration.IcpResult.rmse`
reports - it is usually smaller for the identical data, because a
point-to-plane fit is never penalised for the tangential slop
point-to-point measures. The two numbers are not directly comparable, and
comparing this module against point-to-point honestly means comparing
RECOVERED POSE ERROR against known truth, not comparing the two RMSEs
against each other.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import (
    IDENTITY_MATRIX,
    MIN_VECTOR_NORM,
    Matrix3,
    Point,
    Rigid,
    cross,
    dot,
    norm,
    normalize,
    rotation_about_axis,
    subtract,
)
from scanner3d.registration import (
    DEFAULT_CORRESPONDENCE_FRACTION,
    DEFAULT_KEEP_FRACTION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    DEFAULT_VOXEL_FRACTION,
    DOWNSAMPLE_THRESHOLD,
    IDENTITY_POSE,
    MAX_CLOUD_POINTS,
    MAX_ICP_ITERATIONS,
    MIN_CLOUD_POINTS,
    Cloud,
    cloud_extent,
    kept_count,
    voxel_downsample,
)

SCHEMA_VERSION: Final = 1

#: Points gathered around a query (including the query itself) for its local
#: PCA normal, by default. Small enough that a normal describes a genuinely
#: LOCAL patch rather than blurring across an edge or a corner; large enough
#: that three noisy samples do not masquerade as a plane. Chosen for the
#: point counts ICP already works at post-downsample - a few hundred - not
#: for a raw depth frame.
DEFAULT_NORMAL_NEIGHBOURS: Final = 10

#: Fewest neighbours a normal may be estimated from. Below this the local
#: covariance is barely more than three points, which fit a plane to
#: whatever noise they carry as readily as a real surface.
MIN_NORMAL_NEIGHBOURS: Final = 4

#: How close the smallest and middle local-covariance eigenvalues may be,
#: as a ratio to the largest, before a neighbourhood is treated as having no
#: dominant flat direction at all - an isotropic blob (noise, a corner where
#: several surfaces meet) where "the normal" is not a meaningful question.
#: Comfortably above float noise, comfortably below the ratio a genuinely
#: flat patch with real sensor noise produces.
NORMAL_ISOTROPY_RATIO: Final = 0.35

#: How small a pivot may be, relative to the largest entry in the 6x6
#: linearised system, before that direction is treated as unconstrained by
#: the data rather than solved for. See :func:`_solve_6x6`.
SINGULAR_PIVOT_RATIO: Final = 1e-9


class IcpPlaneNotAllowed(ScannerError):
    """An input was offered that this module refuses to fit."""

    status = DoctorStatus.UNSUPPORTED_MODE


class IcpPlaneCloudUnusable(IcpPlaneNotAllowed):
    """A point cloud cannot take part in a point-to-plane match."""


class IcpPlaneParametersUnusable(IcpPlaneNotAllowed):
    """An iteration bound, tolerance, gate or neighbour count is out of range."""


class IcpPlaneCorrespondencesInsufficient(IcpPlaneNotAllowed):
    """Too few point-normal pairs survived to solve anything.

    Either the clouds do not overlap where the current estimate puts them,
    or too few target points carry a usable normal at all - see
    :func:`estimate_normals`.
    """


class IcpPlaneGeometryUnusable(IcpPlaneNotAllowed):
    """The linearised 6-DOF system is singular: some motion is unconstrained.

    Raised by :func:`_solve_6x6` rather than returning a transform that is
    free to drift in whichever direction the data does not constrain - a
    single flat panel is the textbook example, see the module docstring.
    """


# --- cloud validation -----------------------------------------------------------


def _ensure_cloud(points: Cloud, *, what: str) -> tuple[Point, ...]:
    """Normalise one cloud to validated ``(x, y, z)`` tuples, or refuse it.

    The same checks as :mod:`scanner3d.registration`'s own (private) cloud
    validation, duplicated rather than reached into: see the module
    docstring's note on why small private helpers are copied here instead of
    coupling two modules through a leading-underscore name.
    """
    collected: list[Point] = []
    for index, point in enumerate(points):
        if len(collected) >= MAX_CLOUD_POINTS:
            raise IcpPlaneCloudUnusable(
                f"{what}: refusing more than {MAX_CLOUD_POINTS} points; the producer offered "
                f"at least {index + 1}"
            )
        values = tuple(point)
        if len(values) != 3:
            raise IcpPlaneCloudUnusable(
                f"{what}: point {index} carries {len(values)} coordinates, not the 3 an "
                "(x, y, z) triple has"
            )
        x, y, z = values
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise IcpPlaneCloudUnusable(
                f"{what}: point {index} is {values!r}; NaN and infinities are refused here "
                "rather than allowed to poison a centroid, a normal, and every pose through it"
            )
        collected.append((float(x), float(y), float(z)))
    if len(collected) < MIN_CLOUD_POINTS:
        raise IcpPlaneCloudUnusable(
            f"{what}: {len(collected)} points cannot constrain a rigid transform; at least "
            f"{MIN_CLOUD_POINTS} are needed"
        )
    return tuple(collected)


def _ensure_parameters(
    *,
    max_iterations: int,
    tolerance: float,
    voxel_size: float | None,
    max_correspondence_distance: float | None,
    keep_fraction: float,
    normal_neighbours: int,
) -> None:
    if not 1 <= max_iterations <= MAX_ICP_ITERATIONS:
        raise IcpPlaneParametersUnusable(
            f"max_iterations {max_iterations} is outside [1, {MAX_ICP_ITERATIONS}]"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise IcpPlaneParametersUnusable(
            f"tolerance {tolerance!r} is not a finite non-negative fraction of the target "
            "cloud's extent"
        )
    if not math.isfinite(keep_fraction) or not 0.0 < keep_fraction <= 1.0:
        raise IcpPlaneParametersUnusable(f"keep_fraction {keep_fraction!r} is outside (0, 1]")
    if normal_neighbours < MIN_NORMAL_NEIGHBOURS:
        raise IcpPlaneParametersUnusable(
            f"normal_neighbours {normal_neighbours} is below the {MIN_NORMAL_NEIGHBOURS} a "
            "local covariance needs before it fits noise as readily as a real surface"
        )
    for value, name in (
        (voxel_size, "voxel_size"),
        (max_correspondence_distance, "max_correspondence_distance"),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise IcpPlaneParametersUnusable(f"{name} {value!r} must be a finite positive size")


# --- a small, general symmetric eigensolver ---------------------------------------


def _jacobi(matrix: list[list[float]], sweeps: int = 64) -> tuple[list[float], list[list[float]]]:
    """Symmetric eigen-decomposition by cyclic Jacobi rotations.

    The same method, for the same reason, as
    :func:`scanner3d.multiview._jacobi`, :func:`scanner3d.geometry._jacobi_eigen`
    and :func:`scanner3d.posegraph._jacobi`: unconditionally stable on
    symmetric matrices, no starting guess needed, no numerical library in
    this repository to reach for instead. Duplicated rather than imported
    because every one of those is private to its own module - by now an
    established policy, not a one-off.
    """
    size = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(sweeps):
        off = math.fsum(a[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off < 1e-30:
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
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(size)], v


# --- surface normals ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalEstimate:
    """One point's estimated surface normal, or the honest absence of one.

    ``normal`` is UNORIENTED - see the module docstring - and ``None`` when
    the local neighbourhood could not support one: too few neighbours, every
    neighbour coincident, or no dominant flat direction (an isotropic patch,
    typically noise or a corner where several surfaces meet).
    """

    point: Point
    normal: Point | None
    neighbours: int

    def as_dict(self) -> dict[str, object]:
        return {
            "point": list(self.point),
            "normal": list(self.normal) if self.normal is not None else None,
            "neighbours": self.neighbours,
        }


@dataclass(frozen=True, slots=True)
class NormalField:
    """Per-point normals for one cloud, and how many could not be estimated."""

    estimates: tuple[NormalEstimate, ...]
    requested_neighbours: int
    limitations: tuple[str, ...]

    @property
    def usable(self) -> tuple[NormalEstimate, ...]:
        return tuple(e for e in self.estimates if e.normal is not None)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "points": len(self.estimates),
            "usable": len(self.usable),
            "requested_neighbours": self.requested_neighbours,
            "limitations": list(self.limitations),
        }


def _covariance(points: Sequence[Point], centre: Point) -> list[list[float]]:
    xx = xy = xz = yy = yz = zz = 0.0
    for p in points:
        a, b, c = p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]
        xx += a * a
        xy += a * b
        xz += a * c
        yy += b * b
        yz += b * c
        zz += c * c
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


def estimate_normals(
    points: Sequence[Point], *, neighbours: int = DEFAULT_NORMAL_NEIGHBOURS
) -> NormalField:
    """One unoriented surface normal per point, from a local PCA neighbourhood.

    For each point, the ``neighbours`` nearest OTHER points (by plain
    Euclidean distance, found by brute-force scan - see the note below on why
    that is the right tradeoff here) are gathered together with the point
    itself, and the eigenvector of the SMALLEST eigenvalue of their centred
    covariance is returned as the normal: the direction the local patch
    varies LEAST along, which is "off the surface" for anything that
    resembles one. :data:`NORMAL_ISOTROPY_RATIO` guards the case where no
    direction is clearly least - an isotropic neighbourhood carries no
    surface information at all - and a point with fewer than
    :data:`MIN_NORMAL_NEIGHBOURS` neighbours in the whole cloud is refused
    for the same reason a covariance of three points is refused: not enough
    to distinguish a surface from noise.

    Brute-force neighbour search is quadratic in the cloud size, and that is
    a deliberate, stated tradeoff rather than an oversight: this is called
    once per :func:`icp_point_to_plane` call, on a cloud already reduced to
    ICP's own working size (a few hundred points post-downsample - see
    :mod:`scanner3d.registration`'s own numbers), where it is negligible
    next to the correspondence search ICP repeats every iteration. A voxel
    grid tuned for k-nearest queries would only earn its complexity back on
    clouds this module is never actually asked to normal-estimate.

    Normals are UNORIENTED - see the module docstring for why that is fine
    for point-to-plane and is not corrected here.
    """
    if neighbours < MIN_NORMAL_NEIGHBOURS:
        raise IcpPlaneParametersUnusable(
            f"neighbours {neighbours} is below the {MIN_NORMAL_NEIGHBOURS} a local covariance "
            "needs before it fits noise as readily as a real surface"
        )
    n = len(points)
    estimates: list[NormalEstimate] = []
    sparse = coincident = isotropic = 0
    k_request = min(neighbours, n - 1)
    for i, p in enumerate(points):
        if k_request < MIN_NORMAL_NEIGHBOURS:
            estimates.append(NormalEstimate(p, None, max(k_request, 0)))
            sparse += 1
            continue

        def _squared_distance_from_p(j: int, p: Point = p) -> float:
            return (
                (points[j][0] - p[0]) ** 2 + (points[j][1] - p[1]) ** 2 + (points[j][2] - p[2]) ** 2
            )

        nearest_idx = heapq.nsmallest(k_request + 1, range(n), key=_squared_distance_from_p)
        neighbour_idx = [j for j in nearest_idx if j != i][:k_request]
        patch = [p, *(points[j] for j in neighbour_idx)]
        centre = (
            math.fsum(q[0] for q in patch) / len(patch),
            math.fsum(q[1] for q in patch) / len(patch),
            math.fsum(q[2] for q in patch) / len(patch),
        )
        values, vectors = _jacobi(_covariance(patch, centre))
        order = sorted(range(3), key=lambda idx: values[idx])
        smallest, largest = values[order[0]], values[order[2]]
        if largest <= 0.0:
            estimates.append(NormalEstimate(p, None, len(neighbour_idx)))
            coincident += 1
            continue
        if smallest / largest > NORMAL_ISOTROPY_RATIO:
            estimates.append(NormalEstimate(p, None, len(neighbour_idx)))
            isotropic += 1
            continue
        normal = normalize((vectors[0][order[0]], vectors[1][order[0]], vectors[2][order[0]]))
        estimates.append(NormalEstimate(p, normal, len(neighbour_idx)))
    limitations = [
        "Normals are unoriented: the sign of each is an arbitrary accident of the "
        "eigensolver and carries no meaning. Point-to-plane's squared objective is exactly "
        "invariant to it, so no orientation pass is attempted.",
        "A normal describes a LOCAL patch of the requested neighbourhood size. It says "
        "nothing about the surface's shape beyond that patch, and a patch straddling an "
        "edge or a corner produces a normal for neither surface.",
    ]
    if sparse:
        limitations.append(
            f"{sparse} of {n} points had fewer than {MIN_NORMAL_NEIGHBOURS} neighbours in the "
            "whole cloud and were left without a normal."
        )
    if coincident:
        limitations.append(f"{coincident} of {n} points had an entirely coincident neighbourhood.")
    if isotropic:
        limitations.append(
            f"{isotropic} of {n} points had no dominant flat direction (ratio above "
            f"{NORMAL_ISOTROPY_RATIO}) - likely noise or a corner where surfaces meet - and "
            "were left without a normal."
        )
    return NormalField(
        estimates=tuple(estimates), requested_neighbours=neighbours, limitations=tuple(limitations)
    )


# --- nearest-neighbour correspondence, with the matched normal riding along ----------


_NEIGHBOUR_CELLS: Final = tuple(itertools.product((-1, 0, 1), repeat=3))


def _cell_key(point: Point, cell: float) -> tuple[int, int, int]:
    return (math.floor(point[0] / cell), math.floor(point[1] / cell), math.floor(point[2] / cell))


class _NearestIndex:
    """Nearest-neighbour lookup over target points that carry a usable normal.

    The same 27-cell voxel-hash bound as
    :class:`scanner3d.registration._VoxelIndex`, and the same correctness
    argument: the cell side IS the gate, so any point within one gate of a
    query cannot be outside the 3x3x3 block around the query's own cell.
    Duplicated rather than imported - see the module docstring's note on
    :func:`_jacobi` for why - and extended to carry each point's normal
    alongside it, which the point-to-point index was never asked to do.
    """

    __slots__ = ("_cells", "_gate")

    def __init__(self, points_with_normals: Sequence[tuple[Point, Point]], *, gate: float) -> None:
        self._gate = gate
        cells: dict[tuple[int, int, int], list[tuple[Point, Point]]] = {}
        for point, normal in points_with_normals:
            cells.setdefault(_cell_key(point, gate), []).append((point, normal))
        self._cells = cells

    @property
    def gate(self) -> float:
        return self._gate

    def nearest(self, query: Point) -> tuple[Point, Point] | None:
        base = _cell_key(query, self._gate)
        qx, qy, qz = query
        best: tuple[Point, Point] | None = None
        best_squared = self._gate * self._gate
        for dx, dy, dz in _NEIGHBOUR_CELLS:
            bucket = self._cells.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if bucket is None:
                continue
            for point, normal in bucket:
                ex, ey, ez = point[0] - qx, point[1] - qy, point[2] - qz
                squared = ex * ex + ey * ey + ez * ez
                if squared < best_squared:
                    best_squared = squared
                    best = (point, normal)
        return best


# --- the 6x6 linearised solve, with its singularity guard -----------------------------


def _solve_6x6(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gauss-Jordan elimination with partial pivoting on a symmetric 6x6 system.

    A 6x6 dense solve is entirely reasonable in pure Python - this is not the
    hundreds-of-poses linear system :mod:`scanner3d.posegraph` avoided; it is
    six unknowns, rebuilt and solved once per ICP iteration.

    The singularity guard is the actual point of this function existing as
    its own piece rather than being inlined. A pivot small relative to the
    system's own scale means that column's direction of motion - one of the
    three rotation-vector components or one of the three translations - is
    left almost unconstrained by every paired normal: nothing in the data
    resists moving that way. A single flat panel is the textbook example -
    sliding within it, or turning about its own normal, changes no
    point-to-plane distance at all - and this raises rather than returning
    whatever the arithmetic produces for a direction nothing constrained.
    """
    n = 6
    scale = max((abs(v) for row in matrix for v in row), default=0.0)
    if scale <= 0.0:
        raise IcpPlaneGeometryUnusable(
            "the linearised system is entirely zero: no paired normal contributed any "
            "constraint at all"
        )
    augmented = [[*matrix[i], rhs[i]] for i in range(n)]
    axis_names = (
        "rotation x",
        "rotation y",
        "rotation z",
        "translation x",
        "translation y",
        "translation z",
    )
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < SINGULAR_PIVOT_RATIO * scale:
            raise IcpPlaneGeometryUnusable(
                f"the linearised system is singular in {axis_names[col]}: these correspondences "
                "and their normals leave that direction of motion unconstrained (a single flat "
                "panel is the textbook cause), so no transform is fitted rather than one that is "
                "free to wander there"
            )
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for k in range(col, n + 1):
            augmented[col][k] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = augmented[r][col]
            if factor != 0.0:
                for k in range(col, n + 1):
                    augmented[r][k] -= factor * augmented[col][k]
    return [augmented[i][n] for i in range(n)]


def _solve_point_to_plane(matched: Sequence[tuple[Point, Point, Point]], current: Rigid) -> Rigid:
    """One Gauss-Newton step around ``current``, composed onto it.

    ``matched`` is ``(original_source_point, target_point, target_normal)``
    triples. For small ``e`` (a rotation VECTOR: axis times angle) and
    translation ``t``, the point-to-plane residual after applying the
    correction to the CURRENTLY-transformed source point ``p`` linearises to

        n . (p + e x p + t - q)  ~=  n . (p - q)  +  e . (p x n)  +  n . t

    using the scalar triple product identity ``n . (e x p) = e . (p x n)``.
    Each pair therefore contributes one row ``(p x n, n)`` of a 6x6 normal-
    equations system whose solution is the correction that best cancels
    every pair's current residual at once - unlike point-to-point's
    :func:`~scanner3d.geometry.rigid_from_correspondences`, which refits an
    ABSOLUTE transform from scratch each iteration, this refits a
    RELATIVE correction to the estimate it was handed, which is why it is
    composed onto ``current`` rather than replacing it.
    """
    matrix = [[0.0] * 6 for _ in range(6)]
    rhs = [0.0] * 6
    for original, target_point, target_normal in matched:
        p = current.apply(original)
        pxn = cross(p, target_normal)
        row = (pxn[0], pxn[1], pxn[2], target_normal[0], target_normal[1], target_normal[2])
        residual = dot(target_normal, subtract(p, target_point))
        for a in range(6):
            row_a = row[a]
            if row_a != 0.0:
                for b in range(6):
                    matrix[a][b] += row_a * row[b]
            rhs[a] += -row_a * residual
    solution = _solve_6x6(matrix, rhs)
    e = (solution[0], solution[1], solution[2])
    t = (solution[3], solution[4], solution[5])
    angle = norm(e)
    rotation: Matrix3 = (
        IDENTITY_MATRIX if angle < MIN_VECTOR_NORM else rotation_about_axis(e, angle).rotation
    )
    increment = Rigid.of(rotation, t)
    return increment.compose(current)


def _point_to_plane_rmse(moved: Sequence[tuple[Point, Point, Point]]) -> float:
    """``sqrt(mean((n . (p - q))**2))`` over ``(moved_point, target, normal)`` triples.

    Squared, which is what makes it exactly independent of each normal's
    arbitrary sign - see the module docstring. This is the objective
    :func:`_solve_point_to_plane` actually minimises, so it is the honest
    residual to report; it is not the point-to-point Euclidean RMSE
    :attr:`~scanner3d.registration.IcpResult.rmse` reports, and the two are
    not directly comparable on the same data.
    """
    if not moved:
        return math.inf
    total = math.fsum(dot(normal, subtract(p, q)) ** 2 for p, q, normal in moved)
    return math.sqrt(total / len(moved))


# --- iteration ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FitPlane:
    transform: Rigid
    rmse: float
    iterations: int
    converged: bool
    correspondences: int
    trimmed: int
    rejected: int
    trim_distance: float


def _correspond(
    source: Sequence[Point], transform: Rigid, index: _NearestIndex
) -> tuple[list[tuple[float, Point, Point, Point]], int]:
    """Pair each ORIGINAL source point with its nearest normal-bearing target point.

    Returns ``(distance, original_source_point, target_point, target_normal)``
    tuples and how many source points found nothing inside the gate. The
    ORIGINAL point rides along, not the transformed one - see
    :func:`scanner3d.registration._correspond`'s identical choice and reason
    - so every solve fits (or corrects) an absolute transform from the input
    frame. ``distance`` is the plain Euclidean nearest-neighbour distance,
    the SAME quantity point-to-point gates and trims on, so the two methods'
    correspondence search and trim behave identically and only the SOLVE
    step differs - which is what keeps a comparison between them honest.
    """
    pairs: list[tuple[float, Point, Point, Point]] = []
    rejected = 0
    for original, moved in zip(source, transform.apply_all(source), strict=True):
        found = index.nearest(moved)
        if found is None:
            rejected += 1
            continue
        target_point, target_normal = found
        pairs.append((math.dist(moved, target_point), original, target_point, target_normal))
    return pairs, rejected


def _iterate(
    source: Sequence[Point],
    index: _NearestIndex,
    *,
    initial: Rigid,
    max_iterations: int,
    threshold: float,
    keep_fraction: float,
) -> _FitPlane:
    """Match, trim, solve, repeat - the same shape as
    :func:`scanner3d.registration._iterate`, with a point-to-plane solve in
    place of the closed-form Horn fit and a point-to-plane RMSE for the
    convergence check.
    """
    transform = initial
    rmse = math.inf
    previous: float | None = None
    correspondences = trimmed = rejected = 0
    trim_distance = index.gate
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        pairs, rejected = _correspond(source, transform, index)
        if len(pairs) < 6:
            raise IcpPlaneCorrespondencesInsufficient(
                f"only {len(pairs)} of {len(pairs) + rejected} source points found a "
                f"normal-bearing partner inside the correspondence gate at iteration "
                f"{iterations}; six pairs are the floor a 6-DOF linearised solve needs even "
                "before the singularity guard has a say"
            )
        pairs.sort(key=lambda pair: pair[0])
        keep = min(len(pairs), kept_count(len(pairs), keep_fraction))
        correspondences = keep
        trimmed = len(pairs) - keep
        trim_distance = pairs[keep - 1][0]
        matched = [(pair[1], pair[2], pair[3]) for pair in pairs[:keep]]
        transform = _solve_point_to_plane(matched, transform)
        moved = [(transform.apply(o), q, n) for o, q, n in matched]
        rmse = _point_to_plane_rmse(moved)
        if previous is not None and abs(previous - rmse) <= threshold:
            converged = True
            break
        previous = rmse
    return _FitPlane(
        transform=transform,
        rmse=rmse,
        iterations=iterations,
        converged=converged,
        correspondences=correspondences,
        trimmed=trimmed,
        rejected=rejected,
        trim_distance=trim_distance,
    )


# --- the public result and entry point --------------------------------------------


@dataclass(frozen=True, slots=True)
class IcpPlaneResult:
    """What one point-to-plane alignment did, mirroring
    :class:`~scanner3d.registration.IcpResult` field for field where the two
    are honestly the same thing, and adding what point-to-plane alone needs.

    :attr:`rmse` is a POINT-TO-PLANE residual - see the module docstring for
    why it is not comparable to :attr:`~scanner3d.registration.IcpResult.rmse`
    on the same data. :attr:`target_points_with_normal` is new: point-to-plane
    can only match against target points a usable normal was found for, and
    this says how many of :attr:`target_points` that was.
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
    target_points_with_normal: int
    voxel_size: float | None
    max_correspondence_distance: float
    keep_fraction: float
    trim_distance: float
    convergence_threshold: float
    normal_neighbours: int

    @property
    def offered(self) -> int:
        return self.correspondences + self.trimmed + self.rejected

    @property
    def matched_fraction(self) -> float:
        return (self.correspondences + self.trimmed) / self.offered if self.offered else 0.0

    @property
    def used_fraction(self) -> float:
        return self.correspondences / self.offered if self.offered else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transform": self.transform.as_dict(),
            "rmse": self.rmse,
            "rmse_kind": "point-to-plane: not comparable to registration.IcpResult.rmse",
            "iterations": self.iterations,
            "converged": self.converged,
            "correspondences": self.correspondences,
            "trimmed": self.trimmed,
            "rejected": self.rejected,
            "matched_fraction": self.matched_fraction,
            "used_fraction": self.used_fraction,
            "source_points": self.source_points,
            "target_points": self.target_points,
            "target_points_with_normal": self.target_points_with_normal,
            "voxel_size": self.voxel_size,
            "max_correspondence_distance": self.max_correspondence_distance,
            "keep_fraction": self.keep_fraction,
            "trim_distance": self.trim_distance,
            "convergence_threshold": self.convergence_threshold,
            "normal_neighbours": self.normal_neighbours,
        }


def icp_point_to_plane(
    source: Cloud,
    target: Cloud,
    *,
    initial: Rigid = IDENTITY_POSE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    voxel_size: float | None = None,
    max_correspondence_distance: float | None = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    normal_neighbours: int = DEFAULT_NORMAL_NEIGHBOURS,
) -> IcpPlaneResult:
    """Point-to-plane ICP: the transform carrying ``source`` onto ``target``.

    Same argument order, same transform convention, and the same defaults
    where the two methods share a concern as
    :func:`scanner3d.registration.icp`, deliberately: a caller should be able
    to swap one for the other. ``tolerance``, ``voxel_size`` and
    ``max_correspondence_distance`` are fractions of the target cloud's own
    extent when left to their defaults, exactly as in point-to-point.

    Target normals are estimated ONCE, up front, over the (possibly
    downsampled) target cloud - see :func:`estimate_normals` - and every
    iteration matches against whichever target points carried a usable one.
    Everything else - correspondence search, the distance gate, the trim -
    is identical in mechanism to point-to-point; only the SOLVE step, per
    iteration, differs. See the module docstring for what that step does not
    do and when it refuses rather than guesses.
    """
    source_points = _ensure_cloud(source, what="ICP source")
    target_points = _ensure_cloud(target, what="ICP target")
    _ensure_parameters(
        max_iterations=max_iterations,
        tolerance=tolerance,
        voxel_size=voxel_size,
        max_correspondence_distance=max_correspondence_distance,
        keep_fraction=keep_fraction,
        normal_neighbours=normal_neighbours,
    )

    extent = cloud_extent(target_points)
    gate = (
        max_correspondence_distance
        if max_correspondence_distance is not None
        else extent * DEFAULT_CORRESPONDENCE_FRACTION
    )
    resolved_voxel = voxel_size
    if (
        resolved_voxel is None
        and max(len(source_points), len(target_points)) > DOWNSAMPLE_THRESHOLD
    ):
        resolved_voxel = extent * DEFAULT_VOXEL_FRACTION
    sampled_source = (
        source_points if resolved_voxel is None else voxel_downsample(source_points, resolved_voxel)
    )
    sampled_target = (
        target_points if resolved_voxel is None else voxel_downsample(target_points, resolved_voxel)
    )
    if len(sampled_source) < MIN_CLOUD_POINTS or len(sampled_target) < MIN_CLOUD_POINTS:
        raise IcpPlaneCloudUnusable(
            f"downsampling at a voxel side of {resolved_voxel!r} left "
            f"{len(sampled_source)} source and {len(sampled_target)} target points, fewer "
            f"than the {MIN_CLOUD_POINTS} a rigid transform needs"
        )

    normals = estimate_normals(sampled_target, neighbours=normal_neighbours)
    usable = normals.usable
    if len(usable) < 6:
        raise IcpPlaneCorrespondencesInsufficient(
            f"only {len(usable)} of {len(sampled_target)} target points carry a usable "
            "surface normal - too few for a 6-DOF point-to-plane solve regardless of overlap"
        )
    index = _NearestIndex([(e.point, e.normal) for e in usable if e.normal is not None], gate=gate)

    fit = _iterate(
        sampled_source,
        index,
        initial=initial,
        max_iterations=max_iterations,
        threshold=tolerance * extent,
        keep_fraction=keep_fraction,
    )
    return IcpPlaneResult(
        transform=fit.transform,
        rmse=fit.rmse,
        iterations=fit.iterations,
        converged=fit.converged,
        correspondences=fit.correspondences,
        trimmed=fit.trimmed,
        rejected=fit.rejected,
        source_points=len(sampled_source),
        target_points=len(sampled_target),
        target_points_with_normal=len(usable),
        voxel_size=resolved_voxel,
        max_correspondence_distance=gate,
        keep_fraction=keep_fraction,
        trim_distance=fit.trim_distance,
        convergence_threshold=tolerance * extent,
        normal_neighbours=normal_neighbours,
    )
