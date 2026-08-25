"""Distribute a loop-closure discrepancy back around the loop that produced it.

Why this exists
----------------
Frames are aligned to their predecessor by ICP and the poses chained: frame
``k``'s pose is frame ``k - 1``'s pose composed with what ICP measured between
them (see :func:`scanner3d.registration.register_turntable`'s ``_chain_poses``,
which this module's edge convention deliberately matches). Measured on a real
two-lap capture, that chain is catastrophically wrong at the global scale even
though every per-step and self-consistency signal says it is fine::

    per-pair ICP rmse         0.65 - 0.75  (excellent)
    frame overlap             100%          (excellent)
    self-consistency drift    2.1 - 2.9     (looks fine)
    actual error after 1 lap  65 degrees
    actual error after 2 laps 139 degrees

Each pairwise ICP fit slightly UNDER-rotates; over a revolution the chain
recovers only about 82% of the true rotation, and the shortfall accumulates
linearly with every step. Nothing internal catches it because every internal
metric - rmse, overlap, self-consistency - measures agreement between
neighbouring frames, and a smoothly drifting trajectory is perfectly
self-consistent with itself.

What catches it is loop closure: when the camera revisits a viewpoint it has
already occupied, ICP between the two RAW clouds reports how far apart the
chain has placed them, with no reference to angles, axes, or capture order.
That discrepancy is external evidence - refinement against the chain's own
model cannot see it and cannot fake fixing it, because it never entered the
objective the chain was built from.

This module is the optimiser that acts on that evidence: given the chained
poses and a set of measurements between frame pairs - some adjacent in
capture order, some revisits found elsewhere - it redistributes the closure
discrepancy back around the loop instead of leaving it piled up at the one
seam where the loop happened to close.

No knowledge of capture geometry
---------------------------------
This module takes nodes and edges and nothing else. A node is a pose; an edge
is a measured relative transform between two named nodes and a weight. It does
not know which edges came from sequential chaining and which from a revisit
detector, does not know whether a turntable, a handheld sweep, or something
else produced the poses, and does not know or care whether the graph contains
a loop at all. That distinction belongs entirely to the caller. A capture
method that needed this module to know about it would be a capture method
this module could not serve, and there is exactly one pose-graph optimiser in
this repository for exactly that reason.

The edge convention
--------------------
An edge ``(i, j, transform, weight)`` states what ICP measured relating frame
``i``'s local cloud to frame ``j``'s local cloud: ``transform.apply(p)`` is
approximately the same physical point as ``p``, expressed in frame ``j``'s
local coordinates rather than frame ``i``'s. This is exactly the convention
:mod:`scanner3d.registration` already uses for a pairwise ICP fit (source
``i``, target ``j``), so a sequential edge here is literally that module's
``IcpResult.transform`` carried over unchanged; a closure edge is the same
measurement produced by matching two non-adjacent frames instead.

Method: iterative relaxation, not a linear solve
--------------------------------------------------
A dense Gauss-Newton system over ``6n`` parameters is not viable in pure
Python with no numerical library, for the few hundred poses a real capture
produces. Instead this is Jacobi/Gauss-Seidel relaxation directly on the pose
graph: repeatedly sweep every non-anchor node, and at each one, let every
incident edge PREDICT where that node should sit given its neighbour's
current pose; combine the predictions and move the node there. Repeated to
convergence this settles at the least-squares solution under the chordal
metric - the same metric :func:`average_rotations` projects onto SO(3) below.

Rotation averaging is the one genuinely subtle step, because the arithmetic
mean of rotation matrices is not a rotation. See :func:`average_rotations`.

``anchor`` is the one node relaxation never touches. Every edge constrains
only RELATIVE poses - shift every node by the same rigid transform and every
edge still agrees exactly - so without one node held fixed the whole graph is
free to drift or spin bodily through space while agreeing with itself
perfectly, exactly the failure loop closure exists to catch in the first
place.

What this does NOT do
----------------------
It does not detect loop closures, decide which edges to trust, or know
anything about capture geometry - all of that is the caller's job, upstream
of this module. It is not bundle adjustment over raw points: it optimises
POSES against MEASURED RELATIVE TRANSFORMS between them, and the quality of
that is only ever as good as the edges it is given. A graph with no
redundant path between any two nodes - a bare chain, or a "loop" with only
one edge closing it and no other independent connection - has nothing for
relaxation to reconcile: every edge is already satisfied to first order by
the chain that produced it, and the graph comes back essentially unchanged.
That is not a bug in this module; it is an honest description of what a
single chain of measurements can and cannot correct about itself. See the
``limitations`` on every result, which state this outcome when it is what
happened.

Nor does it detect a genuinely CONTRADICTORY set of edges - two closures that
cannot both hold. Relaxation converges to a compromise in that case (rotation
averaging degrades gracefully rather than producing a non-rotation - see
:func:`average_rotations`), and the compromise is reported honestly as a
large remaining ``worst_edge`` error rather than hidden.

No physical unit is implied anywhere. A weight is a caller-chosen trust score,
not a physical uncertainty, and the "combined" edge-error magnitude mixes a
rotation angle in radians with a translation in whatever unit the caller's
transforms use - a single scalar exists only so edges and iterations can be
RANKED against each other within one run, and is not a claim that the two
halves are commensurate.

Robust mode: a minority of wrong edges can still capture the fit
--------------------------------------------------------------------
The chordal least-squares fit above trusts every edge according to its
weight, with no notion that an edge might simply be wrong. On a real 70-frame
sweep this let 5 of 34 detected "revisits" - each one a genuine failure of
ICP, seeded from the identity, converging to a near-identity transform
between two clouds that were actually a HALF revolution apart rather than a
whole one - drag a good trajectory to 11x worse than doing nothing. Every
LOCAL statistic on those five (rmse, overlap, correspondence count) sat
inside the range the true revisits also occupied; nothing about a false edge
in isolation marks it as false. What marks it is that it disagrees with the
consensus the OTHER edges agree on, and that is only visible globally, which
is exactly this module's job and not :mod:`scanner3d.revisit`'s.

``robust=True`` runs Iteratively Reweighted Least Squares with a Cauchy
kernel on top of the same relaxation: fit, measure every edge's residual
against the fit, treat the median-scaled residual distribution as the
population an edge should agree with, down-weight (and, past
:data:`DEFAULT_OUTLIER_THRESHOLD` robust sigmas, effectively drop) any edge
that does not, and repeat until the rejected set stops changing. See
:func:`_robust_weights`.

This is OPT-IN, never silent: the default is ``robust=False``, bit-identical
to the relaxation described above, and even when enabled the result always
names precisely which edges were downweighted past rejection and why - see
:attr:`PoseGraphResult.rejected_edges`. Discarding a constraint the caller
supplied without saying so is the same category of fault as a silent
backend downgrade, and this module does not do it by default and does not
hide it when asked to.

Rejection is earned by disagreement with the fitted solution, not by being
outnumbered: the Cauchy scale comes from the median of the ELIGIBLE (not
:attr:`PoseEdge.protected`) edges' residuals only, a statistic that stays
small exactly as long as a MAJORITY of THOSE agree, regardless of whether
that majority is 3 edges or 300 - see :data:`MIN_ROBUST_EDGES` for the one
honest exception, where there are too few eligible edges for "majority" to
mean anything at all. Protected edges are excluded from the scale itself,
not merely exempted from rejection: a caller typically protects sequential
edges that were themselves composed to build the poses being fit, so those
edges are satisfied to float64 precision BY CONSTRUCTION, and folding a
population that cannot be rejected into the statistic that decides what
"typical disagreement" looks like among the population that CAN is a
category error - one that, measured on real data, silently crushed the
scale toward zero and rejected nothing at all. See :func:`_robust_weights`
for the full account, including the real numbers that exposed it. A caller
sets :attr:`PoseEdge.protected` for whatever reason it trusts an edge on
grounds this module cannot see; this module still does not know or care
whether that edge came from chaining or a revisit detector, only that the
caller marked it exempt.

If rejecting the settled set of edges would disconnect a node that the FULL
edge set could reach from ``anchor``, :func:`optimise_poses` refuses with
:exc:`PoseGraphRobustDisconnected` rather than returning a pose for a node
nothing constrains any more.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Matrix3, Point, Rigid, ensure_rotation, pose_difference

SCHEMA_VERSION: Final = 1

#: Ratio of the smallest to the largest eigenvalue of ``M^T @ M`` (see
#: :func:`average_rotations`) below which the weighted rotation sum is treated
#: as too close to singular to invert. Below this the "nearest rotation" is
#: no longer well conditioned - two answers on opposite sides are about
#: equally near - so a fallback is used instead of amplifying noise into a
#: confident-looking wrong answer.
MIN_EIGENVALUE_RATIO: Final = 1e-9

#: Sweeps for the Jacobi eigensolver used by :func:`average_rotations`. Small
#: fixed matrix (3x3, from at most a handful of summed rotations), so this
#: converges to float64 precision well before the budget is spent - see
#: :func:`scanner3d.multiview._jacobi`, which uses the same figure for the
#: same reason.
_JACOBI_SWEEPS: Final = 64

#: Outer IRLS passes ``robust=True`` runs, at most. Each pass is a full
#: relaxation to convergence followed by one reweighting; the loop exits
#: early, well before this ceiling, once the rejected set stops changing
#: between passes - see :func:`optimise_poses`. On this module's own
#: headline robust fixture, recovering cleanly from a handful of edges
#: wrong by half a revolution took 6 passes to stabilise; 8 leaves margin
#: without being an invitation to iterate forever on a graph that will not
#: settle.
DEFAULT_ROBUST_PASSES: Final = 8

#: Robust-sigma cutoff (Cauchy kernel) past which an edge's residual is
#: treated as disagreeing with the consensus. 3 sigma is the conventional
#: bar for "clearly not part of this distribution" on a roughly Gaussian
#: population of residuals; the false edges this feature was built for
#: measured roughly 30x the scale of the honest ones, so this is a
#: deliberately unaggressive default, not a knife's-edge tuning.
DEFAULT_OUTLIER_THRESHOLD: Final = 3.0

#: Fewest edges :func:`_robust_weights` will attempt to find outliers among.
#: "The majority disagrees with this edge" is not a meaningful sentence
#: below a handful of edges, so nothing is reweighted or rejected under this
#: floor - rejection must be earned by disagreement with a real consensus,
#: never manufactured by having few edges to begin with.
MIN_ROBUST_EDGES: Final = 4


class PoseGraphNotAllowed(ScannerError):
    """A pose-graph input was offered that this module refuses to accept."""

    status = DoctorStatus.UNSUPPORTED_MODE


class PoseGraphRobustDisconnected(PoseGraphNotAllowed):
    """Robust rejection would isolate a node the full edge set could reach.

    Raised instead of returning a pose for a node nothing constrains any
    more - see the module docstring's section on robust mode.
    """


@dataclass(frozen=True, slots=True)
class PoseEdge:
    """What ICP measured relating frame ``i``'s local cloud to frame ``j``'s.

    ``transform`` carries a point from frame ``i``'s local coordinates to
    frame ``j``'s - the same convention as a pairwise ICP fit with ``i`` as
    source and ``j`` as target elsewhere in this repository. ``weight`` is a
    caller-chosen trust score - not a physical uncertainty - larger meaning
    "move this node further to satisfy this edge". This module does not
    distinguish a sequential edge from a closure edge; both are exactly this
    struct, and drawing that distinction is entirely the caller's job.

    ``protected`` exempts this edge from ``robust=True`` rejection
    regardless of its residual - it is still fitted normally, only never
    downweighted or reported rejected. A caller who trusts an edge on
    grounds this module cannot see (a sequential edge from chaining,
    say) sets this rather than relying on this module to guess which kind
    of edge it is; it does not, and must not, know or care.
    """

    i: int
    j: int
    transform: Rigid
    weight: float = 1.0
    protected: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "i": self.i,
            "j": self.j,
            "transform": self.transform.as_dict(),
            "weight": self.weight,
            "protected": self.protected,
        }


@dataclass(frozen=True, slots=True)
class EdgeResidual:
    """How far one edge's measurement disagrees with the poses it was checked against.

    ``rotation_error_radians`` and ``translation_error`` are kept separate
    because they are not the same unit and summing them would manufacture a
    precision neither half has on its own - the same reason
    :class:`scanner3d.registration.LoopClosure` reports its two halves
    separately rather than combined.
    """

    edge_index: int
    i: int
    j: int
    rotation_error_radians: float
    translation_error: float

    @property
    def combined(self) -> float:
        """A ranking-only magnitude - see the module docstring's unit note."""
        return math.hypot(self.rotation_error_radians, self.translation_error)

    def as_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "i": self.i,
            "j": self.j,
            "rotation_error_radians": self.rotation_error_radians,
            "translation_error": self.translation_error,
            "combined": self.combined,
        }


@dataclass(frozen=True, slots=True)
class PoseGraphResult:
    """The relaxed poses, how the error moved while getting there, and what remains.

    ``edge_residuals`` is deliberately public and reusable: it is exactly what
    a caller needs to check HELD-OUT edges after optimising with only some of
    them - see :func:`edge_residuals`, which this result's own field is built
    from and which works identically on edges that were never given to
    :func:`optimise_poses` at all.
    """

    poses: tuple[Rigid, ...]
    error_history: tuple[float, ...]
    iterations_run: int
    converged: bool
    anchor: int
    edge_residuals: tuple[EdgeResidual, ...]
    robust: bool
    robust_passes_run: int
    rejected_edges: tuple[EdgeResidual, ...]
    limitations: tuple[str, ...]

    @property
    def initial_error(self) -> float | None:
        return self.error_history[0] if self.error_history else None

    @property
    def final_error(self) -> float | None:
        return self.error_history[-1] if self.error_history else None

    @property
    def worst_edge(self) -> EdgeResidual | None:
        """The edge :func:`optimise_poses` agreed with least, by :attr:`EdgeResidual.combined`.

        A large value here after convergence is not necessarily a fault of
        the optimiser - see the module docstring's note on contradictory
        edges, which cannot be resolved and are reported here instead of
        hidden. Includes rejected edges: this reports what remains true of
        every edge given, not only the ones robust mode kept.
        """
        return max(self.edge_residuals, key=lambda r: r.combined, default=None)

    def as_dict(self) -> dict[str, object]:
        worst = self.worst_edge
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": len(self.poses),
            "edges": len(self.edge_residuals),
            "anchor": self.anchor,
            "iterations_run": self.iterations_run,
            "converged": self.converged,
            "initial_error": self.initial_error,
            "final_error": self.final_error,
            "error_history": list(self.error_history),
            "worst_edge": worst.as_dict() if worst else None,
            "robust": self.robust,
            "robust_passes_run": self.robust_passes_run,
            "rejected_edges": [r.as_dict() for r in self.rejected_edges],
            "limitations": list(self.limitations),
        }


# --- rotation averaging --------------------------------------------------------


def _jacobi(
    matrix: list[list[float]], sweeps: int = _JACOBI_SWEEPS
) -> tuple[list[float], list[list[float]]]:
    """Symmetric eigen-decomposition by cyclic Jacobi rotations.

    The same method and the same reason as
    :func:`scanner3d.multiview._jacobi` and
    :func:`scanner3d.geometry._jacobi_eigen`: unconditionally stable on
    symmetric matrices, needs no starting guess, and this repository has no
    numerical library to reach for instead. Not imported from either sibling
    because both are private to their module; duplicating twenty lines of a
    well-understood, well-tested algorithm is cheaper than coupling three
    modules through a leading-underscore name.
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


def average_rotations(rotations: Sequence[Matrix3], weights: Sequence[float]) -> Matrix3:
    """The rotation nearest, in the chordal sense, to a weighted set of rotations.

    The arithmetic mean of rotation matrices is not itself a rotation - in
    general it is not even orthonormal - so this sums the weighted matrices
    into ``M`` and projects that sum back onto SO(3): the nearest rotation to
    ``M`` is ``M @ (M^T @ M)^(-1/2)``, found from a symmetric eigen-
    decomposition of ``M^T @ M`` the same way
    :func:`scanner3d.multiview.principal_frame` decomposes a covariance
    matrix. This is the one genuinely subtle step :func:`optimise_poses`
    needs: at every node, several incident edges each predict a rotation for
    that node, and this is how those predictions become one.

    Degenerate case
    ----------------
    When the weighted rotations largely cancel - contradictory predictions
    pulling opposite ways - ``M`` approaches a singular matrix and
    ``(M^T @ M)^(-1/2)`` does not exist. There is genuinely no single well-
    defined average there: a 0-degree and a 180-degree rotation about the
    same axis, weighted equally, have two equally valid 90-degree answers on
    opposite sides, not one. Rather than divide by a near-zero eigenvalue and
    return NaN or a non-rotation, this falls back to the single highest-
    weighted input rotation - the measurement the caller trusted most - and
    leaves the unresolved disagreement to show up in the residual reported
    elsewhere, rather than pretending to have resolved it.
    """
    if len(rotations) != len(weights):
        raise PoseGraphNotAllowed(
            f"{len(rotations)} rotations against {len(weights)} weights: one weight per rotation"
        )
    if not rotations:
        raise PoseGraphNotAllowed("cannot average zero rotations")
    for weight in weights:
        if not weight > 0.0:
            raise PoseGraphNotAllowed(f"weight must be positive, got {weight!r}")

    m = [[0.0, 0.0, 0.0] for _ in range(3)]
    for rotation, weight in zip(rotations, weights, strict=True):
        for a in range(3):
            for b in range(3):
                m[a][b] += weight * rotation[a][b]

    mtm = [[math.fsum(m[k][a] * m[k][b] for k in range(3)) for b in range(3)] for a in range(3)]
    values, vectors = _jacobi(mtm)
    largest = max(values)
    smallest = min(values)
    if largest <= 0.0 or smallest < MIN_EIGENVALUE_RATIO * largest:
        best = max(range(len(rotations)), key=lambda k: weights[k])
        return rotations[best]

    # (M^T M)^(-1/2) = V @ diag(1/sqrt(values)) @ V^T.
    inv_diag = [1.0 / math.sqrt(v) for v in values]
    inv_sqrt = [
        [math.fsum(vectors[a][k] * inv_diag[k] * vectors[b][k] for k in range(3)) for b in range(3)]
        for a in range(3)
    ]
    nearest = tuple(
        tuple(math.fsum(m[a][k] * inv_sqrt[k][b] for k in range(3)) for b in range(3))
        for a in range(3)
    )
    return ensure_rotation(nearest)  # type: ignore[arg-type]


def _weighted_translation(translations: Sequence[Point], weights: Sequence[float]) -> Point:
    """The weighted arithmetic mean. Unlike rotation, translation space is flat, so a plain
    weighted mean is already the exact least-squares combination - no projection needed."""
    total_weight = math.fsum(weights)
    return (
        math.fsum(t[0] * w for t, w in zip(translations, weights, strict=True)) / total_weight,
        math.fsum(t[1] * w for t, w in zip(translations, weights, strict=True)) / total_weight,
        math.fsum(t[2] * w for t, w in zip(translations, weights, strict=True)) / total_weight,
    )


# --- edge evaluation ------------------------------------------------------------


def edge_residual(poses: Sequence[Rigid], edge: PoseEdge, *, edge_index: int = 0) -> EdgeResidual:
    """How far ``edge`` disagrees with ``poses``, independent of who computed ``poses``.

    This is the small public seam that makes held-out validation possible:
    call it (or :func:`edge_residuals`) with edges that were never passed to
    :func:`optimise_poses` at all, to check whether optimising against OTHER
    edges also improved agreement with the ones held back. Genuine
    improvement there is the strongest evidence available that a graph is
    fixing the underlying trajectory rather than merely fitting whatever
    constraints it was handed.
    """
    predicted_j = poses[edge.i].compose(edge.transform.inverse())
    residual = pose_difference(predicted_j, poses[edge.j])
    return EdgeResidual(
        edge_index=edge_index,
        i=edge.i,
        j=edge.j,
        rotation_error_radians=residual.rotation_angle,
        translation_error=residual.translation_magnitude,
    )


def edge_residuals(poses: Sequence[Rigid], edges: Sequence[PoseEdge]) -> tuple[EdgeResidual, ...]:
    """:func:`edge_residual` for every edge, indexed as given."""
    return tuple(edge_residual(poses, edge, edge_index=index) for index, edge in enumerate(edges))


def _total_error(
    poses: Sequence[Rigid], edges: Sequence[PoseEdge], weights: Sequence[float]
) -> float:
    """A weighted least-squares cost over every edge's residual, for the convergence check.

    ``weights`` is parallel to ``edges`` rather than read from
    ``edge.weight`` directly, so this doubles as the objective for BOTH
    plain relaxation (``weights`` is exactly ``[e.weight for e in edges]``)
    and a robust pass (``weights`` additionally carries that pass's Cauchy
    multipliers) without two versions of this function existing.

    Squared so that one badly-violated edge dominates the figure the way a
    least-squares objective is supposed to - this is what makes the
    fixed point of relaxation the least-squares solution the module docstring
    claims, not an incidental convenience.
    """
    return math.fsum(
        weight * (residual.rotation_error_radians**2 + residual.translation_error**2)
        for weight, residual in zip(weights, edge_residuals(poses, edges), strict=True)
    )


# --- relaxation -------------------------------------------------------------


def _has_redundant_path(node_count: int, edges: Sequence[PoseEdge]) -> bool:
    """Whether any edge closes a loop rather than merely extending a tree.

    Computed from union-find: an edge that connects two nodes already in the
    same component is redundant with some existing path between them, which
    is exactly what a closure edge is and a purely sequential chain never
    has. Used only to phrase an honest limitation - see the module docstring
    on why a chain with no such edge comes back unchanged.
    """
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        root_i, root_j = find(edge.i), find(edge.j)
        if root_i == root_j:
            return True
        parent[root_i] = root_j
    return False


def _reachable_from(
    anchor: int, node_count: int, edges: Sequence[PoseEdge], *, exclude: frozenset[int]
) -> frozenset[int]:
    """Every node reachable from ``anchor`` using every edge except ``exclude``.

    Union-find again, this time to answer a different question than
    :func:`_has_redundant_path`: not "does a loop exist" but "which nodes
    does this specific edge SET connect the anchor to". Used to check
    whether robust rejection disconnects a node the FULL edge set could
    still reach - see :func:`optimise_poses`.
    """
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for index, edge in enumerate(edges):
        if index in exclude:
            continue
        root_i, root_j = find(edge.i), find(edge.j)
        if root_i != root_j:
            parent[root_i] = root_j
    anchor_root = find(anchor)
    return frozenset(node for node in range(node_count) if find(node) == anchor_root)


def _relax(
    poses_in: Sequence[Rigid],
    edges: Sequence[PoseEdge],
    adjacency: dict[int, list[tuple[int, PoseEdge, bool]]],
    *,
    anchor: int,
    iterations: int,
    tolerance: float,
    weights: Sequence[float],
) -> tuple[list[Rigid], list[float], bool]:
    """Gauss-Seidel relaxation, exactly as :func:`optimise_poses` always ran it.

    ``weights`` is parallel to ``edges`` and is what every node update and
    :func:`_total_error` call actually reads - never ``edge.weight``
    directly - which is what lets a robust pass reuse this unchanged with
    Cauchy-adjusted weights while plain relaxation reuses it with
    ``[e.weight for e in edges]`` and gets bit-identical behaviour to before
    this function existed.

    Returns the resulting poses, the total-error value AFTER each sweep (the
    caller supplies the starting value; this never repeats it), and whether
    it converged inside the ``iterations`` budget.
    """
    poses = list(poses_in)
    history: list[float] = []
    previous = _total_error(poses, edges, weights)
    converged = False
    for _ in range(iterations):
        for node in range(len(poses)):
            if node == anchor:
                continue
            incident = adjacency.get(node)
            if not incident:
                continue
            rotations: list[Matrix3] = []
            translations: list[Point] = []
            node_weights: list[float] = []
            for edge_index, edge, node_is_j in incident:
                other = poses[edge.i] if node_is_j else poses[edge.j]
                predicted = (
                    other.compose(edge.transform.inverse())
                    if node_is_j
                    else other.compose(edge.transform)
                )
                rotations.append(predicted.rotation)
                translations.append(predicted.translation)
                node_weights.append(weights[edge_index])
            poses[node] = Rigid.of(
                average_rotations(rotations, node_weights),
                _weighted_translation(translations, node_weights),
            )
        current = _total_error(poses, edges, weights)
        history.append(current)
        if previous - current < tolerance:
            converged = True
            break
        previous = current
    return poses, history, converged


def _robust_weights(
    edges: Sequence[PoseEdge], residuals: Sequence[EdgeResidual], *, threshold: float
) -> tuple[list[float], frozenset[int]]:
    """One IRLS reweighting pass: a Cauchy kernel on a median-scaled combined residual.

    Recomputed from each edge's ORIGINAL ``weight`` every call, not the
    previous pass's weight - standard IRLS, and what keeps this from
    compounding an early bad guess across passes: a pass that looked wrong
    early is given a full, fresh chance once the poses have moved.

    The scale is ``1.4826 * median`` of every edge's
    :attr:`EdgeResidual.combined` - NOT the textbook ``1.4826 * median
    absolute deviation around the median``, and the difference is
    deliberate and measured, not a simplification. MAD-around-the-median is
    calibrated for a SYMMETRIC, two-sided population; ``combined`` is a
    ``hypot`` of two quantities that are never negative, so its population is
    already one-sided (a folded/half-normal shape) even when every edge is
    honest measurement noise around a shared truth. Applying the two-sided
    formula to a one-sided population systematically UNDERESTIMATES the
    scale, and with fifteen to thirty edges - an ordinary revisit count -
    that miscalibration was measured to falsely flag as many as 7 of 30
    edges drawn from a single, honest noise distribution with no outlier in
    it at all, across ten different random seeds. For a folded-normal
    population the median itself is already ``0.6745 * sigma``, so
    ``1.4826 * median`` is the correspondingly calibrated scale, and it
    measured ZERO false rejections across the same seeds. This is what
    keeps :func:`optimise_poses` from discarding good data merely because
    "distance-shaped" residuals cannot be negative - see the honest-input
    robust tests in ``tests/test_posegraph.py``.

    A robust estimate of spread stays small as long as a MAJORITY of edges
    agree, however small a minority disagrees - which is what makes this
    earn rejection through disagreement rather than through being
    outnumbered. Below :data:`MIN_ROBUST_EDGES` ELIGIBLE (unprotected)
    edges there is no majority to speak of, and nothing is reweighted.

    The scale is computed ONLY over unprotected edges' residuals - NOT over
    every edge, which is a real bug this module shipped with and then fixed,
    worth recording so it is not reintroduced. A protected edge is exempt
    from rejection by definition, so folding it into the statistic that
    decides what gets rejected is a category error, and in real usage it is
    an actively misleading one: a caller typically protects the sequential
    edges that were themselves COMPOSED to build the very poses being fit,
    so those edges are satisfied to float64 precision by construction, not
    by being especially trustworthy measurements. Measured on the user's
    real 86-edge graph (69 protected sequential edges at ~0 residual, 17
    closures including 3 falsely claiming 179-degree pairs were the same
    viewpoint), including the protected edges pulled the median to ~0,
    every closure - the honest 0.5-8 degree ones included - looked like an
    astronomical outlier against that floor, and NOTHING was rejected: the
    exact opposite of the intended behaviour, on exactly the data this
    module exists for. The question the scale answers is "what does a
    typical disagreement look like among the things I might reject", and a
    population that cannot be rejected by definition cannot inform that
    answer, however real its own residual is as evidence of something else.

    Cauchy over Huber: measured on this module's own headline fixture with a
    handful of edges wrong by half a revolution, Huber's ``threshold /
    normalised`` falloff was not aggressive enough - a wrong edge kept
    10-15% of its weight even far past the threshold, and Gauss-Seidel
    relaxation propagates that residual influence to its neighbours each
    sweep, which repeatedly re-contaminated the very residuals the next
    pass reweights from. A grossly wrong edge on a Gauss-Seidel graph needs
    to be suppressed towards zero, not merely discounted. Cauchy's
    ``threshold^2 / (threshold^2 + normalised^2)`` falls off quadratically
    instead of linearly - roughly 2% of original weight at 20 robust sigma
    against Huber's 15% - and that difference was the one that let robust
    fitting actually recover the clean result instead of merely improving
    on the contaminated one; see the recovery numbers in
    ``tests/test_posegraph.py``'s headline robust test.

    A :attr:`PoseEdge.protected` edge is still returned at its original
    weight, unconditionally, and still has its own residual computed and
    reported by the caller - it is simply excluded from the population that
    decides what everyone else's "normal" looks like, for the reason above.
    """
    eligible = [
        residual.combined
        for edge, residual in zip(edges, residuals, strict=True)
        if not edge.protected
    ]
    if len(eligible) < MIN_ROBUST_EDGES:
        # Either too few edges overall, or too few left once protected ones
        # are excluded - up to and including every edge being protected, in
        # which case there is no eligible population at all, not merely a
        # small one. Either way "the majority disagrees with this edge" is
        # not a meaningful sentence, so nothing is reweighted - the same
        # honest floor as the too-few-edges-overall case always was.
        return [edge.weight for edge in edges], frozenset()

    median = statistics.median(eligible)
    # A small absolute floor, not a claim about physical scale: it stops a
    # graph that has already converged to float64 precision (every eligible
    # residual within rounding noise of zero) from amplifying that noise
    # into meaningless "normalised" ratios and rejecting edges arbitrarily -
    # the same purpose scanner3d.geometry.ORTHONORMAL_TOLERANCE serves for a
    # different near-zero degeneracy.
    scale = max(1.4826 * median, 1e-9)

    threshold_squared = threshold * threshold
    new_weights: list[float] = []
    rejected: set[int] = set()
    for index, (edge, residual) in enumerate(zip(edges, residuals, strict=True)):
        if edge.protected:
            new_weights.append(edge.weight)
            continue
        normalised = residual.combined / scale
        new_weights.append(edge.weight * (threshold_squared / (threshold_squared + normalised**2)))
        if normalised > threshold:
            rejected.add(index)
    return new_weights, frozenset(rejected)


def optimise_poses(
    nodes: Sequence[Rigid],
    edges: Sequence[PoseEdge],
    *,
    anchor: int = 0,
    iterations: int = 50,
    tolerance: float = 1e-9,
    robust: bool = False,
    robust_passes: int = DEFAULT_ROBUST_PASSES,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> PoseGraphResult:
    """Relax ``nodes`` so they agree with ``edges`` better, holding ``anchor`` fixed.

    Repeatedly sweeps every node but ``anchor`` in index order; at each one,
    every incident edge predicts that node's pose from its neighbour's
    CURRENT estimate (Gauss-Seidel: predictions within a sweep see updates
    already made earlier in the same sweep), and the predictions are combined
    - rotations by :func:`average_rotations`, translations by a plain
    weighted mean - into the node's new pose. See the module docstring for
    why this is the right tool given no numerical library and a few hundred
    poses, and for what it cannot do. See :func:`_relax` for the sweep loop
    itself, factored out so plain and robust fitting share one implementation.

    ``anchor`` is never written to, at any point, by anything - not moved and
    then corrected back, simply never touched - so it comes back bit-for-bit
    identical to ``nodes[anchor]``. A node with no incident edge at all is
    likewise left exactly where it started, because nothing predicts a pose
    for it; that is correct behaviour, not a gap.

    Stops after ``iterations`` sweeps, or earlier once one sweep improves the
    total weighted squared residual (see :func:`optimise_poses`'s error
    metric) by less than ``tolerance``, whichever comes first.

    Robust mode
    ------------
    With ``robust=True``, up to ``robust_passes`` rounds of [relax to
    convergence, reweight by Cauchy kernel at ``outlier_threshold`` robust
    sigmas - see :func:`_robust_weights`] run, stopping early once the set of
    rejected edges stops changing between rounds. See the module docstring's
    section on robust mode for why this exists and what it guarantees:
    ``robust=False`` (the default) is bit-identical to this module before
    the feature existed; a run that rejects nothing behaves like plain
    relaxation; every rejected edge is named in
    :attr:`PoseGraphResult.rejected_edges`; and a rejection set that would
    disconnect a node the full edge set could reach from ``anchor`` raises
    :exc:`PoseGraphRobustDisconnected` instead of returning a partial answer.
    """
    if len(nodes) < 2:
        raise PoseGraphNotAllowed(f"{len(nodes)} node(s) is not enough to have a relative pose")
    if not 0 <= anchor < len(nodes):
        raise PoseGraphNotAllowed(f"anchor {anchor} is not a node index in range(0, {len(nodes)})")
    if iterations < 1:
        raise PoseGraphNotAllowed(f"iterations must be at least 1, got {iterations}")
    if tolerance < 0.0:
        raise PoseGraphNotAllowed(f"tolerance must be non-negative, got {tolerance}")
    if robust_passes < 1:
        raise PoseGraphNotAllowed(f"robust_passes must be at least 1, got {robust_passes}")
    if not math.isfinite(outlier_threshold) or outlier_threshold <= 0.0:
        raise PoseGraphNotAllowed(
            f"outlier_threshold must be finite and positive, got {outlier_threshold!r}"
        )
    for index, edge in enumerate(edges):
        if not 0 <= edge.i < len(nodes) or not 0 <= edge.j < len(nodes):
            raise PoseGraphNotAllowed(
                f"edge {index} references node {edge.i} or {edge.j}, out of range(0, {len(nodes)})"
            )
        if edge.i == edge.j:
            raise PoseGraphNotAllowed(f"edge {index} relates node {edge.i} to itself")
        if not edge.weight > 0.0:
            raise PoseGraphNotAllowed(f"edge {index} has non-positive weight {edge.weight!r}")

    adjacency: dict[int, list[tuple[int, PoseEdge, bool]]] = {}
    for index, edge in enumerate(edges):
        adjacency.setdefault(edge.i, []).append((index, edge, False))
        adjacency.setdefault(edge.j, []).append((index, edge, True))

    base_weights = [edge.weight for edge in edges]
    poses = list(nodes)
    history = [_total_error(poses, edges, base_weights)]
    weights = base_weights
    passes_run = 0
    rejected: frozenset[int] = frozenset()
    # Eligible, not total: a protected edge can never be rejected, so it
    # cannot help meet the floor that decides whether attempting robust
    # fitting is worthwhile at all - see _robust_weights on why the same
    # distinction matters for the scale itself, not only for this count.
    eligible_edge_count = sum(1 for edge in edges if not edge.protected)
    attempt_robust = robust and eligible_edge_count >= MIN_ROBUST_EDGES

    if not attempt_robust:
        poses, sweep_history, converged = _relax(
            poses,
            edges,
            adjacency,
            anchor=anchor,
            iterations=iterations,
            tolerance=tolerance,
            weights=weights,
        )
        history.extend(sweep_history)
    else:
        previous_rejected: frozenset[int] | None = None
        converged = False
        for _ in range(robust_passes):
            poses, sweep_history, converged = _relax(
                poses,
                edges,
                adjacency,
                anchor=anchor,
                iterations=iterations,
                tolerance=tolerance,
                weights=weights,
            )
            history.extend(sweep_history)
            passes_run += 1
            weights, rejected = _robust_weights(
                edges, edge_residuals(poses, edges), threshold=outlier_threshold
            )
            if rejected == previous_rejected:
                break
            previous_rejected = rejected
        # One more relax with the settled weights, so the poses reported
        # reflect exactly the rejection decision being reported, not the
        # pass before it.
        poses, sweep_history, converged = _relax(
            poses,
            edges,
            adjacency,
            anchor=anchor,
            iterations=iterations,
            tolerance=tolerance,
            weights=weights,
        )
        history.extend(sweep_history)

        if rejected:
            full_reach = _reachable_from(anchor, len(nodes), edges, exclude=frozenset())
            surviving_reach = _reachable_from(anchor, len(nodes), edges, exclude=rejected)
            newly_isolated = full_reach - surviving_reach
            if newly_isolated:
                raise PoseGraphRobustDisconnected(
                    f"rejecting edge(s) {sorted(rejected)} disconnects node(s) "
                    f"{sorted(newly_isolated)} from anchor {anchor}, though the full edge set "
                    "could reach them; refusing rather than returning a pose for a node "
                    "nothing constrains any more. Raise outlier_threshold, protect more "
                    "edges, or review the rejected edges before retrying"
                )

    limitations = [
        "This is Gauss-Seidel relaxation on the chordal pose-graph objective, not a joint "
        "linear solve or bundle adjustment: each node repeatedly moves to what its edges "
        "currently predict. It is not guaranteed to reach the global optimum from an "
        "arbitrarily bad start, though correcting a mild systematic bias distributed around "
        "a loop - the case this module exists for - is exactly what it is good at.",
        "A node with no incident edge is never moved, because nothing predicts a pose for "
        "it; it is returned exactly as given, not a failure of relaxation.",
        "No physical unit is implied anywhere. A weight is a caller-chosen trust score, not "
        "a physical uncertainty, and EdgeResidual.combined mixes a rotation angle in radians "
        "with a translation in the caller's unit purely to rank edges within one run.",
        "This module cannot tell a genuine loop closure from a genuinely contradictory pair "
        "of edges that cannot both hold; it converges to a compromise either way. A large "
        "worst_edge after convergence can mean either a hard graph or bad input - it does "
        "not by itself distinguish them.",
    ]
    if not _has_redundant_path(len(nodes), edges):
        limitations.append(
            "No edge in this graph closes a loop or otherwise duplicates a path between two "
            "nodes already connected by another edge - every component is a tree. A tree has "
            "nothing for relaxation to reconcile: every edge is already satisfied, to first "
            "order, by the chain that produced it, so this graph comes back essentially "
            "unchanged. That is expected, not a defect - see the module docstring."
        )
    if robust and not attempt_robust:
        limitations.append(
            f"robust=True was requested but only {eligible_edge_count} edge(s) were eligible "
            f"for rejection (not protected), below the {MIN_ROBUST_EDGES} this needs before "
            "'the majority disagrees with this edge' is a meaningful sentence; no edge was "
            "reweighted or rejected, and this ran exactly as robust=False would have."
        )
    elif robust:
        limitations.append(
            f"Robust mode ran {passes_run} reweighting pass(es) (IRLS, Cauchy kernel, "
            f"{outlier_threshold} robust sigma(s) on the median-scaled combined residual, "
            f"combined residual) and rejected {len(rejected)} of {len(edges)} edge(s) - see "
            "rejected_edges for which and why. A protected edge is never reweighted or "
            "rejected regardless of its residual."
        )

    final_residuals = edge_residuals(poses, edges)
    return PoseGraphResult(
        poses=tuple(poses),
        error_history=tuple(history),
        iterations_run=len(history) - 1,
        converged=converged,
        anchor=anchor,
        edge_residuals=final_residuals,
        robust=robust,
        robust_passes_run=passes_run,
        rejected_edges=tuple(final_residuals[index] for index in sorted(rejected)),
        limitations=tuple(limitations),
    )
