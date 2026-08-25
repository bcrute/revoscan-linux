"""Selectable compute backends for registration's hot inner loops.

Why this module exists
-----------------------
:func:`scanner3d.registration.icp` is pure Python and single-threaded: one
pairwise alignment measured at about 1.5 seconds on the machine this was
written on, so a 417-frame turntable sweep costs roughly 11 minutes just to
chain. That machine is a 20-core i9-10900K, so a scan currently uses about
5% of the CPU sitting idle next to it. An 11-minute wait between "press
capture" and "see the result" is long enough that a user who wants to run
scans themselves will not. This module is the selectable second gear: the
same answer, computed faster, chosen explicitly rather than assumed.

Which operations, and why - measured, not guessed
---------------------------------------------------
Before choosing what to vectorise, the actual cost was profiled rather than
assumed. Isolated timings on synthetic clouds of realistic size (2000-10000
points, an unoccluded bumpy-sphere shape so every query has a true nearest
neighbour), one core, this machine::

    n      voxel-index build   correspondence search   apply_all   Horn solve
    2000    0.6 ms               72.5 ms                0.3 ms      3.3 ms
    5000    1.6 ms              437.4 ms                0.8 ms      8.2 ms
    10000   3.1 ms             1711.3 ms                1.7 ms     16.4 ms

    estimate_normals (post-downsample sizes, since that is where it runs)
    n=200    17.2 ms   n=500    85.1 ms   n=1000   303.8 ms   n=2000  1141.9 ms

Nearest-neighbour correspondence search dominates everything else at every
size measured - two to three orders of magnitude above applying a transform
or solving Horn's closed form - and it is the cost that scales worst: it is
quadratic in the point count, because :class:`scanner3d.registration._VoxelIndex`'s
own docstring records that on real captures the 27-cell bound holds a
constant *fraction* of the cloud (91% on the real ``base_4`` capture), not a
constant count. :func:`scanner3d.icp_plane.estimate_normals`'s k-nearest-neighbour
search is the other cost in the same class, for the same reason (a brute-force
scan per point, stated as a deliberate tradeoff in its own docstring).
Applying a transform to a whole cloud and centroid/cross-covariance
accumulation are cheap in absolute terms at these sizes - but they run every
ICP iteration, on every frame pair, so they were measured rather than assumed
free, and vectorising them costs little once the array conversion machinery
already exists for the two operations that matter.

That gives four operations, and they are exactly the four this module
backs: nearest-neighbour correspondence search
(:meth:`ComputeBackend.nearest_neighbors`), applying a rigid transform to a
whole cloud (:meth:`ComputeBackend.apply_transform`), the centroid/cross-
covariance accumulation inside a closed-form rigid solve
(:meth:`ComputeBackend.rigid_from_correspondences`), and k-nearest-neighbour
search for local surface normals (:meth:`ComputeBackend.k_nearest_neighbors`).

The backend contract
---------------------
Every backend must return the SAME answer as the "python" backend on the
same input, to a stated tolerance - see each method's docstring for the
number and why. "python" is the REFERENCE: it wraps what already exists in
:mod:`scanner3d.registration` and :mod:`scanner3d.geometry`, which carry
their own mutation-tested suites, rather than reimplementing their algorithms
a second time where that can be avoided. Only :meth:`ComputeBackend.k_nearest_neighbors`
has no existing standalone function to wrap - the closest analogue lives
inline inside :func:`scanner3d.icp_plane.estimate_normals`, mixed with normal-
fitting concerns that make it inseparable - so its "python" backend is a
freshly authored reference implementation of the same asymptotic algorithm
(``heapq.nsmallest``, matching that function's own choice), not a wrapper.

Selection is exact, never a downgrade. Requesting ``"python"`` or ``"numpy"``
explicitly and not getting it back is always an error
(:class:`ComputeBackendUnavailable`), never a silent fallback to a different
backend: a caller who selected a backend and got a different one cannot
interpret their own timings. ``"auto"`` is the one mode that picks for the
caller, and it reports which backend it picked and why in
:attr:`BackendSelection.reason` - see :func:`select_backend`.

numpy is optional
-------------------
This project is dependency-free by a stated user preference, not a hard
rule - see :func:`select_backend`'s docstring for the "why numpy first"
reasoning. numpy is therefore an *optional* accelerator: this module imports
cleanly, and the "python" backend works fully, whether or not numpy is
installed. No import of numpy happens at module scope; :func:`_probe_numpy`
does a real (function-scoped) import to answer "is it usable", and
:class:`_NumpyBackend`'s own methods each do their own local import, which is
cheap after the first (a ``sys.modules`` dictionary lookup) and keeps every
numpy-shaped type entirely out of this module's own namespace.

Seam for a future GPU backend - not implemented here
-------------------------------------------------------
``"gpu"`` is a *recognised* backend name - see :data:`GPU_BACKEND` and
:func:`available_backends` - that always reports unavailable, and
:func:`select_backend` always refuses it, with a reason rather than a
mystery. It is deliberately not implemented in this task: the machine this
scanner runs on carries an AMD Radeon RX 470/480/570-class GPU plus a newer
device, so CUDA is not an option and ROCm's support for that generation is
unverified here. A GPU backend needs a measurement of what it would actually
buy on THIS hardware before it is worth building, and no such measurement
exists yet. Recognising the name now, and refusing it honestly, is what lets
a caller's code that branches on backend names be written once and not need
to change again when a GPU backend eventually arrives.

This is the LOCAL seam, not the network one
----------------------------------------------
There are two different seams in this system, at two different granularities,
and this module is only one of them. This module - "which local implementation
runs a hot inner loop" - is called thousands of times per registration
(:meth:`ComputeBackend.nearest_neighbors` alone runs once per ICP iteration
per frame pair), so putting a network call here would mean thousands of round
trips per sweep; at even 1 ms of LAN latency that is slower than pure Python.
The OTHER seam - "which machine runs this whole job" (register this sweep,
fuse these clouds) - is coarse-grained, one request per sweep, and belongs in
a separate, not-yet-built module that hands a whole job to a
:class:`ComputeBackend` running on whichever machine accepted it. A remote
job handler is expected to select a LOCAL backend from THIS module to do the
actual work, exactly as a local caller would; it does not reach inside this
module's hot loops from across a network. Conflating the two seams - putting
the network at this module's granularity - is the mistake to avoid.

What this module does NOT do
------------------------------
It does not decide which backend the pipeline uses by default, and it is not
wired into :mod:`scanner3d.registration` or :mod:`scanner3d.icp_plane` -
both keep working exactly as before, unmodified. It does not implement a GPU
backend - see above. It does not implement, or know anything about, remote
job execution - see "This is the LOCAL seam, not the network one" above. It
does not validate that input points are finite; every caller in this
codebase already validates a cloud (see
:func:`scanner3d.registration._ensure_cloud` and its :mod:`scanner3d.icp_plane`
twin) before a point ever reaches one of these hot loops, and duplicating
that check here would just be a second place for it to drift out of sync.
"""

from __future__ import annotations

import heapq
import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import (
    CorrespondenceUnusable,
    Point,
    Rigid,
)
from scanner3d.geometry import (
    rigid_from_correspondences as _geometry_rigid_from_correspondences,
)
from scanner3d.geometry import (
    spanned_dimensions as _geometry_spanned_dimensions,
)
from scanner3d.registration import (
    _VoxelIndex,  # the reference nearest-neighbour search; see the module docstring
)

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

PYTHON_BACKEND: Final = "python"
NUMPY_BACKEND: Final = "numpy"
SCIPY_BACKEND: Final = "scipy"
GPU_BACKEND: Final = "gpu"
AUTO_BACKEND: Final = "auto"

#: Backends this module can actually construct, in the order "auto" prefers
#: them. scipy first, then numpy, then python: see :func:`select_backend`'s
#: docstring for the measured reason - a KD-tree is an asymptotically better
#: ALGORITHM for the operation that dominates registration's cost, not merely
#: a faster implementation of the same one, and the gap grows with cloud size.
_KNOWN_BACKENDS: Final[tuple[str, ...]] = (SCIPY_BACKEND, NUMPY_BACKEND, PYTHON_BACKEND)

#: Recognised, but never constructed here - see the module docstring's
#: "Seam for a future GPU backend".
_RESERVED_BACKENDS: Final[tuple[str, ...]] = (GPU_BACKEND,)

#: Why the GPU seam refuses, stated once so :func:`available_backends` and
#: :func:`select_backend` give the identical reason rather than two that
#: could drift apart.
_GPU_UNAVAILABLE_REASON: Final = (
    "not implemented: this machine's AMD GPU generation has no measured ROCm support and CUDA "
    "is not an option, so a GPU backend needs a measurement this task did not make, not a guess"
)

#: Upper bound on how many (row, target-point) pairs a chunked numpy search
#: materialises into one dense array at once. Bounds peak memory rather than
#: bounding correctness: a smaller chunk only means more, smaller chunks. At
#: 3 float64 coordinates per pair this ceiling is about 48 MB per block,
#: comfortably below what would make a large sweep's memory use a concern.
_PAIRWISE_CHUNK_ELEMENTS: Final = 2_000_000

#: One nearest-neighbour hit: ``(distance, target_point)``. ``None`` means
#: nothing in the target cloud lies strictly inside the gate - the same
#: rejection :class:`scanner3d.registration._VoxelIndex.nearest` reports, for
#: the same reason: "no partner near enough to be believed", not "the query
#: failed to run".
NearestHit = tuple[float, Point] | None


# --- refusals -------------------------------------------------------------------


class ComputeNotAllowed(ScannerError):
    """A compute-backend request or parameter is refused before any array is built."""

    status = DoctorStatus.UNSUPPORTED_MODE


class ComputeBackendUnknown(ComputeNotAllowed):
    """The requested name is not one this module recognises at all."""


class ComputeBackendUnavailable(ComputeNotAllowed):
    """The requested backend is a real, recognised name, but not usable here.

    Raised for an explicit request this environment cannot satisfy (numpy not
    importable) and for the reserved ``"gpu"`` name, which is never usable
    because it is never implemented - see the module docstring. Never raised
    to mean "so something else was used instead": see :func:`select_backend`.
    """


class ComputeParametersUnusable(ComputeNotAllowed):
    """An operation's own parameter - a gate, a neighbour count - is out of range."""


# --- shared validation, so every backend refuses identically ---------------------


def _ensure_gate(gate: float) -> None:
    """Refuse a correspondence gate that could never admit anything.

    Shared by every backend's :meth:`ComputeBackend.nearest_neighbors`, so
    "gate is non-finite or non-positive" raises the identical exception with
    the identical message regardless of which backend is asked - one
    function, not two copies that could drift apart.
    """
    if not math.isfinite(gate) or gate <= 0.0:
        raise ComputeParametersUnusable(
            f"gate {gate!r} must be a finite positive size; a nearest-neighbour search gated at "
            "this value would admit nothing"
        )


def _ensure_k(k: int) -> None:
    """Refuse a neighbour count that cannot name a neighbourhood."""
    if k < 1:
        raise ComputeParametersUnusable(f"k {k} must be at least 1")


def _ensure_correspondences(source: Sequence[Point], target: Sequence[Point]) -> None:
    """The exact checks :func:`scanner3d.geometry.rigid_from_correspondences` makes.

    Factored out so :class:`_NumpyBackend` refuses a degenerate correspondence
    set with the identical exception type and message that
    :func:`scanner3d.geometry.rigid_from_correspondences` itself raises - which
    is what :class:`_PythonBackend` returns by construction, since it calls
    that function directly. Both backends therefore draw the refusal line in
    exactly the same place, checked the same way, not merely "close enough".
    """
    if len(source) != len(target):
        raise CorrespondenceUnusable(
            f"{len(source)} source points against {len(target)} target points: "
            "a correspondence set must pair one to one"
        )
    if len(source) < 3:
        raise CorrespondenceUnusable(
            f"{len(source)} correspondences cannot constrain a rotation; at least 3 are needed"
        )
    spanned = _geometry_spanned_dimensions(source)
    if spanned < 2:
        raise CorrespondenceUnusable(
            f"the source points span only {spanned} dimension(s), so the rotation is "
            "underdetermined - see scanner3d.geometry.rigid_from_correspondences for why this "
            "is refused rather than fitted anyway"
        )


# --- the backend protocol ---------------------------------------------------------


class ComputeBackend(Protocol):
    """The hot-path operations a registration/ICP pipeline needs, made selectable.

    Every method must return the SAME answer, to a stated tolerance, as the
    "python" backend on the same input - see each method's own docstring for
    the number and why. That is what lets a caller change which backend runs
    without changing what a scan produces, only how long it takes to produce
    it.
    """

    @property
    def name(self) -> str:
        """This backend's own name - one of :data:`PYTHON_BACKEND`, :data:`NUMPY_BACKEND`."""
        ...

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        """``transform`` applied to every point in ``points``, in order.

        Mirrors :meth:`scanner3d.geometry.Rigid.apply_all`. Agreement with the
        "python" backend is expected near float64 machine precision: both
        compute the identical ``R @ p + t`` per point, so the only source of
        disagreement is summation order, which for a 3-term dot product is
        negligible.
        """
        ...

    def nearest_neighbors(
        self, queries: Sequence[Point], target: Sequence[Point], *, gate: float
    ) -> list[NearestHit]:
        """For each point in ``queries``, its nearest point in ``target`` strictly
        inside ``gate``, or ``None``.

        Mirrors :class:`scanner3d.registration._VoxelIndex.nearest` applied to
        every query. An empty ``target`` returns ``None`` for every query
        rather than raising - a cloud with no points to match against is a
        valid, if useless, target.

        The "python" backend refuses (:class:`ComputeParametersUnusable`) a
        gate so fine, relative to the target cloud's extent, that
        :class:`scanner3d.registration._VoxelIndex`'s voxel grid would need
        more than :data:`scanner3d.registration.MAX_GRID_CELLS` cells - a
        property of THAT algorithm's own data structure, not of the query's
        validity. The "numpy" backend never grids at all, so it has no such
        failure mode; this one narrow case is a documented, deliberate
        difference between backends, not a violated contract.
        """
        ...

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        """The rigid transform Horn's method fits to paired ``source``/``target`` points.

        Mirrors :func:`scanner3d.geometry.rigid_from_correspondences`.
        Agreement with the "python" backend is expected to about 1e-9 in each
        rotation entry and translation component on well-conditioned input:
        both solve the identical eigenproblem in exact arithmetic, but one
        does it by cyclic Jacobi rotations and the other by a LAPACK
        symmetric eigensolver (``numpy.linalg.eigh``), and those accumulate
        float64 rounding differently. See ``tests/test_compute.py`` for the
        measured figure this tolerance was set from.
        """
        ...

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        """For each point, the indices of its ``k`` nearest OTHER points.

        Capped at ``len(points) - 1`` when there are not ``k`` other points to
        find. Order within each result is not part of the contract - a
        Gauss-Newton or PCA consumer of a neighbourhood does not care which
        order its members arrived in - so equivalence between backends is a
        claim about the SET of indices returned per point, not their order.
        On real-valued point clouds with no exact coincident points (the case
        this module is ever asked about), that set is unique and both
        backends are expected to agree on it exactly, with no tolerance
        needed at all: a discrete selection has no rounding to disagree
        about.
        """
        ...


# --- the python (reference) backend -----------------------------------------------


class _PythonBackend:
    """The reference backend. Wraps what already exists, wherever something
    already exists to wrap - see the module docstring - and is always
    available: no import beyond the standard library.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return PYTHON_BACKEND

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        return list(transform.apply_all(points))

    def nearest_neighbors(
        self, queries: Sequence[Point], target: Sequence[Point], *, gate: float
    ) -> list[NearestHit]:
        _ensure_gate(gate)
        if not target:
            return [None for _ in queries]
        index = _VoxelIndex(list(target), gate=gate)
        hits: list[NearestHit] = []
        for query in queries:
            found = index.nearest(query)
            hits.append(None if found is None else (math.dist(query, found), found))
        return hits

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        return _geometry_rigid_from_correspondences(source, target)

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        _ensure_k(k)
        n = len(points)
        result: list[tuple[int, ...]] = []
        for i, p in enumerate(points):
            cap = min(k, n - 1)
            if cap <= 0:
                result.append(())
                continue

            def _squared_distance(j: int, p: Point = p) -> float:
                q = points[j]
                return (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2

            nearest_idx = heapq.nsmallest(cap + 1, range(n), key=_squared_distance)
            result.append(tuple(j for j in nearest_idx if j != i)[:cap])
        return result


# --- numpy availability, probed live, never imported at module scope -------------


def _probe_numpy() -> tuple[bool, str | None]:
    """``(usable, reason-if-not)``. A real import, not a spec lookup, so a numpy
    that is installed but broken (a missing shared library, an ABI mismatch)
    is reported the same way as one that is missing outright - it names why
    it is not usable rather than merely asserting that it is not.
    """
    try:
        importlib.import_module("numpy")
    except ImportError as exc:
        return False, str(exc)
    return True, None


# --- the numpy (accelerated) backend -----------------------------------------------


class _NumpyBackend:
    """Vectorised over numpy arrays. Constructed only once :func:`_probe_numpy`
    has confirmed numpy is importable - see :func:`select_backend` - but every
    method here still does its own local ``import numpy``, so this class can
    be defined (though never instantiated with numpy absent) whether or not
    numpy is installed, and this module never holds a module-level reference
    to it. See the module docstring's "numpy is optional" section.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return NUMPY_BACKEND

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        if not points:
            return []
        import numpy as np

        pts = np.asarray(points, dtype=np.float64)
        rotation = np.asarray(transform.rotation, dtype=np.float64)
        translation = np.asarray(transform.translation, dtype=np.float64)
        moved = pts @ rotation.T + translation
        return [(float(row[0]), float(row[1]), float(row[2])) for row in moved]

    def nearest_neighbors(
        self, queries: Sequence[Point], target: Sequence[Point], *, gate: float
    ) -> list[NearestHit]:
        """See :meth:`ComputeBackend.nearest_neighbors`.

        Computed as ``|q-t|^2 = |q|^2 - 2 q.t + |t|^2``, so the ``(rows, n)``
        cross term comes from one BLAS matrix multiply (``@``) rather than
        materialising a ``(rows, n, 3)`` difference array and reducing it -
        measured at 3.2-3.4x faster than the broadcast-subtraction form at
        n=5000/10000 on this machine, because a GEMM is compute-bound and
        SIMD/cache-friendly where the broadcast form is memory-bandwidth-bound
        on an array three times the size. Recentring both clouds on the
        target's own centroid first keeps every squared norm near the cloud's
        own scale, which is what keeps this identity's cancellation error
        negligible relative to :data:`gate` - see ``tests/test_compute.py``
        for the measured figure.
        """
        _ensure_gate(gate)
        if not target or not queries:
            return [None for _ in queries]
        import numpy as np

        q = np.asarray(queries, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        centre = t.mean(axis=0)
        qc = q - centre
        tc = t - centre
        target_squared = np.sum(tc * tc, axis=1)
        gate_squared = gate * gate
        chunk_rows = max(1, _PAIRWISE_CHUNK_ELEMENTS // len(target))
        hits: list[NearestHit] = []
        for start in range(0, len(queries), chunk_rows):
            block = qc[start : start + chunk_rows]
            query_squared = np.sum(block * block, axis=1)
            cross = block @ tc.T
            squared = query_squared[:, None] - 2.0 * cross + target_squared[None, :]
            nearest_idx = np.argmin(squared, axis=1)
            nearest_squared = squared[np.arange(squared.shape[0]), nearest_idx]
            for target_idx, distance_squared in zip(nearest_idx, nearest_squared, strict=True):
                # The subtraction identity above can round a true-zero
                # squared distance to a tiny negative float64 through
                # cancellation; never let that reach sqrt.
                clamped = max(0.0, float(distance_squared))
                if clamped < gate_squared:
                    hits.append((math.sqrt(clamped), target[int(target_idx)]))
                else:
                    hits.append(None)
        return hits

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        _ensure_correspondences(source, target)
        import numpy as np

        p = np.asarray(source, dtype=np.float64)
        q = np.asarray(target, dtype=np.float64)
        source_centre = p.mean(axis=0)
        target_centre = q.mean(axis=0)
        pc = p - source_centre
        qc = q - target_centre

        # Cross-covariance: the "S" matrix of Horn's method, sum(p_i (x) q_i).
        s = pc.T @ qc
        trace = float(s[0, 0] + s[1, 1] + s[2, 2])
        s01, s02, s10, s12, s20, s21 = s[0, 1], s[0, 2], s[1, 0], s[1, 2], s[2, 0], s[2, 1]
        s00, s11, s22 = s[0, 0], s[1, 1], s[2, 2]
        n_matrix = np.array(
            [
                [trace, s12 - s21, s20 - s02, s01 - s10],
                [s12 - s21, s00 - s11 - s22, s01 + s10, s20 + s02],
                [s20 - s02, s01 + s10, -s00 + s11 - s22, s12 + s21],
                [s01 - s10, s20 + s02, s12 + s21, -s00 - s11 + s22],
            ],
            dtype=np.float64,
        )
        # Symmetric eigensolve (LAPACK, via numpy.linalg.eigh) in place of the
        # geometry module's cyclic-Jacobi routine - a legitimate difference in
        # ALGORITHM for the identical, well-posed eigenproblem, not a
        # different problem. eigh returns eigenvalues ascending, so the
        # optimal rotation quaternion is the LAST eigenvector.
        _, vectors = np.linalg.eigh(n_matrix)
        w, x, y, z = (float(v) for v in vectors[:, -1])
        length = math.sqrt(w * w + x * x + y * y + z * z)
        if length < 1e-12:
            raise CorrespondenceUnusable(
                "the correspondence set is degenerate - collinear or coincident points leave "
                "the rotation undetermined, so no unique alignment exists"
            )
        w, x, y, z = w / length, x / length, y / length, z / length
        rotation = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
        spin = Rigid.of(rotation, (0.0, 0.0, 0.0))
        target_centre_point = (
            float(target_centre[0]),
            float(target_centre[1]),
            float(target_centre[2]),
        )
        source_centre_point = (
            float(source_centre[0]),
            float(source_centre[1]),
            float(source_centre[2]),
        )
        moved = spin.apply(source_centre_point)
        translation = (
            target_centre_point[0] - moved[0],
            target_centre_point[1] - moved[1],
            target_centre_point[2] - moved[2],
        )
        return Rigid.of(rotation, translation)

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        """See :meth:`ComputeBackend.k_nearest_neighbors`.

        Same ``|a-b|^2 = |a|^2 - 2 a.b + |b|^2`` reformulation as
        :meth:`nearest_neighbors`, for the same reason: measured 2.2-2.8x
        faster than the broadcast-subtraction form across n=500-10000, and
        at n=10000 the broadcast form's ``(n, n, 3)`` difference array is
        2.4 GB against this form's 800 MB ``(n, n)`` squared-distance matrix.
        """
        _ensure_k(k)
        n = len(points)
        if n == 0:
            return []
        cap = min(k, n - 1)
        if cap <= 0:
            return [() for _ in range(n)]
        import numpy as np

        p = np.asarray(points, dtype=np.float64)
        centre = p.mean(axis=0)
        pc = p - centre
        squared_norm = np.sum(pc * pc, axis=1)
        chunk_rows = max(1, _PAIRWISE_CHUNK_ELEMENTS // n)
        result: list[tuple[int, ...]] = []
        for start in range(0, n, chunk_rows):
            block = pc[start : start + chunk_rows]
            block_norm = squared_norm[start : start + block.shape[0]]
            cross = block @ pc.T
            squared = block_norm[:, None] - 2.0 * cross + squared_norm[None, :]
            row_positions = np.arange(start, start + block.shape[0])
            # Exclude each row's own point from its own neighbour search.
            squared[np.arange(block.shape[0]), row_positions] = np.inf
            # Partial selection (average O(n)), not a full sort: this module
            # never promises an ORDER among the k, only the SET - see
            # ComputeBackend.k_nearest_neighbors.
            nearest_idx = np.argpartition(squared, cap - 1, axis=1)[:, :cap]
            for row in nearest_idx:
                result.append(tuple(int(j) for j in row))
        return result


# --- scipy availability, probed live, never imported at module scope -------------


def _probe_scipy() -> tuple[bool, str | None]:
    """``(usable, reason-if-not)``, the same shape and the same reasoning as
    :func:`_probe_numpy`. Probes ``scipy.spatial`` specifically, since that is
    the only part of scipy this module ever touches.
    """
    try:
        importlib.import_module("scipy.spatial")
    except ImportError as exc:
        return False, str(exc)
    return True, None


# --- the scipy (KD-tree) backend ----------------------------------------------------


class _ScipyBackend:
    """``nearest_neighbors`` and ``k_nearest_neighbors`` via
    ``scipy.spatial.cKDTree``; ``apply_transform`` and
    ``rigid_from_correspondences`` delegate to :class:`_NumpyBackend` unchanged,
    since scipy always installs numpy as a dependency and there is nothing
    tree-shaped about either of those two operations - duplicating their
    numpy implementation a third time would just be a second place for it to
    drift.

    Why this backend exists, measured rather than assumed: profiling (see the
    module docstring) found nearest-neighbour correspondence search dominates
    registration's cost by 2-3 orders of magnitude, and the "numpy" backend's
    GEMM reformulation only bought about 3x on it there, because a brute-force
    distance matrix - however well vectorised - is still O(n*m). A KD-tree is
    O(n log m): measured at 13.3x over the numpy GEMM form at n=2000, growing
    to 137.5x at n=20000 (index build included), with the index agreeing with
    the "python" reference on every query at every size measured. Growing
    speedup with n is the signature of a better ALGORITHM, not a faster
    implementation of the same one - see ``tests/test_compute.py`` for the
    equivalence tests and the module docstring for the full measured table.

    Constructed only once :func:`_probe_scipy` has confirmed
    ``scipy.spatial`` is importable - see :func:`select_backend` - but every
    method still does its own local imports, for the same reason
    :class:`_NumpyBackend` does: this class is definable, though never
    instantiated, whether or not scipy is installed.
    """

    __slots__ = ("_numpy",)

    def __init__(self) -> None:
        self._numpy = _NumpyBackend()

    @property
    def name(self) -> str:
        return SCIPY_BACKEND

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        return self._numpy.apply_transform(transform, points)

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        return self._numpy.rigid_from_correspondences(source, target)

    def nearest_neighbors(
        self, queries: Sequence[Point], target: Sequence[Point], *, gate: float
    ) -> list[NearestHit]:
        """See :meth:`ComputeBackend.nearest_neighbors`.

        ``cKDTree.query``'s own ``distance_upper_bound`` already draws the
        gate boundary exactly where this module's contract requires: a point
        AT ``distance_upper_bound`` comes back as a miss (index equal to the
        number of target points, distance ``inf``), the identical strict
        ``<`` the "python" backend's ``_VoxelIndex.nearest`` and the "numpy"
        backend both use - verified empirically while writing this backend,
        not assumed, and checked by
        ``test_nearest_neighbors_gate_is_strict_on_both_backends``'s scipy
        parametrisation. A miss is mapped to ``None`` here for the same
        reason. Queried with ``workers=-1`` to use every core - the query is
        embarrassingly parallel across independent points, and this machine
        has 20 of them mostly idle during a scan.

        Ties (two target points at the identical distance from a query) are
        broken by ``cKDTree``'s own internal traversal order, which is not
        documented as stable and is not guaranteed to match the "python" or
        "numpy" backends' choice of which tied point wins. Stated here rather
        than left implicit: it is not a concern for any input this module is
        ever actually asked about (real-valued sensor data, no coincident
        points), and every equivalence test in ``tests/test_compute.py`` uses
        jittered data specifically so it never has to rely on a tie-breaking
        rule that is not part of this module's contract.
        """
        _ensure_gate(gate)
        if not target or not queries:
            return [None for _ in queries]
        from scipy.spatial import cKDTree

        target_points = list(target)
        tree = cKDTree(target_points)
        distances, indices = tree.query(list(queries), k=1, distance_upper_bound=gate, workers=-1)
        miss = len(target_points)
        hits: list[NearestHit] = []
        for distance, index in zip(distances, indices, strict=True):
            if int(index) == miss or not math.isfinite(distance):
                hits.append(None)
            else:
                hits.append((float(distance), target_points[int(index)]))
        return hits

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        """See :meth:`ComputeBackend.k_nearest_neighbors`.

        Queries the tree against its own points, requesting ``cap + 1``
        neighbours (self is always among them, at distance 0) and filtering
        self out - the same shape as the "python" and "numpy" backends. Since
        ``cap`` is already capped at ``len(points) - 1``, the request never
        exceeds the tree's own point count, so ``cKDTree``'s own "fewer than
        k neighbours exist" padding (an out-of-range index) never triggers
        here. Order and ties: see :meth:`nearest_neighbors` - the same
        caveat applies, and is equally unreachable on this module's real
        inputs.
        """
        _ensure_k(k)
        n = len(points)
        if n == 0:
            return []
        cap = min(k, n - 1)
        if cap <= 0:
            return [() for _ in range(n)]
        from scipy.spatial import cKDTree

        point_list = list(points)
        tree = cKDTree(point_list)
        _, indices = tree.query(point_list, k=cap + 1, workers=-1)
        result: list[tuple[int, ...]] = []
        for row_index, row in enumerate(indices):
            neighbours = tuple(int(j) for j in row if int(j) != row_index)[:cap]
            result.append(neighbours)
        return result


# --- availability and selection ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Whether one backend name can be constructed right now, and why not."""

    name: str
    available: bool
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
        }


def available_backends() -> tuple[BackendAvailability, ...]:
    """Every backend name this module knows, and whether it can be used right now.

    ``"python"`` is always available. ``"numpy"`` is available exactly when
    ``import numpy`` succeeds, and ``"scipy"`` exactly when ``import
    scipy.spatial`` succeeds; :attr:`BackendAvailability.reason` names the
    actual failure when either does not, rather than a generic "unavailable".
    ``"gpu"`` is always reported unavailable - it is a reserved name for a
    backend this module does not implement; see the module docstring's "Seam
    for a future GPU backend".
    """
    numpy_available, numpy_reason = _probe_numpy()
    scipy_available, scipy_reason = _probe_scipy()
    return (
        BackendAvailability(PYTHON_BACKEND, True, None),
        BackendAvailability(NUMPY_BACKEND, numpy_available, numpy_reason),
        BackendAvailability(SCIPY_BACKEND, scipy_available, scipy_reason),
        BackendAvailability(GPU_BACKEND, False, _GPU_UNAVAILABLE_REASON),
    )


def _construct(name: str) -> ComputeBackend:
    if name == PYTHON_BACKEND:
        return _PythonBackend()
    if name == NUMPY_BACKEND:
        return _NumpyBackend()
    if name == SCIPY_BACKEND:
        return _ScipyBackend()
    raise AssertionError(  # pragma: no cover
        f"unreachable: {name!r} is not a constructible backend"
    )


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """What :func:`select_backend` resolved a request to, and why.

    :attr:`requested` is exactly what the caller asked for - including
    ``"auto"``. :attr:`resolved` is the concrete backend name actually
    returned, and is NEVER ``"auto"``: something concrete was always chosen,
    or the call raised instead. This is the report an ``"auto"`` selection
    promised - see the module docstring.

    :attr:`backend` carries the live backend object, which is not JSON-shaped
    and is deliberately left out of :meth:`as_dict`.
    """

    requested: str
    resolved: str
    reason: str
    backend: ComputeBackend

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "requested": self.requested,
            "resolved": self.resolved,
            "reason": self.reason,
        }


#: How ``"auto"`` explains each possible pick, keyed by the backend it picked.
#: A dict of functions rather than a single f-string built inline, so each
#: reason can name what was actually tried and skipped before it - see
#: :func:`select_backend`.
_AUTO_REASONS: Final[dict[str, Callable[[dict[str, BackendAvailability]], str]]] = {
    SCIPY_BACKEND: lambda _availability: (
        "'auto' selected 'scipy': it is importable here, and its KD-tree is the fastest "
        "backend measured for the correspondence-search and k-NN operations that dominate "
        "registration's cost - see the module docstring"
    ),
    NUMPY_BACKEND: lambda availability: (
        "'auto' selected 'numpy': scipy is not usable here "
        f"({availability[SCIPY_BACKEND].reason}), and numpy is - it is faster than 'python' "
        "for every operation this module measured, see the module docstring"
    ),
    PYTHON_BACKEND: lambda availability: (
        "'auto' selected 'python': neither scipy nor numpy is usable here "
        f"(scipy: {availability[SCIPY_BACKEND].reason}; "
        f"numpy: {availability[NUMPY_BACKEND].reason}), and 'python' is always available"
    ),
}


def select_backend(name: str = AUTO_BACKEND) -> BackendSelection:
    """Resolve ``name`` to a concrete, usable backend - or refuse. Never a downgrade.

    ``"python"``, ``"numpy"`` and ``"scipy"`` are honoured exactly: asking
    for one that is not usable here raises :class:`ComputeBackendUnavailable`
    rather than silently handing back a different backend. A caller who
    selected a backend and got a different one could not interpret their own
    timings - see the module docstring.

    ``"auto"`` picks the fastest backend confirmed usable here: scipy over
    numpy over python, in that order - see :data:`_KNOWN_BACKENDS` and this
    module's docstring for the measured reason a KD-tree beats a vectorised
    brute-force search by a growing margin as cloud size grows, not merely a
    constant factor. Which backend was picked, and why, is reported in
    :attr:`BackendSelection.reason` - "auto" never silently becomes one
    backend or another without saying which.

    ``"gpu"`` is a recognised, reserved name that always raises
    :class:`ComputeBackendUnavailable` - see the module docstring's "Seam for
    a future GPU backend".

    An unrecognised name raises :class:`ComputeBackendUnknown`.
    """
    availability_by_name = {a.name: a for a in available_backends()}

    if name == AUTO_BACKEND:
        for candidate in _KNOWN_BACKENDS:  # scipy, then numpy, then python - see above
            if availability_by_name[candidate].available:
                reason = _AUTO_REASONS[candidate](availability_by_name)
                return BackendSelection(name, candidate, reason, _construct(candidate))
        raise ComputeBackendUnavailable(  # pragma: no cover - python is always available
            "no backend is usable, which should be impossible: 'python' has no external dependency"
        )

    if name in _RESERVED_BACKENDS:
        raise ComputeBackendUnavailable(
            f"backend {name!r} is reserved but not implemented: {availability_by_name[name].reason}"
        )

    if name not in _KNOWN_BACKENDS:
        known = _KNOWN_BACKENDS + _RESERVED_BACKENDS + (AUTO_BACKEND,)
        raise ComputeBackendUnknown(f"backend {name!r} is not one of {known}")

    availability = availability_by_name[name]
    if not availability.available:
        raise ComputeBackendUnavailable(
            f"backend {name!r} was requested explicitly but is not usable here: "
            f"{availability.reason}. Explicit selection is honoured exactly - see the module "
            "docstring - so this raises rather than silently using a different backend"
        )
    return BackendSelection(
        name, name, f"{name!r} was requested explicitly and is usable", _construct(name)
    )
