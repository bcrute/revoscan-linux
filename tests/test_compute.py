"""Prove the selectable compute backends agree with the pure-Python reference,
and that selection itself is honest.

Two kinds of claim live here, and they are not interchangeable.

**Backend agreement.** :mod:`scanner3d.compute`'s whole premise is that a
caller can switch backends and change nothing about the SCAN, only how long
it takes to produce it. Every operation test below therefore feeds the SAME
clouds to both backends and compares the SAME outputs directly - matched
points, matched indices, transform entries - never merely "both look
reasonable". Where floating-point associativity makes exact agreement
impossible (every numeric operation except :meth:`k_nearest_neighbors`,
whose output is a discrete SET of indices with no rounding to disagree
about), the tolerance used is stated next to the assertion and is derived
from figures measured while this module was written - see
``src/scanner3d/compute.py``'s own module docstring for the fuller
measurement writeup.

**Selection honesty.** A caller who asks for one backend must get that
backend or an error, never a different backend silently substituted - see
``test_select_backend_explicit_unavailable_raises_rather_than_falls_back``.
"auto" is the one exception, and it must say what it picked
(``test_select_backend_auto_reports_what_it_chose``).

The strongest single check in this file is
``test_python_backend_reproduces_one_registration_icp_iteration``: it drives
:func:`scanner3d.registration.icp` itself for one iteration, drives the same
inputs through :mod:`scanner3d.compute`'s "python" backend by hand, and
requires the two transforms to match to float64 precision. That is the
direct evidence that the "python" backend is not a lookalike reimplementation
but the same code path :mod:`scanner3d.registration` already runs.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from collections.abc import Sequence

import pytest

from scanner3d.compute import (
    AUTO_BACKEND,
    GPU_BACKEND,
    NUMPY_BACKEND,
    PYTHON_BACKEND,
    SCHEMA_VERSION,
    SCIPY_BACKEND,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendUnavailable,
    ComputeBackendUnknown,
    ComputeParametersUnusable,
    _NumpyBackend,
    _probe_numpy,
    _probe_scipy,
    _PythonBackend,
    _ScipyBackend,
    available_backends,
    select_backend,
)
from scanner3d.geometry import (
    CorrespondenceUnusable,
    Point,
    Rigid,
    alignment_rmse,
    rotation_about_axis,
)
from scanner3d.registration import (
    DEFAULT_CORRESPONDENCE_FRACTION,
    DEFAULT_KEEP_FRACTION,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    DEFAULT_VOXEL_FRACTION,
    DOWNSAMPLE_THRESHOLD,
    IDENTITY_POSE,
    MIN_CORRESPONDENCES,
    cloud_extent,
    icp,
    kept_count,
    voxel_downsample,
)

NUMPY_AVAILABLE, _NUMPY_REASON = _probe_numpy()
SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()

needs_numpy = pytest.mark.skipif(
    not NUMPY_AVAILABLE, reason=f"numpy not installed: {_NUMPY_REASON}"
)
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

#: The two accelerated backends, constructor and skip-marker keyed by name, so
#: an equivalence test can be written once and parametrized over both rather
#: than copy-pasted per backend - see e.g. ``test_apply_transform_agrees_with_python``.
ACCELERATED_BACKENDS: dict[str, type[ComputeBackend]] = {
    NUMPY_BACKEND: _NumpyBackend,
    SCIPY_BACKEND: _ScipyBackend,
}
_ACCELERATED_MARKERS = {NUMPY_BACKEND: needs_numpy, SCIPY_BACKEND: needs_scipy}
ACCELERATED_PARAMS = [
    pytest.param(name, marks=_ACCELERATED_MARKERS[name], id=name) for name in ACCELERATED_BACKENDS
]


# --- shared cloud builders ---------------------------------------------------------


def make_cloud(n: int, *, seed: int) -> list[Point]:
    """A bumpy, filled sphere: enough spread that no two points coincide and no
    query ties for nearest neighbour, so backend agreement can be checked
    exactly rather than "close enough" on the discrete choices.
    """
    rng = random.Random(seed)
    points: list[Point] = []
    for _ in range(n):
        theta = rng.uniform(0, math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        radius = 10.0 + rng.uniform(-0.5, 0.5)
        points.append(
            (
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta) + 5.0,
            )
        )
    return points


def _jitter(points: Sequence[Point], *, amount: float, seed: int) -> list[Point]:
    """Perturb every coordinate slightly, so a target cloud is not an exact
    rotation of the source - the realistic case, and the one where a nearest-
    neighbour search can actually disagree between two implementations.
    """
    rng = random.Random(seed)
    return [
        (
            x + rng.uniform(-amount, amount),
            y + rng.uniform(-amount, amount),
            z + rng.uniform(-amount, amount),
        )
        for x, y, z in points
    ]


PYTHON = _PythonBackend()


# --- available_backends / select_backend --------------------------------------------


def test_python_backend_always_reports_available() -> None:
    by_name = {a.name: a for a in available_backends()}
    assert by_name[PYTHON_BACKEND] == BackendAvailability(PYTHON_BACKEND, True, None)


def test_numpy_availability_matches_real_import() -> None:
    by_name = {a.name: a for a in available_backends()}
    assert by_name[NUMPY_BACKEND].available == NUMPY_AVAILABLE
    if not NUMPY_AVAILABLE:
        assert by_name[NUMPY_BACKEND].reason is not None


def test_scipy_availability_matches_real_import() -> None:
    by_name = {a.name: a for a in available_backends()}
    assert by_name[SCIPY_BACKEND].available == SCIPY_AVAILABLE
    if not SCIPY_AVAILABLE:
        assert by_name[SCIPY_BACKEND].reason is not None


def test_gpu_backend_always_reports_unavailable_with_a_reason() -> None:
    by_name = {a.name: a for a in available_backends()}
    gpu = by_name[GPU_BACKEND]
    assert gpu.available is False
    assert gpu.reason is not None
    assert "not implemented" in gpu.reason


def test_backend_availability_as_dict_shape() -> None:
    availability = BackendAvailability("python", True, None)
    assert availability.as_dict() == {
        "schema_version": SCHEMA_VERSION,
        "name": "python",
        "available": True,
        "reason": None,
    }


def test_select_backend_python_is_explicit_and_usable() -> None:
    selection = select_backend(PYTHON_BACKEND)
    assert selection.requested == PYTHON_BACKEND
    assert selection.resolved == PYTHON_BACKEND
    assert selection.backend.name == PYTHON_BACKEND
    assert isinstance(selection.backend, _PythonBackend)


@needs_numpy
def test_select_backend_numpy_is_explicit_and_usable() -> None:
    selection = select_backend(NUMPY_BACKEND)
    assert selection.requested == NUMPY_BACKEND
    assert selection.resolved == NUMPY_BACKEND
    assert selection.backend.name == NUMPY_BACKEND
    assert isinstance(selection.backend, _NumpyBackend)


@needs_scipy
def test_select_backend_scipy_is_explicit_and_usable() -> None:
    selection = select_backend(SCIPY_BACKEND)
    assert selection.requested == SCIPY_BACKEND
    assert selection.resolved == SCIPY_BACKEND
    assert selection.backend.name == SCIPY_BACKEND
    assert isinstance(selection.backend, _ScipyBackend)


def test_select_backend_auto_reports_what_it_chose() -> None:
    """scipy > numpy > python is the preference order - see the module
    docstring for the measured reason a KD-tree wins by a growing margin.
    """
    selection = select_backend(AUTO_BACKEND)
    assert selection.requested == AUTO_BACKEND
    assert selection.resolved != AUTO_BACKEND
    assert selection.resolved in (PYTHON_BACKEND, NUMPY_BACKEND, SCIPY_BACKEND)
    # The report names which backend and gives a reason - not a bare name.
    assert selection.resolved in selection.reason
    if SCIPY_AVAILABLE:
        assert selection.resolved == SCIPY_BACKEND
    elif NUMPY_AVAILABLE:
        assert selection.resolved == NUMPY_BACKEND
    else:
        assert selection.resolved == PYTHON_BACKEND


def test_select_backend_gpu_raises_reserved_not_implemented() -> None:
    with pytest.raises(ComputeBackendUnavailable, match="not implemented"):
        select_backend(GPU_BACKEND)


def test_select_backend_unknown_name_raises() -> None:
    with pytest.raises(ComputeBackendUnknown):
        select_backend("quantum")


def test_select_backend_explicit_unavailable_raises_rather_than_falls_back(monkeypatch) -> None:
    """The load-bearing selection guarantee: requesting numpy when it cannot be
    used must raise, never silently hand back a different backend instead. A
    caller who selected a backend and got a different one could not interpret
    their own timings.
    """
    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: not installed")
    )
    with pytest.raises(ComputeBackendUnavailable, match="simulated: not installed"):
        select_backend(NUMPY_BACKEND)


def test_select_backend_auto_falls_back_one_tier_at_a_time(monkeypatch) -> None:
    """ "auto" steps down the preference order one tier per unavailable
    backend, and names why at every step - never silently jumping straight
    to "python" just because something upstream failed.
    """
    monkeypatch.setattr(
        "scanner3d.compute._probe_scipy", lambda: (False, "simulated: scipy missing")
    )
    # scipy unavailable, numpy still is (or isn't, on this environment) - auto
    # should land on numpy if it can, and say so.
    selection = select_backend(AUTO_BACKEND)
    if NUMPY_AVAILABLE:
        assert selection.resolved == NUMPY_BACKEND
        assert "scipy" in selection.reason
    else:
        assert selection.resolved == PYTHON_BACKEND

    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: numpy missing")
    )
    # Both unavailable now - auto falls all the way to python, and reports
    # both reasons, not just the last one tried.
    selection = select_backend(AUTO_BACKEND)
    assert selection.resolved == PYTHON_BACKEND
    assert "scipy" in selection.reason
    assert "numpy" in selection.reason


def test_backend_selection_as_dict_omits_the_live_backend_object() -> None:
    selection = select_backend(PYTHON_BACKEND)
    as_dict = selection.as_dict()
    assert as_dict["schema_version"] == SCHEMA_VERSION
    assert as_dict["requested"] == PYTHON_BACKEND
    assert as_dict["resolved"] == PYTHON_BACKEND
    assert "backend" not in as_dict


# --- apply_transform agreement -------------------------------------------------------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n", [3, 50, 500, 2000])
def test_apply_transform_agrees_with_python(backend_name: str, n: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    points = make_cloud(n, seed=1)
    transform = rotation_about_axis((0.2, 0.7, 0.4), 0.37, through=(1.0, 2.0, 3.0))

    reference = PYTHON.apply_transform(transform, points)
    accelerated = backend.apply_transform(transform, points)

    assert len(reference) == len(accelerated) == n
    # Both compute the identical R @ p + t per point; the only possible
    # disagreement is 3-term summation order, which is negligible - measured
    # under 5e-15 at n=10000 while writing this module. ("scipy" delegates
    # this operation to the "numpy" backend unchanged - see _ScipyBackend -
    # so this parametrisation is also a check that the delegation is wired
    # up, not a second independent implementation to verify.
    for a, b in zip(reference, accelerated, strict=True):
        assert math.dist(a, b) < 1e-9


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_apply_transform_empty_cloud_agrees(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    transform = Rigid.identity()
    assert PYTHON.apply_transform(transform, []) == []
    assert backend.apply_transform(transform, []) == []


# --- nearest_neighbors agreement -----------------------------------------------------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n", [10, 100, 800, 2500])
def test_nearest_neighbors_agrees_with_python(backend_name: str, n: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    source = make_cloud(n, seed=2)
    transform = rotation_about_axis((0.0, 0.0, 1.0), 0.1, through=(0.0, 0.0, 5.0))
    target = _jitter([transform.apply(p) for p in source], amount=0.01, seed=9)
    extent = cloud_extent(target)
    gate = extent * DEFAULT_CORRESPONDENCE_FRACTION
    queries = PYTHON.apply_transform(transform, source)

    reference = PYTHON.nearest_neighbors(queries, target, gate=gate)
    accelerated = backend.nearest_neighbors(queries, target, gate=gate)

    assert len(reference) == len(accelerated) == n
    for ref_hit, acc_hit in zip(reference, accelerated, strict=True):
        assert (ref_hit is None) == (acc_hit is None)
        if ref_hit is None or acc_hit is None:
            continue
        ref_distance, ref_point = ref_hit
        acc_distance, acc_point = acc_hit
        # The matched POINT is a discrete choice - on this jittered, non-
        # coincident test data it is unique, so both backends must pick the
        # identical target point, not merely a nearby one.
        assert ref_point == acc_point
        # The DISTANCE is arithmetic - measured agreement under 2.2e-11 at
        # n=10000 for the "numpy" GEMM reformulation and exact (index-based,
        # not reconstructed from an array) for "scipy"'s cKDTree; 1e-6 leaves
        # several orders of magnitude of margin either way.
        assert math.isclose(ref_distance, acc_distance, abs_tol=1e-6)


@needs_numpy
def test_nearest_neighbors_chunking_agrees_with_unchunked(monkeypatch) -> None:
    """Force the numpy backend's chunk loop to run more than one chunk on a
    small input, by shrinking the chunk-size ceiling, and confirm the result
    is unchanged. This is the only way to exercise the chunking branch
    without a multi-thousand-point test, which would make the suite slow.

    Compared with a tolerance, not exact equality: BLAS's matrix multiply can
    sum in a different order for a different block shape, which is the same
    class of float64 non-associativity this file already tolerates between
    backends - here it is one backend against itself, chunked differently.
    """
    monkeypatch.setattr("scanner3d.compute._PAIRWISE_CHUNK_ELEMENTS", 10)
    numpy_backend = _NumpyBackend()
    source = make_cloud(40, seed=21)
    target = _jitter(source, amount=0.001, seed=22)
    extent = cloud_extent(target)
    gate = extent * DEFAULT_CORRESPONDENCE_FRACTION

    chunked = numpy_backend.nearest_neighbors(source, target, gate=gate)

    monkeypatch.undo()
    unchunked = _NumpyBackend().nearest_neighbors(source, target, gate=gate)

    assert len(chunked) == len(unchunked)
    for chunked_hit, unchunked_hit in zip(chunked, unchunked, strict=True):
        assert (chunked_hit is None) == (unchunked_hit is None)
        if chunked_hit is None or unchunked_hit is None:
            continue
        chunked_distance, chunked_point = chunked_hit
        unchunked_distance, unchunked_point = unchunked_hit
        assert chunked_point == unchunked_point
        assert math.isclose(chunked_distance, unchunked_distance, abs_tol=1e-6)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_nearest_neighbors_empty_target_returns_all_none_on_both_backends(
    backend_name: str,
) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(5, seed=3)
    assert PYTHON.nearest_neighbors(queries, [], gate=1.0) == [None] * 5
    assert backend.nearest_neighbors(queries, [], gate=1.0) == [None] * 5


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_nearest_neighbors_empty_queries_returns_empty_on_both_backends(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    target = make_cloud(5, seed=4)
    assert PYTHON.nearest_neighbors([], target, gate=1.0) == []
    assert backend.nearest_neighbors([], target, gate=1.0) == []


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("gate", [0.0, -1.0, math.inf, math.nan])
def test_nearest_neighbors_bad_gate_raises_identically(backend_name: str, gate: float) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    target = make_cloud(5, seed=5)
    queries = make_cloud(3, seed=6)
    with pytest.raises(ComputeParametersUnusable):
        PYTHON.nearest_neighbors(queries, target, gate=gate)
    with pytest.raises(ComputeParametersUnusable):
        backend.nearest_neighbors(queries, target, gate=gate)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_nearest_neighbors_gate_is_strict_on_both_backends(backend_name: str) -> None:
    """A point EXACTLY ``gate`` away is rejected, not admitted - every backend
    uses ``<``, never ``<=``. Random test data essentially never lands exactly
    on a boundary, so this is constructed by hand rather than relying on the
    randomised agreement tests above to ever exercise it. This is also the
    scipy-specific boundary claim requested in review: cKDTree's own
    ``distance_upper_bound`` was verified empirically (not assumed) to be
    exclusive at the boundary, matching ``_VoxelIndex.nearest`` and the numpy
    backend exactly - see ``_ScipyBackend.nearest_neighbors``'s docstring.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    target: list[Point] = [(0.0, 0.0, 0.0)]
    gate = 2.0
    query: list[Point] = [(gate, 0.0, 0.0)]  # exactly `gate` away, not inside it

    assert PYTHON.nearest_neighbors(query, target, gate=gate) == [None]
    assert backend.nearest_neighbors(query, target, gate=gate) == [None]

    # And a point a hair inside the gate is admitted by both.
    just_inside: list[Point] = [(gate - 1e-9, 0.0, 0.0)]
    py_hit = PYTHON.nearest_neighbors(just_inside, target, gate=gate)[0]
    acc_hit = backend.nearest_neighbors(just_inside, target, gate=gate)[0]
    assert py_hit is not None
    assert acc_hit is not None


# --- rigid_from_correspondences agreement --------------------------------------------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n", [3, 4, 50, 500, 2000])
def test_rigid_from_correspondences_agrees_with_python(backend_name: str, n: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    source = make_cloud(n, seed=7)
    transform = rotation_about_axis((0.3, 0.1, 0.9), 0.55, through=(2.0, -1.0, 0.5))
    target = [transform.apply(p) for p in source]

    reference = PYTHON.rigid_from_correspondences(source, target)
    accelerated = backend.rigid_from_correspondences(source, target)

    # Both solve the identical well-posed eigenproblem in exact arithmetic;
    # one by cyclic Jacobi rotations, the other by LAPACK's symmetric
    # eigensolver (numpy.linalg.eigh). Measured agreement was under 3e-14 in
    # every rotation entry and under 3e-14 in translation at n up to 10000;
    # 1e-8 leaves six orders of magnitude of margin. ("scipy" delegates this
    # operation to "numpy" unchanged - see _ScipyBackend.)
    for i in range(3):
        for j in range(3):
            assert math.isclose(reference.rotation[i][j], accelerated.rotation[i][j], abs_tol=1e-8)
    assert math.dist(reference.translation, accelerated.translation) < 1e-8


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_rigid_from_correspondences_mismatched_lengths_raises_identically(
    backend_name: str,
) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    source = make_cloud(5, seed=8)
    target = make_cloud(4, seed=9)
    with pytest.raises(CorrespondenceUnusable):
        PYTHON.rigid_from_correspondences(source, target)
    with pytest.raises(CorrespondenceUnusable):
        backend.rigid_from_correspondences(source, target)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_rigid_from_correspondences_too_few_points_raises_identically(backend_name: str) -> None:
    """Matched on message, not just exception type: 2 points always span at
    most 1 dimension, so a weakened "at least 3" length check (refusing only
    fewer than 2) would still raise CorrespondenceUnusable here via the
    SEPARATE spanned-dimensions check - just with a different message. Only
    checking the exception type would not catch that the length floor itself
    had drifted.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    source = make_cloud(2, seed=10)
    target = make_cloud(2, seed=11)
    with pytest.raises(CorrespondenceUnusable, match="at least 3"):
        PYTHON.rigid_from_correspondences(source, target)
    with pytest.raises(CorrespondenceUnusable, match="at least 3"):
        backend.rigid_from_correspondences(source, target)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_rigid_from_correspondences_collinear_points_raises_identically(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    source: list[Point] = [(float(i), 0.0, 0.0) for i in range(6)]
    target: list[Point] = [(0.0, float(i), 0.0) for i in range(6)]
    with pytest.raises(CorrespondenceUnusable):
        PYTHON.rigid_from_correspondences(source, target)
    with pytest.raises(CorrespondenceUnusable):
        backend.rigid_from_correspondences(source, target)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_rigid_from_correspondences_empty_raises_identically(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    with pytest.raises(CorrespondenceUnusable):
        PYTHON.rigid_from_correspondences([], [])
    with pytest.raises(CorrespondenceUnusable):
        backend.rigid_from_correspondences([], [])


# --- k_nearest_neighbors agreement -----------------------------------------------------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n", [5, 50, 500, 1500])
def test_k_nearest_neighbors_agrees_with_python_exactly(backend_name: str, n: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    points = make_cloud(n, seed=12)
    k = 10

    reference = PYTHON.k_nearest_neighbors(points, k)
    accelerated = backend.k_nearest_neighbors(points, k)

    assert len(reference) == len(accelerated) == n
    for i, (ref_neighbours, acc_neighbours) in enumerate(zip(reference, accelerated, strict=True)):
        # Order is not part of the contract - see ComputeBackend.k_nearest_neighbors
        # - but on this non-coincident test data the SET is unique, and a
        # discrete selection has no rounding to disagree about: this is exact
        # equality, not a tolerance.
        assert set(ref_neighbours) == set(acc_neighbours), f"point {i} disagrees"
        assert i not in ref_neighbours
        assert i not in acc_neighbours


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_neighbors_k_equal_to_one_is_the_valid_boundary(backend_name: str) -> None:
    """``k=1`` is the smallest VALID neighbour count - :data:`ComputeParametersUnusable`
    only starts at 0. A validator that off-by-one refused ``k=1`` too would
    not be caught by ``test_k_nearest_neighbors_bad_k_raises_identically``
    alone, which only tries values already below the floor.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    points = make_cloud(5, seed=19)
    reference = PYTHON.k_nearest_neighbors(points, 1)
    accelerated = backend.k_nearest_neighbors(points, 1)
    assert all(len(neighbours) == 1 for neighbours in reference)
    assert reference == accelerated or all(
        set(r) == set(a) for r, a in zip(reference, accelerated, strict=True)
    )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_neighbors_caps_at_available_points_on_both_backends(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    points = make_cloud(4, seed=13)
    k = 10  # far more than the 3 other points available

    reference = PYTHON.k_nearest_neighbors(points, k)
    accelerated = backend.k_nearest_neighbors(points, k)

    for ref_neighbours, acc_neighbours in zip(reference, accelerated, strict=True):
        assert len(ref_neighbours) == 3
        assert set(ref_neighbours) == set(acc_neighbours)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_neighbors_empty_points_agrees(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    assert PYTHON.k_nearest_neighbors([], 5) == []
    assert backend.k_nearest_neighbors([], 5) == []


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("k", [0, -1])
def test_k_nearest_neighbors_bad_k_raises_identically(backend_name: str, k: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    points = make_cloud(5, seed=14)
    with pytest.raises(ComputeParametersUnusable):
        PYTHON.k_nearest_neighbors(points, k)
    with pytest.raises(ComputeParametersUnusable):
        backend.k_nearest_neighbors(points, k)


# --- k_nearest_between agreement (cross-set: queries against a DIFFERENT cloud) -----------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n_queries,n_targets", [(5, 8), (13, 50), (60, 500)])
def test_k_nearest_between_agrees_with_python_exactly(
    backend_name: str, n_queries: int, n_targets: int
) -> None:
    """The cross-set analogue of ``test_k_nearest_neighbors_agrees_with_python_exactly``:
    two DIFFERENT clouds (different seeds, so no query coincides with a
    target), never one cloud searched against itself.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(n_queries, seed=1001)
    targets = make_cloud(n_targets, seed=2002)
    k = 6

    reference = PYTHON.k_nearest_between(queries, targets, k=k)
    accelerated = backend.k_nearest_between(queries, targets, k=k)

    assert len(reference) == len(accelerated) == n_queries
    for i, (ref_neighbours, acc_neighbours) in enumerate(zip(reference, accelerated, strict=True)):
        # Order is not part of the contract - see ComputeBackend.k_nearest_between -
        # but on this non-coincident test data the SET is unique, and a discrete
        # selection has no rounding to disagree about: this is exact equality.
        assert set(ref_neighbours) == set(acc_neighbours), f"query {i} disagrees"
        assert len(ref_neighbours) == min(k, n_targets)


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_never_excludes_a_target_that_equals_a_query(backend_name: str) -> None:
    """Unlike :meth:`k_nearest_neighbors`, there is no "exclude myself" rule here -
    a query and a target are two different clouds, so a target that happens to
    sit exactly where a query sits is still a legitimate nearest neighbour.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    shared_point: Point = (1.0, 2.0, 3.0)
    queries = [shared_point]
    targets = [shared_point, (100.0, 100.0, 100.0), (200.0, 200.0, 200.0)]

    reference = PYTHON.k_nearest_between(queries, targets, k=1)
    accelerated = backend.k_nearest_between(queries, targets, k=1)
    assert reference == [(0,)]
    assert accelerated == [(0,)]


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_k_equal_to_one_is_the_valid_boundary(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(5, seed=1019)
    targets = make_cloud(9, seed=2029)
    reference = PYTHON.k_nearest_between(queries, targets, k=1)
    accelerated = backend.k_nearest_between(queries, targets, k=1)
    assert all(len(neighbours) == 1 for neighbours in reference)
    assert reference == accelerated or all(
        set(r) == set(a) for r, a in zip(reference, accelerated, strict=True)
    )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_caps_at_available_targets_on_both_backends(backend_name: str) -> None:
    """``cKDTree.query`` pads a request for more neighbours than exist with an
    out-of-range index (equal to the tree's own point count) and an infinite
    distance - a real trap specific to the scipy path, since neither the
    python nor the numpy backend has any such padding behaviour to
    accidentally let through. The length check below would already fail if
    a pad leaked in (it would make the row longer than 4), but the explicit
    bounds check names exactly what a leaked pad looks like rather than
    leaving it to be inferred from a length mismatch.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(3, seed=1013)
    targets = make_cloud(4, seed=2013)
    k = 10  # far more than the 4 targets available

    reference = PYTHON.k_nearest_between(queries, targets, k=k)
    accelerated = backend.k_nearest_between(queries, targets, k=k)

    for ref_neighbours, acc_neighbours in zip(reference, accelerated, strict=True):
        assert len(ref_neighbours) == 4
        assert set(ref_neighbours) == set(acc_neighbours)
        assert all(0 <= index < len(targets) for index in acc_neighbours), (
            f"{acc_neighbours} contains an index outside [0, {len(targets)}) - looks like "
            "cKDTree's own out-of-range padding for 'fewer neighbours than requested' leaked "
            "through uncapped"
        )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_returns_indices_into_targets_not_a_combined_list(
    backend_name: str,
) -> None:
    """Indices must resolve against ``targets`` alone - never against some
    ``queries + targets`` (or ``targets + queries``) concatenation a
    same-set-flavoured implementation might have been tempted to build.

    Every query sits 0.1 away from exactly one target and far from every
    other target and every other query, so which target is "the answer" for
    each query is known by construction, not by search - a transposition or
    an index shifted by ``len(queries)`` either lands out of range (caught
    directly) or resolves, via ``targets[index]``, to a point that is not the
    one this test built for that query (caught by the equality check).
    Deliberately ``len(queries) != len(targets)`` so a combined-list shift
    cannot coincidentally still land in range.
    """
    backend = ACCELERATED_BACKENDS[backend_name]()
    targets: list[Point] = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        (50.0, 0.0, 0.0),
    ]
    queries: list[Point] = [(0.1, 0.0, 0.0), (10.1, 0.0, 0.0), (30.1, 0.0, 0.0), (50.1, 0.0, 0.0)]
    expected_target_index = [0, 1, 3, 5]

    for source in (PYTHON, backend):
        result = source.k_nearest_between(queries, targets, k=1)
        assert len(result) == len(queries)
        for query_index, row in enumerate(result):
            assert len(row) == 1
            (index,) = row
            assert 0 <= index < len(targets), (
                f"index {index} does not name any of the {len(targets)} targets - looks like "
                "an index into something other than `targets` alone"
            )
            expected = targets[expected_target_index[query_index]]
            assert targets[index] == expected, (
                f"query {query_index} resolved to {targets[index]}, not its known nearest "
                f"target {expected} - looks like an index meant for a different list"
            )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_empty_targets_agrees(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(3, seed=1014)
    assert PYTHON.k_nearest_between(queries, [], k=5) == [(), (), ()]
    assert backend.k_nearest_between(queries, [], k=5) == [(), (), ()]


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_empty_queries_agrees(backend_name: str) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    targets = make_cloud(3, seed=1015)
    assert PYTHON.k_nearest_between([], targets, k=5) == []
    assert backend.k_nearest_between([], targets, k=5) == []


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("k", [0, -1])
def test_k_nearest_between_bad_k_raises_identically(backend_name: str, k: int) -> None:
    backend = ACCELERATED_BACKENDS[backend_name]()
    queries = make_cloud(3, seed=1016)
    targets = make_cloud(5, seed=2016)
    with pytest.raises(ComputeParametersUnusable):
        PYTHON.k_nearest_between(queries, targets, k=k)
    with pytest.raises(ComputeParametersUnusable):
        backend.k_nearest_between(queries, targets, k=k)


def test_k_nearest_between_k_is_keyword_only() -> None:
    """Matches the signature the caller (scanner3d.texture) actually depends
    on: ``k`` must be impossible to pass positionally, unlike the plain
    ``int`` third parameter :meth:`k_nearest_neighbors` uses - the gate-like
    parameters in this module (see ``nearest_neighbors``'s ``gate``) are
    keyword-only, and this is one of them.
    """
    with pytest.raises(TypeError):
        PYTHON.k_nearest_between([(0.0, 0.0, 0.0)], [(1.0, 1.0, 1.0)], 1)  # type: ignore[call-arg]


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_k_nearest_between_numpy_chunking_agrees_past_one_chunk(backend_name: str) -> None:
    """The numpy backend chunks over queries when the (queries x targets)
    matrix would be large - see :meth:`_NumpyBackend.k_nearest_between`. This
    picks target and query counts large enough to force more than one chunk
    and checks the chunk boundary does not drop or duplicate a neighbour.
    """
    from scanner3d.compute import _PAIRWISE_CHUNK_ELEMENTS

    backend = ACCELERATED_BACKENDS[backend_name]()
    n_targets = 200
    # Comfortably past one chunk boundary for the numpy backend; scipy has no
    # such boundary but is checked identically for the equivalence claim.
    n_queries = (_PAIRWISE_CHUNK_ELEMENTS // n_targets) * 3 + 7
    queries = make_cloud(n_queries, seed=3001)
    targets = make_cloud(n_targets, seed=3002)
    k = 4

    reference = PYTHON.k_nearest_between(queries, targets, k=k)
    accelerated = backend.k_nearest_between(queries, targets, k=k)
    assert len(accelerated) == n_queries
    for ref_neighbours, acc_neighbours in zip(reference, accelerated, strict=True):
        assert set(ref_neighbours) == set(acc_neighbours)


# --- the strong claim: the python backend IS registration.icp's own search --------------


def test_python_backend_reproduces_one_registration_icp_iteration() -> None:
    """Drive :func:`scanner3d.registration.icp` for exactly one iteration, then
    drive the identical inputs through :mod:`scanner3d.compute`'s "python"
    backend by hand, using the same gate formula and the same
    :func:`scanner3d.registration.kept_count` trim. One iteration only, so
    there is no convergence loop whose stopping point could differ. If the
    two transforms match to float64 precision, the "python" backend is not a
    lookalike: it is the same nearest-neighbour search
    (:class:`scanner3d.registration._VoxelIndex`) and the same closed-form
    solve (:func:`scanner3d.geometry.rigid_from_correspondences`) that
    :mod:`scanner3d.registration` already runs in production.
    """
    source = make_cloud(300, seed=15)
    transform = rotation_about_axis((0.1, 0.2, 0.9), 0.2, through=(0.5, -0.5, 1.0))
    target = _jitter([transform.apply(p) for p in source], amount=0.02, seed=16)

    real = icp(source, target, initial=IDENTITY_POSE, max_iterations=1, keep_fraction=0.6)

    extent = cloud_extent(target)
    gate = extent * DEFAULT_CORRESPONDENCE_FRACTION
    moved = PYTHON.apply_transform(IDENTITY_POSE, source)
    hits = PYTHON.nearest_neighbors(moved, target, gate=gate)

    triples = [
        (distance, source_point, hit[1])
        for source_point, hit in zip(source, hits, strict=True)
        if (hit is not None) and (distance := hit[0]) is not None
    ]
    triples.sort(key=lambda t: t[0])
    keep = min(len(triples), kept_count(len(triples), 0.6))
    matched_source = [t[1] for t in triples[:keep]]
    matched_target = [t[2] for t in triples[:keep]]

    assert keep == real.correspondences
    fitted = PYTHON.rigid_from_correspondences(matched_source, matched_target)

    for i in range(3):
        for j in range(3):
            assert fitted.rotation[i][j] == pytest.approx(real.transform.rotation[i][j], abs=1e-12)
    assert math.dist(fitted.translation, real.transform.translation) < 1e-12


# --- module import and the "python" backend without numpy -------------------------------


def test_module_imports_and_python_backend_works_with_numpy_unavailable() -> None:
    """Simulate numpy being absent by poisoning ``sys.modules`` before the
    import, in a subprocess so this process's already-imported numpy (needed
    by the numpy-agreement tests above) is untouched. ``sys.modules[name] =
    None`` is the documented technique for forcing ``import numpy`` to raise
    ``ImportError`` even when numpy is genuinely installed on disk.

    scipy depends on numpy internally, so blocking numpy this way is expected
    to cascade: ``import scipy.spatial`` fails too, transitively, without
    this module doing anything special to make that happen. Asserted here
    rather than assumed.
    """
    script = (
        "import sys; sys.modules['numpy'] = None\n"
        "import scanner3d.compute as compute\n"
        "availability = {a.name: a for a in compute.available_backends()}\n"
        "assert availability['numpy'].available is False, availability\n"
        "assert availability['numpy'].reason is not None\n"
        "assert availability['scipy'].available is False, availability\n"
        "assert availability['scipy'].reason is not None\n"
        "assert availability['python'].available is True\n"
        "selection = compute.select_backend('auto')\n"
        "assert selection.resolved == 'python', selection\n"
        "backend = compute.select_backend('python').backend\n"
        "from scanner3d.geometry import Rigid\n"
        "result = backend.apply_transform(Rigid.identity(), [(1.0, 2.0, 3.0)])\n"
        "assert result == [(1.0, 2.0, 3.0)], result\n"
        "for name in ('numpy', 'scipy'):\n"
        "    try:\n"
        "        compute.select_backend(name)\n"
        "    except compute.ComputeBackendUnavailable:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'expected ComputeBackendUnavailable for {name}')\n"
        "print('OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_numpy_backend_works_with_scipy_unavailable() -> None:
    """Simulate scipy being absent (numpy present) in a subprocess: "auto"
    steps down exactly one tier, to numpy, and the numpy backend itself is
    unaffected - scipy's absence must not degrade numpy at all, since
    :class:`_NumpyBackend` never imports scipy.
    """
    script = (
        "import sys; sys.modules['scipy'] = None; sys.modules['scipy.spatial'] = None\n"
        "import scanner3d.compute as compute\n"
        "availability = {a.name: a for a in compute.available_backends()}\n"
        "assert availability['scipy'].available is False, availability\n"
        "assert availability['numpy'].available is True, availability\n"
        "selection = compute.select_backend('auto')\n"
        "assert selection.resolved == 'numpy', selection\n"
        "assert 'scipy' in selection.reason, selection.reason\n"
        "backend = compute.select_backend('numpy').backend\n"
        "from scanner3d.geometry import Rigid\n"
        "result = backend.apply_transform(Rigid.identity(), [(1.0, 2.0, 3.0)])\n"
        "assert result == [(1.0, 2.0, 3.0)], result\n"
        "try:\n"
        "    compute.select_backend('scipy')\n"
        "except compute.ComputeBackendUnavailable:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ComputeBackendUnavailable')\n"
        "print('OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


# --- the full-convergence claim: all backends land on the same registration -----------


def _run_icp_with_backend(
    backend: ComputeBackend,
    source: list[Point],
    target: list[Point],
    *,
    initial: Rigid,
    max_iterations: int,
    tolerance: float,
    keep_fraction: float,
) -> tuple[Rigid, float, int, bool]:
    """:func:`scanner3d.registration.icp`'s own algorithm - the same
    downsampling conditional from ``_prepare`` and the same match/trim/solve
    loop from ``_iterate`` - driven through a :class:`ComputeBackend` instead
    of the hardcoded python-only path. Verified against the real
    :func:`scanner3d.registration.icp` in
    ``test_full_icp_convergence_agrees_across_backends_and_matches_registration_icp``
    below: the first cut of this harness matched at n=2000 (below
    :data:`~scanner3d.registration.DOWNSAMPLE_THRESHOLD`, so no downsampling
    triggers) but silently diverged at n=2500 (above it) until the
    downsampling step was added here too - a reminder that "equivalent to
    the reference" has to be checked at the SAME granularity the reference
    itself operates at, not just its inner loop.
    """
    extent = cloud_extent(target)
    gate = extent * DEFAULT_CORRESPONDENCE_FRACTION
    threshold = tolerance * extent
    if max(len(source), len(target)) > DOWNSAMPLE_THRESHOLD:
        voxel_size = extent * DEFAULT_VOXEL_FRACTION
        source = list(voxel_downsample(source, voxel_size))
        target = list(voxel_downsample(target, voxel_size))
    transform = initial
    rmse = math.inf
    previous: float | None = None
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        moved = backend.apply_transform(transform, source)
        hits = backend.nearest_neighbors(moved, target, gate=gate)
        pairs: list[tuple[float, Point, Point]] = []
        for original, hit in zip(source, hits, strict=True):
            if hit is None:
                continue
            distance, nearest_target_point = hit
            pairs.append((distance, original, nearest_target_point))
        if len(pairs) < MIN_CORRESPONDENCES:
            raise RuntimeError(f"only {len(pairs)} correspondences at iteration {iterations}")
        pairs.sort(key=lambda pair: pair[0])
        keep = min(len(pairs), kept_count(len(pairs), keep_fraction))
        matched_source = [pair[1] for pair in pairs[:keep]]
        matched_target = [pair[2] for pair in pairs[:keep]]
        transform = backend.rigid_from_correspondences(matched_source, matched_target)
        rmse = alignment_rmse(matched_source, matched_target, transform)
        if previous is not None and abs(previous - rmse) <= threshold:
            converged = True
            break
        previous = rmse
    return transform, rmse, iterations, converged


@pytest.mark.parametrize("n", [2000, 2500])
def test_full_icp_convergence_agrees_across_backends_and_matches_registration_icp(
    n: int, capsys
) -> None:
    """The strongest equivalence claim in this file, extended from one
    iteration to full convergence, and extended to all three backends: run
    :func:`scanner3d.registration.icp` for real, then run the SAME clouds
    through :func:`_run_icp_with_backend` for python, numpy and scipy, and
    require every accelerated backend's FINAL, CONVERGED transform to match
    both the "python" harness result AND the real ``registration.icp()``
    result - not just one iteration's worth of agreement, the whole
    convergence trajectory's outcome.

    n=2000 is below :data:`~scanner3d.registration.DOWNSAMPLE_THRESHOLD`
    (2048): the harness runs ICP on the full, undownsampled 2000-point pair,
    matching the realistic post-downsample working set size requested for
    end-to-end measurement. n=2500 is above the threshold, so
    ``registration.icp`` (and this harness, which replicates that step)
    downsamples it further first - included to prove the harness's
    downsampling replication is correct too, not because 2500 raw points is
    itself the interesting size.

    Timing is printed (run with ``-s`` to see it), not asserted - a
    real 20-core machine shared with other agents' work showed enough
    variance during this task that a hard speedup assertion here would be
    flaky by construction; the numbers belong in the task report.
    """
    import time

    source = make_cloud(n, seed=42)
    true_transform = rotation_about_axis((0.1, 0.2, 0.9), 0.15, through=(0.5, -0.3, 1.0))
    target = _jitter([true_transform.apply(p) for p in source], amount=0.02, seed=43)

    real = icp(
        source,
        target,
        initial=IDENTITY_POSE,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        tolerance=DEFAULT_TOLERANCE,
        keep_fraction=DEFAULT_KEEP_FRACTION,
    )

    backends_to_try: list[tuple[str, ComputeBackend]] = [(PYTHON_BACKEND, PYTHON)]
    if NUMPY_AVAILABLE:
        backends_to_try.append((NUMPY_BACKEND, _NumpyBackend()))
    if SCIPY_AVAILABLE:
        backends_to_try.append((SCIPY_BACKEND, _ScipyBackend()))

    results: dict[str, tuple[Rigid, float]] = {}
    for name, backend in backends_to_try:
        start = time.perf_counter()
        transform, _rmse, iterations, converged = _run_icp_with_backend(
            backend,
            source,
            target,
            initial=IDENTITY_POSE,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            tolerance=DEFAULT_TOLERANCE,
            keep_fraction=DEFAULT_KEEP_FRACTION,
        )
        elapsed = time.perf_counter() - start
        results[name] = (transform, elapsed)
        print(
            f"\nn={n} {name:8s}: {elapsed * 1000:8.1f} ms  "
            f"({iterations} iters, converged={converged})"
        )
        if name == PYTHON_BACKEND:
            # The python harness must match the real production call exactly
            # - this is test_python_backend_reproduces_one_registration_icp_iteration's
            # claim, extended to a full multi-iteration convergence run.
            rot_err = max(
                abs(real.transform.rotation[i][j] - transform.rotation[i][j])
                for i in range(3)
                for j in range(3)
            )
            trans_err = math.dist(real.transform.translation, transform.translation)
            assert rot_err < 1e-12, f"python harness diverged from registration.icp: {rot_err}"
            assert trans_err < 1e-12

    python_transform, _ = results[PYTHON_BACKEND]
    for name, (transform, _elapsed) in results.items():
        if name == PYTHON_BACKEND:
            continue
        # Measured agreement after FULL convergence (16-33 iterations) was
        # under 6e-15 in every rotation entry and under 6e-15 in translation
        # at n=2000/2500 - tighter than any single-operation tolerance in
        # this file, because ICP's fixed point stabilises the correspondence
        # set: once every backend agrees on which points are paired, the
        # final solve is the same well-posed problem regardless of how the
        # correspondences were found. 1e-6 leaves eight orders of margin.
        rot_err = max(
            abs(python_transform.rotation[i][j] - transform.rotation[i][j])
            for i in range(3)
            for j in range(3)
        )
        trans_err = math.dist(python_transform.translation, transform.translation)
        assert rot_err < 1e-6, f"{name} converged transform disagrees: rot_err={rot_err}"
        assert trans_err < 1e-6, f"{name} converged transform disagrees: trans_err={trans_err}"


# --- an informational speed comparison: no assertion on the speedup factor --------------


def test_speed_report_is_informational_only(capsys) -> None:
    """Times every available backend over a realistic cloud size and prints
    the numbers for a human to read in ``-s`` output. Intentionally asserts
    nothing about which backend is faster or by how much - a timing
    assertion in a test suite is flaky by nature (this machine has been
    shared with other agents' work throughout this task, and that showed up
    as real variance in one-off timings); the measured numbers belong in the
    task report, not in a pass/fail here.
    """
    import time

    n = 3000
    source = make_cloud(n, seed=17)
    transform = rotation_about_axis((0.0, 0.0, 1.0), 0.1, through=(0.0, 0.0, 5.0))
    target = _jitter([transform.apply(p) for p in source], amount=0.01, seed=18)
    extent = cloud_extent(target)
    gate = extent * DEFAULT_CORRESPONDENCE_FRACTION
    queries = PYTHON.apply_transform(transform, source)

    backends: list[tuple[str, ComputeBackend]] = [(PYTHON_BACKEND, PYTHON)]
    if NUMPY_AVAILABLE:
        backends.append((NUMPY_BACKEND, _NumpyBackend()))
    if SCIPY_AVAILABLE:
        backends.append((SCIPY_BACKEND, _ScipyBackend()))

    lengths = set()
    report = [f"\nnearest_neighbors n={n}:"]
    for name, backend in backends:
        start = time.perf_counter()
        result = backend.nearest_neighbors(queries, target, gate=gate)
        elapsed = time.perf_counter() - start
        lengths.add(len(result))
        report.append(f"{name}={elapsed * 1000:.1f}ms")

    assert lengths == {n}
    print(" ".join(report))
