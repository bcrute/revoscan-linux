"""Prove the compute backend wired into :mod:`scanner3d.icp_plane` is real,
not merely reported.

Two different kinds of claim live here, and they need different techniques.

**Equivalence.** ``icp_point_to_plane(a, b, backend="scipy")`` must produce
the same transform as ``icp_point_to_plane(a, b)`` (the default), run to
convergence, on the same input. This is the same shape of claim
``tests/test_compute.py`` already proves for the backend protocol itself;
here it is proven for the real, public function a caller actually reaches.

**Genuine wiring, not a label.** Equivalence alone cannot distinguish "the
backend genuinely ran and, correctly, agreed with the reference" from "the
backend parameter is silently ignored and the legacy path ran regardless" -
both look identical from the outside if the reference IS correct. Catching
that distinction needs a backend that is DELIBERATELY WRONG in an easily
detected way: :class:`_CorrespondenceStub` and :class:`_NormalsStub` return
answers no real search would give, and the tests using them assert the
WRONGNESS propagates into the output - proof the plumbing is live, not
merely equivalent by coincidence. :func:`test_default_backend_never_touches_compute_module`
takes the complementary approach for the default: it makes every
:mod:`scanner3d.compute` backend method raise, and confirms the default path
never triggers one.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence

import pytest

from scanner3d.compute import (
    NUMPY_BACKEND,
    PYTHON_BACKEND,
    SCIPY_BACKEND,
    ComputeBackendUnavailable,
    ComputeBackendUnknown,
    _NumpyBackend,
    _probe_numpy,
    _probe_scipy,
    _PythonBackend,
    _ScipyBackend,
)
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.icp_plane import estimate_normals, icp_point_to_plane

NUMPY_AVAILABLE, _NUMPY_REASON = _probe_numpy()
SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()

needs_numpy = pytest.mark.skipif(
    not NUMPY_AVAILABLE, reason=f"numpy not installed: {_NUMPY_REASON}"
)
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

ACCELERATED_BACKENDS = {NUMPY_BACKEND: _NumpyBackend, SCIPY_BACKEND: _ScipyBackend}
_MARKERS = {NUMPY_BACKEND: needs_numpy, SCIPY_BACKEND: needs_scipy}
ACCELERATED_PARAMS = [
    pytest.param(name, marks=_MARKERS[name], id=name) for name in ACCELERATED_BACKENDS
]


# --- shared scene builder ------------------------------------------------------------


def _sphere_cloud(
    n: int, *, seed: int, radius: float = 20.0, centre: Point = (0.0, 0.0, 0.0)
) -> list[Point]:
    """A bumpy sphere: normals point in every direction, so the 6x6
    linearised system is well conditioned everywhere - unlike a single flat
    panel, which the module's own docstring names as the degenerate case
    :func:`~scanner3d.icp_plane._solve_6x6` exists to refuse.
    """
    rng = random.Random(seed)
    points: list[Point] = []
    for _ in range(n):
        theta = rng.uniform(0, math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        r = radius + rng.uniform(-0.05, 0.05)
        points.append(
            (
                centre[0] + r * math.sin(theta) * math.cos(phi),
                centre[1] + r * math.sin(theta) * math.sin(phi),
                centre[2] + r * math.cos(theta),
            )
        )
    return points


def _scene(n: int, *, seed: int) -> tuple[list[Point], list[Point]]:
    """A source/target pair related by a known, non-degenerate rotation, with
    small per-point noise so the target is not an exact copy of the
    transformed source - the realistic case.
    """
    source = _sphere_cloud(n, seed=seed)
    transform = rotation_about_axis((0.2, 0.4, 0.88), 0.18, through=(1.0, -0.5, 2.0))
    rng = random.Random(seed + 1)
    target = [
        (
            p[0] + rng.uniform(-0.05, 0.05),
            p[1] + rng.uniform(-0.05, 0.05),
            p[2] + rng.uniform(-0.05, 0.05),
        )
        for p in transform.apply_all(source)
    ]
    return source, target


def _rotation_error(a: Rigid, b: Rigid) -> float:
    return max(abs(a.rotation[i][j] - b.rotation[i][j]) for i in range(3) for j in range(3))


# --- equivalence: the deliverable ----------------------------------------------------


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
@pytest.mark.parametrize("n", [300, 800])
def test_icp_point_to_plane_backend_matches_default_to_convergence(
    backend_name: str, n: int
) -> None:
    """The property the task exists to prove: run to convergence, the
    accelerated backend's transform must match the default's.
    """
    source, target = _scene(n, seed=100 + n)

    reference = icp_point_to_plane(source, target)
    accelerated = icp_point_to_plane(source, target, backend=backend_name)

    assert reference.resolved_backend == PYTHON_BACKEND
    assert accelerated.resolved_backend == backend_name
    assert reference.converged
    assert accelerated.converged

    # Measured agreement after full convergence was well under 1e-8 in every
    # rotation entry and well under 1e-8 in translation on this scene size;
    # 1e-4 leaves comfortable margin while still being a real claim, not a
    # rubber stamp - point-to-plane's linearised solve means iteration counts
    # can differ by one or two between backends even when both converge to
    # the same pose, so the comparison is on the FINAL transform, not on
    # matching intermediate steps.
    rot_err = _rotation_error(reference.transform, accelerated.transform)
    trans_err = math.dist(reference.transform.translation, accelerated.transform.translation)
    assert rot_err < 1e-4, f"{backend_name} converged rotation disagrees: {rot_err}"
    assert trans_err < 1e-4, f"{backend_name} converged translation disagrees: {trans_err}"


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_estimate_normals_backend_matches_default_exactly(backend_name: str) -> None:
    """Stronger than a tolerance: the neighbour SETS the backend and the
    legacy heapq search find are proven to agree exactly (no ties on this
    non-coincident data - see ``tests/test_compute.py``), and the local-PCA
    fit downstream of that neighbour set (:func:`~scanner3d.icp_plane._covariance`,
    :func:`~scanner3d.icp_plane._jacobi`) is completely UNCHANGED code
    regardless of backend. So the resulting normals are expected to agree
    to float64 precision, not merely "close".
    """
    points = _sphere_cloud(400, seed=7)

    reference = estimate_normals(points)
    accelerated = estimate_normals(points, backend=backend_name)

    assert len(reference.estimates) == len(accelerated.estimates)
    compared = 0
    for ref_estimate, acc_estimate in zip(reference.estimates, accelerated.estimates, strict=True):
        assert (ref_estimate.normal is None) == (acc_estimate.normal is None)
        if ref_estimate.normal is None or acc_estimate.normal is None:
            continue
        compared += 1
        assert math.dist(ref_estimate.normal, acc_estimate.normal) < 1e-9
    assert compared > len(points) * 0.9


# --- genuine wiring: a deliberately wrong backend must change the output --------------


class _CorrespondenceStub:
    """Correct everywhere except ``nearest_neighbors``, which pairs query
    ``i`` with a target point shifted by a fixed offset - never the true
    nearest point, unless a query happens to already be there. Shifted by an
    offset rather than collapsed onto a single point: pairing every query
    with the SAME target point (and so the same normal) makes the 6x6 solve
    genuinely singular (every row identical), which raises
    :class:`~scanner3d.icp_plane.IcpPlaneGeometryUnusable` instead of
    exercising the comparison this test wants - a real, if different,
    confirmation that the stub is wired in, but not this test's job. The
    offset keeps the correspondences varied enough to stay solvable while
    still being wrong, so a genuine wiring failure shows up as the fitted
    transform silently NOT moving, not as a different kind of refusal.

    If :func:`icp_point_to_plane` genuinely routes its correspondence search
    through the backend, this measurably moves the fitted transform; if the
    parameter is silently ignored for that operation, the legacy path's
    correct answer comes back unchanged and this stub's wrongness never
    shows up in the result.
    """

    name = "stub-correspondence"

    def __init__(self) -> None:
        self._real = _NumpyBackend()

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        return self._real.apply_transform(transform, points)

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        return self._real.rigid_from_correspondences(source, target)

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        return self._real.k_nearest_neighbors(points, k)

    def nearest_neighbors(self, queries, target, *, gate):
        if not target:
            return [None for _ in queries]
        n = len(target)
        offset = max(1, n // 3)
        result = []
        for i, q in enumerate(queries):
            wrong_target = target[(i + offset) % n]
            result.append((math.dist(q, wrong_target), wrong_target))
        return result


class _NormalsStub:
    """Correct everywhere except ``k_nearest_neighbors``, which always
    returns the first ``k`` OTHER indices by position - never the true
    nearest neighbours, unless a point happens to already be there. If
    :func:`estimate_normals` genuinely routes its neighbour search through
    the backend, this measurably changes the fitted normals; if the
    parameter is silently ignored for that operation, the legacy path's
    correct answer comes back unchanged.
    """

    name = "stub-normals"

    def __init__(self) -> None:
        self._real = _NumpyBackend()

    def apply_transform(self, transform: Rigid, points: Sequence[Point]) -> list[Point]:
        return self._real.apply_transform(transform, points)

    def rigid_from_correspondences(self, source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        return self._real.rigid_from_correspondences(source, target)

    def nearest_neighbors(self, queries, target, *, gate):
        return self._real.nearest_neighbors(queries, target, gate=gate)

    def k_nearest_neighbors(self, points: Sequence[Point], k: int) -> list[tuple[int, ...]]:
        n = len(points)
        cap = min(k, n - 1)
        if cap <= 0:
            return [() for _ in range(n)]
        return [tuple(j for j in range(cap + 1) if j != i)[:cap] for i in range(n)]


def test_icp_point_to_plane_actually_uses_the_backends_correspondence_search() -> None:
    source, target = _scene(300, seed=42)
    reference = icp_point_to_plane(source, target)
    stubbed = icp_point_to_plane(source, target, backend=_CorrespondenceStub())

    assert stubbed.resolved_backend == "stub-correspondence"
    rot_err = _rotation_error(reference.transform, stubbed.transform)
    trans_err = math.dist(reference.transform.translation, stubbed.transform.translation)
    assert rot_err > 1e-3 or trans_err > 1e-3, (
        "a deliberately wrong nearest_neighbors left the fitted transform unchanged - "
        "the correspondence search does not appear to be routed through the backend"
    )


def test_estimate_normals_actually_uses_the_backends_neighbour_search() -> None:
    points = _sphere_cloud(300, seed=8)
    reference = estimate_normals(points)
    stubbed = estimate_normals(points, backend=_NormalsStub())

    differences = 0
    for ref_estimate, stub_estimate in zip(reference.estimates, stubbed.estimates, strict=True):
        if ref_estimate.normal is None or stub_estimate.normal is None:
            continue
        if math.dist(ref_estimate.normal, stub_estimate.normal) > 1e-6:
            differences += 1
    assert differences > 0, (
        "a deliberately wrong k_nearest_neighbors left every normal unchanged - the "
        "neighbour search does not appear to be routed through the backend"
    )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_explicit_backend_is_used_for_both_operations_in_icp_point_to_plane(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call-counting spy on the REAL backend class: proves
    ``apply_transform``, ``nearest_neighbors`` (the correspondence search)
    AND ``k_nearest_neighbors`` (the normal estimation) are all genuinely
    invoked for an explicit non-default request - not just one of the two
    operations the task asked to be routed through the backend.
    """
    cls = ACCELERATED_BACKENDS[backend_name]
    calls = {"apply_transform": 0, "nearest_neighbors": 0, "k_nearest_neighbors": 0}
    for method_name in calls:
        original = getattr(cls, method_name)

        def wrapper(self, *args, __original=original, __name=method_name, **kwargs):
            calls[__name] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method_name, wrapper)

    source, target = _scene(300, seed=9)
    result = icp_point_to_plane(source, target, backend=backend_name)

    assert result.resolved_backend == backend_name
    assert calls["apply_transform"] > 0, "apply_transform was never called"
    assert calls["nearest_neighbors"] > 0, "nearest_neighbors was never called"
    assert calls["k_nearest_neighbors"] > 0, "k_nearest_neighbors was never called"


def test_default_backend_never_touches_compute_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The complementary claim to the two stub tests above: the DEFAULT path
    must not merely produce the right answer, it must never call into
    :mod:`scanner3d.compute` at all. Every method on every backend class is
    patched to raise; running with the default ``backend`` (both explicitly
    and by omission) must not trigger any of them.
    """

    def _boom(self, *args, **kwargs):
        raise AssertionError("a scanner3d.compute backend method was called on the default path")

    method_names = (
        "apply_transform",
        "nearest_neighbors",
        "rigid_from_correspondences",
        "k_nearest_neighbors",
    )
    for cls in (_PythonBackend, _NumpyBackend, _ScipyBackend):
        for method_name in method_names:
            monkeypatch.setattr(cls, method_name, _boom)

    source, target = _scene(300, seed=11)
    result_default = icp_point_to_plane(source, target)
    result_explicit = icp_point_to_plane(source, target, backend="python")
    assert result_default.resolved_backend == PYTHON_BACKEND
    assert result_explicit.resolved_backend == PYTHON_BACKEND

    field = estimate_normals(_sphere_cloud(200, seed=12))
    assert len(field.usable) > 0


def test_auto_resolving_to_python_also_never_touches_compute_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same claim as the test above, but reached through ``"auto"``
    rather than the literal string ``"python"`` or omission - the ONLY other
    way :func:`_resolve_backend` can decide "run the legacy path". Both
    numpy and scipy are forced unavailable so "auto" has nowhere else to
    resolve to; every backend method is still patched to raise, so this
    would fail loudly if "auto resolved to python" ever took the live
    :class:`~scanner3d.compute.ComputeBackend` route instead of the untouched
    one - a real, previously untested seam, since this environment normally
    has scipy available and "auto" never otherwise reaches "python".
    """
    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: not installed")
    )
    monkeypatch.setattr(
        "scanner3d.compute._probe_scipy", lambda: (False, "simulated: not installed")
    )

    def _boom(self, *args, **kwargs):
        raise AssertionError("a scanner3d.compute backend method was called on the default path")

    method_names = (
        "apply_transform",
        "nearest_neighbors",
        "rigid_from_correspondences",
        "k_nearest_neighbors",
    )
    for cls in (_PythonBackend, _NumpyBackend, _ScipyBackend):
        for method_name in method_names:
            monkeypatch.setattr(cls, method_name, _boom)

    source, target = _scene(300, seed=17)
    result = icp_point_to_plane(source, target, backend="auto")
    assert result.resolved_backend == PYTHON_BACKEND


# --- resolved_backend reporting -------------------------------------------------------


def test_resolved_backend_defaults_to_python() -> None:
    source, target = _scene(60, seed=13)
    result = icp_point_to_plane(source, target)
    assert result.resolved_backend == PYTHON_BACKEND


def test_result_as_dict_includes_resolved_backend() -> None:
    source, target = _scene(60, seed=18)
    result = icp_point_to_plane(source, target, backend="numpy" if NUMPY_AVAILABLE else "python")
    record = result.as_dict()
    assert "resolved_backend" in record
    assert record["resolved_backend"] == result.resolved_backend


def test_resolved_backend_auto_reports_what_it_chose() -> None:
    source, target = _scene(60, seed=14)
    result = icp_point_to_plane(source, target, backend="auto")
    assert result.resolved_backend in (PYTHON_BACKEND, NUMPY_BACKEND, SCIPY_BACKEND)
    if SCIPY_AVAILABLE:
        assert result.resolved_backend == SCIPY_BACKEND
    elif NUMPY_AVAILABLE:
        assert result.resolved_backend == NUMPY_BACKEND
    else:
        assert result.resolved_backend == PYTHON_BACKEND


def test_unknown_backend_name_raises() -> None:
    source, target = _scene(60, seed=15)
    with pytest.raises(ComputeBackendUnknown):
        icp_point_to_plane(source, target, backend="quantum")


def test_explicit_unavailable_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: not installed")
    )
    source, target = _scene(60, seed=16)
    with pytest.raises(ComputeBackendUnavailable):
        icp_point_to_plane(source, target, backend="numpy")


# --- end-to-end timing: informational only, no assertion on the speedup ---------------


@needs_numpy
def test_speed_report_is_informational_only(capsys: pytest.CaptureFixture[str]) -> None:
    """One full ``icp_point_to_plane`` call per backend, warm, at the sizes
    requested for the task report. Asserts nothing about relative speed -
    see ``tests/test_compute.py``'s identical rationale; the measured numbers
    belong in the task report.
    """
    backends = [PYTHON_BACKEND, NUMPY_BACKEND]
    if SCIPY_AVAILABLE:
        backends.append(SCIPY_BACKEND)

    report = []
    for n in (2000, 2500):
        source, target = _scene(n, seed=1000 + n)
        # Warm each backend once before timing (pays any first-import tax).
        for name in backends:
            icp_point_to_plane(source[:20], target[:20], backend=name, max_iterations=1)
        for name in backends:
            start = time.perf_counter()
            result = icp_point_to_plane(source, target, backend=name)
            elapsed = time.perf_counter() - start
            report.append(
                f"n={n} {name:8s}: {elapsed * 1000:8.1f} ms "
                f"({result.iterations} iters, converged={result.converged})"
            )
    print("\n" + "\n".join(report))
    assert report
