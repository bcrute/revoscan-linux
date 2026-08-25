"""Prove the compute backend wired into :mod:`scanner3d.registration.icp` is
real, not merely reported.

Two different kinds of claim live here, and they need different techniques -
the same split :mod:`tests.test_icp_plane_backend` uses for the sibling
module, since this is the identical pattern applied to point-to-point ICP.

**Equivalence.** ``icp(a, b, backend="scipy")`` must produce the same
transform as ``icp(a, b)`` (the default), run to convergence, on the same
input.

**Genuine wiring, not a label.** Equivalence alone cannot distinguish "the
backend genuinely ran and, correctly, agreed with the reference" from "the
backend parameter is silently ignored and the legacy path ran regardless" -
both look identical from the outside if the reference IS correct. Catching
that distinction needs a backend that is DELIBERATELY WRONG in an easily
detected way: :class:`_CorrespondenceStub` returns an answer no real search
would give, and the tests using it assert the WRONGNESS propagates into the
output - proof the plumbing is live, not merely equivalent by coincidence.
:func:`test_default_backend_never_touches_compute_module` takes the
complementary approach for the default: it makes every
:mod:`scanner3d.compute` backend method raise, and confirms the default path
never triggers one.

``register_turntable`` has no ``backend`` parameter of its own and is not
exercised here at all - ``tests/test_registration.py`` already covers it in
full and must stay untouched; this file only adds new tests, in a new file,
for the new ``icp`` parameter.
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
from scanner3d.registration import DEFAULT_CORRESPONDENCE_FRACTION, _VoxelIndex, cloud_extent, icp

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


def _bumpy_sphere(n: int, *, seed: int, radius: float = 20.0) -> list[Point]:
    """A bumpy sphere: no symmetry a point-to-point fit could hide behind,
    unlike a perfect sphere (useless for registration - it looks identical
    from every angle) or a flat panel (the exact tangential blind spot
    :mod:`scanner3d.icp_plane` exists to fix, irrelevant to point-to-point).
    """
    rng = random.Random(seed)
    points: list[Point] = []
    for _ in range(n):
        theta = rng.uniform(0, math.pi)
        phi = rng.uniform(0, 2 * math.pi)
        r = radius + rng.uniform(-0.05, 0.05)
        points.append(
            (
                r * math.sin(theta) * math.cos(phi),
                r * math.sin(theta) * math.sin(phi),
                r * math.cos(theta),
            )
        )
    return points


def _scene(n: int, *, seed: int) -> tuple[list[Point], list[Point]]:
    """A source/target pair related by a known, non-degenerate rotation, with
    small per-point noise so the target is not an exact copy of the
    transformed source - the realistic case.
    """
    source = _bumpy_sphere(n, seed=seed)
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
def test_icp_backend_matches_default_to_convergence(backend_name: str, n: int) -> None:
    """The property the task exists to prove: run to convergence, the
    accelerated backend's transform must match the default's.
    """
    source, target = _scene(n, seed=100 + n)

    reference = icp(source, target)
    accelerated = icp(source, target, backend=backend_name)

    assert reference.resolved_backend == PYTHON_BACKEND
    assert accelerated.resolved_backend == backend_name
    assert reference.converged
    assert accelerated.converged

    rot_err = _rotation_error(reference.transform, accelerated.transform)
    trans_err = math.dist(reference.transform.translation, accelerated.transform.translation)
    assert rot_err < 1e-4, f"{backend_name} converged rotation disagrees: {rot_err}"
    assert trans_err < 1e-4, f"{backend_name} converged translation disagrees: {trans_err}"


# --- genuine wiring: a deliberately wrong backend must change the output --------------


class _CorrespondenceStub:
    """Correct everywhere except ``nearest_neighbors``, which pairs query
    ``i`` with a target point shifted by a fixed offset - never the true
    nearest point, unless a query happens to already be there. Shifted by an
    offset rather than collapsed onto a single point: pairing every query
    with the SAME target point makes Horn's method degenerate (every
    correspondence identical) and raise, a real but different confirmation
    of wiring than this test wants. The offset keeps correspondences varied
    enough to stay solvable while still being wrong, so a genuine wiring
    failure shows up as the fitted transform silently NOT moving, not as a
    different kind of refusal. Matches
    ``tests.test_icp_plane_backend._CorrespondenceStub``, and its own
    docstring's note about why an offset was chosen over collapsing to one
    point.
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


def test_icp_actually_uses_the_backends_correspondence_search() -> None:
    source, target = _scene(300, seed=42)
    reference = icp(source, target)
    stubbed = icp(source, target, backend=_CorrespondenceStub())

    assert stubbed.resolved_backend == "stub-correspondence"
    rot_err = _rotation_error(reference.transform, stubbed.transform)
    trans_err = math.dist(reference.transform.translation, stubbed.transform.translation)
    assert rot_err > 1e-3 or trans_err > 1e-3, (
        "a deliberately wrong nearest_neighbors left the fitted transform unchanged - "
        "the correspondence search does not appear to be routed through the backend"
    )


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_explicit_backend_is_used_for_the_correspondence_search(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call-counting spy on the REAL backend class: proves
    ``apply_transform`` AND ``nearest_neighbors`` (the correspondence
    search) are both genuinely invoked for an explicit non-default request.
    ``icp`` never calls ``k_nearest_neighbors`` - it has no k-NN operation,
    unlike :mod:`scanner3d.icp_plane` - so that method is not checked here.
    """
    cls = ACCELERATED_BACKENDS[backend_name]
    calls = {"apply_transform": 0, "nearest_neighbors": 0}
    for method_name in calls:
        original = getattr(cls, method_name)

        def wrapper(self, *args, __original=original, __name=method_name, **kwargs):
            calls[__name] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method_name, wrapper)

    source, target = _scene(300, seed=9)
    result = icp(source, target, backend=backend_name)

    assert result.resolved_backend == backend_name
    assert calls["apply_transform"] > 0, "apply_transform was never called"
    assert calls["nearest_neighbors"] > 0, "nearest_neighbors was never called"


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_a_selected_backend_never_builds_the_legacy_voxel_index(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live :mod:`scanner3d.compute` backend "never grids at all" (see
    that module's own docstring), so an accelerated request must not build
    a :class:`~scanner3d.registration._VoxelIndex` at all - not merely
    leave it unused. Patched to raise so an unused-but-still-constructed
    index (a real, if purely wasteful, difference from
    :mod:`scanner3d.icp_plane`'s parallel design) is caught rather than
    silently tolerated.
    """

    def _boom(self, *args, **kwargs):
        raise AssertionError("_VoxelIndex was constructed on an accelerated-backend request")

    monkeypatch.setattr(_VoxelIndex, "__init__", _boom)

    source, target = _scene(120, seed=25)
    result = icp(source, target, backend=backend_name)
    assert result.resolved_backend == backend_name


@pytest.mark.parametrize("backend_name", ACCELERATED_PARAMS)
def test_the_backends_correspondence_search_receives_the_resolved_gate(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call-argument spy, not just a call-count one: the ``gate`` passed to
    ``nearest_neighbors`` on every iteration must be the SAME gate the
    legacy path would have resolved (the target's own extent times
    ``DEFAULT_CORRESPONDENCE_FRACTION`` here, since neither ``voxel_size``
    nor ``max_correspondence_distance`` is overridden) - proving the value
    is actually threaded through, not merely that some call happened.
    """
    cls = ACCELERATED_BACKENDS[backend_name]
    original = getattr(cls, "nearest_neighbors")  # noqa: B009
    seen_gates: list[float] = []

    def wrapper(self, queries, target, *, gate, __original=original):
        seen_gates.append(gate)
        return __original(self, queries, target, gate=gate)

    monkeypatch.setattr(cls, "nearest_neighbors", wrapper)

    source, target = _scene(200, seed=27)
    icp(source, target, backend=backend_name)

    expected_gate = cloud_extent(target) * DEFAULT_CORRESPONDENCE_FRACTION
    assert seen_gates, "nearest_neighbors was never called"
    assert all(g == pytest.approx(expected_gate) for g in seen_gates), seen_gates


def test_default_backend_never_touches_compute_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The complementary claim to the stub test above: the DEFAULT path must
    not merely produce the right answer, it must never call into
    :mod:`scanner3d.compute` at all. Every method on every backend class is
    patched to raise; running with the default ``backend`` (both explicitly
    and by omission) must not trigger any of them.
    """

    def _boom(self, *args, **kwargs):
        raise AssertionError("a scanner3d.compute backend method was called on the default path")

    method_names = ("apply_transform", "nearest_neighbors", "rigid_from_correspondences")
    for cls in (_PythonBackend, _NumpyBackend, _ScipyBackend):
        for method_name in method_names:
            monkeypatch.setattr(cls, method_name, _boom)

    source, target = _scene(300, seed=11)
    result_default = icp(source, target)
    result_explicit = icp(source, target, backend="python")
    assert result_default.resolved_backend == PYTHON_BACKEND
    assert result_explicit.resolved_backend == PYTHON_BACKEND
    assert result_default.converged or result_default.iterations > 0
    assert result_explicit.converged or result_explicit.iterations > 0


def test_the_literal_string_python_never_even_calls_select_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stronger than the raise-guard above: the literal string ``"python"``
    must be a genuine early-return sentinel in ``_resolve_backend``, never a
    request that happens to be routed through
    :func:`scanner3d.compute.select_backend` and resolve to the same place.
    ``"auto"`` DOES need to call ``select_backend`` (that is how it decides
    what "auto" means), so this checks the literal string specifically, not
    every path that can end up reporting ``resolved_backend == "python"``.
    """

    def _boom(name: str = "auto"):
        raise AssertionError("select_backend was called for the literal string 'python'")

    monkeypatch.setattr("scanner3d.compute.select_backend", _boom)

    source, target = _scene(120, seed=21)
    result = icp(source, target, backend="python")
    assert result.resolved_backend == PYTHON_BACKEND


def test_auto_resolving_to_python_also_never_touches_compute_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same claim, reached through ``"auto"`` rather than the literal
    string ``"python"`` or omission - the ONLY other way
    ``_resolve_backend`` can decide "run the legacy path". Both numpy and
    scipy are forced unavailable so "auto" has nowhere else to resolve to.
    """
    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: not installed")
    )
    monkeypatch.setattr(
        "scanner3d.compute._probe_scipy", lambda: (False, "simulated: not installed")
    )

    def _boom(self, *args, **kwargs):
        raise AssertionError("a scanner3d.compute backend method was called on the default path")

    method_names = ("apply_transform", "nearest_neighbors", "rigid_from_correspondences")
    for cls in (_PythonBackend, _NumpyBackend, _ScipyBackend):
        for method_name in method_names:
            monkeypatch.setattr(cls, method_name, _boom)

    source, target = _scene(300, seed=17)
    result = icp(source, target, backend="auto")
    assert result.resolved_backend == PYTHON_BACKEND


# --- resolved_backend reporting -------------------------------------------------------


def test_resolved_backend_defaults_to_python() -> None:
    source, target = _scene(60, seed=13)
    result = icp(source, target)
    assert result.resolved_backend == PYTHON_BACKEND


def test_result_as_dict_includes_resolved_backend() -> None:
    source, target = _scene(60, seed=18)
    result = icp(source, target, backend="numpy" if NUMPY_AVAILABLE else "python")
    record = result.as_dict()
    assert "resolved_backend" in record
    assert record["resolved_backend"] == result.resolved_backend


def test_resolved_backend_auto_reports_what_it_chose() -> None:
    source, target = _scene(60, seed=14)
    result = icp(source, target, backend="auto")
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
        icp(source, target, backend="quantum")


def test_explicit_unavailable_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scanner3d.compute._probe_numpy", lambda: (False, "simulated: not installed")
    )
    source, target = _scene(60, seed=16)
    with pytest.raises(ComputeBackendUnavailable):
        icp(source, target, backend="numpy")


def test_backend_instance_opts_into_the_backend_path_even_named_python() -> None:
    """A live ComputeBackend instance is always the backend-driven path, even
    one whose OWN ``.name`` happens to be ``"python"`` - only the bare
    STRING ``"python"`` means "use the legacy path". Matches
    ``_resolve_backend``'s documented distinction exactly.
    """
    source, target = _scene(120, seed=20)
    stub = _CorrespondenceStub()
    stub.name = "python"
    result = icp(source, target, backend=stub)
    assert result.resolved_backend == "python"
    reference = icp(source, target)
    rot_err = _rotation_error(reference.transform, result.transform)
    trans_err = math.dist(reference.transform.translation, result.transform.translation)
    assert rot_err > 1e-3 or trans_err > 1e-3, (
        "a live backend instance named 'python' was treated as the legacy sentinel"
    )


# --- end-to-end timing: informational only, no assertion on the speedup ---------------


@needs_numpy
def test_speed_report_is_informational_only(capsys: pytest.CaptureFixture[str]) -> None:
    """One full ``icp`` call per backend, warm, at the sizes requested for
    the task report. Asserts nothing about relative speed - the measured
    numbers belong in the task report, not a flaky timing assertion.
    """
    backends = [PYTHON_BACKEND, NUMPY_BACKEND]
    if SCIPY_AVAILABLE:
        backends.append(SCIPY_BACKEND)

    report = []
    for n in (2000, 2500):
        source, target = _scene(n, seed=1000 + n)
        # Warm each backend once before timing (pays any first-import tax).
        for name in backends:
            icp(source[:20], target[:20], backend=name, max_iterations=1)
        for name in backends:
            start = time.perf_counter()
            result = icp(source, target, backend=name)
            elapsed = time.perf_counter() - start
            report.append(
                f"n={n} {name:8s}: {elapsed * 1000:8.1f} ms "
                f"({result.iterations} iters, converged={result.converged})"
            )
    print("\n" + "\n".join(report))
    assert report
