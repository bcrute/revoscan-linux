"""Optional real-GPU kernel checks. Never opens or controls the scanner.

Opt in with SCANNER3D_TEST_GPU=1 and a working desktop OpenGL context.
Replay equality is tested separately by tools/gpu_tracking_probe.py against
physical recordings, since synthetic kernel checks cannot establish it.
"""

from __future__ import annotations

import os

import pytest
from tools.gpu_tracking_probe import ORIGINAL_NORMALS, GpuProbe

from scanner3d import icp_plane
from scanner3d.compute import ComputeParametersUnusable, select_backend

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("SCANNER3D_TEST_GPU") != "1":
        pytest.skip("GPU experiment requires SCANNER3D_TEST_GPU=1")
    pytest.importorskip("OpenGL")
    pytest.importorskip("PySide6")
    probe = GpuProbe()
    yield probe
    probe.close()
    probe.close()  # teardown is idempotent


@pytest.mark.parametrize("gate", [0.0, -1.0, float("inf"), float("nan")])
def test_nearest_invalid_gate_is_refused_before_dispatch(gpu, gate):
    previous = gpu.calls["nearest"]
    with pytest.raises(ComputeParametersUnusable):
        gpu.nearest_neighbors([], [], gate=gate)
    assert gpu.calls["nearest"] == previous


def test_nearest_empty_and_strict_gate(gpu):
    assert gpu.nearest_neighbors([], [(1.0, 0.0, 0.0)], gate=1.0) == []
    assert gpu.nearest_neighbors([(0.0, 0.0, 0.0)], [], gate=1.0) == [None]
    assert gpu.nearest_neighbors([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], gate=1.0) == [None]
    assert gpu.nearest_neighbors([(0.0, 0.0, 0.0)], [(0.5, 0.0, 0.0)], gate=1.0) == [
        (0.5, (0.5, 0.0, 0.0))
    ]


def test_nearest_matches_scipy_with_resident_targets(gpu):
    rng = np.random.default_rng(167)
    queries = [tuple(row) for row in rng.normal(size=(900, 3))]
    targets = [tuple(row) for row in rng.normal(size=(1700, 3))]
    expected = select_backend("scipy").backend.nearest_neighbors(queries, targets, gate=0.2)
    for _ in range(2):
        actual = gpu.nearest_neighbors(queries, targets, gate=0.2)
        for a, b in zip(actual, expected, strict=True):
            if b is None:
                assert a is None
            else:
                assert a is not None
                assert a[1] == b[1]
                assert a[0] == pytest.approx(b[0], abs=1e-14)
    before = gpu.calls["target_uploads"]
    gpu.nearest_neighbors(queries, targets, gate=0.2)
    assert gpu.calls["target_uploads"] == before


def test_target_cache_is_bounded_and_value_keyed(gpu):
    target = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    for index in range(12):
        target[0] = (index + 0.1, 0.0, 0.0)
        hit = gpu.nearest_neighbors([target[0]], target, gate=0.01)
        assert hit == [(0.0, target[0])]
        assert len(gpu.target_buffers) <= 8


def test_batched_jacobi_matches_cpu_rotations(gpu):
    rng = np.random.default_rng(51)
    matrices = [np.zeros((3, 3)).tolist(), np.eye(3).tolist()]
    for _ in range(1000):
        values = rng.normal(size=(12, 3))
        matrices.append((values.T @ values).tolist())
    actual = gpu.jacobi(matrices)
    for matrix, result in zip(matrices, actual, strict=True):
        expected = icp_plane._jacobi(matrix)
        assert result == expected
    assert gpu.jacobi([]) == []


@pytest.mark.parametrize("kind", ["curved", "coincident", "sparse"])
def test_normal_field_matches_reference_including_refusals(gpu, kind):
    rng = np.random.default_rng(117)
    if kind == "curved":
        x = rng.normal(size=(500, 3))
        x[:, 2] = 0.2 * x[:, 0] ** 2 + 0.3 * x[:, 1] ** 2
        points = [tuple(row) for row in x]
    elif kind == "coincident":
        points = [(1.0, 1.0, 1.0)] * 12
    else:
        points = [(1.0, 1.0, 1.0)]
    expected = ORIGINAL_NORMALS(points, backend="scipy")
    actual = gpu.estimate_normals(points, backend="scipy")
    assert actual == expected


def test_production_backend_is_not_changed_by_probe_import():
    assert select_backend("auto").resolved == "scipy"
    assert icp_plane.estimate_normals is ORIGINAL_NORMALS
