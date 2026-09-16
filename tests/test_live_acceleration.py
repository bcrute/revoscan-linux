"""Numerical and lifecycle checks for the live hybrid processing path."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from scanner3d import compute, coverage, gpu_normals, icp_plane, support_plane, tracking
from scanner3d.geometry import Rigid, normalize, rotation_about_axis
from scanner3d.multiview import overlap_fraction
from scanner3d.viewer import render_frame, render_packed_frame

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


def test_pose_pair_cache_covers_reference_archive_and_remains_exact_and_bounded():
    gap = tracking.IncrementalTracker._pose_gap
    gap.cache_clear()
    try:
        poses = [rotation_about_axis((1.0, 2.0, 3.0), i * 0.013) for i in range(65)]
        expected = {}
        for i, a in enumerate(poses):
            for j, b in enumerate(poses):
                if i == j:
                    continue
                relative = a.inverse().compose(b)
                expected[i, j] = (
                    math.degrees(relative.rotation_angle),
                    relative.translation_magnitude,
                )
                assert gap(a, b) == expected[i, j]
        before = gap.cache_info()
        for (i, j), value in expected.items():
            assert gap(poses[i], poses[j]) == value
        after = gap.cache_info()
        assert after.misses == before.misses
        assert after.hits - before.hits == 65 * 64
        assert after.maxsize == tracking.POSE_GAP_CACHE_PAIRS
        identity = Rigid.identity()
        for i in range(tracking.POSE_GAP_CACHE_PAIRS + 1):
            translated = Rigid.of(identity.rotation, (float(i + 100), 0.0, 0.0))
            assert gap(identity, translated) == (0.0, float(i + 100))
        assert gap.cache_info().currsize == tracking.POSE_GAP_CACHE_PAIRS
    finally:
        gap.cache_clear()


@pytest.mark.parametrize("kind", ["blank", "saturated", "all_codes", "random", "flat"])
def test_packed_preview_is_byte_and_stat_identical(kind):
    rng = np.random.default_rng(51)
    codes = rng.integers(0, 65536, 256000, dtype=np.uint16)
    if kind == "blank":
        codes[:] = 0
    elif kind == "saturated":
        codes[:] = 65535
    elif kind == "all_codes":
        codes = np.resize(np.arange(65536, dtype=np.uint16), 256000)
    elif kind == "flat":
        codes[:] = 4500
    codes[::5] = 0
    raw = codes.astype("<u2").tobytes()
    reference, actual = render_frame(raw), render_packed_frame(raw)
    assert actual.rgb == b"".join(reference.pixels)
    assert actual.stats == reference.stats
    assert tuple(actual.samples) == reference.samples
    assert not np.asarray(actual.samples).flags.writeable


@pytest.mark.parametrize("seed", range(8))
def test_sequential_vectorized_solve_keeps_exact_pose(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(180, 3, 3)).tolist()
    matched = [(tuple(a), tuple(b), normalize(tuple(n))) for a, b, n in points]
    current = rotation_about_axis((1.0, 2.0, 3.0), 0.05)
    assert icp_plane._solve_point_to_plane(matched, current, accelerated=True) == (
        icp_plane._solve_point_to_plane(matched, current)
    )


@pytest.mark.parametrize("tolerance", [0.1, 1.0, 2.0])
def test_indexed_overlap_keeps_inclusive_boundary(tolerance):
    source = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (-10.0, 0.0, 0.0)]
    target = [(tolerance, 0.0, 0.0), (10.0 + tolerance, 0.0, 0.0)]
    expected = overlap_fraction(source, target, Rigid.identity(), tolerance=tolerance)
    assert (
        overlap_fraction(source, target, Rigid.identity(), tolerance=tolerance, backend="scipy")
        == expected
    )


def test_indexed_overlap_matches_reference_on_random_clouds():
    rng = np.random.default_rng(15)
    for _ in range(8):
        source, target = rng.normal(size=(2, 200, 3)).tolist()
        pose = rotation_about_axis((1.0, 2.0, 3.0), 0.1)
        assert overlap_fraction(source, target, pose, tolerance=0.1, backend="scipy") == (
            overlap_fraction(source, target, pose, tolerance=0.1)
        )


@pytest.mark.parametrize("count", [2, 8, 12, 40, 64, 65])
def test_coverage_spread_retains_exact_reference_dot(count):
    rng = np.random.default_rng(count)
    for _ in range(5):
        directions = [normalize(tuple(p)) for p in rng.normal(size=(count, 3)).tolist()]
        expected = max(
            math.acos(max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b, strict=True)))))
            for i, a in enumerate(directions)
            for b in directions[i + 1 :]
        )
        assert coverage._max_pairwise_angle(directions) == expected


def test_support_refinement_retains_exact_inliers_and_plane():
    rng = np.random.default_rng(29)
    for _ in range(12):
        sample = [tuple(p) for p in rng.normal(size=(200, 3)).tolist()]
        normal = normalize(tuple(rng.normal(size=3).tolist()))
        spans = [max(p[axis] for p in sample) - min(p[axis] for p in sample) for axis in range(3)]
        diagonal = math.sqrt(sum(span**2 for span in spans))
        assert support_plane._refine(sample, normal, 0.8, diagonal=diagonal) == (
            support_plane._refine(sample, normal, 0.8, diagonal=diagonal, accelerated=False)
        )


@pytest.mark.parametrize("count", [1, 2, 3, 64, 900, 2200])
def test_batched_support_window_lands_on_the_scalar_loops_window(count):
    """The sliding window's end and centre, bit for bit, ties and rounding included."""
    rng = np.random.default_rng(count)
    for band in (1e-12, 0.05, 0.8, 3.0, 1e6):
        for offset in (0.0, 1e9):
            distances = (rng.normal(size=count) + offset).tolist()
            # Repeats and near-repeats, so ties and one-ulp differences occur.
            distances += [distances[0]] * 3 + [distances[-1] + 1e-15 * abs(distances[-1])]
            assert support_plane._best_window(distances, band, accelerated=True) == (
                support_plane._best_window(distances, band)
            )


@pytest.mark.parametrize("count", [3, 64, 900, 2200])
def test_batched_support_covariance_preserves_correctly_rounded_sums(count):
    rng = np.random.default_rng(count)
    for offset in (0.0, 1e9):
        points = [tuple(p) for p in (rng.normal(size=(count, 3)) + offset).tolist()]
        assert support_plane._covariance(points, accelerated=True) == (
            support_plane._covariance(points)
        )


def test_scipy_target_cache_reuses_values_but_never_a_mutated_target():
    from scipy.spatial import cKDTree

    backend = compute._ScipyBackend()
    queries = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    target = [(0.1, 0.0, 0.0), (1.1, 0.0, 0.0)]
    with patch("scipy.spatial.cKDTree", wraps=cKDTree) as build:
        first = backend.nearest_neighbors(queries, target, gate=0.5)
        assert backend.nearest_neighbors(queries, list(target), gate=0.5) == first
        assert build.call_count == 1
        target[0] = (0.6, 0.0, 0.0)
        assert backend.nearest_neighbors(queries, target, gate=0.5)[0] is None
        assert build.call_count == 2
        large = [(float(i), 0.0, 0.0) for i in range(2049)]
        backend.nearest_neighbors(queries, large, gate=0.5)
        assert backend._tree is None and not backend._tree_points


def test_cached_scipy_queries_retain_fresh_tree_answers_including_ties():
    rng = np.random.default_rng(91)
    target = [tuple(p) for p in rng.normal(size=(500, 3)).tolist()]
    target.extend([(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)] * 5)
    cached = compute._ScipyBackend()
    for gate in (0.1, 0.5, 1.0):
        queries = [(0.0, 0.0, 0.0)] + [tuple(p) for p in rng.normal(size=(150, 3)).tolist()]
        assert cached.nearest_neighbors(queries, target, gate=gate) == (
            compute._ScipyBackend().nearest_neighbors(queries, target, gate=gate)
        )


def test_normal_provider_is_scoped_cached_and_thread_local():
    calls = []
    field = icp_plane.estimate_normals([], backend="python")

    def estimate(points, neighbours):
        calls.append((points, neighbours))
        return field

    with pytest.raises(RuntimeError, match="cancelled"), icp_plane.normal_acceleration(estimate):
        provider = icp_plane._NORMAL_ACCELERATOR.get()
        assert provider is not None
        assert provider((), 12) is field
        assert provider((), 12) is field
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(icp_plane._NORMAL_ACCELERATOR.get).result() is None
        assert len(calls) == 1
        raise RuntimeError("cancelled")
    assert icp_plane._NORMAL_ACCELERATOR.get() is None


def test_gpu_initialization_failure_keeps_cpu_and_reports_reason():
    messages = []
    with (
        patch.object(gpu_normals, "GpuNormals", side_effect=RuntimeError("no GL")),
        gpu_normals.live_normals(None, messages.append),
    ):
        assert icp_plane._NORMAL_ACCELERATOR.get() is None
    assert "CPU processing" in messages[0] and "no GL" in messages[0]


def test_gpu_dispatch_failure_falls_back_once_and_closes():
    class FakeGpu:
        renderer = "test renderer"
        closed = False

        def __init__(self, **kwargs):
            self.np = np
            self.calls = {}
            self.seconds = {}

        def jacobi(self, matrices):
            return [icp_plane._jacobi(m) for m in matrices]

        def estimate_normals(self, *args, **kwargs):
            raise RuntimeError("dispatch failed")

        def close(self):
            self.closed = True

    fake = FakeGpu()
    messages = []
    with (
        patch.object(gpu_normals, "GpuNormals", return_value=fake),
        gpu_normals.live_normals(None, messages.append),
    ):
        provider = icp_plane._NORMAL_ACCELERATOR.get()
        assert provider is not None
        assert provider((), 12) == icp_plane.estimate_normals([], neighbours=12, backend="scipy")
        assert provider((), 13) == icp_plane.estimate_normals([], neighbours=13, backend="scipy")
    assert fake.closed
    assert sum("continuing on CPU" in message for message in messages) == 1


def test_gpu_covariance_handoff_stays_packed_without_a_python_list_roundtrip(monkeypatch):
    gpu = object.__new__(gpu_normals.GpuNormals)
    gpu.np = np
    batches = []

    def cpu_kernel(matrices):
        assert isinstance(matrices, np.ndarray)
        assert matrices.dtype == np.float64 and matrices.flags.c_contiguous
        batches.append(matrices)
        return [icp_plane._jacobi(matrix.tolist()) for matrix in matrices]

    monkeypatch.setattr(gpu, "jacobi", cpu_kernel)
    points = [tuple(p) for p in np.random.default_rng(79).normal(size=(80, 3)).tolist()]
    assert gpu.estimate_normals(points) == icp_plane.estimate_normals(points, backend="scipy")
    assert len(batches) == 1 and batches[0].shape == (80, 3, 3)


@pytest.mark.skipif(os.environ.get("SCANNER3D_TEST_GPU") != "1", reason="real GPU opt-in")
def test_production_gpu_covariance_and_normals_match_reference():
    pytest.importorskip("OpenGL")
    pytest.importorskip("PySide6")
    gpu = gpu_normals.GpuNormals()
    rng = np.random.default_rng(51)
    try:
        values = rng.normal(size=(100, 12, 3))
        matrices = np.transpose(values, (0, 2, 1)) @ values
        assert gpu.jacobi(matrices) == gpu.jacobi(matrices.tolist())
        assert gpu.jacobi(np.empty((0, 3, 3))) == []
        for _ in range(8):
            points = [tuple(p) for p in rng.normal(size=(600, 3)).tolist()]
            assert gpu.estimate_normals(points) == icp_plane.estimate_normals(
                points, backend="scipy"
            )
    finally:
        gpu.close()
