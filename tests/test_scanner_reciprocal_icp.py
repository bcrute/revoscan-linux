"""Reciprocity counts, exact ties, invariant mapping, fit and context guards."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis
from tests.test_scanner_solver_mechanisms import fixture_pairs, moved


@pytest.fixture
def reciprocal(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_reciprocal_icp")


def test_many_to_one_matching_keeps_only_reciprocal_source(reciprocal):
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    pairs = [(float(i), p, (0.1, 0.0, 0.0), (0.0, 1.0, 0.0)) for i, p in enumerate(points)]
    actual, removed = reciprocal.reciprocal_pairs(points, Rigid.identity(), pairs)
    assert actual == pairs[:1] and removed == 2


def test_source_without_forward_match_still_participates_in_reverse_search(reciprocal):
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    pairs = [(1.0, points[1], (0.1, 0.0, 0.0), (0.0, 1.0, 0.0))]
    assert reciprocal.reciprocal_pairs(points, Rigid.identity(), pairs) == ([], 1)


def test_reverse_exact_tie_uses_first_original_source_index(reciprocal):
    points = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    pairs = [(1.0, p, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) for p in points[:2]]
    assert reciprocal.reciprocal_pairs(points, Rigid.identity(), pairs) == (pairs[:1], 1)
    reversed_source = tuple(reversed(points))
    assert reciprocal.reciprocal_pairs(reversed_source, Rigid.identity(), pairs) == (pairs[1:], 1)


def test_pairs_order_and_geometry_are_preserved_and_match_brute_force(reciprocal):
    rng = np.random.default_rng(51809)
    points = tuple(tuple(p) for p in rng.normal(size=(80, 3)))
    transform = Rigid.of(rotation_about_axis((1.0, 2.0, 3.0), 0.21).rotation, (2.0, 3.0, -1.0))
    placed = moved(transform, points)
    targets = placed + rng.normal(size=(80, 3)) * 0.25
    pairs = [
        (float(i), p, tuple(q), (0.0, 0.0, 1.0))
        for i, (p, q) in enumerate(zip(points, targets, strict=True))
    ]
    expected = [
        pair
        for i, pair in enumerate(pairs)
        if min(range(len(points)), key=lambda j: (np.sum((placed[j] - targets[i]) ** 2), j)) == i
    ]
    actual, removed = reciprocal.reciprocal_pairs(points, transform, pairs)
    assert actual == expected and removed == len(pairs) - len(expected)


@pytest.mark.parametrize("profile", ["reciprocal-origin", "reciprocal-centered"])
def test_known_nearest_correspondences_recover_truth(reciprocal, profile):
    pairs, truth = fixture_pairs()
    source = [p for p, _, _ in pairs]
    # Small known offset keeps each nearest correspondence reciprocal.
    initial = Rigid.of(truth.rotation, tuple(t + 0.01 for t in truth.translation))

    def correspond(points, transform):
        return [
            (float(np.sum((np.asarray(transform.apply(p)) - q) ** 2)), p, q, n) for p, q, n in pairs
        ], 0

    experiment = reciprocal.ReciprocalExperiment(profile)
    with experiment.active():
        result = icp_plane._iterate(
            source,
            correspond,
            initial=initial,
            max_iterations=10,
            threshold=1e-7,
            keep_fraction=0.8,
            gate=12.0,
            accelerated=True,
        )
    assert (
        np.max(np.linalg.norm(moved(result.transform, source) - moved(truth, source), axis=1))
        < 1e-6
    )
    assert result.converged and experiment.records[0]["status"] == "completed"


@pytest.mark.parametrize(
    "bad",
    [[], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)] * 2, [(float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0)]],
)
def test_invalid_source_is_refused(reciprocal, bad):
    with pytest.raises(ValueError):
        reciprocal.reciprocal_pairs(bad, Rigid.identity(), [])


def test_unmapped_forward_source_refuses(reciprocal):
    with pytest.raises(ValueError, match="absent"):
        reciprocal.reciprocal_pairs(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            Rigid.identity(),
            [(0.0, (2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
        )


def test_patch_restoration_and_cross_solver_nesting(reciprocal):
    old_step, old_iteration = icp_plane._solve_point_to_plane, icp_plane._iterate
    experiment = reciprocal.ReciprocalExperiment("reciprocal-origin")
    with pytest.raises(RuntimeError, match="injected"), experiment.active():
        with pytest.raises(RuntimeError, match="nested"), experiment.active():
            pass
        raise RuntimeError("injected")
    assert icp_plane._solve_point_to_plane is old_step and icp_plane._iterate is old_iteration
    assert experiment._iteration is None and experiment._step is None


def test_iteration_preserves_settling_and_original_step_on_exact_pairs(reciprocal, monkeypatch):
    settling = importlib.import_module("scanner_pose_settling")
    monkeypatch.setattr(icp_plane, "_iterate", settling.iterate)
    experiment = reciprocal.ReciprocalExperiment("reciprocal-origin")
    with experiment.active():
        assert experiment._iteration is settling.iterate
    assert icp_plane._iterate is settling.iterate


def test_closed_profiles_source_indices_and_outside_context_refuse(reciprocal):
    with pytest.raises(ValueError):
        reciprocal.ReciprocalExperiment("retuned")
    experiment = reciprocal.ReciprocalExperiment("reciprocal-centered")
    with pytest.raises(ValueError):
        experiment.mark_frame(3000)
    with pytest.raises(RuntimeError, match="context"):
        experiment.step([], Rigid.identity())


def test_duplicate_forward_source_cannot_inflate_constraints(reciprocal):
    source = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    pair = (0.0, source[0], source[0], (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="one original forward"):
        reciprocal.reciprocal_pairs(source, Rigid.identity(), [pair, pair])


@pytest.mark.parametrize("profile", ["reciprocal-origin", "reciprocal-centered"])
def test_removed_pairs_count_as_rejected_before_unchanged_trim(reciprocal, profile):
    pairs, truth = fixture_pairs()
    source = [p for p, _, _ in pairs]
    target = np.asarray([q for _, q, _ in pairs])
    distances = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    a, b = np.unravel_index(np.argmin(distances), distances.shape)
    assert distances[a, b] < 12
    forward = [(0.0, p, q, n) for p, q, n in pairs]
    forward[a] = (float(distances[a, b] ** 2), pairs[a][0], pairs[b][1], pairs[b][2])
    experiment = reciprocal.ReciprocalExperiment(profile)
    with experiment.active():
        experiment.mark_frame(5)
        result = icp_plane._iterate(
            source,
            lambda *_: (forward, 0),
            initial=truth,
            max_iterations=1,
            threshold=1e-6,
            keep_fraction=0.8,
            gate=12.0,
            accelerated=True,
        )
    assert result.rejected == 1
    assert result.correspondences == icp_plane.kept_count(len(source) - 1, 0.8)
    assert result.rmse < 1e-10
    assert all(c["reciprocal_rejected"] == 1 for c in experiment.records[0]["calls"])
    assert all(r["source_index"] == 5 for r in experiment.records)
