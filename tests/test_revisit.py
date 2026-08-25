"""Tests for :mod:`scanner3d.revisit` - the module that found the user's real 65-degree error.

What is at stake
-----------------
:mod:`scanner3d.posegraph` can only fix a drifted trajectory if it is handed
closure edges, and this module is where those come from: it looks at a
sequence of captured frames and answers "have I seen this view before?"
using nothing but the frames' own content - no turntable, no angular step,
no capture path. Two properties matter more than any other here, and they
pull in opposite directions on purpose:

* **It must find a genuine revisit, at the right index gap.** See
  :func:`test_detect_revisits_finds_the_loop_closing_pair_at_the_right_gap`.
* **It must stay silent when there is nothing to find - and reject a close
  call, not just an obviously wrong one.** A false closure edge is worse
  than none: it would drag :func:`~scanner3d.posegraph.optimise_poses`
  toward a confidently wrong compromise instead of leaving an honest tree.
  See :func:`test_detect_revisits_reports_nothing_on_a_sweep_that_never_revisits`
  and, the sharper version of the same claim,
  :func:`test_verification_rejects_a_near_miss_the_signature_proposes`, where
  two frames are close enough in VIEWPOINT to be proposed by Stage 1 but not
  close enough in POSE to survive Stage 2.

A note on the test geometry: the synthetic box is rotated about a
deliberately oblique axis, not one aligned with any of the box's own
principal axes. A rectangular box is symmetric under a 180-degree turn about
each of its own three axes, and rotating it about one of those (an easy
mistake - the box happens to sit centred on the sweep's pivot) makes
non-adjacent frames look like genuine revisits when they are not, which was
caught here empirically before it could hide a real bug: see the git history
of this fixture's ``_AXIS`` constant for the numbers that showed it.

Every synthetic frame carries the scanner's measured 0.65-sigma depth noise
(:func:`scanner3d.simulate.render`'s own ``jitter``), because a noiseless
capture is a much easier problem than the real one.
"""

from __future__ import annotations

import functools
import json
import math
import random

import pytest

from scanner3d.compute import _probe_scipy
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, rotation_about_axis
from scanner3d.posegraph import PoseEdge, optimise_poses
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.revisit import (
    RevisitNotAllowed,
    _fit_quality,
    depth_signature_from_points,
    depth_signature_from_raw,
    detect_revisits,
    signature_distance,
)
from scanner3d.simulate import Box, points_of, render

SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

# A deliberately oblique axis - see the module docstring on why an axis
# aligned with the box's own symmetry (its local x, y or z) produces false
# revisits that have nothing to do with the loop actually closing.
_AXIS: Point = (0.55, 0.6, 0.58)
_THROUGH: Point = (0.0, 0.0, 117.0)
_BOX = Box(half_extents=(8.0, 12.0, 6.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 117.0)))

#: Downsample target: a few hundred to ~1000 points per frame. Coarse enough
#: that a pure-Python ICP call is fast (a fraction of a second), fine enough
#: that a genuine revisit still clears the default 400-correspondence floor
#: by a wide margin - see the measured numbers in each fixture's docstring.
_VOXEL_FRACTION = 1.0 / 24.0


def _frame_cloud(angle_deg: float, *, seed: int) -> tuple[Point, ...]:
    """One rendered, noisy, downsampled frame at ``angle_deg`` about ``_AXIS``."""
    pose = rotation_about_axis(_AXIS, math.radians(angle_deg), through=_THROUGH)
    frame = render([_BOX], pose=pose, jitter=0.65, seed=seed)
    points = tuple(points_of(frame))
    extent = cloud_extent(points)
    return voxel_downsample(points, extent * _VOXEL_FRACTION)


@functools.cache
def _loop_clouds() -> tuple[tuple[Point, ...], ...]:
    """13 frames, 30 degrees apart, closing exactly at frame 12 == frame 0.

    Frame 12 sits at 360 degrees about ``_AXIS`` - bit-for-bit the same
    viewpoint as frame 0 - which is the genuine revisit this fixture exists
    to produce. Every OTHER pair at least 3 frames apart is at least 90
    degrees away in true rotation, comfortably past any threshold used here.
    """
    return tuple(_frame_cloud(30.0 * k, seed=k) for k in range(13))


@functools.cache
def _loop_report():
    return detect_revisits(_loop_clouds(), min_index_gap=3, max_candidates=20)


@functools.cache
def _arc_clouds() -> tuple[tuple[Point, ...], ...]:
    """9 frames, 15 degrees apart, spanning 0 to 120 degrees - never returns.

    With ``min_index_gap=3`` the smallest true angular separation between
    any two COMPARED frames is 45 degrees, far past every default
    verification threshold. Nothing here should verify.
    """
    return tuple(_frame_cloud(15.0 * k, seed=100 + k) for k in range(9))


@functools.cache
def _arc_report():
    return detect_revisits(_arc_clouds(), min_index_gap=3, max_candidates=15)


@functools.cache
def _near_miss_clouds() -> tuple[tuple[Point, ...], ...]:
    """6 frames: indices 0 and 5 are 25 degrees apart - close, not a revisit.

    Frames 1-4 are filler, spread widely (50, 95, 140, 185 degrees) so they
    do not themselves resemble frame 0 or 5. Measured: at 25 degrees true
    offset, identity-seeded ICP recovers roughly 20 degrees of rotation and
    38-40 units of translation - both well past the default thresholds,
    while the SIGNATURE distance is small enough that Stage 1 still proposes
    the pair (see :func:`test_verification_rejects_a_near_miss_the_signature_proposes`).
    Below 25 degrees, point-to-point ICP's own tangential under-recovery
    (the exact bias :mod:`scanner3d.icp_plane` exists to fix) let a couple of
    tried offsets under-shoot below the rotation threshold and verify by
    accident - 25 degrees was chosen empirically to give a wide, honest
    margin on both sides of the gate instead of a marginal one.
    """
    angles = (0.0, 50.0, 95.0, 140.0, 185.0, 25.0)
    return tuple(_frame_cloud(angle, seed=300 + index) for index, angle in enumerate(angles))


@functools.cache
def _near_miss_report():
    return detect_revisits(_near_miss_clouds(), min_index_gap=3, max_candidates=20)


# --- the local-minimum bug: hand-built to reproduce the real measured pattern -----------
#
# Rendering two ~180-degree box views (as the fixtures above do) does not
# reproduce the real bug: the box is close enough to genuinely rotationally
# symmetric about this axis that identity-seeded ICP finds a real, complete
# match (matched_fraction ~1.0) - not the deceptive PARTIAL match the real
# capture showed (overlap 0.220-0.397, rmse BETTER than the genuine
# revisits'). Reproducing that specific pattern needs precise control over
# how much of each cloud coincidentally agrees under identity versus how
# much only agrees at the TRUE separation - control a rendered scene does not
# give directly. So this is built by hand: a "core" cluster present,
# near-identically, in both clouds (what makes identity look plausible) plus
# a "distinguishing" cluster unique to each, related by the TRUE ~179-degree
# rotation (what identity gets wrong and an alternative seed can recover).
#
# Measured on this fixture (see the task report for the full numbers): the
# identity fit lands at rmse~0.47, matched_fraction~0.37, comfortably inside
# every existing threshold - and its raw rmse is BETTER than the best
# alternative's (~0.55), reproducing the exact inversion that makes rmse
# alone useless here. Coverage-adjusted (:func:`scanner3d.revisit._fit_quality`)
# identity measures ~1.26 against the best alternative's ~0.55 - a ~2.3x gap,
# comfortably past :data:`~scanner3d.revisit.DEFAULT_ALTERNATIVE_MARGIN`.

_FALSE_REVISIT_CENTRE: Point = (0.0, 0.0, 50.0)
_FALSE_REVISIT_AXIS: Point = (0.03, 0.02, 1.0)
_FALSE_REVISIT_TRUE_ANGLE_DEG = 179.0
_FALSE_REVISIT_CORE_N = 700
_FALSE_REVISIT_DIST_N = 1200


def _cluster(n: int, centre: Point, radius: float, *, seed: int) -> list[Point]:
    rng = random.Random(seed)
    return [
        (
            centre[0] + rng.uniform(-radius, radius),
            centre[1] + rng.uniform(-radius, radius),
            centre[2] + rng.uniform(-radius, radius),
        )
        for _ in range(n)
    ]


def _jitter_points(points: list[Point], amount: float, *, seed: int) -> list[Point]:
    rng = random.Random(seed)
    return [
        (p[0] + rng.gauss(0, amount), p[1] + rng.gauss(0, amount), p[2] + rng.gauss(0, amount))
        for p in points
    ]


@functools.cache
def _false_revisit_clouds() -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """Two hand-built clouds ~179 degrees apart that fool identity-seeded
    verification alone - see the section comment above for the construction
    and the measured numbers.
    """
    core = _cluster(_FALSE_REVISIT_CORE_N, _FALSE_REVISIT_CENTRE, 8.0, seed=1)
    dist_a = _cluster(
        _FALSE_REVISIT_DIST_N,
        (_FALSE_REVISIT_CENTRE[0] + 25.0, _FALSE_REVISIT_CENTRE[1], _FALSE_REVISIT_CENTRE[2]),
        10.0,
        seed=2,
    )
    true_transform = rotation_about_axis(
        _FALSE_REVISIT_AXIS,
        math.radians(_FALSE_REVISIT_TRUE_ANGLE_DEG),
        through=_FALSE_REVISIT_CENTRE,
    )
    dist_b = [true_transform.apply(p) for p in dist_a]
    cloud_a = tuple(_jitter_points(core, 0.3, seed=3) + _jitter_points(dist_a, 0.3, seed=4))
    cloud_b = tuple(_jitter_points(core, 0.3, seed=5) + _jitter_points(dist_b, 0.3, seed=6))
    return cloud_a, cloud_b


# --- 0. the local-minimum bug: current code accepts it, fixed code rejects it -----------


def test_identity_seeded_verification_alone_accepts_a_false_180_degree_revisit():
    """THE HEADLINE. Demonstrates the bug is real, not assumed: with the
    local-minimum check disabled (``alternative_seeds=0``, this module's
    pre-fix behaviour - still reachable through the parameter, not deleted -
    identity-seeded verification ALONE wrongly accepts a pair whose true
    separation is 179 degrees, exactly the pattern measured on the user's
    real capture. Confirmed to fail before writing the fix; see the task
    report for the standalone reproduction this test was built from.
    """
    a, b = _false_revisit_clouds()
    report = detect_revisits([a, b], min_index_gap=1, max_candidates=1, alternative_seeds=0)
    assert len(report.verified) == 1, "the false pair was not even accepted with the check off"
    revisit = report.verified[0]
    assert math.degrees(revisit.rotation_angle) < 12.0  # inside DEFAULT_MAX_ROTATION
    assert revisit.correspondences > 400  # inside DEFAULT_MIN_CORRESPONDENCES


def test_detect_revisits_rejects_the_false_180_degree_revisit_by_default():
    """THE FIX. The exact same pair the test above shows current code
    wrongly accepting is correctly rejected with the default (nonzero)
    ``alternative_seeds`` - the only difference between the two tests.
    """
    a, b = _false_revisit_clouds()
    report = detect_revisits([a, b], min_index_gap=1, max_candidates=1)
    assert report.verified == (), "the false 180-degree pair still verified with the fix active"
    assert len(report.rejected_as_local_minimum) == 1
    rejected = report.rejected_as_local_minimum[0]
    assert (rejected.i, rejected.j) == (0, 1)
    assert math.degrees(rejected.alternative_rotation_angle) > 90.0
    assert any("local minimum" in limitation for limitation in report.limitations)


def test_alternative_margin_is_genuinely_read_not_a_fixed_threshold():
    """``alternative_margin`` must actually change the verdict, not just be
    accepted and ignored. Measured on this fixture, the false pair's
    coverage-adjusted ratio is ~2.3x - comfortably past the DEFAULT margin
    (1.5, see the two tests above) but NOT past a margin of 3.0. A caller
    who explicitly asks for a more lenient margin must get the more lenient
    answer.
    """
    a, b = _false_revisit_clouds()
    lenient = detect_revisits([a, b], min_index_gap=1, max_candidates=1, alternative_margin=3.0)
    assert len(lenient.verified) == 1, (
        "a margin of 3.0 (above the measured ~2.3x ratio) should have been too lenient to "
        "reject the false pair - alternative_margin does not appear to be read"
    )
    assert lenient.rejected_as_local_minimum == ()


def test_fit_quality_prefers_higher_coverage_over_lower_raw_rmse():
    """The crux of the fix, isolated: a fit with WORSE raw rmse but much
    higher coverage must score BETTER (lower) than one with better raw rmse
    but partial coverage - otherwise the comparison would be made on exactly
    the quantity the bug hides behind (see the module docstring: the false
    fit's raw rmse measured BETTER than the genuine revisits'). Built from
    real :class:`~scanner3d.registration.IcpResult` objects on the
    false-revisit fixture, not hand-constructed doubles, so this is a claim
    about the real computation.
    """
    from scanner3d.multiview import coarse_align
    from scanner3d.registration import icp

    a, b = _false_revisit_clouds()
    identity_fit = icp(a, b)
    alignment = coarse_align(a, b)
    alt_candidate = next(
        c for c in alignment.candidates if math.degrees(c.transform.rotation_angle) > 90.0
    )
    alt_fit = icp(a, b, initial=alt_candidate.transform)

    assert alt_fit.matched_fraction > identity_fit.matched_fraction
    # Raw rmse comparison gets this backwards - identity's rmse looks better:
    assert identity_fit.rmse < alt_fit.rmse
    # Coverage-adjusted correctly reverses it - identity is the worse fit:
    assert _fit_quality(identity_fit) > _fit_quality(alt_fit)


@needs_scipy
def test_backend_does_not_change_which_pairs_verify():
    """Same verified set with and without an accelerated backend - checked
    on BOTH the genuine loop (identity wins clearly) and the false-positive
    pair (identity is correctly beaten), not just the easy case. Skips
    cleanly, not failing, when scipy is unavailable.
    """
    python_loop = detect_revisits(
        _loop_clouds(), min_index_gap=3, max_candidates=20, backend="python"
    )
    scipy_loop = detect_revisits(
        _loop_clouds(), min_index_gap=3, max_candidates=20, backend="scipy"
    )
    assert {(r.i, r.j) for r in python_loop.verified} == {(r.i, r.j) for r in scipy_loop.verified}

    a, b = _false_revisit_clouds()
    python_false = detect_revisits([a, b], min_index_gap=1, max_candidates=1, backend="python")
    scipy_false = detect_revisits([a, b], min_index_gap=1, max_candidates=1, backend="scipy")
    assert python_false.verified == ()
    assert scipy_false.verified == ()
    assert len(python_false.rejected_as_local_minimum) == 1
    assert len(scipy_false.rejected_as_local_minimum) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alternative_seeds": -1},
        {"alternative_margin": 1.0},
        {"alternative_margin": 0.5},
        {"alternative_margin": math.nan},
    ],
)
def test_detect_revisits_refuses_new_parameters_outside_their_ranges(kwargs: dict[str, object]):
    clouds = [[(0.0, 0.0, float(k + 1))] * 6 for k in range(8)]
    with pytest.raises(RevisitNotAllowed):
        detect_revisits(clouds, **kwargs)  # type: ignore[arg-type]


def test_alternative_seeds_zero_matches_pre_fix_behaviour_on_the_genuine_loop():
    """``alternative_seeds=0`` is not just "off" for the false-positive case -
    it must not change anything for a genuine revisit either, since it is
    documented as restoring this module's exact pre-fix code path.
    """
    report = detect_revisits(
        _loop_clouds(), min_index_gap=3, max_candidates=20, alternative_seeds=0
    )
    assert len(report.verified) == 1
    assert (report.verified[0].i, report.verified[0].j) == (0, 12)
    assert report.rejected_as_local_minimum == ()


def test_local_minimum_check_cost_is_informational_only(capsys: pytest.CaptureFixture[str]) -> None:
    """Measures the added Stage 2 cost this check pays, on the existing loop
    fixture - only for the one candidate that already passes the
    identity-seeded thresholds, per the module docstring. Asserts nothing
    about the specific numbers; they belong in the task report, not a
    pass/fail here - a timing assertion in a test suite is flaky by nature.
    """
    import time

    clouds = list(_loop_clouds())
    start = time.perf_counter()
    detect_revisits(clouds, min_index_gap=3, max_candidates=20, alternative_seeds=0)
    without = time.perf_counter() - start

    start = time.perf_counter()
    detect_revisits(clouds, min_index_gap=3, max_candidates=20)
    with_check = time.perf_counter() - start

    print(
        f"\nlocal-minimum check cost on the 13-frame loop fixture: "
        f"without={without * 1000:.1f}ms with={with_check * 1000:.1f}ms "
        f"(+{(with_check - without) * 1000:.1f}ms)"
    )


# --- 1. the core claim: finds the genuine revisit, at the right gap ---------------------


def test_detect_revisits_finds_the_loop_closing_pair_at_the_right_gap():
    report = _loop_report()

    assert len(report.verified) == 1, [
        (r.i, r.j, math.degrees(r.rotation_angle)) for r in report.verified
    ]
    revisit = report.verified[0]
    assert (revisit.i, revisit.j) == (0, 12)
    assert revisit.j - revisit.i == 12  # the right index gap: a full lap of 30-degree steps

    # Measured: well inside the default thresholds, with the wide margin the
    # module docstring claims - not a marginal pass.
    assert math.degrees(revisit.rotation_angle) < 1.0
    assert revisit.translation_magnitude < 1.0
    assert revisit.correspondences > 400


# --- 2. the critical negative, and the sharper near-miss version of it -------------------


def test_detect_revisits_reports_nothing_on_a_sweep_that_never_revisits():
    """A straight partial arc must produce zero verified revisits.

    A detector that hallucinates a closure is worse than one that finds
    nothing: a false edge drags the pose graph toward a confidently wrong
    compromise instead of leaving an honest, unimproved tree.
    """
    report = _arc_report()
    assert report.verified == ()
    assert report.pose_edges() == ()
    assert "no revisit" in " ".join(report.limitations) or "genuinely" in " ".join(
        report.limitations
    )


def test_verification_rejects_a_near_miss_the_signature_proposes():
    """Close enough to be proposed, not close enough to verify.

    This is the sharper claim than the arc test above: it is not enough for
    Stage 2 to reject an obviously unrelated pair, it must reject one Stage 1
    genuinely thought was promising.
    """
    report = _near_miss_report()
    proposed_pairs = {(c.i, c.j) for c in report.candidates}
    verified_pairs = {(r.i, r.j) for r in report.verified}

    assert (0, 5) in proposed_pairs, "the near-miss pair was not even proposed by signature"
    assert (0, 5) not in verified_pairs, "the near-miss pair survived verification"
    assert report.verified == (), "no pair in this fixture is a genuine revisit"


# --- 4. verification actually rejects -----------------------------------------------------


def test_verification_rejects_when_fed_a_pair_that_is_not_a_real_revisit():
    """The same claim as the near-miss test, read off the report structurally.

    ``proposed`` counts every pair Stage 1 handed to Stage 2; ``verified`` is
    a strict subset whenever verification is doing its job. On this fixture
    it is a much smaller subset - most proposals are not real revisits.
    """
    report = _near_miss_report()
    assert report.proposed > len(report.verified)
    assert report.verification_rate < 0.5


# --- 5. composes with posegraph without index fiddling -----------------------------------


def test_revisit_edges_compose_with_optimise_poses_directly():
    """The seam this module exists to feed: pose_edges() straight into optimise_poses."""
    report = _loop_report()
    ground_truth = [
        rotation_about_axis(_AXIS, math.radians(30.0 * k), through=_THROUGH) for k in range(13)
    ]
    # A minimal set of sequential edges, built the same way registration.py's
    # own chaining would state them - PoseEdge(i, i+1, transform, weight)
    # where transform carries frame i's local cloud onto frame i+1's. Built
    # here from ground truth rather than a second ICP pass, since this test
    # is about API composition, not about re-proving posegraph's own numbers.
    sequential = [
        PoseEdge(k, k + 1, ground_truth[k + 1].compose(ground_truth[k].inverse()).inverse(), 1.0)
        for k in range(12)
    ]
    edges = sequential + list(report.pose_edges())
    assert any(e.i == 0 and e.j == 12 for e in edges), "the closure edge did not carry over"

    result = optimise_poses(ground_truth, edges, anchor=0, iterations=20)
    assert result.poses[0] == ground_truth[0]
    assert len(result.edge_residuals) == len(edges)


# --- 3. signature behaviour ------------------------------------------------------------


def test_signature_of_identical_input_has_zero_distance():
    points = [(0.0, 0.0, z) for z in (5.0, 12.0, 30.0, 44.0, 61.0)]
    a = depth_signature_from_points(points, max_depth=100.0)
    b = depth_signature_from_points(points, max_depth=100.0)
    assert signature_distance(a, b) == 0.0


def test_signature_of_clearly_different_scenes_has_a_large_distance():
    near = [(0.0, 0.0, float(z)) for z in range(5, 15)]
    far = [(0.0, 0.0, float(z)) for z in range(85, 95)]
    a = depth_signature_from_points(near, max_depth=100.0)
    b = depth_signature_from_points(far, max_depth=100.0)
    # Two normalised histograms sharing no mass: L1 distance is at its
    # maximum of 2.0. A large fraction of that ceiling is the honest bar.
    assert signature_distance(a, b) > 1.5


def test_signature_distance_is_symmetric():
    a = depth_signature_from_points([(0.0, 0.0, float(z)) for z in range(1, 20)], max_depth=50.0)
    b = depth_signature_from_points([(0.0, 0.0, float(z)) for z in range(22, 41)], max_depth=50.0)
    assert signature_distance(a, b) == signature_distance(b, a)


def test_depth_signature_from_raw_agrees_with_depth_signature_from_points():
    """The two interfaces describe the same quantity - see the module docstring."""
    frame = render([_BOX], jitter=0.0, seed=7)
    from_raw = depth_signature_from_raw(frame.raw, depth_scale=frame.depth_scale, max_depth=200.0)
    from_points = depth_signature_from_points(tuple(points_of(frame)), max_depth=200.0)
    assert signature_distance(from_raw, from_points) < 1e-9
    assert from_raw.returned == from_points.returned


# --- 6. guards ----------------------------------------------------------------------------


def test_detect_revisits_refuses_too_few_frames():
    with pytest.raises(RevisitNotAllowed):
        detect_revisits([[(0.0, 0.0, 1.0)]])


def test_detect_revisits_refuses_a_min_index_gap_not_smaller_than_the_sequence():
    clouds = [[(0.0, 0.0, float(k + 1))] * 4 for k in range(4)]
    with pytest.raises(RevisitNotAllowed):
        detect_revisits(clouds, min_index_gap=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_index_gap": 0},
        {"max_candidates": 0},
        {"bins": 0},
        {"max_rotation": 0.0},
        {"max_rotation": -1.0},
        {"max_translation": 0.0},
        {"max_rmse": -0.5},
        {"min_correspondences": 0},
    ],
)
def test_detect_revisits_refuses_parameters_outside_their_ranges(kwargs: dict[str, object]):
    clouds = [[(0.0, 0.0, float(k + 1))] * 6 for k in range(8)]
    with pytest.raises(RevisitNotAllowed):
        detect_revisits(clouds, **kwargs)  # type: ignore[arg-type]


def test_depth_signature_from_points_refuses_bad_bins_and_max_depth():
    points = [(0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
    with pytest.raises(RevisitNotAllowed):
        depth_signature_from_points(points, bins=0, max_depth=10.0)
    with pytest.raises(RevisitNotAllowed):
        depth_signature_from_points(points, max_depth=0.0)


def test_signature_distance_refuses_mismatched_signatures():
    a = depth_signature_from_points([(0.0, 0.0, 1.0)], bins=10, max_depth=10.0)
    b = depth_signature_from_points([(0.0, 0.0, 1.0)], bins=20, max_depth=10.0)
    c = depth_signature_from_points([(0.0, 0.0, 1.0)], bins=10, max_depth=5.0)
    with pytest.raises(RevisitNotAllowed):
        signature_distance(a, b)
    with pytest.raises(RevisitNotAllowed):
        signature_distance(a, c)


# --- structure and serialisation -----------------------------------------------------


def test_revisit_report_as_dict_is_json_safe_and_carries_the_schema_version():
    report = _near_miss_report()
    record = json.loads(json.dumps(report.as_dict()))
    assert record["schema_version"] == 1
    assert record["proposed"] == report.proposed
    assert record["verified"] == len(report.verified)


def test_depth_signature_as_dict_is_json_safe():
    signature = depth_signature_from_points([(0.0, 0.0, 5.0)], max_depth=10.0)
    record = json.loads(json.dumps(signature.as_dict()))
    assert record["schema_version"] == 1
    assert isinstance(record["histogram"], list)


def test_a_verified_revisit_becomes_a_pose_edge_with_matching_fields():
    report = _loop_report()
    revisit = report.verified[0]
    edge = revisit.as_pose_edge(weight=2.5)
    assert isinstance(edge, PoseEdge)
    assert (edge.i, edge.j, edge.weight) == (revisit.i, revisit.j, 2.5)
    assert edge.transform == revisit.transform
