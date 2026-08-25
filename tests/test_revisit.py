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

import pytest

from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, rotation_about_axis
from scanner3d.posegraph import PoseEdge, optimise_poses
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.revisit import (
    RevisitNotAllowed,
    depth_signature_from_points,
    depth_signature_from_raw,
    detect_revisits,
    signature_distance,
)
from scanner3d.simulate import Box, points_of, render

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
