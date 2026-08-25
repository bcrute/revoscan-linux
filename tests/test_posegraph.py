"""Tests for :mod:`scanner3d.posegraph`.

What is at stake
-----------------
This module is the fix for a failure that every OTHER quality signal misses.
Measured on a real two-lap capture: per-pair ICP rmse 0.65-0.75 (excellent),
100% frame overlap (excellent), self-consistency drift 2.1-2.9 (looks fine) -
and an actual pose error of 65 degrees after one lap, 139 after two, because
each pairwise ICP fit slightly under-rotates and the shortfall accumulates
linearly around the loop. Only loop closure - an independent ICP fit between
two frames that revisit the same physical viewpoint - catches it, because it
is evidence the chain was never fitted against. :func:`optimise_poses`'s job
is to spend that evidence, redistributing the closure discrepancy back around
the loop instead of leaving it piled at the one seam where the loop closed.

Everything here is built around one synthetic fixture, :func:`_loop_fixture`,
deliberately shaped like the real failure: a chain of frames evenly spaced
around one or more full revolutions, where every SEQUENTIAL edge under-
rotates by the same fixed fraction (simulating the measured ICP bias), and
CLOSURE edges state - correctly - that a frame reappears at the same physical
pose one lap later. The headline test
(:func:`test_optimise_poses_recovers_true_pose_from_biased_chain_via_closure`)
is the whole point of the module: with closure edges, the true pose error
must fall by a large factor; see
:func:`test_optimise_poses_leaves_an_unclosed_chain_essentially_unchanged`
for the honest negative case that shows *why* - a bare chain has nothing to
reconcile, because it was built to satisfy its own edges exactly.

Held-out validation (test 2 in the assignment) is the one that matters most
for real data, where there is no ground truth to check against: optimise
using only some closure edges and confirm the ones held back also improve.
That can only happen if the optimiser is correcting the underlying
trajectory rather than fitting whatever it was shown.
"""

from __future__ import annotations

import functools
import json
import math
import random
import statistics
from dataclasses import dataclass

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import Point, Rigid, ensure_rotation, pose_difference, rotation_about_axis
from scanner3d.posegraph import (
    MIN_ROBUST_EDGES,
    EdgeResidual,
    PoseEdge,
    PoseGraphNotAllowed,
    PoseGraphRobustDisconnected,
    _robust_weights,
    average_rotations,
    edge_residual,
    edge_residuals,
    optimise_poses,
)

# A tilted axis, not aligned with any coordinate axis, so a bug that only
# shows up off-axis (a transposed rotation, a sign flip in one component)
# cannot hide behind an axis-aligned special case.
_AXIS: Point = (1.0, 2.0, 2.0)


# --- fixtures -------------------------------------------------------------------


@dataclass(frozen=True)
class _LoopFixture:
    """A synthetic multi-lap sweep with a fixed per-step ICP under-rotation.

    ``true_poses`` is the ground truth. ``chain`` is what naive chaining (the
    thing :mod:`scanner3d.registration` does today) produces from the biased
    sequential edges alone - it is what a caller would hand to this module
    without any closure information. ``sequential_edges`` are the biased
    frame-to-frame measurements; ``closure_edges`` are the CORRECT, unbiased
    measurements between frames that share a physical pose one lap apart.
    """

    true_poses: tuple[Rigid, ...]
    chain: tuple[Rigid, ...]
    sequential_edges: tuple[PoseEdge, ...]
    closure_edges: tuple[PoseEdge, ...]
    per_lap: int
    laps: int


@functools.cache
def _loop_fixture(
    *, laps: int = 2, per_lap: int = 12, recovery_fraction: float = 0.82
) -> _LoopFixture:
    """Build the fixture described in the module docstring.

    Every sequential edge recovers only ``recovery_fraction`` of the true
    per-step rotation - matching the ~82% recovery measured on the real
    capture that motivated this module. Ground truth poses are honest
    rotations about a shared axis (a stand-in for ANY capture path that
    revisits its own poses periodically; nothing about this fixture or the
    module under test is turntable-specific).
    """
    true_step = 2.0 * math.pi / per_lap
    total = laps * per_lap

    true_poses = tuple(rotation_about_axis(_AXIS, index * true_step) for index in range(total))

    # The measured relative transform for a sequential pair: what ICP found
    # relating frame k's local cloud to frame k+1's, under-rotated by the
    # fixed bias. Matches this module's edge convention exactly.
    biased_step = rotation_about_axis(_AXIS, -true_step * recovery_fraction)
    chain = [Rigid.identity()]
    sequential_edges = []
    for index in range(total - 1):
        sequential_edges.append(PoseEdge(index, index + 1, biased_step, 1.0))
        chain.append(chain[-1].compose(biased_step.inverse()))

    # A closure edge fires whenever the camera returns to a pose it already
    # occupied: frame k and frame k + per_lap sit at the same true angle, one
    # full revolution apart, so the honest ICP measurement between their raw
    # clouds is the identity (up to whatever noise a real fit would have -
    # none is injected here, since the point of this fixture is to isolate
    # the systematic bias, not sensor noise).
    closure_edges = tuple(
        PoseEdge(index, index + per_lap, Rigid.identity(), 1.0)
        for index in range(per_lap * (laps - 1))
    )

    return _LoopFixture(
        true_poses=true_poses,
        chain=tuple(chain),
        sequential_edges=tuple(sequential_edges),
        closure_edges=closure_edges,
        per_lap=per_lap,
        laps=laps,
    )


def _true_error(poses: tuple[Rigid, ...], true_poses: tuple[Rigid, ...], index: int) -> float:
    """The actual pose error at ``index``, in radians - what no internal metric can see."""
    return pose_difference(true_poses[index], poses[index]).rotation_angle


def _rotation_about(angle: float) -> Rigid:
    return rotation_about_axis(_AXIS, angle)


# --- 1. the headline test --------------------------------------------------------


def test_optimise_poses_recovers_true_pose_from_biased_chain_via_closure():
    """With closure edges, the true pose error falls by a large factor.

    This is the whole point of the module. The fixture is built so that
    ignoring the closure edges (or getting their direction backwards) leaves
    the large chain error untouched - see the companion negative test with
    no closures at all, which pins down that the reduction seen here is
    genuinely coming from the closure edges and not from some other effect
    of relaxation.
    """
    fixture = _loop_fixture()
    last = len(fixture.chain) - 1

    naive_error = _true_error(fixture.chain, fixture.true_poses, last)
    # Sanity check on the fixture itself: the bias must actually have
    # produced a large error, or this test would prove nothing.
    assert math.degrees(naive_error) > 40.0

    result = optimise_poses(
        list(fixture.chain),
        list(fixture.sequential_edges) + list(fixture.closure_edges),
        anchor=0,
        iterations=1000,
        tolerance=1e-13,
    )
    optimised_error = _true_error(result.poses, fixture.true_poses, last)

    # Measured: naive chaining reaches 124.2 degrees of true error at the far
    # node; relaxing against the closure edges brings that down to about 11.0
    # degrees, an 11.3x reduction. The remaining error is not a bug - every
    # sequential edge is STILL biased, and this converges to the honest
    # least-squares compromise between 23 biased edges and 12 correct ones,
    # not to zero. A margin is kept below both measured figures so the test
    # is not pinned to floating-point-exact numbers.
    assert result.converged
    assert optimised_error < naive_error / 8.0, (
        f"expected at least an 8x reduction: naive {math.degrees(naive_error):.2f} deg, "
        f"optimised {math.degrees(optimised_error):.2f} deg"
    )
    assert math.degrees(optimised_error) < 15.0


def test_optimise_poses_reduces_error_at_every_node_not_only_the_seam():
    """The closure discrepancy is spread around the loop, not dumped at one node.

    A version that only patched the closing edge (rather than genuinely
    relaxing the whole ring) would fix the last node and leave the interior
    of the loop just as wrong as the naive chain.
    """
    fixture = _loop_fixture()
    result = optimise_poses(
        list(fixture.chain),
        list(fixture.sequential_edges) + list(fixture.closure_edges),
        anchor=0,
        iterations=200,
        tolerance=1e-12,
    )
    # Check a handful of interior nodes across both laps, not just the ends.
    for index in (3, 6, 9, 15, 18, 21, len(fixture.chain) - 1):
        naive = _true_error(fixture.chain, fixture.true_poses, index)
        optimised = _true_error(result.poses, fixture.true_poses, index)
        assert optimised < naive, f"node {index}: naive {naive!r}, optimised {optimised!r}"


# --- 2. held-out validation --------------------------------------------------------


def test_optimise_poses_improves_held_out_closure_edges_it_never_saw():
    """The strongest test available with no ground truth: hide some closures, check them after.

    Optimising against half the closure edges and then measuring the OTHER
    half - never passed to :func:`optimise_poses` - is what proves the
    optimiser is correcting the trajectory itself rather than merely
    satisfying whatever constraints it was given. This is exactly the check
    intended for real captures, where ground truth does not exist.
    """
    fixture = _loop_fixture()
    train = fixture.closure_edges[0::2]
    held_out = fixture.closure_edges[1::2]
    assert train and held_out  # the fixture must actually produce both halves

    baseline = edge_residuals(fixture.chain, held_out)
    baseline_worst = max(r.rotation_error_radians for r in baseline)

    result = optimise_poses(
        list(fixture.chain),
        list(fixture.sequential_edges) + list(train),
        anchor=0,
        iterations=200,
        tolerance=1e-12,
    )
    after = edge_residuals(result.poses, held_out)
    after_worst = max(r.rotation_error_radians for r in after)

    assert after_worst < baseline_worst / 5.0, (
        f"held-out worst edge error: before {math.degrees(baseline_worst):.2f} deg, "
        f"after {math.degrees(after_worst):.2f} deg"
    )


# --- 3. the honest negative case ----------------------------------------------------


def test_optimise_poses_leaves_an_unclosed_chain_essentially_unchanged():
    """No closure edges means nothing pins the drift, so there is nothing to fix.

    A bare chain was built to satisfy its own sequential edges exactly -
    every node's neighbours already predict its current pose - so relaxation
    starting from it is already at a fixed point. This is not a limitation of
    this test; it is the documented, honest behaviour of the module (see its
    ``limitations``), and it is what makes the headline test's reduction
    attributable to the closure edges specifically.
    """
    fixture = _loop_fixture()
    result = optimise_poses(
        list(fixture.chain),
        list(fixture.sequential_edges),
        anchor=0,
        iterations=50,
        tolerance=1e-12,
    )
    for original, optimised in zip(fixture.chain, result.poses, strict=True):
        residual = pose_difference(original, optimised)
        assert residual.rotation_angle < 1e-8
        assert residual.translation_magnitude < 1e-8

    assert any("tree" in note for note in result.limitations)


# --- 4. the anchor never moves -----------------------------------------------------


def test_anchor_pose_is_exactly_unchanged():
    """The anchor is never written to, at any point - not corrected back, simply untouched."""
    fixture = _loop_fixture()
    for anchor in (0, 5, len(fixture.chain) - 1):
        result = optimise_poses(
            list(fixture.chain),
            list(fixture.sequential_edges) + list(fixture.closure_edges),
            anchor=anchor,
            iterations=30,
        )
        assert result.poses[anchor] == fixture.chain[anchor]


def test_optimise_poses_honours_edge_weights_not_just_average_rotations():
    """Weight propagation through the full sweep, not only through :func:`average_rotations`.

    Two edges from the anchor to node 1 disagree: one says 0 degrees with
    heavy weight, the other says 90 degrees with light weight. If
    ``optimise_poses`` silently dropped the weights on their way into
    :func:`average_rotations` - passing every edge as equally trusted - node
    1 would settle near the 45-degree halfway point instead of close to the
    heavily-weighted measurement.
    """
    nodes = [Rigid.identity(), _rotation_about(math.radians(45.0))]
    edges = [
        PoseEdge(0, 1, Rigid.identity(), 9.0),
        PoseEdge(0, 1, _rotation_about(math.radians(-90.0)), 1.0),
    ]
    result = optimise_poses(nodes, edges, anchor=0, iterations=50, tolerance=1e-14)
    angle = pose_difference(Rigid.identity(), result.poses[1]).rotation_angle
    assert math.degrees(angle) < 20.0


# --- 5. contradictory edges converge to a reported compromise ---------------------


def test_contradictory_closure_edges_converge_to_a_compromise_not_a_crash():
    """Two closures that cannot both hold must not diverge or produce a non-rotation.

    Node 0 is anchored at the identity. Node 1 is one small step away.
    Closure A says node 2 coincides with node 0 (a tiny rotation).
    Closure B says node 2 is 90 degrees from node 0. Both cannot be right.
    """
    small_step = _rotation_about(math.radians(5.0))
    nodes = [Rigid.identity(), small_step, small_step.compose(small_step)]
    edges = [
        PoseEdge(0, 1, small_step, 1.0),
        PoseEdge(1, 2, small_step, 1.0),
        PoseEdge(0, 2, Rigid.identity(), 1.0),  # "they coincide"
        PoseEdge(0, 2, _rotation_about(math.radians(-90.0)), 1.0),  # "90 degrees apart"
    ]
    result = optimise_poses(nodes, edges, anchor=0, iterations=100, tolerance=1e-14)

    # Every pose that came out is a genuine rotation: re-validating with
    # ensure_rotation raises NotARotation if relaxation ever produced
    # something that merely looks like one (a scaled or skewed matrix).
    for pose in result.poses:
        ensure_rotation(pose.rotation)

    assert result.converged
    worst = result.worst_edge
    assert worst is not None
    assert math.degrees(worst.rotation_error_radians) > 20.0


# --- 6. rotation averaging itself --------------------------------------------------


def test_average_rotations_of_a_rotation_with_itself_returns_it():
    r = _rotation_about(math.radians(37.0)).rotation
    averaged = average_rotations([r, r], [1.0, 1.0])
    for row_a, row_b in zip(averaged, r, strict=True):
        for a, b in zip(row_a, row_b, strict=True):
            assert a == pytest.approx(b, abs=1e-9)


def test_average_rotations_about_a_shared_axis_is_the_halfway_rotation():
    theta1, theta2 = math.radians(10.0), math.radians(50.0)
    r1 = _rotation_about(theta1).rotation
    r2 = _rotation_about(theta2).rotation
    averaged = average_rotations([r1, r2], [1.0, 1.0])
    expected = _rotation_about((theta1 + theta2) / 2.0).rotation
    for row_a, row_b in zip(averaged, expected, strict=True):
        for a, b in zip(row_a, row_b, strict=True):
            assert a == pytest.approx(b, abs=1e-9)


def test_average_rotations_weights_toward_the_more_trusted_measurement():
    theta1, theta2 = 0.0, math.radians(90.0)
    r1 = _rotation_about(theta1).rotation
    r2 = _rotation_about(theta2).rotation
    averaged = average_rotations([r1, r2], [9.0, 1.0])
    averaged_angle = Rigid.of(averaged, (0.0, 0.0, 0.0)).rotation_angle
    r1_angle = Rigid.of(r1, (0.0, 0.0, 0.0)).rotation_angle
    # Heavily weighted toward r1 (identity, angle 0): the result should sit
    # much closer to r1 than to r2's 90 degrees.
    assert math.degrees(averaged_angle) < 20.0
    assert averaged_angle >= r1_angle


def test_average_rotations_handles_the_near_cancelling_degenerate_case():
    """A 0-degree and a 180-degree rotation about the same axis have no single average.

    The weighted sum is singular there (see the function's docstring): this
    must return a genuine rotation - never NaN, never a non-rotation - rather
    than crash or silently misbehave.
    """
    identity = _rotation_about(0.0).rotation
    half_turn = _rotation_about(math.pi).rotation
    averaged = average_rotations([identity, half_turn], [1.0, 1.0])
    # A valid rotation: orthonormal with determinant +1. Constructing a Rigid
    # from it re-runs exactly that check and raises if it fails.
    Rigid.of(averaged, (0.0, 0.0, 0.0))
    # The documented fallback is the higher-weighted input; with equal
    # weights that is whichever candidate is found first, so accept either -
    # what matters is that it is one of the two genuine inputs, not a
    # manufactured non-rotation in between.
    assert averaged in (identity, half_turn)


# --- 7. guards ----------------------------------------------------------------------


def test_optimise_poses_refuses_an_edge_index_out_of_range():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(nodes, [PoseEdge(0, 5, Rigid.identity(), 1.0)])


def test_optimise_poses_refuses_fewer_than_two_nodes():
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses([Rigid.identity()], [])


def test_optimise_poses_refuses_a_non_positive_edge_weight():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(nodes, [PoseEdge(0, 1, Rigid.identity(), 0.0)])
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(nodes, [PoseEdge(0, 1, Rigid.identity(), -1.0)])


def test_optimise_poses_refuses_an_anchor_out_of_range():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(nodes, [], anchor=7)


def test_optimise_poses_refuses_zero_iterations():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(nodes, [], anchor=0, iterations=0)


def test_average_rotations_refuses_mismatched_lengths_and_empty_input():
    r = _rotation_about(0.1).rotation
    with pytest.raises(PoseGraphNotAllowed):
        average_rotations([r, r], [1.0])
    with pytest.raises(PoseGraphNotAllowed):
        average_rotations([], [])
    with pytest.raises(PoseGraphNotAllowed):
        average_rotations([r], [0.0])


def test_every_posegraph_refusal_is_a_scanner_error_the_doctor_can_classify():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphNotAllowed) as excinfo:
        optimise_poses(nodes, [], anchor=99)
    assert excinfo.value.status == DoctorStatus.UNSUPPORTED_MODE


# --- structure and serialisation -----------------------------------------------------


def test_a_node_with_no_incident_edge_is_left_exactly_where_it_started():
    """Nothing predicts its pose, so relaxation does not touch it - the honest behaviour."""
    nodes = [
        Rigid.identity(),
        _rotation_about(math.radians(20.0)),
        _rotation_about(math.radians(99.0)),
    ]
    edges = [PoseEdge(0, 1, _rotation_about(-math.radians(20.0)), 1.0)]  # node 2 is isolated
    result = optimise_poses(nodes, edges, anchor=0, iterations=10)
    assert result.poses[2] == nodes[2]


def test_edge_residual_is_near_zero_for_edges_the_poses_already_satisfy():
    step = _rotation_about(math.radians(15.0))
    # Built the same way _chain_poses in scanner3d.registration would: pose[1]
    # is pose[0] composed with the inverse of the edge that relates them.
    poses = [Rigid.identity(), Rigid.identity().compose(step.inverse())]
    residual = edge_residual(poses, PoseEdge(0, 1, step, 1.0))
    assert residual.rotation_error_radians < 1e-9
    assert residual.translation_error < 1e-9


def test_pose_graph_result_as_dict_carries_the_schema_version_and_is_json_safe():
    fixture = _loop_fixture()
    result = optimise_poses(
        list(fixture.chain),
        list(fixture.sequential_edges) + list(fixture.closure_edges),
        anchor=0,
        iterations=20,
    )
    record = json.loads(json.dumps(result.as_dict()))
    assert record["schema_version"] == 1
    assert record["anchor"] == 0
    assert isinstance(record["worst_edge"], dict)
    assert isinstance(record["limitations"], list) and record["limitations"]


def test_edge_residual_combined_matches_hypot_of_its_two_halves():
    residual = EdgeResidual(
        edge_index=0, i=0, j=1, rotation_error_radians=0.3, translation_error=0.4
    )
    assert residual.combined == pytest.approx(0.5)


# --- 8. robust estimation -----------------------------------------------------------
#
# WHY THIS EXISTS: on a real 70-frame sweep, 5 of 34 detected "revisits"
# were false - ICP seeded from the identity, given two clouds a HALF
# revolution apart, failed to escape the local minimum and reported a
# near-identity transform between frames that were actually 176-181 degrees
# apart. Every LOCAL statistic (rmse, overlap, correspondence count) on
# those five sat inside the range the true revisits also occupied - there
# is no per-edge threshold that separates them. Held-out closure error went
# from 2.735 degrees (chained, unoptimised) to 30.486 degrees (optimised) -
# 11x WORSE - because the optimiser trusted every edge equally and a
# minority of wrong ones captured the fit. That is the exact failure this
# section exists to prevent.

#: Iteration/tolerance budget for robust tests: loose enough to keep this
#: section's own runtime reasonable on the 120-node headline fixture below,
#: tight enough that the recovery each test asserts is a real property of
#: the fit rather than an artefact of stopping early - see the measured
#: numbers each assertion checks against.
_ROBUST_ITERATIONS = 150
_ROBUST_TOLERANCE = 1e-8


@functools.cache
def _robust_headline_fixture():
    """REAL usage, not the module's own biased-chain headline fixture: poses
    are the EXACT composition of the sequential edges (recovery_fraction=1.0
    - no bias at all), matching how a real caller actually builds the poses
    it hands to this module (see :func:`scanner3d.registration.register_turntable`'s
    ``_chain_poses``) before this module ever runs. That distinction is not
    cosmetic - see ``posegraph._robust_weights``'s own docstring: a first
    version of this fixture reused the module's OTHER headline fixture,
    which protects sequential edges that still carry a real, nonzero
    systematic-bias residual, and it could not have caught the real bug
    this one exists for, because that bug is specifically about a
    protected population satisfied to float64 precision by construction.

    Ten laps (120 nodes, 108 dense closures - one nearly every node, not a
    sparse handful) is kept from the original fixture: measured directly, a
    SPARSER graph (tried at 20 and 80 nodes with a handful of closures
    spanning wide gaps) lets a single 179-degree edge distort a whole
    unconstrained neighbourhood during the very first, still-unweighted
    relaxation pass, before robust reweighting gets a say - collateral
    damage that inflates nearby HONEST closures' residuals enough to blur
    the separation robust fitting needs. Dense coverage keeps a wrong
    edge's distortion local, exactly as the ten-lap count did for the
    original fixture's different reason.

    108 closures get a small, honest amount of independent noise (0.2 to 3
    degrees, matching the scale of real ICP noise on a genuine revisit); 5
    are corrupted to ~179 degrees - two frames a half revolution apart
    mistaken for the same viewpoint, the exact real failure.
    """
    fixture = _loop_fixture(laps=10, per_lap=12, recovery_fraction=1.0)
    # Sequential edges are PROTECTED and, because the poses above are the
    # exact composition of these same edges, satisfied to float64 precision
    # - the real-usage property this fixture exists to exercise.
    protected_sequential = tuple(
        PoseEdge(e.i, e.j, e.transform, e.weight, True) for e in fixture.sequential_edges
    )
    axis: Point = (1.0, 2.0, 2.0)
    rng = random.Random(5)
    wrong_indices = (3, 20, 45, 70, 95)
    closures = []
    wrong_pairs = []
    for idx, edge in enumerate(fixture.closure_edges):
        if idx in wrong_indices:
            wrong = rotation_about_axis(axis, math.radians(179.0))
            closures.append(PoseEdge(edge.i, edge.j, wrong, 1.0))
            wrong_pairs.append((edge.i, edge.j))
        else:
            noise = rotation_about_axis(axis, rng.uniform(math.radians(0.2), math.radians(3.0)))
            closures.append(PoseEdge(edge.i, edge.j, noise.compose(edge.transform), 1.0))
    return fixture, protected_sequential, tuple(closures), tuple(wrong_pairs)


@functools.cache
def _robust_headline_results():
    fixture, protected_sequential, closures, _wrong_pairs = _robust_headline_fixture()
    all_edges = protected_sequential + closures
    nonrobust = optimise_poses(
        list(fixture.chain),
        list(all_edges),
        anchor=0,
        iterations=_ROBUST_ITERATIONS,
        tolerance=_ROBUST_TOLERANCE,
    )
    robust = optimise_poses(
        list(fixture.chain),
        list(all_edges),
        anchor=0,
        iterations=_ROBUST_ITERATIONS,
        tolerance=_ROBUST_TOLERANCE,
        robust=True,
        robust_passes=8,
    )
    return fixture, all_edges, nonrobust, robust


def test_robust_recovers_close_to_clean_when_wrong_edges_are_added():
    """THE HEADLINE robust test, on a REAL-usage-shaped fixture (see
    ``_robust_headline_fixture``): sequential edges satisfied to float64
    precision, exactly like a real caller's composed chain, not merely
    protected while still carrying a residual of their own.

    Non-robust fitting on this fixture must be measurably WRECKED by the 5
    wrong edges; robust fitting must recover most of the way back. Both
    are asserted explicitly - a test that checked only one half would not
    prove the feature does anything.
    """
    fixture, _all_edges, nonrobust, robust = _robust_headline_results()
    last = len(fixture.chain) - 1
    nonrobust_error = _true_error(nonrobust.poses, fixture.true_poses, last)
    robust_error = _true_error(robust.poses, fixture.true_poses, last)

    message = (
        f"non-robust {math.degrees(nonrobust_error):.2f} deg, "
        f"robust {math.degrees(robust_error):.2f} deg"
    )
    # Sanity check on the fixture itself: the wrong edges must actually
    # have hurt, or this proves nothing about recovering from them.
    assert math.degrees(nonrobust_error) > 25.0, message
    assert robust_error < nonrobust_error / 2.0, message
    assert math.degrees(robust_error) < 20.0, message


def test_robust_reports_exactly_the_wrong_edges_as_rejected():
    _fixture, _protected_sequential, _closures, wrong_pairs = _robust_headline_fixture()
    _fixture2, _all_edges, _nonrobust, robust = _robust_headline_results()
    rejected_pairs = {(residual.i, residual.j) for residual in robust.rejected_edges}
    assert rejected_pairs == set(wrong_pairs), (rejected_pairs, wrong_pairs)
    assert robust.robust is True
    assert robust.robust_passes_run >= 1


def test_robust_headline_fixtures_sequential_edges_are_satisfied_by_construction():
    """The property that makes this fixture "real usage" rather than a
    restatement of the module's other headline test: every sequential
    edge's residual, BEFORE any fitting at all, is at the float64 floor,
    because the poses were built by composing exactly these edges - see
    :func:`scanner3d.registration.register_turntable`'s ``_chain_poses``.
    """
    fixture, protected_sequential, _closures, _wrong_pairs = _robust_headline_fixture()
    residuals = edge_residuals(fixture.chain, list(protected_sequential))
    worst = max(r.rotation_error_radians for r in residuals)
    assert worst < 1e-9, worst


def test_robust_weights_scale_excludes_protected_edges_at_the_real_data_scale():
    """THE test that matters, at the exact level the bug lives at: a direct
    reproduction of ``_robust_weights`` on the user's real proportions, not
    routed through a full multi-pass relaxation where residuals shift
    between passes and can obscure the mechanism - see the fixture above's
    own docstring on why even a carefully-built full fixture did not
    reliably reproduce this on its own once relaxation had a few passes to
    smooth things over.

    69 protected edges at EXACTLY zero residual (the real sequential-edge
    count, satisfied to float64 precision by construction) mixed with 17
    closures: 14 honest ones in the real 0.53-8.15 degree range, and 3 at
    the real measured 179.69, 179.63 and 179.42 degrees. The old formula -
    scale from every edge, protected included - crushes the scale to the
    1e-9 floor and rejects NOTHING, reproducing the real failure exactly.
    The fixed formula must reject exactly the 3 wrong closures.
    """
    protected = [PoseEdge(0, 1, Rigid.identity(), 1.0, True) for _ in range(69)]
    protected_residuals = [EdgeResidual(i, 0, 1, 0.0, 0.0) for i in range(69)]

    honest_degrees = [0.53, 0.71, 0.88, 1.08, 1.42, 1.9, 2.3, 2.85, 3.25, 4.4, 5.1, 6.0, 7.2, 8.15]
    wrong_degrees = [179.69, 179.63, 179.42]
    closures = [
        PoseEdge(2, 3, Rigid.identity(), 1.0, False) for _ in honest_degrees + wrong_degrees
    ]
    closure_residuals = [
        EdgeResidual(69 + i, 2, 3, math.radians(deg), 0.0)
        for i, deg in enumerate(honest_degrees + wrong_degrees)
    ]

    edges = protected + closures
    residuals = protected_residuals + closure_residuals

    # The bug, reproduced directly: computing the scale from every edge's
    # residual (protected included) crushes it toward the 1e-9 floor and
    # rejects nothing, on data shaped exactly like the real graph.
    buggy_values = [r.combined for r in residuals]
    buggy_median = statistics.median(buggy_values)
    assert buggy_median < 1e-6, (
        f"fixture sanity check failed: expected the all-edges median to be crushed near "
        f"zero by the 69 protected zeros, got {buggy_median!r}"
    )

    _weights, rejected = _robust_weights(edges, residuals, threshold=3.0)
    rejected_pairs = {(edges[i].i, edges[i].j) for i in rejected}
    assert len(rejected) == len(wrong_degrees), (
        f"expected exactly {len(wrong_degrees)} rejected, got {len(rejected)}: {rejected_pairs}"
    )
    # Every rejected edge must be one of the wrong closures (index 69+14 and
    # up), never a protected edge (never eligible) or an honest closure.
    assert all(index >= 69 + len(honest_degrees) for index in rejected), sorted(rejected)


def test_optimise_poses_does_not_reject_anything_when_every_edge_agrees():
    """A robust estimator that quietly discards good data is worse than none.

    24 independent, individually noisy measurements of the SAME true
    relative pose - exactly what several agreeing closures onto one
    genuine revisit would look like. None of them is wrong; robust fitting
    must keep every one and land on (very nearly) the same answer plain
    relaxation does.
    """
    axis: Point = (0.3, 0.6, 0.58)
    true_angle = math.radians(20.0)
    rng = random.Random(0)
    edges = [
        PoseEdge(
            0, 1, rotation_about_axis(axis, -true_angle + rng.gauss(0.0, math.radians(0.3))), 1.0
        )
        for _ in range(24)
    ]
    nodes = [Rigid.identity(), rotation_about_axis(axis, true_angle)]

    nonrobust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12)
    robust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12, robust=True)

    assert robust.rejected_edges == ()
    # "Very nearly" the same answer, not bit-identical: the Cauchy kernel
    # is a smooth curve, not a hard pass/fail gate, so even a KEPT edge with
    # nonzero residual is weighted fractionally differently than plain
    # relaxation would weight it. What matters is that nothing was rejected
    # and the two answers are close - degrees, not tens of degrees, apart.
    difference = pose_difference(nonrobust.poses[1], robust.poses[1])
    assert math.degrees(difference.rotation_angle) < 1.0
    assert difference.translation_magnitude < 1e-6


def test_robust_refuses_when_rejection_would_disconnect_a_node():
    """A node reachable only through edges that all get rejected must not get a pose.

    Six edges agree tightly on node 1's pose; node 2 is reached only by two
    edges that flatly contradict each other (identity vs. 90 degrees) and
    nothing else. Once both lose the argument with the consensus the first
    six edges establish, node 2 has no surviving path to the anchor at all,
    and this must refuse rather than return a pose for it anyway.
    """
    axis: Point = (0.3, 0.6, 0.58)
    agreed = rotation_about_axis(axis, -math.radians(20.0))
    edges = [PoseEdge(0, 1, agreed, 1.0) for _ in range(6)]
    edges.append(PoseEdge(1, 2, Rigid.identity(), 1.0))
    edges.append(PoseEdge(0, 2, rotation_about_axis(axis, -math.radians(90.0)), 1.0))
    nodes = [Rigid.identity(), rotation_about_axis(axis, math.radians(20.0)), Rigid.identity()]

    with pytest.raises(PoseGraphRobustDisconnected):
        optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12, robust=True)


def test_robust_rejection_is_deterministic():
    """Same input, same rejections - no randomness anywhere in the robust pass."""
    fixture = _loop_fixture(laps=4, per_lap=12, recovery_fraction=0.82)
    per_lap = fixture.per_lap
    protected_sequential = [
        PoseEdge(e.i, e.j, e.transform, e.weight, True) for e in fixture.sequential_edges
    ]
    wrong_edges = [PoseEdge(k, k + per_lap // 2, Rigid.identity(), 1.0) for k in (0, 4, 8, 12, 16)]
    edges = protected_sequential + list(fixture.closure_edges) + wrong_edges

    first = optimise_poses(
        list(fixture.chain), edges, anchor=0, iterations=150, tolerance=1e-8, robust=True
    )
    second = optimise_poses(
        list(fixture.chain), edges, anchor=0, iterations=150, tolerance=1e-8, robust=True
    )

    assert [(r.i, r.j) for r in first.rejected_edges] == [(r.i, r.j) for r in second.rejected_edges]
    for pose_a, pose_b in zip(first.poses, second.poses, strict=True):
        assert pose_a == pose_b


def test_robust_edges_below_the_minimum_are_never_rejected():
    """Rejection must be earned by disagreement, never manufactured by having few edges.

    Three wildly contradictory edges - fewer than MIN_ROBUST_EDGES - cannot
    define a "majority" for anything to disagree with, so robust=True must
    behave exactly like robust=False here.
    """
    axis: Point = (0.3, 0.6, 0.58)
    nodes = [Rigid.identity(), Rigid.identity()]
    edges = [
        PoseEdge(0, 1, Rigid.identity(), 1.0),
        PoseEdge(0, 1, rotation_about_axis(axis, math.radians(90.0)), 1.0),
        PoseEdge(0, 1, rotation_about_axis(axis, math.radians(-90.0)), 1.0),
    ]
    assert len(edges) < MIN_ROBUST_EDGES

    nonrobust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12)
    robust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12, robust=True)
    assert robust.rejected_edges == ()
    assert robust.poses[1] == nonrobust.poses[1]
    assert any("below" in note for note in robust.limitations)


def test_robust_false_is_bit_identical_to_before_the_feature_existed():
    """The default is robust=False, and it must change nothing about existing behaviour."""
    fixture = _loop_fixture()
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)
    explicit_false = optimise_poses(
        list(fixture.chain), edges, anchor=0, iterations=200, tolerance=1e-9, robust=False
    )
    default = optimise_poses(list(fixture.chain), edges, anchor=0, iterations=200, tolerance=1e-9)
    assert explicit_false.poses == default.poses
    assert explicit_false.error_history == default.error_history
    assert default.robust is False
    assert default.robust_passes_run == 0
    assert default.rejected_edges == ()


def test_the_honest_noise_scenario_still_rejects_nothing_after_the_scale_fix():
    """Re-run of the many-honest-measurements-zero-outliers claim, because
    changing which population the scale comes from could plausibly have
    reintroduced the false-positive problem the folded-normal calibration
    fixed. It has not: this is the identical scenario and assertion as
    test_optimise_poses_does_not_reject_anything_when_every_edge_agrees,
    kept here as its own named check specifically for the scale-population
    change.
    """
    axis: Point = (0.3, 0.6, 0.58)
    true_angle = math.radians(20.0)
    rng = random.Random(0)
    edges = [
        PoseEdge(
            0, 1, rotation_about_axis(axis, -true_angle + rng.gauss(0.0, math.radians(0.3))), 1.0
        )
        for _ in range(24)
    ]
    nodes = [Rigid.identity(), rotation_about_axis(axis, true_angle)]
    robust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12, robust=True)
    assert robust.rejected_edges == ()


def test_robust_handles_every_edge_protected_without_dividing_by_an_empty_population():
    """No edge is eligible for rejection at all - not merely few - and this
    must be handled explicitly rather than computing a median of nothing.
    """
    axis: Point = (0.3, 0.6, 0.58)
    nodes = [Rigid.identity(), Rigid.identity(), Rigid.identity()]
    edges = [
        PoseEdge(0, 1, rotation_about_axis(axis, math.radians(20.0)), 1.0, True),
        PoseEdge(1, 2, rotation_about_axis(axis, math.radians(-20.0)), 1.0, True),
        PoseEdge(0, 2, Rigid.identity(), 1.0, True),
        PoseEdge(0, 2, rotation_about_axis(axis, math.radians(90.0)), 1.0, True),
    ]
    nonrobust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12)
    robust = optimise_poses(nodes, edges, anchor=0, iterations=200, tolerance=1e-12, robust=True)
    assert robust.rejected_edges == ()
    assert robust.poses == nonrobust.poses
    assert any("eligible" in note for note in robust.limitations)


def test_robust_weights_itself_handles_zero_eligible_edges_without_crashing():
    """The same claim, aimed directly at ``_robust_weights``: with
    ``optimise_poses``, the OUTER ``attempt_robust`` guard already stops an
    all-protected graph from ever reaching this function at all, so that
    path alone would not prove ``_robust_weights`` handles an empty eligible
    population itself - only that its caller happens not to trigger it. This
    calls it directly with every edge protected, so ``eligible`` is
    genuinely empty (not merely small), and confirms it returns the
    unchanged weights and no rejections rather than raising on
    ``statistics.median([])``.
    """
    edges = [PoseEdge(0, 1, Rigid.identity(), 1.0, True) for _ in range(5)]
    residuals = [EdgeResidual(i, 0, 1, math.radians(50.0), 0.0) for i in range(5)]
    weights, rejected = _robust_weights(edges, residuals, threshold=3.0)
    assert weights == [edge.weight for edge in edges]
    assert rejected == frozenset()


# --- 9. robust guards -----------------------------------------------------------------


def test_optimise_poses_refuses_bad_robust_parameters():
    fixture = _loop_fixture()
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(list(fixture.chain), edges, robust=True, robust_passes=0)
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(list(fixture.chain), edges, robust=True, outlier_threshold=0.0)
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(list(fixture.chain), edges, robust=True, outlier_threshold=-1.0)
    with pytest.raises(PoseGraphNotAllowed):
        optimise_poses(list(fixture.chain), edges, robust=True, outlier_threshold=math.inf)


def test_pose_edge_protected_defaults_false_and_survives_as_dict():
    plain = PoseEdge(0, 1, Rigid.identity(), 1.0)
    assert plain.protected is False
    protected = PoseEdge(0, 1, Rigid.identity(), 1.0, True)
    assert protected.as_dict()["protected"] is True
    assert plain.as_dict()["protected"] is False
