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
from dataclasses import dataclass

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import Point, Rigid, ensure_rotation, pose_difference, rotation_about_axis
from scanner3d.posegraph import (
    EdgeResidual,
    PoseEdge,
    PoseGraphNotAllowed,
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
