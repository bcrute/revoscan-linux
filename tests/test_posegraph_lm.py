"""Tests for :mod:`scanner3d.posegraph_lm`.

Priority order, matching the assignment this module was built against:

1. The headline test: on the SAME fixture ``tests/test_posegraph.py`` uses
   for its own headline test, Levenberg-Marquardt must reach a solution at
   least as good as Gauss-Seidel relaxation, measured against known ground
   truth. Both are run and both numbers are asserted and printed via the
   assertion message, so a failure names the actual figures.
2. SE(3) log/exp round-trip, at tight tolerance, INCLUDING near-identity and
   near-pi rotations - the two ends of the range
   :mod:`scanner3d.posegraph_lm`'s own module docstring says its conditioning
   care is spent on.
3. Analytic Jacobians agree with central finite differences on random poses.
4. A perfect graph (every pose already satisfies every edge) converges
   immediately and moves nothing.
5. Contradictory edges settle to a reported compromise, not a crash or a
   non-rotation.
6. Guards: bad indices, fewer than two nodes, anchor out of range,
   non-positive weights.
"""

from __future__ import annotations

import math
import random

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import Point, Rigid, ensure_rotation, rotation_about_axis
from scanner3d.posegraph import PoseEdge, optimise_poses
from scanner3d.posegraph_lm import (
    PoseGraphLMNotAllowed,
    _apply_update,
    _edge_residual_and_jacobians,
    _so3_exp,
    _so3_log,
    optimise_poses_lm,
    se3_exp,
    se3_log,
)
from tests.test_posegraph import _loop_fixture, _true_error

# The same tilted axis test_posegraph.py uses, so a bug that only shows up
# off-axis cannot hide behind an axis-aligned special case here either.
_AXIS: Point = (1.0, 2.0, 2.0)


def _rotation_about(angle: float) -> Rigid:
    return rotation_about_axis(_AXIS, angle)


# --- 1. the headline test: LM against Gauss-Seidel, same fixture, same ground truth ------


def test_lm_matches_or_beats_gauss_seidel_on_the_shared_headline_fixture():
    """The point of this module: at least as good as relaxation, on relaxation's own fixture.

    ``_loop_fixture`` is ``functools.cache``d in ``tests/test_posegraph.py``,
    so importing and calling it here returns the IDENTICAL fixture object
    ``test_optimise_poses_recovers_true_pose_from_biased_chain_via_closure``
    exercises - not a separately-constructed lookalike. Gauss-Seidel is
    measured to reach about 11.0 degrees of true error from a 124.2-degree
    start (see that test's own comment); LM is required to do at least as
    well, and both figures are reported in the assertion message so a
    regression names actual numbers, not just "worse".
    """
    fixture = _loop_fixture()
    last = len(fixture.chain) - 1
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)

    naive_error = _true_error(fixture.chain, fixture.true_poses, last)
    assert math.degrees(naive_error) > 40.0

    gs_result = optimise_poses(
        list(fixture.chain), edges, anchor=0, iterations=1000, tolerance=1e-13
    )
    gs_error = _true_error(gs_result.poses, fixture.true_poses, last)

    lm_result = optimise_poses_lm(
        list(fixture.chain), edges, anchor=0, iterations=100, tolerance=1e-14
    )
    lm_error = _true_error(lm_result.poses, fixture.true_poses, last)

    assert lm_result.converged
    assert lm_error <= gs_error * 1.05, (
        f"naive {math.degrees(naive_error):.3f} deg, "
        f"Gauss-Seidel {math.degrees(gs_error):.3f} deg, "
        f"LM {math.degrees(lm_error):.3f} deg: LM should match or beat relaxation here"
    )
    assert math.degrees(lm_error) < 15.0
    # LM's whole point is doing this with second-order information instead of
    # hundreds of relaxation sweeps - confirm it actually did.
    assert lm_result.iterations_run < 50


def test_lm_reduces_error_at_every_node_not_only_the_seam():
    fixture = _loop_fixture()
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)
    result = optimise_poses_lm(list(fixture.chain), edges, anchor=0, iterations=100)
    for index in (3, 6, 9, 15, 18, 21, len(fixture.chain) - 1):
        naive = _true_error(fixture.chain, fixture.true_poses, index)
        optimised = _true_error(result.poses, fixture.true_poses, index)
        assert optimised < naive, f"node {index}: naive {naive!r}, optimised {optimised!r}"


def test_lm_improves_held_out_closure_edges_it_never_saw():
    fixture = _loop_fixture()
    train = fixture.closure_edges[0::2]
    held_out = fixture.closure_edges[1::2]
    assert train and held_out

    from scanner3d.posegraph import edge_residuals

    baseline = edge_residuals(fixture.chain, held_out)
    baseline_worst = max(r.rotation_error_radians for r in baseline)

    result = optimise_poses_lm(
        list(fixture.chain),
        list(fixture.sequential_edges) + list(train),
        anchor=0,
        iterations=100,
    )
    after = edge_residuals(result.poses, held_out)
    after_worst = max(r.rotation_error_radians for r in after)

    assert after_worst < baseline_worst / 5.0, (
        f"held-out worst edge error: before {math.degrees(baseline_worst):.2f} deg, "
        f"after {math.degrees(after_worst):.2f} deg"
    )


# --- 2. SE(3) log/exp round-trip, including near-identity and near-pi -----------------


def _random_tangent(rng: random.Random, theta: float) -> tuple[float, ...]:
    axis = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(a * a for a in axis))
    axis = [a / norm for a in axis]
    rho = tuple(rng.uniform(-5.0, 5.0) for _ in range(3))
    phi = tuple(a * theta for a in axis)
    return rho + phi


@pytest.mark.parametrize(
    "theta",
    [0.0, 1e-9, 1e-6, 1e-4, 1e-3, 0.1, 1.0, 2.0, math.pi - 1e-3, math.pi - 1e-6, math.pi - 1e-9],
)
def test_se3_round_trip_across_the_full_angle_range(theta: float):
    """``se3_exp(se3_log(pose))`` reproduces ``pose`` to float64 precision.

    Parametrised across the whole ``[0, pi]`` range with extra density at
    both ends - ``theta`` near 0 (the small-angle Taylor branch) and ``theta``
    near ``pi`` (where the rotation axis extraction is at its most delicate)
    - because those are exactly the two places a naive implementation loses
    precision or breaks outright; see the module docstring.

    ``abs=1e-9`` everywhere except the single closest-to-``pi`` case, which
    gets a looser ``1e-7``: AT ``theta = pi`` the axis sign genuinely has no
    unique answer (see :func:`_so3_log`'s docstring - ``axis`` and ``-axis``
    represent the identical rotation there), so a float64-noise-sized wobble
    in which side of that ambiguity gets picked is expected, not a
    regression, and it propagates once through ``_so3_v``/``_so3_v_inv``
    into the translation. Every other sampled angle, including
    ``pi - 1e-6``, still round-trips at the tight tolerance.
    """
    tolerance = 1e-7 if theta > math.pi - 1e-8 else 1e-9
    rng = random.Random(hash(("round_trip", theta)))
    for _ in range(20):
        tangent = _random_tangent(rng, theta)
        pose = se3_exp(tangent)
        ensure_rotation(pose.rotation)  # a genuine rotation came out, not dust
        round_tripped = se3_exp(se3_log(pose))
        for a, b in zip(pose.rotation, round_tripped.rotation, strict=True):
            for x, y in zip(a, b, strict=True):
                assert x == pytest.approx(y, abs=tolerance)
        for x, y in zip(pose.translation, round_tripped.translation, strict=True):
            assert x == pytest.approx(y, abs=tolerance)


def test_se3_exp_at_zero_tangent_is_the_identity():
    pose = se3_exp((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert pose == Rigid.identity()


def test_se3_log_rotation_angle_matches_rigid_rotation_angle():
    """The rotation half of ``se3_log`` and :attr:`Rigid.rotation_angle` are the
    same geodesic angle, computed the same conditioning-aware way - they must
    agree, including near ``pi``.
    """
    rng = random.Random(11)
    for theta in (1e-8, 1e-3, 1.0, math.pi - 1e-3, math.pi - 1e-8):
        for _ in range(10):
            tangent = _random_tangent(rng, theta)
            pose = se3_exp(tangent)
            _, _, _, phi_x, phi_y, phi_z = se3_log(pose)
            phi_norm = math.sqrt(phi_x**2 + phi_y**2 + phi_z**2)
            assert phi_norm == pytest.approx(pose.rotation_angle, abs=1e-6)


def test_se3_exp_refuses_a_tangent_of_the_wrong_shape():
    with pytest.raises(PoseGraphLMNotAllowed):
        se3_exp((1.0, 2.0, 3.0))


def test_so3_round_trip_is_a_genuine_rotation_at_every_angle():
    """``_so3_exp``/``_so3_log`` round-trip, and every output is a real rotation."""
    rng = random.Random(3)
    for theta in (0.0, 1e-8, 0.5, math.pi - 1e-8, math.pi):
        for _ in range(10):
            tangent = _random_tangent(rng, theta)
            phi = tangent[3:]
            rotation = _so3_exp(_to_array(phi))
            ensure_rotation(_to_matrix(rotation))
            phi2 = _so3_log(rotation)
            rotation2 = _so3_exp(phi2)
            for a, b in zip(rotation.tolist(), rotation2.tolist(), strict=True):
                for x, y in zip(a, b, strict=True):
                    assert x == pytest.approx(y, abs=1e-9)


def _to_array(values: tuple[float, ...]):
    import numpy as np

    return np.asarray(values, dtype=np.float64)


def _to_matrix(array):
    return tuple(tuple(float(v) for v in row) for row in array.tolist())


# --- 3. analytic Jacobians agree with central finite differences ----------------------


def _random_rigid(rng: random.Random, max_theta: float = math.pi - 1e-4) -> Rigid:
    theta = rng.uniform(0.0001, max_theta)
    axis = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(a * a for a in axis))
    axis = [a / norm * theta for a in axis]
    translation = tuple(rng.uniform(-5.0, 5.0) for _ in range(3))
    return se3_exp(tuple(translation) + tuple(axis))


def _numeric_edge_jacobians(pose_i: Rigid, pose_j: Rigid, edge: PoseEdge, h: float = 1e-6):
    """Central finite differences of the SAME residual convention
    :func:`scanner3d.posegraph_lm._edge_residual_and_jacobians` computes,
    built entirely from :func:`se3_log`, :func:`se3_exp` and
    :meth:`Rigid.compose` - independent code paths from the analytic
    Jacobian under test.
    """
    import numpy as np

    def residual(pi: Rigid, pj: Rigid) -> np.ndarray:
        relative = pj.inverse().compose(pi)
        full = edge.transform.inverse().compose(relative)
        return np.asarray(se3_log(full), dtype=np.float64)

    jacobian_i = np.zeros((6, 6))
    jacobian_j = np.zeros((6, 6))
    for k in range(6):
        bump = [0.0] * 6
        bump[k] = h
        plus = residual(pose_i.compose(se3_exp(tuple(bump))), pose_j)
        bump[k] = -h
        minus = residual(pose_i.compose(se3_exp(tuple(bump))), pose_j)
        jacobian_i[:, k] = (plus - minus) / (2.0 * h)
    for k in range(6):
        bump = [0.0] * 6
        bump[k] = h
        plus = residual(pose_i, pose_j.compose(se3_exp(tuple(bump))))
        bump[k] = -h
        minus = residual(pose_i, pose_j.compose(se3_exp(tuple(bump))))
        jacobian_j[:, k] = (plus - minus) / (2.0 * h)
    return jacobian_i, jacobian_j


def test_edge_jacobian_matches_numeric_differences_on_random_poses():
    """The load-bearing correctness check for this module's whole reason to exist:
    the closed-form Jacobians :func:`_edge_residual_and_jacobians` computes must
    agree with central finite differences of the same residual, built via an
    entirely independent path (:func:`se3_log`, :func:`se3_exp`,
    :meth:`Rigid.compose`).

    ``1e-4`` absolute tolerance is set from measurement, not guessed: central
    differences at ``h = 1e-6`` carry their own O(h**2) truncation error,
    largest exactly where this module's own conditioning care concentrates -
    close to theta = pi, where the residual's rotation is itself near pi. On
    500 random trials spanning the full angle range the measured worst-case
    disagreement was about 2.6e-7, comfortably inside this margin; see the
    module's own development validation for that figure.
    """
    import numpy as np

    rng = random.Random(7)
    worst = 0.0
    for _ in range(60):
        pose_i = _random_rigid(rng)
        pose_j = _random_rigid(rng)
        edge_transform = _random_rigid(rng)
        edge = PoseEdge(0, 1, edge_transform, 1.0)

        residual, analytic_i, analytic_j = _edge_residual_and_jacobians(
            np.asarray(pose_i.rotation, dtype=np.float64),
            np.asarray(pose_i.translation, dtype=np.float64),
            np.asarray(pose_j.rotation, dtype=np.float64),
            np.asarray(pose_j.translation, dtype=np.float64),
            np.asarray(edge_transform.rotation, dtype=np.float64),
            np.asarray(edge_transform.translation, dtype=np.float64),
        )
        del residual
        numeric_i, numeric_j = _numeric_edge_jacobians(pose_i, pose_j, edge)
        worst = max(
            worst,
            float(np.abs(analytic_i - numeric_i).max()),
            float(np.abs(analytic_j - numeric_j).max()),
        )
    assert worst < 1e-4, f"worst analytic/numeric Jacobian disagreement: {worst!r}"


def test_edge_jacobian_matches_numeric_differences_near_identity():
    """The same check, specifically at a near-zero residual (an edge that is
    already almost satisfied) - the case every LM run spends most of its
    later iterations in, and a plausible place for a small-angle branch bug
    to hide.
    """
    import numpy as np

    pose_i = se3_exp((1e-6, -2e-6, 3e-6, 4e-7, -1e-7, 2e-7))
    pose_j = Rigid.identity()
    edge_transform = Rigid.identity()
    edge = PoseEdge(0, 1, edge_transform, 1.0)

    _, analytic_i, analytic_j = _edge_residual_and_jacobians(
        np.asarray(pose_i.rotation, dtype=np.float64),
        np.asarray(pose_i.translation, dtype=np.float64),
        np.asarray(pose_j.rotation, dtype=np.float64),
        np.asarray(pose_j.translation, dtype=np.float64),
        np.asarray(edge_transform.rotation, dtype=np.float64),
        np.asarray(edge_transform.translation, dtype=np.float64),
    )
    numeric_i, numeric_j = _numeric_edge_jacobians(pose_i, pose_j, edge)
    assert float(np.abs(analytic_i - numeric_i).max()) < 1e-6
    assert float(np.abs(analytic_j - numeric_j).max()) < 1e-6


# --- 4. a perfect graph converges immediately and moves nothing -----------------------


def test_perfect_graph_converges_immediately_and_moves_nothing():
    """Every edge already satisfied by the given poses: LM must not take a
    single step, let alone many - the gradient is already ~0 everywhere.
    """
    step = _rotation_about(math.radians(15.0))
    poses = [Rigid.identity()]
    for _ in range(5):
        poses.append(poses[-1].compose(step.inverse()))
    edges = [PoseEdge(i, i + 1, step, 1.0) for i in range(5)]

    result = optimise_poses_lm(poses, edges, anchor=0, iterations=100)

    assert result.converged
    assert result.iterations_run == 0
    for original, optimised in zip(poses, result.poses, strict=True):
        assert optimised == original
    assert result.final_error is not None
    assert result.final_error < 1e-18


def test_a_node_with_no_incident_edge_is_left_exactly_where_it_started():
    nodes = [
        Rigid.identity(),
        _rotation_about(math.radians(20.0)),
        _rotation_about(math.radians(99.0)),
    ]
    edges = [PoseEdge(0, 1, _rotation_about(-math.radians(20.0)), 1.0)]  # node 2 is isolated
    result = optimise_poses_lm(nodes, edges, anchor=0, iterations=10)
    assert result.poses[2] == nodes[2]
    assert any("node(s) [2]" in limitation for limitation in result.limitations)


def test_no_edges_at_all_returns_every_node_unchanged():
    nodes = [Rigid.identity(), _rotation_about(math.radians(20.0))]
    result = optimise_poses_lm(nodes, [], anchor=0, iterations=10)
    assert result.poses == tuple(nodes)
    assert result.converged
    assert result.iterations_run == 0


# --- 5. contradictory edges settle to a reported compromise, not a crash --------------


def test_contradictory_closure_edges_converge_to_a_compromise_not_a_crash():
    small_step = _rotation_about(math.radians(5.0))
    nodes = [Rigid.identity(), small_step, small_step.compose(small_step)]
    edges = [
        PoseEdge(0, 1, small_step, 1.0),
        PoseEdge(1, 2, small_step, 1.0),
        PoseEdge(0, 2, Rigid.identity(), 1.0),  # "they coincide"
        PoseEdge(0, 2, _rotation_about(math.radians(-90.0)), 1.0),  # "90 degrees apart"
    ]
    result = optimise_poses_lm(nodes, edges, anchor=0, iterations=200)

    for pose in result.poses:
        ensure_rotation(pose.rotation)  # never a non-rotation, never NaN

    assert result.converged
    worst = result.worst_edge
    assert worst is not None
    assert math.degrees(worst.rotation_error_radians) > 20.0


# --- the damping schedule must actually adapt -----------------------------------------


def _ill_conditioned_fixture(seed: int, n: int = 10):
    """A graph nonlinear enough that at least one LM step overshoots and is
    REJECTED before the run settles: near-``pi`` sequential steps, noisy
    measurements at wildly different weights (``1e-3`` to ``1e3``), and a
    handful of extra edges at ``weight=1e4`` between random node pairs. Built
    specifically to discriminate a working damping schedule from a frozen
    one - see ``test_damping_schedule_recovers_where_frozen_damping_fails``.
    """
    rng = random.Random(seed)
    nodes = [Rigid.identity()]
    for _ in range(n - 1):
        angle = rng.uniform(150.0, 179.0)
        step = _rotation_about(math.radians(angle))
        nodes.append(nodes[-1].compose(step))
    edges = []
    for i in range(n - 1):
        true_step = nodes[i].inverse().compose(nodes[i + 1])
        noise = rotation_about_axis((0.3, 0.1, 0.9), math.radians(rng.uniform(-30, 30)))
        noisy = true_step.compose(noise)
        weight = rng.choice([1.0, 1e-3, 1e3])
        edges.append(PoseEdge(i, i + 1, noisy, weight))
    for _ in range(4):
        a, b = rng.sample(range(n), 2)
        extra = _rotation_about(math.radians(rng.uniform(0.0, 180.0)))
        edges.append(PoseEdge(min(a, b), max(a, b), extra, 1e4))
    return nodes, edges


def test_damping_schedule_recovers_where_frozen_damping_fails():
    """The adaptive damping schedule is load-bearing, not decoration.

    ``_ill_conditioned_fixture(seed=8)`` starts about 5 orders of magnitude
    from its optimum (cost ~230000, converges to ~0.015). The correct
    implementation SHRINKS the damping parameter on every accepted step
    (Nielsen's schedule), so it accelerates from a cautious first move
    toward a fast, near-Gauss-Newton one as the fit improves, and converges
    in well under half the iteration budget. A build that seeds the damping
    parameter once and never adjusts it again - the schedule dropped from
    both the accept and reject branches - never gets to take a large step:
    every iteration stays as cautious as the first, and on this exact
    fixture it exhausts the whole 60-iteration budget without converging,
    final cost still in the thousands (measured while developing this test).
    A well-behaved graph (the headline fixture, a perfect graph) never
    forces this branch to matter at all, which is why it needs its own
    fixture rather than relying on those to catch a broken schedule.
    """
    nodes, edges = _ill_conditioned_fixture(seed=8)
    result = optimise_poses_lm(nodes, edges, anchor=0, iterations=60, tolerance=1e-13)

    assert result.converged
    assert result.final_error is not None
    assert result.final_error < 1.0
    assert result.iterations_run <= 30


# --- 6. guards --------------------------------------------------------------------------


def test_optimise_poses_lm_refuses_an_edge_index_out_of_range():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [PoseEdge(0, 5, Rigid.identity(), 1.0)])


def test_optimise_poses_lm_refuses_an_edge_relating_a_node_to_itself():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [PoseEdge(0, 0, Rigid.identity(), 1.0)])


def test_optimise_poses_lm_refuses_fewer_than_two_nodes():
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm([Rigid.identity()], [])


def test_optimise_poses_lm_refuses_a_non_positive_edge_weight():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [PoseEdge(0, 1, Rigid.identity(), 0.0)])
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [PoseEdge(0, 1, Rigid.identity(), -1.0)])


def test_optimise_poses_lm_refuses_an_anchor_out_of_range():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [], anchor=7)


def test_optimise_poses_lm_refuses_zero_iterations():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [], anchor=0, iterations=0)


def test_optimise_poses_lm_refuses_negative_tolerance():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed):
        optimise_poses_lm(nodes, [], anchor=0, tolerance=-1.0)


def test_every_posegraph_lm_refusal_is_a_scanner_error_the_doctor_can_classify():
    nodes = [Rigid.identity(), Rigid.identity()]
    with pytest.raises(PoseGraphLMNotAllowed) as excinfo:
        optimise_poses_lm(nodes, [], anchor=99)
    assert excinfo.value.status == DoctorStatus.UNSUPPORTED_MODE


# --- anchor semantics and API parity with scanner3d.posegraph -------------------------


def test_anchor_pose_is_exactly_unchanged():
    fixture = _loop_fixture()
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)
    for anchor in (0, 5, len(fixture.chain) - 1):
        result = optimise_poses_lm(list(fixture.chain), edges, anchor=anchor, iterations=30)
        assert result.poses[anchor] == fixture.chain[anchor]


def test_result_as_dict_carries_the_schema_version_and_is_json_safe():
    import json

    fixture = _loop_fixture()
    edges = list(fixture.sequential_edges) + list(fixture.closure_edges)
    result = optimise_poses_lm(list(fixture.chain), edges, anchor=0, iterations=20)
    payload = result.as_dict()
    json.dumps(payload)  # raises if anything is not JSON-safe
    assert payload["schema_version"] == 1
    assert payload["nodes"] == len(fixture.chain)
    assert payload["anchor"] == 0


def test_apply_update_leaves_the_anchor_and_untouched_nodes_alone():
    """A focused check on the retraction step itself: nodes with no parameter
    block come back as the SAME array objects, never recomputed.
    """
    import numpy as np

    rotations = [np.eye(3), np.eye(3)]
    translations = [np.zeros(3), np.array([1.0, 2.0, 3.0])]
    delta = np.zeros(6)
    param_index = {1: 0}
    new_rotations, new_translations = _apply_update(rotations, translations, delta, param_index)
    assert new_rotations[0] is rotations[0]
    assert new_translations[0] is translations[0]
    # Node 1 has a (zero) update applied, so it is recomposed even though the
    # numeric result is unchanged - a fresh array, not the same object.
    assert new_translations[1].tolist() == translations[1].tolist()
