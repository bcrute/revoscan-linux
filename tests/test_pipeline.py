"""Tests for :mod:`scanner3d.pipeline` - the module a user actually runs.

What is at stake
------------------
This module's entire reason to exist is reporting HELD-OUT closure error
instead of training error - see its own module docstring's account of a
65-degree drift that looked healthy on every self-consistency metric. That
means the single most important property to verify is not "does it produce
a small number" but "are the constraints it SCORES against genuinely
different from the constraints it FITS against". An equivalence test on the
final numbers cannot distinguish a real held-out split from one that
silently reused the same edges for both roles - both would look like a
plausible score. So the load-bearing test here
(``test_held_out_partition_no_overlap_no_loss``) does not just check counts;
it instruments :func:`scanner3d.pipeline._score` and
:func:`scanner3d.pipeline.optimise_poses` directly to capture which EDGE
OBJECTS each one actually received, and asserts the held-out set and the
fitting set are disjoint by (i, j) identity - the same "deliberately wrong
stub" philosophy proven out on :mod:`scanner3d.icp_plane`'s backend wiring,
adapted to a monkeypatched spy since there is no wrong-but-solvable answer
to substitute here.

Runtime
--------
Two cached fixtures - see :func:`_looping_clouds` (a closing 2-lap sweep,
verified revisits) and :func:`_arc_clouds` (a short non-closing arc, no
revisits) - are each rendered once via ``functools.cache`` and reused across
every test that needs them, so the ~4.5s combined render cost is paid once
per test session, not once per test. ``min_index_gap``, ``max_candidates``
and ``iterations`` are all kept small on purpose (see the constants below):
this suite tests a CONTRACT, not a realistic sweep, and a smaller synthetic
scene proves the same property faster. Measured total for this file:
under 60 seconds.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence

import pytest

from scanner3d.compute import _probe_scipy
from scanner3d.errors import ScannerError
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, dot, rotation_about_axis
from scanner3d.pipeline import (
    MIN_CLOSURES_TO_SPLIT,
    ClosureScore,
    PipelineNotAllowed,
    SweepReconstruction,
    _fit_pair,
    _PairSettings,
    reconstruct_sweep,
)
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.revisit import detect_revisits
from scanner3d.simulate import Box, points_of, render
from scanner3d.tracking import (
    DEFAULT_POLICY,
    IncrementalTracker,
    bounded_source,
    retained_cloud,
)

SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

# --- shared scene builders -----------------------------------------------------------

# WHY THESE FIXTURES WERE REPLACED, which matters more than what they are now.
#
# Until model/keyframe tracking existed, this file's scenes were 60-degree steps
# of a small jittered box. That was fine for what these tests were written to
# check - the held-out closure contract - because the pose graph was handed the
# chain and asked to fix it. It was NOT a sweep that could be chained: measured
# directly on the old fixture, the adjacent alignment of frame 1 was 63.6
# degrees from the truth and frame 2 was 170.5 degrees from it. The optimisation
# was papering over a chain that was nonsense, and every number this file
# asserted was measured over that.
#
# :func:`~scanner3d.pipeline.reconstruct_sweep` now REFUSES a sweep with a frame
# it cannot place, so a fixture whose chain is meaningless is refused before any
# closure is looked at. The scene below is the same shape of scenario - a path
# that laps and therefore revisits - over geometry and a step size where the
# chain is real: measured, every frame places and no accepted pose is more than
# 2.8 degrees from the truth. The closure machinery is then being tested over a
# trajectory it could plausibly be given, which is what these tests always meant.

#: Standoff of the camera from the object's centre, in the same arbitrary depth
#: units everything else here uses.
_STANDOFF = 150.0

#: The object: a flat asymmetric slab with one lobe stuck off a corner. The lobe
#: is what makes it placeable at all. A box is nearly symmetric under quarter
#: turns, and a near-symmetric object offers several equally good placements -
#: which the tracker refuses, correctly, and which would make this file's
#: subject the ambiguity refusal rather than the closure contract.
_SLAB = Box(half_extents=(30.0, 12.0, 6.0))
_LOBE = Box(
    half_extents=(6.0, 6.0, 6.0),
    pose=Rigid.of(IDENTITY_MATRIX, (24.0, 10.0, 0.0)),
)

#: Voxel fraction the clouds are downsampled to: fine enough that a frame is a
#: surface rather than a scatter (measured: 400 to 1100 points per frame), and
#: coarse enough that this file's several full reconstructions stay inside a
#: test suite's patience.
_VOXEL_FRACTION = 1.0 / 32.0

#: One lap of :func:`_looping_clouds`, in frames. 16, so the step is 22.5
#: degrees - measured as a step this scene tracks all the way round, where 45
#: degrees is refused at the fifth frame and 30 at the tenth.
_PER_LAP = 16

#: Matches the true revisit gap in :func:`_looping_clouds` (one lap apart);
#: pipeline.py's own default (``max(8, frames // 8)``) would be 8 here and would
#: admit near-adjacent pairs that are not laps, so every call in this file
#: passes this explicitly rather than relying on the default.
_MIN_INDEX_GAP = _PER_LAP - 1

#: The arc fixture is far shorter than one lap, so it needs its own gap: a gap
#: at or above the frame count is refused by ``detect_revisits`` outright, which
#: would prove nothing about a path that simply never closes.
_ARC_INDEX_GAP = 3

#: Small on purpose: measured to cover the true lap-closing pairs by signature
#: rank without spending Stage 2 ICP calls on the many pairs that were never
#: going to verify. Measured on the scene below: 7 revisits verify at this
#: value and 10 at 16, and 7 is ODD - which matters, because it is what makes a
#: count-only check able to catch a mutation that used the SAME slice for both
#: the fitting and held-out sets (see
#: ``test_held_out_partition_no_overlap_no_loss``'s docstring).
_MAX_CANDIDATES = 8

#: Small enough to keep the suite fast, large enough that this small graph
#: reaches its own convergence tolerance well within budget - verified
#: empirically, not assumed.
_ITERATIONS = 60


@functools.cache
def _object_points() -> tuple[Point, ...]:
    """The object itself, as one cloud in its own frame.

    Assembled by rendering each shape from six directions and mapping every
    frame back through the pose that produced it, so the result covers all
    sides - which is what a camera walking round it needs, and what a single
    rendered view (one face, no depth cue for rotation about its normal) is not.
    """
    merged: list[Point] = []
    for axis, angle in (
        ((0.0, 1.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), math.pi / 2),
        ((0.0, 1.0, 0.0), math.pi),
        ((0.0, 1.0, 0.0), -math.pi / 2),
        ((1.0, 0.0, 0.0), math.pi / 2),
        ((1.0, 0.0, 0.0), -math.pi / 2),
    ):
        rotation = rotation_about_axis(axis, angle) if angle else Rigid.identity()
        pose = Rigid.of(rotation.rotation, (0.0, 0.0, _STANDOFF))
        for shape in (_SLAB, _LOBE):
            merged.extend(pose.inverse().apply_all(tuple(points_of(render([shape], pose=pose)))))
    return voxel_downsample(tuple(merged), 1.2)


def _camera(index: int, step_deg: float) -> Rigid:
    """The pose carrying object coordinates into the camera's, at one step."""
    orbit = rotation_about_axis((0.0, 1.0, 0.0), math.radians(step_deg * index))
    return Rigid.of(orbit.rotation, (0.0, 0.0, _STANDOFF))


def _frame_cloud(index: int, step_deg: float) -> tuple[Point, ...]:
    """One frame: the half of the object facing the camera, in camera coordinates.

    The visibility rule is the near half of a convex body, which this object is
    close enough to for a fixture. It matters that there IS one: handing every
    frame the whole object would make each pair a rigid re-registration of
    identical data, which is not what a scanner sees and not what tracking has
    to survive.
    """
    pose = _camera(index, step_deg)
    eye = pose.inverse().apply((0.0, 0.0, 0.0))
    visible = [point for point in _object_points() if dot(point, eye) > 0.0]
    points = tuple(pose.apply_all(visible))
    return voxel_downsample(points, cloud_extent(points) * _VOXEL_FRACTION)


def _true_pose(index: int, step_deg: float) -> Rigid:
    """Where frame ``index`` truly sits relative to frame 0, by construction."""
    return _camera(0, step_deg).compose(_camera(index, step_deg).inverse())


@functools.cache
def _looping_clouds() -> tuple[tuple[Point, ...], ...]:
    """24 frames of a camera walking round the object at 22.5 degrees a step.

    One lap is :data:`_PER_LAP` frames, so frame k and frame k + 16 sit at the
    same true viewpoint and every such pair is a genuine revisit. Measured: 7 of
    them verify at ``_MIN_INDEX_GAP``/``_MAX_CANDIDATES`` above - an ODD count,
    which is what several tests below depend on.
    """
    step = 360.0 / _PER_LAP
    return tuple(_frame_cloud(k, step) for k in range(_PER_LAP + 8))


@functools.cache
def _arc_clouds() -> tuple[tuple[Point, ...], ...]:
    """6 frames spanning a 75-degree arc that never returns - no revisit.

    Every pair is at least 15 degrees of true separation, comfortably past
    detect_revisits' 12-degree verification threshold, so this fixture is
    expected to verify zero revisits - the honest answer for a path that never
    closes, and the scenario
    :data:`~scanner3d.pipeline.MIN_CLOSURES_TO_SPLIT` exists to catch.
    """
    return tuple(_frame_cloud(k, 15.0) for k in range(6))


def _run(clouds: Sequence[tuple[Point, ...]], **overrides: object) -> SweepReconstruction:
    kwargs: dict[str, object] = {
        "min_index_gap": _MIN_INDEX_GAP,
        "max_candidates": _MAX_CANDIDATES,
        "iterations": _ITERATIONS,
        # Robust rejection (default True) runs up to DEFAULT_ROBUST_PASSES=8
        # relax-to-convergence rounds instead of one, and is a separate
        # concern from the held-out contract this file tests - not touched
        # here, and off by default to keep this suite's runtime in budget.
        "robust": False,
    }
    kwargs.update(overrides)
    return reconstruct_sweep(clouds, **kwargs)  # type: ignore[arg-type]


@functools.cache
def _main_result() -> SweepReconstruction:
    """The main scenario, computed once and reused by every test that only
    needs to inspect its properties (not re-run it under different settings
    or instrumentation).
    """
    return _run(_looping_clouds(), workers=1, backend="python")


@functools.cache
def _guard_result() -> SweepReconstruction:
    # max_candidates cut further than the main fixture: this scenario only
    # needs to confirm nothing verifies, and every candidate here is
    # genuinely far apart in true angle (see _arc_clouds), so a shallow
    # search is exactly as conclusive as a deep one, for less Stage 2 cost.
    return _run(_arc_clouds(), max_candidates=3, min_index_gap=_ARC_INDEX_GAP)


def _rotation_error(a: Rigid, b: Rigid) -> float:
    return max(abs(a.rotation[i][j] - b.rotation[i][j]) for i in range(3) for j in range(3))


# --- 1. the held-out contract: the property everything else depends on ---------------


def test_main_scenario_has_enough_revisits_to_exercise_the_split() -> None:
    """A precondition for every other test in this section, asserted directly
    rather than assumed: if this ever regresses (a simulate.py change, a
    detect_revisits default change), every test below would be exercising
    the wrong code path and should fail loudly here first, not downstream.
    """
    result = _main_result()
    assert result.revisits_found >= MIN_CLOSURES_TO_SPLIT
    # BOTH scores, not ``optimised``. The split is what this precondition is
    # about, and the split ran whenever both scores exist. Whether the optimised
    # poses are then RETURNED is a separate judgement made against the held-out
    # set, and on this scene it goes the other way: the tracker's chain is
    # already within a few degrees of the truth, so the pose graph makes
    # held-out agreement worse and ``reconstruct_sweep`` correctly hands back
    # the chain. Asserting ``optimised`` here would be asserting that the
    # chain is bad enough to need fixing.
    assert result.chained_score is not None
    assert result.optimised_score is not None


def test_held_out_counts_partition_the_detected_closures() -> None:
    """``held_out + used_for_fitting`` must equal ``revisits_found`` exactly,
    for both scores - no closure lost, and (see the exact-split test below
    for the stronger claim) none double-counted.
    """
    result = _main_result()
    assert result.chained_score is not None
    assert result.optimised_score is not None
    assert result.chained_score.held_out + result.chained_score.used_for_fitting == (
        result.revisits_found
    )
    assert result.optimised_score.held_out + result.optimised_score.used_for_fitting == (
        result.revisits_found
    )
    # Both scores are computed against the SAME held-out set (before and
    # after optimising), so they must report the same split.
    assert result.chained_score.held_out == result.optimised_score.held_out
    assert result.chained_score.used_for_fitting == result.optimised_score.used_for_fitting


def test_held_out_split_matches_the_documented_alternation_exactly() -> None:
    """Cross-checked against an independently recomputed closure list, not
    just internal counts: ``detect_revisits`` is a deterministic pure
    function of its input clouds and parameters, so calling it again here
    with the SAME arguments ``reconstruct_sweep`` used internally reproduces
    the identical closure list, and the documented ``[0::2]``/``[1::2]``
    split can be checked against it directly. The fixture's ODD verified
    count (5) makes this sensitive to a mutation that swapped which half is
    withheld: for an odd count the two halves have different sizes, so
    swapping them changes the reported counts - it would not for an even
    count, which is exactly why the fixture was tuned to produce one.
    """
    closures = list(
        detect_revisits(
            _looping_clouds(), min_index_gap=_MIN_INDEX_GAP, max_candidates=_MAX_CANDIDATES
        ).pose_edges()
    )
    assert len(closures) % 2 == 1, "fixture must verify an odd number of revisits - see docstring"
    expected_used = len(closures[0::2])
    expected_held = len(closures[1::2])
    assert expected_used != expected_held

    result = _main_result()
    assert result.revisits_found == len(closures)
    assert result.chained_score is not None
    assert result.chained_score.used_for_fitting == expected_used
    assert result.chained_score.held_out == expected_held


def test_held_out_partition_no_overlap_no_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing test in this file. Instruments the actual data flow
    rather than inferring it from output counts: captures the exact edge
    OBJECTS handed to ``_score`` (the held-out scorer, called once before and
    once after optimising) and to ``optimise_poses`` (the fitter), and
    asserts by (i, j) identity that no edge given to the scorer was also
    given to the fitter - the module's central claim, made checkable.
    """
    import scanner3d.pipeline as pipeline_module
    from scanner3d.posegraph import optimise_poses as original_optimise

    held_sets_seen: list[list[tuple[int, int]]] = []
    original_score = pipeline_module._score

    def spy_score(poses, held, used):
        held_sets_seen.append([(edge.i, edge.j) for edge in held])
        return original_score(poses, held, used)

    fitting_edge_sets: list[list[tuple[int, int]]] = []

    def spy_optimise(nodes, edges, **kwargs):
        fitting_edge_sets.append([(edge.i, edge.j) for edge in edges])
        return original_optimise(nodes, edges, **kwargs)

    monkeypatch.setattr(pipeline_module, "_score", spy_score)
    monkeypatch.setattr(pipeline_module, "optimise_poses", spy_optimise)

    result = _run(_looping_clouds(), workers=1, backend="python")

    # The SPLIT, not the acceptance - see
    # ``test_main_scenario_has_enough_revisits_to_exercise_the_split`` for why
    # the two came apart on this scene.
    assert result.chained_score is not None
    assert result.optimised_score is not None
    # Scored exactly twice: once against the chained poses, once against the
    # optimised poses - both against the SAME held-out set.
    assert len(held_sets_seen) == 2
    held_before, held_after = held_sets_seen
    assert held_before == held_after
    assert result.chained_score is not None
    assert len(held_before) == result.chained_score.held_out

    # Fitted exactly once, against the sequential edges plus the TRAINING
    # closures only.
    assert len(fitting_edge_sets) == 1
    fitting_pairs = set(fitting_edge_sets[0])
    held_pairs = set(held_before)
    assert held_pairs, "the held-out set must not be empty on this fixture"
    assert held_pairs.isdisjoint(fitting_pairs), (
        "a held-out closure was also given to the optimiser - the module's central "
        "claim (scoring against evidence the optimiser never fit) is false"
    )


# --- 2. too few revisits: the module must refuse to invent confidence -----------------


def test_too_few_revisits_does_not_optimise_or_score() -> None:
    result = _guard_result()
    assert result.revisits_found < MIN_CLOSURES_TO_SPLIT
    assert result.optimised is False
    assert result.converged is False
    assert result.chained_score is None
    assert result.optimised_score is None
    assert any("NOT corrected and NOT scored" in limitation for limitation in result.limitations)


def test_too_few_revisits_returns_the_tracked_poses_unchanged() -> None:
    """The returned poses must be exactly what the tracker produced - not a
    lookalike recomputation: built here by driving
    :class:`scanner3d.tracking.IncrementalTracker` with the results of
    :func:`scanner3d.pipeline._fit_pair`, the same two pieces
    ``reconstruct_sweep`` itself uses and in the same order, so agreement is a
    claim about ``reconstruct_sweep``'s control flow (did it skip optimising?)
    rather than about two independently written pieces of arithmetic happening
    to agree.

    This used to compose the adjacent fits directly, because that was all a
    pose was. It is not any more: an adjacent fit is one candidate of three and
    a pose is the placement they agreed on, so composing them here would be
    asserting the behaviour this file exists to prove was replaced.
    """
    clouds = _arc_clouds()
    settings = _PairSettings(
        point_to_plane=True,
        backend="python",
        max_correspondence_distance=12.0,
        voxel_size=1.5,
        keep_fraction=0.8,
        # The tracker's own convergence setting, because a pre-fit settled
        # against a different threshold is a different fit and its ``converged``
        # flag is now a hard gate.
        convergence_noise_fraction=DEFAULT_POLICY.convergence_noise_fraction,
    )
    tracker = IncrementalTracker(
        backend="python",
        point_to_plane=True,
        max_correspondence_distance=12.0,
        voxel_size=1.5,
        keep_fraction=0.8,
    )
    assert tracker.offer(clouds[0]).accepted
    for i in range(1, len(clouds)):
        # The identical two bounded clouds ``reconstruct_sweep`` pre-fits, not
        # the raw ones: the tracker samples what it is offered, so a pre-fit of
        # the whole cloud would be a fit of a different problem and this test
        # would be asserting that two different computations agree.
        adjacent = _fit_pair(
            (
                settings,
                bounded_source(clouds[i], DEFAULT_POLICY),
                retained_cloud(clouds[i - 1], DEFAULT_POLICY),
            )
        )
        decision = tracker.offer(clouds[i], adjacent_fit=adjacent)
        assert decision.accepted, decision.message()
    expected = tracker.poses

    result = _guard_result()
    assert len(result.poses) == len(expected)
    for actual_pose, expected_pose in zip(result.poses, expected, strict=True):
        assert actual_pose.rotation == expected_pose.rotation
        assert actual_pose.translation == expected_pose.translation


def test_min_closures_to_split_boundary_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly :data:`MIN_CLOSURES_TO_SPLIT` revisits must still trigger the
    split and optimise - the guard is documented as "fewer than", not "at
    most or fewer". Neither fixture in this file exercises the boundary
    VALUE itself (the looping fixture verifies 5, the arc verifies 0), so a
    mutation that widened the guard to ``<=`` would not otherwise be caught
    - confirmed empirically while writing this test.

    Reaching the exact boundary through real geometry is not something a
    scene can be reliably tuned to on demand (see the task report on how
    much tuning the 5-revisit count itself took), so this bypasses geometry
    entirely: monkeypatches ``detect_revisits`` to return a crafted
    ``RevisitReport`` with precisely :data:`MIN_CLOSURES_TO_SPLIT` verified
    revisits, referencing real frame indices from the (cheap, already-built)
    arc fixture so the chaining and optimisation that follow still run for
    real.
    """
    import scanner3d.pipeline as pipeline_module
    from scanner3d.revisit import Revisit, RevisitReport

    clouds = _arc_clouds()
    fake_verified = tuple(
        Revisit(
            i=0,
            j=k + 1,
            transform=Rigid.identity(),
            rmse=0.1,
            correspondences=500,
            rotation_angle=0.001,
            translation_magnitude=0.001,
        )
        for k in range(MIN_CLOSURES_TO_SPLIT)
    )
    assert len(fake_verified) == MIN_CLOSURES_TO_SPLIT
    fake_report = RevisitReport(
        frames=len(clouds),
        candidates=(),
        verified=fake_verified,
        rejected_as_local_minimum=(),
        min_index_gap=1,
        max_candidates=10,
        limitations=(),
    )
    monkeypatch.setattr(pipeline_module, "detect_revisits", lambda *args, **kwargs: fake_report)

    result = _run(clouds, workers=1, min_index_gap=_ARC_INDEX_GAP)

    assert result.revisits_found == MIN_CLOSURES_TO_SPLIT
    assert result.optimised is True
    assert result.chained_score is not None
    assert result.optimised_score is not None


# --- 3. workers do not change the answer ----------------------------------------------


@functools.cache
def _main_result_workers2() -> SweepReconstruction:
    return _run(_looping_clouds(), workers=2, backend="python")


def test_workers_do_not_change_the_answer() -> None:
    """Same input, ``workers=1`` vs ``workers=2``: identical poses. Pairs are
    independent and composition happens afterwards, in index order,
    regardless of which worker finished a given pair first - so this is
    expected to be EXACT agreement, not merely close.
    """
    serial = _main_result()
    parallel = _main_result_workers2()

    assert serial.revisits_found == parallel.revisits_found
    assert len(serial.poses) == len(parallel.poses)
    for serial_pose, parallel_pose in zip(serial.poses, parallel.poses, strict=True):
        assert serial_pose.rotation == parallel_pose.rotation
        assert serial_pose.translation == parallel_pose.translation

    assert serial.optimised == parallel.optimised
    assert serial.converged == parallel.converged
    assert serial.chained_score == parallel.chained_score
    assert serial.optimised_score == parallel.optimised_score


def test_workers_argument_actually_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equivalence alone cannot distinguish "workers correctly used, and
    parallelising a pair-independent computation happens not to change the
    answer" from "the workers argument is silently ignored and everything
    ran serially regardless" - both look identical from the outside, since
    the correct answer is the same either way. This proves the argument is
    genuinely read: spies on ``multiprocessing.Pool`` itself and asserts it
    is constructed with ``processes=2`` when asked, and never constructed at
    all when ``workers=1``.
    """
    import multiprocessing

    calls: list[int | None] = []
    original_pool = multiprocessing.Pool

    def spy_pool(*args, **kwargs):
        calls.append(kwargs.get("processes", args[0] if args else None))
        return original_pool(*args, **kwargs)

    monkeypatch.setattr(multiprocessing, "Pool", spy_pool)

    # max_candidates=1 here: this test does not care whether anything
    # verifies as a revisit, only whether Pool was constructed during the
    # (unavoidable, prior) chaining step - so Stage 2 is cut to the bone to
    # keep this check cheap.
    _run(_arc_clouds(), workers=1, max_candidates=1, min_index_gap=_ARC_INDEX_GAP)
    assert calls == [], "multiprocessing.Pool was constructed even though workers=1"

    _run(_arc_clouds(), workers=2, max_candidates=1, min_index_gap=_ARC_INDEX_GAP)
    assert calls == [2], f"expected one Pool(processes=2), got {calls}"


# --- 4. backend does not change the answer ---------------------------------------------

#: :func:`_looping_clouds` truncated to its first lap minus one frame, and a
#: SEPARATE fixture from it for exactly one reason, measured directly (see
#: below): the full 24-frame sweep sits ON a keyframe-selection threshold, so
#: it is unusable for this test and ``_looping_clouds`` itself is left alone
#: for the other tests that depend on its full length and closed lap.
#:
#: Measured with ``min_index_gap=15`` (the suite-wide default) on prefixes of
#: ``_looping_clouds()``: at 16, 18, 20 and the full 24 frames, the python and
#: scipy backends agree on every placement through frame 14 and then disagree
#: starting exactly at frame 15 - the first frame a lap-closing revisit can
#: seed from. At frame 15 the two backends keep the identical rescue/origin
#: decision but retain a DIFFERENT keyframe (frame 6 in one run, frame 7 in
#: the other) - a genuine decision split, not a rounding difference, and it
#: propagates to every later frame's keyframe set. That is a threshold sitting
#: exactly on the lap-closure event, so no amount of prefix trimming that still
#: includes frame 15 avoids it.
#:
#: The first 15 frames (0 through 14, one shy of the first possible revisit
#: pair) are on neither side of that threshold - measured below - while still
#: exercising real decisions: with ``min_index_gap=8`` six revisits verify, ten
#: of the fifteen placements come from the accepted model rather than an
#: adjacent fit, one is a rescue, and the keyframe set is repeatedly promoted
#: and demoted (frames 1 and 10 are both added and later dropped). ``8`` is
#: used here instead of the suite-wide ``_MIN_INDEX_GAP`` (15) only because 15
#: frames cannot satisfy a gap of 15 at all; ``detect_revisits`` refuses that
#: outright, and the value is otherwise unrelated to the threshold above.
_BACKEND_EQUIVALENCE_INDEX_GAP = 8


@functools.cache
def _backend_equivalence_clouds() -> tuple[tuple[Point, ...], ...]:
    return _looping_clouds()[:15]


@functools.cache
def _backend_equivalence_reference() -> SweepReconstruction:
    return _run(
        _backend_equivalence_clouds(),
        workers=1,
        backend="python",
        min_index_gap=_BACKEND_EQUIVALENCE_INDEX_GAP,
    )


@needs_scipy
def test_backend_does_not_change_the_answer() -> None:
    """``backend="python"`` vs ``"scipy"``: the identical per-frame placement
    DECISIONS, and poses that agree to well inside the sensor's noise floor.
    Skipped cleanly, not failed, when scipy is unavailable.

    A pose is the placement several candidates agreed on, and which frames
    become KEYFRAMES is a threshold on a continuous measurement of pose
    distance - so two backends whose intermediate numbers differ even
    slightly can, in general, land on opposite sides of that threshold and
    from there build different model references. :data:`_looping_clouds`'s
    full 24 frames measurably do this (see
    :data:`_BACKEND_EQUIVALENCE_INDEX_GAP`'s docstring for the frame and the
    two keyframes involved) - that is a fixture sitting on the threshold, not
    a backend computing something different, so this test uses a separate
    15-frame prefix instead, measured clear of it.

    Measured on that fixture: both backends verify the same 6 revisits, place
    all 15 frames, and agree on every placement's ``origin``, ``rescue`` and
    ``keyframes`` (``tracking["placements"]`` below) - zero of 15 differ. Their
    poses' rotation matrices differ by at most 6.1e-4 in any single entry
    (measured; the same max-abs-entry-difference metric
    :func:`tests.test_icp_plane_backend._rotation_error` uses, where a single
    converged ICP call's own two backends agree to within that suite's 1e-4 -
    this test chains 15 such placements plus revisit-driven model fits, so
    5e-3 keeps roughly an order of magnitude of margin over the measured
    6.1e-4 while staying far tighter than the 1.0 the chain could drift
    without the same reference winning every frame). Translations differ by at
    most 0.0505 depth units (measured); 0.2 is the tolerance asserted below -
    under a third of :data:`scanner3d.pipeline.NOISE_FLOOR` (0.65), the
    smallest distance that means anything on this sensor, and over 3x the
    measured value for margin.

    So this asserts what is actually claimed: the same verified-revisit count,
    the same number of poses, the same placement decision on every frame, and
    pose agreement stated in the units compared (rotation-matrix-entry
    magnitude; depth units for translation) and bounded by a cited scale. A
    backend that computed something different would change a decision, not
    survive at these tolerances.
    """
    reference = _backend_equivalence_reference()
    accelerated = _run(
        _backend_equivalence_clouds(),
        workers=1,
        backend="scipy",
        min_index_gap=_BACKEND_EQUIVALENCE_INDEX_GAP,
    )

    assert accelerated.revisits_found == reference.revisits_found
    assert len(accelerated.poses) == len(reference.poses)
    assert (accelerated.tracking or {})["placements"] == (reference.tracking or {})["placements"]
    for ref_pose, acc_pose in zip(reference.poses, accelerated.poses, strict=True):
        assert _rotation_error(ref_pose, acc_pose) < 5e-3
        assert math.dist(ref_pose.translation, acc_pose.translation) < 0.2


# --- 5. improvement_factor --------------------------------------------------------------


def _score(*, held_out: int, used_for_fitting: int, median: float = 0.1) -> ClosureScore:
    return ClosureScore(
        median_error_radians=median,
        worst_error_radians=median * 2,
        held_out=held_out,
        used_for_fitting=used_for_fitting,
    )


def _sweep(*, chained: ClosureScore | None, optimised: ClosureScore | None) -> SweepReconstruction:
    return SweepReconstruction(
        poses=(Rigid.identity(),),
        frames=1,
        revisits_found=0,
        chained_score=chained,
        optimised_score=optimised,
        optimised=chained is not None,
        converged=True,
        limitations=(),
    )


def test_improvement_factor_none_when_no_held_out_set() -> None:
    assert _sweep(chained=None, optimised=None).improvement_factor is None


def test_improvement_factor_none_when_optimised_score_is_not_held_out() -> None:
    """A ``ClosureScore`` with ``held_out=0`` is training error, even if one
    happens to be constructed - :attr:`ClosureScore.is_held_out` exists
    exactly so :attr:`SweepReconstruction.improvement_factor` can refuse to
    report a ratio computed from it, which this checks directly rather than
    trusting that :func:`reconstruct_sweep` never constructs one (it does
    not, structurally - see the module - but the property itself should not
    depend on that being true forever).
    """
    chained = _score(held_out=2, used_for_fitting=3, median=1.0)
    optimised_not_held = _score(held_out=0, used_for_fitting=5, median=0.1)
    result = _sweep(chained=chained, optimised=optimised_not_held)
    assert result.optimised_score is not None
    assert result.optimised_score.is_held_out is False
    assert result.improvement_factor is None


def test_improvement_factor_computes_the_measured_ratio() -> None:
    chained = _score(held_out=2, used_for_fitting=3, median=1.0)
    optimised = _score(held_out=2, used_for_fitting=3, median=0.25)
    result = _sweep(chained=chained, optimised=optimised)
    assert result.improvement_factor == pytest.approx(4.0)


def test_improvement_factor_is_infinite_for_a_zero_held_out_error() -> None:
    chained = _score(held_out=2, used_for_fitting=3, median=1.0)
    optimised = _score(held_out=2, used_for_fitting=3, median=0.0)
    result = _sweep(chained=chained, optimised=optimised)
    assert result.improvement_factor == math.inf


def test_main_scenario_improvement_factor_is_a_real_number() -> None:
    result = _main_result()
    assert result.optimised_score is not None
    assert result.optimised_score.is_held_out
    assert isinstance(result.improvement_factor, float)


def test_guard_scenario_improvement_factor_is_none() -> None:
    assert _guard_result().improvement_factor is None


# --- 6. guards ---------------------------------------------------------------------------

#: Trivial, non-degenerate points - these guards raise before any real cloud
#: is touched, so the content only needs to be shaped like a cloud.
_DUMMY_CLOUD: tuple[Point, ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)


def test_fewer_than_three_frames_refuses() -> None:
    with pytest.raises(PipelineNotAllowed, match="at least three frames"):
        reconstruct_sweep([_DUMMY_CLOUD, _DUMMY_CLOUD])


@pytest.mark.parametrize("keep_fraction", [0.0, -0.5, 1.5, math.nan])
def test_keep_fraction_outside_range_refuses(keep_fraction: float) -> None:
    with pytest.raises(PipelineNotAllowed, match="keep_fraction"):
        reconstruct_sweep([_DUMMY_CLOUD, _DUMMY_CLOUD, _DUMMY_CLOUD], keep_fraction=keep_fraction)


def test_keep_fraction_at_the_boundary_is_not_rejected_by_this_guard() -> None:
    """1.0 is the top of the documented ``(0, 1]`` range and must not be
    refused by the ``keep_fraction`` guard specifically. The dummy cloud
    used for the guard tests above is too small to complete real
    registration once it clears the guards - real processing (unlike the
    guard checks, which never touch a single point) collapses it during
    downsampling - so SOME failure downstream is expected here. This test's
    only claim is that the failure is not this guard's.
    """
    with pytest.raises(ScannerError) as excinfo:
        reconstruct_sweep([_DUMMY_CLOUD, _DUMMY_CLOUD, _DUMMY_CLOUD], keep_fraction=1.0)
    if isinstance(excinfo.value, PipelineNotAllowed):
        assert "keep_fraction" not in str(excinfo.value)


# --- 7. as_dict round-trips ---------------------------------------------------------------


def test_sweep_reconstruction_as_dict_round_trips_and_carries_schema_version() -> None:
    result = _main_result()
    record = json.loads(json.dumps(result.as_dict()))
    assert record["schema_version"] == 1
    assert record["frames"] == result.frames
    assert record["revisits_found"] == result.revisits_found
    assert record["optimised"] == result.optimised
    assert record["tracking"] == dict(result.tracking or {})
    assert record["chained"] == result.chained_score.as_dict()  # type: ignore[union-attr]
    assert record["optimised_score"] == result.optimised_score.as_dict()  # type: ignore[union-attr]
    assert record["improvement_factor"] == pytest.approx(result.improvement_factor)
    assert record["limitations"] == list(result.limitations)


def test_guard_scenario_as_dict_carries_the_limitation() -> None:
    result = _guard_result()
    record = json.loads(json.dumps(result.as_dict()))
    assert record["optimised"] is False
    assert record["chained"] is None
    assert record["optimised_score"] is None
    assert record["improvement_factor"] is None
    assert any("NOT corrected and NOT scored" in limitation for limitation in record["limitations"])


def test_closure_score_as_dict_shape() -> None:
    score = _score(held_out=2, used_for_fitting=3, median=0.5)
    record = score.as_dict()
    assert record["held_out"] == 2
    assert record["used_for_fitting"] == 3
    assert record["is_held_out"] is True
    assert record["median_error_degrees"] == pytest.approx(math.degrees(0.5))
    assert record["worst_error_degrees"] == pytest.approx(math.degrees(1.0))
