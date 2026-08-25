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
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, rotation_about_axis
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

SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

# --- shared scene builders -----------------------------------------------------------

# The same oblique axis and box test_revisit.py's own fixtures use, and the
# same reason: an axis aligned with one of the box's own symmetry axes makes
# non-adjacent frames look like genuine revisits when they are not - caught
# empirically before it could hide a bug (see that file's module docstring).
# Reused here as independent constants, not an import, matching this
# codebase's established policy of duplicating a small fixture rather than
# coupling two test files.
_AXIS: Point = (0.55, 0.6, 0.58)
_THROUGH: Point = (0.0, 0.0, 117.0)
_BOX = Box(half_extents=(8.0, 12.0, 6.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 117.0)))

#: Voxel fraction the clouds are downsampled to. Coarser than test_revisit.py's
#: own 1/24: measured empirically (see the task report) that anything coarser
#: leaves some frames under detect_revisits' default 400-correspondence
#: verification floor after ICP's own trim, which silently drops genuine
#: revisits rather than proving anything about this module's contract.
_VOXEL_FRACTION = 1.0 / 28.0

#: Matches the true revisit gap in both fixtures below (one lap = 6 frames);
#: pipeline.py's own default (``max(8, frames // 8)``) would be 8 here and
#: would exclude every genuine revisit pair, so every call in this file
#: passes this explicitly rather than relying on the default.
_MIN_INDEX_GAP = 3

#: Small on purpose: measured to comfortably cover the true lap-closing pairs
#: by signature rank without spending Stage 2 ICP calls on the many pairs
#: that were never going to verify - see the task report for the measured
#: candidate counts this was tuned against.
_MAX_CANDIDATES = 8

#: Small enough to keep the suite fast, large enough that this tiny graph
#: (12 or 5 nodes) reaches its own convergence tolerance well within budget -
#: verified empirically, not assumed.
_ITERATIONS = 60


def _frame_cloud(angle_deg: float, *, seed: int) -> tuple[Point, ...]:
    """One rendered, noisy, downsampled frame at ``angle_deg`` about ``_AXIS``."""
    pose = rotation_about_axis(_AXIS, math.radians(angle_deg), through=_THROUGH)
    frame = render([_BOX], pose=pose, jitter=0.65, seed=seed)
    points = tuple(points_of(frame))
    extent = cloud_extent(points)
    return voxel_downsample(points, extent * _VOXEL_FRACTION)


@functools.cache
def _looping_clouds() -> tuple[tuple[Point, ...], ...]:
    """12 frames: 2 laps of 6 steps (60 degrees) each about ``_AXIS``.

    Frame k and frame k+6 sit at the same true angle (one full lap apart),
    so (0,6), (1,7), (2,8), (3,9), (4,10), (5,11) are all genuine revisits.
    Measured: 5 of those 6 verify at ``_MIN_INDEX_GAP``/``_MAX_CANDIDATES``
    above (one candidate's signature ranked outside the top 8) - an ODD
    count, which matters: it is what makes a count-only check able to catch
    a mutation that used the SAME slice for both the fitting and held-out
    sets (see ``test_held_out_partition_no_overlap_no_loss``'s docstring).
    """
    per_lap, laps = 6, 2
    step = 360.0 / per_lap
    return tuple(_frame_cloud(step * k, seed=k) for k in range(per_lap * laps))


@functools.cache
def _arc_clouds() -> tuple[tuple[Point, ...], ...]:
    """5 frames spanning a 60-degree arc that never returns - no revisit.

    Every pair is at least 15 degrees of true separation, comfortably past
    detect_revisits' default 12-degree verification threshold, so this
    fixture is expected to verify zero revisits - the honest answer for a
    path that never closes, and the scenario
    :data:`~scanner3d.pipeline.MIN_CLOSURES_TO_SPLIT` exists to catch.
    """
    return tuple(_frame_cloud(15.0 * k, seed=200 + k) for k in range(5))


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
    return _run(_arc_clouds(), max_candidates=3)


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
    assert result.optimised


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

    assert result.optimised
    # Scored exactly twice: once against the chained poses, once against the
    # optimised poses - both against the SAME held-out set.
    assert len(held_sets_seen) == 2
    held_before, held_after = held_sets_seen
    assert held_before == held_after
    assert len(held_before) == result.chained_score.held_out  # type: ignore[union-attr]

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


def test_too_few_revisits_returns_the_chained_poses_unchanged() -> None:
    """The returned poses must be exactly what chaining alone produces - not
    a lookalike recomputation: built here by calling
    :func:`scanner3d.pipeline._fit_pair` directly, the same function
    ``reconstruct_sweep`` itself calls, so agreement is a claim about
    ``reconstruct_sweep``'s control flow (did it skip optimising?), not
    about two independently written pieces of arithmetic happening to
    agree.
    """
    clouds = _arc_clouds()
    settings = _PairSettings(
        point_to_plane=True,
        backend="python",
        max_correspondence_distance=12.0,
        voxel_size=1.5,
        keep_fraction=0.8,
    )
    expected = [Rigid.identity()]
    for i in range(1, len(clouds)):
        transform = _fit_pair((settings, clouds[i], clouds[i - 1]))
        expected.append(expected[-1].compose(transform))

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

    result = _run(clouds, workers=1)

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
    _run(_arc_clouds(), workers=1, max_candidates=1)
    assert calls == [], "multiprocessing.Pool was constructed even though workers=1"

    _run(_arc_clouds(), workers=2, max_candidates=1)
    assert calls == [2], f"expected one Pool(processes=2), got {calls}"


# --- 4. backend does not change the answer ---------------------------------------------


@needs_scipy
def test_backend_does_not_change_the_answer() -> None:
    """``backend="python"`` vs ``"scipy"``, same poses within tolerance -
    not exact, since the two solve the identical well-posed problems by
    different numerical routes (see scanner3d.icp_plane's own backend
    equivalence tests for the measured per-call tolerance this is
    consistent with). Skipped cleanly, not failed, when scipy is
    unavailable.
    """
    reference = _main_result()
    accelerated = _run(_looping_clouds(), workers=1, backend="scipy")

    assert accelerated.revisits_found == reference.revisits_found
    assert len(accelerated.poses) == len(reference.poses)
    for ref_pose, acc_pose in zip(reference.poses, accelerated.poses, strict=True):
        assert _rotation_error(ref_pose, acc_pose) < 1e-4
        assert math.dist(ref_pose.translation, acc_pose.translation) < 1e-4


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
    assert record["optimised"] is True
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
