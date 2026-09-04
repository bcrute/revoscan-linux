"""What :mod:`scanner3d.tracking` will and will not place, and what it costs.

This file is in two halves, and the split is deliberate.

The GEOMETRIC half drives the real tracker over a real synthetic scene with real
ICP, and it is the expensive half. It is what proves the claim the module exists
to make: that placing a frame against keyframes and a bounded model reference
tracks a moving camera materially better than chaining it to its neighbour, and
that a fixed camera watching a turning object is unaffected. Nothing here is
stubbed, because a stub cannot drift and drift is the subject.

The POLICY half hands the tracker fits directly - the ``adjacent_fit``
parameter, or a stand-in for one - and never renders anything. Every gate,
every keyframe decision and every bound is exercised there, because a test that
had to persuade real ICP to return a residual of 99 would be testing ICP's
ability to fail on demand rather than testing the gate that refuses it.

What this file does NOT claim
-----------------------------
That an accepted pose is correct. The scene below is a convex-ish solid sampled
without noise and viewed by a camera on a perfect circle; a real scanner has
noise, missing returns, a support surface and a hand in the way. What the
geometric half establishes is a RELATIONSHIP - tracked against blind - measured
under conditions where both are given exactly the same data.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest

import scanner3d.tracking as tracking_module
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, dot, rotation_about_axis
from scanner3d.icp_plane import IcpPlaneGeometryUnusable, icp_point_to_plane
from scanner3d.pipeline import NOISE_FLOOR, frame_travel
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.relocalize import LocalFit, judge_local_edge
from scanner3d.simulate import Box, points_of, render
from scanner3d.tracking import (
    ADJACENT_ONLY_POLICY,
    ADJACENT_ORIGIN,
    AGREEMENT_RESIDUAL_MULTIPLE,
    ANCHOR_ORIGIN,
    BRIDGE_ORIGIN,
    KEYFRAME_MIN_DEGREES,
    KEYFRAME_ORIGIN,
    MODEL_ORIGIN,
    TRACK_DISAGREEMENT_TYPE,
    TRACK_FRAME_EMPTY_TYPE,
    TRACK_NO_CANDIDATE_TYPE,
    TRACK_RESCUE_UNSUPPORTED_TYPE,
    TRAVEL_GATE,
    IncrementalTracker,
    TrackingNotAllowed,
    TrackingPolicy,
)

# --- the scene ------------------------------------------------------------------------

#: Standoff of the camera from the object's centre, in the arbitrary depth units
#: everything in this repository uses.
_STANDOFF = 150.0

#: The object: a flat asymmetric slab with one lobe stuck off a corner.
#:
#: The lobe is load-bearing rather than decorative. A plain box is nearly
#: symmetric under quarter turns, and a near-symmetric object offers several
#: equally good placements - which this tracker refuses, correctly, and which
#: would make every test below a test of the ambiguity refusal instead of the
#: thing it was written for.
_SLAB = Box(half_extents=(30.0, 12.0, 6.0))
_LOBE = Box(half_extents=(6.0, 6.0, 6.0), pose=Rigid.of(IDENTITY_MATRIX, (24.0, 10.0, 0.0)))

#: Voxel side the assembled object is thinned to, in depth units. Fine enough
#: that a frame taken from it is a surface with usable normals; coarse enough
#: that a sixteen-frame sweep with three ICP solves a frame fits in a test.
_OBJECT_VOXEL = 1.2

#: Voxel fraction each frame is thinned to, as a fraction of its own extent.
_FRAME_VOXEL_FRACTION = 1.0 / 32.0

#: Degrees the camera moves between frames of the moving-camera sweep. Measured
#: on this scene: at this step every frame places, and the blind chain drifts
#: enough over sixteen frames for the comparison below to mean something.
_STEP_DEGREES = 7.0

#: Frames in the moving-camera sweep. Long enough for a chained error to
#: accumulate visibly - the point being made is about GROWTH, and a handful of
#: frames cannot show it - and short enough to stay inside a test's patience at
#: about a minute for both chains together.
_SWEEP_FRAMES = 24


@functools.cache
def _object_points() -> tuple[Point, ...]:
    """The object as one cloud in its own frame, covering every side.

    Assembled by rendering each shape from six directions and mapping every
    frame back through the pose that produced it. A single rendered view is one
    face, and one face has no geometry determining rotation about its own normal
    - so a scene built from one would make every test here a test of that
    singularity rather than of what the tracker does with a real object.
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
    return voxel_downsample(tuple(merged), _OBJECT_VOXEL)


def _camera(index: int, step_deg: float = _STEP_DEGREES) -> Rigid:
    """The pose carrying object coordinates into the camera's, at one step."""
    orbit = rotation_about_axis((0.0, 1.0, 0.0), math.radians(step_deg * index))
    return Rigid.of(orbit.rotation, (0.0, 0.0, _STANDOFF))


def _view(pose: Rigid) -> tuple[Point, ...]:
    """The half of the object facing a camera at ``pose``, in its coordinates.

    A visibility rule matters, and specifically it matters that there is one.
    Handing every frame the whole object would make each pair a rigid
    re-registration of identical data, which is not what a scanner sees and not
    what tracking has to survive: the interesting failures all come from the
    fact that consecutive frames share only part of a surface.
    """
    eye = pose.inverse().apply((0.0, 0.0, 0.0))
    visible = [point for point in _object_points() if dot(point, eye) > 0.0]
    points = tuple(pose.apply_all(visible))
    return voxel_downsample(points, cloud_extent(points) * _FRAME_VOXEL_FRACTION)


@functools.cache
def _sweep() -> tuple[tuple[Point, ...], ...]:
    """A camera walking round the object, one frame per :data:`_STEP_DEGREES`."""
    return tuple(_view(_camera(index)) for index in range(_SWEEP_FRAMES))


def _true_pose(index: int) -> Rigid:
    """Where frame ``index`` truly sits relative to frame 0, by construction."""
    return _camera(0).compose(_camera(index).inverse())


def _rotation_error(actual: Rigid, expected: Rigid) -> float:
    """How far apart two poses are in degrees, which is the units the gates use."""
    return math.degrees(actual.inverse().compose(expected).rotation_angle)


def _blind_chain(clouds: Sequence[Sequence[Point]]) -> list[Rigid]:
    """Poses from chaining each frame to the one before it, judging nothing.

    Exactly what the capture did before this module existed, written out here
    rather than imported, because the point of the comparison is that this
    behaviour is no longer available anywhere to compare against.
    """
    poses = [Rigid.identity()]
    for index in range(1, len(clouds)):
        try:
            fit = icp_point_to_plane(
                clouds[index],
                clouds[index - 1],
                max_correspondence_distance=12.0,
                voxel_size=1.5,
                keep_fraction=0.8,
            )
        except Exception:
            # The blind chain has no answer for a pair ICP refuses, which is
            # itself part of what is being compared: it repeats the last pose,
            # because there is nothing else it can do.
            poses.append(poses[-1])
            continue
        poses.append(poses[-1].compose(fit.transform))
    return poses


# --- stand-ins for the policy half ------------------------------------------------------


def _fit(**overrides: object) -> LocalFit:
    """An adjacent fit that clears every gate, unless a test asks otherwise.

    Every field :func:`~scanner3d.relocalize.judge_local_edge` measures, not
    just the transform: a stand-in carrying only a transform reports a NaN
    residual and no correspondences to that judgement, which is a refusal, and
    would make every test below assert the failure path by accident.
    """
    fields: dict[str, object] = {
        "transform": Rigid.identity(),
        "rmse": 0.2,
        "converged": True,
        "iterations": 9,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
    }
    fields.update(overrides)
    return cast(LocalFit, SimpleNamespace(**fields))


def _grid(offset: float = 0.0, side: int = 8, spacing: float = 2.0) -> tuple[Point, ...]:
    """A small solid lattice: cheap, non-degenerate, and exactly reproducible.

    Three-dimensional on purpose - a plane would be singular in rotation about
    its normal, which is a real case with its own test below and not something
    every other test should have to work around.
    """
    return tuple(
        (x * spacing + offset, y * spacing, z * spacing + 100.0)
        for x in range(side)
        for y in range(side)
        for z in range(3)
    )


def _plane(side: int = 12, spacing: float = 2.0) -> tuple[Point, ...]:
    """One flat panel, which determines no rotation about its own normal."""
    return tuple((x * spacing, y * spacing, 100.0) for x in range(side) for y in range(side))


def _shifted(cloud: Sequence[Point], transform: Rigid) -> tuple[Point, ...]:
    return tuple(transform.apply_all(cloud))


def _adjacent_only(**overrides: object) -> TrackingPolicy:
    """The adjacent-only policy, with named fields changed for one test."""
    import dataclasses

    return dataclasses.replace(ADJACENT_ONLY_POLICY, **overrides)  # type: ignore[arg-type]


# --- 1. the claim: a moving camera tracks better than blind chaining ---------------------

#: Blind chaining must be measurably worse than this by the end of the sweep, or
#: the comparison below is not measuring anything. Measured on this scene, the
#: blind chain climbs almost monotonically - 0.42, 0.56, 0.59, 0.51, ... 1.05,
#: 1.15, 1.33, 1.46, 1.60 - because every pairwise error it makes is permanent.
_BLIND_DRIFT_FLOOR_DEGREES = 1.20

#: What the tracker must stay inside over the same sweep. Measured: 1.00
#: degrees at its worst, and it does not trend - it sits between 0.31 and 1.00
#: from the third frame to the twenty-fourth, ending at 0.84. Set wider than the
#: measurement so the test fails on a regression rather than on the third
#: decimal place of a linear solve.
#:
#: Both this and :data:`_REQUIRED_IMPROVEMENT` are looser than the round-1
#: figures (0.62 worst, 2.75x), and the reason is a CORRECTION rather than a
#: regression, so it is recorded here rather than quietly absorbed. Round 1
#: exempted the adjacent candidate from the point budget and let the model
#: reference contain the selected keyframe's own points; making both bounds true
#: - every candidate fitted from
#: :data:`~scanner3d.tracking.CANDIDATE_SOURCE_POINTS` points, and the model
#: reference built only from accepted frames no other candidate is using - costs
#: precision on this fixture, and it was measured which of the two costs it:
#: with the same disjoint references and a 4000-point source budget the sweep
#: returns to 0.89 worst / 0.66 final, so the price is the point bound, not the
#: disjointness. 900 points is what the live capture actually has
#: (``scanner3d.gui.PREVIEW_POINTS``), so this is the honest number for the
#: appliance and the wider one is the one that was not.
_TRACKED_ERROR_CEILING_DEGREES = 1.20

#: How much better the tracker must be at the end of the sweep. Measured: 1.90x
#: over 24 frames (blind 1.595 degrees, tracked 0.840), and rising with the
#: length of the sweep, because one of the two numbers grows and the other does
#: not. See :data:`_TRACKED_ERROR_CEILING_DEGREES` for why this is below round
#: 1's 2.75x.
_REQUIRED_IMPROVEMENT = 1.75

#: How much worse a chain's second half is allowed to be than its first before
#: it counts as GROWING. Used both ways below: the tracker must stay inside it
#: and the blind chain must not.
_GROWTH_FACTOR = 1.5


@functools.cache
def _blind_sweep() -> tuple[Rigid, ...]:
    """The moving-camera sweep chained blindly, computed once."""
    return tuple(_blind_chain(_sweep()))


@functools.cache
def _tracked_sweep() -> tuple[Rigid, ...]:
    """The moving-camera sweep placed by the tracker, computed once."""
    tracker = IncrementalTracker()
    for cloud in _sweep():
        decision = tracker.offer(cloud)
        assert decision.accepted, f"frame {decision.frame}: {decision.message()}"
    return tracker.poses


def test_every_frame_of_an_ordinary_moving_camera_sweep_is_placed() -> None:
    """A precondition for the comparison below, asserted rather than assumed.

    If this regresses, the improvement test underneath it would be comparing a
    partial chain against a complete one and its number would mean nothing - so
    it should fail loudly here first.
    """
    assert len(_tracked_sweep()) == _SWEEP_FRAMES


def test_a_moving_camera_path_tracks_materially_better_than_blind_adjacent_chaining() -> None:
    """The claim this module was written to make, measured both ways at once.

    Both chains are given the IDENTICAL clouds and the identical ICP settings.
    The only difference is what a pose is allowed to come from: one neighbour,
    or a neighbour and a keyframe and a bounded model reference that have to
    agree. A blind chain accumulates every pairwise error permanently, which is
    why its error grows monotonically along the sweep and the tracker's does
    not.
    """
    blind = _blind_sweep()
    tracked = _tracked_sweep()

    blind_final = _rotation_error(blind[-1], _true_pose(_SWEEP_FRAMES - 1))
    tracked_final = _rotation_error(tracked[-1], _true_pose(_SWEEP_FRAMES - 1))
    worst_tracked = max(
        _rotation_error(pose, _true_pose(index)) for index, pose in enumerate(tracked)
    )

    assert blind_final > _BLIND_DRIFT_FLOOR_DEGREES, (
        "blind chaining did not drift on this scene, so the comparison proves nothing"
    )
    assert worst_tracked < _TRACKED_ERROR_CEILING_DEGREES
    assert blind_final / tracked_final >= _REQUIRED_IMPROVEMENT


def test_the_tracked_error_does_not_grow_along_the_sweep_the_way_a_chain_does() -> None:
    """The mechanism, not just the endpoint.

    A chain's error is the composition of every earlier error, so it can only
    grow. A pose measured against a reference built from many earlier frames is
    not a composition of anything, so it should not. Both halves of that are
    asserted, because either one alone could be satisfied by a scene that was
    simply too easy or too hard: the blind chain over the SAME frames has to
    grow, and the tracked one has to not.

    Stated as a ratio between halves rather than as monotonicity, which noise
    breaks on any given pair.
    """
    half = _SWEEP_FRAMES // 2

    def halves(poses: Sequence[Rigid]) -> tuple[float, float]:
        errors = [_rotation_error(pose, _true_pose(index)) for index, pose in enumerate(poses)]
        return max(errors[:half]), max(errors[half:])

    blind_first, blind_second = halves(_blind_sweep())
    tracked_first, tracked_second = halves(_tracked_sweep())

    assert blind_second > blind_first * _GROWTH_FACTOR
    assert tracked_second <= tracked_first * _GROWTH_FACTOR


def test_a_fixed_camera_watching_a_turning_object_is_placed_the_same_way() -> None:
    """The other capture the owner makes, and the one that already worked.

    Geometrically this is the same relative motion as the sweep above - the
    frames are identical clouds - and that is exactly the point: nothing in this
    module knows or cares which of the two moved, so a change that helped the
    moving camera at the turning object's expense would have to be a change that
    treated them differently, and there is no such code to write.
    """
    tracker = IncrementalTracker()
    poses = []
    for index in range(10):
        # The object turning under a fixed camera produces the same view as the
        # camera walking the other way round a fixed object.
        decision = tracker.offer(_view(_camera(-index)))
        assert decision.accepted, decision.message()
        assert decision.pose is not None
        poses.append(decision.pose)

    for index, pose in enumerate(poses):
        expected = _camera(0).compose(_camera(-index).inverse())
        assert _rotation_error(pose, expected) < _TRACKED_ERROR_CEILING_DEGREES


def test_every_candidate_reports_a_world_pose_whatever_reference_produced_it() -> None:
    """The transform-direction contract, stated and checked.

    An adjacent candidate composes through the previous accepted pose, a
    keyframe candidate through that keyframe's pose, and a model candidate
    through nothing at all because its reference is already in model
    coordinates. Getting any one of those backwards produces a pose that is not
    slightly wrong but nowhere near the object, so requiring every candidate to
    land near the truth is a direct test of all three compositions.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:4]:
        assert tracker.offer(cloud).accepted
    decision = tracker.offer(clouds[4])

    origins = {candidate.origin for candidate in decision.candidates}
    assert origins == {ADJACENT_ORIGIN, KEYFRAME_ORIGIN, MODEL_ORIGIN}
    for candidate in decision.candidates:
        assert _rotation_error(candidate.pose, _true_pose(4)) < 1.0, candidate.origin


def test_the_work_one_frame_costs_never_exceeds_the_policy_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound on ICP solves per offered frame, counted rather than argued.

    A long capture pays this on every frame, so "bounded" has to mean a number
    that can be read off the policy and checked, not "it settles down".
    """
    calls: list[int] = []
    real = icp_point_to_plane

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking_module, "icp_point_to_plane", counting)
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:6]:
        before = len(calls)
        tracker.offer(cloud)
        assert len(calls) - before <= tracker.policy.max_candidates


def test_an_adjacent_fit_handed_in_is_used_rather_than_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam the offline reconstruction depends on.

    ``reconstruct_sweep`` fits every adjacent pair up front and across
    processes, because those pairs are independent of each other. That is only
    honest if handing the result in produces the same candidate the tracker
    would have computed - so the adjacent candidate has to carry the handed-in
    fit's own measurements, not a second solve's.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted

    calls: list[int] = []
    monkeypatch.setattr(
        tracking_module,
        "icp_point_to_plane",
        lambda *a, **k: calls.append(1),
    )
    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(rmse=0.37))

    assert calls == []
    assert decision.accepted
    assert decision.candidates[0].origin == ADJACENT_ORIGIN
    assert decision.candidates[0].edge.rmse == 0.37


# --- 2. refusals, and that a refusal changes nothing --------------------------------------


def _placed_pair() -> IncrementalTracker:
    """A tracker with two accepted frames and nothing else, cheaply."""
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    assert tracker.offer(_grid(offset=0.5), adjacent_fit=_fit()).accepted
    return tracker


def _state(tracker: IncrementalTracker) -> tuple[object, ...]:
    """Everything a refusal is forbidden to move."""
    return (tracker.frames, tracker.poses, tracker.keyframes, tracker.retained_clouds)


def test_an_empty_frame_is_refused_and_moves_nothing() -> None:
    """The most literal form of the object leaving the view.

    A frame with nothing in it cannot be aligned, and the important half of that
    is the second one: it must not quietly leave the model where it was and let
    the capture carry on as though a pose had been placed.
    """
    tracker = _placed_pair()
    before = _state(tracker)

    decision = tracker.offer(())

    assert not decision.accepted
    assert decision.pose is None
    assert decision.failure_type == TRACK_FRAME_EMPTY_TYPE
    assert _state(tracker) == before


def test_a_planar_view_no_reference_can_constrain_is_refused_and_moves_nothing() -> None:
    """A single flat panel against a single flat panel: the textbook singular case.

    Point-to-plane alignment of a plane onto a plane is free to slide in two
    directions and to spin about the normal, so the linearised 6-DOF system is
    singular in rotation about that normal from the FIRST iteration.
    :func:`~scanner3d.icp_plane.icp_point_to_plane` refuses it rather than
    returning a transform free to wander there, no candidate is formed, and
    there is nothing to accept.

    Both clouds are planar deliberately, and that is a change from round 1,
    which offered a plane against a solid lattice. That pair is NOT singular:
    the lattice constrains every direction, and the fit converges in about
    thirty iterations at a residual inside the gate. It refused in round 1 only
    because the solver was being run at a convergence threshold four orders of
    magnitude below anything measurable (see
    :data:`~scanner3d.tracking.CONVERGENCE_NOISE_FRACTION`), so it kept
    iterating long past settling and eventually assembled a correspondence set
    that happened to be singular. A refusal that depends on how far past
    convergence a solver was allowed to run is not a test of the singular case;
    this is.
    """
    tracker = IncrementalTracker()
    assert tracker.offer(_plane()).accepted
    before = _state(tracker)

    decision = tracker.offer(tuple((x + 1.0, y, z) for x, y, z in _plane()))

    assert not decision.accepted
    assert decision.failure_type == TRACK_NO_CANDIDATE_TYPE
    assert any("singular in rotation" in refusal for refusal in decision.refusals)
    assert _state(tracker) == before


def test_candidates_that_disagree_refuse_instead_of_choosing_the_smaller_residual() -> None:
    """The rule that separates this module from "take the best fit".

    Both candidates here clear every gate on their own. One says the frame did
    not move; the other says it moved a long way. A residual cannot arbitrate
    between them, and picking the smaller one would be manufacturing a
    confidence the geometry does not supply - so the frame is refused with both
    placements recorded.
    """
    tracker = IncrementalTracker()
    for offset in (0.0, 0.5, 1.0, 1.5):
        cloud = _grid(offset=offset)
        assert tracker.offer(cloud, adjacent_fit=_fit()).accepted
    before = _state(tracker)

    # Far enough from the other candidates to be a different answer (the gate is
    # 5.2 units of surface movement) and near enough to be a plausible step (the
    # travel gate is 15.6), so what refuses this frame is the disagreement and
    # not the jump.
    liar = Rigid.of(IDENTITY_MATRIX, (10.0, 0.0, 0.0))
    decision = tracker.offer(_grid(offset=2.0), adjacent_fit=_fit(transform=liar))

    assert not decision.accepted
    assert decision.failure_type == TRACK_DISAGREEMENT_TYPE
    assert len(decision.passing) >= 2
    assert decision.agreeing == ()
    assert _state(tracker) == before


def test_a_placement_that_moves_the_surface_too_far_is_refused_as_a_jump() -> None:
    """Travel is a gate, not only a readout.

    A fit can be finite, converged, well matched and still describe the scanner
    arriving somewhere else entirely. Nothing else here would catch that: the
    residual is measured after the jump, so it says the frame agrees with
    wherever it landed.
    """
    tracker = _placed_pair()
    before = _state(tracker)

    leap = Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 4000.0))
    decision = tracker.offer(_grid(offset=1.0), adjacent_fit=_fit(transform=leap))

    assert not decision.accepted
    assert decision.failure_type == TRACK_NO_CANDIDATE_TYPE
    assert any("not a step" in refusal for refusal in decision.candidates[0].refusals)
    assert _state(tracker) == before


def test_a_fit_that_never_settled_is_not_a_candidate_even_when_the_others_agree() -> None:
    """Convergence is a gate, not a caveat, and this is where round 1 was wrong.

    Round 1 rewrote a non-converged fit as a converged one for gating purposes
    and accepted the candidate anyway when another reference agreed with it. That
    made :attr:`~scanner3d.tracking.TrackCandidate.accepted` disagree with
    ``candidate.edge.accepted``, and it let a fit that failed a named gate be
    counted as one of the independent references confirming a placement.

    The observation behind it was real - at a convergence threshold of a
    millionth of the target's extent, healthy fits never settle - and it is
    fixed at the source instead: the tracker now RUNS the solver at a threshold
    stated in the units the decision is made in
    (:data:`~scanner3d.tracking.CONVERGENCE_NOISE_FRACTION`), so a fit that still
    reports ``converged=False`` is a fit that genuinely never settled.

    The adjacent fit handed in here is the step the scene actually applied, so
    it AGREES with what the keyframe and the model say. It is still refused as a
    candidate, and it is still not listed among the references that agreed - and
    the frame is placed anyway, by the two references that did settle.
    """
    tracker = IncrementalTracker(policy=TrackingPolicy(motion_prior=False))
    clouds = _sweep()
    for cloud in clouds[:4]:
        assert tracker.offer(cloud).accepted
    step = _camera(3).compose(_camera(4).inverse())

    decision = tracker.offer(clouds[4], adjacent_fit=_fit(converged=False, transform=step))

    adjacent = next(c for c in decision.candidates if c.origin == ADJACENT_ORIGIN)
    assert adjacent.edge.converged is False
    assert adjacent.accepted is False
    assert adjacent.accepted == adjacent.edge.accepted
    assert any("never settled" in refusal for refusal in adjacent.refusals)
    assert ADJACENT_ORIGIN not in decision.passing
    assert ADJACENT_ORIGIN not in decision.agreeing
    assert decision.accepted, decision.message()
    assert decision.origin in {KEYFRAME_ORIGIN, MODEL_ORIGIN}


def test_a_fit_that_never_settled_and_nothing_confirmed_is_refused() -> None:
    """The same gate where it is the only evidence there is.

    The tentative segment of a relocalization runs under exactly this policy -
    chained against itself, with no model to confirm anything with - so this is
    the rule that keeps its links judged as strictly as they were before this
    module existed.
    """
    tracker = _placed_pair()
    before = _state(tracker)

    decision = tracker.offer(_grid(offset=1.0), adjacent_fit=_fit(converged=False))

    assert not decision.accepted
    assert any("never settled" in item for item in decision.refusals)
    assert _state(tracker) == before


def test_the_start_of_a_capture_is_not_exempt_from_the_convergence_gate() -> None:
    """There is no startup loophole, and round 1 had one.

    Round 1 read "could anything have confirmed this placement" off
    ``len(candidates) > 1``, which meant the frames early in a capture - where no
    second reference exists yet to form a candidate from - skipped the gates a
    lone placement is supposed to clear. Frame 1 of a capture is precisely such a
    frame, and a fit that never settled is precisely such a gate.

    Refusing it is not "no capture can begin": a capture begins with an anchor,
    which is placed at the identity without being fitted at all, and with
    ordinary frames whose adjacent fits DO settle - the frame below settles when
    it is offered honestly, which is asserted here so this test cannot pass by
    refusing everything.
    """
    tracker = IncrementalTracker()
    assert tracker.offer(_grid()).accepted
    before = _state(tracker)

    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(converged=False))

    assert not decision.accepted
    assert any("never settled" in item for item in decision.refusals)
    assert _state(tracker) == before
    assert tracker.offer(_grid(offset=0.5)).accepted


def test_a_frame_no_reference_can_place_is_refused_rather_than_given_the_last_pose() -> None:
    """The 2026-08-28 failure, stated as a property.

    What that capture did was keep going with a model that had stopped moving.
    The tracker must never do the arithmetic that produces that: a frame nothing
    placed gets no pose at all, and the pose count does not move.
    """
    tracker = _placed_pair()
    before_poses = tracker.poses

    decision = tracker.offer(_grid(offset=1.0), adjacent_fit=_fit(rmse=99.0))

    assert not decision.accepted
    assert decision.pose is None
    assert tracker.poses == before_poses


def test_a_candidate_that_barely_touches_its_reference_is_refused() -> None:
    """Overlap is measured against the reference the residual came from.

    A small residual over a corner of a frame is not evidence about the frame,
    and the gate is expressed as a policy field so a test can state the
    threshold it is exercising rather than arrange geometry until one fires.
    """
    # Both overlap gates move together: ``TrackingPolicy`` refuses a rescue
    # gate below the ordinary one, because a placement with no second opinion
    # must never be the easiest one to accept.
    tracker = IncrementalTracker(
        policy=TrackingPolicy(min_reference_overlap=0.999, rescue_min_overlap=0.999)
    )
    clouds = _sweep()
    assert tracker.offer(clouds[0]).accepted
    assert tracker.offer(clouds[1]).accepted
    decision = tracker.offer(clouds[2])

    non_adjacent = [c for c in decision.candidates if c.origin != ADJACENT_ORIGIN]
    assert non_adjacent
    assert all(not candidate.accepted for candidate in non_adjacent)
    assert all(
        any("overlaps the" in refusal for refusal in candidate.refusals)
        for candidate in non_adjacent
    )


# --- 3. the model-only rescue, and what it has to clear -----------------------------------


def _rescue_ready(policy: TrackingPolicy) -> IncrementalTracker:
    """A tracker whose model reference spans several accepted viewpoints.

    Built with real frames rather than a lattice, because the rescue's whole
    subject is whether a reference assembled from several viewpoints can place a
    frame the previous one could not - and a reference of four copies of one
    lattice would be several frames of one viewpoint.
    """
    tracker = IncrementalTracker(policy=policy)
    for cloud in _sweep()[:6]:
        assert tracker.offer(cloud).accepted, "the rescue fixture must start from a real chain"
    return tracker


def test_a_model_only_placement_is_reported_as_a_rescue_and_accepted_when_supported() -> None:
    """The case this module exists for, reached deliberately.

    The adjacent fit is made to fail and the keyframe reference is switched off,
    so the model is the only reference left. With several accepted viewpoints
    behind it and the frame substantially on it, that is a placement - and it is
    labelled a rescue, because an operator whose scan is being held up by
    geometry captured minutes ago should be told so.
    """
    tracker = _rescue_ready(TrackingPolicy(keyframe_candidates=0, motion_prior=False))

    decision = tracker.offer(_sweep()[6], adjacent_fit=_fit(rmse=99.0))

    assert decision.accepted, decision.message()
    assert decision.origin == MODEL_ORIGIN
    assert decision.rescue is True
    assert _rotation_error(decision.pose or Rigid.identity(), _true_pose(6)) < 1.0


def test_a_rescue_with_too_few_accepted_viewpoints_behind_it_refuses() -> None:
    """Remove the multi-view evidence and the rescue must fail.

    This is the acceptance contract's own test. A reference assembled from one
    or two nearby viewpoints is blind in the same direction a weak frame is, so
    a placement made against it is the same singularity wearing a different
    name - and there is no second opinion left to catch it.
    """
    tracker = _rescue_ready(
        TrackingPolicy(keyframe_candidates=0, rescue_min_reference_frames=99, motion_prior=False)
    )
    before = _state(tracker)

    decision = tracker.offer(_sweep()[6], adjacent_fit=_fit(rmse=99.0))

    assert not decision.accepted
    assert decision.failure_type == TRACK_RESCUE_UNSUPPORTED_TYPE
    assert any("accepted frame(s), under the" in refusal for refusal in decision.refusals)
    assert _state(tracker) == before


def test_a_rescue_that_barely_overlaps_its_reference_refuses() -> None:
    """Remove the overlap and the rescue must fail, for the same reason.

    Stated through the policy field rather than by arranging a frame that
    happens to overlap by 0.4, so the test names the gate it is exercising.
    """
    tracker = _rescue_ready(
        TrackingPolicy(keyframe_candidates=0, rescue_min_overlap=0.999, motion_prior=False)
    )
    before = _state(tracker)

    decision = tracker.offer(_sweep()[6], adjacent_fit=_fit(rmse=99.0))

    assert not decision.accepted
    assert decision.failure_type == TRACK_RESCUE_UNSUPPORTED_TYPE
    assert any("overlaps the model reference" in refusal for refusal in decision.refusals)
    assert _state(tracker) == before


def test_an_ordinary_agreed_placement_is_not_labelled_a_rescue() -> None:
    """The flag has to distinguish something, or it distinguishes nothing."""
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:3]:
        assert tracker.offer(cloud).accepted
    decision = tracker.offer(clouds[3])

    assert decision.accepted
    assert decision.rescue is False
    assert len(decision.agreeing) > 1


# --- 4. keyframes: chosen from motion and novelty, never from frame number ----------------


#: A step past the keyframe motion gate that is still an ordinary step.
#:
#: The two constraints pull opposite ways and both are real. It has to exceed
#: :data:`~scanner3d.tracking.KEYFRAME_MIN_DEGREES` or no keyframe is ever
#: inserted; and the surface it moves - these lattices sit about 100 units in
#: front of the origin the rotation is about - has to stay under the travel gate
#: or every frame is refused as a jump. Seven degrees moves this lattice about
#: 12 units, against a 15.6 gate.
_KEYFRAME_STEP_DEGREES = KEYFRAME_MIN_DEGREES + 1.0


def _walk(tracker: IncrementalTracker, steps: Sequence[Rigid]) -> None:
    """Offer one frame per step, with the step supplied as the adjacent fit.

    The clouds are a lattice and the fits are stated, so the poses the tracker
    ends up with are exactly the ones this test chose. That is the point: a
    keyframe rule is a rule about POSES, and testing it through real ICP would
    be testing whether ICP recovered the poses.
    """
    for step in steps:
        decision = tracker.offer(_grid(), adjacent_fit=_fit(transform=step))
        assert decision.accepted, decision.message()


def test_a_long_stationary_sequence_does_not_fill_the_keyframe_set() -> None:
    """The failure a frame-counting rule produces, refused directly.

    A scanner left pointing at one spot delivers frames at exactly the rate a
    moving one does. A keyframe set filled by counting them holds twelve copies
    of one view precisely when the tracker most needs twelve different ones.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    _walk(tracker, [Rigid.identity()] * 24)

    assert tracker.frames == 25
    assert tracker.keyframes == (0,)


def test_distinct_viewpoints_do_become_keyframes() -> None:
    """The other half of the same rule, or the first half proves nothing."""
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    turn = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_KEYFRAME_STEP_DEGREES))
    _walk(tracker, [turn] * 5)

    assert tracker.keyframes == (0, 1, 2, 3, 4, 5)


def test_every_keyframe_decision_is_recorded_with_the_reason_for_it() -> None:
    """ "The set holds 0, 4 and 9" does not say why 6 is absent.

    The difference between "it was never novel" and "it was evicted as
    redundant" is the difference between a capture that sampled its viewpoints
    badly and one whose bound is too small, and only one of those is worth
    changing a setting over.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    anchor = tracker.offer(_grid())
    assert [item.action for item in anchor.keyframe_decisions] == ["anchor"]

    still = tracker.offer(_grid(), adjacent_fit=_fit())
    assert [item.action for item in still.keyframe_decisions] == ["retain"]
    assert "motion gate" in still.keyframe_decisions[0].reason

    turn = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_KEYFRAME_STEP_DEGREES))
    moved = tracker.offer(_grid(), adjacent_fit=_fit(transform=turn))
    assert [item.action for item in moved.keyframe_decisions] == ["insert"]
    assert "past the" in moved.keyframe_decisions[0].reason


def test_eviction_is_deterministic_and_every_bound_holds_over_a_long_sequence() -> None:
    """The bound is the promise; determinism is what makes it inspectable.

    Run twice over identical input, the set has to come out identical - an
    eviction rule that depended on iteration order would make a capture's
    keyframes an accident of dictionary layout, and the report that named them
    would be describing one run rather than the policy.
    """
    turn = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_KEYFRAME_STEP_DEGREES))

    runs = []
    for _ in range(2):
        tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
        assert tracker.offer(_grid()).accepted
        _walk(tracker, [turn] * 40)
        assert len(tracker.keyframes) <= tracker.policy.max_keyframes
        assert tracker.retained_clouds <= tracker.policy.max_retained_clouds
        assert tracker.frames == 41
        runs.append(tracker.keyframes)

    assert runs[0] == runs[1]


def test_the_anchor_and_the_most_recent_keyframes_are_never_evicted() -> None:
    """The anchor defines the model frame and the recent ones are where the
    scanner is now. An eviction rule that dropped either would be trading away
    the two references a placement most needs for the sake of a bound.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    turn = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_KEYFRAME_STEP_DEGREES))
    _walk(tracker, [turn] * 40)

    keyframes = tracker.keyframes
    assert keyframes[0] == 0
    assert keyframes[-1] == tracker.frames - 1
    assert keyframes[-2] == tracker.frames - 2


def test_a_refused_frame_never_touches_the_keyframe_set() -> None:
    """The state-immutability promise, aimed at the part of it most easily lost.

    A tracker that had inserted a keyframe and then refused the frame would make
    the next frame's evidence describe a model no decision ever accepted.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    turn = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_KEYFRAME_STEP_DEGREES))
    _walk(tracker, [turn] * 3)
    before = _state(tracker)

    refused = tracker.offer(_grid(), adjacent_fit=_fit(rmse=99.0))

    assert not refused.accepted
    assert refused.keyframe_decisions == ()
    assert _state(tracker) == before


# --- 5. seeded frames: a recorded boundary is placed, never refitted -----------------------


def test_a_seeded_frame_is_placed_exactly_where_it_was_told_and_never_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a recorded relocalization bridge is.

    The boundary was decided during the capture against evidence in its own
    record. Refitting it here would be doing the one thing the boundary exists
    to prevent - handing ICP two frames that are not a small step apart - so no
    solve may run, and the pose must be the one supplied, to the bit.
    """
    monkeypatch.setattr(
        tracking_module,
        "icp_point_to_plane",
        lambda *a, **k: pytest.fail("a seeded frame must not be fitted"),
    )
    tracker = IncrementalTracker()
    assert tracker.offer(_grid()).accepted

    placed = Rigid.of(IDENTITY_MATRIX, (5.0, 6.0, 7.0))
    decision = tracker.seed(_grid(offset=1.0), placed)

    assert decision.accepted
    assert decision.origin == BRIDGE_ORIGIN
    assert decision.pose == placed
    assert tracker.poses[-1] == placed


def test_the_first_seeded_frame_of_a_batch_is_the_keyframe_and_the_rest_are_not() -> None:
    """A published batch is one viewpoint, not a dozen.

    The boundary itself always becomes a keyframe: it is where the model gains a
    view it has no other way of reaching. The frames published behind it were
    placed by the same recorded decision and are ordinary members of the chain,
    and admitting all of them would fill the whole set with one viewpoint.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    pose = Rigid.of(IDENTITY_MATRIX, (5.0, 0.0, 0.0))
    for offset in range(4):
        tracker.seed(_grid(offset=float(offset)), pose, anchor=offset == 0)

    assert tracker.keyframes == (0, 1)


def test_a_seeded_frame_with_no_geometry_is_refused_rather_than_placed() -> None:
    """A boundary is only as good as the frame it names.

    Seeding a frame nothing can be placed against afterwards would put a pose in
    the chain and leave the next frame with a reference that is not a surface.
    """
    tracker = IncrementalTracker()
    assert tracker.offer(_grid()).accepted
    before = _state(tracker)

    decision = tracker.seed((), Rigid.identity())

    assert not decision.accepted
    assert decision.failure_type == TRACK_FRAME_EMPTY_TYPE
    assert _state(tracker) == before


def test_the_very_first_frame_is_the_anchor_however_it_arrives() -> None:
    """The model frame is defined by the first accepted frame and nothing else."""
    tracker = IncrementalTracker()
    decision = tracker.offer(_grid())

    assert decision.accepted
    assert decision.origin == ANCHOR_ORIGIN
    assert decision.pose == Rigid.identity()
    assert tracker.keyframes == (0,)


# --- 6. the policy is a value, and every decision is a record ------------------------------


def test_the_same_clouds_offered_twice_produce_the_same_decisions() -> None:
    """Determinism is what lets the offline path reproduce the live one.

    Nothing in this module may read a clock, a device or a random number, and
    the way to state that as a test is to run it twice and require the answers
    to be equal rather than close.
    """
    clouds = _sweep()[:5]
    first = IncrementalTracker()
    second = IncrementalTracker()
    for cloud in clouds:
        a = first.offer(cloud)
        b = second.offer(cloud)
        assert a.accepted == b.accepted
        assert a.origin == b.origin
        assert a.pose == b.pose
    assert first.keyframes == second.keyframes


def test_the_policy_is_schema_versioned_and_survives_a_json_round_trip() -> None:
    """The live capture and the offline reconstruction share this policy.

    A reader comparing a capture's decisions with a reconstruction's has to be
    able to SEE that the two ran the same one, which means it has to reach a
    durable record - and a record written with ``allow_nan=False`` refuses
    anything JSON has no word for.
    """
    policy = TrackingPolicy().as_dict()
    assert policy["schema_version"] == tracking_module.SCHEMA_VERSION
    assert policy["max_candidates"] == TrackingPolicy().max_candidates
    assert policy["agreement_cluster_margin"] == tracking_module.AGREEMENT_CLUSTER_MARGIN
    assert json.loads(json.dumps(policy, allow_nan=False)) == policy


def test_a_refusal_serialises_with_every_candidate_and_no_nan() -> None:
    """A refusal a reader cannot inspect is not evidence.

    ``travel`` is infinite on a refused candidate, which is the honest word for
    "there was no step to measure" and is exactly what a JSON encoder cannot
    write - so the record has to say ``null`` rather than crash or substitute a
    finite number nobody measured.
    """
    tracker = _placed_pair()
    decision = tracker.offer(_grid(offset=1.0), adjacent_fit=_fit(rmse=99.0))

    record = decision.as_dict()
    assert record["accepted"] is False
    assert record["pose"] is None
    assert record["travel"] is None
    assert record["candidates"]
    assert json.loads(json.dumps(record, allow_nan=False))["refusals"] == list(decision.refusals)


def test_an_accepted_decision_names_what_agreed_and_what_placed_it() -> None:
    """The record has to distinguish adjacent, keyframe and model acceptance.

    "Still tracking" while every pose comes from the accepted model means the
    individual views have gone weak, and a record that collapsed the three into
    "aligned" could not tell anyone that.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:3]:
        assert tracker.offer(cloud).accepted
    decision = tracker.offer(clouds[3])

    record = decision.as_dict()
    assert record["origin"] in {ADJACENT_ORIGIN, KEYFRAME_ORIGIN, MODEL_ORIGIN}
    assert record["agreeing"] == list(decision.agreeing)
    assert record["passing"] == list(decision.passing)
    assert record["dissenting"] == list(decision.dissenting)
    assert record["keyframes"] == list(tracker.keyframes)
    assert json.loads(json.dumps(record, allow_nan=False))["frame"] == decision.frame


def test_the_tracker_record_carries_the_policy_the_bounds_and_the_limitations() -> None:
    """What a reconstruction embeds, so a reader is not left to assume it."""
    tracker = IncrementalTracker()
    assert tracker.offer(_grid()).accepted
    record = tracker.record()

    assert record["policy"] == TrackingPolicy().as_dict()
    assert record["frames"] == 1
    assert record["keyframes"] == [0]
    assert record["limitations"]
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_keyframes": 0},
        {"max_reference_frames": 0},
        {"model_reference_points": 0},
        {"candidate_source_points": 0},
        {"max_retained_clouds": 0},
        {"keyframe_candidates": -1},
        # Every entry below is a RELATIONSHIP round 1 admitted and then broke
        # somewhere else, which is why each is named rather than folded into a
        # single range check.
        # The anchor plus the protected tail already fill the bound, so an
        # over-full keyframe set could never be brought back inside it.
        {"max_keyframes": 2, "protected_recent_keyframes": 2},
        # Fewer retained clouds than keyframes: retention would release a cloud
        # keyframe selection and the model reference both assume is present.
        {"max_keyframes": 12, "max_retained_clouds": 12},
        # Fewer reference points than reference frames: a frame could be NAMED
        # in a reference it contributed no point to, and the rescue's
        # multi-viewpoint gate is counted against that name.
        {"model_reference_points": 4, "max_reference_frames": 8},
        # A retained cloud thinned below what a placement can be built on.
        {"retained_cloud_points": 2},
        {"min_track_points": 2},
        # A rescue - the placement with no second opinion - made EASIER to
        # accept than an ordinary confirmed one.
        {"rescue_min_overlap": 0.2, "min_reference_overlap": 0.35},
        {"rescue_min_reference_frames": 1},
        {"min_reference_overlap": 0.0},
        {"min_reference_overlap": 1.5},
        {"max_travel_multiple": 0.0},
        # An extrapolation that could never be formed for any frame at all: one
        # step is the ordinary next frame, so this is a policy switching the
        # prediction off by arithmetic instead of by ``motion_prior``.
        {"max_prior_steps": 0},
        {"convergence_noise_fraction": 0.0},
        {"agreement_cluster_margin": 1.0},
        {"agreement_cluster_margin": math.inf},
    ],
)
def test_a_policy_with_an_impossible_bound_is_refused_at_construction(
    overrides: dict[str, int],
) -> None:
    """Refused where it is written, not where it would first misbehave.

    A zero point budget or a keyframe set with no room for the anchor does not
    produce a worse answer, it produces a tracker that cannot form a reference
    at all - and finding that out on frame 400 of a capture is finding it out
    too late.
    """
    with pytest.raises(TrackingNotAllowed):
        TrackingPolicy(**overrides)  # type: ignore[arg-type]


def test_the_adjacent_only_policy_switches_references_off_without_loosening_a_gate() -> None:
    """What the tentative segment of a relocalization runs under.

    A segment chained against itself has no accepted model and no keyframe set
    to refer to. Those two references are absent, which is a fact about the
    situation - and every gate the shared policy applies is still exactly the
    same number, which is the part that has to be true for the two paths to be
    one implementation.
    """
    shared = TrackingPolicy()
    adjacent = ADJACENT_ONLY_POLICY

    assert adjacent.use_keyframes is False
    assert adjacent.use_model is False
    # One REFERENCE, which is what "switches references off" means. The ceiling
    # is one higher because the adjacent candidate may be retried once from the
    # motion prediction - a second attempt at the same reference under the same
    # gates, not a second reference.
    assert adjacent.ordinary_candidates == 1
    assert adjacent.max_candidates == 2
    for field in (
        "min_track_points",
        "min_reference_overlap",
        "rescue_min_overlap",
        "rescue_min_reference_frames",
        "max_travel_multiple",
        "agreement_cluster_margin",
    ):
        assert getattr(adjacent, field) == getattr(shared, field)


def test_a_policy_built_by_adjacent_only_matches_the_shared_constant() -> None:
    """One way of saying it, so a caller cannot invent a second."""
    assert tracking_module.adjacent_only() == ADJACENT_ONLY_POLICY
    assert _adjacent_only(max_keyframes=3).max_keyframes == 3


# --- 7. what the round-1 review found, asserted directly ------------------------------------
#
# Every test in this section exists because the complete round-1 implementation
# passed its own suite while a load-bearing claim was false. They are grouped
# rather than scattered so a reader can see the shape of what was wrong: three
# bounds that were descriptions rather than bounds, an "independent" reference
# that was not, an agreement that was not pairwise, and a rescue gate a whole
# class of rescues walked past.


def _raise_the_first_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the FIRST ICP solve of the next offered frame raise, as a real one does.

    The adjacent candidate is formed first (see
    :meth:`~scanner3d.tracking.IncrementalTracker._candidates`), so this is the
    adjacent pair refusing to fit at all - a singular flat panel is the textbook
    cause, and it is the exact failure this module was written about. Every
    later solve for the same frame runs normally.

    Patched at the solver rather than at the tracker on purpose: what is being
    tested is what the tracker does when a candidate cannot be FORMED, and
    stubbing the tracker's own candidate machinery would test the stub.
    """
    real = icp_point_to_plane
    calls: list[int] = []

    def first_one_raises(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise IcpPlaneGeometryUnusable(
                "the linearised system is singular in rotation z: these correspondences and "
                "their normals leave that direction of motion unconstrained"
            )
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking_module, "icp_point_to_plane", first_one_raises)


def test_a_model_only_survivor_is_a_rescue_when_the_adjacent_solve_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loophole round 1 left open, closed and asserted at its widest point.

    Round 1 decided whether a lone survivor needed checking from
    ``len(candidates) > 1`` - whether another candidate had been FORMED. That
    exempted precisely the case the rescue gates exist for: when the adjacent
    solve RAISES because the geometry is singular, no adjacent candidate exists,
    so the model candidate was the only one and sailed through with no rescue
    check at all. A singular flat panel is the textbook cause of that raise.

    Here the adjacent solve raises and the keyframe reference is switched off,
    so the model is the only candidate that exists - and the multi-viewpoint
    gate is set out of reach. The frame must be refused as an unsupported
    rescue, and nothing may move.
    """
    tracker = _rescue_ready(
        TrackingPolicy(keyframe_candidates=0, rescue_min_reference_frames=99, motion_prior=False)
    )
    before = _state(tracker)
    _raise_the_first_solve(monkeypatch)

    decision = tracker.offer(_sweep()[6])

    assert not decision.accepted
    assert decision.failure_type == TRACK_RESCUE_UNSUPPORTED_TYPE
    assert any("accepted frame(s), under the" in refusal for refusal in decision.refusals)
    assert _state(tracker) == before


def test_a_model_only_survivor_whose_reference_barely_overlaps_refuses_when_the_solve_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same loophole, closed against the other half of the rescue evidence.

    Removing the overlap evidence rather than the multi-viewpoint evidence has
    to refuse for the same reason: with the adjacent solve unable to run, the
    model reference is the only thing saying where this frame is, so the frame
    has to be substantially ON it.
    """
    tracker = _rescue_ready(
        TrackingPolicy(keyframe_candidates=0, rescue_min_overlap=0.999, motion_prior=False)
    )
    before = _state(tracker)
    _raise_the_first_solve(monkeypatch)

    decision = tracker.offer(_sweep()[6])

    assert not decision.accepted
    assert decision.failure_type == TRACK_RESCUE_UNSUPPORTED_TYPE
    assert any("overlaps the model reference by only" in refusal for refusal in decision.refusals)
    assert _state(tracker) == before


def test_a_supported_model_only_survivor_is_still_placed_when_the_adjacent_solve_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the gate is a gate rather than a wall.

    The same unformed-adjacent situation with the rescue evidence INTACT has to
    place the frame, or the two refusals above would be satisfied by a tracker
    that simply never rescues anything - which would remove the one mechanism
    this module has for holding a pose through a weak view.
    """
    tracker = _rescue_ready(TrackingPolicy(keyframe_candidates=0, motion_prior=False))
    _raise_the_first_solve(monkeypatch)

    decision = tracker.offer(_sweep()[6])

    assert decision.accepted, decision.message()
    assert decision.origin == MODEL_ORIGIN
    assert decision.rescue is True
    assert _rotation_error(decision.pose or Rigid.identity(), _true_pose(6)) < 1.5


def test_no_two_candidates_for_one_frame_are_built_from_the_same_accepted_frame() -> None:
    """ "Independent" has to mean disjoint provenance, and round 1's was not.

    Round 1 described the keyframe and model candidates as independent evidence
    while the model reference was assembled from the keyframes - the selected one
    included. A frame that agreed with itself therefore counted as two references
    agreeing, at the highest preference, and as one of the two sides of the
    disagreement gate that is supposed to catch exactly that.

    Asserted on the reference frames the candidates themselves report, over a
    whole sweep, rather than on one frame that happened to work.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    checked = 0
    for cloud in clouds:
        decision = tracker.offer(cloud)
        assert decision.accepted, decision.message()
        if len(decision.candidates) < 2:
            continue
        checked += 1
        seen: set[int] = set()
        for candidate in decision.candidates:
            frames = set(candidate.reference_frames)
            assert frames, candidate.origin
            assert not (frames & seen), (
                f"the {candidate.origin} candidate reuses accepted frame(s) "
                f"{sorted(frames & seen)} another candidate for the same frame is using"
            )
            seen |= frames
    assert checked > 10, "the sweep did not produce enough multi-candidate frames to prove this"


def _candidate_at(origin: str, pose: Rigid) -> tracking_module.TrackCandidate:
    """A surviving candidate placed exactly where a test says, and nowhere near ICP.

    The agreement rule is arithmetic over poses, so it is tested as arithmetic
    over poses. Persuading three real solves to land at three chosen distances
    from each other would be a test of the fixture.
    """
    return tracking_module.TrackCandidate(
        origin=origin,
        reference_frames=(0,),
        reference_points=100,
        source_points=100,
        pose=pose,
        edge=judge_local_edge(_fit(), frame=0),
        overlap=1.0,
        travel=0.0,
        accepted=True,
        refusals=(),
    )


def test_agreement_is_measured_between_every_pair_and_not_only_against_the_preferred_one() -> None:
    """The gap round 1's comparison left, stated as arithmetic.

    Round 1 measured every survivor against the PREFERRED candidate only. Three
    placements can each sit inside the gate of a middle one while the two
    extremes are nearly twice the gate apart - and "the references agreed" was
    then true of a set holding two different answers.

    The middle candidate below is the model, which is the preferred origin, so
    the round-1 comparison would have found nothing. The two extremes are on
    opposite sides of it, each just inside the gate, and therefore just under
    twice the gate from each other.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    points = _grid()
    # frame_travel of a pure translation is that translation's length, so these
    # offsets are the surface distances the gate is expressed in.
    step = gate * 0.9
    survivors = [
        _candidate_at(ADJACENT_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-step, 0.0, 0.0))),
        _candidate_at(MODEL_ORIGIN, Rigid.identity()),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (step, 0.0, 0.0))),
    ]

    against_preferred = tracking_module._disagreements([survivors[1]], points, gate)
    all_pairs = tracking_module._disagreements(survivors, points, gate)

    assert against_preferred == [], "the fixture must be one round 1 would have accepted"
    assert [(a.origin, b.origin) for a, b, _ in all_pairs] == [(ADJACENT_ORIGIN, KEYFRAME_ORIGIN)]
    assert all_pairs[0][2] > gate


def test_every_candidate_and_every_retained_cloud_stays_inside_the_point_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound round 1 advertised and did not keep, counted rather than argued.

    Round 1 exempted the adjacent candidate's source, its travel measurement and
    the retained cloud from the point budget, so a physical capture reached about
    18000 points against a nominal 900-point source budget and took 738.971
    seconds for 176 frames. A bound any candidate may opt out of is not a bound.

    Every ICP source and every ICP target is counted here as it is handed to the
    solver, and the retained clouds are counted as points rather than as clouds,
    because "32 clouds" bounds nothing if a cloud may be any size.
    """
    policy = TrackingPolicy()
    sizes: list[tuple[int, int]] = []
    real = icp_point_to_plane

    def counting(source: object, target: object, *args: object, **kwargs: object) -> object:
        sizes.append((len(cast(Sequence[Point], source)), len(cast(Sequence[Point], target))))
        return real(source, target, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking_module, "icp_point_to_plane", counting)

    # Deliberately far denser than the live preview's own budget, which is what
    # an offline reconstruction hands in and what round 1 passed straight
    # through to the solver.
    dense = [_grid(offset=index * 0.4, side=32, spacing=0.5) for index in range(12)]
    assert len(dense[0]) > 3 * policy.candidate_source_points
    tracker = IncrementalTracker(policy=policy)
    for cloud in dense:
        tracker.offer(cloud)

    assert sizes, "no solve ran, so this proves nothing"
    assert max(source for source, _ in sizes) <= policy.candidate_source_points
    assert max(target for _, target in sizes) <= max(
        policy.retained_cloud_points, policy.model_reference_points
    )
    assert tracker.retained_clouds <= policy.max_retained_clouds
    assert tracker.retained_points <= policy.max_retained_clouds * policy.retained_cloud_points
    assert tracker.retained_points < len(dense[0]) * tracker.frames


def test_a_frame_carrying_a_value_that_is_not_a_number_is_refused_before_anything_is_admitted() -> (
    None
):
    """An unusable cloud is refused at the door, not by the solver behind it.

    The FIRST accepted frame is not fitted at all - it is the anchor, placed at
    the identity - so there is no ICP gate between a non-finite coordinate and
    the accepted model. Everything a later frame is judged against is built from
    the accepted clouds, so a NaN admitted here is reported by some frame much
    later that has nothing wrong with it.
    """
    poisoned = ((0.0, 0.0, 100.0), (1.0, float("nan"), 100.0), (2.0, 0.0, 100.0))

    fresh = IncrementalTracker()
    anchor = fresh.offer(poisoned)
    assert not anchor.accepted
    assert anchor.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert fresh.frames == 0

    tracker = _placed_pair()
    before = _state(tracker)
    ordinary = tracker.offer(poisoned)
    assert not ordinary.accepted
    assert ordinary.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert _state(tracker) == before


def test_a_seed_that_is_not_finite_is_refused_rather_than_written_into_the_model() -> None:
    """The same check where it matters most, because a seed is never fitted.

    A seeded frame's pose was decided elsewhere and is taken on trust - that is
    the whole point of a recorded relocalization bridge. Trust is not the same as
    unchecked: a non-finite pose or cloud entering here is composed into every
    pose after it.
    """
    tracker = _placed_pair()
    before = _state(tracker)

    poisoned_cloud = tracker.seed(
        ((0.0, 0.0, 100.0), (1.0, float("inf"), 100.0), (2.0, 0.0, 100.0)), Rigid.identity()
    )
    assert not poisoned_cloud.accepted
    assert poisoned_cloud.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert _state(tracker) == before

    # Built through the dataclass constructor rather than ``Rigid.of``, because
    # ``Rigid.of`` validates and a pose that reached here would be one that had
    # not been through it - which is exactly the pose this check is for.
    poisoned_pose = tracker.seed(
        _grid(), Rigid(rotation=IDENTITY_MATRIX, translation=(float("nan"), 0.0, 0.0))
    )
    assert not poisoned_pose.accepted
    assert poisoned_pose.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert _state(tracker) == before


@pytest.mark.parametrize(
    "policy",
    [
        TrackingPolicy(),
        TrackingPolicy(max_keyframes=2, protected_recent_keyframes=1, max_retained_clouds=3),
        TrackingPolicy(max_keyframes=3, protected_recent_keyframes=0, max_retained_clouds=4),
        TrackingPolicy(max_reference_frames=1, model_reference_points=1),
        TrackingPolicy(retained_cloud_points=8, candidate_source_points=8),
        TrackingPolicy(model_recent_frames=0, keyframe_candidates=0),
    ],
)
def test_every_constructible_policy_keeps_every_hard_bound_over_a_long_sequence(
    policy: TrackingPolicy,
) -> None:
    """Not only the default one. A bound is a bound for every policy that exists.

    Round 1's validation admitted combinations that then broke a promise made
    elsewhere - an over-full keyframe set that could never be reduced, a
    retention budget that released the anchor, a reference naming frames whose
    points had been truncated away. Constructibility and the bounds are now the
    same statement, so this drives each surviving combination over a sequence
    long enough to reach eviction and release and checks all of it.
    """
    tracker = IncrementalTracker(policy=policy)
    for index in range(40):
        tracker.offer(_grid(offset=index * 0.35), adjacent_fit=_fit())

    # Not all forty. The stand-in adjacent fit says the frame did not move while
    # the real keyframe solve says it did, so once the nearest keyframe is far
    # enough behind, the two genuinely disagree and the frame is refused - which
    # is the tracker working. What matters here is that the sequence is long
    # enough to have reached eviction and cloud release under every policy,
    # which thirty-plus accepted frames is under all of them.
    assert tracker.frames >= 30
    assert len(tracker.keyframes) <= policy.max_keyframes
    assert tracker.keyframes[0] == 0, "the anchor defines the model frame and is never evicted"
    assert list(tracker.keyframes) == sorted(tracker.keyframes)
    assert tracker.retained_clouds <= policy.max_retained_clouds
    assert tracker.retained_points <= policy.max_retained_clouds * policy.retained_cloud_points
    record = tracker.record()
    assert record["retained_points"] == tracker.retained_points
    assert record["retained_points"] <= cast(int, record["max_retained_points"])


def test_two_references_that_agree_place_the_frame_and_report_that_they_agreed() -> None:
    """The ordinary case, and the distinction the record now has to carry.

    "Cleared its gates" and "said the same thing" are different statements, and
    round 1 reported the first under the second's name. An accepted decision now
    names the cluster that agreed, separately from every candidate that merely
    survived its own gates.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:5]:
        assert tracker.offer(cloud).accepted
    decision = tracker.offer(clouds[5])

    assert decision.accepted, decision.message()
    assert len(decision.agreeing) >= 2
    assert decision.origin in decision.agreeing
    assert decision.dissenting == ()
    assert set(decision.agreeing) <= set(decision.passing)
    assert "agreed by" in decision.message()


def test_one_dissenting_reference_is_outvoted_rather_than_given_a_veto() -> None:
    """Why a lone dissenter does not end a capture, measured on the real one.

    The adjacent pair is the reference this module exists because of, and a
    refusal makes it worse mechanically: it is seeded from the identity relative
    to the last ACCEPTED frame, so one skipped frame leaves it a whole step away
    from the answer - the condition :mod:`scanner3d.icp_plane` says its
    linearisation does not survive. On the owner's 2026-08-28 capture that turned
    one badly seeded solve into a terminal loss and a dead capture.

    Here the adjacent fit handed in is deliberately wrong by more than the
    agreement gate while the keyframe and the model, fitted for real, agree with
    each other. The frame is placed by the two that agree, the dissenter is
    NAMED rather than hidden, and the pose does not come from the dissenter.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:5]:
        assert tracker.offer(cloud).accepted
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    truth = _camera(4).compose(_camera(5).inverse())
    liar = truth.compose(Rigid.of(IDENTITY_MATRIX, (gate * 1.6, 0.0, 0.0)))

    decision = tracker.offer(clouds[5], adjacent_fit=_fit(transform=liar))

    assert decision.accepted, decision.message()
    assert set(decision.agreeing) == {KEYFRAME_ORIGIN, MODEL_ORIGIN}
    assert decision.dissenting == (ADJACENT_ORIGIN,)
    assert decision.origin != ADJACENT_ORIGIN
    assert _rotation_error(decision.pose or Rigid.identity(), _true_pose(5)) < 1.5


def test_survivors_that_split_into_two_equal_groups_are_refused_as_two_answers() -> None:
    """A tie is not a majority, and a majority is the whole of the rule above.

    Three placements where each of two pairs agrees internally and the two pairs
    disagree with each other is not one answer with a dissenter; it is two
    answers. There is no majority to outvote anything, and a residual cannot
    arbitrate, so the frame is refused.

    Asserted over the clustering itself, because persuading three real solves to
    land at three chosen distances from each other would be a test of the
    fixture rather than of the rule.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    points = _grid()
    step = gate * 0.9
    survivors = [
        _candidate_at(ADJACENT_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-step, 0.0, 0.0))),
        _candidate_at(MODEL_ORIGIN, Rigid.identity()),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (step, 0.0, 0.0))),
    ]

    clusters = tracking_module._agreement_clusters(survivors, points, gate)

    assert len(clusters) == 2, "the fixture must be a genuine tie"
    assert all(len(cluster) == 2 for cluster in clusters)
    assert {frozenset(c.origin for c in cluster) for cluster in clusters} == {
        frozenset({ADJACENT_ORIGIN, MODEL_ORIGIN}),
        frozenset({MODEL_ORIGIN, KEYFRAME_ORIGIN}),
    }

    tracker = IncrementalTracker(policy=TrackingPolicy(recovery_enabled=False))
    assert tracker.offer(points).accepted
    before = _state(tracker)
    decision = tracker._decide(1, points, points, tracker._accepted[-1], tuple(survivors))

    assert not decision.accepted
    assert decision.failure_type == TRACK_DISAGREEMENT_TYPE
    assert "only 1.00x apart" in decision.message()
    assert _state(tracker) == before


def test_a_distinctly_tighter_equal_size_group_places_the_frame() -> None:
    """A tight consensus is not tied with another pair merely inside the gate.

    This is the arithmetic shape of physical capture ``20260831-175352`` frame
    88: adjacent/keyframe were 1.620 apart, adjacent/model 4.760 apart, and
    keyframe/model 5.752 apart against a 5.200 gate. The old rule called the two
    size-two groups equal and refused; the refusal stranded the following two
    frames even though admitting either branch let both place.
    """
    points = _grid()
    survivors = (
        _candidate_at(ADJACENT_ORIGIN, Rigid.identity()),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-1.6, 0.0, 0.0))),
        _candidate_at(MODEL_ORIGIN, Rigid.of(IDENTITY_MATRIX, (4.7, 0.0, 0.0))),
    )
    tracker = IncrementalTracker(policy=TrackingPolicy(recovery_enabled=False))
    assert tracker.offer(points).accepted

    decision = tracker._decide(
        1,
        points,
        points,
        tracker._accepted[-1],
        survivors,
    )

    assert decision.accepted, decision.message()
    assert decision.origin == ADJACENT_ORIGIN
    assert decision.agreeing == (ADJACENT_ORIGIN, KEYFRAME_ORIGIN)
    assert decision.dissenting == (MODEL_ORIGIN,)
    assert decision.recovery is None


def test_a_failed_recovery_arbitration_keeps_the_tighter_consensus_fallback() -> None:
    """Stronger evidence gets first chance without turning its absence into loss.

    The tracker is deliberately too young to build two disjoint multi-frame
    recovery references. The same distinctly tighter ordinary pair therefore
    remains the placement, while the record says that recovery arbitration was
    considered and why it could not replace that pair.
    """
    points = _grid()
    survivors = (
        _candidate_at(ADJACENT_ORIGIN, Rigid.identity()),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-1.6, 0.0, 0.0))),
        _candidate_at(MODEL_ORIGIN, Rigid.of(IDENTITY_MATRIX, (4.7, 0.0, 0.0))),
    )
    tracker = IncrementalTracker()
    assert tracker.offer(points).accepted

    decision = tracker._decide(1, points, points, tracker._accepted[-1], survivors)

    assert decision.accepted, decision.message()
    assert decision.origin == ADJACENT_ORIGIN
    assert decision.agreeing == (ADJACENT_ORIGIN, KEYFRAME_ORIGIN)
    assert decision.dissenting == (MODEL_ORIGIN,)
    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert recovery.triggered_by == TRACK_DISAGREEMENT_TYPE
    assert recovery.work.coarse_searches == 0
    assert "bounded coarse arbitration ran first" in decision.message()


def test_nearly_equal_consensus_groups_remain_an_ambiguous_refusal() -> None:
    """The new rule has a margin; it is not permission to break every tie.

    The pairs here span 4.5 and 5.0, only 1.11x apart against the shared 1.15x
    ambiguity margin. Both are plausible answers, so the tracker still refuses
    and preserves all accepted state.
    """
    points = _grid()
    survivors = (
        _candidate_at(ADJACENT_ORIGIN, Rigid.identity()),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-4.5, 0.0, 0.0))),
        _candidate_at(MODEL_ORIGIN, Rigid.of(IDENTITY_MATRIX, (5.0, 0.0, 0.0))),
    )
    tracker = IncrementalTracker(policy=TrackingPolicy(recovery_enabled=False))
    assert tracker.offer(points).accepted
    before = _state(tracker)

    decision = tracker._decide(
        1,
        points,
        points,
        tracker._accepted[-1],
        survivors,
    )

    assert not decision.accepted
    assert decision.failure_type == TRACK_DISAGREEMENT_TYPE
    assert "only 1.11x apart" in decision.message()
    assert decision.agreeing == ()
    assert _state(tracker) == before


def test_an_agreement_cluster_requires_every_pair_inside_it_to_agree() -> None:
    """Mutual, not merely "each near the leader". The round-1 reading, refused.

    Two placements nearly twice the gate apart, each just inside the gate of a
    third, are not an agreement. Round 1 compared every survivor against the
    preferred candidate only and would have called this set unanimous; here the
    largest mutually-agreeing set is a PAIR, and since two such pairs exist the
    decision above refuses the frame.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    points = _grid()
    step = gate * 0.9
    extremes = (
        _candidate_at(ADJACENT_ORIGIN, Rigid.of(IDENTITY_MATRIX, (-step, 0.0, 0.0))),
        _candidate_at(KEYFRAME_ORIGIN, Rigid.of(IDENTITY_MATRIX, (step, 0.0, 0.0))),
    )
    middle = _candidate_at(MODEL_ORIGIN, Rigid.identity())

    assert tracking_module._agreement_clusters(list(extremes), points, gate) == [
        (extremes[0],),
        (extremes[1],),
    ]
    assert tracking_module._agreement_clusters([extremes[0], middle], points, gate) == [
        (extremes[0], middle)
    ]


def test_the_majority_rule_needs_a_majority_and_two_survivors_do_not_make_one() -> None:
    """The boundary of the rule, stated where the rule is.

    "At most one dissenter" is not "the biggest group wins". With two survivors
    that disagree, the biggest mutually-agreeing group is ONE candidate and the
    other half of the evidence contradicts it - which is the ambiguity this
    module refuses, not a minority to outvote.
    :func:`test_candidates_that_disagree_refuse_instead_of_choosing_the_smaller_residual`
    is the same rule reached through real geometry; this states the arithmetic
    it rests on.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    points = _grid()
    apart = (
        _candidate_at(ADJACENT_ORIGIN, Rigid.identity()),
        _candidate_at(MODEL_ORIGIN, Rigid.of(IDENTITY_MATRIX, (gate * 1.8, 0.0, 0.0))),
    )

    clusters = tracking_module._agreement_clusters(list(apart), points, gate)

    assert [len(cluster) for cluster in clusters] == [1, 1]
    assert len(clusters) > 1, "two singleton clusters is a tie, and a tie has no majority"


# --- 8. what the round-2 review found, asserted directly ------------------------------------


#: Clouds that are the right SHAPE for a cloud and are not one.
#:
#: Every entry has three points, so the "too few points" gate cannot be what
#: refuses them, and each carries one thing a
#: :data:`~scanner3d.geometry.Point` is not. The type checker says these cannot
#: happen; a replay reading a file, a Qt callback and a test double all say
#: otherwise, and :meth:`~scanner3d.tracking.IncrementalTracker.offer` promises
#: never to raise for ordinary data.
_MALFORMED_CLOUDS: Sequence[tuple[str, object]] = (
    ("two coordinates", ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
    ("four coordinates", ((0.0, 0.0, 100.0, 1.0), (1.0, 0.0, 100.0), (2.0, 0.0, 100.0))),
    ("a string coordinate", ((0.0, 0.0, 100.0), (1.0, "0.0", 100.0), (2.0, 0.0, 100.0))),
    ("a None coordinate", ((0.0, 0.0, 100.0), (1.0, None, 100.0), (2.0, 0.0, 100.0))),
    ("a boolean coordinate", ((0.0, 0.0, 100.0), (True, 0.0, 100.0), (2.0, 0.0, 100.0))),
    ("a complex coordinate", ((0.0, 0.0, 100.0), (complex(1.0, 2.0), 0.0, 100.0), (2.0, 0.0, 1.0))),
    ("a point that is a number", ((0.0, 0.0, 100.0), 1.0, (2.0, 0.0, 100.0))),
    ("a point that is a string", ((0.0, 0.0, 100.0), "abc", (2.0, 0.0, 100.0))),
    ("a point that is a dict", ((0.0, 0.0, 100.0), {"x": 1.0}, (2.0, 0.0, 100.0))),
    ("a nested non-finite", ((0.0, 0.0, 100.0), (1.0, float("nan"), 100.0), (2.0, 0.0, 100.0))),
)


@pytest.mark.parametrize(
    "cloud", [pytest.param(cloud, id=name) for name, cloud in _MALFORMED_CLOUDS]
)
def test_a_malformed_cloud_is_refused_wherever_it_arrives_and_never_raises(
    cloud: object,
) -> None:
    """The cloud check is a TOTAL validator, on all three doors into the model.

    Round 1 asked ``math.isfinite`` of whatever iterating a point produced,
    which answers the question only for input that was already well formed: a
    two- or four-coordinate entry passed the check and entered the accepted
    model, and a string or ``None`` coordinate raised ``TypeError`` out of
    ``offer`` - the one method whose contract is that it never raises for
    ordinary data - so a live capture saw a crash where it should have seen a
    refusal it could publish.

    All three doors, because they are three: the ANCHOR is admitted without
    being fitted, a LATER offer is admitted after a fit that would itself have
    raised, and a SEED is admitted on a recorded pose and never fitted at all.
    """
    points = cast(Sequence[Point], cloud)

    fresh = IncrementalTracker()
    anchor = fresh.offer(points)
    assert not anchor.accepted, "a malformed cloud must not become the model frame"
    assert anchor.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert fresh.frames == 0
    assert fresh.keyframes == ()

    tracker = _placed_pair()
    before = _state(tracker)
    ordinary = tracker.offer(points)
    assert not ordinary.accepted
    assert ordinary.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert ordinary.pose is None
    assert _state(tracker) == before

    seeded = tracker.seed(points, Rigid.identity())
    assert not seeded.accepted
    assert seeded.failure_type == tracking_module.TRACK_FRAME_UNUSABLE_TYPE
    assert seeded.pose is None
    assert _state(tracker) == before


def test_a_cloud_of_lists_of_three_real_numbers_is_still_a_cloud() -> None:
    """The validator refuses malformed input, not merely unfamiliar input.

    A total check written carelessly refuses everything it was not written
    against, which would turn any caller building points as lists - or as whole
    integers, which are real numbers - into a capture that never places a frame.
    Stated here so the refusals above cannot be satisfied by a check that says
    no to everything.
    """
    tracker = IncrementalTracker()
    listed = [list(point) for point in _grid()]

    decision = tracker.offer(cast(Sequence[Point], listed))

    assert decision.accepted, decision.message()
    assert tracker.frames == 1

    integral = ((0, 0, 100), (2, 0, 100), (0, 2, 100), (2, 2, 100))
    assert tracker.seed(cast(Sequence[Point], integral), Rigid.identity()).accepted


def test_agreeing_is_the_selected_cluster_and_passing_is_every_gate_clearer() -> None:
    """The two statements the record must not conflate, in one decision.

    Round 1 published "every candidate that cleared its own gates" under the
    name ``agreeing``, so a reader - and a durable JSON record - was told three
    references had agreed when what had happened was that three references had
    each been individually plausible and two of them had contradicted the third.
    ``agreeing`` is now the cluster that placed the frame; the gate-clearing set
    is still published, under ``passing``, because "the adjacent fit was fine
    and disagreed anyway" is evidence about the capture.
    """
    tracker = IncrementalTracker()
    clouds = _sweep()
    for cloud in clouds[:5]:
        assert tracker.offer(cloud).accepted
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    truth = _camera(4).compose(_camera(5).inverse())
    liar = truth.compose(Rigid.of(IDENTITY_MATRIX, (gate * 1.6, 0.0, 0.0)))

    decision = tracker.offer(clouds[5], adjacent_fit=_fit(transform=liar))

    assert decision.accepted, decision.message()
    assert set(decision.agreeing) == {KEYFRAME_ORIGIN, MODEL_ORIGIN}
    assert decision.dissenting == (ADJACENT_ORIGIN,)
    assert ADJACENT_ORIGIN in decision.passing, "it cleared its own gates and then disagreed"
    assert set(decision.agreeing) < set(decision.passing)
    assert decision.origin in decision.agreeing

    record = decision.as_dict()
    assert record["agreeing"] == list(decision.agreeing)
    assert record["passing"] == list(decision.passing)
    assert record["dissenting"] == [ADJACENT_ORIGIN]
    assert json.loads(json.dumps(record, allow_nan=False))["agreeing"] == list(decision.agreeing)


def test_a_refused_frame_reports_no_agreement_however_many_candidates_passed() -> None:
    """The other half of the same distinction, where it is load-bearing.

    A disagreement is exactly the case where the gate-clearing set is large and
    the agreement is empty. Reporting the first as the second is how a refusal
    could read, in a durable record, as though references had agreed about a
    frame no pose was ever published for.
    """
    tracker = IncrementalTracker()
    for offset in (0.0, 0.5, 1.0, 1.5):
        assert tracker.offer(_grid(offset=offset), adjacent_fit=_fit()).accepted
    before = _state(tracker)

    decision = tracker.offer(
        _grid(offset=2.0), adjacent_fit=_fit(transform=Rigid.of(IDENTITY_MATRIX, (10.0, 0.0, 0.0)))
    )

    assert not decision.accepted
    assert decision.failure_type == TRACK_DISAGREEMENT_TYPE
    assert decision.agreeing == ()
    assert len(decision.passing) >= 2
    assert decision.as_dict()["agreeing"] == []
    assert _state(tracker) == before


def test_every_accepted_decision_names_its_support_the_same_way() -> None:
    """Anchor, seed and single reference report support like every other placement.

    A reader of the record should not have to know which door a frame came
    through to find out what supported it. An accepted decision always names a
    non-empty ``agreeing`` containing its own ``origin`` - the anchor names the
    anchor, a seeded bridge frame names the bridge, and a placement with one
    surviving reference names that reference.
    """
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)

    anchor = tracker.offer(_grid())
    assert anchor.accepted
    assert anchor.agreeing == (ANCHOR_ORIGIN,)
    assert anchor.origin in anchor.agreeing

    single = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit())
    assert single.accepted
    assert single.agreeing == (ADJACENT_ORIGIN,)
    assert single.dissenting == ()

    seeded = tracker.seed(_grid(offset=1.0), Rigid.identity(), anchor=True)
    assert seeded.accepted
    assert seeded.agreeing == (BRIDGE_ORIGIN,)
    assert seeded.origin in seeded.agreeing

    for decision in (anchor, single, seeded):
        assert decision.agreeing, "an accepted placement always names what supported it"
        assert decision.origin in decision.agreeing
        assert decision.as_dict()["agreeing"] == list(decision.agreeing)


# --- 9. the motion prior: where a refinement is started from --------------------------------

#: The step every prediction below is extrapolated from.
#:
#: Chosen so the three bounds around a prediction can be told apart. Well past
#: the stationary band, so a prediction is formed at all; small enough that
#: FOUR of it is still inside the travel gate, so the step ceiling is what stops
#: the extrapolation in the test for the step ceiling rather than the travel
#: gate silently doing it first.
_STEP = Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 3.0))


def _moved_pair(step: Rigid, policy: TrackingPolicy = ADJACENT_ONLY_POLICY) -> IncrementalTracker:
    """A tracker holding two accepted frames a stated step apart.

    The step is the point. :func:`_placed_pair` builds its pair at the identity,
    which is a scanner that has not moved and therefore has no prediction to
    make - so it is the right fixture for everything else here and the wrong one
    for this.
    """
    tracker = IncrementalTracker(policy=policy)
    assert tracker.offer(_grid()).accepted
    assert tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=step)).accepted
    return tracker


def _recording_solver(monkeypatch: pytest.MonkeyPatch, answer: object) -> list[Rigid]:
    """Replace ICP with a stand-in that records the seed it was handed.

    A seed is not a measurement, so it cannot be read off a decision - and it is
    the entire subject here. Recording it at the solver rather than stubbing the
    tracker's own candidate machinery is the same choice
    :func:`_raise_the_first_solve` makes, and for the same reason.

    ``answer`` is asked for a fit per seed, so a test can say "the identity
    attempt finds nothing and the predicted one lands" without persuading real
    ICP to do that on demand.
    """
    seeds: list[Rigid] = []

    def solve(source: object, target: object, *, initial: Rigid, **kwargs: object) -> object:
        seeds.append(initial)
        return answer(initial)  # type: ignore[operator]

    monkeypatch.setattr(tracking_module, "icp_point_to_plane", solve)
    return seeds


def test_an_adjacent_candidate_no_placement_came_out_of_is_retried_from_the_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The change this section exists for, at its smallest.

    The first attempt is seeded from the identity - a statement that the scanner
    has not moved since the last accepted frame - and finds nothing. The second
    is seeded from the step the scanner was last measured making, and places the
    frame. No gate differs between the two attempts; only where the solve was
    started, and the record says which one reached the placement.
    """
    tracker = _moved_pair(_STEP)
    seeds = _recording_solver(
        monkeypatch,
        lambda initial: _fit(rmse=99.0) if initial == Rigid.identity() else _fit(transform=_STEP),
    )

    decision = tracker.offer(_grid(offset=1.0))

    assert seeds == [Rigid.identity(), _STEP]
    assert decision.accepted, decision.message()
    assert decision.origin == ADJACENT_ORIGIN
    assert [candidate.seed for candidate in decision.candidates] == [tracking_module.PRIOR_SEED]


def test_an_adjacent_placement_that_worked_first_time_pays_for_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry is a ceiling, not a cost.

    A long capture places almost every frame at the first attempt, and it must
    not start paying for a second solve on all of them to buy the few it
    rescues.
    """
    tracker = _moved_pair(_STEP)
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(transform=_STEP))

    decision = tracker.offer(_grid(offset=1.0))

    assert seeds == [Rigid.identity()]
    assert decision.accepted, decision.message()
    assert decision.candidates[0].seed == tracking_module.IDENTITY_SEED


def test_a_scanner_that_has_not_moved_makes_no_prediction_and_pays_for_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prediction inside the stationary band is the seed the first attempt used.

    Retrying there would re-run that solve and reach that answer again, so the
    refusal stands on one solve rather than two.

    The band is about the STEP BETWEEN FRAMES, not about the scanner being
    bolted down, and an earlier draft of this docstring got that wrong by
    offering a turntable as the example. A turntable is the opposite case: the
    scanner is fixed and the object turns, so consecutive frames differ by the
    whole rotation step and a prediction is formed exactly as it is for a
    carried scanner. Measured on the owner's 20260901-145316 turntable capture,
    the tracker's own travel runs 1.77 to 10.60 with a median of 3.38 - the
    middle of the guidance band, nowhere near stationary. What this test
    describes is a scanner pointed at something that is not moving at all.
    """
    tracker = _placed_pair()
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))

    decision = tracker.offer(_grid(offset=1.0))

    assert seeds == [Rigid.identity()]
    assert not decision.accepted


def test_the_prediction_is_extrapolated_across_the_frames_that_were_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused frame is invisible to this module's own numbering.

    ``offer`` numbers frames by ACCEPTANCE, so the frame after a refusal sits
    two delivery steps from the reference it is seeded against - which is the
    mechanism docs/tracking.md names when it says one refused frame leaves the
    adjacent candidate a whole skipped step away from the answer. The prediction
    is composed once per step for exactly that reason, and a prediction that
    ignored the gap would under-shoot by a whole step per refusal at precisely
    the moment the seed matters most.
    """
    tracker = _moved_pair(_STEP)
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))

    assert not tracker.offer(_grid(offset=1.0)).accepted
    assert seeds == [Rigid.identity(), _STEP]

    seeds.clear()
    assert not tracker.offer(_grid(offset=1.5)).accepted
    assert seeds == [Rigid.identity(), _STEP.compose(_STEP)]


def test_the_extrapolation_stops_at_the_step_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the ceiling the prediction is multiplying an error, not carrying a measurement.

    Every step of an extrapolation multiplies whatever the previous one got
    wrong, and a scanner that has been unplaceable this long is in the
    tracking-loss streak's own territory. The prediction is withheld there
    rather than made more confidently.
    """
    tracker = _moved_pair(_STEP)
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))

    for step in range(1, tracker.policy.max_prior_steps + 1):
        seeds.clear()
        assert not tracker.offer(_grid(offset=1.0)).accepted
        extrapolated = _STEP
        for _ in range(step - 1):
            extrapolated = extrapolated.compose(_STEP)
        assert seeds == [Rigid.identity(), extrapolated], f"at {step} step(s)"

    seeds.clear()
    assert not tracker.offer(_grid(offset=1.0)).accepted
    assert seeds == [Rigid.identity()]


def test_a_prediction_past_one_travel_gate_is_now_formed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The travel gate is judged against the prediction, so the prediction may cross it.

    Two delivery steps of a step that is inside the gate extrapolate to a
    surface travel past it. That used to withhold the prediction, because a
    fit near it would have been refused as a jump; it is now made, because a
    fit near it is a steady fast step. The runaway cap three gates out is
    tested beside the gate itself, in section 10.
    """
    gate = TrackingPolicy().max_travel_multiple * NOISE_FLOOR
    step = Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 9.0))
    grid = _grid()
    assert frame_travel(grid, step) < gate < frame_travel(grid, step.compose(step))

    tracker = _moved_pair(step)
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))

    assert not tracker.offer(_grid(offset=1.0)).accepted
    assert seeds == [Rigid.identity(), step], "one step is inside the gate"

    seeds.clear()
    assert not tracker.offer(_grid(offset=1.5)).accepted
    assert seeds == [Rigid.identity(), step.compose(step)], "two steps cross it, and are tried"


def test_a_seeded_bridge_frame_establishes_no_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transform across a relocalization seam is not a step the scanner made.

    A bridge is where a capture found the object again after losing it. The
    frames either side of it are explicitly NOT a small step apart - that is the
    whole reason the boundary is recorded rather than fitted - so extrapolating
    it as though it were one frame's motion would seed the next solve from
    somewhere the scanner was never travelling towards.
    """
    tracker = _moved_pair(_STEP)
    tracker.seed(_grid(offset=1.0), tracker.poses[-1].compose(_STEP), anchor=False)
    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))

    assert not tracker.offer(_grid(offset=1.5)).accepted
    assert seeds == [Rigid.identity()]


def test_the_other_references_are_seeded_from_the_prediction_and_not_the_previous_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half of this that costs nothing and was never paid for.

    When the adjacent solve has produced no accepted placement, the keyframe and
    model refinements used to be seeded from ``previous.pose`` - a statement that
    the scanner did not move at all, made on exactly the frames where it had
    already failed to be placed once. They are now seeded from where the last
    measured motion says the frame is: the same solve, started somewhere the
    scanner might actually be, for no extra solves at all.
    """
    policy = TrackingPolicy(use_model=False, recovery_enabled=False)
    tracker = IncrementalTracker(policy=policy)
    assert tracker.offer(_grid()).accepted

    _recording_solver(monkeypatch, lambda initial: _fit(transform=_STEP))
    assert tracker.offer(_grid(offset=0.5)).accepted
    previous = tracker.poses[-1]
    keyframe = tracker.poses[tracker.keyframes[0]]

    seeds = _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))
    assert not tracker.offer(_grid(offset=1.0)).accepted

    stale = keyframe.inverse().compose(previous)
    predicted = keyframe.inverse().compose(previous.compose(_STEP))
    assert predicted != stale, "the fixture has to be able to tell the two apart"
    assert seeds == [Rigid.identity(), _STEP, predicted]


def test_a_frame_every_reference_places_first_time_costs_what_it_always_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_candidates`` went up by one; what an ordinary frame costs did not.

    Counted rather than argued, exactly as the ceiling above it is, because
    "the retry only runs where it is needed" is a claim about every frame of a
    long capture and not about the one frame it rescues.
    """
    calls: list[int] = []
    real = icp_point_to_plane

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking_module, "icp_point_to_plane", counting)
    tracker = IncrementalTracker()

    for cloud in _sweep()[:6]:
        before = len(calls)
        assert tracker.offer(cloud).accepted
        assert len(calls) - before <= tracker.policy.ordinary_candidates


#: Degrees between frames of the sweep the prediction is measured on.
#:
#: Deliberately far past :data:`_STEP_DEGREES`. At the ordinary step the
#: identity seed is already close enough that the prediction changes nothing at
#: all, which is the right behaviour and proves nothing - measured on this
#: scene, the two policies place all 12 frames identically at 7, 14 and 18
#: degrees. This is the speed a carried scanner actually reaches: median surface
#: travel of about 9.6 depth units per frame, past the capture guidance ceiling
#: of :data:`~scanner3d.pipeline.TRAVEL_MAX_MULTIPLE` noise floors and still
#: well inside the tracker's own jump gate, so nothing here is measuring the
#: travel refusal by accident.
_PREDICTED_STEP_DEGREES = 30.0

#: Frames of it. Long enough to reach the part of the sweep where the two
#: policies diverge - measured, that is frame 7 - and to carry on for another
#: eight afterwards, because "it placed one more frame" and "it placed the rest
#: of the sweep" are different claims.
_PREDICTED_SWEEP_FRAMES = 16


#: How much worse the frames only the prediction reaches are allowed to be than
#: the ones both policies place. Measured on this scene: 2.32 degrees worst
#: across all 16 against 2.13 worst across the 7 the identity seed manages, a
#: ratio of 1.09. Set wider than that because the point being asserted is "not
#: materially worse", and a ratio this close to 1 would otherwise be a test of
#: the fixture's arithmetic.
_PREDICTED_ERROR_ALLOWANCE = 1.5


def _fast_sweep() -> tuple[tuple[Point, ...], ...]:
    """A camera moving too fast for an identity seed, at a stated step."""
    return tuple(
        _view(_camera(index, _PREDICTED_STEP_DEGREES)) for index in range(_PREDICTED_SWEEP_FRAMES)
    )


def _placed(policy: TrackingPolicy) -> list[float]:
    """Rotation error per placed frame, stopping where the tracker refuses."""
    tracker = IncrementalTracker(policy=policy)
    errors: list[float] = []
    for index, cloud in enumerate(_fast_sweep()):
        decision = tracker.offer(cloud)
        if not decision.accepted:
            break
        truth = _camera(0, _PREDICTED_STEP_DEGREES).compose(
            _camera(index, _PREDICTED_STEP_DEGREES).inverse()
        )
        errors.append(_rotation_error(decision.pose or Rigid.identity(), truth))
    return errors


def test_a_carried_scanner_places_frames_the_identity_seed_alone_cannot() -> None:
    """The geometric claim, measured both ways over the identical clouds.

    The only difference between the two runs is where a refused adjacent solve
    is allowed to be tried again from. Measured on this scene at
    :data:`_PREDICTED_STEP_DEGREES`: the prediction places all 16 frames with a
    worst error of 2.32 degrees, and without it the sweep refuses at frame 7
    having placed 7 - at the identical errors, because up to that point the two
    runs are the same run.

    The accuracy assertion is the one that matters. A tracker that placed every
    frame by lowering a gate would satisfy the count on its own, so the extra
    nine frames have to be placed about as well as the seven that never needed
    the prediction.
    """
    with_prediction = _placed(TrackingPolicy())
    without = _placed(TrackingPolicy(motion_prior=False))

    assert len(without) < _PREDICTED_SWEEP_FRAMES, (
        "the identity seed placed the whole sweep, so this speed proves nothing"
    )
    assert len(with_prediction) == _PREDICTED_SWEEP_FRAMES
    assert with_prediction[: len(without)] == without, (
        "the two runs must be the same run until the first solve the identity seed loses"
    )
    assert max(with_prediction) < max(without) * _PREDICTED_ERROR_ALLOWANCE


# --- 10. the travel gate, judged against the prediction ---------------------------------------


def _translation(z: float) -> Rigid:
    return Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, z))


def _travel_gate() -> float:
    return ADJACENT_ONLY_POLICY.max_travel_multiple * NOISE_FLOOR


def test_a_steady_step_past_the_travel_gate_is_a_step_when_the_prediction_says_so() -> None:
    """A scanner carried steadily moves the surface far between frames and hardly
    at all from its own prediction; that is a step, however long, not a jump.

    Measured on a real turntable capture replayed at 23 and 34 degrees per
    frame: ICP placed every frame and the gate refused each one at 15.7-24.4
    against 15.6, then the next frame was two steps away and the capture was
    lost. The same gate ended the hand-carried capture at frame 70 at 20.662.
    """
    gate = _travel_gate()
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    # Two steps inside the gate establish a velocity.
    for _ in range(2):
        decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(10.0)))
        assert decision.accepted, decision.message()
    # Then a step past the gate, but only a little past the prediction.
    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(20.0)))
    assert decision.accepted, decision.message()
    (candidate,) = [c for c in decision.candidates if c.origin == ADJACENT_ORIGIN]
    assert candidate.travel == pytest.approx(20.0, abs=1e-6)
    assert candidate.travel > gate
    assert candidate.travel_from_prediction == pytest.approx(10.0, abs=1e-6)
    assert TRAVEL_GATE not in candidate.failed_gates
    recorded = cast(list[dict[str, object]], decision.as_dict()["candidates"])
    assert recorded[0]["travel_from_prediction"] == pytest.approx(10.0)
    # And the next one, steady at that speed, deviates from its prediction by nothing.
    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(20.0)))
    assert decision.accepted, decision.message()
    (candidate,) = [c for c in decision.candidates if c.origin == ADJACENT_ORIGIN]
    assert candidate.travel_from_prediction == pytest.approx(0.0, abs=1e-6)


def test_a_jump_away_from_the_prediction_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate keeps its meaning: far from the last frame AND far from the prediction.

    The solver is stubbed for the retry the refusal triggers, as every test of
    a refusal here does: on a periodic lattice real ICP seeded from the
    prediction can settle on a false small step, and this test is about the
    gate, not about ICP.
    """
    gate = _travel_gate()
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    for _ in range(2):
        assert tracker.offer(
            _grid(offset=0.5), adjacent_fit=_fit(transform=_translation(10.0))
        ).accepted
    _recording_solver(monkeypatch, lambda initial: _fit(rmse=99.0))
    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(-20.0)))
    assert not decision.accepted
    (candidate,) = [c for c in decision.candidates if c.origin == ADJACENT_ORIGIN]
    assert TRAVEL_GATE in candidate.failed_gates
    assert candidate.travel == pytest.approx(20.0, abs=1e-6)
    assert candidate.travel_from_prediction is not None
    assert candidate.travel_from_prediction == pytest.approx(30.0, abs=1e-6)
    assert candidate.travel_from_prediction > gate
    assert any("from where the last accepted motion predicted it" in r for r in decision.refusals)


def test_a_first_step_past_the_gate_with_no_prediction_is_still_a_jump() -> None:
    """Without a measured velocity there is nothing to judge a long step against."""
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(20.0)))
    assert not decision.accepted
    (candidate,) = [c for c in decision.candidates if c.origin == ADJACENT_ORIGIN]
    assert TRAVEL_GATE in candidate.failed_gates
    assert candidate.travel_from_prediction is None
    assert any("no accepted motion predicts a step that size" in r for r in decision.refusals)


def test_a_prediction_is_formed_past_one_gate_but_not_past_three() -> None:
    """The prediction may exceed the gate it is now judged against, up to the runaway cap."""
    gate = _travel_gate()
    tracker = IncrementalTracker(policy=ADJACENT_ONLY_POLICY)
    assert tracker.offer(_grid()).accepted
    for z in (10.0, 20.0, 30.0):
        decision = tracker.offer(_grid(offset=0.5), adjacent_fit=_fit(transform=_translation(z)))
        assert decision.accepted, decision.message()
    # The last accepted step moves the surface 30, nearly two gates: one delivery
    # step of it is a prediction; three of it, 90, is past the cap and is not.
    tracker._offers_since_accepted = 1
    prior = tracker._motion_prior(_grid(offset=0.5))
    assert prior is not None
    assert frame_travel(_grid(offset=0.5), prior) == pytest.approx(30.0, abs=1e-6)
    assert frame_travel(_grid(offset=0.5), prior) > gate
    tracker._offers_since_accepted = 3
    assert tracker._motion_prior(_grid(offset=0.5)) is None
    assert tracking_module.MAX_PRIOR_TRAVEL_MULTIPLE * gate < 3 * 30.0
