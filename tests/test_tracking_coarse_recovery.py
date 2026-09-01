"""What the bounded coarse recovery in :mod:`scanner3d.tracking` will not do.

The recovery exists so that a frame the ordinary references could not agree on
is re-seeded from a shape-level search instead of ending a capture. That makes
it the one place in this module where a pose is produced with NO ordinary
reference agreeing with it, so almost everything worth asserting here is a
refusal: which evidence it insists on, which evidence it refuses to substitute
for, and that refusing moves nothing.

There is one acceptance test, and it is load-bearing for all the rest. Every
refusal below would be satisfied by a tracker that simply never recovered
anything, so the gates are only gates if the same fixture, unmodified, does
recover. It is written first for that reason.

The scene is small and deliberately so. What the recovery needs from a fixture
is one thing - an object whose shape distinguishes its own placements at coarse
resolution - and the tall lobe off the slab's corner is that and nothing else.
The expensive drift comparison belongs to ``tests/test_tracking.py``; this file
would not learn anything from paying for it again.

Reaching the recovery
---------------------
Every test here reaches it the same way: ``_candidates`` is monkeypatched to
return nothing for exactly one offered frame. That is the ordinary
``tracking_no_candidate`` refusal - the first of the three the recovery may be
offered - reached deliberately rather than by arranging a scene in which real
ICP happens to refuse three references at once. The frame, the accepted model,
the partitions, the coarse search and every ICP solve after that point are all
real; what is stubbed is only the door the frame comes through.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import replace

import pytest

from scanner3d.capture_sequence import TRACKING_LOSS_STREAK
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, dot, rotation_about_axis
from scanner3d.multiview import AMBIGUITY_MARGIN
from scanner3d.multiview import coarse_align as _coarse_align
from scanner3d.pipeline import NOISE_FLOOR, frame_travel, reconstruct_sweep
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.simulate import Box, points_of, render
from scanner3d.tracking import (
    AGREEMENT_RESIDUAL_MULTIPLE,
    ANCHOR_ORIGIN,
    COARSE_RANKED_POSES,
    PREFERENCE_ORDER,
    RECOVERY_CONFIRM_ORIGIN,
    RECOVERY_MIN_PARTITION_POSE_SPAN,
    RECOVERY_ORIGIN,
    RECOVERY_PARTITION_FRAMES,
    RECOVERY_PARTITIONS,
    TRACK_DISAGREEMENT_TYPE,
    TRACK_NO_CANDIDATE_TYPE,
    IncrementalTracker,
    TrackCandidate,
    TrackDecision,
    TrackingNotAllowed,
    TrackingPolicy,
    _Agreement,
    _disagreement,
)

# --- the scene ------------------------------------------------------------------------

#: Standoff of the camera from the object's centre, in depth units.
_STANDOFF = 150.0

#: A slab with a TALL lobe off one corner, and the height of that lobe is the
#: whole fixture. A coarse ranking compares whole-object orientations, so an
#: object that looks much the same a quarter turn round offers several equally
#: good placements - which this module refuses, correctly, and which would make
#: every test here a test of the ambiguity refusal. Measured on this scene: the
#: best coarse placement covers the recovery reference 0.53 and the best
#: genuinely different one 0.32, a 1.68x margin.
_SLAB = Box(half_extents=(30.0, 12.0, 6.0))
_LOBE = Box(half_extents=(5.0, 18.0, 5.0), pose=Rigid.of(IDENTITY_MATRIX, (26.0, 22.0, 0.0)))

_OBJECT_VOXEL = 1.6
_FRAME_VOXEL_FRACTION = 1.0 / 26.0
_STEP_DEGREES = 7.0

#: Frames rendered: ten to build a model two disjoint four-frame partitions can
#: be dealt from, and one more to offer the recovery.
_SWEEP_FRAMES = 11

#: The frame every test offers to the recovery, and the frames accepted before
#: it. Named rather than written as ``-1`` so a reader can see that the offered
#: frame is not one of the accepted ones.
_ACCEPTED_FRAMES = 10
_OFFERED_FRAME = 10

#: How far the recovered pose may sit from the truth, in degrees. Measured on
#: this scene: 0.42, which is the same order as the ordinary placements around
#: it (0.2 to 0.5). Set wide enough to fail on a regression rather than on a
#: linear solve's last decimal.
_MAX_RECOVERED_ERROR_DEGREES = 1.5

#: Camera steps beyond the offered frame that make a frame a JUMP rather than a
#: frame. Seven, measured: at three and five steps the recovery still places the
#: frame and should, at seven the best-covering hypothesis refines to a
#: placement 20.1 from the last accepted pose and the travel gate refuses it
#: while both ambiguity verdicts still pass - which is what makes the travel
#: test below a test of the travel gate rather than of the ranking.
_TRAVEL_JUMP_STEPS = 7


@functools.cache
def _object_of(shapes: tuple[Box, ...]) -> tuple[Point, ...]:
    """``shapes`` as one cloud in their own frame, covering every side."""
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
        for shape in shapes:
            merged.extend(pose.inverse().apply_all(tuple(points_of(render([shape], pose=pose)))))
    return voxel_downsample(tuple(merged), _OBJECT_VOXEL)


def _object_points() -> tuple[Point, ...]:
    """The asymmetric object every test defaults to: the slab and its lobe."""
    return _object_of((_SLAB, _LOBE))


def _camera(index: int) -> Rigid:
    """The pose carrying object coordinates into the camera's, at one step."""
    orbit = rotation_about_axis((0.0, 1.0, 0.0), math.radians(_STEP_DEGREES * index))
    return Rigid.of(orbit.rotation, (0.0, 0.0, _STANDOFF))


def _view_of(
    object_points: tuple[Point, ...],
    pose: Rigid,
    *,
    fraction: float = _FRAME_VOXEL_FRACTION,
) -> tuple[Point, ...]:
    """The half of ``object_points`` facing a camera at ``pose``, in its coordinates."""
    eye = pose.inverse().apply((0.0, 0.0, 0.0))
    visible = [point for point in object_points if dot(point, eye) > 0.0]
    points = tuple(pose.apply_all(visible))
    return voxel_downsample(points, cloud_extent(points) * fraction)


def _view(pose: Rigid) -> tuple[Point, ...]:
    """The half of the default object facing a camera at ``pose``."""
    return _view_of(_object_points(), pose)


@functools.cache
def _sweep() -> tuple[tuple[Point, ...], ...]:
    """A camera walking round the object, one frame per :data:`_STEP_DEGREES`."""
    return tuple(_view(_camera(index)) for index in range(_SWEEP_FRAMES))


def _true_pose(index: int) -> Rigid:
    """Where frame ``index`` truly sits relative to frame 0, by construction."""
    return _camera(0).compose(_camera(index).inverse())


def _rotation_error(actual: Rigid, expected: Rigid) -> float:
    return math.degrees(actual.inverse().compose(expected).rotation_angle)


def _plane() -> tuple[Point, ...]:
    """One flat panel: three-dimensionally singular, and a real capture's case."""
    return tuple((x * 2.0, y * 2.0, 100.0) for x in range(20) for y in range(20))


def _cube() -> tuple[Point, ...]:
    """A solid whose three extents are equal, so its principal axes are a guess."""
    return tuple(
        (x * 2.0, y * 2.0, z * 2.0 + 100.0) for x in range(9) for y in range(9) for z in range(9)
    )


def _stranger() -> tuple[Point, ...]:
    """A real box that is not this object: three distinct extents, and no match.

    Not degenerate and not planar, so it passes every evidence check the
    recovery makes about the FRAME and fails on the only thing left - how much
    of it lands on a reference built from a different shape.
    """
    points: list[Point] = []
    for x in range(16):
        for y in range(10):
            for z in (0, 5):
                points.append((x * 2.0, y * 2.0, z * 2.0 + 100.0))
    return tuple(points)


@functools.cache
def _symmetric_sweep() -> tuple[tuple[Point, ...], ...]:
    """The same walk round the SLAB ALONE - the lobe, and nothing else, removed.

    The lobe is the whole of what makes the default scene's whole-object
    orientations distinguishable, so a slab without it is the ambiguous case by
    construction rather than by a coincidence of numbers: a half turn about any
    of its axes puts it back on itself, and a shape-level ranking correctly
    reports that it cannot tell those placements apart.
    """
    points = _object_of((_SLAB,))
    return tuple(_view_of(points, _camera(index)) for index in range(_SWEEP_FRAMES))


#: Degrees a step for the stationary fixture: small enough that the largest
#: pose gap inside a recovery partition stays under one keyframe motion gate,
#: which is what :data:`RECOVERY_MIN_PARTITION_POSE_SPAN` refuses.
_STATIONARY_STEP_DEGREES = 0.15


@functools.cache
def _stationary_sweep() -> tuple[tuple[Point, ...], ...]:
    """The same object seen from an operator who stopped moving.

    Eleven accepted frames, eleven distinct indices, and one viewpoint. Every
    frame-COUNTING gate in the recovery is satisfied by this fixture; only a
    measurement of how far the poses actually are apart can tell it from a real
    walk round the object.
    """
    points = _object_points()
    return tuple(
        _view_of(
            points,
            Rigid.of(
                rotation_about_axis(
                    (0.0, 1.0, 0.0), math.radians(_STATIONARY_STEP_DEGREES * index)
                ).rotation,
                (0.0, 0.0, _STANDOFF),
            ),
        )
        for index in range(_SWEEP_FRAMES)
    )


#: The incompatible-minima scene, and why it is a second scene rather than a
#: parameter on the first.
#:
#: What this file needs to exercise is the case the whole fallback exists for:
#: two independent references, both refined by real ICP, both clearing every
#: gate, and landing in two places that cannot both be right. On the clean scene
#: above that does not happen - measured across camera steps from 7 to 24
#: degrees and offered frames from one step to seven, every reference agrees
#: with every other to within a degree, because a fully three-dimensional
#: object seen from a full viewpoint constrains a point-to-plane solve
#: completely. The failure is a property of REAL captures: partial views, weak
#: geometry, a reference that has gone thin in one direction.
#:
#: So the scene is degraded in exactly that one way and no other. One accepted
#: frame - the one immediately before the offered frame, which is the adjacent
#: reference - is the object's FACING SURFACE ONLY, everything below
#: :data:`_CARRIED_FLAT_DEPTH` in the object's own frame removed. That is a real
#: capture's commonest weak frame and it leaves the adjacent solve free to slide
#: along the surface it can still see. Everything else is unchanged: the object,
#: the step, the gates and the policy. The object is scaled up so that the same
#: angular slide is a larger surface displacement than the agreement gate, which
#: is stated in absolute depth units and does not scale with the fixture.
_CARRIED_SCALE = 2.0
_CARRIED_STANDOFF = _STANDOFF * _CARRIED_SCALE
_CARRIED_SLAB = Box(
    half_extents=(30.0 * _CARRIED_SCALE, 12.0 * _CARRIED_SCALE, 6.0 * _CARRIED_SCALE)
)
_CARRIED_LOBE = Box(
    half_extents=(5.0 * _CARRIED_SCALE, 18.0 * _CARRIED_SCALE, 5.0 * _CARRIED_SCALE),
    pose=Rigid.of(IDENTITY_MATRIX, (26.0 * _CARRIED_SCALE, 22.0 * _CARRIED_SCALE, 0.0)),
)
#: Everything of the object below this, in its own frame, is missing from the
#: one weak accepted frame. Six, which on this object's own downsampling keeps
#: the facing surface and drops the returning sides: the sides are what stop a
#: point-to-plane solve sliding, and the whole point of this frame is that it
#: does not stop one.
_CARRIED_FLAT_DEPTH = 6.0
#: The weak frame is sampled FINER than the ordinary frames, not coarser.
#:
#: Counter-intuitive and load-bearing. A sparse weak reference does not produce
#: a wrong-but-credible placement; it produces a residual over the ordinary gate
#: and the adjacent candidate is refused, which is the ordinary vocabulary
#: working and not the case this fixture is about. Sampled finely, the same
#: geometry fits its slid placement tightly enough to clear every gate - and
#: then disagrees with the model. Measured on this fixture at the ordinary
#: sampling: adjacent residual 3.075 against the 2.600 gate, refused.
_CARRIED_WEAK_REFINEMENT = 3.0
#: Degrees between the carried scene's accepted frames.
#:
#: Its own step rather than :data:`_STEP_DEGREES`, because this scene is fitted
#: to produce a specific measured failure and the clean scene above is not.
#: Measured across steps 6 to 9 and jumps 9 to 14: this pair puts the two real
#: minima 5.847 apart against the unchanged 5.200 agreement gate, the widest
#: separation any of them reached.
_CARRIED_STEP_DEGREES = 8.0
#: Degrees the offered frame sits beyond the last accepted one - more than the
#: steady step behind it, which is what leaves the adjacent reference's slide
#: unmeasured.
_CARRIED_JUMP_DEGREES = 9.0


@functools.cache
def _carried_object() -> tuple[Point, ...]:
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
        pose = Rigid.of(rotation.rotation, (0.0, 0.0, _CARRIED_STANDOFF))
        for shape in (_CARRIED_SLAB, _CARRIED_LOBE):
            merged.extend(pose.inverse().apply_all(tuple(points_of(render([shape], pose=pose)))))
    return voxel_downsample(tuple(merged), _OBJECT_VOXEL * _CARRIED_SCALE)


def _carried_camera(degrees: float) -> Rigid:
    return Rigid.of(
        rotation_about_axis((0.0, 1.0, 0.0), math.radians(degrees)).rotation,
        (0.0, 0.0, _CARRIED_STANDOFF),
    )


def _carried_view(degrees: float, *, facing_only: bool = False) -> tuple[Point, ...]:
    pose = _carried_camera(degrees)
    eye = pose.inverse().apply((0.0, 0.0, 0.0))
    visible = [
        point
        for point in _carried_object()
        if dot(point, eye) > 0.0 and (not facing_only or point[2] >= _CARRIED_FLAT_DEPTH)
    ]
    points = tuple(pose.apply_all(visible))
    fraction = (
        _FRAME_VOXEL_FRACTION / _CARRIED_WEAK_REFINEMENT if facing_only else _FRAME_VOXEL_FRACTION
    )
    return voxel_downsample(points, cloud_extent(points) * fraction)


@functools.cache
def _carried_sweep() -> tuple[tuple[Point, ...], ...]:
    """Nine ordinary accepted frames, one weak one, and the frame after it."""
    last = _CARRIED_STEP_DEGREES * (_ACCEPTED_FRAMES - 1)
    frames = [_carried_view(_CARRIED_STEP_DEGREES * index) for index in range(_ACCEPTED_FRAMES - 1)]
    frames.append(_carried_view(last, facing_only=True))
    frames.append(_carried_view(last + _CARRIED_JUMP_DEGREES))
    return tuple(frames)


def _carried_truth() -> Rigid:
    """Where the offered frame truly sits relative to frame 0, by construction."""
    return _carried_camera(0.0).compose(
        _carried_camera(
            _CARRIED_STEP_DEGREES * (_ACCEPTED_FRAMES - 1) + _CARRIED_JUMP_DEGREES
        ).inverse()
    )


def _placement_error(pose: Rigid, cloud: tuple[Point, ...]) -> float:
    """How far a placement puts the frame's own surface from where it truly is.

    The same measurement the tracker's agreement gate is stated in - one number
    in depth units, over the frame's own points - so an error here and the gate
    that refused the frame are directly comparable rather than two scales a
    reader has to convert between.
    """
    return frame_travel(cloud, pose.inverse().compose(_carried_truth()))


def _state(tracker: IncrementalTracker) -> tuple[object, ...]:
    """Everything a refusal is forbidden to move."""
    return (
        tracker.frames,
        tracker.poses,
        tracker.keyframes,
        tracker.retained_clouds,
        tracker.retained_points,
    )


def _tracked(
    policy: TrackingPolicy,
    frames: int = _ACCEPTED_FRAMES,
    sweep: tuple[tuple[Point, ...], ...] | None = None,
) -> IncrementalTracker:
    """A tracker with a real accepted model behind it, or a failed assertion."""
    tracker = IncrementalTracker(policy=policy)
    for index, cloud in enumerate((sweep or _sweep())[:frames]):
        assert tracker.offer(cloud).accepted, f"the fixture must place frame {index}"
    return tracker


def _no_candidates(monkeypatch: pytest.MonkeyPatch, *, after: int = _ACCEPTED_FRAMES) -> None:
    """Refuse every ordinary candidate ONCE the tracker holds ``after`` frames.

    The first of the three refusals :meth:`_maybe_recover` may be offered, and
    the only one that can be produced without also arranging what real ICP does
    with three real references. It is a SEAM, not the acceptance regression:
    the load-bearing positive case in this file reaches the recovery through a
    real candidate disagreement between two real refinements, and this stub
    exists so the refusals below can each be reached one at a time.

    Conditional on the tracker's own accepted count rather than undone by hand,
    so the SAME stub can be installed for the whole of an offline
    :func:`~scanner3d.pipeline.reconstruct_sweep` run: every frame that builds
    the model is placed normally and only the last one comes through this door.
    That is what makes the live/offline parity assertion possible at all.
    """
    real = IncrementalTracker._candidates

    def gated(self: IncrementalTracker, *args: object, **kwargs: object) -> tuple[object, ...]:
        if self.frames >= after:
            return ()
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(IncrementalTracker, "_candidates", gated)


def _offer(
    monkeypatch: pytest.MonkeyPatch,
    policy: TrackingPolicy,
    *,
    frames: int = _ACCEPTED_FRAMES,
    cloud: tuple[Point, ...] | None = None,
    sweep: tuple[tuple[Point, ...], ...] | None = None,
    build_policy: TrackingPolicy | None = None,
) -> tuple[TrackDecision, IncrementalTracker, tuple[object, ...]]:
    """One frame offered through the recovery door, with the state before it.

    ``build_policy`` builds the accepted chain under a DIFFERENT policy from the
    one the offered frame is decided under. Used only where the policy under
    test would make building the chain absurd rather than interesting - a
    convergence tolerance no solve reaches makes every one of the ten frames
    behind the offered one run its whole iteration budget, which measures
    nothing and costs minutes. The tracker's state at the moment of the offer is
    the same either way.
    """
    tracker = _tracked(build_policy or policy, frames, sweep)
    tracker._policy = policy
    before = _state(tracker)
    _no_candidates(monkeypatch, after=frames)
    try:
        offered = (sweep or _sweep())[_OFFERED_FRAME] if cloud is None else cloud
        decision = tracker.offer(offered)
    finally:
        # Undone immediately, so a test may offer more than one frame through
        # this door without the SECOND fixture's own accepted chain being built
        # by a tracker that cannot form a candidate.
        monkeypatch.undo()
    return decision, tracker, before


# --- 1. the acceptance every refusal below is measured against ---------------------------


def test_a_frame_no_ordinary_reference_placed_is_recovered_and_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case the fallback exists for, and the control for this whole file.

    A frame that reached the ordinary refusal is re-seeded from a bounded coarse
    search, refined, and confirmed against a second reference sharing no
    accepted frame with the first. The pose it lands on is asserted against the
    scene's construction rather than against itself, because "a recovery
    happened" and "the frame is where it actually is" are different claims and
    only the second one is worth having.
    """
    decision, _tracker, _before = _offer(monkeypatch, TrackingPolicy())

    assert decision.accepted, decision.message()
    assert decision.origin == RECOVERY_ORIGIN
    assert decision.rescue is False
    assert decision.pose is not None
    assert _rotation_error(decision.pose, _true_pose(_OFFERED_FRAME)) < (
        _MAX_RECOVERED_ERROR_DEGREES
    )

    recovery = decision.recovery
    assert recovery is not None
    assert recovery.triggered_by == TRACK_NO_CANDIDATE_TYPE
    assert recovery.attempted and recovery.accepted
    assert recovery.agreeing == (RECOVERY_ORIGIN, RECOVERY_CONFIRM_ORIGIN)
    # Independence is the claim, so it is the thing asserted: two references
    # that shared an accepted frame would be one reference read twice.
    first, second = (set(partition.frames) for partition in recovery.partitions)
    assert first and second and not (first & second)
    assert recovery.confirmation_disagreements, "a confirmation has to have been measured"


# --- 2. the rescue gates, inherited whole ------------------------------------------------


def test_the_recovery_inherits_the_rescue_gate_on_reference_viewpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery is a placement with no second opinion, so it clears that gate.

    ``rescue_min_reference_frames`` is the evidence a model-only placement needs
    because nothing else is left to contradict it - and a recovered placement is
    in exactly that position, since it is only reached once every ordinary
    reference has already refused. A better STARTING POSE is all a coarse search
    contributes; it contributes nothing at all about how many accepted
    viewpoints the reference behind the answer spans.

    Set out of reach, the recovery must refuse, and it must refuse without
    paying for a coarse search first: a partition too thin to rescue anything
    cannot become thick enough later in the same attempt.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(rescue_min_reference_frames=99))

    assert not decision.accepted
    assert decision.pose is None
    assert decision.failure_type == TRACK_NO_CANDIDATE_TYPE
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert any("model-only rescue clears" in refusal for refusal in recovery.refusals)
    assert recovery.work.coarse_searches == 0
    assert recovery.work.icp_solves == 0


def test_the_recovery_inherits_the_rescue_gate_on_reference_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the same evidence, and it survives the coarse search.

    Overlap cannot be settled before the search the way the viewpoint count can
    - how much of the frame lands on a reference is a property of the placement,
    which is what the refinement produces - so this gate is applied to every
    refined hypothesis and every confirmation individually. Set out of reach,
    every hypothesis that would otherwise have been accepted is refused by name,
    and nothing moves.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(rescue_min_overlap=0.999))

    assert not decision.accepted
    assert decision.failure_type == TRACK_NO_CANDIDATE_TYPE
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert recovery.hypotheses, "the search must have got as far as refining something"
    assert not any(hypothesis.accepted for hypothesis in recovery.hypotheses)
    assert any(
        "overlaps the recovery reference by only" in refusal
        for hypothesis in recovery.hypotheses
        for refusal in hypothesis.refusals
    )


def test_a_capture_too_young_for_two_disjoint_references_cannot_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the second independent partition and the same frame must refuse.

    One reference set cannot confirm itself, so a capture with too few accepted
    frames to deal two disjoint ones out of has no recovery available to it -
    and says so, rather than confirming a placement against the frames that
    produced it.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), frames=4, cloud=_sweep()[4])

    assert not decision.accepted
    assert _state(tracker) == before
    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert any("cannot confirm itself" in refusal for refusal in recovery.refusals)
    assert recovery.work.coarse_searches == 0


# --- 3. evidence the recovery refuses to work from ---------------------------------------


def test_planar_evidence_is_refused_before_any_orientation_is_ranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat frame determines no orientation, so there is nothing to rank.

    Refused by name rather than left to ICP, which will settle on a plane just
    as tidily as on a real surface and report a small residual for doing it.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), cloud=_plane())

    assert not decision.accepted
    assert _state(tracker) == before
    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert any("does not span three dimensions" in refusal for refusal in recovery.refusals)
    assert recovery.work.icp_solves == 0


def test_a_symmetric_frame_with_undetermined_principal_axes_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degeneracy is not the gate; the ambiguity it produces still is.

    A cube genuinely has several equally good placements. Its principal-frame
    warning is recorded, the shared ranking measures the alternatives at 1.00x,
    and that measured ambiguity refuses before any refinement. This is the
    unsafe case the old blanket degeneracy veto was trying to cover, retained
    without treating every near-equal pair of extents as though it were a cube.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), cloud=_cube())

    assert not decision.accepted
    assert _state(tracker) == before
    recovery = decision.recovery
    assert recovery is not None and not recovery.accepted
    assert recovery.coarse_source_degenerate is True
    assert recovery.coarse_score_unambiguous is False
    assert recovery.coarse_score_margin < AMBIGUITY_MARGIN
    assert any("shared coarse search" in refusal for refusal in recovery.refusals)
    assert recovery.work.icp_solves == 0


def test_principal_axis_warning_does_not_veto_measured_unambiguous_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warning says how seeds were made, not that every seed is ambiguous.

    Mark both principal frames degenerate without changing one candidate, score,
    margin, refinement or reference in the positive control. The recovery must
    still earn its pose from both ambiguity verdicts and the disjoint confirming
    reference. This is the policy shape of physical capture
    ``20260831-202514`` frame 35, where a near-square recovery reference tripped
    the old veto despite 1.72x/2.41x margins and 0.370 agreement.
    """

    def flagged(*args: object, **kwargs: object) -> object:
        result = _coarse_align(*args, **kwargs)  # type: ignore[arg-type]
        return replace(
            result,
            source_principal_degenerate=True,
            target_principal_degenerate=True,
            limitations=(
                *result.limitations,
                "test fixture: both principal frames have near-equal extents",
            ),
        )

    monkeypatch.setattr("scanner3d.tracking.coarse_align", flagged)
    decision, _tracker, _before = _offer(monkeypatch, TrackingPolicy())

    assert decision.accepted, decision.message()
    recovery = decision.recovery
    assert recovery is not None and recovery.accepted
    assert recovery.coarse_source_degenerate is True
    assert recovery.coarse_reference_degenerate is True
    assert recovery.coarse_score_unambiguous is True
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.selected_overlap_unambiguous is True
    assert recovery.confirmation_disagreements


def test_disjoint_recovery_gets_first_chance_before_a_tighter_tied_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery wins the arbitration when its stronger evidence succeeds.

    The real positive fixture ordinarily has one unambiguous agreement cluster.
    This seam marks that cluster as the fallback from an equal-size tie; every
    candidate, reference partition, coarse hypothesis, refinement and
    confirmation remains real. The accepted origin must therefore be recovery,
    not the perfectly usable ordinary cluster retained behind it.
    """
    tracker = _tracked(TrackingPolicy())
    real_agreed = IncrementalTracker._agreed

    def arbitrate(
        self: IncrementalTracker,
        index: int,
        survivors: list[TrackCandidate],
        source: tuple[Point, ...],
        gate: float,
        candidates: tuple[TrackCandidate, ...],
    ) -> _Agreement | TrackDecision:
        result = real_agreed(self, index, survivors, source, gate, candidates)
        assert isinstance(result, _Agreement), "the fixture must have an ordinary placement"
        refusal = self._refused(
            index,
            TRACK_DISAGREEMENT_TYPE,
            ["fixture: two equal-size consensus clusters need stronger arbitration"],
            candidates=candidates,
        )
        return replace(result, arbitration_refusal=refusal)

    monkeypatch.setattr(IncrementalTracker, "_agreed", arbitrate)
    decision = tracker.offer(_sweep()[_OFFERED_FRAME])

    assert decision.accepted, decision.message()
    assert decision.origin == RECOVERY_ORIGIN
    recovery = decision.recovery
    assert recovery is not None and recovery.accepted
    assert recovery.triggered_by == TRACK_DISAGREEMENT_TYPE
    assert recovery.confirmation_disagreements


# --- 4. what the record has to say honestly ----------------------------------------------


def test_the_ordinary_accepted_path_runs_no_coarse_search() -> None:
    """The whole cost claim, asserted where it is made rather than described.

    ``recovery`` is ``None`` exactly when the frame never reached
    :meth:`~scanner3d.tracking.IncrementalTracker._maybe_recover`, so a sweep of
    ordinary placements carrying no recovery evidence at all IS the statement
    that the accepted path paid nothing for this fallback existing.
    """
    tracker = IncrementalTracker()
    decisions = [tracker.offer(cloud) for cloud in _sweep()[:_ACCEPTED_FRAMES]]

    assert all(decision.accepted for decision in decisions)
    assert all(decision.recovery is None for decision in decisions)
    assert all(decision.origin != RECOVERY_ORIGIN for decision in decisions)


def test_the_shared_coarse_ambiguity_verdict_is_required_as_well_as_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two verdicts, both required, and the record carries both numbers.

    Round 1 recorded :func:`~scanner3d.multiview.coarse_align`'s verdict and
    then accepted while it was false, substituting a coverage margin measured on
    a different quantity. That substitution is withdrawn: a recovery now needs
    the shared verdict AND the coverage margin, and the coverage margin is
    additional evidence rather than a replacement.

    Coverage is asked as well because it is a different measurement of a
    different amount of the frame: the shared score is a truncated mean over a
    DECIMATED cloud on a coarse grid, and coverage is measured over the WHOLE
    frame at the same tolerance the accepted placement's own overlap will be
    judged at. Which of the two is the harder question is not asserted - it is
    not a fixed property of the pair, and on this repository's two fixtures it
    goes both ways. What is asserted is that they are two numbers, both kept,
    and that neither was written into the other's field.
    """
    decision, _tracker, _before = _offer(monkeypatch, TrackingPolicy())
    recovery = decision.recovery
    assert recovery is not None and recovery.accepted

    assert recovery.coarse_score_unambiguous is True, "the shared verdict is required, not noted"
    assert recovery.coarse_score_margin >= AMBIGUITY_MARGIN
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.coarse_overlap_margin >= AMBIGUITY_MARGIN

    # Two different numbers measured on two different quantities, both kept. A
    # record that had written the coverage margin into the shared field would
    # pass every other assertion in this file and fail this one.
    assert recovery.coarse_score_margin != recovery.coarse_overlap_margin
    # Carried verbatim, which is checkable because these sentences exist nowhere
    # else: they are written by coarse_align and by nothing in this module.
    assert any(
        note.startswith("A coarse orientation is a starting point for ICP")
        for note in recovery.coarse_limitations
    )
    # The prior this module offers the shared search is ranked by it and said so.
    assert any("came from the caller as priors" in note for note in recovery.coarse_limitations)


def test_the_coarse_and_refinement_stages_are_counted_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One "source points" number for a recovery would understate one of them.

    The ranking is scored from a bounded sample of the frame and every
    refinement is fitted from the larger sample every ordinary candidate uses,
    so the two are different work on different data. The clearest proof that
    they are counted separately rather than reported once is a recovery that
    RANKED and then refused before refining anything: it paid a coarse cost and
    no refinement cost, and a record carrying one number could not say so.
    """
    accepted, _tracker, _before = _offer(monkeypatch, TrackingPolicy())
    recovery = accepted.recovery
    assert recovery is not None and recovery.accepted
    work = recovery.work

    assert work.coarse_source_points <= TrackingPolicy().recovery_source_points
    assert work.refine_source_points <= TrackingPolicy().candidate_source_points
    # The search ranks everything it generates whatever shortlist it is asked to
    # score, so reporting the shortlist as the work done would understate it.
    assert work.coarse_orientations_ranked >= work.coarse_hypotheses_scored
    assert work.confirmations_refined >= 1
    assert work.coarse_reference_points < work.reference_points

    refused, _tracker2, _before2 = _offer(monkeypatch, TrackingPolicy(), sweep=_symmetric_sweep())
    ambiguous = refused.recovery
    assert ambiguous is not None and ambiguous.attempted and not ambiguous.accepted
    assert ambiguous.work.coarse_source_points > 0
    assert ambiguous.work.coarse_searches == 1
    assert ambiguous.work.refine_source_points == 0
    assert ambiguous.work.hypotheses_refined == 0


def test_no_recovery_work_count_exceeds_its_advertised_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ceiling in the policy record, checked against the count beside it.

    The bounds are only bounds if the record carries the numbers to check them
    against, which is the reason both are recorded at all.
    """
    policy = TrackingPolicy()
    decision, _tracker, _before = _offer(monkeypatch, policy)
    recovery = decision.recovery
    assert recovery is not None and recovery.accepted
    work = recovery.work
    bounds = policy.as_dict()["recovery"]
    assert isinstance(bounds, dict)

    assert work.coarse_source_points <= policy.recovery_source_points
    assert work.refine_source_points <= policy.candidate_source_points
    assert work.coarse_searches <= bounds["max_coarse_searches"]
    assert work.coarse_orientations_ranked <= bounds["max_coarse_poses_ranked"]
    assert work.coarse_orientations_ranked <= COARSE_RANKED_POSES
    assert work.coarse_hypotheses_scored <= bounds["max_coarse_hypotheses_scored"]
    assert work.hypotheses_refined <= policy.recovery_hypotheses
    assert work.confirmations_refined <= bounds["max_confirmations_refined"]
    assert work.icp_solves <= policy.max_recovery_solves
    assert work.partitions <= policy.recovery_partitions
    assert work.reference_frames <= policy.max_recovery_reference_frames
    assert work.reference_points <= bounds["max_reference_points"]
    assert work.coarse_reference_points <= policy.recovery_partition_points


def test_the_recovery_evidence_survives_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable record is the whole point, so it has to be writable.

    ``allow_nan=False`` because an infinity is exactly what a margin with no
    genuinely different rival produces, and a record that wrote it as ``NaN``
    would be unreadable by anything but Python.
    """
    accepted, _tracker, _before = _offer(monkeypatch, TrackingPolicy())
    refused, _tracker2, _before2 = _offer(monkeypatch, TrackingPolicy(), cloud=_plane())

    for decision in (accepted, refused):
        written = json.dumps(decision.as_dict(), allow_nan=False)
        read = json.loads(written)
        assert read["recovery"]["schema_version"] >= 1
        assert isinstance(read["recovery"]["work"], dict)
        assert isinstance(read["recovery"]["coarse_limitations"], list)


# --- 6. the refusals a recovery has of its own -------------------------------------------


def test_a_symmetric_shape_ranks_several_placements_alike_and_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slab without its lobe: one shape, several placements, no recovery.

    The lobe is the whole of what distinguishes this scene's whole-object
    orientations, so removing it is the ambiguous case by construction rather
    than by a coincidence of numbers. A ranking that cannot tell two placements
    apart is not evidence for either of them, and refining one of them would
    pick an answer the geometry did not give.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), sweep=_symmetric_sweep())

    assert not decision.accepted
    assert decision.pose is None
    assert decision.failure_type == TRACK_NO_CANDIDATE_TYPE
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    # Whichever of the two verdicts caught it, one of them had to, and the
    # refusal has to be the ambiguity one rather than something further down.
    assert not (recovery.coarse_score_unambiguous and recovery.coarse_overlap_unambiguous)
    assert any(
        "unambiguous" in refusal or "sits several ways" in refusal for refusal in recovery.refusals
    ), recovery.refusals
    assert recovery.work.hypotheses_refined == 0, "refused before paying for any refinement"


def test_partitions_that_span_no_viewpoint_change_cannot_confirm_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eleven accepted frames, eleven distinct indices, and one viewpoint.

    Every frame-COUNTING gate in the recovery is satisfied by an operator who
    stopped moving: the partitions are disjoint, each holds four accepted
    frames, and each clears ``rescue_min_reference_frames``. Only the MEASURED
    pose span can tell that fixture from a real walk round the object, which is
    why the span is measured with the tracker's own keyframe pose-gap
    vocabulary and recorded on every partition it builds.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), sweep=_stationary_sweep())

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert any("accepted VIEWPOINTS" in refusal for refusal in recovery.refusals), recovery.refusals
    assert recovery.partitions, "the partitions are built and recorded before they are refused"
    assert all(
        partition.pose_span < RECOVERY_MIN_PARTITION_POSE_SPAN for partition in recovery.partitions
    )
    # Counting alone would have let this through, which is the point.
    assert all(len(partition.frames) >= 4 for partition in recovery.partitions)
    assert recovery.work.coarse_searches == 0, "refused before paying for a coarse search"


def test_a_hypothesis_that_never_settles_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convergence is the ordinary gate and a recovery clears it unchanged.

    Set out of reach - a convergence tolerance no ordinary solve reaches - the
    hypotheses that are still moving when the solver stops are refused and say
    so, no hypothesis survives, and the whole recovery refuses. The convergence
    gate is the tracker's own, unchanged; what moved is the tolerance, and it
    moved in the tightening direction.

    Only the OFFERED frame is decided under that tolerance. The accepted chain
    behind it is built under the default policy, because a tolerance no solve
    reaches makes each of the ten frames behind this one run its whole iteration
    budget - minutes of arithmetic that measure nothing.
    """
    decision, tracker, before = _offer(
        monkeypatch,
        TrackingPolicy(convergence_noise_fraction=1e-12),
        build_policy=TrackingPolicy(),
    )

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    unsettled = [
        hypothesis
        for hypothesis in recovery.hypotheses
        if any("never settled" in refusal for refusal in hypothesis.refusals)
    ]
    assert unsettled, [hypothesis.refusals for hypothesis in recovery.hypotheses]
    assert all(not hypothesis.accepted for hypothesis in recovery.hypotheses)
    assert recovery.selected_rank is None
    # CAUSAL: convergence is what stopped the best-ranked hypothesis, and no
    # survivor is what stopped the recovery.
    assert recovery.hypotheses[0] in unsettled
    assert any(
        "cleared the same gates an ordinary placement clears" in refusal
        for refusal in recovery.refusals
    ), recovery.refusals


def test_a_recovery_may_not_place_a_frame_the_travel_gate_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A better starting pose does not move the last accepted pose.

    The travel gate measures a recovered placement against the same last
    accepted pose it measures every ordinary placement against, so a frame whose
    true placement is a jump is refused here exactly as it is refused there.
    This is the boundary between this fallback and
    :mod:`scanner3d.relocalize`, and it is asserted rather than described.
    """
    far = _view(_camera(_OFFERED_FRAME + _TRAVEL_JUMP_STEPS))
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), cloud=far)

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    # CAUSAL. The frame is far enough round the object that it is a jump, and
    # close enough that the ranking is still clear - so both ambiguity verdicts
    # pass, the search runs, hypotheses are refined, and TRAVEL is what stops
    # the best one. A refusal earlier than that would prove nothing about the
    # travel gate at all.
    assert recovery.coarse_score_unambiguous is True
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.work.hypotheses_refined > 0
    best = recovery.hypotheses[0]
    assert not best.accepted
    assert any("moves the surface" in refusal for refusal in best.refusals), best.refusals
    assert any(
        "cleared the same gates an ordinary placement clears" in refusal
        for refusal in recovery.refusals
    ), recovery.refusals


def test_a_recovery_may_not_place_a_frame_the_overlap_gate_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A residual measured over part of a frame describes that part.

    The overlap gate is TIGHTENED here rather than the frame made strange: the
    ordinary requirement is raised to 0.98 of the frame and the rescue
    requirement above it, so the acceptance fixture's own frame - which really
    does land on its reference, and which the default policy recovers - now
    fails the one gate that moved. Nothing is widened, and what is asserted is
    that the recovery is measured by that gate rather than around it.
    """
    decision, tracker, before = _offer(
        monkeypatch,
        TrackingPolicy(min_reference_overlap=0.98, rescue_min_overlap=0.99),
        build_policy=TrackingPolicy(),
    )

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert recovery.work.hypotheses_refined > 0
    assert all(not hypothesis.accepted for hypothesis in recovery.hypotheses)
    assert all(
        any("overlaps the coarse_recovery reference" in refusal for refusal in hypothesis.refusals)
        for hypothesis in recovery.hypotheses
    ), [hypothesis.refusals for hypothesis in recovery.hypotheses]


def test_a_frame_that_is_not_this_object_ranks_every_placement_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cloud with nothing of the model in it has no best placement.

    The offered cloud here is a real, non-degenerate box that is not this
    object. Every orientation of it fits the reference about as badly as every
    other, so the shared coarse ranking reports a margin of 1.00 - not a fit it
    is unsure about, a ranking with no winner in it at all - and the recovery
    stops there rather than refining a placement it has no reason to prefer.
    Asserted by name, because "refused" alone would also be true of a recovery
    that had crashed into some later gate by accident.
    """
    decision, tracker, before = _offer(monkeypatch, TrackingPolicy(), cloud=_stranger())

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert recovery.coarse_score_unambiguous is False
    assert recovery.coarse_score_margin < AMBIGUITY_MARGIN
    assert any("shared coarse search" in refusal for refusal in recovery.refusals), (
        recovery.refusals
    )
    assert recovery.work.hypotheses_refined == 0, "refused before paying for any refinement"


def test_several_seeds_reaching_one_answer_are_one_answer() -> None:
    """Refinement merging seeds is fine, and merged seeds are not rivals.

    Four of sixteen refined hypotheses clear every gate on this fixture and all
    four land on the same pose, which is refinement doing exactly its job. The
    claim asserted here is what the recovery then does with them: they form one
    mutual-agreement cluster, the placement is admitted, and none of the merged
    hypotheses is counted as an alternative to the one selected - a margin that
    punished two seeds for reaching one answer would refuse every recovery this
    module can make.

    THE OTHER HALF OF THIS BRANCH IS NOT REACHED FROM A SCENE, AND THAT IS
    MEASURED RATHER THAN ASSUMED. Two survivors in two places needs two
    placements that both clear convergence, residual, overlap and travel and sit
    further apart than the agreement gate - and the ambiguity and travel gates
    that run BEFORE this one refuse exactly those scenes first. Measured, with
    the coarse ranking refining up to sixteen hypotheses each time: the clean
    asymmetric sweep at fifteen offered angles from 90 to 125 degrees, the
    incompatible-minima scene at six, ten and sixteen hypotheses, a slab with
    the distinguishing lobe scaled to 0.35, 0.5 and 0.65, and the symmetric slab
    with the ambiguity margin at its own floor - in every one of them either
    every survivor merged or no survivor cleared a gate. So the disagreeing case
    is asserted through ONE narrow seam on the accepted candidate RECORD the
    agreement decision is handed - a displacement applied after the gates judged
    the pose, not a second gate-clearing minimum this scene produces - in
    :func:`test_two_refined_placements_that_disagree_refuse_the_recovery`, with
    the recovery's own refusal and the unchanged agreement gate asserted by
    name; ``test_tracking.py`` additionally exercises ``_agreement_clusters``
    and ``_disagreements`` directly, including a set whose extremes are nearly
    twice the gate apart.
    """
    sweep = _carried_sweep()
    tracker = IncrementalTracker(policy=TrackingPolicy(keyframe_candidates=0))
    for cloud in sweep[:-1]:
        assert tracker.offer(cloud).accepted
    tracker._policy = TrackingPolicy(keyframe_candidates=0, recovery_hypotheses=16)
    decision = tracker.offer(sweep[-1])

    recovery = decision.recovery
    assert recovery is not None and recovery.accepted
    merged = [hypothesis for hypothesis in recovery.hypotheses if hypothesis.accepted]
    assert len(merged) > 1, "the fixture must actually merge more than one seed"
    assert decision.pose is not None
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    for hypothesis in merged:
        assert hypothesis.refined is not None
        apart = frame_travel(sweep[-1], hypothesis.refined.inverse().compose(decision.pose))
        assert apart <= gate, (hypothesis.rank, apart)
    # And the merged seeds are not rivals: not one of their coverages is the
    # number the selected hypothesis's margin was measured against.
    assert recovery.selected_rival_overlap is not None
    assert all(
        hypothesis.coarse_overlap != recovery.selected_rival_overlap for hypothesis in merged
    ), (recovery.selected_rival_overlap, [h.coarse_overlap for h in merged])
    assert recovery.selected_overlap_margin >= AMBIGUITY_MARGIN


# --- the two branches no scene on this fixture family reaches ---------------------------
#
# Both regressions below need a combination the geometry will not produce: every
# hypothesis that survives refinement on these scenes lands in the SAME place,
# so the survivors pass or fail their gates together. Measured, at four, eight
# and sixteen refined hypotheses, on the clean sweep at every whole camera step
# from one to twelve beyond the offered frame and at every tenth of a step from
# five to six and a half - the interval where the travel gate turns over - the
# best-covering hypothesis was accepted and so was every other survivor, or it
# was refused and so was every other survivor. Never one of each.
#
# So each of these two tests installs ONE narrow deterministic seam, named in
# its own docstring, and nothing else in the recovery is stubbed: the ranking,
# every coverage number, every other refinement and confirmation, the merge
# test, the margin arithmetic, the refusal text and the tracker state are all
# real. What each seam supplies is stated exactly where it is installed.


def _refuse_the_best_covering_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEAM: the solver refuses the best-covering PLACEMENT, and nothing else.

    Every recovery hypothesis whose starting pose sits within the agreement gate
    of the top-ranked one - that placement, however many coarse orientations
    spell it - comes back unformed, exactly as :meth:`_candidate` returns
    nothing when a solve raises or a gate refuses it. Every genuinely different
    hypothesis is refined for real and judged by the real gates.

    What is being reproduced is the owner's capture: coarse rank 0 was the
    recorded clear best and the travel gate refused it, and a lower-ranked
    placement went on to survive. The travel test above shows a real unchanged
    gate really does refuse a rank-0 hypothesis on this scene; what it cannot
    also produce is a lower-ranked survivor beside it, which is what this seam
    supplies and only that.
    """
    real = IncrementalTracker._candidate
    best: list[Rigid] = []

    def gated(
        self: IncrementalTracker,
        index: int,
        source: tuple[Point, ...],
        *,
        origin: str,
        initial: Rigid,
        **kwargs: object,
    ) -> object:
        if origin == RECOVERY_ORIGIN:
            if not best:
                best.append(initial)
            if _disagreement(best[0], initial, source) <= AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR:
                return None
        return real(self, index, source, origin=origin, initial=initial, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(IncrementalTracker, "_candidate", gated)


def _displace_the_second_refined_placement(
    monkeypatch: pytest.MonkeyPatch, *, apart: float
) -> None:
    """SEAM: the second refined hypothesis is RECORDED ``apart`` from the first.

    The candidate is refined for real and judged by the real gates, and then its
    POSE is replaced with one translated far enough that the two survivors are
    two placements rather than one. Stated narrowly, because that ordering is
    what this seam actually is: :meth:`IncrementalTracker._candidate` judged the
    ORIGINAL pose, and the displaced pose it returns is never put back through
    convergence, residual, matched, used, overlap or travel. What the seam
    supplies is the accepted candidate RECORD the recovery's agreement decision
    is then handed.

    So this proves the agreement decision, its refusal text and the absence of
    mutation. It does not prove that this fixture naturally produces a second
    minimum that both clears every gate and disagrees - measured, it does not,
    which is why the seam exists.
    """
    real = IncrementalTracker._candidate
    seen = {"count": 0}
    offset = Rigid.of(IDENTITY_MATRIX, (apart, 0.0, 0.0))

    def gated(
        self: IncrementalTracker,
        index: int,
        source: tuple[Point, ...],
        *,
        origin: str,
        **kwargs: object,
    ) -> object:
        candidate = real(self, index, source, origin=origin, **kwargs)  # type: ignore[arg-type]
        if origin != RECOVERY_ORIGIN:
            return candidate
        seen["count"] += 1
        if seen["count"] == 2 and candidate is not None and candidate.accepted:
            return replace(candidate, pose=candidate.pose.compose(offset))
        return candidate

    monkeypatch.setattr(IncrementalTracker, "_candidate", gated)


def test_two_refined_placements_that_disagree_refuse_the_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two answers are two answers, wherever in the recovery they appear.

    The acceptance fixture, the default policy and every unchanged gate - the
    1.15 ambiguity margin included, and asserted below rather than lowered to
    reach this branch. Both ambiguity verdicts pass, four hypotheses are refined
    against the searching reference, and the recovery is handed two accepted
    candidate records that place the frame further apart than the agreement
    gate. It refuses by name and moves nothing: the smaller residual is not
    evidence for either placement, here or in the ordinary decision.

    The one seam is :func:`_displace_the_second_refined_placement`, and its
    boundary is stated exactly there: the second record's pose is displaced
    AFTER the gates judged the original, so what is asserted below is the
    agreement decision, its text and the unmoved state - not that a second
    gate-clearing minimum arises from this scene. Everything else, including
    the disagreement measurement and the gate it is compared against, is real.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    policy = TrackingPolicy()
    assert policy.recovery_ambiguity_margin == AMBIGUITY_MARGIN, "the margin is not lowered"

    tracker = _tracked(policy)
    before = _state(tracker)
    _no_candidates(monkeypatch)
    _displace_the_second_refined_placement(monkeypatch, apart=3.0 * gate)
    decision = tracker.offer(_sweep()[_OFFERED_FRAME])

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    # Reached the branch honestly: the ranking was clear on both quantities and
    # the refusal is about what refinement did, not about what was ranked.
    assert recovery.coarse_score_unambiguous is True
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.work.hypotheses_refined > 0
    survivors = [hypothesis for hypothesis in recovery.hypotheses if hypothesis.accepted]
    assert len(survivors) >= 2, [hypothesis.refusals for hypothesis in recovery.hypotheses]
    poses = [hypothesis.refined for hypothesis in survivors]
    assert all(pose is not None for pose in poses)
    assert (
        max(
            _disagreement(first, second, _sweep()[_OFFERED_FRAME])
            for first in poses
            for second in poses
            if first is not None and second is not None
        )
        > gate
    )
    # No pose was selected: the refusal happens before any of them is chosen.
    assert recovery.selected_rank is None
    assert any(
        "placed this frame in different places" in refusal for refusal in recovery.refusals
    ), recovery.refusals
    assert any(f"{gate:.3f} agreement gate" in refusal for refusal in recovery.refusals), (
        recovery.refusals
    )


# --- 7. the ambiguity evidence belongs to the hypothesis that was selected ---------------


def test_the_selected_hypothesis_carries_its_own_ambiguity_margin() -> None:
    """A margin is evidence for the placement it was measured on and no other.

    Round 1 recorded the coarse ranking's best margin and let whichever
    hypothesis survived refinement stand on it. On the owner's capture that let
    a rank 0 the travel gate had already refused hand its 3.317 margin to the
    rank 2 the recovery went on to confirm.

    Two separate claims are asserted here, on the incompatible-minima fixture
    because it is the one where they are different numbers.

    First, that the two margins are measured SEPARATELY and can disagree. The
    pre-refinement margin is 1.36, measured from the ranking's best against the
    best coarse placement that put the frame somewhere else. The selected
    hypothesis's margin is 3.13, because that runner-up refined onto the
    selected placement and two seeds reaching one answer are one answer. A
    record that carried one number in both fields would pass every other
    assertion in this file.

    Second - and this is the round-1 defect - that a hypothesis the gates
    REFUSED still counts as a rival. The margin this acceptance stands on is
    measured against coarse rank 2, which the overlap gate refused.

    This test is the ACCEPTING side of that correction: the selection here is
    rank 0, so what it can show is that the two margins are separate numbers and
    that a refused hypothesis is a rival.
    :func:`test_a_lower_rank_may_not_stand_on_a_refused_higher_ranks_confidence`
    is the refusing side, where rank 0 fails a gate, a lower rank survives, and
    the recovery refuses on the selected placement's own margin.
    """
    decision, _after, _before = _carried_offer(TrackingPolicy(keyframe_candidates=0))
    recovery = decision.recovery
    assert recovery is not None and recovery.accepted

    assert recovery.selected_rank is not None
    assert recovery.selected_overlap_unambiguous is True
    assert recovery.selected_overlap_margin >= AMBIGUITY_MARGIN
    selected = recovery.hypotheses[recovery.selected_rank]
    assert selected.accepted, "the recorded rank is the hypothesis that was actually admitted"
    assert selected.coarse_overlap == recovery.selected_coarse_overlap
    # Every hypothesis ranked above the selected one was refused, so none of
    # them could have lent it anything.
    assert all(
        not hypothesis.accepted for hypothesis in recovery.hypotheses[: recovery.selected_rank]
    )

    # Two margins, measured separately, and different.
    assert recovery.selected_overlap_margin != recovery.coarse_overlap_margin
    assert recovery.selected_overlap_margin > recovery.coarse_overlap_margin

    # The rival the selection's margin was measured against is a hypothesis the
    # gates REFUSED - refused hypotheses are still rival placements.
    assert recovery.selected_rival_overlap is not None
    rivals = [
        hypothesis
        for hypothesis in recovery.hypotheses
        if hypothesis.coarse_overlap == recovery.selected_rival_overlap
    ]
    assert rivals and not any(hypothesis.accepted for hypothesis in rivals), [
        (hypothesis.rank, hypothesis.accepted) for hypothesis in rivals
    ]
    assert recovery.selected_overlap_margin == pytest.approx(
        recovery.selected_coarse_overlap / recovery.selected_rival_overlap
    )
    # The inequality the correction rests on: a refused rival ranked above the
    # selection covers more than it, so the same division lands under 1.0.
    assert all(
        hypothesis.coarse_overlap >= recovery.selected_coarse_overlap
        for hypothesis in recovery.hypotheses[: recovery.selected_rank]
    )


#: Refined hypotheses the selected-rank regression below runs with.
#:
#: Sixteen rather than the default four, and measured rather than chosen: the
#: best-covering placement on this fixture is spelled by several coarse
#: orientations, and the first genuinely DIFFERENT hypothesis that survives
#: refinement is rank 9. At four and at eight, every hypothesis offered belongs
#: to the placement the seam refuses, no survivor is left, and the recovery
#: stops at "nothing cleared the gates" without ever selecting anything.
_SELECTED_RANK_HYPOTHESES = 16


def test_a_lower_rank_may_not_stand_on_a_refused_higher_ranks_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round-1 defect, as a branch rather than as an inequality.

    The best-covering placement is refused - see
    :func:`_refuse_the_best_covering_placement` for exactly what that seam does
    and does not do - and a lower-ranked, genuinely DIFFERENT placement goes on
    to clear every unchanged gate. Round 1 would have accepted it on the
    ranking's own margin, which was measured on the placement that had just been
    refused and which is comfortably clear here.

    What the recovery does instead is measure the margin of the hypothesis it
    actually selected, against the best genuinely different placement still
    standing beside it - which is the refused one, because a hypothesis the
    gates refused is still the shape saying a placement is there. The selected
    placement covers the reference LESS than the rival it cannot rule out, so
    its own margin lands under 1.0, and the recovery refuses and moves nothing.
    """
    policy = TrackingPolicy(recovery_hypotheses=_SELECTED_RANK_HYPOTHESES)
    assert policy.recovery_ambiguity_margin == AMBIGUITY_MARGIN, "the margin is not lowered"

    tracker = _tracked(policy)
    tracker._policy = policy
    before = _state(tracker)
    _no_candidates(monkeypatch)
    _refuse_the_best_covering_placement(monkeypatch)
    decision = tracker.offer(_sweep()[_OFFERED_FRAME])

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted

    # Rank 0 failed, and a LOWER rank survived every gate. Both halves are
    # needed: either one alone makes the assertion below vacuous.
    assert recovery.hypotheses[0].accepted is False
    assert recovery.selected_rank is not None and recovery.selected_rank > 0
    selected = recovery.hypotheses[recovery.selected_rank]
    assert selected.accepted, "the recorded rank is the hypothesis that survived refinement"

    # The confidence that could have been borrowed really was there.
    assert recovery.coarse_score_unambiguous is True
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.coarse_overlap_margin >= AMBIGUITY_MARGIN

    # And the selected placement's own margin is not. The rival is the refused
    # rank 0 itself, which covers MORE than the selection, so the division lands
    # under one rather than merely under the margin.
    assert recovery.selected_rival_overlap == recovery.hypotheses[0].coarse_overlap
    assert recovery.selected_coarse_overlap == selected.coarse_overlap
    assert recovery.selected_coarse_overlap < recovery.selected_rival_overlap
    assert recovery.selected_overlap_margin == pytest.approx(
        recovery.selected_coarse_overlap / recovery.selected_rival_overlap
    )
    assert recovery.selected_overlap_margin < 1.0
    assert recovery.selected_overlap_unambiguous is False
    assert any(
        "actually selected" in refusal and "cannot lend it confidence" in refusal
        for refusal in recovery.refusals
    ), recovery.refusals


# --- 8. the machinery around the recovery, under the default enabled policy --------------


def test_a_refused_recovery_changes_nothing_about_the_ordinary_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consecutive-refusal streak is what a durable tracking loss is made of.

    ``capture_sequence.TRACKING_LOSS_STREAK`` refusals in a row is the live
    capture's terminal definition of a loss, and it counts REFUSALS - so a
    fallback that turned one of them into a different kind of event, or that
    left state behind on its way past, would move a boundary this module does
    not own. Three offers of evidence no recovery can use, under the default
    recovery-ENABLED policy, must produce exactly the three refusals a tracker
    with the fallback switched off produces, word for word, and must leave the
    tracker exactly as they found it.
    """
    plain = _tracked(TrackingPolicy(recovery_enabled=False))
    enabled = _tracked(TrackingPolicy())
    before = _state(enabled)
    assert _state(plain) == before

    _no_candidates(monkeypatch)
    refusals = [enabled.offer(_plane()) for _ in range(TRACKING_LOSS_STREAK)]
    controls = [plain.offer(_plane()) for _ in range(TRACKING_LOSS_STREAK)]

    assert len(refusals) == TRACKING_LOSS_STREAK
    for refused, control in zip(refusals, controls, strict=True):
        assert not refused.accepted and not control.accepted
        assert refused.failure_type == control.failure_type
        # The words too: this is what reaches the durable tracking-loss record.
        assert refused.refusals == control.refusals
        assert refused.pose is None
    # The recovery ran and is on the record; the refusal is unchanged anyway.
    assert all(decision.recovery is not None for decision in refusals)
    assert all(control.recovery is None for control in controls)
    assert _state(enabled) == before
    assert _state(plain) == before


def test_a_recorded_bridge_still_seeds_the_chain_with_the_recovery_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bridge is decided elsewhere, and the fallback does not reopen it.

    ``seed`` is how a recorded relocalization boundary enters the chain: the
    pose came from the capture's own record, the pair is deliberately not
    fitted, and no candidate, coarse search or recovery is involved. Asserted
    under the default recovery-enabled policy because that is the policy the
    live capture and the offline reconstruction both run.
    """
    tracker = _tracked(TrackingPolicy())
    _no_candidates(monkeypatch)
    bridged = tracker.seed(_sweep()[_OFFERED_FRAME], _true_pose(_OFFERED_FRAME))

    assert bridged.accepted
    assert bridged.recovery is None, "a seeded frame runs no coarse search at all"
    assert bridged.pose == _true_pose(_OFFERED_FRAME)
    assert tracker.frames == _ACCEPTED_FRAMES + 1


def test_the_live_and_offline_paths_reach_the_same_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One implementation, two callers, and the same decision from both.

    The live capture and :func:`~scanner3d.pipeline.reconstruct_sweep` share
    :class:`~scanner3d.tracking.IncrementalTracker`, so a recovery is either the
    same event on both paths or the two paths have drifted. The poses are
    compared exactly rather than approximately - the offline path pre-fits the
    adjacent pairs across workers, so a difference here would be a difference in
    the arithmetic, not in the noise - and the recovery's own record is checked
    for its schema version and for surviving strict JSON, which is what a reader
    comparing a capture directory with a reconstruction actually reads.
    """
    clouds = list(_sweep()[: _ACCEPTED_FRAMES + 1])
    _no_candidates(monkeypatch)

    live_tracker = IncrementalTracker(policy=TrackingPolicy())
    live = [live_tracker.offer(cloud) for cloud in clouds]
    assert live[-1].accepted and live[-1].origin == RECOVERY_ORIGIN
    # The stub is global and conditional on the tracker's own accepted count, so
    # the frames that BUILD the model have to have come through the ordinary
    # path. If the patch had disabled them too, every frame would be a recovery
    # or a refusal and the parity below would be comparing two empty chains.
    assert all(decision.accepted for decision in live)
    assert all(decision.recovery is None for decision in live[:-1])
    # Frame 0 is the ANCHOR: it defines the model frame and is placed without a
    # reference at all, so its origin is not one of the references the ordinary
    # decision prefers between. Asserted separately rather than folded into the
    # set below, which would have to be widened to admit it and would then also
    # admit an anchor appearing anywhere else in the chain.
    assert live[0].origin == ANCHOR_ORIGIN
    assert {decision.origin for decision in live[1:-1]} <= set(PREFERENCE_ORDER)

    offline = reconstruct_sweep(clouds)
    assert offline.frames == len(clouds)
    assert offline.poses == tuple(decision.pose for decision in live if decision.pose is not None)

    tracking = offline.tracking
    assert tracking is not None
    placements = tracking["placements"]
    assert isinstance(placements, list)
    assert placements[-1]["origin"] == RECOVERY_ORIGIN
    assert placements[-1]["recovered"] is True
    assert [placement["origin"] for placement in placements] == [
        decision.origin for decision in live
    ]

    # The same statement on the offline side: exactly one recovery, at the end,
    # and every earlier frame placed by an ordinary reference.
    assert tracking["coarse_recoveries"] == 1
    assert [placement["recovered"] for placement in placements] == [False] * _ACCEPTED_FRAMES + [
        True
    ]

    # DIRECTION, not merely value. A transform written the other way round would
    # still compare equal frame by frame above if both paths were wrong the same
    # way, so the recovered pose is checked against the fixture's own truth.
    recovered_pose = live[-1].pose
    assert recovered_pose is not None
    assert _rotation_error(recovered_pose, _true_pose(_OFFERED_FRAME)) < (
        _MAX_RECOVERED_ERROR_DEGREES
    )

    recovery = live[-1].recovery
    assert recovery is not None
    written = json.dumps(recovery.as_dict(), allow_nan=False)
    assert json.loads(written)["schema_version"] == recovery.schema_version


# --- 9. real incompatible local minima, and what the corrected recovery does -------------

#: How far the WRONG of the two real minima puts the frame's surface from where
#: it truly is, in depth units. Measured on this fixture: 5.535, and the two
#: placements sit 5.847 apart against the unchanged 5.200 agreement gate. Named
#: as a floor rather than an equality because it is a real solve's answer, not
#: arithmetic.
_CARRIED_WRONG_MINIMUM = 5.0

#: How close the RIGHT of the two ordinary minima gets. Measured: 0.334.
_CARRIED_RIGHT_PLACEMENT = 1.0

#: What the RECOVERY has to reach on this same frame. Measured: 0.278, which is
#: better than either ordinary reference managed and 20 times better than the
#: minimum the ordinary decision could not rule out. Named as a gate rather than
#: quoted, so a regression that placed the frame in the wrong basin - or in the
#: right basin sloppily - fails here rather than in a docstring.
_CARRIED_RECOVERED_PLACEMENT = 1.0

#: The material improvement the acceptance contract asks for, as a ratio of the
#: wrong minimum's error to the recovered pose's. Measured: 19.9x. Named at 5x
#: so this asserts a MATERIAL improvement rather than a decimal place.
_CARRIED_MATERIAL_IMPROVEMENT = 5.0


def _carried_local_only() -> tuple[TrackDecision, dict[str, float]]:
    """The task-023 result on this frame: two real minima and a refusal.

    Nothing is stubbed. Ten frames are placed by real ICP, the last of them the
    object's facing surface only, and then one more frame a jump further round.
    Both surviving references clear every gate - convergence, residual, matched,
    used, overlap and travel - and then put the frame 5.847 apart against the
    5.200 agreement gate. That is not a frame that cannot be placed; it is two
    answers, and the ordinary decision correctly refuses to arbitrate them by
    residual.
    """
    sweep = _carried_sweep()
    offered = sweep[-1]
    plain = IncrementalTracker(policy=TrackingPolicy(recovery_enabled=False, keyframe_candidates=0))
    for index, cloud in enumerate(sweep[:-1]):
        assert plain.offer(cloud).accepted, f"the fixture must place frame {index}"
    before = _state(plain)
    local = plain.offer(offered)
    assert _state(plain) == before, "a refusal moves nothing, with or without the fallback"
    survivors = {
        candidate.origin: _placement_error(candidate.pose, offered)
        for candidate in local.candidates
        if candidate.accepted
    }
    return local, survivors


def _carried_offer(
    policy: TrackingPolicy,
) -> tuple[TrackDecision, tuple[object, ...], tuple[object, ...]]:
    """The same frame offered to a recovery-enabled tracker under ``policy``.

    The accepted chain is always built under the DEFAULT policy, so every
    control below differs from the acceptance case in exactly the one recovery
    bound it names and in nothing about the model behind it.
    """
    sweep = _carried_sweep()
    tracker = IncrementalTracker(policy=TrackingPolicy(keyframe_candidates=0))
    for cloud in sweep[:-1]:
        assert tracker.offer(cloud).accepted
    tracker._policy = policy
    before = _state(tracker)
    decision = tracker.offer(sweep[-1])
    return decision, _state(tracker), before


def test_two_real_references_settle_in_incompatible_minima_and_the_recovery_places_the_frame() -> (
    None
):
    """The case the whole fallback exists for, end to end, with nothing stubbed.

    This is the acceptance regression the rest of this file's refusals are
    measured against, and it is a COMPARISON: the same frame, the same accepted
    model, once without the fallback and once with it.

    Without it - task 023's behaviour - the adjacent reference has slid along
    the weak frame's facing surface and the model reference has not, both clear
    every gate, and the decision refuses because they are two answers. With it,
    the bounded coarse search re-seeds the refinement, both ambiguity verdicts
    clear, two disjoint pose-diverse partitions agree, and the frame is placed
    an order of magnitude closer to the truth than the minimum the ordinary
    decision could not rule out.
    """
    sweep = _carried_sweep()
    offered = sweep[-1]
    local, survivors = _carried_local_only()

    assert not local.accepted
    assert local.failure_type == TRACK_DISAGREEMENT_TYPE
    assert set(survivors) == {"adjacent", "model"}, survivors
    # Two REAL minima: one right, one wrong, both through every gate.
    assert survivors["model"] < _CARRIED_RIGHT_PLACEMENT, survivors
    assert survivors["adjacent"] > _CARRIED_WRONG_MINIMUM, survivors

    recovered, after, before = _carried_offer(TrackingPolicy(keyframe_candidates=0))

    assert recovered.accepted
    assert recovered.origin == RECOVERY_ORIGIN
    assert recovered.pose is not None
    error = _placement_error(recovered.pose, offered)
    assert error < _CARRIED_RECOVERED_PLACEMENT, error
    # The material improvement the contract asks for, over the minimum the
    # ordinary decision could not rule out.
    assert survivors["adjacent"] / error > _CARRIED_MATERIAL_IMPROVEMENT, (survivors, error)
    assert after != before, "an ACCEPTED frame is the one case that does move state"

    recovery = recovered.recovery
    assert recovery is not None and recovery.attempted and recovery.accepted
    assert recovery.triggered_by == TRACK_DISAGREEMENT_TYPE
    # Both verdicts, on both quantities, and neither substituted for the other.
    assert recovery.coarse_score_unambiguous is True
    assert recovery.coarse_score_margin >= AMBIGUITY_MARGIN
    assert recovery.coarse_overlap_unambiguous is True
    assert recovery.selected_overlap_unambiguous is True
    # Two disjoint partitions, each spanning measured viewpoints, one of them
    # confirming a placement it did not search for.
    assert len(recovery.partitions) == 2
    assert not set(recovery.partitions[0].frames) & set(recovery.partitions[1].frames)
    assert all(
        partition.pose_span >= RECOVERY_MIN_PARTITION_POSE_SPAN for partition in recovery.partitions
    )
    assert recovery.agreeing == (RECOVERY_ORIGIN, RECOVERY_CONFIRM_ORIGIN)
    assert len(recovery.confirmation_disagreements) == 1


def test_the_same_frame_refuses_when_the_second_disjoint_partition_is_taken_away() -> None:
    """One reference set cannot confirm itself, and the fixture proves it.

    The scene, the model and the offered frame are the acceptance case above,
    unchanged - the accepted chain is built by the same default policy, so this
    differs from the acceptance case in the recovery's own bounds and in nothing
    else. Each partition is asked for five accepted viewpoints instead of four,
    which is one more than the nine eligible accepted frames can deal into two
    disjoint sets: the second partition comes up short and is dropped rather
    than padded, and one reference set cannot confirm itself. No gate moves.
    """
    policy = TrackingPolicy(
        keyframe_candidates=0,
        recovery_partition_frames=RECOVERY_PARTITION_FRAMES + 1,
        recovery_min_partition_frames=RECOVERY_PARTITION_FRAMES + 1,
    )
    decision, after, before = _carried_offer(policy)

    assert not decision.accepted
    assert decision.pose is None
    assert after == before
    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert len(recovery.partitions) < RECOVERY_PARTITIONS
    assert any("share no accepted frame" in refusal for refusal in recovery.refusals), (
        recovery.refusals
    )
    assert recovery.work.coarse_searches == 0, "refused before paying for a coarse search"


#: Degrees between the accepted frames of :func:`_carried_one_viewpoint_sweep`.
#:
#: Small enough that every partition dealt from that chain spans under the
#: 1.0 keyframe motion gate - measured: 0.644 - and no smaller than that, so the
#: chain is still a chain of real distinct solves rather than eleven copies of
#: one cloud.
_CARRIED_ONE_VIEWPOINT_STEP = 0.08


@functools.cache
def _carried_one_viewpoint_sweep() -> tuple[tuple[Point, ...], ...]:
    """The incompatible-minima case with the operator's walk taken out of it.

    The object, the weak facing-only accepted frame, the offered frame and the
    nine-degree jump onto it are all at the SAME absolute angles as
    :func:`_carried_sweep`. The only thing changed is where the nine ordinary
    accepted frames sit: instead of walking eight degrees at a time up to the
    weak frame they are bunched against it, so the model behind the offer is
    built from one viewpoint seen repeatedly rather than from a walk round the
    object.

    That is the multi-view support the acceptance case rests on, removed
    causally rather than by asking for a partition shape the policy forbids.
    """
    last = _CARRIED_STEP_DEGREES * (_ACCEPTED_FRAMES - 1)
    frames = [
        _carried_view(last - (_ACCEPTED_FRAMES - 1 - index) * _CARRIED_ONE_VIEWPOINT_STEP)
        for index in range(_ACCEPTED_FRAMES - 1)
    ]
    frames.append(_carried_view(last, facing_only=True))
    frames.append(_carried_view(last + _CARRIED_JUMP_DEGREES))
    return tuple(frames)


def test_a_partition_of_one_accepted_frame_is_not_a_policy_this_tracker_will_run() -> None:
    """The blind partition is refused where it is written, not where it is used.

    A reference assembled from a single accepted frame is blind in the direction
    that frame is blind in, which is the same no-second-opinion condition the
    model-only rescue gate exists for. So there is no reachable branch in which
    a recovery confirms a placement against one: the policy that would ask for
    one cannot be constructed, and the refusal names the minimum it protects.
    """
    with pytest.raises(TrackingNotAllowed) as refused:
        TrackingPolicy(recovery_partition_frames=1)

    assert "recovery_partition_frames 1" in str(refused.value)
    assert "frames every partition must span" in str(refused.value)


def test_the_same_frame_refuses_when_the_partitions_hold_one_viewpoint() -> None:
    """Multi-view support removed from the acceptance case, causally.

    Same object, same weak accepted frame, same offered frame, same nine-degree
    jump, same policy and same gates as the acceptance regression above - and
    the nine ordinary accepted frames bunched into one viewpoint instead of
    spread round the object. The ordinary decision still refuses this frame for
    the same reason it does there, so the recovery is still offered it; what it
    no longer has is two references that see the object from anywhere different.

    It refuses before it pays for a coarse search, which is the honest order:
    the reference sets are what the confirmation would be measured against, and
    a search seeded to be confirmed by nothing is work not worth doing.
    """
    sweep = _carried_one_viewpoint_sweep()
    tracker = IncrementalTracker(policy=TrackingPolicy(keyframe_candidates=0))
    for index, cloud in enumerate(sweep[:-1]):
        assert tracker.offer(cloud).accepted, f"the fixture must place frame {index}"
    before = _state(tracker)

    decision = tracker.offer(sweep[-1])

    # The SAME ordinary refusal the acceptance case recovers from.
    assert not decision.accepted
    assert decision.failure_type == TRACK_DISAGREEMENT_TYPE
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert recovery.triggered_by == TRACK_DISAGREEMENT_TYPE
    # Two disjoint partitions, four accepted frames each: every COUNTING gate is
    # satisfied, and only the measured span tells this chain from a walk.
    assert len(recovery.partitions) == RECOVERY_PARTITIONS
    assert all(len(partition.frames) >= 4 for partition in recovery.partitions)
    assert all(
        partition.pose_span < RECOVERY_MIN_PARTITION_POSE_SPAN for partition in recovery.partitions
    )
    assert any("accepted VIEWPOINTS" in refusal for refusal in recovery.refusals), recovery.refusals
    assert recovery.work.coarse_searches == 0, "refused before paying for a coarse search"


# --- 12. the confirmation refusal, emitted once ------------------------------------------

#: The one sentence the independent-confirmation disagreement refusal ends with.
#: Pinned as a constant here because the claim being asserted is about how many
#: times an operator is told this, not about its wording.
_CONFIRMATION_DISAGREEMENT_SENTENCE = (
    "Two reference sets that do not agree are two answers, not a recovery."
)


def _displace_the_confirmation(monkeypatch: pytest.MonkeyPatch, *, apart: float) -> None:
    """SEAM: the confirming reference's solve settles ``apart`` from the searching one.

    The confirmation is refined for real and judged by the real rescue and
    ordinary gates; what is replaced afterwards is its POSE, so the record the
    agreement decision is handed is a confirmation that cleared its own gates
    and landed somewhere else. Like
    :func:`_displace_the_second_refined_placement`, this SYNTHESISES the
    accepted candidate record at the agreement boundary: the displaced pose
    itself is never put back through :meth:`IncrementalTracker._candidate`'s
    gates, so what this proves is the agreement decision, its emitted text and
    the absence of mutation - not that this fixture naturally produces a
    confirming reference that both clears every gate and disagrees.
    """
    real = IncrementalTracker._candidate
    offset = Rigid.of(IDENTITY_MATRIX, (apart, 0.0, 0.0))

    def gated(
        self: IncrementalTracker,
        index: int,
        source: tuple[Point, ...],
        *,
        origin: str,
        **kwargs: object,
    ) -> object:
        candidate = real(self, index, source, origin=origin, **kwargs)  # type: ignore[arg-type]
        if origin != RECOVERY_CONFIRM_ORIGIN or candidate is None or not candidate.accepted:
            return candidate
        return replace(candidate, pose=candidate.pose.compose(offset))

    monkeypatch.setattr(IncrementalTracker, "_candidate", gated)


def test_a_disagreeing_confirmation_refuses_and_says_so_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two reference sets that do not agree are two answers - said once.

    The acceptance fixture and the default policy, with the one seam above on
    the confirming solve's pose. The recovery refuses by name, nothing moves,
    and the sentence the operator is given appears exactly once in the recovery
    record and exactly once in the line
    :meth:`~scanner3d.tracking.TrackDecision.message` builds from it. A refusal
    that repeats its own reasoning reads as two findings rather than one, so
    the count is asserted rather than the mere presence of the words.
    """
    gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
    tracker = _tracked(TrackingPolicy())
    before = _state(tracker)
    _no_candidates(monkeypatch)
    _displace_the_confirmation(monkeypatch, apart=3.0 * gate)

    decision = tracker.offer(_sweep()[_OFFERED_FRAME])

    assert not decision.accepted
    assert decision.pose is None
    assert _state(tracker) == before

    recovery = decision.recovery
    assert recovery is not None and recovery.attempted and not recovery.accepted
    assert recovery.agreeing == (RECOVERY_ORIGIN,), "the confirmation never joined"
    assert len(recovery.confirmation_disagreements) == 1
    assert recovery.confirmation_disagreements[0] > gate
    refused = "\n".join(recovery.refusals)
    assert f"{gate:.3f} agreement gate" in refused, recovery.refusals
    assert refused.count(_CONFIRMATION_DISAGREEMENT_SENTENCE) == 1, recovery.refusals
    assert decision.message().count(_CONFIRMATION_DISAGREEMENT_SENTENCE) == 1, decision.message()
