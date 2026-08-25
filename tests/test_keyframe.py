"""Tests for :mod:`scanner3d.keyframe`.

What is at stake
-----------------
The whole point of keyframing is that it chains fewer LINKS than plain
frame-to-frame registration, and error accumulates with the number of links,
not with elapsed time or how far the object turned. Every test after the
first uses a deterministic FAKE aligner - :func:`_scripted_aligner`, injected
through ``select_keyframes``'s own ``align`` parameter - so that selection
logic, composition and the degenerate cases can be checked exactly, with no
rendering and no real registration in the loop. The fake models constant-rate
1D motion and exploits :func:`scanner3d.pipeline.frame_travel`'s own
arithmetic directly: translating ANY point by ``(d, 0, 0)`` moves it exactly
``d``, so a scripted offset's travel is exactly the number scripted for it,
with no geometry involved at all.

Test 1 is the one exception, and deliberately so: a fake aligner could be
scripted to make keyframing look good by construction, which would prove
nothing about the actual module. It renders a real synthetic turntable sweep
(:mod:`scanner3d.simulate`) of a box tilted to present several faces at once,
runs REAL point-to-plane ICP as the aligner, and compares the keyframe-
chained trajectory against a plain frame-to-frame chain, both measured
against the ground-truth pose :func:`scanner3d.simulate.render` itself
guarantees. See that test's own docstring for how "plain chaining" is built
independently of :func:`~scanner3d.keyframe.select_keyframes` - not a variant
call to it - so a keyframing implementation that silently did nothing (cut
every frame, or never cut at all) would make the two trajectories converge
and the assertion would correctly stop passing.

Runtime
-------
Test 1 is the only one that renders or runs real ICP; it is the dominant
cost of this file (a handful of seconds) and was tuned - frame count, box
tilt, downsample fraction - to be the smallest scene that still shows a
material, repeatable difference (see its own comments for the tuning
history in brief). Every other test uses the scripted aligner and runs in a
small fraction of a second. Measured total for this file: comfortably under
the requested budget.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import pytest

from scanner3d.errors import ScannerError
from scanner3d.geometry import IDENTITY_MATRIX, Point, Rigid, rotation_about_axis
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.keyframe import (
    KEYFRAME_TRAVEL_MULTIPLE,
    SCHEMA_VERSION,
    KeyframeNotAllowed,
    poses_from_keyframes,
    select_keyframes,
)
from scanner3d.pipeline import NOISE_FLOOR
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.simulate import Box, points_of, render

# ============================================================================
# 1. THE HEADLINE: keyframing measurably beats plain chaining, real geometry,
#    known ground truth.
# ============================================================================

#: An oblique axis, not aligned with any of the box's own symmetry axes -
#: matches the reasoning in tests/test_pipeline.py and tests/test_revisit.py
#: for why that matters. Duplicated here rather than imported, matching this
#: codebase's established policy (see test_pipeline.py's own comment) of
#: keeping one test file's fixtures independent of another's.
_AXIS: Point = (0.55, 0.6, 0.58)
_THROUGH: Point = (0.0, 0.0, 60.0)

#: A PLAIN box placed straight in front of the camera shows one dominant
#: face - fine for a coarse rotation estimate, but both point-to-point and
#: point-to-plane ICP need more than one face's worth of surface normal
#: directions to constrain a rotation tightly (point-to-plane failed outright
#: with a "singular in rotation z" refusal on the untilted box - a single
#: near-planar view has no information about rotation around its own
#: normal). Tilting the box's OWN placement, once, before any sweep pose is
#: applied, keeps three faces in view throughout the sweep and was what
#: took the recovered single-step rotation from a consistent ~18% shortfall
#: (a real, if unintuitive, point-to-point ICP artefact of thin overlap - see
#: the module docstring's runtime note) down to sub-tenth-of-a-degree
#: accuracy with point-to-plane. Measured, not guessed - see the module
#: docstring.
_TILT = rotation_about_axis((1.0, 0.3, 0.2), math.radians(35.0), through=_THROUGH)
_BOX = Box(
    half_extents=(12.0, 16.0, 9.0),
    pose=_TILT.compose(Rigid.of(IDENTITY_MATRIX, _THROUGH)),
)

#: Coarse enough to keep real ICP fast (a few hundred points per frame), fine
#: enough that point-to-plane still has real surface structure to match
#: against. Tuned empirically alongside the frame count and sweep angle
#: below - finer sampling did not change the result, only the runtime.
_VOXEL_FRACTION = 1.0 / 25.0

#: 18 frames over 20 degrees: oversampled on purpose, matching the module
#: docstring's own "over-sampled sweep" scenario - each single step is small
#: enough that a naive chain has many links to accumulate error along, and
#: 0.5 is well below what a handful of steps' worth of travel adds up to, so
#: keyframing has real cuts to make rather than degenerating to "every frame"
#: or "one keyframe".
_HEADLINE_FRAMES = 18
_HEADLINE_TOTAL_DEGREES = 20.0
_HEADLINE_STEP_DEGREES = _HEADLINE_TOTAL_DEGREES / (_HEADLINE_FRAMES - 1)
_HEADLINE_THRESHOLD = 0.5


def _rendered_frame(angle_deg: float, *, seed: int) -> tuple[tuple[Point, ...], Rigid]:
    """One rendered, downsampled frame at ``angle_deg`` about ``_AXIS``, and the
    ground-truth pose :func:`scanner3d.simulate.render` guarantees for it.

    ``pose`` moves the SCENE - see that function's own docstring - so a point
    that sits at ``Q`` in frame 0 (pose is the identity there) sits at
    ``pose.apply(Q)`` in this frame. The transform carrying THIS frame's
    points back to frame 0's is therefore ``pose.inverse()``, which is what
    the headline test compares recovered trajectories against.
    """
    pose = rotation_about_axis(_AXIS, math.radians(angle_deg), through=_THROUGH)
    frame = render([_BOX], pose=pose, seed=seed)
    points = tuple(points_of(frame))
    extent = cloud_extent(points)
    return voxel_downsample(points, extent * _VOXEL_FRACTION), pose


def _real_align(source: Sequence[Point], target: Sequence[Point]) -> Rigid:
    """The aligner a real caller of ``select_keyframes`` would inject: point-to-
    plane ICP, ``keep_fraction`` matching :mod:`scanner3d.pipeline`'s own
    calibrated choice rather than :func:`scanner3d.icp_plane.icp_point_to_plane`'s
    less-accurate library default (see that module's own docstring on why 0.8
    recovers materially more of the true rotation than 0.6 does).
    """
    return icp_point_to_plane(source, target, keep_fraction=0.8).transform


def _angular_error(estimated: Rigid, true: Rigid) -> float:
    """The geodesic angle between two rotations - see :attr:`Rigid.rotation_angle`."""
    return estimated.compose(true.inverse()).rotation_angle


def test_keyframing_measurably_reduces_drift_against_known_ground_truth():
    """A real capture is not available for a unit test; a synthetic one with a
    ground truth :func:`scanner3d.simulate.render` GUARANTEES is the next best
    thing, and lets the comparison be made in degrees of ACTUAL error rather
    than by proxy.

    The naive trajectory is built by hand - every frame aligned to its
    immediate predecessor and composed into a running product - which is
    exactly plain chaining, never a call to ``select_keyframes`` at all. If
    keyframing were bypassed (cutting on every frame, which reduces nothing,
    or never cutting at all, which is a different failure the guard tests
    below catch) the keyframe trajectory would collapse onto - or towards -
    this same naive one, and the margin asserted at the end would not hold.
    """
    clouds: list[tuple[Point, ...]] = []
    ground_truth: list[Rigid] = []
    for index in range(_HEADLINE_FRAMES):
        cloud, pose = _rendered_frame(index * _HEADLINE_STEP_DEGREES, seed=index)
        clouds.append(cloud)
        ground_truth.append(pose)

    naive_pose = [Rigid.identity()]
    for index in range(1, _HEADLINE_FRAMES):
        offset = _real_align(clouds[index], clouds[index - 1])
        naive_pose.append(naive_pose[-1].compose(offset))

    selection = select_keyframes(clouds, _real_align, travel_threshold=_HEADLINE_THRESHOLD)
    # A genuine reduction happened - not every frame became a keyframe (that
    # would just be the naive chain again) and not only one did either (that
    # would not be testing accumulation, just a single big jump).
    assert 1 < len(selection.keyframes) < _HEADLINE_FRAMES, (
        f"expected a real subset of keyframes, got {selection.keyframes}"
    )

    keyframe_pose = {selection.keyframes[0]: Rigid.identity()}
    for previous, current in itertools.pairwise(selection.keyframes):
        offset = _real_align(clouds[current], clouds[previous])
        keyframe_pose[current] = keyframe_pose[previous].compose(offset)
    recovered = poses_from_keyframes(
        selection, [keyframe_pose[keyframe] for keyframe in selection.keyframes]
    )

    true_final = ground_truth[-1].inverse()
    naive_error = _angular_error(naive_pose[-1], true_final)
    keyframe_error = _angular_error(recovered[-1], true_final)

    # Materially better, not just numerically different - the module's own
    # docstring reports about 2.4x on a real capture; 0.7x leaves comfortable
    # margin below what this scene actually measures (about 2.5x) while
    # still being a real, stated bar rather than "any improvement at all".
    assert keyframe_error < naive_error * 0.7, (
        f"keyframe chain error {math.degrees(keyframe_error):.3f} deg was not materially "
        f"better than the naive chain's {math.degrees(naive_error):.3f} deg"
    )


# ============================================================================
# Scripted aligner: deterministic, no rendering, models constant-rate motion
# ============================================================================


class _ScriptedAlignmentFailed(ScannerError):
    """Raised by :func:`_scripted_aligner` to simulate a frame that will not
    align - a :class:`~scanner3d.errors.ScannerError`, which is exactly what
    ``select_keyframes`` catches to force a keyframe.
    """


def _indexed_cloud(index: int) -> tuple[Point, ...]:
    """A placeholder cloud carrying nothing but its own frame index.

    :func:`_scripted_aligner` reads the index back off the point to decide
    what to return, and :func:`~scanner3d.pipeline.frame_travel` computes the
    same travel for it as for any other cloud - a single point's distance
    moved is independent of where that point started. What is IN the cloud
    never matters to either; only that it carries an index round-trips.
    """
    return ((float(index), 0.0, 0.0),)


def _travel_rigid(magnitude: float) -> Rigid:
    """A pure translation whose :func:`~scanner3d.pipeline.frame_travel` is
    exactly ``abs(magnitude)`` for ANY cloud - translating a point by
    ``(d, 0, 0)`` moves it exactly ``d``, regardless of where it started.
    """
    return Rigid.of(IDENTITY_MATRIX, (magnitude, 0.0, 0.0))


def _scripted_aligner(
    per_step: float, *, fail_at: frozenset[int] = frozenset()
) -> tuple[object, list[tuple[int, int]]]:
    """A deterministic fake ``align(source, target)`` modelling constant-rate
    1D motion, plus the list of ``(source_index, target_index)`` pairs it was
    actually called with - so a test can check not just the OUTCOME but which
    pairs ``select_keyframes`` chose to align, e.g. that a non-keyframe was
    measured against its KEYFRAME and never against the previous frame.

    Returns the true offset scaled by how many indices apart the two frames
    are - source index ``i`` in ``fail_at`` raises instead.
    """
    calls: list[tuple[int, int]] = []

    def align(source: Sequence[Point], target: Sequence[Point]) -> Rigid:
        source_index = round(source[0][0])
        target_index = round(target[0][0])
        calls.append((source_index, target_index))
        if source_index in fail_at:
            raise _ScriptedAlignmentFailed(f"frame {source_index} deliberately refuses to align")
        return _travel_rigid(per_step * (source_index - target_index))

    return align, calls


# ============================================================================
# 2. Nothing is lost: every original frame gets a pose, composed once - not
#    a running product.
# ============================================================================


def test_poses_from_keyframes_composes_one_measured_offset_not_a_running_product():
    """The property the whole module exists for: a non-keyframe's pose is its
    keyframe's pose composed with the SINGLE offset measured directly between
    them.

    Checked two ways, both independent of what ``poses_from_keyframes``
    itself computes: the attachment's own ``offset`` is compared against a
    value recomputed from the scripted per-step rate (proving it was
    measured directly against the keyframe, not accumulated through
    intermediate frames - a running product would drift from this formula as
    soon as more than one link separated a frame from its keyframe), and the
    composed pose is compared against ``chosen_pose.compose(that offset)``
    for keyframe poses chosen to be unmistakable, not derived from anything
    the scripted aligner returned.
    """
    frame_count = 14
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    per_step = 3.0
    align, _calls = _scripted_aligner(per_step)
    selection = select_keyframes(clouds, align, travel_threshold=10.0)
    assert 1 < len(selection.keyframes) < frame_count, "need a real mix of keyframes and members"

    # Distinct, recognisable per-keyframe poses that bear no relationship to
    # the scripted per-step offsets - a bug that accidentally re-derived
    # them (e.g. by chaining through intermediate frames instead of using
    # the caller-supplied poses) would not be able to reproduce these.
    keyframe_poses = [
        Rigid.of(IDENTITY_MATRIX, (0.0, 1000.0 * (position + 1), 0.0))
        for position in range(len(selection.keyframes))
    ]
    keyframe_pose_by_index = dict(zip(selection.keyframes, keyframe_poses, strict=True))
    recovered = poses_from_keyframes(selection, keyframe_poses)

    assert len(recovered) == frame_count
    for attachment, pose in zip(selection.attachments, recovered, strict=True):
        expected_offset = _travel_rigid(per_step * (attachment.index - attachment.keyframe))
        assert attachment.offset == expected_offset, (
            f"frame {attachment.index}'s measured offset was not the one directly against "
            "its own keyframe"
        )
        expected_pose = keyframe_pose_by_index[attachment.keyframe].compose(expected_offset)
        assert pose == expected_pose, f"frame {attachment.index}'s composed pose is wrong"


def test_a_non_keyframe_is_aligned_against_its_keyframe_never_the_previous_frame():
    """The call log itself, not just the arithmetic: every non-keyframe call's
    target index must be a KEYFRAME index, and specifically the current one -
    never ``index - 1``, which is what plain chaining would do instead.
    """
    frame_count = 20
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    align, calls = _scripted_aligner(per_step=1.0)
    selection = select_keyframes(clouds, align, travel_threshold=4.0)
    assert 1 < len(selection.keyframes) < frame_count

    keyframe_set = set(selection.keyframes)
    for source_index, target_index in calls:
        assert target_index in keyframe_set, (
            f"frame {source_index} was aligned against {target_index}, which is not a keyframe"
        )
        # Only for a frame that stayed a NON-keyframe, which is what this test
        # is named for. A frame that BECOMES a keyframe is recorded as its own,
        # with an identity offset, because its pose comes from the keyframe
        # chain rather than from an offset to the previous one.
        attachment = selection.attachments[source_index]
        if not attachment.is_keyframe:
            assert target_index == attachment.keyframe


# ============================================================================
# 3. It adapts to speed: same total motion, wildly different frame counts,
#    a similar number of keyframes.
# ============================================================================


def test_a_fast_and_a_slow_sweep_covering_the_same_motion_cut_a_similar_number_of_keyframes():
    """This is the property that makes the module settings-free: a fast sweep
    (few, large steps) and a slow one (many, small steps) covering the SAME
    total motion should settle on close to the same keyframe count, even
    though one has far more frames to offer.
    """
    total_motion = 40.0
    threshold = 12.0

    fast_frames = 9
    slow_frames = 81
    assert slow_frames > fast_frames * 5, "the two sweeps must genuinely differ in frame count"

    fast_step = total_motion / (fast_frames - 1)
    slow_step = total_motion / (slow_frames - 1)
    # Both steps must be smaller than the threshold, or every frame becomes a
    # keyframe regardless of speed and the test would not be exercising the
    # adaptive behaviour at all.
    assert fast_step < threshold and slow_step < threshold

    fast_align, _ = _scripted_aligner(fast_step)
    slow_align, _ = _scripted_aligner(slow_step)
    fast_selection = select_keyframes(
        [_indexed_cloud(i) for i in range(fast_frames)], fast_align, travel_threshold=threshold
    )
    slow_selection = select_keyframes(
        [_indexed_cloud(i) for i in range(slow_frames)], slow_align, travel_threshold=threshold
    )

    assert abs(len(fast_selection.keyframes) - len(slow_selection.keyframes)) <= 1, (
        f"fast sweep cut {len(fast_selection.keyframes)} keyframes, slow sweep cut "
        f"{len(slow_selection.keyframes)}, for the same total motion"
    )


# ============================================================================
# 4. The degenerate cases.
# ============================================================================


def test_a_stationary_sequence_is_cut_by_max_frames_per_keyframe_not_by_travel():
    """No motion at all must not make every frame a keyframe - that would be
    exactly the failure ``MAX_FRAMES_PER_KEYFRAME`` exists to prevent (see the
    module docstring: "A capture of a stationary object would otherwise chain
    hundreds of frames to one keyframe"). The cut spacing must match
    ``max_frames_per_keyframe`` exactly, and the "EVERY frame became a
    keyframe" limitation - which correctly fires only when nothing was
    saved - must be absent, since here plenty was.
    """
    frame_count = 23
    max_per_keyframe = 5
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    align, _ = _scripted_aligner(per_step=0.0)  # zero true motion, ever
    selection = select_keyframes(
        clouds, align, travel_threshold=1.0, max_frames_per_keyframe=max_per_keyframe
    )
    assert selection.keyframes == tuple(range(0, frame_count, max_per_keyframe))
    limitations = selection.limitations
    assert not any("EVERY frame became a keyframe" in limitation for limitation in limitations)
    assert not any("could not be aligned" in limitation for limitation in limitations)


def test_a_frame_that_will_not_align_becomes_a_keyframe_and_is_counted():
    frame_count = 10
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    align, _ = _scripted_aligner(per_step=1.0, fail_at=frozenset({4}))
    # A threshold travel could never reach, so ONLY the failure forces a cut.
    selection = select_keyframes(clouds, align, travel_threshold=1000.0)

    assert 4 in selection.keyframes
    failed_attachment = selection.attachments[4]
    assert failed_attachment.is_keyframe
    assert failed_attachment.keyframe == 4
    assert failed_attachment.offset == Rigid.identity()
    assert any(
        "1 frame(s) could not be aligned" in limitation for limitation in selection.limitations
    )


def test_several_frames_that_will_not_align_are_all_counted():
    frame_count = 12
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    align, _ = _scripted_aligner(per_step=1.0, fail_at=frozenset({3, 7}))
    selection = select_keyframes(clouds, align, travel_threshold=1000.0)
    assert {3, 7} <= set(selection.keyframes)
    assert any(
        "2 frame(s) could not be aligned" in limitation for limitation in selection.limitations
    )


# ============================================================================
# 5. link_reduction.
# ============================================================================


def test_link_reduction_matches_the_ratio_of_links_actually_saved():
    frame_count = 25
    max_per_keyframe = 6
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    # A threshold travel could never reach, so the cut is forced entirely by
    # max_frames_per_keyframe, at a spacing this test controls exactly.
    align, _ = _scripted_aligner(per_step=1.0)
    selection = select_keyframes(
        clouds, align, travel_threshold=1000.0, max_frames_per_keyframe=max_per_keyframe
    )
    expected_keyframe_count = len(range(0, frame_count, max_per_keyframe))
    assert len(selection.keyframes) == expected_keyframe_count
    expected_reduction = (frame_count - 1) / (expected_keyframe_count - 1)
    assert selection.link_reduction == pytest.approx(expected_reduction)


def test_link_reduction_is_1_when_every_frame_became_a_keyframe():
    frame_count = 11
    clouds = [_indexed_cloud(index) for index in range(frame_count)]
    align, _ = _scripted_aligner(per_step=1000.0)  # always past any sane threshold
    selection = select_keyframes(clouds, align, travel_threshold=1.0)
    assert selection.keyframes == tuple(range(frame_count))
    assert selection.link_reduction == pytest.approx(1.0)


# ============================================================================
# 6. Guards.
# ============================================================================


def _identity_aligner(source: Sequence[Point], target: Sequence[Point]) -> Rigid:
    return Rigid.identity()


def test_fewer_than_two_frames_is_refused():
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes([], _identity_aligner)
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes([_indexed_cloud(0)], _identity_aligner)


def test_non_positive_travel_threshold_is_refused():
    clouds = [_indexed_cloud(index) for index in range(3)]
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes(clouds, _identity_aligner, travel_threshold=0.0)
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes(clouds, _identity_aligner, travel_threshold=-1.0)


def test_max_frames_per_keyframe_below_one_is_refused():
    clouds = [_indexed_cloud(index) for index in range(3)]
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes(clouds, _identity_aligner, max_frames_per_keyframe=0)
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes(clouds, _identity_aligner, max_frames_per_keyframe=-3)


def test_a_non_callable_aligner_is_refused():
    clouds = [_indexed_cloud(index) for index in range(3)]
    with pytest.raises(KeyframeNotAllowed):
        select_keyframes(clouds, "not callable")


def test_poses_from_keyframes_refuses_a_pose_count_that_does_not_match_the_keyframes():
    clouds = [_indexed_cloud(index) for index in range(6)]
    align, _ = _scripted_aligner(per_step=1.0)
    selection = select_keyframes(clouds, align, travel_threshold=1000.0)
    assert len(selection.keyframes) == 1  # sanity: exactly one pose is required
    with pytest.raises(KeyframeNotAllowed):
        poses_from_keyframes(selection, [])
    with pytest.raises(KeyframeNotAllowed):
        poses_from_keyframes(selection, [Rigid.identity(), Rigid.identity()])


# ============================================================================
# Bonus: the documented default threshold is the number the module docstring
# actually claims.
# ============================================================================


def test_the_default_travel_threshold_is_what_the_module_docstring_claims():
    """The module docstring's own worked example says "threshold 5.85" - a
    claim about the CODE, not merely about prose, and worth pinning down so a
    change to either constant it is built from is caught here rather than
    only in a stale comment.
    """
    clouds = [_indexed_cloud(index) for index in range(2)]
    align, _ = _scripted_aligner(per_step=0.0)
    selection = select_keyframes(clouds, align)
    assert selection.travel_threshold == pytest.approx(KEYFRAME_TRAVEL_MULTIPLE * NOISE_FLOOR)
    assert selection.travel_threshold == pytest.approx(5.85)


def test_schema_version_and_as_dict_are_internally_consistent():
    clouds = [_indexed_cloud(index) for index in range(5)]
    align, _ = _scripted_aligner(per_step=1.0)
    selection = select_keyframes(clouds, align, travel_threshold=1000.0, max_frames_per_keyframe=2)
    record = selection.as_dict()
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["frames"] == 5
    assert record["keyframes"] == len(selection.keyframes)
    assert record["keyframe_indices"] == list(selection.keyframes)
    assert record["link_reduction"] == pytest.approx(selection.link_reduction)
