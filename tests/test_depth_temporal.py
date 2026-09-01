"""Pose-compensated temporal completion, proved conservative rather than merely dense.

``src/scanner3d/depth_temporal.py`` carries the *original raw samples* of other
posed frames into a target camera and asks whether they agree about a sample
the target did not return. This file owns the properties underneath that
claim, on hand-built synthetic frames small enough to check by hand:

* the pose composition ``temporal_transform`` builds is target-inverse-after-
  source and not the other way round, on a motion whose answer is known
  analytically;
* a target sample the sensor returned is never touched;
* a reconstruction only ever rests on another frame's own raw measurement,
  never on the target itself and never on a hole;
* an accepted spatial estimate is confirmed, contradicted or left alone by
  exactly the same evidence a run with no estimate would have had - never
  averaged with it;
* the confidence formula is monotone in the direction each of its terms
  claims and is capped strictly below what a measurement carries;
* the work is bounded by something derivable before the run;
* a result whose own labels are legal but whose value the evidence does not
  give is refused by recomputation, not merely by shape.

Nothing here touches a scanner, a capture, a display server, a GPU, the
network or the clock.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import pytest

from scanner3d.compute import ComputeBackendUnavailable, ComputeBackendUnknown
from scanner3d.depth_completion import MAX_CODE, CompletedPatch, CompletionPolicy, DepthPatch
from scanner3d.depth_completion import complete_patch as complete_spatially
from scanner3d.depth_temporal import (
    AUTO_BACKEND,
    CONFIRMATION_CONFIRMED,
    CONFIRMATIONS,
    DEFAULT_POLICY,
    MAX_CANDIDATES_PER_TARGET,
    MAX_FOOTPRINT_RADIUS,
    MAX_RECONSTRUCTION_CONFIDENCE,
    MAX_SUPPORT_FRAMES,
    MAX_WORK_UNITS,
    OBSERVATION_REJECTIONS,
    PYTHON_BACKEND,
    REJECTION_REASONS,
    STATE_TEMPORALLY_RECONSTRUCTED,
    CameraModel,
    CameraUnusable,
    FrameUnusable,
    PolicyUnusable,
    PoseRecord,
    PoseUnusable,
    ResultUnusable,
    TemporalEvidence,
    TemporalFrame,
    TemporalPolicy,
    TemporalWork,
    WorkTooLarge,
    complete_temporally,
    project_observation,
    select_backend,
    temporal_confidence,
    temporal_transform,
    viewpoint_separation,
)
from scanner3d.geometry import IDENTITY_MATRIX, Rigid

# --- small builders --------------------------------------------------------------------


def _camera(width: int, *, fx: float = 50.0, depth_scale: float = 1.0) -> CameraModel:
    """A pinhole camera centred on a ``width x width`` patch, in whole-code units."""
    centre = (width - 1) / 2.0
    return CameraModel(fx=fx, fy=fx, cx=centre, cy=centre, depth_scale=depth_scale)


def _pose(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    *,
    accepted: bool = True,
    status: str = "accepted",
    quality: float = 1.0,
) -> PoseRecord:
    """A pose translated from the origin by a pure lateral/axial shift, identity rotation."""
    return PoseRecord(
        transform=Rigid.of(IDENTITY_MATRIX, (dx, dy, dz)),
        accepted=accepted,
        status=status,
        quality=quality,
    )


IDENTITY_POSE: Final = _pose()


def _patch(width: int, height: int, codes: list[int]) -> DepthPatch:
    return DepthPatch.of(width, height, codes)


def _frame(
    index: int, patch: DepthPatch, pose: PoseRecord, *, u0: int = 0, v0: int = 0
) -> TemporalFrame:
    return TemporalFrame(index=index, patch=patch, u0=u0, v0=v0, pose=pose)


# =========================================================================================
# 1. Construction validation of every public immutable type
# =========================================================================================

# --- CameraModel -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fx", 0.0),
        ("fx", -1.0),
        ("fx", float("nan")),
        ("fx", float("inf")),
        ("fy", 0.0),
        ("fy", -1.0),
        ("cx", float("nan")),
        ("cy", float("inf")),
        ("depth_scale", 0.0),
        ("depth_scale", -1.0),
        ("depth_scale", float("nan")),
    ],
)
def test_a_camera_that_could_not_project_or_deproject_anything_is_refused(field, value):
    fields = {"fx": 50.0, "fy": 50.0, "cx": 2.0, "cy": 2.0, "depth_scale": 1.0}
    with pytest.raises(CameraUnusable):
        CameraModel(**(fields | {field: value}))


def test_a_usable_camera_deprojects_and_reprojects_its_own_pixel_exactly():
    camera = _camera(5)
    point = camera.deproject(3.0, 1.0, 2000)
    assert camera.project(point) == pytest.approx((3.0, 1.0))
    assert camera.code_of(point[2]) == pytest.approx(2000.0)


# --- PoseRecord ----------------------------------------------------------------------------


def test_a_pose_whose_transform_is_not_a_rigid_is_refused():
    with pytest.raises(PoseUnusable, match="not a Rigid transform"):
        PoseRecord(transform="not-a-rigid", accepted=True, status="ok")  # type: ignore[arg-type]


def test_a_pose_built_directly_from_a_non_finite_translation_is_refused():
    # Rigid.of() would validate; a caller can still build the dataclass fields
    # directly, and PoseRecord is what catches that.
    broken = Rigid(rotation=IDENTITY_MATRIX, translation=(float("nan"), 0.0, 0.0))
    with pytest.raises(PoseUnusable, match="not a finite rigid transform"):
        PoseRecord(transform=broken, accepted=True, status="ok")


def test_a_pose_built_directly_from_a_non_rotation_matrix_is_refused():
    skewed = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    broken = Rigid(rotation=skewed, translation=(0.0, 0.0, 0.0))
    with pytest.raises(PoseUnusable, match="not a finite rigid transform"):
        PoseRecord(transform=broken, accepted=True, status="ok")


def test_a_pose_whose_acceptance_is_not_a_boolean_is_refused():
    with pytest.raises(PoseUnusable, match="acceptance is a boolean"):
        PoseRecord(transform=Rigid.identity(), accepted=1, status="ok")  # type: ignore[arg-type]


def test_a_pose_with_no_status_name_is_refused():
    with pytest.raises(PoseUnusable, match="not a name"):
        PoseRecord(transform=Rigid.identity(), accepted=True, status="")


@pytest.mark.parametrize("quality", [0.0, -0.1, 1.5, float("nan"), True])
def test_a_pose_quality_that_is_not_a_fraction_above_zero_is_refused(quality):
    with pytest.raises(PoseUnusable):
        PoseRecord(transform=Rigid.identity(), accepted=True, status="ok", quality=quality)


def test_acceptance_is_what_the_tracker_decided_and_is_gated_at_the_frame_set():
    """``accepted=False`` alone does not fail ``PoseRecord``'s own validation.

    It is a fact the tracker states, not a number this type can range-check;
    what refuses an unaccepted pose is naming it as a target or a source, in
    :func:`complete_temporally` (see the FrameUnusable/PoseUnusable tests
    below).
    """
    refused = PoseRecord(transform=Rigid.identity(), accepted=False, status="refused")
    assert refused.accepted is False


# --- TemporalFrame -------------------------------------------------------------------------


def test_a_frame_with_a_boolean_or_negative_index_is_refused():
    patch = _patch(2, 2, [0, 0, 0, 0])
    with pytest.raises(FrameUnusable, match="not an integer"):
        TemporalFrame(index=True, patch=patch, u0=0, v0=0, pose=IDENTITY_POSE)
    with pytest.raises(FrameUnusable, match="not an index"):
        TemporalFrame(index=-1, patch=patch, u0=0, v0=0, pose=IDENTITY_POSE)


def test_a_frame_whose_patch_is_not_a_depth_patch_is_refused():
    with pytest.raises(FrameUnusable, match="not a DepthPatch"):
        TemporalFrame(index=0, patch=[0, 0, 0, 0], u0=0, v0=0, pose=IDENTITY_POSE)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["u0", "v0"])
def test_a_frame_whose_origin_is_negative_or_a_boolean_is_refused(field):
    patch = _patch(2, 2, [0, 0, 0, 0])
    fields = {"index": 0, "patch": patch, "u0": 0, "v0": 0, "pose": IDENTITY_POSE}
    with pytest.raises(FrameUnusable):
        TemporalFrame(**(fields | {field: -1}))
    with pytest.raises(FrameUnusable):
        TemporalFrame(**(fields | {field: True}))


def test_a_frame_whose_pose_is_not_a_pose_record_is_refused():
    patch = _patch(2, 2, [0, 0, 0, 0])
    with pytest.raises(FrameUnusable, match="not a PoseRecord"):
        TemporalFrame(index=0, patch=patch, u0=0, v0=0, pose=Rigid.identity())  # type: ignore[arg-type]


def test_a_frame_carries_the_depth_patchs_own_immutability_and_no_extra_copy_is_needed():
    """``TemporalFrame`` has no raw sequence field of its own to alias.

    Its only container-shaped field is ``patch``, which is already the
    immutable, copy-on-construction :class:`DepthPatch`; there is nothing
    further here for a caller-owned list to leak into.
    """
    codes = [0, 1000, 0, 0]
    patch = _patch(2, 2, codes)
    frame = _frame(0, patch, IDENTITY_POSE)
    codes[1] = 5
    assert frame.patch.codes == (0, 1000, 0, 0)


# --- TemporalPolicy ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_support_frames", 0),
        ("min_support_frames", -1),
        ("min_support_frames", True),
        ("min_support_frames", MAX_SUPPORT_FRAMES + 1),
        ("max_support_frames", 0),
        ("max_support_frames", MAX_SUPPORT_FRAMES + 1),
        ("min_distinct_viewpoints", 0),
        ("min_distinct_viewpoints", True),
        ("footprint_radius", -1),
        ("footprint_radius", True),
        ("footprint_radius", 1.5),
        ("footprint_radius", MAX_FOOTPRINT_RADIUS + 1),
        ("max_candidates_per_target", 0),
        ("max_candidates_per_target", MAX_CANDIDATES_PER_TARGET + 1),
        ("min_baseline", 0.0),
        ("min_baseline", float("nan")),
        ("min_baseline", True),
        ("min_rotation_degrees", -1.0),
        ("min_rotation_degrees", 181.0),
        ("max_reprojection_distance", 0.0),
        ("max_reprojection_distance", float("inf")),
        ("max_depth_disagreement_codes", 0.0),
        ("occlusion_margin_codes", float("nan")),
        ("cluster_gap_codes", -1.0),
        ("confirmation_tolerance_codes", 0.0),
        ("min_confidence", 0.0),
        ("min_confidence", MAX_RECONSTRUCTION_CONFIDENCE + 0.01),
        ("min_confidence", True),
        ("min_confidence", float("nan")),
        ("min_code", 0),
        ("min_code", True),
        ("max_code", MAX_CODE + 1),
    ],
)
def test_a_policy_bound_that_is_not_a_bound_is_refused(field, value):
    with pytest.raises(PolicyUnusable):
        TemporalPolicy(**{field: value})


def test_a_policy_may_not_want_more_support_than_it_allows():
    with pytest.raises(PolicyUnusable, match="allows at most"):
        TemporalPolicy(min_support_frames=6, max_support_frames=3)


def test_a_policy_may_not_want_more_viewpoints_than_its_own_support_floor():
    with pytest.raises(PolicyUnusable, match="could never be satisfied"):
        TemporalPolicy(min_support_frames=2, min_distinct_viewpoints=3, max_support_frames=4)


def test_a_policy_predicting_into_no_range_at_all_is_refused():
    with pytest.raises(PolicyUnusable, match="not a range"):
        TemporalPolicy(min_code=900, max_code=900)


def test_footprint_radius_zero_is_a_legal_single_cell_stamp():
    policy = TemporalPolicy(footprint_radius=0)
    assert policy.footprint_cells == 1


@pytest.mark.parametrize(
    ("name", "ceiling"),
    [
        ("min_support_frames", MAX_SUPPORT_FRAMES),
        ("footprint_radius", MAX_FOOTPRINT_RADIUS),
        ("max_candidates_per_target", MAX_CANDIDATES_PER_TARGET),
    ],
)
def test_every_integer_bound_has_a_ceiling_and_it_is_reachable(name, ceiling):
    base = {"max_support_frames": MAX_SUPPORT_FRAMES}
    assert TemporalPolicy(**(base | {name: ceiling}))
    with pytest.raises(PolicyUnusable, match=r"ceiling|outside the"):
        TemporalPolicy(**(base | {name: ceiling + 1}))


def test_worst_case_work_is_exact_and_refuses_a_boolean_or_negative_count():
    policy = DEFAULT_POLICY
    assert policy.worst_case_work(0, 0) == 0
    # The per-hole term covers examining the hole plus every candidate the
    # deposit can retain at it - which is one past the cap, so an over-cap
    # target is provably over it and the sentinel it keeps is budgeted work.
    assert policy.worst_case_work(10, 3) == 10 * policy.footprint_cells + 3 * (
        2 + policy.max_candidates_per_target
    )
    with pytest.raises(PolicyUnusable, match="not a number of"):
        policy.worst_case_work(True, 0)
    with pytest.raises(PolicyUnusable, match="not a number of"):
        policy.worst_case_work(0, -1)


# --- temporal_transform / viewpoint_separation ----------------------------------------------


def test_temporal_transform_is_target_pose_inverse_composed_with_source_pose():
    """The single most likely error in this module, on a motion with a known answer.

    ``target`` sits at world ``(1, 2, 3)``, identity rotation; ``source`` sits
    at the world origin. A point five units in front of the source camera,
    ``(0, 0, 5)``, is at world ``(0, 0, 5)`` (source pose is identity) and at
    ``(-1, -2, 2)`` in the target camera - the target is one unit further
    along +x and two along +y, and its own translation subtracts out. The
    transform composed the other way round gives a different, wrong answer:
    it adds the target's translation instead of subtracting it.
    """
    target_pose = PoseRecord(Rigid.of(IDENTITY_MATRIX, (1.0, 2.0, 3.0)), True, "accepted")
    source_pose = IDENTITY_POSE
    correct = temporal_transform(target_pose, source_pose)
    assert correct.apply((0.0, 0.0, 5.0)) == pytest.approx((-1.0, -2.0, 2.0))
    backwards = source_pose.transform.inverse().compose(target_pose.transform)
    assert backwards.apply((0.0, 0.0, 5.0)) == pytest.approx((1.0, 2.0, 8.0))
    assert correct.apply((0.0, 0.0, 5.0)) != pytest.approx(backwards.apply((0.0, 0.0, 5.0)))


def test_temporal_transform_needs_two_pose_records():
    with pytest.raises(PoseUnusable):
        temporal_transform(Rigid.identity(), IDENTITY_POSE)  # type: ignore[arg-type]


def test_viewpoint_separation_reports_baseline_and_rotation_independently():
    a = Rigid.identity()
    b = Rigid.of(IDENTITY_MATRIX, (0.3, 0.4, 0.0))  # a 3-4-5 triangle
    baseline, degrees = viewpoint_separation(a, b)
    assert baseline == pytest.approx(0.5)
    assert degrees == pytest.approx(0.0)


# --- project_observation ---------------------------------------------------------------------


def test_project_observation_under_identity_returns_the_same_sample():
    camera = _camera(5)
    result = project_observation(camera, Rigid.identity(), source_u=2, source_v=3, code=1000)
    assert result == pytest.approx((2.0, 3.0, 1000.0))


def test_project_observation_under_a_known_translation_lands_where_it_analytically_must():
    """``dx=100`` at depth 1000 moves the pixel by ``dx * fx / depth = 5``."""
    camera = _camera(5)
    transform = Rigid.of(IDENTITY_MATRIX, (100.0, 0.0, 0.0))
    result = project_observation(camera, transform, source_u=2, source_v=3, code=1000)
    assert result == pytest.approx((7.0, 3.0, 1000.0))


def test_project_observation_refuses_a_sample_with_no_return_code():
    camera = _camera(5)
    assert project_observation(camera, Rigid.identity(), source_u=2, source_v=2, code=0) is None


def test_project_observation_refuses_a_point_behind_the_source_camera():
    camera = _camera(5)
    # code=0 is caught earlier; a negative code (not something DepthPatch would
    # ever carry, but project_observation takes a raw int) deprojects to a
    # negative depth, which is behind the source's own optical centre.
    assert project_observation(camera, Rigid.identity(), source_u=2, source_v=2, code=-500) is None


def test_project_observation_refuses_a_point_behind_the_target_camera():
    camera = _camera(5)
    transform = Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, -2000.0))
    result = project_observation(camera, transform, source_u=2, source_v=2, code=1000)
    assert result is None


# =========================================================================================
# 4. complete_temporally end-to-end on small hand-built synthetic frames
# =========================================================================================


WIDE_CAMERA: Final = _camera(9)


def test_identity_poses_and_agreeing_evidence_land_exactly_and_codes_are_exact():
    """Two frames, two viewpoints, complementary holes: the textbook case.

    Target is a 9x9 constant plane at 555 with two holes. Both source frames
    are almost entirely no-return, carrying exactly one measurement at each
    hole's pixel: 6000 at (2, 2) and 7000 at (6, 6). Source 1 sits at the
    identity pose and source 2 is shifted 0.1 units in x - a lateral shift
    that changes which pixel the sample projects to by a fraction of a pixel
    (``dx * fx / depth`` is under 0.001 at these depths) but never changes the
    depth itself, so both frames predict the exact same code and the two
    viewpoints are still genuinely distinct (baseline 0.1 >= min_baseline).
    """
    target_codes = [555] * 81
    target_codes[2 * 9 + 2] = 0
    target_codes[6 * 9 + 6] = 0
    target = _frame(0, _patch(9, 9, target_codes), IDENTITY_POSE)

    source_codes = [0] * 81
    source_codes[2 * 9 + 2] = 6000
    source_codes[6 * 9 + 6] = 7000
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))

    result = complete_temporally(target, [source1, source2], camera=WIDE_CAMERA)

    assert result.work.holes == 2
    assert result.work.accepted == 2
    assert result.work.rejected == 0
    assert result.codes[2 * 9 + 2] == 6000
    assert result.codes[6 * 9 + 6] == 7000
    assert result.state_at(2 * 9 + 2) == "temporally_reconstructed"
    assert result.state_at(6 * 9 + 6) == "temporally_reconstructed"
    assert result.support_at(2 * 9 + 2) == ((1, 2 * 9 + 2), (2, 2 * 9 + 2))
    assert result.support_at(6 * 9 + 6) == ((1, 6 * 9 + 6), (2, 6 * 9 + 6))
    # Every declared reason is reported, and none of them fired.
    assert set(result.work.rejections) == set(REJECTION_REASONS)
    assert all(count == 0 for count in result.work.rejections.values())
    assert set(result.work.observation_rejections) == set(OBSERVATION_REJECTIONS)
    assert all(count == 0 for count in result.work.observation_rejections.values())


def _straddling_target(near: int = 500, far: int = 5000) -> DepthPatch:
    """A 9x9 target whose left half is near, right half is far, with a hole between.

    The hole at (4, 4) has measured neighbours on both surfaces, ``far - near``
    codes apart. That is the silhouette: which surface the missing sample
    belongs to is exactly what its own frame cannot say.
    """
    codes = [near if column < 4 else far for _row in range(9) for column in range(9)]
    codes[4 * 9 + 4] = 0
    return _patch(9, 9, codes)


def test_a_hole_whose_own_neighbours_straddle_a_step_is_refused_as_a_discontinuity():
    """The moving object edge: agreeing evidence is not enough at a silhouette.

    Both sources measure the same code at the hole and agree perfectly, so
    nothing about the evidence itself is suspect. It is refused because the
    target's own measured neighbours straddle a step wider than
    ``cluster_gap_codes`` - the case where two other views can both be looking
    past a foreground surface that only this frame saw.
    """
    near, far = 500, 5000
    target = _frame(0, _straddling_target(near, far), IDENTITY_POSE)
    source_codes = [0] * 81
    source_codes[4 * 9 + 4] = far
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))
    policy = TemporalPolicy(cluster_gap_codes=float(far - near) / 2.0)

    result = complete_temporally(target, [source1, source2], camera=WIDE_CAMERA, policy=policy)

    assert result.state_at(4 * 9 + 4) == "missing"
    assert result.work.accepted == 0
    assert result.work.rejections["edge_discontinuity"] == 1

    # The identical evidence at a hole whose neighbours are all one surface is
    # accepted, so it is the discontinuity that refused this and nothing else.
    flat = [far] * 81
    flat[4 * 9 + 4] = 0
    unambiguous = _frame(0, _patch(9, 9, flat), IDENTITY_POSE)
    settled = complete_temporally(
        unambiguous, [source1, source2], camera=WIDE_CAMERA, policy=policy
    )
    assert settled.state_at(4 * 9 + 4) == "temporally_reconstructed"
    assert settled.codes[4 * 9 + 4] == far


def test_a_prediction_outside_the_policys_own_code_range_is_refused():
    """A value the policy says is not a depth is not published at any confidence."""
    target_codes = [555] * 81
    target_codes[2 * 9 + 2] = 0
    target = _frame(0, _patch(9, 9, target_codes), IDENTITY_POSE)
    source_codes = [0] * 81
    source_codes[2 * 9 + 2] = 6000
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))
    # A ceiling below the evidence's own code. The observations survive - the
    # discard vocabulary has its own reason for a code out of range - so this
    # asserts the hole-level refusal that follows.
    policy = TemporalPolicy(max_code=5000)

    result = complete_temporally(target, [source1, source2], camera=WIDE_CAMERA, policy=policy)

    assert result.state_at(2 * 9 + 2) == "missing"
    assert result.work.observation_rejections["predicted_code_out_of_range"] == 2
    assert result.work.rejections["all_evidence_refused"] == 1


def test_more_candidates_than_the_policy_allows_refuses_rather_than_tie_breaking():
    """Past the cap the sample is ambiguous, not settled by whichever came first."""
    target_codes = [555] * 81
    target_codes[2 * 9 + 2] = 0
    target = _frame(0, _patch(9, 9, target_codes), IDENTITY_POSE)
    source_codes = [0] * 81
    source_codes[2 * 9 + 2] = 6000
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))

    result = complete_temporally(
        target,
        [source1, source2],
        camera=WIDE_CAMERA,
        policy=TemporalPolicy(max_candidates_per_target=1),
    )

    assert result.state_at(2 * 9 + 2) == "missing"
    assert result.work.rejections["too_many_candidates"] == 1


def test_perturbing_a_source_pose_turns_a_correct_recovery_into_a_refusal():
    """A pose that is wrong does not produce a plausible shift - it produces nothing.

    The same two frames recover the sample exactly under their real poses. Move
    one source a hundred units along the optical axis and the depth it now
    claims for that sample is a hundred units further away, so the visibility
    rule discards it as hidden behind the other frame's nearer evidence and the
    hole falls to one supporting frame - which is not enough. Refused, with a
    named reason, rather than reconstructed a little off. That is the property
    that matters: an unnoticed pose error must not survive as geometry.
    """
    target_codes = [555] * 81
    target_codes[4 * 9 + 4] = 0
    target = _frame(0, _patch(9, 9, target_codes), IDENTITY_POSE)
    source_codes = [0] * 81
    source_codes[4 * 9 + 4] = 6000
    truthful = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)

    settled = complete_temporally(target, [source1, truthful], camera=WIDE_CAMERA)
    assert settled.state_at(4 * 9 + 4) == "temporally_reconstructed"
    assert settled.codes[4 * 9 + 4] == 6000

    # The perturbation is along the optical axis, so it changes the depth this
    # source claims for the sample rather than only where it lands.
    perturbed = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1, dz=100.0))
    result = complete_temporally(target, [source1, perturbed], camera=WIDE_CAMERA)

    assert result.state_at(4 * 9 + 4) == "missing"
    assert result.work.accepted == 0
    assert result.work.observation_rejections["occluded_by_nearer_evidence"] == 1
    assert result.work.rejections["insufficient_support_frames"] == 1
    assert sum(result.work.rejections.values()) == 1


def test_a_raw_target_measurement_is_never_overwritten():
    target_codes = [555] * 81
    target_codes[0] = 42
    target_codes[2 * 9 + 2] = 0
    target = _frame(0, _patch(9, 9, target_codes), IDENTITY_POSE)
    source_codes = [0] * 81
    source_codes[2 * 9 + 2] = 6000
    source1 = _frame(1, _patch(9, 9, source_codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(9, 9, source_codes), _pose(dx=0.1))

    result = complete_temporally(target, [source1, source2], camera=WIDE_CAMERA)

    assert result.codes[0] == 42
    assert result.state_at(0) == "measured"
    assert result.confidence[0] == 1.0
    assert result.support_at(0) == ()


NARROW_CAMERA: Final = _camera(5)


def _mixed_support_scene() -> tuple[TemporalFrame, TemporalFrame, TemporalFrame]:
    """A 5x5 target with two holes: one two frames agree on, one only one frame sees.

    Hole (1, 1) [index 6] gets code 4000 from both source frames, at distinct
    viewpoints. Hole (3, 3) [index 18] gets code 5000 from source 1 only;
    source 2 has no measurement there at all, so it stays missing for want of
    a second independent frame.
    """
    target_codes = [3000] * 25
    target_codes[6] = 0
    target_codes[18] = 0
    target = _frame(0, _patch(5, 5, target_codes), IDENTITY_POSE)

    codes1 = [0] * 25
    codes1[6] = 4000
    codes1[18] = 5000
    source1 = _frame(1, _patch(5, 5, codes1), IDENTITY_POSE)

    codes2 = [0] * 25
    codes2[6] = 4000
    source2 = _frame(2, _patch(5, 5, codes2), _pose(dx=0.1))
    return target, source1, source2


def test_a_hole_with_support_from_only_one_other_frame_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.state_at(6) == "temporally_reconstructed"
    assert result.state_at(18) == "missing"
    assert result.codes[18] == 0
    assert result.confidence[18] == 0.0
    assert result.work.rejections["insufficient_support_frames"] == 1


def test_two_frames_at_the_same_pose_are_one_viewpoint():
    target_codes = [3000] * 25
    target_codes[6] = 0
    target = _frame(0, _patch(5, 5, target_codes), IDENTITY_POSE)
    codes = [0] * 25
    codes[6] = 4000
    source1 = _frame(1, _patch(5, 5, codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(5, 5, codes), IDENTITY_POSE)  # same pose as source1

    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)

    assert result.state_at(6) == "missing"
    assert result.work.rejections["insufficient_viewpoint_diversity"] == 1


def test_two_sources_that_disagree_beyond_the_policy_are_refused_rather_than_averaged():
    # occlusion_margin_codes is loosened past the disagreement so the z-buffer
    # does not discard the farther observation before the two are ever
    # compared for disagreement - this test is about the disagreement check,
    # not the occlusion one (see test_a_farther_sample_cannot_vote... above).
    policy = dataclasses.replace(DEFAULT_POLICY, occlusion_margin_codes=1000.0)
    target_codes = [3000] * 25
    target_codes[6] = 0
    target = _frame(0, _patch(5, 5, target_codes), IDENTITY_POSE)
    codes1 = [0] * 25
    codes1[6] = 1000
    source1 = _frame(1, _patch(5, 5, codes1), IDENTITY_POSE)
    codes2 = [0] * 25
    codes2[6] = 1050  # 50 codes away; default max_depth_disagreement_codes is 32
    source2 = _frame(2, _patch(5, 5, codes2), _pose(dx=0.1))

    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA, policy=policy)

    assert result.state_at(6) == "missing"
    assert result.work.rejections["conflicting_depth_evidence"] == 1


def test_a_farther_sample_cannot_vote_through_nearer_occluding_evidence():
    """Foreground occludes background: the z-buffer, not iteration order, decides.

    Three source frames project onto hole (1, 1) [index 6]: two agree at depth
    1000 from distinct viewpoints, and a third lands nearby at depth 1050,
    more than ``occlusion_margin_codes`` (32) beyond the nearest evidence. The
    far one is discarded before it is ever compared, and the reconstruction
    comes from the two that agree.
    """
    target_codes = [3000] * 25
    target_codes[6] = 0
    target = _frame(0, _patch(5, 5, target_codes), IDENTITY_POSE)
    codes_near = [0] * 25
    codes_near[6] = 1000
    source1 = _frame(1, _patch(5, 5, codes_near), IDENTITY_POSE)
    codes_far = [0] * 25
    codes_far[6] = 1050
    source2 = _frame(2, _patch(5, 5, codes_far), _pose(dx=0.2))
    source3 = _frame(3, _patch(5, 5, codes_near), _pose(dx=-0.1))

    result = complete_temporally(target, [source1, source2, source3], camera=NARROW_CAMERA)

    assert result.work.observation_rejections["occluded_by_nearer_evidence"] == 1
    assert result.state_at(6) == "temporally_reconstructed"
    assert result.codes[6] == 1000
    assert result.support_at(6) == ((1, 6), (3, 6))


def test_outside_the_target_rectangle_is_refused_and_named():
    target_codes = [3000] * 9
    target_codes[4] = 0
    camera = _camera(3)
    target = _frame(0, _patch(3, 3, target_codes), IDENTITY_POSE)
    codes = [0] * 9
    codes[4] = 500  # pixel (1, 1), depth 500
    # A huge lateral shift throws the projection far outside the 3x3 target.
    source = _frame(1, _patch(3, 3, codes), _pose(dx=1000.0))
    result = complete_temporally(target, [source], camera=camera, policy=DEFAULT_POLICY)
    assert result.work.observation_rejections["outside_target_patch"] == 1


def test_a_projection_landing_more_than_the_policy_allows_from_any_pixel_is_refused():
    codes = [0] * 25
    codes[12] = 1000  # pixel (2, 2), depth 1000
    target_codes = [3000] * 25
    target_codes[12] = 0
    target = _frame(0, _patch(5, 5, target_codes), IDENTITY_POSE)
    # dx=dy=8 at depth 1000 shifts (u, v) by (0.4, 0.4): the 2D distance to the
    # nearest pixel is hypot(0.4, 0.4) ~= 0.566, past the default 0.5 bound,
    # while every candidate is still inside the 5x5 rectangle.
    source = _frame(1, _patch(5, 5, codes), _pose(dx=8.0, dy=8.0))
    result = complete_temporally(target, [source], camera=NARROW_CAMERA, policy=DEFAULT_POLICY)
    assert result.work.observation_rejections["reprojection_too_far"] == 1


# =========================================================================================
# 5. Spatial confirmation semantics
# =========================================================================================

CONFIRM_CAMERA: Final = _camera(4)
HOLE_INDEX: Final = 5  # (column=1, row=1) of a 4x4 patch


def _spatial_estimate() -> CompletedPatch:
    """A real spatial completion of a 4x4 constant patch missing one sample.

    Mirrors the accepted sibling suite's ``_ONE_HOLE_SOURCE``/``ONE_SUPPORT``
    fixture exactly: constant code 9000, one hole, ``min_support=1`` so the
    default 4-radius disc (which reaches every other sample of a 4x4 patch)
    is enough to reconstruct it. A constant surface's least-squares plane fit
    is exact, so the spatial estimate is 9000 at full precision.
    """
    codes = [9000] * 16
    codes[HOLE_INDEX] = 0
    patch = _patch(4, 4, codes)
    return complete_spatially(patch, policy=CompletionPolicy(min_support=1))


def _confirm_target() -> TemporalFrame:
    codes = [9000] * 16
    codes[HOLE_INDEX] = 0
    return _frame(0, _patch(4, 4, codes), IDENTITY_POSE)


def _confirm_sources(code1: int, code2: int) -> tuple[TemporalFrame, TemporalFrame]:
    codes1 = [0] * 16
    codes1[HOLE_INDEX] = code1
    source1 = _frame(1, _patch(4, 4, codes1), IDENTITY_POSE)
    codes2 = [0] * 16
    codes2[HOLE_INDEX] = code2
    source2 = _frame(2, _patch(4, 4, codes2), _pose(dx=0.1))
    return source1, source2


def test_a_spatial_estimate_independent_evidence_reproduces_is_confirmed():
    target = _confirm_target()
    spatial = _spatial_estimate()
    source1, source2 = _confirm_sources(9000, 9000)  # agrees with the spatial estimate

    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)

    assert result.confirmation_at(HOLE_INDEX) == "confirmed"
    assert result.state_at(HOLE_INDEX) == "temporally_reconstructed"
    assert result.codes[HOLE_INDEX] == 9000
    assert result.support_at(HOLE_INDEX) == ((1, HOLE_INDEX), (2, HOLE_INDEX))
    assert result.spatial_support_at(HOLE_INDEX) == spatial.support_at(HOLE_INDEX)
    assert result.spatial_support_at(HOLE_INDEX) != ()


def test_a_spatial_estimate_independent_evidence_disagrees_with_is_contradicted():
    target = _confirm_target()
    spatial = _spatial_estimate()
    source1, source2 = _confirm_sources(200, 200)  # 8800 away from the 9000 estimate

    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)

    assert result.confirmation_at(HOLE_INDEX) == "contradicted"
    # Temporally reconstructed, carrying the independent value - never the
    # spatial one and never an average of the two.
    assert result.state_at(HOLE_INDEX) == "temporally_reconstructed"
    assert result.codes[HOLE_INDEX] == 200
    assert result.codes[HOLE_INDEX] != spatial.codes[HOLE_INDEX]
    tolerance = DEFAULT_POLICY.confirmation_tolerance_codes
    assert abs(result.codes[HOLE_INDEX] - spatial.codes[HOLE_INDEX]) > tolerance
    assert result.spatial_support_at(HOLE_INDEX) == spatial.support_at(HOLE_INDEX)


def test_a_spatial_estimate_no_independent_evidence_reaches_is_not_observed():
    target = _confirm_target()
    spatial = _spatial_estimate()
    empty = [0] * 16
    source1 = _frame(1, _patch(4, 4, empty), IDENTITY_POSE)
    source2 = _frame(2, _patch(4, 4, empty), _pose(dx=0.1))

    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)

    assert result.confirmation_at(HOLE_INDEX) == "not_observed"
    assert result.state_at(HOLE_INDEX) == "spatially_reconstructed"
    assert result.codes[HOLE_INDEX] == spatial.codes[HOLE_INDEX]
    assert result.confidence[HOLE_INDEX] == spatial.confidence[HOLE_INDEX]


def test_a_hole_with_no_spatial_estimate_is_reconstructed_when_evidence_suffices():
    target = _confirm_target()
    source1, source2 = _confirm_sources(9000, 9000)

    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=None)

    assert result.confirmation_at(HOLE_INDEX) == "no_spatial_estimate"
    assert result.state_at(HOLE_INDEX) == "temporally_reconstructed"
    assert result.codes[HOLE_INDEX] == 9000


def test_a_spatial_guess_does_not_change_which_holes_independent_evidence_settles():
    target = _confirm_target()
    spatial = _spatial_estimate()
    source1, source2 = _confirm_sources(9000, 9000)

    with_spatial = complete_temporally(
        target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial
    )
    without_spatial = complete_temporally(
        target, [source1, source2], camera=CONFIRM_CAMERA, spatial=None
    )
    assert (
        with_spatial.temporally_reconstructed_indices()
        == without_spatial.temporally_reconstructed_indices()
    )


def test_every_confirmation_code_published_is_the_closed_vocabulary():
    target = _confirm_target()
    spatial = _spatial_estimate()
    source1, source2 = _confirm_sources(9000, 9000)
    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)
    for index in range(16):
        assert result.confirmation_at(index) in CONFIRMATIONS
    # Every other sample had no spatial proposal at all.
    for index in range(16):
        if index != HOLE_INDEX:
            assert result.confirmation_at(index) == "no_spatial_estimate"


# =========================================================================================
# 6. Forgery / replay controls on TemporalPatch
# =========================================================================================


def test_a_reconstructed_code_the_evidence_does_not_give_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.codes[6] == 4000
    codes = (*result.codes[:6], 4321, *result.codes[7:])
    with pytest.raises(ResultUnusable, match="these frames under the policy it publishes derive"):
        dataclasses.replace(result, codes=codes)


def test_a_confidence_the_formula_does_not_give_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.confidence[6] != 0.5
    confidence = (*result.confidence[:6], 0.5, *result.confidence[7:])
    with pytest.raises(ResultUnusable, match="not a number a result may state"):
        dataclasses.replace(result, confidence=confidence)


def test_a_support_reference_swapped_for_the_targets_own_frame_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    support = dict(result.temporal_support) | {6: ((1, 6), (target.index, 0))}
    with pytest.raises(ResultUnusable, match="the target's own frame"):
        dataclasses.replace(result, temporal_support=support)


def test_a_support_reference_naming_a_no_return_source_sample_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert source2.patch.codes[0] == 0  # a genuine hole in source 2's own frame
    support = dict(result.temporal_support) | {6: ((1, 6), (2, 0))}
    with pytest.raises(ResultUnusable, match="returned nothing"):
        dataclasses.replace(result, temporal_support=support)


def test_a_duplicated_support_reference_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    support = dict(result.temporal_support) | {6: ((1, 6), (1, 6))}
    with pytest.raises(ResultUnusable, match="same sample twice"):
        dataclasses.replace(result, temporal_support=support)


def test_a_measured_sample_relabelled_reconstructed_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.state_at(0) == "measured"
    states = bytearray(result.state_codes)
    states[0] = STATE_TEMPORALLY_RECONSTRUCTED
    with pytest.raises(ResultUnusable, match="A measurement stays a measurement"):
        dataclasses.replace(result, state_codes=bytes(states))


def test_a_missing_sample_given_a_depth_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.state_at(18) == "missing"
    codes = (*result.codes[:18], 777, *result.codes[19:])
    confidence = (*result.confidence[:18], 0.5, *result.confidence[19:])
    with pytest.raises(ResultUnusable, match="exactly code zero, confidence zero"):
        dataclasses.replace(result, codes=codes, confidence=confidence)


def test_a_forged_work_record_that_moves_a_hole_from_rejected_to_accepted_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.work.accepted == 1
    assert result.work.rejected == 1
    forged_work = dataclasses.replace(
        result.work,
        accepted=result.work.accepted + 1,
        rejected=result.work.rejected - 1,
        # The rejection taxonomy has to stay internally consistent (sum ==
        # rejected), and the support-sample count has to stay inside what the
        # (now higher) accepted count's own policy allows, so this forgery is
        # caught by the *reconstruction count* rule in TemporalPatch, not by
        # TemporalWork's own arithmetic first.
        rejections=dict.fromkeys(REJECTION_REASONS, 0),
        support_samples=2 * DEFAULT_POLICY.min_support_frames,
    )
    with pytest.raises(ResultUnusable, match="beside work that accepted"):
        dataclasses.replace(result, work=forged_work)


def test_a_rejection_count_that_does_not_sum_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    reasons = dict(result.work.rejections)
    reasons["insufficient_support_frames"] += 1
    with pytest.raises(ResultUnusable, match="gives reasons for"):
        dataclasses.replace(result.work, rejections=reasons)


def test_a_confirmation_relabelled_confirmed_where_the_derivation_says_not_observed_is_refused():
    target = _confirm_target()
    spatial = _spatial_estimate()
    empty = [0] * 16
    source1 = _frame(1, _patch(4, 4, empty), IDENTITY_POSE)
    source2 = _frame(2, _patch(4, 4, empty), _pose(dx=0.1))
    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)
    assert result.confirmation_at(HOLE_INDEX) == "not_observed"
    confirmations = bytearray(result.confirmation_codes)
    confirmations[HOLE_INDEX] = CONFIRMATION_CONFIRMED
    with pytest.raises(ResultUnusable):
        dataclasses.replace(result, confirmation_codes=bytes(confirmations))


def test_a_confirmation_relabelled_confirmed_at_an_index_the_derivation_contradicted_is_refused():
    """The important one: a contradiction cannot be laundered into a confirmation.

    The published code here is 200, more than the confirmation tolerance away
    from the spatial estimate's 9000, so the sample really is contradicted.
    Relabelling it confirmed while leaving every other field alone is not
    caught by the shape rules - both states are ``temporally_reconstructed``
    - only recomputation catches it.
    """
    target = _confirm_target()
    spatial = _spatial_estimate()
    source1, source2 = _confirm_sources(200, 200)
    result = complete_temporally(target, [source1, source2], camera=CONFIRM_CAMERA, spatial=spatial)
    assert result.confirmation_at(HOLE_INDEX) == "contradicted"
    confirmations = bytearray(result.confirmation_codes)
    confirmations[HOLE_INDEX] = CONFIRMATION_CONFIRMED
    # Keep the work record's own confirmed/contradicted counts internally
    # consistent with the relabelled confirmation codes, so this forgery is
    # caught by recomputation and not by the earlier, cruder count check.
    forged_work = dataclasses.replace(
        result.work,
        confirmed=result.work.confirmed + 1,
        contradicted=result.work.contradicted - 1,
    )
    with pytest.raises(ResultUnusable, match="these frames under the policy it publishes derive"):
        dataclasses.replace(result, confirmation_codes=bytes(confirmations), work=forged_work)


# =========================================================================================
# 7. Confidence
# =========================================================================================

BASELINE_EVIDENCE: Final = TemporalEvidence(
    frames=3,
    viewpoints=3,
    mean_distance=1.0,
    disagreement_codes=8.0,
    pose_quality=0.8,
)


def test_confidence_is_capped_strictly_below_a_measurements_confidence():
    value = temporal_confidence(BASELINE_EVIDENCE, DEFAULT_POLICY)
    assert value < MAX_RECONSTRUCTION_CONFIDENCE
    assert value < 1.0


def test_confidence_is_never_published_above_its_stated_ceiling_even_at_perfect_evidence():
    perfect = TemporalEvidence(
        frames=DEFAULT_POLICY.max_support_frames,
        viewpoints=DEFAULT_POLICY.max_support_frames,
        mean_distance=0.0,
        disagreement_codes=0.0,
        pose_quality=1.0,
    )
    assert temporal_confidence(perfect, DEFAULT_POLICY) <= MAX_RECONSTRUCTION_CONFIDENCE


def test_confidence_does_not_fall_with_more_distinct_supporting_frames():
    more = dataclasses.replace(BASELINE_EVIDENCE, frames=4, viewpoints=4)
    assert temporal_confidence(more, DEFAULT_POLICY) >= temporal_confidence(
        BASELINE_EVIDENCE, DEFAULT_POLICY
    )


def test_confidence_does_not_rise_with_a_larger_reprojection_distance():
    farther = dataclasses.replace(BASELINE_EVIDENCE, mean_distance=3.0)
    assert temporal_confidence(farther, DEFAULT_POLICY) <= temporal_confidence(
        BASELINE_EVIDENCE, DEFAULT_POLICY
    )


def test_confidence_does_not_rise_with_more_disagreement():
    disagreeing = dataclasses.replace(BASELINE_EVIDENCE, disagreement_codes=20.0)
    assert temporal_confidence(disagreeing, DEFAULT_POLICY) <= temporal_confidence(
        BASELINE_EVIDENCE, DEFAULT_POLICY
    )


def test_confidence_does_not_rise_with_a_worse_supporting_pose():
    worse_pose = dataclasses.replace(BASELINE_EVIDENCE, pose_quality=0.2)
    assert temporal_confidence(worse_pose, DEFAULT_POLICY) <= temporal_confidence(
        BASELINE_EVIDENCE, DEFAULT_POLICY
    )


def test_a_term_at_or_below_zero_zeroes_the_whole_geometric_mean():
    zeroed = dataclasses.replace(BASELINE_EVIDENCE, pose_quality=0.0)
    assert temporal_confidence(zeroed, DEFAULT_POLICY) == 0.0


def test_a_reconstruction_is_never_published_below_the_policys_own_floor():
    strict = dataclasses.replace(DEFAULT_POLICY, min_confidence=MAX_RECONSTRUCTION_CONFIDENCE)
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA, policy=strict)
    assert result.state_at(6) == "missing"
    assert result.work.rejections["confidence_below_threshold"] == 1


def test_every_accepted_confidence_is_at_or_above_the_policys_floor():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    for index in result.temporally_reconstructed_indices():
        assert result.confidence[index] >= DEFAULT_POLICY.min_confidence


# =========================================================================================
# 8. Work bounds
# =========================================================================================


def test_select_backend_accepts_auto_and_python_and_refuses_everything_else():
    assert select_backend() == PYTHON_BACKEND
    assert select_backend(AUTO_BACKEND) == PYTHON_BACKEND
    assert select_backend(PYTHON_BACKEND) == PYTHON_BACKEND


@pytest.mark.parametrize("name", ["numpy", "scipy", "gpu"])
def test_select_backend_refuses_a_recognised_backend_it_has_not_written(name):
    with pytest.raises(ComputeBackendUnavailable, match="only implementation"):
        select_backend(name)


def test_select_backend_refuses_a_name_it_does_not_recognise_at_all():
    with pytest.raises(ComputeBackendUnknown):
        select_backend("cuda")


def test_complete_temporally_never_silently_downgrades_its_backend():
    target, source1, source2 = _mixed_support_scene()
    with pytest.raises(ComputeBackendUnavailable):
        complete_temporally(target, [source1, source2], camera=NARROW_CAMERA, backend="numpy")


def test_a_run_whose_worst_case_exceeds_the_ceiling_is_refused_before_it_projects_anything():
    policy = TemporalPolicy(
        footprint_radius=2,
        max_candidates_per_target=256,
        min_support_frames=1,
        min_distinct_viewpoints=1,
    )
    # footprint_cells = (2*2+1)**2 = 25; 100_000 eligible samples * 25 = 2_500_000
    # 90_000 holes * (2 + 256) = 23_220_000; total 25_720_000 > MAX_WORK_UNITS.
    budget = policy.worst_case_work(100_000, 90_000)
    assert budget == 25_720_000
    assert budget > MAX_WORK_UNITS

    width, height = 500, 400  # 200,000 samples: 90,000 holes + 110,000 measured
    target_codes = [0] * 90_000 + [1] * (width * height - 90_000)
    source_codes = [1] * 100_000 + [0] * (width * height - 100_000)
    target = _frame(0, _patch(width, height, target_codes), IDENTITY_POSE)
    source = _frame(1, _patch(width, height, source_codes), IDENTITY_POSE)
    camera = CameraModel(fx=50.0, fy=50.0, cx=250.0, cy=200.0, depth_scale=1.0)

    with pytest.raises(WorkTooLarge, match="past the ceiling"):
        complete_temporally(target, [source], camera=camera, policy=policy)


def test_work_units_never_exceed_the_published_budget():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert result.work.work_units <= result.work.work_budget
    assert result.work.within_budget


def test_the_published_budget_is_exactly_the_policys_own_derivation():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    eligible = sum(len(frame.measured_indices()) for frame in (source1, source2))
    holes = len(target.patch.hole_indices())
    assert result.work.work_budget == DEFAULT_POLICY.worst_case_work(eligible, holes)


# --- the over-cap sentinel: processed work is published work -----------------------------


#: A 3x3 camera in whole-code units: one unit of depth per code, and a pixel
#: exactly one unit wide at unit depth. Small enough that every projection in
#: the tests below can be worked out by hand.
UNIT_CAMERA: Final = CameraModel(fx=1.0, fy=1.0, cx=1.0, cy=1.0, depth_scale=1.0)

#: A policy that votes only for the sample a projection rounds to and refuses a
#: target that accumulates more than two candidates. One frame of support and
#: one viewpoint are enough here, because what is being measured is the work
#: accounting and not the support rule.
CAPPED_POLICY: Final = TemporalPolicy(
    footprint_radius=0,
    max_candidates_per_target=2,
    min_support_frames=1,
    min_distinct_viewpoints=1,
)


def _centre_stack(frames: int, code: int = 100) -> tuple[TemporalFrame, list[TemporalFrame]]:
    """One 3x3 all-hole target and ``frames`` sources measuring only their centre.

    Every source sits at the identity pose and measures code ``code`` at its
    own centre sample, so every one of them projects onto the target's centre
    pixel exactly - which is how a chosen number of candidates is put on one
    target sample.
    """
    target = _frame(0, _patch(3, 3, [0] * 9), IDENTITY_POSE)
    codes = [0] * 9
    codes[4] = code
    sources = [_frame(index + 1, _patch(3, 3, codes), IDENTITY_POSE) for index in range(frames)]
    return target, sources


def test_a_target_exactly_at_the_candidate_cap_spends_and_publishes_the_same_units():
    target, sources = _centre_stack(2)
    result = complete_temporally(target, sources, camera=UNIT_CAMERA, policy=CAPPED_POLICY)
    work = result.work

    assert work.observations == 2
    assert work.candidates == 2
    assert work.accepted == 1
    assert result.codes[4] == 100
    # 2 projections at a 1-cell footprint, then 9 holes examined, then the two
    # candidates deposited at the centre hole.
    assert work.work_units == 2 * CAPPED_POLICY.footprint_cells + 9 + 2
    assert work.work_units == 13
    assert work.work_budget == CAPPED_POLICY.worst_case_work(2, 9) == 2 + 9 * 4
    assert work.within_budget


def test_a_target_one_over_the_candidate_cap_counts_the_sentinel_it_processed():
    """The refusal costs the run a candidate, so the run's budget pays for it.

    ``_deposit`` keeps ``max_candidates_per_target + 1`` observations so an
    over-cap target is provably over the cap rather than silently truncated to
    it. That retained sentinel is judged for visibility like any other
    candidate, so the per-hole term is ``2 + max_candidates_per_target`` and
    the units published here include it.
    """
    target, sources = _centre_stack(3)
    result = complete_temporally(target, sources, camera=UNIT_CAMERA, policy=CAPPED_POLICY)
    work = result.work

    assert work.observations == 3
    assert work.candidates == 3  # the cap, plus the one sentinel past it
    assert work.rejections["too_many_candidates"] == 1
    assert work.accepted == 0
    assert result.state_at(4) == "missing"
    assert work.work_units == 3 * CAPPED_POLICY.footprint_cells + 9 + 3
    assert work.work_units == 15
    assert work.work_budget == CAPPED_POLICY.worst_case_work(3, 9) == 3 + 9 * 4
    assert work.within_budget


def test_candidates_at_one_target_never_pass_the_deposit_ceiling_however_many_project():
    """Four projections, three candidates: the ceiling holds and the units follow it."""
    target, sources = _centre_stack(4)
    result = complete_temporally(target, sources, camera=UNIT_CAMERA, policy=CAPPED_POLICY)
    work = result.work

    assert work.observations == 4
    assert work.candidates == CAPPED_POLICY.max_candidates_per_target + 1 == 3
    assert work.rejections["too_many_candidates"] == 1
    assert work.work_units == 4 * CAPPED_POLICY.footprint_cells + 9 + 3
    assert work.work_units == 16
    assert work.work_budget == CAPPED_POLICY.worst_case_work(4, 9) == 4 + 9 * 4
    assert work.within_budget


def test_a_projection_on_an_exact_half_pixel_boundary_supports_both_adjacent_samples():
    """The case a bare ``candidates <= observations`` invariant refused.

    Both source frames measure their own centre sample and both land at target
    ``(1.5, 1.0)``: exactly on the boundary between the centre sample and the
    one to its right, at exactly the policy's ``max_reprojection_distance`` of
    half a pixel from each. The policy says a point on the boundary supports
    both sides of it, so two projected observations legitimately become four
    visible candidates and complete two target holes.

    Source 1 measures 100 from a camera shifted 50 units in x; source 2
    measures 80 from a camera shifted 50 in x and 20 nearer in z, so its
    sample lands at the same target point and the same depth from a genuinely
    different viewpoint.
    """
    target = _frame(0, _patch(3, 3, [0] * 9), IDENTITY_POSE)
    first = [0] * 9
    first[4] = 100
    second = [0] * 9
    second[4] = 80
    source1 = _frame(1, _patch(3, 3, first), _pose(dx=50.0))
    source2 = _frame(2, _patch(3, 3, second), _pose(dx=50.0, dz=20.0))

    # The projection itself, before any completion: exactly the half-pixel.
    for source in (source1, source2):
        transform = temporal_transform(target.pose, source.pose)
        landed = project_observation(
            UNIT_CAMERA, transform, source_u=1, source_v=1, code=source.patch.codes[4]
        )
        assert landed == (1.5, 1.0, 100.0)

    result = complete_temporally(target, [source1, source2], camera=UNIT_CAMERA)
    work = result.work

    assert work.observations == 2
    assert work.candidates == 4
    assert work.accepted == 2
    for index in (4, 5):
        assert result.state_at(index) == "temporally_reconstructed"
        assert result.codes[index] == 100
        assert result.support_at(index) == ((1, 4), (2, 4))
    # Deposited into 9 footprint cells each, then 9 holes examined, then the
    # two candidates that survived at each of the two holes they landed in.
    assert work.work_units == 2 * DEFAULT_POLICY.footprint_cells + 9 + 4
    assert work.work_units == 31
    assert work.work_budget == DEFAULT_POLICY.worst_case_work(2, 9)


# --- TemporalWork: shape, closed vocabulary, cross-field consistency --------------------


def _work_of(**overrides: object) -> TemporalWork:
    fields: dict[str, object] = {
        "backend": PYTHON_BACKEND,
        "policy": DEFAULT_POLICY,
        "source_frames": 1,
        "eligible_samples": 1,
        "observations": 1,
        "observation_rejections": dict.fromkeys(OBSERVATION_REJECTIONS, 0),
        "candidates": 1,
        "holes": 1,
        "attempted": 1,
        "accepted": 0,
        "rejected": 1,
        "rejections": dict.fromkeys(REJECTION_REASONS, 0) | {"no_posed_evidence": 1},
        "confirmed": 0,
        "contradicted": 0,
        "not_observed": 0,
        "support_samples": 0,
        "work_units": 1,
        "work_budget": DEFAULT_POLICY.worst_case_work(1, 1),
    }
    return TemporalWork(**(fields | overrides))  # type: ignore[arg-type]


def test_a_work_record_naming_a_backend_that_cannot_have_produced_it_is_refused():
    with pytest.raises(ResultUnusable, match="only one that can have produced a run"):
        _work_of(backend="numpy")


def test_a_work_records_policy_must_be_the_validated_type():
    with pytest.raises(ResultUnusable, match="not a TemporalPolicy"):
        _work_of(policy=DEFAULT_POLICY.as_dict())


def test_a_work_record_stating_a_schema_version_this_module_does_not_write_is_refused():
    with pytest.raises(ResultUnusable, match="schema version"):
        _work_of(schema_version=0)


@pytest.mark.parametrize(
    "name",
    [
        "source_frames",
        "eligible_samples",
        "observations",
        "candidates",
        "holes",
        "attempted",
        "accepted",
        "rejected",
        "confirmed",
        "contradicted",
        "not_observed",
        "support_samples",
        "work_units",
        "work_budget",
    ],
)
def test_a_work_count_that_is_a_boolean_is_refused(name):
    with pytest.raises(ResultUnusable, match="not a count"):
        _work_of(**{name: True})


def test_a_negative_work_count_is_refused():
    with pytest.raises(ResultUnusable, match="not negative"):
        _work_of(work_units=-1)


def test_a_work_record_that_does_not_add_up_is_refused():
    with pytest.raises(ResultUnusable, match="accounts for"):
        _work_of(accepted=1)


def test_a_work_record_that_looked_at_fewer_holes_than_it_had_is_refused():
    with pytest.raises(ResultUnusable, match="every hole is examined once"):
        _work_of(holes=2)


def test_a_work_record_whose_reasons_do_not_cover_its_refusals_is_refused():
    with pytest.raises(ResultUnusable, match="gives reasons for"):
        _work_of(rejections=dict.fromkeys(REJECTION_REASONS, 0))


def test_a_work_record_with_a_hole_reason_nobody_declared_is_refused():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"felt_wrong": 1}
    with pytest.raises(ResultUnusable, match="declared vocabulary"):
        _work_of(rejections=reasons)


def test_a_work_record_with_an_observation_reason_nobody_declared_is_refused():
    reasons = dict.fromkeys(OBSERVATION_REJECTIONS, 0) | {"felt_wrong": 1}
    with pytest.raises(ResultUnusable, match="declared vocabulary"):
        _work_of(observation_rejections=reasons)


def test_a_work_record_where_candidates_exceed_the_footprint_of_its_observations_is_refused():
    """Bounded by the footprint, not by the observation count.

    One projection may be deposited into every target sample of its footprint,
    so ``candidates > observations`` is legitimate; ``candidates >
    observations * footprint_cells`` is not.
    """
    policy = DEFAULT_POLICY
    cells = policy.footprint_cells
    assert cells == 9
    assert _work_of(observations=1, candidates=cells)
    with pytest.raises(ResultUnusable, match="the only honest relation between them"):
        _work_of(observations=1, candidates=cells + 1)
    with pytest.raises(ResultUnusable, match="the only honest relation between them"):
        _work_of(observations=0, candidates=1)


def test_a_work_record_at_a_zero_footprint_bounds_candidates_by_its_observations():
    policy = TemporalPolicy(footprint_radius=0)
    assert policy.footprint_cells == 1
    assert _work_of(policy=policy, observations=1, candidates=1)
    with pytest.raises(ResultUnusable, match="the only honest relation between them"):
        _work_of(policy=policy, observations=1, candidates=2)


def test_a_work_record_where_confirmations_exceed_accepted_is_refused():
    with pytest.raises(ResultUnusable, match="one of those reconstructions"):
        _work_of(
            accepted=1,
            rejected=0,
            rejections=dict.fromkeys(REJECTION_REASONS, 0),
            confirmed=1,
            contradicted=1,
        )


def test_a_work_record_that_overspent_its_own_budget_is_refused():
    with pytest.raises(ResultUnusable, match="not advisory"):
        _work_of(work_units=10_000, work_budget=10)


def test_a_work_record_naming_support_its_policy_could_not_have_used_is_refused():
    with pytest.raises(ResultUnusable, match="its own policy allows"):
        _work_of(accepted=0, rejected=1, support_samples=5)


def test_a_work_records_rejection_map_cannot_be_written_to_or_changed_underneath_it():
    reasons = dict.fromkeys(REJECTION_REASONS, 0) | {"no_posed_evidence": 1}
    work = _work_of(rejections=reasons)
    reasons["no_posed_evidence"] = 99
    assert work.rejections["no_posed_evidence"] == 1
    with pytest.raises(TypeError):
        work.rejections["no_posed_evidence"] = 99  # type: ignore[index]


# =========================================================================================
# 9. Determinism
# =========================================================================================


def test_the_same_inputs_always_give_a_byte_identical_result():
    target, source1, source2 = _mixed_support_scene()
    first = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    second = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert first.codes == second.codes
    assert first.state_codes == second.state_codes
    assert first.confidence == second.confidence
    assert first.confirmation_codes == second.confirmation_codes
    assert first.temporal_support == second.temporal_support
    assert first.work.as_dict() == second.work.as_dict()


def test_source_frames_must_be_offered_in_strictly_ascending_distinct_index_order():
    """There is no alternate valid order to test order-independence against.

    A support reference names a frame by its index, so
    :func:`complete_temporally` requires the caller's source list to already
    be sorted and distinct - offering the same two frames in the other order
    is refused outright rather than silently reordered, which is what a
    caller checking "did the order matter" would actually need to see.
    """
    target, source1, source2 = _mixed_support_scene()
    with pytest.raises(FrameUnusable, match="distinct and in index order"):
        complete_temporally(target, [source2, source1], camera=NARROW_CAMERA)


# =========================================================================================
# Frame-set construction: acceptance, distinctness, matching rectangles
# =========================================================================================


def test_an_unaccepted_target_pose_is_refused():
    target_codes = [3000] * 25
    target_codes[6] = 0
    unaccepted = _pose(accepted=False, status="refused")
    target = _frame(0, _patch(5, 5, target_codes), unaccepted)
    codes = [0] * 25
    codes[6] = 4000
    source1 = _frame(1, _patch(5, 5, codes), IDENTITY_POSE)
    source2 = _frame(2, _patch(5, 5, codes), _pose(dx=0.1))
    with pytest.raises(PoseUnusable, match="the tracker did not accept"):
        complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)


def test_an_unaccepted_source_pose_is_refused():
    target, source1, _ = _mixed_support_scene()
    unaccepted_source = _frame(2, source1.patch, _pose(accepted=False, status="refused"))
    with pytest.raises(PoseUnusable, match="the tracker did not accept"):
        complete_temporally(target, [source1, unaccepted_source], camera=NARROW_CAMERA)


def test_the_target_frame_cannot_also_be_named_as_a_source():
    target, source1, source2 = _mixed_support_scene()
    retargeted = _frame(0, source2.patch, source2.pose)  # index 0, same as target
    # Sorted ascending (0, 1) so the distinctness/order check passes and the
    # membership check - "both target and source" - is the one that fires.
    with pytest.raises(FrameUnusable, match="both target and source"):
        complete_temporally(target, [retargeted, source1], camera=NARROW_CAMERA)


def test_a_source_of_a_different_rectangle_than_the_target_is_refused():
    target, source1, _ = _mixed_support_scene()
    mismatched = _frame(2, _patch(4, 4, [0] * 16), IDENTITY_POSE)
    with pytest.raises(FrameUnusable, match="one rectangle of one camera"):
        complete_temporally(target, [source1, mismatched], camera=NARROW_CAMERA)


def test_no_source_frames_at_all_is_refused():
    target, _, _ = _mixed_support_scene()
    with pytest.raises(FrameUnusable, match="at least one other frame"):
        complete_temporally(target, [], camera=NARROW_CAMERA)


# =========================================================================================
# TemporalPatch: shape controls not already exercised as forgeries above
# =========================================================================================


def test_the_valid_result_of_a_real_run_is_accepted_by_its_own_recomputation():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    assert dataclasses.replace(result).codes == result.codes


def test_a_result_whose_dimensions_are_not_its_targets_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    with pytest.raises(ResultUnusable, match="does not get to move it"):
        dataclasses.replace(result, width=8, height=2)


def test_a_result_naming_a_target_that_is_not_a_temporal_frame_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    with pytest.raises(ResultUnusable, match="not a TemporalFrame"):
        dataclasses.replace(result, target={"index": 0})  # type: ignore[arg-type]


def test_a_result_naming_a_camera_that_is_not_a_camera_model_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    with pytest.raises(ResultUnusable, match="which is not one"):
        dataclasses.replace(result, camera={"fx": 1})  # type: ignore[arg-type]


def test_a_result_stating_a_schema_version_this_module_does_not_write_is_refused():
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    with pytest.raises(ResultUnusable, match="schema version"):
        dataclasses.replace(result, schema_version=2)


def test_a_results_temporal_support_mapping_does_not_change_when_the_dict_it_was_built_from_does():
    """``TemporalWork.rejections``/``TemporalPatch.temporal_support`` copy on construction.

    ``CameraModel``, ``PoseRecord``, ``TemporalFrame`` and ``TemporalPolicy``
    hold only scalar fields - there is no caller-owned container for them to
    alias, so the aliasing property is meaningful only for the two mapping-
    shaped results, which are exercised here and in the rejections test above.
    """
    target, source1, source2 = _mixed_support_scene()
    result = complete_temporally(target, [source1, source2], camera=NARROW_CAMERA)
    support = dict(result.temporal_support)
    rebuilt = dataclasses.replace(result, temporal_support=support)
    support[6] = ((9, 9),)
    assert rebuilt.support_at(6) == result.support_at(6)
    with pytest.raises(TypeError):
        rebuilt.temporal_support[6] = ((9, 9),)  # type: ignore[index]
