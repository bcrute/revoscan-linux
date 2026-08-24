"""End to end: synthetic frames in, a mesh out, checked against ground truth.

Every other test file checks one module against its own contract. This one
checks that the modules compose, which is a different question and the one that
actually matters: registration that converges, fusion that merges and a mesh
that closes can all be individually correct and still be wrong together.

What makes these tests worth having is that the answer is known *independently*
of the code under test. :mod:`scanner3d.simulate` renders frames of shapes whose
geometry is exact and returns the ground-truth pose of every frame, so the
recovered transforms can be compared against transforms the pipeline was never
told. Nothing here is self-referential.

Deliberately structural rather than precise. Tolerances are generous and stated
as what the pipeline must not be *worse* than, because the point is to catch
composition breaking - a convention flipped, a transform inverted, a stage
silently dropping data - and not to pin numbers that legitimate tuning would
move. The per-module test files own precision.

Two conventions are worth stating once, because both are easy to get backwards
and each produces a plausible-looking wrong scan:

* :func:`simulate.turntable_sweep` returns the poses applied to the *scene*.
  :func:`registration.register_turntable` returns the poses that bring each
  cloud into a *common frame*. Those are inverses of one another.
* A sphere is useless as a turntable target: it looks identical from every
  angle, so it carries no information about how far it turned. The scene here is
  deliberately asymmetric about the spindle.

No assertion in this file states a physical unit, and none asserts accuracy
against a real object - no measurement in this repository has established any.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from scanner3d.fusion import fuse
from scanner3d.geometry import Rigid, pose_difference
from scanner3d.mesh import reconstruct
from scanner3d.registration import register_turntable, voxel_downsample
from scanner3d.simulate import (
    FACTORY_INTRINSICS,
    Box,
    Shape,
    Sphere,
    points_of,
    render,
    sheared_intrinsics,
    turntable_sweep,
)
from scanner3d.validation import (
    CaliperMeasurement,
    FittedQuantity,
    Tolerance,
    Verdict,
    fit_sphere,
    validate,
)

#: Where the object sits: the median working distance measured across the nine
#: committed frames.
SPINDLE: tuple[float, float, float] = (0.0, 0.0, 117.0)
AXIS: tuple[float, float, float] = (0.0, 1.0, 0.0)

#: Few enough to keep this file fast, many enough that consecutive frames
#: overlap substantially - which is what registration actually aligns. Twelve
#: frames is a 30-degree step. Eight (45 degrees) is genuinely too coarse and
#: registration refuses it, which matches the guidance in
#: ``docs/capture-checklist.md`` that steps past about 20 degrees tend to fail.
FRAMES = 12

#: Downsample cell. The correspondence gate must stay comfortably above this or
#: no nearest neighbour can fall inside it.
CELL = 0.6
GATE = 3.0

BOX_HALF = (8.0, 12.0, 14.0)

#: The synthetic ball the M4 rehearsal below measures.
SPHERE_DIAMETER = 40.0


def _at(point: tuple[float, float, float]) -> Rigid:
    return Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), point)


def _scene() -> list[Shape]:
    """An asymmetric scene: a box off-square in x and z, plus an off-axis sphere.

    The box alone is nearly all flat faces, where point-to-point ICP can slide
    along a plane without penalty. The sphere adds curvature that pins the
    remaining freedom. Both matter, and a scene of only one of them is a weaker
    test than it looks.
    """
    return [
        Box(half_extents=BOX_HALF, pose=_at(SPINDLE)),
        Sphere(centre=(10.0, 10.0, 105.0), radius=6.0),
    ]


@pytest.fixture(scope="module")
def swept() -> tuple[tuple, tuple, list]:
    frames, truth = turntable_sweep(_scene(), frames=FRAMES, axis=AXIS, through=SPINDLE)
    clouds = [voxel_downsample(list(points_of(frame)), CELL) for frame in frames]
    return frames, truth, clouds


@pytest.fixture(scope="module")
def registered(swept):
    _, _, clouds = swept
    return register_turntable(
        clouds,
        angle_step=2.0 * math.pi / FRAMES,
        max_correspondence_distance=GATE,
        max_iterations=60,
    )


def test_every_frame_returns_points_so_registration_has_something_to_align(swept):
    frames, _, clouds = swept
    assert len(frames) == FRAMES
    assert all(frame.returned > 0 for frame in frames)
    assert all(len(cloud) > 50 for cloud in clouds)


def test_registration_produces_one_pose_per_frame_with_a_recorded_residual(registered):
    assert len(registered.poses) == FRAMES
    assert [pose.index for pose in registered.poses] == list(range(FRAMES))
    # A residual is a measurement; a transform without one is only an assertion.
    residuals = [pose.rmse for pose in registered.poses if pose.rmse is not None]
    assert len(residuals) >= FRAMES - 1
    assert all(value >= 0.0 for value in residuals)


def test_the_recovered_poses_match_ground_truth_they_were_never_given(swept, registered):
    """The composition check. Poses are compared against the simulator's truth.

    Compared against the *inverse* of the scene pose, because registration
    returns clouds-into-a-common-frame while the sweep returns scene motion. If
    that convention were flipped the error here would be enormous rather than
    subtle, which is exactly what makes this test worth running.
    """
    _, truth, _ = swept
    worst_rotation = max(
        pose_difference(actual.inverse(), pose.transform).rotation_angle
        for pose, actual in zip(registered.poses, truth, strict=True)
    )
    assert math.degrees(worst_rotation) < 5.0, (
        f"worst pose error {math.degrees(worst_rotation):.3f} deg against known truth"
    )


def test_the_first_frame_is_the_reference_and_is_left_alone(registered):
    assert registered.poses[0].transform.rotation_angle == pytest.approx(0.0, abs=1e-9)


def test_loop_closure_is_reported_and_is_small_for_a_closed_sweep(registered):
    """The headline metric: a full revolution must land back where it started."""
    closure = registered.loop_closure
    assert closure is not None, "a full 360-degree sweep must report a closure error"
    assert closure.frames == FRAMES
    assert math.degrees(closure.rotation_error_radians) < 5.0
    assert closure.rotation_error_radians >= 0.0
    assert closure.translation_error >= 0.0


def test_fusion_merges_rather_than_concatenates(swept, registered):
    _, _, clouds = swept
    fused = fuse(clouds, [pose.transform for pose in registered.poses], voxel_size=0.8)
    assert fused.source_frames == FRAMES
    assert fused.input_points == sum(len(cloud) for cloud in clouds)
    # Merging must actually merge: overlapping views collapse onto shared voxels.
    assert len(fused.points) < fused.input_points
    assert fused.occupied_voxels == len(fused.points)


def test_fusion_carries_its_limitations_forward(swept, registered):
    _, _, clouds = swept
    fused = fuse(clouds, [pose.transform for pose in registered.poses], voxel_size=0.8)
    assert fused.limitations, "fusing must not launder away the inputs' uncertainty"


def test_the_pipeline_reaches_a_closed_mesh_of_about_the_right_size(swept, registered):
    """The whole chain, and the one dimensional check the pipeline can honestly make.

    The mesh is compared against the *synthetic* object's own extents, which are
    exact. This says nothing about real-world accuracy - it says the arithmetic
    composes. The tolerance is one-sided and generous on the upper end because
    the mesh module documents, and this repository has measured, that an
    occupancy-grid surface stands off outward by up to about one voxel, so a
    mesh reads larger than the cloud it came from.
    """
    _, _, clouds = swept
    voxel = 1.6
    fused = fuse(clouds, [pose.transform for pose in registered.poses], voxel_size=0.8)
    mesh = reconstruct(fused.points, voxel_size=voxel)

    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert mesh.is_watertight

    xs = [vertex[0] for vertex in mesh.vertices]
    zs = [vertex[2] for vertex in mesh.vertices]
    # The sphere sits off-axis, so it widens x beyond the box alone; z is the
    # box's own extent and is the cleaner check.
    true_z = 2.0 * BOX_HALF[2]
    measured_z = max(zs) - min(zs)
    assert true_z <= measured_z <= true_z + 4.0 * voxel, (
        f"z extent {measured_z:.2f} against a true {true_z:.2f} and a documented "
        f"outward offset of up to one voxel per side"
    )
    assert max(xs) - min(xs) > 0.0


def test_a_mesh_from_unvalidated_depth_says_so(swept, registered):
    _, _, clouds = swept
    fused = fuse(clouds, [pose.transform for pose in registered.poses], voxel_size=0.8)
    mesh = reconstruct(fused.points, voxel_size=1.6)
    joined = " ".join(mesh.limitations).lower()
    assert "unvalidated" in joined or "no physical measurement" in joined
    # The closure caveat must travel with the artifact: a closed mesh is a
    # property of the algorithm, not evidence the object was seen all round.
    assert "every side" in joined or "closes" in joined


# --- the M4 workflow, rehearsed against ground truth --------------------------


def _caliper() -> CaliperMeasurement:
    return CaliperMeasurement(
        label="ball bearing diameter",
        value=SPHERE_DIAMETER,
        uncertainty=0.05,
        measured_by="tests/test_pipeline.py",
        instrument="synthetic; stands in for digital calipers",
    )


def _tolerance() -> Tolerance:
    return Tolerance.declare(
        quantity=FittedQuantity.SPHERE_DIAMETER,
        limit=0.5,
        declared_by="tests/test_pipeline.py",
        rationale="declared before the comparison, which is the whole point of the type",
    )


def _fit_of(intrinsics=None, depth_scale_factor: float = 1.0):
    frame = render([Sphere(centre=(0.0, 0.0, 117.0), radius=SPHERE_DIAMETER / 2.0)])
    if intrinsics is not None:
        frame = dataclasses.replace(frame, intrinsics=intrinsics)
    if depth_scale_factor != 1.0:
        frame = dataclasses.replace(frame, depth_scale=frame.depth_scale * depth_scale_factor)
    return fit_sphere(list(points_of(frame)))


def test_a_correct_capture_passes_against_its_caliper_measurement():
    # Declared BEFORE the fit is computed. The module enforces this and it is
    # not a formality: writing these two lines the other way round raises
    # ToleranceDeclaredLate, which is how this test was first written.
    tolerance = _tolerance()
    result = validate(_fit_of(), _caliper(), tolerance)
    assert result.passed is True
    assert result.verdict is Verdict.PASS
    assert result.difference is not None
    assert abs(result.difference) < 0.01


def test_a_wrong_depth_scale_fails_rather_than_being_absorbed():
    """A scale error is a real disagreement and must read as one."""
    tolerance = _tolerance()
    result = validate(_fit_of(depth_scale_factor=1.03), _caliper(), tolerance)
    assert result.passed is False
    assert result.verdict is Verdict.FAIL
    assert result.difference is not None and result.difference > tolerance.limit


def test_a_sheared_capture_is_inconclusive_and_not_a_failure():
    """The distinction the harness exists to make, and it is not a subtlety.

    A wrong principal point does not make the diameter *wrong by a knowable
    amount* - it makes the fitted diameter a measurement of the target convolved
    with an unknown shear, which is not a measurement of the target at all. So
    the honest verdict is INCONCLUSIVE, not FAIL: the experiment could not
    decide, and reporting a failure would imply a disagreement whose size means
    something. Reporting a pass would be far worse.
    """
    tolerance = _tolerance()
    sheared = sheared_intrinsics(FACTORY_INTRINSICS, cx=42.43747329711914 + 277.56)
    result = validate(_fit_of(intrinsics=sheared), _caliper(), tolerance)
    assert result.passed is None
    assert result.verdict is Verdict.INCONCLUSIVE
    assert any("shear" in reason for reason in result.inconclusive_reasons)
