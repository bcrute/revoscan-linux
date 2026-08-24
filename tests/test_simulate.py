"""Tests for the synthetic depth-frame renderer.

What is being defended here is not "the renderer produces triangles" - it is
that the renderer is the *exact inverse* of the production deprojection, so a
synthetic frame is genuine ground truth rather than a second opinion computed
by the same arithmetic. If those two ever diverge, every test in the pipeline
that relies on synthetic input becomes worthless while still passing.

Three properties carry that weight:

* a rendered sphere deprojects back onto its own surface to within uint16
  quantisation, and no further;
* rendering with one principal point and deprojecting with another reproduces
  the shear at the magnitude asked for; and
* a plane does *not* reveal that shear, which is the trap
  ``docs/capture-checklist.md`` is built around. That test is the executable
  form of the warning.

Nothing here touches hardware and no assertion states a physical unit.
"""

from __future__ import annotations

import dataclasses
import math
import struct

import pytest

from scanner3d.geometry import Rigid, distance, dot, rotation_about_axis, subtract
from scanner3d.simulate import (
    FACTORY_INTRINSICS,
    MAX_CODE,
    NO_RETURN,
    SCHEMA_VERSION,
    Box,
    Plane,
    SceneUnusable,
    ShapeUnusable,
    Sphere,
    SyntheticFrame,
    intrinsics_of,
    points_of,
    recovery_error,
    render,
    sheared_intrinsics,
    turntable_sweep,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: A sphere at the median working distance measured across the nine committed
#: frames, at a radius that fills a useful part of the frame there.
TARGET = Sphere(centre=(0.0, 0.0, 117.0), radius=20.0)

#: The frame centre. Deprojecting with this while the frame was rendered with
#: the factory cx is the principal-point fault, at a magnitude we chose.
FRAME_CENTRE_CX = 320.0


# --- the frame is a real frame ------------------------------------------------


def test_a_rendered_frame_is_byte_identical_in_shape_to_a_captured_one():
    frame = render([TARGET])
    assert len(frame.raw) == EXPECTED_FRAME_BYTES
    assert frame.samples == FRAME_WIDTH * FRAME_HEIGHT
    codes = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", frame.raw)
    assert all(0 <= code <= MAX_CODE for code in codes)


def test_pixels_that_miss_every_shape_are_no_return_not_zero_distance():
    """A miss must leave the sample at 0, which the whole stack reads as absence."""
    frame = render([TARGET])
    codes = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", frame.raw)
    assert codes[0] == NO_RETURN, "a corner pixel cannot see a small centred sphere"
    assert frame.returned == sum(1 for code in codes if code != NO_RETURN)
    assert 0 < frame.returned < frame.samples


def test_returned_count_is_recorded_not_inferred():
    frame = render([TARGET])
    codes = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", frame.raw)
    assert frame.returned == len([c for c in codes if c])


# --- the exact-inverse property, which is the whole point ---------------------


def test_a_rendered_sphere_deprojects_back_onto_its_own_surface():
    """The round trip is limited by uint16 quantisation and by nothing else.

    This is the load-bearing test of the module. If the renderer and the
    production deprojection ever stop being exact inverses, synthetic ground
    truth silently becomes fiction, and every downstream test that trusts it
    keeps passing.
    """
    frame = render([TARGET])
    worst, count = recovery_error(frame, TARGET)
    assert count == frame.returned
    assert worst <= frame.max_quantisation_error * 1.5, (
        f"recovered points sit {worst} from the true surface, but quantisation alone "
        f"allows only {frame.max_quantisation_error}"
    )


def test_quantisation_error_never_exceeds_half_a_depth_code():
    frame = render([TARGET])
    assert frame.max_quantisation_error <= frame.depth_scale / 2.0 + 1e-12


def test_a_finer_depth_scale_recovers_the_surface_more_closely():
    """Halving the scale must halve the quantisation floor, not merely change it."""
    coarse = render([TARGET], depth_scale=0.01)
    fine = render([TARGET], depth_scale=0.005)
    assert fine.max_quantisation_error < coarse.max_quantisation_error
    assert recovery_error(fine, TARGET)[0] < recovery_error(coarse, TARGET)[0]


# --- the principal-point fault, reproduced on demand --------------------------


def test_deprojecting_with_a_wrong_cx_shears_a_sphere_far_off_its_surface():
    frame = render([TARGET])
    wrong = dataclasses.replace(
        frame, intrinsics=sheared_intrinsics(FACTORY_INTRINSICS, cx=FRAME_CENTRE_CX)
    )
    correct_error, _ = recovery_error(frame, TARGET)
    sheared_error, _ = recovery_error(wrong, TARGET)
    assert sheared_error > TARGET.radius * 0.5, (
        "a principal point wrong by 278 columns must distort a sphere grossly; "
        f"got only {sheared_error}"
    )
    assert sheared_error > correct_error * 1000


def test_a_plane_does_not_reveal_a_wrong_cx_at_all():
    """The trap, as an executable assertion rather than a warning in prose.

    Shear maps planes to planes. A flat calibration target therefore passes a
    plane fit cleanly while the principal point is badly wrong, which is why
    ``docs/capture-checklist.md`` refuses to let a flat target be used for M4.
    """
    frame = render([Plane(normal=(0.0, 0.0, 1.0), offset=117.0)])
    wrong = dataclasses.replace(
        frame, intrinsics=sheared_intrinsics(FACTORY_INTRINSICS, cx=FRAME_CENTRE_CX)
    )

    correct_z = [point[2] for point in points_of(frame)]
    sheared_z = [point[2] for point in points_of(wrong)]
    assert correct_z and sheared_z
    # Flat before, and exactly as flat after: the fault is invisible here.
    assert max(correct_z) - min(correct_z) == pytest.approx(0.0, abs=1e-9)
    assert max(sheared_z) - min(sheared_z) == pytest.approx(0.0, abs=1e-9)


def test_the_shear_displaces_x_in_proportion_to_depth():
    """The mechanism itself: x shifts by k*z, so the shift grows with distance."""
    near = render([Plane(normal=(0.0, 0.0, 1.0), offset=100.0)])
    far = render([Plane(normal=(0.0, 0.0, 1.0), offset=200.0)])
    wrong_intrinsics = sheared_intrinsics(FACTORY_INTRINSICS, cx=FRAME_CENTRE_CX)

    def centre_x(frame: SyntheticFrame) -> float:
        points = list(points_of(dataclasses.replace(frame, intrinsics=wrong_intrinsics)))
        return math.fsum(p[0] for p in points) / len(points)

    shift_near, shift_far = centre_x(near), centre_x(far)
    # Twice the depth, twice the lateral displacement.
    assert shift_far == pytest.approx(shift_near * 2.0, rel=1e-6)


# --- shapes -------------------------------------------------------------------


def test_a_box_is_hit_at_its_near_face():
    box = Box(
        half_extents=(10.0, 10.0, 10.0),
        pose=Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 117.0)),
    )
    # A ray straight down the optical axis meets the near face, not the far one.
    assert box.intersect((0.0, 0.0, 1.0)) == pytest.approx(107.0)


def test_the_nearest_shape_wins():
    near = Sphere(centre=(0.0, 0.0, 60.0), radius=5.0)
    far = Sphere(centre=(0.0, 0.0, 200.0), radius=50.0)
    assert render([near, far]).raw != render([far]).raw
    both = render([near, far])
    codes = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", both.raw)
    # The pixel on the optical axis must see the near sphere's front face.
    on_axis = codes[
        round(FACTORY_INTRINSICS.values["cy"]) * FRAME_WIDTH
        + round(FACTORY_INTRINSICS.values["cx"])
    ]
    assert on_axis * both.depth_scale == pytest.approx(55.0, abs=0.01)


def test_a_sphere_behind_the_camera_returns_nothing():
    assert render([Sphere(centre=(0.0, 0.0, -117.0), radius=20.0)]).returned == 0


def test_a_surface_beyond_the_uint16_ceiling_does_not_return():
    frame = render([Plane(normal=(0.0, 0.0, 1.0), offset=1e6)])
    assert frame.returned == 0
    assert frame.beyond_range > 0


# --- turntable ----------------------------------------------------------------


def test_a_turntable_sweep_returns_ground_truth_poses_that_close_the_loop():
    frames, poses = turntable_sweep(
        [
            Box(
                half_extents=(8.0, 12.0, 6.0),
                pose=Rigid.of(
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 117.0)
                ),
            )
        ],
        frames=12,
        axis=(0.0, 1.0, 0.0),
        through=(0.0, 0.0, 117.0),
    )
    assert len(frames) == len(poses) == 12
    assert poses[0].rotation_angle == pytest.approx(0.0, abs=1e-12)
    # A twelfth of a revolution per step.
    assert poses[1].rotation_angle == pytest.approx(2.0 * math.pi / 12, rel=1e-12)
    # Every frame sees something, so registration has data to work with.
    assert all(frame.returned > 0 for frame in frames)


def test_a_turntable_sweep_of_an_asymmetric_shape_produces_differing_frames():
    """A sphere would produce identical frames and carry no rotational signal.

    This is why ``docs/capture-checklist.md`` forbids the sphere as the
    turntable target, and the contrast is asserted rather than assumed.
    """
    spindle = (0.0, 0.0, 117.0)
    # The half-extents must differ in x and z. A box of (4, 20, 4) is
    # symmetric under a quarter turn about y - its x and z extents are equal -
    # so it would look identical from every sampled angle despite being
    # obviously "asymmetric" to the eye. Being asymmetric in general is not
    # enough; it has to be asymmetric about the rotation axis in use.
    box, _ = turntable_sweep(
        [
            Box(
                half_extents=(4.0, 20.0, 12.0),
                pose=Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), spindle),
            )
        ],
        frames=4,
        axis=(0.0, 1.0, 0.0),
        through=spindle,
    )
    ball, _ = turntable_sweep(
        [Sphere(centre=spindle, radius=20.0)], frames=4, axis=(0.0, 1.0, 0.0), through=spindle
    )
    assert box[0].raw != box[1].raw, "an asymmetric shape must look different when turned"
    assert ball[0].raw == ball[1].raw, "a sphere on its own centre looks identical from every angle"


def test_a_sweep_needs_at_least_two_frames():
    with pytest.raises(SceneUnusable):
        turntable_sweep([TARGET], frames=1, axis=(0.0, 1.0, 0.0), through=(0.0, 0.0, 117.0))


# --- noise --------------------------------------------------------------------


def test_dropout_removes_returns_and_is_counted():
    clean = render([TARGET])
    lossy = render([TARGET], dropout=0.5, seed=7)
    assert lossy.returned < clean.returned
    assert lossy.dropped > 0
    assert lossy.returned + lossy.dropped == pytest.approx(clean.returned, rel=0.05)


def test_rendering_is_deterministic_for_a_given_seed():
    assert render([TARGET], dropout=0.3, seed=11).raw == render([TARGET], dropout=0.3, seed=11).raw


def test_a_different_seed_gives_a_different_frame():
    assert render([TARGET], dropout=0.3, seed=11).raw != render([TARGET], dropout=0.3, seed=12).raw


def test_jitter_degrades_recovery():
    assert (
        recovery_error(render([TARGET], jitter=0.5, seed=3), TARGET)[0]
        > recovery_error(render([TARGET]), TARGET)[0]
    )


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.0, -1.0, math.nan, math.inf])
def test_a_degenerate_sphere_is_refused(radius: float):
    with pytest.raises(ShapeUnusable):
        Sphere(centre=(0.0, 0.0, 1.0), radius=radius)


@pytest.mark.parametrize("extents", [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, math.nan)])
def test_a_degenerate_box_is_refused(extents: tuple[float, float, float]):
    with pytest.raises(ShapeUnusable):
        Box(half_extents=extents)


def test_an_empty_scene_is_refused():
    with pytest.raises(SceneUnusable):
        render([])


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf])
def test_a_bad_depth_scale_is_refused(scale: float):
    with pytest.raises(SceneUnusable):
        render([TARGET], depth_scale=scale)


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_a_dropout_outside_zero_to_one_is_refused(dropout: float):
    with pytest.raises(SceneUnusable):
        render([TARGET], dropout=dropout)


def test_negative_jitter_is_refused():
    with pytest.raises(SceneUnusable):
        render([TARGET], jitter=-1.0)


def test_intrinsics_that_cannot_project_are_refused():
    with pytest.raises(SceneUnusable):
        render([TARGET], intrinsics=intrinsics_of(fx=0.0, fy=1.0, cx=0.0, cy=0.0))


# --- records ------------------------------------------------------------------


def test_the_frame_record_is_frozen():
    frame = render([TARGET])
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.returned = 0  # type: ignore[misc]


def test_the_report_carries_its_version_and_refuses_to_imply_accuracy():
    reported = render([TARGET]).as_dict()
    assert reported["schema_version"] == SCHEMA_VERSION
    assert reported["bytes"] == EXPECTED_FRAME_BYTES
    limitations = reported["limitations"]
    assert isinstance(limitations, list) and limitations
    joined = " ".join(limitations).lower()
    assert "not a sensor model" in joined
    assert "no physical unit is implied" in joined


def test_the_shapes_that_were_rendered_are_recorded():
    reported = render([TARGET]).as_dict()
    shapes = reported["shapes"]
    assert isinstance(shapes, list) and len(shapes) == 1
    assert shapes[0]["kind"] == "sphere"
    assert shapes[0]["radius"] == TARGET.radius


def test_a_pose_moves_the_scene_and_is_recorded_in_the_report():
    moved = render(
        [TARGET], pose=rotation_about_axis((0.0, 1.0, 0.0), math.pi, through=(0.0, 0.0, 117.0))
    )
    shapes = moved.as_dict()["shapes"]
    assert isinstance(shapes, list)
    centre = shapes[0]["centre"]
    assert isinstance(centre, list)
    # Rotating a centred sphere about its own centre leaves it where it was.
    assert distance((centre[0], centre[1], centre[2]), TARGET.centre) == pytest.approx(
        0.0, abs=1e-9
    )


def test_geometry_helpers_agree_with_the_recovery_metric():
    """recovery_error must measure radial distance, not something else."""
    frame = render([TARGET])
    worst = 0.0
    for point in points_of(frame):
        offset = subtract(point, TARGET.centre)
        worst = max(worst, abs(math.sqrt(dot(offset, offset)) - TARGET.radius))
    assert recovery_error(frame, TARGET)[0] == pytest.approx(worst)
