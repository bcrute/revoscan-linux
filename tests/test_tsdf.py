"""What the reference TSDF field must be true of before anything is built on it.

The claims that matter here are geometric, not structural. A field that
allocates the right number of voxels and reports tidy counters can still have
its sign inverted, its pose composed backwards or its truncation applied to the
wrong side of the surface, and every one of those defects produces a result
that still looks like a scan. So the tests below measure recovered geometry
against shapes whose truth is known exactly, and the refusal tests exist to
prove that a bad input is stopped before it can become one of those
plausible-looking wrong answers.
"""

from __future__ import annotations

import itertools
import math
import statistics
import struct
from types import SimpleNamespace

import pytest

from scanner3d import depth_completion, depth_temporal, tsdf_input
from scanner3d import tsdf as tsdf_module
from scanner3d.cloud_export import camera_points
from scanner3d.depth_completion import DepthPatch, complete_patch
from scanner3d.depth_metric import (
    UNVALIDATED_UNIT,
    DepthFrameSizeMismatch,
    DepthIntrinsicsUnusable,
    DepthScaleUnusable,
    deproject_depth_frame,
)
from scanner3d.errors import ScannerError
from scanner3d.geometry import IDENTITY_MATRIX, Rigid, rotation_about_axis
from scanner3d.mesh import reconstruct
from scanner3d.simulate import FACTORY_INTRINSICS, Plane, Sphere, intrinsics_of, render
from scanner3d.tsdf import (
    AUTO_BACKEND,
    INHERENT_LIMITATIONS,
    MAX_ACCUMULATED_WEIGHT,
    MAX_FRAMES_PER_CALL,
    MAX_VOXELS,
    MAX_ZERO_SAMPLES,
    PYTHON_BACKEND_MAX_VOXELS,
    DepthRangeUnusable,
    FrameCountExceeded,
    IgnoredReason,
    PoseCountMismatch,
    TsdfBackendUnavailable,
    TsdfBackendUnknown,
    TsdfConfig,
    TsdfNotAllowed,
    TsdfVolume,
    VolumeBounds,
    VolumeBoundsUnusable,
    VolumeEmpty,
    VolumeTooLarge,
    VoxelParametersUnusable,
    WeightPolicy,
    WeightPolicyUnusable,
    ZeroLevelCeilingExceeded,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_WIDTH
from tests import tsdf_benchmark

# A centred, symmetric camera. The device's own factory record puts cx at 42 of
# 640 columns, which is real and is exercised elsewhere; here a centred record
# keeps the synthetic scene inside the frame from every direction so that a
# geometric miss is a defect in the fusion rather than in the framing.
INTRINSICS = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)

#: The unit's measured depth scale. Its physical meaning is unestablished; it is
#: used because the quantisation it implies is the real one.
DEPTH_SCALE = 0.004999999888241291

#: The synthetic target, in MODEL coordinates: ground truth that no recovered
#: surface is allowed to disagree with.
SPHERE_CENTRE = (0.0, 0.0, 100.0)
SPHERE_RADIUS = 20.0

#: The camera orbits this point, which is deliberately NOT the sphere's centre.
#: An orbit about the centre of a sphere is a symmetry of the scene: every frame
#: would be identical, a reversed pose would fuse just as well as a right one,
#: and a rotation error would be invisible. Twenty units in front of the centre
#: is enough to break that symmetry and make the transform tests bite.
ORBIT_CENTRE = (0.0, 0.0, 80.0)

VIEWS = 8


def orbit_poses(count: int = VIEWS, *, offset_degrees: float = 0.0) -> list[Rigid]:
    """The scene transforms one per view; ``render``'s ``pose`` argument."""
    return [
        rotation_about_axis(
            (0.0, 1.0, 0.0),
            2.0 * math.pi * index / count + math.radians(offset_degrees),
            through=ORBIT_CENTRE,
        )
        for index in range(count)
    ]


def camera_to_model(scene_poses: list[Rigid]) -> list[Rigid]:
    """``render(pose=P)`` shows the camera ``P(scene)``, so camera->model is ``P^-1``."""
    return [pose.inverse() for pose in scene_poses]


def sphere_views(*, jitter: float) -> tuple[list[bytes], list[Rigid]]:
    scene = orbit_poses()
    frames = [
        render(
            [Sphere(centre=SPHERE_CENTRE, radius=SPHERE_RADIUS)],
            intrinsics=INTRINSICS,
            depth_scale=DEPTH_SCALE,
            pose=pose,
            jitter=jitter,
            # A different stream per view: the point of several views is that
            # their errors are independent, and one seed would make them
            # identical and the averaging a no-op.
            seed=index,
        ).raw
        for index, pose in enumerate(scene)
    ]
    return frames, camera_to_model(scene)


def sphere_config(**overrides: object) -> TsdfConfig:
    settings: dict[str, object] = {
        "bounds": VolumeBounds.of((-25.0, -25.0, 75.0), (25.0, 25.0, 125.0)),
        "voxel_size": 1.0,
        "truncation": 3.0,
        "min_depth": 30.0,
        "max_depth": 200.0,
    }
    settings.update(overrides)
    return TsdfConfig.of(**settings)  # type: ignore[arg-type]


def fuse(
    frames: list[bytes],
    poses: list[Rigid],
    *,
    config: TsdfConfig | None = None,
    backend: str = AUTO_BACKEND,
) -> TsdfVolume:
    volume = TsdfVolume(config if config is not None else sphere_config(), backend=backend)
    volume.integrate(frames, poses, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    return volume


def radial_error(points: tuple[tuple[float, float, float], ...]) -> tuple[float, float]:
    """``(median, rms)`` distance from the true sphere surface. The whole measure."""
    assert points, "no points were recovered at all, so there is nothing to measure"
    errors = [abs(math.dist(point, SPHERE_CENTRE) - SPHERE_RADIUS) for point in points]
    return statistics.median(errors), math.sqrt(statistics.fmean(e * e for e in errors))


@pytest.fixture(scope="module")
def plane_frame() -> bytes:
    """One frame of a plane at exactly z = 100, seen head on."""
    return render(
        [Plane(normal=(0.0, 0.0, 1.0), offset=100.0)],
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
    ).raw


@pytest.fixture(scope="module")
def plane_volume(plane_frame: bytes) -> TsdfVolume:
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-40.0, -30.0, 90.0), (40.0, 30.0, 110.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=50.0,
        max_depth=200.0,
    )
    return fuse([plane_frame], [Rigid.identity()], config=config)


#: A power-of-two depth scale, used only where an EXACT depth is the point of
#: the test: 100.0 / 0.0078125 is 12800 exactly, and 12800 * 0.0078125 is
#: 100.0 exactly, so the plane survives the uint16 round trip with no
#: quantisation at all. The device's own DEPTH_SCALE is not a power of two and
#: puts the same plane at 99.99999776 - true to the hardware, and precisely
#: wrong for a test about a value landing on a voxel centre exactly.
EXACT_DEPTH_SCALE = 0.0078125

#: Where the grid-aligned plane is, in model coordinates.
GRID_PLANE_Z = 100.0


def grid_plane_config() -> TsdfConfig:
    """Bounds whose z centres are the integers: 89.5 + (k + 0.5) is 90.0 + k.

    So the k = 10 slab of voxel centres lies EXACTLY on the plane at z = 100.
    Small enough, at 41 x 31 x 20 cells, for the pure-Python backend's own
    ceiling to allow it, because both backends have to agree here.
    """
    return TsdfConfig.of(
        bounds=VolumeBounds.of((-20.5, -15.5, 89.5), (20.5, 15.5, 109.5)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=50.0,
        max_depth=200.0,
    )


def fuse_grid_plane(frame: bytes, *, backend: str = AUTO_BACKEND) -> TsdfVolume:
    volume = TsdfVolume(grid_plane_config(), backend=backend)
    volume.integrate(
        [frame], [Rigid.identity()], intrinsics=INTRINSICS, depth_scale=EXACT_DEPTH_SCALE
    )
    return volume


@pytest.fixture(scope="module")
def grid_plane_frame() -> bytes:
    """One frame of the plane at z = 100, at the scale that keeps it exact."""
    return render(
        [Plane(normal=(0.0, 0.0, 1.0), offset=GRID_PLANE_Z)],
        intrinsics=INTRINSICS,
        depth_scale=EXACT_DEPTH_SCALE,
    ).raw


@pytest.fixture(scope="module")
def grid_plane_volume(grid_plane_frame: bytes) -> TsdfVolume:
    return fuse_grid_plane(grid_plane_frame)


@pytest.fixture(scope="module")
def clean_views() -> tuple[list[bytes], list[Rigid]]:
    return sphere_views(jitter=0.0)


@pytest.fixture(scope="module")
def noisy_views() -> tuple[list[bytes], list[Rigid]]:
    return sphere_views(jitter=0.5)


# --- the configuration refuses before it allocates ----------------------------


def test_a_configuration_states_its_shape_cells_and_cost_before_a_voxel_exists():
    config = sphere_config()
    assert config.shape == (50, 50, 50)
    assert config.cell_count == 125_000
    # Two float32 values per voxel, and a reader can check the arithmetic.
    assert config.owned_bytes == 125_000 * 8
    assert config.realised_maximum == (25.0, 25.0, 125.0)
    assert config.voxel_centre(0, 0, 0) == (-24.5, -24.5, 75.5)


def test_a_volume_whose_extent_does_not_divide_evenly_covers_the_bounds_not_less():
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((0.0, 0.0, 0.0), (2.5, 1.0, 1.0)),
        voxel_size=1.0,
        truncation=2.0,
        min_depth=1.0,
        max_depth=10.0,
    )
    assert config.shape == (3, 1, 1)
    assert config.realised_maximum[0] == 3.0
    assert config.realised_maximum[0] > config.bounds.max_x


def test_bounds_that_are_not_a_finite_strictly_ordered_box_are_refused():
    with pytest.raises(VolumeBoundsUnusable) as raised:
        VolumeBounds.of((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    assert "strictly ordered on x" in str(raised.value)
    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.of((0.0, 2.0, 0.0), (1.0, 1.0, 1.0))
    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.of((0.0, 0.0, 0.0), (1.0, math.inf, 1.0))
    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.of((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_a_box_measured_around_points_grows_by_the_margin_and_refuses_an_empty_cloud():
    bounds = VolumeBounds.around([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)], margin=0.5)
    assert bounds.minimum == (-0.5, -0.5, -0.5)
    assert bounds.maximum == (1.5, 2.5, 3.5)
    # A flat cloud still yields a box, because the margin opens it on its own.
    flat = VolumeBounds.around([(0.0, 0.0, 5.0), (1.0, 1.0, 5.0)], margin=0.25)
    assert flat.extent[2] == pytest.approx(0.5)
    with pytest.raises(VolumeEmpty) as raised:
        VolumeBounds.around([], margin=1.0)
    assert "no points were offered" in str(raised.value)
    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.around([(0.0, 0.0, math.nan)], margin=1.0)
    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.around([(0.0, 0.0, 0.0)], margin=-1.0)


def test_a_point_that_is_not_exactly_three_coordinates_is_refused_by_index_and_arity():
    """A cloud of the wrong shape is the caller's mistake, and is named as one.

    Not an IndexError leaking out of the middle of the measurement for a short
    point, and emphatically not a box silently built from the first three
    coordinates of a longer one - which would be a volume around a quantity
    nobody asked for, reported as though it were the scan's extent.
    """
    with pytest.raises(VolumeBoundsUnusable) as short:
        VolumeBounds.around([(0.0, 0.0, 0.0), (1.0, 2.0)], margin=1.0)
    assert "point 1" in str(short.value)
    assert "2 coordinates" in str(short.value)

    with pytest.raises(VolumeBoundsUnusable) as extra:
        VolumeBounds.around([(0.0, 0.0, 0.0, 4.0)], margin=1.0)
    assert "point 0" in str(extra.value)
    assert "4 coordinates" in str(extra.value)

    with pytest.raises(VolumeBoundsUnusable):
        VolumeBounds.around([()], margin=1.0)

    # Three coordinates is a point, whatever sequence type carries them.
    assert VolumeBounds.around([[1.0, 2.0, 3.0]], margin=1.0).minimum == (0.0, 1.0, 2.0)


def test_a_voxel_size_or_truncation_that_cannot_define_a_field_is_refused():
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(VoxelParametersUnusable):
            sphere_config(voxel_size=bad)
        with pytest.raises(VoxelParametersUnusable):
            sphere_config(truncation=bad)


def test_a_truncation_narrower_than_the_voxel_size_is_refused_because_the_surface_would_vanish():
    with pytest.raises(VoxelParametersUnusable) as raised:
        sphere_config(voxel_size=1.0, truncation=0.5)
    message = str(raised.value)
    assert "BETWEEN neighbouring voxel centres" in message
    assert "disappear" in message
    # Equal is allowed; it is narrow, not incoherent.
    assert sphere_config(voxel_size=1.0, truncation=1.0).truncation == 1.0


def test_a_depth_interval_no_observation_could_fall_inside_is_refused():
    with pytest.raises(DepthRangeUnusable):
        sphere_config(min_depth=200.0, max_depth=30.0)
    with pytest.raises(DepthRangeUnusable):
        sphere_config(min_depth=100.0, max_depth=100.0)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(DepthRangeUnusable):
            sphere_config(min_depth=bad)
        with pytest.raises(DepthRangeUnusable):
            sphere_config(max_depth=bad)


def test_the_weight_cap_is_bounded_and_the_policy_must_be_one_this_module_implements():
    for bad in (0.0, -1.0, math.nan, math.inf, MAX_ACCUMULATED_WEIGHT + 1.0):
        with pytest.raises(WeightPolicyUnusable):
            sphere_config(max_weight=bad)
    with pytest.raises(WeightPolicyUnusable) as raised:
        sphere_config(weight_policy="exponential")
    assert "not one of" in str(raised.value)
    # A valid member's own string is accepted, so a config can survive JSON.
    assert sphere_config(weight_policy="ramped_behind").weight_policy is WeightPolicy.RAMPED_BEHIND


def test_a_volume_past_the_cell_ceiling_is_refused_rather_than_allocated():
    with pytest.raises(VolumeTooLarge) as raised:
        sphere_config(voxel_size=0.01)
    assert "nothing was allocated" in str(raised.value)
    # And a caller's own lower ceiling binds too.
    sphere_config(voxel_size=5.0, truncation=10.0, max_voxels=2000)
    with pytest.raises(VolumeTooLarge):
        sphere_config(voxel_size=5.0, truncation=10.0, max_voxels=999)


def test_the_cell_ceiling_cannot_be_widened_at_run_time():
    with pytest.raises(VolumeTooLarge):
        sphere_config(max_voxels=MAX_VOXELS + 1)
    with pytest.raises(VolumeTooLarge):
        sphere_config(max_voxels=0)


def test_the_python_backend_carries_its_own_lower_wall_clock_ceiling():
    config = sphere_config(voxel_size=0.5, truncation=1.5)
    assert config.cell_count > PYTHON_BACKEND_MAX_VOXELS
    assert config.cell_count <= MAX_VOXELS
    with pytest.raises(VolumeTooLarge) as raised:
        TsdfVolume(config, backend="python")
    message = str(raised.value)
    assert "pure-Python backend" in message
    assert "nothing was allocated" in message
    # The same volume is legal on the accelerated path.
    assert TsdfVolume(config, backend=AUTO_BACKEND).config.cell_count == config.cell_count


# --- backends -----------------------------------------------------------------


def test_an_unknown_backend_is_refused_and_no_gpu_backend_is_offered():
    config = sphere_config()
    with pytest.raises(TsdfBackendUnknown) as raised:
        TsdfVolume(config, backend="gpu")
    message = str(raised.value)
    assert "not one of" in message
    assert "ROCm" in message
    for name in ("rocm", "hip", "opencl", "vulkan", "cuda", "scipy"):
        with pytest.raises(TsdfBackendUnknown):
            TsdfVolume(config, backend=name)


def test_an_unavailable_backend_raises_rather_than_silently_using_another(monkeypatch):
    monkeypatch.setattr(
        tsdf_module, "_probe_numpy", lambda: (False, "numpy is absent in this test")
    )
    with pytest.raises(TsdfBackendUnavailable) as raised:
        TsdfVolume(sphere_config(), backend="numpy")
    message = str(raised.value)
    assert "numpy is absent in this test" in message
    assert "rather than silently using a different backend" in message


def test_without_numpy_auto_resolves_to_python_and_the_field_still_fuses(monkeypatch, plane_frame):
    monkeypatch.setattr(tsdf_module, "_probe_numpy", lambda: (False, "numpy is absent"))
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-24.0, -24.0, 88.0), (24.0, 24.0, 112.0)),
        voxel_size=3.0,
        truncation=6.0,
        min_depth=50.0,
        max_depth=200.0,
    )
    volume = TsdfVolume(config, backend=AUTO_BACKEND)
    assert volume.backend == "python"
    volume.integrate(
        [plane_frame], [Rigid.identity()], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE
    )
    assert volume.voxel_updates > 0
    assert volume.zero_level_samples().point_count > 0


def test_numpy_is_never_imported_into_this_module_s_namespace():
    """The optionality claim, checked rather than asserted in prose."""
    assert "numpy" not in vars(tsdf_module)
    assert "np" not in vars(tsdf_module)


# --- the transform contract ---------------------------------------------------


def test_a_pose_carries_camera_points_into_the_model_frame_and_integration_inverts_it(
    clean_views,
):
    """The direction, proved by where the surface lands rather than by reading code.

    Every camera in this sweep both rotates AND translates relative to the
    model - the orbit is about a point 20 units in front of the sphere - so a
    transform that composed the other way, or applied the rotation without the
    translation, would place each frame's surface somewhere else and the fused
    zero level would not be a sphere of radius 20 at (0, 0, 100).
    """
    frames, poses = clean_views
    assert any(pose.translation_magnitude > 1.0 for pose in poses)
    assert any(pose.rotation_angle > 0.5 for pose in poses)
    median, rms = radial_error(fuse(frames, poses).zero_level_samples().points)
    assert median < 0.15, f"median radial error {median!r} is more than a sixth of a voxel"
    assert rms < 0.30, f"rms radial error {rms!r}"


def test_feeding_the_poses_the_other_way_round_is_caught_by_the_recovered_surface(clean_views):
    """A reversed composition must not merely look worse; it must be measurable."""
    frames, poses = clean_views
    right = radial_error(fuse(frames, poses).zero_level_samples().points)
    reversed_direction = [pose.inverse() for pose in poses]
    wrong = radial_error(fuse(frames, reversed_direction).zero_level_samples().points)
    assert wrong[1] > 5.0 * right[1], (
        f"reversing the pose direction moved the rms radial error only from {right[1]!r} to "
        f"{wrong[1]!r}; this scene does not discriminate the composition order"
    )


def test_a_voxel_in_front_of_the_observed_surface_is_positive_and_one_behind_is_negative(
    plane_volume,
):
    """The sign convention, at four voxels whose true distances are exact.

    The plane sits at z = 100 and the voxel centres either side of it are at
    99.5 and 100.5, so the signed distances are +0.5 and -0.5 exactly, and the
    truncation normalises them to +/- 1/6.
    """
    in_front = plane_volume.sample_at(40, 30, 9)
    behind = plane_volume.sample_at(40, 30, 10)
    assert in_front.centre[2] == 99.5
    assert behind.centre[2] == 100.5
    assert in_front.value == pytest.approx(0.5 / 3.0, abs=1e-6)
    assert behind.value == pytest.approx(-0.5 / 3.0, abs=1e-6)
    assert in_front.distance(3.0) == pytest.approx(0.5, abs=1e-5)
    assert behind.distance(3.0) == pytest.approx(-0.5, abs=1e-5)
    assert in_front.observed and behind.observed


def test_values_are_clamped_in_front_and_left_unobserved_too_far_behind(plane_volume):
    far_in_front = plane_volume.sample_at(40, 30, 5)  # z = 95.5, 4.5 in front
    assert far_in_front.value == pytest.approx(1.0)
    assert far_in_front.observed
    just_inside = plane_volume.sample_at(40, 30, 12)  # z = 102.5, 2.5 behind
    assert just_inside.value == pytest.approx(-2.5 / 3.0, abs=1e-6)
    beyond = plane_volume.sample_at(40, 30, 13)  # z = 103.5, 3.5 behind
    assert not beyond.observed, "a voxel past the truncation must take nothing from the frame"
    assert beyond.value == 0.0


def test_a_voxel_index_outside_the_volume_is_refused(plane_volume):
    nx, ny, nz = plane_volume.config.shape
    for index in ((nx, 0, 0), (0, ny, 0), (0, 0, nz), (-1, 0, 0)):
        with pytest.raises(VolumeBoundsUnusable):
            plane_volume.sample_at(*index)


# --- what contributes nothing, and why ----------------------------------------


def counts(volume: TsdfVolume) -> dict[IgnoredReason, int]:
    return {entry.reason: entry.count for entry in volume.evidence().ignored}


def test_a_zero_sample_is_no_return_and_is_never_fused_as_a_surface_at_the_camera():
    """An all-zero frame is a frame the sensor returned nothing for, not a wall at z = 0."""
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-5.0, -5.0, 95.0), (5.0, 5.0, 105.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    volume = fuse([bytes(EXPECTED_FRAME_BYTES)], [Rigid.identity()], config=config)
    assert volume.voxel_updates == 0
    assert volume.frames_offered == 1
    assert volume.frames_integrated == 0
    assert counts(volume)[IgnoredReason.NO_RETURN] == config.cell_count
    evidence = volume.evidence()
    assert evidence.weights.observed_voxels == 0
    assert any("NO voxel was observed" in note for note in evidence.limitations)


def test_depths_outside_the_accepted_interval_contribute_nothing(plane_frame):
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-5.0, -5.0, 95.0), (5.0, 5.0, 105.0)),
        voxel_size=1.0,
        truncation=3.0,
        # The plane really is at 100; this interval excludes it deliberately.
        min_depth=150.0,
        max_depth=200.0,
    )
    volume = fuse([plane_frame], [Rigid.identity()], config=config)
    assert volume.voxel_updates == 0
    assert counts(volume)[IgnoredReason.DEPTH_OUT_OF_RANGE] == config.cell_count


def test_voxels_behind_the_camera_or_outside_the_image_contribute_nothing(plane_frame):
    behind = TsdfConfig.of(
        bounds=VolumeBounds.of((-5.0, -5.0, -105.0), (5.0, 5.0, -95.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    volume = fuse([plane_frame], [Rigid.identity()], config=behind)
    assert counts(volume)[IgnoredReason.BEHIND_CAMERA] == behind.cell_count
    beside = TsdfConfig.of(
        bounds=VolumeBounds.of((500.0, -5.0, 95.0), (510.0, 5.0, 105.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    aside = fuse([plane_frame], [Rigid.identity()], config=beside)
    assert counts(aside)[IgnoredReason.OUTSIDE_IMAGE] == beside.cell_count
    assert aside.voxel_updates == 0


def test_the_accounting_partitions_every_frame_times_voxel_pair(noisy_views):
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    evidence = volume.evidence()
    expected = len(frames) * volume.config.cell_count
    assert evidence.voxel_observations == expected
    assert evidence.voxel_updates + sum(e.count for e in evidence.ignored) == expected
    assert evidence.voxel_updates > 0
    assert evidence.frames_integrated == len(frames)
    assert evidence.frames_without_updates == 0


# --- weights ------------------------------------------------------------------


def test_accumulated_weight_is_bounded_by_the_cap_and_the_cap_is_reported(noisy_views):
    frames, poses = noisy_views
    volume = fuse(frames, poses, config=sphere_config(max_weight=3.0))
    weights = volume.evidence().weights
    assert weights.maximum is not None
    assert weights.maximum <= 3.0
    assert weights.minimum is not None and weights.minimum > 0.0
    assert weights.at_cap > 0
    assert any("weight cap" in note for note in volume.evidence().limitations)
    # And the range is measured over observed voxels only, never over the whole
    # volume, which would report a minimum of zero and say nothing.
    assert weights.observed_voxels + weights.unobserved_voxels == volume.config.cell_count
    assert weights.unobserved_voxels > 0


def test_the_ramped_policy_weighs_the_occluded_band_behind_a_surface_less_than_the_front(
    plane_frame,
):
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-5.0, -5.0, 90.0), (5.0, 5.0, 110.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=30.0,
        max_depth=200.0,
        weight_policy=WeightPolicy.RAMPED_BEHIND,
    )
    volume = fuse([plane_frame], [Rigid.identity()], config=config)
    in_front = volume.sample_at(5, 5, 5)  # z = 95.5, in front of the plane
    behind = volume.sample_at(5, 5, 12)  # z = 102.5, 2.5 behind it
    assert in_front.weight == pytest.approx(1.0)
    assert behind.weight == pytest.approx(1.0 - 2.5 / 3.0, abs=1e-6)
    assert 0.0 < behind.weight < in_front.weight
    # Uniform, by contrast, weighs both the same.
    uniform = fuse(
        [plane_frame],
        [Rigid.identity()],
        config=TsdfConfig.of(
            bounds=config.bounds,
            voxel_size=1.0,
            truncation=3.0,
            min_depth=30.0,
            max_depth=200.0,
            weight_policy=WeightPolicy.UNIFORM,
        ),
    )
    assert uniform.sample_at(5, 5, 12).weight == pytest.approx(1.0)


# --- recovered geometry -------------------------------------------------------


def test_an_exact_plane_is_recovered_at_the_plane(plane_volume):
    """No noise, no dropout: the only error left is depth quantisation and float32."""
    samples = plane_volume.zero_level_samples()
    assert samples.point_count > 1000
    depths = [point[2] for point in samples.points]
    worst = max(abs(depth - 100.0) for depth in depths)
    # The bound is the quantisation step, not an arbitrary tolerance: a code is
    # 0.005 units and the field is float32.
    assert worst < 0.01, f"a zero-level sample stood {worst!r} off an exactly known plane"


def test_a_curved_surface_is_recovered_within_a_fraction_of_a_voxel(clean_views):
    frames, poses = clean_views
    volume = fuse(frames, poses)
    points = volume.zero_level_samples().points
    median, rms = radial_error(points)
    voxel = volume.config.voxel_size
    assert median < 0.15 * voxel, f"median radial error {median!r} at voxel {voxel!r}"
    assert rms < 0.30 * voxel, f"rms radial error {rms!r} at voxel {voxel!r}"
    # And no sample wanders off the surface entirely.
    worst = max(abs(math.dist(point, SPHERE_CENTRE) - SPHERE_RADIUS) for point in points)
    assert worst < 1.5 * voxel, f"a zero-level sample stood {worst!r} off the sphere"


def test_several_noisy_views_recover_the_surface_better_than_one(noisy_views):
    """The claim that makes a signed-distance field worth having: averaging.

    Occupancy cannot do this at all - a union only grows - so if this ever
    stopped holding, the whole reason for the module would have gone with it.
    """
    frames, poses = noisy_views
    one = radial_error(fuse(frames[:1], poses[:1]).zero_level_samples().points)
    many = radial_error(fuse(frames, poses).zero_level_samples().points)
    assert many[0] < 0.75 * one[0], f"median went {one[0]!r} -> {many[0]!r} on {len(frames)} views"
    assert many[1] < 0.75 * one[1], f"rms went {one[1]!r} -> {many[1]!r} on {len(frames)} views"


def test_the_fused_zero_level_beats_the_occupancy_surface_on_the_same_noisy_views(noisy_views):
    """Head to head, same frames, same poses, same cell size, same measure.

    The comparison the task turns on. Occupancy marks a cell solid if any
    sample fell in it and marches the boundary of that solid, so its surface
    stands about a voxel outside the samples and cannot average noise away.
    """
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    voxel = volume.config.voxel_size
    fused = radial_error(volume.zero_level_samples().points)

    # The same frames, deprojected through the production path and placed with
    # the same poses: the only difference between the two surfaces is what is
    # done with the points once they are in the model frame.
    model_points: list[tuple[float, float, float]] = []
    for raw, pose in zip(frames, poses, strict=True):
        conversion = deproject_depth_frame(
            raw, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE, unit=UNVALIDATED_UNIT
        )
        model_points.extend(pose.apply_all(camera_points(conversion)))
    occupancy = radial_error(reconstruct(model_points, voxel_size=voxel).vertices)

    assert fused[0] < 0.5 * occupancy[0], (
        f"median distance to truth: tsdf {fused[0]!r} vs occupancy {occupancy[0]!r}"
    )
    assert fused[1] < 0.5 * occupancy[1], (
        f"rms distance to truth: tsdf {fused[1]!r} vs occupancy {occupancy[1]!r}"
    )
    # Not merely smoother: the occupancy surface really does stand outside.
    assert occupancy[0] > 0.5 * voxel


def test_a_wrong_pose_measurably_worsens_the_fused_zero_level(noisy_views):
    """Fusion depends on tracking and must not appear to substitute for it."""
    frames, poses = noisy_views
    truthful = radial_error(fuse(frames, poses).zero_level_samples().points)

    drifted = camera_to_model(orbit_poses(offset_degrees=5.0))
    everywhere = radial_error(fuse(frames, drifted).zero_level_samples().points)
    assert everywhere[0] > 3.0 * truthful[0], (
        f"a 5-degree pose error moved the median only from {truthful[0]!r} to {everywhere[0]!r}"
    )
    assert everywhere[1] > 3.0 * truthful[1]

    one_bad = list(poses)
    one_bad[3] = rotation_about_axis(
        (0.0, 1.0, 0.0),
        2.0 * math.pi * 3 / VIEWS + math.radians(8.0),
        through=ORBIT_CENTRE,
    ).inverse()
    single = radial_error(fuse(frames, one_bad).zero_level_samples().points)
    assert single[1] > 3.0 * truthful[1], (
        f"one wrong pose in {VIEWS} moved the rms only from {truthful[1]!r} to {single[1]!r}"
    )


# --- zero-level samples -------------------------------------------------------


def test_zero_level_samples_are_taken_only_between_observed_opposite_sign_neighbours(
    noisy_views,
):
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    generous = volume.zero_level_samples(min_weight=1.0)
    strict = volume.zero_level_samples(min_weight=4.0)
    assert 0 < strict.point_count < generous.point_count, (
        "raising the weight threshold must narrow the surface to the better-observed part"
    )
    assert generous.crossings_found == generous.point_count
    with pytest.raises(TsdfNotAllowed):
        volume.zero_level_samples(min_weight=0.0)
    with pytest.raises(TsdfNotAllowed):
        volume.zero_level_samples(min_weight=math.nan)


def test_a_plane_lying_exactly_on_the_voxel_centres_is_still_recovered(grid_plane_volume):
    """A zero AT a voxel centre is evidence, and discarding it erases the plane.

    A surface aligned to the grid has no strictly opposite-sign neighbour pair
    anywhere along it: the voxels on the plane hold exactly zero, and zero
    times either neighbour is not negative. A rule that reports only sign
    changes returns NOTHING here - not a worse plane, no plane at all - which
    is why a voxel that was observed and holds exactly zero is a point in its
    own right, at its own centre.
    """
    config = grid_plane_volume.config
    nx, ny, _ = config.shape
    samples = grid_plane_volume.zero_level_samples()

    assert samples.edge_crossings == 0, "an aligned plane has no strict sign change to cross"
    assert samples.exact_zero_voxels == nx * ny, "every voxel centre on the plane is a sample"
    assert samples.point_count == samples.crossings_found == nx * ny

    # On the plane by equality, not by tolerance: the depth scale is a power of
    # two and the centre is 89.5 + 10.5, so both sides are exact.
    assert {point[2] for point in samples.points} == {GRID_PLANE_Z}
    # Once each. A voxel centre on the level set is one point, not one per axis.
    assert len(set(samples.points)) == samples.point_count

    # The voxels on the plane hold zero with a weight behind it...
    on_plane = grid_plane_volume.sample_at(nx // 2, ny // 2, 10)
    assert on_plane.centre[2] == GRID_PLANE_Z
    assert on_plane.value == 0.0
    assert on_plane.observed
    # ...and the same volume is full of the OTHER zero, which stays excluded:
    # had those counted, this field would report thousands more points.
    unobserved = grid_plane_volume.sample_at(nx // 2, ny // 2, 15)
    assert unobserved.value == 0.0
    assert not unobserved.observed
    # The weight threshold governs exact zeros exactly as it governs crossings.
    assert grid_plane_volume.zero_level_samples(min_weight=2.0).point_count == 0


def test_the_ceiling_counts_observed_exact_zeros_before_a_point_is_built(grid_plane_volume):
    """What the limit is checked against is what materialising would produce."""
    found = grid_plane_volume.zero_level_samples().crossings_found
    assert found > 1000
    with pytest.raises(ZeroLevelCeilingExceeded) as raised:
        grid_plane_volume.zero_level_samples(limit=found - 1)
    message = str(raised.value)
    assert "nothing was materialised" in message
    assert str(found) in message
    assert "exactly-zero voxels" in message
    # Exactly at the count is allowed, and it is all exact zeros here.
    exact = grid_plane_volume.zero_level_samples(limit=found)
    assert exact.point_count == found
    assert exact.exact_zero_voxels == found


def test_an_unobserved_volume_yields_no_zero_level_at_all():
    """Zero means "no evidence", and pairing against it would invent a surface."""
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-5.0, -5.0, 95.0), (5.0, 5.0, 105.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    volume = TsdfVolume(config)
    samples = volume.zero_level_samples()
    assert samples.point_count == 0
    assert samples.crossings_found == 0
    # Every voxel in this field holds exactly zero, and not one of them is a
    # point: what makes an exact zero a sample is the weight behind it.
    assert samples.exact_zero_voxels == 0
    assert samples.edge_crossings == 0


def test_the_zero_sample_ceiling_is_enforced_before_a_single_point_is_built(noisy_views):
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    found = volume.zero_level_samples().crossings_found
    assert found > 10
    with pytest.raises(ZeroLevelCeilingExceeded) as raised:
        volume.zero_level_samples(limit=found - 1)
    message = str(raised.value)
    assert "nothing was materialised" in message
    assert str(found) in message
    # Exactly at the count is allowed.
    assert volume.zero_level_samples(limit=found).point_count == found


def test_the_zero_sample_ceiling_cannot_be_widened_at_run_time(plane_volume):
    with pytest.raises(ZeroLevelCeilingExceeded):
        plane_volume.zero_level_samples(limit=MAX_ZERO_SAMPLES + 1)
    with pytest.raises(ZeroLevelCeilingExceeded):
        plane_volume.zero_level_samples(limit=0)


def test_zero_level_samples_say_plainly_that_they_are_not_a_mesh(plane_volume):
    samples = plane_volume.zero_level_samples()
    rendered = samples.as_dict()
    assert rendered["connectivity"] == (
        "none: these are unconnected samples of the zero level, not a mesh"
    )
    assert any("NOT a mesh" in note for note in samples.limitations)
    assert any("connected TSDF surface exists" in note for note in samples.limitations)


# --- determinism and backend agreement ----------------------------------------


def coarse_config(policy: WeightPolicy = WeightPolicy.UNIFORM) -> TsdfConfig:
    """Small enough for the pure-Python backend's own ceiling to allow it."""
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-24.0, -24.0, 76.0), (24.0, 24.0, 124.0)),
        voxel_size=3.0,
        truncation=6.0,
        min_depth=30.0,
        max_depth=200.0,
        weight_policy=policy,
    )
    assert config.cell_count <= PYTHON_BACKEND_MAX_VOXELS
    return config


@pytest.mark.parametrize("policy", list(WeightPolicy))
def test_the_two_backends_produce_the_same_field_and_the_same_zero_level(noisy_views, policy):
    """One arithmetic, two implementations: any divergence is a defect in one of them."""
    frames, poses = noisy_views
    accelerated = fuse(frames, poses, config=coarse_config(policy), backend="numpy")
    reference = fuse(frames, poses, config=coarse_config(policy), backend="python")
    assert accelerated.backend == "numpy"
    assert reference.backend == "python"
    assert accelerated.voxel_updates == reference.voxel_updates
    assert counts(accelerated) == counts(reference)

    nx, ny, nz = coarse_config(policy).shape
    for i in range(0, nx, 3):
        for j in range(0, ny, 3):
            for k in range(0, nz, 3):
                one = accelerated.sample_at(i, j, k)
                two = reference.sample_at(i, j, k)
                assert one.value == pytest.approx(two.value, abs=1e-6), (i, j, k)
                assert one.weight == pytest.approx(two.weight, abs=1e-6), (i, j, k)
    assert accelerated.zero_level_samples().points == reference.zero_level_samples().points


def test_the_two_backends_agree_on_a_surface_that_lies_on_the_voxel_centres(grid_plane_frame):
    """The exact-zero pass is two implementations as well, so it gets the same test."""
    accelerated = fuse_grid_plane(grid_plane_frame, backend="numpy").zero_level_samples()
    reference = fuse_grid_plane(grid_plane_frame, backend="python").zero_level_samples()
    assert accelerated.exact_zero_voxels == reference.exact_zero_voxels > 0
    assert accelerated.edge_crossings == reference.edge_crossings
    assert accelerated.points == reference.points


def test_the_same_frames_and_poses_produce_the_same_result_every_time(noisy_views):
    frames, poses = noisy_views
    first = fuse(frames, poses)
    second = fuse(frames, poses)
    assert first.zero_level_samples().points == second.zero_level_samples().points
    assert first.evidence().as_dict() == second.evidence().as_dict()


def test_integrating_the_same_frames_in_two_calls_matches_integrating_them_in_one(noisy_views):
    frames, poses = noisy_views
    together = fuse(frames, poses)
    split = TsdfVolume(sphere_config())
    split.integrate(frames[:3], poses[:3], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    split.integrate(frames[3:], poses[3:], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert split.frames_offered == together.frames_offered
    assert split.voxel_updates == together.voxel_updates
    assert split.zero_level_samples().points == together.zero_level_samples().points


# --- what integration refuses -------------------------------------------------


def test_a_frame_and_pose_count_mismatch_is_refused_before_anything_is_integrated(clean_views):
    frames, poses = clean_views
    volume = TsdfVolume(sphere_config())
    with pytest.raises(PoseCountMismatch) as raised:
        volume.integrate(frames, poses[:-1], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert "Nothing was integrated" in str(raised.value)
    assert volume.frames_offered == 0
    assert volume.voxel_updates == 0


def test_more_frames_than_the_per_call_ceiling_are_refused(plane_frame):
    volume = TsdfVolume(sphere_config())
    frames = [plane_frame] * (MAX_FRAMES_PER_CALL + 1)
    poses = [Rigid.identity()] * (MAX_FRAMES_PER_CALL + 1)
    with pytest.raises(FrameCountExceeded):
        volume.integrate(frames, poses, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert volume.frames_offered == 0


def test_a_buffer_that_is_not_one_frame_of_the_capture_mode_is_refused(plane_frame):
    volume = TsdfVolume(sphere_config())
    with pytest.raises(DepthFrameSizeMismatch):
        volume.integrate(
            [plane_frame, plane_frame[:-2]],
            [Rigid.identity(), Rigid.identity()],
            intrinsics=INTRINSICS,
            depth_scale=DEPTH_SCALE,
        )
    # The first frame was legal and is still not integrated: the gate runs over
    # every frame before the first one is projected.
    assert volume.frames_offered == 0
    assert volume.voxel_updates == 0


def test_intrinsics_or_a_scale_the_projection_cannot_use_are_refused(plane_frame):
    volume = TsdfVolume(sphere_config())
    with pytest.raises(DepthIntrinsicsUnusable):
        volume.integrate(
            [plane_frame],
            [Rigid.identity()],
            intrinsics=intrinsics_of(fx=0.0, fy=400.0, cx=320.0, cy=200.0),
            depth_scale=DEPTH_SCALE,
        )
    for bad_scale in (0.0, -1.0, math.nan):
        with pytest.raises(DepthScaleUnusable):
            volume.integrate(
                [plane_frame], [Rigid.identity()], intrinsics=INTRINSICS, depth_scale=bad_scale
            )
    assert volume.frames_offered == 0


# --- evidence -----------------------------------------------------------------


def test_the_evidence_reports_what_ran_and_claims_no_accuracy(noisy_views):
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    evidence = volume.evidence(elapsed_seconds=1.25)
    rendered = evidence.as_dict()

    assert rendered["backend"] == volume.backend
    assert rendered["dimensional_accuracy_claimed"] is False
    frames_block = rendered["frames"]
    assert isinstance(frames_block, dict)
    assert frames_block["offered"] == VIEWS
    assert frames_block["integrated"] == VIEWS
    timing = rendered["timing"]
    assert isinstance(timing, dict)
    assert timing["elapsed_seconds"] == 1.25
    config_block = rendered["config"]
    assert isinstance(config_block, dict)
    assert config_block["sign_convention"].startswith("sdf = observed camera-z")
    grid = config_block["grid"]
    assert isinstance(grid, dict)
    assert grid["cells"] == volume.config.cell_count
    assert grid["shape"] == [50, 50, 50]
    assert grid["module_cell_ceiling"] == MAX_VOXELS
    for note in INHERENT_LIMITATIONS:
        assert note in evidence.limitations
    assert any("PROJECTIVE" in note for note in evidence.limitations)
    assert any("does not replace it" in note for note in evidence.limitations)


def test_a_timing_is_the_callers_measurement_and_is_never_invented(noisy_views):
    frames, poses = noisy_views
    volume = fuse(frames, poses)
    assert volume.evidence().elapsed_seconds is None
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(TsdfNotAllowed):
            volume.evidence(elapsed_seconds=bad)


def test_an_empty_field_says_it_is_not_a_scan_of_anything():
    volume = TsdfVolume(sphere_config())
    evidence = volume.evidence()
    assert evidence.frames_offered == 0
    assert evidence.voxel_updates == 0
    assert evidence.weights.minimum is None
    assert evidence.weights.maximum is None
    assert any("not a scan of anything" in note for note in evidence.limitations)


def test_evidence_is_a_snapshot_that_later_integration_does_not_reach_back_into(clean_views):
    frames, poses = clean_views
    volume = TsdfVolume(sphere_config())
    volume.integrate(frames[:1], poses[:1], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    first = volume.evidence()
    volume.integrate(frames[1:], poses[1:], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert first.frames_offered == 1
    assert volume.evidence().frames_offered == VIEWS
    assert first.voxel_updates < volume.evidence().voxel_updates


# --- the benchmark harness's camera boundary ----------------------------------
#
# tests/tsdf_benchmark.py is deliberately not collected, because what it
# measures - wall clock, resident memory - is a property of the host and
# asserting bounds on it would produce a test that fails on a slower machine.
# WHICH CAMERA a measurement projected with is not that kind of fact. It is a
# property of the measurement, and getting it wrong is exactly how round 1 of
# this task produced real-capture counts for a camera the frames were never
# taken with: the harness registered through the device's factory record and
# fused through its own synthetic one. So the harness's camera boundary is
# tested here, deterministically, and nothing below asserts a timing.


#: Neither the harness's synthetic record nor the device's: a longer focal
#: length and an off-centre principal point, so a fusion that quietly used the
#: harness's own record instead of this one lands somewhere else and counts
#: something else.
BENCHMARK_CALLER_CAMERA = tsdf_benchmark.FusionCamera(
    label="the caller's camera, which is not the harness's own",
    intrinsics=intrinsics_of(fx=520.0, fy=520.0, cx=250.0, cy=170.0),
    depth_scale=DEPTH_SCALE,
)


def test_the_benchmark_fuses_with_the_camera_it_was_given_and_not_its_own(clean_views):
    frames, poses = clean_views
    two_frames, two_poses = list(frames[:2]), list(poses[:2])

    def fused_through(camera: tsdf_benchmark.FusionCamera) -> TsdfVolume:
        volume = TsdfVolume(sphere_config(), backend="numpy")
        volume.integrate(
            two_frames, two_poses, intrinsics=camera.intrinsics, depth_scale=camera.depth_scale
        )
        return volume

    row = tsdf_benchmark.measure(
        two_frames,
        two_poses,
        sphere_config(),
        backend="numpy",
        truth=False,
        camera=BENCHMARK_CALLER_CAMERA,
    )
    supplied = fused_through(BENCHMARK_CALLER_CAMERA)
    harness_own = fused_through(tsdf_benchmark.SYNTHETIC_CAMERA)

    assert row["voxel_updates"] == supplied.evidence().voxel_updates
    assert row["voxel_observations"] == supplied.evidence().voxel_observations
    assert row["observed_voxels"] == supplied.evidence().weights.observed_voxels
    assert row["zero_samples"] == supplied.zero_level_samples().point_count

    # The two records are genuinely distinguishable on these frames, which is
    # what gives the assertions above their force: had `measure` projected with
    # its own INTRINSICS, every one of them would have failed.
    assert supplied.evidence().voxel_updates != harness_own.evidence().voxel_updates
    assert row["voxel_updates"] != harness_own.evidence().voxel_updates
    assert row["zero_samples"] != harness_own.zero_level_samples().point_count


def test_the_benchmark_reports_the_exact_camera_and_depth_scale_it_fused_with(clean_views):
    frames, poses = clean_views
    row = tsdf_benchmark.measure(
        list(frames[:1]),
        list(poses[:1]),
        sphere_config(),
        backend="numpy",
        truth=False,
        camera=BENCHMARK_CALLER_CAMERA,
    )

    reported = row["fusion_camera"]
    assert isinstance(reported, dict)
    assert reported == BENCHMARK_CALLER_CAMERA.as_dict()
    assert reported["fx"] == 520.0
    assert reported["fy"] == 520.0
    assert reported["cx"] == 250.0
    assert reported["cy"] == 170.0
    assert reported["width"] == 640
    assert reported["height"] == 400
    assert reported["depth_scale"] == DEPTH_SCALE
    # The report answers "which camera was this?" on its own. A reader auditing
    # the JSON never has to infer the projection from the harness's source.
    assert reported != tsdf_benchmark.SYNTHETIC_CAMERA.as_dict()
    assert reported != tsdf_benchmark.CAPTURE_CAMERA.as_dict()


def test_the_registration_clouds_are_deprojected_through_the_camera_supplied(plane_frame):
    through_synthetic = tsdf_benchmark.registration_clouds(
        [plane_frame], camera=tsdf_benchmark.SYNTHETIC_CAMERA
    )
    through_factory = tsdf_benchmark.registration_clouds(
        [plane_frame], camera=tsdf_benchmark.CAPTURE_CAMERA
    )

    assert through_synthetic[0] and through_factory[0]
    assert through_synthetic != through_factory
    # A real capture's camera is the device's own record, and only that.
    assert tsdf_benchmark.CAPTURE_CAMERA.intrinsics is FACTORY_INTRINSICS
    assert tsdf_benchmark.CAPTURE_CAMERA.intrinsics != tsdf_benchmark.SYNTHETIC_CAMERA.intrinsics


def test_the_real_capture_path_deprojects_and_fuses_through_one_record(
    tmp_path, monkeypatch, plane_frame
):
    """The round-2 defect itself: registration and fusion must not diverge.

    Registration and the fusion's cost are stubbed out - neither is what this
    test is about, and running them would make it a slow test of somebody
    else's code. What is exercised for real is the harness's own path from
    capture directory to both projections.
    """
    from scanner3d import depth_metric, pipeline

    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "frame_0000.raw").write_bytes(plane_frame)

    deprojections: list[tuple[object, float]] = []
    genuine_deproject = depth_metric.deproject_depth_frame

    def watched_deproject(raw, *, intrinsics, depth_scale, unit):
        deprojections.append((intrinsics, depth_scale))
        return genuine_deproject(raw, intrinsics=intrinsics, depth_scale=depth_scale, unit=unit)

    fusions: list[tsdf_benchmark.FusionCamera] = []

    def stub_measure(frames, poses, config, *, backend, truth, camera):
        fusions.append(camera)
        return {"case": "cost not measured here"}

    def stub_sweep(clouds, **options):
        return SimpleNamespace(
            poses=tuple(Rigid.identity() for _ in clouds),
            frames=len(clouds),
            revisits_found=0,
            optimised=False,
            converged=True,
            limitations=(),
        )

    monkeypatch.setattr(depth_metric, "deproject_depth_frame", watched_deproject)
    monkeypatch.setattr(pipeline, "reconstruct_sweep", stub_sweep)
    monkeypatch.setattr(tsdf_benchmark, "measure", stub_measure)

    report = tsdf_benchmark.real_capture(capture, stride=1, backend="python", workers=1)

    assert len(deprojections) == 1
    assert len(fusions) == 1
    registered_with, registered_scale = deprojections[0]
    # The SAME record object reaches both halves, so no future edit can move one
    # of them without moving the other.
    assert registered_with is fusions[0].intrinsics
    assert registered_scale == fusions[0].depth_scale
    assert registered_with is FACTORY_INTRINSICS
    assert fusions[0].intrinsics is not tsdf_benchmark.SYNTHETIC_CAMERA.intrinsics

    registration = report["registration"]
    assert isinstance(registration, dict)
    assert registration["deprojection_camera"] == fusions[0].as_dict()


# =========================================================================================
# Provenance-weighted integration
# =========================================================================================
#
# Everything below is about ``integrate_weighted`` and the accepted
# ``scanner3d.tsdf_input`` contract behind it. The claims that matter are
# arithmetic rather than structural: a weight that does not move the zero
# crossing in the direction its evidence points has not been applied, and a
# weight of zero that is not exactly absence would let unconfirmed geometry
# into the field through a discount.
#
# The scene is one 16x16 patch of a real 640x400 frame, placed so that a small
# volume in front of it projects entirely into the patch's interior. It is
# small on purpose: every number here is checkable by hand, and the multi-view
# corpus in tests/depth_completeness.py is where this is exercised at scale.

WEIGHTED_PATCH = 16
WEIGHTED_ORIGIN = (312, 192)

#: Whole codes at this scale, so a depth is exactly ``code * 0.05``.
WEIGHTED_SCALE = 0.05
NEAR_CODE = 2000  # depth 100.0
FAR_CODE = 2020  # depth 101.0

#: The camera the accepted temporal primitive is given. Its principal point is
#: stated in PATCH coordinates, because a TemporalFrame's patch carries its own
#: origin in the frame; the TSDF is given the frame-coordinate record above.
WEIGHTED_CAMERA = depth_temporal.CameraModel(
    fx=INTRINSICS.values["fx"],
    fy=INTRINSICS.values["fy"],
    cx=INTRINSICS.values["cx"] - WEIGHTED_ORIGIN[0],
    cy=INTRINSICS.values["cy"] - WEIGHTED_ORIGIN[1],
    depth_scale=WEIGHTED_SCALE,
)


def weighted_config(**overrides: object) -> TsdfConfig:
    """A volume small enough that every voxel projects into the patch interior."""
    settings: dict[str, object] = {
        "bounds": VolumeBounds.of((-0.5, -0.5, 99.0), (0.5, 0.5, 102.0)),
        "voxel_size": 0.25,
        "truncation": 0.75,
        "min_depth": 50.0,
        "max_depth": 200.0,
    }
    settings.update(overrides)
    return TsdfConfig.of(**settings)  # type: ignore[arg-type]


def patch_frame(code: int) -> bytes:
    """A real 512000-byte frame carrying ``code`` across the patch and zero elsewhere."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    u0, v0 = WEIGHTED_ORIGIN
    for index in range(WEIGHTED_PATCH * WEIGHTED_PATCH):
        row, column = divmod(index, WEIGHTED_PATCH)
        at = ((v0 + row) * FRAME_WIDTH + (u0 + column)) * 2
        struct.pack_into("<H", buffer, at, code)
    return bytes(buffer)


def temporal_pose(dx: float, quality: float) -> depth_temporal.PoseRecord:
    return depth_temporal.PoseRecord(
        transform=Rigid.of(IDENTITY_MATRIX, (dx, 0.0, 0.0)),
        accepted=True,
        status="synthetic weighted-input pose, exact by construction",
        quality=quality,
    )


def temporal_frame(
    index: int, code: int, dx: float, quality: float
) -> depth_temporal.TemporalFrame:
    return depth_temporal.TemporalFrame(
        index=index,
        patch=DepthPatch.of(
            WEIGHTED_PATCH, WEIGHTED_PATCH, [code] * (WEIGHTED_PATCH * WEIGHTED_PATCH)
        ),
        u0=WEIGHTED_ORIGIN[0],
        v0=WEIGHTED_ORIGIN[1],
        pose=temporal_pose(dx, quality),
    )


def accepted_temporal(code: int, *, quality: float = 1.0) -> depth_temporal.TemporalPatch:
    """The accepted temporal completion of an all-holes patch from two posed views.

    ``quality`` is the supporting poses' own accepted quality, which is one of
    the five terms of :func:`scanner3d.depth_temporal.temporal_confidence` and
    the only one this scene varies. Lowering it lowers the accepted confidence
    and nothing else, which is exactly the input the weighting is monotone in.
    """
    return depth_temporal.complete_temporally(
        temporal_frame(0, 0, 0.0, quality),
        [temporal_frame(1, code, 0.0, quality), temporal_frame(2, code, 0.5, quality)],
        camera=WEIGHTED_CAMERA,
    )


def measured_input(code: int) -> tsdf_input.WeightedDepthFrame:
    return tsdf_input.weighted_raw_frame(patch_frame(code))


def reconstructed_input(code: int, *, quality: float = 1.0) -> tsdf_input.WeightedDepthFrame:
    return tsdf_input.weighted_temporal_frame(
        bytes(EXPECTED_FRAME_BYTES),
        accepted_temporal(code, quality=quality),
        origin=WEIGHTED_ORIGIN,
    )


def _place(codes: list[int]) -> bytes:
    """One real frame carrying ``codes`` across the patch and zero elsewhere."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    u0, v0 = WEIGHTED_ORIGIN
    for index, value in enumerate(codes):
        row, column = divmod(index, WEIGHTED_PATCH)
        struct.pack_into("<H", buffer, ((v0 + row) * FRAME_WIDTH + (u0 + column)) * 2, value)
    return bytes(buffer)


def _dropout_codes(code: int) -> list[int]:
    """``code`` across the patch with every third sample dropped.

    Enough returns for the accepted single-frame primitive to have support, and
    enough holes for it to propose values for - which is what makes the
    excluded-provenance claim testable rather than vacuous.
    """
    codes = [code] * (WEIGHTED_PATCH * WEIGHTED_PATCH)
    for index in range(0, len(codes), 3):
        codes[index] = 0
    return codes


def excluded_input(code: int) -> tsdf_input.WeightedDepthFrame:
    """An accepted single-frame estimate: its reconstructions are all excluded."""
    codes = _dropout_codes(code)
    completed = complete_patch(
        DepthPatch.of(WEIGHTED_PATCH, WEIGHTED_PATCH, codes),
        policy=depth_completion.CompletionPolicy(min_support=1),
    )
    return tsdf_input.weighted_completed_frame(_place(codes), completed, origin=WEIGHTED_ORIGIN)


def unweighted_counterfactual(code: int) -> bytes:
    """The same completion's codes as a raw frame: provenance ignored entirely.

    What fusing a single-frame estimate at full weight would offer, which is
    the comparison that makes the exclusion load-bearing rather than a label.
    """
    codes = _dropout_codes(code)
    completed = complete_patch(
        DepthPatch.of(WEIGHTED_PATCH, WEIGHTED_PATCH, codes),
        policy=depth_completion.CompletionPolicy(min_support=1),
    )
    return _place(list(completed.codes))


def weighted_field(
    frames: list[tsdf_input.WeightedDepthFrame],
    *,
    config: TsdfConfig | None = None,
    backend: str = "python",
) -> TsdfVolume:
    volume = TsdfVolume(weighted_config() if config is None else config, backend=backend)
    volume.integrate_weighted(
        frames,
        [Rigid.identity()] * len(frames),
        intrinsics=INTRINSICS,
        depth_scale=WEIGHTED_SCALE,
    )
    return volume


def crossing_depth(volume: TsdfVolume) -> float:
    """Where the field's zero level sits along the central column, in model z.

    The FIRST crossing walking away from the camera, and deliberately not an
    average over every zero-level sample the volume holds. A field fused from
    two surfaces at different depths has a second, spurious crossing where the
    nearer frame's truncation band ends and only the farther frame's evidence
    remains; that one is an artefact of the band and not a surface, and an
    average over both would move with it rather than with the geometry these
    tests are about.

    Read straight out of the stored values through
    :meth:`~scanner3d.tsdf.TsdfVolume.sample_at`, so what is measured is the
    field itself and not the sample bag built from it.
    """
    nx, ny, nz = volume.config.shape
    column = [volume.sample_at(nx // 2, ny // 2, k) for k in range(nz)]
    for near, far in itertools.pairwise(column):
        if not (near.observed and far.observed):
            continue
        if near.value > 0.0 >= far.value:
            span = near.value - far.value
            step = far.centre[2] - near.centre[2]
            return near.centre[2] + step * near.value / span
    raise AssertionError("this field has no zero crossing along its central column")


# --- the legacy path is untouched ---------------------------------------------------------


@pytest.mark.parametrize("backend", ["python", "numpy"])
@pytest.mark.parametrize("policy", list(WeightPolicy))
def test_a_raw_frame_weighted_is_byte_for_byte_the_raw_frame_integrated(
    backend, policy, plane_frame
):
    """The identity that makes the whole contract checkable.

    A measured sample weighs exactly one, so weighted integration of a raw
    frame is multiplication by one - not an approximation of the old path but
    the same arithmetic - and both stored buffers and the whole ignored
    accounting have to come out identical on both backends and both policies.
    """
    config = sphere_config(weight_policy=policy)
    poses = [Rigid.identity()]
    legacy = TsdfVolume(config, backend=backend)
    legacy.integrate([plane_frame], poses, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    weighted = TsdfVolume(config, backend=backend)
    weighted.integrate_weighted(
        [tsdf_input.weighted_raw_frame(plane_frame)],
        poses,
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
    )

    assert bytes(weighted.values) == bytes(legacy.values)
    assert bytes(weighted.weights) == bytes(legacy.weights)
    assert weighted.evidence().as_dict() == legacy.evidence().as_dict()


def test_the_weighted_accounting_is_empty_on_a_volume_built_the_old_way(plane_volume):
    evidence = plane_volume.weighted_evidence()
    assert evidence.frames_offered == 0
    assert evidence.voxel_updates == 0
    assert evidence.contribution == 0.0
    assert evidence.inputs.samples == 0
    assert any("No weighted frame has been offered" in note for note in evidence.limitations)


@pytest.mark.parametrize("policy", list(WeightPolicy))
def test_the_two_backends_produce_the_identical_weighted_field_and_contribution(policy):
    frames = [measured_input(NEAR_CODE), reconstructed_input(FAR_CODE)]
    config = weighted_config(weight_policy=policy)
    reference = weighted_field(frames, config=config, backend="python")
    accelerated = weighted_field(frames, config=config, backend="numpy")

    assert accelerated.backend == "numpy"
    assert bytes(accelerated.values) == bytes(reference.values)
    assert bytes(accelerated.weights) == bytes(reference.weights)
    assert (
        accelerated.weighted_evidence().contribution == reference.weighted_evidence().contribution
    )
    assert (
        accelerated.weighted_evidence().voxel_updates == reference.weighted_evidence().voxel_updates
    )


# --- zero weight is absence ---------------------------------------------------------------


def test_a_weight_of_zero_is_exactly_absence_and_never_a_discount():
    """The property the two excluded provenances exist to have.

    An accepted single-frame completion of a patch full of holes proposes real
    values for them, and every one arrives here excluded. The field it produces
    has to be byte-identical to the field produced by not offering the frame at
    all - not close, identical - because a zero-weight sample that changed a
    voxel would be invented geometry arriving at a discount.
    """
    excluded = excluded_input(NEAR_CODE)
    assert excluded.summary.spatially_excluded > 0

    with_estimate = weighted_field([measured_input(FAR_CODE), excluded])
    without = weighted_field(
        [measured_input(FAR_CODE), tsdf_input.weighted_raw_frame(excluded.raw)]
    )

    assert bytes(with_estimate.values) == bytes(without.values)
    assert bytes(with_estimate.weights) == bytes(without.weights)
    evidence = with_estimate.weighted_evidence()
    assert evidence.inputs.spatially_excluded == excluded.summary.spatially_excluded
    assert evidence.zero_weight_observations > 0
    assert any("contributed exactly zero" in note for note in evidence.limitations)


def test_an_excluded_estimate_observes_no_more_space_than_the_measurements_did():
    """And the counterfactual beside it, so the exclusion is shown to be load-bearing.

    Fusing the same completion's codes at uniform weight - which is what
    ignoring provenance means - observes strictly more voxels. That difference
    is the invented geometry the weighting refuses.
    """
    excluded = excluded_input(NEAR_CODE)
    weighted = weighted_field([excluded])
    measured_only = weighted_field([tsdf_input.weighted_raw_frame(excluded.raw)])

    assert bytes(weighted.values) == bytes(measured_only.values)
    assert bytes(weighted.weights) == bytes(measured_only.weights)

    ignoring_provenance = TsdfVolume(weighted_config(), backend="python")
    ignoring_provenance.integrate(
        [unweighted_counterfactual(NEAR_CODE)],
        [Rigid.identity()],
        intrinsics=INTRINSICS,
        depth_scale=WEIGHTED_SCALE,
    )
    assert (
        ignoring_provenance.evidence().weights.observed_voxels
        > weighted.evidence().weights.observed_voxels
    )


# --- the weighting moves the field in the direction its evidence points --------------------


def test_two_depths_at_unequal_confidence_put_the_crossing_nearer_the_stronger_evidence():
    """The analytical claim of this task, in one measurement.

    Two temporal reconstructions of the same voxels disagree by a whole unit of
    depth. Nothing distinguishes them except the accepted confidence behind
    each, and the zero level has to land nearer the one the evidence is
    stronger for - and to move the other way when the two confidences are
    swapped.
    """
    near_strong = weighted_field(
        [
            reconstructed_input(NEAR_CODE, quality=1.0),
            reconstructed_input(FAR_CODE, quality=0.25),
        ]
    )
    far_strong = weighted_field(
        [
            reconstructed_input(NEAR_CODE, quality=0.25),
            reconstructed_input(FAR_CODE, quality=1.0),
        ]
    )
    near = NEAR_CODE * WEIGHTED_SCALE
    far = FAR_CODE * WEIGHTED_SCALE

    assert near < crossing_depth(near_strong) < crossing_depth(far_strong) < far


def test_lowering_an_accepted_confidence_monotonically_reduces_its_influence():
    near = NEAR_CODE * WEIGHTED_SCALE
    crossings = [
        crossing_depth(
            weighted_field(
                [measured_input(NEAR_CODE), reconstructed_input(FAR_CODE, quality=quality)]
            )
        )
        for quality in (1.0, 0.8, 0.5, 0.25)
    ]
    # Every step down in accepted confidence pulls the crossing back towards
    # the measurement, and none of them ever reaches it.
    assert all(high > low for high, low in itertools.pairwise(crossings))
    assert all(crossing > near for crossing in crossings)


def test_a_measurement_dominates_an_otherwise_equal_reconstruction():
    """Same geometry, same disagreement, and only the provenance differs."""
    near = NEAR_CODE * WEIGHTED_SCALE
    far = FAR_CODE * WEIGHTED_SCALE
    measured_near = weighted_field([measured_input(NEAR_CODE), reconstructed_input(FAR_CODE)])
    measured_far = weighted_field([reconstructed_input(NEAR_CODE), measured_input(FAR_CODE)])
    midpoint = 0.5 * (near + far)

    assert crossing_depth(measured_near) < midpoint < crossing_depth(measured_far)


def _outvoting_frames() -> list[tsdf_input.WeightedDepthFrame]:
    """One measurement, and two reconstructions that disagree with it."""
    return [
        measured_input(NEAR_CODE),
        reconstructed_input(FAR_CODE),
        reconstructed_input(FAR_CODE),
    ]


def test_two_reconstructions_do_not_outvote_the_one_measurement_they_contradict():
    """The reason the ceiling is where it is, under the uniform policy.

    Two agreeing reconstructions at the top of the accepted confidence range
    together weigh less than one measurement, so the zero level stays on the
    measured side of the midpoint between the two depths.
    """
    near = NEAR_CODE * WEIGHTED_SCALE
    outvoted = weighted_field(
        _outvoting_frames(), config=weighted_config(weight_policy=WeightPolicy.UNIFORM)
    )
    assert crossing_depth(outvoted) < 0.5 * (near + FAR_CODE * WEIGHTED_SCALE)


def test_the_ramped_policy_and_the_provenance_weight_multiply_rather_than_replace():
    """The two weightings compose, and a reader choosing a policy has to know it.

    :attr:`WeightPolicy.RAMPED_BEHIND` already discounts an observation towards
    zero BEHIND the surface it observed, which is what it exists to do: that
    band is occluded and is where the projective approximation is worst. A
    reconstruction of a farther surface sits in FRONT of its own observation
    there, at full geometric weight, so in that band it can outweigh the
    measurement whose ramp has run down - and the crossing moves away from the
    measurement rather than towards it.

    That is the ramped policy's own documented trade, not the provenance
    weighting failing: the fusion weight multiplied the geometric weight in
    both runs, and the geometric weights differ. It is recorded rather than
    smoothed over, because a caller that wants a reconstruction never to move a
    contradicted measurement wants the uniform policy.
    """
    frames = _outvoting_frames()
    uniform = crossing_depth(
        weighted_field(frames, config=weighted_config(weight_policy=WeightPolicy.UNIFORM))
    )
    ramped = crossing_depth(
        weighted_field(frames, config=weighted_config(weight_policy=WeightPolicy.RAMPED_BEHIND))
    )
    assert uniform < ramped


# --- the published accounting -------------------------------------------------------------


def test_the_weighted_evidence_publishes_exact_input_counts_and_contribution():
    measured = measured_input(NEAR_CODE)
    reconstruction = reconstructed_input(FAR_CODE)
    volume = weighted_field([measured, reconstruction])
    evidence = volume.weighted_evidence()

    assert evidence.frames_offered == 2
    assert evidence.frames_integrated == 2
    assert evidence.inputs.measured == measured.summary.measured
    assert (
        evidence.inputs.temporally_reconstructed == reconstruction.summary.temporally_reconstructed
    )
    assert evidence.inputs.samples == 2 * tsdf_input.FRAME_SAMPLES
    assert evidence.inputs.offered_weight == math.fsum(
        (measured.summary.total_weight, reconstruction.summary.total_weight)
    )
    # Every accepted voxel observation is worth at most one, and this run has
    # some that are worth strictly less.
    assert 0.0 < evidence.contribution < float(evidence.voxel_updates)
    assert evidence.refusals == 0
    record = evidence.as_dict()
    assert record["dimensional_accuracy_claimed"] is False
    assert record["physical_improvement_claimed"] is False


def test_the_base_accounting_still_partitions_every_voxel_observation():
    volume = weighted_field([measured_input(NEAR_CODE), reconstructed_input(FAR_CODE)])
    evidence = volume.evidence()
    assert evidence.voxel_observations == evidence.frames_offered * volume.config.cell_count
    assert volume.weighted_evidence().voxel_updates == evidence.voxel_updates


def test_the_contribution_splits_by_provenance_and_the_parts_add_back_to_it():
    """The total, and where every part of it came from, counted in one pass.

    The split is what makes the zero this contract claims a MEASUREMENT. A
    measured frame and an accepted temporal reconstruction both reach voxels,
    so both parts are strictly positive; the accepted single-frame estimate
    beside them reaches none, and its entry is zero because nothing was ever
    accumulated against it rather than because a line of code says zero.
    """
    volume = weighted_field(
        [measured_input(NEAR_CODE), reconstructed_input(FAR_CODE), excluded_input(NEAR_CODE)]
    )
    evidence = volume.weighted_evidence()
    parts = evidence.contribution_by_provenance

    assert set(parts) == set(tsdf_input.PROVENANCE_NAMES)
    assert parts["measured"] > 0.0
    assert parts["temporally_reconstructed"] > 0.0
    assert parts["spatially_excluded"] == 0.0
    assert parts["missing"] == 0.0
    assert math.fsum(parts.values()) == evidence.contribution
    accounting = evidence.as_dict()["accounting"]
    assert isinstance(accounting, dict)
    assert accounting["contribution_by_provenance"] == dict(parts)


@pytest.mark.parametrize("policy", list(WeightPolicy))
def test_the_two_backends_split_the_contribution_identically(policy):
    """Parity of the split, not merely of the total it adds up to."""
    frames = [measured_input(NEAR_CODE), reconstructed_input(FAR_CODE)]
    config = weighted_config(weight_policy=policy)
    reference = weighted_field(frames, config=config, backend="python")
    accelerated = weighted_field(frames, config=config, backend="numpy")

    assert dict(accelerated.weighted_evidence().contribution_by_provenance) == dict(
        reference.weighted_evidence().contribution_by_provenance
    )


def test_the_split_is_a_snapshot_that_later_integration_does_not_reach_back_into():
    volume = weighted_field([measured_input(NEAR_CODE)])
    parts = volume.weighted_evidence().contribution_by_provenance
    before = dict(parts)
    volume.integrate_weighted(
        [measured_input(NEAR_CODE)],
        [Rigid.identity()],
        intrinsics=INTRINSICS,
        depth_scale=WEIGHTED_SCALE,
    )
    assert dict(parts) == before
    assert volume.weighted_evidence().contribution_by_provenance["measured"] > before["measured"]


def test_a_run_with_no_reconstruction_says_so_rather_than_claiming_a_gain():
    volume = weighted_field([measured_input(NEAR_CODE), excluded_input(NEAR_CODE)])
    limitations = volume.weighted_evidence().limitations
    assert any("The weighting changed nothing" in note for note in limitations)


# --- the accumulated-weight cap ------------------------------------------------------------


def test_weighted_observations_accumulate_and_stop_at_the_configured_cap():
    cap = 2.0
    config = weighted_config(max_weight=cap)
    frames = [measured_input(NEAR_CODE)] * 6
    volume = weighted_field(frames, config=config)

    summary = volume.evidence().weights
    assert summary.maximum == cap
    assert summary.at_cap > 0
    assert any(f"reached the {cap} weight cap" in note for note in volume.evidence().limitations)


def test_a_reconstruction_reaches_the_cap_more_slowly_than_a_measurement_does():
    """Capping is where a lower weight has to keep meaning a lower weight.

    Six measured frames and six reconstructed ones offer the same number of
    voxel observations. If the cap were reached by counting observations rather
    than by accumulating weight, the two would be indistinguishable here.
    """
    config = weighted_config(max_weight=4.0)
    measured = weighted_field([measured_input(NEAR_CODE)] * 6, config=config)
    reconstructed = weighted_field([reconstructed_input(NEAR_CODE)] * 6, config=config)

    assert measured.evidence().weights.at_cap > 0
    assert reconstructed.evidence().weights.at_cap == 0
    heaviest = reconstructed.evidence().weights.maximum
    assert heaviest is not None
    assert heaviest < 4.0


# --- refusals leave the field exactly as they found it ---------------------------------------


def _unchanged(volume: TsdfVolume, action) -> None:
    """Run ``action`` expecting a refusal, and prove nothing at all moved."""
    values, weights = bytes(volume.values), bytes(volume.weights)
    evidence = volume.evidence().as_dict()
    refusals = volume.weighted_refusals
    with pytest.raises(ScannerError):
        action()
    assert bytes(volume.values) == values
    assert bytes(volume.weights) == weights
    assert volume.evidence().as_dict() == evidence
    assert volume.weighted_refusals == refusals + 1


def test_raw_bytes_offered_to_weighted_integration_are_refused_and_change_nothing():
    volume = weighted_field([measured_input(NEAR_CODE)])
    _unchanged(
        volume,
        lambda: volume.integrate_weighted(
            [patch_frame(NEAR_CODE)],  # type: ignore[list-item]
            [Rigid.identity()],
            intrinsics=INTRINSICS,
            depth_scale=WEIGHTED_SCALE,
        ),
    )


def test_a_weighted_call_whose_poses_do_not_match_its_frames_changes_nothing():
    volume = weighted_field([measured_input(NEAR_CODE)])
    _unchanged(
        volume,
        lambda: volume.integrate_weighted(
            [measured_input(NEAR_CODE)],
            [Rigid.identity(), Rigid.identity()],
            intrinsics=INTRINSICS,
            depth_scale=WEIGHTED_SCALE,
        ),
    )


def test_a_weighted_call_past_the_per_call_frame_ceiling_changes_nothing():
    volume = weighted_field([measured_input(NEAR_CODE)])
    frame = measured_input(NEAR_CODE)
    over = [frame] * (MAX_FRAMES_PER_CALL + 1)
    _unchanged(
        volume,
        lambda: volume.integrate_weighted(
            over,
            [Rigid.identity()] * len(over),
            intrinsics=INTRINSICS,
            depth_scale=WEIGHTED_SCALE,
        ),
    )


def test_a_weighted_call_with_an_unusable_depth_scale_changes_nothing():
    volume = weighted_field([measured_input(NEAR_CODE)])
    _unchanged(
        volume,
        lambda: volume.integrate_weighted(
            [measured_input(NEAR_CODE)],
            [Rigid.identity()],
            intrinsics=INTRINSICS,
            depth_scale=0.0,
        ),
    )


def test_every_refusal_is_counted_and_published():
    volume = TsdfVolume(weighted_config(), backend="python")
    for _ in range(3):
        with pytest.raises(ScannerError):
            volume.integrate_weighted(
                [measured_input(NEAR_CODE)],
                [],
                intrinsics=INTRINSICS,
                depth_scale=WEIGHTED_SCALE,
            )
    assert volume.weighted_evidence().refusals == 3
    assert volume.weighted_evidence().frames_offered == 0
