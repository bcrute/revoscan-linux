"""The gates that decide whether a reconstruction may be published at all.

What this module is about
-------------------------
:mod:`scanner3d.reconstruction_readiness` exists because a reconstruction can
be perfectly self-consistent and worthless. Forty frames taken from one place
register beautifully against each other; a lost capture's later frames are
related to its earlier ones by nothing at all and still look like frames on
disk; a TSDF surface three truncations away from the points it was fused from
is topologically flawless and opens in any viewer.

So the subject here is REFUSAL, and the two halves of getting a refusal right:
that a bad reconstruction is refused, and - at least as load-bearing - that a
legitimate one is not. Half the tests below are negative controls for exactly
that reason. An orbit that comes back to its starting view, a scan the operator
paused in the middle of, a validated recovery from a tracking loss, and an open
one-sided surface with real boundaries are all ordinary, correct things, and a
gate that refused any of them would be worse than no gate.

What is real here and what is not
----------------------------------
The geometry is synthetic and exact: a Fibonacci-sampled sphere, cameras placed
on named orbits, and surfaces derived from that sphere by a stated defect -
displaced by a multiple of the fusion truncation, floated away, or thickened
into a lumpy shell. Nothing is fitted, so every fixture's verdict is a property
of the gate rather than of a registration that happened to converge.

The tracking fixtures are the opposite: they are REAL capture folders, written
by :mod:`scanner3d.capture_sequence`'s own publishers, because the one thing
this module must not do to that gate is reimplement it. Everything is offline;
nothing opens a device node, and no capture outside ``tmp_path`` is touched.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import pytest

from scanner3d.capture_sequence import (
    FRAME_TEMPLATE,
    TRACKING_LOSS_STREAK,
    AlignmentFailure,
    publish_recovery_segment,
    publish_tracking_loss,
    read_tracking_loss,
)
from scanner3d.coverage import coverage_map
from scanner3d.depth_metric import PIXELS_PER_FRAME
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.pipeline import NOISE_FLOOR, TRAVEL_MIN_MULTIPLE
from scanner3d.quality import cloud_agreement
from scanner3d.reconstruction_readiness import (
    ADVICE,
    COVERAGE_NOT_MEASURABLE,
    COVERAGE_UNSUPPORTED,
    DEFAULT_POLICY,
    KIND,
    LIMITATIONS,
    POSE_EXCURSION_INSUFFICIENT,
    POSE_MOTION_BELOW_NOISE,
    POSE_MOTION_NOT_MEASURABLE,
    POSE_VIEWPOINTS_INSUFFICIENT,
    REASONS,
    SCHEMA_VERSION,
    SURFACE_ABANDONS_EVIDENCE,
    SURFACE_NOT_MEASURABLE,
    SURFACE_UNSUPPORTED,
    TRACKING_LOSS_UNREADABLE,
    TRACKING_LOST,
    FusionEvidence,
    ReadinessFailure,
    ReadinessNotMeasurable,
    ReadinessPolicy,
    ReadinessRecordUnusable,
    ReconstructionReadiness,
    ReconstructionRefused,
    TrackingEvidence,
    _directed_support,
    _strided,
    coverage_evidence,
    evaluate_readiness,
    fusion_evidence_sample,
    pose_evidence,
    surface_evidence,
    tracking_evidence,
)
from scanner3d.simulate import intrinsics_of
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_WIDTH

# -- the object, and the ways a camera can look at it -----------------------

#: Radius of the synthetic object, and how far the camera orbits from it. Both
#: are in the capture's own depth units, the units
#: :data:`scanner3d.pipeline.NOISE_FLOOR` is expressed in, so a threshold stated
#: as a multiple of the noise floor means the same thing here as in a capture.
RADIUS = 40.0
DISTANCE = 150.0

#: Points per synthetic frame. Deliberately well above the coverage cell's
#: sampling floor, so the coverage fixtures below are judged on their view
#: directions rather than on being thin.
FRAME_POINTS = 6000


def sphere(count: int = 20000, radius: float = RADIUS) -> list[Point]:
    """A deterministic near-uniform sampling of a sphere, in world coordinates."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points: list[Point] = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * golden
        points.append(
            (radius * ring * math.cos(angle), radius * y, radius * ring * math.sin(angle))
        )
    return points


WORLD = sphere()


def camera(theta: float) -> Rigid:
    """The camera-to-world pose of a camera on the orbit at angle ``theta``."""
    rotation = rotation_about_axis((0.0, 1.0, 0.0), theta)
    return Rigid.of(rotation.rotation, rotation.apply((0.0, 0.0, -DISTANCE)))


def seen_from(pose: Rigid) -> list[Point]:
    """The front-facing half of the sphere, in that camera's own coordinates.

    Front-facing rather than the whole shell, because a camera cannot see
    through an object and a fixture that pretended otherwise would give every
    cell of the model the same observation count whatever the trajectory did -
    which is precisely the distinction the coverage gate is measuring.
    """
    inverse = pose.inverse()
    eye = pose.translation
    visible = [
        point
        for point in WORLD
        if sum(point[axis] * (eye[axis] - point[axis]) for axis in range(3)) > 0.0
    ]
    step = max(1, len(visible) // FRAME_POINTS)
    return [inverse.apply(point) for point in visible[::step]]


def sweep(angles: Sequence[float]) -> tuple[list[list[Point]], list[Rigid]]:
    """One frame per angle: what that camera saw, and where it was."""
    poses = [camera(angle) for angle in angles]
    return [seen_from(pose) for pose in poses], poses


def orbit(count: int, span: float = 2.0 * math.pi) -> tuple[list[list[Point]], list[Rigid]]:
    return sweep([index * span / count for index in range(count)])


# -- pose diversity ---------------------------------------------------------


def test_an_orbit_of_a_real_object_passes_every_pose_gate() -> None:
    """The negative control that everything else here is measured against.

    Twenty-four views around one object is what a scan looks like. If this
    failed, every refusal below would be uninterpretable: a gate that refuses
    everything is not evidence about anything.
    """
    clouds, poses = orbit(24)

    evidence, failures = pose_evidence(clouds, poses)

    assert failures == ()
    assert evidence.poses == 24
    assert evidence.adjacent_steps == 23
    assert evidence.moving_steps == 23
    assert evidence.stationary_steps == 0
    assert evidence.distinct_viewpoints >= DEFAULT_POLICY.min_distinct_viewpoints
    assert evidence.excursion >= DEFAULT_POLICY.min_excursion


def test_a_sequence_of_identical_poses_fails_every_pose_gate() -> None:
    """The manufactured sequence, and it must fail all three instruments.

    Not one: an all-identical trajectory has no viewpoints, no excursion and no
    motion, and a result that named only one of those would be describing the
    first check that happened to run rather than the capture.

    Object identity is deliberately not what catches it. The poses below are
    forty separately constructed :class:`~scanner3d.geometry.Rigid` values that
    happen to be equal, so nothing here can be passing by comparing ``is``.
    """
    clouds, poses = sweep([0.0] * 40)
    assert len({id(pose) for pose in poses}) == 40

    evidence, failures = pose_evidence(clouds, poses)

    assert [failure.reason for failure in failures] == [
        POSE_VIEWPOINTS_INSUFFICIENT,
        POSE_EXCURSION_INSUFFICIENT,
        POSE_MOTION_BELOW_NOISE,
    ]
    assert evidence.distinct_viewpoints == 1
    assert evidence.moving_steps == 0
    assert evidence.excursion == 0.0


def test_a_trajectory_that_only_jitters_below_the_noise_floor_is_refused() -> None:
    """Near-stationary, and refused for being near-stationary rather than equal.

    Every pose here is different - no two are the same object and no two are
    equal float for float - and the whole sweep moves the surface by less than
    the scanner's own per-point measurement noise. A gate that only caught
    exact repetition would pass this, and the model it produces is one view.
    """
    clouds, poses = sweep([index * 1e-4 for index in range(20)])
    assert len({pose.translation for pose in poses}) == 20

    evidence, failures = pose_evidence(clouds, poses)

    assert POSE_MOTION_BELOW_NOISE in {failure.reason for failure in failures}
    assert evidence.travel_worst < NOISE_FLOOR
    assert evidence.stationary_steps == evidence.adjacent_steps


def test_a_trajectory_stuck_between_two_viewpoints_is_refused() -> None:
    """Plenty of motion, and still only two places to have seen it from.

    This is the one the adjacent-travel instrument alone cannot catch: every
    step moves the surface a long way, so travel is healthy and the excursion
    is real, and the sweep still knows nothing about the object it did not see.
    The viewpoint count is the instrument that answers it.
    """
    clouds, poses = sweep([0.0 if index % 2 == 0 else 0.9 for index in range(20)])

    evidence, failures = pose_evidence(clouds, poses)

    assert [failure.reason for failure in failures] == [POSE_VIEWPOINTS_INSUFFICIENT]
    assert evidence.distinct_viewpoints == 2
    assert evidence.moving_steps == evidence.adjacent_steps


def test_an_orbit_that_returns_to_its_starting_view_still_passes() -> None:
    """The revisit a closure score is BUILT on must not read as "never moved".

    Coming back to where the scan started is not a defect; it is how a
    trajectory gets a constraint that was never fitted. A gate that measured
    the whole sweep by the distance between the first and last pose would
    refuse the best scans in this repository, so the excursion is the extent of
    the trajectory rather than its endpoints, and this is the test that says so.
    """
    clouds, poses = orbit(24)
    clouds.append(clouds[0])
    poses.append(camera(0.0))

    evidence, failures = pose_evidence(clouds, poses)

    assert failures == ()
    assert math.dist(poses[0].translation, poses[-1].translation) < NOISE_FLOOR
    assert evidence.excursion > DEFAULT_POLICY.min_excursion


def test_an_ordinary_pause_in_the_middle_of_a_scan_still_passes() -> None:
    """Stopping to reposition is a thing operators do and must keep doing.

    Two thirds of the adjacent steps here moved nothing at all. The moving-step
    requirement is a small ABSOLUTE count rather than a fraction precisely so
    that a scan is judged on the motion it has and not on the proportion of its
    frames that were taken while standing still.
    """
    angles = [index * math.tau / 12 for index in range(12) for _ in range(3)]
    clouds, poses = sweep(angles)

    evidence, failures = pose_evidence(clouds, poses)

    assert failures == ()
    assert evidence.stationary_steps > evidence.moving_steps
    assert evidence.moving_steps >= DEFAULT_POLICY.min_moving_steps


def test_the_viewpoint_gate_passes_at_its_threshold_and_fails_one_below() -> None:
    """The boundary itself, from both sides, at the exact policy value.

    A gate is only as trustworthy as its edge: "at least three" has to mean
    three passes and two fails, not "about three".
    """
    separation = DEFAULT_POLICY.viewpoint_separation
    assert separation == NOISE_FLOOR * TRAVEL_MIN_MULTIPLE
    wide = [0.0, 0.6, 1.2]
    clouds, poses = sweep(wide)
    evidence, failures = pose_evidence(clouds, poses)
    assert evidence.distinct_viewpoints == DEFAULT_POLICY.min_distinct_viewpoints
    assert POSE_VIEWPOINTS_INSUFFICIENT not in {failure.reason for failure in failures}

    clouds, poses = sweep(wide[:2])
    evidence, failures = pose_evidence(clouds, poses)
    assert evidence.distinct_viewpoints == DEFAULT_POLICY.min_distinct_viewpoints - 1
    assert POSE_VIEWPOINTS_INSUFFICIENT in {failure.reason for failure in failures}


def test_the_excursion_gate_passes_exactly_at_its_required_value() -> None:
    """A measured excursion equal to the requirement PASSES, and one hair below fails.

    Stated once here rather than described in prose everywhere: every gate in
    this module compares with "at least", so the threshold is inside the
    passing set. The policy is moved rather than the geometry, because moving
    the geometry to land on a threshold to nine decimal places is a fixture
    that tests arithmetic rather than the boundary.
    """
    clouds, poses = orbit(24)
    evidence, _ = pose_evidence(clouds, poses)
    exact = replace(
        DEFAULT_POLICY, min_excursion_multiple=evidence.excursion / DEFAULT_POLICY.noise_floor
    )

    _, at_threshold = pose_evidence(clouds, poses, policy=exact)
    assert POSE_EXCURSION_INSUFFICIENT not in {failure.reason for failure in at_threshold}

    just_above = replace(exact, min_excursion_multiple=exact.min_excursion_multiple * 1.000001)
    _, past_threshold = pose_evidence(clouds, poses, policy=just_above)
    assert POSE_EXCURSION_INSUFFICIENT in {failure.reason for failure in past_threshold}


def test_a_uniformly_scaled_world_is_judged_by_the_same_calibrated_noise_floor() -> None:
    """Scale is not free here, and that is the honest answer rather than a gap.

    The noise floor is a property of the SCANNER, in the units its depth
    samples are expressed in. So a sweep whose motion is scaled up stays a
    scan, and one scaled down until every step is under that floor stops being
    one - which is exactly right: the second scanner really did not move enough
    to see round anything.
    """
    clouds, poses = orbit(8, span=math.pi / 2)
    _, ordinary = pose_evidence(clouds, poses)
    assert ordinary == ()

    def stretch(point: Point, factor: float) -> Point:
        return (point[0] * factor, point[1] * factor, point[2] * factor)

    def scaled(factor: float) -> tuple[list[list[Point]], list[Rigid]]:
        return (
            [[stretch(point, factor) for point in cloud] for cloud in clouds],
            [Rigid.of(pose.rotation, stretch(pose.translation, factor)) for pose in poses],
        )

    bigger_clouds, bigger_poses = scaled(4.0)
    _, bigger = pose_evidence(bigger_clouds, bigger_poses)
    assert bigger == ()

    smaller_clouds, smaller_poses = scaled(0.001)
    _, smaller = pose_evidence(smaller_clouds, smaller_poses)
    assert POSE_MOTION_BELOW_NOISE in {failure.reason for failure in smaller}


def test_a_sweep_with_no_points_to_measure_against_says_so_rather_than_passing() -> None:
    """No probe means no claim - in either direction.

    "Nothing could be measured" is a refusal here and not a pass, because the
    alternative is a reconstruction published on the strength of a gate that
    never ran.
    """
    poses = [camera(index * 0.5) for index in range(4)]

    evidence, failures = pose_evidence([[], [], [], []], poses)

    assert [failure.reason for failure in failures] == [POSE_MOTION_NOT_MEASURABLE]
    assert evidence.probe_points == 0


def test_poses_that_do_not_describe_the_frames_are_a_caller_error_not_a_refusal() -> None:
    """A mismatch is not a bad scan; it is a bug, and it raises rather than fails.

    Returning "this capture is not ready" for it would let a caller that lost a
    pose somewhere read the answer as evidence about the object.
    """
    clouds, poses = orbit(4)

    with pytest.raises(ReadinessNotMeasurable):
        pose_evidence(clouds, poses[:3])


# -- coverage ---------------------------------------------------------------


def test_a_high_point_count_from_one_angle_fails_coverage() -> None:
    """The failure this gate exists for, and the number it deliberately ignores.

    Forty frames of the same view is a quarter of a million points and knows
    one silhouette. The point count does not enter the decision at all: what
    does is how many distinct frames saw each occupied cell and how far apart
    they were looking.
    """
    clouds, poses = sweep([0.0] * 40)

    evidence, failures = coverage_evidence(clouds, poses)

    assert [failure.reason for failure in failures] == [COVERAGE_UNSUPPORTED]
    assert evidence.points_used > 40_000
    assert evidence.supported_cells == 0
    assert evidence.supported_fraction == 0.0


def test_a_comparable_surface_seen_from_many_angles_passes_coverage() -> None:
    """The control for the above: the same object, the same instrument, more angles.

    Without this, the failure above would be evidence that the gate refuses
    spheres.
    """
    clouds, poses = orbit(24)

    evidence, failures = coverage_evidence(clouds, poses)

    assert failures == ()
    assert evidence.supported_fraction >= DEFAULT_POLICY.coverage_min_supported_fraction
    assert evidence.total_cells > 0
    assert evidence.supported_cells <= evidence.total_cells


def test_unseen_empty_volume_contributes_no_cell_and_no_shortfall() -> None:
    """Unseen volume is not an unsupported surface.

    The fixture makes the point by itself: a hollow sphere is mostly empty, and
    nothing ever observed its interior. A gate that gridded the bounding box
    would find that interior missing, call it unsupported surface, and refuse a
    scan that is entirely correct.

    So the assertion is that the occupied cells are a small minority of the
    cells the bounding box holds - proving the grid is the geometry rather than
    the box - and that the verdict is nonetheless a pass.
    """
    clouds, poses = orbit(24)

    evidence, failures = coverage_evidence(clouds, poses)

    assert failures == ()
    # Every cell in the denominator is a cell some frame put a point in. That is
    # the exact statement of the claim, and it does not depend on how coarse the
    # grid happened to be for this object.
    report = coverage_map(
        [_strided(cloud, DEFAULT_POLICY.coverage_points_per_frame) for cloud in clouds],
        poses,
        cell=evidence.cell_size,
        min_observations=DEFAULT_POLICY.coverage_min_observations,
        min_spread=DEFAULT_POLICY.coverage_min_spread,
    )
    assert report.total_cells == evidence.total_cells
    assert min(cell.observations for cell in report.cells) >= 1
    # And the box the object sits in holds strictly more cells than the object
    # does: the hollow interior and the corners are not in the denominator.
    placed = [
        pose.apply(point) for cloud, pose in zip(clouds, poses, strict=True) for point in cloud
    ]
    low = tuple(min(point[axis] for point in placed) for axis in range(3))
    high = tuple(max(point[axis] for point in placed) for axis in range(3))
    box_cells = 1
    for axis in range(3):
        box_cells *= max(1, math.ceil((high[axis] - low[axis]) / evidence.cell_size))
    assert evidence.total_cells < box_cells
    assert evidence.supported_cells <= evidence.total_cells


def test_the_coverage_policy_and_the_support_it_measured_are_both_serialized() -> None:
    """A verdict nobody can re-derive is an opinion.

    So the record carries the cell size that was used, the observation and
    spread policy that was applied, and the counts the fraction came from.
    """
    clouds, poses = orbit(24)

    evidence, _ = coverage_evidence(clouds, poses)
    record = evidence.as_dict()

    assert record["measured"] is True
    assert record["total_cells"] == evidence.total_cells
    assert record["supported_cells"] == evidence.supported_cells
    assert record["cell_size"] == pytest.approx(evidence.cell_size)
    policy = DEFAULT_POLICY.as_dict()
    assert policy["coverage_min_observations"] == DEFAULT_POLICY.coverage_min_observations
    assert policy["coverage_min_spread_radians"] == pytest.approx(
        DEFAULT_POLICY.coverage_min_spread
    )
    assert policy["coverage_min_supported_fraction"] == pytest.approx(
        DEFAULT_POLICY.coverage_min_supported_fraction
    )


def test_a_capture_coverage_cannot_be_measured_from_is_refused_not_passed() -> None:
    """The unmeasurable case is a refusal, with the measurement's own words in it."""
    poses = [camera(index * 0.5) for index in range(3)]

    evidence, failures = coverage_evidence([[], [], []], poses)

    assert [failure.reason for failure in failures] == [COVERAGE_NOT_MEASURABLE]
    assert evidence.measured is False
    assert evidence.detail


# -- surface agreement ------------------------------------------------------

#: A fusion resolution to judge surfaces at. Real exports derive these from the
#: placed extent and the voxel budget; stating them here keeps every surface
#: fixture below judged at one known tolerance rather than at whatever a
#: budget happened to produce.
VOXEL = 0.34
TRUNCATION = 1.02


def evidence_of(points: Sequence[Point]) -> FusionEvidence:
    """A ``FusionEvidence`` standing for a cloud whose frames are not the subject.

    :func:`scanner3d.reconstruction_readiness.surface_evidence` takes the
    sample and the bound it was drawn under together, so that a caller cannot
    hand it the decimated registration cloud by accident. The tests below are
    about the AGREEMENT arithmetic and build the cloud directly, so the frame
    counts here are the SMALLEST ones the real sampler could have produced this
    many points under this policy - a stand-in has to be a shape a real
    evaluation could have emitted, because these records are round-tripped
    through a strict reader that refuses ones that are not.
    :func:`test_the_fusion_sample_deprojects_and_places_with_the_poses_it_was_given`
    and :mod:`tests.test_reconstruction_export_readiness` are where the real
    sampler is exercised.
    """
    frames = max(1, -(-len(points) // DEFAULT_POLICY.fusion_pixels_per_frame))
    return FusionEvidence(
        frames_offered=frames,
        frames_sampled=frames,
        samples_examined=len(points),
        points=tuple(points),
    )


def displaced(points: Sequence[Point], distance: float) -> list[Point]:
    return [(point[0] + distance, point[1], point[2]) for point in points]


def lumpy(points: Sequence[Point], amplitude: float) -> list[Point]:
    """The same sphere with its radius wobbling by ``amplitude``, deterministically."""
    out: list[Point] = []
    for index, point in enumerate(points):
        wobble = 1.0 + amplitude * math.sin(index * 2.399963) / RADIUS
        out.append((point[0] * wobble, point[1] * wobble, point[2] * wobble))
    return out


def test_a_surface_that_sits_on_its_own_evidence_passes_both_directions() -> None:
    """The control, and the shape of an honest agreement.

    Both directions are always computed and both are always reported, because
    either alone passes a reconstruction the other fails.
    """
    evidence_points = sphere(20000)
    surface = sphere(60000)

    agreement, failures = surface_evidence(
        surface, evidence_of(evidence_points), voxel_size=VOXEL, truncation=TRUNCATION
    )

    assert failures == ()
    assert agreement.measured is True
    assert agreement.surface_supported_fraction == 1.0
    assert agreement.evidence_explained_fraction == 1.0
    assert agreement.tolerance >= TRUNCATION


def test_an_open_one_sided_surface_with_real_boundaries_passes() -> None:
    """Watertightness is NOT an implicit requirement anywhere in this gate.

    A scan of one side of an object produces an open surface, and the evidence
    it is compared against is the one-sided set of points its own frames
    produced - not an imagined whole object. Nothing here counts a boundary
    edge or asks whether the mesh closes.
    """
    half_evidence = [point for point in sphere(20000) if point[2] > 0.0]
    half_surface = [point for point in sphere(60000) if point[2] > 0.0]

    agreement, failures = surface_evidence(
        half_surface, evidence_of(half_evidence), voxel_size=VOXEL, truncation=TRUNCATION
    )

    assert failures == ()
    assert agreement.surface_supported_fraction == 1.0
    assert agreement.evidence_explained_fraction == 1.0


def test_a_deliberately_displaced_surface_is_refused() -> None:
    """Three truncations off the evidence, and topologically perfect.

    This is the defect that a vertex count, an edge census and a watertightness
    flag are all completely blind to.
    """
    evidence_points = sphere(20000)

    agreement, failures = surface_evidence(
        displaced(sphere(60000), 3.0 * TRUNCATION),
        evidence_of(evidence_points),
        voxel_size=VOXEL,
        truncation=TRUNCATION,
    )

    assert SURFACE_UNSUPPORTED in {failure.reason for failure in failures}
    assert agreement.surface_supported_fraction < DEFAULT_POLICY.surface_min_supported_fraction


def test_a_floating_surface_is_refused_in_both_directions() -> None:
    """Far enough away and both instruments say so, which is the point of having two."""
    evidence_points = sphere(20000)

    agreement, failures = surface_evidence(
        displaced(sphere(60000), 50.0),
        evidence_of(evidence_points),
        voxel_size=VOXEL,
        truncation=TRUNCATION,
    )

    assert {failure.reason for failure in failures} == {
        SURFACE_UNSUPPORTED,
        SURFACE_ABANDONS_EVIDENCE,
    }
    assert agreement.surface_supported_fraction < 0.2
    assert agreement.evidence_explained_fraction < 0.2


def test_a_grossly_lumpy_surface_is_refused_and_a_gently_noisy_one_is_not() -> None:
    """The pair, because either half alone proves nothing.

    A gate that refused every surface would "catch" the lumpy one. What makes
    the refusal mean something is that the same instrument, at the same
    tolerance, passes a surface wobbling by less than the scanner's own noise.
    """
    evidence_points = sphere(20000)

    _, gross = surface_evidence(
        lumpy(sphere(60000), 4.0 * TRUNCATION),
        evidence_of(evidence_points),
        voxel_size=VOXEL,
        truncation=TRUNCATION,
    )
    assert SURFACE_UNSUPPORTED in {failure.reason for failure in gross}

    _, gentle = surface_evidence(
        lumpy(sphere(60000), NOISE_FLOOR),
        evidence_of(evidence_points),
        voxel_size=VOXEL,
        truncation=TRUNCATION,
    )
    assert gentle == ()


def test_the_tolerance_scales_with_the_declared_truncation() -> None:
    """No fixed world-coordinate number decides this, and here is the proof.

    The same displaced surface is refused at one truncation and accepted at
    four times it - because a field truncated four times as far genuinely
    cannot resolve that displacement, and saying otherwise would be inventing
    precision the fusion does not have.
    """
    evidence_points = sphere(20000)
    surface = displaced(sphere(60000), 3.0 * TRUNCATION)

    fine, refused = surface_evidence(
        surface, evidence_of(evidence_points), voxel_size=VOXEL, truncation=TRUNCATION
    )
    coarse, accepted = surface_evidence(
        surface, evidence_of(evidence_points), voxel_size=4.0 * VOXEL, truncation=4.0 * TRUNCATION
    )

    assert SURFACE_UNSUPPORTED in {failure.reason for failure in refused}
    assert accepted == ()
    assert coarse.tolerance > fine.tolerance
    assert coarse.truncation == 4.0 * TRUNCATION


def test_the_neighbourhood_search_agrees_with_the_repositorys_own_all_pairs_measure() -> None:
    """This module's search is a cheaper route to the same answer, not a second one.

    :func:`scanner3d.quality.cloud_agreement` is the repository's measurement of
    exactly this quantity and it searches all pairs, which is unaffordable at
    the resolution this gate needs. So the voxel-hash search is checked against
    it here, on clouds small enough for the all-pairs version to be affordable,
    over both a coinciding case and a displaced one - and it has to agree
    exactly, because the hash is claimed to be exact inside its tolerance
    rather than approximately right.
    """
    tolerance = 2.0
    queries = sphere(400)
    for offset in (0.0, 1.0, 1.9, 2.5, 10.0):
        targets = displaced(sphere(500), offset)
        within, _ = _directed_support(queries, targets, tolerance=tolerance)

        reference = cloud_agreement(queries, targets, name="control", coverage_tolerance=tolerance)
        expected = round(reference.observed_to_reference.coverage_fraction * len(queries))

        assert within == expected, f"disagreed at offset {offset}"


# -- the capture's own tracking authority -----------------------------------


def write_frames(directory: Path, indices: Sequence[int]) -> None:
    for index in indices:
        (directory / FRAME_TEMPLATE.format(index=index)).write_bytes(
            bytes([index % 251]) * (64 + index)
        )


def failures_from(first: int) -> list[AlignmentFailure]:
    return [
        AlignmentFailure(frame_index=first + offset, failure_type="RegistrationError", message="x")
        for offset in range(TRACKING_LOSS_STREAK)
    ]


def test_a_healthy_capture_costs_one_stat_and_does_not_read_the_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap path, asserted rather than assumed.

    Every export pays for this gate, so a capture with no loss record must cost
    one ``stat`` that finds nothing. Proved by booby-trapping the authority: a
    healthy folder that reached it would fail this test.
    """
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(6))

    import scanner3d.capture_sequence as authority

    def refuse(_directory: Path) -> object:
        raise AssertionError("a capture with no loss record must not be read as a sequence")

    monkeypatch.setattr(authority, "read_capture_sequence", refuse)

    evidence, failures = tracking_evidence(capture)

    assert failures == ()
    assert evidence.records_found == 0
    assert evidence.unresolved is False


def test_an_unresolved_tracking_loss_is_refused_in_the_authoritys_own_words(
    tmp_path: Path,
) -> None:
    """The 2026-08-28 capture's shape, and the refusal is quoted rather than reworded.

    Nothing about what closes a loss is decided here. The record is published by
    the module that owns it, and the refusal that comes back is that module's
    sentence, so there is exactly one interpretation of a lost capture in this
    repository.
    """
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(10))
    publish_tracking_loss(capture, last_aligned_frame_index=5, failures=failures_from(6))

    evidence, failures = tracking_evidence(capture)

    assert [failure.reason for failure in failures] == [TRACKING_LOST]
    assert evidence.records_found == 1
    assert evidence.unresolved is True
    assert failures[0].advice == ADVICE[TRACKING_LOST]
    assert evidence.detail


def test_a_loss_record_that_cannot_be_read_is_refused_rather_than_ignored(
    tmp_path: Path,
) -> None:
    """ "Cannot read it, so the capture is fine" is the original failure exactly."""
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(6))
    (capture / "tracking-loss.json").write_text("{not json at all")

    evidence, failures = tracking_evidence(capture)

    assert [failure.reason for failure in failures] == [TRACKING_LOSS_UNREADABLE]
    assert evidence.unresolved is True


def test_a_validated_recovery_is_history_and_not_a_refusal(tmp_path: Path) -> None:
    """A capture that lost tracking and genuinely found the object again is USABLE.

    This is the negative control the tracking gate needs most. A gate that
    refused every capture whose folder mentions a loss would throw away exactly
    the captures the recovery mechanism was built to save, and would do it
    while looking correct.
    """
    from tests.test_capture_sequence import accepted_evidence, transform

    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(6))
    record = publish_tracking_loss(capture, last_aligned_frame_index=2, failures=failures_from(3))
    loss = read_tracking_loss(capture)
    assert loss is not None
    publish_recovery_segment(
        capture,
        loss_dir=record.parent,
        loss=loss,
        frames=[(6, b"a" * 70), (7, b"b" * 71), (8, b"c" * 72)],
        transform=transform(0.25),
        evidence=accepted_evidence(3),
    )

    evidence, failures = tracking_evidence(capture)

    assert failures == ()
    assert evidence.records_found == 1
    assert evidence.recovered_episodes == 1
    assert evidence.unresolved is False


def test_a_tampered_recovery_is_still_refused_by_the_authority(tmp_path: Path) -> None:
    """Weakening the binding is not something this module is able to do.

    The recovery above is published correctly and then one excluded frame on
    disk is altered, which breaks the byte binding the capture reader enforces.
    The refusal has to come back through this gate unchanged - if it did not,
    this module would have become a way around the check.
    """
    from tests.test_capture_sequence import accepted_evidence, transform

    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(6))
    record = publish_tracking_loss(capture, last_aligned_frame_index=2, failures=failures_from(3))
    loss = read_tracking_loss(capture)
    assert loss is not None
    publish_recovery_segment(
        capture,
        loss_dir=record.parent,
        loss=loss,
        frames=[(6, b"a" * 70), (7, b"b" * 71), (8, b"c" * 72)],
        transform=transform(0.25),
        evidence=accepted_evidence(3),
    )
    excluded = capture / FRAME_TEMPLATE.format(index=3)
    excluded.write_bytes(excluded.read_bytes() + b"tampered")

    _, failures = tracking_evidence(capture)

    assert [failure.reason for failure in failures] == [TRACKING_LOST]


# -- everything at once -----------------------------------------------------


def test_every_applicable_gate_reports_and_the_result_names_all_of_them(
    tmp_path: Path,
) -> None:
    """One capture, several things wrong, and one result that says all of it.

    Stopping at the first failed check would send an operator round a loop:
    fix the pose problem, run it again, be told about the coverage problem,
    run it again. Every gate that can be answered is answered.
    """
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(10))
    publish_tracking_loss(capture, last_aligned_frame_index=5, failures=failures_from(6))
    clouds, poses = sweep([0.0] * 12)

    readiness = evaluate_readiness(capture, clouds, poses)

    assert readiness.ready is False
    assert readiness.reasons == (
        TRACKING_LOST,
        POSE_VIEWPOINTS_INSUFFICIENT,
        POSE_EXCURSION_INSUFFICIENT,
        POSE_MOTION_BELOW_NOISE,
        COVERAGE_UNSUPPORTED,
    )
    text = readiness.refusal_text()
    for reason in readiness.reasons:
        assert reason in text
    # The dialog shows the LAST line, so the last line has to be advice.
    assert text.splitlines()[-1] == f"What to do: {ADVICE[TRACKING_LOST]}."
    # And a REAL five-failure verdict survives the strict reader, which is the
    # control the cross-checks below need: a reader that derives the required
    # reason set from the evidence must accept the set an evaluation actually
    # produced, in the order it actually produced it.
    assert (
        ReconstructionReadiness.from_dict(
            json.loads(json.dumps(readiness.as_dict(), allow_nan=False))
        )
        == readiness
    )


def test_a_good_capture_passes_every_gate_that_precedes_fusion(tmp_path: Path) -> None:
    """The whole-result control."""
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(4))
    clouds, poses = orbit(16)

    readiness = evaluate_readiness(capture, clouds, poses)

    assert readiness.ready is True
    assert readiness.reasons == ()
    assert readiness.surface is None
    assert "every applicable gate passed" in readiness.summary()


def test_the_surface_result_is_added_to_the_earlier_failures_rather_than_replacing_them(
    tmp_path: Path,
) -> None:
    """A post-fusion failure must not erase what was already known to be wrong."""
    capture = tmp_path / "capture"
    capture.mkdir()
    write_frames(capture, range(4))
    clouds, poses = sweep([0.0] * 12)
    before = evaluate_readiness(capture, clouds, poses)
    agreement, surface_failures = surface_evidence(
        displaced(sphere(60000), 50.0),
        evidence_of(sphere(20000)),
        voxel_size=VOXEL,
        truncation=TRUNCATION,
    )

    after = before.with_surface(agreement, surface_failures)

    assert after.reasons[: len(before.reasons)] == before.reasons
    assert SURFACE_UNSUPPORTED in after.reasons
    assert after.surface is not None
    assert after.ready is False


def test_a_refusal_carries_the_whole_result_and_not_only_a_sentence() -> None:
    """A front end has to be able to render every reason, so the exception carries them."""
    readiness = ReconstructionReadiness(
        policy=DEFAULT_POLICY,
        tracking=blank().tracking,
        poses=blank().poses,
        coverage=blank().coverage,
        surface=None,
        failures=blank_failures(),
    )

    refusal = ReconstructionRefused(readiness)

    assert refusal.readiness is readiness
    assert str(refusal) == readiness.refusal_text()


# -- the record --------------------------------------------------------------


def blank() -> ReconstructionReadiness:
    """A ready result over a real, if small, evaluation of a real orbit."""
    clouds, poses = orbit(8)
    pose_record, _ = pose_evidence(clouds, poses)
    coverage_record, _ = coverage_evidence(clouds, poses)
    return ReconstructionReadiness(
        policy=DEFAULT_POLICY,
        tracking=TrackingEvidence(0, 0, unresolved=False, detail=""),
        poses=pose_record,
        coverage=coverage_record,
        surface=None,
        failures=(),
    )


def blank_failures() -> tuple[ReadinessFailure, ...]:
    return (
        ReadinessFailure(
            reason=POSE_MOTION_BELOW_NOISE,
            detail="nothing moved",
            advice=ADVICE[POSE_MOTION_BELOW_NOISE],
        ),
    )


def full_record() -> dict:
    """A complete, valid record with BOTH halves present, as JSON text would hold it."""
    surface, _ = surface_evidence(
        sphere(4000), evidence_of(sphere(3000)), voxel_size=VOXEL, truncation=TRUNCATION
    )
    readiness = blank().with_surface(surface, ())
    return json.loads(json.dumps(readiness.as_dict(), allow_nan=False))


def test_a_complete_record_round_trips_through_json_text() -> None:
    """Written, encoded with ``allow_nan=False``, decoded, and read back equal.

    Through actual JSON text rather than the dict, because that is the journey
    a record in an export folder makes and it is where a non-finite value or a
    tuple pretending to be a list would be caught.
    """
    readiness = blank().with_surface(
        surface_evidence(
            sphere(4000), evidence_of(sphere(3000)), voxel_size=VOXEL, truncation=TRUNCATION
        )[0],
        (),
    )

    decoded = ReconstructionReadiness.from_dict(
        json.loads(json.dumps(readiness.as_dict(), allow_nan=False))
    )

    assert decoded == readiness
    assert decoded.ready is True


@pytest.mark.parametrize(
    "path",
    [
        ("kind",),
        ("ready",),
        ("failures",),
        ("policy",),
        ("tracking",),
        ("poses",),
        ("coverage",),
        ("limitations",),
        ("policy", "noise_floor"),
        ("poses", "excursion"),
        ("coverage", "total_cells"),
        ("surface", "tolerance"),
    ],
)
def test_a_missing_field_is_refused(path: tuple[str, ...]) -> None:
    """Every field is required. A record missing one is not a smaller record."""
    record = full_record()
    target = record
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    "path", [(), ("policy",), ("tracking",), ("poses",), ("coverage",), ("surface",)]
)
def test_an_unknown_field_is_refused(path: tuple[str, ...]) -> None:
    """The interesting direction: a record from a richer schema is not half-read.

    Ignoring a field this version does not know is how a record that means
    something different gets treated as one that means this.
    """
    record = full_record()
    target = record
    for key in path:
        target = target[key]
    target["something_this_version_never_wrote"] = 1

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("poses", "moving_steps"), True),
        (("coverage", "total_cells"), True),
        (("policy", "noise_floor"), True),
        (("surface", "surface_points"), False),
        (("poses", "excursion"), True),
    ],
)
def test_a_boolean_where_a_number_belongs_is_refused(path: tuple[str, str], value: bool) -> None:
    """``True`` is an ``int`` in Python and is not a number in JSON.

    A record saying ``"moving_steps": true`` must be refused, never counted
    as one moving step.
    """
    record = full_record()
    record[path[0]][path[1]] = value

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused(value: float) -> None:
    """JSON has no word for these and neither does this reader."""
    record = full_record()
    record["poses"]["excursion"] = value

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_a_record_that_claims_it_is_ready_beside_failures_is_refused() -> None:
    """The one shape this whole module exists to refuse, in record form."""
    record = full_record()
    record["failures"] = [
        {
            "reason": POSE_MOTION_BELOW_NOISE,
            "detail": "nothing moved",
            "advice": ADVICE[POSE_MOTION_BELOW_NOISE],
        }
    ]

    with pytest.raises(ReadinessRecordUnusable, match="contradicts its own evidence"):
        ReconstructionReadiness.from_dict(record)


def test_a_record_that_claims_it_is_refused_with_nothing_wrong_is_also_refused() -> None:
    """Both directions of the same rule: the decision is derived, never stated."""
    record = full_record()
    record["ready"] = False

    with pytest.raises(ReadinessRecordUnusable, match="contradicts its own evidence"):
        ReconstructionReadiness.from_dict(record)


def test_more_supported_cells_than_occupied_cells_is_refused() -> None:
    """An internally impossible total, refused rather than reported."""
    record = full_record()
    record["coverage"]["supported_cells"] = record["coverage"]["total_cells"] + 1

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_a_supported_fraction_that_does_not_follow_from_its_counts_is_refused() -> None:
    """The fraction is written for a reader's convenience and is therefore checked."""
    record = full_record()
    record["coverage"]["supported_fraction"] = 0.999

    with pytest.raises(ReadinessRecordUnusable, match="contradicts itself"):
        ReconstructionReadiness.from_dict(record)


def test_moving_and_stationary_steps_that_do_not_sum_to_the_adjacent_steps_are_refused() -> None:
    record = full_record()
    record["poses"]["moving_steps"] = record["poses"]["moving_steps"] + 1

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_a_derived_policy_value_that_does_not_follow_from_its_multiple_is_refused() -> None:
    record = full_record()
    record["policy"]["viewpoint_separation"] = 999.0

    with pytest.raises(ReadinessRecordUnusable, match="contradicts itself"):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize("version", [0, 1, SCHEMA_VERSION + 1, True, "1", None])
def test_any_schema_version_other_than_this_one_is_refused(version: object) -> None:
    """A future format is refused, never guessed at."""
    record = full_record()
    record["schema_version"] = version

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_a_record_of_a_different_kind_is_refused() -> None:
    """So a file found loose is identified rather than assumed."""
    record = full_record()
    record["kind"] = "scanner3d.tracking-loss"

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)
    assert KIND == "scanner3d.reconstruction-readiness"


def test_a_reason_this_version_does_not_publish_is_refused() -> None:
    """An unknown reason code cannot be rendered or acted on, so it is not accepted."""
    record = full_record()
    record["ready"] = False
    record["failures"] = [{"reason": "everything_is_fine_actually", "detail": "d", "advice": "a"}]

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_advice_that_is_not_this_versions_advice_for_that_reason_is_refused() -> None:
    """The advice is vocabulary, not free text a record may substitute for."""
    record = full_record()
    record["ready"] = False
    record["failures"] = [
        {"reason": POSE_MOTION_BELOW_NOISE, "detail": "d", "advice": "just publish it anyway"}
    ]

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_the_same_reason_twice_is_refused() -> None:
    record = full_record()
    record["ready"] = False
    failure = {
        "reason": POSE_MOTION_BELOW_NOISE,
        "detail": "d",
        "advice": ADVICE[POSE_MOTION_BELOW_NOISE],
    }
    record["failures"] = [failure, dict(failure)]

    with pytest.raises(ReadinessRecordUnusable, match="more than once"):
        ReconstructionReadiness.from_dict(record)


def test_the_limitations_are_part_of_the_record_and_not_optional() -> None:
    """They are the record's own statement of what it does not establish."""
    record = full_record()
    record["limitations"] = []

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)
    assert any("INTERNAL CONSISTENCY" in text for text in LIMITATIONS)
    assert any("watertight" in text.lower() for text in LIMITATIONS)


def test_every_reason_this_version_publishes_has_advice_and_they_agree() -> None:
    """The vocabulary is one mapping, so a reason cannot exist without advice."""
    assert set(REASONS) == set(ADVICE)
    assert all(ADVICE[reason].strip() for reason in REASONS)


def test_a_policy_value_that_is_not_a_number_is_refused_before_anything_is_measured() -> None:
    """A policy is validated where it is constructed, not where it is used."""
    with pytest.raises(ReadinessNotMeasurable):
        ReadinessPolicy(noise_floor=float("nan"))
    with pytest.raises(ReadinessNotMeasurable):
        ReadinessPolicy(min_distinct_viewpoints=0)


# -- the fusion evidence sample ----------------------------------------------
#
# The gate above is only as honest as the points it is given, and the points it
# used to be given were the wrong ones: the decimated, filtered clouds the
# registration was fitted to rather than the full-resolution frames the volume
# integrated. These are about the sampler that closes that gap - what it keeps,
# what it drops, and the bound it works inside.
#: A depth scale that makes a raw code and a distance easy to read against each
#: other in a test. It measures nothing.
SAMPLE_SCALE = 0.01

#: A synthetic calibration, built through the real 40-byte parser so it is
#: exactly as constrained as one read off the device.
SAMPLE_INTRINSICS = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)

#: The pixel stride :data:`DEFAULT_POLICY` works at, recomputed here from the
#: policy rather than written down, so a changed budget changes this too.
PIXEL_STRIDE = -(-PIXELS_PER_FRAME // DEFAULT_POLICY.fusion_pixels_per_frame)


def raw_frame(codes: dict[int, int]) -> bytes:
    """One ``Y16 `` frame of the capture's exact size, zero except where named.

    Zero is what the sensor writes for "no depth returned", and it is also how
    a support-plane mask reaches this sampler: a masked sample is a zero in the
    frame that was handed over. So a frame that is mostly zeros is the ordinary
    case here rather than a degenerate one.
    """
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    for offset, code in codes.items():
        struct.pack_into("<H", buffer, offset * 2, code)
    return bytes(buffer)


def test_the_fusion_sample_keeps_exactly_what_the_fusion_would_have_integrated() -> None:
    """Zeros, the inclusive depth window, and nothing else.

    Every rule here is the volume's own rule rather than a second version of
    it: a zero sample is no depth returned and is skipped, a distance outside
    the accepted interval is skipped exactly as
    :meth:`scanner3d.tsdf.TsdfVolume.integrate` skips it, and the interval is
    INCLUSIVE at both ends - a sample landing precisely on ``min_depth`` or
    ``max_depth`` is integrated by the fusion and is therefore evidence here.
    """
    min_depth, max_depth = 60.0, 300.0
    frame = raw_frame(
        {
            0 * PIXEL_STRIDE: 0,  # no return
            1 * PIXEL_STRIDE: 5000,  # 50.0, below the window
            2 * PIXEL_STRIDE: 6000,  # 60.0, exactly min_depth: kept
            3 * PIXEL_STRIDE: 20000,  # 200.0, inside
            4 * PIXEL_STRIDE: 30000,  # 300.0, exactly max_depth: kept
            5 * PIXEL_STRIDE: 30001,  # 300.01, above the window
            1: 20000,  # a valid sample this stride never examines
        }
    )

    evidence = fusion_evidence_sample(
        [frame],
        [Rigid.identity()],
        intrinsics=SAMPLE_INTRINSICS,
        depth_scale=SAMPLE_SCALE,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    assert sorted(round(point[2], 6) for point in evidence.points) == [60.0, 200.0, 300.0]
    assert evidence.frames_offered == 1
    assert evidence.frames_sampled == 1
    assert evidence.samples_examined == DEFAULT_POLICY.fusion_pixels_per_frame


def test_the_fusion_sample_deprojects_and_places_with_the_poses_it_was_given() -> None:
    """The same pinhole arithmetic and the same placement the export applies.

    The pose passed in is already composed with whatever presentation rotation
    the surface will be compared in - that composition is the caller's, so this
    module never grows a second opinion about which way up a reconstruction is
    - and this proves the sampler applies it rather than returning camera-frame
    points a caller has to remember to place.
    """
    offset = 3 * PIXEL_STRIDE
    row, column = divmod(offset, FRAME_WIDTH)
    frame = raw_frame({offset: 20000})
    distance = 20000 * SAMPLE_SCALE
    placement = Rigid.of(rotation=Rigid.identity().rotation, translation=(7.0, -3.0, 11.0))

    evidence = fusion_evidence_sample(
        [frame],
        [placement],
        intrinsics=SAMPLE_INTRINSICS,
        depth_scale=SAMPLE_SCALE,
        min_depth=60.0,
        max_depth=300.0,
    )

    expected = placement.apply(
        (
            (column - SAMPLE_INTRINSICS.values["cx"]) * distance / SAMPLE_INTRINSICS.values["fx"],
            (row - SAMPLE_INTRINSICS.values["cy"]) * distance / SAMPLE_INTRINSICS.values["fy"],
            distance,
        )
    )
    assert evidence.points == (expected,)


def test_the_fusion_sample_is_bounded_in_frames_and_in_pixels() -> None:
    """A long capture costs the same as a short one, and says so in the record.

    The bound is the whole reason this is affordable on the export path: a
    hundred and seventy frames at full resolution is forty-three million
    samples, and this looks at
    ``fusion_sample_frames x fusion_pixels_per_frame`` of them whatever the
    capture's length. The frames are chosen by uniform stride rather than by
    taking the first N, so the angular spread of the sweep survives the bound.
    """
    frame = raw_frame({index * PIXEL_STRIDE: 20000 for index in range(8)})
    # The same bytes object 200 times: this is about the count, not the content.
    frames = [frame] * 200

    evidence = fusion_evidence_sample(
        frames,
        [Rigid.identity()] * len(frames),
        intrinsics=SAMPLE_INTRINSICS,
        depth_scale=SAMPLE_SCALE,
        min_depth=60.0,
        max_depth=300.0,
    )

    # A uniform stride, so 200 frames at a ceiling of 64 are sampled every
    # fourth rather than the first 64 - fewer than the ceiling, and spread over
    # the whole sweep, which is the property that matters.
    stride = -(-200 // DEFAULT_POLICY.fusion_sample_frames)
    sampled = len(range(0, 200, stride))
    assert evidence.frames_offered == 200
    assert sampled <= DEFAULT_POLICY.fusion_sample_frames
    assert evidence.frames_sampled == sampled
    assert evidence.samples_examined == sampled * DEFAULT_POLICY.fusion_pixels_per_frame
    assert len(evidence.points) == 8 * sampled


def test_the_fusion_sample_refuses_frames_that_do_not_have_poses() -> None:
    """A caller error, not a verdict about the scan."""
    with pytest.raises(ReadinessNotMeasurable):
        fusion_evidence_sample(
            [raw_frame({}), raw_frame({})],
            [Rigid.identity()],
            intrinsics=SAMPLE_INTRINSICS,
            depth_scale=SAMPLE_SCALE,
            min_depth=60.0,
            max_depth=300.0,
        )


def test_the_record_states_which_frames_the_evidence_came_from() -> None:
    """The counts are in the schema so a reader does not have to take it on trust."""
    frame = raw_frame({index * PIXEL_STRIDE: 20000 for index in range(64)})
    evidence = fusion_evidence_sample(
        [frame] * 3,
        [Rigid.identity()] * 3,
        intrinsics=SAMPLE_INTRINSICS,
        depth_scale=SAMPLE_SCALE,
        min_depth=60.0,
        max_depth=300.0,
    )

    agreement, _ = surface_evidence(
        list(evidence.points), evidence, voxel_size=VOXEL, truncation=TRUNCATION
    )

    assert agreement.source_frames_offered == 3
    assert agreement.source_frames_sampled == 3
    assert agreement.source_samples_examined == 3 * DEFAULT_POLICY.fusion_pixels_per_frame
    assert agreement.source_points == len(evidence.points)
    assert "FULL-RESOLUTION" in agreement.detail


# -- the record and its own verdict, read as one statement --------------------
#
# Everything above about the record checks one field or one object. These check
# the thing no per-field rule can: that the decision a record states is the
# decision its own evidence and policy produce. A reader that parses the
# failure list without recomputing it accepts a forged verdict - an unresolved
# tracking loss with an empty failure list, or a coverage refusal beside
# adequate coverage - and reads it as a real one.


def refused_record() -> dict:
    """A valid record of a reconstruction refused for TWO different reasons.

    Coverage and surface, so the canonical ordering has something to order.
    """
    record = full_record()
    record["coverage"]["total_cells"] = 10
    record["coverage"]["supported_cells"] = 1
    record["coverage"]["supported_fraction"] = 0.1
    record["surface"]["surface_supported_fraction"] = 0.1
    record["ready"] = False
    record["failures"] = [
        {"reason": reason, "detail": "d", "advice": ADVICE[reason]}
        for reason in (COVERAGE_UNSUPPORTED, SURFACE_UNSUPPORTED)
    ]
    return record


def test_a_record_of_two_refusals_in_canonical_order_reads_back() -> None:
    """The positive control, so nothing below can pass by refusing everything."""
    readiness = ReconstructionReadiness.from_dict(refused_record())

    assert readiness.ready is False
    assert readiness.reasons == (COVERAGE_UNSUPPORTED, SURFACE_UNSUPPORTED)


def test_reasons_out_of_the_canonical_order_this_schema_publishes_are_refused() -> None:
    """One verdict has one shape.

    The order is not decoration: a front end renders the first failure's advice
    as the thing to do next, and two records of the same verdict that disagree
    about which failure is first are two different instructions.
    """
    record = refused_record()
    record["failures"] = list(reversed(record["failures"]))

    with pytest.raises(ReadinessRecordUnusable, match="canonical order"):
        ReconstructionReadiness.from_dict(record)


def test_an_unresolved_tracking_loss_with_an_empty_failure_list_is_refused() -> None:
    """The forged verdict in its most dangerous shape.

    A capture whose own authority did not close a tracking loss is refused -
    that is the whole first gate - so a record stating the loss and claiming
    readiness beside it is not a readable record of a permissive policy. It is
    a record of a decision nothing here ever made.
    """
    record = full_record()
    record["tracking"]["unresolved"] = True

    with pytest.raises(ReadinessRecordUnusable, match="unresolved tracking loss"):
        ReconstructionReadiness.from_dict(record)


def test_a_tracking_refusal_beside_resolved_tracking_evidence_is_refused() -> None:
    """And the other direction, so the rule is a rule rather than a one-way check."""
    record = full_record()
    record["ready"] = False
    record["failures"] = [{"reason": TRACKING_LOST, "detail": "d", "advice": ADVICE[TRACKING_LOST]}]

    with pytest.raises(ReadinessRecordUnusable, match="tracking evidence is resolved"):
        ReconstructionReadiness.from_dict(record)


def test_a_validated_recovery_in_the_history_is_not_a_tracking_refusal() -> None:
    """The control that keeps the rule above from becoming "never recover".

    A capture that lost tracking and whose recovery the authority validated
    carries loss records and is NOT refused. The record says both things, and
    reading it back must preserve both.
    """
    record = full_record()
    record["tracking"]["records_found"] = 2
    record["tracking"]["recovered_episodes"] = 2

    readiness = ReconstructionReadiness.from_dict(record)

    assert readiness.ready is True
    assert readiness.tracking.records_found == 2
    assert readiness.tracking.recovered_episodes == 2


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("coverage", "supported_fraction"), None, COVERAGE_UNSUPPORTED),
        (("poses", "distinct_viewpoints"), 1, POSE_VIEWPOINTS_INSUFFICIENT),
        (("poses", "excursion"), 0.0, POSE_EXCURSION_INSUFFICIENT),
        (("poses", "moving_steps"), 0, POSE_MOTION_BELOW_NOISE),
        (("surface", "surface_supported_fraction"), 0.1, SURFACE_UNSUPPORTED),
        (("surface", "evidence_explained_fraction"), 0.1, SURFACE_ABANDONS_EVIDENCE),
    ],
)
def test_a_threshold_failure_missing_from_the_reason_set_is_refused(
    path: tuple[str, str], value: object, reason: str
) -> None:
    """Evidence below a policy threshold obliges its refusal, in the record too.

    This is the check that makes the failure list evidence rather than a claim:
    a record whose numbers fail a gate and whose failure list does not say so
    is refused, whichever gate it was.
    """
    record = full_record()
    if path == ("coverage", "supported_fraction"):
        record["coverage"]["total_cells"] = 10
        record["coverage"]["supported_cells"] = 1
        record["coverage"]["supported_fraction"] = 0.1
    else:
        record[path[0]][path[1]] = value
    if path == ("poses", "moving_steps"):
        # The two still have to sum to the adjacent steps; that rule is checked
        # first and would otherwise be the one that fired.
        record["poses"]["stationary_steps"] = record["poses"]["adjacent_steps"]

    with pytest.raises(ReadinessRecordUnusable, match="own evidence and policy give"):
        ReconstructionReadiness.from_dict(record)
    assert reason in REASONS


@pytest.mark.parametrize(
    "reason",
    [
        COVERAGE_UNSUPPORTED,
        POSE_VIEWPOINTS_INSUFFICIENT,
        POSE_EXCURSION_INSUFFICIENT,
        POSE_MOTION_BELOW_NOISE,
        SURFACE_UNSUPPORTED,
        SURFACE_ABANDONS_EVIDENCE,
        COVERAGE_NOT_MEASURABLE,
        POSE_MOTION_NOT_MEASURABLE,
        SURFACE_NOT_MEASURABLE,
    ],
)
def test_a_reason_asserted_while_its_own_evidence_passes_is_refused(reason: str) -> None:
    """A refusal nothing in the record supports is as wrong as a missing one.

    Both directions matter for the same reason: the failure list is what a
    front end renders and what an operator acts on. A reason with no evidence
    behind it sends somebody to rescan an object that was fine.
    """
    record = full_record()
    record["ready"] = False
    record["failures"] = [{"reason": reason, "detail": "d", "advice": ADVICE[reason]}]

    with pytest.raises(ReadinessRecordUnusable, match="own evidence and policy give"):
        ReconstructionReadiness.from_dict(record)


def unmeasured_coverage(record: dict) -> None:
    """The exact shape :func:`coverage_evidence` emits when the map refused."""
    record["coverage"].update(
        measured=False, total_cells=0, supported_cells=0, supported_fraction=0.0
    )


def unmeasured_surface(record: dict) -> None:
    """The exact shape :func:`surface_evidence` emits with nothing to compare."""
    record["surface"].update(
        measured=False,
        surface_points=0,
        evidence_points=0,
        surface_supported_fraction=0.0,
        evidence_explained_fraction=0.0,
        surface_distance_median=0.0,
        surface_distance_p95=0.0,
        evidence_distance_median=0.0,
        evidence_distance_p95=0.0,
    )


def unmeasured_poses(record: dict) -> None:
    """The exact shape :func:`pose_evidence` emits with no probe to measure with."""
    poses = record["poses"]
    poses.update(
        probe_points=0,
        moving_steps=0,
        stationary_steps=poses["adjacent_steps"],
        travel_median=0.0,
        travel_worst=0.0,
        excursion=0.0,
        distinct_viewpoints=min(poses["poses"], 1),
        viewpoints_capped=False,
    )


@pytest.mark.parametrize(
    ("unmeasure", "reason"),
    [
        (unmeasured_coverage, COVERAGE_NOT_MEASURABLE),
        (unmeasured_surface, SURFACE_NOT_MEASURABLE),
        (unmeasured_poses, POSE_MOTION_NOT_MEASURABLE),
    ],
)
def test_unmeasured_evidence_without_its_own_refusal_is_refused(
    unmeasure: Callable[[dict], None], reason: str
) -> None:
    """ "Not measured" is never "measured and fine".

    That fallback is the exact failure this whole module exists to prevent, and
    a record is one of the places it could still happen: a gate that could not
    run leaves the reconstruction unchecked, so its refusal is obligatory and
    a record without it is not readable.

    Each mutation writes the WHOLE unmeasured shape the evaluator emits, not
    just the flag, so what is being tested here is the missing refusal and not
    the shape rules beside it - the record is otherwise one a real evaluation
    could have produced, and its only defect is the verdict.
    """
    record = full_record()
    unmeasure(record)

    with pytest.raises(ReadinessRecordUnusable, match="own evidence and policy give"):
        ReconstructionReadiness.from_dict(record)
    assert reason in REASONS


@pytest.mark.parametrize(
    ("unmeasure", "reason"),
    [
        (unmeasured_coverage, COVERAGE_NOT_MEASURABLE),
        (unmeasured_surface, SURFACE_NOT_MEASURABLE),
        (unmeasured_poses, POSE_MOTION_NOT_MEASURABLE),
    ],
)
def test_a_completely_unmeasured_gate_reads_back_with_its_own_refusal(
    unmeasure: Callable[[dict], None], reason: str
) -> None:
    """The positive control for the three shapes above.

    Without it the rules added for each unmeasured shape could be passing by
    refusing every unmeasured record, which would make an honest "this could
    not be measured" unreadable - and an unreadable refusal is a refusal
    nobody sees.
    """
    record = full_record()
    unmeasure(record)
    record["ready"] = False
    record["failures"] = [{"reason": reason, "detail": "d", "advice": ADVICE[reason]}]

    readiness = ReconstructionReadiness.from_dict(record)

    assert readiness.ready is False
    assert readiness.reasons == (reason,)


def test_a_pre_fusion_record_whose_surface_is_honestly_null_still_reads_back() -> None:
    """``None`` obliges nothing: the volume does not exist and no gate has run.

    A pre-fusion refusal is a complete, honest answer about everything knowable
    at that boundary, and the cross-check above must not turn its missing
    surface into a missing refusal.
    """
    readiness = blank()

    decoded = ReconstructionReadiness.from_dict(
        json.loads(json.dumps(readiness.as_dict(), allow_nan=False))
    )

    assert decoded == readiness
    assert decoded.surface is None
    assert decoded.ready is True


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("poses", "poses"), -1),
        (("poses", "excursion"), -1.0),
        (("poses", "travel_median"), -1.0),
        (("poses", "distinct_viewpoints"), -1),
        (("poses", "probe_points"), -1),
        (("coverage", "points_used"), -1),
        (("coverage", "cell_size"), -1.0),
        (("coverage", "frames_used"), -1),
        (("surface", "surface_points"), -1),
        (("surface", "source_points"), -1),
        (("surface", "source_frames_sampled"), -1),
        (("surface", "source_samples_examined"), -1),
        (("surface", "tolerance"), -1.0),
        (("surface", "distance_clamp"), -1.0),
        (("surface", "surface_supported_fraction"), -0.1),
        (("surface", "evidence_explained_fraction"), 1.5),
        (("tracking", "records_found"), -1),
        (("tracking", "recovered_episodes"), -1),
    ],
)
def test_a_count_distance_or_fraction_outside_its_domain_is_refused(
    path: tuple[str, str], value: object
) -> None:
    """No evaluation can produce any of these, so a record carrying one is not an evaluation."""
    record = full_record()
    record[path[0]][path[1]] = value

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("poses", "adjacent_steps"), 99),
        (("poses", "distinct_viewpoints"), 99),
        (("poses", "travel_median"), 1e9),
        (("surface", "source_frames_sampled"), 99),
        (("surface", "source_points"), 10**9),
        (("surface", "surface_distance_p95"), 1e9),
        (("coverage", "frames_used"), 10**6),
    ],
)
def test_a_count_or_distance_its_own_fields_contradict_is_refused(
    path: tuple[str, str], value: object
) -> None:
    """A number can be in its domain and still be impossible beside the rest.

    More adjacent steps than the pose count allows, more distinct viewpoints
    than there are poses, a median travel above the worst one, more fusion
    frames sampled than were offered or than the policy samples, more evidence
    points than examined samples, a p95 past the search's own clamp: each is a
    record contradicting itself rather than a record of something unusual.
    """
    record = full_record()
    record[path[0]][path[1]] = value

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


#: Policy values that are finite and non-negative and still describe a gate no
#: run could have been judged under: a required share above all of it, an angle
#: wider than two directions can span, or a scale of zero in the unit the
#: values beside it are expressed in.
IMPOSSIBLE_POLICY_VALUES: Final = [
    ("coverage_min_supported_fraction", 1.5),
    ("surface_min_supported_fraction", 1.5),
    ("surface_min_evidence_explained", 1.5),
    ("coverage_min_spread", 4.0),
    ("noise_floor", 0.0),
    ("viewpoint_separation_multiple", 0.0),
    ("surface_cell_fraction", 0.0),
    ("max_viewpoints_examined", 2),
    ("coverage_max_frames", 1),
]

#: The same values, under the names the record publishes them by. Two of them
#: differ, so a JSON control cannot simply reuse the attribute name.
POLICY_RECORD_NAMES: Final = {
    "coverage_min_spread": "coverage_min_spread_radians",
}


@pytest.mark.parametrize(("name", "value"), IMPOSSIBLE_POLICY_VALUES)
def test_a_policy_that_no_gate_could_be_judged_under_is_refused(name: str, value: object) -> None:
    """Refused where it is BUILT, before anything is measured against it.

    Each of these is in range for its type and still not a policy: a gate
    demanding more than all of something, more angle than exists between two
    directions, a zero scale, viewpoint counting that stops below its own
    minimum, or a coverage frame budget under the observations it requires.
    Left standing, each one silently refuses every reconstruction forever, and
    the record it writes says the scan was bad.
    """
    keyword: dict[str, Any] = {name: value}

    with pytest.raises(ReadinessNotMeasurable):
        ReadinessPolicy(**keyword)


def test_a_surface_tolerance_of_exactly_zero_by_construction_is_refused() -> None:
    """Both terms of the tolerance at zero is a tolerance of zero.

    Each multiple alone may legitimately be zero - a policy can derive the
    tolerance from the truncation alone or from the noise floor alone - so this
    is a relationship between them and not a domain on either.
    """
    ReadinessPolicy(surface_tolerance_truncations=0.0)
    ReadinessPolicy(surface_tolerance_noise_multiple=0.0)

    with pytest.raises(ReadinessNotMeasurable):
        ReadinessPolicy(surface_tolerance_truncations=0.0, surface_tolerance_noise_multiple=0.0)


def test_a_strict_but_reachable_policy_is_still_accepted() -> None:
    """The positive control: strict is not the same as impossible.

    Every value here sits exactly on a boundary the rules above refuse to
    cross, and each one is a policy somebody could reasonably state.
    """
    policy = ReadinessPolicy(
        coverage_min_supported_fraction=1.0,
        surface_min_supported_fraction=1.0,
        surface_min_evidence_explained=0.0,
        coverage_min_spread=math.pi,
        min_distinct_viewpoints=4,
        max_viewpoints_examined=4,
        coverage_min_observations=2,
        coverage_max_frames=2,
    )

    assert policy.max_viewpoints_examined == policy.min_distinct_viewpoints
    assert policy.coverage_max_frames == policy.coverage_min_observations


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("noise_floor", -1.0),
        ("min_excursion_multiple", -1.0),
        ("coverage_min_supported_fraction", -0.5),
        ("min_distinct_viewpoints", 0),
        ("fusion_sample_frames", 0),
        ("fusion_pixels_per_frame", -1),
        *IMPOSSIBLE_POLICY_VALUES,
    ],
)
def test_a_policy_value_outside_its_domain_is_refused(name: str, value: object) -> None:
    """The policy is validated where it is constructed, and again where it is read."""
    record = full_record()
    record["policy"][POLICY_RECORD_NAMES.get(name, name)] = value

    with pytest.raises((ReadinessRecordUnusable, ReadinessNotMeasurable)):
        ReconstructionReadiness.from_dict(record)


# -- records the evaluator could not have produced ---------------------------
#
# Codex's round-2 review named three records that parsed and that no run of
# this module emits. The rules below are stated as the SHAPES the evaluator
# actually has - a resolved tracking state, a measured gate, an unmeasured one
# - so that the reproductions are refused by an invariant rather than by three
# special cases, and each has a positive control beside it.


def test_a_resolved_tracking_state_that_recovered_nothing_it_found_is_refused() -> None:
    """Codex's first reproduction. One loss found, none recovered, resolved anyway.

    The capture reader refuses any loss no validated recovery segment closed,
    so "resolved" and "found a loss nothing closed" cannot both be true of one
    capture. A record asserting them is asserting that the authority passed a
    capture it refuses.
    """
    record = full_record()
    record["tracking"].update(records_found=1, recovered_episodes=0, unresolved=False)

    with pytest.raises(ReadinessRecordUnusable, match="resolved tracking state"):
        ReconstructionReadiness.from_dict(record)


def test_an_unresolved_tracking_state_cannot_claim_a_validated_recovery() -> None:
    """The other side of it: the unresolved paths never reach the validator.

    :func:`tracking_evidence` returns unresolved only by giving up - an
    unlistable directory, an unreadable record, a capture the reader refused -
    and every one of those returns before a recovery could have been
    validated.
    """
    record = full_record()
    record["tracking"].update(records_found=2, recovered_episodes=1, unresolved=True)
    record["ready"] = False
    record["failures"] = [{"reason": TRACKING_LOST, "detail": "d", "advice": ADVICE[TRACKING_LOST]}]

    with pytest.raises(ReadinessRecordUnusable, match="unresolved or unreadable"):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    ("found", "recovered", "unresolved", "reasons"),
    [
        (0, 0, False, ()),
        (2, 2, False, ()),
        (1, 0, True, (TRACKING_LOST,)),
        (0, 0, True, (TRACKING_LOSS_UNREADABLE,)),
    ],
)
def test_every_tracking_state_the_authority_can_leave_behind_reads_back(
    found: int, recovered: int, unresolved: bool, reasons: tuple[str, ...]
) -> None:
    """The positive control, over each shape :func:`tracking_evidence` returns.

    A healthy capture, a capture whose losses were all validly recovered, a
    capture the authority refused, and a capture that could not be listed or
    read at all. All four must still read back, because a rule that refused
    any of them would make an honest refusal unreadable.
    """
    record = full_record()
    record["tracking"].update(
        records_found=found, recovered_episodes=recovered, unresolved=unresolved
    )
    record["ready"] = not reasons
    record["failures"] = [
        {"reason": reason, "detail": "d", "advice": ADVICE[reason]} for reason in reasons
    ]

    readiness = ReconstructionReadiness.from_dict(record)

    assert readiness.tracking.records_found == found
    assert readiness.tracking.recovered_episodes == recovered
    assert readiness.reasons == reasons


def test_a_measured_surface_drawn_from_no_points_at_all_is_refused() -> None:
    """Codex's second reproduction. A downsampled cloud bigger than its source.

    The evidence pipeline only ever shrinks - examined candidates, then
    deprojected points, then one point per comparison cell, then the strided
    queries - so a record whose thinned count exceeds the count it was thinned
    from has invented one of the two.
    """
    record = full_record()
    record["surface"]["source_points"] = 0

    with pytest.raises(ReadinessRecordUnusable, match="downsampling never produces more"):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize("name", ["surface_points", "evidence_points", "source_frames_sampled"])
def test_a_measured_surface_with_an_empty_population_is_refused(name: str) -> None:
    """A direction with no query or no target is exactly when it does NOT measure."""
    record = full_record()
    record["surface"][name] = 0
    if name == "evidence_points":
        record["surface"]["source_points"] = 0

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    "names",
    [
        ("surface_supported_fraction",),
        ("evidence_explained_fraction",),
        ("surface_distance_median", "surface_distance_p95"),
        ("evidence_distance_median", "evidence_distance_p95"),
    ],
)
def test_an_unmeasured_surface_that_reports_a_measurement_is_refused(
    names: tuple[str, ...],
) -> None:
    """ "Not measured" carries no fraction and no distance; that is its whole shape.

    The two distance fields of a direction move together, so that this refusal
    is the unmeasured rule and not the median-past-its-own-p95 rule beside it.
    """
    record = full_record()
    unmeasured_surface(record)
    record["ready"] = False
    record["failures"] = [
        {"reason": SURFACE_NOT_MEASURABLE, "detail": "d", "advice": ADVICE[SURFACE_NOT_MEASURABLE]}
    ]
    for name in names:
        record["surface"][name] = 0.5

    with pytest.raises(ReadinessRecordUnusable, match="could not be measured"):
        ReconstructionReadiness.from_dict(record)


def test_unmeasured_coverage_that_carries_a_populated_map_is_refused() -> None:
    """Codex's third reproduction. A refused coverage map produces no cells.

    ``measured=False`` is reached only where
    :func:`scanner3d.coverage.coverage_map` raised, so there is no map behind
    it to have counted anything. The frames and points offered TO it stay,
    because those describe the call rather than its result.
    """
    record = full_record()
    unmeasured_coverage(record)
    record["ready"] = False
    record["failures"] = [
        {
            "reason": COVERAGE_NOT_MEASURABLE,
            "detail": "d",
            "advice": ADVICE[COVERAGE_NOT_MEASURABLE],
        }
    ]
    record["coverage"].update(total_cells=10, supported_cells=10, supported_fraction=1.0)

    with pytest.raises(ReadinessRecordUnusable, match="refused coverage map produces no cells"):
        ReconstructionReadiness.from_dict(record)


def test_an_unmeasurable_trajectory_that_reports_travel_is_refused() -> None:
    """With no probe there is nothing any of those numbers could be a measure of."""
    record = full_record()
    unmeasured_poses(record)
    record["ready"] = False
    record["failures"] = [
        {
            "reason": POSE_MOTION_NOT_MEASURABLE,
            "detail": "d",
            "advice": ADVICE[POSE_MOTION_NOT_MEASURABLE],
        }
    ]
    record["poses"]["excursion"] = 5.0

    with pytest.raises(ReadinessRecordUnusable, match="nothing here to measure it against"):
        ReconstructionReadiness.from_dict(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("poses", "probe_points"), DEFAULT_POLICY.probe_points + 1),
        (("coverage", "points_used"), 10**9),
        (("surface", "source_samples_examined"), 10**9),
        (("surface", "surface_points"), DEFAULT_POLICY.surface_sample_points + 1),
        (("surface", "evidence_points"), DEFAULT_POLICY.surface_sample_points + 1),
    ],
)
def test_a_count_above_the_bound_its_own_policy_states_is_refused(
    path: tuple[str, str], value: object
) -> None:
    """Every gate here is bounded, and the policy that bounded it is in the record.

    So a count above its own stated ceiling is not a large scan; it is a count
    no run under that policy produced. ``points_used`` and
    ``source_samples_examined`` are the per-frame ceilings times the frames
    actually used, which is the arithmetic a forged record has to satisfy.
    """
    record = full_record()
    record[path[0]][path[1]] = value

    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(record)


def test_viewpoint_counting_that_claims_a_cap_it_did_not_reach_is_refused() -> None:
    """Counting stops AT the cap, so a cap at any other number is two runs."""
    record = full_record()
    record["poses"]["viewpoints_capped"] = True

    with pytest.raises(ReadinessRecordUnusable, match="hit its cap"):
        ReconstructionReadiness.from_dict(record)


def test_a_genuinely_capped_viewpoint_count_still_reads_back() -> None:
    """The positive control: a long orbit legitimately stops counting at the cap."""
    clouds, poses = orbit(40)
    evidence, failures = pose_evidence(clouds, poses)

    assert evidence.viewpoints_capped is True
    assert evidence.distinct_viewpoints == DEFAULT_POLICY.max_viewpoints_examined
    assert failures == ()

    record = full_record()
    record["poses"].update(
        json.loads(json.dumps(evidence.as_dict(), allow_nan=False)),
    )

    assert ReconstructionReadiness.from_dict(record).poses == evidence


# -- the exact support boundary ----------------------------------------------


def test_a_point_exactly_at_the_tolerance_is_supported() -> None:
    """ "Within" is INCLUSIVE, and it is inclusive because the repository says so.

    :func:`scanner3d.quality.cloud_agreement` counts ``distance <= tolerance``
    and is the repository's own name for this measurement. The hash here had
    counted ``<`` , so one exact pair of clouds got two different answers
    depending on which instrument was asked. The direct case, first.
    """
    tolerance = 2.0

    within, distances = _directed_support([(0.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)], tolerance=tolerance)

    assert within == 1
    reference = cloud_agreement(
        [(0.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)], name="control", coverage_tolerance=tolerance
    )
    assert reference.observed_to_reference.coverage_fraction == 1.0
    # And the distance is still the clamped one: at exactly the tolerance the
    # search cannot tell "this far" from "this far or further", and says so.
    assert distances == (tolerance,)


@pytest.mark.parametrize(
    ("distance", "supported"), [(1.9999999999, True), (2.0, True), (2.0000000001, False)]
)
def test_the_support_boundary_is_exercised_from_both_sides(
    distance: float, supported: bool
) -> None:
    """Just inside, exactly on, and just outside - the controls for the case above."""
    within, _ = _directed_support([(0.0, 0.0, 0.0)], [(distance, 0.0, 0.0)], tolerance=2.0)

    assert bool(within) is supported


def test_the_hash_stays_exact_at_the_boundary_wherever_the_cell_walls_fall() -> None:
    """The inequality changed, so the search's exactness argument is re-proved.

    The hash indexes at a cell of one tolerance and searches 27 cells. That is
    exact because a target within one tolerance of a query differs from it by
    at most one tolerance on every axis, so their cell keys differ by at most
    one - and the arrangement where that is tightest is exactly the one the
    inclusive boundary introduces: a target landing precisely ON a cell wall.

    So every relevant arrangement is walked: the query at many fractions of a
    cell, the target displaced by exactly the tolerance along each axis, each
    sign and a diagonal, and the answer compared with brute force every time.
    """
    tolerance = 2.0
    directions = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    ]
    fractions = [0.0, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-12, 1.0, 1.5, 1.999, 2.0, -0.5, -1.0]
    for fraction in fractions:
        base = fraction * tolerance
        query = (base, base * 0.5, -base)
        for direction in directions:
            for scale in (tolerance, tolerance * (1.0 - 1e-9), tolerance * (1.0 + 1e-9)):
                target: Point = (
                    query[0] + direction[0] * scale,
                    query[1] + direction[1] * scale,
                    query[2] + direction[2] * scale,
                )
                within, _ = _directed_support([query], [target], tolerance=tolerance)
                brute = 1 if math.dist(query, target) <= tolerance else 0
                assert within == brute, f"{query} -> {target} at scale {scale}"
