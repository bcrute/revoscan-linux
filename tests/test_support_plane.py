"""What the support-plane filter removes, what it refuses to, and what it costs.

Every scene here is built from arithmetic rather than captured, and deliberately
so: the question this module answers is whether a GATE fires, and a gate is
tested by putting a scene exactly on either side of it. A real frame proves
neither - it is one point in the space, and whichever way it came out the test
would pass for a reason nobody chose.

The three properties worth stating about the fixtures:

* no random seed anywhere, including in the "noisy" scenes, whose jitter is a
  closed-form function of the sample index. A filter this repository claims is
  deterministic cannot be tested by a fixture that is not;
* every scene sits in the depth interval :mod:`scanner3d.preparation` keeps, so
  the same points can be pushed through the shared preparation path unchanged;
* the object is always a thing you could put on a table - a box, a plate, a
  ridge - because the gate that matters most is the one separating "flat thing
  the object is standing on" from "flat thing the object is made of".
"""

from __future__ import annotations

import math

import pytest

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, normalize, rotation_about_axis
from scanner3d.support_plane import (
    BAND_DIAGONAL_FRACTION,
    CANDIDATE_SEEDS,
    CANDIDATES,
    COARSE_BAND_FRACTION,
    DEFAULT_THRESHOLDS,
    MAX_INPUT_POINTS,
    MAX_OBJECT_FOOTPRINT_FRACTION,
    MIN_INLIER_FRACTION,
    MIN_POINTS_TO_FIT,
    POLICY_VERSION,
    REASON_APPLIED,
    REASON_DEGENERATE,
    REASON_FEW_INLIERS,
    REASON_LITTLE_OBJECT,
    REASON_NOT_SURROUNDING,
    REASON_SPARSE,
    REASON_TWO_SIDED,
    SCHEMA_VERSION,
    BandUnusable,
    InputTooLarge,
    NothingToFilter,
    PointUnusable,
    SupportPlaneNotAllowed,
    ThresholdUnusable,
    suppress_support_plane,
)

#: Where every scene sits along the camera's depth axis. Inside the interval
#: scanner3d.preparation keeps, so the same points survive the shared path.
BED_DEPTH = 120.0

#: The largest rotation error, in degrees, the bed-dominance regression accepts
#: from the prepared clouds. Named rather than inlined: this is the number the
#: acceptance contract for this work is written against, and a test that fails
#: should say which gate it failed.
TRANSFORM_ERROR_GATE_DEGREES = 1.5


def plane_points(
    *,
    span: float,
    steps: int,
    height: float,
    tilt: tuple[float, float] = (0.0, 0.0),
    jitter: float = 0.0,
    centre: tuple[float, float] = (0.0, 0.0),
) -> list[Point]:
    """A flat sheet, optionally slanted and optionally noisy, on a regular lattice.

    ``jitter`` displaces each sample along the plane's own normal direction by a
    closed-form function of its index. It is not random: the same scene is the
    same points on every machine, which is the only way a determinism claim can
    be checked by a test.
    """
    points: list[Point] = []
    for i in range(steps):
        for j in range(steps):
            x = centre[0] - span / 2.0 + span * i / (steps - 1)
            y = centre[1] - span / 2.0 + span * j / (steps - 1)
            wobble = jitter * math.sin(2.399 * (i * steps + j)) if jitter else 0.0
            points.append((x, y, height + tilt[0] * x + tilt[1] * y + wobble))
    return points


def block(
    *,
    span: float,
    steps: int,
    base: float,
    tall: float,
    centre: tuple[float, float] = (0.0, 0.0),
) -> list[Point]:
    """A closed box shell standing on ``base``: a top face and four walls.

    A shell rather than a solid, because a depth camera only ever sees surface,
    and a solid would give the plane fit interior points no frame contains.
    """
    points: list[Point] = []
    for i in range(steps):
        for j in range(steps):
            x = centre[0] - span / 2.0 + span * i / (steps - 1)
            y = centre[1] - span / 2.0 + span * j / (steps - 1)
            points.append((x, y, base + tall))
            on_edge = i in (0, steps - 1) or j in (0, steps - 1)
            if on_edge:
                for k in range(1, steps):
                    points.append((x, y, base + tall * k / steps))
    return points


def ridge(*, length: float, steps: int, base: float, height: float) -> list[Point]:
    """An asymmetric fin, so a scene's object is not invariant under a quarter turn."""
    return [
        (length * i / (steps - 1) - length / 4.0, 0.0, base + height * i / (steps - 1))
        for i in range(steps)
        for _ in range(3)
    ]


def bed_with_block(*, jitter: float = 0.0, tilt: tuple[float, float] = (0.0, 0.0)) -> list[Point]:
    """The motivating scene: a large stationary support surface with a box on it."""
    bed = plane_points(span=220.0, steps=45, height=BED_DEPTH, tilt=tilt, jitter=jitter)
    box = block(span=60.0, steps=16, base=BED_DEPTH + 6.0, tall=30.0)
    lifted = [(p[0], p[1], p[2] + tilt[0] * p[0] + tilt[1] * p[1]) for p in box]
    return bed + lifted


def transformed(points: list[Point], move: Rigid) -> list[Point]:
    return [move.apply(point) for point in points]


def as_multiset(points: tuple[Point, ...]) -> list[tuple[float, float, float]]:
    """The points as a comparable set, rounded past the noise of a rigid transform."""
    return sorted((round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p in points)


# --- what it removes ----------------------------------------------------------


def test_an_exact_flat_support_plane_is_removed_and_the_object_survives():
    scene = bed_with_block()
    result = suppress_support_plane(scene)

    assert result.applied is True
    assert result.reason == REASON_APPLIED
    assert result.plane is not None
    # The bed is flat and horizontal in this scene, so the fitted normal is the
    # depth axis and the offset is where the bed actually is - checked against
    # the scene rather than against whatever the fit happened to produce.
    assert abs(abs(result.plane.normal[2]) - 1.0) < 1e-6
    assert abs(result.plane.offset - BED_DEPTH) < 0.5
    assert all(point[2] > BED_DEPTH + 1.0 for point in result.points)
    assert all(abs(point[2] - BED_DEPTH) < 2.0 for point in result.removed)


def test_a_slanted_support_plane_is_removed_without_any_assumption_about_up():
    # A tilt no camera axis is aligned with. Nothing in the module knows which
    # way gravity points, which is the whole reason this case is separate.
    scene = bed_with_block(tilt=(0.28, -0.17))
    result = suppress_support_plane(scene)

    assert result.applied is True
    assert result.plane is not None
    normal = result.plane.normal
    expected = normalize((-0.28, 0.17, 1.0))
    assert abs(abs(sum(a * b for a, b in zip(normal, expected, strict=True))) - 1.0) < 1e-4
    assert result.removed_points > result.kept_points


def test_a_noisy_slanted_support_plane_is_still_removed():
    scene = bed_with_block(tilt=(0.28, -0.17), jitter=0.4)
    result = suppress_support_plane(scene)

    assert result.applied is True
    # Every bed sample was inside the band despite the noise: the band is a
    # fraction of the cloud, and the point of the test is that a real surface's
    # roughness does not push a third of it out of its own plane.
    assert result.removed_points > 2000


def test_a_flat_chassis_plate_above_a_bed_survives_while_the_bed_goes():
    """The case that must not be got wrong: two parallel planes, one is the bed."""
    bed = plane_points(span=220.0, steps=45, height=BED_DEPTH)
    plate = plane_points(span=80.0, steps=30, height=BED_DEPTH + 30.0)
    result = suppress_support_plane(bed + plate)

    assert result.applied is True
    kept = as_multiset(result.points)
    assert kept == as_multiset(tuple(plate))
    assert len(result.removed) == len(bed)


# --- what it refuses to remove ------------------------------------------------


def test_an_almost_entirely_planar_panel_keeps_every_point():
    """A frame pointed at a chassis panel is a frame whose one plane IS the object."""
    panel = plane_points(span=220.0, steps=45, height=BED_DEPTH)
    speck = block(span=20.0, steps=4, base=BED_DEPTH + 4.0, tall=8.0)
    scene = panel + speck
    result = suppress_support_plane(scene)

    assert result.applied is False
    assert result.reason == REASON_LITTLE_OBJECT
    assert as_multiset(result.points) == as_multiset(tuple(scene))
    assert result.removed == ()


def test_a_compact_planar_face_inside_a_larger_scene_keeps_every_point():
    """A flat face is not a support surface if the frame is much bigger than it."""
    face = plane_points(span=45.0, steps=26, height=BED_DEPTH + 40.0)
    body = block(span=200.0, steps=20, base=BED_DEPTH, tall=55.0)
    scene = face + body
    result = suppress_support_plane(scene)

    assert result.applied is False
    assert as_multiset(result.points) == as_multiset(tuple(scene))


def test_a_plane_with_geometry_on_both_sides_keeps_every_point():
    """Things rest ON a support surface. Geometry either side means it cuts something."""
    sheet = plane_points(span=220.0, steps=45, height=BED_DEPTH)
    above = block(span=60.0, steps=16, base=BED_DEPTH + 6.0, tall=25.0)
    below = [(p[0], p[1], 2.0 * BED_DEPTH - p[2]) for p in above]
    result = suppress_support_plane(sheet + above + below)

    assert result.applied is False
    assert result.reason == REASON_TWO_SIDED


def test_the_lid_of_a_box_is_not_a_bed_even_though_it_looks_exactly_like_one():
    """A plane with all the geometry on one side of it, which is still the object.

    Seen from above, a box's top face has everything a support surface has: it
    is the largest plane in the frame, it spans the frame's whole extent, and
    every other point is on one side of it. The one thing it does not have is
    room around the object - a lid is exactly as wide as the box it closes,
    while a thing standing on a bed leaves bed showing around it. That is the
    only evidence available without knowing which way up the scanner was, and
    it is what :data:`MAX_OBJECT_FOOTPRINT_FRACTION` tests.
    """
    scene = block(span=120.0, steps=18, base=BED_DEPTH, tall=110.0)
    result = suppress_support_plane(scene)

    assert result.applied is False
    assert result.reason == REASON_NOT_SURROUNDING
    assert result.object_footprint_major > 0.99
    assert as_multiset(result.points) == as_multiset(tuple(scene))


def test_a_cloud_too_sparse_to_fit_withdraws_rather_than_guessing():
    scene = plane_points(span=100.0, steps=7, height=BED_DEPTH)
    assert len(scene) < MIN_POINTS_TO_FIT
    result = suppress_support_plane(scene)

    assert result.applied is False
    assert result.reason == REASON_SPARSE
    assert len(result.points) == len(scene)


def test_a_cloud_with_no_extent_withdraws_rather_than_dividing_by_it():
    result = suppress_support_plane([(1.0, 2.0, BED_DEPTH)] * (MIN_POINTS_TO_FIT + 10))

    assert result.applied is False
    assert result.reason == REASON_DEGENERATE
    assert result.diagonal == 0.0


# --- invariances --------------------------------------------------------------


def test_the_answer_does_not_depend_on_the_order_the_points_arrived_in():
    scene = bed_with_block(tilt=(0.2, 0.1), jitter=0.3)
    straight = suppress_support_plane(scene)
    # Two permutations, because a reversal alone can be undone by a symmetric
    # accumulation and would not prove anything about summation order.
    reversed_result = suppress_support_plane(list(reversed(scene)))
    interleaved = suppress_support_plane(scene[1::2] + scene[0::2])

    for permuted in (reversed_result, interleaved):
        assert permuted.applied == straight.applied
        assert permuted.plane == straight.plane
        assert permuted.band == straight.band
        assert as_multiset(permuted.points) == as_multiset(straight.points)


def test_the_answer_does_not_depend_on_where_the_scene_is_or_which_way_it_faces():
    scene = bed_with_block(tilt=(0.15, -0.22))
    move = rotation_about_axis(
        normalize((0.3, -0.7, 0.65)), 0.9, through=(12.0, -8.0, BED_DEPTH)
    ).compose(Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (40.0, 15.0, -5.0)))
    straight = suppress_support_plane(scene)
    moved = suppress_support_plane(transformed(scene, move))

    assert moved.applied is straight.applied is True
    assert moved.removed_points == straight.removed_points
    # The same POINTS survive, carried through the same transform: a rigid
    # motion cannot change which points are the bed, and a filter whose answer
    # depended on the camera's pose would be measuring the pose.
    assert as_multiset(moved.points) == as_multiset(tuple(transformed(list(straight.points), move)))


# --- refusals -----------------------------------------------------------------


def test_an_empty_cloud_is_refused_rather_than_answered():
    with pytest.raises(NothingToFilter):
        suppress_support_plane([])


def test_a_cloud_past_the_ceiling_is_refused_without_being_read():
    with pytest.raises(InputTooLarge):
        suppress_support_plane([(0.0, 0.0, 0.0)] * (MAX_INPUT_POINTS + 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_point_is_refused_before_anything_is_fitted(bad: float):
    scene = bed_with_block()
    scene[17] = (scene[17][0], bad, scene[17][2])
    with pytest.raises(PointUnusable):
        suppress_support_plane(scene)


def test_a_point_that_is_not_a_triple_is_refused():
    scene: list[object] = list(bed_with_block())
    scene[3] = (1.0, 2.0)
    with pytest.raises(PointUnusable):
        suppress_support_plane(scene)  # type: ignore[arg-type]


@pytest.mark.parametrize("band", [0.0, -1.0, float("nan"), 1e-12])
def test_a_band_that_cannot_describe_a_plane_is_refused(band: float):
    with pytest.raises(BandUnusable):
        suppress_support_plane(bed_with_block(), band=band)


def test_a_band_that_is_a_quarter_of_the_cloud_is_refused():
    with pytest.raises(BandUnusable):
        suppress_support_plane(bed_with_block(), band=500.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_inlier_fraction": 1.5},
        {"min_spread_fraction": -0.1},
        {"min_object_fraction": float("nan")},
        {"min_one_sided_fraction": 2.0},
        {"max_removed_fraction": 0.0},
        {"max_removed_fraction": 1.5},
    ],
)
def test_a_threshold_that_is_not_a_gate_is_refused(kwargs: dict[str, float]):
    with pytest.raises(ThresholdUnusable):
        suppress_support_plane(bed_with_block(), **kwargs)


def test_every_refusal_is_a_scanner_error_with_a_status():
    assert issubclass(SupportPlaneNotAllowed, ScannerError)
    assert SupportPlaneNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE


# --- the record ---------------------------------------------------------------


def test_the_record_carries_the_evidence_the_decision_was_made_on():
    result = suppress_support_plane(bed_with_block())
    record = result.as_dict()

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["policy_version"] == POLICY_VERSION
    assert record["applied"] is True
    assert record["reason"] == REASON_APPLIED
    assert record["dimensional_accuracy_claimed"] is False
    evidence = record["support_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["one_sided_fraction"] > 0.99
    assert evidence["object_footprint_major"] < MAX_OBJECT_FOOTPRINT_FRACTION
    assert evidence["spread_major"] > 0.9
    assert record["band"] == pytest.approx(BAND_DIAGONAL_FRACTION * result.diagonal)
    assert any("not background subtraction" in note for note in result.limitations)


def test_a_withdrawal_still_reports_what_it_measured_and_why_it_stopped():
    panel = plane_points(span=220.0, steps=45, height=BED_DEPTH)
    speck = block(span=20.0, steps=4, base=BED_DEPTH + 4.0, tall=8.0)
    result = suppress_support_plane(panel + speck)
    record = result.as_dict()

    assert record["applied"] is False
    assert record["reason"] == REASON_LITTLE_OBJECT
    # The candidate it looked at is on the record. A refusal whose numbers are
    # invisible cannot be told from one that never looked.
    assert record["plane"] is not None
    assert result.inlier_points > 0
    assert any(REASON_LITTLE_OBJECT in note for note in result.limitations)


def test_the_record_states_the_gates_that_actually_ran_not_the_defaults():
    """A caller's own threshold is what decided the frame, so it is what is reported.

    The scene below is removed under the shipped policy and withdrawn under a
    stricter inlier gate. Both records are checked, because the failure this
    pins is not "the number is missing" - it is a record that quotes the
    shipped 0.15 beside a decision 0.99 made, which reads as a bug in the
    filter rather than as the caller's own setting.
    """
    scene = bed_with_block()

    shipped = suppress_support_plane(scene)
    strict = suppress_support_plane(scene, min_inlier_fraction=0.99)

    assert shipped.applied is True
    assert strict.applied is False
    assert strict.reason == REASON_FEW_INLIERS
    # Both clouds are intact under the stricter gate: a withdrawal returns
    # everything, so the two runs differ in the DECISION and in nothing else.
    assert as_multiset(strict.points) == as_multiset(tuple(scene))

    shipped_gates = shipped.as_dict()["thresholds"]
    strict_gates = strict.as_dict()["thresholds"]
    assert isinstance(shipped_gates, dict)
    assert isinstance(strict_gates, dict)
    assert shipped_gates["min_inlier_fraction"] == MIN_INLIER_FRACTION
    assert shipped_gates["overridden_by_caller"] == []
    assert strict_gates["min_inlier_fraction"] == 0.99
    assert strict_gates["overridden_by_caller"] == ["min_inlier_fraction"]
    # The gates a caller cannot move are still stated, on both records: they
    # judged the frame too.
    assert strict_gates["min_spread_aspect"] == shipped_gates["min_spread_aspect"]
    assert strict.thresholds.min_inlier_fraction == 0.99
    assert shipped.thresholds == DEFAULT_THRESHOLDS


def test_a_withdrawal_reports_the_candidates_it_scored_not_the_ceiling():
    """Nine is the bound on the search, not a count of what this frame cost.

    An exactly planar frame is claimed WHOLE by the first round's winner, so
    there is nothing left to re-seed the second round on and the deflation loop
    stops after three candidates. A record saying nine would describe six
    refinements that never ran.
    """
    result = suppress_support_plane(plane_points(span=220.0, steps=45, height=BED_DEPTH))

    assert result.applied is False
    assert result.candidates_scored == CANDIDATE_SEEDS
    assert result.candidates_scored < CANDIDATES
    search = result.as_dict()["search"]
    assert isinstance(search, dict)
    assert search["candidates_scored"] == CANDIDATE_SEEDS


def test_a_withdrawal_before_any_candidate_reports_no_candidates_at_all():
    """A cloud too sparse to fit scored nothing, and says nothing was scored."""
    result = suppress_support_plane(
        [(float(index), 0.0, BED_DEPTH) for index in range(MIN_POINTS_TO_FIT - 1)]
    )

    assert result.applied is False
    assert result.reason == REASON_SPARSE
    assert result.candidates_scored == 0
    # And the gates are still on the record, because a refusal this early was
    # still a refusal made under some policy.
    assert result.thresholds == DEFAULT_THRESHOLDS


# --- the regression this work exists for --------------------------------------


def rotating_scene(angle: float) -> list[Point]:
    """A stationary bed with an object that has turned by ``angle`` radians on it.

    The bed is three times the object. That ratio is the failure: pairwise ICP
    has no notion of which points belong to the thing being reconstructed, and
    the cheapest way to explain the next frame is to leave the bed where it is
    and call the pose identity.
    """
    bed = plane_points(span=240.0, steps=45, height=BED_DEPTH)
    turn = rotation_about_axis((0.0, 0.0, 1.0), angle, through=(0.0, 0.0, BED_DEPTH))
    object_points = block(span=70.0, steps=17, base=BED_DEPTH + 5.0, tall=34.0) + ridge(
        length=60.0, steps=24, base=BED_DEPTH + 40.0, height=12.0
    )
    return bed + transformed(object_points, turn)


def rotation_error_degrees(recovered: Rigid, truth: float) -> float:
    return abs(math.degrees(recovered.rotation_angle) - math.degrees(truth))


def test_a_dominant_bed_defeats_adjacent_icp_and_suppression_recovers_the_motion():
    from scanner3d.icp_plane import icp_point_to_plane

    turn = math.radians(8.0)
    first = rotating_scene(0.0)
    second = rotating_scene(turn)

    raw = icp_point_to_plane(first, second, max_iterations=60)
    raw_error = rotation_error_degrees(raw.transform, turn)

    prepared_first = suppress_support_plane(first)
    prepared_second = suppress_support_plane(second)
    assert prepared_first.applied and prepared_second.applied
    cleaned = icp_point_to_plane(prepared_first.points, prepared_second.points, max_iterations=60)
    cleaned_error = rotation_error_degrees(cleaned.transform, turn)

    # Without suppression the bed pins the fit near identity, so the error is
    # most of the motion itself. With it, the named gate.
    assert raw_error > 4.0
    assert cleaned_error < TRANSFORM_ERROR_GATE_DEGREES
    assert cleaned_error < raw_error / 3.0


# --- a bed the covariance seed points well away from --------------------------

#: A support surface seen at a grazing angle: ``y = GRAZING_SLOPE * z -
#: GRAZING_DROP`` leans about nineteen degrees off the camera's line of sight
#: and recedes through the object's own extent, the way a turntable does from a
#: low tripod. Its normal and offset are stated so the fit can be checked
#: against the scene rather than against whatever the fit produced.
GRAZING_SLOPE = 0.35
GRAZING_DROP = 45.5
GRAZING_NORMAL = normalize((0.0, -1.0, GRAZING_SLOPE))
GRAZING_OFFSET = -GRAZING_DROP * GRAZING_NORMAL[1]
BALL_RADIUS = 30.0


def grazing_bed(x_steps: int = 22, z_steps: int = 60) -> list[Point]:
    """The slanted bed, denser near the camera the way a pixel grid samples one.

    Off-centre in x on purpose: a bed and an object that are not symmetric
    about the camera's axis give the whole cloud a smallest covariance
    direction well away from the bed's normal, and that direction is the seed
    the search has to recover from.
    """
    points: list[Point] = []
    for i in range(x_steps):
        x = -5.0 + 105.0 * i / (x_steps - 1)
        for j in range(z_steps):
            reciprocal = 1 / 80.0 + (1 / 290.0 - 1 / 80.0) * j / (z_steps - 1)
            z = 1 / reciprocal
            points.append((x, GRAZING_SLOPE * z - GRAZING_DROP, z))
    return points


def resting_ball(steps: int) -> list[Point]:
    """The camera-facing cap of a ball resting on the grazing bed at z = 130.

    ``steps`` sets how densely the cap is sampled, and so what share of the
    cloud the ball is. A depth camera sees only the surface that faces it and
    only what is on its side of the bed, so both halves are cut away.
    """
    contact = (8.0, GRAZING_SLOPE * 130.0 - GRAZING_DROP, 130.0)
    centre = tuple(c - BALL_RADIUS * n for c, n in zip(contact, GRAZING_NORMAL, strict=True))
    points: list[Point] = []
    for a in range(steps):
        theta = math.pi * (a + 0.5) / steps
        for b in range(2 * steps):
            phi = 2 * math.pi * (b + 0.5) / (2 * steps)
            direction = (
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            )
            point = (
                centre[0] + BALL_RADIUS * direction[0],
                centre[1] + BALL_RADIUS * direction[1],
                centre[2] + BALL_RADIUS * direction[2],
            )
            faces_camera = sum(-d * p for d, p in zip(direction, point, strict=True)) > 0.0
            camera_side = (
                sum(n * p for n, p in zip(GRAZING_NORMAL, point, strict=True)) < GRAZING_OFFSET
            )
            if faces_camera and camera_side:
                points.append(point)
    return points


def seed_angle_degrees(points: list[Point]) -> float:
    """How far the cloud's smallest covariance direction is from the bed's normal."""
    import numpy as np

    _, vectors = np.linalg.eigh(np.cov(np.asarray(points).T))
    cosine = abs(float(np.dot(vectors[:, 0], GRAZING_NORMAL)))
    return math.degrees(math.acos(min(cosine, 1.0)))


@pytest.mark.parametrize(
    ("ball_steps", "expected_applied"),
    [(16, False), (20, True), (24, True), (28, True), (32, True), (36, True), (40, True)],
)
def test_a_grazing_bed_is_found_however_far_off_the_covariance_seed_points(
    ball_steps: int, expected_applied: bool
):
    """A curved object must not capture the refinement.

    The seed is the cloud's smallest covariance direction, and on every scene
    here it is more than ten degrees off the bed's normal. A window one applied
    band wide along such a seed catches a ribbon of the bed at best, while some
    patch of the ball always lies square to the seed and is caught whole - and
    a patch square to the seed refits to the seed, so the version-1 loop
    settled on the ball and reported too few inliers, for some ball densities
    and not others. The search now starts wide and narrows, placing the window
    by density at each width, and the bed is found at every density down to a
    bed that is half the cloud: the same 1320 points, to a twentieth of a
    degree. (At 40% of the cloud it is not: the deflation rounds do not
    recover it either, and that limit is recorded here rather than claimed
    away.)

    The gates then judge what was found. At the sparsest ball the bed is more
    than the removal bound of the cloud, and the frame is rightly kept whole.
    """
    bed = grazing_bed()
    ball = resting_ball(ball_steps)
    scene = bed + ball
    assert seed_angle_degrees(scene) > 10.0

    result = suppress_support_plane(scene)

    assert result.plane is not None
    cosine = abs(sum(a * b for a, b in zip(result.plane.normal, GRAZING_NORMAL, strict=True)))
    assert math.degrees(math.acos(min(cosine, 1.0))) < 0.05
    assert abs(abs(result.plane.offset) - GRAZING_OFFSET) < 0.05
    assert result.inlier_points == len(bed)
    assert result.applied is expected_applied
    if expected_applied:
        assert as_multiset(result.removed) == as_multiset(tuple(bed))
        assert as_multiset(result.points) == as_multiset(tuple(ball))
    search = result.as_dict()["search"]
    assert isinstance(search, dict)
    assert search["coarse_band_fraction"] == COARSE_BAND_FRACTION
