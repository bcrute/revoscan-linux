"""Lock down the rigid-body primitives that registration, fusion and meshing share.

Nothing here touches hardware, opens a device node or reads a device. The module
under test is closed-form arithmetic, so every input is built by the test and
every expected value is either derived by hand or predicted from theory before
the code is asked for it.

Four claims carry most of the weight, and none of them can be settled by reading
:mod:`scanner3d.geometry`:

* **A reflection is refused.** This is the highest-value assertion in the file.
  A determinant ``-1`` matrix composes, inverts and applies exactly like a
  rotation; it even reports a ``rotation_angle`` of zero for the one built here.
  A mirror-image scan is not detectable by eye, so the refusal - and the fact
  that the message says *why* it matters - is the only thing standing between a
  transposed sign somewhere upstream and a silently wrong reconstruction.
* **Composition order is what the docstring says.** Every check uses two
  rotations that do not commute, and additionally asserts how far apart the two
  orders land, so an implementation that swapped them cannot pass by accident.
* **``rotation_angle`` still resolves near the identity.** The module uses
  ``atan2`` rather than ``acos`` specifically so a well-closed turntable loop
  reports its real error instead of a numerical noise floor. The loop-closure
  test computes the ``acos`` form alongside and asserts it is worse by more than
  five orders of magnitude, so the test is demonstrably able to catch a
  regression to it rather than merely hoping to.
* **A scale error cannot be fitted away.** Horn's method is rigid-only, and the
  residual it leaves on a scaled point set is compared against the value theory
  predicts exactly. A scale quietly absorbed would be a calibration fault turned
  invisible, which is the failure mode this repository can least afford.

Two behaviours are recorded rather than endorsed - a documented refusal that
does not happen, and a constructor that does not validate. They are marked as
such where they appear, and are reported upstream rather than fixed here.

No assertion in this file concerns a physical unit, and none can: a translation
of ``1.0`` is one unit of whatever the depth scale produced, and that scalar's
physical meaning remains unestablished on this device. Angles are the one
exception, and they are radians because the module fixes them so.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Iterator, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import (
    IDENTITY_MATRIX,
    MIN_VECTOR_NORM,
    ORTHONORMAL_TOLERANCE,
    SCHEMA_VERSION,
    CorrespondenceUnusable,
    DegenerateVector,
    GeometryNotAllowed,
    Matrix3,
    NonFiniteValue,
    NotARotation,
    Point,
    Rigid,
    add,
    alignment_rmse,
    centroid,
    cross,
    determinant,
    distance,
    dot,
    ensure_finite,
    ensure_rotation,
    norm,
    normalize,
    pose_difference,
    rigid_from_correspondences,
    rotation_about_axis,
    scaled,
    spanned_dimensions,
    subtract,
)

# --- inputs the tests build ---------------------------------------------------

#: A spin axis chosen to be aligned with nothing: not a coordinate axis, not a
#: unit vector, and with a negative component. An axis-aligned test axis hides
#: transposition and sign defects, because so many of the matrix entries are
#: zero that a wrong one lands on a zero anyway.
AXIS: Point = (0.3, -0.9, 0.2)

#: A second axis, non-parallel to the first, for the tests that need two
#: rotations that genuinely do not commute.
OTHER_AXIS: Point = (-0.7, 0.15, 0.65)

#: A turntable spindle nowhere near the camera origin. The whole point of the
#: ``through`` argument is that a real turntable does not rotate about the origin
#: of anyone's frame, and a spindle at a few hundred units out is what makes an
#: "I forgot to translate back" defect show up as a gross error rather than a
#: rounding one.
SPINDLE: Point = (137.0, -89.0, 41.0)

#: A full revolution in this many equal steps. Chosen to match the loop-closure
#: figure the module's ``rotation_angle`` docstring is written to defend.
TURNTABLE_STEPS = 36

#: Nine points that are genuinely three-dimensional, asymmetric, and share no
#: plane, so a fit against them constrains all three rotational degrees of
#: freedom. A symmetric or planar set would let a wrong rotation fit as well as
#: the right one.
CORRESPONDENCE_POINTS: tuple[Point, ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 3.0),
    (-1.5, 0.5, 1.0),
    (2.0, -1.0, 0.5),
    (0.25, 1.75, -2.0),
    (4.0, 4.0, -4.0),
    (-3.0, 2.5, 6.0),
    (0.5, -5.0, 1.5),
)

#: A Householder reflection through the plane normal to ``(1, 1, 1)``. Its
#: columns are orthonormal to the last bit, its determinant is exactly -1, and
#: every entry on its diagonal is positive - so nothing about its *shape* looks
#: wrong, and a check that merely sniffed at diagonal signs would wave it
#: through. Its trace is 1, which means the honest geodesic angle metric reports
#: it as a zero-magnitude transform: this matrix is indistinguishable from the
#: identity by every scalar the module publishes.
_THIRD = 1.0 / 3.0
PLANE_REFLECTION: Matrix3 = (
    (_THIRD, -2.0 * _THIRD, -2.0 * _THIRD),
    (-2.0 * _THIRD, _THIRD, -2.0 * _THIRD),
    (-2.0 * _THIRD, -2.0 * _THIRD, _THIRD),
)

#: Words no assertion, message or serialised field in this module may contain.
#: The depth scale's physical unit is unestablished on this device, so a
#: transform built from it measures nothing nameable.
#:
#: The bare stems ``metre``/``meter`` cannot be listed on their own, because
#: "parameter", "diameter" and "perimeter" all contain them and are legitimate
#: prose. So each is listed twice: once preceded by a space, which catches the
#: prose form, and once preceded by an underscore, which catches the far more
#: likely form - a serialised field named ``translation_metres``. Listing only
#: the spaced form left exactly that hole, and a field name is the shape this
#: violation would actually take.
FORBIDDEN_UNIT_WORDS = (
    "millimetre",
    "millimeter",
    "centimetre",
    "centimeter",
    "micrometre",
    "micrometer",
    "micron",
    " metre",
    " meter",
    "_metre",
    "_meter",
    "inch",
    "_mm",
    "mm_",
    "_cm",
    "cm_",
)


def rotation_of(axis: Point, angle: float) -> Matrix3:
    """The rotation part alone, for tests that need a matrix rather than a pose."""
    return rotation_about_axis(axis, angle).rotation


def mirrored(rotation: Matrix3) -> Matrix3:
    """``rotation`` with its third column negated: still orthonormal, determinant -1.

    A reflection built this way sits arbitrarily close to a legitimate rotation -
    it is one, composed with a mirror - which is exactly the shape the defect
    takes when a sign is transposed upstream rather than invented from nothing.
    """
    return (
        (rotation[0][0], rotation[0][1], -rotation[0][2]),
        (rotation[1][0], rotation[1][1], -rotation[1][2]),
        (rotation[2][0], rotation[2][1], -rotation[2][2]),
    )


def acos_rotation_angle(transform: Rigid) -> float:
    """The rotation magnitude the naive way, kept here purely as the thing to beat.

    ``acos((trace - 1) / 2)`` is the obvious formula and the one
    :attr:`Rigid.rotation_angle` deliberately does not use. Computing it beside
    the real answer is what lets a test *demonstrate* that it would catch a
    regression, instead of asserting a tight bound and trusting that some other
    implementation would have missed it.
    """
    r = transform.rotation
    trace = math.fsum((r[0][0], r[1][1], r[2][2]))
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def compose_n(step: Rigid, times: int) -> Rigid:
    """``step`` applied ``times`` over, accumulated the way a sweep accumulates."""
    accumulated = Rigid.identity()
    for _ in range(times):
        accumulated = step.compose(accumulated)
    return accumulated


def rms_radius(points: Sequence[Point]) -> float:
    """Root-mean-square distance from the set's own centroid."""
    centre = centroid(points)
    return math.sqrt(math.fsum(distance(p, centre) ** 2 for p in points) / len(points))


def transformed(transform: Rigid, points: Sequence[Point]) -> tuple[Point, ...]:
    return tuple(transform.apply(point) for point in points)


def covariance_condition(points: Sequence[Point]) -> float:
    """Largest over smallest eigenvalue of the centred covariance.

    Solved from the characteristic cubic in closed form (Smith's trigonometric
    method) rather than by calling into :mod:`scanner3d.geometry`, so a claim
    about how well conditioned a point set is does not rest on the same
    eigensolver the module under test uses.

    That closed form loses about half its significant digits when two eigenvalues
    nearly coincide, so it is fit for comparing two well-separated, full-rank
    sets - which is all it is used for here - and is *not* a substitute for the
    module's Jacobi routine when the question is degeneracy. That difference is
    the reason :func:`scanner3d.geometry.spanned_dimensions` uses Jacobi.
    """
    centre = centroid(points)
    offsets = [subtract(point, centre) for point in points]
    a = [[math.fsum(o[i] * o[j] for o in offsets) for j in range(3)] for i in range(3)]
    p1 = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
    q = (a[0][0] + a[1][1] + a[2][2]) / 3.0
    if p1 == 0.0:
        diagonal = sorted((a[0][0], a[1][1], a[2][2]), reverse=True)
        return diagonal[0] / diagonal[2]
    p2 = (a[0][0] - q) ** 2 + (a[1][1] - q) ** 2 + (a[2][2] - q) ** 2 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    b = [[(a[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
    half_det = determinant((tuple(b[0]), tuple(b[1]), tuple(b[2]))) / 2.0  # type: ignore[arg-type]
    phi = math.acos(max(-1.0, min(1.0, half_det))) / 3.0
    first = q + 2.0 * p * math.cos(phi)
    third = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    second = 3.0 * q - first - third
    ordered = sorted((first, second, third), reverse=True)
    return ordered[0] / ordered[2]


# --- vector arithmetic and the refusals around it -----------------------------


def test_vector_arithmetic_matches_values_worked_by_hand():
    a: Point = (1.0, -2.0, 3.0)
    b: Point = (-4.0, 0.5, 2.0)
    assert add(a, b) == (-3.0, -1.5, 5.0)
    assert subtract(a, b) == (5.0, -2.5, 1.0)
    assert scaled(a, -2.0) == (-2.0, 4.0, -6.0)
    assert dot(a, b) == -4.0 - 1.0 + 6.0
    assert cross(a, b) == (-2.0 * 2.0 - 3.0 * 0.5, 3.0 * -4.0 - 1.0 * 2.0, 1.0 * 0.5 - -2.0 * -4.0)
    assert norm((3.0, 4.0, 0.0)) == 5.0
    assert distance((1.0, 2.0, 3.0), (1.0, 6.0, 6.0)) == 5.0


def test_cross_is_perpendicular_and_anticommutes():
    """The two properties every downstream axis construction leans on."""
    a: Point = (1.0, -2.0, 3.0)
    b: Point = (-4.0, 0.5, 2.0)
    perpendicular = cross(a, b)
    assert abs(dot(perpendicular, a)) < 1e-12
    assert abs(dot(perpendicular, b)) < 1e-12
    assert cross(b, a) == scaled(perpendicular, -1.0)


def test_normalize_returns_a_unit_vector_along_the_original_direction():
    unit = normalize(AXIS)
    assert math.isclose(norm(unit), 1.0, rel_tol=0.0, abs_tol=1e-15)
    # Same direction, not merely the same line: the cross product vanishes and
    # the dot product is positive.
    assert norm(cross(unit, AXIS)) < 1e-15
    assert dot(unit, AXIS) > 0.0


def test_normalize_refuses_a_vector_with_no_reliable_direction():
    """Below the floor there is no direction, only amplified rounding noise."""
    with pytest.raises(DegenerateVector) as caught:
        normalize((0.0, 0.0, 0.0))
    assert "no reliable direction" in str(caught.value)
    assert repr(MIN_VECTOR_NORM) in str(caught.value)


def test_the_degenerate_vector_floor_sits_exactly_where_it_is_documented():
    """A tolerance nobody has pinned down drifts. This pins it down."""
    assert MIN_VECTOR_NORM == 1e-12
    assert norm(normalize((MIN_VECTOR_NORM, 0.0, 0.0))) == pytest.approx(1.0)
    with pytest.raises(DegenerateVector):
        normalize((MIN_VECTOR_NORM / 2.0, 0.0, 0.0))


def test_normalize_refuses_a_non_finite_vector_before_measuring_it():
    with pytest.raises(NonFiniteValue) as caught:
        normalize((1.0, float("nan"), 3.0))
    assert "entry 1" in str(caught.value)


def test_ensure_finite_names_the_offending_entry_and_the_thing_it_belongs_to():
    ensure_finite((1.0, -2.0, 3.0), "translation")  # no exception
    with pytest.raises(NonFiniteValue) as caught:
        ensure_finite((0.0, 0.0, float("inf")), "translation")
    message = str(caught.value)
    assert "translation" in message
    assert "entry 2" in message
    assert "inf" in message


def test_centroid_is_the_arithmetic_mean_and_an_empty_set_has_none():
    assert centroid(((1.0, 2.0, 3.0), (3.0, 4.0, 9.0))) == (2.0, 3.0, 6.0)
    with pytest.raises(CorrespondenceUnusable) as caught:
        centroid(())
    assert "undefined" in str(caught.value)


def test_centroid_of_a_translated_set_moves_by_exactly_that_translation():
    offset: Point = (100.0, -250.0, 7.5)
    moved = [add(point, offset) for point in CORRESPONDENCE_POINTS]
    before = centroid(CORRESPONDENCE_POINTS)
    after = centroid(moved)
    assert distance(subtract(after, before), offset) < 1e-12


# --- the reflection gate: the highest-value refusal in the module -------------


def test_of_refuses_a_reflection_and_says_what_it_would_have_produced():
    """The one that matters most.

    A determinant ``-1`` matrix is orthonormal. It composes, inverts and applies
    with no numerical complaint whatsoever, and the scan it produces is a mirror
    image that looks entirely reasonable until something is measured against a
    real object. Refusing it is necessary; saying *why* in the message is what
    stops the next reader from loosening the check to make a test pass.
    """
    with pytest.raises(NotARotation) as caught:
        Rigid.of(PLANE_REFLECTION, (0.0, 0.0, 0.0))
    message = str(caught.value)
    assert "reflection" in message
    assert "mirror-image" in message
    assert "not detectable by eye" in message
    assert "determinant" in message


def test_the_refused_reflection_really_is_orthonormal_and_otherwise_plausible():
    """Proof the gate is doing real work rather than tripping on a malformed input.

    Every column of :data:`PLANE_REFLECTION` is a unit vector, every pair is
    orthogonal, and every diagonal entry is positive. Its determinant is the only
    thing wrong with it, which is precisely why a determinant check is the only
    thing that can catch it.
    """
    for i in range(3):
        column = (PLANE_REFLECTION[0][i], PLANE_REFLECTION[1][i], PLANE_REFLECTION[2][i])
        assert math.isclose(norm(column), 1.0, rel_tol=0.0, abs_tol=1e-15)
        assert PLANE_REFLECTION[i][i] > 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            a = (PLANE_REFLECTION[0][i], PLANE_REFLECTION[1][i], PLANE_REFLECTION[2][i])
            b = (PLANE_REFLECTION[0][j], PLANE_REFLECTION[1][j], PLANE_REFLECTION[2][j])
            assert abs(dot(a, b)) < 1e-15
    assert math.isclose(determinant(PLANE_REFLECTION), -1.0, rel_tol=0.0, abs_tol=1e-14)


def test_no_scalar_the_module_publishes_can_distinguish_that_reflection():
    """Why the refusal has to happen at construction and cannot be deferred.

    The reflection's trace is 1, so the geodesic angle - the honest scalar this
    module offers for "how much transform is this" - reports zero, the same value
    it reports for the identity. Its translation magnitude is zero too. Nothing
    downstream could notice, which is the whole argument for refusing it at the
    door.
    """
    smuggled = Rigid(rotation=PLANE_REFLECTION, translation=(0.0, 0.0, 0.0))
    assert smuggled.rotation_angle == 0.0
    assert smuggled.translation_magnitude == 0.0
    assert Rigid.identity().rotation_angle == 0.0
    # And it is a genuine mirror: a chirality-sensitive triple flips sign.
    a: Point = (1.0, 0.0, 0.0)
    b: Point = (0.0, 1.0, 0.0)
    c: Point = (0.0, 0.0, 1.0)
    handed = dot(cross(a, b), c)
    flipped = dot(
        cross(smuggled.apply(a), smuggled.apply(b)),
        smuggled.apply(c),
    )
    assert handed > 0.0
    assert flipped < 0.0


def test_of_refuses_a_reflection_that_is_a_real_rotation_with_one_column_mirrored():
    """The realistic shape of the defect: a correct rotation with a sign transposed."""
    with pytest.raises(NotARotation) as caught:
        Rigid.of(mirrored(rotation_of(AXIS, 0.6)), (1.0, 2.0, 3.0))
    assert "mirror-image" in str(caught.value)


# --- the rest of the rotation validation --------------------------------------


def test_of_refuses_a_uniform_scale_because_it_is_not_orthonormal():
    doubled: Matrix3 = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    with pytest.raises(NotARotation) as caught:
        Rigid.of(doubled, (0.0, 0.0, 0.0))
    message = str(caught.value)
    assert "not orthonormal" in message
    assert "scales or skews" in message


def test_of_refuses_a_skew_whose_columns_are_unit_length_but_not_perpendicular():
    """Guards the off-diagonal half of the orthonormality check specifically.

    Both columns here have norm 1, so a check that only verified column lengths
    would accept this matrix. It is caught by the pair test, and the message says
    which pair, which is what distinguishes this failure from a determinant one.
    """
    angle = 0.1
    skewed: Matrix3 = (
        (1.0, math.sin(angle), 0.0),
        (0.0, math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    with pytest.raises(NotARotation) as caught:
        Rigid.of(skewed, (0.0, 0.0, 0.0))
    message = str(caught.value)
    assert "columns 0 and 1" in message
    assert "not orthonormal" in message


def test_the_determinant_check_is_a_second_independent_gate():
    """A scale small enough to slip past the column check is still refused.

    Column dot products drift as the square of a uniform scale and the
    determinant as its cube, so there is a narrow band where the first check
    passes and the second does not. Landing a test in that band is what proves
    the determinant check is not merely redundant with the orthonormality one.
    """
    barely: Matrix3 = ((1.0 + 4.9e-7, 0.0, 0.0), (0.0, 1.0 + 4.9e-7, 0.0), (0.0, 0.0, 1.0 + 4.9e-7))
    with pytest.raises(NotARotation) as caught:
        Rigid.of(barely, (0.0, 0.0, 0.0))
    message = str(caught.value)
    assert "determinant" in message
    assert "not orthonormal" not in message
    assert "reflection" not in message


def test_the_orthonormal_tolerance_sits_exactly_where_it_is_documented():
    """The tolerance is deliberate - it has to survive a composed sweep - so pin it.

    A drift small enough to be the residue of composing a few hundred transforms
    is accepted on purpose. Recording that as a test is what keeps a later reader
    from either tightening it until a legitimate sweep is refused, or loosening it
    until a real defect fits through.
    """
    assert ORTHONORMAL_TOLERANCE == 1e-6
    within: Matrix3 = ((1.0 + 3e-7, 0.0, 0.0), (0.0, 1.0 + 3e-7, 0.0), (0.0, 0.0, 1.0 + 3e-7))
    assert Rigid.of(within, (0.0, 0.0, 0.0)).rotation is within


@pytest.mark.parametrize("through", [(0.0, 0.0, 0.0), (5.0, -10.0, 200.0)])
def test_stationary_coordinate_roundtrips_do_not_amplify_rotation_roundoff(through):
    original = rotation_about_axis((0.1, 0.3, 0.7), 0.2, through=through)
    pose = original
    probe = (10.0, -20.0, 150.0)
    for _ in range(1000):
        pose = pose.compose(pose.inverse().compose(pose))
        r = pose.rotation
        defect = max(
            abs(math.fsum(r[k][a] * r[k][b] for k in range(3)) - float(a == b))
            for a in range(3)
            for b in range(3)
        )
        assert defect <= 1e-12
        assert distance(pose.apply(probe), original.apply(probe)) < 1e-7
        assert determinant(r) > 0.0


@pytest.mark.parametrize("scale", [1.01, -1.0, 1.0 + 1e-6])
def test_roundoff_repair_does_not_admit_scale_or_reflection(scale):
    matrix: Matrix3 = ((scale, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(NotARotation):
        Rigid.of(matrix, (0.0, 0.0, 0.0), repair_roundoff=True)


def test_roundoff_repair_is_closed_to_tiny_defects_and_leaves_translation_unchanged():
    shift = (1.0, 2.0, 3.0)
    tiny: Matrix3 = ((1.0 + 1e-11, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    fixed = Rigid.of(tiny, shift, repair_roundoff=True)
    assert fixed.rotation != tiny
    assert fixed.rotation == IDENTITY_MATRIX
    assert fixed.translation is shift
    assert Rigid.of(tiny, shift).rotation is tiny
    larger: Matrix3 = ((1.0 + 1e-8, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert Rigid.of(larger, shift, repair_roundoff=True).rotation is larger


def test_of_refuses_a_non_finite_rotation_entry_naming_its_position():
    broken: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, float("inf")), (0.0, 0.0, 1.0))
    with pytest.raises(NonFiniteValue) as caught:
        Rigid.of(broken, (0.0, 0.0, 0.0))
    message = str(caught.value)
    assert "rotation matrix" in message
    # Flattened row-major, so entry 5 is row 1, column 2.
    assert "entry 5" in message


def test_of_refuses_a_non_finite_translation():
    with pytest.raises(NonFiniteValue) as caught:
        Rigid.of(IDENTITY_MATRIX, (0.0, float("nan"), 0.0))
    assert "translation" in str(caught.value)
    assert "entry 1" in str(caught.value)


def test_the_translation_is_checked_before_the_rotation_is():
    """Refusal order is part of the contract, not an accident of line ordering.

    Both arguments are bad here. Which error comes out says which check ran
    first, and a caller that branches on the exception type is entitled to a
    stable answer.
    """
    with pytest.raises(NonFiniteValue):
        Rigid.of(PLANE_REFLECTION, (float("nan"), 0.0, 0.0))


def test_ensure_rotation_returns_the_very_matrix_it_validated():
    """No normalisation, no repair, no copy: what went in is what comes out."""
    rotation = rotation_of(AXIS, 0.4)
    assert ensure_rotation(rotation) is rotation


def test_determinant_matches_a_value_computed_by_hand():
    assert determinant(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0))) == -3.0
    assert determinant(IDENTITY_MATRIX) == 1.0


def test_every_geometry_refusal_is_a_scanner_error_the_doctor_can_classify():
    for error in (NonFiniteValue, DegenerateVector, NotARotation, CorrespondenceUnusable):
        assert issubclass(error, GeometryNotAllowed)
        assert issubclass(error, ScannerError)
        assert error.status is DoctorStatus.UNSUPPORTED_MODE


# --- applying a transform -----------------------------------------------------


def test_identity_leaves_every_point_exactly_where_it_was():
    identity = Rigid.identity()
    assert identity.rotation == IDENTITY_MATRIX
    assert identity.translation == (0.0, 0.0, 0.0)
    for point in CORRESPONDENCE_POINTS:
        assert identity.apply(point) == point


def test_apply_is_rotation_then_translation_not_the_other_way_round():
    """``x' = R @ x + t``. Translating first would give ``R @ (x + t)``.

    The two differ for any non-identity rotation and non-zero translation, so a
    single worked example settles which convention is implemented.
    """
    rotation = rotation_of((0.0, 0.0, 1.0), math.pi / 2.0)
    transform = Rigid.of(rotation, (10.0, 0.0, 0.0))
    # A quarter turn about z carries (1, 0, 0) to (0, 1, 0); then the offset.
    assert distance(transform.apply((1.0, 0.0, 0.0)), (10.0, 1.0, 0.0)) < 1e-15
    # Translate-then-rotate would have put it at (0, 11, 0).
    assert distance(transform.apply((1.0, 0.0, 0.0)), (0.0, 11.0, 0.0)) > 1.0


def test_a_rigid_transform_preserves_every_distance_and_every_angle():
    """The defining property. If this fails, the word "rigid" is a lie."""
    transform = Rigid.of(rotation_of(AXIS, 1.3), (12.5, -3.25, 7.75))
    moved = transformed(transform, CORRESPONDENCE_POINTS)
    for i in range(len(CORRESPONDENCE_POINTS)):
        for j in range(i + 1, len(CORRESPONDENCE_POINTS)):
            before = distance(CORRESPONDENCE_POINTS[i], CORRESPONDENCE_POINTS[j])
            after = distance(moved[i], moved[j])
            assert math.isclose(before, after, rel_tol=1e-14)


def test_apply_all_streams_lazily_and_agrees_with_apply_point_by_point():
    """A fused cloud is too large to double in memory, so this must stay a stream."""
    transform = Rigid.of(rotation_of(AXIS, 0.5), (1.0, 2.0, 3.0))
    pulled: list[Point] = []

    def source() -> Iterator[Point]:
        for point in CORRESPONDENCE_POINTS:
            pulled.append(point)
            yield point

    stream = transform.apply_all(source())
    assert iter(stream) is stream
    assert pulled == []  # nothing consumed before the first pull
    first = next(stream)
    assert len(pulled) == 1
    assert first == transform.apply(CORRESPONDENCE_POINTS[0])
    assert list(stream) == [transform.apply(p) for p in CORRESPONDENCE_POINTS[1:]]


# --- composition --------------------------------------------------------------


def test_compose_applies_self_after_inner_exactly_as_documented():
    """Two rotations that do not commute, so a swapped implementation cannot pass.

    The second assertion is the one with teeth: it measures how far apart the two
    composition orders actually land. Without it, a test that only checked one
    order would pass against an implementation that had them backwards and
    happened to be checked with commuting inputs.
    """
    outer = rotation_about_axis((0.0, 0.0, 1.0), 0.7)
    inner = rotation_about_axis((1.0, 0.0, 0.0), 1.1)
    point: Point = (0.3, -1.7, 2.9)

    assert distance(outer.compose(inner).apply(point), outer.apply(inner.apply(point))) < 1e-15
    assert distance(outer.compose(inner).apply(point), inner.compose(outer).apply(point)) > 1.0


def test_compose_carries_the_inner_translation_through_the_outer_rotation():
    """The half of composition that a rotation-only test would never reach."""
    outer = Rigid.of(rotation_of((0.0, 0.0, 1.0), math.pi / 2.0), (0.0, 0.0, 5.0))
    inner = Rigid.of(IDENTITY_MATRIX, (2.0, 0.0, 0.0))
    combined = outer.compose(inner)
    # The inner offset (2, 0, 0) is rotated a quarter turn to (0, 2, 0), then
    # the outer offset is added. Merely summing the translations would give
    # (2, 0, 5).
    assert distance(combined.translation, (0.0, 2.0, 5.0)) < 1e-15


def test_composition_is_associative_over_three_non_commuting_transforms():
    a = Rigid.of(rotation_of(AXIS, 0.9), (1.0, -2.0, 3.0))
    b = Rigid.of(rotation_of(OTHER_AXIS, 1.4), (-4.0, 0.5, 2.0))
    c = Rigid.of(rotation_of((1.0, 1.0, 0.0), 0.3), (0.25, 0.0, -1.5))
    point: Point = (2.5, -3.5, 4.5)
    left = a.compose(b).compose(c)
    right = a.compose(b.compose(c))
    assert distance(left.apply(point), right.apply(point)) < 1e-13
    assert pose_difference(left, right).rotation_angle < 1e-14


def test_composing_with_the_identity_changes_nothing_on_either_side():
    transform = Rigid.of(rotation_of(AXIS, 0.8), (3.0, -1.0, 0.5))
    identity = Rigid.identity()
    for combined in (transform.compose(identity), identity.compose(transform)):
        assert distance(combined.translation, transform.translation) < 1e-15
        assert pose_difference(transform, combined).rotation_angle < 1e-15


def test_compose_revalidates_so_a_composed_transform_is_still_a_rotation():
    """Every product goes back through the gate; drift cannot accumulate unnoticed."""
    accumulated = compose_n(rotation_about_axis(AXIS, 0.37), 200)
    assert ensure_rotation(accumulated.rotation) is accumulated.rotation


# --- inversion ----------------------------------------------------------------


def test_inverse_round_trips_a_point_through_a_transform_with_translation():
    transform = Rigid.of(rotation_of(AXIS, 1.3), (12.5, -3.25, 7.75))
    inverse = transform.inverse()
    for point in CORRESPONDENCE_POINTS:
        assert distance(inverse.apply(transform.apply(point)), point) < 1e-13
        assert distance(transform.apply(inverse.apply(point)), point) < 1e-13


def test_the_inverse_translation_is_not_simply_the_negated_translation():
    """The classic inversion defect, and the reason this test exists at all.

    ``(R, t)`` inverts to ``(R^T, -R^T t)``, not ``(R^T, -t)``. The wrong form
    round-trips perfectly whenever the rotation is the identity, so a test that
    only ever inverts a pure translation cannot see the bug.
    """
    transform = Rigid.of(rotation_of(AXIS, 1.3), (12.5, -3.25, 7.75))
    inverse = transform.inverse()
    naive: Point = scaled(transform.translation, -1.0)
    assert distance(inverse.translation, naive) > 1.0
    assert math.isclose(norm(inverse.translation), transform.translation_magnitude, rel_tol=1e-14)


def test_a_transform_composed_with_its_inverse_is_the_identity_both_ways():
    transform = Rigid.of(rotation_of(OTHER_AXIS, 2.4), (-40.0, 15.5, 3.125))
    inverse = transform.inverse()
    for combined in (transform.compose(inverse), inverse.compose(transform)):
        assert combined.rotation_angle < 1e-15
        assert combined.translation_magnitude < 1e-13


def test_inverting_twice_returns_the_original_transform():
    transform = Rigid.of(rotation_of(AXIS, 0.65), (7.0, -8.0, 9.0))
    back = transform.inverse().inverse()
    assert pose_difference(transform, back).rotation_angle < 1e-15
    assert distance(back.translation, transform.translation) < 1e-13


def test_the_inverse_rotation_is_the_transpose():
    transform = Rigid.of(rotation_of(AXIS, 0.65), (7.0, -8.0, 9.0))
    inverse = transform.inverse()
    for i in range(3):
        for j in range(3):
            assert inverse.rotation[i][j] == transform.rotation[j][i]


# --- rotation about a line that is not through the origin ---------------------


def test_points_on_a_far_off_spindle_are_left_exactly_where_they_are():
    """The turntable invariant, tested where a forgotten translate-back would show.

    The spindle sits a few hundred units from the origin, and the sampled points
    run a thousand units along it in both directions. Any defect in the
    translate-rotate-translate-back sandwich moves them by a distance of the same
    order as the spindle's offset, not by a rounding error.
    """
    spin = rotation_about_axis(AXIS, 0.77, through=SPINDLE)
    unit = normalize(AXIS)
    for along in (-1000.0, -1.0, 0.0, 1.0, 1000.0):
        on_axis = add(SPINDLE, scaled(unit, along))
        assert distance(spin.apply(on_axis), on_axis) < 1e-9


def test_a_point_off_the_spindle_moves_and_keeps_its_distance_from_the_line():
    """Rotation about a line is a rotation *about that line*, not merely near it."""
    spin = rotation_about_axis(AXIS, 0.77, through=SPINDLE)
    unit = normalize(AXIS)
    off_axis: Point = add(SPINDLE, (5.0, 0.0, 0.0))
    moved = spin.apply(off_axis)
    assert distance(moved, off_axis) > 1.0

    def radius(point: Point) -> float:
        offset = subtract(point, SPINDLE)
        return norm(subtract(offset, scaled(unit, dot(offset, unit))))

    assert math.isclose(radius(moved), radius(off_axis), rel_tol=1e-12)


def test_rotating_about_the_origin_is_the_default_and_leaves_the_origin_fixed():
    spin = rotation_about_axis(AXIS, 0.77)
    assert spin.translation == (0.0, 0.0, 0.0)
    assert spin.apply((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_a_far_off_spindle_produces_a_large_translation_even_though_it_is_a_pure_spin():
    """Why ``through`` exists: the equivalent origin-centred pose is not obvious.

    Expressing the same motion as a rotation about the origin requires a
    translation of the same order as the spindle offset. A caller composing that
    by hand is exactly the caller who gets the order wrong.
    """
    spin = rotation_about_axis(AXIS, 0.77, through=SPINDLE)
    assert spin.translation_magnitude > 10.0
    assert math.isclose(spin.rotation_angle, 0.77, rel_tol=1e-14)


def test_a_full_revolution_in_equal_steps_returns_to_the_identity():
    for through in ((0.0, 0.0, 0.0), SPINDLE):
        step = rotation_about_axis(AXIS, 2.0 * math.pi / TURNTABLE_STEPS, through=through)
        closed = compose_n(step, TURNTABLE_STEPS)
        assert closed.rotation_angle < 1e-12
        assert closed.translation_magnitude < 1e-9
        for point in CORRESPONDENCE_POINTS:
            assert distance(closed.apply(point), point) < 1e-9


def test_a_rotation_and_its_negation_about_the_same_axis_cancel():
    forward = rotation_about_axis(AXIS, 1.1, through=SPINDLE)
    backward = rotation_about_axis(AXIS, -1.1, through=SPINDLE)
    combined = forward.compose(backward)
    assert combined.rotation_angle < 1e-15
    assert combined.translation_magnitude < 1e-12


def test_rotation_about_axis_follows_the_right_hand_rule():
    """Handedness is a convention, and an unrecorded convention is a coin flip."""
    quarter = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2.0)
    assert distance(quarter.apply((1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)) < 1e-15
    assert distance(quarter.apply((0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0)) < 1e-15


def test_the_axis_length_does_not_matter_only_its_direction():
    small = rotation_about_axis(scaled(AXIS, 1e-6), 0.9)
    large = rotation_about_axis(scaled(AXIS, 1e6), 0.9)
    assert pose_difference(small, large).rotation_angle < 1e-14


def test_rotation_about_axis_refuses_an_axis_with_no_direction():
    with pytest.raises(DegenerateVector):
        rotation_about_axis((0.0, 0.0, 0.0), 1.0)


def test_rotation_about_axis_checks_the_angle_before_it_touches_the_axis():
    """Both arguments are unusable; the angle is the one that is reported."""
    with pytest.raises(NonFiniteValue) as caught:
        rotation_about_axis((0.0, 0.0, 0.0), float("nan"))
    assert "angle" in str(caught.value)


def test_rotation_about_axis_refuses_a_non_finite_anchor_point():
    with pytest.raises(NonFiniteValue) as caught:
        rotation_about_axis(AXIS, 1.0, through=(float("inf"), 0.0, 0.0))
    assert "axis anchor point" in str(caught.value)


# --- the rotation magnitude, which is the loop-closure metric -----------------


@pytest.mark.parametrize("degrees", [1e-6, 0.001, 1.0, 45.0, 90.0, 179.999])
def test_rotation_angle_recovers_the_angle_it_was_built_from(degrees: float):
    """Across nine orders of magnitude, because the metric is used across them.

    A closure error is fractions of a microradian when a sweep is good and most
    of a radian when it is not, and one scalar has to be trustworthy over that
    whole span.
    """
    expected = math.radians(degrees)
    assert rotation_about_axis(AXIS, expected).rotation_angle == pytest.approx(expected, rel=1e-12)


def test_rotation_angle_stays_within_zero_and_pi_and_reports_the_short_way_round():
    """It is a geodesic magnitude, so a rotation past ``pi`` comes back the other way."""
    for angle in (0.0, 0.5, math.pi / 2.0, math.pi, 3.5, 2.0 * math.pi - 0.25):
        reported = rotation_about_axis(AXIS, angle).rotation_angle
        assert 0.0 <= reported <= math.pi + 1e-15
    assert rotation_about_axis(AXIS, math.pi).rotation_angle == pytest.approx(math.pi, rel=1e-14)
    assert rotation_about_axis(AXIS, 3.5).rotation_angle == pytest.approx(
        2.0 * math.pi - 3.5, rel=1e-14
    )
    assert rotation_about_axis(AXIS, -0.9).rotation_angle == pytest.approx(0.9, rel=1e-14)


def test_a_thirty_six_step_revolution_closes_to_better_than_a_picoradian():
    """The headline loop-closure figure, and the reason ``acos`` is not used.

    Thirty-six equal steps of a full turn are composed and the residual measured
    against the identity. The ``atan2`` form leaves a residual at the float64
    floor; ``acos`` is ill-conditioned near the identity and cannot report better
    than about ``6e-8`` no matter how well the loop actually closed - a metric
    that stops resolving exactly where a good answer lives.

    The second assertion is what makes this test a regression guard rather than a
    tight bound with a hopeful comment: it computes the naive form on the same
    matrix and requires the real one to be better by more than five orders of
    magnitude. An implementation that switched back would fail here on the
    comparison, not merely on the threshold.
    """
    step = rotation_about_axis(AXIS, 2.0 * math.pi / TURNTABLE_STEPS, through=SPINDLE)
    closure = pose_difference(Rigid.identity(), compose_n(step, TURNTABLE_STEPS))

    assert closure.rotation_angle < 1e-12
    assert acos_rotation_angle(closure) > 1e5 * max(closure.rotation_angle, 1e-300)


def test_the_angle_metric_still_resolves_a_rotation_far_below_the_acos_noise_floor():
    """A microdegree of residual rotation is a real quantity, and must survive.

    ``acos`` returns exactly zero here - not an approximation, zero - so a
    regression would report a perfectly closed loop for a loop that is not
    closed. That is the failure this test exists to make impossible.
    """
    tiny = math.radians(1e-6)
    transform = rotation_about_axis(AXIS, tiny)
    assert transform.rotation_angle == pytest.approx(tiny, rel=1e-12)
    assert acos_rotation_angle(transform) == 0.0


def test_translation_magnitude_is_the_length_of_the_offset():
    transform = Rigid.of(IDENTITY_MATRIX, (3.0, 4.0, 12.0))
    assert transform.translation_magnitude == 13.0
    assert Rigid.identity().translation_magnitude == 0.0


# --- best-fit alignment -------------------------------------------------------


def test_a_known_transform_is_recovered_from_its_own_correspondences():
    """Significant translation and a rotation about an axis aligned with nothing.

    The residual is asserted alongside the recovered pose, because a pose without
    its residual is an assertion rather than a measurement - and because a fit
    that recovered the wrong transform would still return *a* transform.
    """
    known = Rigid.of(rotation_of((1.0, -2.0, 3.0), 0.9), (12.5, -3.25, 7.75))
    target = transformed(known, CORRESPONDENCE_POINTS)

    fitted = rigid_from_correspondences(CORRESPONDENCE_POINTS, target)

    assert alignment_rmse(CORRESPONDENCE_POINTS, target, fitted) < 1e-9
    difference = pose_difference(known, fitted)
    assert difference.rotation_angle < 1e-9
    assert difference.translation_magnitude < 1e-9
    for i in range(3):
        for j in range(3):
            assert fitted.rotation[i][j] == pytest.approx(known.rotation[i][j], abs=1e-12)


def test_a_pure_translation_is_recovered_with_no_spurious_rotation():
    offset: Point = (-250.0, 90.5, 3.125)
    target = tuple(add(point, offset) for point in CORRESPONDENCE_POINTS)
    fitted = rigid_from_correspondences(CORRESPONDENCE_POINTS, target)
    assert fitted.rotation_angle < 1e-12
    assert distance(fitted.translation, offset) < 1e-12
    assert alignment_rmse(CORRESPONDENCE_POINTS, target, fitted) < 1e-12


def test_a_pure_rotation_about_a_far_off_spindle_is_recovered():
    """The turntable case: the fit sees points, not the spindle, and must still agree."""
    known = rotation_about_axis(AXIS, 0.42, through=SPINDLE)
    source = tuple(add(SPINDLE, point) for point in CORRESPONDENCE_POINTS)
    target = transformed(known, source)

    fitted = rigid_from_correspondences(source, target)

    assert alignment_rmse(source, target, fitted) < 1e-9
    assert fitted.rotation_angle == pytest.approx(0.42, rel=1e-10)
    assert pose_difference(known, fitted).translation_magnitude < 1e-8


def test_a_scale_error_surfaces_as_a_residual_instead_of_being_fitted_away():
    """A correctness property, not a nicety.

    If two clouds differ in scale that is a calibration fault, and a solver that
    quietly absorbed it would hide the fault behind a clean-looking alignment.
    Horn's method is rigid-only, so the best it can do on a set scaled by ``s``
    about its own centroid is the identity rotation, leaving a residual of
    exactly ``(s - 1)`` times the set's root-mean-square radius.

    That prediction is computed here from the source points alone, before the
    solver is called, and the residual is required to match it - so the test
    fails both if the scale is absorbed (residual collapses) and if the solver
    merely does something else wrong (residual misses the prediction).
    """
    factor = 1.5
    centre = centroid(CORRESPONDENCE_POINTS)
    target = tuple(
        add(centre, scaled(subtract(point, centre), factor)) for point in CORRESPONDENCE_POINTS
    )
    predicted = (factor - 1.0) * rms_radius(CORRESPONDENCE_POINTS)

    fitted = rigid_from_correspondences(CORRESPONDENCE_POINTS, target)
    residual = alignment_rmse(CORRESPONDENCE_POINTS, target, fitted)

    assert residual == pytest.approx(predicted, rel=1e-12)
    assert residual > 1.0
    # The fit is still rigid: every distance among the moved points is the
    # distance it started with, so no scale entered the transform.
    moved = transformed(fitted, CORRESPONDENCE_POINTS)
    for i in range(len(CORRESPONDENCE_POINTS)):
        for j in range(i + 1, len(CORRESPONDENCE_POINTS)):
            assert distance(moved[i], moved[j]) == pytest.approx(
                distance(CORRESPONDENCE_POINTS[i], CORRESPONDENCE_POINTS[j]), rel=1e-14
            )


def test_a_scale_error_hidden_inside_a_genuine_rigid_motion_still_surfaces():
    """The realistic version: a real pose plus a scale fault, only the pose absorbed.

    A rigid motion preserves distances, so it cannot change what the scale costs.
    The residual is the same one theory predicts for the scale alone, which is
    what makes it a usable alarm on a real registration rather than only on a
    contrived one.
    """
    factor = 1.5
    centre = centroid(CORRESPONDENCE_POINTS)
    pose = Rigid.of(rotation_of((1.0, -2.0, 3.0), 0.9), (12.5, -3.25, 7.75))
    target = tuple(
        pose.apply(add(centre, scaled(subtract(point, centre), factor)))
        for point in CORRESPONDENCE_POINTS
    )
    predicted = (factor - 1.0) * rms_radius(CORRESPONDENCE_POINTS)

    fitted = rigid_from_correspondences(CORRESPONDENCE_POINTS, target)

    assert alignment_rmse(CORRESPONDENCE_POINTS, target, fitted) == pytest.approx(
        predicted, rel=1e-12
    )


def test_rigid_from_correspondences_refuses_mismatched_lengths():
    with pytest.raises(CorrespondenceUnusable) as caught:
        rigid_from_correspondences(CORRESPONDENCE_POINTS[:4], CORRESPONDENCE_POINTS[:5])
    message = str(caught.value)
    assert "4 source points against 5 target points" in message
    assert "one to one" in message


def test_rigid_from_correspondences_refuses_fewer_than_three_pairs():
    """Two points cannot pin down rotation about the line joining them."""
    for count in (0, 1, 2):
        with pytest.raises(CorrespondenceUnusable) as caught:
            rigid_from_correspondences(CORRESPONDENCE_POINTS[:count], CORRESPONDENCE_POINTS[:count])
        assert "at least 3" in str(caught.value)
    # Three is enough, and is accepted.
    known = Rigid.of(rotation_of(AXIS, 0.5), (1.0, 2.0, 3.0))
    triple = CORRESPONDENCE_POINTS[:3]
    fitted = rigid_from_correspondences(triple, transformed(known, triple))
    assert pose_difference(known, fitted).rotation_angle < 1e-12


def test_the_pairing_check_runs_before_the_point_data_is_looked_at():
    """A mismatched set is refused as mismatched, whatever else is wrong with it."""
    with pytest.raises(CorrespondenceUnusable):
        rigid_from_correspondences([(float("nan"), 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])


def test_the_fit_is_the_least_squares_optimum_not_merely_a_plausible_one():
    """Perturb the answer in every direction and confirm nothing beats it.

    A solver can return a transform that looks right on clean data and is
    slightly wrong on noisy data. Comparing the residual against neighbours of
    the returned pose is what distinguishes "solved" from "close".
    """
    known = Rigid.of(rotation_of((1.0, -2.0, 3.0), 0.9), (12.5, -3.25, 7.75))
    target = tuple(add(known.apply(point), scaled(point, 0.01)) for point in CORRESPONDENCE_POINTS)
    fitted = rigid_from_correspondences(CORRESPONDENCE_POINTS, target)
    best = alignment_rmse(CORRESPONDENCE_POINTS, target, fitted)
    assert best > 0.0  # the perturbation is real, so a perfect fit is impossible

    for axis in (AXIS, OTHER_AXIS, (1.0, 0.0, 0.0)):
        for nudge in (0.01, -0.01):
            worse = rotation_about_axis(axis, nudge).compose(fitted)
            assert alignment_rmse(CORRESPONDENCE_POINTS, target, worse) > best
    for shift in ((0.01, 0.0, 0.0), (0.0, -0.01, 0.0), (0.0, 0.0, 0.01)):
        worse = Rigid.of(fitted.rotation, add(fitted.translation, shift))
        assert alignment_rmse(CORRESPONDENCE_POINTS, target, worse) > best


# --- the residual that has to accompany every fit -----------------------------


def test_alignment_rmse_is_exactly_zero_for_a_perfect_fit():
    known = Rigid.of(rotation_of(AXIS, 1.3), (12.5, -3.25, 7.75))
    target = transformed(known, CORRESPONDENCE_POINTS)
    assert alignment_rmse(CORRESPONDENCE_POINTS, target, known) == 0.0


def test_alignment_rmse_equals_the_perturbation_it_was_given():
    """A uniform displacement of every point makes the answer checkable by hand."""
    known = Rigid.of(rotation_of(AXIS, 1.3), (12.5, -3.25, 7.75))
    displacement: Point = (0.0, 0.3, 0.4)  # length exactly 0.5
    target = tuple(add(known.apply(point), displacement) for point in CORRESPONDENCE_POINTS)
    assert alignment_rmse(CORRESPONDENCE_POINTS, target, known) == pytest.approx(0.5, rel=1e-14)


def test_alignment_rmse_is_a_root_mean_square_not_a_mean_or_a_sum():
    """Three points displaced by 1, 2 and 3 give ``sqrt(14 / 3)``, and nothing else.

    A mean of the magnitudes would give 2, and a sum of squares would give 14.
    Distinct values, so the arithmetic is pinned rather than merely bounded.
    """
    source = CORRESPONDENCE_POINTS[:3]
    target = (
        add(source[0], (1.0, 0.0, 0.0)),
        add(source[1], (0.0, 2.0, 0.0)),
        add(source[2], (0.0, 0.0, 3.0)),
    )
    assert alignment_rmse(source, target, Rigid.identity()) == pytest.approx(
        math.sqrt(14.0 / 3.0), rel=1e-15
    )


def test_alignment_rmse_scales_with_the_size_of_the_error():
    source = CORRESPONDENCE_POINTS
    identity = Rigid.identity()
    single = alignment_rmse(source, tuple(add(p, (0.1, 0.0, 0.0)) for p in source), identity)
    double = alignment_rmse(source, tuple(add(p, (0.2, 0.0, 0.0)) for p in source), identity)
    assert single == pytest.approx(0.1, rel=1e-14)
    assert double == pytest.approx(2.0 * single, rel=1e-14)


def test_alignment_rmse_refuses_a_set_it_cannot_pair_or_average():
    with pytest.raises(CorrespondenceUnusable) as caught:
        alignment_rmse(CORRESPONDENCE_POINTS[:2], CORRESPONDENCE_POINTS[:3], Rigid.identity())
    assert "2 source points against 3 target points" in str(caught.value)

    with pytest.raises(CorrespondenceUnusable) as caught:
        alignment_rmse((), (), Rigid.identity())
    assert "undefined" in str(caught.value)


# --- pose difference and loop closure ----------------------------------------


def test_pose_difference_is_the_transform_carrying_the_first_pose_to_the_second():
    a = Rigid.of(rotation_of(AXIS, 0.9), (1.0, -2.0, 3.0))
    b = Rigid.of(rotation_of(OTHER_AXIS, 1.4), (-4.0, 0.5, 2.0))
    bridge = pose_difference(a, b)
    for point in CORRESPONDENCE_POINTS:
        assert distance(bridge.apply(a.apply(point)), b.apply(point)) < 1e-12
    # And not the other way round: these poses are far enough apart to tell.
    assert (
        distance(
            pose_difference(b, a).apply(a.apply(CORRESPONDENCE_POINTS[0])),
            b.apply(CORRESPONDENCE_POINTS[0]),
        )
        > 1.0
    )


def test_the_difference_between_a_pose_and_itself_is_the_identity():
    a = Rigid.of(rotation_of(AXIS, 0.9), (100.0, -200.0, 300.0))
    same = pose_difference(a, a)
    assert same.rotation_angle < 1e-15
    assert same.translation_magnitude < 1e-12


def test_a_deliberately_broken_sweep_reports_both_halves_of_its_closure_error():
    """The metric has to move when the loop is actually open, not only when it is shut.

    Thirty-six steps that are each slightly too large leave a residual rotation,
    and a spindle offset means that residual also displaces the object. Both
    numbers are reported, and both are checked, because a closure error stated as
    an angle alone hides a drifting turntable centre.
    """
    overshoot = 1e-4
    step = rotation_about_axis(
        AXIS, 2.0 * math.pi / TURNTABLE_STEPS + overshoot / TURNTABLE_STEPS, through=SPINDLE
    )
    closure = pose_difference(Rigid.identity(), compose_n(step, TURNTABLE_STEPS))
    assert closure.rotation_angle == pytest.approx(overshoot, rel=1e-6)
    assert closure.translation_magnitude > 1e-4


# --- the serialised record ----------------------------------------------------


def test_a_transform_is_frozen_and_slotted_so_nothing_can_be_edited_after_validation():
    """Validation happens once, at construction. Mutability would make it a lie."""
    transform = Rigid.of(rotation_of(AXIS, 0.4), (1.0, 2.0, 3.0))
    with pytest.raises(FrozenInstanceError):
        transform.rotation = IDENTITY_MATRIX  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        transform.translation = (0.0, 0.0, 0.0)  # type: ignore[misc]
    # Slots: a typo cannot quietly attach a new attribute either.
    with pytest.raises(AttributeError):
        transform.rotaton = IDENTITY_MATRIX  # type: ignore[attr-defined]
    assert not hasattr(transform, "__dict__")


def test_as_dict_carries_the_schema_version_and_survives_json():
    transform = Rigid.of(rotation_of(AXIS, 0.4), (1.0, 2.0, 3.0))
    reported = transform.as_dict()
    assert reported["schema_version"] == SCHEMA_VERSION == 1
    assert reported["rotation"] == [list(row) for row in transform.rotation]
    assert reported["translation"] == [1.0, 2.0, 3.0]
    assert reported["rotation_angle_radians"] == transform.rotation_angle
    assert reported["translation_magnitude"] == transform.translation_magnitude
    assert json.loads(json.dumps(reported)) == reported


def test_the_serialised_record_names_no_physical_unit():
    """The honesty discipline, enforced rather than remembered.

    The depth scale's physical meaning is unestablished on this device, so a
    translation of ``1.0`` is one unit of nothing nameable. The angle is the sole
    quantity the record is entitled to name a unit for, and it does, which is why
    the assertion is not simply "no unit words anywhere".
    """
    transform = Rigid.of(rotation_of(AXIS, 0.4), (1.0, 2.0, 3.0))
    serialised = json.dumps(transform.as_dict()).lower()
    for word in FORBIDDEN_UNIT_WORDS:
        assert word not in serialised
    assert "rotation_angle_radians" in transform.as_dict()


def test_the_unit_guard_would_actually_catch_a_unit_named_field():
    """The guard checked against the shapes the violation would really take.

    A blacklist that happens to miss the realistic spelling is worse than none,
    because it reads as enforcement while enforcing nothing. Every name below is
    a plausible field somebody could add to :meth:`Rigid.as_dict` in good faith,
    and each has to trip at least one entry in the list.

    The spaced ``" metre"``/``" meter"`` entries alone did not catch any of the
    snake_case forms - which is the form a serialised field actually takes - so
    the underscore variants were added.
    """
    plausible_violations = (
        "translation_metres",
        "translation_meters",
        "distance_meters",
        "length_metres",
        "size_micrometres",
        "translation_millimetres",
        "translation_mm",
        "offset_in_mm",
        "scale_cm",
        "width_inches",
        "depth_centimeters",
    )
    for name in plausible_violations:
        payload = json.dumps({name: 1.0}).lower()
        assert any(word in payload for word in FORBIDDEN_UNIT_WORDS), (
            f"{name!r} is a physical-unit field name and the guard lets it through"
        )

    # And the guard must not fire on legitimate prose that merely contains the
    # stems, or the next reader will delete it as a nuisance.
    for innocent in ("parameter", "diameter", "perimeter", "rotation_angle_radians"):
        payload = json.dumps({innocent: 1.0}).lower()
        assert not any(word in payload for word in FORBIDDEN_UNIT_WORDS), (
            f"{innocent!r} is not a unit and the guard must not flag it"
        )


def test_the_module_docstring_disclaims_the_units_it_does_not_establish():
    """The claim travels with the code, so a reader of the module cannot miss it."""
    import scanner3d.geometry as geometry

    assert geometry.__doc__ is not None
    text = geometry.__doc__.lower()
    assert "no claim about physical units" in text
    assert "millimetres" in text  # named only to be disclaimed
    assert "nothing here converts to millimetres" in text


# --- behaviours recorded as they are, and reported rather than fixed ----------


def test_the_dataclass_constructor_does_not_validate_and_only_of_does():
    """Recorded, not endorsed: the guarantee lives in :meth:`Rigid.of` alone.

    The class docstring says "every way of building one goes through
    :meth:`of`", but the generated ``__init__`` is public and accepts anything -
    including the reflection this module exists to refuse. The module's own
    :meth:`Rigid.inverse` uses that bare constructor internally for a scratch
    value, so the door is not merely theoretical.

    Nothing here argues the bypass is desirable. It is pinned so that a reader
    knows where the guarantee actually is, and so that closing it has to be a
    deliberate change rather than a silent one. Reported upstream; not fixed
    here, because this file does not own ``geometry.py``.
    """
    smuggled = Rigid(rotation=PLANE_REFLECTION, translation=(0.0, 0.0, 0.0))
    assert determinant(smuggled.rotation) < 0.0
    with pytest.raises(NotARotation):
        Rigid.of(smuggled.rotation, smuggled.translation)


def test_collinear_correspondences_are_refused_because_the_residual_would_lie():
    """The degeneracy that arrives with a clean residual, and is now refused.

    A set of points on a line leaves rotation *about that line* undetermined.
    Horn's method still returns an answer, that answer is generally not the
    transform that produced the data, and the residual comes back at the float64
    floor - so a caller doing exactly what this module tells them to do, namely
    trusting the residual, would get no warning at all.

    The solver cannot detect this itself: the Jacobi routine accumulates
    orthogonal rotations from the identity, so its eigenvector is always unit
    length however degenerate the input. The rank of the point set is therefore
    checked directly, and this test pins that it is.
    """
    known = Rigid.of(rotation_of(AXIS, 0.9), (1.0, 2.0, 3.0))
    line = tuple((float(i), 2.0 * float(i), -3.0 * float(i)) for i in range(6))
    target = transformed(known, line)

    with pytest.raises(CorrespondenceUnusable) as raised:
        rigid_from_correspondences(line, target)
    message = str(raised.value)
    assert "span only 1 dimension" in message
    assert "rotation about that line changes nothing" in message


def test_the_refused_collinear_fit_really_would_have_looked_perfect():
    """Why the refusal is worth having: the residual alone could not catch it.

    Asserted against the machinery directly, so the claim in the docstring above
    is demonstrated rather than trusted. If a future change made the fit
    non-degenerate this test would stop being meaningful, so it checks the
    premise rather than the refusal.
    """
    line = tuple((float(i), 2.0 * float(i), -3.0 * float(i)) for i in range(6))
    assert spanned_dimensions(line) == 1, "the premise: this set really is rank 1"
    # Rotation about the line's own direction moves none of its points, which is
    # precisely why the rotation is undetermined and the residual uninformative.
    spin = rotation_about_axis((1.0, 2.0, -3.0), 0.9)
    for point in line:
        assert distance(spin.apply(point), point) == pytest.approx(0.0, abs=1e-12)


def test_coincident_correspondences_are_refused_as_determining_no_rotation():
    repeated = ((1.0, 2.0, 3.0),) * 5
    with pytest.raises(CorrespondenceUnusable) as raised:
        rigid_from_correspondences(repeated, repeated)
    assert "span only 0 dimension" in str(raised.value)


def test_the_rank_gate_examines_the_source_set_and_not_the_target():
    """Which of the two sets is measured is a real decision, so pin it.

    Every other degeneracy test here pairs a degenerate source with a target
    derived from it by a rigid motion - and rigid motion preserves rank, so both
    sets are equally degenerate and the two choices are indistinguishable. That
    left the gate free to read either one without any test noticing.

    It matters in both directions:

    * A collinear *source* is genuinely underdetermined - rotation about that
      line changes nothing - and must be refused however well-spread the target
      is. A gate reading the target would accept it and return a confident,
      arbitrary rotation.
    * A collinear *target* against a well-spread source is not underdetermined
      at all; it is simply data that no rigid transform fits. That must come back
      as a large residual, not as a refusal whose message claims the *source* was
      degenerate - which would be a false statement about the caller's input.
    """
    spread = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0), (-1.5, 0.5, 1.0))
    line = tuple((float(i), 2.0 * float(i), -3.0 * float(i)) for i in range(4))
    assert spanned_dimensions(spread) == 3
    assert spanned_dimensions(line) == 1

    # Degenerate source, well-spread target: refused, because the rotation
    # really is undetermined.
    with pytest.raises(CorrespondenceUnusable) as raised:
        rigid_from_correspondences(line, spread)
    assert "span only 1 dimension" in str(raised.value)

    # Well-spread source, degenerate target: accepted, and the mismatch is
    # reported honestly as a residual rather than as a refusal.
    fitted = rigid_from_correspondences(spread, line)
    assert alignment_rmse(spread, line, fitted) > 1.0


def test_a_planar_set_is_still_accepted_because_rank_two_determines_a_rotation():
    """The boundary: refuse rank 0 and 1, accept rank 2. Real depth data is often
    nearly planar and must not be refused for it."""
    known = Rigid.of(rotation_of(AXIS, 0.4), (0.5, -1.0, 2.0))
    planar = ((0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (0.0, 1.0, 10.0), (2.0, 3.0, 10.0))
    assert spanned_dimensions(planar) == 2
    target = transformed(known, planar)
    fitted = rigid_from_correspondences(planar, target)
    assert alignment_rmse(planar, target, fitted) < 1e-12


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (((1.0, 2.0, 3.0),) * 4, 0),
        (tuple((float(i), 0.0, 0.0) for i in range(4)), 1),
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)), 2),
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 3),
    ],
)
def test_spanned_dimensions_counts_the_directions_a_set_actually_occupies(points, expected):
    assert spanned_dimensions(points) == expected


def test_spanned_dimensions_of_an_empty_set_is_zero():
    assert spanned_dimensions(()) == 0


# --- one real captured cloud, for conditioning rather than accuracy -----------
#
# The capture, calibration and export modules are imported inside the two
# helpers below rather than at module scope, on purpose. ``geometry`` depends on
# none of them - it is arithmetic over tuples - and keeping that true of the test
# module as well means the whole rigid-transform suite still imports and runs
# when something in the depth path is mid-change or broken. Only the two tests
# that genuinely need a real frame pay for them.

#: The nine real captures, committed beside a manifest carrying each frame's
#: sha256. Used here for the *shape* of a real depth cloud and nothing else.
FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"

#: An explicit override, so a real frame kept elsewhere can still be used.
FRAME_PATH_ENV = "SCANNER3D_REAL_DEPTH_FRAME"

#: One point in this many is kept. A real frame yields tens of thousands of
#: points and this file has a runtime budget; the conditioning question the test
#: asks does not need all of them, only their distribution.
REAL_FRAME_STRIDE = 7


def real_frame_path() -> Path | None:
    """One real 512000-byte capture, or ``None`` if none is on this machine."""
    from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

    override = os.environ.get(FRAME_PATH_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    if not FIXTURE_FRAMES.is_dir():
        return None
    for path in sorted(FIXTURE_FRAMES.glob("*.bin")):
        if path.is_file() and path.stat().st_size == EXPECTED_FRAME_BYTES:
            return path
    return None


def real_cloud() -> tuple[Point, ...]:
    """A hard-downsampled cloud from one real frame, deprojected by the real path."""
    from scanner3d.cloud_export import camera_points
    from scanner3d.depth_metric import DepthUnitAssumption, deproject_depth_frame
    from scanner3d.intrinsics import INTRINSICS_FORMAT, parse_intrinsics

    path = real_frame_path()
    if path is None:
        pytest.skip(
            f"no real 512000-byte depth frame under {FIXTURE_FRAMES} and ${FRAME_PATH_ENV} is unset"
        )
    # This unit's factory record, quoted in the test rather than imported from
    # any module, exactly as the depth tests do: the calibration is injected.
    blob = struct.pack(
        INTRINSICS_FORMAT,
        640,
        400,
        898.6829833984375,
        0.0,
        42.43747329711914,
        0.0,
        898.6829833984375,
        192.8326873779297,
        0.0,
        0.0,
        1.0,
    )
    intrinsics = parse_intrinsics(blob)
    assert intrinsics is not None
    conversion = deproject_depth_frame(
        path.read_bytes(),
        intrinsics=intrinsics,
        depth_scale=0.004999999888241291,
        unit=DepthUnitAssumption(
            label="unvalidated depth unit",
            stated_by="tests/test_geometry.py",
            basis="the geometry tests need the cloud's shape, not its meaning",
        ),
    )
    return tuple(camera_points(conversion))[::REAL_FRAME_STRIDE]


def test_a_known_pose_is_recovered_from_a_real_less_well_conditioned_cloud():
    """Conditioning, not accuracy - and this file can assert nothing else.

    A real depth cloud is not the well-spread blob the synthetic tests use, and
    its cross-covariance is measurably worse conditioned. How much worse is
    asserted below rather than asserted in prose, because the earlier wording
    here ("nearly a plane", "far worse conditioned") overstated it: the cloud's
    thinnest principal extent is about 14% of its thickest, which is a squat
    solid, not a plane. The honest claim is the comparative one - worse than the
    hand-built set, by a factor this test measures - and that is what is checked.

    A closed-form solver that is fine on a tidy set can degrade on a worse
    conditioned one, so: a known transform is applied to the real points, the
    solver is asked to recover it, and the residual is reported alongside. Not
    one assertion here concerns whether the cloud's coordinates are correct, or
    what they measure. Nothing in this repository has established either.
    """
    cloud = real_cloud()
    assert len(cloud) >= 3

    # The premise, measured rather than asserted in prose. Both sets are full
    # rank, and the real one is the worse conditioned of the two.
    assert spanned_dimensions(cloud) == 3
    assert spanned_dimensions(CORRESPONDENCE_POINTS) == 3
    assert covariance_condition(cloud) > covariance_condition(CORRESPONDENCE_POINTS)

    extent = max(distance(point, cloud[0]) for point in cloud)
    known = rotation_about_axis((0.2, 1.0, -0.4), 0.35, through=centroid(cloud))
    known = Rigid.of(known.rotation, add(known.translation, (0.05, -0.02, 0.11)))
    target = transformed(known, cloud)

    fitted = rigid_from_correspondences(cloud, target)
    residual = alignment_rmse(cloud, target, fitted)

    # Relative to the cloud's own extent, because an absolute threshold on an
    # unvalidated scale would be a number pretending to mean something.
    assert residual < 1e-9 * extent
    assert pose_difference(known, fitted).rotation_angle < 1e-9


def test_a_real_cloud_survives_a_full_turntable_revolution_without_drifting():
    """The composed-sweep case on real data, measured against the points themselves.

    Composing thirty-six steps and applying the product is not the same
    arithmetic as applying thirty-six steps one at a time, and the difference is
    the accumulated float64 error a real sweep would carry. Bounding it against
    the cloud's own extent keeps the claim scale-free, which is the only honest
    way to state it here.
    """
    cloud = real_cloud()
    extent = max(distance(point, cloud[0]) for point in cloud)
    step = rotation_about_axis(AXIS, 2.0 * math.pi / TURNTABLE_STEPS, through=centroid(cloud))

    closed = compose_n(step, TURNTABLE_STEPS)
    assert closed.rotation_angle < 1e-12
    for point in cloud:
        assert distance(closed.apply(point), point) < 1e-9 * max(extent, 1.0)
