"""Tests for support: matching a scan to its own flipped-over twin.

What is at stake
-----------------
:mod:`~scanner3d.support` exists because :mod:`~scanner3d.multiview` genuinely
cannot do this job alone. The team lead measured it directly on the real truck
bed geometry that motivated this module: coarse alignment reports
``unambiguous=False`` with a margin near 1.0, and rescoring the candidates
with full-resolution ICP has the WRONG candidate - an upside-down twin of the
object - win by about one per cent. More compute does not fix this, because
point-to-point distance genuinely cannot separate a shallow object from its
own near-symmetric flip. The fix is a second, independent source of evidence:
the support-plane normal, which is a fact about which way the object was
resting, not about its shape.

Building a fixture where this actually matters is the hard part of testing
this module, and it turned out to be harder than it looked. A first attempt -
a box with three distinct extents, matching the reasoning
:func:`~test_multiview._asymmetric_box` uses - did not work here: any plain
box with exactly one face missing from each scan (the face that rested on the
table, which a real capture never sees) has an EXACT extra symmetry that has
nothing to do with its aspect ratio - the four side faces, each a plain
rectangle, are invariant under an additional 180-degree turn about the
vertical axis, regardless of how long or short the box is. That symmetry, not
noise, is what produces the near-tie, and no amount of stretching the box
fixed it. What did fix it was breaking that symmetry directly: cutting an
off-centre notch out of one side face, small enough not to look like the
whole picture but large enough that ICP's own correspondence trim cannot
average it away.

Test data
---------
The two flip-ambiguity fixtures below (:func:`_flip_fixture`,
:func:`_asymmetric_fixture`) are built directly from box geometry rather than
through :func:`~scanner3d.simulate.render`, because what is under test is the
*occlusion pattern* of a real flip capture - each scan permanently missing the
face that rested on the table - not the renderer's ray-surface intersection,
which :mod:`test_coverage` and :mod:`test_multiview` already exercise
elsewhere. Each scan is sampled directly on five of a box's six faces, with
independent Gaussian noise at sigma 0.65 added to every point - the magnitude
the team lead measured this scanner actually produces - so neither fixture is
the noiseless, easier version of the problem.

:func:`_asymmetric_fixture`, used for the "shape alone already works" test, IS
built through the renderer (reusing test_multiview's own
``_fused_box_cloud`` technique) precisely because that test wants the more
realistic, full-pipeline construction where it is not fighting to create an
ambiguity - shape genuinely resolves it there, and the point is only that the
support-plane filter does not spoil a correct answer that did not need it.
"""

from __future__ import annotations

import functools
import json
import math
import random
from collections.abc import Sequence

import pytest

from scanner3d.geometry import (
    Point,
    Rigid,
    normalize,
    pose_difference,
    rotation_about_axis,
)
from scanner3d.multiview import coarse_align
from scanner3d.registration import cloud_extent, icp, voxel_downsample
from scanner3d.simulate import Box, points_of, render
from scanner3d.support import (
    DEFAULT_UP_TEST_TOLERANCE,
    SCHEMA_VERSION,
    TRUST_MARGIN,
    EmptyCloud,
    ExpectedSignUnusable,
    FlipAlignment,
    NoCandidateSurvivedUpTest,
    UpTestToleranceUnusable,
    UpVectorUnusable,
    align_across_flip,
)

# --------------------------------------------------------------------------
# The near-symmetric flip fixture: the whole point of this module.
# --------------------------------------------------------------------------

#: A shallow box. The exact extents do not matter to the ambiguity itself -
#: see the module docstring - only the notch below does.
_HALF_EXTENTS: Point = (30.0, 3.0, 12.0)
_NOISE_SIGMA = 0.65
_FACE_DENSITY = 14
_FLIP_SEED = 4
_UP: Point = (0.0, 1.0, 0.0)


def _face_points(
    half_extents: Point, *, exclude: str, notch_fraction: float | None, n: int
) -> list[Point]:
    """Points sampled directly on five of a box's six faces.

    ``exclude`` is the face never captured - the one resting on the table.
    ``notch_fraction``, when given, cuts a corner off the +x face, keeping
    only the part with ``b <= notch_fraction * extent_b``: a plain box,
    missing only one face, has an exact 180-degree-about-vertical symmetry
    across its four side faces regardless of aspect ratio, and the notch is
    what breaks it - see the module docstring for why stretching the box does
    not. A smaller fraction cuts more away and breaks the symmetry harder;
    :func:`_near_tie_fixture` deliberately uses a gentler cut than
    :func:`_flip_fixture` to leave a genuine near-tie rather than a clean win.
    """
    hx, hy, hz = half_extents
    specs: dict[str, tuple[int, float, float, float]] = {
        "+x": (0, 1.0, hy, hz),
        "-x": (0, -1.0, hy, hz),
        "+y": (1, 1.0, hx, hz),
        "-y": (1, -1.0, hx, hz),
        "+z": (2, 1.0, hx, hy),
        "-z": (2, -1.0, hx, hy),
    }
    points: list[Point] = []
    for name, (axis, sign, extent_a, extent_b) in specs.items():
        if name == exclude:
            continue
        for i in range(n):
            for j in range(n):
                a = -extent_a + 2.0 * extent_a * (i + 0.5) / n
                b = -extent_b + 2.0 * extent_b * (j + 0.5) / n
                if notch_fraction is not None and name == "+x" and b > notch_fraction * extent_b:
                    continue
                coordinates = [0.0, 0.0, 0.0]
                coordinates[axis] = sign * half_extents[axis]
                other_axes = [k for k in range(3) if k != axis]
                coordinates[other_axes[0]] = a
                coordinates[other_axes[1]] = b
                points.append((coordinates[0], coordinates[1], coordinates[2]))
    return points


def _add_noise(points: Sequence[Point], sigma: float, rng: random.Random) -> tuple[Point, ...]:
    return tuple(
        (p[0] + rng.gauss(0.0, sigma), p[1] + rng.gauss(0.0, sigma), p[2] + rng.gauss(0.0, sigma))
        for p in points
    )


@functools.cache
def _flip_fixture() -> tuple[tuple[Point, ...], tuple[Point, ...], Rigid]:
    """Two scans of the same shallow, notched box: one flipped, one not.

    Returns ``(source, target, true_flip)``. ``source`` is scan B - the
    flipped capture, expressed in its own local frame - and ``target`` is
    scan A; ``true_flip`` is the ground-truth rotation the module has to
    recover. Each scan shows five of six faces - never the one that rested on
    the table - and the two scans are missing OPPOSITE faces, exactly as a
    real flip-and-rescan produces.
    """
    rng = random.Random(_FLIP_SEED)
    scan_a_body = _face_points(_HALF_EXTENTS, exclude="-y", notch_fraction=0.4, n=_FACE_DENSITY)
    scan_b_body = _face_points(_HALF_EXTENTS, exclude="+y", notch_fraction=0.4, n=_FACE_DENSITY)
    true_flip = rotation_about_axis((1.0, 0.0, 0.0), math.pi)
    target = _add_noise(scan_a_body, _NOISE_SIGMA, rng)
    source = tuple(true_flip.apply_all(_add_noise(scan_b_body, _NOISE_SIGMA, rng)))
    return source, target, true_flip


@functools.cache
def _near_tie_fixture() -> tuple[tuple[Point, ...], tuple[Point, ...], Rigid]:
    """A gentler cut than :func:`_flip_fixture`'s: a genuine near-tie, not a clean win.

    Measured: the winner is still the true flip (well under a degree of
    error), but only by a margin of about 0.6 per cent over its runner-up -
    the kind of near-tie this object family produces even when the answer
    happens to be right, and exactly what ``trustworthy`` exists to flag.
    """
    rng = random.Random(3)
    scan_a_body = _face_points(_HALF_EXTENTS, exclude="-y", notch_fraction=0.75, n=_FACE_DENSITY)
    scan_b_body = _face_points(_HALF_EXTENTS, exclude="+y", notch_fraction=0.75, n=_FACE_DENSITY)
    true_flip = rotation_about_axis((1.0, 0.0, 0.0), math.pi)
    target = _add_noise(scan_a_body, _NOISE_SIGMA, rng)
    source = tuple(true_flip.apply_all(_add_noise(scan_b_body, _NOISE_SIGMA, rng)))
    return source, target, true_flip


@functools.cache
def _unfiltered_best_error_degrees() -> tuple[float, float]:
    """The error (degrees, from the true flip) and RMSE of the best candidate

    coarse_align + ICP finds with NO support-plane filtering at all. This is
    what :func:`align_across_flip` must NOT return: measured, it is the
    twin, not the truth.
    """
    source, target, true_flip = _flip_fixture()
    coarse = coarse_align(source, target)
    best_rmse = math.inf
    best_transform = Rigid.identity()
    for candidate in coarse.candidates:
        result = icp(source, target, initial=candidate.transform, max_iterations=50)
        if result.rmse < best_rmse:
            best_rmse = result.rmse
            best_transform = result.transform
    error = pose_difference(best_transform, true_flip).rotation_angle
    return math.degrees(error), best_rmse


@functools.cache
def _primary_result() -> FlipAlignment:
    source, target, _ = _flip_fixture()
    return align_across_flip(source, target, source_up=_UP, target_up=_UP, expected_sign=-1.0)


# --------------------------------------------------------------------------
# The near-symmetric flip: the whole point of this module.
# --------------------------------------------------------------------------


def test_support_plane_test_rescues_the_true_flip_from_its_shape_twin():
    """The test that fails if the up-test is removed.

    Measured: unfiltered ICP-best is the twin, about 180 degrees from the
    truth. Filtered by the support-plane test with the correct
    ``expected_sign``, the winner is the true flip, under a degree off, with
    a comfortable margin over its runner-up.
    """
    _, _, true_flip = _flip_fixture()
    unfiltered_error, unfiltered_rmse = _unfiltered_best_error_degrees()
    assert unfiltered_error > 170.0, (
        f"the fixture must reproduce the twin problem: unfiltered best should be near a "
        f"180-degree twin, got {unfiltered_error:.2f} degrees (rmse {unfiltered_rmse:.4f})"
    )

    result = _primary_result()
    recovered_error = math.degrees(pose_difference(result.transform, true_flip).rotation_angle)
    assert recovered_error < 5.0, (
        f"the support-plane test should recover the true flip, got {recovered_error:.2f} "
        f"degrees of error (rmse {result.rmse:.4f})"
    )
    assert result.candidates_proposed == 24
    assert result.candidates_rejected > 0
    assert result.candidates_kept > 0
    assert result.candidates_rejected + result.candidates_kept == result.candidates_proposed
    assert result.trustworthy
    assert result.margin >= TRUST_MARGIN


def test_wrong_expected_sign_selects_the_twin_not_the_truth():
    """Proof the sign is actually doing the work, not merely decorative.

    With ``expected_sign`` stating the object was NOT flipped - which is
    false for this fixture - the filter should keep the candidates the
    correct sign rejects, and its winner should be the same twin the
    unfiltered search finds, not some other candidate.
    """
    source, target, true_flip = _flip_fixture()
    _unfiltered_error, unfiltered_rmse = _unfiltered_best_error_degrees()

    wrong_signed = align_across_flip(
        source, target, source_up=_UP, target_up=_UP, expected_sign=1.0
    )
    error = pose_difference(wrong_signed.transform, true_flip).rotation_angle
    recovered_error = math.degrees(error)
    assert recovered_error > 170.0, (
        "expected_sign of the wrong sign should select the twin, not the true flip, got "
        f"{recovered_error:.2f} degrees of error"
    )
    assert wrong_signed.rmse == pytest.approx(unfiltered_rmse, rel=1e-6)


def test_near_tie_is_correctly_flagged_as_not_trustworthy():
    """ "Say so rather than claiming confidence": a correct-but-narrow win is not confident.

    A gentler notch than the primary fixture's leaves a genuine near-tie
    among the survivors. The winner is still correct - well under a degree of
    error - but wins by only a fraction of a per cent, comfortably under
    :data:`~scanner3d.support.TRUST_MARGIN`, and the module must say so rather
    than returning a confident-looking result.
    """
    source, target, true_flip = _near_tie_fixture()
    result = align_across_flip(source, target, source_up=_UP, target_up=_UP, expected_sign=-1.0)
    error = math.degrees(pose_difference(result.transform, true_flip).rotation_angle)
    assert error < 5.0, f"the winner should still be correct, got {error:.2f} degrees of error"
    assert result.margin < TRUST_MARGIN
    assert not result.trustworthy
    assert any("NOT TRUSTWORTHY" in limitation for limitation in result.limitations)


def test_nothing_survives_the_up_test_refuses_rather_than_falling_back():
    """The one non-negotiable behaviour: refuse, never silently fall back.

    A target_up pointing off in a direction none of the 24 candidates' support
    normals land anywhere near, combined with a tight tolerance, leaves
    nothing surviving. If this fell back to the unfiltered best instead of
    raising, it would silently return the twin - the exact bug this module
    exists to prevent.
    """
    source, target, _ = _flip_fixture()
    with pytest.raises(NoCandidateSurvivedUpTest):
        align_across_flip(
            source,
            target,
            source_up=_UP,
            target_up=(1.0, 1.0, 1.0),
            expected_sign=-1.0,
            tolerance=math.radians(5.0),
        )


# --------------------------------------------------------------------------
# Shape alone already works: the filter must not break it.
# --------------------------------------------------------------------------


def _fused_box_cloud(half_extents: Point, *, distance: float = 150.0) -> tuple[Point, ...]:
    """Four single-view renders of a box, merged into one full-surface cloud.

    The same technique :mod:`test_multiview` uses for its own asymmetric-box
    fixture: a single render only shows the camera-facing side, so four
    rotated views are merged, each un-rotated back into one shared frame.
    """
    rotations = (
        Rigid.identity().rotation,
        rotation_about_axis((0.0, 1.0, 0.0), math.pi).rotation,
        rotation_about_axis((0.0, 1.0, 0.0), math.pi / 2.0).rotation,
        rotation_about_axis((1.0, 0.0, 0.0), math.pi / 2.0).rotation,
    )
    box = Box(half_extents=half_extents)
    merged: list[Point] = []
    for rotation in rotations:
        pose = Rigid.of(rotation, (0.0, 0.0, distance))
        frame = render([box], pose=pose)
        merged.extend(pose.inverse().apply_all(points_of(frame)))
    return tuple(merged)


@functools.cache
def _asymmetric_fixture() -> tuple[tuple[Point, ...], tuple[Point, ...], Rigid, Point, Point]:
    """Two independently-noised scans of a plainly asymmetric box, related by

    a modest rotation - no flip, no near-symmetry, the case coarse_align and
    ICP already resolve unambiguously on their own. Returns ``(source,
    target, true_rotation, source_up, target_up)``.
    """
    raw = _fused_box_cloud((30.0, 12.0, 5.0))
    clean = voxel_downsample(raw, cloud_extent(raw) / 25.0)
    rng = random.Random(0)
    target = _add_noise(clean, _NOISE_SIGMA, rng)

    true_rotation = rotation_about_axis((0.3, 1.0, 0.1), math.radians(20.0))
    source_at_pose = tuple(true_rotation.inverse().apply_all(clean))
    source = _add_noise(source_at_pose, _NOISE_SIGMA, rng)

    # No physical flip between these two scans, so the support-plane normal
    # is the same physical direction in both - it just looks different in
    # each scan's own local frame, exactly as target_up does not.
    source_up = true_rotation.inverse().apply(_UP)
    return source, target, true_rotation, source_up, _UP


def test_asymmetric_object_where_shape_alone_already_works_still_works_with_filter_on():
    source, target, true_rotation, source_up, target_up = _asymmetric_fixture()
    result = align_across_flip(
        source, target, source_up=source_up, target_up=target_up, expected_sign=1.0
    )
    error = math.degrees(pose_difference(result.transform, true_rotation).rotation_angle)
    assert error < 3.0, f"shape alone should already resolve this; got {error:.2f} degrees of error"
    assert result.trustworthy
    assert result.candidates_kept >= 1


# --------------------------------------------------------------------------
# Rotation only, never translation.
# --------------------------------------------------------------------------


def test_up_test_uses_rotation_only_and_ignores_translation():
    """Moving both clouds far from the origin must not change the answer.

    Registration is translation-invariant in its own right, so the only way
    a large, shared offset could change the result is if the support-plane
    test were - wrongly - applying a candidate's full transform, translation
    included, to a direction vector. A direction has no position to be moved
    by; if this changed, that would be the bug.
    """
    source, target, _ = _flip_fixture()
    near = align_across_flip(source, target, source_up=_UP, target_up=_UP, expected_sign=-1.0)

    offset: Point = (5000.0, -8000.0, 3000.0)
    far_source = tuple((p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]) for p in source)
    far_target = tuple((p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]) for p in target)
    far = align_across_flip(
        far_source, far_target, source_up=_UP, target_up=_UP, expected_sign=-1.0
    )

    spin_near = Rigid.of(near.transform.rotation, (0.0, 0.0, 0.0))
    spin_far = Rigid.of(far.transform.rotation, (0.0, 0.0, 0.0))
    rotation_gap = pose_difference(spin_near, spin_far).rotation_angle
    assert rotation_gap < 1e-6
    assert near.rmse == pytest.approx(far.rmse, rel=1e-6)
    assert near.candidates_kept == far.candidates_kept
    assert near.candidates_rejected == far.candidates_rejected


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

_ONE_CLOUD: tuple[Point, ...] = ((0.0, 0.0, 1.0),)


def test_align_across_flip_raises_for_empty_source_cloud():
    with pytest.raises(EmptyCloud):
        align_across_flip((), _ONE_CLOUD, source_up=_UP, target_up=_UP, expected_sign=-1.0)


def test_align_across_flip_raises_for_empty_target_cloud():
    with pytest.raises(EmptyCloud):
        align_across_flip(_ONE_CLOUD, (), source_up=_UP, target_up=_UP, expected_sign=-1.0)


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (math.nan, 0.0, 0.0), (math.inf, 0.0, 0.0)])
def test_align_across_flip_raises_for_unusable_source_up(bad: Point):
    with pytest.raises(UpVectorUnusable):
        align_across_flip(_ONE_CLOUD, _ONE_CLOUD, source_up=bad, target_up=_UP, expected_sign=-1.0)


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (math.nan, 0.0, 0.0), (math.inf, 0.0, 0.0)])
def test_align_across_flip_raises_for_unusable_target_up(bad: Point):
    with pytest.raises(UpVectorUnusable):
        align_across_flip(_ONE_CLOUD, _ONE_CLOUD, source_up=_UP, target_up=bad, expected_sign=-1.0)


@pytest.mark.parametrize("bad_sign", [0.0, 0.5, -0.9, 2.0, math.nan, math.inf])
def test_align_across_flip_raises_for_unusable_expected_sign(bad_sign: float):
    with pytest.raises(ExpectedSignUnusable):
        align_across_flip(
            _ONE_CLOUD, _ONE_CLOUD, source_up=_UP, target_up=_UP, expected_sign=bad_sign
        )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_expected_sign_boundary_values_are_not_rejected_by_the_guard(sign: float):
    # A single-point cloud is too small for coarse_align's own principal-frame
    # requirement, so this exercises the sign guard specifically: it must not
    # be what raises here.
    with pytest.raises(Exception) as excinfo:
        align_across_flip(_ONE_CLOUD, _ONE_CLOUD, source_up=_UP, target_up=_UP, expected_sign=sign)
    assert not isinstance(excinfo.value, ExpectedSignUnusable)


@pytest.mark.parametrize("bad_tolerance", [0.0, -1.0, math.pi + 0.1, math.nan, math.inf])
def test_align_across_flip_raises_for_unusable_tolerance(bad_tolerance: float):
    with pytest.raises(UpTestToleranceUnusable):
        align_across_flip(
            _ONE_CLOUD,
            _ONE_CLOUD,
            source_up=_UP,
            target_up=_UP,
            expected_sign=-1.0,
            tolerance=bad_tolerance,
        )


def test_tolerance_boundary_value_pi_is_not_rejected_by_the_guard():
    with pytest.raises(Exception) as excinfo:
        align_across_flip(
            _ONE_CLOUD,
            _ONE_CLOUD,
            source_up=_UP,
            target_up=_UP,
            expected_sign=-1.0,
            tolerance=math.pi,
        )
    assert not isinstance(excinfo.value, UpTestToleranceUnusable)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_flip_alignment_as_dict_round_trips_and_carries_schema_version():
    result = _primary_result()
    record = json.loads(json.dumps(result.as_dict()))
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["rmse"] == pytest.approx(result.rmse)
    assert record["overlap"] == pytest.approx(result.overlap)
    assert record["candidates_proposed"] == result.candidates_proposed
    assert record["candidates_rejected"] == result.candidates_rejected
    assert record["candidates_kept"] == result.candidates_kept
    assert record["margin"] == pytest.approx(result.margin)
    assert record["trustworthy"] == result.trustworthy
    assert record["expected_sign"] == result.expected_sign
    assert record["transform"]["schema_version"] == 1
    assert record["limitations"]


def test_default_up_test_tolerance_is_a_finite_positive_radian_value():
    assert 0.0 < DEFAULT_UP_TEST_TOLERANCE <= math.pi


def test_normalize_of_the_shared_up_vector_is_a_unit_vector():
    # Sanity on the fixture itself, not the module - a malformed _UP constant
    # would silently invalidate every test above.
    unit = normalize(_UP)
    assert unit == pytest.approx((0.0, 1.0, 0.0))
