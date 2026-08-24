"""Tests for coarse alignment between fused clouds that share no turntable axis.

What is at stake
-----------------
:mod:`~scanner3d.multiview` exists because plain ICP has no starting point once
the scanner has been moved: measured on the real object it slides to within a
degree of the identity and calls that ``converged``. This module's whole job is
to supply the starting point ICP cannot find, by searching the principal-axis
and octahedral candidates and reporting whether the winner is *clearly* the
winner. Two properties matter more than any other and get the most weight here:

* **A large, honest recovery.** A coarse aligner that only proves itself on
  gentle rotations proves nothing - ICP can already do gentle rotations. The
  tests below include a 180-degree flip because that is exactly the shape of
  bug this module shipped with once: the octahedral correction has to be
  composed *between* the two principal frames, and a version that instead
  post-multiplies the already-combined transform recovers a small turn exactly
  (it needs no sign correction) while missing a flip by 140 degrees. See
  :func:`test_coarse_align_recovers_a_180_degree_flip_exactly`.
* **Honest refusal.** A symmetric object has more than one orientation that
  fits equally well, and reporting one of them as confident would be worse
  than not aligning at all - that is the specific failure this module exists
  to prevent. See :func:`test_coarse_align_reports_ambiguous_for_a_genuinely_symmetric_cloud`.

None of this establishes that the module produces a *correct* final pose: it
is a coarse starting point for ICP, not an alignment, and every test cloud's
extent is a bare number with no physical unit - see the module's own
docstring for why.

Test data
---------
A single-view :func:`~scanner3d.simulate.render` - like a real depth frame -
only ever shows the faces turned toward the camera, which is a flat sliver
with no principal frame worth having. This module operates on *fused* clouds
(its own docstring is explicit about that), so the box used throughout is
built from four single-view renders, each rotated back into one shared object
frame before merging - a crude stand-in for the registration-then-fusion
pipeline this module expects to sit downstream of. The symmetric prism used
for the ambiguity test is built directly, by rotating one face's samples into
the other three, because a rendered capture would only be symmetric in the
limit of infinite sampling and the ambiguity test needs the point *set* itself
to tie, not merely the continuous shape it approximates.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import (
    Point,
    Rigid,
    determinant,
    ensure_rotation,
    pose_difference,
    rotation_about_axis,
)
from scanner3d.multiview import (
    AMBIGUITY_MARGIN,
    MIN_ALIGN_POINTS,
    SCHEMA_VERSION,
    CloudDegenerate,
    CloudTooSmall,
    CoarseAlignment,
    MultiviewNotAllowed,
    coarse_align,
    octahedral_rotations,
    overlap_fraction,
    principal_frame,
)
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.simulate import Box, points_of, render

# --------------------------------------------------------------------------
# Test clouds
# --------------------------------------------------------------------------


def _fused_box_cloud(half_extents: Point, *, distance: float = 150.0) -> tuple[Point, ...]:
    """Merge four single-view renders of a box into one full-surface cloud.

    Each view rotates the box before rendering and then rotates the resulting
    points back by the inverse, so all four land in one shared object frame -
    a stand-in for the register-then-fuse pipeline this module expects to sit
    downstream of, without pulling that whole pipeline into this test file.
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
        # A rotation about the origin moves the origin nowhere, so the pose
        # that places the box at `distance` in front of the camera is just
        # this rotation plus that translation - no separate correction needed.
        pose = Rigid.of(rotation, (0.0, 0.0, distance))
        frame = render([box], pose=pose)
        merged.extend(pose.inverse().apply_all(points_of(frame)))
    return tuple(merged)


@functools.cache
def _asymmetric_box() -> tuple[Point, ...]:
    """A box with three distinct extents, fused and downsampled.

    Distinct extents mean the principal frame has no eigenvalue tie, so every
    coarse-alignment recovery test built on this cloud has exactly one correct
    candidate rather than a family of equally good ones - that family is what
    the *ambiguity* tests exist to cover, deliberately kept separate.
    """
    raw = _fused_box_cloud((30.0, 12.0, 5.0))
    return voxel_downsample(raw, cloud_extent(raw) / 25.0)


def _symmetric_square_prism(
    *,
    half_side: float = 4.0,
    half_length: float = 25.0,
    edge_points: int = 12,
    z_layers: int = 15,
) -> tuple[Point, ...]:
    """A square prism's four side faces, exactly invariant under a quarter turn.

    Built by sampling one face and rotating those samples by 0, 90, 180 and
    270 degrees about the long (z) axis, rather than by rendering: rendering
    would only approximate the symmetry in the limit of infinite sampling,
    and the ambiguity test needs the point *set* to be exactly self-similar,
    so that a quarter-turned copy really does tie with holding still.
    """
    points: list[Point] = []
    for z_index in range(z_layers):
        z = -half_length + 2.0 * half_length * z_index / (z_layers - 1)
        for t_index in range(edge_points):
            t = -half_side + 2.0 * half_side * t_index / (edge_points - 1)
            edge_point: Point = (half_side, t, z)
            for turn in range(4):
                spin = rotation_about_axis((0.0, 0.0, 1.0), turn * math.pi / 2.0)
                points.append(spin.apply(edge_point))
    return tuple(points)


@functools.cache
def _symmetric_prism() -> tuple[Point, ...]:
    return _symmetric_square_prism()


def _sphere_points(
    *, radius: float = 10.0, rings: int = 10, per_ring: int = 16
) -> tuple[Point, ...]:
    """A near-uniform sphere sample: equal extents along every axis, by construction."""
    points: list[Point] = []
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(per_ring):
            theta = 2.0 * math.pi * j / per_ring
            points.append(
                (
                    radius * math.sin(phi) * math.cos(theta),
                    radius * math.cos(phi),
                    radius * math.sin(phi) * math.sin(theta),
                )
            )
    return tuple(points)


#: Cheap to build (144 points, no rendering), so a plain module constant.
_SPHERE: tuple[Point, ...] = _sphere_points()

#: A regular grid, used only for :func:`overlap_fraction` - its exact spacing
#: (1.0) is what makes the expected overlap fractions below exact numbers
#: rather than approximations.
_OVERLAP_GRID: tuple[Point, ...] = tuple(
    (float(i), float(j), float(k)) for i in range(10) for j in range(10) for k in range(6)
)


@functools.cache
def _small_turn_result() -> CoarseAlignment:
    """One ordinary, unambiguous alignment, cached for the structural tests below."""
    src = _asymmetric_box()
    truth = rotation_about_axis((0.1, 1.0, 0.05), math.radians(8.0))
    return coarse_align(src, tuple(truth.apply_all(src)), sample=150)


# --------------------------------------------------------------------------
# The octahedral candidate set
# --------------------------------------------------------------------------


def test_octahedral_rotations_are_24_distinct_proper_rotations():
    """The rotation group of the cube: order 24, and every element a genuine rotation.

    Applied on top of the principal-axis candidates so a cloud whose principal
    frame is degenerate is still covered - see the module docstring.
    """
    rotations = octahedral_rotations()

    assert len(rotations) == 24
    seen: set[tuple[float, ...]] = set()
    for rotation in rotations:
        # Not raising here IS the acceptance check; determinant is asserted
        # explicitly too so a broken determinant path inside ensure_rotation
        # could not silently wave one through.
        assert ensure_rotation(rotation) == rotation
        assert determinant(rotation) == pytest.approx(1.0, abs=1e-9)
        seen.add(tuple(round(value, 9) for row in rotation for value in row))
    assert len(seen) == 24, "24 distinct matrices, not 24 calls that happen to repeat one"


# --------------------------------------------------------------------------
# The principal frame
# --------------------------------------------------------------------------


def test_principal_frame_of_a_clearly_elongated_cloud_is_not_degenerate():
    """Three distinct extents: the frame is determined by the geometry, not guessed."""
    rotation, degenerate = principal_frame(_asymmetric_box())

    assert degenerate is False
    assert determinant(rotation) == pytest.approx(1.0, abs=1e-9)


def test_principal_frame_of_a_near_spherical_cloud_is_degenerate():
    """Equal extents in every direction: no axis is distinguishable from another.

    This does not establish WHICH frame is returned - only that the module
    says the frame is not to be trusted, rather than reporting an arbitrary
    one with the same confidence as a determined frame.
    """
    rotation, degenerate = principal_frame(_SPHERE)

    assert degenerate is True
    # A guess, but never a reflection: right-handedness is not optional here.
    assert determinant(rotation) == pytest.approx(1.0, abs=1e-9)


def test_principal_frame_refuses_fewer_than_the_minimum_points():
    """The boundary in both directions: one short of the minimum, and exactly it."""
    with pytest.raises(CloudTooSmall, match=str(MIN_ALIGN_POINTS)):
        principal_frame(_SPHERE[: MIN_ALIGN_POINTS - 1])

    principal_frame(_SPHERE[:MIN_ALIGN_POINTS])  # exactly the minimum: not refused


def test_principal_frame_refuses_a_planar_cloud():
    """All points at one height: two dimensions of spread, not the three needed."""
    planar = tuple((float(i % 20), float(i // 20), 5.0) for i in range(120))

    with pytest.raises(CloudDegenerate, match="does not span three dimensions"):
        principal_frame(planar)


def test_principal_frame_refuses_a_collinear_cloud():
    """All points on one line: rotation about that line changes nothing about it."""
    collinear = tuple((float(i), 0.0, 0.0) for i in range(MIN_ALIGN_POINTS + 10))

    with pytest.raises(CloudDegenerate, match="does not span three dimensions"):
        principal_frame(collinear)


# --------------------------------------------------------------------------
# Coarse alignment: recovering known reorientations
# --------------------------------------------------------------------------


def test_coarse_align_recovers_a_180_degree_flip_exactly():
    """Pins the composition-order bug this module shipped with once.

    The wrong order, ``dst_frame @ src_frame^T @ extra``, applies the
    octahedral correction in the SOURCE's own coordinates instead of between
    the two principal frames. Measured on the real object it recovered a
    35-degree turn exactly - a small turn needs no sign correction - while
    missing a 180-degree flip by 140 degrees, because a flip is nothing BUT
    the sign correction. A test built only from gentle rotations would have
    passed against that version; this one would not have.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.2, 1.0, -0.15), math.pi)
    target = tuple(truth.apply_all(src))

    result = coarse_align(src, target, sample=150)

    assert result.best is not None
    error = pose_difference(truth, result.best.transform)
    assert error.rotation_angle < 1e-6
    assert result.unambiguous is True, "an asymmetric box has exactly one right answer here"


_RECOVERY_CASES = (
    pytest.param(rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2.0), id="90_degree_roll"),
    pytest.param(rotation_about_axis((0.37, -0.61, 0.7), math.radians(53.0)), id="odd_axis_tilt"),
    pytest.param(rotation_about_axis((0.1, 1.0, 0.05), math.radians(8.0)), id="small_turn"),
)


@pytest.mark.parametrize("truth", _RECOVERY_CASES)
def test_coarse_align_recovers_known_reorientations(truth: Rigid):
    """The other reorientations a two-position scan needs: a right-angle roll, an
    oblique tilt unrelated to the box's own axes, and a small turn close enough
    that ICP alone might almost have found it without help.
    """
    src = _asymmetric_box()
    target = tuple(truth.apply_all(src))

    result = coarse_align(src, target, sample=150)

    assert result.best is not None
    error = pose_difference(truth, result.best.transform)
    assert error.rotation_angle < 1e-6
    assert result.unambiguous is True


def test_coarse_align_reports_ambiguous_for_a_genuinely_symmetric_cloud():
    """The module's most important property: refusing to manufacture confidence.

    A square prism is invariant under a quarter turn about its own long axis,
    so aligning it to a quarter-turned copy of itself has at least two equally
    valid answers - the true turn, and holding still. Picking one and calling
    it unambiguous would be worse than not aligning at all, which is exactly
    the failure this module exists to prevent.
    """
    src = _symmetric_prism()
    truth = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2.0)
    target = tuple(truth.apply_all(src))

    result = coarse_align(src, target, sample=150)

    assert result.unambiguous is False
    assert result.margin < AMBIGUITY_MARGIN
    assert result.source_principal_degenerate is True
    assert result.target_principal_degenerate is True
    joined = " ".join(result.limitations)
    assert "does not clearly distinguish" in joined
    assert "near-equal principal extents" in joined


# --------------------------------------------------------------------------
# overlap_fraction
# --------------------------------------------------------------------------


def test_overlap_fraction_of_a_cloud_aligned_to_itself_is_complete():
    assert overlap_fraction(_OVERLAP_GRID, _OVERLAP_GRID, Rigid.identity(), tolerance=0.4) == 1.0


def test_overlap_fraction_of_a_cloud_translated_far_away_is_zero():
    far = tuple((p[0] + 1000.0, p[1], p[2]) for p in _OVERLAP_GRID)

    assert overlap_fraction(_OVERLAP_GRID, far, Rigid.identity(), tolerance=0.4) == 0.0


def test_overlap_fraction_of_a_partial_overlap_lands_between_the_extremes():
    """Shifting the grid by half its x-span puts exactly half of it back on itself."""
    shifted = tuple((p[0] + 5.0, p[1], p[2]) for p in _OVERLAP_GRID)

    fraction = overlap_fraction(_OVERLAP_GRID, shifted, Rigid.identity(), tolerance=0.4)

    assert fraction == pytest.approx(0.5)


def test_a_low_overlap_with_a_locally_perfect_match_is_reported_as_low():
    """The property the module's scoring docstring warns about, checked on its own.

    The coarse scorer excludes points with no partner rather than penalising
    them, so a candidate that only covers one corner of the cloud can still
    show a near-zero residual on the points it did match - a good-looking
    number that describes a corner, not the object. ``overlap_fraction`` is
    what a caller reads alongside that residual to catch it: it is computed
    over the WHOLE source cloud, so a corner cannot pass itself off as a fit.
    """
    corner = tuple(p for p in _OVERLAP_GRID if p[0] >= 9.0)
    assert len(corner) < len(_OVERLAP_GRID) / 5, "the fixture must actually be a small corner"

    fraction = overlap_fraction(_OVERLAP_GRID, corner, Rigid.identity(), tolerance=0.4)

    # Every point that matched, matched exactly - the corner is a literal
    # subset of the grid's own points - so a residual computed only over the
    # matches would read as a perfect fit. The overlap fraction says otherwise.
    assert fraction == pytest.approx(len(corner) / len(_OVERLAP_GRID))
    assert fraction < 0.2, "a perfect match on a tenth of the cloud is not a good fit"


# --------------------------------------------------------------------------
# Records: frozen, versioned, unit-free
# --------------------------------------------------------------------------


def test_alignment_candidate_and_coarse_alignment_are_frozen():
    result = _small_turn_result()
    assert result.best is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.best.score = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.unambiguous = False  # type: ignore[misc]


def test_coarse_alignment_as_dict_carries_the_schema_version():
    result = _small_turn_result()

    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["considered"] == len(result.candidates)
    assert len(record["candidates"]) == min(5, len(result.candidates))
    assert record["candidates"][0]["transform"]["schema_version"] == SCHEMA_VERSION


def test_no_output_states_or_implies_a_physical_unit():
    """This module's clouds have no established unit; nothing it emits may claim one."""
    text = json.dumps(_small_turn_result().as_dict()).lower()

    for forbidden in ("millimet", "centimet", "inch", " mm "):
        assert forbidden not in text


def test_the_refusals_are_scanner_errors_carrying_a_doctor_status():
    assert issubclass(CloudTooSmall, MultiviewNotAllowed)
    assert issubclass(CloudDegenerate, MultiviewNotAllowed)
    assert MultiviewNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE
