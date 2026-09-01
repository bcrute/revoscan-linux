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

The other two properties tested here are just as central:

* **Drift is measured without reference to angles, by comparing frames far
  apart in capture order.** A cleanly registered sweep must show that
  agreement is FLAT across every gap; a sweep with accumulating error must
  show it RISE with the gap. Either half alone is a weak test - a flat clean
  sweep could just mean the metric never reacts to anything - so
  :func:`test_model_drift_is_flat_when_clean_and_rises_with_the_gap_when_drifting`
  asserts both in one place, deliberately.
* **`refine_against_model` is a documented NEGATIVE result.** Measured against
  synthetic ground truth it cut the self-consistency residual 40% while
  DOUBLING the true pose error - free six-degree-of-freedom refinement against
  a self-consistency objective is under-constrained, so poses can agree with
  each other better while drifting together away from the truth. Nothing here
  asserts that it improves a scan; the tests below check only what it
  documents about itself: that ``residual_improved`` names the residual and
  not accuracy, that a refinement which does not improve the residual leaves
  the input transforms untouched, and that the anchor frame never moves.

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
from collections.abc import Sequence

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import (
    IDENTITY_MATRIX,
    Point,
    Rigid,
    determinant,
    distance,
    ensure_rotation,
    normalize,
    pose_difference,
    rotation_about_axis,
)
from scanner3d.multiview import (
    AMBIGUITY_MARGIN,
    MIN_ALIGN_POINTS,
    PRIOR_ORIGIN,
    SCHEMA_VERSION,
    SCORE_CELL_FRACTION,
    SCORE_MISS_CELLS,
    SCORE_NEIGHBOUR_REACH,
    CloudDegenerate,
    CloudTooSmall,
    CoarseAlignment,
    ModelDrift,
    ModelRefinement,
    MultiviewNotAllowed,
    TurntableRefinement,
    coarse_align,
    model_drift,
    octahedral_rotations,
    overlap_fraction,
    principal_frame,
    refine_against_model,
    refine_on_turntable,
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
    # Ambiguous because a genuinely DIFFERENT placement ties with the best one,
    # not because the search never found an alternative to compare against - the
    # two are opposite pieces of evidence and only one of them is symmetry.
    assert result.rival_index is not None
    assert result.source_principal_degenerate is True
    assert result.target_principal_degenerate is True
    joined = " ".join(result.limitations)
    assert "does not clearly distinguish" in joined
    assert "near-equal principal extents" in joined


# --------------------------------------------------------------------------
# Coarse alignment: what the ranking is measured on
# --------------------------------------------------------------------------


def _by_transform(result: CoarseAlignment, transform: Rigid):
    """The one ranked candidate that is exactly ``transform``, or a failure."""
    matches = [candidate for candidate in result.candidates if candidate.transform == transform]
    assert len(matches) == 1, f"expected exactly one candidate at {transform}, got {len(matches)}"
    return matches[0]


def _mean_displacement(points: Sequence[Point], a: Rigid, b: Rigid) -> float:
    """How far apart two transforms put the same cloud, averaged over it."""
    return sum(
        distance(first, second)
        for first, second in zip(a.apply_all(points), b.apply_all(points), strict=True)
    ) / len(points)


#: Distance between the repeated blocks of :func:`_repeated_blocks`, chosen so
#: that a block slid onto the next one lands EXACTLY on it while the block slid
#: off the end lands nowhere near anything: far beyond both the miss cost and
#: the reach of the scoring pass's own neighbour search, so a miss is a miss
#: rather than a distant partner found and charged its true distance.
_BLOCK_PERIOD = 100.0


def _repeated_blocks() -> tuple[Point, ...]:
    """Four identical small blocks in a row, one :data:`_BLOCK_PERIOD` apart.

    Built for one property: sliding this cloud by a whole number of periods
    puts some of its blocks EXACTLY onto the others and the rest of them into
    empty space. There is no third case - no point lands near a target point
    without landing on it - which is what lets the score below be an exact
    number rather than an approximation of one.
    """
    return tuple(
        (b * _BLOCK_PERIOD + i, float(j), float(k))
        for b in range(4)
        for i in range(4)
        for j in range(3)
        for k in range(2)
    )


def _coarse_cell(target: Sequence[Point]) -> float:
    """The scoring grid's cell for ``target``, at :func:`coarse_align`'s own scale.

    The LARGEST SINGLE-AXIS span, not the bounding-box diagonal
    :func:`~scanner3d.registration.cloud_extent` reports - those are different
    numbers on any cloud that is not a line, and it is the axis span the coarse
    grid is built from. Written out here rather than imported because a test
    that asserts an exact score has to state the arithmetic it expects rather
    than borrow it from the code under test.
    """
    return max(
        max(max(p[axis] for p in target) - min(p[axis] for p in target) for axis in range(3))
        * SCORE_CELL_FRACTION,
        1e-6,
    )


def test_unmatched_geometry_lowers_a_coarse_score():
    """The defect the tracking recovery was refused by, pinned at its source.

    Three placements of the same cloud on the same target, sliding it one, two
    and three block periods along. Every point that matches, matches EXACTLY,
    and every point that does not is in empty space; only HOW MANY match
    differs - three quarters, then half, then a quarter.

    A mean taken over MATCHED points only scores all three of these 0.0. They
    tie, exactly, and the ranking cannot tell a nearly complete fit from a
    mostly-missing one. That is what this module used to do, and downstream it
    meant a tracking recovery was refused as ambiguous while the geometry
    distinguished the placements clearly. The score is a TRUNCATED mean over ALL
    of the sampled points now: a point with no partner costs
    :data:`~scanner3d.multiview.SCORE_MISS_CELLS` cells, so each placement costs
    exactly the fraction of itself it left off the target.
    """
    blocks = _repeated_blocks()
    slides = {
        periods: Rigid.of(IDENTITY_MATRIX, (_BLOCK_PERIOD * periods, 0.0, 0.0))
        for periods in (1, 2, 3)
    }

    # ``sample`` at the cloud's own size so nothing is decimated away and the
    # matched counts below are out of the whole cloud rather than a subset.
    result = coarse_align(
        blocks,
        blocks,
        sample=len(blocks),
        priors=tuple(slides.values()),
        include_octahedral=False,
    )

    miss = SCORE_MISS_CELLS * _coarse_cell(blocks)
    scored = {periods: _by_transform(result, slide) for periods, slide in slides.items()}
    for periods, candidate in scored.items():
        # Exactly the blocks that landed on another block matched, and every one
        # of them matched perfectly - so the whole of the score is the miss cost.
        assert candidate.matched == len(blocks) * (4 - periods) // 4, periods
        missed = 1.0 - candidate.matched / len(blocks)
        assert candidate.score == pytest.approx(missed * miss, rel=1e-9), periods

    # The property, stated without the arithmetic: unmatched geometry costs, and
    # more of it costs more. Under a matched-only mean all three would be 0.0.
    assert scored[1].score < scored[2].score < scored[3].score
    positions = [result.candidates.index(scored[periods]) for periods in (1, 2, 3)]
    assert positions == sorted(positions)
    # And it costs a BOUNDED amount, which is what keeps a partial view rankable
    # at all rather than ranked entirely on how much of it overlaps.
    assert scored[3].score < miss


def test_a_caller_prior_is_ranked_by_the_same_measurement_as_a_generated_orientation():
    """A prior earns its rank on the shared pass or it does not get one.

    The reference here holds geometry the source never saw - a second copy of
    the object off to one side, which is what a recovery reference built from
    several accepted frames actually looks like. That pulls the target's
    centroid and principal axes away from the source's, so every orientation the
    search generates for ITSELF is offset and none of them is the truth. The
    caller's prior is, so it wins - and it is a genuinely different candidate
    from all of them rather than one of them spelled twice.

    What the assertions are about is that it won on the SAME measurement: it
    carries the prior origin, it is counted in ``priors_ranked``, the
    limitations say so in words, and its margin is a margin over everything that
    was offered.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.2, 1.0, -0.15), math.radians(37.0))
    body = tuple(truth.apply_all(src))
    unseen = max(p[0] for p in body) + 100.0
    target = body + tuple((x + unseen, y, z) for x, y, z in body[::3])

    result = coarse_align(src, target, sample=150, priors=(truth,))

    assert result.priors_ranked == 1
    assert result.best is not None
    assert result.best.origin == PRIOR_ORIGIN
    assert result.best.transform == truth
    assert any("came from the caller as priors" in note for note in result.limitations)
    # It is not privileged: the score it won on is the score the generated
    # orientations lost on, on the same decimated pass.
    assert all(
        candidate.score >= result.best.score
        for candidate in result.candidates
        if candidate.origin != PRIOR_ORIGIN
    )


def test_a_prior_that_repeats_a_generated_orientation_is_not_ranked_twice():
    """One placement offered twice is one candidate, and it is not a prior.

    On a target the search recovers exactly, a prior at the truth IS the
    orientation the principal candidate already generated. Ranking it a second
    time would put the same placement in the shortlist twice, and - because the
    margin is measured against the best genuinely different candidate - a
    duplicate that the deduplication missed would show up as a rival to itself.
    So the transform is kept once, under the origin that produced it first, and
    ``priors_ranked`` counts what actually earned a rank rather than what was
    offered.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.2, 1.0, -0.15), math.radians(37.0))
    target = tuple(truth.apply_all(src))

    with_prior = coarse_align(src, target, sample=150, priors=(truth,))
    without = coarse_align(src, target, sample=150)

    assert with_prior.priors_ranked == 0
    assert not any(candidate.origin == PRIOR_ORIGIN for candidate in with_prior.candidates)
    assert len(with_prior.candidates) == len(without.candidates)
    assert [candidate.transform for candidate in with_prior.candidates] == [
        candidate.transform for candidate in without.candidates
    ]
    assert not any("came from the caller as priors" in note for note in with_prior.limitations)


def test_a_prior_that_fits_worse_than_a_generated_orientation_ranks_below_it():
    """The other half of the same claim, which is the half that could be wrong.

    A prior that wins proves nothing about privilege on its own - the truth
    would win from anywhere. So a deliberately bad prior is offered here as
    well, on a target the generated orientations do recover exactly, and it has
    to rank below them.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2.0)
    target = tuple(truth.apply_all(src))
    nonsense = Rigid.of(
        rotation_about_axis((1.0, 0.0, 0.0), math.radians(80.0)).rotation, (400.0, -250.0, 90.0)
    )

    result = coarse_align(src, target, sample=150, priors=(nonsense,))

    assert result.priors_ranked <= 1
    assert result.best is not None
    assert result.best.origin != PRIOR_ORIGIN
    assert pose_difference(truth, result.best.transform).rotation_angle < 1e-6
    offered = [c for c in result.candidates if c.transform == nonsense]
    if offered:
        assert offered[0].score > result.best.score


def test_the_margin_skips_a_near_duplicate_and_measures_a_genuinely_different_rival():
    """A candidate that agrees with the best is not an alternative to it.

    The prior offered here is the best placement nudged by a fraction of one
    scoring cell - the same placement spelled twice. It scores within a hair of
    the best, so ``candidates[1]`` is it, and a margin taken there would be
    arithmetic noise reported as an ambiguity. The runner-up the margin is
    actually measured against has to be a candidate that puts the source
    SOMEWHERE ELSE, and the record has to say how many near-duplicates were
    stepped over to reach it.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.2, 1.0, -0.15), math.radians(37.0))
    target = tuple(truth.apply_all(src))
    cell = _coarse_cell(target)
    shift = cell * 0.05
    x, y, z = truth.translation
    nudge = Rigid.of(truth.rotation, (x + shift, y + shift, z + shift))

    result = coarse_align(src, target, sample=150, priors=(truth, nudge))

    assert result.best is not None
    assert result.duplicates_skipped >= 1
    assert result.rival_index is not None and result.rival_index > 1
    rival = result.candidates[result.rival_index]
    reach = cell * SCORE_NEIGHBOUR_REACH
    # The skipped one really is the same placement, and the rival really is not.
    for skipped in result.candidates[1 : result.rival_index]:
        assert _mean_displacement(src, result.best.transform, skipped.transform) <= reach
    assert _mean_displacement(src, result.best.transform, rival.transform) > reach
    assert result.margin == pytest.approx(rival.score / result.best.score)
    assert any("spelled more than once" in note for note in result.limitations)


def test_an_asymmetric_cloud_has_a_genuinely_different_rival_and_still_beats_it():
    """Unambiguous has to mean "beat a real alternative", not "found none".

    An infinite margin is what a search reports when every candidate landed in
    the same place, and that is a different and weaker piece of evidence than
    beating something. On an asymmetric box the search does generate genuinely
    different placements - flips and quarter turns of it - so the verdict here
    is the strong one, and the test says which.
    """
    src = _asymmetric_box()
    truth = rotation_about_axis((0.37, -0.61, 0.7), math.radians(53.0))
    target = tuple(truth.apply_all(src))

    result = coarse_align(src, target, sample=150)

    assert result.best is not None
    assert result.rival_index is not None, "a real alternative was found and beaten"
    assert result.margin >= AMBIGUITY_MARGIN
    assert result.margin != math.inf
    assert result.unambiguous is True
    cell = _coarse_cell(target)
    rival = result.candidates[result.rival_index]
    assert (
        _mean_displacement(src, result.best.transform, rival.transform)
        > cell * SCORE_NEIGHBOUR_REACH
    )


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

    The coarse scorer charges a point with no partner a BOUNDED miss cost, so a
    candidate that covers one corner of the cloud is penalised but not ruled
    out, and a score in the middle of the range cannot on its own say whether
    it came from a mediocre fit over everything or a perfect fit over a little.
    ``overlap_fraction`` is what a caller reads alongside it to tell those
    apart: it is computed over the WHOLE source cloud at the caller's own
    tolerance, so a corner cannot pass itself off as a fit.
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


# --------------------------------------------------------------------------
# model_drift: self-agreement between frames far apart in capture order
# --------------------------------------------------------------------------

#: A short turntable-like sweep of the asymmetric box, reused for both
#: model_drift and refine_against_model below. Six frames keeps every call in
#: this section well under a tenth of a second - model_drift is quadratic in
#: frame count and refine_against_model re-indexes the whole model per frame
#: per iteration, so both stay cheap only because the sweep is kept small.
_SWEEP_AXIS: Point = (0.1, 1.0, 0.05)
_SWEEP_STEP = 0.35
_SWEEP_FRAMES = 6

#: Off the origin in all three axes - not the point ``rotation_about_axis``
#: defaults to - so a bug that silently dropped ``through`` and rotated about
#: the origin instead would show up as a real displacement, not a coincidence.
_SWEEP_THROUGH: Point = (0.4, -0.2, 0.3)

#: Accumulates with capture index, standing in for the kind of error a
#: chained pairwise registration builds up - a different axis from the sweep
#: itself, so the drift is not just "the same turn, slightly wrong".
_DRIFT_AXIS: Point = (0.0, 0.3, 1.0)
_DRIFT_EPSILON = 0.02


def _turntable_frames(
    canonical: Sequence[Point],
    *,
    frames: int = _SWEEP_FRAMES,
    drift_epsilon: float = 0.0,
    drift_axis: Point = _DRIFT_AXIS,
) -> tuple[list[tuple[Point, ...]], list[Rigid]]:
    """A sweep of one canonical cloud, and the registered pose that undoes it.

    With ``drift_epsilon`` at zero the registered poses are exact inverses of
    the capture poses, so every frame lands back on ``canonical`` precisely -
    a perfectly registered model. A nonzero value composes an extra rotation
    onto each registered pose that grows with the frame index, the way error
    accumulates down a chain of pairwise fits; because it turns about a fixed
    axis by an angle proportional to the index, two frames i and j disagree by
    almost exactly ``(j - i) * drift_epsilon`` radians, which is what makes the
    residual-versus-gap growth below a controlled, not merely plausible, effect.

    ``drift_axis`` defaults to an axis unrelated to the sweep's own, for
    :func:`model_drift` and :func:`refine_against_model`, which have six free
    parameters and do not care where the error came from. :func:`refine_on_turntable`
    only has one, so its own fixture passes ``_SWEEP_AXIS`` here instead: the
    "drift" is then indistinguishable from a mis-measured turntable step -
    exactly the error that function exists to correct - rather than an error
    it has no way to express at all.
    """
    clouds: list[tuple[Point, ...]] = []
    transforms: list[Rigid] = []
    for index in range(frames):
        capture_pose = rotation_about_axis(_SWEEP_AXIS, index * _SWEEP_STEP, through=_SWEEP_THROUGH)
        clouds.append(tuple(capture_pose.apply_all(canonical)))
        registered = capture_pose.inverse()
        if drift_epsilon:
            drift = rotation_about_axis(drift_axis, index * drift_epsilon, through=_SWEEP_THROUGH)
            registered = drift.compose(registered)
        transforms.append(registered)
    return clouds, transforms


@functools.cache
def _clean_sweep() -> tuple[tuple[tuple[Point, ...], ...], tuple[Rigid, ...]]:
    clouds, transforms = _turntable_frames(_asymmetric_box())
    return tuple(clouds), tuple(transforms)


@functools.cache
def _drifted_sweep() -> tuple[tuple[tuple[Point, ...], ...], tuple[Rigid, ...]]:
    clouds, transforms = _turntable_frames(_asymmetric_box(), drift_epsilon=_DRIFT_EPSILON)
    return tuple(clouds), tuple(transforms)


@functools.cache
def _turntable_drifted_sweep() -> tuple[tuple[tuple[Point, ...], ...], tuple[Rigid, ...]]:
    """Like :func:`_drifted_sweep`, but the injected error turns about the sweep's
    own axis - a mis-measured step, the one kind of error a one-DOF-per-frame
    turntable refinement can actually express and therefore genuinely correct.
    """
    clouds, transforms = _turntable_frames(
        _asymmetric_box(), drift_epsilon=_DRIFT_EPSILON, drift_axis=_SWEEP_AXIS
    )
    return tuple(clouds), tuple(transforms)


def test_model_drift_is_flat_when_clean_and_rises_with_the_gap_when_drifting():
    """The one test this whole batch stands or falls on.

    A flat, near-zero residual on the perfectly registered sweep proves the
    metric is not manufacturing disagreement out of nothing; a residual that
    climbs with the capture-order gap on the drifted sweep proves it is
    actually sensitive to accumulated error, and does so in the one place
    loop closure cannot look - between frames that were never fitted to each
    other. Neither half alone would prove much: a flat result on its own could
    just mean the metric never reacts, and a rising result on its own could be
    noise rather than a trend tied to the gap.
    """
    clean_clouds, clean_transforms = _clean_sweep()
    drifted_clouds, drifted_transforms = _drifted_sweep()

    clean = model_drift(clean_clouds, clean_transforms, tolerance=6.0, min_index_gap=1, sample=100)
    drifted = model_drift(
        drifted_clouds, drifted_transforms, tolerance=6.0, min_index_gap=1, sample=100
    )

    assert all(pair.compared > 0 for pair in clean.pairs), "the fixture must actually overlap"
    assert all(pair.compared > 0 for pair in drifted.pairs), "the fixture must actually overlap"
    assert all(pair.residual < 1e-9 for pair in clean.pairs), "flat and near zero when clean"

    def _mean_residual_by_gap(result: ModelDrift) -> list[float]:
        by_gap: dict[int, list[float]] = {}
        for pair in result.pairs:
            by_gap.setdefault(pair.index_gap, []).append(pair.residual)
        return [sum(by_gap[gap]) / len(by_gap[gap]) for gap in sorted(by_gap)]

    drifted_means = _mean_residual_by_gap(drifted)
    assert drifted_means == sorted(drifted_means), "residual must rise monotonically with the gap"
    assert drifted_means[0] > 0.0, "a real drift, not a degenerate zero everywhere"
    # The largest gap must disagree far more than the smallest - not merely by
    # some rounding-sized amount - and the drifted sweep must clearly exceed
    # what the clean sweep showed at the same gap.
    assert drifted_means[-1] > drifted_means[0] * 3.0
    assert drifted_means[0] > max((p.residual for p in clean.pairs), default=0.0) * 1e6


def test_model_drift_residual_is_bounded_above_by_the_tolerance():
    """By construction: nothing further apart than ``tolerance`` counts as matched."""
    clouds, transforms = _drifted_sweep()
    tolerance = 6.0

    result = model_drift(clouds, transforms, tolerance=tolerance, min_index_gap=1, sample=100)

    assert result.pairs, "the fixture must produce at least one pair"
    for pair in result.overlapping:
        assert pair.residual <= tolerance


def test_a_non_overlapping_pair_is_reported_with_compared_zero_not_dropped():
    """A pair with no spatial overlap still appears in the record, just as evidence of nothing."""
    cluster = tuple((float(i), float(i % 3), 0.0) for i in range(20))
    far = tuple((p[0] + 10_000.0, p[1], p[2]) for p in cluster)
    clouds = [cluster, cluster, far]
    transforms = [Rigid.identity(), Rigid.identity(), Rigid.identity()]

    result = model_drift(clouds, transforms, tolerance=0.5, min_index_gap=1, sample=50)

    assert len(result.pairs) == 3, "no pair is silently dropped, overlapping or not"
    empty = next(p for p in result.pairs if p.first == 0 and p.second == 2)
    assert empty.compared == 0
    assert empty.overlap == 0.0
    assert math.isinf(empty.residual)
    assert empty not in result.overlapping
    # And the pair that does overlap is not swept up by the same fate.
    overlapping = next(p for p in result.pairs if p.first == 0 and p.second == 1)
    assert overlapping.compared > 0
    assert overlapping in result.overlapping


def test_model_drift_refuses_invalid_input():
    clouds, transforms = _clean_sweep()

    with pytest.raises(MultiviewNotAllowed, match="one transform per cloud"):
        model_drift(clouds, transforms[:-1], tolerance=1.0)
    with pytest.raises(MultiviewNotAllowed, match="at least two frames"):
        model_drift(clouds[:1], transforms[:1], tolerance=1.0)
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        model_drift(clouds, transforms, tolerance=0.0)
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        model_drift(clouds, transforms, tolerance=-1.0)


def test_model_drift_as_dict_carries_the_schema_version():
    clouds, transforms = _drifted_sweep()

    result = model_drift(clouds, transforms, tolerance=6.0, min_index_gap=1, sample=100)
    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["pairs_examined"] == len(result.pairs)
    assert record["pairs_overlapping"] == len(result.overlapping)
    assert record["worst"]["index_gap"] == result.worst.index_gap  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# refine_against_model: a documented negative result
#
# Measured against synthetic ground truth (see the function's own docstring),
# this cut the self-consistency residual 40% while DOUBLING the true pose
# error. Nothing below asserts that it improves a scan - that would be
# asserting something the module itself reports is false. What is tested is
# only what the module documents about its own behaviour: what the residual
# flag means, what happens when nothing improves, and that the anchor holds.
# --------------------------------------------------------------------------


@functools.cache
def _refined_result() -> ModelRefinement:
    """One ordinary refinement pass over the drifted sweep.

    Self-consistency has real room to improve here, so this reliably exercises
    the branch where ``residual_improved`` is True - which the "nothing
    happens" fixture below cannot, since that one returns the input untouched
    by construction.
    """
    clouds, transforms = _drifted_sweep()
    return refine_against_model(
        clouds, transforms, tolerance=6.0, iterations=3, sample=80, anchor=0
    )


def test_residual_improved_reflects_the_residual_not_an_accuracy_claim():
    """The flag is defined by the residual history alone - nothing else feeds it.

    This does NOT establish that a falling residual means a better scan; the
    module's own limitations say plainly that it measured the opposite on
    synthetic ground truth. This only pins down what the flag actually is.
    """
    result = _refined_result()

    assert result.initial_residual is not None
    assert result.final_residual is not None
    assert result.residual_improved == (result.final_residual < result.initial_residual)
    joined = " ".join(result.limitations)
    assert "MEASURED WARNING" in joined
    assert "is not a claim about accuracy" in joined
    # The warning must keep saying that a falling residual can accompany a WORSE
    # scan. An earlier version claimed refinement roughly doubles pose error;
    # that was measured with the two rotations compared in opposite senses and
    # was wrong. What replaced it is the finding that survived remeasurement:
    # the residual improved in all eight runs while the true pose error improved
    # in only six, so this flag cannot tell the good cases from the bad ones.
    assert "cannot distinguish" in joined


def test_refine_against_model_leaves_transforms_unchanged_when_the_residual_does_not_improve():
    """No correspondence can survive a tolerance this far below the point spacing.

    With nothing ever found to correct against, every pose is left exactly as
    given on every pass, so the residual history cannot fall - and the module's
    own contract is to hand back the ORIGINAL transforms in that case, not an
    approximation of them.
    """
    clouds, transforms = _drifted_sweep()

    result = refine_against_model(clouds, transforms, tolerance=1e-9, iterations=2, sample=60)

    assert result.residual_improved is False
    assert result.transforms == transforms, "the exact input, not merely a close pose"
    assert any("did NOT improve the residual" in note for note in result.limitations)


def test_refine_against_model_keeps_the_anchor_frame_exactly_fixed():
    """Re-anchoring is what stops the whole model drifting bodily through space.

    Checked on a run where refinement actually did something (see
    :func:`_refined_result`): a fixture where nothing changes would pass this
    trivially and prove nothing about the re-anchoring step itself.
    """
    _, transforms = _drifted_sweep()
    result = _refined_result()
    assert result.residual_improved is True, "the fixture must actually refine something"

    anchored = pose_difference(transforms[result.anchor], result.transforms[result.anchor])
    assert anchored.rotation_angle < 1e-9
    assert anchored.translation_magnitude < 1e-9


def test_refine_against_models_residual_history_has_one_entry_per_pass_plus_the_start():
    """The starting residual counts too, so the history is never merely the deltas.

    Checked on two very different runs - one that never corrects anything, one
    that runs to convergence - because the invariant has to hold regardless of
    *why* the loop stopped, not just in the one case that happens to be tested
    elsewhere.
    """
    clouds, transforms = _drifted_sweep()
    never_corrects = refine_against_model(
        clouds, transforms, tolerance=1e-9, iterations=2, sample=60
    )
    runs_to_convergence = _refined_result()

    for result in (never_corrects, runs_to_convergence):
        assert len(result.residual_history) == result.iterations_run + 1


def test_refine_against_model_stops_early_once_improvement_falls_below_the_threshold():
    """A generous ``min_improvement`` is satisfied almost immediately, well short
    of the iteration budget - and the record says so rather than just running quietly to the cap.
    """
    clouds, transforms = _drifted_sweep()

    result = refine_against_model(
        clouds, transforms, tolerance=6.0, iterations=5, sample=80, min_improvement=0.5
    )

    assert result.converged is True
    assert result.iterations_run < 5
    assert len(result.residual_history) == result.iterations_run + 1


def test_a_looser_improvement_threshold_converges_no_later_than_a_tighter_one():
    """The direction of the convergence comparison, pinned down by its effect.

    A single run at one threshold cannot tell "stops early because the
    threshold is loose" apart from "stops early because the threshold is
    tight" - the residual's own shape on this fixture actually plateaus by
    iteration 3 regardless of the threshold, so a run at only one
    ``min_improvement`` could pass against a version with the comparison
    backwards (``>`` instead of ``<``) by coincidence. Comparing a loose
    threshold against a tight one is what pins the direction down: a loose
    bar must be cleared sooner, never later, than a tight one.
    """
    clouds, transforms = _drifted_sweep()

    loose = refine_against_model(
        clouds, transforms, tolerance=6.0, iterations=5, sample=80, min_improvement=0.8
    )
    tight = refine_against_model(
        clouds, transforms, tolerance=6.0, iterations=5, sample=80, min_improvement=0.05
    )

    assert loose.iterations_run <= tight.iterations_run
    assert loose.iterations_run < 5, "the fixture must actually demonstrate an early stop"


def test_refine_against_model_refuses_invalid_input():
    clouds, transforms = _clean_sweep()

    with pytest.raises(MultiviewNotAllowed, match="one each"):
        refine_against_model(clouds, transforms[:-1], tolerance=1.0)
    with pytest.raises(MultiviewNotAllowed, match="at least three frames"):
        refine_against_model(clouds[:2], transforms[:2], tolerance=1.0)
    with pytest.raises(MultiviewNotAllowed, match="not a frame index"):
        refine_against_model(clouds, transforms, tolerance=1.0, anchor=-1)
    with pytest.raises(MultiviewNotAllowed, match="not a frame index"):
        refine_against_model(clouds, transforms, tolerance=1.0, anchor=len(clouds))
    with pytest.raises(MultiviewNotAllowed, match="iterations must be at least 1"):
        refine_against_model(clouds, transforms, tolerance=1.0, iterations=0)
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        refine_against_model(clouds, transforms, tolerance=0.0)
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        refine_against_model(clouds, transforms, tolerance=-1.0)


def test_refine_against_model_as_dict_carries_the_schema_version():
    result = _refined_result()

    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["frames"] == len(result.transforms)
    assert record["anchor"] == result.anchor
    assert record["residual_improved"] == result.residual_improved
    assert len(record["residual_history"]) == len(result.residual_history)
    assert any("MEASURED WARNING" in note for note in record["limitations"])


# --------------------------------------------------------------------------
# refine_on_turntable: the same idea as refine_against_model, with one DOF
#
# Whether this actually reduces true pose error against known ground truth is
# a separate, still-open measurement - nothing below asserts that it does, or
# that it doesn't. What is tested is the one claim that does not depend on
# that answer: every correction this function makes IS a pure rotation about
# the given axis, never anything else. That claim is what makes the six-DOF
# failure of refine_against_model unreachable here, and it can be checked
# today regardless of how the accuracy question eventually turns out.
# --------------------------------------------------------------------------


@functools.cache
def _turntable_refined_result() -> TurntableRefinement:
    """One ordinary turntable-constrained refinement pass over the turntable-axis
    drifted sweep - a mis-measured step is exactly what this function can correct,
    so this reliably exercises the branch where corrections are genuinely nonzero.
    """
    clouds, transforms = _turntable_drifted_sweep()
    return refine_on_turntable(
        clouds,
        transforms,
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=6.0,
        iterations=3,
        sample=80,
        anchor=0,
    )


def test_refine_on_turntable_confines_every_correction_to_a_pure_rotation_about_the_axis():
    """The entire reason this function exists, checked directly rather than assumed.

    ``refine_against_model`` hands every frame six free parameters and
    measurably makes scans worse by finding a self-consistent WRONG answer.
    This function is supposed to make that failure unreachable by allowing
    only one degree of freedom per frame - but a test that only checks the
    residual moved would not tell the two apart; a self-consistent wrong
    answer lowers the residual too. So this checks the actual MOTION applied
    to each frame instead, treating the function as a black box: take its own
    input and output poses, compose out everything that was already there, and
    demand that what is left is nothing but a rotation about ``direction``
    through ``through`` by exactly ``result.corrections[i]``. Three separate
    facts, not one:

    * the rotation's own axis is parallel to ``direction`` - checked by
      confirming the rotation's own axis direction is left FIXED by the
      relative motion's rotation part, which is what "this vector IS the
      axis" means geometrically (equivalent to, but numerically safer than,
      extracting the axis via a skew-symmetric decomposition, which is
      ill-conditioned for the small corrections this fixture produces);
    * the rotation fixes the ``through`` line, not merely one that happens to
      be parallel to it;
    * its angle equals the correction this function itself reports.

    Checked on the anchor-excluded frames of a fixture built so corrections
    are genuinely non-zero - a run where nothing moved would pass this
    vacuously and prove nothing.
    """
    _, transforms = _turntable_drifted_sweep()
    result = _turntable_refined_result()
    assert result.residual_improved is True, "the fixture must actually refine something"
    assert any(c != 0.0 for c in result.corrections[1:]), "and by a genuinely nonzero amount"

    unit_axis = normalize(_SWEEP_AXIS)
    through = _SWEEP_THROUGH
    for index in range(1, len(transforms)):  # frame 0 is the anchor; covered separately below
        observed = result.transforms[index].compose(transforms[index].inverse())
        rotation_only = Rigid.of(observed.rotation, (0.0, 0.0, 0.0))

        assert distance(rotation_only.apply(unit_axis), unit_axis) < 1e-9, "axis is not `direction`"
        assert distance(observed.apply(through), through) < 1e-9, "does not fix the `through` line"
        assert observed.rotation_angle == pytest.approx(abs(result.corrections[index]), abs=1e-9)


def test_refine_on_turntable_keeps_the_anchor_frame_exactly_fixed():
    """The re-anchoring subtracts the anchor's own angle from every correction,
    so the anchor's correction is exactly - not approximately - 0.0, and a
    rotation by exactly 0.0 is exactly the identity: the anchor's returned pose
    must be bit-for-bit its input, not merely close to it.
    """
    _, transforms = _turntable_drifted_sweep()
    result = _turntable_refined_result()
    assert result.residual_improved is True, "the fixture must actually refine something"

    assert result.corrections[result.anchor] == 0.0
    assert result.transforms[result.anchor] == transforms[result.anchor]


def test_refine_on_turntables_residual_history_has_one_entry_per_pass_plus_the_start():
    clouds, transforms = _turntable_drifted_sweep()
    never_corrects = refine_on_turntable(
        clouds,
        transforms,
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=1e-9,
        iterations=2,
        sample=60,
    )
    runs_to_convergence = _turntable_refined_result()

    for result in (never_corrects, runs_to_convergence):
        assert len(result.residual_history) == result.iterations_run + 1


def test_refine_on_turntable_leaves_transforms_unchanged_when_the_residual_does_not_improve():
    """As with :func:`refine_against_model`: a tolerance this far below the point
    spacing finds no correspondence to correct against, ever, so nothing moves.
    """
    clouds, transforms = _turntable_drifted_sweep()

    result = refine_on_turntable(
        clouds,
        transforms,
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=1e-9,
        iterations=2,
        sample=60,
    )

    assert result.residual_improved is False
    assert result.transforms == transforms, "the exact input, not merely a close pose"
    assert all(c == 0.0 for c in result.corrections)
    assert result.worst_correction == 0.0
    assert any("did NOT improve the residual" in note for note in result.limitations)


def test_a_looser_improvement_threshold_converges_no_later_on_turntable_refinement_too():
    """Same reasoning, and the same trap, as the unconstrained version's equivalent test:
    a single run cannot tell "stopped early because the threshold is loose" apart
    from "the comparison is backwards"; comparing a loose against a tight
    threshold on one fixture can - but only if the two thresholds actually land
    on different sides of every pass's improvement fraction. This fixture's
    residual falls to exactly zero after its second pass, so a threshold as
    loose as 0.8 turned out to still separate identically from 0.05 either way
    round; 0.99 (looser than the very first pass's ~93% improvement) against
    0.01 (tighter than a zero delta can ever satisfy, so it runs to the full
    budget) is what actually pins the comparison's direction down.
    """
    clouds, transforms = _turntable_drifted_sweep()

    loose = refine_on_turntable(
        clouds,
        transforms,
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=6.0,
        iterations=6,
        sample=80,
        min_improvement=0.99,
    )
    tight = refine_on_turntable(
        clouds,
        transforms,
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=6.0,
        iterations=6,
        sample=80,
        min_improvement=0.01,
    )

    assert loose.iterations_run <= tight.iterations_run
    assert loose.iterations_run < 6, "the fixture must actually demonstrate an early stop"
    assert tight.iterations_run == 6, "and the tight threshold must actually run to the budget"


def test_worst_correction_is_the_largest_magnitude_correction():
    zero = refine_on_turntable(
        *_turntable_drifted_sweep(),
        direction=_SWEEP_AXIS,
        through=_SWEEP_THROUGH,
        tolerance=1e-9,
        iterations=2,
        sample=60,
    )
    nonzero = _turntable_refined_result()

    assert zero.worst_correction == 0.0, "nothing moved, so nothing to report as worst"
    assert nonzero.worst_correction == max(abs(c) for c in nonzero.corrections)
    assert nonzero.worst_correction > 0.0


def test_refine_on_turntable_refuses_invalid_input():
    clouds, transforms = _clean_sweep()
    through: Point = _SWEEP_THROUGH

    with pytest.raises(MultiviewNotAllowed, match="one each"):
        refine_on_turntable(
            clouds, transforms[:-1], direction=_SWEEP_AXIS, through=through, tolerance=1.0
        )
    with pytest.raises(MultiviewNotAllowed, match="at least three frames"):
        refine_on_turntable(
            clouds[:2], transforms[:2], direction=_SWEEP_AXIS, through=through, tolerance=1.0
        )
    with pytest.raises(MultiviewNotAllowed, match="not a frame index"):
        refine_on_turntable(
            clouds, transforms, direction=_SWEEP_AXIS, through=through, tolerance=1.0, anchor=-1
        )
    with pytest.raises(MultiviewNotAllowed, match="not a frame index"):
        refine_on_turntable(
            clouds,
            transforms,
            direction=_SWEEP_AXIS,
            through=through,
            tolerance=1.0,
            anchor=len(clouds),
        )
    with pytest.raises(MultiviewNotAllowed, match="iterations must be at least 1"):
        refine_on_turntable(
            clouds, transforms, direction=_SWEEP_AXIS, through=through, tolerance=1.0, iterations=0
        )
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        refine_on_turntable(
            clouds, transforms, direction=_SWEEP_AXIS, through=through, tolerance=0.0
        )
    with pytest.raises(MultiviewNotAllowed, match="positive"):
        refine_on_turntable(
            clouds, transforms, direction=_SWEEP_AXIS, through=through, tolerance=-1.0
        )
    with pytest.raises(MultiviewNotAllowed, match="finite non-zero"):
        refine_on_turntable(
            clouds, transforms, direction=(0.0, 0.0, 0.0), through=through, tolerance=1.0
        )
    with pytest.raises(MultiviewNotAllowed, match="finite non-zero"):
        refine_on_turntable(
            clouds, transforms, direction=(math.nan, 1.0, 0.0), through=through, tolerance=1.0
        )


def test_refine_on_turntable_as_dict_carries_schema_version_and_corrections_in_degrees():
    result = _turntable_refined_result()

    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["corrections_deg"] == [math.degrees(c) for c in result.corrections]
    assert record["worst_correction_deg"] == pytest.approx(math.degrees(result.worst_correction))
    assert record["limitations"], "some caveat must travel with every result"
