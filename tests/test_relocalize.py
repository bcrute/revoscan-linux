"""Tests for :mod:`scanner3d.relocalize` - the decision that lets a resumed

capture rejoin a model it was picked up away from.

What is at stake
-----------------
:mod:`~scanner3d.relocalize` exists because a global search over a repositioned
scanner's frames ALWAYS finds a best candidate, including for a segment that
shares nothing with the model, and turning "best" into "accepted" is exactly
the mistake that stitches two halves of a scan together at a plausible-looking
angle that is simply wrong. Every gate in the module - coarse-search
ambiguity, overlap, residual, matched fraction, used fraction, runner-up
margin, per-frame agreement - is a named threshold with a measurement beside
it, and this file exists to prove each one is load-bearing on real geometry,
not merely present in the source. The coarse-search gate is the newest and the
bluntest of them: :func:`~scanner3d.multiview.coarse_align` reporting
``unambiguous=False`` - its best orientation within the named
``AMBIGUITY_MARGIN`` of a genuinely different one, recorded here as
``coarse_ambiguity_margin`` - is a refusal on its own, and no refined
measurement taken afterwards can overturn it.

Two properties matter more than any other and get the most weight here:

* **A large, honest recovery on genuinely asymmetric geometry.** A test that
  only proves itself on a gentle nudge proves nothing a caller could not
  already get from adjacent-frame ICP - the whole point of this module is the
  case adjacent ICP cannot handle, so the acceptance tests below use a
  100-degree-plus rotation with a real translation. See
  :func:`test_relocalize_accepts_a_large_reposition_and_recovers_the_known_transform`.
* **Honest refusal**, on every distinct way a caller can hand this module
  something it should not trust: too few frames, no shared surface, partial
  overlap, a symmetric object, a shape whose COARSE search could not separate
  its best placement from a genuinely different one, and - the one specific to
  this module rather than to coarse alignment generally - one bad frame hiding
  inside an otherwise-good segment. The coarse-ambiguity refusal is what this
  file's newest regressions are about, because the bridge that provoked them
  cleared every refined gate and was accepted while its own evidence said
  ``coarse_unambiguous: false``. See
  :func:`test_coarse_ambiguity_refuses_even_when_every_other_gate_passes` and
  :func:`test_a_coarse_ambiguous_shape_is_refused_rather_than_placed_the_wrong_way_up`;
  for the one-bad-frame case, see
  :func:`test_one_disagreeing_tentative_frame_refuses_even_though_the_union_would_pass`.

None of this establishes that an accepted bridge is *correct*, only that the
evidence cleared conservative thresholds - see the module's own docstring for
what it does not claim. Every test cloud's extent is a bare number with no
physical unit, for the same reason :mod:`scanner3d.multiview`'s tests carry
none: see that module's docstring.

Test geometry
-------------
Two shapes, and which one a test uses is the point rather than a detail.

:func:`_lobed_object` is the ACCEPTING geometry: a Fibonacci ellipsoid with
three different axis lengths and one lobe grown out of a single off-axis
direction, so its OUTLINE - not merely its point pattern - is different from
every direction. It is the same shape
:mod:`tests.test_gui_pause` drives the end-to-end reposition with, for the same
reason. The reference is the whole object as a scan before a pause would have
it; the tentative segment is the part of it a scanner sees from the new
position, sampled at a different Fibonacci count so the two clouds describe one
surface with different points rather than repeating each other exactly.

:func:`_asymmetric_box` is the REFUSING geometry. A plain rectangular box, even
with three distinct half-extents, is centrally symmetric under a 180-degree turn
about any of its three principal axes, because every pair of opposite faces is
congruent. The fused four-view render breaks that symmetry in the point PATTERN
(each face is sampled from a different simulated viewpoint), which is enough for
:mod:`tests.test_multiview`'s exact recoveries - and measurably not enough for
:func:`~scanner3d.multiview.coarse_align`, which ranks decimated clouds on a
coarse grid and separates this box's best orientation from its runner-up by only
1.02x to 1.08x, under the 1.15 margin. Overlapping slices of one such box are
therefore a shape that genuinely sits several ways on itself: the module refuses
them now, and
:func:`test_a_coarse_ambiguous_shape_is_refused_rather_than_placed_the_wrong_way_up`
records what the module used to do with one instead - accept it, at a placement
180 degrees from the truth.

Nothing here uses the ``random`` module: every cloud is a deterministic
lattice or a Fibonacci sphere, so a failure reproduces identically on every
run and every machine.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Final, cast

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import Point, Rigid, normalize, rotation_about_axis
from scanner3d.multiview import AMBIGUITY_MARGIN, CoarseAlignment, coarse_align
from scanner3d.pipeline import NOISE_FLOOR
from scanner3d.registration import cloud_extent, voxel_downsample
from scanner3d.relocalize import (
    LOCAL_MIN_MATCHED_FRACTION,
    LOCAL_MIN_USED_FRACTION,
    LOCAL_RESIDUAL_MULTIPLE,
    MAX_TENTATIVE_FRAMES,
    MIN_FRAME_OVERLAP,
    MIN_MATCHED_FRACTION,
    MIN_OVERLAP,
    MIN_TENTATIVE_FRAMES,
    MIN_USED_FRACTION,
    RESIDUAL_MULTIPLE,
    RUNNER_UP_MARGIN,
    LocalFit,
    Relocalization,
    RelocalizationNotAllowed,
    as_record,
    bounded_reference,
    evidence_summary,
    judge_local_edge,
    local_chain_gates,
    relocalize,
    segment_union,
    thresholds,
)
from scanner3d.simulate import Box, points_of, render

#: The identity rotation, spelled out where a deliberately broken transform is
#: built beside it and the contrast is the point.
_IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# --------------------------------------------------------------------------
# Test geometry
# --------------------------------------------------------------------------


def _fused_box_cloud(half_extents: Point, *, distance: float = 150.0) -> tuple[Point, ...]:
    """Four single-view renders of a box, merged into one shared object frame.

    Identical in construction to :mod:`tests.test_multiview`'s own fixture of
    the same name and for the same reason: a rendered box is asymmetric in a
    way a mathematically perfect box is not, because each face is sampled
    from wherever a simulated camera actually saw it, and that sampling
    pattern does not repeat under any of the box's own 180-degree symmetries.
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
def _asymmetric_box() -> tuple[Point, ...]:
    """~366 points, three distinct half-extents, asymmetric only in its SAMPLING.

    Named for what it was believed to be. It is asymmetric enough for ICP given
    a start near the truth - which is what :mod:`tests.test_multiview` uses it
    for - and it is not asymmetric enough for a coarse ranking to say which way
    up it is, which is why every use of it below is a refusal.
    """
    raw = _fused_box_cloud((30.0, 12.0, 5.0))
    return voxel_downsample(raw, cloud_extent(raw) / 25.0)


#: Points in the sampling of :func:`_lobed_object` that stands for the accepted
#: model, and in the one that stands for what the scanner sees after the
#: reposition. Different on purpose: two scans of a surface never return the
#: same points, and a segment that is a byte-exact subset of its own reference
#: aligns to a residual of 1e-12, which is a number no gate can be straddled
#: around and no real capture will ever produce.
_MODEL_SAMPLES: Final = 460
_SEGMENT_SAMPLES: Final = 397

#: The far end of the object the tentative segment does NOT see, so that the
#: segment is a view of the model rather than a copy of it.
_SEGMENT_CUT: Final = 20.0


def _lobed_object(count: int) -> tuple[Point, ...]:
    """An ellipsoid with three different axis lengths and one off-axis lobe.

    The accepting geometry, and the asymmetry is the whole of its job. A shape
    that maps onto itself under a half turn offers several equally good
    placements, and :mod:`~scanner3d.relocalize` refuses those outright - so a
    symmetric test object cannot prove that a correct placement is FOUND, only
    that an ambiguous one is declined. Measured on this shape, the coarse
    ranking separates the true orientation from the best rival by 1.49x against
    the 1.15 margin it has to clear.

    Deterministic: a Fibonacci lattice, so ``count`` fixes every point and a
    failure reproduces identically on every machine.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        x, z = ring * math.cos(angle), ring * math.sin(angle)
        lobe = 1.0 + 0.9 * max(0.0, 0.60 * x + 0.50 * y + 0.62 * z)
        points.append((15.0 * x * lobe, 9.0 * y * lobe, 6.0 * z * lobe))
    return tuple(points)


#: A large, honest reposition: over 100 degrees about an oblique axis with a
#: real translation, not a nudge ICP could plausibly have found on its own.
_TRUTH: Rigid = Rigid.of(
    rotation_about_axis(normalize((0.3, 0.7, 0.64)), math.radians(130.0)).rotation,
    (50.0, -40.0, 30.0),
)


def _split_frames(points: tuple[Point, ...], count: int) -> list[tuple[Point, ...]]:
    """``points`` divided into ``count`` interleaved tentative frames.

    Interleaving rather than contiguous chunks so every frame samples the
    whole slice's shape, not just one end of it.
    """
    return [points[index::count] for index in range(count)]


class _Scene:
    """The reusable accepting case: reference, a good tentative segment, and truth.

    The reference is the whole object, because that is what an accepted model is
    by the time a pause is even offered - eight or more frames all the way round
    it. The tentative segment is the ``x <= _SEGMENT_CUT`` part of a DIFFERENT
    sampling of the same surface, moved by ``_TRUTH``: a scanner put down the
    other way round sees most of what was already scanned, from somewhere else,
    with its own points rather than the model's.

    That shape - most of a distinctive object, sampled independently - is what
    the module is built to accept, and what its refusal wording asks the
    operator to produce. A segment that shares only a slice of the model is a
    separate case with its own tests
    (:func:`test_low_partial_overlap_refuses`), and it refuses.
    """

    def __init__(self) -> None:
        model = _lobed_object(_MODEL_SAMPLES)
        segment = _lobed_object(_SEGMENT_SAMPLES)
        self.reference = bounded_reference([model], [Rigid.identity()])
        seen = tuple(p for p in segment if p[0] <= _SEGMENT_CUT)
        local = tuple(_TRUTH.inverse().apply(p) for p in seen)
        self.good_clouds = _split_frames(local, 4)
        self.good_poses = [Rigid.identity() for _ in self.good_clouds]


@functools.cache
def _scene() -> _Scene:
    return _Scene()


@functools.cache
def _accepted_result() -> Relocalization:
    """The one call every threshold-straddling test patches around."""
    scene = _scene()
    return relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")


def _sphere_cloud(
    *, radius: float = 20.0, count: int = 200, centre: Point = (500.0, 500.0, 500.0)
) -> tuple[Point, ...]:
    """An evenly-sampled sphere a long way from the box, by the usual Fibonacci construction."""
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        points.append(
            (
                centre[0] + radius * ring * math.cos(angle),
                centre[1] + radius * y,
                centre[2] + radius * ring * math.sin(angle),
            )
        )
    return tuple(points)


# --------------------------------------------------------------------------
# 1. The headline recovery
# --------------------------------------------------------------------------


def test_relocalize_accepts_a_large_reposition_and_recovers_the_known_transform():
    """The module's entire reason to exist: a 130-degree, 70-unit reposition, recovered.

    If this does not hold - or holds only for small nudges - nothing else in
    this file matters. ``rotation_angle`` and ``translation_magnitude`` on
    ``bridge.inverse().compose(truth)`` are the same pose-difference idiom
    :mod:`tests.test_icp_plane` uses. The scene is :class:`_Scene`, built on
    :func:`_lobed_object` rather than on a box: the lobe is what makes one
    placement of the segment fit better than every other, which is why the
    coarse search separates this shape by 1.49x against its 1.15 margin where
    :func:`_asymmetric_box` does not, and so why this scene is allowed to be
    the accepting one at all.

    Both bounds are envelopes rather than the precision achieved. Measured on
    this geometry, the recovered bridge sits 0.10 degrees and 0.14 units from a
    truth 130 degrees and 71 units away, so each assertion has room for the
    backend and the noise floor to move without becoming a test of the exact
    arithmetic. The rotation bound is the load-bearing half: one degree about
    the centre of an object roughly 40 units across displaces its far points by
    about a third of a unit, so a placement that clears it cannot be seated
    anything like the 20 units the translation bound still allows. The
    translation tolerance is the coarse backstop against a candidate that is
    turned the right way and sitting in the wrong place.
    """
    scene = _scene()

    result = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")

    assert result.accepted, result.refusals
    assert result.bridge is not None
    assert result.coarse_unambiguous is True, (
        "this scene has to be one the coarse search itself separates; an accepting "
        "case that started ambiguous would be accepting the very thing the module refuses"
    )
    difference = result.bridge.inverse().compose(_TRUTH)
    assert math.degrees(difference.rotation_angle) < 1.0, (
        f"recovered bridge disagreed with the known truth by "
        f"{math.degrees(difference.rotation_angle):.3f} degrees of rotation"
    )
    assert difference.translation_magnitude < 20.0, (
        f"recovered bridge disagreed with the known truth by "
        f"{difference.translation_magnitude:.3f} units of translation"
    )


def test_accepted_evidence_is_populated_and_every_frame_clears_its_own_gate():
    """The accepted record carries the numbers an operator would want to read back.

    Not just that a bridge exists: the per-candidate evidence and the
    per-frame overlaps are what :func:`evidence_summary` and a durable bridge
    record are built from, so an accepted result with a hole in either is as
    unusable as a refusal with no reason.
    """
    result = _accepted_result()
    assert result.accepted
    best = result.best
    assert best is not None

    assert math.isfinite(best.overlap)
    assert math.isfinite(best.rmse)
    assert math.isfinite(best.matched_fraction)
    assert math.isfinite(best.used_fraction)
    assert best.converged is True
    assert result.frame_overlaps
    assert all(overlap >= MIN_FRAME_OVERLAP for overlap in result.frame_overlaps), (
        f"frame_overlaps {result.frame_overlaps} include one under the "
        f"{MIN_FRAME_OVERLAP} gate on an ACCEPTED result"
    )

    summary = evidence_summary(result)
    assert "relocalized" in summary
    assert f"{best.overlap:.2f}" in summary


# --------------------------------------------------------------------------
# 2. Refusals: too little to decide from
# --------------------------------------------------------------------------


def test_fewer_than_the_minimum_tentative_frames_refuses_naming_the_count():
    """One or two views are not a surface - the module says exactly how many it got."""
    scene = _scene()
    clouds = scene.good_clouds[: MIN_TENTATIVE_FRAMES - 1]
    poses = scene.good_poses[: MIN_TENTATIVE_FRAMES - 1]

    result = relocalize(clouds, poses, scene.reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    assert str(MIN_TENTATIVE_FRAMES - 1) in result.refusals[0]
    assert str(MIN_TENTATIVE_FRAMES) in result.refusals[0]


def test_no_shared_surface_at_all_refuses():
    """A segment describing a different object entirely has nothing to be decided from.

    A sphere a long way from the box shares essentially no surface with it
    under any rigid placement, so the coarse-to-fine pipeline runs to
    completion and still refuses - on overlap specifically, among others.
    """
    scene = _scene()
    sphere = _sphere_cloud()
    clouds = _split_frames(sphere, 4)
    poses = [Rigid.identity() for _ in clouds]

    result = relocalize(clouds, poses, scene.reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    assert result.best is not None
    assert result.best.overlap < MIN_OVERLAP
    joined = " ".join(result.refusals)
    assert "overlap" in joined


def test_low_partial_overlap_refuses():
    """A segment that only clips the edge of the accepted model is refused, not guessed.

    Reference and segment here are drawn from slices of the SAME box that
    barely reach each other (``x >= 18`` against ``x <= -3``), so what little
    correspondence ICP finds is real but too small a fraction of either side
    to found a decision on. Measured on this fixture: the best candidate covers
    0.30 of the segment against the 0.35 gate. The ranking is ALSO ambiguous
    here - two slivers of one box sit several ways against each other - and
    both reasons are recorded; what this test asserts is that the overlap gate
    is one of them and that nothing was accepted.
    """
    box = _asymmetric_box()
    model_subset = tuple(p for p in box if p[0] >= 18.0)
    segment_subset = tuple(p for p in box if p[0] <= -3.0)
    reference = bounded_reference([model_subset], [Rigid.identity()])
    local = tuple(_TRUTH.inverse().apply(p) for p in segment_subset)
    clouds = _split_frames(local, 4)
    poses = [Rigid.identity() for _ in clouds]

    result = relocalize(clouds, poses, reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    assert result.best is not None
    assert result.best.overlap < MIN_OVERLAP
    joined = " ".join(result.refusals)
    assert "overlap" in joined


def test_a_symmetric_object_refuses_and_names_the_margin_or_symmetry():
    """The module's most important property, restated for the full pipeline.

    A square prism, invariant under its own quarter turn (see
    :mod:`tests.test_multiview`'s identical fixture), aligned against a
    NOISY copy of itself so the four rotationally-tied candidates carry a
    real, non-floating-point-scale residual rather than a bit-identical one -
    without the noise, the identity candidate wins by sheer numerical luck
    (its residual is float noise, ~1e-16, against the true 90-degree
    candidate's ~1e-14) and the ambiguity never gets a chance to show. A tiny
    amount of injected noise, at this scanner's own measured noise floor, is
    what makes the four candidates competitors instead of a coincidence.
    """
    import random

    def symmetric_square_prism(
        *,
        half_side: float = 4.0,
        half_length: float = 25.0,
        edge_points: int = 12,
        z_layers: int = 15,
    ) -> tuple[Point, ...]:
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

    def with_noise(points: tuple[Point, ...], *, seed: int) -> tuple[Point, ...]:
        rng = random.Random(seed)
        return tuple(
            (
                p[0] + rng.gauss(0.0, NOISE_FLOOR),
                p[1] + rng.gauss(0.0, NOISE_FLOOR),
                p[2] + rng.gauss(0.0, NOISE_FLOOR),
            )
            for p in points
        )

    prism = symmetric_square_prism()
    quarter_turn = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2.0)
    reference = bounded_reference([with_noise(prism, seed=1)], [Rigid.identity()])
    local = tuple(quarter_turn.inverse().apply(p) for p in with_noise(prism, seed=2))
    clouds = _split_frames(local, 4)
    poses = [Rigid.identity() for _ in clouds]

    result = relocalize(clouds, poses, reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    joined = " ".join(result.refusals)
    assert "margin" in joined or "symmetr" in joined


def test_a_coarse_ambiguous_shape_is_refused_rather_than_placed_the_wrong_way_up():
    """The 2026-08-28 defect, in the geometry that produces it.

    Two overlapping slices of :func:`_asymmetric_box`. Every refined gate this
    module owns passes on them, and the placement the refinement settles on is
    180 degrees from the truth - the object upside down, at a residual that
    looks exactly like a residual of a placement nobody doubted. The only
    evidence that anything is wrong is the coarse search saying it could not
    tell the best orientation from the runner-up, which is what this now
    refuses on.

    The wrong transform is asserted, not just the refusal, because a test that
    only checked "refused" would go on passing if the geometry drifted into a
    case where the answer was right anyway and prove nothing about the danger.
    """
    box = _asymmetric_box()
    model_subset = tuple(p for p in box if p[0] >= -20.0)
    segment_subset = tuple(p for p in box if p[0] <= 20.0)
    reference = bounded_reference([model_subset], [Rigid.identity()])
    local = tuple(_TRUTH.inverse().apply(p) for p in segment_subset)
    clouds = _split_frames(local, 4)
    poses = [Rigid.identity() for _ in clouds]

    result = relocalize(clouds, poses, reference, backend="python")

    assert result.coarse_unambiguous is False
    assert result.accepted is False
    assert result.bridge is None
    # What it would have been given, had the margin been allowed to speak for
    # the decision: the object placed the other way up.
    best = result.best
    assert best is not None
    wrong = best.transform.inverse().compose(_TRUTH)
    assert math.degrees(wrong.rotation_angle) > 90.0, (
        "this scene is only worth testing while its best refined candidate is "
        f"grossly wrong; it is now {math.degrees(wrong.rotation_angle):.1f} degrees out"
    )
    joined = " ".join(result.refusals)
    assert "coarse search" in joined
    assert "sits several ways" in joined


def test_coarse_ambiguity_refuses_even_when_every_other_gate_passes(monkeypatch):
    """The gate cannot be argued out of by the numbers measured after it.

    The accepting scene, with nothing changed except the coarse search's own
    report of whether it separated its winner from the runner-up. Every refined
    gate still passes - each one is asserted below against its own measured
    value - and the decision is still a refusal with no bridge, because those
    numbers are measurements of where the optimiser stopped and not of which
    placement is right.

    The runner-up margin is the one that matters here. The bridge accepted on
    2026-08-28 carried a margin of 1.50 against a 1.25 gate, five chained
    tentative frames and a residual of 0.515, and was wrong: its own record said
    ``coarse_unambiguous: false``, and the candidate it beat covered more of the
    model than it did (0.598 against 0.483). A comfortable margin is what that
    failure looked like from the inside.
    """
    scene = _scene()

    def ambiguous(source: Sequence[Point], target: Sequence[Point]) -> CoarseAlignment:
        """The real ranking, reporting the one thing this test is about."""
        return replace(coarse_align(source, target), unambiguous=False)

    monkeypatch.setattr("scanner3d.relocalize.coarse_align", ambiguous)
    result = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    assert result.coarse_unambiguous is False
    # Every other gate cleared on its own measured number, so the refusal below
    # is the ONLY thing standing between this segment and a published bridge.
    best = result.best
    assert best is not None
    assert best.converged is True
    assert best.rmse <= RESIDUAL_MULTIPLE * NOISE_FLOOR
    assert best.matched_fraction >= MIN_MATCHED_FRACTION
    assert best.used_fraction >= MIN_USED_FRACTION
    assert best.overlap >= MIN_OVERLAP
    assert result.runner_up_margin >= RUNNER_UP_MARGIN
    assert all(overlap >= MIN_FRAME_OVERLAP for overlap in result.frame_overlaps)
    assert len(result.refusals) == 1, result.refusals
    assert str(AMBIGUITY_MARGIN) in result.refusals[0]
    # And it is the operator's instruction, not a diagnosis they cannot act on.
    assert "does NOT look like the rest of it" in result.refusals[0]


def test_an_ambiguous_refusal_keeps_the_evidence_a_durable_record_needs(monkeypatch):
    """Refused, but not silent: the candidates were still refined and recorded.

    A refusal a reader cannot inspect is not evidence. The coarse ambiguity
    ends the DECISION, so the record carries the same populated candidate list,
    per-frame overlaps and margin an acceptance would have carried - and a null
    bridge beside them.
    """
    scene = _scene()

    def ambiguous(source: Sequence[Point], target: Sequence[Point]) -> CoarseAlignment:
        """The real ranking, reporting the one thing this test is about."""
        return replace(coarse_align(source, target), unambiguous=False)

    monkeypatch.setattr("scanner3d.relocalize.coarse_align", ambiguous)
    result = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")

    assert result.candidates, "an ambiguous coarse search still refines what it ranked"
    assert len(result.frame_overlaps) == len(scene.good_clouds)
    assert result.segment_points > 0
    assert result.reference_points > 0
    # The claim this replaced: the refinement disambiguated a weak start, said
    # as a limitation beside an accepted bridge. It is a refusal now, and it
    # must not also be sitting in the list a reader treats as "accepted, with
    # caveats".
    assert not any("coarse ranking" in limitation for limitation in result.limitations)
    assert evidence_summary(result).startswith("not yet placed: the coarse search")

    record = json.loads(json.dumps(as_record(result), allow_nan=False))
    assert record["accepted"] is False
    assert record["bridge"] is None
    assert record["coarse_unambiguous"] is False
    assert record["candidates"]
    assert record["thresholds"]["coarse_ambiguity_margin"] == AMBIGUITY_MARGIN


def test_one_disagreeing_tentative_frame_refuses_even_though_the_union_would_pass():
    """The per-frame gate exists for exactly this: a union score carried by a majority.

    Four good frames plus one small frame given a wildly wrong LOCAL pose (a
    90-degree turn no other frame shares): the eight stray points are too few
    to drag the union's residual, matched fraction, used fraction, overlap or
    runner-up margin below their own gates - every one of those still passes
    - but that one frame's own overlap against the reference is zero, and
    that alone is what this test proves refuses it.
    """
    scene = _scene()
    # Eight of the segment's own points, so the frame is real geometry the
    # local chain mis-posed rather than a foreign object smuggled in.
    bad_local = scene.good_clouds[0][:8]
    bad_pose = rotation_about_axis((0.0, 1.0, 0.0), math.radians(90.0))

    clouds = [*scene.good_clouds, bad_local]
    poses = [*scene.good_poses, bad_pose]

    result = relocalize(clouds, poses, scene.reference, backend="python")

    assert result.accepted is False
    assert result.bridge is None
    assert result.best is not None
    # The union-level gates all passed on their own measured numbers.
    assert result.best.rmse <= RESIDUAL_MULTIPLE * NOISE_FLOOR
    assert result.best.matched_fraction >= MIN_MATCHED_FRACTION
    assert result.best.used_fraction >= MIN_USED_FRACTION
    assert result.best.overlap >= MIN_OVERLAP
    assert result.runner_up_margin >= RUNNER_UP_MARGIN
    assert result.best.converged is True
    # Only the last frame - the corrupted one - is under the per-frame gate.
    assert result.frame_overlaps[-1] < MIN_FRAME_OVERLAP
    assert all(overlap >= MIN_FRAME_OVERLAP for overlap in result.frame_overlaps[:-1])
    joined = " ".join(result.refusals)
    assert "tentative frames overlap" in joined


# --------------------------------------------------------------------------
# 3. Degenerate and non-finite input: relocalize must never raise
# --------------------------------------------------------------------------


def test_relocalize_never_raises_on_an_empty_reference():
    scene = _scene()
    result = relocalize(scene.good_clouds, scene.good_poses, [])
    assert isinstance(result, Relocalization)
    assert result.accepted is False
    assert result.bridge is None


def test_relocalize_never_raises_on_an_empty_segment_cloud():
    scene = _scene()
    empty_clouds = [() for _ in scene.good_clouds]
    result = relocalize(empty_clouds, scene.good_poses, scene.reference)
    assert isinstance(result, Relocalization)
    assert result.accepted is False
    assert result.bridge is None


def test_relocalize_never_raises_on_a_collinear_reference():
    """A reference with no orientation to find - refused, not crashed through."""
    scene = _scene()
    collinear = tuple((float(index), 0.0, 0.0) for index in range(100))
    result = relocalize(scene.good_clouds, scene.good_poses, collinear)
    assert isinstance(result, Relocalization)
    assert result.accepted is False
    assert result.bridge is None
    assert "does not span three dimensions" in result.refusals[0]


def test_relocalize_never_raises_when_clouds_and_poses_do_not_pair_up():
    scene = _scene()
    result = relocalize(scene.good_clouds, scene.good_poses[:-1], scene.reference)
    assert isinstance(result, Relocalization)
    assert result.accepted is False
    assert result.bridge is None


# --------------------------------------------------------------------------
# 4. Threshold straddling: proving each gate is load-bearing on one fixed input
# --------------------------------------------------------------------------
#
# Every test below runs the SAME accepting scene, patches exactly one gate
# constant to just outside the measured value that scene produced, and checks
# the decision flips - then patches it to just inside and checks it does not.
# The measured values come from the unpatched call itself rather than a
# literal copied into the test, so a change to the fixture geometry cannot
# silently leave these straddling the wrong number.


def test_residual_multiple_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    assert baseline.best is not None
    measured_multiple = baseline.best.rmse / NOISE_FLOOR

    monkeypatch.setattr("scanner3d.relocalize.RESIDUAL_MULTIPLE", measured_multiple * 0.5)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False, "a residual gate below the measured residual must refuse"

    monkeypatch.setattr("scanner3d.relocalize.RESIDUAL_MULTIPLE", measured_multiple * 2.0)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True, "a residual gate above the measured residual must accept"


def test_min_matched_fraction_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    assert baseline.best is not None
    measured = baseline.best.matched_fraction

    monkeypatch.setattr("scanner3d.relocalize.MIN_MATCHED_FRACTION", measured + 0.01)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False

    monkeypatch.setattr("scanner3d.relocalize.MIN_MATCHED_FRACTION", measured - 0.01)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True


def test_min_used_fraction_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    assert baseline.best is not None
    measured = baseline.best.used_fraction

    monkeypatch.setattr("scanner3d.relocalize.MIN_USED_FRACTION", measured + 0.01)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False

    monkeypatch.setattr("scanner3d.relocalize.MIN_USED_FRACTION", measured - 0.01)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True


def test_min_overlap_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    assert baseline.best is not None
    measured = baseline.best.overlap

    monkeypatch.setattr("scanner3d.relocalize.MIN_OVERLAP", measured + 0.01)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False

    monkeypatch.setattr("scanner3d.relocalize.MIN_OVERLAP", measured - 0.01)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True


def test_min_frame_overlap_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    measured_worst = min(baseline.frame_overlaps)

    monkeypatch.setattr("scanner3d.relocalize.MIN_FRAME_OVERLAP", measured_worst + 0.01)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False

    monkeypatch.setattr("scanner3d.relocalize.MIN_FRAME_OVERLAP", measured_worst - 0.01)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True


def test_runner_up_margin_gate_is_load_bearing(monkeypatch):
    baseline = _accepted_result()
    measured = baseline.runner_up_margin
    assert math.isfinite(measured), "this scene must have a genuine (finite) rival to straddle"

    monkeypatch.setattr("scanner3d.relocalize.RUNNER_UP_MARGIN", measured + 1.0)
    scene = _scene()
    tightened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert tightened.accepted is False

    monkeypatch.setattr("scanner3d.relocalize.RUNNER_UP_MARGIN", measured - 1.0)
    loosened = relocalize(scene.good_clouds, scene.good_poses, scene.reference, backend="python")
    assert loosened.accepted is True


# --------------------------------------------------------------------------
# 5. bounded_reference and segment_union
# --------------------------------------------------------------------------


def test_bounded_reference_is_bounded_by_its_limit():
    box = _asymmetric_box()
    reference = bounded_reference([box], [Rigid.identity()], limit=40)
    assert 0 < len(reference) <= 40


def test_bounded_reference_draws_from_every_accepted_frame_not_just_the_last_few():
    """A point from the FIRST accepted frame survives among twenty, placed by its pose.

    The whole design of :func:`bounded_reference` is that the reference is
    NOT dominated by the most recently accepted frames, because the surface a
    resumed segment overlaps is very often one scanned minutes earlier. Each
    frame here carries one point at its own local origin and a pose that
    translates it to a distinct, identifiable location, so a reference built
    only from the tail of the list could not possibly contain frame 0's
    point.
    """
    frame_count = 20
    clouds: list[tuple[Point, ...]] = [((0.0, 0.0, 0.0),) for _ in range(frame_count)]
    identity_rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    poses = [
        Rigid.of(identity_rotation, (float(index) * 10.0, 0.0, 0.0)) for index in range(frame_count)
    ]

    reference = bounded_reference(clouds, poses, limit=128)

    assert (0.0, 0.0, 0.0) in reference, "the first accepted frame's point did not survive"
    # And it is placed by ITS OWN pose, not some other frame's.
    assert (190.0, 0.0, 0.0) in reference, "the last accepted frame's point did not survive"


def test_bounded_reference_refuses_mismatched_clouds_and_poses_without_raising_past_the_boundary():
    with pytest.raises(RelocalizationNotAllowed):
        bounded_reference([[(0.0, 0.0, 0.0)]], [Rigid.identity(), Rigid.identity()])


def test_segment_union_is_bounded():
    box = _asymmetric_box()
    clouds = _split_frames(box, 4)
    poses = [Rigid.identity() for _ in clouds]

    union = segment_union(clouds, poses, limit=50)

    assert 0 < len(union) <= 50


def test_segment_union_places_each_frame_by_its_own_local_pose():
    """The union is the segment's OWN local frame - each cloud placed by its chained pose.

    Two single-point frames at the same raw sensor coordinate but with
    different chained poses land at two different places in the union,
    exactly where their own pose puts them.
    """
    clouds: tuple[tuple[Point, ...], ...] = (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),))
    poses = (
        Rigid.identity(),
        Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (100.0, 0.0, 0.0)),
    )

    union = segment_union(clouds, poses)

    assert (0.0, 0.0, 0.0) in union
    assert (100.0, 0.0, 0.0) in union


# --------------------------------------------------------------------------
# 6. Records: thresholds(), as_dict(), and the doctor status
# --------------------------------------------------------------------------


def test_thresholds_names_every_gate_constant():
    record = thresholds()

    assert record["residual_multiple"] == RESIDUAL_MULTIPLE
    assert record["min_matched_fraction"] == MIN_MATCHED_FRACTION
    assert record["min_used_fraction"] == MIN_USED_FRACTION
    assert record["min_overlap"] == MIN_OVERLAP
    assert record["min_frame_overlap"] == MIN_FRAME_OVERLAP
    assert record["runner_up_margin"] == RUNNER_UP_MARGIN
    assert record["coarse_ambiguity_margin"] == AMBIGUITY_MARGIN
    assert record["min_tentative_frames"] == MIN_TENTATIVE_FRAMES
    assert record["max_tentative_frames"] == MAX_TENTATIVE_FRAMES
    assert record["noise_floor"] == NOISE_FLOOR


def test_thresholds_round_trips_through_json_with_no_nan_allowed():
    record = json.loads(json.dumps(thresholds(), allow_nan=False))
    assert record["min_overlap"] == MIN_OVERLAP


def test_accepted_relocalization_as_dict_round_trips_through_json():
    result = _accepted_result()
    assert result.accepted

    record = json.loads(json.dumps(as_record(result), allow_nan=False))

    assert record["accepted"] is True
    assert record["bridge"] is not None
    assert record["thresholds"]["min_overlap"] == MIN_OVERLAP


def test_refused_relocalization_as_dict_round_trips_through_json_with_a_null_bridge():
    """A refusal that found a genuine, finite-margin rival round-trips cleanly.

    The companion case - a refusal with NO rival at all, whose
    ``runner_up_margin`` is ``inf`` - is covered by
    :func:`test_a_refusal_with_no_rival_serialises_its_margin_as_null` below.
    ``inf`` is not JSON, so :meth:`Relocalization.as_dict` puts it through
    :func:`~scanner3d.relocalize._json_number` and writes ``null`` beside a
    ``had_runner_up`` flag; both shapes therefore survive
    ``json.dumps(..., allow_nan=False)``, which is what a durable bridge
    record is written with.
    """
    scene = _scene()
    # Eight of the segment's own points, so the frame is real geometry the
    # local chain mis-posed rather than a foreign object smuggled in.
    bad_local = scene.good_clouds[0][:8]
    bad_pose = rotation_about_axis((0.0, 1.0, 0.0), math.radians(90.0))
    clouds = [*scene.good_clouds, bad_local]
    poses = [*scene.good_poses, bad_pose]

    result = relocalize(clouds, poses, scene.reference, backend="python")
    assert result.accepted is False
    assert math.isfinite(result.runner_up_margin)

    record = json.loads(json.dumps(as_record(result), allow_nan=False))

    assert record["accepted"] is False
    assert record["bridge"] is None
    assert record["refusals"]


def test_a_refusal_with_no_rival_serialises_its_margin_as_null():
    """A margin against a rival that does not exist is not a quantity, and is written.

    A refusal reached before any candidate was refined - too few tentative
    frames is the plainest of them - carries ``runner_up_margin == inf``,
    because a margin against a rival that does not exist is not a quantity.
    The durable bridge record is written with ``allow_nan=False``, so
    embedding that float unsanitised would make exactly the records taken at
    the least confident moments the ones that cannot be read back. ``null``
    plus ``had_runner_up: false`` is the honest pair: not measurable, and
    which of the two reasons it was not.
    """
    scene = _scene()

    result = relocalize(
        list(scene.good_clouds[:1]),
        list(scene.good_poses[:1]),
        scene.reference,
        backend="python",
    )

    assert result.accepted is False
    assert math.isinf(result.runner_up_margin)

    record = json.loads(json.dumps(as_record(result), allow_nan=False))

    assert record["runner_up_margin"] is None
    assert record["had_runner_up"] is False
    assert record["bridge"] is None
    assert record["refusals"]


# --------------------------------------------------------------------------
# 7. The local chain: each tentative link, judged before it is chained
# --------------------------------------------------------------------------


def _local(**overrides: object) -> LocalFit:
    """An adjacent tentative ICP result that clears every local gate.

    A namespace rather than a real :class:`~scanner3d.icp_plane.IcpPlaneResult`
    because what is under test is the JUDGEMENT, and building each failing case
    out of real geometry would be testing ICP's ability to fail on demand
    instead. The real solve's own behaviour is ``test_icp_plane.py``'s subject;
    every field read here is one that class carries.
    """
    fit = {
        "transform": Rigid.identity(),
        "rmse": 0.2,
        "converged": True,
        "iterations": 9,
        "matched_fraction": 0.9,
        "used_fraction": 0.8,
        "resolved_backend": "python",
    }
    fit.update(overrides)
    return cast(LocalFit, SimpleNamespace(**fit))


def test_a_good_local_link_is_accepted_and_records_what_it_measured():
    """The measurements travel with the verdict, because the record embeds both."""
    edge = judge_local_edge(_local(), frame=1)

    assert edge.accepted is True
    assert edge.refusals == ()
    assert edge.frame == 1
    assert edge.rmse == 0.2
    assert edge.matched_fraction == 0.9
    assert edge.as_dict()["converged"] is True


def test_a_local_link_that_never_converged_is_refused():
    """Finishing the iteration budget is not settling.

    Until this gate existed a solve that ran out of iterations while still
    moving was chained into the tentative segment exactly like one that
    converged, and a global search then placed a segment whose own chain
    nothing had looked at.
    """
    edge = judge_local_edge(_local(converged=False, iterations=60), frame=2)

    assert edge.accepted is False
    assert "still moving" in edge.refusals[0]
    assert "60" in edge.refusals[0]


def test_the_local_residual_gate_straddles_the_threshold():
    gate = LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR

    assert judge_local_edge(_local(rmse=gate * 0.99), frame=1).accepted is True
    assert judge_local_edge(_local(rmse=gate * 1.01), frame=1).accepted is False


def test_the_local_matched_fraction_gate_straddles_the_threshold():
    just_under = LOCAL_MIN_MATCHED_FRACTION - 0.01
    just_over = LOCAL_MIN_MATCHED_FRACTION + 0.01

    assert judge_local_edge(_local(matched_fraction=just_over), frame=1).accepted is True
    refused = judge_local_edge(_local(matched_fraction=just_under), frame=1)
    assert refused.accepted is False
    assert "found a partner" in refused.refusals[0]


def test_the_local_used_fraction_gate_straddles_the_threshold():
    just_under = LOCAL_MIN_USED_FRACTION - 0.01
    just_over = LOCAL_MIN_USED_FRACTION + 0.01

    assert judge_local_edge(_local(used_fraction=just_over), frame=1).accepted is True
    refused = judge_local_edge(_local(used_fraction=just_under), frame=1)
    assert refused.accepted is False
    assert "survived the trim" in refused.refusals[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rmse": float("nan")},
        {"rmse": float("inf")},
        {"matched_fraction": float("nan")},
        {"used_fraction": float("inf")},
        {
            "transform": Rigid(
                rotation=((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                translation=(0.0, 0.0, 0.0),
            )
        },
        {"transform": Rigid(rotation=_IDENTITY_ROTATION, translation=(float("inf"), 0.0, 0.0))},
    ],
)
def test_a_non_finite_local_link_is_refused_whichever_number_is_not_a_number(overrides):
    """Including the TRANSFORM.

    A solve that produced a NaN rotation reports an ordinary-looking residual on
    some backends, so judging the numbers alone would chain a step that means
    nothing and then measure everything downstream against it.
    """
    edge = judge_local_edge(_local(**overrides), frame=1)

    assert edge.accepted is False
    assert "finite numbers" in edge.refusals[0]


def test_a_local_link_names_every_gate_it_failed_not_just_the_first():
    edge = judge_local_edge(
        _local(converged=False, rmse=99.0, matched_fraction=0.01, used_fraction=0.01), frame=3
    )

    assert edge.accepted is False
    assert len(edge.refusals) == 4


def test_local_chain_gates_are_named_and_round_trip_through_json():
    gates = local_chain_gates()

    assert gates["local_residual_multiple"] == LOCAL_RESIDUAL_MULTIPLE
    assert gates["local_max_residual"] == LOCAL_RESIDUAL_MULTIPLE * NOISE_FLOOR
    assert gates["local_min_matched_fraction"] == LOCAL_MIN_MATCHED_FRACTION
    assert gates["local_min_used_fraction"] == LOCAL_MIN_USED_FRACTION
    assert gates["noise_floor"] == NOISE_FLOOR
    # The same gates are embedded in the whole threshold record, so a bridge
    # written without a separate local block still carries them.
    assert thresholds()["local_chain"] == gates
    assert json.loads(json.dumps(gates, allow_nan=False)) == gates


def test_a_refused_local_edge_round_trips_through_json_with_its_refusals():
    record = json.loads(json.dumps(judge_local_edge(_local(rmse=99.0), frame=2).as_dict()))

    assert record["accepted"] is False
    assert record["frame"] == 2
    assert record["refusals"]


def test_a_non_finite_local_edge_still_serialises_its_measurements_as_null():
    """The record of a link that failed is written too, and ``nan`` is not JSON."""
    record = json.loads(
        json.dumps(judge_local_edge(_local(rmse=float("nan")), frame=1).as_dict(), allow_nan=False)
    )

    assert record["rmse"] is None
    assert record["accepted"] is False


def test_relocalization_not_allowed_carries_a_doctor_status():
    assert RelocalizationNotAllowed.status == DoctorStatus.UNSUPPORTED_MODE
