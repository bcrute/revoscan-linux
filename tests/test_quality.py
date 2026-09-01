"""The offline quality harness, proved load-bearing rather than merely serialisable.

Every case here is a reconstruction whose truth is known exactly, built from
the pose and surface idioms the registration, multiview and relocalization
suites already use. Four of them are deliberate failures: a harness that only
ever ran on good input would report "all gates passed" for the rest of its
life and nobody would find out until a scan was wrong.

The failures are chosen to be the ones a person looking at the model would
miss:

- a lumpy plane whose bumps are half a percent of the panel's own size;
- an incomplete cloud whose every surviving point sits EXACTLY on the truth,
  so the observed-to-reference direction is perfect;
- a wrong placement of a symmetric object, where every surface metric is
  perfect because the object maps onto itself;
- a two percent uniform scale, which leaves rotation and flatness untouched.

Nothing here needs a GPU, a display server, a scanner or a network, and the
default compute backend is the pure-Python reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scanner3d.compute import PYTHON_BACKEND, available_backends
from scanner3d.geometry import Point, Rigid, normalize, rotation_about_axis, scaled
from scanner3d.multiview import AMBIGUITY_MARGIN
from scanner3d.relocalize import (
    LIMITATIONS,
    CandidateEvidence,
    Relocalization,
    as_record,
)
from scanner3d.validation import DegenerateFit

from scanner3d import quality  # isort: skip

FIXTURES = Path(__file__).parent / "fixtures" / "quality"

#: The recorded identity of the 2026-08-28 false acceptance, as the task packet
#: states it. Written here as a literal, not read from the fixture it is
#: supposed to be checking: a hash computed from the file it verifies proves
#: only that hashing is deterministic.
PHYSICAL_BRIDGE_SHA256 = "76007d2a7870cb122859eafcd502b62d97ae166daa2b6fd2cf5087b148277f68"
PHYSICAL_BRIDGE_SOURCE = "/home/ben/scans/scan/20260828-062457/segment-0001/bridge.json"
PHYSICAL_BRIDGE_FIXTURE = FIXTURES / "bridge-20260828-062457-segment-0001.json"


# --- synthetic truth ----------------------------------------------------------------


def _lobed_surface(count: int = 240) -> tuple[Point, ...]:
    """An ellipsoid with one off-axis lobe, sampled on a Fibonacci lattice.

    The same shape idiom ``tests/test_relocalize.py`` uses for its accepting
    scene, and for the same reason: the lobe is what makes one placement of
    this object better than every other, so a metric that trips here is
    tripping on geometry rather than on an accidental symmetry.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * index
        x, z = math.cos(theta) * radius, math.sin(theta) * radius
        bulge = 1.0 + 0.35 * max(0.0, x * 0.6 + z * 0.8) ** 3
        points.append((x * 20.0 * bulge, y * 14.0 * bulge, z * 17.0 * bulge))
    return tuple(points)


#: Side of the square panel every flatness case is measured on.
PANEL_SIDE = 60.0
#: Amplitude of the deliberate lumpiness, in the same units. Half a percent of
#: the panel it sits on - see ``test_a_subtle_lumpy_plane_trips_its_roughness_gate``.
LUMP_AMPLITUDE = 0.30

#: The panel is tilted off every coordinate axis so a plane fit has to do real
#: work, rather than an implementation that quietly regressed z on x and y
#: passing by accident.
_PANEL_POSE = rotation_about_axis(normalize((0.21, 0.87, 0.44)), 0.7, through=(3.0, -2.0, 5.0))


def _flat_panel(*, steps: int = 20, lump: float = 0.0) -> tuple[Point, ...]:
    """A square planar patch, optionally given a gentle two-axis ripple.

    The ripple is applied along the panel's own normal BEFORE the panel is
    tilted into place, so ``lump`` is exactly the peak displacement from the
    plane rather than a number that has to be projected to mean anything.

    Both axes are sampled over exactly one whole period and the endpoint is
    not repeated, so the ripple's mean and its correlation with either
    in-plane axis are zero. That is what keeps the fitted plane in the same
    place the clean panel's is: a ripple with a bias in it would tilt the fit
    toward itself and report less roughness than it actually has.
    """
    points: list[Point] = []
    for row in range(steps):
        for column in range(steps):
            u = -PANEL_SIDE / 2.0 + PANEL_SIDE * row / steps
            v = -PANEL_SIDE / 2.0 + PANEL_SIDE * column / steps
            w = (
                lump
                * math.sin(2.0 * math.pi * row / steps)
                * math.cos(2.0 * math.pi * column / steps)
            )
            points.append(_PANEL_POSE.apply((u, v, w)))
    return tuple(points)


#: An exact quarter turn about z, written as the matrix rather than built from
#: an angle. Every entry is 0 or +-1, so applying it is exact in float64 and a
#: shape with four-fold symmetry maps onto itself with distance exactly zero -
#: which is the whole point of the test that uses it.
QUARTER_TURN = Rigid.of(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))


def _symmetric_prism() -> tuple[Point, ...]:
    """A square prism whose point SET is invariant under :data:`QUARTER_TURN`.

    One quarter of the perimeter is sampled and the other three are generated
    by applying the exact quarter turn, so the invariance is a property of the
    construction rather than of a tolerance. This is the object
    :mod:`scanner3d.relocalize` warns about: it sits four ways on itself and
    no surface measurement can tell those four apart.
    """
    quadrant: list[Point] = []
    for layer in range(9):
        z = -20.0 + layer * 5.0
        for step in range(4):
            quadrant.append((10.0, -10.0 + step * 5.0, z))
    points = list(quadrant)
    turned = quadrant
    for _ in range(3):
        turned = [QUARTER_TURN.apply(point) for point in turned]
        points.extend(turned)
    return tuple(points)


#: A sweep axis that is off the origin in all three coordinates and parallel to
#: none of them, copied in spirit from ``tests/test_registration.py``: an
#: implementation that assumed the origin or a vertical axis cannot pass by
#: accident.
_SWEEP_AXIS = normalize((0.13, 0.96, 0.07))
_SWEEP_THROUGH: Point = (0.4, -0.2, 0.3)
_SWEEP_STEP = 0.35


def _sweep_poses(frames: int = 6) -> tuple[Rigid, ...]:
    """Exact camera poses for a moving-camera sweep. This is the truth."""
    return tuple(
        rotation_about_axis(_SWEEP_AXIS, index * _SWEEP_STEP, through=_SWEEP_THROUGH)
        for index in range(frames)
    )


def _views(surface: tuple[Point, ...], poses: tuple[Rigid, ...]) -> list[tuple[Point, ...]]:
    """What each camera sees: the world surface brought into that camera's frame.

    Every frame sees the whole surface. Occlusion is a capture concern and
    modelling it here would make "the reconstruction is incomplete" a property
    of the fixture rather than of the case that deliberately introduces it.
    """
    return [tuple(pose.inverse().apply_all(surface)) for pose in poses]


def _placed(views: list[tuple[Point, ...]], poses: tuple[Rigid, ...]) -> tuple[Point, ...]:
    """The reconstruction: every frame put back into the world by its pose."""
    return tuple(
        point for pose, view in zip(poses, views, strict=True) for point in pose.apply_all(view)
    )


def _drifted(poses: tuple[Rigid, ...], *, degrees: float, shift: float) -> tuple[Rigid, ...]:
    """Poses with a small error that accumulates, as a real chain's does."""
    axis = normalize((0.2, -0.9, 0.35))
    direction = normalize((1.0, 0.4, -0.3))
    return tuple(
        pose.compose(
            Rigid.of(
                rotation_about_axis(axis, math.radians(degrees * index)).rotation,
                scaled(direction, shift * index),
            )
        )
        for index, pose in enumerate(poses)
    )


# --- the clean case -----------------------------------------------------------------


def _clean_case() -> quality.CaseReport:
    """A moving camera over a known surface, with a small honest pose error."""
    surface = _lobed_surface()
    truth = _sweep_poses(3)
    estimated = _drifted(truth, degrees=0.01, shift=0.01)
    reconstruction = _placed(_views(surface, truth), estimated)

    poses = quality.pose_sequence_error(estimated, truth, name="sweep")
    agreement = quality.cloud_agreement(
        reconstruction, surface, name="surface", coverage_tolerance=0.10
    )
    flatness = quality.plane_roughness(_flat_panel(), name="panel", region="the reference panel")
    return quality.CaseReport(
        name="clean-sweep",
        measurements=(poses, agreement, flatness),
        gates=(
            quality.at_most(
                "pose.rotation.p95",
                poses.rotation.p95,
                0.05,
                unit=quality.DEGREES_UNIT,
                context="the camera is turning away from truth; re-check the chain before "
                "trusting the model",
            ),
            quality.at_most(
                "pose.translation.p95",
                poses.translation.p95,
                0.05,
                unit=quality.COORDINATE_UNIT,
                context="the camera is sliding away from truth; re-check the chain",
            ),
            quality.at_most(
                "cloud.observed_to_reference.p95",
                agreement.observed_to_reference.distances.p95,
                0.10,
                unit=quality.COORDINATE_UNIT,
                context="reconstructed surface sits off the truth; look for displacement or lumps",
            ),
            quality.at_most(
                "cloud.reference_to_observed.p95",
                agreement.reference_to_observed.distances.p95,
                0.10,
                unit=quality.COORDINATE_UNIT,
                context="truth surface has no reconstruction near it; look for a hole",
            ),
            quality.at_least(
                "cloud.reference_to_observed.coverage",
                agreement.reference_to_observed.coverage_fraction,
                0.99,
                unit=quality.FRACTION_UNIT,
                context="part of the object was never reconstructed; capture the missing side",
            ),
            quality.at_most(
                "plane.residual.p95",
                flatness.residuals.p95,
                0.05,
                unit=quality.COORDINATE_UNIT,
                context="a surface known to be flat is not flat; suspect noise or fusion artefacts",
            ),
        ),
    )


def test_a_clean_known_sweep_over_a_known_surface_passes_every_gate():
    """The harness is not a reject-everything stub.

    The pose error here is real - 0.01 degrees and 0.01 units per frame,
    accumulating - not zero, so the gates are clearing a measurement rather
    than clearing a constant.
    """
    report = _clean_case()
    assert report.passed, report.render()
    assert report.failures == ()

    poses = report.measurements[0]
    assert isinstance(poses, quality.PoseSequenceError)
    assert poses.rotation.worst > 0.0
    assert poses.translation.worst > 0.0


def test_the_gauge_frame_is_removed_without_removing_the_error_being_measured():
    """A shared world origin is not an error; per-frame divergence is.

    Two sequences differing by ONE global transform must report zero, and the
    same sequences with a per-frame divergence added must report it. An
    implementation that fitted an alignment over the whole trajectory would
    pass the first assertion and quietly shrink the second.
    """
    truth = _sweep_poses(6)
    gauge = rotation_about_axis(normalize((0.4, 0.3, 0.87)), 1.1, through=(12.0, -7.0, 4.0))
    shifted = tuple(gauge.compose(pose) for pose in truth)

    identical = quality.pose_sequence_error(shifted, truth, name="gauge-only")
    assert identical.rotation.worst == pytest.approx(0.0, abs=1e-9)
    assert identical.translation.worst == pytest.approx(0.0, abs=1e-9)

    diverging = quality.pose_sequence_error(
        tuple(gauge.compose(pose) for pose in _drifted(truth, degrees=0.4, shift=0.5)),
        truth,
        name="gauge-and-drift",
    )
    # The last frame is five drift steps from the first; the gauge removal must
    # not have absorbed any of them.
    assert diverging.rotation.worst == pytest.approx(2.0, abs=0.05)
    assert diverging.translation.worst == pytest.approx(2.5, abs=0.05)
    assert diverging.frames[0].index == 1
    assert len(diverging.frames) == 5


def test_the_gauge_frame_is_excluded_from_the_summaries_it_would_flatter():
    """Frame 0's error is zero by construction and is not a measurement.

    Including it would pull every median toward passing by an amount that
    grows as the sequence gets shorter - worst exactly where the evidence is
    thinnest.
    """
    truth = _sweep_poses(3)
    error = quality.pose_sequence_error(
        _drifted(truth, degrees=0.5, shift=0.5), truth, name="short"
    )
    assert error.pose_count == 3
    assert error.gauge_frame_index == 0
    assert [frame.index for frame in error.frames] == [1, 2]
    assert error.rotation.count == 2
    assert min(frame.rotation_degrees for frame in error.frames) > 0.0


# --- the four deliberate failures ---------------------------------------------------


def test_a_subtle_lumpy_plane_trips_its_roughness_gate():
    """An error too small to see, and far too big to accept.

    The ripple's peak is 0.30 against a panel 60 across: half a percent, which
    on a rendered model is indistinguishable from shading. The same panel with
    the ripple removed clears the same gate by twelve orders of magnitude, so
    the gate is reading the perturbation and nothing else.
    """
    assert LUMP_AMPLITUDE / PANEL_SIDE < 0.01

    limit = 0.05
    context = "a surface known to be flat is not flat; suspect noise or fusion artefacts"
    clean = quality.plane_roughness(_flat_panel(), name="panel", region="the reference panel")
    lumpy = quality.plane_roughness(
        _flat_panel(lump=LUMP_AMPLITUDE), name="panel", region="the reference panel"
    )

    clean_gate = quality.at_most(
        "plane.residual.p95",
        clean.residuals.p95,
        limit,
        unit=quality.COORDINATE_UNIT,
        context=context,
    )
    lumpy_gate = quality.at_most(
        "plane.residual.p95",
        lumpy.residuals.p95,
        limit,
        unit=quality.COORDINATE_UNIT,
        context=context,
    )
    assert clean_gate.passed
    assert clean.residuals.p95 < 1e-12
    assert not lumpy_gate.passed
    # The ripple is symmetric about the fitted plane, which is a fact the
    # signed extremes carry and an absolute residual cannot. It is also the
    # check that the fit did not tilt itself into the perturbation.
    assert lumpy.signed_max == pytest.approx(-lumpy.signed_min, rel=1e-9)
    assert lumpy.residuals.worst == pytest.approx(LUMP_AMPLITUDE, rel=1e-9)
    assert lumpy.residuals.p95 > 5.0 * limit


def test_an_incomplete_cloud_is_perfect_one_way_and_fails_the_other():
    """One-way cloud distance can never pass an incomplete reconstruction.

    Every point that survives here is byte-for-byte a point of the truth, so
    observed-to-reference is exactly zero and its coverage is exactly one - a
    directed score alone would call this reconstruction flawless. The missing
    cap is only visible from the other side.
    """
    surface = _lobed_surface()
    incomplete = tuple(point for point in surface if point[1] <= 6.0)
    assert 0.5 < len(incomplete) / len(surface) < 0.9

    agreement = quality.cloud_agreement(
        incomplete, surface, name="incomplete", coverage_tolerance=0.10
    )
    forward = quality.at_most(
        "cloud.observed_to_reference.p95",
        agreement.observed_to_reference.distances.p95,
        0.10,
        unit=quality.COORDINATE_UNIT,
        context="reconstructed surface sits off the truth",
    )
    reverse = quality.at_most(
        "cloud.reference_to_observed.p95",
        agreement.reference_to_observed.distances.p95,
        0.10,
        unit=quality.COORDINATE_UNIT,
        context="truth surface has no reconstruction near it; capture the missing side",
    )
    coverage = quality.at_least(
        "cloud.reference_to_observed.coverage",
        agreement.reference_to_observed.coverage_fraction,
        0.99,
        unit=quality.FRACTION_UNIT,
        context="part of the object was never reconstructed; capture the missing side",
    )

    assert forward.passed
    assert agreement.observed_to_reference.distances.worst == 0.0
    assert agreement.observed_to_reference.coverage_fraction == 1.0
    assert not reverse.passed
    assert not coverage.passed


def test_a_wrong_placement_of_a_symmetric_object_is_only_visible_in_pose_truth():
    """A flattering surface residual over a placement that is simply wrong.

    The prism maps onto itself under an exact quarter turn, so a reconstruction
    that placed half its frames a quarter turn out produces a point set
    identical to the truth: both cloud directions are exactly zero and every
    surface gate passes. This is the 2026-08-28 failure in miniature, and the
    only thing that catches it is comparison against exact poses.
    """
    prism = _symmetric_prism()
    truth = _sweep_poses(6)
    estimated = tuple(
        pose if index < 3 else QUARTER_TURN.compose(pose) for index, pose in enumerate(truth)
    )
    reconstruction = _placed(_views(prism, truth), estimated)

    agreement = quality.cloud_agreement(
        reconstruction, prism, name="prism", coverage_tolerance=0.10
    )
    assert agreement.observed_to_reference.distances.worst == pytest.approx(0.0, abs=1e-9)
    assert agreement.reference_to_observed.distances.worst == pytest.approx(0.0, abs=1e-9)
    assert agreement.reference_to_observed.coverage_fraction == 1.0

    poses = quality.pose_sequence_error(estimated, truth, name="prism-sweep")
    rotation = quality.at_most(
        "pose.rotation.worst",
        poses.rotation.worst,
        1.0,
        unit=quality.DEGREES_UNIT,
        context="a frame is placed at the wrong orientation; the object may be symmetric enough "
        "to hide it from every surface metric",
    )
    assert not rotation.passed
    assert poses.rotation.worst == pytest.approx(90.0, abs=1e-6)
    assert [frame.rotation_degrees for frame in poses.frames[:2]] == pytest.approx(
        [0.0, 0.0], abs=1e-9
    )


def test_a_uniform_scale_error_trips_the_dimension_and_landmark_gates():
    """The failure every self-consistency check is blind to.

    A world scaled by 2% has the same pose ROTATIONS, the same flat surfaces
    and the same internal agreement it always had. Held-out closure as
    :func:`scanner3d.pipeline.reconstruct_sweep` scores it is the rotation
    residual of withheld revisit edges, and rotations are untouched here, so
    that number stays exactly as good as it was.

    This case runs four measurements - pose error, plane roughness, dimensions
    and landmarks - and only the last two fail. It deliberately runs no
    ``cloud_agreement``: given a genuinely dense reference surface in the same
    frame, that measurement would see this scale too, because a 2% larger
    model is 2% away from truth everywhere. The narrower claim proved here is
    the one that matters for a physical object nobody has a dense truth
    surface for - a few lengths someone measured independently are enough.
    """
    scale = 1.02
    surface = _lobed_surface()
    scaled_surface = tuple(scaled(point, scale) for point in surface)
    truth_poses = _sweep_poses(4)
    # Scaling the world scales each pose's translation and leaves its rotation
    # alone; that is what makes the rotation half of the comparison pass.
    scaled_poses = tuple(
        Rigid.of(pose.rotation, scaled(pose.translation, scale)) for pose in truth_poses
    )

    poses = quality.pose_sequence_error(scaled_poses, truth_poses, name="scaled-sweep")
    assert poses.rotation.worst == pytest.approx(0.0, abs=1e-9)
    assert poses.translation.worst > 0.0

    flatness = quality.plane_roughness(
        tuple(scaled(point, scale) for point in _flat_panel()),
        name="panel",
        region="the reference panel",
    )
    assert flatness.residuals.p95 < 1e-12

    def _span(points: tuple[Point, ...], axis: int) -> float:
        return max(point[axis] for point in points) - min(point[axis] for point in points)

    truth_dimensions = {
        name: _span(surface, axis) for axis, name in enumerate(("width", "height", "depth"))
    }
    measured_dimensions = {
        name: _span(scaled_surface, axis) for axis, name in enumerate(("width", "height", "depth"))
    }
    dimensions = quality.dimension_errors(
        measured_dimensions,
        truth_dimensions,
        name="hull",
        truth_source="exact synthetic construction: the surface this case was built from",
    )
    landmarks = quality.landmark_offsets(
        {"tip": scaled_surface[0], "tail": scaled_surface[-1]},
        {"tip": surface[0], "tail": surface[-1]},
        name="extremes",
        truth_source="exact synthetic construction: the surface this case was built from",
    )

    rotation_gate = quality.at_most(
        "pose.rotation.worst",
        poses.rotation.worst,
        0.05,
        unit=quality.DEGREES_UNIT,
        context="the camera is turning away from truth",
    )
    plane_gate = quality.at_most(
        "plane.residual.p95",
        flatness.residuals.p95,
        0.05,
        unit=quality.COORDINATE_UNIT,
        context="a surface known to be flat is not flat",
    )
    dimension_gate = quality.at_most(
        "dimension.absolute.worst",
        dimensions.absolute.worst,
        0.20,
        unit=quality.COORDINATE_UNIT,
        context="the object is the right shape at the wrong size; re-check the depth scale",
    )
    landmark_gate = quality.at_most(
        "landmark.offset.worst",
        landmarks.offsets.worst,
        0.20,
        unit=quality.COORDINATE_UNIT,
        context="a known feature is not where it belongs; re-check the depth scale",
    )
    report = quality.CaseReport(
        name="uniform-scale",
        measurements=(poses, flatness, dimensions, landmarks),
        gates=(rotation_gate, plane_gate, dimension_gate, landmark_gate),
    )

    assert not report.passed
    assert [gate.name for gate in report.failures] == [
        "dimension.absolute.worst",
        "landmark.offset.worst",
    ]
    assert dimensions.relative is not None
    assert dimensions.relative.worst == pytest.approx(scale - 1.0, rel=1e-9)


# --- the physical bridge regression -------------------------------------------------


def _fixture_bytes() -> bytes:
    return PHYSICAL_BRIDGE_FIXTURE.read_bytes()


def test_the_physical_bridge_fixture_is_the_bytes_that_were_recorded():
    """Byte-bound to the capture, and to the provenance recorded beside it.

    A fixture that could be regenerated is not evidence of what happened. If
    this fails, the file was edited or reformatted and the regression below is
    no longer testing the 2026-08-28 record.
    """
    assert hashlib.sha256(_fixture_bytes()).hexdigest() == PHYSICAL_BRIDGE_SHA256

    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["name"] == PHYSICAL_BRIDGE_FIXTURE.name)
    assert entry["sha256"] == PHYSICAL_BRIDGE_SHA256
    assert entry["source_sha256"] == PHYSICAL_BRIDGE_SHA256
    assert entry["source"] == PHYSICAL_BRIDGE_SOURCE
    assert entry["bytes"] == len(_fixture_bytes())


def test_the_recorded_2026_08_28_bridge_is_refused_by_current_policy():
    """The mandatory known-bad regression.

    It is not enough that this record is refused; it has to be refused for the
    reason the policy was changed for. Every other gate in the record is
    healthy - residual 0.515 against a 2.6 limit, matched fraction 1.00, a
    1.50x runner-up margin - and a test asserting only "refused" would keep
    passing if the refusal came from somewhere else entirely.
    """
    record = json.loads(_fixture_bytes())
    audit = quality.audit_bridge_record(
        record,
        name="2026-08-28 segment-0001",
        source=PHYSICAL_BRIDGE_SOURCE,
        source_sha256=PHYSICAL_BRIDGE_SHA256,
    )

    assert audit.recorded_accepted is True
    assert audit.accepted_under_current_policy is False
    assert audit.regressed is True
    assert audit.refused_for_coarse_ambiguity is True
    assert len(audit.refusals) == 1
    assert "coarse" in audit.refusals[0]
    assert str(AMBIGUITY_MARGIN) in audit.refusals[0]

    # The decisive numbers, exactly as the capture wrote them.
    assert audit.winner_residual == pytest.approx(0.5148863692487536)
    assert audit.winner_overlap == pytest.approx(0.48333333333333334)
    assert audit.other_candidate_max_overlap == pytest.approx(0.598)
    assert audit.a_rival_covers_more is True
    assert audit.runner_up_margin == pytest.approx(1.5026462192780772)

    gate = quality.must_be(
        "bridge.accepted_under_current_policy",
        audit.accepted_under_current_policy,
        False,
        context="this historical bridge must stay refused; if it is accepted again the "
        "coarse-ambiguity policy has been weakened",
    )
    assert gate.passed
    assert (
        quality.must_be(
            "bridge.recorded_accepted", audit.recorded_accepted, False, context="unused"
        ).passed
        is False
    )


def _healthy_record() -> dict[str, object]:
    """A bridge current policy has no complaint about, built from the real types.

    Constructed through :class:`scanner3d.relocalize.Relocalization` and
    :func:`scanner3d.relocalize.as_record` rather than hand-written JSON, so
    the audit is proved to read the schema the product actually emits. If
    that schema changes, this record changes with it and the audit has to keep
    up - which is the point.
    """
    winner = CandidateEvidence(
        transform=Rigid.identity(),
        origin="principal+octahedral",
        coarse_score=3.10,
        coarse_matched=400,
        rmse=0.30,
        converged=True,
        iterations=14,
        matched_fraction=0.98,
        used_fraction=0.80,
        overlap=0.62,
        resolved_backend=PYTHON_BACKEND,
    )
    rival = CandidateEvidence(
        transform=rotation_about_axis(normalize((0.2, 0.9, 0.3)), 1.4),
        origin="principal+octahedral",
        coarse_score=4.20,
        coarse_matched=390,
        rmse=0.95,
        converged=True,
        iterations=22,
        matched_fraction=0.91,
        used_fraction=0.74,
        overlap=0.41,
        resolved_backend=PYTHON_BACKEND,
    )
    result = Relocalization(
        accepted=True,
        bridge=winner.transform,
        refusals=(),
        candidates=(winner, rival),
        frame_overlaps=(0.55, 0.58, 0.61),
        runner_up_margin=rival.rmse / winner.rmse,
        coarse_unambiguous=True,
        segment_points=1500,
        reference_points=4000,
        frames=3,
        limitations=LIMITATIONS,
    )
    return dict(as_record(result))


def test_a_healthy_bridge_record_is_still_accepted_by_current_policy():
    """The audit is not a reject-everything stub either.

    Without this, "the known-bad record is refused" would be satisfied by a
    function that refuses everything, and the regression above would stop
    meaning anything the day the policy became unconditional.
    """
    audit = quality.audit_bridge_record(
        _healthy_record(), name="synthetic-healthy", source="constructed in tests/test_quality.py"
    )
    assert audit.recorded_accepted is True
    assert audit.accepted_under_current_policy is True
    assert audit.regressed is False
    assert audit.refused_for_coarse_ambiguity is False
    assert audit.refusals == ()
    assert audit.a_rival_covers_more is False


def test_the_audit_reapplies_each_gate_it_claims_to():
    """Every branch of the re-applied policy, one at a time.

    A policy re-application whose only exercised path is the one the physical
    record happens to trip would silently stop checking the others.
    """
    healthy = _healthy_record()

    def _with(
        *,
        top: dict[str, object] | None = None,
        winner: dict[str, object] | None = None,
        rival: dict[str, object] | None = None,
    ) -> quality.BridgeAudit:
        record = json.loads(json.dumps(healthy))
        record.update(top or {})
        record["candidates"][0].update(winner or {})
        record["candidates"][1].update(rival or {})
        return quality.audit_bridge_record(record, name="variant", source="tests")

    def _only_refusal(audit: quality.BridgeAudit) -> str:
        assert len(audit.refusals) == 1, audit.refusals
        return audit.refusals[0]

    assert _only_refusal(_with(top={"coarse_unambiguous": False})).startswith(
        "the record says the coarse"
    )
    assert "never converged" in _only_refusal(_with(winner={"converged": False}))
    # The rival moves with the winner here: raising only the winner's residual
    # would hand the win to the rival, which is the re-selection working rather
    # than the residual gate being tested.
    assert "over the" in _only_refusal(_with(winner={"rmse": 2.9}, rival={"rmse": 9.0}))
    assert "found a partner" in _only_refusal(_with(winner={"matched_fraction": 0.20}))
    assert "survived the trim" in _only_refusal(_with(winner={"used_fraction": 0.10}))
    assert "overlaps the accepted model" in _only_refusal(_with(winner={"overlap": 0.10}))
    assert "fits almost as well" in _only_refusal(_with(top={"runner_up_margin": 1.10}))
    assert "recorded frames overlap" in _only_refusal(
        _with(top={"frame_overlaps": [0.55, 0.05, 0.61]})
    )


def test_the_audit_reselects_the_winner_rather_than_trusting_the_recorded_order():
    """Today's ranking rule, applied to yesterday's list.

    A record whose candidates are stored in some other order must still be
    judged on the candidate current policy would choose.
    """
    record = _healthy_record()
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    record["candidates"] = list(reversed(candidates))
    audit = quality.audit_bridge_record(record, name="reordered", source="tests")
    assert audit.winner_residual == pytest.approx(0.30)
    assert audit.winner_overlap == pytest.approx(0.62)


def test_a_record_that_is_not_a_decision_is_refused_as_unreadable():
    """A corrupt file must not masquerade as a caught regression."""
    healthy = _healthy_record()
    with pytest.raises(quality.BridgeRecordUnusable, match="candidates"):
        quality.audit_bridge_record({**healthy, "candidates": []}, name="empty", source="tests")
    with pytest.raises(quality.BridgeRecordUnusable, match="not a boolean"):
        quality.audit_bridge_record(
            {**healthy, "coarse_unambiguous": "yes"}, name="stringly", source="tests"
        )
    with pytest.raises(quality.BridgeRecordUnusable, match="had_runner_up"):
        quality.audit_bridge_record(
            {**healthy, "had_runner_up": False}, name="mismatched", source="tests"
        )


# --- the gate layer -----------------------------------------------------------------


def test_a_gate_reports_everything_a_reader_needs_without_the_source():
    gate = quality.at_most(
        "cloud.p95", 0.42, 0.10, unit=quality.COORDINATE_UNIT, context="capture the missing side"
    )
    assert not gate.passed
    rendered = gate.render()
    assert rendered.startswith("FAIL cloud.p95: measured 0.42 <= 0.1")
    assert quality.COORDINATE_UNIT in rendered
    assert "capture the missing side" in rendered
    assert gate.as_dict()["comparison"] == "<="


def test_a_gate_at_its_limit_passes_in_both_directions():
    """The boundary is stated once, here, rather than argued about per case."""
    assert quality.at_most("a", 1.0, 1.0, unit="mm", context="c").passed
    assert quality.at_least("b", 1.0, 1.0, unit="mm", context="c").passed
    assert not quality.at_most("a", 1.0 + 1e-12, 1.0, unit="mm", context="c").passed
    assert not quality.at_least("b", 1.0 - 1e-12, 1.0, unit="mm", context="c").passed


def test_a_gate_without_a_name_unit_or_action_is_refused():
    with pytest.raises(quality.GateUnusable, match="gate name"):
        quality.at_most("", 1.0, 2.0, unit="mm", context="c")
    with pytest.raises(quality.GateUnusable, match="gate unit"):
        quality.at_most("a", 1.0, 2.0, unit="  ", context="c")
    with pytest.raises(quality.GateUnusable, match="context"):
        quality.at_most("a", 1.0, 2.0, unit="mm", context="")
    with pytest.raises(quality.GateUnusable, match="context"):
        quality.must_be("a", True, True, context="")


def test_a_gate_refuses_a_non_finite_measurement_or_an_unbounded_limit():
    with pytest.raises(quality.GateUnusable, match="not finite"):
        quality.at_most("a", math.nan, 1.0, unit="mm", context="c")
    with pytest.raises(quality.GateUnusable, match="cannot fail"):
        quality.at_most("a", 1.0, math.inf, unit="mm", context="c")


def test_a_flag_and_a_magnitude_cannot_be_confused_for_each_other():
    """``True >= 0.5`` is a fact about Python's type system, not a measurement."""
    with pytest.raises(quality.GateUnusable, match="boolean"):
        quality.at_least("a", True, 0.5, unit="mm", context="c")
    with pytest.raises(quality.GateUnusable, match="not both booleans"):
        quality.must_be("a", 1.0, True, context="c")  # type: ignore[arg-type]


def _direct(
    *,
    name="direct",
    measured=0.5,
    comparison=quality.Comparison.AT_MOST,
    limit=1.0,
    unit="mm",
    context="look at the panel",
) -> quality.Gate:
    """A valid direct ``Gate(...)`` with fields replaced, bypassing the factories."""
    return quality.Gate(
        name=name,
        measured=measured,
        comparison=comparison,
        limit=limit,
        unit=unit,
        context=context,
    )


def test_a_gate_built_without_the_factories_is_validated_the_same_way():
    """The class's invariants belong to the class, not to how it was built.

    ``Gate`` is public and frozen, and a caller may construct one directly.
    Every refusal the factories make must therefore be made here too, or the
    report guarantees - a named, actionable line that round-trips through
    strict JSON - would only hold for gates that happened to come through
    ``at_most``/``at_least``/``must_be``.
    """
    with pytest.raises(quality.GateUnusable, match="gate name"):
        _direct(name="")
    with pytest.raises(quality.GateUnusable, match="gate name"):
        _direct(name="   ")
    with pytest.raises(quality.GateUnusable, match="gate unit"):
        _direct(unit="")
    with pytest.raises(quality.GateUnusable, match="context"):
        _direct(context="  ")
    with pytest.raises(quality.GateUnusable, match="not finite"):
        _direct(measured=math.inf)
    with pytest.raises(quality.GateUnusable, match="not finite"):
        _direct(measured=math.nan)
    with pytest.raises(quality.GateUnusable, match="cannot fail"):
        _direct(limit=-math.inf)
    with pytest.raises(quality.GateUnusable, match="boolean"):
        _direct(measured=True)
    with pytest.raises(quality.GateUnusable, match="boolean"):
        _direct(limit=False)
    with pytest.raises(quality.GateUnusable, match="not both booleans"):
        _direct(measured=1.0, comparison=quality.Comparison.IS, limit=True, unit="boolean")
    with pytest.raises(quality.GateUnusable, match="not both booleans"):
        _direct(measured=True, comparison=quality.Comparison.IS, limit=1, unit="boolean")


def test_a_comparison_this_module_does_not_define_is_refused_not_treated_as_an_ordering():
    """An unknown comparison must never quietly become ``>=`` and report PASS.

    The string here is the VALUE of ``Comparison.AT_MOST`` but not the member,
    which is exactly the mistake a caller building a gate from parsed JSON
    would make. If it fell through to an ordering it would answer ``5 >= 1``
    and report a PASS for a question nobody asked.
    """
    for comparison in ("<=", ">=", "is", "", None, 0):
        with pytest.raises(quality.GateUnusable, match="not a Comparison"):
            quality.Gate(
                name="cloud.p95",
                measured=5.0,
                comparison=comparison,  # type: ignore[arg-type]
                limit=1.0,
                unit="mm",
                context="capture the far side",
            )


def test_a_valid_direct_gate_behaves_exactly_like_a_factory_built_one():
    """The validation refuses bad gates without refusing good ones."""
    direct = _direct(name="cloud.p95", measured=2.0, limit=1.0, context="capture the far side")
    assert not direct.passed
    assert direct.render().startswith("FAIL cloud.p95: measured 2 <= 1 (mm)")
    assert direct == quality.at_most(
        "cloud.p95", 2.0, 1.0, unit="mm", context="capture the far side"
    )

    flag = quality.Gate(
        name="bridge.refused",
        measured=True,
        comparison=quality.Comparison.IS,
        limit=True,
        unit=quality.BOOLEAN_UNIT,
        context="current policy would refuse this bridge",
    )
    assert flag.passed
    assert flag == quality.must_be(
        "bridge.refused", True, True, context="current policy would refuse this bridge"
    )
    report = quality.CaseReport(name="direct", measurements=(), gates=(direct, flag))
    assert json.loads(json.dumps(report.as_dict(), allow_nan=False)) == report.as_dict()


# --- report assembly ----------------------------------------------------------------


def _mixed_case() -> quality.CaseReport:
    truth = _sweep_poses(3)
    estimated = _drifted(truth, degrees=0.6, shift=0.7)
    return quality.CaseReport(
        name="mixed",
        measurements=(quality.pose_sequence_error(estimated, truth, name="sweep"),),
        gates=(
            quality.at_most("a", 0.1, 1.0, unit="mm", context="fine"),
            quality.at_most("b", 5.0, 1.0, unit="mm", context="look here"),
            quality.at_most("c", 0.2, 1.0, unit="mm", context="fine"),
            quality.at_least("d", 0.2, 1.0, unit="mm", context="and here"),
        ),
        notes=("landmark and dimension error: not measured - no truth was supplied",),
    )


def test_a_report_lists_every_failure_rather_than_stopping_at_the_first():
    """The second failure is usually what says which failure mode happened."""
    report = _mixed_case()
    assert not report.passed
    assert [gate.name for gate in report.failures] == ["b", "d"]
    rendered = report.render()
    assert rendered.splitlines()[0] == "[FAIL] case mixed: 2/4 failed"
    assert rendered.count("PASS ") == 2
    assert rendered.count("FAIL ") == 2
    assert "note: landmark and dimension error: not measured" in rendered


def test_a_report_round_trips_through_strict_json_on_both_paths():
    """``allow_nan=False``, on the failing path as well as the passing one.

    A harness whose failure report cannot be serialised loses exactly the
    evidence it was built to produce.
    """
    for report in (_clean_case(), _mixed_case()):
        text = quality.to_json(report)
        assert json.loads(text) == report.as_dict()
        assert "NaN" not in text and "Infinity" not in text

    suite = quality.SuiteReport(name="fast", cases=(_clean_case(), _mixed_case()))
    assert not suite.passed
    assert [name for name, _ in suite.failures] == ["mixed", "mixed"]
    assert json.loads(quality.to_json(suite)) == suite.as_dict()
    assert suite.render().splitlines()[0] == "[FAIL] suite fast: 2 failed gate(s) across 2 case(s)"


def test_a_report_is_byte_identical_when_the_measurement_is():
    """Deterministic, so a CI diff between two runs means something changed."""
    assert quality.to_json(_clean_case()) == quality.to_json(_clean_case())
    assert _mixed_case().render() == _mixed_case().render()


def test_a_suite_of_only_passing_cases_passes():
    suite = quality.SuiteReport(name="fast", cases=(_clean_case(),))
    assert suite.passed
    assert suite.failures == ()
    assert suite.render().startswith("[PASS] suite fast: 0 failed gate(s)")


# --- input refusal ------------------------------------------------------------------


def test_an_unusable_cloud_is_refused_before_anything_is_allocated():
    surface = _lobed_surface(50)
    with pytest.raises(quality.MeasurementInputUnusable, match="0 point"):
        quality.cloud_agreement((), surface, name="empty", coverage_tolerance=0.1)
    with pytest.raises(quality.MeasurementInputUnusable, match="not finite"):
        quality.cloud_agreement([(0.0, math.nan, 0.0)], surface, name="nan", coverage_tolerance=0.1)
    two_coordinates: list[Point] = [(0.0, 0.0)]  # type: ignore[list-item]
    with pytest.raises(quality.MeasurementInputUnusable, match="coordinate"):
        quality.cloud_agreement(two_coordinates, surface, name="short", coverage_tolerance=0.1)
    with pytest.raises(quality.MeasurementInputUnusable, match="must be positive"):
        quality.cloud_agreement(surface, surface, name="zero", coverage_tolerance=0.0)


def test_an_unbounded_comparison_is_refused_by_point_count_and_by_work():
    """Both bounds, because they catch different mistakes.

    A caller can stay under the point-count bound on each side and still ask
    for a comparison whose cost is the product of the two.
    """
    too_many = [(0.0, 0.0, 0.0)] * (quality.MAX_POINTS + 1)
    with pytest.raises(quality.MeasurementInputUnusable, match=str(quality.MAX_POINTS)):
        quality.cloud_agreement(too_many, [(0.0, 0.0, 0.0)], name="huge", coverage_tolerance=0.1)

    wide = [(float(index), 0.0, 0.0) for index in range(6000)]
    tall = [(0.0, float(index), 0.0) for index in range(5000)]
    assert len(wide) * len(tall) > quality.MAX_NEAREST_WORK
    with pytest.raises(quality.MeasurementInputUnusable, match="query-target pairs"):
        quality.cloud_agreement(wide, tall, name="expensive", coverage_tolerance=0.1)


def test_mismatched_pose_sequences_are_refused_rather_than_truncated():
    truth = _sweep_poses(4)
    with pytest.raises(quality.MeasurementInputUnusable, match="do not pair up"):
        quality.pose_sequence_error(truth[:3], truth, name="short")
    with pytest.raises(quality.MeasurementInputUnusable, match="nothing left to measure"):
        quality.pose_sequence_error(truth[:1], truth[:1], name="single")
    broken = Rigid(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(math.inf, 0.0, 0.0),
    )
    with pytest.raises(quality.MeasurementInputUnusable, match="not finite"):
        quality.pose_sequence_error((truth[0], broken), truth[:2], name="infinite")


def test_named_truth_and_measurement_must_name_the_same_things():
    """A silent intersection would let a bad feature pass by not being reported."""
    with pytest.raises(quality.MeasurementInputUnusable, match="different things"):
        quality.dimension_errors(
            {"width": 1.0}, {"width": 1.0, "height": 2.0}, name="hull", truth_source="calipers"
        )
    with pytest.raises(quality.MeasurementInputUnusable, match="different things"):
        quality.landmark_offsets(
            {"tip": (0.0, 0.0, 0.0), "extra": (1.0, 1.0, 1.0)},
            {"tip": (0.0, 0.0, 0.0)},
            name="features",
            truth_source="drawing",
        )
    with pytest.raises(quality.MeasurementInputUnusable, match="no truth"):
        quality.dimension_errors({}, {}, name="hull", truth_source="calipers")


def test_a_dimension_truth_of_zero_reports_no_relative_error_rather_than_a_huge_one():
    result = quality.dimension_errors(
        {"gap": 0.4}, {"gap": 0.0}, name="hull", truth_source="exact synthetic construction"
    )
    assert result.entries[0].relative_error is None
    assert result.relative is None
    assert result.absolute.worst == pytest.approx(0.4)
    assert (
        json.loads(
            quality.to_json(quality.CaseReport(name="zero", measurements=(result,), gates=()))
        )["measurements"][0]["relative"]
        is None
    )


def test_a_region_that_spans_no_plane_is_refused_rather_than_called_rough():
    line = tuple((float(index), 2.0 * index, 3.0 * index) for index in range(12))
    with pytest.raises(DegenerateFit):
        quality.plane_roughness(line, name="line", region="a region that is not a surface")


def test_a_measurement_without_a_stated_truth_source_is_refused():
    """A dimension gate is only as good as the instrument behind its truth."""
    with pytest.raises(quality.MeasurementInputUnusable, match="truth source"):
        quality.dimension_errors({"a": 1.0}, {"a": 1.0}, name="hull", truth_source="")


# --- backend agreement --------------------------------------------------------------


def _accelerated_backends() -> list[str]:
    return [
        entry.name
        for entry in available_backends()
        if entry.available and entry.name != PYTHON_BACKEND
    ]


def test_the_default_backend_is_the_pure_python_reference():
    """The correctness tier must not depend on numpy or scipy being installed."""
    agreement = quality.cloud_agreement(
        _lobed_surface(60), _lobed_surface(60), name="default", coverage_tolerance=0.1
    )
    assert agreement.resolved_backend == PYTHON_BACKEND


@pytest.mark.parametrize("backend", _accelerated_backends() or [PYTHON_BACKEND])
def test_an_accelerated_backend_agrees_with_the_reference(backend):
    """The appliance path returns the reference's answer, to a stated tolerance.

    Timing is not asserted anywhere: it is a property of the machine, and a
    correctness suite that gated on it would fail on a slower host and tell
    nobody anything.
    """
    if backend == PYTHON_BACKEND:  # pragma: no cover - depends on what is installed
        # _accelerated_backends() filters "python" out, so reaching here means
        # nothing accelerated is installed and the parametrisation fell back.
        # Skip loudly rather than compare the reference against itself: that
        # comparison is equal by construction and would report parity as
        # proved on a host where it was never measured.
        pytest.skip(
            "neither numpy nor scipy is usable here, so there is no accelerated backend to "
            "prove agreement against"
        )
    observed = _placed(
        _views(_lobed_surface(200), _sweep_poses(2)),
        _drifted(_sweep_poses(2), degrees=0.3, shift=0.4),
    )
    reference = _lobed_surface(200)

    expected = quality.cloud_agreement(
        observed, reference, name="parity", coverage_tolerance=0.5, backend=PYTHON_BACKEND
    )
    actual = quality.cloud_agreement(
        observed, reference, name="parity", coverage_tolerance=0.5, backend=backend
    )
    assert actual.resolved_backend == backend
    # The tolerance is a fraction of the scene, not an absolute length, because
    # the numpy backend's expanded squared-distance identity leaves an error
    # proportional to the coordinates' own magnitude - see BACKEND_AGREEMENT.
    tolerance = quality.BACKEND_AGREEMENT * expected.search_gate
    for direction in ("observed_to_reference", "reference_to_observed"):
        want = getattr(expected, direction)
        got = getattr(actual, direction)
        for statistic in ("median", "rms", "p95", "worst"):
            assert getattr(got.distances, statistic) == pytest.approx(
                getattr(want.distances, statistic), abs=tolerance
            ), f"{backend} {direction} {statistic}"
        assert got.coverage_fraction == pytest.approx(want.coverage_fraction, abs=1e-9)


# --- distribution arithmetic --------------------------------------------------------


def test_every_reported_quantile_is_a_value_that_was_actually_observed():
    """Nearest rank, not interpolation, so "which point was that?" has an answer."""
    values = [float(index) for index in range(1, 101)]
    surface = tuple((value, 0.0, 0.0) for value in values)
    reference = ((0.0, 0.0, 0.0),)
    agreement = quality.cloud_agreement(surface, reference, name="ruler", coverage_tolerance=10.0)
    distances = agreement.observed_to_reference.distances
    assert distances.count == 100
    assert distances.median == 50.0
    assert distances.p95 == 95.0
    assert distances.worst == 100.0
    assert distances.rms == pytest.approx(math.sqrt(sum(v * v for v in values) / 100))
    assert agreement.observed_to_reference.coverage_fraction == pytest.approx(0.10)


# --- the opt-in physical tier -------------------------------------------------------


def test_a_case_that_applied_no_gate_says_so_beside_its_verdict():
    """ "Nothing failed" and "nothing was asked" must not read the same.

    The physical replay tier reports cases with no gates at all whenever the
    caller supplied no limits, and those cases pass trivially. The count sits
    beside the verdict so a reader skimming the JSON can tell which happened.
    """
    ungated = quality.CaseReport(name="reported-only", measurements=(), gates=())
    assert ungated.passed
    assert ungated.as_dict()["gate_count"] == 0
    assert ungated.render().startswith("[PASS] case reported-only: 0/0 failed")
    assert (
        quality.SuiteReport(name="s", cases=(ungated, _mixed_case())).as_dict()["gate_count"] == 4
    )


def test_the_physical_replay_tier_is_opt_in_and_reports_what_it_cannot_measure():
    """The replay entry point runs, labels missing truth, and writes nothing.

    Run against an empty directory so this stays a check on the harness rather
    than on twelve megabytes of fixture bytes; ``tests/quality_replay.py`` is
    deliberately not collected by pytest, and the tier it drives stays opt-in.
    """
    from tests import quality_replay

    assert quality_replay.DEFAULT_CAPTURE.name == "turntable"

    report, timing = quality_replay.run(
        FIXTURES,
        stride=32,
        frame_count=0,
        register=False,
        coverage_tolerance=1.0,
        plane_roi=None,
        plane_name=None,
        truth_path=None,
        max_plane_p95=None,
        max_dimension_error=None,
        max_landmark_offset=None,
        max_agreement_p95=None,
        max_icp_rmse=None,
        require_integrity=False,
        require_bridge_pass=False,
    )
    suite = report["suite"]
    assert isinstance(suite, dict)
    assert suite["gate_count"] == 0
    notes = [note for case in suite["cases"] for note in case["notes"]]
    assert any("plane roughness: not measured" in note for note in notes)
    assert any("landmark and dimension error: not measured" in note for note in notes)
    assert any(
        "110 x 95 x 24 mm" in note and "will not turn that sentence" in note for note in notes
    )
    # Timing is evidence about this machine, and it is kept out of the report
    # the quality module serialises so that report stays reproducible.
    assert "total_seconds" in timing
    assert "timing" not in suite


def test_the_replay_tier_refuses_the_recorded_bad_bridge_through_its_own_gate(tmp_path):
    """End to end through the tool: a capture carrying the 2026-08-28 record fails.

    The fixture is copied into a temporary directory rather than read in
    place, so the committed evidence cannot be touched by a tool that scans a
    capture root.
    """
    from tests import quality_replay

    segment = tmp_path / "segment-0001"
    segment.mkdir()
    (segment / "bridge.json").write_bytes(_fixture_bytes())

    assert quality_replay.main(["--capture", str(tmp_path), "--require-bridge-pass"]) == 1
    assert quality_replay.main(["--capture", str(tmp_path)]) == 0
    assert hashlib.sha256(_fixture_bytes()).hexdigest() == PHYSICAL_BRIDGE_SHA256


def test_the_replay_tier_refuses_a_capture_directory_that_is_not_one(tmp_path):
    """A missing capture is a sentence, not a traceback from deep inside a glob."""
    from tests import quality_replay

    with pytest.raises(ValueError, match="is not a directory"):
        quality_replay.run(
            tmp_path / "absent",
            stride=32,
            frame_count=0,
            register=False,
            coverage_tolerance=1.0,
            plane_roi=None,
            plane_name=None,
            truth_path=None,
            max_plane_p95=None,
            max_dimension_error=None,
            max_landmark_offset=None,
            max_agreement_p95=None,
            max_icp_rmse=None,
            require_integrity=False,
            require_bridge_pass=False,
        )
