"""What the surface oracle's contract must be true of, arm by arm and refusal by refusal.

The corpus itself is run by :mod:`tests.surface_oracle_harness`, and running it
is a command rather than a test - it fuses ninety-one arms and takes about a
minute. What is tested here is the machinery that makes that run mean anything:
that a perturbation cannot be filed under the wrong cause, that the two arms are
derived from the same frames and the same poses, that every measurement is
computed here rather than taken from an arm, and that each of the seven
corruption controls really does earn the refusal its fixture declares.

Two cases are loaded once for the whole module - the exact control and the
pose-translation case - because building a scene renders five 640x400 frames and
nothing in these tests needs a third scene to say what it says.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import scanner3d.pipeline as pipeline
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.reconstruction_export import (
    TSDF_MAX_DEPTH,
    TSDF_MIN_DEPTH,
    UPRIGHT_FROM_OPTICAL,
    model_points,
    upright_pose,
)
from scanner3d.simulate import FACTORY_INTRINSICS, Box, Shape, Sphere, render
from scanner3d.tsdf import TsdfConfig, TsdfVolume
from scanner3d.tsdf_surface import NoSurfaceFound, extract_surface
from tests import surface_oracle as contract
from tests import surface_oracle_harness as harness
from tests import surface_oracle_replay as replay

pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(scope="module")
def corpus() -> harness.Corpus:
    return harness.load_corpus()


def _case(corpus: harness.Corpus, case_id: str) -> harness.CaseSpec:
    entry = next(item for item in corpus.files if item.case_id == case_id)
    return harness.load_case(harness.CORPUS / entry.name, entry=entry, corpus=corpus)


@pytest.fixture(scope="module")
def exact(corpus: harness.Corpus) -> contract.Scene:
    return _case(corpus, "exact-control").scene


@pytest.fixture(scope="module")
def perturbed(corpus: harness.Corpus) -> contract.Scene:
    return _case(corpus, "pose-translation").scene


@pytest.fixture(scope="module")
def noisy(corpus: harness.Corpus) -> contract.Scene:
    return _case(corpus, "depth-noise").scene


CELL = "v15-t3-w1-uniform"


# --- the label is load-bearing --------------------------------------------------------


def test_a_perturbation_label_nobody_defined_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="is not one of"):
        contract.check_perturbation_label(contract.Perturbation(label="slightly-wobbly"))


def test_an_exact_case_that_actually_perturbs_something_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="control that is itself perturbed"):
        contract.check_perturbation_label(
            contract.Perturbation(label=contract.EXACT, depth_noise_codes=3)
        )


def test_a_depth_case_that_also_moves_the_poses_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="cannot be compared"):
        contract.check_perturbation_label(
            contract.Perturbation(
                label=contract.DEPTH_NOISE,
                depth_noise_codes=5,
                translation_bias=(0.1, 0.0, 0.0),
            )
        )


def test_a_pose_case_that_also_edits_the_depth_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="cannot be compared"):
        contract.check_perturbation_label(
            contract.Perturbation(
                label=contract.POSE_ROTATION,
                rotation_bias_degrees=0.5,
                depth_noise_codes=5,
            )
        )


def test_a_case_filed_under_the_wrong_depth_family_member_is_refused() -> None:
    """The forged-label control: outlier fields under the 'depth-noise' name."""
    with pytest.raises(contract.ScoringUnusable, match="does not carry the field"):
        contract.check_perturbation_label(
            contract.Perturbation(label=contract.DEPTH_NOISE, outlier_period=7, outlier_codes=40)
        )


def test_a_depth_case_carrying_a_second_familys_field_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="also carries"):
        contract.check_perturbation_label(
            contract.Perturbation(
                label=contract.DEPTH_NOISE,
                depth_noise_codes=5,
                band_period=7,
                band_width=2,
            )
        )


def test_the_combined_case_must_move_both_families() -> None:
    with pytest.raises(contract.ScoringUnusable, match="duplicate of that family"):
        contract.check_perturbation_label(
            contract.Perturbation(label=contract.COMBINED, depth_noise_codes=5)
        )


def test_every_declared_perturbation_belongs_to_exactly_one_family_or_is_the_interaction() -> None:
    families = set(contract.DEPTH_FAMILY) | set(contract.POSE_FAMILY)
    assert families < set(contract.PERTURBATIONS)
    assert not set(contract.DEPTH_FAMILY) & set(contract.POSE_FAMILY)
    assert set(contract.PERTURBATIONS) - families == {contract.EXACT, contract.COMBINED}


# --- the perturbation itself ----------------------------------------------------------


def test_draws_depend_on_the_seed_text_and_nothing_else() -> None:
    first = contract.draws("surface-oracle:depth-noise", 64)
    assert first == contract.draws("surface-oracle:depth-noise", 64)
    assert first != contract.draws("surface-oracle:depth-outliers", 64)
    assert all(0.0 <= value < 1.0 for value in first)


def test_depth_noise_is_bounded_and_never_invents_a_return(noisy: contract.Scene) -> None:
    """Zero-mean bounded noise moves a code and never turns a no-return into one."""
    magnitude = noisy.perturbation.depth_noise_codes
    assert magnitude > 0
    for view in noisy.views:
        truth = noisy.truth_codes(view.index)
        measured = noisy.raw_codes(view.index)
        assert len(truth) == len(measured)
        for exact_code, measured_code in zip(truth, measured, strict=True):
            if exact_code == contract.NO_RETURN:
                assert measured_code == contract.NO_RETURN
            else:
                assert measured_code != contract.NO_RETURN
                assert abs(measured_code - exact_code) <= magnitude


def test_a_pose_perturbation_leaves_every_measured_frame_byte_identical(
    perturbed: contract.Scene,
) -> None:
    """The whole point of separating pose from depth: only the pose moved."""
    assert not perturbed.perturbation.touches_depth
    for view in perturbed.views:
        assert view.raw == view.truth
        assert view.fused_pose.as_dict() != view.true_pose.as_dict()


def test_an_exact_scene_gives_the_fusion_the_pose_the_camera_really_had(
    exact: contract.Scene,
) -> None:
    for view in exact.views:
        assert view.fused_pose.as_dict() == view.true_pose.as_dict()
        assert view.raw == view.truth


def test_perturb_codes_refuses_a_window_it_was_not_given(exact: contract.Scene) -> None:
    with pytest.raises(contract.ScoringUnusable, match="code"):
        contract.perturb_codes(
            (1, 2, 3),
            contract.Perturbation(label=contract.DEPTH_NOISE, depth_noise_codes=2),
            window=exact.window,
            view=0,
            seed="x",
        )


# --- the honest arms ------------------------------------------------------------------


def test_both_honest_arms_pass_their_own_validation(exact: contract.Scene) -> None:
    for result in (contract.raw_points(exact), contract.fused(exact, CELL)):
        contract.validate(exact, result)


def test_the_two_arms_are_given_the_same_frames_and_the_same_poses(
    perturbed: contract.Scene,
) -> None:
    dots = contract.raw_points(perturbed)
    surface = contract.fused(perturbed, CELL)
    assert dots.input_sha256 == surface.input_sha256
    assert [pose.as_dict() for pose in dots.poses_used] == [
        pose.as_dict() for pose in surface.poses_used
    ]
    assert dots.views_used == surface.views_used
    assert dots.arm != surface.arm


def test_fusion_is_a_pure_function_of_the_case_and_the_cell(exact: contract.Scene) -> None:
    first, first_record = contract.fuse(exact, exact.cell(CELL))
    second, second_record = contract.fuse(exact, exact.cell(CELL))
    assert first == second
    assert first_record == second_record


def test_a_coarser_cell_costs_less_work_and_a_finer_one_costs_more(exact: contract.Scene) -> None:
    coarse = contract.fused(exact, "v25-t3-w1-uniform")
    fine = contract.fused(exact, "v10-t4-w1-uniform")
    assert coarse.work.claims["work_units"] < fine.work.claims["work_units"]
    assert fine.work.claims["work_units"] <= exact.work_budget


def test_no_arm_may_spend_past_the_fixtures_own_ceiling(exact: contract.Scene) -> None:
    for cell in exact.grid:
        result = contract.fused(exact, cell.cell_id)
        assert result.work.claims["work_units"] <= exact.work_budget
        assert result.work.work_budget == exact.work_budget


# --- the measurements -----------------------------------------------------------------


def test_every_measurement_is_published_with_a_kind_the_reader_knows(
    exact: contract.Scene,
) -> None:
    measurements = contract.measure(exact, contract.fused(exact, CELL))
    kinds = {measurement.as_dict()["kind"] for measurement in measurements}
    assert kinds <= set(harness.MEASUREMENT_KINDS)
    assert "surface_oracle_field" in kinds


def test_the_raw_arm_publishes_no_field_because_it_allocated_none(
    exact: contract.Scene,
) -> None:
    dots = contract.raw_points(exact)
    kinds = {measurement.as_dict()["kind"] for measurement in contract.measure(exact, dots)}
    assert "surface_oracle_field" not in kinds


def test_the_dots_agree_exactly_with_the_returns_that_produced_them(
    exact: contract.Scene,
) -> None:
    """The corpus's known-good case, and the floor every other number is read against.

    Note WHAT is exactly zero here. The dots are the ideal returns, so their
    distance to the ideal returns is zero by construction and says nothing about
    accuracy - which is the confusion that produced round two's wrong headline.
    The independent statement beside it is the analytic one.
    """
    published = {
        measurement.as_dict()["name"]: measurement.as_dict()
        for measurement in contract.measure(exact, contract.raw_points(exact))
    }
    support = published["support"]
    assert support["observed_to_reference"]["coverage_fraction"] == 1.0  # type: ignore[index]
    assert support["observed_to_reference"]["distances"]["p95"] == 0.0  # type: ignore[index]
    assert "not a truth-accuracy measure" in contract.SUPPORT_SOURCE
    assert published["dimensions"]["absolute"]["worst"] == 0.0  # type: ignore[index]
    assert published["geometry"]["outside_volume"] == 0


def test_the_two_references_are_published_under_different_names(exact: contract.Scene) -> None:
    """Analytic accuracy and measurement support are separate measurements, always."""
    published = {
        measurement.as_dict()["name"]: measurement.as_dict()
        for measurement in contract.measure(exact, contract.fused(exact, CELL))
    }
    assert published["analytic"]["truth_source"] == contract.ANALYTIC_SOURCE
    assert published["support"]["kind"] == "cloud_agreement"
    assert "closed form" in contract.ANALYTIC_SOURCE
    # The one that matters: the analytic reference is denser than the sensor's
    # own sampling, so a vertex between two rays is measured rather than missed.
    analytic = published["analytic"]
    geometry = published["geometry"]
    assert int(str(analytic["reference_points"])) > int(str(geometry["truth_points"]))


def test_a_dimension_is_never_offered_as_a_physical_length(exact: contract.Scene) -> None:
    published = {
        measurement.as_dict()["name"]: measurement.as_dict()
        for measurement in contract.measure(exact, contract.raw_points(exact))
    }
    assert "No physical length" in str(published["geometry"]["unit"])
    assert "no physical unit" in str(published["dimensions"]["truth_source"]).lower()


def test_decimation_is_a_fixed_lattice_and_applies_to_both_sides(exact: contract.Scene) -> None:
    points = contract.measured_points(exact)
    once = contract.decimate(points, exact.regions.truth_pitch)
    assert contract.decimate(once, exact.regions.truth_pitch) == once
    assert len(once) <= len(points)


def test_components_are_a_partition_of_the_population(exact: contract.Scene) -> None:
    points = contract.decimate(contract.measured_points(exact), exact.regions.truth_pitch)
    sizes = contract.component_sizes(points, exact.regions.linkage_radius)
    assert sum(sizes) == len(points)
    assert sizes == sorted(sizes, reverse=True)
    assert contract.component_sizes((), 1.0) == []


# --- the corruption controls ------------------------------------------------------------


def _refusal(scene: contract.Scene, result: contract.ArmResult) -> str:
    with pytest.raises(contract.CandidateRefused) as caught:
        contract.validate(scene, result)
    return caught.value.refusal


def test_fusing_the_truth_frames_is_refused_as_an_edited_measurement(
    noisy: contract.Scene,
) -> None:
    """The impossible-improvement control. Its surface is better; that is the point."""
    forged = contract.fused_from_truth_frames(noisy, CELL)
    assert _refusal(noisy, forged) == "raw_frame_edited"


def test_dropping_the_view_that_disagrees_is_refused(exact: contract.Scene) -> None:
    assert _refusal(exact, contract.fused_dropping_a_view(exact, CELL)) == "views_omitted"


def test_reaching_for_the_poses_the_scene_kept_back_is_refused(
    perturbed: contract.Scene,
) -> None:
    forged = contract.fused_with_true_poses(perturbed, CELL)
    assert _refusal(perturbed, forged) == "pose_substituted"


def test_editing_the_surface_after_extraction_is_refused(perturbed: contract.Scene) -> None:
    forged = contract.fused_with_edited_surface(perturbed, CELL)
    assert _refusal(perturbed, forged) == "surface_not_derived"


def test_the_edited_surface_control_is_refused_even_when_nothing_was_bridged(
    exact: contract.Scene,
) -> None:
    """A control that quietly becomes a copy of the honest arm tests nothing.

    On the exact control the honest surface puts no vertex in the protected
    volume, so the realistic deletion removes nothing. The control still has to
    publish geometry the field did not produce, or the reader would be graded
    on an attack that was never made.
    """
    honest = contract.fused(exact, CELL)
    forged = contract.fused_with_edited_surface(exact, CELL)
    assert forged.points != honest.points
    assert len(forged.points) == len(honest.points) - 1
    assert _refusal(exact, forged) == "surface_not_derived"


def test_the_raw_arm_dropping_its_hardest_samples_is_refused(exact: contract.Scene) -> None:
    forged = contract.raw_points_dropping_hard_samples(exact)
    assert _refusal(exact, forged) == "sample_count_mismatch"


def test_a_work_record_that_flatters_the_parameter_choice_is_refused(
    exact: contract.Scene,
) -> None:
    forged = contract.fused_overclaiming_work(exact, CELL)
    assert _refusal(exact, forged) == "claim_not_supported"


def test_an_arm_that_publishes_a_bar_of_its_own_is_refused(exact: contract.Scene) -> None:
    forged = contract.fused_claiming_the_unmeasurable(exact, CELL)
    assert _refusal(exact, forged) == "claim_unknown"


def test_every_declared_refusal_is_one_this_module_can_actually_raise() -> None:
    assert len(set(contract.REFUSALS)) == len(contract.REFUSALS)
    assert set(harness.CONTROL_REFUSALS) <= set(contract.REFUSALS)


# --- the remaining refusals, which no fixture arm produces --------------------------------


def _honest(scene: contract.Scene) -> contract.ArmResult:
    return contract.fused(scene, CELL)


def test_an_arm_this_oracle_cannot_re_derive_is_refused(exact: contract.Scene) -> None:
    assert _refusal(exact, replace(_honest(exact), arm="hand_polished")) == "arm_unknown"


def test_a_fused_arm_that_names_no_cell_is_refused(exact: contract.Scene) -> None:
    assert _refusal(exact, replace(_honest(exact), cell_id=None)) == "grid_cell_unknown"


def test_the_raw_arm_naming_a_parameter_cell_is_refused(exact: contract.Scene) -> None:
    forged = replace(contract.raw_points(exact), cell_id=CELL)
    assert _refusal(exact, forged) == "grid_cell_unknown"


def test_a_cell_the_corpus_never_declared_is_refused(exact: contract.Scene) -> None:
    with pytest.raises(contract.CandidateRefused) as caught:
        contract.fused(exact, "v01-t9-w0-magic")
    assert caught.value.refusal == "grid_cell_unknown"


def test_a_point_that_is_not_finite_is_refused(exact: contract.Scene) -> None:
    honest = _honest(exact)
    forged = replace(honest, points=(*honest.points[:-1], (0.0, math.nan, 0.0)))
    assert _refusal(exact, forged) == "point_not_finite"


def test_geometry_outside_the_shared_volume_is_refused(exact: contract.Scene) -> None:
    honest = _honest(exact)
    forged = replace(honest, points=(*honest.points, (0.0, 0.0, 1000.0)))
    assert _refusal(exact, forged) == "points_outside_volume"

    dots = contract.raw_points(exact)
    outside = replace(dots, points=(*dots.points[:-1], (0.0, 0.0, 1000.0)))
    assert _refusal(exact, outside) == "points_outside_volume"


def test_a_raw_arm_offering_points_it_did_not_measure_is_refused(exact: contract.Scene) -> None:
    dots = contract.raw_points(exact)
    moved = (dots.points[0][0] + 0.01, dots.points[0][1], dots.points[0][2])
    forged = replace(dots, points=(moved, *dots.points[1:]))
    assert _refusal(exact, forged) == "points_not_measured"


def test_a_spend_past_the_ceiling_is_refused(exact: contract.Scene) -> None:
    honest = _honest(exact)
    starved = replace(exact, work_budget=1)
    assert _refusal(starved, replace(honest, work=honest.work)) in {
        "work_budget_exceeded",
        "claim_not_supported",
    }


# --- what a fixture may name ---------------------------------------------------------------


def test_the_appliance_arm_takes_no_cell() -> None:
    with pytest.raises(contract.ScoringUnusable, match="takes no parameter cell"):
        contract.candidate("fused_at_appliance_resolution", CELL)


# --- two references, and the difference between them ------------------------------------------


def test_analytic_distance_is_the_closed_form_distance_to_each_declared_shape(
    exact: contract.Scene,
) -> None:
    """Exact, not a bound, on the plane and on the curved surface."""
    shapes = exact.shapes
    assert contract.analytic_distance(shapes, (0.0, 0.0, 12.0)) == pytest.approx(0.0, abs=1e-12)
    assert contract.analytic_distance(shapes, (0.0, 0.0, 12.5)) == pytest.approx(0.5)
    # The sphere's own centre is exactly its radius from its surface, and the
    # union's nearest boundary there is the sphere rather than anything else.
    assert contract.analytic_distance(shapes, (-1.2, 0.0, 9.2)) == pytest.approx(0.9)


def test_shapes_that_are_not_disjoint_are_refused_rather_than_approximated() -> None:
    """``min`` over boundaries is the union's boundary only while nothing is buried."""
    from scanner3d.simulate import Sphere

    with pytest.raises(contract.ScoringUnusable, match="not strictly disjoint"):
        contract.check_shapes_are_separable(
            [Sphere(centre=(0.0, 0.0, 9.0), radius=1.0), Sphere(centre=(0.5, 0.0, 9.0), radius=1.0)]
        )


# --- the shipped resolution policy, which is not a grid cell -----------------------------------


def test_the_appliance_resolution_is_finer_than_every_cell_the_grid_declares(
    exact: contract.Scene,
) -> None:
    """The reason this arm exists: the declared envelope is a proxy, not the appliance."""
    derived = contract.appliance_cell(exact)
    assert derived.cell_id == contract.APPLIANCE_CELL_ID
    assert derived.voxel_size < min(cell.voxel_size for cell in exact.grid)
    record = contract.appliance_resolution(exact)
    assert record["cell"]["voxel_size"] == derived.voxel_size  # type: ignore[index]
    assert record["voxel_budget"] == exact.appliance.voxel_budget


def test_the_appliance_policy_must_be_the_one_the_product_ships_today(
    exact: contract.Scene,
) -> None:
    drifted = replace(exact.appliance, truncation_voxels=exact.appliance.truncation_voxels + 1.0)
    with pytest.raises(contract.ScoringUnusable, match="drifted"):
        contract.check_appliance_policy(drifted)


def test_the_appliance_budget_is_the_desktop_reconstruct_actions_own(
    exact: contract.Scene,
) -> None:
    """The budget is what sets the resolution, so it is bound rather than declared.

    Offline and importable: the constant is read from :mod:`scanner3d.gui`
    without constructing anything. Checking the truncation, the minimum weight
    and the weight policy alone would leave the one number that decides how
    fine this arm is free for a fixture to choose, and then "the appliance
    resolution" would be a fourth voxel size under a borrowed name.
    """
    from scanner3d.gui import RECONSTRUCT_VOXEL_BUDGET

    assert exact.appliance.voxel_budget == RECONSTRUCT_VOXEL_BUDGET
    drifted = replace(exact.appliance, voxel_budget=RECONSTRUCT_VOXEL_BUDGET * 2)
    with pytest.raises(contract.ScoringUnusable, match="voxel_budget"):
        contract.check_appliance_policy(drifted)


def test_a_fixture_may_not_declare_a_cell_under_the_appliances_name(
    exact: contract.Scene,
) -> None:
    stolen = replace(exact.grid[0], cell_id=contract.APPLIANCE_CELL_ID)
    with pytest.raises(contract.ScoringUnusable, match="belongs to the derived"):
        replace(exact, grid=(stolen, *exact.grid[1:]))


# --- the surface is not its vertices ----------------------------------------------------------


def _hand_built(scene: contract.Scene, points, faces) -> contract.ArmResult:
    """One arm whose geometry is written out here rather than extracted.

    These do not go through :func:`contract.validate` - they are not results and
    could never be - because what is under test is the MEASUREMENT: given this
    exact surface, does the oracle see what is wrong with it?
    """
    return contract.ArmResult(
        name="hand-built",
        arm=contract.FUSED,
        cell_id=CELL,
        views_used=tuple(range(len(scene.views))),
        poses_used=scene.fused_poses(),
        input_sha256=tuple(view.raw_sha256 for view in scene.views),
        points=tuple(points),
        work=contract.WorkClaim(claims={}, work_budget=scene.work_budget),
        faces=tuple(faces),
    )


def test_the_interior_lattice_is_strictly_inside_the_triangle_and_deterministic() -> None:
    for order in (1, 2, 3, 4):
        lattice = contract.barycentric_lattice(order)
        assert len(lattice) == order * order
        assert lattice == contract.barycentric_lattice(order)
        for u, v, w in lattice:
            assert u > 0.0 and v > 0.0 and w > 0.0
            assert u + v + w == pytest.approx(1.0)


def test_the_interior_lattice_samples_equal_areas_so_an_area_may_be_inferred() -> None:
    """The property ``protected_area`` rests on, checked rather than assumed.

    A regular subdivision at order n cuts a triangle into n**2 sub-triangles of
    identical area, and one centroid per sub-triangle therefore samples every
    part of the face equally. The exactly checkable consequence is that the
    samples' own centroid is the triangle's centroid: the mean of the
    barycentric coordinates is (1/3, 1/3, 1/3), so the mean of any linear
    function over the lattice is its exact mean over the triangle.

    The upright-only half lattice this replaced does not have that property -
    it leaves every inverted sub-triangle unsampled and pulls its centroid
    towards the corners - which is why a fraction of samples could not have
    been read as a fraction of area under it.
    """
    for order in (1, 2, 3, 4, 7):
        lattice = contract.barycentric_lattice(order)
        assert len(lattice) == order * order
        for axis in range(3):
            mean = sum(point[axis] for point in lattice) / len(lattice)
            assert mean == pytest.approx(1.0 / 3.0)


def test_a_lattice_that_samples_no_interior_at_all_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="takes no sample inside"):
        contract.barycentric_lattice(0)


def test_a_mesh_past_the_fixtures_own_sample_ceiling_is_refused(exact: contract.Scene) -> None:
    """Bounded before anything is built, and a refusal rather than a thinner sampling."""
    starved = replace(exact, regions=replace(exact.regions, max_triangle_samples=1))
    honest = contract.fused(exact, CELL)
    with pytest.raises(contract.ScoringUnusable, match="against the fixture's ceiling"):
        contract.surface_samples(starved, honest, support=contract.truth_points(starved))


def test_a_triangle_interpolated_between_two_camera_rays_is_accurate(
    exact: contract.Scene,
) -> None:
    """The failure round two reported. A valid between-ray surface must PASS accuracy.

    Every corner of this triangle lies exactly on the backing plane the scene
    really has, at offsets no camera ray landed on. Measured against the ideal
    returns it is unsupported everywhere; measured against the object it is
    exactly right, and only the second of those is an accuracy claim.
    """
    corners = ((0.013, 0.017, 12.0), (0.41, -0.23, 12.0), (-0.29, 0.37, 12.0))
    arm = _hand_built(exact, corners, [(0, 1, 2)])
    support = contract.truth_points(exact)
    evidence = contract.surface_samples(exact, arm, support=support).as_dict()
    assert evidence["within_tolerance_fraction"] == 1.0
    assert evidence["distances"]["worst"] == pytest.approx(0.0, abs=1e-12)  # type: ignore[index]
    assert evidence["bridging_faces"] == 0


def test_a_triangle_spanning_the_gap_is_invisible_to_every_vertex_measurement(
    exact: contract.Scene,
) -> None:
    """The defect the ablation cannot see, and the reason interiors are sampled.

    All three corners are outside the protected volume - so the vertex-only
    bridging count, which is what ``edge.bridged_points`` measures, is zero -
    while the triangle's own interior crosses it.
    """
    corners = ((2.10, -0.50, 9.00), (2.10, 0.50, 9.00), (2.60, 0.00, 10.50))
    assert not any(exact.regions.protected.contains(corner) for corner in corners)
    arm = _hand_built(exact, corners, [(0, 1, 2)])
    support = contract.truth_points(exact)
    evidence = contract.surface_samples(exact, arm, support=support).as_dict()
    assert int(str(evidence["protected_samples"])) > 0
    assert evidence["bridging_faces"] == 1
    assert evidence["bridging_faces_with_every_vertex_outside"] == 1
    protected_area = float(str(evidence["protected_area"]))
    assert 0.0 < protected_area <= float(str(evidence["surface_area"]))


def test_a_sealed_shell_nothing_measured_is_invisible_to_point_linkage(
    exact: contract.Scene,
) -> None:
    """Two identical tetrahedra, one where the sensor looked and one where it did not.

    ``components`` links POINTS by radius and reports the same thing for both,
    because four points half a unit across are four points half a unit across
    wherever they stand. The topology knows one of them is a closed solid the
    reconstruction invented.
    """
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]

    def tetra(centre):
        x, y, z = centre
        return (
            (x - 0.05, y - 0.05, z - 0.05),
            (x + 0.05, y - 0.05, z - 0.05),
            (x, y + 0.05, z - 0.05),
            (x, y, z + 0.05),
        )

    support = contract.truth_points(exact)
    # Inside the protected volume, which the scene declares empty.
    invented = contract.surface_topology(
        exact, _hand_built(exact, tetra((2.375, 0.0, 9.9)), faces), support=support
    ).as_dict()
    # On the backing plane, where five views measured returns.
    measured = contract.surface_topology(
        exact, _hand_built(exact, tetra((0.0, 0.0, 12.0)), faces), support=support
    ).as_dict()

    assert invented["closed"] is True
    assert measured["closed"] is True
    assert invented["closed_shells"] == 1
    assert measured["closed_shells"] == 1
    assert invented["unsupported_closed_shells"] == 1
    assert invented["unsupported_closed_shell_faces"] == 4
    assert measured["unsupported_closed_shells"] == 0
    # And the measurement that cannot tell them apart, saying so.
    radius = exact.regions.linkage_radius
    assert contract.component_sizes(tetra((2.375, 0.0, 9.9)), radius) == contract.component_sizes(
        tetra((0.0, 0.0, 12.0)), radius
    )


def test_an_unsupported_shell_is_split_by_truth_and_not_by_support_alone(
    exact: contract.Scene,
) -> None:
    """Two sealed shells nothing measured: one on the object, one in free space.

    This is the counterexample the round-three review found in the corpus,
    reduced to its smallest form. Both shells are unsupported - no ideal return
    is anywhere near either of them - so a rule that pruned unsupported shells
    would delete both. Only one of them is false surface: the other lies
    exactly on the backing plane the scene really has, outside the window any
    view sampled, and deleting it would delete correct geometry.
    """
    tetra = (
        (2.325, -0.05, 9.85),
        (2.425, -0.05, 9.85),
        (2.375, 0.05, 9.85),
        (2.375, 0.0, 9.95),
    )
    # Two oppositely wound triangles on the plane z = 12: every edge is used
    # twice, so the shell closes, and every point of it - corner or interior -
    # is exactly on the plane.
    sheet = ((6.0, 3.0, 12.0), (6.4, 3.0, 12.0), (6.2, 3.4, 12.0))
    points = (*tetra, *sheet)
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 1),
        (1, 3, 2),
        (4, 5, 6),
        (4, 6, 5),
    ]
    arm = _hand_built(exact, points, faces)
    evidence = contract.surface_topology(exact, arm, support=contract.truth_points(exact)).as_dict()

    assert evidence["closed_shells"] == 2
    assert evidence["unsupported_closed_shells"] == 2
    assert evidence["unsupported_closed_shell_faces"] == 6
    # The split, and the whole point of it.
    assert evidence["unsupported_inaccurate_closed_shells"] == 1
    assert evidence["unsupported_inaccurate_closed_shell_faces"] == 4
    assert evidence["unsupported_accurate_closed_shells"] == 1
    assert evidence["unsupported_accurate_closed_shell_faces"] == 2
    tolerance = exact.regions.coverage_tolerance
    assert float(str(evidence["unsupported_inaccurate_worst_distance"])) > tolerance
    assert float(str(evidence["unsupported_accurate_worst_distance"])) == pytest.approx(
        0.0, abs=1e-12
    )
    # And the two groups account for every unsupported shell, so no rule can
    # quietly read one of them as the whole.
    assert (
        int(str(evidence["unsupported_inaccurate_closed_shells"]))
        + int(str(evidence["unsupported_accurate_closed_shells"]))
        == evidence["unsupported_closed_shells"]
    )


def test_the_extractors_own_surface_is_manifold_and_shares_its_vertices(
    exact: contract.Scene,
) -> None:
    """The promise scanner3d.tsdf_surface documents, checked rather than assumed."""
    topology = contract.surface_topology(
        exact, contract.fused(exact, CELL), support=contract.truth_points(exact)
    ).as_dict()
    assert topology["nonmanifold_edge_count"] == 0
    assert topology["degenerate_faces"] == 0
    assert topology["referenced_vertices"] == topology["vertices"]


def test_the_bridging_triangle_control_forges_the_faces_and_nothing_else(
    exact: contract.Scene,
) -> None:
    """The corruption control the round-three review asked for, end to end."""
    honest = contract.fused(exact, CELL)
    forged = contract.fused_with_bridging_triangle(exact, CELL)
    assert forged.points == honest.points
    assert forged.faces[:-1] == honest.faces
    assert _refusal(exact, forged) == "surface_not_derived"
    support = contract.truth_points(exact)
    before = contract.surface_samples(exact, honest, support=support).as_dict()
    after = contract.surface_samples(exact, forged, support=support).as_dict()
    assert int(str(after["bridging_faces"])) == int(str(before["bridging_faces"])) + 1
    assert int(str(after["bridging_faces_with_every_vertex_outside"])) >= 1


def test_the_raw_arm_may_not_bring_faces_it_never_meshed(exact: contract.Scene) -> None:
    dots = contract.raw_points(exact)
    assert _refusal(exact, replace(dots, faces=((0, 1, 2),))) == "surface_not_derived"


def test_a_candidate_form_nobody_wrote_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="is not one of"):
        contract.candidate("fused_but_nicer", None)


def test_a_fused_form_without_a_cell_is_refused() -> None:
    with pytest.raises(contract.ScoringUnusable, match="needs the id of a grid cell"):
        contract.candidate("fused", None)


def test_the_raw_forms_take_no_cell() -> None:
    with pytest.raises(contract.ScoringUnusable, match="takes no parameter cell"):
        contract.candidate("raw_points", CELL)
    with pytest.raises(contract.ScoringUnusable, match="takes no parameter cell"):
        contract.candidate("raw_points_dropping_hard_samples", CELL)


# --- task 032's negative result, preserved by construction ------------------------------------


def test_no_completed_sample_is_ever_offered_to_a_volume() -> None:
    """Task 032 proved re-fusing completed samples adds no coverage. Nothing here re-opens it.

    The check is structural rather than a comment: this corpus's contract does
    not import the weighted-input module at all, so there is no path by which a
    reconstructed sample could reach a field here.
    """
    source = Path(str(contract.__file__)).read_text(encoding="utf-8")
    assert "tsdf_input" not in source
    assert "depth_completion" not in source
    assert "depth_temporal" not in source


# --- the physical replay, which has no truth and must not pretend otherwise --------------------


def test_the_replay_binds_a_capture_by_its_manifest_and_checks_every_frame() -> None:
    identity, paths = replay.bind_identity(replay.DEFAULT_CAPTURE, limit=3)
    assert identity["bound_by"] == (
        "manifest.json; every frame's digest was checked against it before use"
    )
    assert identity["frames_used"] == 3
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_a_capture_whose_bytes_do_not_match_its_manifest_is_refused(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "bed-01.bin").write_bytes(b"\x00\x01" * 8)
    (capture / "manifest.json").write_text(
        json.dumps({"frames": [{"name": "bed-01", "sha256": "0" * 64}]}), encoding="utf-8"
    )
    with pytest.raises(replay.ReplayRefused, match="not the capture this manifest describes"):
        replay.bind_identity(capture, limit=None)


def test_a_manifest_naming_a_path_rather_than_a_frame_is_refused(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text(
        json.dumps({"frames": [{"name": "../secrets"}]}), encoding="utf-8"
    )
    with pytest.raises(replay.ReplayRefused, match="which is a path"):
        replay.bind_identity(capture, limit=None)


def test_a_folder_with_neither_a_manifest_nor_frames_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(replay.ReplayRefused, match="will not measure a folder"):
        replay.bind_identity(empty, limit=None)


def test_return_coverage_counts_the_sensors_own_no_return_code() -> None:
    assert replay.return_coverage(b"\x00\x00\x01\x00\x00\x00\x02\x00") == {
        "samples": 4,
        "returned": 2,
        "returned_fraction": 0.5,
    }


def test_a_roughness_region_is_never_chosen_for_the_caller() -> None:
    absent = replay._plane([(0.0, 0.0, 0.0)], None, "unnamed")
    assert absent["measured"] is False
    assert "never chooses a region for the caller" in str(absent["why"])


def test_the_replay_proves_it_did_not_write_to_the_capture() -> None:
    before = replay.fingerprint(replay.DEFAULT_CAPTURE)
    after = replay.fingerprint(replay.DEFAULT_CAPTURE)
    assert replay.verify_unchanged(before, after)["unchanged"] is True
    tampered = {name: dict(record) for name, record in after.items()}
    first = sorted(tampered)[0]
    tampered[first]["sha256"] = "0" * 64
    verdict = replay.verify_unchanged(before, tampered)
    assert verdict["unchanged"] is False
    assert verdict["changed"] == [first]


def test_the_replay_publishes_no_truth_referenced_measurement() -> None:
    """There is no truth for a capture, so the words for one may not appear."""
    published = " ".join(str(item) for item in replay.LIMITATIONS)
    assert "no truth model for a physical capture" in published
    assert "no dimensional accuracy" in published
    source = Path(str(replay.__file__)).read_text(encoding="utf-8")
    assert "dimension_errors" not in source
    assert "cloud_agreement" not in source


# --- one frame, from the bounds to the fusion -------------------------------------------------

#: How many frames the fused replay control renders. Nine is the fewest
#: :func:`scanner3d.pipeline.reconstruct_sweep` will accept - its revisit search
#: needs a gap of eight - so ten is the smallest sweep with anything to spare.
REPLAY_FRAMES = 10

#: The voxel ceiling the fused control hands the replay. Far below the shipped
#: :data:`scanner3d.tsdf.MAX_VOXELS` a physical run uses, because this control
#: is about WHICH POSES reach the field rather than how finely it is diced, and
#: a full-budget field would spend minutes proving the same thing. The chooser
#: refuses anything above the module ceiling, so this argument narrows and can
#: never widen.
REPLAY_VOXEL_BUDGET = 20_000


@dataclass(frozen=True)
class FusedReplay:
    """One complete fused replay, and everything it was asked and answered."""

    capture: Path
    raws: tuple[bytes, ...]
    report: dict[str, object]
    #: The poses the REGISTRATION fitted, straight off its own result.
    registered_at: tuple[Rigid, ...]
    #: The poses the FIELD BOUNDS were measured from, as ``model_points`` saw them.
    measured_at: tuple[Rigid, ...]
    #: The poses each cell's fusion was given, as ``TsdfVolume.integrate`` saw them.
    fused_at: tuple[tuple[Rigid, ...], ...]
    #: The configs the replay built, so a control can fuse the SAME field again.
    configs: tuple[TsdfConfig, ...]
    before: dict[str, dict[str, object]]
    after: dict[str, dict[str, object]]


def _replay_capture(root: Path) -> Path:
    """A real capture folder: ten 512000-byte frames of an ASYMMETRIC scene.

    Rendered at the scale this repository's recorded captures are in - raw codes
    times :data:`~scanner3d.cli.DEPTH_SCALE_METRES`, which puts this object at a
    depth of about 120 - so the fusion projects through the appliance's own
    depth window rather than past it.

    A sphere alone would be useless here: it looks identical from every angle,
    so a turntable of one carries no information about how far it turned and the
    registration recovers the identity. The box beside it is what makes the
    recovered trajectory a real one, which is what this control needs - an
    identity trajectory would leave "the poses" and "the poses turned upright"
    telling apart only by the presentation half turn, and the point of the
    control is that BOTH the pose and the frame have to be right.
    """
    capture = root / "capture"
    capture.mkdir()
    shapes: list[Shape] = [
        Sphere(centre=(0.0, 0.0, 120.0), radius=28.0),
        Box(
            half_extents=(9.0, 16.0, 9.0),
            pose=Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (26.0, 10.0, 112.0)),
        ),
    ]
    for index in range(REPLAY_FRAMES):
        turned = rotation_about_axis((0.0, 1.0, 0.0), 0.06 * index, through=(0.0, 0.0, 120.0))
        frame = render(
            shapes, intrinsics=FACTORY_INTRINSICS, depth_scale=DEPTH_SCALE_METRES, pose=turned
        )
        (capture / f"frame_{index:04d}.raw").write_bytes(frame.raw)
    return capture


@pytest.fixture(scope="module")
def fused(tmp_path_factory: pytest.TempPathFactory) -> FusedReplay:
    """One real replay that gets past readiness, with both pose handoffs recorded.

    Built once for the module because it renders ten full frames and runs the
    real registration; nothing below needs a second one. Nothing is stubbed:
    the preparation, the sweep, the readiness gates, the chooser, the fields and
    the extractor are all the shipped code. The two recorders are pass-throughs
    that write down what they were handed and delegate, so what is asserted
    afterwards is what really happened rather than what a stand-in was told.
    """
    capture = _replay_capture(tmp_path_factory.mktemp("surface-oracle-replay"))
    registered_at: list[Rigid] = []
    measured_at: list[list[Rigid]] = []
    fused_at: list[list[Rigid]] = []
    configs: list[TsdfConfig] = []

    sweep = pipeline.reconstruct_sweep

    def recording_sweep(clouds: object, **options: object) -> object:
        result = sweep(clouds, **options)  # type: ignore[arg-type]
        registered_at.extend(result.poses)
        return result

    def recording_model_points(
        clouds: Sequence[Sequence[tuple[float, float, float]]], poses: Sequence[Rigid]
    ) -> object:
        measured_at.append(list(poses))
        return model_points(clouds, poses)

    class RecordingVolume(TsdfVolume):
        def __init__(self, config: TsdfConfig, *, backend: str) -> None:
            configs.append(config)
            super().__init__(config, backend=backend)

        def integrate(
            self,
            frames: Sequence[bytes],
            poses: Sequence[Rigid],
            *,
            intrinsics: IntrinsicsRecord,
            depth_scale: float,
        ) -> None:
            fused_at.append(list(poses))
            super().integrate(frames, poses, intrinsics=intrinsics, depth_scale=depth_scale)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "reconstruct_sweep", recording_sweep)
        patch.setattr(replay, "model_points", recording_model_points)
        patch.setattr(replay, "TsdfVolume", RecordingVolume)
        before = replay.fingerprint(capture)
        report = replay.replay(
            capture,
            limit=None,
            backend=replay.BACKEND,
            plane_roi=None,
            plane_name="no region was named",
            voxel_budget=REPLAY_VOXEL_BUDGET,
        )
        after = replay.fingerprint(capture)

    return FusedReplay(
        capture=capture,
        raws=tuple(path.read_bytes() for path in sorted(capture.glob("frame_*.raw"))),
        report=report,
        registered_at=tuple(registered_at),
        measured_at=tuple(measured_at[0]),
        fused_at=tuple(tuple(poses) for poses in fused_at),
        configs=tuple(configs),
        before=before,
        after=after,
    )


def _cells(report: dict[str, object]) -> list[dict[str, object]]:
    cells = report["sensitivity"]
    assert isinstance(cells, list)
    return cells


def test_the_replay_reaches_fusion_at_all_so_the_controls_below_mean_something(
    fused: FusedReplay,
) -> None:
    """The premise. A replay that stopped early would pass every claim vacuously."""
    assert "stopped_at" not in fused.report
    assert fused.report["ready_before_fusion"] is True
    assert fused.report["readiness_reasons"] == []
    assert len(fused.measured_at) == REPLAY_FRAMES
    assert len(fused.fused_at) == len(_cells(fused.report))


def test_the_replay_fuses_at_the_very_poses_its_field_was_measured_from(
    fused: FusedReplay,
) -> None:
    """The defect, stated as the invariant it broke.

    ``TsdfVolume.integrate`` inverts each pose to carry a voxel back into its
    camera, so the poses that placed the points a field was BOUNDED from must be
    the poses that field is FUSED at. The replay used to bound with
    ``model_points(clouds, result.poses)`` and fuse with
    ``upright_pose(result.poses)`` - the same trajectory a presentation half turn
    later - and every voxel then projected half a turn away from the raw
    evidence that should have filled it.

    Asserted on the objects themselves rather than on the source text: the two
    recorders wrote down what the shipped chooser and the shipped volume were
    really handed.
    """
    assert Rigid.identity() != UPRIGHT_FROM_OPTICAL
    assert any(pose != Rigid.identity() for pose in fused.measured_at)
    assert all(upright_pose(pose) != pose for pose in fused.measured_at)

    # ABSOLUTE, not merely self-consistent. Bounds and fusion moved into the
    # SAME wrong frame together would still agree with each other and would
    # still fuse, so agreement alone is not the property: both must be the
    # poses the registration actually fitted, with no rotation in between.
    assert len(fused.registered_at) == REPLAY_FRAMES
    assert fused.measured_at == fused.registered_at

    assert fused.fused_at
    for offered in fused.fused_at:
        assert offered == fused.measured_at
        assert offered == fused.registered_at
        assert all(
            offered[index] != upright_pose(fused.measured_at[index])
            for index in range(REPLAY_FRAMES)
        )


def test_fusing_that_capture_at_the_presentation_poses_instead_would_find_nothing(
    fused: FusedReplay,
) -> None:
    """The control that makes the claim above substantive rather than a tautology.

    The SAME frames, the SAME field, the same shipped volume and extractor - and
    the poses one presentation half turn away. Nothing is observed and no surface
    exists, so a replay that handed upright poses to a model-frame field would
    fail the assertions above on outcome as well as on identity.
    """
    volume = TsdfVolume(fused.configs[0], backend=replay.BACKEND)
    volume.integrate(
        list(fused.raws),
        [upright_pose(pose) for pose in fused.measured_at],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    evidence = volume.evidence()
    assert evidence.frames_offered == REPLAY_FRAMES
    assert evidence.frames_integrated == 0
    assert evidence.weights.observed_voxels == 0
    with pytest.raises(NoSurfaceFound):
        extract_surface(volume, min_weight=1.0, backend=replay.BACKEND)


def test_every_allocated_cell_observes_voxels_and_extracts_a_surface(
    fused: FusedReplay,
) -> None:
    """What the corrected frame buys: a field with evidence in it.

    A zero-voxel sweep is not a finding about a capture, it is a harness that
    measured nothing, and this is the assertion that will not let one be
    published as the other again.
    """
    cells = _cells(fused.report)
    assert cells
    for entry in cells:
        assert "refused" not in entry, entry.get("refused")
        field = entry["field"]
        assert isinstance(field, dict)
        assert field["frames_offered"] == REPLAY_FRAMES
        assert field["frames_integrated"] == REPLAY_FRAMES
        assert isinstance(field["observed_voxels"], int) and field["observed_voxels"] > 0
        surface = entry["surface"]
        assert isinstance(surface, dict)
        assert surface["extracted"] is True
        assert isinstance(surface["faces"], int) and surface["faces"] > 0


def test_a_replay_that_fused_still_wrote_nothing_into_the_capture(fused: FusedReplay) -> None:
    """Read-only is a property of the FUSING path too, not only of the refusals."""
    verdict = replay.verify_unchanged(fused.before, fused.after)
    assert verdict == {
        "files": REPLAY_FRAMES,
        "unchanged": True,
        "changed": [],
        "missing": [],
        "appeared": [],
    }
    assert sorted(path.name for path in fused.capture.iterdir()) == [
        f"frame_{index:04d}.raw" for index in range(REPLAY_FRAMES)
    ]


def test_the_report_names_the_one_frame_every_number_in_it_is_in(fused: FusedReplay) -> None:
    """A reader must not have to infer the frame from the code that produced it."""
    frame = str(fused.report["frame"])
    assert "MODEL frame" in frame
    assert "UPRIGHT_FROM_OPTICAL" in frame
    assert "applied to nothing here" in frame
    published = " ".join(str(item) for item in replay.LIMITATIONS)
    assert "not directly comparable with an exported PLY" in published


def test_the_fusion_window_is_the_appliances_own_band_rather_than_a_scenes(
    fused: FusedReplay,
) -> None:
    """The second defect the physical run exposed, and why it hid the first.

    The window was ``0.05``/``4.0`` - a synthetic scene's range, not a
    capture's - so every observation of a real capture was discarded during
    projection and every cell came back with zero observed voxels whatever the
    poses were. Bound to the appliance's own constants here so the two cannot
    drift, and it is the fixture above that proves the band actually admits a
    capture's returns.
    """
    assert replay.MIN_DEPTH == TSDF_MIN_DEPTH
    assert replay.MAX_DEPTH == TSDF_MAX_DEPTH
    for config in fused.configs:
        assert config.min_depth == TSDF_MIN_DEPTH
        assert config.max_depth == TSDF_MAX_DEPTH
