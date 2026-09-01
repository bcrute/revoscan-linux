"""The depth-completeness contract, proved load-bearing rather than merely runnable.

``tests/test_depth_completeness_harness.py`` owns the corpus and the command.
This file owns the rules underneath them: that a window's denominators cannot
be escaped, that every one of the fourteen refusals actually refuses, that a
measurement is computed from the population it says it is, and that the
candidates behave the way the corpus declares they do.

The refusal controls are the load-bearing part. A completion that could report
reconstructed samples as measurements, mutate a raw frame, carry a depth on a
sample it called missing, or name support that was never measured would make
every other number in this benchmark meaningless, so each of those is a test
rather than a comment.

Nothing here touches a scanner, a capture under ``/home/ben/scans``, a display
server, a GPU, the network or the clock.
"""

from __future__ import annotations

import math
import struct

import pytest

import scanner3d.depth_temporal as depth_temporal
import scanner3d.quality as quality
from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.geometry import IDENTITY_MATRIX, Rigid, rotation_about_axis
from scanner3d.simulate import Box, Plane, Sphere, intrinsics_of, render
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_WIDTH
from tests import depth_completeness as contract
from tests import depth_completeness_harness as harness

INTRINSICS = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)
DEPTH_SCALE = 0.0078125
WINDOW = contract.Window(300, 188, 20, 12)


def _scene(
    shapes,
    *,
    window=WINDOW,
    masks=((),),
    poses=(None,),
    regions=None,
    case_id="unit",
    seed="unit-test",
):
    """One scene, built the way a fixture builds one but without a fixture."""
    frames = []
    for index, pose in enumerate(poses):
        rendered = render(
            shapes,
            intrinsics=INTRINSICS,
            depth_scale=DEPTH_SCALE,
            pose=Rigid.identity() if pose is None else pose,
        )
        truth = contract.codes_of(rendered.raw, window)
        removed: set[int] = set()
        for position, mask in enumerate(masks[index]):
            removed |= harness.mask_indices(
                mask,
                window=window,
                truth=truth,
                seed=f"{seed}:{index}:{position}",
                what="a unit-test mask",
            )
        masked = [0 if sample in removed else code for sample, code in enumerate(truth)]
        frames.append(
            contract.FrameEvidence(
                index=index,
                truth=contract.frame_of_codes(truth, window),
                raw=contract.frame_of_codes(masked, window),
                pose=(Rigid.identity() if pose is None else pose).inverse(),
            )
        )
    return contract.Scene(
        case_id=case_id,
        window=window,
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
        frames=tuple(frames),
        regions=regions or contract.Regions(),
    )


PLANE = [Plane(normal=(0.0, 0.0, 1.0), offset=100.0)]
SPECKLE = ({"kind": "speckle", "fraction": 0.2},)


def _named(measurements):
    return {measurement.name: measurement.as_dict() for measurement in measurements}


# --- the window -----------------------------------------------------------------------


def test_a_window_that_leaves_the_frame_is_refused():
    with pytest.raises(contract.ScoringUnusable):
        contract.Window(630, 188, 20, 12)
    with pytest.raises(contract.ScoringUnusable):
        contract.Window(300, 188, 0, 12)


def test_window_indices_and_pixels_are_inverse():
    window = contract.Window(300, 188, 20, 12)
    for index in range(window.sample_count):
        u, v = window.pixel_of(index)
        assert window.index_of(u, v) == index
    assert window.index_of(299, 188) is None
    assert window.index_of(300, 187) is None


def test_a_frame_of_codes_leaves_everything_outside_the_window_unmeasured():
    window = contract.Window(300, 188, 4, 2)
    frame = contract.frame_of_codes([1, 2, 3, 4, 5, 6, 7, 8], window)
    assert len(frame) == EXPECTED_FRAME_BYTES
    assert contract.codes_of(frame, window) == (1, 2, 3, 4, 5, 6, 7, 8)
    (outside,) = struct.unpack_from("<H", frame, (188 * FRAME_WIDTH + 299) * 2)
    assert outside == NO_RETURN_SAMPLE


def test_a_zero_code_is_no_measurement_and_never_a_point():
    window = contract.Window(300, 188, 2, 1)
    cloud = contract.cloud_of([0, 12800], window, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert len(cloud) == 1
    assert cloud[0][2] == pytest.approx(100.0)


# --- the refusals ---------------------------------------------------------------------


def _refusal(candidate, scene) -> str:
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.measure(scene, candidate(scene))
    return raised.value.refusal


@pytest.mark.parametrize(
    ("candidate", "refusal"),
    [
        ("forged_measured_states", "measured_value_changed"),
        ("mutated_raw_frame", "raw_frame_mutated"),
        ("unsupported_fill", "unsupported_reconstruction"),
        ("forged_confidence", "confidence_out_of_range"),
        ("missing_carrying_depth", "missing_carries_depth"),
        ("unknown_provenance_state", "unknown_state"),
        ("truncated_window", "sample_count_mismatch"),
        ("phantom_support", "support_not_measured"),
        ("temporal_support_from_own_frame", "temporal_support_not_another_frame"),
    ],
)
def test_every_corruption_control_earns_its_exact_refusal(candidate, refusal):
    scene = _scene(PLANE, masks=(SPECKLE,))
    assert _refusal(contract.CANDIDATES[candidate], scene) == refusal


def test_a_refusal_names_a_declared_reason():
    with pytest.raises(ValueError, match="declared refusals"):
        contract.CandidateRefused("something_went_wrong", "detail")


def _edit(scene, *, codes=None, states=None, confidence=None, support=None, raw=None):
    """One frame of the baseline, with named fields replaced."""
    frame = contract.baseline(scene).frames[0]
    edited = contract.CompletedFrame(
        index=frame.index,
        raw=frame.raw if raw is None else raw,
        codes=frame.codes if codes is None else codes,
        states=frame.states if states is None else states,
        confidence=frame.confidence if confidence is None else confidence,
        support=frame.support if support is None else support,
    )
    return contract.CandidateResult(name="edited", frames=(edited,))


def test_a_reconstruction_without_depth_is_refused():
    scene = _scene(PLANE, masks=(SPECKLE,))
    states = list(contract.baseline(scene).frames[0].states)
    hole = states.index(contract.MISSING)
    states[hole] = contract.SPATIALLY_RECONSTRUCTED
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, _edit(scene, states=tuple(states)))
    assert raised.value.refusal == "reconstructed_without_depth"


def test_support_on_a_measured_sample_is_refused():
    scene = _scene(PLANE, masks=(SPECKLE,))
    result = _edit(scene, support={0: (contract.support_name(0, 300, 188),)})
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, result)
    assert raised.value.refusal == "support_without_reconstruction"


def test_a_confidence_outside_the_fraction_is_refused_in_both_directions():
    scene = _scene(PLANE)
    for value in (-0.1, 1.5, math.nan, math.inf):
        confidence = list(contract.baseline(scene).frames[0].confidence)
        confidence[0] = value
        with pytest.raises(contract.CandidateRefused) as raised:
            contract.validate(scene, _edit(scene, confidence=tuple(confidence)))
        assert raised.value.refusal == "confidence_out_of_range"


def test_a_result_with_the_wrong_number_of_frames_is_refused():
    scene = _scene(PLANE, masks=((), ()), poses=(None, None))
    single = contract.CandidateResult(name="short", frames=contract.baseline(scene).frames[:1])
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, single)
    assert raised.value.refusal == "frame_set_mismatch"


def test_a_spatial_reconstruction_may_not_name_another_frame():
    scene = _scene(PLANE, masks=(SPECKLE, ()), poses=(None, None))
    result = contract.dense_nearest(scene)
    first = result.frames[0]
    hole = next(
        index for index, state in enumerate(first.states) if state in contract.RECONSTRUCTED_STATES
    )
    support = dict(first.support)
    support[hole] = (contract.support_name(1, 300, 188),)
    edited = contract.CandidateResult(
        name="crossed",
        frames=(
            contract.CompletedFrame(
                index=0,
                raw=first.raw,
                codes=first.codes,
                states=first.states,
                confidence=first.confidence,
                support=support,
            ),
            result.frames[1],
        ),
    )
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, edited)
    assert raised.value.refusal == "spatial_support_not_this_frame"


def test_support_that_is_not_a_sample_reference_is_refused():
    scene = _scene(PLANE, masks=(SPECKLE,))
    result = contract.dense_nearest(scene)
    frame = result.frames[0]
    hole = next(
        index for index, state in enumerate(frame.states) if state in contract.RECONSTRUCTED_STATES
    )
    support = dict(frame.support)
    support[hole] = ("a neighbour, roughly",)
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(
            scene,
            contract.CandidateResult(
                name="vague",
                frames=(
                    contract.CompletedFrame(
                        index=0,
                        raw=frame.raw,
                        codes=frame.codes,
                        states=frame.states,
                        confidence=frame.confidence,
                        support=support,
                    ),
                ),
            ),
        )
    assert raised.value.refusal == "unsupported_reconstruction"


# --- the candidates -------------------------------------------------------------------


def test_the_baseline_invents_nothing_and_keeps_its_holes_visible():
    scene = _scene(PLANE, masks=(SPECKLE,))
    measurements = _named(contract.measure(scene, contract.baseline(scene)))
    coverage = measurements["coverage"]
    assert coverage["reconstructed"] == 0
    assert coverage["false_fill"] == 0
    assert coverage["missing"] > 0
    assert coverage["supported_fraction"] == pytest.approx(coverage["measured_fraction"])
    assert "reconstructed_depth" not in measurements


def test_an_identity_over_complete_input_reproduces_every_measured_value():
    scene = _scene(PLANE, regions=contract.Regions(plane=True, dimensions=True, landmarks=True))
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    identity = _named(contract.measure(scene, contract.identity(scene)))
    truth = contract.codes_of(scene.frames[0].truth, scene.window)
    produced = contract.identity(scene).frames[0]
    assert produced.codes == truth
    assert set(produced.states) == {contract.MEASURED}
    for name, measurement in baseline.items():
        assert identity[name] == measurement
    assert identity["coverage"]["measured_fraction"] == 1.0
    assert identity["plane"]["residuals"]["worst"] == 0.0
    assert identity["dimensions"]["absolute"]["worst"] == 0.0
    assert identity["landmarks"]["offsets"]["worst"] == 0.0


def test_a_dense_fill_gains_coverage_and_invents_geometry_in_empty_space():
    scene = _scene(
        [Sphere(centre=(0.0, 0.0, 150.0), radius=6.0)],
        window=contract.Window(300, 182, 40, 36),
        masks=(SPECKLE,),
        regions=contract.Regions(surface_tolerance=0.1),
    )
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    dense = _named(contract.measure(scene, contract.dense_nearest(scene)))
    assert dense["coverage"]["missing"] == 0
    assert dense["coverage"]["reconstructed"] > 0
    assert dense["coverage"]["false_fill"] > 0
    assert baseline["coverage"]["false_fill"] == 0
    # The direction that notices geometry that is there and should not be.
    assert (
        dense["surface"]["observed_to_reference"]["distances"]["p95"]
        > baseline["surface"]["observed_to_reference"]["distances"]["p95"]
    )


def test_filling_only_the_easy_holes_scores_well_and_covers_little():
    scene = _scene(
        [Plane(normal=(0.3, 0.15, 1.0), offset=105.0)],
        masks=(SPECKLE,),
        regions=contract.Regions(plane=True),
    )
    easy = _named(contract.measure(scene, contract.easy_samples_only(scene)))
    dense = _named(contract.measure(scene, contract.dense_nearest(scene)))
    assert (
        easy["reconstructed_depth"]["absolute"]["p95"]
        < (dense["reconstructed_depth"]["absolute"]["p95"])
    )
    # And cannot hide behind that: the denominator is the whole window.
    assert easy["coverage"]["supported_fraction"] < dense["coverage"]["supported_fraction"]
    assert easy["coverage"]["missing"] > 0


def test_error_is_scored_only_on_reconstructed_samples_with_truth_behind_them():
    scene = _scene(
        [Sphere(centre=(0.0, 0.0, 150.0), radius=6.0)],
        window=contract.Window(300, 182, 40, 36),
        masks=(SPECKLE,),
    )
    dense = _named(contract.measure(scene, contract.dense_nearest(scene)))
    coverage, error = dense["coverage"], dense["reconstructed_depth"]
    assert error["entry_count"] == coverage["reconstructed_on_surface"]
    assert error["entry_count"] < coverage["reconstructed"]
    assert error["dropout_samples"] == coverage["input_dropout"]


# --- the geometry-specific measurements -------------------------------------------------

STEP = [
    Box(half_extents=(4.0, 6.0, 4.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 100.0))),
    Plane(normal=(0.0, 0.0, 1.0), offset=140.0),
]
EDGE_MASK = ({"kind": "edge_band", "foreground_below": 15000, "width": 3},)


def test_averaging_across_a_depth_step_bridges_it():
    scene = _scene(
        STEP,
        window=contract.Window(310, 188, 40, 24),
        masks=(EDGE_MASK,),
        regions=contract.Regions(foreground_below=15000),
    )
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    broad = _named(contract.measure(scene, contract.broad_interpolation(scene)))
    assert baseline["edge"]["bridging"] == 0
    assert baseline["edge"]["misplaced"] == 0
    assert broad["edge"]["bridging"] > 0
    assert broad["edge"]["misplaced"] > 0
    assert broad["edge"]["band_high"] > broad["edge"]["band_low"]


def test_a_slot_stays_open_only_if_nothing_puts_foreground_in_it():
    slot = [
        Box(half_extents=(4.0, 6.0, 4.0), pose=Rigid.of(IDENTITY_MATRIX, (-4.36, 0.0, 100.0))),
        Box(half_extents=(4.0, 6.0, 4.0), pose=Rigid.of(IDENTITY_MATRIX, (4.36, 0.0, 100.0))),
        Plane(normal=(0.0, 0.0, 1.0), offset=140.0),
    ]
    scene = _scene(
        slot,
        window=contract.Window(300, 188, 40, 24),
        masks=(({"kind": "strip", "first_column": 19, "last_column": 21},),),
        regions=contract.Regions(foreground_below=15000, slot_columns=(19, 20, 21)),
    )
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    closed = _named(contract.measure(scene, contract.foreground_bias(scene)))
    assert baseline["slot"]["open"] is True
    assert baseline["slot"]["slot_reconstructed"] == 0
    assert closed["slot"]["open"] is False
    assert closed["slot"]["closed_samples"] == closed["slot"]["slot_samples"]


MULTIVIEW = [
    Box(half_extents=(2.0, 3.0, 2.0), pose=Rigid.of(IDENTITY_MATRIX, (0.0, 0.0, 100.0))),
    Sphere(centre=(1.0, -2.0, 97.0), radius=1.2),
]
TURNS = tuple(
    rotation_about_axis((0.0, 1.0, 0.0), math.radians(angle), through=(0.0, 0.0, 100.0))
    for angle in (-4.0, 0.0, 4.0)
)
#: The corpus case's own masks: a different third of the columns in each frame,
#: and one block that returns in none of them.
COMPLEMENTARY = tuple(
    (
        {"kind": "complementary", "period": 3, "phase": index},
        {"kind": "never_return", "first_column": 2, "last_column": 3},
    )
    for index in range(3)
)


def _multiview_scene(**regions):
    return _scene(
        MULTIVIEW,
        window=contract.Window(310, 190, 20, 16),
        masks=COMPLEMENTARY,
        poses=TURNS,
        regions=contract.Regions(**regions),
    )


def test_carrying_the_same_pixel_across_motion_is_worse_than_carrying_it_through_the_poses():
    scene = _multiview_scene()
    naive = _named(contract.measure(scene, contract.naive_temporal_carry(scene)))
    posed = _named(contract.measure(scene, contract.posed_temporal_carry(scene)))
    # Four raw codes: the tolerance the corpus gates reconstructed error at.
    tolerance = 4 * DEPTH_SCALE
    assert posed["reconstructed_depth"]["absolute"]["p95"] <= tolerance
    assert naive["reconstructed_depth"]["absolute"]["p95"] > tolerance
    assert naive["coverage"]["temporally_reconstructed"] > 0
    assert posed["coverage"]["temporally_reconstructed"] > 0


def test_fusion_runs_through_the_unchanged_tsdf_and_leaves_unseen_space_unobserved():
    spec = contract.FusionSpec(
        minimum=(-3.5, -4.5, 95.0),
        maximum=(3.5, 4.5, 103.0),
        voxel_size=0.25,
        truncation=0.75,
        min_depth=50.0,
        max_depth=200.0,
        min_weight=1.0,
        coverage_tolerance=0.5,
    )
    scene = _multiview_scene(fusion=spec)
    fused = _named(contract.measure(scene, contract.baseline(scene)))
    fusion = fused["fusion"]
    assert fusion["frames_integrated"] == 3
    assert fusion["surface_extracted"] is True
    assert fusion["unobserved_voxels"] > fusion["observed_voxels"]
    assert (
        fused["fusion_surface"]["observed_to_reference"]["query_count"]
        == (fusion["surface_vertices"])
    )


def test_the_tracking_replay_is_the_real_tracker_over_these_exact_clouds():
    scene = _multiview_scene(tracking=True)
    measured = _named(contract.measure(scene, contract.baseline(scene)))
    tracking = measured["tracking"]
    assert tracking["frames_offered"] == 3
    assert tracking["frames_accepted"] == 3
    assert tracking["accepted_all"] is True
    assert tracking["origins"][0] == "anchor"
    assert tracking["failure_types"] == []
    # Where the frames were placed, not only whether they were.
    poses = measured["tracking_pose"]
    assert poses["pose_count"] == 3
    assert len(poses["frames"]) == 2
    assert poses["gauge"] == quality.GAUGE_STATEMENT
    # On the baseline the two replays are the same clouds, because the baseline
    # reconstructs nothing - so they are the same record but for their names.
    counterfactual = measured["tracking_completed_cloud"]
    assert counterfactual["frames_accepted"] == 3
    assert (
        measured["tracking_completed_cloud_pose"]["rotation"]["worst"] == poses["rotation"]["worst"]
    )


def test_the_gated_tracking_route_sees_measured_samples_only_whatever_a_candidate_filled():
    """The acceptance route is the intended dataflow, and it is candidate-independent.

    This project's tracker consumes measured geometry only; neither completion
    stage feeds it. So the gated record is a property of the scene's own
    frames, and a dense in-frame fill, a naive same-pixel carry and the real
    pose-compensated stage all reproduce the baseline's exactly.
    """
    scene = _multiview_scene(tracking=True)
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    for candidate in (
        contract.dense_nearest,
        contract.naive_temporal_carry,
        contract.temporal_completion,
    ):
        measured = _named(contract.measure(scene, candidate(scene)))
        assert measured["tracking"] == baseline["tracking"]
        assert measured["tracking_pose"] == baseline["tracking_pose"]


def test_the_counterfactual_replay_shows_what_feeding_completions_to_the_tracker_costs():
    """The measured reason completion output enters no tracker in this task.

    The counterfactual is not an acceptance route and carries no gate. It is
    published because the answer is unfavourable: a dense in-frame fill moves
    the worst placement error by more than a factor of four, and the real
    pose-compensated stage moves it too - even though every one of its
    reconstructions is within a single raw code of the truth.
    """
    scene = _multiview_scene(tracking=True)
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    dense = _named(contract.measure(scene, contract.dense_nearest(scene)))
    posed = _named(contract.measure(scene, contract.posed_temporal_carry(scene)))
    temporal = _named(contract.measure(scene, contract.temporal_completion(scene)))
    bar = baseline["tracking_completed_cloud_pose"]["rotation"]["worst"]

    assert dense["tracking_completed_cloud"]["accepted_all"] is True
    # Every frame still placed, and placed somewhere else: a count-only record
    # would call this neutral.
    assert dense["tracking_completed_cloud_pose"]["rotation"]["worst"] > bar * 4.0
    assert posed["tracking_completed_cloud_pose"]["rotation"]["worst"] < bar + 0.25
    assert temporal["tracking_completed_cloud_pose"]["rotation"]["worst"] > bar
    # And none of it reaches the gated route.
    assert temporal["tracking_pose"] == baseline["tracking_pose"]


# --- the measurements are the quality authority's ----------------------------------------


def test_every_measurement_carries_a_kind_this_benchmark_has_a_schema_for():
    scene = _multiview_scene(dimensions=True, tracking=True)
    for measurement in contract.measure(scene, contract.dense_nearest(scene)):
        # quality.QualityMeasurement is a plain Protocol, so this is the shape
        # check rather than an isinstance one.
        assert isinstance(measurement.name, str)
        assert measurement.kind in harness.MEASUREMENT_KINDS
        assert measurement.as_dict()["name"] == measurement.name


def test_coverage_accounts_for_every_sample_of_the_window():
    scene = _scene(PLANE, masks=(SPECKLE,))
    for name in ("baseline", "dense_nearest", "easy_samples_only", "broad_interpolation"):
        coverage = _named(contract.measure(scene, contract.CANDIDATES[name](scene)))["coverage"]
        assert (
            coverage["measured"]
            + coverage["spatially_reconstructed"]
            + coverage["temporally_reconstructed"]
            + coverage["missing"]
            == coverage["domain_samples"]
            == scene.window.sample_count
        )
        assert coverage["truth_surface"] + coverage["truth_empty"] == coverage["domain_samples"]


# --- the product primitive, through its adapter -------------------------------------------


@pytest.fixture(scope="module")
def cases():
    """Every shipped case, read once, keyed by case id.

    The corpus rather than a unit scene, because the claims below are about
    geometry the corpus was built to contain - a silhouette, a depth step, a
    real gap, complementary loss across posed frames - and re-rendering
    approximations of them here would be testing a second set of scenes.
    """
    corpus = harness.load_corpus()
    return {
        entry.case_id: harness.load_case(harness.CORPUS / entry.name, entry=entry).scene
        for entry in corpus.files
    }


def _completion(scene):
    result = contract.spatial_completion(scene)
    contract.validate(scene, result)
    return result, _named(contract.measure(scene, result))


def test_the_completion_never_edits_a_raw_frame_or_a_measured_sample(cases):
    for case_id, scene in cases.items():
        result, _ = _completion(scene)
        for index, frame in enumerate(result.frames):
            assert frame.raw == scene.frames[index].raw, case_id
            given = scene.input_codes(index)
            for position, code in enumerate(given):
                if code == NO_RETURN_SAMPLE:
                    continue
                assert frame.codes[position] == code, case_id
                assert frame.states[position] == contract.MEASURED, case_id


def test_complete_input_is_an_exact_identity(cases):
    """The control that says a gain is a gain and not this benchmark's noise.

    With nothing missing there is nothing to reconstruct, and every downstream
    number must be the unchanged baseline's - not close to it.
    """
    scene = cases["plane-fronto-complete"]
    _, completed = _completion(scene)
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    assert completed["coverage"]["spatially_reconstructed"] == 0
    assert "confidence" not in completed
    for name, measurement in baseline.items():
        assert completed[name] == measurement, name
    assert completed["completion_work"]["holes"] == 0
    assert completed["completion_work"]["accepted"] == 0


@pytest.mark.parametrize("case_id", ["plane-fronto", "plane-oblique"])
def test_a_plane_gains_strictly_over_the_baseline_where_its_evidence_is_local(cases, case_id):
    scene = cases[case_id]
    _, completed = _completion(scene)
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    assert completed["coverage"]["measured"] == baseline["coverage"]["measured"]
    assert completed["coverage"]["supported"] > baseline["coverage"]["supported"]
    assert completed["coverage"]["reconstructed_on_surface"] > 0
    assert completed["coverage"]["false_fill"] == 0


def test_the_oblique_plane_is_interpolated_rather_than_copied(cases):
    """The difference between a local fit and a nearest-code copy, in one number.

    On a tilted surface the nearest returned sample is the wrong answer at
    every hole, and the two candidates are handed identical input. The fit is
    exact to within the truth's own quantisation; the copy is a staircase.
    """
    scene = cases["plane-oblique"]
    _, completed = _completion(scene)
    copied = _named(contract.measure(scene, contract.dense_nearest(scene)))
    assert (
        completed["reconstructed_depth"]["absolute"]["worst"]
        < (copied["reconstructed_depth"]["absolute"]["worst"])
    )
    assert completed["reconstructed_depth"]["absolute"]["worst"] <= DEPTH_SCALE


def test_the_silhouette_is_refused_whole_and_nothing_is_painted_in_empty_space(cases):
    scene = cases["curved-surface"]
    _, completed = _completion(scene)
    assert completed["coverage"]["false_fill"] == 0
    assert completed["completion_work"]["rejections"]["component_too_large"] > 0


def test_a_depth_step_is_neither_bridged_nor_crossed(cases):
    scene = cases["depth-step"]
    _, completed = _completion(scene)
    assert completed["edge"]["bridging"] == 0
    assert completed["edge"]["misplaced"] == 0
    assert completed["coverage"]["reconstructed_on_surface"] > 0


def test_a_real_gap_between_two_objects_stays_open(cases):
    scene = cases["narrow-slot"]
    _, completed = _completion(scene)
    assert completed["slot"]["open"] is True
    assert completed["slot"]["closed_samples"] == 0
    assert completed["slot"]["slot_reconstructed"] == 0


def test_running_it_per_frame_costs_the_multi_view_pipeline_nothing(cases):
    """Item by item against the baseline, on the case that fuses and tracks.

    Every hole in this scene is a whole column of complementary loss, which no
    support inside one frame spans, so the completion returns each frame
    unchanged - and the fused surface, the tracking record, the poses and the
    dimensions are the baseline's exactly rather than merely near it. What
    another view saw belongs to the temporal task and is not claimed here.
    """
    scene = cases["multiview-object"]
    _, completed = _completion(scene)
    baseline = _named(contract.measure(scene, contract.baseline(scene)))
    for name in ("coverage", "fusion", "fusion_surface", "tracking", "tracking_pose", "dimensions"):
        assert completed[name] == baseline[name], name


def test_every_accepted_sample_names_support_the_input_actually_returned(cases):
    for case_id, scene in cases.items():
        result, _ = _completion(scene)
        for index, frame in enumerate(result.frames):
            given = scene.input_codes(index)
            for position, state in enumerate(frame.states):
                if state != contract.SPATIALLY_RECONSTRUCTED:
                    continue
                names = frame.support[position]
                assert names, case_id
                for name in names:
                    match = contract.SUPPORT_NAME.match(name)
                    assert match is not None, name
                    source_frame, u, v = (int(part) for part in match.groups())
                    assert source_frame == index, case_id
                    source = scene.window.index_of(u, v)
                    assert source is not None
                    assert given[source] != NO_RETURN_SAMPLE, case_id


def test_the_work_record_is_bound_to_what_the_result_actually_shows(cases):
    for case_id, scene in cases.items():
        result, completed = _completion(scene)
        work = completed["completion_work"]
        holes = sum(
            1
            for index in range(len(scene.frames))
            for code in scene.input_codes(index)
            if code == NO_RETURN_SAMPLE
        )
        assert work["holes"] == work["attempted"] == holes, case_id
        assert work["accepted"] + work["rejected"] == work["attempted"], case_id
        assert sum(work["rejections"].values()) == work["rejected"], case_id
        assert work["accepted"] == completed["coverage"]["reconstructed"], case_id
        assert work["within_budget"] is True, case_id
        assert work["work_units"] <= work["work_budget"], case_id
        assert work["support_samples"] == sum(
            len(sources) for frame in result.frames for sources in frame.support.values()
        ), case_id


def test_the_confidence_block_covers_every_reconstruction_and_is_never_a_measurement(cases):
    for case_id, scene in cases.items():
        _, completed = _completion(scene)
        coverage = completed["coverage"]
        if coverage["reconstructed"] == 0:
            assert "confidence" not in completed, case_id
            continue
        confidence = completed["confidence"]
        assert confidence["reconstructed"] == coverage["reconstructed"], case_id
        assert confidence["on_surface"] == coverage["reconstructed_on_surface"], case_id
        assert confidence["false_fill"] == coverage["false_fill"], case_id
        assert sum(entry["count"] for entry in confidence["bins"]) == coverage["reconstructed"]
        assert confidence["maximum"] < 1.0, case_id


#: The one case where the single-frame primitive is confidently wrong about a
#: sample that is on real surface, and is meant to be. Its pillar is one sample
#: wide, so the held-out sample has a full ring of plane samples around it and
#: nothing local says it is not plane: the estimate lands on the plane, 512
#: codes behind the pillar's own front face. That is the whole reason the case
#: exists, and the corpus declares the failing gate rather than hiding it - so
#: this test asserts the exact wrong answer instead of skipping the case.
CONFIDENTLY_WRONG_CASE = "temporal-confirmation"
CONFIDENTLY_WRONG_CODES = 512


def test_a_confident_reconstruction_is_one_that_was_actually_right(cases):
    """Calibration, against held-out truth rather than against itself.

    One raw code, which is the truth's own quantisation: the rendered scenes
    carry whole codes, so a fit that lands on the right surface still rounds.
    Not a tolerance chosen for this candidate - it is the smallest difference
    these frames can express.

    One case is exempt and is asserted the other way round. See
    :data:`CONFIDENTLY_WRONG_CASE`: single-frame completion cannot know the
    pillar there is not the plane, and a corpus in which one frame is never
    confidently wrong would be a corpus with no reason to have a second stage.
    """
    for case_id, scene in cases.items():
        _, completed = _completion(scene)
        if "confidence" not in completed:
            continue
        worst = completed["confidence"]["high_confidence_worst"]
        if case_id == CONFIDENTLY_WRONG_CASE:
            assert worst == CONFIDENTLY_WRONG_CODES * DEPTH_SCALE, case_id
            continue
        assert worst <= DEPTH_SCALE, case_id


def test_a_completion_that_restates_what_it_refused_is_refused(cases):
    scene = cases["plane-fronto"]
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, contract.spatial_completion_forged_work(scene))
    assert raised.value.refusal == "work_counts_inconsistent"


def test_a_completion_that_cites_its_own_output_as_evidence_is_refused(cases):
    scene = cases["plane-fronto"]
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(scene, contract.spatial_completion_recursive_support(scene))
    assert raised.value.refusal == "support_not_measured"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("holes", 0, "its input returned no measurement at"),
        ("attempted", 0, "hole(s) and accounts for"),
        ("support_samples", 0, "and the result names"),
        ("work_units", 10**9, "not advisory"),
        ("work_budget", 0, "not advisory"),
        ("rejected", 0, "hole(s) and accounts for"),
    ],
)
def test_every_work_fact_fails_closed_when_it_is_changed(cases, field, value, reason):
    """One field at a time, on an otherwise real result. Fail closed, not fail loud."""
    scene = cases["plane-fronto"]
    result = contract.spatial_completion(scene)
    assert result.work is not None
    forged = contract.WorkRecord(
        **{
            **{
                key: getattr(result.work, key)
                for key in (
                    "backend",
                    "policy",
                    "holes",
                    "attempted",
                    "accepted",
                    "rejected",
                    "rejections",
                    "support_samples",
                    "work_units",
                    "work_budget",
                )
            },
            field: value,
        }
    )
    with pytest.raises(contract.CandidateRefused) as raised:
        contract.validate(
            scene, contract.CandidateResult(name="forged", frames=result.frames, work=forged)
        )
    assert raised.value.refusal == "work_counts_inconsistent"
    assert reason in str(raised.value)


# --- the disclosed small-gap ambiguity ------------------------------------------------


def _neighbourhood(codes, window, position, *, radius=4):
    """The scored 9x9 box around one sample, as a plain list of codes.

    Read through :meth:`Window.index_of` rather than raw row/column
    arithmetic, so a mistake in this helper would show up as an index error
    rather than as a silently different neighbourhood.
    """
    column, row = window.pixel_of(position)
    box = []
    for offset_row in range(-radius, radius + 1):
        for offset_column in range(-radius, radius + 1):
            index = window.index_of(column + offset_column, row + offset_row)
            assert index is not None, "the case places both samples away from any edge"
            box.append(codes[index])
    return box


def test_the_aperture_and_the_held_out_dropout_score_byte_identical_input(cases):
    """The ambiguity this case exists to expose, proved rather than described.

    A small true gap in a surface (the aperture: truth code 0, nothing behind
    it) and a small sensor dropout on the same surface (the held-out
    dropout: truth code nonzero, input code 0) must render the same input
    bytes in the neighbourhood a spatial completion scores, or this case is
    not testing the ambiguity it claims to.
    """
    scene = cases["aperture-ambiguity"]
    window = scene.window
    truth = scene.truth_codes(0)
    given = scene.input_codes(0)

    truth_empty = [position for position, code in enumerate(truth) if code == NO_RETURN_SAMPLE]
    dropped_on_surface = [
        position
        for position, code in enumerate(given)
        if code == NO_RETURN_SAMPLE and truth[position] != NO_RETURN_SAMPLE
    ]
    assert len(truth_empty) == 1, "exactly one sample this scene genuinely leaves empty"
    assert len(dropped_on_surface) == 1, "exactly one dropped sample on real surface"

    (aperture,) = truth_empty
    (dropout,) = dropped_on_surface
    assert truth[aperture] == NO_RETURN_SAMPLE
    assert given[aperture] == NO_RETURN_SAMPLE
    assert truth[dropout] != NO_RETURN_SAMPLE
    assert given[dropout] == NO_RETURN_SAMPLE

    aperture_box = _neighbourhood(given, window, aperture)
    dropout_box = _neighbourhood(given, window, dropout)
    assert len(aperture_box) == len(dropout_box) == 81
    assert aperture_box == dropout_box


def test_the_pair_is_scored_as_the_fixture_declares(cases):
    """The paired outcome the corpus commits to, so the pair cannot quietly stop being one.

    A deterministic single-frame method cannot resolve which side of the
    aperture/dropout pair is real. This corpus does not let it pretend to:
    the dropout, which is real surface, must come back reconstructed onto
    surface; the aperture, which is really empty, must come back as false
    fill; and neither ambiguous reconstruction may be declared with the
    confidence that would make it look like settled geometry rather than a
    spatial guess. Until independent temporal or physical evidence
    disambiguates it, that guess must not be treated as tracking or fused
    geometry.
    """
    scene = cases["aperture-ambiguity"]
    _, completed = _completion(scene)
    coverage = completed["coverage"]
    confidence = completed["confidence"]
    assert coverage["reconstructed_on_surface"] == 1, "the dropout is recovered onto surface"
    assert coverage["false_fill"] == 1, "the aperture is filled with invented surface"
    assert confidence["high_confidence_false_fill"] >= 1, (
        "the invented fill is reported with the confidence that this case proves cannot "
        "distinguish it from the real recovery beside it"
    )


# --- the motion control, asserted on exact pixels rather than on a score ------------------
#
# The claim the temporal stage exists to make is that the pose between two
# frames matters. An aggregate score cannot say that: a denser answer and a
# correct answer both move a coverage number, and on this scene the WRONG
# answer is the denser one. So the pair below is asserted at named pixels, at
# named raw codes, on a scene whose camera measurably moved.

#: The scene's silhouette, read off its rendered truth: the last window column
#: the box occupies in each frame. Five apart, because the case shifts the
#: scene by exactly five pixels at the box's own depth per frame.
MOTION_SILHOUETTE_COLUMNS = (26, 31, 36)

#: One sample of frame 2 that frame 2 did not return, whose truth is the box's
#: front face, and where the two carries answer differently. Frames 0 and 1
#: measured the same point five and ten columns to the left, which is exactly
#: the motion.
MOTION_TARGET_PIXEL = (342, 188)
MOTION_SUPPORT_PIXELS = ((0, 332, 188), (1, 337, 188))
BOX_FRONT_CODE = 12288
BACKING_PLANE_CODE = 17920


def _motion_row(scene, index):
    window = scene.window
    truth = scene.truth_codes(index)
    row = window.v0 + window.height // 2
    return [truth[window.index_of(window.u0 + column, row)] for column in range(window.width)]


def test_the_motion_case_really_moves_its_silhouette_five_columns_a_frame(cases):
    """The premise of the pair, checked against the rendered truth.

    If the silhouette did not move, carrying a sample at the same pixel would
    be right and this case would prove nothing. So the truth is read directly:
    each frame's box occupies one column further right than the last, by five.
    """
    scene = cases["temporal-motion"]
    for index, expected in enumerate(MOTION_SILHOUETTE_COLUMNS):
        row = _motion_row(scene, index)
        last_box = max(column for column, code in enumerate(row) if code == BOX_FRONT_CODE)
        assert last_box == expected, index
        assert row[last_box + 1] == BACKING_PLANE_CODE, index
    poses = scene.poses()
    assert poses[0].translation != poses[2].translation, "the camera has to have actually moved"


def test_the_pose_composition_direction_is_the_one_that_lands_on_the_sample(cases):
    """Target-inverse composed with source, asserted rather than inferred.

    The two orders differ by twice the motion, so the wrong one is not a small
    error - it lands twenty columns away, on the other surface. This is checked
    through :func:`scanner3d.depth_temporal.temporal_transform` and the scene's
    own intrinsics, with no candidate and no score involved.
    """
    scene = cases["temporal-motion"]
    values = scene.intrinsics.values
    fx, fy, cx, cy = values["fx"], values["fy"], values["cx"], values["cy"]
    source_index, u, v = MOTION_SUPPORT_PIXELS[0]
    poses = scene.poses()

    def record(pose):
        return depth_temporal.PoseRecord(
            transform=pose, accepted=True, status="depth-completeness scene pose", quality=1.0
        )

    depth = BOX_FRONT_CODE * scene.depth_scale
    point = ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)
    correct = depth_temporal.temporal_transform(record(poses[2]), record(poses[source_index]))
    reversed_order = poses[source_index].inverse().compose(poses[2])
    landed = correct.apply(point)
    elsewhere = reversed_order.apply(point)
    assert (cx + landed[0] * fx / landed[2], cy + landed[1] * fy / landed[2]) == MOTION_TARGET_PIXEL
    assert cx + elsewhere[0] * fx / elsewhere[2] == 322.0
    assert elsewhere[1] == landed[1]


def test_same_pixel_carry_lands_on_the_wrong_surface_where_the_product_does_not(cases):
    """The pair, at one named sample, in raw codes.

    Frame 2 lost this sample; its truth is the box's front face. Frames 0 and 1
    measured that same point, five and ten columns to the left. Carrying their
    samples at the SAME pixel brings back the backing plane instead - 5,632 raw
    codes away, on the wrong surface - while carrying them through the poses
    brings back the box, exactly, and names the two pixels it came from.
    """
    scene = cases["temporal-motion"]
    window = scene.window
    position = window.index_of(*MOTION_TARGET_PIXEL)
    assert position is not None
    assert scene.input_codes(2)[position] == NO_RETURN_SAMPLE
    assert scene.truth_codes(2)[position] == BOX_FRONT_CODE
    for index, u, v in MOTION_SUPPORT_PIXELS:
        assert scene.input_codes(index)[window.index_of(u, v)] == BOX_FRONT_CODE

    naive = contract.naive_temporal_carry(scene).frames[2]
    product = contract.temporal_completion(scene).frames[2]
    assert naive.states[position] == contract.TEMPORALLY_RECONSTRUCTED
    assert naive.codes[position] == BACKING_PLANE_CODE
    assert product.states[position] == contract.TEMPORALLY_RECONSTRUCTED
    assert product.codes[position] == BOX_FRONT_CODE
    assert naive.codes[position] - product.codes[position] == 5632
    assert product.support[position] == tuple(
        contract.support_name(index, u, v) for index, u, v in MOTION_SUPPORT_PIXELS
    )


def test_the_same_pixel_carry_is_wrong_wherever_the_two_disagree(cases):
    """Not one lucky sample: every sample the two answer differently about.

    Seventy-seven of them, on this scene. At every one, the pose-compensated
    answer is the truth's own code and the same-pixel answer is not - so the
    difference between the two carries is which surface a sample is on, and
    never a rounding argument.
    """
    scene = cases["temporal-motion"]
    naive = contract.naive_temporal_carry(scene)
    product = contract.temporal_completion(scene)
    disagreements = 0
    for index in range(len(scene.frames)):
        truth = scene.truth_codes(index)
        given = scene.input_codes(index)
        for position, code in enumerate(given):
            if code != NO_RETURN_SAMPLE or truth[position] == NO_RETURN_SAMPLE:
                continue
            if naive.frames[index].states[position] != contract.TEMPORALLY_RECONSTRUCTED:
                continue
            if product.frames[index].states[position] != contract.TEMPORALLY_RECONSTRUCTED:
                continue
            if naive.frames[index].codes[position] == product.frames[index].codes[position]:
                continue
            disagreements += 1
            assert product.frames[index].codes[position] == truth[position]
            assert naive.frames[index].codes[position] != truth[position]
    assert disagreements == 77


def test_a_contradicted_estimate_publishes_the_independent_value_and_never_a_blend(cases):
    """The one sample where one frame is confidently wrong and two others are not.

    The pillar is one sample wide, so the single-frame estimate lands on the
    plane behind it - the best answer one frame supports, and wrong by 512
    codes. Two other posed views measured the pillar itself. The result is the
    pillar's own code, carried whole: not the estimate, and not anything
    between the two.
    """
    scene = cases["temporal-confirmation"]
    spatial = contract.spatial_completion(scene).frames[0]
    result = contract.spatial_then_temporal_completion(scene).frames[0]
    truth = scene.truth_codes(0)
    contradicted = [
        position
        for position, confirmation in enumerate(result.confirmations)
        if confirmation == contract.CONTRADICTED
    ]
    assert len(contradicted) == 1
    (position,) = contradicted
    estimate = spatial.codes[position]
    published = result.codes[position]
    assert result.states[position] == contract.TEMPORALLY_RECONSTRUCTED
    assert published == truth[position]
    assert estimate != published
    assert estimate - published == 512
    assert published != round((estimate + published) / 2)
    assert all(
        name.startswith("frame:1:") or name.startswith("frame:2:")
        for name in result.support[position]
    )


# --- the provenance-weighted fusion evidence -------------------------------------------
#
# Two cases carry a weighted volume, and they carry it for opposite reasons.
# On the multi-view case the accepted temporal result has to reach MORE of the
# field than the raw baseline does. On the aperture case the accepted spatial
# result has to reach exactly as much of it as the measurements alone do - and
# the counterfactual beside it shows that ignoring the provenance would not.


def _weighted(scene, candidate):
    return _named(contract.measure(scene, candidate(scene)))["weighted_fusion"]


def test_an_unconfirmed_single_frame_estimate_contributes_nothing_to_the_field(cases):
    """The aperture ambiguity, carried all the way to the fused volume.

    :mod:`scanner3d.depth_completion` cannot tell this scene's one-sample true
    gap from the one-sample dropout six columns away - the two scored
    neighbourhoods are byte-identical - so it fills both. What this asserts is
    that neither fill reaches the field: the weighted volume is byte-identical
    to the one the measurements alone produce, and the estimates are counted
    rather than silently dropped.
    """
    scene = cases["aperture-ambiguity"]
    baseline = _weighted(scene, contract.baseline)
    completed = _weighted(scene, contract.spatial_completion)

    assert completed["spatially_excluded_samples"] > 0
    assert completed["spatially_excluded_contribution"] == 0.0
    assert completed["matches_measured_only"] is True
    assert completed["observed_voxels"] == baseline["observed_voxels"]
    assert completed["voxel_updates"] == baseline["voxel_updates"]
    assert completed["contribution"] == baseline["contribution"]
    assert completed["surface_vertices"] == baseline["surface_vertices"]


def test_ignoring_the_provenance_would_have_put_that_estimate_in_the_field(cases):
    """Why the exclusion is load-bearing rather than decorative.

    The same candidate's completed codes fused at uniform weight - which is
    what a fusion path that read only the depths would do - observes strictly
    more voxels and pulls out a bigger surface. Those extra voxels are the
    aperture being filled, and they are exactly what the weighting refuses.
    """
    scene = cases["aperture-ambiguity"]
    completed = _weighted(scene, contract.spatial_completion)

    assert completed["unweighted_observed_voxels"] > completed["observed_voxels"]
    assert completed["unweighted_surface_vertices"] > completed["surface_vertices"]


def test_independently_supported_depth_reaches_the_field_and_changes_no_output(cases):
    """The accounting moves. The geometry does not. Both halves are asserted.

    The accepted temporal result puts strictly more voxel observations and
    strictly more accepted weight into the same volume than the raw baseline
    does, because samples other posed views really measured now arrive at a
    weight derived from their own confidence.

    And that buys nothing an output measure can see. Against the SAME raw
    multi-view field - the same three frames, the same poses, the same volume,
    the same extractor, the same truth model - observed space, the extracted
    surface and its truth-referenced coverage are identical. This is the task's
    negative result and it is asserted rather than noted, so a later change
    that produced a real gain would have to come and edit it.
    """
    scene = cases["multiview-object"]
    baseline = _weighted(scene, contract.baseline)
    temporal = _weighted(scene, contract.temporal_completion)

    assert temporal["temporally_reconstructed_samples"] > 0
    assert temporal["voxel_updates"] > baseline["voxel_updates"]
    assert temporal["contribution"] > baseline["contribution"]
    assert temporal["offered_weight"] > baseline["offered_weight"]
    assert temporal["matches_measured_only"] is False

    # The baseline is every frame, support included.
    assert temporal["measured_only_frames"] == temporal["frames_offered"]
    assert temporal["observed_voxel_gain"] == 0
    assert temporal["surface_vertex_gain"] == 0
    assert temporal["surface_coverage_gain"] == 0.0
    assert temporal["observed_voxels"] == baseline["observed_voxels"]
    assert temporal["unobserved_voxels"] == baseline["unobserved_voxels"]
    assert temporal["surface_coverage"] == temporal["measured_only_surface_coverage"]


def test_the_recovered_geometry_is_real_and_the_support_frames_already_have_it(cases):
    """Why the gain is zero, measured rather than argued.

    Fuse the one frame carrying the reconstructions ON ITS OWN and they are
    plainly load-bearing: the weighted field observes strictly more of the
    volume than that frame's raw bytes do. Fuse the whole scene - which is what
    a scan is - and the same reconstructions add nothing, because what they
    recover is what the supporting views measured, and those views are in the
    volume already.

    That pair is the argument against routing completed samples into an export
    that fuses every frame anyway, and it is the reason this corpus gates no
    gain.
    """
    scene = cases["multiview-object"]
    baseline = _weighted(scene, contract.baseline)
    temporal = _weighted(scene, contract.temporal_completion)

    assert temporal["withheld_support_frames"] == 1
    assert temporal["withheld_support_frames"] < temporal["frames_offered"]
    withheld_gain = (
        temporal["withheld_support_weighted_observed_voxels"]
        - temporal["withheld_support_observed_voxels"]
    )
    assert withheld_gain > 0
    # The very same voxel observations the weighted run applies to the whole
    # scene - and there they change nothing, because the supporting frames
    # observed those voxels first.
    assert withheld_gain == temporal["voxel_updates"] - baseline["voxel_updates"]
    assert temporal["observed_voxel_gain"] == 0


def test_a_reconstruction_never_weighs_as_much_as_the_measurement_beside_it(cases):
    """Read off the published counts, on the corpus rather than a unit scene."""
    scene = cases["multiview-object"]
    temporal = _weighted(scene, contract.temporal_completion)
    measured = temporal["measured_samples"]
    reconstructed = temporal["temporally_reconstructed_samples"]

    assert measured < temporal["offered_weight"] < measured + reconstructed


def test_a_candidate_with_no_accepted_result_publishes_no_weighted_evidence(cases):
    """The route a self-declared provenance does not have.

    ``dense_nearest`` fills this scene's aperture at confidence 1.0 and could
    just as easily have filled it at 0.1. It carries no accepted product
    result, so it reaches no weighted volume at all - which is the difference
    between provenance and a label, stated as an absence rather than as a gate
    it happens to fail.
    """
    scene = cases["aperture-ambiguity"]
    for candidate in (contract.dense_nearest, contract.broad_interpolation):
        result = candidate(scene)
        assert result.accepted is None
        assert "weighted_fusion" not in _named(contract.measure(scene, result))


def test_an_accepted_result_that_does_not_match_the_published_frames_is_refused(cases):
    """The binding that stops an accepted result being a passenger.

    A candidate carrying accepted results is claiming its published codes,
    states and confidences ARE those results. Changing one published state
    while leaving the accepted result alone is refused before anything is
    scored, because the weighted volume reads the accepted result and every
    other metric reads the published frames.
    """
    scene = cases["aperture-ambiguity"]
    result = contract.spatial_completion(scene)
    frame = result.frames[0]
    position = next(
        index
        for index, state in enumerate(frame.states)
        if state == contract.SPATIALLY_RECONSTRUCTED
    )
    states = list(frame.states)
    states[position] = contract.MISSING
    codes = list(frame.codes)
    codes[position] = 0
    confidence = list(frame.confidence)
    confidence[position] = 0.0
    support = {key: value for key, value in frame.support.items() if key != position}
    forged = contract.CandidateResult(
        name="forged_accepted_states",
        frames=(
            contract.CompletedFrame(
                index=frame.index,
                raw=frame.raw,
                codes=tuple(codes),
                states=tuple(states),
                confidence=tuple(confidence),
                support=support,
            ),
        ),
        accepted=result.accepted,
    )
    with pytest.raises(contract.CandidateRefused) as refusal:
        contract.validate(scene, forged)
    assert refusal.value.refusal == "accepted_result_mismatch"


def test_a_baseline_that_claims_more_than_its_raw_frame_returned_is_refused(cases):
    """The same binding on the raw route, which claims nothing beyond measurement."""
    scene = cases["aperture-ambiguity"]
    result = contract.baseline(scene)
    frame = result.frames[0]
    position = next(index for index, state in enumerate(frame.states) if state == contract.MISSING)
    codes = list(frame.codes)
    codes[position] = 12800
    states = list(frame.states)
    states[position] = contract.SPATIALLY_RECONSTRUCTED
    confidence = list(frame.confidence)
    confidence[position] = 0.5
    u, v = scene.window.pixel_of(0)
    forged = contract.CandidateResult(
        name="forged_raw_route",
        frames=(
            contract.CompletedFrame(
                index=frame.index,
                raw=frame.raw,
                codes=tuple(codes),
                states=tuple(states),
                confidence=tuple(confidence),
                support={position: (contract.support_name(0, u, v),)},
            ),
        ),
        accepted=(None,),
    )
    with pytest.raises(contract.CandidateRefused) as refusal:
        contract.validate(scene, forged)
    assert refusal.value.refusal == "accepted_result_mismatch"


def test_the_weighted_volume_is_the_same_volume_the_uniform_record_uses(cases):
    """One geometry, two fusions. The pair is the evidence; two volumes are not."""
    scene = cases["multiview-object"]
    assert scene.regions.fusion is not None
    assert scene.regions.weighted_fusion == scene.regions.fusion
