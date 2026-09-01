"""What tests.temporal_weighted_replay promises, exercised rather than read.

Six promises, and every one of them is the kind that fails silently on the day
something else has already gone wrong:

* the identity control really is an identity - the weighted seam reproduces the
  raw field byte for byte - and the replay STOPS rather than attributing a
  difference to temporal evidence when it does not;
* every field is bounded from, and fused at, the MODEL-frame poses the
  registration fitted, and the presentation half turn reaches nothing;
* the source set is exactly the declared nearest-eight/lower-index rule, which
  is a pure function of two integers and so cannot see an outcome;
* nothing is post-selected: every selected frame is completed, every completion
  is adapted and every adapted frame is fused, in order, and every hole is
  accounted for;
* the one arm that runs is named ``uncalibrated_pose_quality_upper_bound`` at
  pose quality 1.0 and says in the artifact itself that it is not calibrated;
* the replay is read-only on every exit path, publishes no truth-referenced or
  accuracy claim, and can reach no product path and no path opened for writing.

Everything here is offline. The capture is a small temporary directory of
synthetic 512000-byte frames, no device is opened, and the physical capture the
handoff runs the command against is never touched by pytest.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from scanner3d import pipeline, tsdf_input
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_temporal import MAX_SUPPORT_FRAMES, REJECTION_REASONS, PoseRecord
from scanner3d.errors import ScannerError
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.reconstruction_export import (
    TSDF_MAX_DEPTH,
    TSDF_MAX_WEIGHT,
    TSDF_MIN_DEPTH,
    TSDF_MIN_WEIGHT,
    TSDF_WEIGHT_POLICY,
    model_points,
    upright_pose,
)
from scanner3d.simulate import FACTORY_INTRINSICS, Box, Shape, Sphere, render
from scanner3d.tsdf import TsdfConfig, TsdfVolume, VolumeBounds
from scanner3d.tsdf_surface import TsdfSurfaceNotAllowed, extract_surface
from tests import temporal_weighted_replay as module
from tests.surface_oracle_replay import ReplayRefused, fingerprint, verify_unchanged

#: How many frames the synthetic replay renders. Nine is the fewest
#: :func:`scanner3d.pipeline.reconstruct_sweep` accepts - its revisit search
#: needs a gap of eight - so ten is the smallest sweep with anything to spare,
#: and it is two more than the eight sources one completion may name, which is
#: what makes the source rule pick a different set for different targets.
REPLAY_FRAMES = 10

#: The voxel ceiling the synthetic replay hands the harness. Far below the
#: desktop :data:`tests.temporal_weighted_replay.VOXEL_BUDGET` a physical run
#: uses, because these tests are about which bytes, poses and sources reach the
#: field rather than how finely it is diced. The chooser refuses anything above
#: the module ceiling, so this argument narrows and can never widen.
REPLAY_VOXEL_BUDGET = 20_000

#: The pure-Python field and extraction backend: exact, and needing no optional
#: dependency. A physical run names the accelerated one.
REPLAY_BACKEND = "python"


# --- reading the artifact without guessing at its types ---------------------------------------


def _dig(document: object, *keys: str) -> dict[str, object]:
    """One nested mapping out of the report, asserting the shape on the way down."""
    for key in keys:
        assert isinstance(document, dict), f"{key} is not inside a mapping"
        document = document[key]
    assert isinstance(document, dict)
    return document


def _int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _number(value: object) -> float:
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def _records(report: dict[str, object]) -> list[dict[str, object]]:
    per_frame = _dig(report, "temporal")["per_frame"]
    assert isinstance(per_frame, list)
    records: list[dict[str, object]] = []
    for item in per_frame:
        assert isinstance(item, dict)
        records.append(item)
    return records


# --- one real replay, recorded ------------------------------------------------------------------


def _capture(root: Path) -> Path:
    """A real capture folder: ten 512000-byte frames of an ASYMMETRIC scene.

    Rendered at the scale this repository's recorded captures are in - raw codes
    times :data:`~scanner3d.cli.DEPTH_SCALE_METRES`, which puts this object at a
    depth of about 120 - so every arm projects through the appliance's own depth
    window rather than past it.

    A sphere alone would be useless: it looks identical from every angle, so a
    turntable of one carries no information about how far it turned and the
    registration recovers the identity. The box beside it is what makes the
    recovered trajectory a real one, which is what the pose controls below need.
    """
    capture = root / "capture"
    capture.mkdir()
    shapes: list[Shape] = [
        Sphere(centre=(0.0, 0.0, 120.0), radius=14.0),
        Box(
            half_extents=(5.0, 9.0, 5.0),
            pose=Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (15.0, 6.0, 113.0)),
        ),
    ]
    for index in range(REPLAY_FRAMES):
        turned = rotation_about_axis((0.0, 1.0, 0.0), 0.06 * index, through=(0.0, 0.0, 120.0))
        frame = render(
            shapes,
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=DEPTH_SCALE_METRES,
            pose=turned,
        )
        (capture / f"frame_{index:04d}.raw").write_bytes(frame.raw)
    return capture


@dataclass(frozen=True)
class Replayed:
    """One complete synthetic replay, and everything it was asked and answered."""

    capture: Path
    raws: tuple[bytes, ...]
    report: dict[str, object]
    #: The registration result itself, so a later control can reuse the exact
    #: trajectory this run fitted rather than spend a second registration.
    sweep: object
    registered_at: tuple[Rigid, ...]
    measured_at: tuple[Rigid, ...]
    fused_at: tuple[tuple[Rigid, ...], ...]
    weighted_fused_at: tuple[tuple[Rigid, ...], ...]
    weighted_offered: tuple[tuple[tsdf_input.WeightedDepthFrame, ...], ...]
    configs: tuple[TsdfConfig, ...]
    before: dict[str, dict[str, object]]
    after: dict[str, dict[str, object]]


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> Replayed:
    """One real replay that reaches all three arms, with every handoff recorded.

    Built once for the module because it renders ten full frames and then runs
    the real registration, the real full-frame temporal completion of every one
    of them and three real fusions. Nothing is stubbed: the preparation, the
    sweep, the readiness gates, the chooser, the completion, the weighted
    adapters, the fields and the extractor are all the shipped code. The
    recorders are pass-throughs that write down what they were handed and
    delegate, so what is asserted below is what really happened rather than what
    a stand-in was told.
    """
    capture = _capture(tmp_path_factory.mktemp("temporal-weighted-replay"))
    registered_at: list[Rigid] = []
    measured_at: list[list[Rigid]] = []
    fused_at: list[list[Rigid]] = []
    weighted_fused_at: list[list[Rigid]] = []
    weighted_offered: list[tuple[tsdf_input.WeightedDepthFrame, ...]] = []
    configs: list[TsdfConfig] = []
    sweeps: list[object] = []

    sweep = pipeline.reconstruct_sweep

    def recording_sweep(clouds: object, **options: object) -> object:
        result = sweep(clouds, **options)  # type: ignore[arg-type]
        registered_at.extend(result.poses)
        sweeps.append(result)
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

        def integrate_weighted(
            self,
            frames: Sequence[tsdf_input.WeightedDepthFrame],
            poses: Sequence[Rigid],
            *,
            intrinsics: IntrinsicsRecord,
            depth_scale: float,
        ) -> None:
            weighted_fused_at.append(list(poses))
            weighted_offered.append(tuple(frames))
            super().integrate_weighted(
                frames, poses, intrinsics=intrinsics, depth_scale=depth_scale
            )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "reconstruct_sweep", recording_sweep)
        patch.setattr(module, "model_points", recording_model_points)
        patch.setattr(module, "TsdfVolume", RecordingVolume)
        before = fingerprint(capture)
        report = module.replay(
            capture,
            limit=None,
            backend=REPLAY_BACKEND,
            voxel_budget=REPLAY_VOXEL_BUDGET,
        )
        after = fingerprint(capture)

    return Replayed(
        capture=capture,
        raws=tuple(path.read_bytes() for path in sorted(capture.glob("frame_*.raw"))),
        report=report,
        sweep=sweeps[0],
        registered_at=tuple(registered_at),
        measured_at=tuple(measured_at[0]),
        fused_at=tuple(tuple(poses) for poses in fused_at),
        weighted_fused_at=tuple(tuple(poses) for poses in weighted_fused_at),
        weighted_offered=tuple(weighted_offered),
        configs=tuple(configs),
        before=before,
        after=after,
    )


# --- the premise ----------------------------------------------------------------------------


def test_the_replay_reaches_all_three_arms_so_every_control_below_means_something(
    replayed: Replayed,
) -> None:
    """A replay that stopped early would pass every claim below vacuously."""
    assert "stopped_at" not in replayed.report
    assert replayed.report["ready_before_fusion"] is True
    assert replayed.report["readiness_reasons"] == []
    for arm in ("raw_arm", "weighted_raw_arm", "candidate_arm"):
        assert arm in replayed.report
    assert _dig(replayed.report, "raw_arm", "surface")["extracted"] is True
    assert _dig(replayed.report, "candidate_arm", "surface")["extracted"] is True


def test_the_candidate_arm_really_reached_full_frame_temporal_and_weighted_fusion(
    replayed: Replayed,
) -> None:
    """The substantive one: real reconstructions, really weighted, really fused."""
    totals = _dig(replayed.report, "temporal", "totals")
    accepted = _int(totals["accepted"])
    assert _int(totals["frames_completed"]) == REPLAY_FRAMES
    assert accepted > 0
    assert _int(totals["temporally_reconstructed_samples"]) == accepted
    assert _number(totals["offered_weight"]) > 0.0

    inputs = _dig(replayed.report, "candidate_arm", "weighted_integration", "inputs")
    assert _int(inputs["temporally_reconstructed"]) == accepted
    assert _int(inputs["samples"]) == REPLAY_FRAMES * 640 * 400
    by_provenance = _dig(
        replayed.report,
        "candidate_arm",
        "weighted_integration",
        "accounting",
        "contribution_by_provenance",
    )
    assert _number(by_provenance["temporally_reconstructed"]) > 0.0
    assert _number(by_provenance["spatially_excluded"]) == 0.0
    assert _number(by_provenance["missing"]) == 0.0

    delta = _dig(replayed.report, "candidate_arm", "field_delta_against_raw")
    assert delta["identical"] is False
    assert _int(delta["observed_gained"]) > 0
    assert _int(delta["observed_lost"]) == 0
    assert _int(delta["weight_changed_voxels"]) > 0


def test_the_whole_frame_is_completed_at_the_origin_and_no_window_is_cut(
    replayed: Replayed,
) -> None:
    for record in _records(replayed.report):
        result = _dig(record, "result")
        assert _int(result["width"]) == 640
        assert _int(result["height"]) == 400
        assert _int(result["sample_count"]) == 640 * 400
        target = _dig(record, "result", "target")
        assert _int(target["u0"]) == 0
        assert _int(target["v0"]) == 0
    for frames in replayed.weighted_offered:
        for frame in frames:
            assert frame.origin == (0, 0)
            assert frame.adapter in {
                tsdf_input.ADAPTER_RAW,
                tsdf_input.ADAPTER_TEMPORAL,
            }


# --- the identity control ----------------------------------------------------------------------


def test_the_weighted_raw_control_reproduces_the_raw_field_and_surface_exactly(
    replayed: Replayed,
) -> None:
    control = _dig(replayed.report, "weighted_raw_arm")
    delta = _dig(control, "field_delta_against_raw")
    assert delta["values_identical"] is True
    assert delta["weights_identical"] is True
    assert delta["identical"] is True
    assert control["surface_identical_to_raw"] is True
    assert control["reproduces_raw"] is True
    assert control["surface"] == _dig(replayed.report, "raw_arm")["surface"]


def test_field_delta_is_not_blind_to_a_field_that_really_moved() -> None:
    """The control's teeth. A comparator that always says 'identical' proves nothing."""
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-30.0, -30.0, 95.0), (30.0, 30.0, 145.0)),
        voxel_size=5.0,
        truncation=10.0,
        min_depth=TSDF_MIN_DEPTH,
        max_depth=TSDF_MAX_DEPTH,
    )
    empty = TsdfVolume(config, backend=REPLAY_BACKEND)
    other = TsdfVolume(config, backend=REPLAY_BACKEND)
    assert module.field_delta(empty, other)["identical"] is True

    frame = render(
        [Sphere(centre=(0.0, 0.0, 120.0), radius=14.0)],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    other.integrate(
        [frame.raw],
        [Rigid.identity()],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    delta = module.field_delta(empty, other)
    assert delta["identical"] is False
    assert delta["values_identical"] is False
    assert _int(delta["weight_changed_voxels"]) > 0
    assert _int(delta["observed_gained"]) > 0
    assert _number(delta["max_abs_weight_delta"]) > 0.0


def test_a_weighted_seam_that_moves_the_field_stops_the_replay_before_the_candidate(
    replayed: Replayed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken identity control is a stop, not a difference blamed on temporal evidence."""
    real = tsdf_input.weighted_raw_frame

    def broken(raw: bytes) -> tsdf_input.WeightedDepthFrame:
        half = len(raw) // 2
        return real(bytes(half) + bytes(raw[half:]))

    monkeypatch.setattr(pipeline, "reconstruct_sweep", lambda clouds, **options: replayed.sweep)
    monkeypatch.setattr(tsdf_input, "weighted_raw_frame", broken)

    report = module.replay(
        replayed.capture,
        limit=None,
        backend=REPLAY_BACKEND,
        voxel_budget=REPLAY_VOXEL_BUDGET,
    )

    assert report["stopped_at"] == module.STOPPED_IDENTITY_CONTROL
    assert _dig(report, "weighted_raw_arm")["reproduces_raw"] is False
    assert _dig(report, "decision")["outcome"] == module.OUTCOME_COMPARATOR_INVALID
    assert "temporal" not in report
    assert "candidate_arm" not in report


# --- the model frame -------------------------------------------------------------------------


def test_every_arm_is_fused_at_the_very_poses_the_shared_field_was_measured_from(
    replayed: Replayed,
) -> None:
    """``integrate`` inverts each pose to carry a voxel back into its camera.

    So the poses that placed the points the field was BOUNDED from must be the
    poses every arm is FUSED at. A field bounded in the model frame and fused at
    presentation-rotated poses is not a scan drawn the wrong way up; it is every
    voxel projected half a turn away from the raw evidence that should have
    filled it, which is exactly the defect task 037 repaired.
    """
    assert replayed.measured_at == replayed.registered_at
    assert replayed.fused_at == (replayed.registered_at,)
    assert replayed.weighted_fused_at == (
        replayed.registered_at,
        replayed.registered_at,
    )
    assert tuple(upright_pose(pose) for pose in replayed.registered_at) != replayed.registered_at


def test_the_presentation_half_turn_would_have_left_no_surface_at_all(
    replayed: Replayed,
) -> None:
    """The counterfactual, run on the SAME field definition the replay used."""
    volume = TsdfVolume(replayed.configs[0], backend=REPLAY_BACKEND)
    volume.integrate(
        list(replayed.raws),
        [upright_pose(pose) for pose in replayed.registered_at],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    assert volume.evidence().weights.observed_voxels == 0
    with pytest.raises(TsdfSurfaceNotAllowed):
        extract_surface(volume, min_weight=TSDF_MIN_WEIGHT, backend=REPLAY_BACKEND)


def test_the_report_says_which_frame_every_number_in_it_is_in(
    replayed: Replayed,
) -> None:
    frame = replayed.report["frame"]
    assert isinstance(frame, str)
    assert "MODEL frame" in frame
    assert "applied to nothing here" in frame


# --- the fixed source rule ---------------------------------------------------------------------


def _nearest_by_hand(target: int, frames: int, limit: int) -> tuple[int, ...]:
    """An independent statement of the rule, written from its published prose alone."""
    ranked = sorted(
        (index for index in range(frames) if index != target),
        key=lambda index: (abs(index - target), index),
    )
    return tuple(sorted(ranked[:limit]))


def test_the_source_rule_is_the_nearest_eight_other_frames_by_index_distance() -> None:
    assert module.SOURCE_FRAMES == MAX_SUPPORT_FRAMES == 8
    assert module.nearest_sources(0, 12) == (1, 2, 3, 4, 5, 6, 7, 8)
    assert module.nearest_sources(5, 12) == (1, 2, 3, 4, 6, 7, 8, 9)
    assert module.nearest_sources(11, 12) == (3, 4, 5, 6, 7, 8, 9, 10)
    for target in range(12):
        chosen = module.nearest_sources(target, 12)
        assert len(chosen) == MAX_SUPPORT_FRAMES
        assert target not in chosen
        assert list(chosen) == sorted(set(chosen))
        assert chosen == _nearest_by_hand(target, 12, MAX_SUPPORT_FRAMES)


def test_a_tie_in_index_distance_is_broken_by_the_lower_index() -> None:
    """Target 2 of five: 1 and 3 tie at distance one, 0 and 4 tie at distance two."""
    assert module.nearest_sources(2, 5, limit=1) == (1,)
    assert module.nearest_sources(2, 5, limit=2) == (1, 3)
    assert module.nearest_sources(2, 5, limit=3) == (0, 1, 3)
    assert module.nearest_sources(2, 5, limit=4) == (0, 1, 3, 4)


def test_the_rule_takes_two_integers_and_so_cannot_see_an_outcome() -> None:
    for target in range(REPLAY_FRAMES):
        assert module.nearest_sources(target, REPLAY_FRAMES) == _nearest_by_hand(
            target, REPLAY_FRAMES, MAX_SUPPORT_FRAMES
        )
    with pytest.raises(ReplayRefused):
        module.nearest_sources(3, 3)
    with pytest.raises(ReplayRefused):
        module.nearest_sources(-1, 5)
    with pytest.raises(ReplayRefused):
        module.nearest_sources(0, 1)
    with pytest.raises(ReplayRefused):
        module.nearest_sources(0, 5, limit=0)


def test_the_rule_refuses_a_valid_target_when_only_seven_other_frames_exist() -> None:
    with pytest.raises(ReplayRefused, match=r"only 7 other frame.*requires exactly 8"):
        module.nearest_sources(0, 8)


def test_the_replay_offered_exactly_the_declared_source_set_to_every_target(
    replayed: Replayed,
) -> None:
    records = _records(replayed.report)
    assert [_int(record["index"]) for record in records] == list(range(REPLAY_FRAMES))
    for record in records:
        expected = _nearest_by_hand(_int(record["index"]), REPLAY_FRAMES, MAX_SUPPORT_FRAMES)
        chosen = record["sources"]
        assert isinstance(chosen, list)
        assert tuple(_int(index) for index in chosen) == expected
        offered = _dig(record, "result")["sources"]
        assert isinstance(offered, list)
        assert tuple(_int(_dig(frame)["index"]) for frame in offered) == expected
        assert _int(_dig(record, "result", "work")["source_frames"]) == len(expected)
        assert record["source_rule"] == module.SOURCE_RULE


# --- nothing is post-selected --------------------------------------------------------------------


def test_every_selected_frame_is_completed_adapted_and_fused_in_order(
    replayed: Replayed,
) -> None:
    assert len(_records(replayed.report)) == REPLAY_FRAMES
    candidate_frames = replayed.weighted_offered[1]
    assert len(candidate_frames) == REPLAY_FRAMES
    for index, frame in enumerate(candidate_frames):
        assert frame.raw == replayed.raws[index]
        assert frame.adapter == tsdf_input.ADAPTER_TEMPORAL
    frames = _dig(replayed.report, "candidate_arm", "integration", "frames")
    assert _int(frames["offered"]) == REPLAY_FRAMES


def test_every_hole_is_accounted_for_so_no_refusal_is_quietly_dropped(
    replayed: Replayed,
) -> None:
    for record in _records(replayed.report):
        work = _dig(record, "result", "work")
        rejections = _dig(record, "result", "work", "rejections")
        assert _int(work["attempted"]) == _int(work["holes"])
        assert _int(work["accepted"]) + _int(work["rejected"]) == _int(work["holes"])
        assert sum(_int(count) for count in rejections.values()) == _int(work["rejected"])
        assert set(rejections) == set(REJECTION_REASONS)
        assert _int(work["work_units"]) <= _int(work["work_budget"])
        assert _int(work["work_budget"]) <= _int(record["work_ceiling"])
        assert _int(record["declared_work_budget"]) == _int(work["work_budget"])


def test_every_arm_was_handed_the_same_field_definition(replayed: Replayed) -> None:
    assert len(replayed.configs) == 3
    assert replayed.configs[0] is replayed.configs[1] is replayed.configs[2]
    field = _dig(replayed.report, "field")
    assert field["config"] == replayed.configs[0].as_dict()
    assert _number(field["extraction_min_weight"]) == TSDF_MIN_WEIGHT
    assert field["shared_by"] == [
        module.ARM_RAW,
        module.ARM_WEIGHTED_RAW,
        module.ARM_CANDIDATE,
    ]


# --- the declared upper bound ---------------------------------------------------------------------


def test_the_only_arm_that_runs_is_named_and_labelled_uncalibrated(
    replayed: Replayed,
) -> None:
    assert module.POSE_QUALITY == 1.0
    assert module.UPPER_BOUND_ARM == "uncalibrated_pose_quality_upper_bound"
    temporal = _dig(replayed.report, "temporal")
    assert temporal["arm"] == module.UPPER_BOUND_ARM
    assert _number(temporal["pose_quality"]) == 1.0
    assert temporal["pose_quality_calibrated"] is False
    meaning = str(temporal["meaning"])
    assert "pose-quality upper bound under the fixed nearest-eight source rule" in meaning
    assert "not a global upper bound on the method or capture" in meaning
    assert "not a calibrated physical confidence" in meaning
    assert "most the current method could admit" not in meaning.lower()
    decision = _dig(replayed.report, "decision")
    assert decision["arm"] == module.UPPER_BOUND_ARM
    assert decision["pose_quality_calibrated"] is False


def test_every_pose_the_completion_is_given_carries_the_declared_quality_and_status(
    replayed: Replayed,
) -> None:
    frames, camera = module._temporal_frames(replayed.raws[:3], replayed.registered_at[:3])
    for index, frame in enumerate(frames):
        assert isinstance(frame.pose, PoseRecord)
        assert frame.pose.quality == module.POSE_QUALITY == 1.0
        assert frame.pose.accepted is True
        assert frame.pose.status == module.POSE_STATUS
        assert frame.index == index
        assert (frame.u0, frame.v0) == (0, 0)
        assert frame.pose.transform == replayed.registered_at[index]
    assert camera.depth_scale == DEPTH_SCALE_METRES
    assert camera.fx == FACTORY_INTRINSICS.values["fx"]
    assert camera.cy == FACTORY_INTRINSICS.values["cy"]


def test_the_decision_is_the_declared_rule_and_nothing_else(replayed: Replayed) -> None:
    decision = _dig(replayed.report, "decision")
    assert decision["rule"] == list(module.DECISION_RULE)
    assert decision["outcome"] in {
        module.OUTCOME_COMPARATOR_INVALID,
        module.OUTCOME_NEGATIVE,
        module.OUTCOME_NOT_COMPARABLE,
        module.OUTCOME_REJECTED,
        module.OUTCOME_CANDIDATE,
    }
    assert replayed.report["decision_rule"] == list(module.DECISION_RULE)


def test_no_accepted_temporal_sample_anywhere_is_a_negative_same_input_result() -> None:
    verdict = module.decide(
        identical_control=True,
        accepted_samples=0,
        field_identical=False,
        surface_identical=False,
        raw_surface={},
        candidate_surface={},
    )
    assert verdict["outcome"] == module.OUTCOME_NEGATIVE
    assert "no temporal sample was accepted anywhere" in str(verdict["because"])


def test_an_unchanged_field_and_surface_is_a_negative_same_input_result() -> None:
    verdict = module.decide(
        identical_control=True,
        accepted_samples=17,
        field_identical=True,
        surface_identical=True,
        raw_surface={},
        candidate_surface={},
    )
    assert verdict["outcome"] == module.OUTCOME_NEGATIVE
    assert "negative same-input result" in str(verdict["because"])


def test_a_broken_identity_control_invalidates_the_comparator_before_anything_else() -> None:
    verdict = module.decide(
        identical_control=False,
        accepted_samples=99,
        field_identical=False,
        surface_identical=False,
        raw_surface={},
        candidate_surface={},
    )
    assert verdict["outcome"] == module.OUTCOME_COMPARATOR_INVALID
    assert "proxies" not in verdict


def test_a_worse_observable_proxy_rejects_the_candidate_however_much_it_added() -> None:
    good = {
        "measured_support": {
            "measured": True,
            "surface_supported_fraction": 0.9,
            "evidence_explained_fraction": 0.8,
        },
        "components": {"component_count": 3, "isolated_fraction": 0.01},
    }
    lumpier = {
        "measured_support": {
            "measured": True,
            "surface_supported_fraction": 0.9,
            "evidence_explained_fraction": 0.8,
        },
        "components": {"component_count": 9, "isolated_fraction": 0.30},
    }
    verdict = module.decide(
        identical_control=True,
        accepted_samples=4242,
        field_identical=False,
        surface_identical=False,
        raw_surface=good,
        candidate_surface=lumpier,
    )
    assert verdict["outcome"] == module.OUTCOME_REJECTED
    assert verdict["worse_proxies"] == ["component_count", "isolated_fraction"]

    unsupported = {
        "measured_support": {
            "measured": True,
            "surface_supported_fraction": 0.5,
            "evidence_explained_fraction": 0.8,
        },
        "components": {"component_count": 3, "isolated_fraction": 0.01},
    }
    worse = module.decide(
        identical_control=True,
        accepted_samples=4242,
        field_identical=False,
        surface_identical=False,
        raw_surface=good,
        candidate_surface=unsupported,
    )
    assert worse["outcome"] == module.OUTCOME_REJECTED
    assert worse["worse_proxies"] == ["surface_supported_fraction"]


def test_a_changed_field_with_no_worse_proxy_is_only_a_scoped_uncalibrated_upper_bound() -> None:
    same = {
        "measured_support": {
            "measured": True,
            "surface_supported_fraction": 0.9,
            "evidence_explained_fraction": 0.8,
        },
        "components": {"component_count": 3, "isolated_fraction": 0.01},
    }
    verdict = module.decide(
        identical_control=True,
        accepted_samples=4242,
        field_identical=False,
        surface_identical=False,
        raw_surface=same,
        candidate_surface=same,
    )
    assert verdict["outcome"] == module.OUTCOME_CANDIDATE
    assert verdict["worse_proxies"] == []
    because = str(verdict["because"])
    assert "pose-quality upper bound under the fixed nearest-eight source rule" in because
    assert "not a global upper bound on the method or capture" in because
    assert "not a calibrated physical confidence" in because
    assert "added geometry" not in because.lower()
    for claim in module.CLAIMS:
        assert verdict[claim] is False


def test_an_unmeasurable_proxy_produces_no_verdict_rather_than_an_invented_one() -> None:
    verdict = module.decide(
        identical_control=True,
        accepted_samples=4242,
        field_identical=False,
        surface_identical=False,
        raw_surface={"measured_support": {"measured": False}, "components": {}},
        candidate_surface={"measured_support": {"measured": False}, "components": {}},
    )
    assert verdict["outcome"] == module.OUTCOME_NOT_COMPARABLE
    assert "no verdict is invented" in str(verdict["because"])


# --- what the artifact may never say --------------------------------------------------------------

#: Keys that would only exist in this artifact if somebody had invented the
#: truth they are measured against, promoted the upper bound to a result, or
#: chosen a region of the scan after looking at it.
FORBIDDEN_KEYS = frozenset(
    {
        "accuracy",
        "surface_accuracy",
        "dimension_error",
        "dimensional_accuracy",
        "edge_coverage",
        "edge_preservation",
        "thin_feature_survival",
        "truth",
        "truth_coverage",
        "improvement",
        "calibrated_confidence",
        "known_dimension",
        "object_dimensions",
        "plane_roi",
        "roughness",
    }
)


def _keys(document: object) -> set[str]:
    found: set[str] = set()
    if isinstance(document, dict):
        for key, value in document.items():
            found.add(str(key))
            found |= _keys(value)
    elif isinstance(document, list):
        for item in document:
            found |= _keys(item)
    return found


def test_the_artifact_carries_no_truth_referenced_or_accuracy_claim(
    replayed: Replayed,
) -> None:
    assert not (_keys(replayed.report) & FORBIDDEN_KEYS)
    for section in (
        replayed.report,
        _dig(replayed.report, "temporal"),
        _dig(replayed.report, "decision"),
    ):
        for claim in module.CLAIMS:
            assert section[claim] is False


def test_the_measured_support_is_labelled_support_and_never_accuracy(
    replayed: Replayed,
) -> None:
    support = _dig(replayed.report, "candidate_arm", "surface", "measured_support")
    assert "measurement support and not accuracy" in str(support["is"])
    evidence = _dig(replayed.report, "measured_evidence")
    assert "raw bytes alone" in str(evidence["is"])
    assert _int(evidence["points"]) > 0


def test_the_module_publishes_its_limitations_and_claims_no_physical_improvement(
    replayed: Replayed,
) -> None:
    published = " ".join(str(item) for item in module.LIMITATIONS)
    assert "read-only" in published
    assert "no truth model for a physical capture" in published
    assert "measurement SUPPORT rather than accuracy" in published
    assert "nothing here is routed into the product" in published
    assert replayed.report["limitations"] == list(module.LIMITATIONS)


def test_the_replay_never_names_a_region_of_the_scan_for_the_caller() -> None:
    """No plane, edge or region-of-interest option exists to be chosen after the fact."""
    for name in ("_plane", "plane_roi", "PLANE_ROI", "EDGE_BAND"):
        assert not hasattr(module, name)
    parsed = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
    options = {
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
    }
    assert options == {"--capture", "--limit", "--backend"}


# --- read-only, on every exit path ----------------------------------------------------------------


def test_the_replay_left_the_capture_byte_identical(replayed: Replayed) -> None:
    verdict = verify_unchanged(replayed.before, replayed.after)
    assert verdict["unchanged"] is True
    assert verdict["changed"] == []
    assert verdict["missing"] == []
    assert verdict["appeared"] == []
    assert verdict["files"] == REPLAY_FRAMES


def _tiny_capture(root: Path, frames: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        (root / f"frame_{index:04d}.raw").write_bytes(bytes([index + 1]) * 64)
    return root


def _digests(capture: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(capture.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("frames", range(1, MAX_SUPPORT_FRAMES + 1))
def test_too_few_frames_stop_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int
) -> None:
    capture = _tiny_capture(tmp_path / f"capture-{frames}", frames)

    def preparation_must_not_run(raw: bytes, **options: object) -> object:
        raise AssertionError("preparation ran before the exact-eight frame boundary")

    monkeypatch.setattr(module, "prepare_frame", preparation_must_not_run)
    report = module.replay(
        capture,
        limit=None,
        backend=REPLAY_BACKEND,
        voxel_budget=REPLAY_VOXEL_BUDGET,
    )

    assert report["stopped_at"] == module.STOPPED_FRAME_COUNT
    assert f"needs at least {MAX_SUPPORT_FRAMES + 1}" in str(report["stopped_because"])
    assert f"exactly {MAX_SUPPORT_FRAMES} other frames" in str(report["stopped_because"])
    assert "returns" not in report


def test_the_report_goes_to_stdout_as_strict_json_and_no_file_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _tiny_capture(tmp_path / "capture", 2)
    before = _digests(capture)
    listing = sorted(path.name for path in tmp_path.iterdir())

    code = module.main(["--capture", str(capture)])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["stopped_at"] == module.STOPPED_FRAME_COUNT
    assert document["read_only"]["unchanged"] is True
    assert _digests(capture) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == listing


def test_a_capture_that_cannot_be_bound_is_refused_and_still_proves_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 2
    assert document["stopped_at"] == "refused"
    assert document["read_only"]["unchanged"] is True


def test_an_error_part_way_still_publishes_the_after_fingerprint_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _tiny_capture(tmp_path / "capture", 2)

    def explode(capture_path: Path, **options: object) -> dict[str, object]:
        raise ScannerError("the replay fell over half way through")

    monkeypatch.setattr(module, "replay", explode)

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert document["stopped_at"] == "error"
    assert "fell over half way through" in document["stopped_because"]
    assert document["read_only"]["unchanged"] is True


def test_a_run_that_mutates_the_capture_exits_three_whatever_else_happened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _tiny_capture(tmp_path / "capture", 2)

    def explode_and_write(capture_path: Path, **options: object) -> dict[str, object]:
        (capture_path / "frame_0000.raw").write_bytes(b"rewritten by a run that promised not to")
        raise ScannerError("the replay fell over, after touching the capture")

    monkeypatch.setattr(module, "replay", explode_and_write)

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 3
    assert document["read_only"]["unchanged"] is False
    assert document["read_only"]["changed"] == ["frame_0000.raw"]


def test_a_file_that_appears_in_the_capture_is_also_a_read_only_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _tiny_capture(tmp_path / "capture", 2)

    def refuse_and_write(capture_path: Path, **options: object) -> dict[str, object]:
        (capture_path / "frame_9999.raw").write_bytes(b"a file this run had no business creating")
        raise ReplayRefused("this capture cannot be bound")

    monkeypatch.setattr(module, "replay", refuse_and_write)

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 3
    assert document["read_only"]["appeared"] == ["frame_9999.raw"]


def test_an_unexpected_error_is_reported_rather_than_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _tiny_capture(tmp_path / "capture", 2)

    def explode(capture_path: Path, **options: object) -> dict[str, object]:
        raise RuntimeError("something nobody predicted")

    monkeypatch.setattr(module, "replay", explode)

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert document["stopped_at"] == "unexpected_error"


def test_the_capture_must_be_named_because_there_is_no_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        module.main([])
    assert "--capture" in capsys.readouterr().err
    assert not hasattr(module, "DEFAULT_CAPTURE")


def test_a_report_that_is_not_strict_json_is_a_failure_rather_than_something_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Infinity is not a number JSON has a word for, and neither is a guess at one."""
    capture = _tiny_capture(tmp_path / "capture", 2)

    def unpublishable(capture_path: Path, **options: object) -> dict[str, object]:
        return {"kind": "not-strict", "measure": float("inf")}

    monkeypatch.setattr(module, "replay", unpublishable)

    code = module.main(["--capture", str(capture)])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert document["stopped_at"] == "unpublishable_report"
    assert document["read_only"]["unchanged"] is True
    assert "measure" not in document


# --- nothing is routed anywhere -------------------------------------------------------------------

#: Modules whose import would let this harness reach a product path, a device, a
#: network or the filesystem outside the capture it reads.
FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "urllib",
        "urllib.request",
        "PySide6",
        "scanner3d.capture_sequence",
        "scanner3d.cloud_export",
        "scanner3d.gui",
        "scanner3d.preview_provenance",
        "scanner3d.tracking",
        "scanner3d.v4l2_stream",
    }
)

#: Calls that would write, delete or export something, or route the candidate
#: into a production path. Matched on the called NAME, so a comment or a
#: docstring that merely mentions one of them does not trip the check.
FORBIDDEN_CALLS = frozenset(
    {
        "chmod",
        "export_reconstruction",
        "mkdir",
        "open",
        "reconstruct_capture",
        "remove",
        "rename",
        "rmdir",
        "system",
        "touch",
        "unlink",
        "write_bytes",
        "write_ply",
        "write_text",
    }
)


def _module_source() -> str:
    return Path(str(module.__file__)).read_text(encoding="utf-8")


def _imported(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _called(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_the_module_imports_nothing_that_could_route_the_candidate_into_the_product() -> None:
    imported = _imported(_module_source())
    assert not (imported & FORBIDDEN_MODULES), sorted(imported & FORBIDDEN_MODULES)


def test_the_module_calls_nothing_that_writes_deletes_or_exports() -> None:
    called = _called(_module_source())
    assert not (called & FORBIDDEN_CALLS), sorted(called & FORBIDDEN_CALLS)
    assert "read_bytes" in called


def _imported_names(source: str) -> dict[str, set[str]]:
    """``{module: {name, ...}}`` for every ``from ... import ...`` in the source."""
    names: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return names


def test_the_module_imports_the_shipped_constants_rather_than_restating_them() -> None:
    """Every threshold the arms fuse and extract at is the appliance's own."""
    from scanner3d.depth_temporal import DEFAULT_POLICY as SHIPPED_TEMPORAL_POLICY

    imported = _imported_names(_module_source())
    assert imported["scanner3d.reconstruction_export"] >= {
        "TSDF_MAX_DEPTH",
        "TSDF_MAX_WEIGHT",
        "TSDF_MIN_DEPTH",
        "TSDF_MIN_WEIGHT",
        "TSDF_WEIGHT_POLICY",
        "choose_tsdf_resolution",
        "model_points",
    }
    assert imported["tests.surface_oracle_replay"] >= {
        "ISOLATED_MAX_POINTS",
        "LINKAGE_RADIUS_VOXELS",
        "MODEL_FRAME",
        "bind_identity",
        "fingerprint",
        "return_coverage",
        "verify_unchanged",
    }
    assert module.TEMPORAL_POLICY is SHIPPED_TEMPORAL_POLICY
    assert module.VOXEL_BUDGET == 128**3
    assert module.SOURCE_FRAMES == MAX_SUPPORT_FRAMES
    assert module.TEMPORAL_BACKEND == "python"
    assert TSDF_WEIGHT_POLICY.value == "uniform"
    assert (TSDF_MIN_DEPTH, TSDF_MAX_DEPTH, TSDF_MIN_WEIGHT) == (60.0, 300.0, 1.0)
    assert TSDF_MAX_WEIGHT > 0.0


def test_the_replay_module_is_not_collected_as_a_test_module() -> None:
    assert not Path(str(module.__file__)).name.startswith("test_")
