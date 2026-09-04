"""One preparation path, and the proof that both entry points really use it.

The window and ``scanner3d reconstruct`` used to build their registration
clouds separately, with the same four steps written out twice. The tests that
matter most here are therefore not about any single step: they are the ones
that would fail if the two entry points ever started disagreeing again, and the
ones that check the record a report carries says what actually happened rather
than what the code was expected to do.

Nothing here opens a device. The frames are rendered by
:mod:`scanner3d.simulate` into real 512000-byte ``Y16 `` buffers, so the
deprojection under test is the deprojection a capture gets.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import cli
from scanner3d.depth_metric import SAMPLES_FORMAT, deproject_depth_frame
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.preparation import (
    DEPTH_CLIP_FAR,
    DEPTH_CLIP_NEAR,
    PREPARATION_ORDER,
    PREPARATION_VERSION,
    RECONSTRUCT_POINTS,
    SCHEMA_VERSION,
    DepthBand,
    DepthBandUnusable,
    FusionFrameUnusable,
    FusionMaskSummary,
    PointsUnusable,
    PreparationNotAllowed,
    PreparationSummary,
    PreparedCloud,
    TargetPointsUnusable,
    filter_statement,
    mask_support_band,
    preparation_unit,
    prepare_frame,
    prepare_points,
)
from scanner3d.simulate import FACTORY_INTRINSICS, Plane, Shape, Sphere, render
from scanner3d.support_plane import POLICY_VERSION, REASON_APPLIED, REASON_LITTLE_OBJECT

#: A scene with an object standing on a slanted surface, at a size where the
#: surface is genuinely the thing surrounding the object. Rendered rather than
#: constructed, so what is prepared is what a capture would have delivered.
BED = Plane(normal=(0.0, -0.55, 1.0), offset=170.0)
OBJECT = Sphere(centre=(0.0, -20.0, 115.0), radius=35.0)


def scene_frame(*, shapes: Sequence[Shape] | None = None) -> bytes:
    frame = render(
        shapes or [BED, OBJECT],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=cli.DEPTH_SCALE_METRES,
    )
    return frame.raw


def sheet(*, span: float, steps: int, depth: float) -> list[Point]:
    return [
        (
            -span / 2.0 + span * i / (steps - 1),
            -span / 2.0 + span * j / (steps - 1),
            depth,
        )
        for i in range(steps)
        for j in range(steps)
    ]


# --- the order, stated once ---------------------------------------------------


def test_the_order_the_steps_run_in_is_stated_on_every_record():
    prepared = prepare_frame(scene_frame(), target_points=RECONSTRUCT_POINTS)
    record = prepared.as_dict()

    assert record["order"] == [
        "deproject",
        "depth clip",
        "deterministic thinning",
        "speckle filtering",
        "support-plane filtering",
    ]
    assert list(PREPARATION_ORDER) == record["order"]
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["preparation_version"] == PREPARATION_VERSION
    assert record["support_plane_policy_version"] == POLICY_VERSION


def test_only_points_inside_the_depth_interval_survive():
    near = [(0.0, 0.0, DEPTH_CLIP_NEAR - 1.0)] * 40
    far = [(0.0, 0.0, DEPTH_CLIP_FAR + 1.0)] * 40
    inside = sheet(span=100.0, steps=12, depth=150.0)
    prepared = prepare_points(near + inside + far, target_points=RECONSTRUCT_POINTS)

    assert prepared.deprojected_points == len(near) + len(inside) + len(far)
    assert prepared.clipped_points == len(inside)


def test_thinning_is_a_stride_and_gives_the_same_points_every_time():
    dense = sheet(span=100.0, steps=80, depth=150.0)
    first = prepare_points(dense, target_points=500)
    second = prepare_points(list(dense), target_points=500)

    assert first.thinned_points == pytest.approx(500, abs=60)
    assert first.points == second.points


def test_a_frame_that_caught_nothing_is_a_frame_and_says_neither_filter_ran():
    """An empty cloud is not a refusal, and "not run" is not "removed nothing"."""
    prepared = prepare_points([(0.0, 0.0, 10.0)] * 30, target_points=RECONSTRUCT_POINTS)

    assert prepared.points == ()
    assert prepared.speckle is None
    assert prepared.support is None
    assert prepared.support_reason == "not-run"
    record = prepared.as_dict()
    assert record["speckle"] is None
    assert record["support_plane"] is None


@pytest.mark.parametrize("target", [0, -5, 2.5, True])
def test_a_thinning_target_that_is_not_a_count_is_refused_before_any_work(target: object):
    with pytest.raises(TargetPointsUnusable):
        prepare_points(sheet(span=100.0, steps=12, depth=150.0), target_points=target)  # type: ignore[arg-type]
    with pytest.raises(TargetPointsUnusable):
        prepare_frame(b"", target_points=target)  # type: ignore[arg-type]


def test_every_preparation_refusal_is_a_scanner_error_with_a_status():
    assert issubclass(PreparationNotAllowed, ScannerError)
    assert PreparationNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE


# --- the support plane, through the whole path --------------------------------


def test_a_rendered_object_on_a_slanted_bed_loses_the_bed_and_keeps_the_object():
    prepared = prepare_frame(scene_frame(), target_points=RECONSTRUCT_POINTS)

    assert prepared.support_applied is True
    assert prepared.support_reason == REASON_APPLIED
    assert prepared.support_removed > prepared.kept_points
    # The object is nearer than the surface it stands on, and what survives is
    # the object: a check on the scene rather than on the filter's own report.
    assert max(point[2] for point in prepared.points) < 175.0


def test_a_frame_that_is_almost_all_plane_keeps_every_point_it_had():
    flat = sheet(span=200.0, steps=45, depth=150.0)
    speck = [(0.0, 0.0, 160.0)] * 5
    prepared = prepare_points(flat + speck, target_points=RECONSTRUCT_POINTS)

    assert prepared.support is not None
    assert prepared.support_applied is False
    assert prepared.support_reason == REASON_LITTLE_OBJECT
    assert prepared.support_removed == 0
    # Withdrawal is byte-for-point equivalent to the cloud the previous step
    # produced: the plane filter contributed nothing at all.
    assert prepared.speckle is not None
    assert prepared.points == prepared.speckle.points


# --- one path, two entry points -----------------------------------------------


def test_the_window_prepares_a_raw_frame_exactly_as_the_shared_path_does():
    from scanner3d import gui

    raw = scene_frame()
    through_window = gui._prepare_frame(raw, gui.RECONSTRUCT_POINTS)
    through_path = prepare_frame(raw, target_points=RECONSTRUCT_POINTS)

    assert gui.RECONSTRUCT_POINTS == RECONSTRUCT_POINTS
    assert through_window.points == through_path.points
    assert through_window.as_dict() == through_path.as_dict()


def test_both_capture_paths_read_their_cloud_from_the_shared_preparation_record(tmp_path: Path):
    """Ordinary capture and tentative recovery take the SAME prepared cloud.

    The two differ only in where the bytes came from: a committed frame is read
    back off disk by ``_load_cloud``, and a frame a resumed capture is holding
    tentatively has no file yet and is deprojected from memory by
    ``_cloud_from_bytes``. Both are thin wrappers over one record, and this
    is the test that says so - if either grew its own arithmetic again, a
    recovered segment would be registered against clouds the rest of the
    capture was not.
    """
    from scanner3d import gui

    marker = ((7.0, 8.0, 9.0),)
    asked: list[int] = []

    def spy(_raw: bytes, target_points: int, _depth_band: object = None) -> PreparedCloud:
        asked.append(target_points)
        return PreparedCloud.of_points(marker, target_points=target_points)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(gui, "_prepare_frame", spy)
        path = tmp_path / "frame_0000.raw"
        path.write_bytes(b"\x00\x01")

        assert gui._cloud_from_bytes(b"\x00\x01", gui.PREVIEW_POINTS) == marker
        assert gui._load_cloud(path, gui.RECONSTRUCT_POINTS) == marker
    finally:
        monkeypatch.undo()

    assert asked == [gui.PREVIEW_POINTS, gui.RECONSTRUCT_POINTS]


def test_the_window_and_the_shared_path_state_the_same_depth_assumption():
    from scanner3d import gui

    assert gui._depth_unit() == preparation_unit()


def test_the_command_line_prepares_a_capture_through_the_shared_path(
    tmp_path, monkeypatch, capsys, readiness_stood_down
):
    """The CLI's reported preparation is the shared path's, frame for frame.

    Registration is stood in for. What is under test is which clouds the command
    builds and what it says about them, and a real sweep registration here would
    measure the optimiser instead.
    """
    raw = scene_frame()
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    for index in range(4):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)

    seen: dict[str, object] = {}

    def fake_reconstruct(clouds, **kwargs):
        seen["clouds"] = [tuple(cloud) for cloud in clouds]
        # One pose per cloud, because that is what a registration returns and
        # what everything downstream is entitled to assume. The stub used to
        # hand back none at all, which no real result can be.
        return SimpleNamespace(
            as_dict=lambda: {"stub": True},
            poses=[Rigid.identity() for _ in clouds],
        )

    monkeypatch.setattr("scanner3d.pipeline.reconstruct_sweep", fake_reconstruct)
    assert cli.main([cli.RECONSTRUCT_COMMAND, str(frames_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    expected = PreparationSummary.of(
        [prepare_frame(raw, target_points=RECONSTRUCT_POINTS) for _ in range(4)]
    )
    assert payload["command"]["preparation"] == expected.as_dict()
    assert payload["command"]["preparation_statement"] == filter_statement(expected)
    # This command registers and reports; it fuses no surface, so it masks no
    # full-resolution frame and says that rather than leaving the block out.
    assert payload["command"]["preparation"]["full_resolution_fusion"]["ran"] is False
    assert "No surface was fused on this path" in payload["command"]["preparation_statement"]
    assert payload["command"]["per_frame_filter"]["points_rejected"] == (
        expected.speckle_removed_points
    )
    # And the clouds it handed the registration are the prepared ones.
    assert seen["clouds"] == [
        prepare_frame(raw, target_points=RECONSTRUCT_POINTS).points for _ in range(4)
    ]


def test_preparing_a_frame_leaves_the_frame_on_disk_exactly_as_it_was(tmp_path: Path):
    raw = scene_frame()
    path = tmp_path / "frame_0000.raw"
    path.write_bytes(raw)

    prepare_frame(path.read_bytes(), target_points=RECONSTRUCT_POINTS)

    assert path.read_bytes() == raw


# --- the aggregate a report carries -------------------------------------------


def applied_cloud() -> PreparedCloud:
    return prepare_frame(scene_frame(), target_points=RECONSTRUCT_POINTS)


def withdrawn_cloud() -> PreparedCloud:
    flat = sheet(span=200.0, steps=45, depth=150.0)
    return prepare_points(flat + [(0.0, 0.0, 160.0)] * 5, target_points=RECONSTRUCT_POINTS)


def empty_cloud() -> PreparedCloud:
    return prepare_points([(0.0, 0.0, 10.0)] * 30, target_points=RECONSTRUCT_POINTS)


def test_the_summary_counts_applied_withdrawn_and_never_run_frames_separately():
    applied = applied_cloud()
    withdrawn = withdrawn_cloud()
    summary = PreparationSummary.of([applied, applied, withdrawn, empty_cloud()])

    assert summary.frames == 4
    assert summary.support_applied_frames == 2
    assert summary.support_withdrawn_frames == 1
    assert summary.support_not_run_frames == 1
    assert summary.support_removed_points == 2 * applied.support_removed
    assert summary.withdrawal_reasons == ((REASON_LITTLE_OBJECT, 1),)


def test_the_summary_line_names_the_withdrawals_rather_than_only_the_successes():
    line = PreparationSummary.of([applied_cloud(), withdrawn_cloud()]).summary_line()

    assert "1/2 frames" in line
    assert REASON_LITTLE_OBJECT in line


def test_the_exported_statement_says_what_both_filters_did_and_what_was_kept():
    summary = PreparationSummary.of([applied_cloud(), withdrawn_cloud()])
    statement = filter_statement(summary)

    assert "scanner3d.preparation.prepare_frame" in statement
    assert "scanner3d.denoise.strip_speckle" in statement
    assert "scanner3d.support_plane.suppress_support_plane" in statement
    assert str(summary.support_removed_points) in statement
    # The two claims a reader must not be left to guess at.
    assert "withdrew on 1 frame(s)" in statement
    assert "deliberately preserved" in statement


def test_a_cloud_that_arrived_already_prepared_claims_no_filtering_of_its_own():
    points = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    prepared = PreparedCloud.of_points(points)

    assert prepared.points == points
    assert prepared.speckle is None
    assert prepared.support is None
    assert prepared.support_applied is False
    assert PreparationSummary.of([prepared]).support_not_run_frames == 1


# --- the public boundary on already-deprojected points ------------------------


def test_a_cloud_that_is_not_a_sequence_is_refused_before_anything_is_read():
    """A generator cannot be clipped, and consuming one to find out would eat it."""
    with pytest.raises(PointsUnusable):
        prepare_points((p for p in sheet(span=100.0, steps=12, depth=150.0)), target_points=10)  # type: ignore[arg-type]
    with pytest.raises(PointsUnusable):
        prepare_points(150.0, target_points=10)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "point",
    [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), 150.0, "xyz", (1.0, 2.0, "z")],
)
def test_a_point_that_is_not_a_usable_triple_is_refused_rather_than_crashing(point: object):
    """Typed, actionable, and raised by this module - not IndexError or TypeError.

    These used to escape as built-in exceptions from inside a comprehension,
    which told the caller a coordinate was malformed only by naming a line of
    somebody else's code.
    """
    scene = [*sheet(span=100.0, steps=12, depth=150.0), point]
    with pytest.raises(PointsUnusable):
        prepare_points(scene, target_points=RECONSTRUCT_POINTS)  # type: ignore[arg-type]
    with pytest.raises(PointsUnusable):
        PreparedCloud.of_points(scene)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_coordinate_is_refused_rather_than_silently_clipped(bad: float):
    """The worst of the three, because it left no trace at all.

    A NaN depth compares false against BOTH ends of the depth clip, so the
    point simply was not in the output and the record counted it as out of
    range. The frame came out shorter and every number about it was wrong in a
    way nothing could detect.
    """
    scene = [*sheet(span=100.0, steps=12, depth=150.0), (0.0, 0.0, bad)]

    with pytest.raises(PointsUnusable) as raised:
        prepare_points(scene, target_points=RECONSTRUCT_POINTS)

    assert "not finite" in str(raised.value)
    # Also on the axes the clip never looks at, which the clip could not have
    # caught even in principle.
    with pytest.raises(PointsUnusable):
        prepare_points([*scene[:-1], (bad, 0.0, 150.0)], target_points=RECONSTRUCT_POINTS)


def test_the_points_boundary_is_a_preparation_refusal_with_a_status():
    assert issubclass(PointsUnusable, PreparationNotAllowed)
    assert PointsUnusable.status is DoctorStatus.UNSUPPORTED_MODE


# --- the same plane, at full resolution ---------------------------------------


def unpacked(raw: bytes) -> tuple[int, ...]:
    return struct.unpack(SAMPLES_FORMAT, raw)


def test_an_applied_support_band_is_masked_out_of_the_frame_that_gets_fused():
    """The bed leaves the registration cloud AND the frame the mesh is fused from.

    This is the whole of the correction: the plane used to be removed only from
    the decimated cloud, so the surface the operator opens still had the bed in
    it. The masked samples are set to the sensor's own no-return value, which
    the fusion already skips, and every surviving return is checked against the
    band by deprojecting it independently of the code under test.
    """
    raw = scene_frame()
    prepared = prepare_frame(raw, target_points=RECONSTRUCT_POINTS)
    assert prepared.support is not None and prepared.support.applied

    fused = mask_support_band(
        raw,
        prepared.support,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=cli.DEPTH_SCALE_METRES,
    )

    assert fused.applied is True
    assert fused.masked is True
    assert fused.samples_masked > 0
    assert fused.samples_fused > 0
    # Only returns were taken, and only by being zeroed: the frame keeps its
    # length, and no sample was changed to some other depth.
    assert len(fused.raw) == len(raw)
    before, after = unpacked(raw), unpacked(fused.raw)
    changed = [index for index, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == fused.samples_masked
    assert all(after[index] == 0 for index in changed)
    assert all(before[index] != 0 for index in changed)

    # And what survived is outside the band, checked by a deprojection this
    # module did not perform.
    plane = prepared.support.plane
    assert plane is not None
    half = prepared.support.band / 2.0
    kept = deproject_depth_frame(
        fused.raw,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=cli.DEPTH_SCALE_METRES,
        unit=preparation_unit(),
    )
    assert kept.points
    assert all(abs(plane.signed_distance((p.x, p.y, p.z))) > half * 0.999 for p in kept.points)
    # The band that was masked is the band the registration cloud lost, not a
    # second decision: every removed point of the decimated cloud is inside it.
    assert all(abs(plane.signed_distance(p)) <= half * 1.001 for p in prepared.support.removed)


def test_a_withdrawn_frame_is_fused_exactly_as_it_was_captured():
    """Conservative withdrawal reaches the mesh too, or it is not conservative."""
    raw = scene_frame(shapes=[Plane(normal=(0.0, 0.0, 1.0), offset=150.0)])
    prepared = prepare_frame(raw, target_points=RECONSTRUCT_POINTS)
    assert prepared.support is not None and not prepared.support.applied

    fused = mask_support_band(
        raw,
        prepared.support,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=cli.DEPTH_SCALE_METRES,
    )

    assert fused.applied is False
    assert fused.masked is False
    assert fused.samples_masked == 0
    assert fused.reason == prepared.support.reason
    # The SAME object, not an equal copy: a frame that was not masked was not
    # rebuilt either.
    assert fused.raw is raw


def test_a_frame_the_filter_never_ran_on_is_fused_exactly_as_it_was_captured():
    raw = scene_frame()

    fused = mask_support_band(
        raw, None, intrinsics=FACTORY_INTRINSICS, depth_scale=cli.DEPTH_SCALE_METRES
    )

    assert fused.raw is raw
    assert fused.applied is False
    assert fused.reason == "not-run"
    assert fused.samples_masked == 0


def test_a_frame_or_calibration_the_fusion_cannot_use_is_refused_untouched():
    prepared = prepare_frame(scene_frame(), target_points=RECONSTRUCT_POINTS)

    with pytest.raises(FusionFrameUnusable):
        mask_support_band(
            b"short",
            prepared.support,
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=cli.DEPTH_SCALE_METRES,
        )
    with pytest.raises(FusionFrameUnusable):
        mask_support_band(
            scene_frame(),
            prepared.support,
            intrinsics=FACTORY_INTRINSICS,
            depth_scale=0.0,
        )
    assert issubclass(FusionFrameUnusable, PreparationNotAllowed)


def test_the_fusion_summary_and_the_cloud_summary_describe_different_artifacts():
    """One aggregate per artifact, so neither one's counts can be read into the other."""
    raw = scene_frame()
    prepared = prepare_frame(raw, target_points=RECONSTRUCT_POINTS)
    masked = mask_support_band(
        raw,
        prepared.support,
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=cli.DEPTH_SCALE_METRES,
    )
    untouched = mask_support_band(
        raw, None, intrinsics=FACTORY_INTRINSICS, depth_scale=cli.DEPTH_SCALE_METRES
    )
    fusion = FusionMaskSummary.of([masked, untouched])

    assert fusion.frames == 2
    assert fusion.masked_frames == 1
    assert fusion.unmasked_frames == 1
    assert fusion.samples_masked == masked.samples_masked
    assert fusion.reasons == (("not-run", 1),)
    assert "masked out of 1/2 full-resolution frames" in fusion.summary_line()

    record = PreparationSummary.of([prepared]).as_dict(fusion)
    cloud_block = record["registered_cloud"]
    fusion_block = record["full_resolution_fusion"]
    assert isinstance(cloud_block, dict)
    assert isinstance(fusion_block, dict)
    assert cloud_block["support_plane"]["applied_frames"] == 1
    assert cloud_block["kept_points"] == prepared.kept_points
    assert cloud_block["speckle"]["removed_points"] == prepared.speckle_removed
    assert fusion_block["ran"] is True
    assert fusion_block["samples_masked"] == masked.samples_masked
    # The speckle filter's numbers appear only in the registered-cloud block:
    # it never touched a full-resolution frame.
    assert "speckle" not in fusion_block


def test_a_path_that_fuses_nothing_says_so_rather_than_omitting_the_block():
    """An absent block would read as "masking ran and found nothing"."""
    record = PreparationSummary.of([applied_cloud()]).as_dict()
    fusion_block = record["full_resolution_fusion"]

    assert isinstance(fusion_block, dict)
    assert fusion_block["ran"] is False
    assert "no full-resolution fusion happened" in str(fusion_block["means"])


def test_the_summary_line_names_the_points_that_were_kept():
    """An operator cannot read a removal count without the survivor count beside it."""
    summary = PreparationSummary.of([applied_cloud(), withdrawn_cloud()])

    assert f"{summary.kept_points} points kept for registration" in summary.summary_line()


def test_the_statement_keeps_the_two_filters_reach_exact():
    """Speckle touched the registration cloud only, and the statement has to say so."""
    raw = scene_frame()
    prepared = prepare_frame(raw, target_points=RECONSTRUCT_POINTS)
    fusion = FusionMaskSummary.of(
        [
            mask_support_band(
                raw,
                prepared.support,
                intrinsics=FACTORY_INTRINSICS,
                depth_scale=cli.DEPTH_SCALE_METRES,
            )
        ]
    )
    summary = PreparationSummary.of([prepared])

    with_fusion = filter_statement(summary, fusion)
    without = filter_statement(summary)

    assert "ran on the DECIMATED registration clouds only" in with_fusion
    assert "still contain every stray it rejected" in with_fusion
    assert f"{fusion.samples_masked} of {fusion.samples_returned} depth samples" in with_fusion
    assert "The raw frames on disk are unchanged and complete" in with_fusion
    # The claim the round-1 statement got wrong: nothing may say a removed
    # point is in no exported file unless the fusion was actually masked.
    assert "in no file this export wrote" not in with_fusion
    assert "No surface was fused on this path" in without
    assert "no full-resolution frame was masked for one" in without
    # And it does not borrow the fusion's claim: nothing here says a removed
    # point is missing from an exported mesh, because none was written.
    assert "no masked sample reached the fused field" not in without


# --- the depth band ---------------------------------------------------------------


def test_a_stated_depth_band_is_the_interval_the_clip_keeps():
    band = DepthBand(near=120.0, far=200.0)
    inside = sheet(span=100.0, steps=12, depth=150.0)
    nearer = [(0.0, 0.0, 100.0)] * 40  # inside the sensor's interval, outside the band
    further = [(0.0, 0.0, 250.0)] * 40
    prepared = prepare_points(
        nearer + inside + further, target_points=RECONSTRUCT_POINTS, depth_band=band
    )

    assert prepared.clipped_points == len(inside)
    assert prepared.depth_band == band
    assert prepared.as_dict()["depth_clip"] == {"near": 120.0, "far": 200.0}


def test_stating_no_band_keeps_the_sensor_interval_and_the_record_says_so():
    prepared = prepare_points(
        sheet(span=100.0, steps=12, depth=150.0), target_points=RECONSTRUCT_POINTS
    )

    assert prepared.depth_band == DepthBand.default()
    assert DepthBand.default() == DepthBand(near=DEPTH_CLIP_NEAR, far=DEPTH_CLIP_FAR)
    assert prepared.as_dict()["depth_clip"] == {"near": DEPTH_CLIP_NEAR, "far": DEPTH_CLIP_FAR}
    assert PreparedCloud.of_points([(0.0, 0.0, 100.0)]).depth_band == DepthBand.default()


@pytest.mark.parametrize(
    "near, far",
    [
        (200.0, 100.0),
        (100.0, 100.0),
        (0.0, 100.0),
        (-5.0, 100.0),
        (float("nan"), 100.0),
        (100.0, float("inf")),
        (True, 100.0),
        (100.0, "200"),
    ],
)
def test_a_band_the_clip_cannot_apply_is_refused_before_any_point_is_read(near, far):
    with pytest.raises(DepthBandUnusable):
        prepare_points(
            [(0.0, 0.0, 150.0)] * 5,
            target_points=RECONSTRUCT_POINTS,
            depth_band=DepthBand(near=near, far=far),
        )


def test_the_frame_path_carries_the_band_through_to_the_clip():
    raw = scene_frame(shapes=[Sphere(centre=(20.0, 0.0, 150.0), radius=20.0)])

    kept = prepare_frame(raw, target_points=RECONSTRUCT_POINTS, depth_band=DepthBand(100.0, 200.0))
    excluded = prepare_frame(
        raw, target_points=RECONSTRUCT_POINTS, depth_band=DepthBand(200.0, 300.0)
    )

    assert kept.clipped_points > 0
    assert kept.depth_band == DepthBand(100.0, 200.0)
    assert excluded.clipped_points == 0
    assert excluded.points == ()
    assert excluded.depth_band == DepthBand(200.0, 300.0)
