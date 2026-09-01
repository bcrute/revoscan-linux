"""What the export boundary does with a reconstruction it must not publish.

:mod:`tests.test_reconstruction_readiness` owns the gates themselves - what
each one measures and which fixtures it is right to refuse. Nothing here
re-checks any of that. The subject of this module is the BOUNDARY: that
:func:`scanner3d.reconstruction_export.export_reconstruction` consults those
gates at all, that it consults the cheap ones before spending minutes on a
fusion nobody may keep, that a refusal leaves a capture folder exactly as it
found it, and that a published export carries the evidence for the decision
that let it exist.

The fifth claim is the one this module exists for. ``export_reconstruction``
takes a capture directory and a list of poses, and it used to believe both: a
caller holding clouds and poses from somewhere else never asked the capture
anything, so a capture whose tracking loss nothing closed could be reconstructed
by going straight to this function. There is no metadata argument to omit any
more, and the last test below is what says so.

Everything is offline and synthetic, reusing :mod:`tests.test_reconstruction_export`'s
own sphere capture so the geometry an export sees here is the geometry it sees
there. Nothing opens a device node, and no capture outside ``tmp_path`` is read
or written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from scanner3d.capture_sequence import (
    FRAME_TEMPLATE,
    TRACKING_LOSS_STREAK,
    AlignmentFailure,
    publish_tracking_loss,
)
from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import deproject_depth_frame
from scanner3d.geometry import Point, Rigid
from scanner3d.reconstruction_export import (
    CLOUD_NAME,
    MESH_OBJ_NAME,
    MESH_PLY_NAME,
    OUTPUT_PREFIX,
    RECORD_NAME,
    STAGING_PREFIX,
    ReconstructionExport,
    export_reconstruction,
    registered_points,
)
from scanner3d.reconstruction_readiness import (
    ADVICE,
    POSE_VIEWPOINTS_INSUFFICIENT,
    SURFACE_ABANDONS_EVIDENCE,
    SURFACE_UNSUPPORTED,
    TRACKING_LOST,
    FusionEvidence,
    ReadinessFailure,
    ReconstructionReadiness,
    ReconstructionRefused,
    SurfaceAgreement,
)
from scanner3d.simulate import Sphere, render
from tests.test_reconstruction_export import (
    BUDGET,
    DECIMATION,
    DEPTH_SCALE,
    INTRINSICS,
    SPHERE_CENTRE,
    SPHERE_RADIUS,
    Capture,
    orbit,
    registration,
    sphere_capture,
    unit,
)

#: A capture of one sphere from eight points of an orbit. Shared and read-only:
#: nothing below mutates it, and every test that needs a manufactured pose
#: sequence replaces the POSES rather than the frames.
CAPTURE = sphere_capture()

#: The same capture with every pose the identity - eight frames taken from one
#: place, which is exactly what the pose and coverage gates exist to refuse.
MOTIONLESS = Capture(
    raw_frames=CAPTURE.raw_frames,
    clouds=CAPTURE.clouds,
    poses=tuple(Rigid.identity() for _ in CAPTURE.poses),
)


def stray_capture(count: int = 8) -> Capture:
    """A capture whose FRAMES hold an object its registration clouds do not.

    Not a contrived one. ``export_reconstruction`` takes the decimated clouds
    and the full-resolution frames as separate arguments, precisely because the
    window registers a thinned, filtered cloud and fuses the original bytes -
    so a per-frame filter that drops real geometry, or a registration built
    from a narrower view of the capture, produces exactly this: frames that saw
    a second object and clouds that did not.

    What follows is what makes it load-bearing. The field's extent is chosen
    from the CLOUDS, so the stray object lies outside the volume entirely and
    fuses to nothing; the surface is a faithful model of the sphere and sits
    perfectly on the registration sample. Comparing the surface against that
    sample therefore says the reconstruction is sound, while more than half of
    what the scanner actually returned is missing from it.
    """
    subject = Sphere(centre=SPHERE_CENTRE, radius=SPHERE_RADIUS)
    stray = Sphere(centre=(-45.0, 0.0, SPHERE_CENTRE[2]), radius=18.0)
    fused = render([subject, stray], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    registered = render([subject], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    conversion = deproject_depth_frame(
        registered.raw, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE, unit=unit()
    )
    cloud = tuple(camera_points(conversion))[::DECIMATION]
    poses = orbit(count)
    return Capture(
        raw_frames=tuple(fused.raw for _ in poses),
        clouds=tuple(cloud for _ in poses),
        poses=tuple(poses),
    )


#: Built once and read-only, like :data:`CAPTURE`.
STRAY = stray_capture()


def frames_dir_of(tmp_path: Path, capture: Capture, name: str = "capture") -> Path:
    """A capture folder holding the frames this export is about to be given.

    Named as :data:`scanner3d.capture_sequence.FRAME_TEMPLATE` names them, so
    the tracking authority the readiness gate consults reads this folder as the
    capture it is rather than as something it cannot parse.
    """
    directory = tmp_path / name
    directory.mkdir()
    for index, raw in enumerate(capture.raw_frames):
        (directory / FRAME_TEMPLATE.format(index=index)).write_bytes(raw)
    return directory


def contents(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory``, by relative path, with its bytes.

    Names AND bytes, because "nothing was left behind" has to mean both. A
    staging directory that was created and removed leaves no name; a frame that
    was rewritten leaves the same name and different bytes, and either would be
    this boundary failing its contract.
    """
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def export(frames_dir: Path, capture: Capture, **extra: object) -> ReconstructionExport:
    """One export of ``capture`` into ``frames_dir``, at the shared test budget."""
    return export_reconstruction(
        frames_dir,
        list(capture.clouds),
        list(capture.poses),
        raw_frames=list(capture.raw_frames),
        intrinsics=INTRINSICS,
        registration=registration(capture.poses),
        unit=unit(),
        depth_scale=DEPTH_SCALE,
        calibration_source="a synthetic camera; this measures nothing",
        voxel_budget=BUDGET,
        **extra,  # type: ignore[arg-type]
    )


def test_the_pre_fusion_gates_run_before_a_byte_of_field_is_allocated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A motionless capture is refused without any fusion happening at all.

    This is not a nicety. Fusing is the longest stage of an export and the only
    one whose cost grows with the capture - a hundred and seventy frames is a
    hundred and seventy full-resolution projections - and every second of it
    would be spent building a field that is about to be thrown away.

    Proved rather than argued: the volume constructor is replaced with one that
    raises. If the gate ran late, this test would fail with that error instead
    of the refusal, so it cannot pass by accident.
    """
    frames_dir = frames_dir_of(tmp_path, MOTIONLESS)

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fusion must not start for a capture that failed a pre-fusion gate")

    monkeypatch.setattr("scanner3d.reconstruction_export.TsdfVolume", refuse)

    with pytest.raises(ReconstructionRefused) as raised:
        export(frames_dir, MOTIONLESS)

    readiness = raised.value.readiness
    assert readiness.ready is False
    assert POSE_VIEWPOINTS_INSUFFICIENT in readiness.reasons
    # The gates that could be answered were ALL answered: a pre-fusion refusal
    # is still a complete statement about everything knowable at that boundary.
    assert readiness.surface is None
    assert len(readiness.reasons) > 1


def test_a_refused_export_leaves_the_capture_folder_exactly_as_it_found_it(
    tmp_path: Path,
) -> None:
    """No output directory, no staging directory, no half-written anything.

    The export's atomicity was already load-bearing and is unchanged: the whole
    reason the readiness gates run where they do is that at both of those points
    nothing has been created yet, so a refusal has nothing to clean up rather
    than a cleanup that has to be got right.

    Asserted over the complete recursive listing, names and bytes, rather than
    by looking for the four filenames an export writes - a partial artifact
    under some other name would be exactly as misleading.
    """
    frames_dir = frames_dir_of(tmp_path, MOTIONLESS)
    before = contents(frames_dir)

    with pytest.raises(ReconstructionRefused):
        export(frames_dir, MOTIONLESS)

    assert contents(frames_dir) == before
    assert sorted(frames_dir.iterdir()) == sorted(frames_dir.glob("frame_*.raw"))
    for name in (CLOUD_NAME, MESH_PLY_NAME, MESH_OBJ_NAME, RECORD_NAME):
        assert list(frames_dir.rglob(name)) == []
    assert list(frames_dir.glob(f"{OUTPUT_PREFIX}*")) == []
    assert list(frames_dir.glob(f"{STAGING_PREFIX}*")) == []


def test_a_post_fusion_refusal_publishes_nothing_and_keeps_the_earlier_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The surface gate fires after a field exists, and still leaves no output.

    The failure is injected rather than provoked with a deliberately displaced
    surface, because provoking it would mean fusing a defect this module has no
    business constructing - :mod:`tests.test_reconstruction_readiness` owns what
    the surface gate refuses. What is under test here is the BOUNDARY: that its
    verdict is honoured, that it is honoured before an output name is reserved,
    and that it is ADDED to whatever was already wrong rather than replacing it.

    The capture is a good one, so the only thing wrong with it is the injected
    surface verdict - which is what makes "nothing was published" attributable
    to that verdict rather than to a capture that was never going to get this
    far.
    """
    frames_dir = frames_dir_of(tmp_path, CAPTURE)
    before = contents(frames_dir)
    fused: list[int] = []

    def failing_surface(
        surface_points: Sequence[Point], evidence: FusionEvidence, **_kwargs: object
    ) -> tuple[SurfaceAgreement, tuple[ReadinessFailure, ...]]:
        # Recorded so the test can prove this ran AFTER a real fusion produced a
        # real surface, rather than being called with nothing.
        fused.append(len(surface_points))
        assert evidence.points
        return (
            SurfaceAgreement(
                measured=True,
                voxel_size=1.0,
                truncation=3.0,
                tolerance=3.0,
                sample_cell=1.5,
                surface_points=len(surface_points),
                evidence_points=len(evidence.points),
                source_frames_offered=evidence.frames_offered,
                source_frames_sampled=evidence.frames_sampled,
                source_samples_examined=evidence.samples_examined,
                source_points=len(evidence.points),
                surface_supported_fraction=0.1,
                evidence_explained_fraction=0.1,
                surface_distance_median=9.0,
                surface_distance_p95=9.0,
                evidence_distance_median=9.0,
                evidence_distance_p95=9.0,
                distance_clamp=3.0,
                detail="injected by tests.test_reconstruction_export_readiness",
            ),
            (
                ReadinessFailure(
                    reason=SURFACE_UNSUPPORTED,
                    detail="injected",
                    advice=ADVICE[SURFACE_UNSUPPORTED],
                ),
            ),
        )

    monkeypatch.setattr("scanner3d.reconstruction_export.surface_evidence", failing_surface)

    with pytest.raises(ReconstructionRefused) as raised:
        export(frames_dir, CAPTURE)

    assert fused and fused[0] > 0
    readiness = raised.value.readiness
    assert readiness.surface is not None
    assert readiness.surface.measured is True
    assert SURFACE_UNSUPPORTED in readiness.reasons
    assert contents(frames_dir) == before
    assert list(frames_dir.glob(f"{OUTPUT_PREFIX}*")) == []
    assert list(frames_dir.glob(f"{STAGING_PREFIX}*")) == []


def test_a_pre_fusion_refusal_names_every_gate_that_failed_and_not_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One refusal, several things wrong, all of them reported.

    Separated from "nothing was published" so the two fail independently: a
    boundary that published nothing while reporting only the first check that
    happened to run would still send an operator round a loop - fix the poses,
    run it again, be told about the coverage, run it again.

    Note what this test does NOT do, and cannot. A capture carrying a pre-fusion
    failure never reaches the surface gate, because it is stopped before the
    fusion that gate needs; that is the intended ordering and not an omission.
    The additivity of the two halves is proved directly on the result object in
    :mod:`tests.test_reconstruction_readiness`.
    """
    frames_dir = frames_dir_of(tmp_path, MOTIONLESS)
    seen: list[ReconstructionReadiness] = []
    from scanner3d import reconstruction_export

    # Reached through the module object rather than imported by name: this is
    # the reference the export actually calls, and patching anything else would
    # leave the real one running.
    real_evaluate = reconstruction_export.evaluate_readiness  # type: ignore[attr-defined]

    def evaluate(*args: object, **kwargs: object) -> ReconstructionReadiness:
        readiness = real_evaluate(*args, **kwargs)  # type: ignore[arg-type]
        seen.append(readiness)
        return readiness

    monkeypatch.setattr(reconstruction_export, "evaluate_readiness", evaluate)

    with pytest.raises(ReconstructionRefused) as raised:
        export(frames_dir, MOTIONLESS)

    assert seen and seen[0].ready is False
    assert raised.value.readiness.reasons == seen[0].reasons
    assert len(raised.value.readiness.reasons) >= 2
    # And the refusal an operator reads names each of them.
    text = raised.value.readiness.refusal_text()
    for reason in raised.value.readiness.reasons:
        assert reason in text


def test_a_published_export_records_the_readiness_and_the_values_actually_used(
    tmp_path: Path,
) -> None:
    """The evidence for the decision that let these files exist.

    A refused export publishes nothing, so a record that exists always carries a
    READY readiness - it is not a warning beside the geometry. What matters is
    that it is complete enough to be re-read by the strict reader and that the
    surface half states the volume's own numbers rather than a policy default,
    so a reader can tell what tolerance this particular fusion was judged at.
    """
    frames_dir = frames_dir_of(tmp_path, CAPTURE)

    result = export(frames_dir, CAPTURE)

    directory = result.directory
    record = json.loads((directory / RECORD_NAME).read_text(encoding="ascii"))
    readiness = ReconstructionReadiness.from_dict(record["readiness"])
    assert readiness.ready is True
    assert readiness.failures == ()
    assert readiness.policy.as_dict() == record["readiness"]["policy"]
    assert readiness.surface is not None
    assert readiness.surface.measured is True
    resolution = result.resolution
    assert readiness.surface.voxel_size == pytest.approx(resolution.voxel_size)
    assert readiness.surface.truncation == pytest.approx(resolution.truncation)
    assert readiness.surface.tolerance >= readiness.surface.truncation
    assert readiness.coverage.measured is True
    assert readiness.tracking.unresolved is False


def test_an_unresolved_tracking_loss_refuses_this_direct_api_with_no_metadata_offered(
    tmp_path: Path,
) -> None:
    """The bypass this gate closes, in the one shape it actually had.

    The caller below is doing nothing unusual: it hands over clouds, poses and
    raw frames it obtained somewhere else and never mentions tracking. Before
    this gate existed, that was enough to reconstruct a capture whose loss
    nothing closed - the capture reader is the sole authority on that, and this
    entry point simply never asked it.

    There is no argument to omit now. The refusal comes back with the capture
    reader's own words in it, and it is the tracking reason rather than a
    generic one, so an operator is told the actionable thing.
    """
    frames_dir = frames_dir_of(tmp_path, CAPTURE)
    publish_tracking_loss(
        frames_dir,
        last_aligned_frame_index=2,
        failures=[
            AlignmentFailure(frame_index=3 + offset, failure_type="RegistrationError", message="x")
            for offset in range(TRACKING_LOSS_STREAK)
        ],
    )
    before = contents(frames_dir)

    with pytest.raises(ReconstructionRefused) as raised:
        export(frames_dir, CAPTURE)

    readiness = raised.value.readiness
    assert TRACKING_LOST in readiness.reasons
    assert readiness.tracking.unresolved is True
    assert readiness.tracking.records_found == 1
    assert contents(frames_dir) == before
    assert list(frames_dir.glob(f"{OUTPUT_PREFIX}*")) == []


def test_a_capture_with_no_loss_record_is_still_exported(tmp_path: Path) -> None:
    """The control for the test above, so the tracking gate cannot pass by refusing all.

    The same frames, the same poses, the same call - and no ``tracking-loss.json``
    beside them. It exports.
    """
    frames_dir = frames_dir_of(tmp_path, CAPTURE)

    result = export(frames_dir, CAPTURE)

    assert result.directory.name.startswith(OUTPUT_PREFIX)
    assert list(frames_dir.glob(f"{STAGING_PREFIX}*")) == []


def test_a_defect_only_the_full_resolution_frames_can_see_does_not_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is fed the evidence the VOLUME was fused from, not a rehearsal of it.

    ``export_reconstruction`` holds two point sets, and only one of them is the
    surface's own evidence. The registered cloud is the decimated, filtered
    per-frame sample the registration was fitted to; the volume integrated the
    full-resolution frames, most of which that sample never contained. Handing
    the first to the surface gate answers a question nobody asked.

    :func:`stray_capture` is the shape that makes the difference visible: the
    frames hold a second object, the clouds do not, the field's extent comes
    from the clouds so the object fuses to nothing, and the surface that comes
    out is a flawless model of the sphere. The two halves below are the whole
    point - the export refuses it, and the registration sample the gate used to
    be given says there is nothing wrong.
    """
    frames_dir = frames_dir_of(tmp_path, STRAY, name="stray")
    before = contents(frames_dir)
    from scanner3d import reconstruction_export

    measure = reconstruction_export.surface_evidence  # type: ignore[attr-defined]
    surface_seen: list[tuple[Point, ...]] = []
    kwargs_seen: list[dict[str, object]] = []

    def record(
        surface_points: Sequence[Point], evidence: FusionEvidence, **kwargs: object
    ) -> object:
        surface_seen.append(tuple(surface_points))
        kwargs_seen.append(kwargs)
        return measure(surface_points, evidence, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reconstruction_export, "surface_evidence", record)

    with pytest.raises(ReconstructionRefused) as raised:
        export(frames_dir, STRAY)

    readiness = raised.value.readiness
    agreement = readiness.surface
    assert agreement is not None
    assert SURFACE_ABANDONS_EVIDENCE in readiness.reasons
    # And the surface itself is not the defect: it sits on the evidence
    # completely. What is wrong is everything the scanner returned that this
    # surface does not explain.
    assert agreement.surface_supported_fraction == 1.0
    assert agreement.evidence_explained_fraction < 0.6
    # The evidence really was drawn from the fused frames, and the record says
    # from how many and at what bound.
    assert agreement.source_frames_offered == len(STRAY.raw_frames)
    assert agreement.source_frames_sampled == len(STRAY.raw_frames)
    assert agreement.source_points > 0

    # The control, and the reason this test exists: the decimated registration
    # sample the gate used to be handed sees nothing wrong at all.
    sample = tuple(registered_points(STRAY.clouds, STRAY.poses))
    _, missed = measure(
        surface_seen[0],
        FusionEvidence(
            frames_offered=0, frames_sampled=0, samples_examined=len(sample), points=sample
        ),
        **kwargs_seen[0],  # type: ignore[arg-type]
    )
    assert missed == ()

    assert contents(frames_dir) == before
    assert list(frames_dir.glob(f"{OUTPUT_PREFIX}*")) == []
    assert list(frames_dir.glob(f"{STAGING_PREFIX}*")) == []
