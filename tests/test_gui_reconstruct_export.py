"""Reconstruct, from the button to the files, through the real window method.

The action used to end in memory: it registered a folder of frames, drew the
result and returned a score. Whoever pressed it got a picture and a number and
nothing they could open anywhere else, which for a four-minute scan of a real
object is most of the point missing.

So the subject of this module is the whole of what Reconstruct now does - read
the frames a capture wrote, register them, write a registered point cloud, a
mesh in two formats and a record of all three into a folder of their own, and
only THEN put the model on the screen. The ordering is not a detail: a picture
of a reconstruction whose export failed is the only evidence the operator would
have been given that the run succeeded.

Everything is offline. The frames on disk are real, identical renders of one
sphere - the export now fuses its mesh from these bytes directly, and refuses
anything shorter - but their DEPROJECTION into a registered cloud is stood in
for by a synthetic shell, and the registration is stood in for by fixed poses.
The window is :mod:`tests.gui_harness`'s stand-in rather than a Qt window.
What is real is :meth:`~scanner3d.gui.ScannerWindow._start_reconstruct` and
everything it calls into.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from scanner3d import gui, pipeline
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.geometry import Point, Rigid
from scanner3d.gui import (
    LIVE_MODEL_CLOUDS,
    LIVE_MODEL_POINTS,
    ModelSnapshot,
    ReconstructionOutcome,
)
from scanner3d.pipeline import SweepReconstruction
from scanner3d.preparation import (
    PREPARATION_VERSION,
    FusionMaskSummary,
    PreparationSummary,
    PreparedCloud,
    filter_statement,
    mask_support_band,
    preparation_unit,
    prepare_frame,
    prepare_points,
)
from scanner3d.preparation import (
    SCHEMA_VERSION as PREPARATION_SCHEMA_VERSION,
)
from scanner3d.reconstruction_export import (
    CLOUD_NAME,
    MESH_OBJ_NAME,
    MESH_PLY_NAME,
    OUTPUT_PREFIX,
    RECORD_NAME,
    STAGING_PREFIX,
    UPRIGHT_FROM_OPTICAL,
    export_reconstruction,
)
from scanner3d.simulate import FACTORY_INTRINSICS, Sphere, render
from scanner3d.simulate import Plane as SimulatedPlane
from scanner3d.support_plane import POLICY_VERSION, REASON_LITTLE_OBJECT
from tests.gui_harness import (
    CaptureWindow,
    mesh_budget,
    meshable_clouds,
    registration,
    shell_cloud,
    sphere_raw_frame,
    start_reconstruct,
    stub_readiness,
)

__all__ = ["mesh_budget", "meshable_clouds", "registration"]


def capture_of(tmp_path: Path, frames: int = 4) -> Path:
    """A folder of raw frames: real, identical, 512000-byte renders of the same
    sphere ``meshable_clouds`` stands in for the deprojection of.

    The export now fuses its mesh from these bytes directly rather than from a
    decimated cloud, and validates each one's length up front - an empty file,
    which is what this used to write back when deprojection was stood in for
    completely, is refused before a byte of field is allocated. The cloud each
    frame contributes to registration is still the stub's dense shell; only the
    bytes on disk had to become real.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def outputs(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.name.startswith(OUTPUT_PREFIX))


def ply_points(path: Path) -> list[tuple[float, float, float]]:
    """The vertices of a binary PLY, decoded by a reader that did not write it."""
    data = path.read_bytes()
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    lines = data[:end].decode("ascii").splitlines()
    (count,) = [int(line.split()[2]) for line in lines if line.startswith("element vertex ")]
    body = data[end:]
    return [struct.unpack_from("<fff", body, index * 12) for index in range(count)]


# -- the finished thing -----------------------------------------------------


def test_reconstruct_writes_a_fresh_folder_of_geometry_and_says_where_it_went(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """One press, one new folder, four files, and both places the operator looks.

    The report and the status bar name the folder, because "where is my scan"
    is the question this action is pressed to answer and a window that answers
    it only where the operator is not looking has not answered it.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (directory,) = outputs(frames_dir)
    assert directory.name == f"{OUTPUT_PREFIX}0001"
    assert sorted(path.name for path in directory.iterdir()) == [
        MESH_OBJ_NAME,
        MESH_PLY_NAME,
        RECORD_NAME,
        CLOUD_NAME,
    ]
    outcome = window.result
    assert isinstance(outcome, ReconstructionOutcome)
    export = outcome.export
    assert str(directory) in window.report
    for name in (CLOUD_NAME, MESH_PLY_NAME, MESH_OBJ_NAME, RECORD_NAME):
        assert name in window.report
    assert f"{export.point_count} registered points" in window.report
    assert f"{export.vertex_count} vertices" in window.report
    assert f"{export.face_count} faces" in window.report
    assert window._status.messages == [f"exported: {export.summary}"]
    # And the score is still reported, after the files rather than instead.
    assert "revisited viewpoints found" in window.report


def test_the_report_says_which_points_each_half_was_built_from_and_names_the_fusion(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """Two files, two different inputs, and the report has to say so.

    The registered cloud is written from the points that survived decimation and
    the isolation filter; the mesh is fused from the ORIGINAL full-length frames
    at the accepted poses. A closing line that said only "a mesh" left the
    operator to assume the surface they can measure was fitted to the cloud they
    can see - which it is not, and which matters the moment the two disagree.

    So the finished report names the construction (a TSDF fusion, not the
    occupancy shrink-wrap this replaced) and the two numbers that decide what it
    could resolve: the voxel it was fused at and the truncation. Both are read
    off the export rather than spelled out here, so the prose cannot drift from
    the field that was actually built.
    """
    frames = 4
    frames_dir = capture_of(tmp_path, frames=frames)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    outcome = window.result
    assert isinstance(outcome, ReconstructionOutcome)
    export = outcome.export
    report = window.report
    # The cloud: the thinned and filtered points.
    assert f"{export.point_count} registered points (thinned and filtered)" in report
    # The surface: named as a fusion, from every raw frame, at the resolution
    # the field was actually built at.
    assert "TSDF surface" in report
    assert f"fused from {frames} full-resolution frames" in report
    assert export.integration.frames_offered == frames
    assert export.resolution.summary in report
    assert f"truncated at {export.resolution.truncation!r}" in report
    # And the progress said the same thing while it was happening, so an
    # operator watching a long export is not told one story and reported another.
    assert (
        f"exporting the registered cloud from the filtered points, and a mesh fused from all "
        f"{frames} full-resolution frames, 0 of them with a proved support band masked out"
    ) in report
    # Nothing here claims the occupancy reconstruction that used to run.
    assert "occupancy" not in report.lower()


def test_the_report_and_the_chooser_say_which_folder_is_read_and_which_is_written(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """This action both reads a folder and creates one, which is easy to confuse.

    The dialog's own caption says "input"; the report says the frames are the
    input and that a new folder is being created inside that one. Neither is
    decoration: picking the wrong folder here is how somebody ends up
    reconstructing an empty directory and being told nothing about why.
    """
    frames_dir = capture_of(tmp_path)
    registration()
    window = CaptureWindow(frames=1)
    captions: list[str] = []

    class FakeFileDialog:
        @staticmethod
        def getExistingDirectory(_parent: object, caption: str, _start: str) -> str:  # Qt's name
            captions.append(caption)
            return str(frames_dir)

    monkeypatch.setattr(gui, "QFileDialog", FakeFileDialog)
    gui.ScannerWindow._start_reconstruct(window)  # type: ignore[arg-type]

    assert captions == [gui.RECONSTRUCT_DIALOG_CAPTION]
    assert "input" in captions[0].lower()
    assert "frames" in captions[0].lower()
    assert "INPUT" in window.report
    assert "NEW folder" in window.report
    # The running commentary is the worker's progress signal, which the real
    # window appends to the same report the lines above went into.
    progress = " ".join(str(message) for (message,) in window.worker.progress.emitted)
    assert "input frames loaded" in progress
    assert "registered points" in progress
    assert f"writing {CLOUD_NAME}" in progress


def test_the_menu_wording_says_it_reads_captured_frames_and_writes_new_files() -> None:
    """The label on the toolbar, before any dialog has been opened.

    The wording is the part of this action most able to mislead: it both READS a
    folder and CREATES one, and an operator who reads a bare "Reconstruct..."
    and is then shown a folder chooser has every reason to think they are being
    asked where the output should go. Asserted against the constants the toolbar
    is built from, because building a real QToolBar needs a display server and
    the words do not.
    """
    assert gui.RECONSTRUCT_TITLE == "Reconstruct + export..."
    tooltip = gui.RECONSTRUCT_TOOLTIP
    # Which folder is read, and that it is the input.
    assert "CAPTURED FRAMES" in tooltip
    assert "the input" in tooltip
    # And that a different, new one is written, with what in it.
    assert "NEW folder" in tooltip
    for word in ("point cloud", "PLY", "OBJ", "checksums"):
        assert word in tooltip
    # The two promises the operator cannot verify for themselves beforehand.
    assert "never overwritten" in tooltip
    assert "Cancelling the chooser writes nothing." in tooltip


def test_cancelling_the_chooser_writes_nothing_and_starts_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """Before the folder is picked there is nothing to undo, and nothing is done."""
    frames_dir = capture_of(tmp_path)
    window = CaptureWindow(frames=1)

    class Cancelled:
        @staticmethod
        def getExistingDirectory(*_a: object, **_k: object) -> str:  # Qt's name
            return ""

    monkeypatch.setattr(gui, "QFileDialog", Cancelled)
    gui.ScannerWindow._start_reconstruct(window)  # type: ignore[arg-type]

    assert window.busy == []
    assert window.logged == []
    assert sorted(path.name for path in frames_dir.iterdir()) == [
        f"frame_{index:04d}.raw" for index in range(4)
    ]


def test_a_second_reconstruction_of_one_capture_leaves_the_first_alone(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    frames_dir = capture_of(tmp_path)
    registration()
    first_window = CaptureWindow(frames=1)
    start_reconstruct(first_window, frames_dir, monkeypatch)
    (first,) = outputs(frames_dir)
    before = {path.name: path.read_bytes() for path in first.iterdir()}

    start_reconstruct(CaptureWindow(frames=1), frames_dir, monkeypatch)

    names = [path.name for path in outputs(frames_dir)]
    assert names == [f"{OUTPUT_PREFIX}0001", f"{OUTPUT_PREFIX}0002"]
    assert {path.name: path.read_bytes() for path in first.iterdir()} == before


# -- the model comes last ---------------------------------------------------


def test_the_model_reaches_the_screen_only_after_the_export_has_succeeded(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """An export that fails must not leave a picture standing in for it.

    The picture would be the only thing the operator was shown, and it would say
    the run worked. So the failure is allowed out of the job - the window turns
    it into a FAILED report - and nothing is drawn.
    """
    frames_dir = capture_of(tmp_path)
    registration()

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise OSError("a synthetic failure inside the export")

    monkeypatch.setattr(gui, "export_reconstruction", refuse)
    window = CaptureWindow(frames=1)

    with pytest.raises(OSError, match="synthetic failure"):
        start_reconstruct(window, frames_dir, monkeypatch)

    assert window.worker.coverage.emitted == []
    assert window._cloud.clouds == []
    assert outputs(frames_dir) == []


def test_stop_pressed_during_the_export_publishes_nothing_and_draws_nothing(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """Stop unwinds through the export's own cleanup.

    What must not survive is a folder under the final name holding some of the
    files: from outside the process that is indistinguishable from a scan that
    finished.
    """
    frames_dir = capture_of(tmp_path)
    registration()
    window = CaptureWindow(frames=1)

    def stop_once_the_cloud_is_written(message: str) -> None:
        if CLOUD_NAME in message:
            window.worker.cancel()

    window.worker.progress.connect(stop_once_the_cloud_is_written)

    start_reconstruct(window, frames_dir, monkeypatch)

    assert window.stopped == 1
    assert window.completed == []
    assert window.worker.coverage.emitted == []
    assert outputs(frames_dir) == []
    assert not [path for path in frames_dir.iterdir() if path.name.startswith(STAGING_PREFIX)]


# -- what is drawn is what was written --------------------------------------


def test_the_model_on_screen_is_the_geometry_that_was_exported(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """Point for point, in order, through the same upright basis.

    Two paths that each apply "the poses, then the presentation rotation" could
    disagree about the order or the sign and both still look like an object.
    They are the same arithmetic here, and this is what says so.
    """
    frames_dir = capture_of(tmp_path)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (directory,) = outputs(frames_dir)
    ((payload,),) = window.worker.coverage.emitted
    snapshot = cast(ModelSnapshot, payload)
    written = np.array(ply_points(directory / CLOUD_NAME), dtype=np.float32)
    assert np.array_equal(snapshot.points, written)
    assert window._cloud.clouds[0][0] is snapshot.points


def test_the_exported_cloud_is_each_frames_pose_applied_and_then_turned_upright(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """Against the poses the registration returned, recomputed by hand.

    The object the scanner saw is +y DOWN; the object in the file is +y up, and
    it got there by one rotation applied after the pose rather than by anything
    that also moved, mirrored or rescaled it.
    """
    frames_dir = capture_of(tmp_path, frames=3)
    seen = registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (directory,) = outputs(frames_dir)
    poses = cast(ReconstructionOutcome, window.result).registration.poses
    expected = [
        UPRIGHT_FROM_OPTICAL.apply(pose.apply(point))
        for index, pose in enumerate(poses)
        for point in shell_cloud(spin=0.01 * index)
    ]
    written = ply_points(directory / CLOUD_NAME)
    assert len(written) == len(expected)
    assert np.allclose(np.array(written), np.array(expected), atol=1e-3)
    # Every frame that was read went to the registration, and every one of those
    # to the export: no sampling anywhere along the way.
    assert [len(cloud) for cloud in seen.clouds[0]] == [len(shell_cloud())] * 3


def test_the_windows_backend_and_worker_controls_reach_the_registration(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """The two controls beside the button are the only way to steer this run.

    A window that reads them, shows them and then registers with something else
    is one where changing them appears to do nothing - and where a report naming
    a backend would be naming the wrong one.
    """
    # Four frames rather than two: two views of one object are seventeen degrees
    # apart at this fixture's step, which the coverage gate correctly refuses,
    # and this test is about the two controls rather than about that.
    frames_dir = capture_of(tmp_path, frames=4)
    seen = registration()
    window = CaptureWindow(frames=1, workers=5)
    window._backend.setCurrentText("scipy")

    start_reconstruct(window, frames_dir, monkeypatch)

    assert seen.settings[0]["backend"] == "scipy"
    assert seen.settings[0]["workers"] == 5


def test_the_export_uses_every_frame_and_every_point_not_the_live_previews_samples(
    tmp_path: Path, monkeypatch, registration, mesh_budget: int
) -> None:
    """More frames and more points than the live model would ever carry.

    ``LIVE_MODEL_CLOUDS`` and ``LIVE_MODEL_POINTS`` bound a picture that is
    rebuilt every few seconds and thrown away. This is the one the operator is
    asked to trust, and the record says how many points went into it.
    """
    frames = LIVE_MODEL_CLOUDS + 10
    frames_dir = capture_of(tmp_path, frames=frames)
    per_frame = 1200
    seen: list[int] = []

    def load(_raw: bytes, _target: int) -> PreparedCloud:
        # Frames are told apart by CALL ORDER: _prepare_frame takes bytes, not a
        # path, and the reconstruct job reads its frames in one ordered pass, so
        # the call count is the same index a filename would have given.
        index = len(seen)
        seen.append(index)
        return PreparedCloud.of_points(shell_cloud(count=per_frame, spin=0.01 * index))

    monkeypatch.setattr(gui, "_prepare_frame", load)
    registration()
    # Four placements of one flat scene that do not overlap in space: the
    # coverage gate is right that this is not one object seen from four angles,
    # and what this test asks about is the withdrawal record.
    stub_readiness(monkeypatch)
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (directory,) = outputs(frames_dir)
    total = frames * per_frame
    assert total > LIVE_MODEL_POINTS
    assert frames > LIVE_MODEL_CLOUDS
    record = json.loads((directory / RECORD_NAME).read_text())
    assert record["point_cloud"]["point_count"] == total
    assert record["input"]["frames"] == frames
    assert record["input"]["points_per_frame_target"] == gui.RECONSTRUCT_POINTS
    assert len(ply_points(directory / CLOUD_NAME)) == total


# -- what preparation did, carried into the record the operator is given ------


def bed_sheet(span: float = 100.0, steps: int = 40, depth: float = 152.0) -> list[Point]:
    """A flat surface BEHIND the sphere the raw frames hold, in the optical frame."""
    return [
        (-span / 2.0 + span * i / (steps - 1), -span / 2.0 + span * j / (steps - 1), depth)
        for i in range(steps)
        for j in range(steps)
    ]


def bed_scene() -> list[Point]:
    """The sphere standing on that surface: a frame the support filter will act on."""
    return bed_sheet() + list(shell_cloud())


def panel_scene() -> list[Point]:
    """Almost nothing but plane: a frame the support filter will withdraw on."""
    return bed_sheet() + [(0.0, 0.0, 130.0)] * 5


def test_the_exported_record_says_which_frames_kept_their_support_plane(
    tmp_path: Path, monkeypatch, registration, mesh_budget: int
) -> None:
    """A withdrawal is reported, not hidden behind the frames that succeeded.

    The interesting frame in a sweep is the one whose bed was NOT removed: it
    still contains whatever the object was standing on, and a reader comparing
    two reconstructions has no other way to find out. So the aggregate names
    applied frames, withdrawn frames and the reason each withdrew, and the
    exported record carries the sentence built from it.
    """
    frames = 4
    frames_dir = capture_of(tmp_path, frames=frames)
    before = {path.name: path.read_bytes() for path in sorted(frames_dir.glob("frame_*.raw"))}
    prepared: list[PreparedCloud] = []

    def load(_raw: bytes, target: int) -> PreparedCloud:
        # Alternating, so the record has to distinguish the two outcomes rather
        # than report one of them for the whole sweep.
        scene = bed_scene() if len(prepared) % 2 == 0 else panel_scene()
        frame = prepare_points(scene, target_points=target)
        prepared.append(frame)
        return frame

    monkeypatch.setattr(gui, "_prepare_frame", load)
    registration()
    # Four placements of one flat scene that do not overlap in space: the
    # coverage gate is right that this is not one object seen from four angles,
    # and what this test asks about is the withdrawal record.
    stub_readiness(monkeypatch)
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    summary = PreparationSummary.of(prepared)
    assert summary.support_applied_frames == 2
    assert summary.support_withdrawn_frames == 2

    (directory,) = outputs(frames_dir)
    record = json.loads((directory / RECORD_NAME).read_text())
    rejection = [
        line
        for line in record["point_cloud"]["provenance"]["limitations"]
        if line.startswith("per-frame rejection: ")
    ]
    # Rebuilt here the way the job builds it, because the sentence has to be
    # about BOTH artifacts: what left the registration clouds, and what was
    # masked out of the full-resolution frames the mesh was fused from.
    raws = [path.read_bytes() for path in sorted(frames_dir.glob("frame_*.raw"))]
    fusion = FusionMaskSummary.of(
        [
            mask_support_band(
                raw,
                frame.support,
                intrinsics=FACTORY_INTRINSICS,
                depth_scale=DEPTH_SCALE_METRES,
            )
            for raw, frame in zip(raws, prepared, strict=True)
        ]
    )
    assert rejection == [f"per-frame rejection: {filter_statement(summary, fusion)}"]
    assert "withdrew on 2 frame(s)" in rejection[0]
    assert (
        f"across {summary.support_applied_frames} of {frames} registration clouds"
    ) in rejection[0]

    # The structured aggregate beside the prose, and the two artifacts kept
    # apart in it: a reader must be able to tell what left the registration
    # cloud from what was masked out of the fused frames.
    aggregate = record["input"]["preparation"]
    assert aggregate["schema_version"] == PREPARATION_SCHEMA_VERSION
    assert aggregate["preparation_version"] == PREPARATION_VERSION
    assert aggregate["support_plane_policy_version"] == POLICY_VERSION
    assert aggregate["frames"] == frames
    cloud_block = aggregate["registered_cloud"]
    assert cloud_block["support_plane"]["applied_frames"] == 2
    assert cloud_block["support_plane"]["withdrawn_frames"] == 2
    assert cloud_block["support_plane"]["not_run_frames"] == 0
    assert cloud_block["support_plane"]["removed_points"] == summary.support_removed_points
    assert cloud_block["kept_points"] == summary.kept_points
    assert [entry["reason"] for entry in cloud_block["support_plane"]["withdrawal_reasons"]] == [
        name for name, _count in summary.withdrawal_reasons
    ]
    assert cloud_block["speckle"]["removed_points"] == summary.speckle_removed_points
    fusion_block = aggregate["full_resolution_fusion"]
    assert fusion_block["ran"] is True
    assert fusion_block["frames"] == frames
    assert fusion_block["masked_frames"] == fusion.masked_frames
    assert fusion_block["samples_masked"] == fusion.samples_masked

    # And nothing that was captured moved. The filters run on derived clouds;
    # the frames on disk are the evidence and are never rewritten.
    after = {path.name: path.read_bytes() for path in sorted(frames_dir.glob("frame_*.raw"))}
    assert after == before


def test_the_operator_is_told_what_preparation_did_while_it_is_happening(
    tmp_path: Path, monkeypatch, registration, mesh_budget: int, meshable_clouds
) -> None:
    """The progress log names the filter's running tally, at a bounded cadence."""
    frames_dir = capture_of(tmp_path, frames=4)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    said = window.report
    assert "support plane removed from" in said
    # The KEPT count, not only the removed one: an operator cannot tell a filter
    # that took the bed off a healthy cloud from one that took the cloud without
    # the number that is left.
    assert "points kept for registration" in said
    # And what the fusion input was told to drop, which is a different artifact.
    assert "full-resolution frames" in said
    # Bounded: the tally rides the read line's cadence rather than printing per
    # frame, so it appears once during the read and once at the end.
    assert said.count("support plane removed from") <= 2 + 4 // gui.PREPARATION_PROGRESS_EVERY


# -- the bed, and the artifact it used to survive into ------------------------

#: A support surface slanted steeply enough to be seen at a grazing angle - a
#: bed the object stands on and which recedes past it - with a sphere resting on
#: it. Slant matters: a plane square to the camera is entirely BEHIND the
#: object's visible cap, so a field sized to the object never reaches it and the
#: regression below could not fail. This scene puts the bed through the object's
#: own extent, which is what a real turntable capture does and what produces the
#: bed-shaped blob this work exists to remove.
BED_NORMAL: Final = (0.0, -1.0, 0.35)
BED_CONTACT: Final = (0.0, 0.0, 130.0)
BED_OBJECT_RADIUS: Final = 30.0

#: Enough voxels for the field to reach the bed as well as the object. The
#: default budget would resolve both; this one keeps the export inside a test's
#: patience while leaving the surface big enough to count vertices on.
BED_VOXEL_BUDGET: Final = 32**3


def bed_scene_frame() -> bytes:
    """One real 512000-byte frame of that scene, rendered like a capture."""
    length = math.sqrt(sum(component**2 for component in BED_NORMAL))
    unit = tuple(component / length for component in BED_NORMAL)
    offset = sum(n * c for n, c in zip(BED_NORMAL, BED_CONTACT, strict=True))
    # On the CAMERA'S side of the plane, resting on it: the contact point is on
    # the sphere and on the bed at once.
    x, y, z = (c - BED_OBJECT_RADIUS * u for c, u in zip(BED_CONTACT, unit, strict=True))
    frame = render(
        [
            SimulatedPlane(normal=BED_NORMAL, offset=offset),
            Sphere(centre=(x, y, z), radius=BED_OBJECT_RADIUS),
        ],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    )
    return frame.raw


def identity_registration(monkeypatch, frames: int) -> None:
    """Registration reduced to the identity, so the model frame IS the camera's.

    Non-identity poses are the right default everywhere else in this file - they
    catch an export that ignores them. Here they would only obscure the
    question, which is where a fitted plane ended up in the exported surface:
    with the identity, the plane the filter recorded in camera coordinates is
    the plane the mesh can be measured against, through nothing but the one
    upright rotation the export applies at the end.
    """

    def run(clouds, **_kwargs):
        return SweepReconstruction(
            poses=tuple(Rigid.identity() for _ in clouds),
            frames=len(clouds),
            revisits_found=0,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=("a synthetic registration, fitted to nothing",),
        )

    monkeypatch.setattr(pipeline, "reconstruct_sweep", run)
    assert frames > 0


def band_vertices(surface, plane, band: float) -> int:
    """How many exported vertices lie in a camera-frame plane's own band.

    The surface is written in the upright presentation frame, so the plane's
    normal is carried through the same rotation before the distance is taken.
    That rotation is proper and moves no length, which is why the offset is
    unchanged.
    """
    rotation = UPRIGHT_FROM_OPTICAL.rotation
    normal = [
        sum(rotation[row][column] * plane.normal[column] for column in range(3)) for row in range(3)
    ]
    half = band / 2.0
    return sum(
        1
        for vertex in surface.vertices
        if abs(sum(n * v for n, v in zip(normal, vertex, strict=True)) - plane.offset) <= half
    )


def test_a_proved_bed_is_absent_from_the_exported_mesh_not_only_from_the_poses(
    tmp_path: Path, monkeypatch
) -> None:
    """The correction, stated as the artifact the operator actually opens.

    Removing the support plane from the DECIMATED clouds fixes registration and
    nothing else: the mesh is fused separately, from the full-resolution frames,
    and a bed left in those is a bed in the PLY. So the same plane and the same
    band are masked out of an in-memory copy of each applied frame before it is
    fused.

    Both halves are asserted, and the control is the point of the test: the same
    clouds, the same poses and the same budget fused from the UNMASKED frames
    put hundreds of vertices in the bed's band, so the zero below is a property
    of the masking rather than of a field that never reached the bed.
    """
    frames = 4
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = bed_scene_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    before = {path.name: path.read_bytes() for path in sorted(frames_dir.glob("frame_*.raw"))}

    prepared = prepare_frame(raw, target_points=gui.RECONSTRUCT_POINTS)
    assert prepared.support is not None
    assert prepared.support.applied is True
    plane = prepared.support.plane
    assert plane is not None

    identity_registration(monkeypatch, frames)
    # Four identical frames at the identity: a manufactured pose sequence the
    # readiness gates are right to refuse, and not what this test is asking.
    stub_readiness(monkeypatch)
    monkeypatch.setattr(gui, "RECONSTRUCT_VOXEL_BUDGET", BED_VOXEL_BUDGET)
    fused_frames: list[Sequence[bytes]] = []
    real_export = export_reconstruction

    def spy(*args: object, **kwargs: object):
        fused_frames.append(list(cast(Sequence[bytes], kwargs["raw_frames"])))
        return real_export(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gui, "export_reconstruction", spy)
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    outcome = window.result
    assert isinstance(outcome, ReconstructionOutcome)
    export = outcome.export
    assert band_vertices(export.surface, plane, prepared.support.band) == 0

    # The control. Nothing else changes: same clouds, same poses, same budget,
    # the frames as they were captured.
    (offered,) = fused_frames
    assert all(frame != raw for frame in offered)
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control = export_reconstruction(
        control_dir,
        [prepared.points] * frames,
        [Rigid.identity()] * frames,
        raw_frames=[raw] * frames,
        intrinsics=FACTORY_INTRINSICS,
        registration=SweepReconstruction(
            poses=tuple(Rigid.identity() for _ in range(frames)),
            frames=frames,
            revisits_found=0,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=(),
        ),
        unit=preparation_unit(),
        depth_scale=DEPTH_SCALE_METRES,
        calibration_source="a control fusion, for this test only",
        voxel_budget=BED_VOXEL_BUDGET,
    )
    assert band_vertices(control.surface, plane, prepared.support.band) > 100
    assert export.resolution.voxel_size == control.resolution.voxel_size

    # The evidence says the same thing the geometry does.
    record = json.loads((sorted(outputs(frames_dir))[0] / RECORD_NAME).read_text())
    fusion_block = record["input"]["preparation"]["full_resolution_fusion"]
    assert fusion_block["ran"] is True
    assert fusion_block["masked_frames"] == frames
    assert fusion_block["unmasked_frames"] == 0
    assert 0 < fusion_block["samples_masked"] < fusion_block["samples_returned"]
    assert "no masked sample reached the fused field" in record["input"]["per_frame_filter"]

    # And nothing on disk moved. The masking is a derived frame; the capture is
    # the evidence and is never rewritten.
    after = {path.name: path.read_bytes() for path in sorted(frames_dir.glob("frame_*.raw"))}
    assert after == before


def test_a_withdrawn_frame_is_fused_exactly_as_it_was_captured(
    tmp_path: Path, monkeypatch, registration, mesh_budget: int
) -> None:
    """Conservative withdrawal has to reach the mesh too.

    A frame the filter could not prove a support plane in is handed to the
    fusion as the bytes that were captured - the same object, not a rebuilt
    copy - so a withdrawal costs a frame nothing at all.
    """
    frames = 4
    frames_dir = capture_of(tmp_path, frames=frames)
    raw = (frames_dir / "frame_0000.raw").read_bytes()
    prepared: list[PreparedCloud] = []

    def load(_frame: bytes, target: int) -> PreparedCloud:
        # A frame that is almost nothing but plane: the filter runs, finds its
        # plane and withdraws, which is the case under test.
        record = prepare_points(panel_scene(), target_points=target)
        prepared.append(record)
        return record

    monkeypatch.setattr(gui, "_prepare_frame", load)
    fused_frames: list[Sequence[bytes]] = []
    real_export = export_reconstruction

    def spy(*args: object, **kwargs: object):
        fused_frames.append(list(cast(Sequence[bytes], kwargs["raw_frames"])))
        return real_export(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gui, "export_reconstruction", spy)
    registration()
    # As above: four placements of one panel, judged by a gate this test is not
    # about.
    stub_readiness(monkeypatch)
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    assert all(record.support is not None and not record.support.applied for record in prepared)
    (offered,) = fused_frames
    assert [frame for frame in offered] == [raw] * frames

    record = json.loads((sorted(outputs(frames_dir))[0] / RECORD_NAME).read_text())
    fusion_block = record["input"]["preparation"]["full_resolution_fusion"]
    assert fusion_block["masked_frames"] == 0
    assert fusion_block["unmasked_frames"] == frames
    assert fusion_block["samples_masked"] == 0
    assert [entry["reason"] for entry in fusion_block["unmasked_reasons"]] == [REASON_LITTLE_OBJECT]
