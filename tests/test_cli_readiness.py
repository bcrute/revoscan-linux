"""The ``reconstruct`` command's readiness refusal, proved through ``main()``.

Why this exists
----------------
:func:`scanner3d.cli._cmd_reconstruct` runs the SAME
:func:`scanner3d.reconstruction_readiness.evaluate_readiness` gate the window
runs on its Reconstruct action, over the pose sequence
:func:`scanner3d.pipeline.reconstruct_sweep` returns, and prints
``readiness.refusal_text()`` to stderr and ``readiness.as_dict()`` into the
JSON payload before returning the reason's own exit code. Nothing about that
path is exercised by :mod:`tests.test_cli`, which predates this gate, so this
module drives :func:`scanner3d.cli.main` end to end and checks the exit code,
both output shapes, and the one file this command must NEVER write on a
refusal.

What is real and what is synthetic
-----------------------------------
Everything is offline. The frames on disk are real, identical, 512000-byte
``Y16 `` renders of one sphere - the same construction
:mod:`tests.test_gui_reconstruct_export` uses for the window's own path - read
and prepared by the REAL :func:`scanner3d.preparation.prepare_frame`. Only the
registration engine is stood in for:
:func:`scanner3d.pipeline.reconstruct_sweep` is monkeypatched to hand back a
manufactured pose sequence, exactly as the task that asked for this module
specified, so what is under test is the gate and the command's own reporting
of it rather than the registration arithmetic, which is proved elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest

from scanner3d import pipeline
from scanner3d.cli import DEPTH_SCALE_METRES, main
from scanner3d.errors import EXIT_CODES
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.pipeline import SweepReconstruction
from scanner3d.preparation import RECONSTRUCT_POINTS, prepare_frame
from scanner3d.reconstruction_readiness import (
    ADVICE,
    ReconstructionReadiness,
    ReconstructionRefused,
    evaluate_readiness,
)
from scanner3d.simulate import FACTORY_INTRINSICS, Sphere, render

#: The sphere every synthetic frame below renders, and the point every
#: manufactured pose sequence rotates about. A pose about the object's own
#: centre keeps the placed extent unchanged while still describing genuine
#: viewpoint motion - the same trick :mod:`tests.gui_harness`'s own
#: ``registration`` fixture uses for the window's path.
_CENTRE: Final = (0.0, 0.0, 120.0)
_RADIUS: Final = 20.0

#: The step a "good" manufactured sweep below turns by between frames. Wide
#: enough to clear the coverage gate's view-direction spread requirement
#: comfortably rather than sitting on the threshold - see
#: :data:`tests.gui_harness.REGISTRATION_STEP_RADIANS` for the same reasoning
#: applied to the window's own fixture.
_STEP_RADIANS: Final = 0.30


def sphere_frame() -> bytes:
    """One real, 512000-byte ``Y16 `` render of :data:`_CENTRE`'s sphere."""
    return render(
        [Sphere(centre=_CENTRE, radius=_RADIUS)],
        intrinsics=FACTORY_INTRINSICS,
        depth_scale=DEPTH_SCALE_METRES,
    ).raw


def capture_of(tmp_path: Path, frames: int = 4) -> Path:
    """A folder of real, identical frames of one sphere - a synthetic capture."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def _sweep(
    poses_of: Callable[[int], Sequence[Rigid]], limitations: tuple[str, ...] = ("synthetic",)
) -> Callable[..., SweepReconstruction]:
    def run(clouds: Sequence[Sequence[Point]], **_kwargs: object) -> SweepReconstruction:
        return SweepReconstruction(
            poses=tuple(poses_of(len(clouds))),
            frames=len(clouds),
            revisits_found=0,
            chained_score=None,
            optimised_score=None,
            optimised=False,
            converged=False,
            limitations=limitations,
        )

    return run


def motionless_registration(monkeypatch) -> None:
    """Registration collapsed to one manufactured, motionless pose sequence.

    Every pose is the identity: no distinct viewpoint, no excursion, no
    adjacent-frame motion, and every frame looks the same direction, so the
    coverage gate's view-spread requirement is never met either. Confirmed
    empirically against this exact fixture combination before this module was
    written: :func:`~scanner3d.reconstruction_readiness.evaluate_readiness`
    refuses it on ``pose_viewpoints_insufficient``,
    ``pose_excursion_insufficient``, ``pose_motion_below_noise`` and
    ``coverage_unsupported`` together.
    """
    monkeypatch.setattr(
        pipeline,
        "reconstruct_sweep",
        _sweep(lambda count: [Rigid.identity() for _ in range(count)]),
    )


def scanning_registration(monkeypatch) -> None:
    """Registration collapsed to one manufactured, genuinely diverse sweep.

    A rotation about the sphere's own centre, stepped by :data:`_STEP_RADIANS`
    per frame: real viewpoint motion, real excursion, and real view-direction
    spread, all confirmed empirically to clear every gate against this exact
    fixture combination before this module was written. The positive control:
    the guard under test must not pass by refusing every sweep it is offered.
    """
    monkeypatch.setattr(
        pipeline,
        "reconstruct_sweep",
        _sweep(
            lambda count: [
                rotation_about_axis((0.0, 1.0, 0.0), _STEP_RADIANS * index, through=_CENTRE)
                for index in range(count)
            ]
        ),
    )


def _expected_refusal(frames_dir: Path) -> ReconstructionReadiness:
    """The readiness this capture and the motionless poses actually produce.

    Recomputed rather than hardcoded, so this module keeps testing something
    real if the gates' internal thresholds ever change: it reads the same
    frames through the same :func:`~scanner3d.preparation.prepare_frame` the
    command itself calls, judges them against the same identity poses, and
    asks the same function the command asks.
    """
    paths = sorted(frames_dir.glob("frame_*.raw"))
    clouds = [
        prepare_frame(path.read_bytes(), target_points=RECONSTRUCT_POINTS).points for path in paths
    ]
    poses = [Rigid.identity() for _ in clouds]
    return evaluate_readiness(frames_dir, clouds, poses)


def test_a_refused_reconstruction_exits_with_the_readiness_status_and_names_every_reason(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-zero exit, and stderr carries every failed reason and its advice.

    The advice sentence asserted for is the one gate's own
    :meth:`~scanner3d.reconstruction_readiness.ReconstructionReadiness.refusal_text`
    actually prints - the first failure's - read from
    :data:`scanner3d.reconstruction_readiness.ADVICE` rather than retyped here,
    so this and the window's own report
    (:mod:`tests.test_gui_readiness_refusal`) cannot silently drift apart from
    the vocabulary both are supposed to share.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    motionless_registration(monkeypatch)
    expected = _expected_refusal(frames_dir)
    assert not expected.ready  # the fixture actually exercises the refusal

    code = main(["reconstruct", str(frames_dir)])

    err = capsys.readouterr().err
    assert code == EXIT_CODES[ReconstructionRefused.status]
    assert "RECONSTRUCTION REFUSED" in err
    for reason in expected.reasons:
        assert reason in err
    assert ADVICE[expected.reasons[0]] in err


def test_the_json_payload_reports_the_refusal_and_that_nothing_was_exported(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` parses, and every field the command promises is there."""
    frames_dir = capture_of(tmp_path, frames=4)
    motionless_registration(monkeypatch)
    expected = _expected_refusal(frames_dir)

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_CODES[ReconstructionRefused.status]
    assert payload["readiness"]["ready"] is False
    assert [failure["reason"] for failure in payload["readiness"]["failures"]] == list(
        expected.reasons
    )
    assert payload["command"]["exported"] is False
    assert payload["command"]["refused"] is True


def test_poses_out_is_not_written_when_the_reconstruction_is_refused(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal writes NOTHING, including the one file ``--poses-out`` asks for.

    A pose file left behind by a refused run would be evidence of a trajectory
    this command explicitly did not vouch for, sitting on disk with nothing
    marking it as such.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    motionless_registration(monkeypatch)
    poses_out = tmp_path / "poses.json"

    code = main(["reconstruct", str(frames_dir), "--poses-out", str(poses_out)])

    assert code == EXIT_CODES[ReconstructionRefused.status]
    assert not poses_out.exists()


def test_a_sweep_that_passes_every_gate_exits_zero_and_reports_ready(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive control: a real, diverse sweep is not refused by accident.

    Without this, a bug that refused every sweep unconditionally would pass
    every test above.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["readiness"]["ready"] is True
    assert payload["readiness"]["failures"] == []
    # The refusal path's own two fields are absent rather than false: a
    # successful run's payload never claims "not exported" because this
    # command writes no artifact of its own either way - it registers and
    # reports - so "exported" and "refused" are refusal-only vocabulary.
    assert "refused" not in payload["command"]
