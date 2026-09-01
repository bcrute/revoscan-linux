"""The reconstruction readiness gate, proved through the REAL window path.

Why this exists
----------------
:mod:`scanner3d.reconstruction_readiness` is wired into
:func:`scanner3d.reconstruction_export.export_reconstruction`, which
:meth:`scanner3d.gui.ScannerWindow._start_reconstruct` calls. Every other test
of that window's Reconstruct action - :mod:`tests.test_gui_reconstruct_export`
- opts INTO ``tests.gui_harness.stub_readiness`` precisely because it is
testing something else and a manufactured pose sequence would otherwise be
refused before the thing under test ever ran. That stub is correct for those
tests and would be exactly the wrong tool here: this module's subject IS the
gate, so it never stubs it. Every test below drives
:meth:`~scanner3d.gui.ScannerWindow._start_reconstruct` for real, over real
readiness evaluation, and checks what the window told the operator and what
did - and did not - reach disk and the screen.

What is real and what is synthetic
-----------------------------------
The frames on disk are real, identical, 512000-byte ``Y16 `` renders of one
sphere (:func:`tests.gui_harness.sphere_raw_frame`) - the export fuses its mesh
from these bytes directly. Their DEPROJECTION into a registration cloud is
stood in for by a dense synthetic shell (:mod:`tests.gui_harness`'s
``meshable_clouds`` fixture), because what the readiness gate judges is the
POSE SEQUENCE and the placed geometry it implies, not the deprojection
arithmetic - that is proved elsewhere. The registration itself is a
manufactured pose sequence: every test names exactly what its poses are and
why the gate is right to accept or refuse them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui, pipeline
from scanner3d.geometry import Point, Rigid
from scanner3d.gui import ReconstructionOutcome
from scanner3d.pipeline import SweepReconstruction
from scanner3d.reconstruction_export import OUTPUT_PREFIX, RECORD_NAME
from scanner3d.reconstruction_readiness import ADVICE, ReconstructionRefused
from tests.gui_harness import (
    CaptureWindow,
    mesh_budget,
    meshable_clouds,
    registration,
    sphere_raw_frame,
    start_reconstruct,
)

__all__ = ["mesh_budget", "meshable_clouds", "registration"]


def capture_of(tmp_path: Path, frames: int = 4) -> Path:
    """A folder of real, identical, 512000-byte renders of one sphere.

    The same construction :mod:`tests.test_gui_reconstruct_export`'s own
    ``capture_of`` uses: the export fuses its mesh from these bytes directly,
    so they have to be real frames of the right length, while the cloud each
    one contributes to registration is still the stubbed dense shell.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def outputs(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.name.startswith(OUTPUT_PREFIX))


def motionless_registration(monkeypatch) -> None:
    """Registration collapsed to one manufactured, motionless pose sequence.

    Every pose is the identity, so the placed geometry never moves between
    frames: no distinct viewpoint, no excursion, no adjacent-frame motion, and
    every frame "looks" the same direction, so the coverage gate's view-spread
    requirement is never met either. Confirmed empirically (not merely argued)
    against this exact fixture combination before this module was written:
    :meth:`scanner3d.reconstruction_readiness.evaluate_readiness` refuses it on
    ``pose_viewpoints_insufficient``, ``pose_excursion_insufficient``,
    ``pose_motion_below_noise`` and ``coverage_unsupported`` together. This is
    the shape a real capture that never actually moved the scanner - a jammed
    turntable, a registration that silently gave up and returned its seed
    poses - would also produce, which is exactly the case this gate exists to
    catch.
    """

    def run(clouds: Sequence[Sequence[Point]], **_kwargs: object) -> SweepReconstruction:
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


def test_a_motionless_pose_sequence_is_refused_by_the_real_gate(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], mesh_budget: int
) -> None:
    """The window refuses to publish, names every reason, and draws nothing.

    Three separate claims, because a guard that merely raised without telling
    the operator anything - or that raised AFTER something had already reached
    disk or the screen - would technically "refuse" while failing the actual
    point of the gate:

    * the report the operator reads carries every failed gate's reason code
      and the advice sentence that goes with the first of them - the SAME
      words :mod:`scanner3d.reconstruction_readiness.ReconstructionReadiness.
      refusal_text` produces for the command line, read from
      :data:`scanner3d.reconstruction_readiness.ADVICE` rather than retyped
      here, so the two front ends cannot silently drift apart;
    * no ``reconstruction-*`` folder exists afterwards - a refused export must
      leave the capture directory exactly as it found it;
    * nothing was ever handed to the 3D view - no cloud, no surface - because
      a picture of a reconstruction that was refused is the one piece of
      evidence that would tell the operator the run had succeeded.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    motionless_registration(monkeypatch)
    window = CaptureWindow(frames=1)

    with pytest.raises(ReconstructionRefused) as excinfo:
        start_reconstruct(window, frames_dir, monkeypatch)

    readiness = excinfo.value.readiness
    assert not readiness.ready
    assert readiness.reasons  # at least one gate genuinely failed
    for reason in readiness.reasons:
        assert reason in window.report
    # refusal_text() names every reason but states the advice sentence only
    # for the FIRST failure - that is the one line meant to be actionable -
    # so that is the one this asserts against the shared vocabulary.
    assert ADVICE[readiness.reasons[0]] in window.report

    assert outputs(frames_dir) == []
    assert window.worker.coverage.emitted == []
    assert window._cloud.clouds == []
    assert window._cloud.surfaces == []

    # The failure dialog a live run reaches: the worker thread's own exception
    # handler turns any exception the job raises into exactly this call, with
    # exactly this text - see scanner3d.gui.Worker.run and ScannerWindow._fail.
    # Reproduced here because the test harness's own ``_run`` deliberately lets
    # the exception escape instead (see tests.gui_harness's module docstring),
    # so nothing exercises that link unless a test does it explicitly.
    monkeypatch.setattr(gui, "QMessageBox", SimpleNamespace(critical=lambda *_a: None))
    window._fail(str(excinfo.value))
    assert "FAILED" in window.report
    assert readiness.refusal_text() in window.report


def test_a_capture_that_passes_every_gate_still_exports(
    tmp_path: Path,
    monkeypatch,
    meshable_clouds: list[int],
    registration,
    mesh_budget: int,
) -> None:
    """The positive control: the guard must not pass by refusing everything.

    ``registration`` is :mod:`tests.gui_harness`'s own fixture, a real,
    non-identity pose sequence stepped widely enough to give the coverage gate
    genuine view-direction spread - it is explicitly NOT stubbed here, so this
    proves the gate that refused the motionless sequence above lets a real,
    healthy sweep through unmolested.
    """
    frames_dir = capture_of(tmp_path, frames=4)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (directory,) = outputs(frames_dir)
    outcome = window.result
    assert isinstance(outcome, ReconstructionOutcome)
    record = json.loads((directory / RECORD_NAME).read_text())
    assert record["readiness"]["ready"] is True
    assert record["readiness"]["failures"] == []
    # And the model DID reach the screen - the mirror image of the refusal
    # test above, so the same file cannot pass by breaking both directions.
    assert window.worker.coverage.emitted != []
    assert window._cloud.clouds != []
