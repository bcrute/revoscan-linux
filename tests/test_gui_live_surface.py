"""Live capture is points-only; experimental occupancy meshing is retired.

Exercise the real capture job with dense clouds that previously built a mesh.
Offline reconstruction/export and generic surface rendering remain covered by
test_gui_reconstruct_export and test_gui_preview_provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.geometry import Rigid
from scanner3d.mesh import SurfaceAccumulator
from scanner3d.preview_provenance import POINTS_PROVENANCE, SurfaceKind
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    meshable_clouds,
    models,
    start_capture,
)
from tests.test_gui_preview_provenance import published

__all__ = [
    "alignments",
    "capture_dir",
    "meshable_clouds",
    "models",
    "stream_kernel",
    "stream_node",
]


def frames(count: int) -> tuple[object, ...]:
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


def snapshots(window: CaptureWindow) -> list[gui.ModelSnapshot]:
    return [cast(gui.ModelSnapshot, value) for (value,) in window.worker.coverage.emitted]


@pytest.mark.parametrize("wanted_surface", [False, True])
def test_capture_never_constructs_or_marches_an_experimental_mesh(
    stream_kernel,
    capture_dir: Path,
    meshable_clouds,
    alignments,
    models,
    monkeypatch,
    wanted_surface,
):
    total = 4 * gui.COVERAGE_EVERY
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    alignments(set())
    models(set())
    window._toggle_surface(wanted_surface)
    constructor = Mock(side_effect=AssertionError("live mesh constructed"))
    march = Mock(side_effect=AssertionError("live mesh marched"))
    monkeypatch.setattr(SurfaceAccumulator, "__init__", constructor)
    monkeypatch.setattr(SurfaceAccumulator, "surface", march)
    monkeypatch.setattr(gui, "LIVE_MODEL_REFRESH_FLOOR", float("inf"))

    start_capture(window)

    constructor.assert_not_called()
    march.assert_not_called()
    shown = snapshots(window)
    assert len(shown) >= 3
    assert all(len(model.points) > 0 for model in shown)
    assert all(not model.has_surface and model.surface_kind is None for model in shown)
    assert all(model.surface_voxel == 0.0 and model.surface_note == "" for model in shown)
    assert len(window.frames_on_disk()) == total
    assert window._cloud.surfaces == []
    assert not window._surface_toggle.enabled
    assert not window._surface_toggle.checked
    assert not window._cloud.show_surface
    assert window._model_provenance.text == POINTS_PROVENANCE
    assert "live captures show points only" in window._surface_toggle.tooltip.lower()


def test_live_snapshot_does_not_even_prepare_triangle_arrays(monkeypatch):
    triangles = Mock(side_effect=AssertionError("live triangle conversion"))
    monkeypatch.setattr(gui, "_surface_arrays", triangles)
    cloud = ((0.0, 0.0, 100.0), (1.0, 0.0, 100.0), (0.0, 1.0, 100.0))

    model = gui._live_snapshot([cloud], [Rigid.identity()], saved=1, aligned=1)

    triangles.assert_not_called()
    assert len(model.points) == 3
    assert not model.has_surface
    assert model.surface_kind is None


def test_new_capture_clears_a_previous_reconstructed_surface(
    stream_kernel,
    capture_dir: Path,
    meshable_clouds,
    alignments,
    models,
):
    window = CaptureWindow(frames=8)
    window._show_coverage(published(surface=True, kind=SurfaceKind.RECONSTRUCTED_TSDF))
    window._toggle_surface(True)
    assert window._cloud.show_surface
    before = len(window._cloud.surfaces)
    stream_kernel(frames(8))
    alignments(set())
    models(set())

    start_capture(window)

    assert len(window._cloud.surfaces) == before
    assert all(not model.has_surface for model in snapshots(window))
    assert window._surface_evidence is None
    assert not window._surface_toggle.enabled
    assert not window._cloud.show_surface
    assert window._model_provenance.text == POINTS_PROVENANCE


def test_point_model_refreshes_remain_duty_limited(
    stream_kernel,
    capture_dir: Path,
    meshable_clouds,
    alignments,
    models,
    monkeypatch,
):
    total = 4 * gui.COVERAGE_EVERY
    # Deterministic time: each model build advances the clock by one second.
    # No test depends on how quickly a real CPU renders the fixture.
    now = [100.0]
    monkeypatch.setattr(gui.time, "monotonic", lambda: now[0])
    original = gui._live_snapshot

    def expensive(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] += 1.0
        return result

    monkeypatch.setattr(gui, "_live_snapshot", expensive)
    alignments(set())
    models(set())
    monkeypatch.setattr(gui, "LIVE_MODEL_REFRESH_FLOOR", float("inf"))
    stream_kernel(frames(total))
    ungated = CaptureWindow(frames=total)
    start_capture(ungated)
    assert len(snapshots(ungated)) >= 3

    monkeypatch.setattr(gui, "LIVE_MODEL_REFRESH_FLOOR", 0.0)
    stream_kernel(frames(total))
    gated = CaptureWindow(frames=total)
    start_capture(gated)
    assert 1 <= len(snapshots(gated)) < len(snapshots(ungated))
    assert all(not model.has_surface for model in snapshots(gated))
    assert len(gated.frames_on_disk()) == total
