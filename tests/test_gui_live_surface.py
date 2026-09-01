"""A surface that builds while the scan runs, and what it is allowed to claim.

The model panel used to show one picture: the registered points, coloured by
coverage. Points are the wrong picture for "what shape has this become" - a
cloud of a hollow object looks the same from inside and out, and the hole in an
underside reads as a thin patch rather than as a hole. So the capture now also
keeps an occupancy grid and marches it every few frames.

What this module is about is the seam between that and everything already there.
The grid takes EVERY accepted frame, not the sample the point picture uses, and
each frame must reach it exactly once whichever of the two paths accepted it. A
surface that fails to build must cost the points nothing. And the picture must
never quietly become the one the coverage numbers describe, because it is not
measured the same way and it is built from poses the pose graph has not yet
corrected.

Deprojection and alignment are stubbed, for the reason
:mod:`tests.test_gui_live_model` gives: the questions here are about what is
folded in and what is published, and a test that cannot decide what a frame
contains cannot ask them. The accumulator's own arithmetic is
:mod:`tests.test_mesh`'s subject.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.denoise import estimate_spacing
from scanner3d.gui import COVERAGE_EVERY
from scanner3d.mesh import MAX_GRID_CELLS, Mesh, SurfaceAccumulator
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    meshable_clouds,
    models,
    shell_cloud,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "meshable_clouds",
    "models",
    "stream_kernel",
    "stream_node",
]


@pytest.fixture
def live_meshable(meshable_clouds, alignments, models):
    """The live stubs, but with clouds dense enough for a grid to find a surface.

    :func:`~tests.gui_harness.live` hands back three points per frame, which
    proves which frames were read and nothing about a surface - three isolated
    cells straddle no node and march to no triangle. This is the same set of
    stubs over :func:`~tests.gui_harness.shell_cloud` instead.
    """

    def install(
        *, failed_alignments: set[int] | None = None, failed_models: set[int] | None = None
    ) -> tuple[list[int], list[int], list[int]]:
        return (
            meshable_clouds,
            alignments(failed_alignments or set()),
            models(failed_models or set()),
        )

    return install


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, cycling the four buffers a real driver grants."""
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


def snapshots(window: CaptureWindow) -> list[gui.ModelSnapshot]:
    return [cast(gui.ModelSnapshot, snapshot) for (snapshot,) in window.worker.coverage.emitted]


# -- the surface reaches the screen -------------------------------------------


def test_a_capture_produces_a_surface_and_the_window_draws_it(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """The whole feature, end to end through the real capture path."""
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()

    start_capture(window)

    published = snapshots(window)
    assert published
    assert all(snapshot.has_surface for snapshot in published)
    assert len(window._cloud.surfaces) == len(published)
    vertices, faces = window._cloud.surfaces[-1]
    assert len(faces) == len(published[-1].surface_faces)
    assert faces.max() < len(vertices)
    assert window._surface_toggle.enabled


def test_the_surface_says_which_cell_size_it_was_marched_at(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """An operator watching detail disappear as the grid coarsens is owed the
    reason, and the panel is where it is said."""
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()

    start_capture(window)

    assert snapshots(window)[-1].surface_voxel > 0.0
    assert "Surface:" in window._coverage_summary.text
    assert "triangles at cell" in window._coverage_summary.text


def test_the_grid_takes_every_accepted_frame_and_not_the_pictures_sample(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """The point picture samples :data:`LIVE_MODEL_CLOUDS` frames; the grid does
    not sample at all. Folding a frame in costs the frame, and what it costs to
    MARCH is bounded by the object rather than by the sweep.
    """
    stream_kernel(frames(12))
    window = CaptureWindow(frames=12)
    live_meshable()
    folded: list[int] = []
    original = SurfaceAccumulator.add

    def counting_add(self: SurfaceAccumulator, points: object) -> int:
        added = list(cast("list[tuple[float, float, float]]", points))
        folded.append(len(added))
        return original(self, added)

    SurfaceAccumulator.add = counting_add  # type: ignore[method-assign]
    try:
        start_capture(window)
    finally:
        SurfaceAccumulator.add = original  # type: ignore[method-assign]

    aligned = snapshots(window)[-1].aligned
    # One fold per accepted frame, exactly: never twice, never skipped.
    assert len(folded) == aligned
    assert all(count == len(shell_cloud()) for count in folded)


def test_a_surface_that_will_not_build_costs_the_points_nothing(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch, caplog
) -> None:
    """The part of the picture the operator can do without. A capture whose mesh
    fails must still show its points, still write its frames and still finish."""
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()

    def refuse(self: SurfaceAccumulator, **_kwargs: object) -> object:
        raise ValueError("synthetic surface failure")

    monkeypatch.setattr(SurfaceAccumulator, "surface", refuse)

    with caplog.at_level("WARNING"):
        start_capture(window)

    published = snapshots(window)
    assert published
    assert not any(snapshot.has_surface for snapshot in published)
    assert len(published[-1].points) > 0
    assert len(window.frames_on_disk()) == 6
    assert window._cloud.surface_clears == len(published)
    assert not window._surface_toggle.enabled


def test_a_grid_that_marches_to_nothing_withdraws_rather_than_leaving_a_stale_surface(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch
) -> None:
    """An empty mesh is a real answer, not a failure: samples too sparse for the
    cell edge straddle no node and there is no surface to draw. The panel is
    told so rather than left showing the last one that worked."""
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()
    monkeypatch.setattr(
        SurfaceAccumulator,
        "surface",
        lambda self, **_kwargs: Mesh(vertices=(), faces=()),
    )

    start_capture(window)

    published = snapshots(window)
    assert published
    assert not any(snapshot.has_surface for snapshot in published)
    assert window._cloud.surfaces == []
    assert window._cloud.surface_clears == len(published)
    assert not window._surface_toggle.enabled


def test_the_surface_and_the_points_are_produced_together(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """A panel showing a surface built at frame 40 beside points built at frame
    52 is describing something that never existed."""
    stream_kernel(frames(2 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=2 * COVERAGE_EVERY + 4)
    live_meshable()

    start_capture(window)

    for snapshot in snapshots(window):
        assert snapshot.has_surface
        assert len(snapshot.surface_vertices) > 0
        assert snapshot.surface_vertices.dtype == np.float32
        assert snapshot.surface_faces.dtype == np.int32


def test_the_surface_grows_with_the_scan(stream_kernel, capture_dir: Path, live_meshable) -> None:
    """Occupancy is a union, so a later surface can only have more of the object
    in it than an earlier one - which is what makes folding a frame in and then
    forgetting it correct."""
    stream_kernel(frames(3 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=3 * COVERAGE_EVERY + 4)
    live_meshable()

    start_capture(window)

    published = snapshots(window)
    assert len(published) >= 2
    counts = [len(snapshot.surface_faces) for snapshot in published]
    assert counts[-1] >= counts[0]


# -- which picture the panel opens on ------------------------------------------
#
# The owner's decision of 2026-08-27, and it is about fidelity rather than cost.
# Capture 20260827-152842 wrote 116 frames, held cell 1.981 from its first
# refresh to its last, reached 774300 cells and 66616 faces and never coarsened
# - the finest surface this preview can build - and the owner still directly
# observed it as lumpy beside dots that looked clean. An occupancy cell joins
# the surface when its neighbours are occupied, so what is marched is faceted at
# the cell size, stands as much as a cell outside the samples, and draws a small
# disagreement between two accepted poses as solid shape. So the panel opens on
# the points and the surface is an opt-in diagnostic. What must NOT change is
# the control logic underneath: an explicit choice still outlives a surface that
# goes away, and a withdrawal is still answered truthfully with the points.


def _pictures(window: CaptureWindow) -> list[tuple[bool, bool | None]]:
    """Record the control and the view at EVERY refresh, not just the last.

    "The panel opened on the points" is a claim about the first refresh a
    surface reached and about every refresh after it. The window's own state
    only ever holds the most recent one, so the question is asked where each
    picture is drawn.
    """
    seen: list[tuple[bool, bool | None]] = []
    drawn = window._show_coverage

    def record(snapshot: object) -> None:
        drawn(snapshot)
        seen.append((window._surface_toggle.checked, window._cloud.show_surface))

    window._show_coverage = record  # type: ignore[method-assign]
    return seen


def test_the_first_surface_enables_the_control_without_selecting_it(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """A surface becoming available is not the operator asking for it.

    Every refresh of this capture HAS a surface, including the first, and the
    points are the picture at every one of them. The surface still crosses to
    the view, so ticking the box is instant rather than a wait for the next
    rebuild.
    """
    stream_kernel(frames(2 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=2 * COVERAGE_EVERY + 4)
    live_meshable()
    drawn = _pictures(window)

    start_capture(window)

    published = snapshots(window)
    assert published
    assert all(snapshot.has_surface for snapshot in published)
    assert drawn == [(False, False)] * len(published)
    # Offered rather than selected: usable control, geometry already delivered.
    assert window._surface_toggle.enabled
    assert len(window._cloud.surfaces) == len(published)
    # And the panel says what the picture the operator is NOT being shown is.
    assert "Experimental occupancy preview" in window._coverage_summary.text


def test_an_explicit_surface_choice_holds_through_every_refresh(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """Points first is a default and not a veto. An operator who ticks the box
    gets the surface at that refresh and at every refresh after it, exactly as
    the accepted control logic already did."""
    stream_kernel(frames(3 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=3 * COVERAGE_EVERY + 4)
    live_meshable()
    window._toggle_surface(True)
    drawn = _pictures(window)

    start_capture(window)

    published = snapshots(window)
    assert len(published) >= 2
    assert all(snapshot.has_surface for snapshot in published)
    assert drawn == [(True, True)] * len(published)


def test_the_surface_control_offers_an_experimental_preview_and_says_so() -> None:
    """What the control CLAIMS, which is the part the operator acts on.

    Asserted against the constants rather than a widget, for the reason
    :data:`~scanner3d.gui.SURFACE_TOGGLE_TITLE` gives: this is the only place the
    wording is checked at all. It has to offer the preview as experimental, say
    the two things the owner actually saw - a faceted shell standing outside the
    dots - and keep the corrected final mesh separate. The final mesh IS a fitted
    surface now, so what the tooltip must not do has moved: it may no longer say
    the two are the same mesher, and it may not let "better" be read as
    "measured".

    Phrases are read off the UNWRAPPED text: where the tooltip's fixed line
    breaks happen to fall is not part of what it claims, and a rewording that
    moves a break must not be able to pass by moving one.
    """
    title = gui.SURFACE_TOGGLE_TITLE.lower()
    tooltip = gui.SURFACE_TOGGLE_TOOLTIP.lower()
    said = " ".join(tooltip.split())

    assert gui.SURFACE_SELECTED_BY_DEFAULT is False
    assert "experimental" in title
    assert "points" in title
    assert "experimental" in said
    assert "faceted" in said
    assert "lumpy" in said
    assert "outside" in said
    assert "separate" in said
    assert "corrected poses" in said
    # The final mesh is now fused rather than filled, so the tooltip has to keep
    # the two apart instead of calling them the same surface. What it may NOT do
    # is let the better one be read as a measurement.
    assert "signed-distance" in said
    assert "not a measurement" in said


def test_reconstruct_is_offered_as_a_different_surface_and_never_as_a_measurement() -> None:
    """Reconstruct now meshes them better too, and both texts have to say which.

    The wording this guards was written when the final mesh called the SAME
    occupancy mesher as the preview, so both texts said so and refused the
    sentence "the one to trust" that the owner had rejected. The final mesh is
    now the zero level set of a fused signed-distance field, which passes through
    the samples rather than wrapping the cells they filled - so the old claim has
    become the false one, and repeating it would understate the export by exactly
    the defect that was removed.

    What did NOT change is the refusal underneath it. A better surface is still
    not a validated one: nothing in this repository has compared any dimension of
    it to a physical reference. So "the one to trust" stays out of both texts,
    and both have to keep the live preview's own stand-off visible rather than
    letting the improvement be read backwards onto it. Both are read here because
    it was one claim written twice rather than two separate pieces of wording.
    """
    said = " ".join(gui.SURFACE_TOGGLE_TOOLTIP.lower().split())
    prose = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    readme = " ".join(prose.lower().split())

    for text in (said, readme):
        assert "the one to trust" not in text
        # The claim that became false, in the exact words both texts carried it in.
        assert "same occupancy" not in text
        assert "does not exist yet" not in text
        # What replaced it: a named method, and the preview's own limitation kept
        # separate from it rather than quietly inheriting the improvement.
        assert "signed-distance" in text
        assert "outside" in text
    assert "places the frames better" in said
    assert "not a measurement" in said


# -- sizing the grid -----------------------------------------------------------


def test_the_cell_edge_is_measured_from_the_geometry_and_not_chosen_in_advance(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch
) -> None:
    """A constant edge cannot work: what makes cells join into a surface is how
    far apart the samples are, which depends on the object's size, its distance
    and the decimation - none of which this window knows before it has a frame.
    """
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()
    measured: list[int] = []

    def spy(points: object) -> object:
        measured.append(len(cast("list[object]", points)))
        return estimate_spacing(cast("list[tuple[float, float, float]]", points))

    monkeypatch.setattr("scanner3d.gui.estimate_spacing", spy)

    start_capture(window)

    # Once, over the geometry that existed at the first refresh, and never again.
    assert len(measured) == 1
    assert measured[0] > 0
    assert snapshots(window)[-1].surface_voxel > 0.0


def test_a_grid_too_fine_for_its_samples_doubles_its_cell_edge_until_it_is_not(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch, caplog
) -> None:
    """An empty march is a correct answer to a wrong cell edge, and is
    indistinguishable on screen from a scan that has not started - so the guess
    is checked against what it produced rather than trusted.
    """
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()
    # Exactly one measured spacing to a cell, so neighbouring samples land in
    # neighbouring cells only by luck and the grid marches to nothing. This used
    # to be a hundredth of a spacing, which no longer reaches this path at all:
    # a guess that far off implies a grid of billions of cells, and the budget
    # withdraws such a surface rather than doubling its way back to one - see
    # :func:`test_an_object_that_outgrows_the_budget_keeps_its_cell_edge_and_
    # loses_its_surface`. What is left here is the case the doubling is FOR: an
    # edge that is too fine for the samples but implies a grid that could be
    # marched.
    monkeypatch.setattr(gui, "LIVE_MESH_SPACING_MULTIPLE", 1.0)

    with caplog.at_level("INFO", logger="scanner3d.gui"):
        start_capture(window)

    published = snapshots(window)
    assert published
    assert any("doubling the cell edge" in record.message for record in caplog.records)
    # And it stopped once there was a surface, rather than coarsening on for ever.
    assert published[-1].has_surface


def test_the_doubling_gives_up_rather_than_coarsening_until_anything_appears(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch
) -> None:
    """Past the bound the honest answer is that this scan has no surface yet."""
    stream_kernel(frames(6))
    window = CaptureWindow(frames=6)
    live_meshable()
    monkeypatch.setattr(gui, "LIVE_MESH_COARSEN_ATTEMPTS", 0)
    monkeypatch.setattr(gui, "LIVE_MESH_SPACING_MULTIPLE", 0.01)

    start_capture(window)

    published = snapshots(window)
    assert published
    assert not any(snapshot.has_surface for snapshot in published)
    assert not window._surface_toggle.enabled


def test_the_doubling_budget_is_spent_once_for_the_capture_and_not_once_a_refresh(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch, caplog
) -> None:
    """A scan that never produces a surface - of air, or of something too dark to
    return depth - would otherwise spend the budget again every few frames, and
    the cell edge would become a number with no relationship to anything measured.
    """
    stream_kernel(frames(4 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=4 * COVERAGE_EVERY + 4)
    live_meshable()
    monkeypatch.setattr(
        SurfaceAccumulator,
        "surface",
        lambda self, **_kwargs: Mesh(vertices=(), faces=()),
    )

    with caplog.at_level("INFO", logger="scanner3d.gui"):
        start_capture(window)

    doublings = [r for r in caplog.records if "doubling the cell edge" in r.message]
    assert len(snapshots(window)) > 1, "several refreshes, so several chances to overspend"
    assert len(doublings) == gui.LIVE_MESH_COARSEN_ATTEMPTS


# -- outgrowing the budget -----------------------------------------------------
#
# The defect this section exists for. The live grid used to be handed to
# ``coarsen_to_fit`` at every refresh, so an object that turned out bigger than
# the first frames suggested had its ONE accumulator destructively re-cut at
# twice the cell edge, and the panel went on showing the result as though
# nothing had changed. The owner watched exactly that happen: capture
# 20260827-143636 ran at voxel 2.713 over 246796 cells and then showed the same
# object at 5.425 over 39900, and reported the live mesh as a blob beside dots
# that had not changed. The occupancy mesher's surface already stands as much as
# a voxel outside its samples, so doubling the edge doubles that error.


def growing_object(monkeypatch, radius: Callable[[int], float]) -> list[int]:
    """Deprojection reduced to a shell whose SIZE is the test's parameter.

    :func:`~tests.gui_harness.meshable_clouds` hands back the same 20 mm shell
    for every frame, which can never leave the budget behind however long the
    capture runs. Everything in this section is about a scan that turns out to
    cover more than its first frames did, so the radius is what varies.

    The point count per frame does NOT vary, so a bigger shell is a sparser one
    and contributes isolated cells rather than more surface. That is deliberate
    and it is enough: what these tests ask about is the grid's EXTENT and what
    the window does when it outgrows a budget, and the surface that is there to
    be kept or withdrawn comes from the first frames either way.
    """
    read: list[int] = []

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        index = int(path.stem.split("_")[1])
        read.append(index)
        return shell_cloud(radius=radius(index), spin=0.01 * index)

    monkeypatch.setattr(gui, "_load_cloud", load)
    return read


def outgrowing(index: int) -> float:
    """20 mm while the grid is being sized, then 8 mm bigger every frame.

    The three frames the cell edge is measured from are all the same size, so
    the edge this capture chooses is the one every other test here gets. What
    follows is the defect's shape: an object that keeps turning out bigger, past
    the 250000 cells that used to be the whole budget.
    """
    return 20.0 + 8.0 * max(0, index - 2)


def far_outgrowing(index: int) -> float:
    """The same object until a surface exists, and then far too big for one."""
    if index <= 8:
        return outgrowing(index)
    return outgrowing(8) + 200.0 * (index - 8)


def marches(monkeypatch) -> list[tuple[float, int]]:
    """Every grid the capture actually marched: its cell edge, and its cells."""
    seen: list[tuple[float, int]] = []
    original = SurfaceAccumulator.surface

    def spy(self: SurfaceAccumulator, **kwargs: object) -> Mesh:
        seen.append((self.voxel_size, self.grid_cells))
        return original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SurfaceAccumulator, "surface", spy)
    return seen


def folds(monkeypatch) -> list[int]:
    """How many points each fold into the grid carried, in order."""
    seen: list[int] = []
    original = SurfaceAccumulator.add

    def spy(self: SurfaceAccumulator, points: object) -> int:
        folded = list(cast("list[tuple[float, float, float]]", points))
        seen.append(len(folded))
        return original(self, folded)

    monkeypatch.setattr(SurfaceAccumulator, "add", spy)
    return seen


def test_the_live_budget_covers_both_recorded_physical_scan_envelopes() -> None:
    """The budget is sized from the owner's two captures, not from a round number.

    Their logs record the COARSENED grids they finished on. Coarsening maps a
    cell index to ``(i + 1) // 2``, so a fine span is at most twice the coarse
    span plus one and a fine grid is therefore strictly under eight times the
    coarse one - which bounds, without re-running either scan, the grid each
    would have kept had it never been coarsened at all.
    """
    envelopes = {
        "20260827-143407, 158 frames, 97760 cells at voxel 4.944": 97_760,
        "20260827-143636, 90 frames, 73892 cells at voxel 5.425": 73_892,
    }
    for capture, coarse_cells in envelopes.items():
        assert 8 * coarse_cells < gui.LIVE_MESH_CELLS, capture
    # And it stays inside the mesh module's own ceiling, so the library refuses
    # nothing this window is willing to ask for.
    assert gui.LIVE_MESH_CELLS <= MAX_GRID_CELLS


def test_a_grid_that_crosses_the_old_boundary_keeps_the_cell_edge_it_measured(
    stream_kernel, capture_dir: Path, alignments, models, monkeypatch, caplog
) -> None:
    """The direct regression. A live grid that grows through 250000 cells is no
    longer re-cut at twice the edge; it is marched at the edge it measured."""
    stream_kernel(frames(16))
    window = CaptureWindow(frames=16)
    growing_object(monkeypatch, outgrowing)
    alignments(set())
    models(set())
    marched = marches(monkeypatch)

    with caplog.at_level("INFO", logger="scanner3d.gui"):
        start_capture(window)

    published = snapshots(window)
    assert published
    assert all(snapshot.has_surface for snapshot in published)
    # The grid really did cross the boundary that used to trigger the doubling,
    # so this is the transition the owner reported and not a smaller scan.
    assert max(cells for _voxel, cells in marched) > 250_000
    # One cell edge for every surface that was drawn, and it is the edge the
    # sizing settled on rather than any multiple of it.
    edges = {snapshot.surface_voxel for snapshot in published}
    assert len(edges) == 1
    assert edges == {marched[-1][0]}
    assert not [record for record in caplog.records if "coarsened" in record.message]


def test_an_object_that_outgrows_the_budget_keeps_its_cell_edge_and_loses_its_surface(
    stream_kernel, capture_dir: Path, alignments, models, monkeypatch, caplog
) -> None:
    """Past the budget there is no coarser answer worth showing, so none is shown.

    The points are the authoritative picture and they stay current; the surface
    is withdrawn, the control that selects it goes down with it, and the reason
    is on the screen rather than only in the log.
    """
    stream_kernel(frames(12))
    window = CaptureWindow(frames=12)
    growing_object(monkeypatch, far_outgrowing)
    alignments(set())
    models(set())
    marched = marches(monkeypatch)

    with caplog.at_level("WARNING", logger="scanner3d.gui"):
        start_capture(window)

    published = snapshots(window)
    assert len(published) >= 2
    assert published[0].has_surface
    assert not published[-1].has_surface
    # Withdrawn, NOT coarsened. The sizing doublings that precede the first
    # surface are still there - that is what settles the edge - but nothing ever
    # marched a grid coarser than the edge the surfaces were drawn at, which is
    # what the old coarsen_to_fit did every time the object grew.
    assert max(voxel for voxel, _cells in marched) == published[0].surface_voxel
    assert {snapshot.surface_voxel for snapshot in published if snapshot.has_surface} == {
        published[0].surface_voxel
    }
    # The points are still the current picture, and they are still growing.
    assert published[-1].aligned > published[0].aligned
    assert len(published[-1].points) > 0
    # Said in the operator's words, with the cell size and the budget in it, and
    # said on the panel rather than only in the log.
    note = published[-1].surface_note
    assert str(gui.LIVE_MESH_CELLS) in note
    assert f"{published[0].surface_voxel:.2f}" in note
    assert note in window._coverage_summary.text
    # And what it says about the alternative has to be true of the alternative.
    # Reconstruct reads every captured frame and rebuilds from the corrected
    # poses, but it thins each frame, drops isolated samples and derives its own
    # mesh voxel from the registered extent and its own cell budget - so this
    # message must not offer "full resolution" as the consolation for refusing a
    # picture it will not overstate.
    assert "full resolution" not in note
    assert "reads every captured frame" in note
    assert "corrected poses" in note
    assert "its own separate budget" in note
    assert any("live surface withdrawn" in record.message for record in caplog.records)
    # And once in the running report, which is where the operator reads WHEN
    # something changed rather than what is true now.
    assert [line for line in window.logged if note in line] == [f"\n{note}"]
    # Nothing stale is left on the panel, and the control no longer claims a
    # picture that is not there.
    assert window._cloud.surface_clears >= 1
    assert not window._surface_toggle.enabled
    assert not window._surface_toggle.checked
    assert window._cloud.show_surface is False


def test_a_withdrawn_surface_is_not_rebuilt_on_every_refresh(
    stream_kernel, capture_dir: Path, alignments, models, monkeypatch
) -> None:
    """An occupancy union only grows, so a grid that did not fit cannot come to
    fit. Folding and re-measuring it every few frames would spend the whole cost
    of the surface for the rest of the capture to learn the same answer."""
    stream_kernel(frames(20))
    window = CaptureWindow(frames=20)
    growing_object(monkeypatch, far_outgrowing)
    alignments(set())
    models(set())
    folded = folds(monkeypatch)
    marched = marches(monkeypatch)
    # What each published model had COST the grid by the time it was drawn, so
    # that "bounded" can be read off the sequence rather than inferred from a
    # total. Recorded on the window, which is the far end of the emission.
    spent: list[tuple[int, int]] = []
    drawn = window._show_coverage

    def record(snapshot: object) -> None:
        drawn(snapshot)
        spent.append((len(folded), len(marched)))

    window._show_coverage = record  # type: ignore[method-assign]

    start_capture(window)

    published = snapshots(window)
    withdrawn = [index for index, snapshot in enumerate(published) if not snapshot.has_surface]
    assert withdrawn, "the capture has to have outgrown the budget for this to be about anything"
    # Several refreshes AFTER it was given up on, and not one of them folded a
    # frame in or marched a grid.
    assert len(published) - withdrawn[0] >= 3
    assert len(set(spent[withdrawn[0] :])) == 1
    assert len(folded) == published[withdrawn[0]].aligned
    # The points went on being published throughout, which is the whole point of
    # withdrawing the surface rather than the model.
    assert all(len(snapshot.points) > 0 for snapshot in published[withdrawn[0] :])
    assert all(snapshot.surface_note for snapshot in published[withdrawn[0] :])


def test_a_second_capture_starts_with_a_grid_of_its_own(
    stream_kernel, capture_dir: Path, alignments, models, monkeypatch
) -> None:
    """The refusal belongs to the capture that outgrew the budget, not to the
    window. A new scan of a smaller object gets a surface, and the control that
    selects it becomes usable again - unticked, because nobody asked for it."""
    stream_kernel(frames(12))
    window = CaptureWindow(frames=12)
    growing_object(monkeypatch, far_outgrowing)
    alignments(set())
    models(set())

    start_capture(window)

    assert not snapshots(window)[-1].has_surface
    # Read out rather than asserted in place: what this test is about is the
    # state CHANGING across the two captures, and an assertion on the attribute
    # itself would narrow it to the value it had before the second one ran. The
    # ENABLED flag is what moves here, now that the panel opens on the points
    # and the box is down either way.
    after_withdrawal = window._surface_toggle.enabled
    assert not after_withdrawal

    stream_kernel(frames(12))
    window.worker.coverage.emitted.clear()
    growing_object(monkeypatch, lambda _index: 20.0)

    start_capture(window)

    published = snapshots(window)
    assert published
    assert published[-1].has_surface
    assert window._surface_toggle.enabled
    # Usable and not selected: this capture's surface is offered to an operator
    # who has never asked for one, so the points are still the picture.
    assert not window._surface_toggle.checked
    assert window._cloud.show_surface is False


def test_the_operators_choice_of_picture_outlives_the_surface_going_away(
    stream_kernel, capture_dir: Path, alignments, models, monkeypatch
) -> None:
    """A surface that goes away must not be able to change what the operator
    asked for. The control follows the screen; the choice is remembered.

    Asked for the SURFACE here. Now that the panel's own default is the points,
    a preference that agreed with the default would prove nothing about either -
    the choice that has to survive a withdrawal is the one the window would not
    have made itself.
    """
    stream_kernel(frames(12))
    window = CaptureWindow(frames=12)
    growing_object(monkeypatch, far_outgrowing)
    alignments(set())
    models(set())
    # The operator has explicitly asked for the experimental surface.
    window._toggle_surface(True)

    start_capture(window)

    # This capture outgrew the budget, so there is no surface to draw and the
    # panel says so with the points rather than with a control that claims one.
    # The control and the picture are read out together rather than asserted in
    # place, because what this test is about is those same two things having
    # different values either side of the second capture.
    after_withdrawal = (window._surface_toggle.checked, window._cloud.show_surface)
    assert not snapshots(window)[-1].has_surface
    assert after_withdrawal == (False, False)

    stream_kernel(frames(12))
    window.worker.coverage.emitted.clear()
    growing_object(monkeypatch, lambda _index: 20.0)

    start_capture(window)

    # A surface exists again - so the control is usable again - and it comes
    # back TICKED, because that is what was asked for and the withdrawal in
    # between was the screen's answer rather than the operator's.
    restored = (window._surface_toggle.checked, window._cloud.show_surface)
    assert snapshots(window)[-1].has_surface
    assert window._surface_toggle.enabled
    assert restored == (True, True)


def test_an_expensive_refresh_does_not_take_the_stream_thread_with_it(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch
) -> None:
    """A cell budget alone would spend whatever a refresh costs once every
    :data:`~scanner3d.gui.COVERAGE_EVERY` alignments, however long that is. The
    cost of the last one is measured and the next is held off until the thread
    has had :data:`~scanner3d.gui.LIVE_MODEL_DUTY` of its time back.

    The floor is what keeps that off the ordinary case, so lowering it here is
    what makes an otherwise instant synthetic refresh count as expensive.
    """
    ungated = CaptureWindow(frames=4 * COVERAGE_EVERY)
    stream_kernel(frames(4 * COVERAGE_EVERY))
    live_meshable()
    start_capture(ungated)
    freely = len(snapshots(ungated))
    assert freely >= 3, "several refreshes to hold off, or this proves nothing"

    monkeypatch.setattr(gui, "LIVE_MODEL_REFRESH_FLOOR", 0.0)
    gated = CaptureWindow(frames=4 * COVERAGE_EVERY)
    stream_kernel(frames(4 * COVERAGE_EVERY))
    start_capture(gated)

    published = snapshots(gated)
    # Held off, not broken: the capture still finished, still wrote every frame
    # and still published a model with a surface in it.
    assert 1 <= len(published) < freely
    assert published[-1].has_surface
    assert len(gated.frames_on_disk()) == 4 * COVERAGE_EVERY
