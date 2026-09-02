"""What each preview claims to be, at every state the panel can actually reach.

Why this module exists
----------------------
The appliance draws three pictures of one scan and they do not have the same
standing. The depth panel is the sensor's own codes with a checker where it
returned nothing; the dots are those returns placed by calibration and by poses
the scan accepted as it ran; the opt-in shell is an occupancy grid this
application marched into triangles. Until now the only place that difference was
written down was a tooltip on one checkbox, which is a sentence nobody reads
unless they already suspect there is something to read. An operator comparing
clean dots, a lumpy shell and a sparse depth frame should not have to hover to
find out which pixels are measurements and which shape was made up.

What is asked here, and where
-----------------------------
The formatter's own refusals are asked of :mod:`scanner3d.preview_provenance`
directly, because they are pure. The depth panel is asked of the REAL
:meth:`~scanner3d.gui.ScannerWindow._build_depth_panel` on the offscreen
platform :mod:`tests.qt_offscreen` describes, because "visible without hovering"
is a question about a widget. Everything about which picture is mounted is asked
of the real window methods against :mod:`tests.gui_harness`'s stand-in, and the
renderer fallback is asked of a REAL :class:`~scanner3d.cloud_viewer.CloudViewer`
mounted into that stand-in - a swap is exactly the event a stand-in viewer would
answer by assumption.

What must not have changed
--------------------------
Nothing about the depth frame itself. This is presentation work, and the last
section proves it twice: the rendered pixels and every statistic are identical
either side of the panel drawing them, and the live and reconstruct paths still
run with all three completion adapters booby-trapped to raise if anything
reaches them.
"""

from __future__ import annotations

import importlib
import struct
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QWidget

from scanner3d import gui
from scanner3d.cloud_view import CloudCamera, CloudView
from scanner3d.cloud_viewer import CloudViewer
from scanner3d.depth_plausibility import StaticRegionTracker
from scanner3d.gui import COVERAGE_EVERY, ModelSnapshot, ScannerWindow
from scanner3d.preview_provenance import (
    LIVE_SURFACE_PROVENANCE,
    NO_MODEL_PROVENANCE,
    POINTS_PROVENANCE,
    RAW_DEPTH_PROVENANCE,
    RECONSTRUCTED_SURFACE_CONTROL_TITLE,
    RECONSTRUCTED_SURFACE_PROVENANCE,
    ModelPicture,
    ProvenanceUnusable,
    SurfaceEvidence,
    SurfaceKind,
    model_provenance,
    surface_control,
    surface_summary,
)
from scanner3d.return_health import ReturnHealthTracker
from scanner3d.tsdf import TsdfVolume
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH
from scanner3d.viewer import NO_DATA_PRIMARY, NO_DATA_SECONDARY, FrameStats, render_frame
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    FakeSpin,
    alignments,
    capture_dir,
    mesh_budget,
    meshable_clouds,
    models,
    registration,
    sphere_raw_frame,
    start_capture,
    start_reconstruct,
)
from tests.qt_offscreen import qt_app

__all__ = [
    "alignments",
    "capture_dir",
    "mesh_budget",
    "meshable_clouds",
    "models",
    "qt_app",
    "registration",
    "stream_kernel",
    "stream_node",
]


@pytest.fixture
def live_meshable(meshable_clouds, alignments, models):
    """The live stubs over clouds dense enough for a grid to find a surface.

    The same fixture :mod:`tests.test_gui_live_surface` uses, for the same
    reason: three isolated points per frame straddle no grid node and march to
    no triangle, so a capture built on them could never reach the state this
    module is about.
    """

    def install() -> tuple[list[int], list[int], list[int]]:
        return meshable_clouds, alignments(set()), models(set())

    return install


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, cycling the four buffers a real driver grants."""
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


def a_cloud(count: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, their coverage flags and their newest-frame flags."""
    points = np.array(
        [[float(index), 1.0, 100.0 + index] for index in range(count)], dtype=np.float32
    )
    thin = np.arange(count) % 2 == 0
    newest = np.zeros(count, dtype=bool)
    return points, thin, newest


#: A tetrahedron: the smallest thing with a triangle count worth printing.
SURFACE_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32
)
SURFACE_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)

#: The cell edge those triangles are said to have been marched at.
SURFACE_CELL = 1.98


def evidence(
    kind: SurfaceKind = SurfaceKind.LIVE_OCCUPANCY,
    *,
    faces: int = len(SURFACE_FACES),
    cell: float = SURFACE_CELL,
) -> SurfaceEvidence:
    """One surface's own account of itself, defaulting to the live one."""
    return SurfaceEvidence(kind=kind, faces=faces, cell=cell)


def published(
    *, surface: bool, saved: int = 6, kind: SurfaceKind = SurfaceKind.LIVE_OCCUPANCY
) -> ModelSnapshot:
    """One model as a producer publishes it, with or without a surface.

    ``kind`` is what the PRODUCER said it built, and it is stated here for the
    same reason the product states it: the two surfaces this panel can hold are
    described by two sentences that are each false about the other.
    """
    points, thin, newest = a_cloud()
    if not surface:
        return ModelSnapshot(
            points=points,
            thin=thin,
            newest=newest,
            well_covered=0.5,
            aligned=saved,
            saved=saved,
            sampled=saved,
            note="preview -",
        )
    return ModelSnapshot(
        points=points,
        thin=thin,
        newest=newest,
        well_covered=0.5,
        aligned=saved,
        saved=saved,
        sampled=saved,
        note="preview -",
        surface_vertices=SURFACE_VERTICES,
        surface_faces=SURFACE_FACES,
        surface_voxel=SURFACE_CELL,
        surface_kind=kind,
    )


def said(window: CaptureWindow) -> str:
    """What the model panel currently claims its picture is."""
    return window._model_provenance.text


def flat(text: str) -> str:
    """One line, lower-cased: where a wrap falls is not part of a claim."""
    return " ".join(text.lower().split())


# -- the formatter's own boundary ---------------------------------------------


def test_the_raw_depth_line_says_all_three_things_and_promises_no_unit() -> None:
    """The three claims the depth panel was missing, and the one it must not make.

    Read off the constant rather than a widget for the reason
    :data:`~scanner3d.gui.SURFACE_TOGGLE_TITLE` is: this is the only place the
    wording itself is checked, and the panel test below proves the constant is
    what reaches the screen.
    """
    line = " ".join(RAW_DEPTH_PROVENANCE.lower().split())

    assert "raw sensor depth codes" in line
    assert "no return" in line
    assert "no completion" in line
    # The absences that matter: no filled-in samples, and no distance claimed
    # for a code that establishes none.
    for invented in ("filling", "smoothing", "carry-over"):
        assert invented in line
    assert "not millimetres or any other distance" in line


def test_the_points_are_measured_and_are_not_called_ground_truth() -> None:
    """What the dots are, and the two qualifications that keep it honest.

    One sentence for BOTH clouds this panel can hold - the live preview and the
    completed registered reconstruction - so it says what is true of either and
    calls neither of them the other. "The live cloud" was the round-one wording,
    and it was false the moment Reconstruct published.
    """
    line = flat(model_provenance(ModelPicture.POINTS))

    assert "measured depth returns" in line
    assert "the sensor returned" in line
    # Placed, not surveyed: calibration and pose ESTIMATES, and a cloud that is
    # sampled and thinned on its way to the screen.
    assert "calibration" in line
    assert "pose estimates" in line
    assert "estimates rather than surveyed coordinates" in line
    assert "sampled and thinned for display" in line
    assert "thinned for display" in line
    for forbidden in ("ground truth", "camera coordinates", "exact"):
        assert forbidden not in line
    # And it does not describe a finished reconstruction as the live picture.
    assert "the live cloud" not in line


def test_the_live_surface_says_it_is_the_derived_occupancy_grid_it_is() -> None:
    """Derived first, then how coarse, then whose numbers the coverage line is.

    The numbers are the owner's own capture 20260827-152842: 66616 faces held at
    cell 1.981 from the first refresh to the last.
    """
    line = model_provenance(
        ModelPicture.SURFACE, evidence(SurfaceKind.LIVE_OCCUPANCY, faces=66616, cell=1.981)
    )
    lowered = flat(line)

    assert "derived occupancy surface" in lowered
    assert "not measured" in lowered
    assert "66616 triangles" in line
    assert "cell 1.98" in line
    assert "poses accepted during capture" in lowered
    # The warning the tooltip already carried, kept where it is now readable.
    assert "faceted" in lowered
    assert "lumpy" in lowered
    assert "outside the dots" in lowered
    # And coverage is still a statement about the dots.
    assert "describe the points, not this surface" in lowered
    # The live grid is not the exported fusion and may not borrow its words.
    for borrowed in ("signed-distance", "tsdf", "corrected"):
        assert borrowed not in lowered


def test_the_final_surface_says_it_is_the_fused_signed_distance_one_it_is() -> None:
    """The correction itself, in the formatter.

    Reconstruct's mesh is the zero level of a field fused from the captured
    full-resolution frames at the poses the registration CORRECTED. Round one had
    no way to say that, so it said "DERIVED occupancy surface ... from the poses
    accepted during capture" over the final result - a specific, believable and
    false account of the one surface the operator is asked to trust.
    """
    line = model_provenance(
        ModelPicture.SURFACE, evidence(SurfaceKind.RECONSTRUCTED_TSDF, faces=90210, cell=1.04)
    )
    lowered = flat(line)

    # What it IS: derived, fused, signed-distance, at corrected poses.
    assert "signed-distance" in lowered
    assert "tsdf" in lowered
    assert "derived" in lowered
    assert "not measured" in lowered
    assert "corrected" in lowered
    assert "full-resolution" in lowered
    assert "90210 triangles" in line
    assert "voxel 1.04" in line
    # Still not a measurement, which is the claim a better surface must not
    # quietly acquire.
    assert "checked against a physical reference" in lowered
    assert "describe the points, not this surface" in lowered
    # And what it is NOT: the live preview's account of itself, in any form.
    for wrong in (
        "occupancy",
        "accepted during capture",
        "poses accepted",
        "capture-time",
        "as the scan ran",
        "grid cells",
    ):
        assert wrong not in lowered


def test_the_two_surfaces_are_two_different_sentences() -> None:
    """The whole finding in one assertion: they cannot collapse into each other."""
    live = model_provenance(ModelPicture.SURFACE, evidence(SurfaceKind.LIVE_OCCUPANCY))
    final = model_provenance(ModelPicture.SURFACE, evidence(SurfaceKind.RECONSTRUCTED_TSDF))

    assert live != final
    assert live == LIVE_SURFACE_PROVENANCE.format(faces=len(SURFACE_FACES), cell=SURFACE_CELL)
    assert final == RECONSTRUCTED_SURFACE_PROVENANCE.format(
        faces=len(SURFACE_FACES), cell=SURFACE_CELL
    )


def test_every_surface_kind_has_a_line_a_summary_and_a_control_of_its_own() -> None:
    """No kind may borrow another's words, now or when a third one is added.

    Asked over the enum rather than over the two known members, so a producer
    added later without its own wording fails here instead of being described as
    whichever kind happened to be first.
    """
    kinds = list(SurfaceKind)
    lines = {model_provenance(ModelPicture.SURFACE, evidence(kind)) for kind in kinds}
    summaries = {surface_summary(evidence(kind)) for kind in kinds}
    controls = {surface_control(evidence(kind)) for kind in kinds}

    assert len(kinds) >= 2
    assert len(lines) == len(kinds)
    assert len(summaries) == len(kinds)
    assert len(controls) == len(kinds)


def test_an_empty_panel_says_so_rather_than_describing_a_picture() -> None:
    assert model_provenance(ModelPicture.NOTHING) == NO_MODEL_PROVENANCE
    assert "no measured points" in NO_MODEL_PROVENANCE
    assert "no derived surface" in NO_MODEL_PROVENANCE


def test_the_states_are_identifiers_rather_than_the_sentences_they_produce() -> None:
    """Display wording is not pipeline state.

    A member whose value WAS the sentence would invite a caller to branch on the
    words on the screen, and would make every rewording a change of state. A
    plain ``Enum`` rather than a ``StrEnum`` for the same reason: a state that is
    already a string is a state that gets compared against display text.
    """
    assert [picture.value for picture in ModelPicture] == ["nothing", "points", "surface"]
    assert [kind.value for kind in SurfaceKind] == ["live_occupancy", "reconstructed_tsdf"]
    sentences = {
        NO_MODEL_PROVENANCE,
        POINTS_PROVENANCE,
        LIVE_SURFACE_PROVENANCE,
        RECONSTRUCTED_SURFACE_PROVENANCE,
    }
    for state in [*ModelPicture, *SurfaceKind]:
        assert state.value == state.value.lower()
        assert " " not in state.value
        assert state.value not in sentences
        assert not isinstance(state, str)


def test_a_state_that_is_not_a_picture_is_refused_rather_than_defaulted() -> None:
    """There is no safe default here: any default invents a claim about pixels."""
    for wrong in ("surface", 2, None, True):
        with pytest.raises(ProvenanceUnusable):
            model_provenance(cast(ModelPicture, wrong))


def test_a_surface_may_not_be_described_without_its_numbers() -> None:
    with pytest.raises(ProvenanceUnusable):
        model_provenance(ModelPicture.SURFACE)
    with pytest.raises(ProvenanceUnusable):
        model_provenance(ModelPicture.SURFACE, cast(SurfaceEvidence, (4, 1.98)))
    # And the two consumers driven from the same record refuse it too, rather
    # than describing a surface that never said what it was.
    for wrong in (None, (4, 1.98), "the live one"):
        with pytest.raises(ProvenanceUnusable):
            surface_summary(cast(SurfaceEvidence, wrong))
    with pytest.raises(ProvenanceUnusable):
        surface_control(cast(SurfaceEvidence, "the live one"))


def test_a_points_picture_may_not_carry_a_surfaces_numbers() -> None:
    """The exact confusion this seam exists to prevent, refused at the door."""
    for picture in (ModelPicture.POINTS, ModelPicture.NOTHING):
        with pytest.raises(ProvenanceUnusable):
            model_provenance(picture, evidence(faces=4))


@pytest.mark.parametrize(
    "wrong", [None, "live_occupancy", "surface", 0, True, ModelPicture.SURFACE]
)
def test_a_surface_that_does_not_say_which_surface_it_is_is_refused(wrong) -> None:
    """No default, and no nearest match.

    A missing kind is the state round one was in permanently, and the defect it
    produced was not a vague label: it was the live preview's specific account of
    itself printed under the final reconstruction. Either available answer is
    false about the other surface, so neither may be assumed.
    """
    with pytest.raises(ProvenanceUnusable):
        SurfaceEvidence(kind=cast(SurfaceKind, wrong), faces=4, cell=1.98)


def bypassed(kind: object) -> SurfaceEvidence:
    """A record whose kind never passed validation, as only ``object`` can make one.

    Not a hypothetical: the question is whether the FORMATTERS re-check what they
    were handed, or trust a record's own construction to have done it.
    """
    broken = object.__new__(SurfaceEvidence)
    object.__setattr__(broken, "kind", kind)
    object.__setattr__(broken, "faces", 4)
    object.__setattr__(broken, "cell", 1.98)
    return broken


def test_an_internally_inconsistent_surface_is_refused_by_every_consumer() -> None:
    for kind in ("reconstructed_tsdf", None, ModelPicture.SURFACE):
        broken = bypassed(kind)
        with pytest.raises(ProvenanceUnusable):
            model_provenance(ModelPicture.SURFACE, broken)
        with pytest.raises(ProvenanceUnusable):
            surface_summary(broken)
        with pytest.raises(ProvenanceUnusable):
            surface_control(broken)


@pytest.mark.parametrize(
    "faces, cell",
    [
        (0, 1.98),  # nothing was marched, so nothing is being described
        (-4, 1.98),
        (True, 1.98),  # a bool is not a count
        (4.0, 1.98),  # nor is a float
        (4, 0.0),  # a cell edge of zero would print as an infinitely fine one
        (4, -1.98),
        (4, float("nan")),
        (4, float("inf")),
        (4, True),
        (4, "1.98"),
    ],
)
def test_a_surface_that_cannot_state_its_own_resolution_is_refused(faces, cell) -> None:
    with pytest.raises(ProvenanceUnusable):
        SurfaceEvidence(kind=SurfaceKind.LIVE_OCCUPANCY, faces=faces, cell=cell)


# -- the control and the summary, driven by the same kind ----------------------


def test_the_control_names_whichever_surface_the_panel_holds() -> None:
    """Title and tooltip come from the kind, not from what the box last drew."""
    live = surface_control(evidence(SurfaceKind.LIVE_OCCUPANCY))
    final = surface_control(evidence(SurfaceKind.RECONSTRUCTED_TSDF))

    assert "experimental" in flat(live.title)
    assert "occupancy" in flat(live.tooltip)
    assert flat(final.title) == flat(RECONSTRUCTED_SURFACE_CONTROL_TITLE)
    said_final = flat(final.tooltip)
    assert "signed-distance" in said_final
    assert "corrected" in said_final
    assert "not a measurement" in said_final
    # The claims that were false over the exported surface and are now absent.
    for wrong in ("occupancy", "accepted while", "as the scan ran"):
        assert wrong not in said_final


def test_a_panel_holding_no_surface_names_neither_producer_as_its_own() -> None:
    """A disabled box is not evidence about the mesh it does not have.

    It still explains both, because that is what somebody looking at a control
    they cannot use wants to know - but it does not claim to be showing either.
    """
    control = surface_control(None)

    assert control not in {
        surface_control(evidence(SurfaceKind.LIVE_OCCUPANCY)),
        surface_control(evidence(SurfaceKind.RECONSTRUCTED_TSDF)),
    }
    assert "no surface" in flat(control.title)
    assert "no surface to show yet" in flat(control.tooltip)


def test_the_coverage_clause_and_the_line_agree_about_which_surface_it_is() -> None:
    """Two sentences a few pixels apart, from one record.

    Round one's summary called the exported reconstruction an "Experimental
    occupancy preview" built from "poses accepted so far", directly under a line
    that was about to be corrected. Whichever is right, they cannot be allowed to
    disagree.
    """
    live = surface_summary(evidence(SurfaceKind.LIVE_OCCUPANCY))
    final = surface_summary(evidence(SurfaceKind.RECONSTRUCTED_TSDF))

    assert "occupancy" in flat(live)
    assert "poses accepted so far" in flat(live)
    assert f"{len(SURFACE_FACES)} triangles at cell {SURFACE_CELL:.2f}" in live
    assert "signed-distance" in flat(final)
    assert "corrected poses" in flat(final)
    assert f"{len(SURFACE_FACES)} triangles at voxel {SURFACE_CELL:.2f}" in final
    assert "occupancy" not in flat(final)
    assert "accepted so far" not in flat(final)


# -- the depth panel, built by the real window --------------------------------


def make_frame(returns: int, *, code: int = 5000) -> bytes:
    """A whole frame with exactly ``returns`` returning pixels along the top."""
    samples = [0] * (FRAME_WIDTH * FRAME_HEIGHT)
    for index in range(returns):
        samples[index] = code
    return struct.pack(f"<{len(samples)}H", *samples)


SPARSE_FRAME = make_frame(FRAME_WIDTH * FRAME_HEIGHT // 40)
HEALTHY_FRAME = make_frame(FRAME_WIDTH * FRAME_HEIGHT // 3)


class DepthPanel:
    """The real depth panel and the real slots that write it, and nothing else.

    Built exactly as :mod:`tests.test_gui_return_health` builds it, so a slot
    that reached for a capture, a worker or the model panel raises
    ``AttributeError`` here rather than passing quietly.
    """

    _view: QLabel
    _return_label: QLabel
    _return_advice: QLabel
    _frame_label: QLabel
    _depth_provenance: QLabel

    def __init__(self) -> None:
        self._return_health = ReturnHealthTracker()
        self._return_verdict: str | None = None
        #: The value half of frame health. Named here rather than by the
        #: builder because the real window holds it on itself, not on the panel.
        self._static_regions = StaticRegionTracker()
        self._depth_run = 0
        self.panel: QWidget = ScannerWindow._build_depth_panel(cast(ScannerWindow, self))

    def show_frame(self, raw: bytes) -> None:
        """The REAL slot, reached the way a worker connection reaches it."""
        ScannerWindow._show_frame(cast(ScannerWindow, self), raw)

    def _show_return_health(self, stats: FrameStats) -> None:
        ScannerWindow._show_return_health(cast(ScannerWindow, self), stats)

    def _show_plausibility(self, samples: tuple[int, ...]) -> None:
        ScannerWindow._show_plausibility(cast(ScannerWindow, self), samples)

    def _reset_return_health(self) -> None:
        """The REAL one, reached both directly and by the run boundary below."""
        ScannerWindow._reset_return_health(cast(ScannerWindow, self))

    def reset(self) -> None:
        self._reset_return_health()

    def end_run(self) -> None:
        ScannerWindow._end_depth_run(cast(ScannerWindow, self))


def test_the_depth_panel_states_what_it_is_showing_without_being_hovered(qt_app) -> None:
    """The correction itself: a permanent line, not a tooltip.

    A tooltip is read by whoever already suspected there was something to read.
    This has to be readable by somebody who did not.
    """
    panel = DepthPanel()

    assert panel._depth_provenance.text() == RAW_DEPTH_PROVENANCE
    assert panel._depth_provenance.toolTip() == ""
    assert not panel._depth_provenance.isHidden()
    assert panel._depth_provenance.wordWrap()
    # In the depth panel, under the picture it describes.
    assert panel._depth_provenance.parent() is panel.panel


def test_the_raw_depth_statement_survives_frames_warnings_and_run_boundaries(qt_app) -> None:
    """It describes the PANEL, so nothing about a frame may take it away.

    The return-health line beside it comes and goes with the frame; this one
    does not, because "these are raw codes and the checker is an absence" is
    exactly as true of a collapsing frame as of a healthy one - and a scan whose
    returns are disappearing is precisely when somebody starts wondering whether
    the picture is real.
    """
    panel = DepthPanel()

    panel.show_frame(HEALTHY_FRAME)
    after_healthy = panel._depth_provenance.text()
    panel.show_frame(SPARSE_FRAME)
    warned = panel._return_label.text()
    panel.reset()
    panel.end_run()

    assert warned, "this frame is sparse enough that the return-health line should be showing"
    assert after_healthy == RAW_DEPTH_PROVENANCE
    assert panel._depth_provenance.text() == RAW_DEPTH_PROVENANCE


def test_the_checker_the_line_names_is_the_checker_that_is_drawn(qt_app) -> None:
    """The sentence points at a real thing: no-return pixels ARE the two greys.

    A statement about a checker that the renderer did not draw would be worse
    than no statement, so the claim is checked against the pixels rather than
    against the wording alone.
    """
    panel = DepthPanel()
    blank = make_frame(0)

    rendered = render_frame(blank)
    panel.show_frame(blank)

    greys = {NO_DATA_PRIMARY, NO_DATA_SECONDARY}
    drawn = {tuple(pixel) for pixel in rendered.pixels}
    assert drawn == greys
    assert rendered.stats.returned == 0
    assert "no return" in panel._depth_provenance.text().lower()


def test_the_presentation_work_changed_no_pixel_and_no_statistic(qt_app) -> None:
    """Value for value, and byte for byte on the frame itself.

    The panel gained a sentence and nothing else: the same bytes produce the
    same false colour and the same numbers before and after they are drawn, the
    frame the worker handed over is not modified, and the numeric line still
    reports exactly what :func:`~scanner3d.viewer.render_frame` measured.
    """
    panel = DepthPanel()
    raw = HEALTHY_FRAME
    before = render_frame(raw)
    original = bytes(raw)

    panel.show_frame(raw)
    after = render_frame(raw)

    assert raw == original
    assert after.pixels == before.pixels
    assert after.stats == before.stats
    numbers = panel._frame_label.text()
    stats = before.stats
    assert f"returns {stats.returned} of {stats.samples}" in numbers
    assert f"depth codes {stats.min_code} to {stats.max_code}" in numbers
    assert f"median {stats.median_code}" in numbers


# -- the model panel's own widgets, built by the real window --------------------


class ModelPanel:
    """The REAL model panel and the real slots that write it, and nothing else.

    Built the way :class:`DepthPanel` is, and for the same reason: "the checkbox
    beside the line does not contradict it" is a question about widgets. The
    viewer inside it is a real
    :class:`~scanner3d.cloud_viewer.CloudViewer`, which finds no OpenGL on the
    offscreen platform and falls back to the software renderer exactly as it
    would on a machine without a card.
    """

    _capturing = False
    _pause: object = None
    _lost_tracking = False
    #: Everything :meth:`~scanner3d.gui.ScannerWindow._build_model_panel` creates
    #: that the assertions below read, declared for the reason
    #: :class:`DepthPanel`'s are: these are the real widgets, and naming them
    #: keeps the stand-in answerable to the same static check the product is.
    _cloud: CloudViewer
    _surface_evidence: SurfaceEvidence | None
    _surface_wanted: bool
    _surface_toggle: QCheckBox
    _model_provenance: QLabel
    _coverage_summary: QLabel

    def __init__(self) -> None:
        self._model_refreshes = 0
        self._model_refreshed_at: int | None = None
        self._aligned_frames = 0
        self.panel: QWidget = ScannerWindow._build_model_panel(cast(ScannerWindow, self))

    def _log(self, message: str) -> None:
        return None

    def _mounted_picture(self) -> ModelPicture:
        return ScannerWindow._mounted_picture(cast(ScannerWindow, self))

    def _refresh_model_provenance(self) -> None:
        ScannerWindow._refresh_model_provenance(cast(ScannerWindow, self))

    def _show_viewer_backend(self, description: str) -> None:
        ScannerWindow._show_viewer_backend(cast(ScannerWindow, self), description)

    def _toggle_surface(self, on: bool) -> None:
        ScannerWindow._toggle_surface(cast(ScannerWindow, self), on)

    def _show_surface_control(self, has_surface: bool) -> None:
        ScannerWindow._show_surface_control(cast(ScannerWindow, self), has_surface)

    def _show_coverage(self, snapshot: ModelSnapshot) -> None:
        ScannerWindow._show_coverage(cast(ScannerWindow, self), snapshot)

    def tick(self) -> None:
        """Ask for the surface the way an operator does: by clicking the box."""
        self._surface_toggle.setChecked(True)


def test_the_built_control_names_no_surface_before_the_panel_holds_one(qt_app) -> None:
    """A disabled box over an empty panel claims neither producer.

    Naming the live preview here would be the same guess the provenance line
    refuses - and it is the guess that made round one's control describe the
    occupancy grid while the panel was drawing the exported reconstruction.
    """
    panel = ModelPanel()

    assert panel._surface_toggle.text() == surface_control(None).title
    assert panel._surface_toggle.toolTip() == surface_control(None).tooltip
    assert not panel._surface_toggle.isEnabled()
    assert not panel._surface_toggle.isChecked()
    assert panel._model_provenance.text() == NO_MODEL_PROVENANCE


def test_the_real_control_line_and_summary_all_follow_the_surface_that_arrives(
    qt_app,
) -> None:
    """The three visible claims about one mesh, on the real widgets, both ways.

    Round one corrected only the line. This asserts what an operator can actually
    see beside it: the checkbox's own title and tooltip, and the coverage summary
    under the picture. The reconstructed surface arrives first because that is
    the one round one mislabelled, and the live one follows it to prove the
    control goes back rather than latching.

    Retitling a real ``QCheckBox`` must also not look like the operator moving
    it, which is why the tick state is asserted either side of the arrival.
    """
    panel = ModelPanel()

    panel._show_coverage(published(surface=True, kind=SurfaceKind.RECONSTRUCTED_TSDF))
    renamed_without_being_ticked = (
        panel._surface_toggle.isChecked(),
        panel._surface_wanted,
    )
    panel.tick()
    final_line = flat(panel._model_provenance.text())
    final_title = panel._surface_toggle.text()
    final_tooltip = flat(panel._surface_toggle.toolTip())
    final_summary = flat(panel._coverage_summary.text())

    panel._show_coverage(published(surface=True, kind=SurfaceKind.LIVE_OCCUPANCY))
    live_line = flat(panel._model_provenance.text())
    live_title = flat(panel._surface_toggle.text())
    live_tooltip = flat(panel._surface_toggle.toolTip())
    live_summary = flat(panel._coverage_summary.text())

    assert renamed_without_being_ticked == (False, False)
    # The exported surface: one account, in three places, and no occupancy.
    assert panel._mounted_picture() is ModelPicture.SURFACE
    assert final_title == RECONSTRUCTED_SURFACE_CONTROL_TITLE
    for shown in (final_line, final_tooltip, final_summary):
        assert "signed-distance" in shown
        assert "occupancy" not in shown
    assert "corrected" in final_line
    assert "corrected" in final_tooltip
    assert "corrected poses" in final_summary
    # And then the live grid, described as itself again rather than latching.
    for shown in (live_line, live_title, live_summary):
        assert "signed-distance" not in shown
    assert "derived occupancy surface" in live_line
    assert "experimental occupancy preview" in live_summary
    assert "experimental" in live_title
    # The live tooltip does name the fused surface - it is where the operator is
    # told the two are different - so what it must say is which one is HERE.
    assert "what is drawn here is still this occupancy grid" in live_tooltip


# -- the model panel, through the real window ---------------------------------


def test_an_empty_model_panel_claims_nothing() -> None:
    """Before the first model there is no picture to describe, and it says so."""
    window = CaptureWindow(frames=1)

    ScannerWindow._refresh_model_provenance(cast(ScannerWindow, window))

    assert window._mounted_picture() is ModelPicture.NOTHING
    assert said(window) == NO_MODEL_PROVENANCE


def test_the_first_points_snapshot_is_described_as_measurement() -> None:
    window = CaptureWindow(frames=1)

    window._show_coverage(published(surface=False))

    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE


def test_a_surface_arriving_enables_its_control_without_relabelling_the_picture() -> None:
    """Availability is not selection, and the label follows the picture.

    The surface really is delivered to the viewer - ticking the box is instant -
    but the dots are what is drawn, so the dots are what is claimed.
    """
    window = CaptureWindow(frames=1)

    window._show_coverage(published(surface=True))

    assert window._surface_toggle.enabled
    assert not window._surface_toggle.checked
    assert window._cloud.has_surface
    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE
    assert "triangles" not in said(window)


def test_choosing_the_surface_moves_the_picture_and_the_claim_together() -> None:
    """One call changes both, so there is no state in between to be read."""
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=True))

    window._toggle_surface(True)

    assert window._mounted_picture() is ModelPicture.SURFACE
    assert said(window) == model_provenance(ModelPicture.SURFACE, evidence())
    assert f"{len(SURFACE_FACES)} triangles" in said(window)
    assert f"cell {SURFACE_CELL:.2f}" in said(window)
    assert "NOT measured" in said(window)


def test_toggling_back_returns_the_claim_to_the_dots() -> None:
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)

    window._toggle_surface(False)

    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE


def test_every_refresh_while_the_surface_shows_keeps_the_surfaces_own_numbers() -> None:
    """A count that is one refresh behind its mesh describes nothing that exists."""
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)
    bigger = np.vstack([SURFACE_FACES, SURFACE_FACES])
    points, thin, newest = a_cloud()

    window._show_coverage(
        ModelSnapshot(
            points=points,
            thin=thin,
            newest=newest,
            well_covered=0.5,
            aligned=9,
            saved=9,
            sampled=9,
            note="preview -",
            surface_vertices=SURFACE_VERTICES,
            surface_faces=bigger,
            surface_voxel=SURFACE_CELL,
            surface_kind=SurfaceKind.LIVE_OCCUPANCY,
        )
    )

    assert f"{len(bigger)} triangles" in said(window)


def test_a_withdrawn_surface_takes_its_claim_with_it_and_the_preference_stays() -> None:
    """The disagreement the label must resolve towards the SCREEN.

    The operator still wants the surface. There is no surface to draw. So the
    picture is the dots and the sentence is about the dots - while the choice
    itself is untouched, ready for the next surface.
    """
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)

    window._show_coverage(published(surface=False))

    assert window._surface_wanted is True
    assert not window._surface_toggle.enabled
    assert not window._surface_toggle.checked
    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE
    assert "triangles" not in said(window)


def test_a_restored_surface_brings_the_claim_back_with_the_picture() -> None:
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)
    window._show_coverage(published(surface=False))

    window._show_coverage(published(surface=True))

    assert window._surface_toggle.checked
    assert window._mounted_picture() is ModelPicture.SURFACE
    assert "DERIVED occupancy surface" in said(window)


def test_a_ticked_box_over_a_scan_with_no_surface_still_says_measured_points() -> None:
    """Checkbox and screen made to disagree deliberately, in the other order.

    Here the preference is set BEFORE any model arrives, so the box has been
    ticked for a picture that has never existed. What the panel says has to come
    from the viewer either way.
    """
    window = CaptureWindow(frames=1)

    window._toggle_surface(True)
    window._show_coverage(published(surface=False))

    assert window._surface_wanted is True
    assert window._cloud.show_surface is False
    assert said(window) == POINTS_PROVENANCE


# -- across runs ---------------------------------------------------------------


def test_a_new_scan_does_not_open_on_the_last_scans_derived_shape(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """A finished capture's triangle count is not evidence about the next one.

    The first capture ends with the surface selected and claimed. The second one
    is started with too few frames to publish anything at all, which is exactly
    the window in which a stale label would be visible - and the panel has
    already fallen back to the dots by then, with the operator's choice kept for
    the moment this scan has a surface of its own.
    """
    stream_kernel(frames(3 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=3 * COVERAGE_EVERY + 4)
    live_meshable()
    window._toggle_surface(True)
    start_capture(window)
    ended_showing = said(window)

    stream_kernel(frames(2))
    window.worker.coverage.emitted.clear()
    window._frames_spin = FakeSpin(2)
    start_capture(window)

    assert "DERIVED occupancy surface" in ended_showing
    assert not window.worker.coverage.emitted, "this run is too short to publish a model"
    assert window._surface_wanted is True
    assert not window._surface_toggle.enabled
    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE
    assert "triangles" not in said(window)


# -- which surface, end to end -------------------------------------------------


def emitted(window: CaptureWindow) -> list[ModelSnapshot]:
    """Every model this window's producer actually published, in order."""
    return [cast(ModelSnapshot, args[0]) for args in window.worker.coverage.emitted]


def sphere_frames(root: Path, count: int = 4) -> Path:
    """A folder of real 512000-byte frames for Reconstruct to read."""
    frames_dir = root / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(count):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def test_the_live_capture_surface_is_described_as_the_occupancy_grid_it_is(
    stream_kernel, capture_dir: Path, live_meshable
) -> None:
    """The live half of the distinction, through the real capture that builds it.

    Not a hand-made snapshot: the surface here is the one
    :class:`~scanner3d.tsdf_surface.SurfaceAccumulator` marched from the cells
    the poses accepted DURING this capture filled, published by the real
    producer, and the kind travels with it.
    """
    stream_kernel(frames(3 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=3 * COVERAGE_EVERY + 4)
    live_meshable()

    start_capture(window)
    window._toggle_surface(True)

    published_models = emitted(window)
    assert published_models, "this capture should have published at least one model"
    assert all(
        model.surface_kind is SurfaceKind.LIVE_OCCUPANCY
        for model in published_models
        if model.has_surface
    )
    assert window._mounted_picture() is ModelPicture.SURFACE
    line = flat(said(window))
    assert "derived occupancy surface" in line
    assert "not measured" in line
    assert "poses accepted during capture" in line
    # The exported reconstruction's account, which is false about this mesh.
    for wrong in ("signed-distance", "tsdf", "corrected"):
        assert wrong not in line
    # And the two adjacent claims about the same mesh agree with it.
    assert "Experimental occupancy preview" in window._coverage_summary.text
    assert "signed-distance" not in flat(window._coverage_summary.text)
    assert "experimental" in flat(window._surface_toggle.text)
    assert "occupancy" in flat(window._surface_toggle.tooltip)


def test_the_reconstructed_surface_is_described_as_the_fused_one_that_was_exported(
    tmp_path: Path, monkeypatch, meshable_clouds, registration, mesh_budget: int
) -> None:
    """The finding itself, end to end, through the real Reconstruct.

    Codex reproduced round one's defect on exactly this path: the surface
    ``_start_reconstruct`` publishes is ``export.surface`` - the mesh that was
    just written to the capture folder, fused from the full-resolution frames at
    the registration's corrected poses - and the panel described it as "DERIVED
    occupancy surface ... from the poses accepted during capture". Nothing here
    is constructed by the test: the frames are read, registered, fused and
    exported by the product, and the surface that is then SELECTED is the one the
    export kept.
    """
    frames_dir = sphere_frames(tmp_path)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)
    window._toggle_surface(True)

    (model,) = emitted(window)
    assert model.has_surface, "Reconstruct should have published its exported surface"
    assert model.surface_kind is SurfaceKind.RECONSTRUCTED_TSDF
    assert window._mounted_picture() is ModelPicture.SURFACE
    line = said(window)
    lowered = flat(line)
    # What it is: fused, signed-distance, at corrected poses, and still derived.
    assert "signed-distance" in lowered
    assert "tsdf" in lowered
    assert "corrected" in lowered
    assert "full-resolution" in lowered
    assert "derived" in lowered
    assert "not measured" in lowered
    # Its OWN numbers, from the export rather than from the live grid.
    assert f"{len(model.surface_faces)} triangles" in line
    assert f"voxel {model.surface_voxel:.2f}" in line
    # And none of the live preview's claims, which is the correction.
    for wrong in ("occupancy", "accepted during capture", "capture-time", "as the scan ran"):
        assert wrong not in lowered
    # The coverage summary beside it, which said "Experimental occupancy preview"
    # and "poses accepted so far" over this very surface in round one.
    summary = flat(window._coverage_summary.text)
    assert "signed-distance" in summary
    assert "corrected poses" in summary
    assert "occupancy" not in summary
    assert "accepted so far" not in summary
    # And the control that is drawing it, which claimed the occupancy grid.
    assert flat(window._surface_toggle.text) == flat(RECONSTRUCTED_SURFACE_CONTROL_TITLE)
    tooltip = flat(window._surface_toggle.tooltip)
    assert "signed-distance" in tooltip
    assert "corrected" in tooltip
    assert "occupancy" not in tooltip


def test_the_surface_kind_is_taken_from_the_producer_and_not_read_off_the_snapshot() -> None:
    """Everything that LOOKS like a reconstruction, with the live kind, and back.

    A registered model is the one that carries ``note='registered -'`` and
    ``reframe=True``, and those are exactly the fields a shortcut would reach
    for. They are given here to a live occupancy surface, and the live producer's
    own note is given to a fused one; the sentence follows the kind both times.
    """
    points, thin, newest = a_cloud()

    def snapshot(kind: SurfaceKind, *, note: str, reframe: bool) -> ModelSnapshot:
        return ModelSnapshot(
            points=points,
            thin=thin,
            newest=newest,
            well_covered=0.5,
            aligned=4,
            saved=4,
            sampled=4,
            note=note,
            reframe=reframe,
            surface_vertices=SURFACE_VERTICES,
            surface_faces=SURFACE_FACES,
            surface_voxel=SURFACE_CELL,
            surface_kind=kind,
        )

    window = CaptureWindow(frames=1)
    window._toggle_surface(True)

    window._show_coverage(snapshot(SurfaceKind.LIVE_OCCUPANCY, note="registered -", reframe=True))
    dressed_as_registered = said(window)
    window._show_coverage(
        snapshot(
            SurfaceKind.RECONSTRUCTED_TSDF, note="preview after 8 aligned frames -", reframe=False
        )
    )
    dressed_as_live = said(window)

    assert "occupancy" in flat(dressed_as_registered)
    assert "signed-distance" not in flat(dressed_as_registered)
    assert "signed-distance" in flat(dressed_as_live)
    assert "occupancy" not in flat(dressed_as_live)


def test_a_published_surface_that_says_nothing_about_itself_is_refused_not_drawn() -> None:
    """A missing kind stops the picture rather than picking one of the two.

    Loudly, and before the mesh reaches the viewer: the panel keeps the picture
    and the sentence it already had, so there is no moment at which an
    undescribable surface is on the screen under a guess.
    """
    window = CaptureWindow(frames=1)
    window._show_coverage(published(surface=False))
    points, thin, newest = a_cloud()
    nameless = ModelSnapshot(
        points=points,
        thin=thin,
        newest=newest,
        well_covered=0.5,
        aligned=6,
        saved=6,
        sampled=6,
        note="preview -",
        surface_vertices=SURFACE_VERTICES,
        surface_faces=SURFACE_FACES,
        surface_voxel=SURFACE_CELL,
    )

    with pytest.raises(ProvenanceUnusable):
        window._show_coverage(nameless)

    assert nameless.surface_kind is None
    assert not window._cloud.has_surface
    assert window._surface_evidence is None
    assert window._mounted_picture() is ModelPicture.POINTS
    assert said(window) == POINTS_PROVENANCE


# -- and across a renderer swap ------------------------------------------------


class FakeGLView(CloudCamera, QWidget):
    """A backend shaped exactly like the real GPU one, and as controllable.

    The same two signals and the same shared camera, with no driver behind it:
    what the WINDOW says after a swap is the question, and a real context would
    answer it only on machines that have one.
    """

    ready = Signal(str)
    unavailable = Signal(str)

    def __init__(self) -> None:
        QWidget.__init__(self)
        self._init_camera()


class Recording:
    """A label that keeps every sentence it was given, not just the last.

    "The replacement was not relabelled" is a claim about what was WRITTEN, and
    a stand-in holding only the final text cannot tell a line that never moved
    from one that was wrongly rewritten and then corrected.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.texts: list[str] = []

    def setText(self, text: str) -> None:  # Qt's name
        self.text = text
        self.texts.append(text)


def viewing(window: CaptureWindow) -> CloudViewer:
    """Mount a REAL viewer, and the real backend reporting, into the stand-in."""
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    window._cloud = viewer  # type: ignore[assignment]
    window._viewer_backend = Recording()  # type: ignore[attr-defined]
    window._model_provenance = Recording(said(window))  # type: ignore[assignment]
    viewer.backend_changed.connect(
        lambda description: ScannerWindow._show_viewer_backend(
            cast(ScannerWindow, window), description
        )
    )
    return viewer


def test_a_fallback_keeps_the_claim_describing_what_is_now_drawing_it(qt_app) -> None:
    """The swap carries the surface, the camera and the sentence about them.

    A fallback that left the panel claiming the dots while the replacement drew
    the shell - or the other way round - would be the one moment an operator has
    a concrete reason to distrust the picture.
    """
    window = CaptureWindow(frames=1)
    viewer = viewing(window)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)
    before = said(window)

    backend.unavailable.emit("the driver stopped answering")
    qt_app.processEvents()

    assert isinstance(viewer.view, CloudView)
    assert viewer.showing_surface
    assert window._mounted_picture() is ModelPicture.SURFACE
    assert said(window) == before
    assert f"{len(SURFACE_FACES)} triangles" in said(window)


def test_a_fallback_under_the_dots_still_describes_the_dots(qt_app) -> None:
    window = CaptureWindow(frames=1)
    viewer = viewing(window)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    window._show_coverage(published(surface=True))

    backend.unavailable.emit("the driver stopped answering")
    qt_app.processEvents()

    assert isinstance(viewer.view, CloudView)
    assert not viewer.showing_surface
    assert said(window) == POINTS_PROVENANCE


def test_a_report_from_the_renderer_that_was_replaced_relabels_nothing(qt_app) -> None:
    """Several failures can be in flight when the first one is acted on.

    Every one after the first names a widget that has already been unparented,
    and the viewer drops them. This asserts the consequence for the sentence:
    the line is not rewritten at all, so it cannot be rewritten wrongly.
    """
    window = CaptureWindow(frames=1)
    viewer = viewing(window)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    window._show_coverage(published(surface=True))
    window._toggle_surface(True)
    backend.unavailable.emit("the driver stopped answering")
    qt_app.processEvents()
    label = cast(Recording, window._model_provenance)
    after_swap = list(label.texts)

    backend.unavailable.emit("the driver stopped answering again")
    qt_app.processEvents()

    assert label.texts == after_swap
    assert window._mounted_picture() is ModelPicture.SURFACE
    assert f"{len(SURFACE_FACES)} triangles" in label.text


# -- the boundary this task must not cross ------------------------------------

#: Every offline completion adapter. None of them may be reached by the GUI, the
#: capture, the tracker, the preview, the fusion or the export - see
#: ``docs/preview-provenance.md`` for why they stay offline evidence.
COMPLETION_MODULES = (
    "scanner3d.depth_completion",
    "scanner3d.depth_temporal",
    "scanner3d.tsdf_input",
)

#: The product modules that make, place or show a picture. Named rather than
#: globbed, and :mod:`scanner3d.tsdf` is deliberately NOT among them: that module
#: offers a weighted entry point whose argument type comes from
#: :mod:`scanner3d.tsdf_input`, and what matters is that no product path calls
#: it. That half is asserted at runtime by the reconstruction probe below.
PICTURE_MODULES = (
    "gui",
    "cloud_view",
    "cloud_viewer",
    "gl_cloud_view",
    "preview_provenance",
    "viewer",
    "capture_sequence",
    "preparation",
    "tracking",
    "pipeline",
    "reconstruction_export",
    "mesh",
    "tsdf_surface",
)


def test_no_module_that_makes_a_picture_so_much_as_names_a_completion_adapter() -> None:
    """The structural half: nothing on the way to the screen imports one.

    Cheaper and stricter than any runtime probe, because it holds for paths this
    suite never happens to drive.
    """
    package = Path(gui.__file__).resolve().parent
    named = {
        name: sorted(
            adapter
            for adapter in COMPLETION_MODULES
            if adapter.split(".")[-1] in (package / f"{name}.py").read_text(encoding="utf-8")
        )
        for name in PICTURE_MODULES
    }

    assert {name: found for name, found in named.items() if found} == {}


def booby_trap(monkeypatch) -> list[str]:
    """Make every public callable in all three adapters raise if it is called.

    Attributes on the imported module objects, so a lazy ``from ... import`` at
    call time gets the trap too: it would be handed the very object this
    replaced.
    """
    reached: list[str] = []

    def refuse(where: str) -> Callable[..., object]:
        def trap(*_args: object, **_kwargs: object) -> object:
            reached.append(where)
            raise AssertionError(f"{where} was reached from a product path")

        return trap

    for name in COMPLETION_MODULES:
        module = importlib.import_module(name)
        for attribute in dir(module):
            value = getattr(module, attribute)
            if attribute.startswith("_") or not callable(value):
                continue
            if getattr(value, "__module__", None) != name:
                continue
            monkeypatch.setattr(module, attribute, refuse(f"{name}.{attribute}"))
    assert reached == []
    return reached


def test_a_live_capture_draws_both_pictures_with_every_completion_adapter_trapped(
    stream_kernel, capture_dir: Path, live_meshable, monkeypatch
) -> None:
    """The negative probe over the live path, with the pictures still produced.

    Spatial and temporal completion and the weighted TSDF input are offline
    evidence. Making the preview say what is measured must not become a reason
    to route any of them into what the operator is shown, so the capture that
    produces both pictures is run with all three set to raise.
    """
    stream_kernel(frames(3 * COVERAGE_EVERY + 4))
    window = CaptureWindow(frames=3 * COVERAGE_EVERY + 4)
    live_meshable()
    reached = booby_trap(monkeypatch)

    start_capture(window)
    window._toggle_surface(True)
    while_surfaced = said(window)
    window._toggle_surface(False)

    assert reached == []
    assert window.worker.coverage.emitted
    assert len(window.frames_on_disk()) == 3 * COVERAGE_EVERY + 4
    assert "DERIVED occupancy surface" in while_surfaced
    assert said(window) == POINTS_PROVENANCE


def test_a_reconstruction_draws_its_model_with_every_completion_adapter_trapped(
    tmp_path: Path, monkeypatch, meshable_clouds, registration, mesh_budget: int
) -> None:
    """The same probe over Reconstruct, which is the other path that publishes."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(4):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    registration()
    window = CaptureWindow(frames=1)
    reached = booby_trap(monkeypatch)
    # The one product module that names a completion type at all: the fusion
    # volume's weighted entry point. Trapped by the method rather than by the
    # module, because ``scanner3d.tsdf`` bound those names at import time and an
    # attribute swap on the adapter would not be seen from inside it.
    monkeypatch.setattr(
        TsdfVolume,
        "integrate_weighted",
        lambda *_args, **_kwargs: reached.append("scanner3d.tsdf.TsdfVolume.integrate_weighted"),
    )

    start_reconstruct(window, frames_dir, monkeypatch)

    assert reached == []
    assert window.worker.coverage.emitted
    assert window._mounted_picture() in (ModelPicture.POINTS, ModelPicture.SURFACE)
    assert said(window) != NO_MODEL_PROVENANCE


# -- and what the operator is told away from the screen ------------------------


def test_the_prose_says_what_each_preview_is_and_why_completion_stays_offline() -> None:
    """The README and the focused document, read for the same three claims.

    Read here because it was one claim written in three places - the panel, the
    README and the operator document - and three copies of a claim drift unless
    something asks all of them. Phrases are read off the UNWRAPPED text: where a
    line break happens to fall is not part of what is claimed.
    """
    root = Path(__file__).resolve().parents[1]
    readme = " ".join(
        (root / "README.md").read_text(encoding="utf-8").lower().replace("*", "").split()
    )
    focused = " ".join(
        (root / "docs" / "preview-provenance.md")
        .read_text(encoding="utf-8")
        .lower()
        .replace("*", "")
        .split()
    )

    for prose in (readme, focused):
        # What the depth picture is, and what the checker means.
        assert "raw sensor depth codes" in prose
        assert "no return" in prose
        assert "no completion" in prose
        # Measured versus derived, in the words the panel itself uses.
        assert "measured depth returns" in prose
        assert "derived occupancy surface" in prose
        # And the two derived surfaces kept apart, which is what round one's
        # single surface wording could not do.
        assert "signed-distance" in prose
        assert "corrected" in prose
        assert "voxel size" in prose
        # And why the offline adapters stay offline.
        for adapter in COMPLETION_MODULES:
            assert adapter.lower() in prose
        assert "offline evidence" in prose
