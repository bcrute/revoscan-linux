"""Whether the accumulated model SHOWS that it is accumulating.

The owner's 15:50:53 run produced eight live models and the window drew all
eight of them - nothing was starved, nothing was superseded, and the saved-frame
numbers beside the picture kept climbing. The picture still looked static, for
two reasons that have nothing to do with throughput:

- Every snapshot re-centred and re-scaled the view, so a model that had grown
  was normalised straight back to the same apparent size, and whatever the
  operator had panned or zoomed to went with it.
- A snapshot is built from a fixed number of frames sampled across the whole
  sweep, so once a sweep is longer than that bound the payload stops growing.
  Late snapshots genuinely do look alike - even though each one contains a frame
  the one before it did not.

So the newest contributing frame is now flagged all the way from the build to
the pixels, drawn in its own coverage colour lifted towards white, and the view
frames itself only at a boundary. What "a boundary" means is the subject of the
second half of this module: a new capture and a registered reconstruction, and
nothing else.

Nothing here builds a widget. :class:`CloudView`'s methods are called unbound
against a stand-in carrying exactly the attributes they touch, for the reason
:mod:`tests.gui_harness` gives: a Qt event loop answers none of these questions
and would make the tests unrunnable on a machine with no display server.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.cloud_view import (
    COVERED_COLOUR,
    MAX_RENDER_POINTS,
    NEWEST_COVERED_COLOUR,
    NEWEST_THIN_COLOUR,
    THIN_COLOUR,
    CloudCamera,
    CloudView,
)
from scanner3d.geometry import Rigid
from scanner3d.gui import (
    LIVE_MODEL_CLOUDS,
    LIVE_MODEL_POINTS,
    ModelSnapshot,
    _live_snapshot,
)
from scanner3d.reconstruction_export import UPRIGHT_FROM_OPTICAL
from tests.gui_harness import (
    CaptureWindow,
    clouds,
    mesh_budget,
    meshable_clouds,
    registration,
    sphere_raw_frame,
    start_reconstruct,
)

__all__ = ["clouds", "mesh_budget", "meshable_clouds", "registration"]


def upright(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """One point of the depth camera's optical frame, as the view is given it.

    The live model and the exported files go through the same rotation, so a
    test that expects one of them can be written with the other's arithmetic.
    """
    return UPRIGHT_FROM_OPTICAL.apply(point)


class Viewport(CloudCamera):
    """The REAL camera that both renderers share, with no Qt behind it.

    :class:`~scanner3d.cloud_view.CloudCamera` holds the model, the pose and the
    gestures and is deliberately not a widget, so a method that reached for
    something only a widget has raises ``AttributeError`` here instead of passing
    quietly. The orbit it carries is the accumulated rotation:
    :mod:`tests.test_cloud_view_orbit` owns what turns it, and this module owns
    what must not.
    """

    def __init__(self) -> None:
        self._init_camera()
        self.updates = 0

    def update(self, *_args: object) -> None:  # Qt's name
        self.updates += 1


def show(
    view: Viewport,
    points: np.ndarray,
    thin: np.ndarray,
    newest: np.ndarray | None = None,
    *,
    fit: bool = False,
) -> None:
    """The real :meth:`CloudView.set_cloud`, against the stand-in."""
    CloudView.set_cloud(cast(CloudView, view), points, thin, newest, fit=fit)


def _turned(yaw: float = 1.2, pitch: float = -0.3) -> np.ndarray:
    """An orientation an operator could have dragged to.

    Built rather than written down, so it is a real rotation: an assertion that
    something survived a matrix means nothing if the matrix was never one.
    """
    about_y = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ]
    )
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    return about_x @ about_y


#: The two angles the old clamped orbit would have been left at, as the matrix
#: that has replaced them.
TURNED = _turned()


def cube(count: int, scale: float = 1.0) -> np.ndarray:
    """``count`` points on a line, so a bounding box is trivially predictable."""
    return np.array(
        [[float(index) * scale, 0.0, 100.0] for index in range(count)], dtype=np.float32
    )


# -- the highlight ---------------------------------------------------------


def test_the_newest_points_are_the_lifted_form_of_what_they_already_meant() -> None:
    """Four colours, two meanings. The highlight adds, it does not replace.

    Red still means under-scanned and grey still means covered on the newest
    frame as much as anywhere else - the point is lighter, not recoloured. A
    third hue would have taken those points out of the coverage picture at the
    exact moment the operator is reading it.
    """
    view = Viewport()
    points = cube(4)
    thin = np.array([False, True, False, True])
    newest = np.array([False, False, True, True])

    show(view, points, thin, newest)

    assert [tuple(colour) for colour in view._colours] == [
        COVERED_COLOUR,
        THIN_COLOUR,
        NEWEST_COVERED_COLOUR,
        NEWEST_THIN_COLOUR,
    ]
    # Brighter than the meaning it carries, and still recognisably that meaning:
    # the red channel of a lifted gap stays the highest of its three.
    assert NEWEST_THIN_COLOUR > THIN_COLOUR
    assert NEWEST_COVERED_COLOUR > COVERED_COLOUR


def test_a_cloud_with_no_newest_frame_is_coloured_exactly_as_it_always_was() -> None:
    """The default, which is what a caller that knows nothing about this gets."""
    view = Viewport()
    thin = np.array([True, False])

    show(view, cube(2), thin)

    assert [tuple(colour) for colour in view._colours] == [THIN_COLOUR, COVERED_COLOUR]


def test_the_highlight_survives_the_views_own_point_ceiling() -> None:
    """Decimating at the door must not leave the flags describing the old array.

    ``points`` and both masks are stepped together or the colouring slides off
    the geometry - and a highlight that lands on the wrong points is worse than
    none, because it points the operator at a part of the object the scanner has
    not just seen.
    """
    view = Viewport()
    total = MAX_RENDER_POINTS * 2 + 7
    points = cube(total)
    thin = np.zeros(total, dtype=bool)
    thin[::3] = True
    newest = np.zeros(total, dtype=bool)
    newest[-900:] = True

    show(view, points, thin, newest)

    assert len(view._points) <= MAX_RENDER_POINTS
    assert len(view._colours) == len(view._points)
    lifted = [
        index
        for index, colour in enumerate(view._colours)
        if tuple(colour) in (NEWEST_COVERED_COLOUR, NEWEST_THIN_COLOUR)
    ]
    # Still there, still at the END - which is where the newest frame's points
    # are - and still a minority of the picture rather than all of it.
    assert lifted
    assert max(lifted) == len(view._points) - 1
    assert len(lifted) < len(view._points)


# -- what the producer flags -----------------------------------------------


def test_the_newest_frame_survives_being_sampled_out_of_a_long_sweep() -> None:
    """A sweep longer than the sampling bound still highlights its last frame.

    This is the second half of why late snapshots looked identical: past
    ``LIVE_MODEL_CLOUDS`` frames the payload stops growing, so "there is a new
    frame in this" has to be VISIBLE rather than inferable from the size. The
    sampler always ends its span on the newest frame, and the flag follows it.
    """
    total = LIVE_MODEL_CLOUDS * 3
    poses = [Rigid.identity() for _ in range(total)]
    built = [((float(index), 0.0, 100.0), (0.0, float(index), 101.0)) for index in range(total)]

    snapshot = _live_snapshot(built, poses, saved=total, aligned=total)

    assert snapshot.sampled == LIVE_MODEL_CLOUDS
    assert len(snapshot.newest) == len(snapshot.points)
    assert len(snapshot.thin) == len(snapshot.points)
    # Exactly the last frame's two points, and they are that frame's own.
    assert int(snapshot.newest.sum()) == 2
    # The last frame's own two points, turned upright: the view draws +y up and
    # +z towards it, and the live model reaches it through the same
    # presentation rotation the exported files are written in.
    assert [tuple(point) for point in snapshot.points[snapshot.newest]] == [
        upright((float(total - 1), 0.0, 100.0)),
        upright((0.0, float(total - 1), 101.0)),
    ]


def test_the_newest_flags_are_stepped_with_the_points_they_describe(monkeypatch) -> None:
    """Producer-side point limiting keeps all three arrays parallel.

    The bound exists so a payload is never assembled in full and then thrown
    away at the far end; carrying two masks through it is what stops that saving
    from costing the picture its meaning.
    """
    total = LIVE_MODEL_POINTS * 3
    points = np.arange(3 * total, dtype=np.float32).reshape(-1, 3)
    thin = np.zeros(total, dtype=bool)
    thin[1::2] = True
    newest = np.zeros(total, dtype=bool)
    newest[-total // 3 :] = True
    monkeypatch.setattr(gui, "_coloured_cloud", lambda _clouds, _poses: (points, thin, 0.5, newest))

    snapshot = _live_snapshot([((0.0, 0.0, 1.0),)], [Rigid.identity()], saved=1, aligned=1)

    assert len(snapshot.points) <= LIVE_MODEL_POINTS
    assert len(snapshot.newest) == len(snapshot.points)
    assert len(snapshot.thin) == len(snapshot.points)
    step = total // LIVE_MODEL_POINTS + 1
    assert np.array_equal(snapshot.newest, newest[::step])
    assert np.array_equal(snapshot.points, points[::step])


def test_the_build_flags_the_last_cloud_it_was_given() -> None:
    """The flag is produced where the point order is decided, not guessed at.

    A caller counting backwards from the end would agree with this exactly until
    the day the build changed how it lays points out, and would then mislabel
    the highlight silently.
    """
    built = [((0.0, 0.0, 100.0),), ((1.0, 0.0, 100.0), (2.0, 0.0, 100.0))]
    poses = [Rigid.identity(), Rigid.identity()]

    points, _thin, _covered, newest = gui._coloured_cloud(built, poses)

    assert len(points) == 3
    assert list(newest) == [False, True, True]


def test_the_live_model_is_handed_over_in_the_same_upright_frame_as_the_export() -> None:
    """One frame for one widget, whether the model is live or registered.

    :class:`~scanner3d.cloud_view.CloudView` draws +y up and +z towards the
    viewer. The depth camera's optical frame counts +y DOWN the image and +z
    away, so a live model handed over unrotated is drawn upside down AND back to
    front - and next to a registered reconstruction that is neither, which is
    the state the owner reported. Both paths now compose the same
    :func:`~scanner3d.reconstruction_export.upright_pose`.

    The pose itself is not mutated: the composition is done on a copy for the
    picture, and the argument this test passes in is asserted afterwards to be
    the pose it was.
    """
    pose = Rigid.of(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (3.0, -2.0, 5.0))
    rotation, translation = pose.rotation, pose.translation
    built = [(((7.0, 11.0, 120.0), (-4.0, 2.0, 130.0)))]

    snapshot = _live_snapshot(built, [pose], saved=1, aligned=1)

    assert [tuple(point) for point in snapshot.points] == [
        upright(placed) for placed in pose.apply_all(built[0])
    ]
    # Said again without the helper, so this cannot pass by agreeing with the
    # same arithmetic twice: x is kept and y and z are negated.
    placed = pose.apply(built[0][0])
    assert tuple(snapshot.points[0]) == pytest.approx((placed[0], -placed[1], -placed[2]))
    # The pose the caller handed in is the pose it still holds.
    assert (pose.rotation, pose.translation) == (rotation, translation)


# -- the framing boundary ---------------------------------------------------


def test_the_first_cloud_is_framed_even_though_it_did_not_ask_to_be() -> None:
    """There is no pose to preserve before the first one.

    Left unframed, the first model of a session is drawn with the camera at its
    constructed distance, pointing at the origin - which for a cloud a metre
    away is an empty window that looks exactly like a capture that produced
    nothing.
    """
    view = Viewport()
    points = cube(10, scale=10.0)

    show(view, points, np.zeros(10, dtype=bool))

    assert view._framed is True
    assert view._centre[0] == pytest.approx(45.0)
    assert view._distance == pytest.approx(90.0 * 1.4)


def test_incremental_updates_leave_the_orbit_zoom_and_pan_exactly_where_they_were() -> None:
    """The correction, stated as the three things the operator set themselves.

    Each of these updates has a bigger cloud than the last, which under the old
    behaviour is precisely when the view re-scaled and the growth stopped being
    visible.
    """
    view = Viewport()
    show(view, cube(10, scale=10.0), np.zeros(10, dtype=bool))
    # The operator turns it, zooms in, and pans it off centre.
    view._rotation = TURNED
    view._distance = 55.0
    view._pan[:] = (12.0, -8.0)
    framed_at = np.array(view._centre)

    for count in (20, 40, 80):
        show(view, cube(count, scale=10.0), np.zeros(count, dtype=bool))

    assert view._distance == pytest.approx(55.0)
    assert list(view._pan) == [12.0, -8.0]
    assert np.array_equal(view._centre, framed_at)
    assert np.array_equal(view._rotation, TURNED)
    assert view.updates == 4


def test_an_update_that_asks_to_be_framed_re_centres_and_clears_the_pan() -> None:
    """The boundary itself: a new subject, framed, with the pan reset.

    The orbit is deliberately NOT reset with it. Which side of the object you
    are standing on is a preference the operator expressed about themselves;
    where the object is and how big it is are facts about a different cloud.
    """
    view = Viewport()
    show(view, cube(10, scale=10.0), np.zeros(10, dtype=bool))
    view._rotation = TURNED
    view._pan[:] = (12.0, -8.0)

    show(view, cube(10, scale=100.0), np.zeros(10, dtype=bool), fit=True)

    assert view._centre[0] == pytest.approx(450.0)
    assert view._distance == pytest.approx(900.0 * 1.4)
    assert list(view._pan) == [0.0, 0.0]
    assert np.array_equal(view._rotation, TURNED)


def test_the_window_passes_the_snapshots_own_framing_answer_to_the_view() -> None:
    """The window decides nothing here; it carries what the producer decided.

    Which is what keeps "is this the first model of this capture" a fact known
    where the capture is - on the worker, at build time - rather than something
    the GUI thread re-derives from counters that have since moved on.
    """
    window = CaptureWindow(frames=1)
    payload = np.zeros((2, 3), dtype=np.float32)

    for reframe in (True, False, False):
        window._show_coverage(
            ModelSnapshot(
                points=payload,
                thin=np.array([True, False]),
                newest=np.array([False, True]),
                well_covered=0.5,
                aligned=3,
                saved=3,
                sampled=3,
                note="preview -",
                reframe=reframe,
            )
        )

    assert window._cloud.fits == [True, False, False]
    assert "brightest points are the newest frame" in window._coverage_summary.text


# -- and none of it applies to a registered reconstruction ------------------


def test_a_registered_model_highlights_nothing_and_frames_the_view(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    """Reconstruct's own result, through the real Reconstruct path.

    A reconstruction places every frame at once, possibly hours after the sweep,
    so none of its frames "just arrived" and highlighting the last file would be
    an animation cue attached to nothing that happened. It IS a new subject
    though - every frame, every point, and poses the optimiser moved - so it is
    entitled to frame the view, which is the other half of this assertion.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(4):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    registration()
    window = CaptureWindow(frames=1)

    start_reconstruct(window, frames_dir, monkeypatch)

    (snapshot,) = [cast(ModelSnapshot, payload) for (payload,) in window.worker.coverage.emitted]
    assert snapshot.note == "registered -"
    assert len(snapshot.newest) == len(snapshot.points)
    assert not snapshot.newest.any()
    assert snapshot.reframe is True
    # And what the view was handed says the same: framed, nothing lifted.
    assert window._cloud.fits == [True]
    assert not cast(np.ndarray, window._cloud.newest[0]).any()
    assert "brightest points" not in window._coverage_summary.text
