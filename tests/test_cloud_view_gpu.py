"""The GPU viewer, asked everything that can be asked without a graphics card.

What "the same viewer, faster" has to mean
------------------------------------------
The complaint was that the program stutters while the scanned points are being
turned, and the answer was to stop projecting sixty thousand points on the CPU
for every mouse event and let the card draw geometry it already holds. That is
only an improvement if the picture is otherwise IDENTICAL: same way up, same
face in front, same colours, same framing rules, same gestures. A faster viewer
that mirrors the object, or loses the depth order, or re-frames when it should
not, is not a faster viewer - it is a different and wrong one.

So the software renderer is treated here as the specification and
:func:`~scanner3d.gl_cloud_view.camera_matrix` is held to it numerically, point
by point, at several poses. Two descriptions of one projection that are never
compared do not stay in step.

The other half is the thing the change is actually FOR: that a drag costs a
matrix and not an upload, and that a hundred mouse events cost one frame rather
than a hundred. Both are asserted directly, on counters that exist for that.

Nothing here needs a driver, a context, a window or a display server - see
:mod:`tests.test_cloud_view_gl_smoke` for the part that does.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt

from scanner3d import gl_cloud_view
from scanner3d.cloud_view import (
    COVERED_COLOUR,
    DRAG_RADIANS_PER_PIXEL,
    NEWEST_COVERED_COLOUR,
    NEWEST_THIN_COLOUR,
    THIN_COLOUR,
    CloudView,
)
from scanner3d.cloud_viewer import AUTO, OPENGL, SOFTWARE, VIEWER_ENV, requested_backend
from scanner3d.gl_cloud_view import (
    FAR_MARGIN,
    NEAR_PLANE,
    GLCamera,
    GLCloudView,
    PointRenderer,
    RendererUnavailable,
    _colour_attributes,
)
from scanner3d.reconstruction_export import UPRIGHT_FROM_OPTICAL

WIDTH = 331
HEIGHT = 217


class Viewport(GLCamera):
    """The real GPU camera - matrix, planes and point size - with no widget.

    :class:`~scanner3d.gl_cloud_view.GLCamera` is exactly what
    :meth:`~scanner3d.gl_cloud_view.GLCloudView.paintGL` reads, and it is
    inherited rather than reimplemented: a stand-in that computed its own matrix
    would only prove that the stand-in is right. What is supplied here is the
    widget's part - a size, a device pixel ratio, and a repaint that is counted
    instead of scheduled.
    """

    def __init__(self) -> None:
        self._init_camera()
        self.updates = 0
        self.painted = 0

    def update(self, *_args: object) -> None:  # Qt's name
        self.updates += 1

    def width(self) -> int:  # Qt's name
        return WIDTH

    def height(self) -> int:  # Qt's name
        return HEIGHT

    def devicePixelRatioF(self) -> float:  # Qt's name
        return 1.0

    def paint(self) -> None:
        """What a displayed frame does to the pacing state."""
        self.painted += 1
        self._frame_painted()


class Move:
    """One mouse event: where the pointer is, and what is held down."""

    def __init__(
        self,
        x: float,
        y: float,
        buttons: Qt.MouseButton = Qt.MouseButton.NoButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._at = QPointF(x, y)
        self._buttons = buttons
        self._modifiers = modifiers

    def position(self) -> QPointF:  # Qt's name
        return self._at

    def buttons(self) -> Qt.MouseButton:  # Qt's name
        return self._buttons

    def modifiers(self) -> Qt.KeyboardModifier:  # Qt's name
        return self._modifiers


class Wheel:
    def __init__(self, notches: int) -> None:
        self._delta = QPoint(0, 120 * notches)

    def angleDelta(self) -> QPoint:  # Qt's name
        return self._delta


def press(view: Viewport) -> None:
    CloudView.mousePressEvent(cast(CloudView, view), cast(object, Move(0.0, 0.0)))  # type: ignore[arg-type]


def move(view: Viewport, x: float, y: float, **kwargs: object) -> None:
    with_button(view, x, y, Qt.MouseButton.LeftButton, **kwargs)


def with_button(
    view: Viewport, x: float, y: float, button: Qt.MouseButton, **kwargs: object
) -> None:
    CloudView.mouseMoveEvent(
        cast(CloudView, view),
        cast(object, Move(x, y, button, **kwargs)),  # type: ignore[arg-type]
    )


def release(view: Viewport) -> None:
    CloudView.mouseReleaseEvent(cast(CloudView, view), cast(object, Move(0.0, 0.0)))  # type: ignore[arg-type]


def upright(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """One point of the depth camera's optical frame, as the view is given it."""
    return UPRIGHT_FROM_OPTICAL.apply(point)


def a_cloud(count: int = 400, seed: int = 7) -> np.ndarray:
    """A deterministic blob of points in front of where the scanner stood."""
    rng = np.random.default_rng(seed)
    points = (rng.random((count, 3)) * 40.0 - 20.0).astype(np.float32)
    points[:, 2] += 120.0
    return np.array([upright(tuple(point)) for point in points], dtype=np.float32)


def framed_view(points: np.ndarray | None = None) -> Viewport:
    view = Viewport()
    cloud = a_cloud() if points is None else points
    view.set_cloud(cloud, np.zeros(len(cloud), dtype=bool), fit=True)
    return view


def on_screen(view: Viewport) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the card would put every point: columns, rows, and eye distance.

    The clip-space arithmetic the driver does, done here in numpy so it can be
    compared with the software renderer's answer. Everything after the matrix -
    the perspective divide, the viewport transform, the near clip, the window
    edges - is fixed by OpenGL and is not this project's choice.
    """
    matrix = view.camera().astype(np.float64)
    homogeneous = np.c_[view._points, np.ones(len(view._points))]
    clip = homogeneous @ matrix.T
    distance = clip[:, 3]
    visible = distance > NEAR_PLANE
    normalised = clip[visible, :3] / distance[visible, None]
    columns = np.rint((normalised[:, 0] + 1.0) * WIDTH / 2.0).astype(np.int32)
    rows = np.rint((1.0 - normalised[:, 1]) * HEIGHT / 2.0).astype(np.int32)
    inside = (columns >= 0) & (columns < WIDTH) & (rows >= 0) & (rows < HEIGHT)
    return columns[inside], rows[inside], distance[visible][inside]


# -- the same picture ------------------------------------------------------


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll", "pan", "distance"),
    [
        (0.0, 0.0, 0.0, (0.0, 0.0), None),
        (1.3, -0.7, 0.4, (17.0, -9.0), 88.0),
        (-2.9, 2.2, -1.1, (-40.0, 55.0), 300.0),
        (math.pi, 0.0, 0.0, (0.0, 0.0), 30.0),
    ],
)
def test_the_card_puts_every_point_where_the_software_viewer_puts_it(
    yaw: float, pitch: float, roll: float, pan: tuple[float, float], distance: float | None
) -> None:
    """The whole claim, as one assertion, at four poses.

    Not "close": the same integer pixel, for every point. The two paths share no
    arithmetic - one is a numpy projection and the other is a 4x4 handed to a
    shader - so agreeing to the pixel is evidence that the second is a restatement
    of the first rather than a second opinion about what this view means.
    """
    view = framed_view()
    view._rotation = _about_x(pitch) @ _about_y(yaw) @ np.array(_about_z(roll), dtype=np.float64)
    view._pan[:] = pan
    if distance is not None:
        view._distance = distance

    software = sorted(
        zip(*CloudView.projected(cast(CloudView, view), WIDTH, HEIGHT)[:2], strict=True)
    )
    columns, rows, _distance = on_screen(view)

    assert len(columns), "the pose under test put nothing on the screen"
    assert sorted(zip(columns.tolist(), rows.tolist(), strict=True)) == [
        (int(column), int(row)) for column, row in software
    ]


def _about_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _about_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _about_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_matrix_keeps_the_object_upright_and_the_right_way_round() -> None:
    """Which way up and which way round, asked of the matrix on its own.

    The same question :mod:`tests.test_cloud_view_orbit` asks of the software
    path, repeated here because a shader is where a sign error would now live and
    a mirrored scan still looks like a scan.
    """
    view = framed_view()
    matrix = view.camera().astype(np.float64)

    def screen(point: tuple[float, float, float]) -> tuple[float, float]:
        clip = matrix @ np.array([*point, 1.0])
        normalised = clip[:3] / clip[3]
        return (normalised[0] + 1.0) * WIDTH / 2.0, (1.0 - normalised[1]) * HEIGHT / 2.0

    _column, top = screen(upright((0.0, -10.0, 120.0)))
    _column, bottom = screen(upright((0.0, 10.0, 120.0)))
    right, _row = screen(upright((10.0, 0.0, 120.0)))
    left, _row = screen(upright((-10.0, 0.0, 120.0)))

    assert top < HEIGHT / 2 < bottom
    assert left < WIDTH / 2 < right


def test_what_is_in_front_is_what_the_depth_buffer_will_keep() -> None:
    """Occlusion, as the depth buffer decides it rather than as a sort does.

    The software path paints far to near so the near point lands on top. Here
    there is no order at all - the card draws in buffer order - so the ONLY thing
    that makes the near surface the visible one is that its clip-space depth is
    smaller, monotonically, all the way through the model.
    """
    view = framed_view()
    matrix = view.camera().astype(np.float64)
    homogeneous = np.c_[view._points, np.ones(len(view._points))]
    clip = homogeneous @ matrix.T
    eye_distance = clip[:, 3]
    depth = clip[:, 2] / eye_distance

    order = np.argsort(eye_distance)
    assert np.all(np.diff(depth[order]) >= -1e-12), "nearer points must win the depth test"
    assert depth.min() >= -1.0 and depth.max() <= 1.0, "the whole model is between the planes"


def test_the_far_plane_stays_behind_the_object_however_it_is_turned() -> None:
    """A far plane at the object clips its own back surface as rounding decides.

    Turning the model moves points AROUND the centre, never away from it, so the
    radius measured when the model arrived bounds every pose - which is why it is
    measured once per model and not once per frame.
    """
    view = framed_view()
    for pitch in np.linspace(0.0, 2.0 * math.pi, 17):
        view._rotation = _about_x(float(pitch))
        matrix = view.camera().astype(np.float64)
        clip = np.c_[view._points, np.ones(len(view._points))] @ matrix.T
        depth = clip[:, 2] / clip[:, 3]
        assert depth.max() < 1.0, "a point fell behind the far plane"
    assert view._far_plane() > view._distance + view._radius


def test_the_far_plane_is_measured_from_the_model_and_not_from_a_guess() -> None:
    """It has to move with the cloud, or a big scan clips and a small one z-fights."""
    small = framed_view(a_cloud(count=50))
    big = framed_view(np.concatenate([a_cloud(), a_cloud() * 8.0]))

    assert big._radius > small._radius
    assert big._far_plane() > small._far_plane()
    assert small._far_plane() == pytest.approx(
        NEAR_PLANE + small._distance + small._radius + FAR_MARGIN
    )


# -- the colours -----------------------------------------------------------


def test_the_four_meaning_colours_survive_the_trip_to_the_card() -> None:
    """Red still means "scan here" after it has become a vertex attribute.

    Converted to floats on the way rather than in the shader, so that what a
    colour MEANS stays decided in one place. Exactly, not nearly: 255ths are
    representable and a coverage colour that drifted would drift on every point.
    """
    colours = np.array(
        [COVERED_COLOUR, THIN_COLOUR, NEWEST_COVERED_COLOUR, NEWEST_THIN_COLOUR], dtype=np.uint8
    )

    attributes = _colour_attributes(colours)

    assert attributes.dtype == np.float32
    assert np.array_equal(attributes, colours.astype(np.float32) / 255.0)
    assert attributes.min() >= 0.0 and attributes.max() <= 1.0
    assert np.array_equal(np.rint(attributes * 255.0).astype(np.uint8), colours)


def test_the_colours_are_still_the_ones_the_shared_camera_chose() -> None:
    """The GPU path colours nothing itself; it uploads what the camera decided."""
    view = Viewport()
    points = np.zeros((4, 3), dtype=np.float32)
    thin = np.array([False, True, False, True])
    newest = np.array([False, False, True, True])

    view.set_cloud(points, thin, newest)

    assert [tuple(colour) for colour in view._colours] == [
        COVERED_COLOUR,
        THIN_COLOUR,
        NEWEST_COVERED_COLOUR,
        NEWEST_THIN_COLOUR,
    ]


# -- what costs an upload --------------------------------------------------


def test_only_a_new_model_makes_the_resident_geometry_stale() -> None:
    """The point of the whole change, stated as the flag that drives it.

    Every gesture there is, then a resize's worth of camera change, and the
    geometry is still the geometry that was uploaded. If any of these set the
    flag, an orbit puts sixty thousand points back on the bus every frame and the
    GPU path is the CPU path with an upload added.
    """
    view = framed_view()
    view._geometry_dirty = False

    press(view)
    move(view, 40.0, 25.0)
    move(view, 80.0, 60.0, modifiers=Qt.KeyboardModifier.ShiftModifier)
    release(view)
    with_button(view, 5.0, 5.0, Qt.MouseButton.MiddleButton)
    CloudView.wheelEvent(cast(CloudView, view), cast(object, Wheel(1)))  # type: ignore[arg-type]
    CloudView.wheelEvent(cast(CloudView, view), cast(object, Wheel(-3)))  # type: ignore[arg-type]

    assert view._geometry_dirty is False


def test_a_new_model_does_make_it_stale() -> None:
    """And the other half: the one event that must reach the card, does."""
    view = framed_view()
    view._geometry_dirty = False

    view.set_cloud(a_cloud(count=90), np.zeros(90, dtype=bool))

    assert view._geometry_dirty is True


# -- what costs a frame ----------------------------------------------------


def test_one_drag_asks_for_one_frame_however_many_events_it_is() -> None:
    """Mouse events arrive faster than a monitor can show them.

    Fifty positions between two displayed frames is an ordinary fast drag on a
    high-rate mouse. Asking for fifty repaints is what accumulates: the work
    piles up behind the pointer and the picture trails it, which is exactly the
    stutter that was reported.
    """
    view = framed_view()
    view.paint()
    before = view.updates

    press(view)
    for step in range(1, 51):
        move(view, float(step), float(step) * 0.5)

    assert view.updates - before == 1


def test_coalescing_drops_frames_and_never_drops_motion() -> None:
    """The thing coalescing must not cost: how far the object actually turned.

    The state a drag changes is cumulative and every event is applied in full;
    what is skipped is only the drawing of positions nobody could have seen. So
    fifty coalesced events and fifty drawn ones leave the SAME orientation, and
    it is the analytic rotation for the pixels that were dragged.
    """
    coalesced = framed_view()
    drawn = framed_view()

    press(coalesced)
    press(drawn)
    for step in range(1, 51):
        move(coalesced, float(step) * 2.0, 0.0)
        move(drawn, float(step) * 2.0, 0.0)
        drawn.paint()

    assert np.allclose(coalesced._rotation, drawn._rotation)
    assert np.allclose(coalesced._rotation, _about_y(100.0 * DRAG_RADIANS_PER_PIXEL), atol=1e-9)
    assert drawn.updates > coalesced.updates


def test_a_painted_frame_lets_the_next_gesture_ask_again() -> None:
    """Otherwise the first frame is the only one and the view freezes mid-drag."""
    view = framed_view()
    view.paint()
    press(view)

    move(view, 10.0, 0.0)
    move(view, 20.0, 0.0)
    first = view.updates
    view.paint()
    move(view, 30.0, 0.0)

    assert view.updates == first + 1


def test_letting_go_of_the_pointer_always_asks_for_a_frame() -> None:
    """Wherever the user let go is where they meant to leave it.

    A release that only asked politely could land while a frame was already
    pending, be coalesced into it, and leave the last few degrees of the drag
    undrawn - the object stopping just short of where the hand stopped.
    """
    view = framed_view()
    view.paint()
    press(view)
    move(view, 10.0, 4.0)
    pending = view.updates

    move(view, 22.0, 9.0)
    release(view)

    assert view.updates == pending + 1, "the release must ask even with a frame pending"
    assert view._frame_pending is True


def test_an_idle_view_asks_for_nothing() -> None:
    """No timer, no redraw loop: a view nobody is touching costs nothing."""
    view = framed_view()
    view.paint()
    before = view.updates

    with_button(view, 300.0, 300.0, Qt.MouseButton.NoButton)

    assert view.updates == before


# -- failing safely --------------------------------------------------------


def test_the_renderer_refuses_to_pretend_it_was_set_up() -> None:
    """Upload and draw before initialisation are a fallback, not a blank panel."""
    renderer = PointRenderer()

    with pytest.raises(RendererUnavailable):
        renderer.upload(np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(RendererUnavailable):
        renderer.draw(np.eye(4, dtype=np.float32), 10, 10, 1.0)
    assert renderer.uploads == 0
    assert renderer.draws == 0


def test_the_renderer_will_not_initialise_without_a_context() -> None:
    """The commonest real failure, and it has to be a readable sentence."""
    with pytest.raises(RendererUnavailable, match="context"):
        PointRenderer().initialise()


class Reporter(Viewport):
    """A GPU view whose failure report is recorded rather than emitted."""

    def __init__(self) -> None:
        super().__init__()
        self._failure: str | None = None
        self.reported: list[str] = []
        self.unavailable = self

    def emit(self, reason: str) -> None:  # Qt's name, on the stand-in signal
        self.reported.append(reason)


def fail(view: Reporter, reason: str) -> None:
    """The REAL failure path, against the recorder."""
    GLCloudView._fail(cast(GLCloudView, view), reason)


def test_a_driver_that_gives_up_is_reported_once_and_with_its_reason() -> None:
    """A blank panel is the failure mode this path exists to prevent.

    Once, because a driver that fails in one paint fails in every later one and a
    log filling at the repaint rate buries the first line - which is the one that
    says what went wrong.
    """
    view = Reporter()

    fail(view, "shader program would not link: syntax error")
    fail(view, "and again")

    assert view.reported == ["shader program would not link: syntax error"]
    assert view._failure == "shader program would not link: syntax error"


# -- what the driver only tells you if you ask -----------------------------


class ErrorQueue:
    """``glGetError``'s actual contract: one queued error per call, then zero.

    The calls this stands in for - allocating a buffer, binding it, submitting a
    draw - return NOTHING to Qt. On a real driver the only evidence that one of
    them was refused is this queue, which is why the queue is what is faked: a
    stub that raised instead would be testing an error path that does not exist.
    """

    def __init__(self) -> None:
        self.queued: list[int] = []
        self.viewports: list[tuple[int, int]] = []
        self.drawn: list[tuple[int, int]] = []
        #: The primitive each draw asked for, kept beside :attr:`drawn` rather
        #: than inside it so the existing assertions about first-and-count stay
        #: about first and count.
        self.modes: list[int] = []

    def refuse(self, code: int = 0x0505) -> None:
        """What a driver out of memory leaves behind. 0x0505 is GL_OUT_OF_MEMORY."""
        self.queued.append(code)

    def glGetError(self) -> int:  # Qt's name
        return self.queued.pop(0) if self.queued else 0

    def glViewport(self, _x: int, _y: int, width: int, height: int) -> None:  # Qt's name
        self.viewports.append((width, height))

    def glClearColor(self, *_args: float) -> None: ...  # Qt's name

    def glClear(self, _bits: int) -> None: ...  # Qt's name

    def glDrawArrays(self, mode: int, first: int, count: int) -> None:  # Qt's name
        self.modes.append(mode)
        self.drawn.append((first, count))


class FakeBuffer:
    """A vertex buffer that accepts everything, and a driver that may not."""

    def __init__(self, errors: ErrorQueue, *, refuses: str | None = None) -> None:
        self._errors = errors
        self._refuses = refuses
        self.bytes = 0

    def bind(self) -> bool:  # Qt's name
        if self._refuses == "bind":
            self._errors.refuse()
        return True

    def release(self) -> None: ...  # Qt's name

    def allocate(self, _data: object, size: int) -> None:  # Qt's name
        if self._refuses == "allocate":
            self._errors.refuse()
            return
        self.bytes = size


class FakeProgram:
    """The linked shader, doing nothing, so the error checks are what is tested."""

    def bind(self) -> bool:  # Qt's name
        return True

    def release(self) -> None: ...  # Qt's name

    def setUniformValue(self, *_args: object) -> None: ...  # Qt's name

    def setUniformValue1f(self, *_args: object) -> None: ...  # Qt's name

    def enableAttributeArray(self, _location: int) -> None: ...  # Qt's name

    def disableAttributeArray(self, _location: int) -> None: ...  # Qt's name

    def setAttributeBuffer(self, *_args: object) -> None: ...  # Qt's name


def with_driver(
    monkeypatch: pytest.MonkeyPatch,
    errors: ErrorQueue,
    *,
    refuses: str | None = None,
    which: str = "positions",
) -> PointRenderer:
    """An initialised renderer whose GL is this queue, and no context anywhere.

    The current context is replaced rather than created because the question is
    what happens when a driver refuses, and a working machine's driver does not.
    """
    monkeypatch.setattr(
        gl_cloud_view,
        "QOpenGLContext",
        SimpleNamespace(currentContext=lambda: SimpleNamespace(functions=lambda: errors)),
    )
    renderer = PointRenderer()
    renderer._program = cast(object, FakeProgram())  # type: ignore[assignment]
    renderer._positions = cast(  # type: ignore[assignment]
        object, FakeBuffer(errors, refuses=refuses if which == "positions" else None)
    )
    renderer._colours = cast(  # type: ignore[assignment]
        object, FakeBuffer(errors, refuses=refuses if which == "colours" else None)
    )
    renderer._position_attribute = 0
    renderer._colour_attribute = 1
    renderer._mvp_uniform = 2
    renderer._point_size_uniform = 3
    return renderer


def a_model(count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A tiny cloud and its colours, in the shape ``upload`` is given them."""
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    colours = np.full((count, 3), 200, dtype=np.uint8)
    return points, colours


def test_an_upload_the_driver_refused_is_a_fallback_not_a_blank_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode this check exists for, in full.

    ``allocate`` tells Qt nothing, so without asking the driver afterwards the
    renderer would count the upload, record sixty thousand points as resident,
    draw them out of a buffer that has no storage, and report success at every
    step while the operator looks at nothing.
    """
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors, refuses="allocate")
    points, colours = a_model()

    with pytest.raises(RendererUnavailable) as refusal:
        renderer.upload(points, colours)

    assert "position buffer allocation" in str(refusal.value)
    assert "0x0505" in str(refusal.value), "the driver's own code, for somebody to look up"
    assert renderer.uploads == 0, "a refused upload is not an upload"
    assert renderer.resident_points == 0


def test_a_refused_colour_allocation_is_reported_as_the_colour_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each allocation is asked about separately, or the reason names the wrong one."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors, refuses="allocate", which="colours")

    with pytest.raises(RendererUnavailable, match="colour buffer allocation"):
        renderer.upload(*a_model())

    assert renderer.uploads == 0


def test_an_error_left_behind_by_something_else_is_not_blamed_on_the_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue is emptied first, so a check is about the call it follows.

    Otherwise anything else in the process that ever provoked an error would
    send this viewer to software the next time the model changed.
    """
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)
    errors.refuse(0x0500)
    errors.refuse(0x0502)

    renderer.upload(*a_model())

    assert renderer.uploads == 1
    assert renderer.resident_points == 4


def test_a_draw_the_driver_refused_is_not_counted_as_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submission reports nothing either, and a rejected frame is a blank one."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)
    renderer.upload(*a_model())
    drawn = renderer.draws

    def refuse_the_draw(_mode: int, _first: int, _count: int) -> None:
        errors.refuse()

    monkeypatch.setattr(errors, "glDrawArrays", refuse_the_draw)
    with pytest.raises(RendererUnavailable, match="draw submission"):
        renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.draws == drawn, "a frame the driver rejected is not a frame"


def test_a_binding_the_driver_refused_is_reported_before_the_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointing an attribute at a buffer can fail on its own, and silently."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)
    renderer.upload(*a_model())
    renderer._positions = cast(  # type: ignore[assignment]
        object, FakeBuffer(errors, refuses="bind")
    )

    with pytest.raises(RendererUnavailable, match="binding the resident geometry"):
        renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert errors.drawn == [], "nothing is submitted against geometry that would not bind"


def test_a_frame_the_driver_accepted_is_counted_and_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the same contract: a working driver still just works."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)
    renderer.upload(*a_model())

    renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.draws == 1
    assert errors.drawn == [(0, 4)]
    assert errors.viewports == [(100, 80)]


# -- which backend was asked for -------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, AUTO),
        ("", AUTO),
        ("   ", AUTO),
        ("software", SOFTWARE),
        ("SOFTWARE", SOFTWARE),
        ("  OpenGL  ", OPENGL),
        ("auto", AUTO),
        ("gpu", AUTO),
        ("yes", AUTO),
    ],
)
def test_the_override_is_read_the_way_a_person_would_type_it(
    value: str | None, expected: str
) -> None:
    """It exists to get somebody out of trouble at a shell or in a desktop file.

    So it is case- and space-insensitive, and a value it does not recognise is
    "choose automatically" with a warning rather than a refusal to start: a
    viewer that will not open because of a typo in a troubleshooting variable is
    the opposite of what the variable is for.
    """
    environ = {} if value is None else {VIEWER_ENV: value}

    assert requested_backend(environ) == expected


# -- drawing a surface out of the same buffers ---------------------------------


def a_soup(faces: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A triangle soup and its colours, in the shape ``upload`` is given them."""
    points = np.arange(faces * 9, dtype=np.float32).reshape(faces * 3, 3)
    colours = np.full((faces * 3, 3), 180, dtype=np.uint8)
    return points, colours


def test_a_surface_is_the_same_buffers_and_a_different_draw_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole reason the lighting is baked into the colours on the CPU: no
    second program, no normal attribute, no index buffer - one enumerant."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)

    renderer.upload(*a_soup(3), triangles=True)
    renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.resident_primitive == gl_cloud_view.GL_TRIANGLES
    assert renderer.resident_points == 9
    assert errors.modes == [gl_cloud_view.GL_TRIANGLES]
    assert errors.drawn == [(0, 9)]


def test_points_are_still_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default did not move: everything that drew points still draws them."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)

    renderer.upload(*a_model())
    renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.resident_primitive == gl_cloud_view.GL_POINTS
    assert errors.modes == [gl_cloud_view.GL_POINTS]


def test_a_soup_that_is_not_whole_triangles_is_refused_before_it_is_uploaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial triangle draws a garbage face out of whatever follows it, which
    is a shape the reconstruction never produced."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    colours = np.full((4, 3), 180, dtype=np.uint8)

    with pytest.raises(RendererUnavailable, match="multiple of three"):
        renderer.upload(points, colours, triangles=True)

    assert renderer.uploads == 0
    assert renderer.resident_points == 0


def test_switching_back_to_points_switches_the_draw_call_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The toggle in the panel is one re-upload, and the primitive travels with it."""
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)

    renderer.upload(*a_soup(2), triangles=True)
    renderer.upload(*a_model())
    renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.resident_primitive == gl_cloud_view.GL_POINTS
    assert errors.modes == [gl_cloud_view.GL_POINTS]


def test_an_empty_surface_leaves_nothing_resident_and_nothing_drawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = ErrorQueue()
    renderer = with_driver(monkeypatch, errors)

    renderer.upload(
        np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), triangles=True
    )
    renderer.draw(np.eye(4, dtype=np.float32), 100, 80, 1.0)

    assert renderer.resident_points == 0
    assert errors.drawn == []
    assert renderer.draws == 1
