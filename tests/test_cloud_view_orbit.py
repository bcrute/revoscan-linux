"""Turning the model: every direction, all the way round, and still a rotation.

Two complaints, one cause. The object came up upside down, and it could not be
turned over to look at - the orbit was a yaw and a pitch, and the pitch was
clamped just short of vertical because at vertical it stopped responding
altogether. So the top and the bottom of a scan could be approached and never
passed, and "let me see the underside" was not a thing this view could do.

The orbit is now an accumulated 3x3 rotation with no angle in it to wrap and no
pole to avoid, and the points it is given are already the right way up. This
module is about what that buys and what it costs: that a drag passes straight
through where the clamp used to be, that a full revolution comes back to where
it started, that roll exists, and that after thousands of drags the matrix is
still a rotation rather than a slowly accumulating shear.

Nothing here builds a widget. The orbit lives in
:class:`~scanner3d.cloud_view.CloudCamera`, which both renderers share and which
is deliberately not a widget, so it is subclassed directly and the widget's
methods are called unbound against it - for the reason :mod:`tests.gui_harness`
gives: a Qt event loop answers none of these questions and would make the tests
unrunnable on a machine with no display server. It also means these answers
belong to the GPU viewer and the software one alike, because there is one copy
of them.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt

from scanner3d.cloud_view import DRAG_RADIANS_PER_PIXEL, CloudCamera, CloudView
from scanner3d.reconstruction_export import UPRIGHT_FROM_OPTICAL

#: The pitch limit the old orbit clamped at, in radians. Kept as a number in
#: this module because what it names no longer exists in the widget: the tests
#: below are about reaching past it, so it has to be written down somewhere.
OLD_PITCH_CLAMP = 1.55

WIDTH = 200
HEIGHT = 200


class Viewport(CloudCamera):
    """The REAL camera - state, orbit and all - with no Qt behind it.

    :class:`~scanner3d.cloud_view.CloudCamera` is deliberately not a widget, so
    everything these tests are about can be built without a display server. What
    is missing is only the drawing, and the one widget method the camera calls
    into: ``update``, counted here instead of scheduling a repaint.
    """

    def __init__(self) -> None:
        self._init_camera()
        self.updates = 0

    def update(self, *_args: object) -> None:  # Qt's name
        self.updates += 1


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


def drag(
    view: Viewport,
    steps: list[tuple[float, float]],
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Press, move through ``steps`` as pointer positions, release."""
    widget = cast(CloudView, view)
    CloudView.mousePressEvent(widget, cast(object, Move(0.0, 0.0)))  # type: ignore[arg-type]
    for x, y in steps:
        CloudView.mouseMoveEvent(widget, cast(object, Move(x, y, button, modifiers)))  # type: ignore[arg-type]
    CloudView.mouseReleaseEvent(widget, cast(object, Move(0.0, 0.0)))  # type: ignore[arg-type]


def turn(view: Viewport, dx: float, dy: float, steps: int = 1, **kwargs: object) -> None:
    """Drag ``dx`` by ``dy`` pixels in ``steps`` equal moves."""
    path = [(dx * (index + 1) / steps, dy * (index + 1) / steps) for index in range(steps)]
    drag(view, path, **kwargs)  # type: ignore[arg-type]


def show(view: Viewport, points: np.ndarray, *, fit: bool = False) -> None:
    CloudView.set_cloud(cast(CloudView, view), points, np.zeros(len(points), dtype=bool), fit=fit)


def screen_of(view: Viewport, point: tuple[float, float, float]) -> tuple[int, int]:
    """Where one point lands, with the framing the view already has."""
    view._points = np.array([point], dtype=np.float32)
    view._colours = np.zeros((1, 3), dtype=np.uint8)
    px, py, _colours = CloudView.projected(cast(CloudView, view), WIDTH, HEIGHT)
    assert len(px) == 1, "the point fell outside the window"
    return int(px[0]), int(py[0])


def upright(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """One point of the depth camera's optical frame, as it is exported."""
    return UPRIGHT_FROM_OPTICAL.apply(point)


def is_a_rotation(matrix: np.ndarray) -> bool:
    return (
        bool(np.all(np.isfinite(matrix)))
        and bool(np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9))
        and float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-9)
    )


def framed_view() -> Viewport:
    """A view framed on a 20-unit cube around a point 120 in front of the camera."""
    view = Viewport()
    corners = np.array(
        [upright((x, y, 120.0 + z)) for x in (-10, 10) for y in (-10, 10) for z in (-10, 10)],
        dtype=np.float32,
    )
    show(view, corners, fit=True)
    return view


# -- which way up ----------------------------------------------------------


def test_the_object_starts_upright_and_the_right_way_round() -> None:
    """The first thing shown, before anybody has touched the mouse.

    The scanner's own frame counts +y DOWN the image. Drawn as it stands, the
    top of the object is at the bottom of the window, which is what the owner
    reported. Exported through the upright frame and drawn by the identity
    orientation, the picture matches what the scanner saw: what was above the
    middle of the depth image is above the middle of the window, what was to the
    right is on the right, and it is not mirrored.
    """
    view = framed_view()
    assert np.array_equal(view._rotation, np.eye(3))

    # In the OPTICAL frame, -y is up the image and +y is down it.
    _, top = screen_of(view, upright((0.0, -10.0, 120.0)))
    _, bottom = screen_of(view, upright((0.0, 10.0, 120.0)))
    right, _ = screen_of(view, upright((10.0, 0.0, 120.0)))
    left, _ = screen_of(view, upright((-10.0, 0.0, 120.0)))

    assert top < HEIGHT // 2 < bottom
    assert left < WIDTH // 2 < right


def test_the_surface_nearest_the_scanner_is_the_one_you_see() -> None:
    """Depth order, which is the other half of "not inside out".

    Points are painted far first so the near ones land on top. With the depth
    sign the wrong way round the picture is of the BACK of the object seen
    through it, which for a single-view scan looks like a plausible object.
    """
    view = framed_view()
    near, far = upright((0.0, 0.0, 110.0)), upright((0.0, 0.0, 130.0))
    view._points = np.array([near, far], dtype=np.float32)
    view._colours = np.array([[10, 0, 0], [20, 0, 0]], dtype=np.uint8)

    _px, _py, colours = CloudView.projected(cast(CloudView, view), WIDTH, HEIGHT)

    # Far first, near last: the last thing painted is the closest to the scanner.
    assert [int(colour[0]) for colour in colours] == [20, 10]


# -- turning ----------------------------------------------------------------


def test_a_drag_passes_straight_through_where_the_pitch_used_to_be_clamped() -> None:
    """The complaint, as an assertion: the object can be turned right over.

    A drag of well past the old limit used to stop at it. Here it keeps going,
    and the proof is that what was the top of the object ends up at the bottom
    of the window - which is only reachable by passing through vertical.
    """
    view = framed_view()
    top = upright((0.0, -10.0, 120.0))
    _, before = screen_of(view, top)
    assert before < HEIGHT // 2

    # Half a turn, well past the old 1.55 rad clamp, in small steps.
    pixels = math.pi / DRAG_RADIANS_PER_PIXEL
    assert pixels * DRAG_RADIANS_PER_PIXEL > OLD_PITCH_CLAMP
    turn(view, 0.0, pixels, steps=60)

    _, after = screen_of(view, top)
    assert after > HEIGHT // 2
    assert is_a_rotation(view._rotation)


def test_a_full_revolution_comes_back_to_where_it_started() -> None:
    """No wrap, no jump, no drift: a lap of dragging is the identity again.

    628 pixels rather than ``2 pi / 0.01`` = 628.3, because a pointer lands on
    whole pixels: the widget is given integer positions, so a test that asked
    for a third of a pixel would be asserting against an input the widget can
    never receive. What is left is the 0.3 px the lap is short by, which is 0.2
    of a degree, and the tolerance below is exactly that and no larger.
    """
    view = framed_view()
    lap = 628

    turn(view, float(lap), 0.0, steps=200)

    assert is_a_rotation(view._rotation)
    assert np.allclose(view._rotation, np.eye(3), atol=4e-3)
    # And it got there by turning, not by wrapping: the accumulated matrix is
    # the analytic rotation for the pixels that were actually dragged.
    angle = lap * DRAG_RADIANS_PER_PIXEL
    about_y = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    assert np.allclose(view._rotation, about_y, atol=1e-9)


def test_the_orbit_reaches_every_direction() -> None:
    """Any face of the object can be brought to the front, from any other.

    A quarter turn down puts the world's up axis where the viewer is, which is
    the "look at it from directly above" the clamped pitch could only ever
    approach.
    """
    view = framed_view()

    # 157 whole pixels of drag, which is a quarter turn to within a thousandth
    # of a radian - a pointer cannot land on the 157.08th pixel.
    turn(view, 0.0, 157.0, steps=30)

    # World +y (up) has been turned onto the camera's own +z, which points at
    # the viewer: the top of the object is now facing them.
    assert np.allclose(view._rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-3)


def test_shift_and_left_drag_rolls_and_a_plain_drag_does_not() -> None:
    """Roll is the third degree of freedom, and it is a distinct gesture.

    Without it the object can be pointed in any direction and never turned in
    the plane of the window, so a long object cannot be laid flat across it.
    """
    view = framed_view()
    top = upright((0.0, -10.0, 120.0))
    column_before, row_before = screen_of(view, top)

    turn(view, 100.0, 0.0, steps=10, modifiers=Qt.KeyboardModifier.ShiftModifier)

    column_after, row_after = screen_of(view, top)
    assert column_after > column_before, "the top of the object should roll to the right"
    assert row_after > row_before, "and downwards as it goes"
    assert is_a_rotation(view._rotation)
    # A roll leaves the axis you are looking along alone.
    assert np.allclose(view._rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-9)

    # And the other half of the name: the SAME drag without the modifier is not
    # a roll. It moves the view axis, which is exactly what a roll does not do,
    # so the two gestures cannot be confused for one another.
    plain = framed_view()
    turn(plain, 100.0, 0.0, steps=10)

    assert is_a_rotation(plain._rotation)
    assert not np.allclose(plain._rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-3)
    # Horizontally, where a plain drag takes it, and not down the window.
    _column, row_plain = screen_of(plain, top)
    assert row_plain == pytest.approx(row_before, abs=1)


def test_the_orientation_is_still_a_rotation_after_thousands_of_drags() -> None:
    """Drift is not hypothetical: every drag multiplies the matrix again.

    Left alone the axes stop being perpendicular and stop being unit length, and
    a camera matrix that shears makes a scan look distorted - which is
    indistinguishable, to the person looking at it, from a scan that IS
    distorted.
    """
    view = framed_view()
    # Deterministic and deliberately awkward: two incommensurable step sizes so
    # the drags never repeat a cycle, and a roll every seventh one.
    for index in range(2000):
        modifiers = (
            Qt.KeyboardModifier.ShiftModifier if index % 7 == 0 else Qt.KeyboardModifier.NoModifier
        )
        turn(view, 13.0 * math.sin(index), 7.0 * math.cos(index * 0.5), modifiers=modifiers)

    assert is_a_rotation(view._rotation)


def test_the_orientation_is_only_ever_handed_out_as_a_copy() -> None:
    """The one thing that must not happen to it is being changed from outside."""
    view = framed_view()

    taken = CloudView.orientation.fget(cast(CloudView, view))  # type: ignore[attr-defined]
    taken[0][0] = 99.0

    assert view._rotation[0][0] == 1.0


# -- and the other two gestures, unchanged ----------------------------------


def test_middle_drag_still_pans_and_the_wheel_still_zooms() -> None:
    """The two gestures this change was not about, asserted so it was not about them."""
    view = framed_view()
    distance = view._distance

    turn(view, 30.0, -12.0, steps=3, button=Qt.MouseButton.MiddleButton)

    assert list(view._pan) == [30.0, -12.0]
    assert np.array_equal(view._rotation, np.eye(3)), "panning must not turn the object"

    CloudView.wheelEvent(cast(CloudView, view), cast(object, Wheel(1)))  # type: ignore[arg-type]
    assert view._distance == pytest.approx(distance * 0.9)
    CloudView.wheelEvent(cast(CloudView, view), cast(object, Wheel(-1)))  # type: ignore[arg-type]
    assert view._distance == pytest.approx(distance)
    for _ in range(200):
        CloudView.wheelEvent(cast(CloudView, view), cast(object, Wheel(1)))  # type: ignore[arg-type]
    assert view._distance == pytest.approx(1.0), "zoom stays in front of the object"


def test_a_new_cloud_never_turns_the_object_however_it_is_framed() -> None:
    """Which side you are standing on is a preference, not a property of the points.

    Framing is a boundary - a new capture, a registered reconstruction - and it
    moves the centre, the distance and the pan. It has never moved the orbit and
    it still does not.
    """
    view = framed_view()
    turn(view, 40.0, 25.0, steps=4)
    turned = view._rotation.copy()

    show(view, np.array([[0.0, 0.0, 0.0], [500.0, 500.0, 500.0]], dtype=np.float32), fit=True)
    show(view, np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32))

    assert np.array_equal(view._rotation, turned)
