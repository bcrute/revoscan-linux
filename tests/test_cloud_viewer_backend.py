"""Choosing a renderer, and surviving having chosen the wrong one.

Why this cannot be answered without widgets
-------------------------------------------
Everything else about the viewer is arithmetic and is asked without Qt. This is
not: the promise is that when the graphics card cannot draw, the panel the
operator is looking at gets a working software renderer IN PLACE of the broken
one, still showing the model it was showing and still pointing the same way. That
is a statement about a layout, and it is only true if a layout does it.

The application object is offscreen and nothing is ever shown, so no window
appears and the owner's desktop is untouched - see :mod:`tests.qt_offscreen`.

Why the failure is injected
---------------------------
A working machine cannot produce a broken driver on demand, and a test that only
ever ran on a broken one would never run. So the backend is supplied through
:class:`~scanner3d.cloud_viewer.CloudViewer`'s factory seam and made to fail the
two ways a real one can: refusing to be built at all, which is a PySide6 with no
OpenGL module, and building and then reporting that it cannot draw, which is
every driver, shader and buffer failure there is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget

from scanner3d.cloud_view import THIN_COLOUR, CloudCamera, CloudView
from scanner3d.cloud_viewer import OPENGL, SOFTWARE, VIEWER_ENV, CloudViewer
from scanner3d.gl_cloud_view import (
    CONTEXT_CHECK_ATTEMPTS,
    GLCloudView,
    GLDescription,
)
from scanner3d.gui import ScannerWindow
from scanner3d.preview_provenance import ModelPicture
from tests.qt_offscreen import qt_app

__all__ = ["qt_app"]


class FakeGLView(CloudCamera, QWidget):
    """A backend shaped exactly like the real GPU one, and as controllable.

    The same two signals, the same shared camera, and no driver: what the viewer
    does with a backend is the question, and a real context would answer it only
    on machines that have one.
    """

    ready = Signal(str)
    unavailable = Signal(str)

    def __init__(self) -> None:
        QWidget.__init__(self)
        self._init_camera()


def unbuildable() -> CloudCamera:
    """A PySide6 with no ``QtOpenGLWidgets``, as this module can produce one."""
    raise ImportError("No module named 'PySide6.QtOpenGLWidgets'")


def deliver(qt_app) -> None:
    """Let the event loop run the work a queued connection put on it.

    The fallback is deliberately NOT synchronous: a real backend reports its
    failure from inside ``initializeGL`` or ``paintGL``, and replacing the
    widget Qt is part-way through rendering is a use-after-free. So the swap is
    queued, and a test that never turned the loop over would be asserting the
    unsafe behaviour rather than the shipped one.
    """
    qt_app.processEvents()


def a_model(count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [[float(index), 1.0, 100.0 + index] for index in range(count)], dtype=np.float32
    )
    thin = np.arange(count) % 2 == 0
    return points, thin


# -- choosing --------------------------------------------------------------


def test_the_graphics_card_is_what_you_get_by_default(qt_app) -> None:
    """Nothing set: prefer the card. The owner's machine has one and should use it."""
    viewer = CloudViewer(environ={}, factory=FakeGLView)

    assert isinstance(viewer.view, FakeGLView)
    assert viewer.is_software is False


def test_the_documented_override_gets_the_software_renderer_and_never_builds_the_other(
    qt_app,
) -> None:
    """The troubleshooting escape hatch, and it has to be a real escape.

    A backend that was still constructed and then discarded would still have
    created a context - which is precisely the thing somebody setting this
    variable is trying to stop happening.
    """
    built = []

    def factory() -> CloudCamera:
        built.append(True)
        return FakeGLView()

    viewer = CloudViewer(environ={VIEWER_ENV: SOFTWARE}, factory=factory)

    assert built == []
    assert isinstance(viewer.view, CloudView)
    assert viewer.is_software is True
    assert SOFTWARE in viewer.backend_description


def test_the_active_backend_is_said_out_loud_when_the_driver_answers(qt_app) -> None:
    """The owner's actual question: is this machine using its graphics card.

    Answered with the driver's own words rather than with "GPU", because "GPU"
    is what a viewer would say whether or not it were true.
    """
    said: list[str] = []
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    viewer.backend_changed.connect(said.append)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)

    backend.ready.emit("Radeon RX 9070 XT - 4.6 (Compatibility Profile) Mesa 25.2.8")

    assert "Radeon RX 9070 XT" in viewer.backend_description
    assert "OpenGL" in viewer.backend_description
    assert said == [viewer.backend_description]


# -- surviving -------------------------------------------------------------


def test_a_backend_that_will_not_build_leaves_a_working_viewer(qt_app) -> None:
    """The application still opens, and the panel still draws. With the reason."""
    viewer = CloudViewer(environ={}, factory=unbuildable)

    assert isinstance(viewer.view, CloudView)
    assert "OpenGL unavailable" in viewer.backend_description
    assert "QtOpenGLWidgets" in viewer.backend_description


def test_a_driver_that_gives_up_is_replaced_under_the_model_being_shown(qt_app) -> None:
    """The whole promise of the fallback, in one test.

    The model does not blank, the pose the operator dragged to is still the pose,
    and the coverage colours still mean what they meant. A fallback that reset
    the view would be a fallback the operator sees as the application losing
    their place.
    """
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    points, thin = a_model()
    viewer.set_cloud(points, thin, fit=True)
    backend._distance = 61.0
    backend._pan[:] = (13.0, -4.0)
    turned = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    backend._rotation = turned

    backend.unavailable.emit("shader program would not link: syntax error")
    assert viewer.view is backend, "the reporting widget must outlive the callback it reported from"
    deliver(qt_app)

    replacement = viewer.view
    assert isinstance(replacement, CloudView)
    assert np.array_equal(replacement._points, points)
    assert tuple(replacement._colours[0]) == THIN_COLOUR
    assert viewer.thin_fraction == pytest.approx(0.5)
    assert np.array_equal(viewer.orientation, turned)
    assert replacement._distance == pytest.approx(61.0)
    assert list(replacement._pan) == [13.0, -4.0]
    assert "syntax error" in viewer.backend_description


def test_the_fallback_happens_once_however_often_the_driver_complains(qt_app) -> None:
    """A driver that fails in one paint fails in every later one.

    Swapping the widget out from under a second report would be replacing a
    renderer that has already been deleted.
    """
    said: list[str] = []
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    viewer.backend_changed.connect(said.append)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)

    backend.unavailable.emit("first")
    deliver(qt_app)
    first = viewer.view
    backend.unavailable.emit("second")
    deliver(qt_app)

    assert viewer.view is first
    assert said == [viewer.backend_description]
    assert "first" in viewer.backend_description


def test_the_replacement_waits_for_the_rendering_callback_to_return(qt_app) -> None:
    """The correction this round is about, stated as behaviour.

    A driver gives up inside ``paintGL``. At that instant Qt is executing the GL
    widget, and the fallback unparents it, hides it and schedules its deletion.
    Doing that from inside the callback is a use-after-free that shows up as a
    crash on somebody else's machine, so the report is queued: the widget is
    still mounted when the emit returns, and is replaced once the loop runs.
    """
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)

    backend.unavailable.emit("the driver reset while drawing")

    assert viewer.view is backend, "the widget Qt is drawing must not be taken away yet"
    assert "OpenGL unavailable" not in viewer.backend_description

    deliver(qt_app)

    assert viewer.is_software
    assert "the driver reset while drawing" in viewer.backend_description


def test_several_reports_queued_before_the_swap_still_swap_once(qt_app) -> None:
    """A driver that fails once fails on every later paint.

    So by the time the first queued report is acted on there can be more behind
    it, every one of them naming a widget that has already been unparented and
    scheduled for deletion. Acting on the second would replace a renderer that
    did nothing wrong - and would free the first one twice.
    """
    said: list[str] = []
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    viewer.backend_changed.connect(said.append)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)

    backend.unavailable.emit("first")
    backend.unavailable.emit("second")
    backend.unavailable.emit("third")
    deliver(qt_app)

    assert viewer.is_software is True
    assert said == [viewer.backend_description]
    assert "first" in viewer.backend_description


def test_a_report_that_arrives_after_the_swap_leaves_the_new_backend_alone(qt_app) -> None:
    """The stale report, arriving late, must not tear out its own replacement."""
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    viewer.fall_back("the shader would not link")
    replacement = viewer.view

    backend.unavailable.emit("and it is still broken")
    deliver(qt_app)

    assert viewer.view is replacement
    assert "the shader would not link" in viewer.backend_description


def test_a_demanded_graphics_card_that_fails_says_so_and_still_opens(qt_app) -> None:
    """``opengl`` is not "crash if it fails".

    An operator who set it wants to know it did not happen; an application that
    refused to start would tell them that and nothing else, and would take the
    log panel away with it.
    """
    viewer = CloudViewer(environ={VIEWER_ENV: OPENGL}, factory=unbuildable)

    assert isinstance(viewer.view, CloudView)
    assert f"{VIEWER_ENV}={OPENGL}" in viewer.backend_description
    assert "OpenGL unavailable" in viewer.backend_description


def test_a_model_that_arrives_after_the_fallback_is_still_drawn(qt_app) -> None:
    """The swap is not a one-shot rescue; the panel goes on being the panel."""
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    backend.unavailable.emit("the driver would not create a vertex buffer")
    deliver(qt_app)

    points, thin = a_model(count=6)
    viewer.set_cloud(points, thin)

    assert np.array_equal(viewer.view._points, points)
    assert viewer.thin_fraction == pytest.approx(0.5)


# -- the failure with no callback ------------------------------------------


class Stalled(GLCloudView):
    """A real GL widget with the two things this machine will not do on demand.

    Whether it is on screen, and whether Qt manages to give it a context, are
    exactly the conditions the post-show check reads - and a working desktop
    cannot be made to fail at the second one. So they are supplied, and
    everything else about the widget, including the check being tested, is the
    shipped code.
    """

    def __init__(
        self,
        *,
        visible: bool = True,
        context: object | None = None,
        initialises: bool = False,
    ) -> None:
        super().__init__()
        self._visible = visible
        self._context = context
        self._initialises = initialises
        self.forced = 0
        self.reported: list[str] = []
        self.unavailable.connect(self.reported.append)

    def isVisible(self) -> bool:  # Qt's name
        return self._visible

    def context(self) -> object | None:  # type: ignore[override]  # Qt's name
        return self._context

    def _force_initialisation(self) -> None:
        """The nudge, with the driver's answer decided by the test."""
        self.forced += 1
        if self._initialises:
            self._description = GLDescription("Test", "Renderer", "4.6", "4.60")


def a_valid_context() -> SimpleNamespace:
    """A context Qt did create, belonging to a widget that has not drawn yet."""
    return SimpleNamespace(isValid=lambda: True)


def test_being_shown_starts_the_check_for_a_context_that_never_arrives(qt_app) -> None:
    """Nothing else in the widget would ever notice this failure.

    ``initializeGL`` raising and ``paintGL`` raising are both reports from code
    that RAN. The blank panel is the case where neither is ever called, so the
    only way to learn about it is to look after being shown.
    """
    widget = Stalled(visible=False)

    widget.showEvent(QShowEvent())

    assert widget._context_checks == 1
    assert widget.failure is None, "being shown is not itself a verdict"


def test_a_widget_that_is_not_on_screen_yet_is_early_and_not_broken(qt_app) -> None:
    """The false fallback this check must not produce.

    Qt creates the context on the first paint after exposure, so a widget that
    has been constructed and not yet drawn has simply not been asked anything.
    Falling back here would send every machine to software for being built
    before it was shown - which is every machine.
    """
    widget = Stalled(visible=False)

    widget._check_context()

    assert widget.failure is None
    assert widget.reported == []
    assert widget.forced == 0, "an unexposed widget must not be nudged"
    assert widget._context_checks == 1, "and it must be looked at again"


def test_a_widget_that_initialises_when_nudged_is_not_a_fallback(qt_app) -> None:
    """The ordinary machine: the check forces the context and finds it working."""
    widget = Stalled(context=a_valid_context(), initialises=True)

    widget._check_context()

    assert widget.forced == 1
    assert widget.failure is None
    assert widget.reported == []


def test_a_mounted_widget_with_no_context_is_reported_rather_than_left_blank(qt_app) -> None:
    """Qt says it on stderr and carries on; the operator sees an empty panel.

    That is the whole defect: a widget which is mounted, laid out, and will
    never draw anything. It has to become the same visible fallback every other
    driver failure becomes, with a reason a person can read.
    """
    widget = Stalled(context=None)

    widget._check_context()

    assert widget.forced == 1
    assert widget.failure is not None
    assert "context" in widget.failure
    assert widget.reported == [widget.failure]


def test_an_invalid_context_is_the_same_answer_as_no_context(qt_app) -> None:
    """A context object that exists and is not valid draws exactly as much."""
    widget = Stalled(context=SimpleNamespace(isValid=lambda: False))

    widget._check_context()

    assert widget.failure is not None
    assert widget.reported == [widget.failure]


def test_the_wait_for_an_initialisation_that_never_happens_is_bounded(qt_app) -> None:
    """A context that exists but never initialises must not be waited on forever.

    Each look is cheap and none of them is a verdict on its own; what makes this
    safe is that there are finitely many, after which the panel stops being
    blank and starts being the software renderer.
    """
    widget = Stalled(context=a_valid_context())

    for _ in range(CONTEXT_CHECK_ATTEMPTS + 2):
        widget._check_context()

    assert widget.failure is not None
    assert "not initialised" in widget.failure
    assert widget.reported == [widget.failure], "once, whatever else happens"
    assert widget._context_checks == CONTEXT_CHECK_ATTEMPTS


def test_a_widget_that_already_failed_is_not_reported_twice_by_the_check(qt_app) -> None:
    """The check is one more way to notice, not one more thing to announce."""
    widget = Stalled(context=None)
    widget._check_context()
    first = widget.failure

    widget._check_context()

    assert widget.reported == [first]


def test_a_blank_panel_becomes_the_software_renderer_through_the_viewer(qt_app) -> None:
    """End to end: the check reports, the report is queued, the panel works."""
    widget = Stalled(context=None)
    viewer = CloudViewer(environ={}, factory=lambda: widget)
    points, thin = a_model()
    viewer.set_cloud(points, thin, fit=True)

    widget._check_context()
    deliver(qt_app)

    assert viewer.is_software is True
    assert "OpenGL unavailable" in viewer.backend_description
    assert np.array_equal(viewer.view._points, points)


# -- and where the owner reads it ------------------------------------------


class Panel:
    """:class:`~scanner3d.gui.ScannerWindow`'s reporting surface, and no more.

    Called unbound for the reason :mod:`tests.gui_harness` gives: what the window
    SAYS about its renderer is answered by its own logic, and an event loop would
    only make the question unaskable without a display server.
    """

    def __init__(self, *, software: bool) -> None:
        self.label = ""
        self.report: list[str] = []
        #: What the panel says the picture IS. Carried because a fallback has to
        #: leave that sentence describing the replacement rather than the
        #: renderer that failed, and this stand-in is where the window's own
        #: reporting is asked about.
        self.provenance = ""
        #: What the surface control is CALLED and what it says when hovered.
        #: The same refresh writes them, from the surface the panel holds, so a
        #: stand-in without them would not run the real method at all.
        self.control_title = ""
        self.control_tooltip = ""
        self._viewer_backend = SimpleNamespace(setText=self._set)
        self._model_provenance = SimpleNamespace(setText=self._say)
        self._surface_toggle = SimpleNamespace(setText=self._name, setToolTip=self._hover)
        self._surface_evidence = None
        self._cloud = SimpleNamespace(is_software=software, showing_surface=False, has_cloud=False)

    def _set(self, text: str) -> None:  # Qt's name, on the stand-in label
        self.label = text

    def _say(self, text: str) -> None:  # Qt's name, on the provenance label
        self.provenance = text

    def _name(self, text: str) -> None:  # Qt's name, on the stand-in control
        self.control_title = text

    def _hover(self, text: str) -> None:  # Qt's name, on the stand-in control
        self.control_tooltip = text

    def _mounted_picture(self) -> ModelPicture:
        """The REAL one, against this stand-in's viewer."""
        return ScannerWindow._mounted_picture(cast(ScannerWindow, self))

    def _refresh_model_provenance(self) -> None:
        """The REAL one, so the sentence is the window's rather than the test's."""
        ScannerWindow._refresh_model_provenance(cast(ScannerWindow, self))

    def _log(self, message: str) -> None:
        self.report.append(message)


def show_backend(panel: Panel, description: str) -> None:
    ScannerWindow._show_viewer_backend(cast(ScannerWindow, panel), description)


def test_the_window_names_the_renderer_beside_the_model_it_is_drawing() -> None:
    """Where the question is actually asked: under the picture, in words."""
    panel = Panel(software=False)

    show_backend(panel, "GPU / OpenGL: AMD Radeon RX 9070 XT - 4.6 Mesa 25.2.8")

    assert "AMD Radeon RX 9070 XT" in panel.label
    assert panel.report == [], "a working graphics card is not news for the results panel"


def test_a_fallback_is_also_put_where_results_are_read() -> None:
    """Because it is the one the operator did not choose and may want to act on."""
    panel = Panel(software=True)

    show_backend(panel, "software (OpenGL unavailable: the driver would not create a buffer)")

    assert "software" in panel.label
    assert len(panel.report) == 1
    assert "would not create a buffer" in panel.report[0]


def test_a_deliberate_software_choice_is_not_reported_as_a_problem() -> None:
    """Somebody who set the override already knows; saying it twice is noise."""
    panel = Panel(software=True)

    show_backend(panel, f"software, by {VIEWER_ENV}={SOFTWARE}")

    assert "software" in panel.label
    assert panel.report == []


def test_a_fallback_carries_the_surface_across_as_well_as_the_points(qt_app) -> None:
    """Both pictures survive the swap, and so does which of them was showing.

    A fallback that dropped the surface would be a fallback the operator sees as
    the scan losing its shape, and one that dropped the toggle would silently
    put back a preference they had turned off.
    """
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    points, thin = a_model()
    viewer.set_cloud(points, thin, fit=True)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    viewer.set_surface(vertices, faces)
    viewer.set_show_surface(False)

    backend.unavailable.emit("the driver stopped answering")
    deliver(qt_app)

    replacement = viewer.view
    assert isinstance(replacement, CloudView)
    assert viewer.has_surface
    # Off, because that is what was asked for before the swap.
    kept_off = viewer.showing_surface
    assert not kept_off
    viewer.set_show_surface(True)
    turned_on = viewer.showing_surface
    assert turned_on
    assert np.array_equal(replacement._surface, vertices)


def test_clearing_the_surface_survives_a_fallback_too(qt_app) -> None:
    """The withdrawal has to be remembered as well as the surface, or a swap
    resurrects a shape the capture had already said it no longer has."""
    viewer = CloudViewer(environ={}, factory=FakeGLView)
    backend = viewer.view
    assert isinstance(backend, FakeGLView)
    viewer.set_cloud(*a_model(), fit=True)
    viewer.set_surface(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    viewer.clear_surface()

    backend.unavailable.emit("the driver stopped answering")
    deliver(qt_app)

    assert isinstance(viewer.view, CloudView)
    assert not viewer.has_surface
