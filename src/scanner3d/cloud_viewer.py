"""Which renderer the model panel gets, and what happens when it cannot have one.

Why this is a separate thing from either renderer
-------------------------------------------------
Because the decision has to survive being wrong. The graphics card cannot be
asked whether it will work until a context has been created, a shader compiled
and a buffer allocated - all of which happen inside a widget that is already on
the screen. So "use the GPU" is not a choice made once at startup; it is a choice
made optimistically and then possibly taken back, with the model that was being
shown carried across.

That is what this module is: one widget that stands where the viewer stands,
mounts a backend inside itself, and can swap it. Neither renderer knows about the
other, and neither of them decides which is mounted.

How a person overrides it
-------------------------
``SCANNER3D_VIEWER=software`` forces the CPU path, ``SCANNER3D_VIEWER=opengl``
demands the graphics card, and anything else - including the variable being
unset - is ``auto``, which is prefer the card and fall back. It follows the same
convention as ``SCANNER3D_SERIAL`` and ``SCANNER3D_CAPTURES_DIR`` for the same
reason: this application is launched from a desktop menu as often as from a
shell, and an environment variable is something that can be set in a ``.desktop``
file without editing code.

``opengl`` does NOT mean "crash if it fails". It means the fallback is reported
as the failure it is rather than as a normal automatic choice - the application
still opens, because a viewer that will not start teaches the owner nothing that
a viewer saying why it fell back does not.

Why the backend is said out loud
--------------------------------
The owner's question was whether this machine is actually using its RX 9070 XT.
Nothing about a smooth picture answers that - a fast enough CPU path looks the
same until the cloud grows - so the answer is written where it can be read: in
the model panel, and in the log the application always keeps.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Final, cast

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from scanner3d.cloud_view import CloudCamera, CloudView
from scanner3d.logs import get_logger

LOG = get_logger("cloud_viewer")

#: The override, read from the environment.
VIEWER_ENV: Final = "SCANNER3D_VIEWER"

#: Prefer the graphics card, fall back silently-but-loggedly if it cannot.
AUTO: Final = "auto"

#: The CPU renderer, chosen deliberately. For troubleshooting: it is the way to
#: find out whether a display problem is the driver's or this application's.
SOFTWARE: Final = "software"

#: The graphics card, demanded. A fallback from here is reported as a failure.
OPENGL: Final = "opengl"

#: Every value :data:`VIEWER_ENV` accepts, and what it means.
MODES: Final = (AUTO, SOFTWARE, OPENGL)


def requested_backend(environ: Mapping[str, str] | None = None) -> str:
    """Which backend the environment asks for, as one of :data:`MODES`.

    An unrecognised value is ``auto`` with a warning rather than an error. The
    variable exists to get somebody out of trouble; refusing to start because
    they typed ``gpu`` would be the opposite of that, and the warning is in the
    log they would already be reading.
    """
    values = os.environ if environ is None else environ
    raw = values.get(VIEWER_ENV, "").strip().lower()
    if not raw:
        return AUTO
    if raw not in MODES:
        LOG.warning(
            "%s=%r is not one of %s; choosing the renderer automatically",
            VIEWER_ENV,
            raw,
            ", ".join(MODES),
        )
        return AUTO
    return raw


def _build_gl_view() -> CloudCamera:
    """The OpenGL widget, imported at the moment it is wanted.

    Late because :mod:`PySide6.QtOpenGLWidgets` is a separate extension module
    that a stripped or unusual PySide6 build can be missing, and "the GL module
    would not import" is one more thing that has to become a fallback rather
    than a window that does not open.
    """
    from scanner3d.gl_cloud_view import GLCloudView

    return GLCloudView()


def _widget(view: CloudCamera) -> QWidget:
    """A backend as the widget it also is.

    Both renderers are a :class:`~scanner3d.cloud_view.CloudCamera` mixed in
    front of a ``QWidget``, and this module needs each half of that at different
    moments: the camera half to show a model, the widget half to mount and
    unmount one. Stated once, here, rather than at every call site.
    """
    return cast(QWidget, view)


class CloudViewer(QWidget):
    """The model panel's viewer: one backend, mounted, and swappable.

    Forwards the small surface the window actually uses - :meth:`set_cloud`,
    :attr:`thin_fraction` and :attr:`orientation` - rather than exposing the
    mounted widget, so nothing outside can hold a reference to a backend that is
    about to be replaced.
    """

    #: Emitted whenever :attr:`backend_description` changes: at the moment the
    #: driver identifies itself, and at the moment a fallback happens.
    backend_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        factory: Callable[[], CloudCamera] = _build_gl_view,
    ) -> None:
        """``factory`` is the seam for tests and for nothing else.

        It exists because every interesting question here - what happens when
        the context cannot be created, when the shader will not link, when the
        module will not import - is a question about a failure that a working
        machine cannot produce on demand.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout
        self._factory = factory
        self._mode = requested_backend(environ)
        #: The last cloud handed to this viewer, kept so a backend swap does not
        #: blank the panel. Held as it arrived, before any decimation: replaying
        #: it is exactly the call the window made.
        self._last_cloud: tuple[np.ndarray, np.ndarray, np.ndarray | None, bool] | None = None
        #: The last surface handed to this viewer, kept for the same reason and
        #: separately, because a surface arrives on its own schedule: a capture
        #: can produce several models before its grid holds a surface worth
        #: marching, and a reconstruction produces both at once.
        self._last_surface: tuple[np.ndarray, np.ndarray] | None = None
        #: Which of the two pictures the operator has asked for. Held here as
        #: well as on the backend so a swap restores the choice rather than
        #: silently reverting to the default.
        self._show_surface = True
        self._backend_description = ""
        #: The GL backend this viewer connected to, while it is still the
        #: mounted one. Compared against on a failure report so a report that
        #: was queued before a swap cannot arrive afterwards and tear out the
        #: renderer that replaced it.
        self._gl_backend: CloudCamera | None = None
        if self._mode == SOFTWARE:
            self._view: CloudCamera = CloudView()
            layout.addWidget(_widget(self._view))
            self._describe(f"software, by {VIEWER_ENV}={SOFTWARE}")
            return
        try:
            self._view = self._factory()
        except Exception as exc:  # an unbuildable backend is a fallback, not a crash
            self._view = CloudView()
            layout.addWidget(_widget(self._view))
            self._describe(self._fallback_description(f"the viewer could not be built: {exc}"))
            return
        layout.addWidget(_widget(self._view))
        self._gl_backend = self._view
        # Both outcomes, connected before the widget can be shown: the driver
        # identifying itself, and the driver refusing.
        self._view.ready.connect(self._gl_ready)  # type: ignore[attr-defined]
        # QUEUED, and this is the whole reason the swap is safe. A driver failure
        # is discovered inside initializeGL or paintGL - that is, while Qt is
        # part-way through a rendering callback ON the widget that fall_back
        # unparents, hides and schedules for deletion. Doing that to a widget Qt
        # is still executing is a use-after-free waiting for the right timing.
        # A queued connection puts the replacement back on the event loop, so the
        # callback has returned before the widget it belongs to is taken away.
        self._view.unavailable.connect(  # type: ignore[attr-defined]
            self.fall_back, Qt.ConnectionType.QueuedConnection
        )
        self._describe("OpenGL, waiting for the driver to answer")

    # -- what is drawing ---------------------------------------------------
    @property
    def backend_description(self) -> str:
        """One line naming the active renderer, and the reason if it is a fallback."""
        return self._backend_description

    @property
    def view(self) -> CloudCamera:
        """The mounted backend. For tests and diagnostics, not for holding on to."""
        return self._view

    @property
    def is_software(self) -> bool:
        """Whether the CPU renderer is what is currently mounted."""
        return isinstance(self._view, CloudView)

    # -- the window's surface ----------------------------------------------
    def set_cloud(
        self,
        points: np.ndarray,
        thin: np.ndarray,
        newest: np.ndarray | None = None,
        *,
        fit: bool = False,
    ) -> None:
        """Show a model, and remember it in case the backend has to change."""
        self._last_cloud = (points, thin, newest, fit)
        self._view.set_cloud(points, thin, newest, fit=fit)

    def set_surface(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Show a lit surface over the model, and remember it for a backend change."""
        self._last_surface = (vertices, faces)
        self._view.set_surface(vertices, faces)

    def clear_surface(self) -> None:
        """Forget the surface, here and on the mounted backend."""
        self._last_surface = None
        self._view.clear_surface()

    def set_show_surface(self, showing: bool) -> None:
        """Choose between the lit surface and the coverage-coloured points."""
        self._show_surface = showing
        self._view.set_show_surface(showing)

    @property
    def has_surface(self) -> bool:
        """Whether a surface has been given to this viewer."""
        return self._view.has_surface

    @property
    def has_cloud(self) -> bool:
        """Whether measured points have been given to this viewer."""
        return self._view.has_cloud

    @property
    def showing_surface(self) -> bool:
        """Whether the surface is what is actually on the screen."""
        return self._view.showing_surface

    @property
    def thin_fraction(self) -> float:
        """The fraction of the drawn points that still need scanning."""
        return self._view.thin_fraction

    @property
    def orientation(self) -> np.ndarray:
        """A copy of the accumulated world-to-camera rotation."""
        return self._view.orientation

    # -- swapping ----------------------------------------------------------
    def _gl_ready(self, summary: str) -> None:
        self._describe(f"GPU / OpenGL: {summary}")

    def fall_back(self, reason: str) -> None:
        """Replace the mounted backend with the software one, keeping the model.

        Public because the GL widget reports its failure by signal and this is
        what the signal is connected to - by a QUEUED connection, so this runs
        after Qt's rendering callback has returned rather than inside it.

        Once only, and only for the backend that is still mounted. A driver that
        fails in one paint fails in every later one, so several reports can be
        in flight at the moment the first one is acted on; every one of them
        after the first names a widget that has already been unparented and
        scheduled for deletion, and acting on it would replace a renderer that
        did nothing wrong - or free one twice.
        """
        if self.is_software or self._view is not self._gl_backend:
            return
        self._gl_backend = None
        old = self._view
        self._layout.removeWidget(_widget(old))
        _widget(old).setParent(None)
        _widget(old).deleteLater()

        view = CloudView()
        self._view = view
        self._layout.addWidget(view)
        # The surface first only in the sense that both are restored: set_cloud
        # is what frames a view that has never been framed, and _adopt_camera
        # undoes that framing afterwards, so the order between the two model
        # calls does not matter and the order against the camera does.
        view.set_show_surface(self._show_surface)
        if self._last_cloud is not None:
            points, thin, newest, fit = self._last_cloud
            view.set_cloud(points, thin, newest, fit=fit)
        if self._last_surface is not None:
            view.set_surface(*self._last_surface)
        if self._last_cloud is not None:
            self._adopt_camera(old, view)
        self._describe(self._fallback_description(reason))

    def _fallback_description(self, reason: str) -> str:
        """What the panel says about a fallback, including whether it was asked for.

        ``SCANNER3D_VIEWER=opengl`` does not make this fatal - the application
        still opens - but it does change what the sentence means, from a normal
        automatic choice into the failure of something the operator demanded.
        """
        if self._mode == OPENGL:
            return (
                f"software, although {VIEWER_ENV}={OPENGL} asked for the graphics card "
                f"(OpenGL unavailable: {reason})"
            )
        return f"software (OpenGL unavailable: {reason})"

    @staticmethod
    def _adopt_camera(old: CloudCamera, new: CloudView) -> None:
        """Carry the pose across, so a fallback does not also re-frame the object.

        Applied AFTER the model, because ``set_cloud`` frames the first cloud a
        view has ever been given and would otherwise undo this.
        """
        if not old._framed:
            return
        new._rotation = old._rotation.copy()
        new._centre = np.array(old._centre)
        new._distance = old._distance
        new._pan = np.array(old._pan)
        new._framed = True

    def _describe(self, description: str) -> None:
        self._backend_description = description
        LOG.info("model view backend: %s", description)
        self.backend_changed.emit(description)
