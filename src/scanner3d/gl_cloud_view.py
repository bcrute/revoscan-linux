"""The same picture, drawn by the graphics card.

What was actually slow
----------------------
Not the arithmetic, and not the frame rate the model was produced at. The
software viewer did the SAME work for every mouse event: transform sixty
thousand points, perspective-divide them, sort them by depth, allocate a
full-window canvas, clear it, scatter pixels into it, wrap it in a QImage and
hand it to Qt to copy. Measured on this host that is about 5.2 ms of projection
and sort plus about 3.6 ms of canvas for a 1200x800 panel, before composition -
all of it on the GUI thread, and all of it repeated per event rather than per
displayed frame, while the depth and colour previews wanted the same thread.

The fix is not a faster projection. It is that a drag does not change the
points: it changes a 3x3 orientation, a distance and two pan offsets. So the
points are uploaded ONCE, when :meth:`~scanner3d.cloud_view.CloudCamera.set_cloud`
is given a new model, and an orbit becomes sixteen floats and a draw call. The
per-frame cost stops scaling with the size of the cloud, which is the property
the CPU pipeline could not have at any constant factor.

Why the projection is written out here instead of reused
--------------------------------------------------------
:meth:`~scanner3d.cloud_view.CloudView.projected` decides what this view MEANS -
+y up, +z towards the viewer, the near surface in front, this focal length, this
pan convention - and a shader cannot call it. So :func:`camera_matrix` restates
it as the 4x4 that produces the same screen position for the same point, and the
tests hold the two to each other numerically rather than trusting that two
descriptions of one projection stayed in step. That is the whole reason the
matrix is built in numpy in a module-level function: it is checkable without a
context, a window or a display server.

Occlusion is the one thing that is not restated but replaced. The software path
paints far to near so the near points land on top; here there is a depth buffer,
which is the same answer arrived at without the sort - and the sort was a third
of the per-event cost.

Why it can fail, and what happens then
--------------------------------------
Context creation, a version that will not do, a shader that will not compile or
link, a buffer that will not allocate: all of them are real on somebody's
machine, and none of them may produce a blank panel. Every one of them raises
:class:`RendererUnavailable`, is logged with its reason, and is reported through
:attr:`GLCloudView.unavailable` so :mod:`scanner3d.cloud_viewer` can put the
software viewer in this widget's place with the model it was showing.

There is a fourth way, and it has no callback: Qt can fail to create the context
at all, in which case ``initializeGL`` and ``paintGL`` are NEVER called and the
widget sits there, mounted and blank, reporting nothing. So being shown starts a
bounded check that nudges the initialisation and, if there is still no valid
context, reports that as the same readable failure. It has to tell "not yet" from
"never": a widget that has been constructed and not yet painted is the normal
state of every machine for a moment, and falling back there would send them all
to software. The report itself is delivered through a queued connection, because
it is raised while Qt is part-way through the very widget the replacement
unparents and deletes.

Why the shaders are GLSL 1.20
-----------------------------
Because the point of the GPU path is that it works, not that it is modern.
1.20 with ``attribute``/``varying`` is what every desktop driver this could meet
compiles, including the compatibility profile Mesa hands out by default here; a
core-profile-only shader would turn a working machine into a fallback for
nothing gained. ``gl_PointSize`` needs ``GL_PROGRAM_POINT_SIZE`` enabled, which
is the same enumerant as the compatibility profile's
``GL_VERTEX_PROGRAM_POINT_SIZE``, so one call covers both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import (
    QMatrix4x4,
    QOpenGLContext,
    QOpenGLFunctions,
    QShowEvent,
    QSurfaceFormat,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from scanner3d.cloud_view import BACKGROUND, CloudCamera
from scanner3d.logs import get_logger

LOG = get_logger("gl_cloud_view")

#: The handful of GL enumerants this module needs. Written out rather than
#: imported from a GL binding: PySide6's :class:`QOpenGLFunctions` takes the
#: numbers, and adding PyOpenGL as a dependency to name six constants would put
#: a second GL loader in the process for no behaviour.
GL_POINTS: Final = 0x0000
GL_TRIANGLES: Final = 0x0004
GL_LESS: Final = 0x0201
GL_DEPTH_TEST: Final = 0x0B71
GL_FLOAT: Final = 0x1406
GL_VENDOR: Final = 0x1F00
GL_RENDERER: Final = 0x1F01
GL_VERSION: Final = 0x1F02
GL_SHADING_LANGUAGE_VERSION: Final = 0x8B8C
GL_PROGRAM_POINT_SIZE: Final = 0x8642
GL_COLOR_BUFFER_BIT: Final = 0x4000
GL_DEPTH_BUFFER_BIT: Final = 0x0100

#: Nothing closer than this to the camera is drawn, matching the software path's
#: ``depth > 1.0``. Stated once because both backends have to agree about which
#: points are in front of the viewer rather than behind them.
NEAR_PLANE: Final = 1.0

#: How much clearance the far plane is given behind the furthest point. A far
#: plane exactly at the object clips its own back surface as rounding decides.
FAR_MARGIN: Final = 1.0

#: Bits of depth buffer asked for. 24 is what every driver this could meet
#: offers, and 16 visibly z-fights at the depth ratios a close-up scan produces.
DEPTH_BUFFER_BITS: Final = 24

#: How long a mounted GL widget is given to produce a context before the panel
#: gives up on it. Qt creates the context lazily, on the first paint after the
#: widget is exposed, so "nothing has happened yet" is the NORMAL state for a
#: short while after the window is shown and must not be read as a failure.
CONTEXT_CHECK_MS: Final = 250

#: How many of those waits are allowed. Bounded on purpose: a widget that never
#: initialises has to become a visible fallback rather than a panel that stays
#: blank for as long as the application is open.
CONTEXT_CHECK_ATTEMPTS: Final = 8

#: The most consecutive stale GL errors drained before a call. A driver is
#: expected to have at most a handful queued; the loop is bounded so a broken
#: implementation that always reports an error cannot hang the GUI thread.
MAX_DRAINED_ERRORS: Final = 16

VERTEX_SHADER: Final = """
#version 120
attribute vec3 position;
attribute vec3 colour;
varying vec3 v_colour;
uniform mat4 mvp;
uniform float point_size;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = point_size;
    v_colour = colour;
}
"""

FRAGMENT_SHADER: Final = """
#version 120
varying vec3 v_colour;
void main() {
    gl_FragColor = vec4(v_colour, 1.0);
}
"""


class RendererUnavailable(RuntimeError):
    """The graphics card cannot draw this, and the reason a person can read.

    Carried all the way to the log and the model panel. "The 3D view fell back
    to software" answers nothing on its own; "vertex shader would not compile"
    with the driver's own message is something the owner can act on.
    """


@dataclass(frozen=True)
class GLDescription:
    """Who is actually drawing: the driver's own answer, not an assumption."""

    vendor: str
    renderer: str
    version: str
    shading_language: str

    def summary(self) -> str:
        """One line naming the hardware, for the log and the model panel."""
        return f"{self.renderer} - {self.version} (GLSL {self.shading_language}, {self.vendor})"


def camera_matrix(
    rotation: np.ndarray,
    centre: np.ndarray,
    distance: float,
    pan: np.ndarray,
    width: int,
    height: int,
    far: float,
) -> np.ndarray:
    """The 4x4 that puts a world point where the software viewer would put it.

    Read alongside :meth:`~scanner3d.cloud_view.CloudView.projected`, which this
    reproduces term for term:

    * the camera-space point is ``rotation @ (point - centre)``, so +x is right,
      +y is up and +z points at the viewer - a right-handed basis, and the thing
      that must not be got backwards, because a mirrored scan still looks like
      a scan;
    * it is then moved ``distance`` down the view axis, which makes the eye-space
      ``-z`` exactly the ``distance - z`` the software path divides by;
    * the focal length is ``0.9 * min(width, height)`` in the same pixels, and
      pan is added in pixels afterwards - hence the pan terms in the ``z``
      column, which is how "a constant number of pixels, whatever the depth"
      is expressed inside a projective divide;
    * ``y`` is negated relative to the screen because a screen row counts
      downwards, and here that falls out of the clip-space convention where
      ``+1`` is the TOP of the window.

    The depth column is the ordinary perspective one. It exists for the depth
    buffer's sake and it is monotonic in distance from the eye, which is the only
    property occlusion needs; the software path gets the same answer by painting
    far to near.

    ``far`` is passed in rather than derived because it depends on the model's
    extent, which is measured once when the model changes and not per frame.
    """
    field = 0.9 * min(width, height)
    near = NEAR_PLANE
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ np.asarray(centre, dtype=np.float64) - np.array([0.0, 0.0, distance])
    projection = np.array(
        [
            [2.0 * field / width, 0.0, -2.0 * float(pan[0]) / width, 0.0],
            [0.0, 2.0 * field / height, 2.0 * float(pan[1]) / height, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(projection @ view, dtype=np.float32)


def _colour_attributes(colours: np.ndarray) -> np.ndarray:
    """8-bit display colours as the 0..1 floats a vertex attribute carries.

    Converted here rather than uploaded as bytes because Qt's
    ``setAttributeBuffer`` cannot ask for a normalised integer attribute, and a
    shader that divided by 255 itself would put the coverage colours' definition
    in two places. At the point ceiling this is 720 KB, uploaded once per model.
    """
    return np.ascontiguousarray(colours, dtype=np.float32) / 255.0


def _current_functions(operation: str) -> QOpenGLFunctions:
    """The current context's entry points, or the reason there are none."""
    context = QOpenGLContext.currentContext()
    if context is None:
        raise RendererUnavailable(f"no OpenGL context is current for {operation}")
    return context.functions()


def _drain_gl_errors(functions: QOpenGLFunctions) -> None:
    """Empty the driver's error queue so the next check is about THIS call.

    ``glGetError`` returns one queued error per call and Qt does not clear the
    queue for us, so an error left behind by anything earlier in the process
    would otherwise be blamed on the operation that happens to look next.
    """
    for _ in range(MAX_DRAINED_ERRORS):
        if not functions.glGetError():
            return


def _raise_on_gl_error(functions: QOpenGLFunctions, operation: str) -> None:
    """Turn whatever the driver queued during ``operation`` into a fallback.

    The calls this guards - allocating a buffer, binding an attribute array,
    submitting a draw - return nothing Qt can check, so a driver that refuses
    one of them (out of memory is the realistic case) is silent. Without this
    the code would count the upload as done, record the points as resident, and
    leave the operator looking at an empty panel with nothing in the log.
    """
    error = functions.glGetError()
    if error:
        raise RendererUnavailable(f"OpenGL reported error 0x{error:04x} during {operation}")


class PointRenderer:
    """Shader, buffers and one draw call, against whatever context is current.

    Separate from the widget so the same code that the window runs can be driven
    on an offscreen surface: a smoke test that built its own second renderer
    would prove that the second one works. It touches no widget state and holds
    no camera - it is given a matrix and told to draw.
    """

    def __init__(self) -> None:
        self._program: QOpenGLShaderProgram | None = None
        self._positions: QOpenGLBuffer | None = None
        self._colours: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._position_attribute = -1
        self._colour_attribute = -1
        self._mvp_uniform = -1
        self._point_size_uniform = -1
        self._count = 0
        self._primitive = GL_POINTS
        self._uploads = 0
        self._draws = 0

    # -- what happened, for the tests and the probe ------------------------
    @property
    def uploads(self) -> int:
        """How many times geometry has been sent to the card.

        The number the whole change is about: it must go up when the model does
        and stay put across any amount of orbiting.
        """
        return self._uploads

    @property
    def draws(self) -> int:
        """How many frames have been submitted."""
        return self._draws

    @property
    def resident_points(self) -> int:
        """How many vertices are currently in the card's buffers.

        Vertices, not primitives: for a surface these are three per triangle,
        because :func:`~scanner3d.cloud_view.shade_faces` expands the mesh
        rather than indexing it. The name is the one the tests and the probe
        have always used and is kept.
        """
        return self._count

    @property
    def resident_primitive(self) -> int:
        """Whether the resident vertices are drawn as points or as triangles."""
        return self._primitive

    # -- setup -------------------------------------------------------------
    def initialise(self) -> GLDescription:
        """Compile, link and allocate against the current context.

        Raises :class:`RendererUnavailable`, with the driver's own message where
        there is one, for every way this can fail. Nothing here is allowed to
        half-succeed: a program that linked but whose attributes are missing
        would draw an empty panel and report success, which is the failure mode
        this whole path exists to avoid.
        """
        functions = _current_functions("the renderer setup")
        _drain_gl_errors(functions)

        program = QOpenGLShaderProgram()
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER):
            raise RendererUnavailable(f"vertex shader would not compile: {program.log().strip()}")
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER
        ):
            raise RendererUnavailable(f"fragment shader would not compile: {program.log().strip()}")
        if not program.link():
            raise RendererUnavailable(f"shader program would not link: {program.log().strip()}")

        self._position_attribute = program.attributeLocation("position")
        self._colour_attribute = program.attributeLocation("colour")
        self._mvp_uniform = program.uniformLocation("mvp")
        self._point_size_uniform = program.uniformLocation("point_size")
        if (
            min(
                self._position_attribute,
                self._colour_attribute,
                self._mvp_uniform,
                self._point_size_uniform,
            )
            < 0
        ):
            raise RendererUnavailable("the linked shader is missing an input this renderer needs")

        positions = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        colours = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        if not positions.create() or not colours.create():
            raise RendererUnavailable("the driver would not create a vertex buffer")

        # Only in a core profile is this required, and asking for it in a
        # compatibility one costs nothing - so it is not conditional on which
        # profile the driver decided to hand out.
        vao = QOpenGLVertexArrayObject()
        self._vao = vao if vao.create() else None

        functions.glEnable(GL_DEPTH_TEST)
        functions.glDepthFunc(GL_LESS)
        # Lets the vertex shader size the points. The compatibility profile
        # spells this GL_VERTEX_PROGRAM_POINT_SIZE and gives it the same number.
        functions.glEnable(GL_PROGRAM_POINT_SIZE)
        _raise_on_gl_error(functions, "the renderer setup")

        self._program = program
        self._positions = positions
        self._colours = colours
        return GLDescription(
            vendor=str(functions.glGetString(GL_VENDOR)),
            renderer=str(functions.glGetString(GL_RENDERER)),
            version=str(functions.glGetString(GL_VERSION)),
            shading_language=str(functions.glGetString(GL_SHADING_LANGUAGE_VERSION)),
        )

    # -- geometry ----------------------------------------------------------
    def upload(self, points: np.ndarray, colours: np.ndarray, *, triangles: bool = False) -> None:
        """Make ``points`` and ``colours`` the card's resident geometry.

        ``triangles`` draws the same buffers as a triangle soup instead of as
        points - three consecutive vertices per face, which is the form
        :func:`~scanner3d.cloud_view.shade_faces` produces. It changes the draw
        call and nothing else: the shader, the buffers and the attributes are
        the ones the point path already uses, because the lighting is in the
        colours by the time it arrives here.

        Called when the model changes and at no other time. Everything else this
        class does - every orbit, roll, pan, zoom and resize - reads what this
        left behind.

        ``QOpenGLBuffer.allocate`` returns nothing, so the only way to learn that
        a driver refused the storage - out of memory is the realistic way for
        sixty thousand points to be refused - is to ask it afterwards. Nothing
        here counts as done until it has been asked: a failed upload that
        incremented :attr:`uploads` and recorded the points as resident would
        leave an empty panel claiming to be a full one.
        """
        if self._positions is None or self._colours is None:
            raise RendererUnavailable("upload before the renderer was initialised")
        vertices = np.ascontiguousarray(points, dtype=np.float32)
        attributes = _colour_attributes(colours)
        if len(vertices) != len(attributes):
            raise RendererUnavailable("points and colours are different lengths")
        self._primitive = GL_TRIANGLES if triangles else GL_POINTS
        if triangles and len(vertices) % 3:
            raise RendererUnavailable(
                f"a triangle soup needs a multiple of three vertices, got {len(vertices)}"
            )
        if not len(vertices):
            # An empty model is what the panel holds before the first capture.
            # Allocating a zero-byte buffer is a driver question with no useful
            # answer; there is simply nothing resident.
            self._count = 0
            self._uploads += 1
            return
        functions = _current_functions("the geometry upload")
        _drain_gl_errors(functions)
        if not self._positions.bind():
            raise RendererUnavailable("the position buffer would not bind")
        self._positions.allocate(vertices.tobytes(), vertices.nbytes)
        self._positions.release()
        _raise_on_gl_error(functions, "the position buffer allocation")
        if not self._colours.bind():
            raise RendererUnavailable("the colour buffer would not bind")
        self._colours.allocate(attributes.tobytes(), attributes.nbytes)
        self._colours.release()
        _raise_on_gl_error(functions, "the colour buffer allocation")
        self._count = len(vertices)
        self._uploads += 1

    # -- drawing -----------------------------------------------------------
    def draw(self, matrix: np.ndarray, width: int, height: int, point_size: float) -> None:
        """Clear to the background and draw the resident points through ``matrix``.

        ``width`` and ``height`` are DEVICE pixels, because that is what a
        viewport is in; the matrix was built from logical ones, which is correct
        because everything in it is a ratio.

        Binding a buffer, pointing an attribute at it and submitting the draw
        are all calls that report nothing to Qt, so the error queue is emptied
        first and read afterwards. A frame the driver rejected is not counted as
        a frame: it is a fallback, with the operation that failed named in it.
        """
        program = self._program
        if program is None:
            raise RendererUnavailable("draw before the renderer was initialised")
        functions = _current_functions("the frame")
        _drain_gl_errors(functions)
        red, green, blue = BACKGROUND
        functions.glViewport(0, 0, max(1, width), max(1, height))
        functions.glClearColor(red / 255.0, green / 255.0, blue / 255.0, 1.0)
        functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        _raise_on_gl_error(functions, "clearing the frame")
        if not self._count:
            self._draws += 1
            return

        if self._vao is not None:
            self._vao.bind()
        if not program.bind():
            raise RendererUnavailable("the shader program would not bind")
        program.setUniformValue(self._mvp_uniform, _as_qt_matrix(matrix))
        # setUniformValue1f, and not the overloaded setUniformValue: given a
        # Python float the binding picks an integer overload, which is
        # GL_INVALID_OPERATION against a float uniform and leaves the point size
        # unset. Found by the error check below, which is what it is for.
        program.setUniformValue1f(self._point_size_uniform, float(point_size))
        if self._positions is not None:
            if not self._positions.bind():
                raise RendererUnavailable("the position buffer would not bind")
            program.enableAttributeArray(self._position_attribute)
            program.setAttributeBuffer(self._position_attribute, GL_FLOAT, 0, 3, 0)
        if self._colours is not None:
            if not self._colours.bind():
                raise RendererUnavailable("the colour buffer would not bind")
            program.enableAttributeArray(self._colour_attribute)
            program.setAttributeBuffer(self._colour_attribute, GL_FLOAT, 0, 3, 0)
        _raise_on_gl_error(functions, "binding the resident geometry")
        functions.glDrawArrays(self._primitive, 0, self._count)
        _raise_on_gl_error(functions, "the draw submission")
        program.disableAttributeArray(self._position_attribute)
        program.disableAttributeArray(self._colour_attribute)
        program.release()
        if self._vao is not None:
            self._vao.release()
        self._draws += 1

    def release(self) -> None:
        """Give the buffers and the program back. Safe to call twice, and late.

        This runs from the context's ``aboutToBeDestroyed``, which during
        application teardown can arrive AFTER Qt has already destroyed the
        objects it is about to hand back - observed as shiboken reporting the
        shader program's C++ side as already deleted. Releasing what is already
        released is the successful outcome, not something to raise about while
        the window is closing.
        """
        for owned in (self._positions, self._colours, self._vao, self._program):
            if owned is None:
                continue
            try:
                if isinstance(owned, QOpenGLShaderProgram):
                    owned.removeAllShaders()
                else:
                    owned.destroy()
            except RuntimeError:
                pass
        self._positions = self._colours = None
        self._vao = None
        self._program = None
        self._count = 0
        self._primitive = GL_POINTS


def _as_qt_matrix(matrix: np.ndarray) -> QMatrix4x4:
    """A row-major 4x4 as the thing Qt hands to ``glUniformMatrix4fv``."""
    return QMatrix4x4(*(float(value) for value in np.asarray(matrix, dtype=np.float64).ravel()))


class GLCamera(CloudCamera):
    """The shared camera, plus the arithmetic the GPU path adds to it.

    Separate from the widget for the same reason
    :meth:`~scanner3d.cloud_view.CloudView.projected` is separate from a paint
    event: what a frame is looking at, how far the planes are and how big a point
    is are all answerable without a context, a window or an event loop, and the
    tests and the offscreen probe answer them that way.
    """

    def camera(self) -> np.ndarray:
        """This frame's matrix. Sixteen floats, and the only per-frame work."""
        return camera_matrix(
            self._rotation,
            self._centre,
            self._distance,
            self._pan,
            max(1, self.width()),
            max(1, self.height()),
            self._far_plane(),
        )

    def _far_plane(self) -> float:
        """Behind the whole object, whatever it is and wherever it is turned.

        The furthest a point can be from the eye is ``distance`` plus the radius
        measured when the model arrived, because turning it moves points around
        the centre and not away from it.
        """
        return NEAR_PLANE + self._distance + self._radius + FAR_MARGIN

    def _point_size(self) -> float:
        """One device pixel, matching the software path's single-pixel points."""
        return max(1.0, self.devicePixelRatioF())


class GLCloudView(GLCamera, QOpenGLWidget):
    """The model panel's viewer, with the points resident on the card.

    Identical to :class:`~scanner3d.cloud_view.CloudView` in everything a person
    can do to it: the gestures, the framing rules, the colours and the point
    ceiling are all inherited from
    :class:`~scanner3d.cloud_view.CloudCamera` rather than restated. What is
    different is that a gesture costs a matrix and a draw call instead of a pass
    over the whole cloud.
    """

    #: Emitted once, with a readable reason, when this widget cannot draw. The
    #: widget does not replace itself: :class:`~scanner3d.cloud_viewer.CloudViewer`
    #: owns which backend is mounted, so there is one place that decides it.
    unavailable = Signal(str)

    #: Emitted once the driver has answered, carrying
    #: :meth:`GLDescription.summary`. The owner asked to be able to tell whether
    #: this machine is really using its graphics card, and the honest moment to
    #: say so is after a context exists, not when the widget was constructed.
    ready = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        QOpenGLWidget.__init__(self, parent)
        self._init_camera()
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        surface = QSurfaceFormat()
        surface.setDepthBufferSize(DEPTH_BUFFER_BITS)
        # Pace to the display. Combined with the one-pending-frame rule in
        # CloudCamera this is what bounds interactive work to what can be seen:
        # a swap that does not wait would let a fast card spin.
        surface.setSwapInterval(1)
        self.setFormat(surface)
        self._renderer = PointRenderer()
        self._description: GLDescription | None = None
        self._failure: str | None = None
        #: How many times the bounded post-show check has looked. Zero means it
        #: has not been scheduled, which is the state of a widget nobody has
        #: shown yet - and NOT a failure; see :meth:`_check_context`.
        self._context_checks = 0

    # -- what is drawing ---------------------------------------------------
    @property
    def description(self) -> GLDescription | None:
        """The driver's own identification, or None before the context exists."""
        return self._description

    @property
    def failure(self) -> str | None:
        """Why this widget stopped drawing, or None while it is fine."""
        return self._failure

    @property
    def renderer(self) -> PointRenderer:
        """The resident geometry and its counters."""
        return self._renderer

    # -- being shown, and finding out whether that worked ------------------
    def showEvent(self, event: QShowEvent) -> None:  # Qt's name
        """Start the clock on the context this widget has been promised.

        Everything else in this class learns about failure by being called:
        ``initializeGL`` raising, ``paintGL`` raising. The failure this exists
        for is the one where NEITHER is ever called - Qt cannot create the
        context, says so on stderr, and leaves the widget mounted and blank.
        That has no callback, so the only way to notice it is to look.
        """
        QOpenGLWidget.showEvent(self, event)
        self._schedule_context_check()

    def _settled(self) -> bool:
        """Whether there is already a verdict: a working context, or a failure."""
        return self._description is not None or self._failure is not None

    def _schedule_context_check(self) -> None:
        """Look again shortly, up to :data:`CONTEXT_CHECK_ATTEMPTS` times."""
        if self._context_checks >= CONTEXT_CHECK_ATTEMPTS:
            return
        self._context_checks += 1
        # With this widget as the timer's context object, a check that is still
        # pending when the viewer replaces this backend dies with it instead of
        # firing at a widget whose C++ side has already gone.
        QTimer.singleShot(CONTEXT_CHECK_MS, self, self._check_context)

    def _check_context(self) -> None:
        """Decide whether this widget has a context, is still getting one, or never will.

        The three states have to be kept apart, because reporting the wrong one
        is worse than not looking at all:

        * initialised, or already failed - there is a verdict, and it is not
          this method's to give;
        * shown but not yet exposed - Qt initialises on the first paint, so a
          widget that has not been painted yet is simply EARLY. Falling back
          here would send every machine to software for being constructed
          before it was on screen, which is every machine;
        * asked to initialise and still without a valid context - that is the
          blank panel, and it becomes a reported fallback.

        The nudge is :meth:`makeCurrent`, which is what forces Qt to attempt the
        creation it has been putting off; where that succeeds ``initializeGL``
        runs inside it and there is a description before it returns.
        """
        if self._settled():
            return
        if not self.isVisible():
            # Hidden again, or never really shown. Nothing has been asked of the
            # driver, so there is nothing to conclude about it; keep waiting
            # within the bound rather than blaming it for our timing.
            self._schedule_context_check()
            return
        self._force_initialisation()
        if self._settled():
            return
        context = self.context()
        if context is None or not context.isValid():
            self._fail("Qt could not create an OpenGL context for the model view")
            return
        if self._context_checks >= CONTEXT_CHECK_ATTEMPTS:
            waited = CONTEXT_CHECK_ATTEMPTS * CONTEXT_CHECK_MS
            self._fail(f"the OpenGL context was still not initialised after {waited} ms")
            return
        self._schedule_context_check()

    def _force_initialisation(self) -> None:
        """Make Qt attempt the context now rather than at some later paint.

        Wrapped because this is the call that runs on a machine whose driver is
        broken: a binding-level error here is the same answer as a refusal, and
        it must arrive as a fallback rather than as a traceback out of a timer.
        """
        try:
            self.makeCurrent()
            self.doneCurrent()
        except Exception as exc:  # every driver failure has the same answer
            self._fail(f"the OpenGL context could not be made current: {exc}")

    # -- the GL widget's three methods -------------------------------------
    def initializeGL(self) -> None:  # Qt's name
        try:
            self._description = self._renderer.initialise()
        except Exception as exc:  # every driver failure has the same answer
            self._fail(str(exc))
            return
        LOG.info("model view: OpenGL renderer ready - %s", self._description.summary())
        context = self.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._renderer.release)
        self.ready.emit(self._description.summary())

    def resizeGL(self, width: int, height: int) -> None:  # Qt's name
        """Deliberately nothing.

        A resize changes the viewport and the focal length, which are read from
        the widget when the matrix is built. It does not change the points, so
        it must not touch the buffers - and the way to be sure of that is for
        there to be no code here that could.
        """

    def paintGL(self) -> None:  # Qt's name
        self._frame_painted()
        if self._failure is not None:
            return
        try:
            if self._geometry_dirty:
                positions, colours, triangles = self._drawable()
                self._renderer.upload(positions, colours, triangles=triangles)
                self._geometry_dirty = False
            self._renderer.draw(
                self.camera(),
                round(self.width() * self.devicePixelRatioF()),
                round(self.height() * self.devicePixelRatioF()),
                self._point_size(),
            )
        except Exception as exc:  # every driver failure has the same answer
            self._fail(str(exc))

    # -- failure -----------------------------------------------------------
    def _fail(self, reason: str) -> None:
        """Stop drawing, say why, and let the viewer put something else here.

        Once only. A driver that fails in ``paintGL`` fails on every subsequent
        paint too, and a log filling at the repaint rate would bury the first
        line, which is the one that says what went wrong.
        """
        if self._failure is not None:
            return
        self._failure = reason
        LOG.warning("model view: OpenGL unavailable, falling back to software - %s", reason)
        self.unavailable.emit(reason)
