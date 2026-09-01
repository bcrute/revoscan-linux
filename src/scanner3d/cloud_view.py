"""An orbitable point cloud, with the gaps in soft red.

Why this is now the FALLBACK and not the renderer
-------------------------------------------------
This module used to say that a GL surface was the obvious choice and had not
been taken, because a viewer that fails to initialise on the user's own desktop
is worth less than a slower one that always works. Both halves of that turned
out to be true and only one of them mattered: the machine this runs on has an
accelerated OpenGL 4.6 stack, and the owner could SEE the software path - every
mouse event transformed and depth-sorted sixty thousand points, allocated and
cleared a full-window canvas and pushed an image across, on the GUI thread,
while the depth and colour previews wanted the same thread.

So :mod:`scanner3d.gl_cloud_view` draws the model on the graphics card and this
stays, unchanged in meaning, as the thing that runs when it cannot. The choice
is made in :mod:`scanner3d.cloud_viewer` and it is automatic; the fallback is
production-usable rather than a stub, which is the only reason it is safe to
prefer the driver at all.

What is left here is also the SPECIFICATION. :meth:`CloudView.projected` is
where which way up, which face is in front and where a point lands are decided,
in arithmetic that needs no driver and no display server, and the GPU backend is
held to agreeing with it.

Why holes are red and everything else is muted
----------------------------------------------
The picture answers one question: WHERE DO I STILL NEED TO SCAN. Anything that
does not serve that is noise, so covered surface is deliberately drab and only
the gaps carry colour. The red is soft rather than saturated because a large
under-scanned region should read as "go here", not as an error condition - on a
first pass most of the object is legitimately red and a harsh colour would make
a normal early capture look broken.

Why the newest frame is BRIGHTER rather than a different colour
---------------------------------------------------------------
Hue is already spoken for: red means under-scanned and grey means covered, and
a third colour for "this one just arrived" would have to take points away from
that meaning at exactly the moment the operator is reading it. Brightness is the
free channel, so the newest contributing frame keeps its own coverage colour and
is lifted towards white by :data:`NEWEST_LIFT`. A bright red rim still says
"this edge needs more scanning" while also saying "and this is the part that
landed a second ago" - which together are what tell a live model that is
following the scanner apart from one that has stopped.

Which way is up, and where the camera starts
--------------------------------------------
This widget draws +y up and +z towards the viewer, a right-handed camera looking
along its own -z. Points arrive already in the upright presentation frame
:mod:`scanner3d.reconstruction_export` defines, so the identity orientation -
where every session starts, and where a cloud that has never been turned stays -
is the view from where the scanner stood, the right way up and not mirrored.

That is a change of basis and not a change of data. Depth coordinates put +y
DOWN the image because that is the camera's own optical frame; drawing them
without accounting for it is why the object used to open upside down. The
correction is one rotation applied where the points are produced, not a sign
flipped here, so the picture and the exported files agree.

Why the orbit is a matrix and not two angles
--------------------------------------------
Yaw plus pitch has a pole. At exactly vertical the yaw axis and the view axis
coincide, left-drag stops doing anything, and the usual fix - clamping the pitch
just short of vertical - trades a control that breaks for a control that stops:
the top and the bottom of the object can be approached and never passed, so
"turn it over and look at the underside" is not a thing this view could do.

So the orientation is an accumulated 3x3 rotation, and every drag multiplies one
more increment onto it in CAMERA space. There is no angle to wrap, no pole to
avoid and nothing to clamp; the object turns through the vertical and keeps
going, and a full revolution ends where it started. The cost is that the matrix
must be kept orthonormal, which :func:`_orthonormalised` does on every drag -
cheaply, on nine numbers, and with the third row taken from a cross product so
the determinant is exactly +1 and the view can never accumulate into a mirror.

Roll is the one gesture that does not fall out of that for free, since a drag
has two axes and an orientation has three degrees of freedom. It is shift and
left-drag, and it is stated in the model panel's own key rather than left to be
discovered.

Why the view does not re-frame itself on every update
-----------------------------------------------------
Re-centring and re-fitting a cloud that has just grown is what makes the growth
invisible: the object is normalised back to the same apparent size every time,
so a snapshot with a new frame in it looks like the snapshot before it. It also
discards the pan and the zoom the operator chose, which during a capture is the
one thing they set deliberately. So framing happens at a BOUNDARY - the first
cloud this view has ever been given, a new capture, a registered
reconstruction - and every incremental update after it leaves the camera where
it was. The orbit direction is never touched at all: which side of the object
you are looking from is a preference, not a property of the points.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QWidget

#: Colour for surface that has been scanned enough, from enough directions.
#: Deliberately drab: it is context, not the message.
COVERED_COLOUR: Final = (150, 158, 168)

#: Colour for surface that still needs scanning. Soft rather than harsh - on an
#: early capture most of the object is legitimately this colour.
THIN_COLOUR: Final = (206, 110, 104)

#: How far towards white the newest frame's points are lifted. Enough that a rim
#: of them reads as a highlight from across a room, and not so far that either
#: coverage colour stops being recognisable as itself.
NEWEST_LIFT: Final = 0.62


def _lift(colour: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """``colour``, moved ``amount`` of the way towards white and still 8-bit."""
    red, green, blue = colour
    return (
        round(red + (255 - red) * amount),
        round(green + (255 - green) * amount),
        round(blue + (255 - blue) * amount),
    )


#: The two coverage colours as they appear on the newest contributing frame.
#: Derived rather than chosen, so a change to either meaning colour carries into
#: its own highlight instead of the two quietly drifting apart.
NEWEST_COVERED_COLOUR: Final = _lift(COVERED_COLOUR, NEWEST_LIFT)
NEWEST_THIN_COLOUR: Final = _lift(THIN_COLOUR, NEWEST_LIFT)

#: Background. Dark enough that both colours above stay legible on it.
BACKGROUND: Final = (24, 24, 30)


def shade_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A triangle soup with the light already in its colours, plus per-vertex shades.

    Returns ``(soup, soup_colours, vertex_colours)``. The soup is three vertices
    per face in face order - the same coordinates repeated rather than indexed -
    which is what lets the GPU path draw a flat-shaded surface through the
    position-and-colour shader it already has, with no index buffer, no normal
    attribute and no second program. Flat shading is also the honest one for
    this geometry: the surface IS faceted at the voxel scale, and smoothing the
    normals would draw a rounder object than the reconstruction found.

    ``vertex_colours`` is the same lighting averaged back onto the mesh's own
    unique vertices, for the software backend - which paints points, cannot
    rasterise a triangle at a useful rate, and draws the surface's vertices
    instead. That is a real difference between the two backends and it is stated
    here rather than left to be discovered: on the GPU the surface is solid, on
    the CPU it is the surface's vertices, and at any reasonable zoom they look
    alike because the vertices sit one voxel apart.

    The light is two-sided - the shade uses the magnitude of the dot product -
    because marching cubes over an occupancy grid orients its faces by a
    convention, and a surface half of which came out black would be reporting a
    winding rule as if it were a shape.
    """
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(faces, dtype=np.int32)
    if not len(indices) or not len(positions):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, 3), dtype=np.uint8),
        )
    corners = positions[indices]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    # A degenerate face has no direction. It gets the ambient term rather than a
    # division by zero, and it is left in the soup: dropping faces here would
    # make the drawn surface disagree with the mesh that was measured.
    safe = lengths > 0.0
    unit = np.zeros_like(normals)
    unit[safe] = normals[safe] / lengths[safe, None]
    light = np.asarray(SURFACE_LIGHT, dtype=np.float32)
    light = light / float(np.linalg.norm(light))
    shade = SURFACE_AMBIENT + (1.0 - SURFACE_AMBIENT) * np.abs(unit @ light)
    shade[~safe] = SURFACE_AMBIENT

    base = np.asarray(SURFACE_COLOUR, dtype=np.float32)
    face_colours = np.clip(shade[:, None] * base, 0.0, 255.0).astype(np.uint8)
    soup = corners.reshape(-1, 3)
    soup_colours = np.repeat(face_colours, 3, axis=0)

    # The per-vertex shade is the mean of the faces touching each vertex, which
    # is the same lighting seen at a lower resolution rather than a second
    # lighting model. A vertex no face uses keeps the ambient term.
    totals = np.zeros(len(positions), dtype=np.float32)
    counts = np.zeros(len(positions), dtype=np.float32)
    flat = indices.ravel()
    np.add.at(totals, flat, np.repeat(shade, 3))
    np.add.at(counts, flat, 1.0)
    averaged = np.where(counts > 0.0, totals / np.maximum(counts, 1.0), SURFACE_AMBIENT)
    vertex_colours = np.clip(averaged[:, None] * base, 0.0, 255.0).astype(np.uint8)
    return soup, soup_colours, vertex_colours


#: Points beyond this are decimated away before rendering. Chosen so a redraw
#: stays under a frame time on a plain CPU.
MAX_RENDER_POINTS: Final = 60_000

#: The surface's own colour. One colour and not two, deliberately: coverage is
#: measured per CELL on the points and shown there, and a marching-cubes vertex
#: is invented on a grid edge rather than observed, so painting it "thin" or
#: "covered" would be attributing a measurement to a coordinate nothing measured.
#: The surface answers a different question - what shape has this become - and
#: the panel's toggle is how an operator asks the first one instead.
SURFACE_COLOUR: Final = (176, 180, 190)

#: How lit a face pointing straight away from the light still is. Not zero: an
#: unlit face on a scan reads as a hole, and a hole is the one thing this
#: picture must not invent.
SURFACE_AMBIENT: Final = 0.34

#: Where the light is, in the model's own frame. FIXED to the model rather than
#: to the camera, so orbiting turns the object under a steady light the way a
#: hand-held object turns under a lamp; a headlight would leave the shading
#: identical from every angle, which is exactly the cue that makes a shaded
#: surface easier to read than a cloud. Normalised on use.
SURFACE_LIGHT: Final = (0.38, 0.76, 0.53)

#: Faces beyond this are dropped before rendering.
#:
#: Much larger than :data:`MAX_RENDER_POINTS`, and deliberately not derived from
#: it, because the two ceilings exist for different costs. The point ceiling
#: bounds work the software path does EVERY FRAME - it projects and depth-sorts
#: the whole array for each mouse event. The face ceiling bounds a buffer the
#: card is handed ONCE per model and then draws without the CPU touching it, so
#: what it has to be small enough for is video memory, and 200000 faces is about
#: 7 MB of positions and the same again of colours.
#:
#: Dropping faces from a closed surface is also much worse than dropping points
#: from a cloud: a strided cloud is a sparser cloud, and a strided mesh is a
#: lacework with holes through it that the reconstruction never found. So this
#: is set high enough that a live grid at its own cell budget never reaches it,
#: and the software path deals with its own cost by thinning the surface's
#: VERTICES - which it draws as points, where thinning means what it usually
#: means.
MAX_RENDER_FACES: Final = 200_000

#: How far one pixel of drag turns the object. The old value, kept: the gesture
#: is meant to feel the same as it did, and only what it can REACH has changed.
DRAG_RADIANS_PER_PIXEL: Final = 0.01

#: Below this, a Gram-Schmidt step has nothing left to normalise and the matrix
#: it came from was not a rotation. Only reachable by handing this widget a
#: matrix from outside, which is why :func:`_orthonormalised` answers with the
#: identity rather than a division by almost zero.
MIN_AXIS_NORM: Final = 1e-9


def _rotation_about_x(angle: float) -> np.ndarray:
    """Right-handed rotation about the camera's own +x, which points right."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_about_y(angle: float) -> np.ndarray:
    """Right-handed rotation about the camera's own +y, which points up."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_about_z(angle: float) -> np.ndarray:
    """Right-handed rotation about the camera's own +z, which points at you."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _orthonormalised(matrix: np.ndarray) -> np.ndarray:
    """The nearest rotation to ``matrix``, by Gram-Schmidt on its rows.

    Drift is inevitable: an accumulated orientation is a product of thousands of
    matrices, each rounded. Left alone it stops being a rotation - the axes
    stop being unit length and stop being perpendicular - and a camera matrix
    that shears makes a scan look distorted, which is indistinguishable from a
    scan that IS distorted.

    The third row is the cross product of the first two rather than another
    Gram-Schmidt step, which fixes the determinant at exactly +1: the view can
    never accumulate its way into a reflection, however many drags it has been
    through.
    """
    x = matrix[0]
    x_norm = float(np.linalg.norm(x))
    if not math.isfinite(x_norm) or x_norm < MIN_AXIS_NORM:
        return np.eye(3, dtype=np.float64)
    x = x / x_norm
    y = matrix[1] - x * float(np.dot(x, matrix[1]))
    y_norm = float(np.linalg.norm(y))
    if not math.isfinite(y_norm) or y_norm < MIN_AXIS_NORM:
        return np.eye(3, dtype=np.float64)
    y = y / y_norm
    return np.array([x, y, np.cross(x, y)], dtype=np.float64)


class CloudCamera:
    """The model, the camera and the gestures - everything both backends share.

    There are two renderers now, one on the graphics card and one on the CPU,
    and the whole value of the second is that it behaves like the first. That
    is only true if there is exactly ONE copy of what a drag means, what a
    wheel notch means, which cloud is being shown and where the camera is
    standing; a second copy written for the second backend would be a second
    set of answers, and the fallback would slowly stop being the same viewer.

    So everything that is not "put pixels on the screen" lives here, and each
    backend is left with the drawing. This is deliberately NOT a widget: it is
    mixed in FRONT of one, and a mixin that defined Qt's own methods would
    shadow them.
    """

    if TYPE_CHECKING:
        # Supplied by the QWidget each backend is also built on. Declared for
        # the type checker only - defining them for real would shadow Qt's own,
        # which is the one thing a mixin in front of a widget must not do.
        # ``update`` takes its arguments loosely because Qt overloads it four
        # ways and a narrower declaration here would CONTRADICT the widget it is
        # mixed in front of rather than describe it.
        def update(self, *args: object) -> None: ...

        def width(self) -> int: ...

        def height(self) -> int: ...

        def devicePixelRatioF(self) -> float: ...

    def _init_camera(self) -> None:
        """Every piece of state the two backends share, set once."""
        #: What is actually drawn by the software path, and what the camera
        #: measures itself against: whichever of the cloud and the surface is
        #: showing. Only :meth:`_select` assigns it.
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._colours = np.zeros((0, 3), dtype=np.uint8)
        #: The registered points, as :meth:`set_cloud` was given them.
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._cloud_colours = np.zeros((0, 3), dtype=np.uint8)
        #: The surface's own vertices, shaded, for the software path.
        self._surface = np.zeros((0, 3), dtype=np.float32)
        self._surface_colours = np.zeros((0, 3), dtype=np.uint8)
        #: The same surface as three vertices per face, shaded, for the GPU
        #: path - see :func:`shade_faces` for why it is expanded rather than
        #: indexed.
        self._soup = np.zeros((0, 3), dtype=np.float32)
        self._soup_colours = np.zeros((0, 3), dtype=np.uint8)
        #: Whether to show the surface when there is one. The operator's choice,
        #: kept across model updates: a preference that reset itself every time
        #: a frame arrived would be unusable during a capture, which is the only
        #: time both pictures exist.
        self._show_surface = True
        self._centre = np.zeros(3, dtype=np.float32)
        #: The whole orbit: one accumulated world-to-camera rotation, in float64.
        #: Identity is the view from where the scanner stood - see this module's
        #: docstring for why that is the right way up rather than an arbitrary
        #: starting angle.
        self._rotation = np.eye(3, dtype=np.float64)
        self._distance = 400.0
        self._pan = np.zeros(2, dtype=np.float32)
        self._last: QPoint | None = None
        self._thin_fraction = 0.0
        #: Whether this view has ever framed a cloud. Until it has, the camera
        #: is pointing at the origin from an arbitrary distance and there is no
        #: viewing pose worth preserving, so the first cloud always gets one.
        self._framed = False
        #: How far the furthest point is from :attr:`_centre`. Measured once per
        #: model rather than per frame, because it is only wanted to put the far
        #: clip plane behind the object and a per-frame pass over every point is
        #: exactly the CPU work this view exists to stop doing.
        self._radius = 0.0
        #: Whether the resident GPU buffers still describe :attr:`_points`. Only
        #: :meth:`set_cloud` sets it; see there for why nothing else may.
        self._geometry_dirty = True
        #: Whether a repaint has been asked for and not yet happened. One drag
        #: can produce dozens of mouse events between two displayed frames, and
        #: without this each of them asks for its own paint - which is the
        #: accumulating work that makes a fast machine feel slow.
        self._frame_pending = False

    # -- pacing ------------------------------------------------------------
    def _request_frame(self) -> None:
        """Ask for one repaint, and only one, before the next one is drawn.

        Mouse events arrive faster than a monitor can show them. The state they
        change - the orientation, the pan, the distance - is CUMULATIVE and is
        applied in full whatever this does, so nothing is lost by drawing the
        result once instead of once per event: what is dropped is duplicate
        paints of intermediate states nobody could have seen.

        There is no timer here on purpose. A view nobody is touching asks for
        nothing and costs nothing, which a redraw loop would not.
        """
        if self._frame_pending:
            return
        self._frame_pending = True
        self.update()

    def _force_frame(self) -> None:
        """Ask for a repaint even if one is already pending.

        For the two moments where "eventually" is not good enough: a new model,
        and the release of the pointer. Letting go of a drag must leave the
        orientation the user stopped at on the screen, not the one the last
        drawn frame happened to catch.
        """
        self._frame_pending = True
        self.update()

    def _frame_painted(self) -> None:
        """Called by a backend as it starts drawing: the request is now met."""
        self._frame_pending = False

    # -- data --------------------------------------------------------------
    def set_cloud(
        self,
        points: np.ndarray,
        thin: np.ndarray,
        newest: np.ndarray | None = None,
        *,
        fit: bool = False,
    ) -> None:
        """Show ``points``, colouring those flagged in ``thin`` as gaps.

        ``thin`` is a boolean array parallel to ``points``. Passing it in rather
        than computing it here keeps this module about DISPLAY - what counts as
        under-scanned is a measurement question answered in
        :mod:`scanner3d.coverage`, and it should not be re-decided by a widget.

        ``newest`` is a second parallel boolean array flagging the points that
        came from the frame that has just arrived. They are drawn in the same
        coverage colours lifted towards white, so the highlight adds a fact
        instead of replacing one. ``None`` means "nothing here is new", which is
        the honest answer for a registered reconstruction: every frame in that
        one was placed at once by the optimiser, and the last file in a finished
        capture is not news.

        ``fit`` frames the cloud - centre, distance and pan - and is the
        BOUNDARY described in this module's docstring. Left false, an update
        keeps the pose the operator is looking through, so a model that has
        grown looks like a model that has grown rather than like the same model
        rescaled. The first cloud is framed whatever ``fit`` says, because a
        pose that has never been established is not one worth keeping.
        """
        if newest is None:
            newest = np.zeros(len(points), dtype=bool)
        if len(points) > MAX_RENDER_POINTS:
            # All three together, or the flags stop describing the points they
            # are indexed against and the colouring silently moves.
            step = len(points) // MAX_RENDER_POINTS + 1
            points, thin, newest = points[::step], thin[::step], newest[::step]
        self._cloud = np.ascontiguousarray(points, dtype=np.float32)
        colours = np.empty((len(self._cloud), 3), dtype=np.uint8)
        colours[:] = COVERED_COLOUR
        colours[thin] = THIN_COLOUR
        # After both meaning colours, and split by the same flag, so the
        # highlight is the lifted form of whatever that point already meant.
        colours[newest & ~thin] = NEWEST_COVERED_COLOUR
        colours[newest & thin] = NEWEST_THIN_COLOUR
        self._cloud_colours = colours
        self._thin_fraction = float(thin.mean()) if len(thin) else 0.0
        # Framed from the CLOUD and never from the surface, whichever is
        # showing. The two describe the same object, and a camera that jumped
        # when the operator flipped the toggle would make the toggle feel like a
        # different model rather than a different picture of one.
        if len(self._cloud) and (fit or not self._framed):
            lo = self._cloud.min(axis=0)
            hi = self._cloud.max(axis=0)
            self._centre = (lo + hi) / 2.0
            self._distance = float(np.linalg.norm(hi - lo)) * 1.4 or 400.0
            self._pan[:] = 0.0
            self._framed = True
        self._select()

    # -- the surface -------------------------------------------------------
    def set_surface(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Show ``faces`` over ``vertices`` as a lit surface, when the toggle allows it.

        The mesh arrives as the reconstruction produced it - shared vertices,
        indexed triangles - and the lighting is baked here by
        :func:`shade_faces`, once per model. Nothing about it is recomputed
        while the operator orbits: that is the whole reason the light is fixed
        to the model rather than to the camera.

        A mesh past :data:`MAX_RENDER_FACES` is strided down to it, which
        punches holes in a closed surface and is why that ceiling is set where a
        live grid cannot reach it. The vertex array is never strided with the
        faces - a face that survives has to index the coordinates it was
        measured with - but the copy the software path draws as POINTS is
        thinned to :data:`MAX_RENDER_POINTS`, where thinning costs a sparser
        picture and nothing else.

        An empty ``faces`` is the way to say "there is no surface yet" and is
        the same as :meth:`clear_surface`.
        """
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if len(faces) > MAX_RENDER_FACES:
            faces = faces[:: len(faces) // MAX_RENDER_FACES + 1]
        soup, soup_colours, vertex_colours = shade_faces(vertices, faces)
        if not len(faces):
            vertices = np.zeros((0, 3), dtype=np.float32)
        if len(vertices) > MAX_RENDER_POINTS:
            # Both together, so a vertex keeps the shade that was computed for
            # it rather than one belonging to a vertex that was dropped.
            step = len(vertices) // MAX_RENDER_POINTS + 1
            vertices, vertex_colours = vertices[::step], vertex_colours[::step]
        self._surface = vertices
        self._surface_colours = vertex_colours
        self._soup = soup
        self._soup_colours = soup_colours
        self._select()

    def clear_surface(self) -> None:
        """Forget the surface and go back to the points."""
        self._surface = np.zeros((0, 3), dtype=np.float32)
        self._surface_colours = np.zeros((0, 3), dtype=np.uint8)
        self._soup = np.zeros((0, 3), dtype=np.float32)
        self._soup_colours = np.zeros((0, 3), dtype=np.uint8)
        self._select()

    @property
    def has_surface(self) -> bool:
        """Whether a surface has been given to this view."""
        return bool(len(self._soup))

    @property
    def has_cloud(self) -> bool:
        """Whether any measured points have been given to this view.

        The cloud rather than :attr:`_points`, which for a showing surface is
        the surface's own vertices: what this answers is "is there a model in
        here at all", and a panel that has only ever been handed a surface has
        no measured points in it however much geometry it is drawing.
        """
        return bool(len(self._cloud))

    @property
    def show_surface(self) -> bool:
        """Whether the surface is what would be drawn if there were one."""
        return self._show_surface

    def set_show_surface(self, showing: bool) -> None:
        """Choose between the lit surface and the coverage-coloured points."""
        if self._show_surface == showing:
            return
        self._show_surface = showing
        self._select()

    @property
    def showing_surface(self) -> bool:
        """Whether the surface is what is ACTUALLY on the screen right now."""
        return self._show_surface and self.has_surface

    def _drawable(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Positions, colours, and whether they are triangles rather than points.

        The GPU path asks this; the software path draws :attr:`_points`, which
        for a showing surface is the same surface at its own vertices. See
        :func:`shade_faces` for why the two backends differ here and nowhere
        else.
        """
        if self.showing_surface:
            return self._soup, self._soup_colours, True
        return self._cloud, self._cloud_colours, False

    def _select(self) -> None:
        """Point the drawn arrays at whichever picture is showing, and redraw.

        The one place :attr:`_points` and :attr:`_colours` are assigned, and the
        one place :attr:`_geometry_dirty` is set. New geometry is the only event
        that can make the resident GPU buffers wrong, and a gesture that set it
        would put the whole model back on the bus every frame - which is the CPU
        pipeline this view replaced, with an upload added.
        """
        if self.showing_surface:
            self._points, self._colours = self._surface, self._surface_colours
        else:
            self._points, self._colours = self._cloud, self._cloud_colours
        self._radius = (
            float(np.linalg.norm(self._points - self._centre, axis=1).max())
            if len(self._points)
            else 0.0
        )
        self._geometry_dirty = True
        self._force_frame()

    @property
    def thin_fraction(self) -> float:
        return self._thin_fraction

    # -- camera ------------------------------------------------------------
    @property
    def orientation(self) -> np.ndarray:
        """A copy of the accumulated world-to-camera rotation.

        A copy, because the one thing that must never happen to this matrix is
        being modified from outside the two methods that keep it orthonormal.
        """
        return self._rotation.copy()

    def _matrix(self) -> np.ndarray:
        return self._rotation.astype(np.float32)

    def _turn(self, rotation: np.ndarray) -> None:
        """Apply one incremental rotation in CAMERA space and renormalise.

        Left-multiplication is what makes the gesture mean the same thing
        wherever the object has already been turned to: the increment is
        expressed in the axes the operator is looking along, not in the object's
        own, so "drag right" is always "spin it right" and never "spin it about
        whatever the world's y axis has ended up being".

        The renormalisation is not decoration either. Every drag multiplies this
        matrix again, and a product of a few thousand nearly-orthonormal
        matrices is not orthonormal: it shears, and a sheared camera matrix
        distorts the object slowly enough that it looks like the scan is wrong.
        Re-deriving the third row from the cross product of the first two also
        pins the determinant at +1 exactly, so no accumulation of rounding can
        turn the view inside out.
        """
        self._rotation = _orthonormalised(rotation @ self._rotation)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last = event.position().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last = None
        # Unconditionally, because this is the frame that must not be dropped:
        # wherever the user let go is where they meant to leave it.
        self._force_frame()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last is None:
            return
        delta = event.position().toPoint() - self._last
        self._last = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # ROLL, about the axis you are looking along. The other two
                # gestures cannot produce it on their own from every attitude,
                # and an object that cannot be rolled is one whose long axis
                # cannot be laid flat across the window to be looked at.
                self._turn(_rotation_about_z(-delta.x() * DRAG_RADIANS_PER_PIXEL))
            else:
                # Both axes of one drag, in camera space, with no clamp anywhere:
                # the picture follows the cursor through the poles and round
                # again, because there is no pole here to be at.
                self._turn(
                    _rotation_about_x(delta.y() * DRAG_RADIANS_PER_PIXEL)
                    @ _rotation_about_y(delta.x() * DRAG_RADIANS_PER_PIXEL)
                )
            self._request_frame()
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self._pan[0] += delta.x()
            self._pan[1] += delta.y()
            self._request_frame()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._distance *= 0.9 if event.angleDelta().y() > 0 else 1.0 / 0.9
        self._distance = max(1.0, self._distance)
        self._request_frame()


class CloudView(CloudCamera, QWidget):
    """A point cloud you can turn around, drawn a pixel at a time on the CPU.

    The fallback, and still the definition. Every meaning the GPU backend has
    to reproduce - which way up, which face is in front, what the colours say -
    is decided by :meth:`projected` below, in arithmetic that needs no driver,
    no context and no display server to check.

    Left-drag orbits in any direction and as far as you like, shift+left-drag
    rolls, middle-drag pans, the wheel zooms. Those are the things a person does
    to look at an object, and no mode has to be selected first.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_camera()
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # -- painting ----------------------------------------------------------
    def projected(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Screen columns, screen rows and colours of every visible point.

        Ordered far to near, which is the order they must be painted in: the
        near ones land on top, which is cheaper than a z-buffer and correct
        enough for points that are all one size.

        Separate from :meth:`paintEvent` because this is arithmetic and that is
        a widget. Which way up the object is drawn, which way round it is, and
        which face of it is in front are all decided here, and all three are
        answerable without a display server, a window or an event loop.
        """
        local = (self._points - self._centre) @ self._matrix().T
        # The camera sits on its own +z and looks along -z, which is what makes
        # (right, up, towards you) = (+x, +y, +z) a RIGHT-handed basis. Getting
        # this backwards is not a cosmetic error: it mirrors the object, and a
        # mirrored scan is wrong in a way that still looks like a scan.
        depth = self._distance - local[:, 2]
        visible = depth > 1.0
        if not visible.any():
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 3), dtype=np.uint8),
            )
        focal = 0.9 * min(width, height)
        x = local[visible, 0] * focal / depth[visible] + width / 2 + self._pan[0]
        # Negated because a screen row counts DOWNWARDS and +y is up.
        y = -local[visible, 1] * focal / depth[visible] + height / 2 + self._pan[1]
        px = np.rint(x).astype(np.int32)
        py = np.rint(y).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px, py = px[inside], py[inside]
        colours = self._colours[visible][inside]
        order = np.argsort(-depth[visible][inside])
        return px[order], py[order], colours[order]

    def paintEvent(self, _event: object) -> None:
        self._frame_painted()
        width = max(1, self.width())
        height = max(1, self.height())
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:] = BACKGROUND

        if len(self._points):
            px, py, colours = self.projected(width, height)
            canvas[py, px] = colours

        image = QImage(canvas.data, width, height, width * 3, QImage.Format.Format_RGB888)
        QPainter(self).drawImage(0, 0, image)
