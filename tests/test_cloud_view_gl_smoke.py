"""The real driver, on this machine, drawing the real thing.

Why a test that talks to the graphics card at all
-------------------------------------------------
Every other test of this feature holds the GPU path to the software path's
arithmetic, and none of them would notice if the shader never compiled here, if
the buffers were never bound, or if the depth buffer the whole occlusion argument
rests on were not attached. Those are exactly the things that are true or false
per machine, and the owner's question was about THIS machine.

So this creates a genuine context, runs the production
:class:`~scanner3d.gl_cloud_view.PointRenderer` against it, uploads sixty
thousand points once, draws many camera-only frames, and reads the pixels back.
It reports the vendor, renderer and version it got, so a run of this suite says
in its own output whether the RX 9070 XT answered or something else did.

Why there is no timing assertion
--------------------------------
Because the honest measurement and the useful gate are different things. A frame
budget asserted here would fail on a loaded machine, on a slower card and under a
software rasteriser, none of which are defects in this code. What IS asserted is
the structural property the speed comes from and which cannot be got by luck: one
upload for many frames. The measured submission cost is reported rather than
gated, which is where a number that varies with the host belongs.

Nothing here is shown. The surface is offscreen, no window is mapped and the
owner's desktop is not touched; where no context can be had, everything skips,
because that is a machine where the software fallback is the right answer.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QOpenGLContext
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat

from scanner3d.cloud_view import BACKGROUND, MAX_RENDER_POINTS, THIN_COLOUR, shade_faces
from scanner3d.gl_cloud_view import (
    NEAR_PLANE,
    GLCamera,
    PointRenderer,
    camera_matrix,
)
from scanner3d.reconstruction_export import UPRIGHT_FROM_OPTICAL
from tests.qt_offscreen import gl_context, qt_app

__all__ = ["gl_context", "qt_app"]

#: The probe's framebuffer. Small on purpose: this is about geometry submission
#: and about the depth buffer existing, not about fill rate.
SIZE = 256

#: How many camera-only frames the probe draws. Enough that a per-frame upload
#: would be unmistakable in the counter, and quick enough not to hold the suite.
FRAMES = 120


class Probe(GLCamera):
    """The real GPU camera with no widget behind it.

    :class:`~scanner3d.gl_cloud_view.GLCamera` is what the widget itself reads
    every frame; a probe that computed its own projection would be exercising the
    probe.
    """

    def __init__(self) -> None:
        self._init_camera()

    def update(self, *_args: object) -> None:  # Qt's name
        """A frame the probe draws itself; nothing to schedule."""

    def width(self) -> int:  # Qt's name
        return SIZE

    def height(self) -> int:  # Qt's name
        return SIZE


@pytest.fixture
def framebuffer(gl_context: QOpenGLContext) -> QOpenGLFramebufferObject:
    """Somewhere to draw that is not a window, WITH a depth attachment.

    The depth attachment is not incidental: without it the card keeps whichever
    point it drew last rather than whichever is nearest, and the picture is of
    the inside of the object.
    """
    fmt = QOpenGLFramebufferObjectFormat()
    fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
    buffer = QOpenGLFramebufferObject(SIZE, SIZE, fmt)
    if not buffer.isValid() or not buffer.bind():
        pytest.skip("this machine's OpenGL stack has no usable framebuffer object")
    return buffer


def a_big_cloud() -> tuple[np.ndarray, np.ndarray]:
    """The view's own ceiling of points, deterministically placed."""
    rng = np.random.default_rng(11)
    points = (rng.random((MAX_RENDER_POINTS, 3)) * 60.0 - 30.0).astype(np.float32)
    points[:, 2] += 200.0
    thin = rng.random(MAX_RENDER_POINTS) < 0.4
    return points, thin


def test_this_machine_draws_sixty_thousand_resident_points_without_re_uploading_them(
    framebuffer: QOpenGLFramebufferObject,
) -> None:
    """The performance claim, as the structural fact underneath it.

    One upload, many frames, and the camera moving on every one of them. If an
    orbit re-uploaded the model this counter would read 121 instead of 1, and
    the per-frame cost would still be scaling with the size of the scan - which
    is the thing that made the old viewer stutter.
    """
    renderer = PointRenderer()
    description = renderer.initialise()
    print(f"\nOpenGL vendor:   {description.vendor}")
    print(f"OpenGL renderer: {description.renderer}")
    print(f"OpenGL version:  {description.version} (GLSL {description.shading_language})")

    view = Probe()
    points, thin = a_big_cloud()
    view.set_cloud(points, thin, fit=True)
    renderer.upload(view._points, view._colours)
    assert renderer.resident_points == MAX_RENDER_POINTS
    assert renderer.uploads == 1

    started = time.perf_counter()
    for frame in range(FRAMES):
        view._rotation = _spun(frame)
        view._distance = float(view._distance) * (1.0 + 0.0005 * (1 if frame % 2 else -1))
        view._pan[:] = (frame % 7, -(frame % 5))
        renderer.draw(view.camera(), SIZE, SIZE, 1.0)
    QOpenGLContext.currentContext().functions().glFinish()
    elapsed = time.perf_counter() - started
    print(
        f"backend: GPU / OpenGL; {FRAMES} camera-only frames of "
        f"{MAX_RENDER_POINTS} resident points in {elapsed * 1000:.1f} ms "
        f"({elapsed / FRAMES * 1000:.3f} ms per frame submitted), uploads: {renderer.uploads}"
    )

    assert renderer.uploads == 1, "the geometry must not have gone to the card again"
    assert renderer.draws == FRAMES
    assert renderer.resident_points == MAX_RENDER_POINTS

    renderer.release()
    # Twice, because the real second call comes from the context's own teardown
    # after the window has closed - and a viewer that raised while it was being
    # shut down would be a crash on exit.
    renderer.release()
    assert renderer.resident_points == 0


def _spun(frame: int) -> np.ndarray:
    """One orbit position, as an orbit gets there: a rotation, no clamp."""
    angle = frame * 0.05
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float64)


def test_the_card_paints_the_coverage_colours_it_was_given(
    framebuffer: QOpenGLFramebufferObject,
) -> None:
    """A picture actually came out, in the colours that carry the meaning.

    Everything else could pass with a shader that discarded every fragment. This
    reads the pixels back: the background is the background, and the point that
    was flagged as needing more scanning is the soft red that says so.
    """
    renderer = PointRenderer()
    renderer.initialise()
    view = Probe()
    view.set_cloud(np.array([[0.0, 0.0, 200.0]], dtype=np.float32), np.array([True]), fit=True)
    renderer.upload(view._points, view._colours)

    renderer.draw(view.camera(), SIZE, SIZE, 1.0)
    image = framebuffer.toImage()

    centre = image.pixelColor(SIZE // 2, SIZE // 2)
    corner = image.pixelColor(1, 1)
    assert (centre.red(), centre.green(), centre.blue()) == THIN_COLOUR
    assert (corner.red(), corner.green(), corner.blue()) == BACKGROUND
    renderer.release()


def test_the_depth_buffer_keeps_the_surface_nearest_the_scanner(
    framebuffer: QOpenGLFramebufferObject,
) -> None:
    """Occlusion, proved by the card rather than argued from the matrix.

    Two points on the same line of sight, the far one uploaded LAST so that
    without a depth test it would be the one on the screen. What comes back has
    to be the near one, or a single-view scan is being shown inside out.
    """
    renderer = PointRenderer()
    renderer.initialise()
    view = Probe()
    # In the upright presentation frame the camera looks down its own -z, so the
    # surface nearer the scanner is the one with the LARGER z. Written through
    # the same rotation the model arrives through, so it reads in the scanner's
    # own terms rather than in the widget's.
    near = UPRIGHT_FROM_OPTICAL.apply((0.0, 0.0, 190.0))
    far = UPRIGHT_FROM_OPTICAL.apply((0.0, 0.0, 260.0))
    view.set_cloud(np.array([near, far], dtype=np.float32), np.array([True, False]), fit=True)
    # Draw the FAR point second, so that without a depth test it is the one left
    # on the screen and this test would fail rather than pass by accident.
    order = np.argsort(-view._points[:, 2])
    assert view._points[order][1][2] < view._points[order][0][2]
    renderer.upload(view._points[order], view._colours[order])

    renderer.draw(view.camera(), SIZE, SIZE, 1.0)
    image = framebuffer.toImage()

    centre = image.pixelColor(SIZE // 2, SIZE // 2)
    assert (centre.red(), centre.green(), centre.blue()) == THIN_COLOUR, (
        "the far point was drawn over the near one"
    )
    renderer.release()


def test_nothing_behind_the_viewer_is_drawn(framebuffer: QOpenGLFramebufferObject) -> None:
    """The near plane and the software path's ``depth > 1`` are the same rule."""
    renderer = PointRenderer()
    renderer.initialise()
    view = Probe()
    view.set_cloud(np.array([[0.0, 0.0, 200.0]], dtype=np.float32), np.array([True]), fit=True)
    renderer.upload(view._points, view._colours)
    # Stand inside the object, so its one point is behind the eye.
    view._distance = NEAR_PLANE / 2.0

    renderer.draw(view.camera(), SIZE, SIZE, 1.0)
    image = framebuffer.toImage()

    centre = image.pixelColor(SIZE // 2, SIZE // 2)
    assert (centre.red(), centre.green(), centre.blue()) == BACKGROUND
    renderer.release()


def test_the_matrix_the_card_is_given_is_the_one_the_widget_would_build(
    framebuffer: QOpenGLFramebufferObject,
) -> None:
    """The probe drives production code, so the matrix has to be production's."""
    view = Probe()
    points, thin = a_big_cloud()
    view.set_cloud(points, thin, fit=True)

    assert np.allclose(
        view.camera(),
        camera_matrix(
            view._rotation,
            view._centre,
            view._distance,
            view._pan,
            SIZE,
            SIZE,
            view._far_plane(),
        ),
    )


def test_this_machine_fills_a_triangle_and_not_just_its_corners(
    framebuffer: QOpenGLFramebufferObject,
) -> None:
    """The surface path, proved by pixels rather than by a draw counter.

    A single large triangle across the middle of the frame. If the primitive
    were still ``GL_POINTS`` the three corners would appear and the centre would
    stay background, which is exactly what the panel looked like before the mode
    travelled with the upload - so the assertion is on the CENTRE, and it is on
    a colour the shading produced rather than on any colour this test chose.
    """
    renderer = PointRenderer()
    renderer.initialise()
    view = Probe()
    # Big enough to cover the middle of the frame, and square-on to the camera,
    # which in the upright presentation frame means a constant z.
    vertices = np.array(
        [[-60.0, -40.0, -200.0], [60.0, -40.0, -200.0], [0.0, 60.0, -200.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    soup, soup_colours, _ = shade_faces(vertices, faces)
    view.set_cloud(vertices, np.zeros(3, dtype=bool), fit=True)
    view.set_surface(vertices, faces)
    renderer.upload(soup, soup_colours, triangles=True)

    renderer.draw(view.camera(), SIZE, SIZE, 1.0)
    image = framebuffer.toImage()

    centre = image.pixelColor(SIZE // 2, SIZE // 2)
    corner = image.pixelColor(1, 1)
    expected = tuple(int(value) for value in soup_colours[0])
    assert (centre.red(), centre.green(), centre.blue()) == expected, (
        "the middle of the triangle is background: the draw is still points"
    )
    assert (corner.red(), corner.green(), corner.blue()) == BACKGROUND
    renderer.release()
