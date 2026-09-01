"""One Qt application for the tests that genuinely need one, and no window.

Why any of them need one
------------------------
Almost nothing in this suite does: :mod:`tests.gui_harness` exists precisely so
that capture logic, coalescing and the model panel's arithmetic can be asked
without an event loop. But two questions cannot be answered that way. Whether
the model panel really SWAPS one renderer for another is a question about
widgets and layouts, and whether this machine's OpenGL stack really draws sixty
thousand resident points is a question about a driver.

Why offscreen, and why that is not a compromise
-----------------------------------------------
``QT_QPA_PLATFORM=offscreen`` is a real Qt platform that maps no window. On this
host it still hands out a hardware context - measured: the same AMD Radeon RX
9070 XT and the same Mesa 4.6 compatibility profile the window would get - so
the GPU probe exercises the actual driver rather than a software rasteriser,
while nothing appears on the owner's desktop and nothing takes their focus.

Where it does NOT, everything here skips. A machine with no GL stack is a
machine where the fallback is the answer, and a suite that failed there would be
asserting the opposite of what this feature promises.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import QApplication

#: Depth bits asked of the probe's context, matching the widget's own request.
DEPTH_BITS = 24


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """The one application object, on a platform that maps no window.

    Session-scoped because Qt allows exactly one, and the environment is set
    before it is built because the platform cannot be changed afterwards.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


@pytest.fixture
def gl_context(qt_app: QApplication) -> Iterator[QOpenGLContext]:
    """A real current OpenGL context on an offscreen surface, or a skip.

    Everything the widget's context is asked for, asked for here: the same depth
    buffer, and a genuine driver. If any step fails this is a machine where the
    software fallback is the honest answer, so the test says so rather than
    failing.
    """
    surface_format = QSurfaceFormat()
    surface_format.setDepthBufferSize(DEPTH_BITS)
    surface = QOffscreenSurface()
    surface.setFormat(surface_format)
    surface.create()
    if not surface.isValid():
        pytest.skip("no offscreen surface on this machine")
    context = QOpenGLContext()
    context.setFormat(surface_format)
    if not context.create():
        pytest.skip("this machine has no usable OpenGL context")
    if not context.makeCurrent(surface):
        pytest.skip("this machine's OpenGL context would not become current")
    try:
        yield context
    finally:
        context.doneCurrent()
