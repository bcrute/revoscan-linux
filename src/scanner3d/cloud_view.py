"""An orbitable point cloud, with the gaps in soft red.

Why software rendering rather than OpenGL
-----------------------------------------
The obvious choice is a GL surface, and it was not taken. This machine has AMD
graphics with no working compute stack, driver support for the older card is
uneven, and a viewer that fails to initialise on the user's own desktop is
worth less than a slower one that always works - a lesson already learned once
this session, when the window would not start at all because a Qt platform
plugin was missing.

Projecting a few tens of thousands of points is arithmetic numpy does in a
couple of milliseconds. The cost is that this cannot scale to millions of
points; the benefit is that it has no driver dependency at all. If that trade
ever stops making sense, this module is the only thing that has to change.

Why holes are red and everything else is muted
----------------------------------------------
The picture answers one question: WHERE DO I STILL NEED TO SCAN. Anything that
does not serve that is noise, so covered surface is deliberately drab and only
the gaps carry colour. The red is soft rather than saturated because a large
under-scanned region should read as "go here", not as an error condition - on a
first pass most of the object is legitimately red and a harsh colour would make
a normal early capture look broken.
"""

from __future__ import annotations

import math
from typing import Final

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

#: Background. Dark enough that both colours above stay legible on it.
BACKGROUND: Final = (24, 24, 30)

#: Points beyond this are decimated away before rendering. Chosen so a redraw
#: stays under a frame time on a plain CPU.
MAX_RENDER_POINTS: Final = 60_000


class CloudView(QWidget):
    """A point cloud you can turn around.

    Left-drag orbits, middle-drag pans, the wheel zooms. Those are the three
    things a person does to look at an object, and no mode has to be selected
    first.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._colours = np.zeros((0, 3), dtype=np.uint8)
        self._centre = np.zeros(3, dtype=np.float32)
        self._azimuth = 0.6
        self._elevation = 0.4
        self._distance = 400.0
        self._pan = np.zeros(2, dtype=np.float32)
        self._last: QPoint | None = None
        self._thin_fraction = 0.0

    # -- data --------------------------------------------------------------
    def set_cloud(self, points: np.ndarray, thin: np.ndarray) -> None:
        """Show ``points``, colouring those flagged in ``thin`` as gaps.

        ``thin`` is a boolean array parallel to ``points``. Passing it in rather
        than computing it here keeps this module about DISPLAY - what counts as
        under-scanned is a measurement question answered in
        :mod:`scanner3d.coverage`, and it should not be re-decided by a widget.
        """
        if len(points) > MAX_RENDER_POINTS:
            step = len(points) // MAX_RENDER_POINTS + 1
            points, thin = points[::step], thin[::step]
        self._points = np.ascontiguousarray(points, dtype=np.float32)
        colours = np.empty((len(self._points), 3), dtype=np.uint8)
        colours[:] = COVERED_COLOUR
        colours[thin] = THIN_COLOUR
        self._colours = colours
        self._thin_fraction = float(thin.mean()) if len(thin) else 0.0
        if len(self._points):
            lo = self._points.min(axis=0)
            hi = self._points.max(axis=0)
            self._centre = (lo + hi) / 2.0
            self._distance = float(np.linalg.norm(hi - lo)) * 1.4 or 400.0
            self._pan[:] = 0.0
        self.update()

    @property
    def thin_fraction(self) -> float:
        return self._thin_fraction

    # -- camera ------------------------------------------------------------
    def _matrix(self) -> np.ndarray:
        ca, sa = math.cos(self._azimuth), math.sin(self._azimuth)
        ce, se = math.cos(self._elevation), math.sin(self._elevation)
        yaw = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]], dtype=np.float32)
        pitch = np.array([[1.0, 0.0, 0.0], [0.0, ce, se], [0.0, -se, ce]], dtype=np.float32)
        return pitch @ yaw

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last = event.position().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last is None:
            return
        delta = event.position().toPoint() - self._last
        self._last = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._azimuth += delta.x() * 0.01
            # Clamped just short of the poles: at exactly vertical the yaw axis
            # and the view axis coincide and left-drag stops doing anything,
            # which feels like the control has broken.
            self._elevation = max(-1.55, min(1.55, self._elevation + delta.y() * 0.01))
            self.update()
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self._pan[0] += delta.x()
            self._pan[1] += delta.y()
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._distance *= 0.9 if event.angleDelta().y() > 0 else 1.0 / 0.9
        self._distance = max(1.0, self._distance)
        self.update()

    # -- painting ----------------------------------------------------------
    def paintEvent(self, _event: object) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:] = BACKGROUND

        if len(self._points):
            local = (self._points - self._centre) @ self._matrix().T
            depth = local[:, 2] + self._distance
            visible = depth > 1.0
            if visible.any():
                focal = 0.9 * min(width, height)
                x = local[visible, 0] * focal / depth[visible] + width / 2 + self._pan[0]
                y = -local[visible, 1] * focal / depth[visible] + height / 2 + self._pan[1]
                px = np.rint(x).astype(np.int32)
                py = np.rint(y).astype(np.int32)
                inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                px, py = px[inside], py[inside]
                colours = self._colours[visible][inside]
                # Paint far points first so near ones land on top. Cheaper than
                # a z-buffer and correct enough for points of a single size.
                order = np.argsort(-depth[visible][inside])
                canvas[py[order], px[order]] = colours[order]

        image = QImage(canvas.data, width, height, width * 3, QImage.Format.Format_RGB888)
        QPainter(self).drawImage(0, 0, image)
