"""Showing the scan as a surface, and what that picture is allowed to claim.

Two pictures now share one panel: the coverage-coloured points, which carry a
measurement, and a lit surface, which carries a shape. This module is about the
seam between them - that the lighting is baked once and never recomputed while
the object is turned, that switching between them does not move the camera, that
a backend swap keeps both, and that the surface never quietly becomes the
picture the coverage percentages are describing.

Nothing here builds a window. The state and the shading live in
:class:`~scanner3d.cloud_view.CloudCamera`, which both renderers share and which
is deliberately not a widget, so it is subclassed directly - for the reason
:mod:`tests.test_cloud_view_orbit` gives.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from scanner3d.cloud_view import (
    COVERED_COLOUR,
    MAX_RENDER_FACES,
    MAX_RENDER_POINTS,
    SURFACE_AMBIENT,
    SURFACE_COLOUR,
    CloudCamera,
    shade_faces,
)


class Viewport(CloudCamera):
    """The real camera and the real model state, with no Qt behind it."""

    def __init__(self) -> None:
        self._init_camera()
        self.updates = 0

    def update(self, *_args: object) -> None:  # Qt's name
        self.updates += 1

    def width(self) -> int:
        return 200

    def height(self) -> int:
        return 200

    def devicePixelRatioF(self) -> float:
        return 1.0


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    """Four vertices, four faces, no two of them parallel. Every normal differs."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


def cloud_of(count: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.linspace(-5.0, 5.0, count * 3, dtype=np.float32).reshape(count, 3)
    return points, np.zeros(count, dtype=bool)


# --- the shading --------------------------------------------------------------


def test_a_face_becomes_three_vertices_carrying_its_own_shade():
    """Flat shading without an index buffer or a normal attribute: the whole
    reason the GPU path can draw a surface through the point shader."""
    vertices, faces = tetrahedron()

    soup, soup_colours, vertex_colours = shade_faces(vertices, faces)

    assert len(soup) == 3 * len(faces)
    assert len(soup_colours) == len(soup)
    assert len(vertex_colours) == len(vertices)
    for index, face in enumerate(faces):
        for corner in range(3):
            assert tuple(soup[3 * index + corner]) == tuple(vertices[face[corner]])
        # One colour for the whole triangle. That is what makes it flat.
        assert len({tuple(soup_colours[3 * index + c]) for c in range(3)}) == 1


def test_faces_pointing_different_ways_are_shaded_differently():
    """Otherwise the surface is a silhouette and the lighting is decoration."""
    vertices, faces = tetrahedron()

    _, soup_colours, _ = shade_faces(vertices, faces)

    per_face = {tuple(soup_colours[3 * index]) for index in range(len(faces))}
    assert len(per_face) > 1


def test_no_face_is_ever_unlit():
    """An unlit face on a scan reads as a hole, and a hole is the one thing this
    picture must not invent - so the ambient term is a floor, not a hope."""
    vertices, faces = tetrahedron()

    _, soup_colours, _ = shade_faces(vertices, faces)

    floor = SURFACE_AMBIENT * min(SURFACE_COLOUR)
    assert soup_colours.min() >= floor - 1.0


def test_the_light_is_two_sided_so_a_flipped_face_looks_the_same():
    """Marching cubes orients its faces by a convention. A surface half of which
    came out black would be reporting a winding rule as if it were a shape."""
    vertices, faces = tetrahedron()
    flipped = faces[:, ::-1].copy()

    _, forwards, _ = shade_faces(vertices, faces)
    _, backwards, _ = shade_faces(vertices, flipped)

    for index in range(len(faces)):
        assert tuple(forwards[3 * index]) == tuple(backwards[3 * index])


def test_a_degenerate_face_gets_the_ambient_term_rather_than_a_division_by_zero():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    _, soup_colours, _ = shade_faces(vertices, faces)

    expected = np.clip(SURFACE_AMBIENT * np.asarray(SURFACE_COLOUR), 0, 255).astype(np.uint8)
    assert tuple(soup_colours[0]) == tuple(expected)


def test_an_empty_mesh_shades_to_nothing_rather_than_raising():
    soup, soup_colours, vertex_colours = shade_faces(
        np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    )

    assert len(soup) == len(soup_colours) == len(vertex_colours) == 0


# --- what the view does with it -----------------------------------------------


def test_a_surface_becomes_the_picture_and_the_points_are_still_there():
    view = Viewport()
    points, thin = cloud_of(50)
    view.set_cloud(points, thin)

    view.set_surface(*tetrahedron())

    assert view.has_surface
    showing = view.showing_surface
    assert showing
    positions, colours, triangles = view._drawable()
    assert triangles
    assert len(positions) == 12
    assert len(colours) == 12
    # The cloud is untouched and one toggle away.
    view.set_show_surface(False)
    switched_back = view.showing_surface
    assert not switched_back
    back_to_points, _, still_triangles = view._drawable()
    assert not still_triangles
    assert len(back_to_points) == 50


def test_the_software_path_draws_the_surfaces_own_vertices_and_not_the_soup():
    """The one place the two backends genuinely differ, asserted so it stays a
    stated difference rather than a discovered one."""
    view = Viewport()
    view.set_cloud(*cloud_of(20))
    vertices, faces = tetrahedron()

    view.set_surface(vertices, faces)

    assert len(view._points) == len(vertices)
    assert len(view._colours) == len(vertices)
    soup, _, _ = view._drawable()
    assert len(soup) == 3 * len(faces)


def test_the_toggle_does_not_move_the_camera():
    """Both pictures are the same object. A camera that jumped would make the
    toggle feel like a different model rather than a different picture of one."""
    view = Viewport()
    view.set_cloud(*cloud_of(40))
    view.set_surface(*tetrahedron())
    centre, distance = np.array(view._centre), view._distance

    view.set_show_surface(False)
    view.set_show_surface(True)

    assert np.array_equal(view._centre, centre)
    assert view._distance == distance


def test_the_surface_never_frames_the_view_even_as_the_first_thing_given():
    """Framing comes from the points, always: they are what the coverage
    measurement and every other panel number describe."""
    view = Viewport()

    view.set_surface(*tetrahedron())

    assert not view._framed


def test_clearing_the_surface_goes_back_to_the_points():
    view = Viewport()
    view.set_cloud(*cloud_of(30))
    view.set_surface(*tetrahedron())

    view.clear_surface()

    assert not view.has_surface
    assert not view.showing_surface
    positions, _, triangles = view._drawable()
    assert triangles is False
    assert len(positions) == 30


def test_an_empty_face_list_is_the_same_as_no_surface():
    view = Viewport()
    view.set_cloud(*cloud_of(30))

    view.set_surface(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))

    assert not view.has_surface
    assert not view.showing_surface


def test_a_mesh_past_the_face_ceiling_is_strided_down_to_it():
    """The ceiling is on the expanded soup the card holds, so it is counted in
    faces and applied to faces."""
    view = Viewport()
    count = MAX_RENDER_FACES + 100
    vertices = np.linspace(0.0, 100.0, (count + 2) * 3, dtype=np.float32).reshape(-1, 3)
    faces = np.stack([np.arange(count), np.arange(count) + 1, np.arange(count) + 2], axis=1).astype(
        np.int32
    )

    view.set_surface(vertices, faces)

    soup, _, triangles = view._drawable()
    assert triangles
    assert len(soup) <= 3 * MAX_RENDER_FACES


def test_the_softwares_copy_of_the_surface_is_thinned_and_the_cards_is_not():
    """The two ceilings bound different costs - a per-frame CPU pass and a
    one-off upload - so they are different numbers applied to different arrays."""
    view = Viewport()
    count = MAX_RENDER_POINTS + 5000
    vertices = np.linspace(0.0, 500.0, (count + 2) * 3, dtype=np.float32).reshape(-1, 3)
    faces = np.stack([np.arange(count), np.arange(count) + 1, np.arange(count) + 2], axis=1).astype(
        np.int32
    )

    view.set_surface(vertices, faces)

    assert len(view._points) <= MAX_RENDER_POINTS
    assert len(view._points) == len(view._colours)
    soup, soup_colours, _ = view._drawable()
    # Every face is still on the card: a strided mesh is a lacework.
    assert len(soup) == 3 * count
    assert len(soup_colours) == len(soup)


def test_every_model_change_marks_the_resident_geometry_stale_and_nothing_else_does():
    """The GPU buffers are re-uploaded only when the geometry changes. A gesture
    that set this would put the whole model back on the bus every frame."""
    view = Viewport()
    view.set_cloud(*cloud_of(20))
    view._geometry_dirty = False

    view.set_surface(*tetrahedron())
    assert view._geometry_dirty

    view._geometry_dirty = False
    view.set_show_surface(False)
    assert view._geometry_dirty

    view._geometry_dirty = False
    view.set_show_surface(False)
    assert not view._geometry_dirty


def test_the_thin_fraction_keeps_describing_the_points_while_a_surface_is_shown():
    """The coverage percentage is measured on the cloud. It must not start
    describing a surface nobody measured."""
    view = Viewport()
    points = np.zeros((4, 3), dtype=np.float32)
    view.set_cloud(points, np.array([True, True, False, False]))

    view.set_surface(*tetrahedron())

    assert view.thin_fraction == pytest.approx(0.5)


def test_the_far_plane_follows_whichever_picture_is_showing():
    """The radius is what puts the far plane behind the object, and the two
    pictures do not have the same extent."""
    view = Viewport()
    view.set_cloud(*cloud_of(20))
    with_points = view._radius

    view.set_surface(*tetrahedron())

    assert view._radius != with_points
    view.set_show_surface(False)
    assert view._radius == with_points


def test_the_surface_colours_are_the_surface_colour_and_not_the_coverage_ones():
    """Deliberate: a marching-cubes vertex is invented on a grid edge, and
    painting it 'covered' would attribute a measurement to a coordinate nothing
    measured."""
    view = Viewport()
    view.set_cloud(*cloud_of(10))

    view.set_surface(*tetrahedron())

    _, colours, _ = view._drawable()
    assert not any(tuple(colour) == COVERED_COLOUR for colour in colours)
