"""Prove the surface is closed because it was counted, not because it was hoped.

Three claims carry this module and none of them can be settled by reading it.

The first is that the marching cubes table cannot crack. That is checked
exhaustively rather than by example: for all 256 cases, all three axes and all
sixteen ways a neighbouring cell can differ off the shared face, the two cells
are made to agree on the shared face's contour segments and to disagree on their
direction - which is what a closed, consistently wound surface needs and what
the original algorithm's published table famously does not always give.

The second is that the geometry is right. A synthetic sphere is reconstructed
and every vertex is measured against the sphere it came from, in both
directions; the triangles are measured for area; the normals are measured
against the centre. "It produced triangles" is not a result here.

The third is that a file is written whole or not at all, and that everything it
knows about its own limits is inside it. Every encoded mesh is parsed back by a
reader in this file that knows OBJ and PLY and nothing about
:mod:`scanner3d.mesh`, and every refusal path is followed to a path that is then
checked to be absent.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import struct
from pathlib import Path

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.geometry import NonFiniteValue, Point, cross, dot, norm, subtract
from scanner3d.intrinsics import INTRINSICS_FORMAT, parse_intrinsics
from scanner3d.mesh import (
    CORNER_OFFSETS,
    DEFAULT_ISO_LEVEL,
    EDGE_CORNERS,
    EDGE_TABLE,
    INHERENT_LIMITATIONS,
    MAX_GRID_CELLS,
    MAX_OUTPUT_BYTES,
    SCHEMA_VERSION,
    TRIANGLE_TABLE,
    DegenerateFace,
    EmptyMesh,
    GridTooLarge,
    IsoLevelUnusable,
    Mesh,
    MeshCoordinateNotEncodable,
    MeshCountExceeded,
    MeshFileFormat,
    MeshIndexOutOfRange,
    MeshNotAllowed,
    MeshProvenance,
    MeshWriteIncomplete,
    NoPointsOffered,
    OutputSizeExceeded,
    PointCountExceeded,
    VoxelSizeUnusable,
    binary32,
    case_index,
    edge_census,
    encode_obj,
    encode_ply_mesh,
    reconstruct,
    write_obj,
    write_ply_mesh,
)
from scanner3d.pointcloud import (
    CAMERA_OPTICAL_FRAME,
    CalibrationProvenance,
    CloudProvenance,
    PlyFormat,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

# --------------------------------------------------------------------------
# Synthetic shapes, exactly known, so a vertex can be measured against one
# --------------------------------------------------------------------------


def sphere_points(count: int, radius: float) -> list[Point]:
    """``count`` points spread evenly over a sphere of ``radius`` about the origin.

    The Fibonacci spiral rather than a latitude-longitude grid: it has no poles
    where the samples bunch up, so the occupancy grid sees the same sample
    density everywhere and a vertex measured against the sphere is measuring the
    reconstruction rather than the sampling.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        points.append(
            (radius * ring * math.cos(angle), radius * ring * math.sin(angle), radius * z)
        )
    return points


def cube_surface_points(steps: int, half: float) -> list[Point]:
    """The six faces of a cube of side ``2 * half``, sampled on a regular grid."""
    points: list[Point] = []
    for i in range(steps + 1):
        for j in range(steps + 1):
            u = -half + 2.0 * half * i / steps
            v = -half + 2.0 * half * j / steps
            points.extend(
                (
                    (u, v, -half),
                    (u, v, half),
                    (u, -half, v),
                    (u, half, v),
                    (-half, u, v),
                    (half, u, v),
                )
            )
    return points


def box_points(steps: int, half: float, skip: tuple[int, int] | None = None) -> list[Point]:
    """The six faces of a box, optionally leaving one of them off.

    Sampled finely enough that each face is a solid plane of cells with no gap a
    six-connected fill could squeeze through, so the only way into the middle is
    the face that was left off.
    """
    points: list[Point] = []
    for axis in range(3):
        for side in (-1, 1):
            if skip == (axis, side):
                continue
            across = [index for index in range(3) if index != axis]
            for i in range(steps + 1):
                for j in range(steps + 1):
                    place = [0.0, 0.0, 0.0]
                    place[axis] = side * half
                    place[across[0]] = -half + 2.0 * half * i / steps
                    place[across[1]] = -half + 2.0 * half * j / steps
                    points.append((place[0], place[1], place[2]))
    return points


def triangle_area(mesh: Mesh, face: tuple[int, int, int]) -> float:
    first, second, third = (mesh.vertices[index] for index in face)
    return 0.5 * norm(cross(subtract(second, first), subtract(third, first)))


def face_normal(mesh: Mesh, face: tuple[int, int, int]) -> Point:
    first, second, third = (mesh.vertices[index] for index in face)
    return cross(subtract(second, first), subtract(third, first))


def face_centroid(mesh: Mesh, face: tuple[int, int, int]) -> Point:
    first, second, third = (mesh.vertices[index] for index in face)
    return tuple((first[axis] + second[axis] + third[axis]) / 3.0 for axis in range(3))  # type: ignore[return-value]


def euler_characteristic(mesh: Mesh) -> int:
    """``V - E + F``. Two for a closed surface of genus zero, and only for one."""
    return mesh.vertex_count - mesh.census.edge_count + mesh.face_count


def sub(record: dict[str, object], key: str) -> dict[str, object]:
    """One nested section of a serialised record, checked to be one."""
    value = record[key]
    assert isinstance(value, dict), f"{key} should be a mapping, not {value!r}"
    return value


def rebuilt(parsed: ParsedObj | ParsedPlyMesh) -> Mesh:
    """A mesh assembled from what a reader got out of the file, not from the writer.

    The round trip is only worth something if the surface survives it: a mesh
    rebuilt from the parsed numbers has to measure closed the same way the
    original did.
    """
    return Mesh(
        vertices=tuple((x, y, z) for x, y, z in parsed.vertices),
        faces=tuple((face[0], face[1], face[2]) for face in parsed.faces),
    )


#: A provenance a test can pass without pretending a calibration happened.
def provenance(**overrides: object) -> MeshProvenance:
    cloud = CloudProvenance(
        units="test units; this label measures nothing and is not a length",
        convention=CAMERA_OPTICAL_FRAME,
        calibration=CalibrationProvenance.none_applied("tests/test_mesh.py: synthetic geometry"),
    )
    fields: dict[str, object] = {
        "cloud": cloud,
        "method": "voxel occupancy + marching cubes",
        "voxel_size": 0.1,
        "iso_level": DEFAULT_ISO_LEVEL,
    }
    fields.update(overrides)
    return MeshProvenance(**fields)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def sphere_mesh() -> Mesh:
    """One reconstruction of a sphere, reused: several tests measure this shape."""
    return reconstruct(sphere_points(4000, 1.0), voxel_size=0.1)


@pytest.fixture(scope="module")
def cube_mesh() -> Mesh:
    return reconstruct(cube_surface_points(24, 1.0), voxel_size=0.125)


# --------------------------------------------------------------------------
# The table: 256 cases that must agree with each other at every shared face
# --------------------------------------------------------------------------


def cut_edges(mask: int) -> set[int]:
    """The edges whose two corners disagree, derived from the corners alone."""
    return {
        edge
        for edge, (first, second) in enumerate(EDGE_CORNERS)
        if (mask >> first & 1) != (mask >> second & 1)
    }


def directed_boundary_segments(mask: int) -> list[tuple[int, int]]:
    """The contour segments of a case: directed triangle edges used exactly once.

    A fan's internal diagonal is used by two triangles of the same cell, once
    each way, so it cancels. What survives is the polygon boundary - the
    segments that lie on a face of the cell and have to be matched by the cell
    on the other side of it.
    """
    used: dict[tuple[int, int], int] = {}
    for first, second, third in TRIANGLE_TABLE[mask]:
        for start, end in ((first, second), (second, third), (third, first)):
            used[(start, end)] = used.get((start, end), 0) + 1
    return [pair for pair, count in used.items() if count == 1 and used.get(pair[::-1], 0) == 0]


def on_face(edge: int, axis: int, side: int) -> bool:
    return all(CORNER_OFFSETS[corner][axis] == side for corner in EDGE_CORNERS[edge])


def shared_face_key(edge: int, axis: int) -> frozenset[tuple[int, ...]]:
    """An edge of a shared face, named in a way both neighbouring cells agree on.

    The two cells number this edge differently - it is on one cell's ``axis==1``
    face and on the other's ``axis==0`` face - so it is named by the two
    coordinates that are *not* the shared axis, which both cells see the same
    way round.
    """
    return frozenset(
        tuple(value for index, value in enumerate(CORNER_OFFSETS[corner]) if index != axis)
        for corner in EDGE_CORNERS[edge]
    )


def face_segments(mask: int, axis: int, side: int) -> set[tuple[frozenset[tuple[int, ...]], ...]]:
    """The case's contour segments lying on one face, in shared-face names."""
    return {
        (shared_face_key(start, axis), shared_face_key(end, axis))
        for start, end in directed_boundary_segments(mask)
        if on_face(start, axis, side) and on_face(end, axis, side)
    }


def test_the_table_has_one_entry_per_case_and_nothing_for_the_two_uniform_ones():
    assert len(TRIANGLE_TABLE) == 256
    assert len(EDGE_TABLE) == 256
    # All corners on the same side of the level: no surface passes through.
    assert TRIANGLE_TABLE[0] == ()
    assert TRIANGLE_TABLE[255] == ()
    assert EDGE_TABLE[0] == 0
    assert EDGE_TABLE[255] == 0
    assert all(TRIANGLE_TABLE[mask] for mask in range(1, 255))
    # Five triangles is the standard maximum for a marching cubes cell.
    assert max(len(entry) for entry in TRIANGLE_TABLE) == 5


def test_the_edge_table_is_exactly_the_edges_whose_corners_disagree():
    for mask in range(256):
        expected = sum(1 << edge for edge in cut_edges(mask))
        assert EDGE_TABLE[mask] == expected, mask


def test_every_triangle_names_three_distinct_edges_the_case_actually_cuts():
    for mask in range(256):
        for triangle in TRIANGLE_TABLE[mask]:
            assert len(set(triangle)) == 3, (mask, triangle)
            for edge in triangle:
                assert EDGE_TABLE[mask] >> edge & 1, (mask, triangle, edge)


def test_no_case_uses_one_mesh_edge_more_than_twice():
    """The fan apex is chosen to make this true; here it is measured.

    A diagonal joining two cut edges that share a grid face would be drawn by
    both cells on that face and used four times - a pinch in the surface that
    edge counting would report as non-manifold for shapes that are nothing of
    the kind.
    """
    for mask in range(256):
        used: dict[frozenset[int], int] = {}
        for first, second, third in TRIANGLE_TABLE[mask]:
            for start, end in ((first, second), (second, third), (third, first)):
                key = frozenset((start, end))
                used[key] = used.get(key, 0) + 1
        assert all(count <= 2 for count in used.values()), mask
        # Every once-used edge is a contour segment, so it must lie on a face,
        # where the neighbouring cell can meet it.
        for pair, count in used.items():
            if count == 1:
                first, second = sorted(pair)
                assert any(
                    on_face(first, axis, side) and on_face(second, axis, side)
                    for axis in range(3)
                    for side in (0, 1)
                ), (mask, pair)


def test_two_cells_agree_on_every_shared_face_for_all_256_cases():
    """The no-cracks proof, taken over every case pair that can actually occur.

    For each case, each of the three axes, and each of the sixteen patterns the
    neighbouring cell can have off the shared face, the segments both cells draw
    on that face must be the same segments - and drawn in opposite directions,
    which is what leaves the two triangles that meet there consistently wound.
    """
    checked = 0
    for mask in range(256):
        for axis in range(3):
            # The neighbour's near-face corners are this cell's far-face
            # corners, at the same position in space.
            for free in range(16):
                neighbour = 0
                far = 0
                for corner, offset in enumerate(CORNER_OFFSETS):
                    if offset[axis] == 0:
                        mirror = tuple(
                            1 if index == axis else value for index, value in enumerate(offset)
                        )
                        source = CORNER_OFFSETS.index(mirror)
                        if mask >> source & 1:
                            neighbour |= 1 << corner
                    else:
                        if free >> far & 1:
                            neighbour |= 1 << corner
                        far += 1
                mine = face_segments(mask, axis, 1)
                theirs = face_segments(neighbour, axis, 0)
                assert mine == {pair[::-1] for pair in theirs}, (mask, axis, free)
                checked += 1
    assert checked == 256 * 3 * 16


def test_the_case_index_reads_corners_strictly_above_the_level():
    assert case_index((0, 0, 0, 0, 0, 0, 0, 0)) == 0
    assert case_index((8, 8, 8, 8, 8, 8, 8, 8)) == 255
    assert case_index((4, 0, 0, 0, 0, 0, 0, 0)) == 1
    assert case_index((0, 0, 0, 0, 0, 0, 0, 4)) == 128
    # Exactly on the level is outside: one rule, applied by every cell that
    # shares the corner.
    assert case_index((3.5, 0, 0, 0, 0, 0, 0, 0)) == 0
    with pytest.raises(MeshNotAllowed):
        case_index((1, 2, 3))


# --------------------------------------------------------------------------
# Reconstruction: measured against the shape the samples came from
# --------------------------------------------------------------------------


def test_a_sphere_of_samples_becomes_one_closed_surface(sphere_mesh):
    census = sphere_mesh.census
    assert sphere_mesh.is_watertight
    assert census.boundary_edge_count == 0
    assert census.nonmanifold_edge_count == 0
    assert census.consistently_oriented
    # One closed surface of genus zero, and nothing else: a second component, a
    # handle or a tunnel would all show up here.
    assert euler_characteristic(sphere_mesh) == 2
    # A sphere of radius 1 at 0.1 per voxel has a surface area of about 12.6
    # and cells of about 0.01, so a few thousand triangles. The window is wide
    # on purpose - it is a sanity bound, not a measurement.
    assert 1000 < sphere_mesh.face_count < 20000
    assert sphere_mesh.vertex_count * 2 == sphere_mesh.face_count + 4  # Euler, rearranged


def test_every_vertex_of_a_reconstructed_sphere_lies_on_the_sphere(sphere_mesh):
    """The real geometric check: where each vertex is, not merely that it exists.

    Both bounds matter and they are not symmetric. A cell is solid when a sample
    falls anywhere in it, so the solid contains the samples: no vertex may fall
    *inside* the sampled sphere, and none may stand more than about a voxel
    outside it. A reconstruction that had drifted, scaled or shifted would fail
    one bound or the other however many triangles it produced.
    """
    voxel = 0.1
    radii = [norm(vertex) for vertex in sphere_mesh.vertices]
    assert min(radii) >= 1.0, f"a vertex fell inside the samples at radius {min(radii)!r}"
    assert max(radii) <= 1.0 + 1.5 * voxel, f"a vertex stood off at radius {max(radii)!r}"
    root_mean_square = math.sqrt(math.fsum((r - 1.0) ** 2 for r in radii) / len(radii))
    assert root_mean_square < voxel


def test_a_reconstructed_sphere_has_no_zero_area_triangle(sphere_mesh):
    """No degenerate triangles, which the half-integer level is what guarantees.

    A vertex can never land exactly on a grid node, so two cut edges can never
    produce one point, so no triangle can collapse to a segment.
    """
    for face in sphere_mesh.faces:
        assert len(set(face)) == 3, face
        assert triangle_area(sphere_mesh, face) > 0.0, face


def test_the_normals_of_a_reconstructed_sphere_point_outwards(sphere_mesh):
    """Winding is a fact about the file, so it is checked against known geometry.

    For a surface around the origin, a triangle wound so its normal points away
    from the solid has a positive dot product with its own centroid. Every one
    of them, or the mesh is inside out - which no edge count would ever notice.
    """
    for face in sphere_mesh.faces:
        assert dot(face_normal(sphere_mesh, face), face_centroid(sphere_mesh, face)) > 0.0, face


def test_a_cube_of_samples_becomes_a_closed_surface_around_them(cube_mesh):
    voxel = 0.125
    assert cube_mesh.is_watertight
    assert cube_mesh.census.consistently_oriented
    assert euler_characteristic(cube_mesh) == 2
    extent = cube_mesh.bounds
    assert extent is not None
    low, high = extent
    for axis in range(3):
        # Outside the sampled cube, by less than a voxel and a half, on all six
        # sides: the solid contains the samples and stands off them.
        assert -1.0 - 1.5 * voxel <= low[axis] <= -1.0
        assert 1.0 <= high[axis] <= 1.0 + 1.5 * voxel
        # And the standoff is the same on both sides. A grid quantised without
        # its half-cell offset shifts the whole shape by half a voxel, which
        # each bound above would still admit and this one will not: a
        # reconstruction may stand off its samples, and may not move them.
        assert abs(low[axis] + high[axis]) < 0.5 * voxel
    for face in cube_mesh.faces:
        assert triangle_area(cube_mesh, face) > 0.0


def test_an_open_surface_is_measured_as_not_watertight(sphere_mesh):
    """A hemisphere: the same vertices, half the faces, and a rim of open edges.

    Cut from a mesh that measured closed, so the difference between the two
    results is the hole and nothing else.
    """
    assert sphere_mesh.is_watertight
    upper = tuple(face for face in sphere_mesh.faces if face_centroid(sphere_mesh, face)[2] > 0.0)
    hemisphere = Mesh(vertices=sphere_mesh.vertices, faces=upper, limitations=("cut by a test",))
    assert 0 < hemisphere.face_count < sphere_mesh.face_count
    census = hemisphere.census
    assert not hemisphere.is_watertight
    assert census.boundary_edge_count > 0, "an open hemisphere has a rim"
    assert census.nonmanifold_edge_count == 0
    assert not census.closed
    measured = hemisphere.as_dict()["watertight"]
    assert isinstance(measured, dict)
    assert measured["measured"] is False


def test_a_hemisphere_of_samples_encloses_nothing_and_the_mesh_says_so():
    """An open cloud leaks, and the record says it leaked rather than hiding it.

    The fill reaches the inside through the missing half, so nothing is enclosed
    and what comes back is a shell wrapped round the samples - closed, because
    marching cubes over a padded grid always closes, and closed is exactly the
    result that would mislead a reader without this limitation attached.
    """
    bowl = [point for point in sphere_points(4000, 1.0) if point[2] >= 0.0]
    mesh = reconstruct(bowl, voxel_size=0.1)
    assert mesh.face_count > 0
    assert mesh.is_watertight
    leak = [line for line in mesh.limitations if "encloses no volume" in line]
    assert len(leak) == 1, mesh.limitations
    assert "shell around the samples" in leak[0]

    solid = reconstruct(sphere_points(4000, 1.0), voxel_size=0.1)
    enclosed = [line for line in solid.limitations if "cells were enclosed" in line]
    assert len(enclosed) == 1, solid.limitations
    assert not any("encloses no volume" in line for line in solid.limitations)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_box_with_one_face_left_off_encloses_nothing(axis, side):
    """The fill has to be able to walk in through a hole on any of the six sides.

    Six directions, six ways to get this wrong, and a fill missing one of them
    still classifies a sphere perfectly - every slice of a sphere reaches the
    grid border sideways. It fails only on a shape whose cavity is reachable
    from exactly one direction, which is what a box with one face missing is.
    The closed box in the same test is the control: it must enclose.
    """
    closed = reconstruct(box_points(24, 1.0), voxel_size=0.15)
    assert any("cells were enclosed" in line for line in closed.limitations), (
        "a closed box has an inside"
    )
    opened = reconstruct(box_points(24, 1.0, skip=(axis, side)), voxel_size=0.15)
    assert any("encloses no volume" in line for line in opened.limitations), (
        f"the fill could not reach the cavity through the missing {axis}/{side} face"
    )
    # And the open box keeps both of its walls, so it has strictly more surface
    # than the solid the closed one becomes.
    assert opened.face_count > closed.face_count


def test_a_reconstruction_records_its_grid_and_its_own_closure(sphere_mesh):
    assert INHERENT_LIMITATIONS[0] in sphere_mesh.limitations
    assert all(limitation in sphere_mesh.limitations for limitation in INHERENT_LIMITATIONS)
    grid = [line for line in sphere_mesh.limitations if "reconstructed at voxel size" in line]
    assert len(grid) == 1
    assert "0.1" in grid[0]
    closure = [line for line in sphere_mesh.limitations if line.startswith("measured closed")]
    assert len(closure) == 1
    assert str(sphere_mesh.census.edge_count) in closure[0]


def test_an_isolated_sample_produces_no_surface_and_the_record_says_so():
    """The documented sparse-sample behaviour, exercised rather than asserted.

    One sample alone fills one cell, which puts at most one solid cell around
    any node - short of the four the level asks for. Nothing is drawn, and the
    result says nothing was drawn instead of returning a shape.
    """
    mesh = reconstruct([(0.0, 0.0, 0.0)], voxel_size=0.1)
    assert mesh.face_count == 0
    assert mesh.vertex_count == 0
    assert not mesh.is_watertight, "an empty mesh is not a closed surface"
    assert mesh.bounds is None
    assert any("no surface was found" in line for line in mesh.limitations)


def test_lowering_the_level_keeps_the_sample_the_default_drops():
    """``iso_level`` is the documented knob, so it is shown doing what is claimed."""
    kept = reconstruct([(0.0, 0.0, 0.0)], voxel_size=0.1, iso_level=0.5)
    assert kept.face_count > 0
    assert kept.is_watertight


@pytest.mark.parametrize("seed", range(30))
def test_random_clouds_all_reconstruct_to_closed_surfaces(seed):
    """Arbitrary blobs, not just the two shapes chosen to work.

    Random occupancy at a coarse voxel size is what actually reaches the awkward
    cases - ambiguous faces, cells touching at a corner, one-cell bridges - and
    every one of them still has to close.
    """
    state = seed * 2654435761 + 12345
    values: list[float] = []
    for _ in range(3 * 40):
        state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
        values.append((state >> 33) / float(2**31))
    points = [
        (values[index], values[index + 1], values[index + 2]) for index in range(0, len(values), 3)
    ]
    mesh = reconstruct(points, voxel_size=0.22)
    if mesh.face_count == 0:
        return
    census = mesh.census
    assert census.closed, (seed, census.as_dict())
    assert census.consistently_oriented, (seed, census.as_dict())
    # A closed orientable surface has an even Euler characteristic, whatever its
    # genus or how many pieces it is in.
    assert euler_characteristic(mesh) % 2 == 0, seed
    for face in mesh.faces:
        assert triangle_area(mesh, face) > 0.0


def test_two_runs_over_one_cloud_produce_the_same_mesh():
    """Determinism, because two exports of one cloud must be the same file."""
    points = sphere_points(500, 1.0)
    first = reconstruct(points, voxel_size=0.2)
    second = reconstruct(list(reversed(points)), voxel_size=0.2)
    assert first.vertices == second.vertices
    assert first.faces == second.faces
    assert first.limitations == second.limitations


# --------------------------------------------------------------------------
# Watertightness is a measurement, so the measurement gets its own tests
# --------------------------------------------------------------------------


def tetrahedron() -> Mesh:
    """Four vertices, four faces, wound so every normal points outwards."""
    return Mesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        faces=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        limitations=("a hand-built tetrahedron; it measures nothing",),
    )


def test_a_closed_hand_built_surface_measures_closed():
    mesh = tetrahedron()
    census = mesh.census
    assert mesh.is_watertight
    assert census.edge_count == 6
    assert census.shared_edge_count == 6
    assert census.boundary_edge_count == 0
    assert census.consistently_oriented
    assert euler_characteristic(mesh) == 2


def test_a_single_triangle_measures_open():
    mesh = Mesh(vertices=tetrahedron().vertices, faces=((0, 1, 2),))
    assert not mesh.is_watertight
    assert mesh.census.boundary_edge_count == 3


def test_an_edge_shared_by_three_faces_measures_non_manifold():
    """A fin: three triangles hinged on one edge. Closed by no honest reading."""
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
        faces=((0, 1, 2), (0, 1, 3), (0, 1, 4)),
    )
    census = mesh.census
    assert census.nonmanifold_edge_count == 1
    assert not mesh.is_watertight
    assert not census.closed


def test_an_empty_mesh_is_not_watertight():
    mesh = Mesh(vertices=(), faces=())
    assert not mesh.is_watertight
    assert mesh.census == edge_census(())
    assert mesh.bounds is None


def test_a_closed_surface_wound_inconsistently_is_reported_as_such():
    """Closed and consistently oriented are two measurements, not one."""
    flipped = tetrahedron()
    faces = (flipped.faces[0][::-1], *flipped.faces[1:])
    mesh = Mesh(vertices=flipped.vertices, faces=faces)
    census = mesh.census
    assert mesh.is_watertight, "flipping one face changes no edge's multiplicity"
    assert not census.consistently_oriented
    assert census.reversed_edge_count < census.shared_edge_count


def test_the_mesh_record_carries_counts_and_caveats_and_no_geometry(sphere_mesh):
    record = sphere_mesh.as_dict()
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["vertex_count"] == sphere_mesh.vertex_count
    assert record["face_count"] == sphere_mesh.face_count
    assert record["dimensional_accuracy_claimed"] is False
    assert record["watertight"]["measured"] is True
    assert record["limitations"] == list(sphere_mesh.limitations)
    # The geometry belongs in the exported file, not in a second copy here.
    assert "vertices" not in record
    assert "faces" not in record


# --------------------------------------------------------------------------
# Refusals. Every one of them happens before anything is allocated or opened
# --------------------------------------------------------------------------


def test_a_grid_past_the_cap_is_refused_rather_than_allocated():
    points = sphere_points(200, 1.0)
    with pytest.raises(GridTooLarge) as raised:
        reconstruct(points, voxel_size=1e-6)
    assert "nothing was allocated" in str(raised.value)
    # And a caller's own lower cap binds too.
    reconstruct(points, voxel_size=0.25, max_grid_cells=4096)
    with pytest.raises(GridTooLarge):
        reconstruct(points, voxel_size=0.25, max_grid_cells=100)


def test_the_cell_cap_cannot_be_widened_at_run_time():
    with pytest.raises(MeshNotAllowed):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=1.0, max_grid_cells=MAX_GRID_CELLS + 1)
    with pytest.raises(MeshNotAllowed):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=1.0, max_grid_cells=0)


@pytest.mark.parametrize("voxel_size", [0.0, -1.0, -0.0001])
def test_a_voxel_size_that_defines_no_grid_is_refused(voxel_size):
    with pytest.raises(VoxelSizeUnusable):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=voxel_size)


def test_a_non_finite_voxel_size_is_refused():
    with pytest.raises(NonFiniteValue):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=float("nan"))


@pytest.mark.parametrize("iso_level", [0.0, 8.0, -1.0, 9.0])
def test_an_iso_level_no_surface_can_live_at_is_refused(iso_level):
    with pytest.raises(IsoLevelUnusable):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=0.1, iso_level=iso_level)


def test_an_empty_point_set_is_refused_rather_than_answered_with_an_empty_mesh():
    with pytest.raises(NoPointsOffered):
        reconstruct([], voxel_size=0.1)


def test_more_points_than_the_cap_is_refused_while_they_stream_past():
    consumed = 0

    def endless():
        nonlocal consumed
        while True:
            consumed += 1
            yield (0.0, 0.0, 0.0)

    with pytest.raises(PointCountExceeded):
        reconstruct(endless(), voxel_size=0.1, max_points=10)
    # Stopped at the cap, not drained: an unbounded producer must not be
    # allowed to run to exhaustion first.
    assert consumed == 11


def test_the_point_cap_cannot_be_widened_at_run_time():
    with pytest.raises(MeshNotAllowed):
        reconstruct([(0.0, 0.0, 0.0)], voxel_size=0.1, max_points=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_coordinate_is_refused(bad):
    with pytest.raises(NonFiniteValue):
        reconstruct([(0.0, 0.0, 0.0), (bad, 0.0, 0.0)], voxel_size=0.1)


def test_a_point_that_is_not_a_triple_is_refused():
    with pytest.raises(MeshNotAllowed):
        reconstruct([(0.0, 0.0)], voxel_size=0.1)


# --------------------------------------------------------------------------
# Independent readers: they know OBJ and PLY, and nothing about the writer
# --------------------------------------------------------------------------


class ParsedObj:
    """An OBJ parsed by the format's own rules alone."""

    def __init__(self, raw: bytes) -> None:
        self.comments: list[str] = []
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []
        for line in raw.decode("ascii").splitlines():
            if line.startswith("# "):
                self.comments.append(line[2:])
                continue
            tokens = line.split()
            assert tokens, "an OBJ from this writer has no blank lines"
            if tokens[0] == "v":
                assert len(tokens) == 4
                x, y, z = (float(value) for value in tokens[1:])
                self.vertices.append((x, y, z))
            elif tokens[0] == "f":
                assert len(tokens) == 4
                # OBJ indexes from one, and a reader that forgets is off by one
                # in a way that still renders.
                a, b, c = (int(value) - 1 for value in tokens[1:])
                self.faces.append((a, b, c))
            else:
                raise AssertionError(f"unexpected OBJ statement {tokens[0]!r}")


class ParsedPlyMesh:
    """A PLY mesh parsed by the format's own rules alone, ASCII or binary."""

    def __init__(self, raw: bytes) -> None:
        lines: list[str] = []
        offset = 0
        while True:
            newline = raw.index(b"\n", offset)
            line = raw[offset:newline].decode("ascii")
            offset = newline + 1
            lines.append(line)
            if line.split()[:1] == ["end_header"]:
                break
        assert lines[0] == "ply"
        _, self.format, self.version = lines[1].split()
        self.comments = [line[len("comment ") :] for line in lines if line.startswith("comment ")]
        self.elements: list[tuple[str, int]] = []
        self.vertex_properties: list[tuple[str, str]] = []
        self.face_property: str | None = None
        for line in lines:
            tokens = line.split()
            if tokens[:1] == ["element"]:
                self.elements.append((tokens[1], int(tokens[2])))
            elif tokens[:1] == ["property"] and self.elements[-1][0] == "vertex":
                self.vertex_properties.append((tokens[1], tokens[2]))
            elif tokens[:1] == ["property"] and self.elements[-1][0] == "face":
                self.face_property = " ".join(tokens[1:])
        counts = dict(self.elements)
        body = raw[offset:]
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, ...]] = []
        if self.format == "ascii":
            rows = body.decode("ascii").splitlines()
            for row in rows[: counts["vertex"]]:
                x, y, z = (float(value) for value in row.split())
                self.vertices.append((x, y, z))
            for row in rows[counts["vertex"] : counts["vertex"] + counts["face"]]:
                tokens = row.split()
                assert int(tokens[0]) == len(tokens) - 1
                self.faces.append(tuple(int(value) for value in tokens[1:]))
            assert len(rows) == counts["vertex"] + counts["face"]
        else:
            assert self.format == "binary_little_endian"
            for index in range(counts["vertex"]):
                self.vertices.append(struct.unpack_from("<fff", body, index * 12))
            at = counts["vertex"] * 12
            for _ in range(counts["face"]):
                (size,) = struct.unpack_from("<B", body, at)
                at += 1
                self.faces.append(struct.unpack_from(f"<{size}i", body, at))
                at += 4 * size
            assert at == len(body)


# --------------------------------------------------------------------------
# Export: whole files, or none
# --------------------------------------------------------------------------


def test_an_obj_round_trips_through_a_reader_that_knows_only_obj(tmp_path, sphere_mesh):
    path = tmp_path / "sphere.obj"
    export = write_obj(path, sphere_mesh, provenance=provenance())
    parsed = ParsedObj(path.read_bytes())
    assert len(parsed.vertices) == sphere_mesh.vertex_count
    assert len(parsed.faces) == sphere_mesh.face_count
    assert parsed.faces == list(sphere_mesh.faces)
    for read_back, original in zip(parsed.vertices, sphere_mesh.vertices, strict=True):
        assert read_back == tuple(binary32(value) for value in original)
    assert export.path == str(path)
    assert export.file_format is MeshFileFormat.OBJ
    assert export.byte_count == len(path.read_bytes())
    assert export.watertight is True
    # A mesh parsed back out of the file is the same closed surface.
    assert rebuilt(parsed).is_watertight


@pytest.mark.parametrize("ply_format", list(PlyFormat))
def test_a_ply_mesh_round_trips_through_a_reader_that_knows_only_ply(
    tmp_path, sphere_mesh, ply_format
):
    path = tmp_path / f"sphere-{ply_format}.ply"
    export = write_ply_mesh(path, sphere_mesh, provenance=provenance(), ply_format=ply_format)
    parsed = ParsedPlyMesh(path.read_bytes())
    assert parsed.format == str(ply_format)
    assert parsed.elements == [
        ("vertex", sphere_mesh.vertex_count),
        ("face", sphere_mesh.face_count),
    ]
    assert parsed.vertex_properties == [("float", "x"), ("float", "y"), ("float", "z")]
    assert parsed.face_property == "list uchar int vertex_indices"
    assert [tuple(face) for face in parsed.faces] == list(sphere_mesh.faces)
    for read_back, original in zip(parsed.vertices, sphere_mesh.vertices, strict=True):
        assert read_back == tuple(binary32(value) for value in original)
    assert export.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rebuilt(parsed).is_watertight


def test_the_three_encodings_of_one_mesh_carry_identical_coordinates(cube_mesh):
    """Choosing a file format must not change the data inside it."""
    obj = ParsedObj(encode_obj(cube_mesh, provenance=provenance()).payload)
    ascii_ply = ParsedPlyMesh(
        encode_ply_mesh(cube_mesh, provenance=provenance(), ply_format=PlyFormat.ASCII).payload
    )
    binary_ply = ParsedPlyMesh(
        encode_ply_mesh(
            cube_mesh, provenance=provenance(), ply_format=PlyFormat.BINARY_LITTLE_ENDIAN
        ).payload
    )
    assert obj.vertices == ascii_ply.vertices == binary_ply.vertices
    assert obj.faces == [tuple(face) for face in ascii_ply.faces]
    assert obj.faces == [tuple(face) for face in binary_ply.faces]


def test_every_export_carries_its_caveats_and_claims_no_unit(tmp_path, cube_mesh):
    """The caveats travel in the file, because there is nowhere else to put them."""
    obj_path = tmp_path / "cube.obj"
    ply_path = tmp_path / "cube.ply"
    obj = write_obj(obj_path, cube_mesh, provenance=provenance())
    ply = write_ply_mesh(ply_path, cube_mesh, provenance=provenance())
    obj_comments = " ".join(ParsedObj(obj_path.read_bytes()).comments)
    ply_comments = " ".join(ParsedPlyMesh(ply_path.read_bytes()).comments)
    for block in (obj_comments, ply_comments):
        for limitation in INHERENT_LIMITATIONS:
            # Wrapped across lines in the file, so the check is on the words.
            for fragment in limitation.split(";")[0].split():
                assert fragment in block
        assert "dimensional accuracy: none is claimed" in block
        assert "closed surface: yes, measured by counting edges" in block
        # A physical unit may appear only inside a sentence denying it. The
        # limitation that says these numbers are "not millimetres" has to name
        # millimetres to say so; what must never appear is one used as a claim.
        lowered = block.lower()
        for unit in ("millimetre", "millimeter", " mm ", "centimetre", "metres", "inches"):
            at = lowered.find(unit)
            while at != -1:
                window = lowered[max(0, at - 220) : at + 60]
                assert any(
                    denial in window
                    for denial in ("not ", "no ", "unknown", "unestablished", "unvalidated")
                ), f"{unit!r} is stated without a denial around it: {window!r}"
                at = lowered.find(unit, at + 1)
    for export in (obj, ply):
        assert set(INHERENT_LIMITATIONS) <= set(export.limitations)
        assert set(cube_mesh.limitations) <= set(export.limitations)
        assert len(export.limitations) == len(set(export.limitations)), "de-duplicated"


def test_the_export_record_says_what_was_written(tmp_path, cube_mesh):
    path = tmp_path / "cube.ply"
    export = write_ply_mesh(path, cube_mesh, provenance=provenance())
    record = export.as_dict()
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["artifact"] == {
        "path": str(path),
        "written": True,
        "format": "ply-ascii",
        "byte_count": path.stat().st_size,
        "sha256": export.sha256,
    }
    assert record["dimensional_accuracy_claimed"] is False
    assert sub(record, "watertight")["measured"] is True
    assert sub(record, "edges") == export.census.as_dict()
    assert sub(record, "bounds")["module_max_output_bytes"] == MAX_OUTPUT_BYTES
    # Encoded but unwritten is a different record, and says so.
    unwritten = encode_ply_mesh(cube_mesh, provenance=provenance()).as_dict()["artifact"]
    assert unwritten == {**sub(record, "artifact"), "path": None, "written": False}


# --- refusals that must leave nothing behind ----------------------------------


def bad_meshes() -> list[tuple[str, Mesh, type[Exception]]]:
    good = tetrahedron()
    return [
        ("no faces at all", Mesh(vertices=good.vertices, faces=()), EmptyMesh),
        (
            "an index past the end",
            Mesh(vertices=good.vertices, faces=((0, 1, 9),)),
            MeshIndexOutOfRange,
        ),
        (
            "a negative index",
            Mesh(vertices=good.vertices, faces=((0, 1, -1),)),
            MeshIndexOutOfRange,
        ),
        (
            "one vertex named twice",
            Mesh(vertices=good.vertices, faces=((0, 1, 1),)),
            DegenerateFace,
        ),
        (
            "a coordinate that is not finite",
            Mesh(vertices=((float("nan"), 0.0, 0.0), *good.vertices[1:]), faces=good.faces),
            NonFiniteValue,
        ),
        (
            "a coordinate no PLY float can hold",
            Mesh(vertices=((1e39, 0.0, 0.0), *good.vertices[1:]), faces=good.faces),
            MeshCoordinateNotEncodable,
        ),
    ]


@pytest.mark.parametrize(("writer", "suffix"), [(write_obj, ".obj"), (write_ply_mesh, ".ply")])
@pytest.mark.parametrize(("what", "mesh", "expected"), bad_meshes(), ids=lambda value: value)
def test_a_refused_mesh_leaves_no_file(tmp_path, writer, suffix, what, mesh, expected):
    path = tmp_path / f"refused{suffix}"
    with pytest.raises(expected):
        writer(path, mesh, provenance=provenance())
    assert not path.exists(), f"{what} left a file behind"


def test_an_encoding_past_the_byte_ceiling_is_refused_before_the_path_is_touched(
    tmp_path, cube_mesh
):
    path = tmp_path / "too-big.obj"
    with pytest.raises(OutputSizeExceeded):
        write_obj(path, cube_mesh, provenance=provenance(), max_output_bytes=4096)
    assert not path.exists()


def test_the_byte_ceiling_cannot_be_widened_at_run_time(tmp_path, cube_mesh):
    path = tmp_path / "widened.obj"
    with pytest.raises(MeshNotAllowed):
        write_obj(path, cube_mesh, provenance=provenance(), max_output_bytes=MAX_OUTPUT_BYTES + 1)
    with pytest.raises(MeshNotAllowed):
        write_obj(path, cube_mesh, provenance=provenance(), max_output_bytes=0)
    assert not path.exists()


def test_a_comment_block_past_the_ceiling_is_refused_rather_than_truncated(tmp_path):
    path = tmp_path / "shouty.obj"
    wordy = provenance(limitations=tuple(f"caveat {index} " + "x" * 200 for index in range(200)))
    with pytest.raises(MeshNotAllowed) as raised:
        write_obj(path, tetrahedron(), provenance=wordy)
    assert "refused rather than truncated" in str(raised.value)
    assert not path.exists()
    with pytest.raises(MeshNotAllowed):
        write_ply_mesh(path, tetrahedron(), provenance=wordy)
    assert not path.exists()


def test_the_vertex_and_face_caps_refuse(tmp_path, monkeypatch):
    """The real caps sit above what a grid can produce, so they are lowered here.

    A mesh with four million vertices cannot be built inside a test that fits
    its time budget, and a guard that is never executed is a guard nobody knows
    the state of.
    """
    from scanner3d import mesh as module

    path = tmp_path / "capped.obj"
    monkeypatch.setattr(module, "MAX_VERTICES", 2)
    with pytest.raises(MeshCountExceeded):
        write_obj(path, tetrahedron(), provenance=provenance())
    assert not path.exists()
    monkeypatch.setattr(module, "MAX_VERTICES", 1024)
    monkeypatch.setattr(module, "MAX_FACES", 2)
    with pytest.raises(MeshCountExceeded):
        write_obj(path, tetrahedron(), provenance=provenance())
    assert not path.exists()


@pytest.mark.parametrize(("writer", "suffix"), [(write_obj, ".obj"), (write_ply_mesh, ".ply")])
def test_an_existing_name_is_never_overwritten(tmp_path, writer, suffix):
    path = tmp_path / f"taken{suffix}"
    path.write_bytes(b"someone else's file")
    with pytest.raises(MeshWriteIncomplete) as raised:
        writer(path, tetrahedron(), provenance=provenance())
    assert raised.value.errno == errno.EEXIST
    assert path.read_bytes() == b"someone else's file"


class WriteFault:
    """Misbehave on one file's descriptor and on no other.

    Targeted by path, because a blanket patch of ``os.write`` breaks pytest's
    own output before the writer under test ever runs. Descriptor numbers are
    reused, so a watched fd stops being watched the moment it is closed or the
    next unrelated file inherits the fault.
    """

    def __init__(self, target: Path, behaviour: str) -> None:
        self.target = target
        self.behaviour = behaviour
        self.calls = 0
        self.watched: set[int] = set()
        self.real_open = os.open
        self.real_write = os.write
        self.real_close = os.close

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(os, "open", self._open)
        monkeypatch.setattr(os, "write", self._write)
        monkeypatch.setattr(os, "close", self._close)

    def _open(self, path, flags, mode=0o777, **kwargs):
        fd = self.real_open(path, flags, mode, **kwargs)
        if Path(path) == self.target:
            self.watched.add(fd)
        return fd

    def _close(self, fd: int) -> None:
        self.watched.discard(fd)
        self.real_close(fd)

    def _write(self, fd: int, data) -> int:
        if fd not in self.watched:
            return self.real_write(fd, data)
        self.calls += 1
        payload = bytes(data)
        if self.behaviour == "stall":
            return 0
        if self.behaviour == "short":
            # One byte at a time: the writer has to resume from where the
            # kernel actually stopped, over and over, and still produce the
            # whole file.
            return self.real_write(fd, payload[:1])
        if self.calls == 1:
            return self.real_write(fd, payload[:2])
        raise OSError(errno.EIO, "synthetic failure")


def test_a_stalled_write_is_an_error_rather_than_a_spin(tmp_path, monkeypatch):
    path = tmp_path / "stalled.obj"
    WriteFault(path, "stall").install(monkeypatch)
    with pytest.raises(MeshWriteIncomplete) as raised:
        write_obj(path, tetrahedron(), provenance=provenance())
    assert raised.value.errno == errno.EIO
    assert not path.exists()


def test_a_write_that_fails_part_way_through_leaves_nothing(tmp_path, monkeypatch):
    path = tmp_path / "half.ply"
    WriteFault(path, "eio").install(monkeypatch)
    with pytest.raises(OSError) as raised:
        write_ply_mesh(path, tetrahedron(), provenance=provenance())
    assert raised.value.errno == errno.EIO
    assert not path.exists(), "a truncated mesh is worse than no mesh"


def test_a_short_write_is_resumed_until_the_file_is_whole(tmp_path, monkeypatch):
    """``os.write`` may write fewer bytes than it was given; one call is not a file."""
    path = tmp_path / "dribbled.obj"
    fault = WriteFault(path, "short")
    fault.install(monkeypatch)
    export = write_obj(path, tetrahedron(), provenance=provenance())
    assert fault.calls == export.byte_count, "one call per byte, all of them resumed"
    assert path.read_bytes() == export.payload
    assert len(ParsedObj(path.read_bytes()).faces) == 4


# --------------------------------------------------------------------------
# One real capture, reconstructed offline
# --------------------------------------------------------------------------

#: The unit's own factory numbers, injected rather than read from a device.
REAL_FX = REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297
REAL_SCALE = 0.004999999888241291

FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"


def real_frame() -> bytes:
    """One committed 512000-byte capture, or an explicit skip. A skip is not a pass."""
    if FIXTURE_FRAMES.is_dir():
        for candidate in sorted(FIXTURE_FRAMES.glob("*.bin")):
            if candidate.stat().st_size == EXPECTED_FRAME_BYTES:
                return candidate.read_bytes()
    pytest.skip(
        f"no real {EXPECTED_FRAME_BYTES}-byte depth frame under {FIXTURE_FRAMES}; the path "
        "from real captured bytes to a mesh was NOT exercised on this run, and this skip is "
        "not a pass"
    )


def test_one_real_captured_frame_reconstructs_and_exports(tmp_path):
    """Real bytes in, a parseable mesh file out, with nothing dimensional claimed.

    Structure only. Nothing here asserts an extent, a size or a distance,
    because no measurement in this repository has established any. One frame,
    at a voxel size coarse enough to keep the grid small - this is a shape test,
    not a benchmark.
    """
    blob = struct.pack(
        INTRINSICS_FORMAT, 640, 400, REAL_FX, 0.0, REAL_CX, 0.0, REAL_FY, REAL_CY, 0.0, 0.0, 1.0
    )
    intrinsics = parse_intrinsics(blob)
    assert intrinsics is not None
    conversion = deproject_depth_frame(
        real_frame(), intrinsics=intrinsics, depth_scale=REAL_SCALE, unit=UNVALIDATED_UNIT
    )
    points = list(camera_points(conversion))
    assert points, "a frame with no returns cannot exercise this path"
    span = max(
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    )
    mesh = reconstruct(points, voxel_size=span / 40.0)
    assert mesh.face_count > 0
    assert mesh.is_watertight
    assert mesh.census.consistently_oriented
    for face in mesh.faces:
        assert triangle_area(mesh, face) > 0.0

    # Every vertex is inside the cloud's own extent, widened by the standoff
    # the grid adds. Nothing here says what that extent measures.
    voxel = span / 40.0
    extent = mesh.bounds
    assert extent is not None
    low, high = extent
    for axis in range(3):
        assert low[axis] >= min(point[axis] for point in points) - 2.0 * voxel
        assert high[axis] <= max(point[axis] for point in points) + 2.0 * voxel

    path = tmp_path / "real.ply"
    export = write_ply_mesh(
        path,
        mesh,
        provenance=provenance(
            voxel_size=voxel,
            cloud=CloudProvenance(
                units="device depth sample units times the factory scalar; unit unestablished",
                convention=CAMERA_OPTICAL_FRAME,
                calibration=CalibrationProvenance(
                    source="tests/test_mesh.py: the unit's factory record, injected",
                    depth_scale=REAL_SCALE,
                ),
            ),
        ),
    )
    parsed = ParsedPlyMesh(path.read_bytes())
    assert len(parsed.vertices) == mesh.vertex_count
    assert len(parsed.faces) == mesh.face_count
    assert export.watertight is True
    assert any("no physical measurement has validated" in line for line in export.limitations)
