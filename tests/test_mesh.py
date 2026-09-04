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
from types import MappingProxyType

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
    MAX_HEADER_BYTES,
    MAX_OUTPUT_BYTES,
    METHOD_LIMITATIONS,
    PAD_CELLS,
    SCHEMA_VERSION,
    TRIANGLE_TABLE,
    TSDF_INHERENT_LIMITATIONS,
    DegenerateFace,
    EmptyMesh,
    GridTooLarge,
    IsoLevelUnusable,
    Mesh,
    MeshCoordinateNotEncodable,
    MeshCountExceeded,
    MeshFileFormat,
    MeshIndexOutOfRange,
    MeshMethod,
    MeshNotAllowed,
    MeshProvenance,
    MeshWriteIncomplete,
    NoPointsOffered,
    OutputSizeExceeded,
    PointCountExceeded,
    SurfaceAccumulator,
    VoxelSizeUnusable,
    binary32,
    case_index,
    edge_census,
    encode_obj,
    encode_ply_mesh,
    provenance_comments,
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
from scanner3d.pointcloud import MAX_POINTS as MAX_INPUT_POINTS
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


def flattened(lines: list[str]) -> str:
    """Every line with its internal whitespace collapsed away, then concatenated.

    A comment block is prose that :func:`textwrap.wrap` has broken onto several
    lines - sometimes mid-word, at a hyphen. A literal substring search over the
    joined lines is exactly what that wrapping defeats. Stripping whitespace out
    of each line before joining removes the one thing wrapping is allowed to
    change - where a space or a newline falls - while a hyphen, which wrapping
    never inserts or deletes, survives untouched; comparing two strings put
    through the same reduction is then a search for the original sentence
    regardless of where it happened to be broken.
    """
    return "".join("".join(line.split()) for line in lines)


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


@pytest.mark.parametrize("iso_level", [1.0, 4.0, 7.0])
def test_an_integer_level_still_closes_and_never_collapses_a_triangle(iso_level):
    """The level the module warns about, exercised instead of only described.

    ``_ensure_iso_level`` admits any level in ``(0, 8)``, and an *integer* one is
    the case the half-integer default exists to avoid: the field is an integer
    count, so a node can sit exactly on the level, and the crossing on every edge
    touching that node would land exactly on it. Two cut edges would then produce
    one point and the triangle between them would have no area at all.

    ``MIN_EDGE_FRACTION`` is what stops that, and nothing exercised it: the whole
    suite used half-integer levels, so the clamp could be deleted and the
    ``>``-at-the-level rule in :func:`_march` could be relaxed to ``>=`` with
    every test still passing. Both are pinned here.

    What is asserted is what the module actually promises - closed, consistently
    oriented, no face naming a vertex twice, no face of zero area, and no two
    vertices at the same point. Not that the triangles are well shaped: at an
    integer level they are slivers, which is the documented cost of the choice
    and the reason the default is not one.
    """
    mesh = reconstruct(sphere_points(2000, 1.0), voxel_size=0.12, iso_level=iso_level)
    assert mesh.face_count > 0
    assert mesh.is_watertight, "an integer level must not tear the surface open"
    assert mesh.census.consistently_oriented
    for face in mesh.faces:
        assert len(set(face)) == 3, face
        assert triangle_area(mesh, face) > 0.0, face
    # No two vertices coincide, which is the property the clamp buys and the
    # property a vertex landing exactly on a node would destroy.
    assert len(set(mesh.vertices)) == mesh.vertex_count, "two cut edges produced one point"
    # And the surface still exports: a mesh this module hands back is a mesh its
    # own writers accept, at every level it accepts.
    export = encode_obj(mesh, provenance=provenance(iso_level=iso_level))
    assert export.face_count == mesh.face_count


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
# MeshProvenance's method family: two constructions, two sets of caveats, one
# record that cannot be talked into dropping either
# --------------------------------------------------------------------------


def test_a_provenance_built_without_naming_a_family_carries_every_inherent_limitation_first():
    """The legacy safety claim: every call site that predates ``method_family``
    still gets occupancy's whole caveat list, in order, at the front of the
    record.

    Nothing in this repository names a family yet outside this test file - every
    real call to :class:`MeshProvenance` looks exactly like :func:`provenance`
    here, with no ``method_family`` keyword at all. If the default silently
    dropped or reordered a caveat, every one of those existing files would be
    weaker than the ones written before this field existed, and nothing about
    the call site would have changed to warn a reader.
    """
    prov = provenance()
    assert prov.method_family == MeshMethod.OCCUPANCY
    assert prov.inherent_limitations == INHERENT_LIMITATIONS
    assert prov.all_limitations[: len(INHERENT_LIMITATIONS)] == INHERENT_LIMITATIONS


def test_selecting_tsdf_carries_only_the_tsdf_caveats_and_none_of_occupancys_own():
    """A fused field has different inherent defects, and occupancy's must not follow it.

    Six of occupancy's eight caveats are about the occupancy construction
    specifically and nothing a TSDF fusion does: the surface standing off the
    samples because a cell counts as solid when any sample falls anywhere in
    it, an iso level counted over a node's eight touching cells, a
    six-connected flood fill from the grid border deciding what is interior,
    and a padded grid that always closes whether or not the object was seen
    from every side. None of that happened to a level set of a fused signed
    distance field, so a TSDF-family provenance must not carry any of them.
    """
    prov = provenance(method_family=MeshMethod.TSDF)
    assert prov.inherent_limitations == TSDF_INHERENT_LIMITATIONS
    occupancy_only = set(INHERENT_LIMITATIONS) - set(TSDF_INHERENT_LIMITATIONS)
    # Sanity: there really is something occupancy-only to exclude, or this test
    # would pass for a reason that has nothing to do with what it claims.
    assert len(occupancy_only) == 6
    for limitation in occupancy_only:
        assert limitation not in prov.all_limitations


def test_the_caveats_shared_by_both_families_are_the_same_string_object():
    """Shared prose has to be repeated verbatim, not respelled, or the two
    families drift into two sentences that were supposed to be one.

    The module docstring says this is done by reusing the same string object
    rather than typing it twice, which is checked here with ``is`` rather than
    ``==``: two equal-but-distinct strings would already satisfy equality and
    still be exactly the drift this is meant to catch.
    """
    assert TSDF_INHERENT_LIMITATIONS[0] is INHERENT_LIMITATIONS[0]
    assert TSDF_INHERENT_LIMITATIONS[-1] is INHERENT_LIMITATIONS[-1]
    shared = set(INHERENT_LIMITATIONS) & set(TSDF_INHERENT_LIMITATIONS)
    assert shared == {INHERENT_LIMITATIONS[0], INHERENT_LIMITATIONS[-1]}


def test_a_caller_cannot_drop_or_replace_an_inherent_caveat_via_limitations():
    """``limitations`` adds to a family's caveats; it is not a way to compose them.

    Passing an empty tuple, passing something unrelated, and restating one of
    the inherent caveats verbatim are three different ways a caller might try
    to end up with a shorter or edited list. None of them can: the family's set
    is always there, in front, and restating one does not duplicate it.
    """
    dropped = provenance(limitations=())
    assert dropped.all_limitations[: len(INHERENT_LIMITATIONS)] == INHERENT_LIMITATIONS

    unrelated = provenance(limitations=("the lighting rig was not colour calibrated",))
    assert unrelated.all_limitations[: len(INHERENT_LIMITATIONS)] == INHERENT_LIMITATIONS
    assert "the lighting rig was not colour calibrated" in unrelated.all_limitations

    restated = provenance(limitations=(INHERENT_LIMITATIONS[0],))
    assert restated.all_limitations.count(INHERENT_LIMITATIONS[0]) == 1


def test_method_limitations_is_exhaustive_over_meshmethod_and_not_mutable():
    """A family with no entry would be a family whose meshes carry no inherent
    caveats at all, silently, so the mapping is checked to cover every member -
    and to be a :class:`~types.MappingProxyType`, so a caller holding a
    reference to it cannot edit what every future provenance reads from.
    """
    assert set(METHOD_LIMITATIONS) == set(MeshMethod)
    assert isinstance(METHOD_LIMITATIONS, MappingProxyType)
    with pytest.raises(TypeError):
        METHOD_LIMITATIONS[MeshMethod.OCCUPANCY] = ()  # type: ignore[index]


def test_as_dict_and_the_written_file_agree_on_which_family_a_tsdf_mesh_belongs_to(tmp_path):
    """The record and the artifact must tell the same story about a TSDF mesh.

    A tiny hand-built mesh stands in for a real TSDF surface - what matters here
    is the provenance, not the geometry. ``as_dict`` names the family
    explicitly, and the OBJ and PLY files it is written into must carry TSDF's
    caveats and none of occupancy's own, the same way :func:`provenance_comments`
    assembles them.
    """
    prov = provenance(method_family=MeshMethod.TSDF, method="tsdf fusion + marching cubes")
    record = prov.as_dict()
    assert record["method_family"] == "tsdf"

    mesh = tetrahedron()
    occupancy_only = set(INHERENT_LIMITATIONS) - set(TSDF_INHERENT_LIMITATIONS)
    # The six caveats about the stand-off of a solid that contains its
    # samples, an iso level counted over a node's cells, a six-connected fill
    # deciding an interior, and a padded grid that always closes - none of
    # which happened to a fused signed-distance field - must be excluded here
    # even though the words alone were checked absent at the provenance level
    # already; this checks the FILE, not just the record.
    assert len(occupancy_only) == 6

    block = flattened(provenance_comments(mesh, prov))
    for limitation in TSDF_INHERENT_LIMITATIONS:
        assert flattened([limitation]) in block
    for limitation in occupancy_only:
        assert flattened([limitation]) not in block

    obj_path = tmp_path / "tsdf.obj"
    ply_path = tmp_path / "tsdf.ply"
    obj_export = write_obj(obj_path, mesh, provenance=prov)
    ply_export = write_ply_mesh(ply_path, mesh, provenance=prov)
    for export, path, parser in (
        (obj_export, obj_path, ParsedObj),
        (ply_export, ply_path, ParsedPlyMesh),
    ):
        assert set(TSDF_INHERENT_LIMITATIONS) <= set(export.limitations)
        assert not (occupancy_only & set(export.limitations))
        file_block = flattened(parser(path.read_bytes()).comments)
        for limitation in TSDF_INHERENT_LIMITATIONS:
            assert flattened([limitation]) in file_block
        for limitation in occupancy_only:
            assert flattened([limitation]) not in file_block


def test_each_family_states_the_level_it_actually_has_and_never_the_other_ones(tmp_path):
    """An iso level is occupancy's parameter, and a TSDF file must not print it.

    Two claims, and the second is only safe because of the first. The occupancy
    line is byte for byte what it has always been - including ``iso level: not
    stated`` when a caller left the parameter out - because those bytes are in
    every occupancy file this repository has written and a reader diffing an old
    export against a new one is entitled to see no difference.

    A TSDF surface has no such quantity: nothing counted how many of a node's
    eight touching cells were solid, and the level it does have is the zero of a
    signed distance, fixed by arithmetic rather than chosen. So the LABEL is
    absent from the comment block of both file formats and from the record, and
    the level that exists is stated in its place. The label is searched for with
    its colon, because the words themselves still appear inside the caveat that
    says no iso level was counted.
    """
    occupancy = provenance()
    assert occupancy.summary == "voxel occupancy + marching cubes; voxel size: 0.1; iso level: 3.5"
    assert provenance(iso_level=None).summary == (
        "voxel occupancy + marching cubes; voxel size: 0.1; iso level: not stated"
    )
    assert provenance(voxel_size=None, iso_level=None).summary == (
        "voxel occupancy + marching cubes; voxel size: not stated; iso level: not stated"
    )
    assert occupancy.as_dict()["iso_level"] == DEFAULT_ISO_LEVEL
    assert "level_set" not in occupancy.as_dict()

    fused = provenance(method_family=MeshMethod.TSDF, method="tsdf fusion + marching cubes")
    assert (
        fused.summary
        == "tsdf fusion + marching cubes; voxel size: 0.1; level set: zero (the sign change)"
    )
    assert "iso level:" not in fused.summary
    record = fused.as_dict()
    assert "iso_level" not in record, "the record states an occupancy parameter"
    assert record["level_set"] == 0.0

    # The FILES, not just the record: the summary reaches them through
    # provenance_comments, and a reader opening the mesh sees that line and not
    # this object.
    mesh = tetrahedron()
    obj_path = tmp_path / "level.obj"
    ply_path = tmp_path / "level.ply"
    write_obj(obj_path, mesh, provenance=fused)
    write_ply_mesh(ply_path, mesh, provenance=fused)
    for path, parser in ((obj_path, ParsedObj), (ply_path, ParsedPlyMesh)):
        block = flattened(parser(path.read_bytes()).comments)
        assert flattened(["iso level:"]) not in block, f"{path.suffix} states an iso level"
        assert flattened(["level set: zero"]) in block
    # And the occupancy files still carry theirs, so the absence above is this
    # family's and not a line that stopped being written for everybody.
    legacy = tmp_path / "legacy.obj"
    write_obj(legacy, mesh, provenance=occupancy)
    assert flattened(["iso level: 3.5"]) in flattened(ParsedObj(legacy.read_bytes()).comments)


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


def test_provenance_text_cannot_forge_a_line_of_either_format(tmp_path):
    """The guard :func:`scanner3d.mesh._one_line` exists for is exercised, not assumed.

    Both formats are line structured and both take strings straight from a
    caller. A units label carrying a newline and ``end_header`` ends a PLY header
    early and turns the caveats that follow it into vertex data; one carrying a
    newline and ``f 1 2 3`` adds a face to an OBJ that no reconstruction
    produced. Nothing in the suite covered this: the guard could be reduced to
    ``printable = text`` and every other test still passed, which is exactly the
    state a security-relevant guard must not be left in.

    So the payloads are checked, not the helper: every line of the written file
    is either one this module's own encoder emits or a comment carrying its
    format's prefix, and the counts the header declares still match the body.
    """
    # Non-ASCII is in here deliberately and is the half that only this guard
    # stops: newlines alone are already flattened by ``textwrap.wrap``, but a
    # single non-ASCII character reaches ``encode("ascii")`` in the header
    # builder and raises ``UnicodeEncodeError`` - a crash from outside this
    # module's refusal hierarchy, on a string a caller is free to supply.
    poison = "\nend_header\n0 0 0\nf 1 2 3\nelement vertex 99\ncafé 90° 中\t\r\x00"
    prov = MeshProvenance(
        cloud=CloudProvenance(
            units=f"unvalidated{poison}",
            convention=CAMERA_OPTICAL_FRAME,
            calibration=CalibrationProvenance.none_applied(f"none{poison}"),
        ),
        method=f"voxel occupancy{poison}",
        voxel_size=0.1,
        iso_level=DEFAULT_ISO_LEVEL,
        limitations=(f"a caveat{poison}",),
    )
    mesh = tetrahedron()

    obj = write_obj(tmp_path / "poisoned.obj", mesh, provenance=prov)
    body = [line for line in obj.payload.decode("ascii").splitlines() if not line.startswith("# ")]
    assert [line[0] for line in body] == ["v"] * 4 + ["f"] * 4, body
    parsed_obj = ParsedObj(obj.payload)
    assert len(parsed_obj.vertices) == 4
    assert len(parsed_obj.faces) == 4, "a forged 'f' line would make a fifth"

    ply = write_ply_mesh(tmp_path / "poisoned.ply", mesh, provenance=prov)
    lines = ply.payload.decode("ascii").splitlines()
    # A directive is a whole line, so that is what must not be forged. The words
    # may appear inside a comment - flattened onto one, which is the point.
    assert lines.count("end_header") == 1, "a provenance string forged a header terminator"
    assert [line for line in lines if line.startswith("element ")] == [
        "element vertex 4",
        "element face 4",
    ], "a provenance string forged an element declaration"
    end = lines.index("end_header")
    assert lines[0] == "ply"
    assert lines[1].startswith("format ")
    for line in lines[2 : end - 6]:
        assert line.startswith("comment "), f"a provenance string became its own line: {line!r}"
    rows = [line.split() for line in lines[end + 1 :]]
    assert [len(tokens) for tokens in rows] == [3] * 4 + [4] * 4, "a forged line reached the body"
    assert all(tokens[0] == "3" for tokens in rows[4:]), rows[4:]
    parsed_ply = ParsedPlyMesh(ply.payload)
    assert len(parsed_ply.vertices) == 4
    assert len(parsed_ply.faces) == 4
    # The caveats survive the flattening; they are neutered, not dropped.
    assert any("a caveat" in line for line in ply.limitations)


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
    # Sized from the ceiling itself, so raising the ceiling keeps this a test
    # of the refusal rather than of one particular number.
    count = MAX_HEADER_BYTES // 200 + 8
    wordy = provenance(limitations=tuple(f"caveat {index} " + "x" * 200 for index in range(count)))
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


# --- reconstructing while the capture runs ------------------------------------
#
# The accumulator's whole claim is that folding frames in one at a time reaches
# the same surface as reading them all at once. That is asserted directly rather
# than approximately: the same points, the same voxel size, the same vertices
# and the same faces. Everything else here is about what it costs and what it
# gives up.


def shell(count: int, *, inner: float, outer: float, seed: int) -> list[Point]:
    """Points in a spherical shell: a cloud with a real surface, and repeatable."""
    import random

    rng = random.Random(seed)
    points: list[Point] = []
    while len(points) < count:
        candidate = (
            rng.uniform(-outer, outer),
            rng.uniform(-outer, outer),
            rng.uniform(-outer, outer),
        )
        if inner <= norm(candidate) <= outer:
            points.append(candidate)
    return points


def sphere_surface(count: int, radius: float) -> list[Point]:
    """A dense, evenly spread sphere SURFACE - a Fibonacci lattice, no randomness.

    Different from :func:`shell` on purpose. A shell is a thick volume sampled
    sparsely, where nearly every point owns a cell; this is a two-dimensional
    surface sampled densely, where several points share one. That is what a
    depth frame of an object actually is, and it is the only shape on which
    "re-observing costs nothing" is a meaningful claim.
    """
    points: list[Point] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden * index
        points.append(
            (radius * ring * math.cos(angle), radius * y, radius * ring * math.sin(angle))
        )
    return points


def test_folding_frames_in_one_at_a_time_reaches_the_same_surface_as_reading_them_at_once():
    """The claim the whole class exists to make, asserted exactly."""
    points = shell(2000, inner=25.0, outer=30.0, seed=11)
    accumulator = SurfaceAccumulator(voxel_size=2.0)
    for start in range(0, len(points), 137):
        accumulator.add(points[start : start + 137])

    incremental = accumulator.surface()
    at_once = reconstruct(points, voxel_size=2.0)

    assert incremental.vertices == at_once.vertices
    assert incremental.faces == at_once.faces
    assert accumulator.added_points == len(points)


def test_the_order_frames_arrive_in_does_not_change_the_surface():
    """Occupancy is a set union, so it cannot. Asserted because it is the reason
    a frame can be folded in and then forgotten."""
    points = shell(900, inner=20.0, outer=24.0, seed=3)
    forwards = SurfaceAccumulator(voxel_size=1.5)
    forwards.add(points)
    backwards = SurfaceAccumulator(voxel_size=1.5)
    backwards.add(list(reversed(points)))

    assert forwards.surface().faces == backwards.surface().faces


def test_adding_the_same_frame_twice_changes_nothing_but_the_point_count():
    points = shell(500, inner=15.0, outer=18.0, seed=5)
    accumulator = SurfaceAccumulator(voxel_size=1.5)
    accumulator.add(points)
    cells = accumulator.occupied_cells
    first = accumulator.surface().faces

    accumulator.add(points)

    assert accumulator.occupied_cells == cells
    assert accumulator.surface().faces == first
    assert accumulator.added_points == 2 * len(points)


def test_an_empty_frame_is_a_no_op_and_not_a_refusal():
    """The difference from :func:`reconstruct`, and the reason for it: a frame
    that caught nothing is a normal event during a capture."""
    accumulator = SurfaceAccumulator(voxel_size=1.0)

    assert accumulator.add([]) == 0
    assert accumulator.is_empty
    assert accumulator.grid_shape == (0, 0, 0)
    assert accumulator.grid_cells == 0


def test_an_accumulator_with_nothing_in_it_refuses_to_produce_a_surface():
    with pytest.raises(NoPointsOffered):
        SurfaceAccumulator(voxel_size=1.0).surface()


def test_add_reports_how_many_cells_are_new():
    accumulator = SurfaceAccumulator(voxel_size=1.0)

    first = accumulator.add([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    again = accumulator.add([(0.0, 0.0, 0.0), (9.0, 0.0, 0.0)])

    assert first == 2
    assert again == 1
    assert accumulator.occupied_cells == 3


def test_coarsening_halves_each_axis_of_the_grid_and_doubles_the_cell_edge():
    points = shell(1500, inner=20.0, outer=26.0, seed=17)
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add(points)
    before = accumulator.grid_shape

    assert accumulator.coarsen() == 2.0

    after = accumulator.grid_shape
    assert accumulator.voxel_size == 2.0
    assert accumulator.coarsenings == 1
    for axis in range(3):
        # Halved, give or take the padding, which is a constant on each side and
        # so a larger share of a smaller grid.
        assert after[axis] < before[axis]
        assert after[axis] >= before[axis] // 2 - 2 * PAD_CELLS


def test_a_coarsened_cell_is_where_its_own_centre_lands_on_the_new_grid():
    """The exact arithmetic, on indices whose answer is written down by hand.

    A cell's centre is ``index * voxel_size``, so at twice the edge its index is
    ``floor(index / 2 + 0.5)``. The negative cases are the ones worth pinning:
    Python's floor division is what makes ``(index + 1) // 2`` agree with the
    floor of a negative half.
    """
    indices = (-4, -3, -2, -1, 0, 1, 2, 3)
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add([(float(index), 0.0, 0.0) for index in indices])

    accumulator.coarsen()

    lanes = sorted({cell[0] for cell in accumulator._occupied})
    assert lanes == sorted({math.floor(index / 2 + 0.5) for index in indices})
    # Written out, because the negative half is where an implementation that
    # truncated instead of flooring would differ and still look plausible.
    assert lanes == [-2, -1, 0, 1, 2]


def test_coarsening_is_not_the_grid_the_points_would_have_made_at_the_new_edge():
    """Stated in the docstring as a limitation, so it is asserted as one.

    Coarsening re-bins from the old cells' centres, not from where the points
    actually were - which is information that was discarded when they were
    added. The two grids are close and they are not the same, and a caller told
    otherwise would treat a preview as if it were a choice of resolution.
    """
    points = shell(1200, inner=18.0, outer=23.0, seed=23)
    coarsened = SurfaceAccumulator(voxel_size=1.0)
    coarsened.add(points)
    coarsened.coarsen()
    fresh = SurfaceAccumulator(voxel_size=2.0)
    fresh.add(points)

    assert coarsened.voxel_size == fresh.voxel_size
    assert coarsened._occupied != fresh._occupied


def test_coarsen_to_fit_reaches_the_budget_and_says_how_many_steps_it_took():
    points = shell(2000, inner=25.0, outer=30.0, seed=29)
    accumulator = SurfaceAccumulator(voxel_size=0.4)
    accumulator.add(points)
    assert accumulator.grid_cells > 200_000

    steps = accumulator.coarsen_to_fit(50_000)

    assert steps >= 1
    assert accumulator.grid_cells <= 50_000
    assert accumulator.coarsenings == steps


def test_coarsen_to_fit_does_nothing_when_the_grid_already_fits():
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add(shell(300, inner=8.0, outer=10.0, seed=31))
    edge = accumulator.voxel_size

    assert accumulator.coarsen_to_fit(MAX_GRID_CELLS) == 0
    assert accumulator.voxel_size == edge


def test_coarsen_to_fit_terminates_on_a_budget_no_grid_can_meet():
    """It stops when doubling stops helping, not when the budget is met.

    The trap this guards is specific and it is not "the grid gets small enough".
    Re-binning maps an index to ``(index + 1) // 2``, whose FIXED POINTS are 0
    and 1, so occupied cells can settle on a grid a few cells wide that no
    further doubling shrinks - a grid that is still bigger than a small enough
    budget. A loop watching only the budget spins there for ever.
    """
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add(shell(400, inner=10.0, outer=14.0, seed=37))

    steps = accumulator.coarsen_to_fit(1)

    assert steps > 0
    assert accumulator.grid_cells > 1, "the impossible budget was not met, and that is correct"
    # And it really is the fixed point: another doubling changes nothing.
    settled = accumulator.grid_shape
    accumulator.coarsen()
    assert accumulator.grid_shape == settled


def test_the_fixed_point_of_re_binning_is_reached_and_recognised():
    """The two indices that survive halving, made explicit on a two-cell grid.

    ``(0 + 1) // 2`` is 0 and ``(1 + 1) // 2`` is 1, so this grid is six cells
    wide for ever. Written down here because it is the whole reason
    :meth:`coarsen_to_fit` watches progress rather than the budget.
    """
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    shape = accumulator.grid_shape

    for _ in range(5):
        accumulator.coarsen()
        assert accumulator.grid_shape == shape

    assert accumulator.coarsen_to_fit(1) == 1, "one doubling to learn it, and no more"


def test_a_budget_that_cannot_be_met_surfaces_as_a_refusal_and_not_as_a_hang():
    """Where an impossible budget is actually reported: the march refuses it."""
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    accumulator.coarsen_to_fit(1)

    with pytest.raises(GridTooLarge):
        accumulator.surface(max_grid_cells=1)


def test_a_surface_past_the_budget_is_refused_rather_than_silently_coarsened():
    """The decision belongs to the caller: a class that halved its own
    resolution would change the picture for a reason nobody saw."""
    accumulator = SurfaceAccumulator(voxel_size=0.5)
    accumulator.add(shell(1500, inner=25.0, outer=30.0, seed=41))

    with pytest.raises(GridTooLarge):
        accumulator.surface(max_grid_cells=1000)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"voxel_size": 0.0}, VoxelSizeUnusable),
        ({"voxel_size": -1.0}, VoxelSizeUnusable),
        ({"voxel_size": float("nan")}, NonFiniteValue),
        ({"voxel_size": 1.0, "max_points": 0}, MeshNotAllowed),
        ({"voxel_size": 1.0, "max_points": MAX_INPUT_POINTS + 1}, MeshNotAllowed),
    ],
)
def test_an_unusable_accumulator_is_refused_at_construction(kwargs, expected):
    with pytest.raises(expected):
        SurfaceAccumulator(**kwargs)


def test_an_unusable_iso_level_is_refused_before_the_grid_is_allocated():
    accumulator = SurfaceAccumulator(voxel_size=1.0)
    accumulator.add(shell(200, inner=8.0, outer=11.0, seed=43))

    with pytest.raises(IsoLevelUnusable):
        accumulator.surface(iso_level=0.0)


def test_a_frame_past_the_point_cap_is_refused_and_what_preceded_it_is_kept():
    """Documented as deliberate: the union does not need an all-or-nothing, and
    buffering a whole frame to provide one would cost what this class saves."""
    accumulator = SurfaceAccumulator(voxel_size=1.0, max_points=4)

    with pytest.raises(PointCountExceeded):
        accumulator.add([(float(index), 0.0, 0.0) for index in range(10)])

    assert accumulator.occupied_cells == 4


def test_a_non_finite_point_is_refused_by_the_accumulator_too():
    accumulator = SurfaceAccumulator(voxel_size=1.0)

    with pytest.raises(NonFiniteValue):
        accumulator.add([(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)])


def test_re_observing_one_surface_barely_grows_the_grid():
    """The property that makes this affordable: cost is bounded by the OBJECT.

    Ten views of one surface, each nudged by less than a cell the way overlapping
    frames of a sweep are. The points go in ten times over and the occupied set
    ends up about the size one view produced, which is what makes the march's
    cost flat over a long sweep instead of proportional to its length.
    """
    import random

    points = sphere_surface(4000, 20.0)
    accumulator = SurfaceAccumulator(voxel_size=3.0)
    accumulator.add(points)
    one_view = accumulator.occupied_cells
    rng = random.Random(53)
    for _ in range(9):
        dx, dy, dz = (rng.uniform(-0.3, 0.3) for _ in range(3))
        accumulator.add([(x + dx, y + dy, z + dz) for x, y, z in points])

    assert accumulator.added_points == 10 * len(points)
    assert one_view > 0
    # Ten times the points, and about a fifth more cells.
    assert accumulator.occupied_cells < 1.5 * one_view
    assert accumulator.surface().is_watertight
