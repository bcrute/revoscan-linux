"""Tests for :mod:`scanner3d.texture`.

What is at stake
-----------------
A mesh vertex from :mod:`scanner3d.mesh` is a point marching cubes invented on
a grid edge - never a point anyone actually observed - so colouring one is
always an inference from NEARBY real samples, never a lookup. Two ways that
inference can go quietly wrong, and what catches each:

* A vertex with nothing genuinely nearby gets painted some default colour
  instead of being reported as uncoloured. Indistinguishable, once rendered,
  from a real dark surface - test 1 asserts the exact uncoloured count and
  indices, not merely that SOME vertices came back coloured.
* A single sample from the wrong surface - the classic depth-discontinuity
  case, where a vertex sits right at the boundary between what it is
  actually on and something else nearby - drags the assigned colour toward
  itself. Test 3 is built specifically to make that outlier the CLOSEST
  candidate, which is the case a naive "nearest wins" or a distance-weighted
  average would get wrong and a per-channel median does not.

Test 2 is a file-format claim, not a geometry one: a PLY written without
colours must be bytes nobody could tell apart from a PLY this module never
touched, and a PLY written with them must declare exactly the three uchar
properties PLY's own convention uses, with the right values in the right
order.
"""

from __future__ import annotations

import random
import statistics
import struct
from collections.abc import Sequence

import pytest

from scanner3d.compute import PYTHON_BACKEND, SCIPY_BACKEND, _probe_scipy, select_backend
from scanner3d.errors import DoctorStatus
from scanner3d.geometry import NonFiniteValue, Point, distance
from scanner3d.mesh import (
    Mesh,
    MeshProvenance,
    VertexColourChannelOutOfRange,
    VertexColourCountMismatch,
    encode_ply_mesh,
)
from scanner3d.pointcloud import (
    CAMERA_OPTICAL_FRAME,
    CalibrationProvenance,
    CloudProvenance,
    PlyFormat,
)
from scanner3d.texture import (
    SCHEMA_VERSION,
    ColourChannelOutOfRange,
    NeighbourCountUnusable,
    NoColouredPointsOffered,
    NoVerticesOffered,
    RadiusUnusable,
    VertexColours,
    colour_vertices,
)

SCIPY_AVAILABLE, _SCIPY_REASON = _probe_scipy()
needs_scipy = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason=f"scipy not installed: {_SCIPY_REASON}"
)

RgbTriple = tuple[int, int, int]


def _mesh_provenance() -> MeshProvenance:
    """A provenance a test can pass without pretending a real capture happened."""
    cloud = CloudProvenance(
        units="test units; this label measures nothing and is not a length",
        convention=CAMERA_OPTICAL_FRAME,
        calibration=CalibrationProvenance.none_applied("tests/test_texture.py: synthetic geometry"),
    )
    return MeshProvenance(
        cloud=cloud, method="hand-built for a test", voxel_size=None, iso_level=None
    )


def _tetrahedron() -> Mesh:
    """A minimal valid mesh: four vertices, four triangles, nothing degenerate.

    Small on purpose - the PLY-format tests below care about the exact bytes
    written, and a tetrahedron is the smallest solid that has one.
    """
    vertices: tuple[Point, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    return Mesh(vertices=vertices, faces=faces)


# --------------------------------------------------------------------------
# 1. Vertices among coloured points get those colours; a far vertex is
#    reported uncoloured, not painted a default.
# --------------------------------------------------------------------------


def test_a_nearby_vertex_is_coloured_and_a_far_one_is_reported_uncoloured():
    consistent: RgbTriple = (12, 34, 56)
    vertices: list[Point] = [
        (0.0, 0.0, 0.0),  # sits right among the coloured points below
        (500.0, 500.0, 500.0),  # nothing within any sane radius
    ]
    coloured_points: list[tuple[Point, RgbTriple]] = [
        ((0.01, 0.0, 0.0), consistent),
        ((-0.01, 0.0, 0.0), consistent),
        ((0.0, 0.01, 0.0), consistent),
    ]
    result = colour_vertices(vertices, coloured_points, radius=0.5)
    assert isinstance(result, VertexColours)
    assert result.colours[0] == consistent
    assert result.colours[1] is None
    # The report, not merely a spot check: the exact count and the exact
    # index of the vertex that got no colour.
    assert result.uncoloured_count == 1
    assert result.coloured_count == 1
    assert result.uncoloured_indices == (1,)
    assert result.vertex_count == 2
    record = result.as_dict()
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["uncoloured_indices"] == [1]
    assert record["colours"] == [list(consistent), None]


def test_every_vertex_can_be_reported_uncoloured_at_once():
    """Not a special case of "some vertices" - zero coloured vertices is a valid,
    fully-reported outcome when the radius genuinely excludes everything.
    """
    vertices: list[Point] = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)]
    coloured_points: list[tuple[Point, RgbTriple]] = [((100.0, 100.0, 100.0), (1, 2, 3))]
    result = colour_vertices(vertices, coloured_points, radius=0.1)
    assert result.colours == (None, None)
    assert result.uncoloured_count == 2
    assert result.coloured_count == 0
    assert result.uncoloured_indices == (0, 1)


# --------------------------------------------------------------------------
# 2. PLY round trip: header and bytes, with and without colour.
# --------------------------------------------------------------------------


def _read_ply_header(raw: bytes) -> tuple[list[str], bytes]:
    """The header lines (without a trailing newline) and the raw body bytes."""
    lines: list[str] = []
    offset = 0
    while True:
        newline = raw.index(b"\n", offset)
        line = raw[offset:newline].decode("ascii")
        offset = newline + 1
        lines.append(line)
        if line == "end_header":
            break
    return lines, raw[offset:]


def test_ply_with_colours_declares_the_three_uchar_properties_and_the_values_survive():
    mesh = _tetrahedron()
    colours: tuple[RgbTriple, ...] = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30))
    coloured_mesh = Mesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colours=colours)

    ascii_export = encode_ply_mesh(coloured_mesh, provenance=_mesh_provenance())
    ascii_lines, ascii_body = _read_ply_header(ascii_export.payload)
    assert "property float x" in ascii_lines
    assert "property float y" in ascii_lines
    assert "property float z" in ascii_lines
    assert "property uchar red" in ascii_lines
    assert "property uchar green" in ascii_lines
    assert "property uchar blue" in ascii_lines
    # Declared in the standard PLY order: position, then colour.
    assert ascii_lines.index("property float z") < ascii_lines.index("property uchar red")
    ascii_rows = ascii_body.decode("ascii").splitlines()
    for row, (r, g, b) in zip(ascii_rows[: mesh.vertex_count], colours, strict=True):
        tokens = row.split()
        assert tuple(int(value) for value in tokens[3:6]) == (r, g, b)

    binary_export = encode_ply_mesh(
        coloured_mesh, provenance=_mesh_provenance(), ply_format=PlyFormat.BINARY_LITTLE_ENDIAN
    )
    binary_lines, binary_body = _read_ply_header(binary_export.payload)
    assert "property uchar red" in binary_lines
    for index, (r, g, b) in enumerate(colours):
        _x, _y, _z, red, green, blue = struct.unpack_from("<fffBBB", binary_body, index * 15)
        assert (red, green, blue) == (r, g, b)


def test_ply_without_colours_is_byte_identical_regardless_of_how_the_field_was_left_default():
    """The core claim: attaching this new field to Mesh must not move one byte of
    a colourless export. Checked two ways - by constructing "the same mesh
    both ways" (default-omitted vs. explicit ``None``) and by confirming the
    header carries none of the three colour properties at all.
    """
    mesh = _tetrahedron()
    explicit_none = Mesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colours=None)
    assert mesh.vertex_colours is None  # the default every pre-existing caller relies on

    left_default = encode_ply_mesh(mesh, provenance=_mesh_provenance())
    given_explicitly = encode_ply_mesh(explicit_none, provenance=_mesh_provenance())
    assert left_default.payload == given_explicitly.payload
    assert left_default.header == given_explicitly.header
    assert left_default.sha256 == given_explicitly.sha256

    lines, body = _read_ply_header(left_default.payload)
    vertex_properties = [
        line for line in lines if line.startswith("property") and "vertex" not in line
    ]
    assert vertex_properties == ["property float x", "property float y", "property float z"]
    assert "property uchar red" not in lines
    assert "property uchar green" not in lines
    assert "property uchar blue" not in lines
    # Exactly 12 bytes (fff) per vertex, none of the extra 3 uchar bytes a
    # coloured export would add.
    ascii_rows = body.decode("ascii").splitlines()
    for row in ascii_rows[: mesh.vertex_count]:
        assert len(row.split()) == 3


# --------------------------------------------------------------------------
# 3. The outlier case: one wildly different close point must not win.
# --------------------------------------------------------------------------


def test_a_single_close_outlier_does_not_win_over_several_consistent_points():
    vertex: Point = (0.0, 0.0, 0.0)
    consistent: RgbTriple = (10, 200, 10)
    outlier: RgbTriple = (250, 5, 5)
    coloured_points: list[tuple[Point, RgbTriple]] = [
        ((0.001, 0.0, 0.0), outlier),  # the single closest candidate, and wrong
        ((0.05, 0.0, 0.0), consistent),
        ((-0.05, 0.0, 0.0), consistent),
        ((0.0, 0.05, 0.0), consistent),
        ((0.0, -0.05, 0.0), consistent),
        ((0.0, 0.0, 0.05), consistent),
    ]
    result = colour_vertices([vertex], coloured_points, radius=0.5, neighbours=6)
    assert result.colours[0] == consistent
    assert result.colours[0] != outlier


def test_a_majority_of_outliers_does_win_which_is_the_advertised_break_point():
    """States plainly what the chosen rule does NOT protect against: a median
    has no defence once wrong samples are the majority, and this pins that
    down rather than leaving it as an unstated assumption.
    """
    vertex: Point = (0.0, 0.0, 0.0)
    minority: RgbTriple = (10, 200, 10)
    majority: RgbTriple = (250, 5, 5)
    coloured_points: list[tuple[Point, RgbTriple]] = [
        ((0.01, 0.0, 0.0), minority),
        ((0.02, 0.0, 0.0), minority),
        ((-0.02, 0.0, 0.0), majority),
        ((0.0, 0.02, 0.0), majority),
        ((0.0, -0.02, 0.0), majority),
    ]
    result = colour_vertices([vertex], coloured_points, radius=0.5, neighbours=5)
    assert result.colours[0] == majority


# --------------------------------------------------------------------------
# 4. Backend equivalence: "python" and "scipy" agree exactly.
# --------------------------------------------------------------------------


@needs_scipy
def test_python_and_scipy_backends_produce_the_same_colours():
    rng = random.Random(20260823)
    # Jittered, real-valued coordinates - no exact coincidences - so neither
    # backend's tie-breaking (undocumented, and allowed to differ; see
    # scanner3d.compute's own module docstring) can make this test flaky.
    coloured_points: list[tuple[Point, RgbTriple]] = [
        (
            (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
        for _ in range(60)
    ]
    vertices: list[Point] = [
        (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(15)
    ]
    # Two vertices placed well outside every coloured point's reach, so at
    # least one uncoloured report is a guaranteed outcome and not left to
    # the luck of the random draw above.
    vertices += [(50.0, 50.0, 50.0), (-50.0, -50.0, -50.0)]
    python_result = colour_vertices(
        vertices, coloured_points, radius=0.2, neighbours=5, backend=PYTHON_BACKEND
    )
    scipy_result = colour_vertices(
        vertices, coloured_points, radius=0.2, neighbours=5, backend=SCIPY_BACKEND
    )
    assert python_result.colours == scipy_result.colours
    assert python_result.backend == PYTHON_BACKEND
    assert scipy_result.backend == SCIPY_BACKEND
    # Some vertices should land in both camps in this configuration, or the
    # test is not exercising what it claims to.
    assert any(c is not None for c in python_result.colours)
    assert any(c is None for c in python_result.colours)


# --------------------------------------------------------------------------
# 6. Fast-path equivalence: the current colour_vertices (one batched
#    ComputeBackend.k_nearest_between call) against an independent,
#    deliberately slow per-vertex reference (one ComputeBackend.k_nearest_neighbors
#    call per vertex, on a list built from that vertex plus every coloured
#    point - the shape colour_vertices itself used before it was rewritten
#    to close the six-minutes-per-real-mesh gap; see scanner3d.texture's own
#    "Cost" section). Written fresh here rather than imported, so a bug
#    shared between the two would not cancel itself out of this comparison.
# --------------------------------------------------------------------------


def _reference_combine(candidates: Sequence[RgbTriple]) -> RgbTriple:
    channels = tuple(
        round(statistics.median(candidate[axis] for candidate in candidates)) for axis in range(3)
    )
    return (channels[0], channels[1], channels[2])


def _reference_colour_vertices(
    vertices: Sequence[Point],
    coloured_points: Sequence[tuple[Point, RgbTriple]],
    *,
    radius: float,
    neighbours: int,
    backend: str,
) -> tuple[RgbTriple | None, ...]:
    resolved = select_backend(backend).backend
    colour_points = [point for point, _colour in coloured_points]
    point_colours = [colour for _point, colour in coloured_points]
    results: list[RgbTriple | None] = []
    for vertex in vertices:
        combined: tuple[Point, ...] = (vertex, *colour_points)
        neighbour_rows = resolved.k_nearest_neighbors(combined, neighbours)
        candidates = [
            point_colours[combined_index - 1]
            for combined_index in neighbour_rows[0]
            if distance(vertex, colour_points[combined_index - 1]) < radius
        ]
        results.append(_reference_combine(candidates) if candidates else None)
    return tuple(results)


@pytest.mark.parametrize(
    "backend_name", [PYTHON_BACKEND, pytest.param(SCIPY_BACKEND, marks=needs_scipy)]
)
def test_the_batched_fast_path_matches_the_independent_per_vertex_reference(backend_name: str):
    rng = random.Random(9920260825)
    coloured_points: list[tuple[Point, RgbTriple]] = [
        (
            (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)),
            (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
        for _ in range(120)
    ]
    vertices: list[Point] = [
        (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)) for _ in range(40)
    ]
    # A few vertices placed outside every coloured point's reach, so the
    # "reported None" path is exercised on both sides of the comparison too.
    vertices += [(80.0, 80.0, 80.0), (-80.0, -80.0, -80.0)]
    radius = 0.5
    neighbours = 6

    fast = colour_vertices(
        vertices, coloured_points, radius=radius, neighbours=neighbours, backend=backend_name
    )
    reference = _reference_colour_vertices(
        vertices, coloured_points, radius=radius, neighbours=neighbours, backend=backend_name
    )
    assert fast.colours == reference
    assert any(c is not None for c in fast.colours)
    assert any(c is None for c in fast.colours)


# --------------------------------------------------------------------------
# 5. Guards.
# --------------------------------------------------------------------------


def test_non_positive_radius_is_refused():
    with pytest.raises(RadiusUnusable) as excinfo:
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (1, 2, 3))], radius=0.0)
    assert excinfo.value.status is DoctorStatus.UNSUPPORTED_MODE
    with pytest.raises(RadiusUnusable):
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (1, 2, 3))], radius=-1.0)


def test_non_finite_radius_is_refused():
    # Matches scanner3d.mesh's own convention for this exact shape of check
    # (see _ensure_voxel_size): finiteness is scanner3d.geometry's own
    # question and its own exception, raised unwrapped; only the "usable
    # value in range" check past it is this module's to name.
    with pytest.raises(NonFiniteValue):
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (1, 2, 3))], radius=float("nan"))


def test_neighbour_count_below_one_is_refused():
    with pytest.raises(NeighbourCountUnusable):
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (1, 2, 3))], radius=1.0, neighbours=0)


def test_empty_vertices_is_refused():
    with pytest.raises(NoVerticesOffered):
        colour_vertices([], [((0.0, 0.0, 0.0), (1, 2, 3))], radius=1.0)


def test_empty_coloured_points_is_refused():
    with pytest.raises(NoColouredPointsOffered):
        colour_vertices([(0.0, 0.0, 0.0)], [], radius=1.0)


def test_out_of_range_colour_channel_is_refused():
    with pytest.raises(ColourChannelOutOfRange):
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (256, 0, 0))], radius=1.0)
    with pytest.raises(ColourChannelOutOfRange):
        colour_vertices([(0.0, 0.0, 0.0)], [((0.0, 0.0, 0.0), (0, -1, 0))], radius=1.0)


def test_mesh_refuses_a_vertex_colour_count_that_does_not_match_the_vertex_count():
    mesh = _tetrahedron()
    short = Mesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colours=((1, 2, 3),))
    with pytest.raises(VertexColourCountMismatch):
        encode_ply_mesh(short, provenance=_mesh_provenance())


def test_mesh_refuses_a_vertex_colour_channel_out_of_range():
    mesh = _tetrahedron()
    bad_colours: tuple[RgbTriple, ...] = ((0, 0, 0), (0, 0, 0), (0, 0, 0), (999, 0, 0))
    bad = Mesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colours=bad_colours)
    with pytest.raises(VertexColourChannelOutOfRange):
        encode_ply_mesh(bad, provenance=_mesh_provenance())


def test_vertex_colours_type_carries_what_was_asked_for():
    vertices: list[Point] = [(0.0, 0.0, 0.0)]
    coloured_points: list[tuple[Point, RgbTriple]] = [((0.0, 0.0, 0.0), (7, 8, 9))]
    result = colour_vertices(vertices, coloured_points, radius=1.0, neighbours=3)
    assert result.radius == 1.0
    assert result.neighbours == 3
    assert result.backend == PYTHON_BACKEND
