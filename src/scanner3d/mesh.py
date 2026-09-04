"""Surface reconstruction: a fused point cloud turned into a triangle mesh, and exported.

What this does
--------------
Points go into a uniform voxel grid, the grid is solidified, and marching cubes
walks the cells that touch the boundary of that solid and emits triangles. Three
stages, each with a property worth naming:

1. **Occupancy.** Every point is quantised to ``floor(coordinate / voxel_size)``
   and the resulting integer triple is the only thing kept. The voxel hash *is*
   the spatial index and it is also the downsample: a million points in one
   voxel are one voxel. Nothing here computes a distance between two points, so
   no part of this module is quadratic in the point count.
2. **Solidification.** A point cloud is a surface, not a solid, and the boundary
   of a one-voxel-thick shell is two surfaces - an inner wall and an outer wall
   with the samples between them. So the empty cells are flood-filled six-ways
   from the grid border, and every cell the fill could not reach is called
   interior. A cloud that encloses a volume then reconstructs as one closed
   surface. A cloud with a gap in it leaks, encloses nothing, and reconstructs
   as the two-walled shell - which is a real result about the input, so
   :func:`reconstruct` measures it and records it in
   :attr:`Mesh.limitations` rather than letting it pass as a shape.
3. **Marching cubes**, on the full 256-case table, over the cells that touch the
   solid boundary and no others.

The scalar field
----------------
The field lives at grid *nodes* (cell corners) and its value is an integer: how
many of the eight cells touching that node are solid, 0 to 8. The default
``iso_level`` of 3.5 is a half-integer on purpose. No node value can equal it,
so no intersection point can ever land exactly on a node, so two cut edges can
never produce the same vertex and a zero-area triangle cannot be built out of
one. It also bounds every edge fraction into ``[1/16, 15/16]``, which keeps the
triangles well away from slivers.

The table, and why it is derived rather than transcribed
--------------------------------------------------------
:data:`TRIANGLE_TABLE` has the standard 256 entries in the standard corner and
edge numbering, and it is *computed* at import by :func:`_build_tables` from one
rule, applied to each of the cube's six faces:

    walking a face counter-clockwise as seen from outside the cube, join each
    corner-to-outside crossing to the nearest preceding outside-to-corner
    crossing.

Every rule that decides how a face's crossings pair up must be a function of
that face's four corner signs alone, because the neighbouring cube sees the same
four signs and has to reach the same answer. This one is, and the neighbour
walks the shared face in the opposite direction, which yields the same pairs
with the directions reversed - so the two cubes agree on the segment and on its
winding. That is what makes the surface closed and consistently oriented, and it
is checked in ``tests/test_mesh.py`` over every case, every axis, and all
sixteen ways a neighbouring cell can differ away from the shared face - 12288
face pairs, which is all of them. Transcribing a published table would have been shorter
and would have brought the original algorithm's well-known complementary-case
holes with it, undetected.

On an ambiguous face - one where the two corners above the level are diagonally
opposite - the rule separates them. That is a convention. It is applied
identically everywhere, which is what matters for closure, but it is not a
measurement of anything.

What this does badly
--------------------
* **Thin features.** The surface is the boundary of a voxel set. A feature
  narrower than a voxel is either swallowed or inflated to one voxel; a sheet
  one voxel thick comes back as a closed box around itself, not as a sheet.
* **Open boundaries.** There are none. Marching cubes over a bounded grid whose
  border is empty always closes, so an open scan - a single depth view, a
  hemisphere of samples - is not reconstructed as an open patch. It is
  reconstructed as a closed shell wrapped around the visible samples, with a
  back wall that was never observed. :attr:`Mesh.limitations` says so; the
  geometry cannot.
* **Resolution, and a bias with a direction.** A cell is solid when any sample
  falls anywhere in it, so the solid contains the samples and the surface stands
  *outside* them - by up to about a voxel, never through them. A mesh here reads
  larger than its cloud, and a coarse grid reads larger still. Nothing is
  smoothed or refitted afterwards, so it is also faceted at the grid's scale.
* **Sparse samples.** The default level asks for four of a node's eight cells to
  be solid. A lone cell has at most one, so an isolated sample vanishes and so
  does a filament one cell wide. That rejects speckle and loses thin real
  structure, and the algorithm cannot tell which it did. ``iso_level`` is the
  knob: lower it towards 0.5 to keep every sample, at the cost of keeping every
  outlier and standing off further.
* **Cost.** The flood fill and the node counts are linear in the *cells* of the
  grid, not in the points, which is why :data:`MAX_GRID_CELLS` exists and why it
  is a time bound as much as a memory bound.

What this does not establish
----------------------------
Nothing dimensional. The coordinates come in unvalidated - see
:mod:`scanner3d.pointcloud` for why the depth scale's physical unit is unknown -
and a surface fitted to unvalidated coordinates has unvalidated dimensions. No
extent, radius, thickness or volume read off a mesh from this module is a
measured length, and every artifact it writes carries that in its own comments.
"""

from __future__ import annotations

import contextlib
import enum
import errno as errno_mod
import math
import os
import struct
import textwrap
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, ensure_finite
from scanner3d.pointcloud import (
    MAX_POINTS as MAX_INPUT_POINTS,
)
from scanner3d.pointcloud import (
    CloudProvenance,
    PlyFormat,
    sha256_of,
)

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

# --- bounds, none of them widened at run time ---------------------------------

#: The most grid cells one reconstruction may occupy: 128^3. The flood fill and
#: the node counts each touch every cell once in pure Python, so this is a
#: wall-clock ceiling as much as a memory one - a grid at this cap takes tens of
#: seconds. A caller may lower it and may never raise it, and it is checked
#: before a single byte of grid is allocated.
MAX_GRID_CELLS: Final = 128 * 128 * 128

#: Empty cells on every side of the sampled extent. Two, because the fill needs
#: an unoccupied border to start from and the node counts need a ring that
#: reaches zero; with both, the reconstructed surface cannot touch the edge of
#: the grid and be clipped by it.
PAD_CELLS: Final = 2

#: How many of a node's eight touching cells must be solid for the node to count
#: as inside. A half-integer: see the module docstring for why that is the whole
#: point of the default.
DEFAULT_ISO_LEVEL: Final = 3.5

#: The number of cells touching one grid node, and so the top of the field's
#: range. The field is an integer count in ``[0, CELLS_PER_NODE]``.
CELLS_PER_NODE: Final = 8

#: How far an intersection may be pushed off a node when a caller picks an
#: integer ``iso_level`` and a node lands exactly on it. Only that case can
#: reach the clamp; the default level cannot.
MIN_EDGE_FRACTION: Final = 1e-6

#: The most vertices and faces one exported mesh may carry. A grid at
#: :data:`MAX_GRID_CELLS` cannot produce more than these, so they bind only a
#: mesh assembled by hand.
MAX_VERTICES: Final = 4 * 1024 * 1024
MAX_FACES: Final = 8 * 1024 * 1024

#: The most bytes one exported mesh may occupy, matching
#: :data:`scanner3d.pointcloud.MAX_OUTPUT_BYTES` for the same reason: a cap that
#: binds the ASCII encodings, which are the fat ones.
MAX_OUTPUT_BYTES: Final = 128 * 1024 * 1024

#: The comment block is the only place either format can state its provenance,
#: so it is generous - and still a ceiling. A header past it is refused, never
#: truncated, because half a stated caveat reads like a whole one.
#:
#: 64 KiB rather than the 16 KiB it was: on 2026-09-04 the first export of a
#: real, coherent turntable registration was refused here with a 17086-byte
#: mesh header - the fusion and extraction evidence plus five registration
#: limitations, every one of them true. The bound is against runaway
#: provenance, not against complete prose; :mod:`scanner3d.pointcloud` was
#: raised the same day for the same reason.
MAX_HEADER_BYTES: Final = 64 * 1024

#: Where comment text is wrapped, leaving room for a ``comment `` or ``# ``
#: prefix inside 79 columns.
COMMENT_WRAP_WIDTH: Final = 70

PLY_MAGIC: Final = "ply"
PLY_VERSION: Final = "1.0"

#: One vertex in the binary PLY encoding, and one face: a ``uchar`` count
#: followed by three little-endian ``int`` indices, which is what
#: ``property list uchar int vertex_indices`` declares.
VERTEX_STRUCT: Final = struct.Struct("<fff")
FACE_STRUCT: Final = struct.Struct("<Biii")

#: One vertex in the binary PLY encoding when :attr:`Mesh.vertex_colours` is
#: present: the same three little-endian floats, followed by three
#: unsigned bytes - what ``property uchar red`` / ``green`` / ``blue`` declare.
#: A second struct rather than packing position and colour separately, so a
#: coloured vertex is one call, one contiguous write, matching how
#: :data:`VERTEX_STRUCT` is used for an uncoloured one.
VERTEX_COLOUR_STRUCT: Final = struct.Struct("<fffBBB")

#: One binary32, used to round a coordinate to the width the PLY carries.
_BINARY32: Final = struct.Struct("<f")

#: Caveats every mesh this repository produces inherits, whatever the caller
#: says. Prepended by :attr:`MeshProvenance.all_limitations`, so no call site can
#: drop one by passing a shorter list.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "this surface was fitted to coordinates that no physical measurement has validated, so "
    "its dimensions are unvalidated with them; no extent, radius or thickness read off this "
    "mesh is a measured length, in millimetres or in anything else",
    "the surface is the boundary of a voxel occupancy grid and is resolution-limited by it: "
    "no feature narrower than about one voxel survives, and every vertex sits within about "
    "one voxel of the samples it was fitted to",
    "a cell counts as solid when any sample falls anywhere inside it, so the solid contains "
    "the samples rather than passing through them and the surface stands off them outward by "
    "up to about one voxel; a mesh from this module reads larger than the cloud it came from",
    "a sample alone in its cell produces no surface at all: the level asks for four of a "
    "node's eight cells to be solid, so an isolated cell or a one-cell filament is dropped "
    "in silence. That is outlier rejection when the sample was noise and lost data when it "
    "was not, and nothing here can tell those two apart",
    "occupancy carries no surface direction: the samples were treated as a solid whose "
    "interior is whatever a six-connected fill from the grid border could not reach, so a "
    "cloud with a gap in it encloses nothing and comes back as a shell wrapped around the "
    "samples rather than as the boundary of an object",
    "marching cubes over a bounded grid always closes, so an open scan is reconstructed with "
    "a back wall that was never observed; a closed mesh here is not evidence that the object "
    "was seen from every side",
    "an ambiguous cube face is resolved by a fixed convention - the corners above the level "
    "are always separated - so where the sampled surface is genuinely ambiguous at this "
    "resolution the connectivity is a convention and not a measurement",
    "no normals, colours, texture coordinates or materials are exported; nothing in this "
    "repository has established any of them",
)

#: Caveats a mesh built by fusing a truncated signed-distance field inherits.
#:
#: A different construction has different inherent defects, and attaching one
#: method's to the other's output is not extra caution - it is a false statement
#: about what is in the file. Six of the eight caveats above are about a voxel
#: OCCUPANCY grid specifically: the stand-off of a solid that contains its
#: samples, an iso level counted over a node's eight cells, a six-connected
#: flood fill deciding an interior, and a padded grid that always closes. None
#: of those happened to a level set of a fused field, so none of them is
#: repeated here. What IS shared is repeated verbatim - the same string object,
#: so a reader diffing the two files sees one sentence and not two spellings of
#: it, and so no edit can move one without the other.
#:
#: Fixed, and not the caller's to compose: :class:`MeshProvenance` selects a
#: family by name and can add to what the family already states, never replace
#: it.
TSDF_INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    INHERENT_LIMITATIONS[0],
    "the surface is the zero level set of a fused truncated signed-distance field sampled at "
    "voxel centres, and is resolution-limited by it: no feature narrower than about one voxel "
    "survives, and every vertex lies on a straight line between two neighbouring voxel centres "
    "rather than on a measured sample",
    "this is NOT an occupancy reconstruction, and the caveats of one are deliberately absent "
    "rather than forgotten: no cell was called solid because a sample landed in it, so this "
    "surface does not inherit occupancy's deterministic cell-containment stand-off - the "
    "outward offset of a solid built to CONTAIN every cell a sample landed in. That is not a "
    "claim that this surface is unbiased: a fused level set has bias of its own, from the "
    "truncation, from averaging observations that disagree, and from being sampled at voxel "
    "centres, and this repository has not measured it. What is claimed is only that "
    "occupancy's stand-off is not among them. Likewise no iso level was counted over a node's "
    "cells; no six-connected fill from a grid border decided an interior; and no padding was "
    "added to make the grid close, so a closed mesh here is closed because the evidence "
    "surrounded the object and an open one is where the evidence stopped",
    "where a cube face is ambiguous - two corners on one side of the level and their diagonal "
    "neighbours on the other - the connectivity is decided by a fixed convention in the shared "
    "marching-cubes tables rather than measured, so an ambiguous face at this resolution is "
    "resolved consistently and not necessarily correctly",
    INHERENT_LIMITATIONS[-1],
)


class MeshMethod(enum.StrEnum):
    """Which construction produced a surface, and so which caveats it inherits.

    Named rather than inferred from :attr:`MeshProvenance.method`, whose text is
    prose a caller writes: a family that could be guessed from a sentence would
    be guessed wrong the first time somebody reworded one.
    """

    #: :func:`reconstruct` - a voxel occupancy grid, flood-filled and marched.
    OCCUPANCY = "occupancy"

    #: :func:`scanner3d.tsdf_surface.extract_surface` - the zero level set of a
    #: field fused by :mod:`scanner3d.tsdf`.
    TSDF = "tsdf"


#: The caveats each family inherits. Read through
#: :attr:`MeshProvenance.all_limitations`; a family with no entry here would be
#: a family whose meshes carry no inherent caveats at all, which is why this is
#: exhaustive over :class:`MeshMethod` and checked by the suite.
METHOD_LIMITATIONS: Final[Mapping[MeshMethod, tuple[str, ...]]] = MappingProxyType(
    {
        MeshMethod.OCCUPANCY: INHERENT_LIMITATIONS,
        MeshMethod.TSDF: TSDF_INHERENT_LIMITATIONS,
    }
)


# --- refusals -----------------------------------------------------------------


class MeshNotAllowed(ScannerError):
    """A reconstruction or an export was requested that this module refuses.

    Always raised before a grid is allocated and before a path is opened, so a
    refused mesh costs no memory it did not already have and leaves no file -
    not even an empty one - behind it.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class VoxelSizeUnusable(MeshNotAllowed):
    """The voxel size cannot define a grid: it is zero, negative or not finite."""


class IsoLevelUnusable(MeshNotAllowed):
    """The iso level is outside ``(0, 8)``, where no surface can exist.

    The field counts solid cells around a node, so a level at or below 0 puts
    every node inside and a level at or above 8 puts every node outside. Either
    one returns an empty mesh from a call that looks like it asked for a
    surface, which is worse than a refusal.
    """


class GridTooLarge(MeshNotAllowed):
    """The grid the extent and voxel size imply is past the cell cap.

    Raised from the cell count alone, before any array exists, so the cap is a
    refusal rather than an allocation followed by regret.
    """


class PointCountExceeded(MeshNotAllowed):
    """More points were offered than the cap, which is never widened at run time.

    Raised on the first point past the cap, so an unbounded producer is stopped
    rather than drained.
    """


class NoPointsOffered(MeshNotAllowed):
    """No points at all were offered, so there is nothing to reconstruct.

    Refused rather than answered with an empty mesh: an empty mesh returned
    from a call that was given nothing reads, downstream, exactly like a
    reconstruction that failed to find a surface in real data.
    """


class MeshCoordinateNotEncodable(MeshNotAllowed):
    """A vertex coordinate is finite but outside the binary32 range a PLY carries.

    NaN and the infinities are refused before this, by
    :func:`scanner3d.geometry.ensure_finite`, which already owns that question
    for the whole geometry stack. This class covers only what the file format
    adds: a float64 as ordinary as ``1e39`` is finite everywhere else and an
    infinity once written as a PLY ``float``.
    """


class MeshIndexOutOfRange(MeshNotAllowed):
    """A face names a vertex the mesh does not have.

    Fatal for an exporter rather than skippable: the index is written as-is into
    the file, and a consumer resolving it reads whatever vertex happens to be
    there, or crashes.
    """


class DegenerateFace(MeshNotAllowed):
    """A face names the same vertex twice, so it has no area and no normal.

    Refused at export. Marching cubes here cannot produce one - the half-integer
    level keeps two cut edges from ever sharing a vertex - so a degenerate face
    reaching the writer means the mesh was assembled somewhere else, and passing
    it through would put a triangle no renderer can shade into a file that
    claims to be a surface.
    """


class VertexColourCountMismatch(MeshNotAllowed):
    """``Mesh.vertex_colours`` is present but does not carry one entry per vertex.

    The field is all-or-nothing by design - see :class:`Mesh` - so a caller
    who attaches colour is attaching it for every vertex; a shorter or longer
    tuple means the caller's own indexing has already drifted from the mesh
    it is describing, and writing it out regardless would silently pair some
    vertex with the wrong colour, or with none the file format has room to
    represent.
    """


class VertexColourChannelOutOfRange(MeshNotAllowed):
    """A vertex colour channel is outside ``[0, 255]``, what a PLY ``uchar``
    vertex property can hold.

    Checked before the header is built, so the failure names the actual bad
    channel rather than surfacing later as a ``struct.error`` from packing a
    binary body.
    """


class EmptyMesh(MeshNotAllowed):
    """The mesh has no faces, so it is not a surface and is not written.

    An empty reconstruction is a real and reportable outcome - the grid found no
    boundary - but a zero-face file under a name promising a scan reads as a
    successful scan of nothing. The record says so instead.
    """


class MeshCountExceeded(MeshNotAllowed):
    """More vertices or faces were offered than the cap allows."""


class OutputSizeExceeded(MeshNotAllowed):
    """The encoded file would pass the byte ceiling, so nothing was written."""


class MeshWriteIncomplete(OSError):
    """The file could not be written in full, so none was kept.

    An ``OSError`` for the same reason
    :class:`scanner3d.pointcloud.PointCloudWriteIncomplete` is: it is raised on
    the same path as a real write failure and carries an ``errno``, so a short
    write, a zero-progress write, an existing name and a kernel error all stay
    describable in one field.
    """


# --- the marching cubes tables ------------------------------------------------

#: The eight cube corners, in the numbering every published marching cubes table
#: uses. Corner ``c`` sits at ``CORNER_OFFSETS[c]`` in cell-local coordinates.
CORNER_OFFSETS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

#: The twelve cube edges as corner pairs, in the same standard numbering.
EDGE_CORNERS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

#: The six faces as corner cycles, each ordered counter-clockwise *seen from
#: outside the cube*. The direction is load-bearing, not cosmetic: it is what
#: makes two cubes traverse their shared face in opposite senses, which is what
#: makes them agree on a segment and disagree on its winding - and a closed
#: surface needs exactly that.
FACE_CORNER_CYCLES: Final[tuple[tuple[int, int, int, int], ...]] = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
)

_EDGE_BY_CORNERS: Final[dict[frozenset[int], int]] = {
    frozenset(pair): index for index, pair in enumerate(EDGE_CORNERS)
}

#: The four edges of each face, as sets. Two cut edges that appear together in
#: one of these are the two edges of a grid face, and so belong to *two* cells
#: at once; two that do not belong to at most one. :func:`_fan` turns on that
#: distinction - see it for why.
_FACE_EDGE_SETS: Final[tuple[frozenset[int], ...]] = tuple(
    frozenset(
        _EDGE_BY_CORNERS[frozenset((cycle[index], cycle[(index + 1) % 4]))] for index in range(4)
    )
    for cycle in FACE_CORNER_CYCLES
)

#: Which axis each edge runs along, so a cut edge can be named by its lower node
#: and a direction and shared between the (up to four) cells that meet on it.
EDGE_AXIS: Final[tuple[int, ...]] = tuple(
    next(axis for axis in range(3) if CORNER_OFFSETS[a][axis] != CORNER_OFFSETS[b][axis])
    for a, b in EDGE_CORNERS
)


def _face_segments(mask: int, corner_cycle: tuple[int, int, int, int]) -> Iterator[tuple[int, int]]:
    """The directed contour segments one face contributes for one corner pattern.

    Walking the cycle, a crossing is *leaving* when it runs from a corner above
    the level to one below. Each leaving crossing is joined to the nearest
    preceding entering one. With two crossings that pairing is forced; with four
    it separates the corners above the level, and it is a function of the four
    face corners alone - which is the whole requirement, because the cube on the
    other side of this face computes from the same four signs.
    """
    crossings: list[tuple[int, bool]] = []
    for index, corner in enumerate(corner_cycle):
        following = corner_cycle[(index + 1) % 4]
        inside_here = bool(mask >> corner & 1)
        if inside_here != bool(mask >> following & 1):
            crossings.append((_EDGE_BY_CORNERS[frozenset((corner, following))], inside_here))
    for index, (edge, leaving) in enumerate(crossings):
        if not leaving:
            continue
        for back in range(1, len(crossings) + 1):
            other_edge, other_leaving = crossings[index - back]
            if not other_leaving:
                yield (edge, other_edge)
                break


def _polygons(mask: int) -> list[tuple[int, ...]]:
    """The closed contour polygons for one corner pattern, as cycles of edge ids.

    Every cut edge lies on exactly two faces and is paired once on each, so
    every cut edge has in-degree one and out-degree one and the segments fall
    apart into disjoint directed cycles. That is the closure argument for the
    whole algorithm, and it does not depend on which case this is.

    Each cycle is emitted reversed. Traced forwards it keeps the region above
    the level on the side a triangle fanned from it would call front, which puts
    every normal *into* the solid; reversed, the normals point out of it, which
    is what every consumer of a surface expects. The flip is uniform, so the
    property closure rests on survives it: two cubes sharing a face still emit
    that face's segment in opposite directions.
    """
    successor: dict[int, int] = {}
    for corner_cycle in FACE_CORNER_CYCLES:
        for start, end in _face_segments(mask, corner_cycle):
            successor[start] = end
    polygons: list[tuple[int, ...]] = []
    while successor:
        first = next(iter(successor))
        cycle = [first]
        current = successor.pop(first)
        while current != first:
            cycle.append(current)
            current = successor.pop(current)
        cycle.reverse()
        polygons.append(tuple(cycle))
    return polygons


def _fan(polygon: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Triangulate one contour polygon, choosing an apex that stays manifold.

    Fanning from an arbitrary vertex is the obvious thing and it is subtly
    wrong. The fan's diagonals are used twice inside this cell, which is
    correct - unless the two cut edges a diagonal joins happen to be two edges
    of the same *grid face*. Those two edges belong to two cells, both cells
    draw the same diagonal, and the result is one mesh edge used by four
    triangles: a pinch that no geometry asked for and that no later stage can
    undo.

    Two cut edges lie in more than one cell exactly when they are two edges of a
    common grid face, so an apex whose diagonals all avoid such pairs cannot
    produce that collision. One always exists - every polygon this table
    contains has at least one, which ``tests/test_mesh.py`` checks for all 256
    cases - and the lowest-numbered one is taken so the table is deterministic.
    """
    size = len(polygon)
    apex = next(
        candidate
        for candidate in range(size)
        if not any(
            {polygon[candidate], polygon[(candidate + step) % size]} <= face
            for step in range(2, size - 1)
            for face in _FACE_EDGE_SETS
        )
    )
    rotated = polygon[apex:] + polygon[:apex]
    return [(rotated[0], rotated[index], rotated[index + 1]) for index in range(1, size - 1)]


def _build_tables() -> tuple[tuple[int, ...], tuple[tuple[tuple[int, int, int], ...], ...]]:
    """Both 256-entry tables, computed once at import from :func:`_polygons`.

    The polygon's own segments lie on a grid face and are used by one triangle
    here and one in the cell across that face; :func:`_fan` guarantees its
    diagonals are used by exactly two triangles of this cell and by nothing
    else. Every edge of the emitted surface therefore has exactly two faces -
    which is why :attr:`Mesh.is_watertight` is worth *measuring*: the algorithm
    has an argument for it, and a measurement is what turns an argument into
    evidence.
    """
    edges: list[int] = []
    triangles: list[tuple[tuple[int, int, int], ...]] = []
    for mask in range(256):
        cut = 0
        fan: list[tuple[int, int, int]] = []
        for polygon in _polygons(mask):
            for edge in polygon:
                cut |= 1 << edge
            fan.extend(_fan(polygon))
        edges.append(cut)
        triangles.append(tuple(fan))
    return tuple(edges), tuple(triangles)


_TABLES: Final = _build_tables()

#: Which of the twelve edges each case cuts, as a bitmask. Derived from the
#: corner signs and used to decide which vertices a cell has to place.
EDGE_TABLE: Final = _TABLES[0]

#: The triangles each case emits, as triples of edge ids. 256 entries, standard
#: numbering, computed rather than transcribed - see the module docstring.
TRIANGLE_TABLE: Final = _TABLES[1]


def case_index(values: Sequence[float], iso_level: float = DEFAULT_ISO_LEVEL) -> int:
    """The 8-bit case for one cell's corner values, in the table's own numbering.

    Bit ``c`` is set when corner ``c`` is strictly above the level. Strictly, so
    that a value sitting exactly on the level falls outside - one rule, applied
    the same way by every cell that shares that corner.
    """
    if len(values) != CELLS_PER_NODE:
        raise MeshNotAllowed(
            f"a cell has {CELLS_PER_NODE} corners; {len(values)} values were offered"
        )
    return sum(1 << corner for corner, value in enumerate(values) if value > iso_level)


# --- the mesh -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeCensus:
    """What counting a mesh's edges actually found. Evidence, not a claim.

    Every field here is a count taken from the faces themselves. A closed
    surface is one where every edge is used by exactly two faces; a consistently
    oriented one is a closed surface where those two uses run in opposite
    directions. Both are measured, and both are reported alongside the counts
    that produced them so a reader can see what was checked.
    """

    edge_count: int
    boundary_edge_count: int
    shared_edge_count: int
    nonmanifold_edge_count: int
    reversed_edge_count: int

    @property
    def closed(self) -> bool:
        """No edge used once, none used three or more times."""
        return (
            self.edge_count > 0
            and self.boundary_edge_count == 0
            and self.nonmanifold_edge_count == 0
        )

    @property
    def consistently_oriented(self) -> bool:
        """Every twice-used edge is traversed once each way."""
        return (
            self.nonmanifold_edge_count == 0 and self.reversed_edge_count == self.shared_edge_count
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "edge_count": self.edge_count,
            "boundary_edge_count": self.boundary_edge_count,
            "shared_edge_count": self.shared_edge_count,
            "nonmanifold_edge_count": self.nonmanifold_edge_count,
            "reversed_edge_count": self.reversed_edge_count,
            "closed": self.closed,
            "consistently_oriented": self.consistently_oriented,
        }


def edge_census(faces: Sequence[tuple[int, int, int]]) -> EdgeCensus:
    """Count how many faces use each edge, and which way round.

    Linear in the faces and allocating one counter per edge, which is why the
    result is a record a caller can keep rather than something recomputed by
    every property that wants a piece of it.
    """
    undirected: Counter[tuple[int, int]] = Counter()
    directed: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for start, end in ((a, b), (b, c), (c, a)):
            undirected[(start, end) if start < end else (end, start)] += 1
            directed[(start, end)] += 1
    boundary = sum(1 for count in undirected.values() if count == 1)
    shared = sum(1 for count in undirected.values() if count == 2)
    nonmanifold = sum(1 for count in undirected.values() if count > 2)
    reversed_pairs = sum(
        1
        for (low, high), count in undirected.items()
        if count == 2 and directed[(low, high)] == 1 and directed[(high, low)] == 1
    )
    return EdgeCensus(
        edge_count=len(undirected),
        boundary_edge_count=boundary,
        shared_edge_count=shared,
        nonmanifold_edge_count=nonmanifold,
        reversed_edge_count=reversed_pairs,
    )


@dataclass(frozen=True, slots=True)
class Mesh:
    """A triangle surface: shared vertices, indexed faces, and what is not known.

    Vertices are bare :data:`scanner3d.geometry.Point` tuples, for the reason
    that module gives: bulk geometry is never wrapped in objects. Faces index
    into them, and every face of a mesh built by :func:`reconstruct` indexes a
    vertex that is *shared* with its neighbours rather than duplicated per
    triangle - without that, edge counting measures nothing and
    :attr:`is_watertight` would be a coin toss.

    :attr:`limitations` travels with the geometry so that a consumer holding
    only this object still holds the caveats. It is not optional in practice:
    :func:`reconstruct` always fills it, and the exporters put it in the file.

    :attr:`vertex_colours` is ``None`` by default - what :func:`reconstruct`
    always produces, since marching cubes has no colour to give a vertex it
    invents on a grid edge; see :mod:`scanner3d.texture` for how a colour is
    actually transferred onto these vertices from a separately coloured
    point cloud. When present it is all-or-nothing: one ``(red, green,
    blue)`` triple per vertex, in vertex order, never a mix of a triple and
    a missing entry - :func:`encode_ply_mesh` refuses
    (:class:`VertexColourCountMismatch`) anything else rather than guess
    which vertices the caller meant to leave out.
    """

    vertices: tuple[Point, ...]
    faces: tuple[tuple[int, int, int], ...]
    limitations: tuple[str, ...] = ()
    vertex_colours: tuple[tuple[int, int, int], ...] | None = None

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def census(self) -> EdgeCensus:
        """The edge count, recomputed on each access.

        Deliberately not cached: the dataclass is frozen so that a mesh cannot
        drift from its own measurement, and a cached census would be exactly
        such a drift if anything ever built a mesh by replacing one field.
        """
        return edge_census(self.faces)

    @property
    def is_watertight(self) -> bool:
        """Measured, never assumed: is every edge shared by exactly two faces?

        An empty mesh is not watertight. Vacuous truth is the wrong answer to
        "is this a closed surface" when the honest answer is "there is no
        surface here".
        """
        return self.face_count > 0 and self.census.closed

    @property
    def bounds(self) -> tuple[Point, Point] | None:
        """The axis-aligned extent, or ``None`` when there are no vertices.

        In the input's own coordinates, whose unit is unestablished. This is a
        span of numbers, not a size.
        """
        if not self.vertices:
            return None
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def as_dict(self) -> dict[str, object]:
        """The record to serialise beside the mesh - counts and caveats, no geometry.

        The vertices and faces are deliberately absent. They belong in the
        exported file, which has a format built for them; a metadata document
        that also carried a million coordinates would be a second copy of the
        artifact, free to disagree with the first.
        """
        census = self.census
        extent = self.bounds
        return {
            "schema_version": SCHEMA_VERSION,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edges": census.as_dict(),
            "watertight": {
                "measured": self.is_watertight,
                "how": (
                    "every undirected edge of every face was counted; a closed surface has "
                    "each of them used by exactly two faces"
                ),
            },
            "bounds": None if extent is None else {"min": list(extent[0]), "max": list(extent[1])},
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


# --- reconstruction -----------------------------------------------------------

_UNKNOWN: Final = 0
_OCCUPIED: Final = 1
_EXTERIOR: Final = 2


def _ensure_voxel_size(voxel_size: float) -> float:
    ensure_finite((voxel_size,), "voxel size")
    if voxel_size <= 0.0:
        raise VoxelSizeUnusable(
            f"voxel size {voxel_size!r} is not positive; a grid cannot be built from it"
        )
    return voxel_size


def _ensure_iso_level(iso_level: float) -> float:
    ensure_finite((iso_level,), "iso level")
    if not 0.0 < iso_level < float(CELLS_PER_NODE):
        raise IsoLevelUnusable(
            f"iso level {iso_level!r} is outside (0, {CELLS_PER_NODE}); the field counts the "
            "solid cells around a node, so a level outside that range puts every node on the "
            "same side and no surface can exist"
        )
    return iso_level


def _quantise_into(
    points: Iterable[Sequence[float]],
    voxel_size: float,
    max_points: int,
    occupied: set[tuple[int, int, int]],
) -> int:
    """Add the points to ``occupied`` as voxel coordinates, and say how many were read.

    The floats are never kept. Quantising as they stream past is both the
    spatial index and the downsample, and it bounds the memory by the number of
    *occupied voxels* rather than by the number of points. The cap is enforced
    while the iterable is being consumed, so an unbounded producer is stopped
    rather than exhausted.

    The half in ``floor(coordinate / voxel_size + 0.5)`` is the difference
    between a cell a sample sits *in* and a cell a sample sits at the *centre*
    of, and it is worth a whole half-voxel of accuracy. Truncating instead puts
    every sample at the low corner of its cell, and the reconstructed boundary
    then lies entirely outside the samples, up to a full voxel out. Rounding
    centres the cell on the sample, and :func:`_march` shifts the grid origin
    back by the matching half voxel so the two agree.

    Adds INTO a set the caller owns rather than returning a new one, which is
    what lets :class:`SurfaceAccumulator` keep one growing set across a whole
    capture instead of re-reading every frame it has ever seen. It says nothing
    about an empty input: emptiness is a refusal for :func:`reconstruct` and a
    normal event for an accumulator, and those two answers belong to the callers
    rather than to the arithmetic.
    """
    read = 0
    for index, point in enumerate(points):
        if index >= max_points:
            raise PointCountExceeded(
                f"refusing to read more than {max_points} points; the producer offered at "
                f"least {index + 1}, and the cap is never widened at run time"
            )
        values = tuple(point)
        if len(values) != 3:
            raise MeshNotAllowed(
                f"point {index} carries {len(values)} coordinates, not the 3 an (x, y, z) "
                "triple has"
            )
        ensure_finite(values, f"point {index}")
        occupied.add(
            (
                math.floor(values[0] / voxel_size + 0.5),
                math.floor(values[1] / voxel_size + 0.5),
                math.floor(values[2] / voxel_size + 0.5),
            )
        )
        read = index + 1
    return read


def _quantise(
    points: Iterable[Sequence[float]], voxel_size: float, max_points: int
) -> set[tuple[int, int, int]]:
    """Every point's voxel coordinate, refusing a cloud with no points in it."""
    occupied: set[tuple[int, int, int]] = set()
    _quantise_into(points, voxel_size, max_points, occupied)
    if not occupied:
        raise NoPointsOffered(
            "no points were offered, so there is no cloud to reconstruct; refused rather "
            "than answered with an empty mesh, which downstream is indistinguishable from a "
            "reconstruction that found no surface in real data"
        )
    return occupied


def _grid_extent(
    occupied: set[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The padded low corner and the grid shape the occupied cells imply."""
    low = tuple(min(cell[axis] for cell in occupied) - PAD_CELLS for axis in range(3))
    high = tuple(max(cell[axis] for cell in occupied) + PAD_CELLS for axis in range(3))
    shape = tuple(high[axis] - low[axis] + 1 for axis in range(3))
    return (low[0], low[1], low[2]), (shape[0], shape[1], shape[2])


def _surface_of(occupied: set[tuple[int, int, int]], voxel: float, level: float) -> Mesh:
    """Solidify the occupied cells and march their boundary. No bounds checked here.

    The half of :func:`reconstruct` that is arithmetic, split out so that
    :class:`SurfaceAccumulator` runs the identical fill and the identical march
    rather than a second implementation of either. Two marching cubes in one
    module that disagreed by a cell would put the live preview and the exported
    mesh on different surfaces, and that discrepancy is invisible until someone
    compares them.
    """
    low, (nx, ny, nz) = _grid_extent(occupied)
    state = bytearray(nx * ny * nz)
    for i, j, k in occupied:
        state[((k - low[2]) * ny + (j - low[1])) * nx + (i - low[0])] = _OCCUPIED
    _solidify(state, nx, ny, nz)
    enclosed = state.count(_UNKNOWN)

    vertices, faces = _march(state, (nx, ny, nz), low, voxel, level)
    return Mesh(
        vertices=vertices,
        faces=faces,
        limitations=(
            *INHERENT_LIMITATIONS,
            _grid_limitation(voxel, nx, ny, nz, len(occupied)),
            _enclosure_limitation(enclosed),
            _closure_limitation(faces),
        ),
    )


def _border_cells(nx: int, ny: int, nz: int) -> Iterator[int]:
    """Every cell on the outer shell of the grid, as linear indices.

    The fill starts from all of them rather than from one corner. The padding
    does guarantee the shell is empty and connected, so one seed would do, but
    seeding the whole shell makes the fill correct without depending on that.
    """
    for k in range(nz):
        k_edge = k in (0, nz - 1)
        for j in range(ny):
            j_edge = j in (0, ny - 1)
            base = (k * ny + j) * nx
            if k_edge or j_edge:
                yield from range(base, base + nx)
            else:
                yield base
                yield base + nx - 1


def _solidify(state: bytearray, nx: int, ny: int, nz: int) -> None:
    """Mark every cell a six-connected fill from the border can reach as exterior.

    Six-connected and not twenty-six: a diagonal step slips between two cells
    that share only an edge, and a fill that can do that leaks through a wall
    the samples actually built. Whatever the fill cannot reach is interior,
    which is the only way an unoriented point cloud can be told apart from the
    solid it is the surface of.
    """
    plane = nx * ny
    queue: deque[int] = deque()
    for cell in _border_cells(nx, ny, nz):
        if state[cell] == _UNKNOWN:
            state[cell] = _EXTERIOR
            queue.append(cell)
    while queue:
        cell = queue.popleft()
        k, remainder = divmod(cell, plane)
        j, i = divmod(remainder, nx)
        if i > 0 and state[cell - 1] == _UNKNOWN:
            state[cell - 1] = _EXTERIOR
            queue.append(cell - 1)
        if i + 1 < nx and state[cell + 1] == _UNKNOWN:
            state[cell + 1] = _EXTERIOR
            queue.append(cell + 1)
        if j > 0 and state[cell - nx] == _UNKNOWN:
            state[cell - nx] = _EXTERIOR
            queue.append(cell - nx)
        if j + 1 < ny and state[cell + nx] == _UNKNOWN:
            state[cell + nx] = _EXTERIOR
            queue.append(cell + nx)
        if k > 0 and state[cell - plane] == _UNKNOWN:
            state[cell - plane] = _EXTERIOR
            queue.append(cell - plane)
        if k + 1 < nz and state[cell + plane] == _UNKNOWN:
            state[cell + plane] = _EXTERIOR
            queue.append(cell + plane)


def reconstruct(
    points: Iterable[Sequence[float]],
    *,
    voxel_size: float,
    iso_level: float = DEFAULT_ISO_LEVEL,
    max_grid_cells: int = MAX_GRID_CELLS,
    max_points: int = MAX_INPUT_POINTS,
) -> Mesh:
    """Turn a point cloud into a closed triangle surface at ``voxel_size`` resolution.

    The result's coordinates are the input's coordinates - each cell is centred
    on the coordinates it quantises and nothing is recentred or rescaled - so a
    mesh can be compared against the cloud it came from without a transform in
    between. It inherits the cloud's units, which for this repository means it
    inherits the fact that there are none established.

    ``iso_level`` is how many of a grid node's eight touching cells must be
    solid for it to count as inside. The default 3.5 asks for a majority, which
    drops an isolated sample and a one-cell filament along with the speckle;
    lowering it towards 0.5 keeps every sample and every outlier, and stands the
    surface off further. A half-integer is worth keeping: on one no node can sit
    exactly on the level, which is what stops two cut edges from ever meeting at
    a point and making a triangle with no area.

    ``max_grid_cells`` and ``max_points`` may be *lowered* by a caller and never
    raised: the module ceilings are the ceilings, and an argument that could
    widen one would make the bound a suggestion. The grid cap is checked from
    the cell count alone, before any array is allocated.

    Every refusal happens before the grid exists:
    :class:`VoxelSizeUnusable`, :class:`IsoLevelUnusable`,
    :class:`NoPointsOffered`, :class:`PointCountExceeded` and
    :class:`GridTooLarge`.

    The returned mesh's :attr:`Mesh.limitations` records what this particular
    run did - the grid it used, how many cells it enclosed, and whether the
    surface it produced measured as closed. None of that is inferred from the
    absence of an exception.
    """
    voxel = _ensure_voxel_size(voxel_size)
    level = _ensure_iso_level(iso_level)
    if not 0 < max_grid_cells <= MAX_GRID_CELLS:
        raise MeshNotAllowed(
            f"max_grid_cells {max_grid_cells} is outside (0, {MAX_GRID_CELLS}]; the module "
            "ceiling is never widened at run time"
        )
    if not 0 < max_points <= MAX_INPUT_POINTS:
        raise MeshNotAllowed(
            f"max_points {max_points} is outside (0, {MAX_INPUT_POINTS}]; the module ceiling "
            "is never widened at run time"
        )

    occupied = _quantise(points, voxel, max_points)
    _, (nx, ny, nz) = _grid_extent(occupied)
    cells = nx * ny * nz
    if cells > max_grid_cells:
        raise GridTooLarge(
            f"a voxel size of {voxel!r} over this extent needs a {nx}x{ny}x{nz} grid of "
            f"{cells} cells, past the {max_grid_cells}-cell cap; nothing was allocated. Use "
            "a larger voxel size, or raise the cap up to the module ceiling"
        )
    return _surface_of(occupied, voxel, level)


def _grid_limitation(voxel: float, nx: int, ny: int, nz: int, occupied: int) -> str:
    return (
        f"reconstructed at voxel size {voxel!r} in the input's own unestablished units, on a "
        f"{nx}x{ny}x{nz} grid of {nx * ny * nz} cells of which {occupied} were occupied by at "
        "least one sample; a different voxel size gives a different surface, and none of them "
        "is the object"
    )


def _enclosure_limitation(enclosed: int) -> str:
    if enclosed == 0:
        return (
            "the six-connected fill reached every cell the samples did not occupy, so this "
            "cloud encloses no volume: what was reconstructed is a shell around the samples, "
            "with an inner and an outer wall about one voxel apart, and neither wall is the "
            "surface - the samples lie between them"
        )
    return (
        f"{enclosed} cells were enclosed by the samples and could not be reached from the "
        "grid border, so this surface is the boundary of that solid; the fill decides what is "
        "inside, and it has no evidence beyond the samples' own connectivity"
    )


def _closure_limitation(faces: tuple[tuple[int, int, int], ...]) -> str:
    census = edge_census(faces)
    if not faces:
        return (
            "no surface was found: no cell of the grid straddled the level, so this mesh has "
            "no faces at all. An empty mesh is a result, not a failure to report one"
        )
    if census.closed:
        return (
            f"measured closed: all {census.edge_count} edges are used by exactly two faces, "
            f"and {census.reversed_edge_count} of them are traversed once each way. Closure is "
            "a property of the algorithm over a padded grid, not evidence that the object was "
            "observed from every side"
        )
    return (
        f"measured NOT closed: of {census.edge_count} edges, {census.boundary_edge_count} are "
        f"used by one face and {census.nonmanifold_edge_count} by three or more. The surface "
        "has holes or self-contact and should not be treated as a solid"
    )


# --- reconstructing WHILE the capture runs ------------------------------------


class SurfaceAccumulator:
    """The occupied cells of one growing scan, and the surface they currently imply.

    Why this exists
    ---------------
    :func:`reconstruct` reads a whole cloud and throws its grid away. A capture
    that wants to show a surface every few frames cannot afford that: at 900
    points a frame and several hundred frames, re-reading everything seen so far
    costs more every time it is asked, on a thread that is between two frame
    deliveries.

    What makes the incremental form possible is that the grid is the ONLY state
    the reconstruction has. Occupancy is a set union - a cell a frame marks
    solid is solid whatever arrives later - so a frame can be folded in once and
    then forgotten, and the memory is bounded by the cells the object occupies
    rather than by how long the scan has run. Measured over a 120-frame
    synthetic sweep: 0.6 ms per frame to fold in, and an occupied set that stops
    growing once the sweep has been round once.

    What it costs, and what it is not
    ---------------------------------
    :meth:`surface` is NOT incremental. The fill and the march are linear in the
    grid's cells, and they are redone in full every time one is asked for -
    measured at 64 ms over a 79000-cell grid and 289 ms over a 238000-cell one.
    That is the number a caller has to budget against; folding frames in is free
    by comparison.

    And the deeper limitation, which no amount of speed answers: a surface from
    this accumulator is built from the poses that were accepted AT THE TIME each
    frame was added. During a live capture those are the sequential chain, which
    drifts. Nothing here revises a cell once it is set, so a pose correction -
    the pose graph closing the loop at the end of a sweep - does not reach the
    cells that were placed with the old poses. A live surface is a PREVIEW of
    where the scan has been; the exported mesh is reconstructed again from the
    optimised poses, and the two are not expected to be identical.

    Coarsening, and why it is the caller's call
    -------------------------------------------
    An object that turns out to be bigger than the first frames suggested
    eventually needs more cells than the caller budgeted for. :meth:`coarsen`
    doubles the cell edge and re-bins the cells' own CENTRES, which is exact
    integer arithmetic on the indices - a cell centre is ``index * voxel_size``,
    so at twice the edge its new index is ``(index + 1) // 2`` - and costs one
    pass over the occupied cells rather than over the points, which are long
    gone. It loses resolution, permanently and without any way to get it back.

    It is not done automatically. :attr:`grid_cells` says what the current cells
    would cost to march, and the caller decides what it can afford; a class that
    silently halved its own resolution would change the picture an operator is
    watching for a reason they never saw.
    """

    __slots__ = ("_added", "_coarsenings", "_max_points", "_occupied", "_voxel_size")

    def __init__(self, *, voxel_size: float, max_points: int = MAX_INPUT_POINTS) -> None:
        """Start empty, at ``voxel_size``.

        ``max_points`` bounds ONE :meth:`add`, not the accumulator's lifetime:
        the whole point of the grid is that the number of points that have
        passed through it does not bound anything. It may be lowered by a caller
        and never raised past the module ceiling.
        """
        if not 0 < max_points <= MAX_INPUT_POINTS:
            raise MeshNotAllowed(
                f"max_points {max_points} is outside (0, {MAX_INPUT_POINTS}]; the module "
                "ceiling is never widened at run time"
            )
        self._voxel_size = _ensure_voxel_size(voxel_size)
        self._max_points = max_points
        self._occupied: set[tuple[int, int, int]] = set()
        self._added = 0
        self._coarsenings = 0

    # -- what it holds -----------------------------------------------------
    @property
    def voxel_size(self) -> float:
        """The current cell edge, which :meth:`coarsen` doubles."""
        return self._voxel_size

    @property
    def occupied_cells(self) -> int:
        """How many cells any point has ever landed in."""
        return len(self._occupied)

    @property
    def added_points(self) -> int:
        """How many points have been folded in, across every :meth:`add`."""
        return self._added

    @property
    def coarsenings(self) -> int:
        """How many times the cell edge has been doubled since this was empty."""
        return self._coarsenings

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """The padded grid the current cells imply, or zeros while it is empty.

        This is what :meth:`surface` would have to allocate and walk, so it is
        the quantity a caller budgets against - not the occupied count, which
        for a hollow object is a small fraction of it.
        """
        if not self._occupied:
            return (0, 0, 0)
        return _grid_extent(self._occupied)[1]

    @property
    def grid_cells(self) -> int:
        """Cells in that grid: the cost of the next :meth:`surface`, before paying it."""
        nx, ny, nz = self.grid_shape
        return nx * ny * nz

    @property
    def is_empty(self) -> bool:
        """Whether anything has been added that landed in a cell."""
        return not self._occupied

    # -- growing it --------------------------------------------------------
    def add(self, points: Iterable[Sequence[float]]) -> int:
        """Fold one frame's points into the grid, and say how many cells are new.

        The points must already be in the shared frame - this applies no pose
        and knows of none. An empty iterable is a no-op returning zero, because
        a frame that caught nothing is a normal event during a capture and not
        an error the way it is for :func:`reconstruct`.

        Refuses exactly what :func:`reconstruct` refuses about points -
        :class:`PointCountExceeded` past ``max_points``, :class:`MeshNotAllowed`
        for something that is not a triple and the non-finite refusal from
        :func:`~scanner3d.geometry.ensure_finite` - and it refuses them
        MID-STREAM, so a bad point leaves the cells that preceded it in the
        grid. That is deliberate: the alternative is buffering a whole frame to
        preserve an all-or-nothing that the union does not need, and a cell is
        as true as the point that put it there whatever came next in the same
        frame.
        """
        before = len(self._occupied)
        self._added += _quantise_into(points, self._voxel_size, self._max_points, self._occupied)
        return len(self._occupied) - before

    def coarsen(self) -> float:
        """Double the cell edge, re-binning the cells that are already here.

        Returns the new edge. Exact integer arithmetic on the indices, as the
        class docstring explains, so this costs one pass over the occupied cells
        and never touches a coordinate.

        Lossy and one-way, in two distinct ways worth separating. Two cells that
        merge cannot be separated again, and the points that distinguished them
        were discarded when they were added. And the result is NOT the grid the
        original points would have produced at the new edge: a point is re-binned
        from its old cell's centre rather than from where it actually was, so a
        cell can land up to half the new edge away from where a fresh quantise
        would have put it. Coarsening is a way to keep drawing something on a
        budget, not a way to have chosen a coarser voxel size in the first place.
        """
        self._voxel_size *= 2.0
        self._occupied = {((i + 1) // 2, (j + 1) // 2, (k + 1) // 2) for i, j, k in self._occupied}
        self._coarsenings += 1
        return self._voxel_size

    def coarsen_to_fit(self, max_grid_cells: int) -> int:
        """Coarsen until :attr:`grid_cells` is within ``max_grid_cells``, or until
        coarsening stops helping. Returns how many times it doubled the edge.

        The loop a live caller would otherwise write, offered here so that the
        termination argument lives with the arithmetic - and it is NOT "the grid
        shrinks until it reaches the padding". Re-binning maps a cell index to
        ``(index + 1) // 2``, which has 0 and 1 as FIXED POINTS, so a grid whose
        occupied cells have settled on indices 0 and 1 in some axis stays that
        size for ever however many times it is halved. That grid is a handful of
        cells wide and still not within a small enough budget, and a loop that
        only watched the budget would spin.

        So what is watched is progress: when a doubling leaves the grid the same
        shape it was, this is as small as the grid gets and the budget cannot be
        met. The caller is not told twice - :meth:`surface` refuses a grid past
        the budget it is given, which is where an impossible budget surfaces as
        a refusal rather than as a hang.
        """
        if not 0 < max_grid_cells <= MAX_GRID_CELLS:
            raise MeshNotAllowed(
                f"max_grid_cells {max_grid_cells} is outside (0, {MAX_GRID_CELLS}]; the module "
                "ceiling is never widened at run time"
            )
        done = 0
        while self._occupied and self.grid_cells > max_grid_cells:
            before = self.grid_shape
            self.coarsen()
            done += 1
            if self.grid_shape == before:
                break
        return done

    # -- reading it --------------------------------------------------------
    def surface(
        self,
        *,
        iso_level: float = DEFAULT_ISO_LEVEL,
        max_grid_cells: int = MAX_GRID_CELLS,
    ) -> Mesh:
        """March the cells as they stand, and return the surface they imply.

        The same fill and the same march :func:`reconstruct` runs, over the same
        kind of grid, so a live preview and an exported mesh built from the same
        cells at the same edge are the same surface.

        Refuses rather than allocating: :class:`NoPointsOffered` while nothing
        has been added, :class:`IsoLevelUnusable` for a level no node can
        straddle, and :class:`GridTooLarge` when the cells imply a grid past
        ``max_grid_cells`` - which is the signal to call :meth:`coarsen_to_fit`,
        not something this decides on the caller's behalf.
        """
        level = _ensure_iso_level(iso_level)
        if not 0 < max_grid_cells <= MAX_GRID_CELLS:
            raise MeshNotAllowed(
                f"max_grid_cells {max_grid_cells} is outside (0, {MAX_GRID_CELLS}]; the module "
                "ceiling is never widened at run time"
            )
        if not self._occupied:
            raise NoPointsOffered(
                "nothing has been added to this accumulator, so there is no grid to march; "
                "refused rather than answered with an empty mesh, which downstream is "
                "indistinguishable from a scan that found no surface"
            )
        nx, ny, nz = self.grid_shape
        cells = nx * ny * nz
        if cells > max_grid_cells:
            raise GridTooLarge(
                f"the cells added so far need a {nx}x{ny}x{nz} grid of {cells} cells at a "
                f"voxel size of {self._voxel_size!r}, past the {max_grid_cells}-cell cap; "
                "nothing was allocated. Call coarsen_to_fit to trade resolution for it"
            )
        return _surface_of(self._occupied, self._voxel_size, level)


def _march(
    state: bytearray,
    dims: tuple[int, int, int],
    low: tuple[int, ...],
    voxel: float,
    level: float,
) -> tuple[tuple[Point, ...], tuple[tuple[int, int, int], ...]]:
    """Marching cubes over the cells that touch the boundary of the solid.

    The field at a node is the number of its eight touching cells that are
    solid, and a cell is only visited when at least one of its corners is
    *mixed* - somewhere strictly between 0 and 8. That is not an approximation.
    Two adjacent nodes share four cells, so their counts differ by at most four;
    a crossing needs one corner at or below the level and one strictly above it,
    and for any level in ``(0, 8)`` the arithmetic forces at least one of that
    pair to be mixed. Skipping the rest therefore skips only cells that could
    not have produced a triangle, which is what keeps this linear in the
    boundary rather than in the volume.

    Vertices are keyed by the grid edge they were cut from, so the four cells
    meeting on an edge share one vertex. Without that the mesh would be a soup
    of unattached triangles that no edge count could measure.
    """
    nx, ny, nz = dims
    node_x, node_y = nx + 1, ny + 1
    node_plane = node_x * node_y
    counts = bytearray(node_x * node_y * (nz + 1))

    #: Linear node offsets of a cell's eight corners, in the table's numbering.
    corner_offsets = tuple((dz * node_y + dy) * node_x + dx for dx, dy, dz in CORNER_OFFSETS)
    plane = nx * ny
    for cell in range(len(state)):
        if state[cell] == _EXTERIOR:
            continue
        k, remainder = divmod(cell, plane)
        j, i = divmod(remainder, nx)
        base = (k * node_y + j) * node_x + i
        for offset in corner_offsets:
            counts[base + offset] += 1

    active: set[int] = set()
    for node in range(len(counts)):
        value = counts[node]
        if value == 0 or value == CELLS_PER_NODE:
            continue
        k, remainder = divmod(node, node_plane)
        j, i = divmod(remainder, node_x)
        for dz in (0, 1):
            cell_k = k - dz
            if not 0 <= cell_k < nz:
                continue
            for dy in (0, 1):
                cell_j = j - dy
                if not 0 <= cell_j < ny:
                    continue
                for dx in (0, 1):
                    cell_i = i - dx
                    if 0 <= cell_i < nx:
                        active.add((cell_k * ny + cell_j) * nx + cell_i)

    vertices: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    by_edge: dict[int, int] = {}
    # Half a voxel back, because :func:`_quantise` rounded rather than
    # truncated: cell (i, j, k) is centred on the coordinate it quantises, not
    # cornered on it.
    origin = ((low[0] - 0.5) * voxel, (low[1] - 0.5) * voxel, (low[2] - 0.5) * voxel)
    # Sorted, so two runs over the same cloud produce byte-identical exports.
    for cell in sorted(active):
        k, remainder = divmod(cell, plane)
        j, i = divmod(remainder, nx)
        base = (k * node_y + j) * node_x + i
        corners = tuple(counts[base + offset] for offset in corner_offsets)
        # Through :func:`case_index` rather than inline, so the rule at the level
        # exists once. A second copy here could drift to ``>=`` while the public
        # function still documented and tested ``>``, and the two would then
        # disagree about which side a node sitting exactly on an integer level
        # falls - silently, because either rule closes as long as it is the only
        # one in use.
        mask = case_index(corners, level)
        table = TRIANGLE_TABLE[mask]
        if not table:
            continue
        local: dict[int, int] = {}
        for edge in range(12):
            if not EDGE_TABLE[mask] >> edge & 1:
                continue
            first, second = EDGE_CORNERS[edge]
            low_node = base + corner_offsets[first]
            high_node = base + corner_offsets[second]
            low_value, high_value = corners[first], corners[second]
            if low_node > high_node:
                low_node, high_node = high_node, low_node
                low_value, high_value = high_value, low_value
            axis = EDGE_AXIS[edge]
            key = low_node * 3 + axis
            index = by_edge.get(key)
            if index is None:
                index = len(vertices)
                by_edge[key] = index
                vertices.append(
                    _cut_point(
                        low_node,
                        axis,
                        low_value,
                        high_value,
                        level,
                        node_x,
                        node_y,
                        origin,
                        voxel,
                    )
                )
            local[edge] = index
        faces.extend((local[a], local[b], local[c]) for a, b, c in table)
    return tuple(vertices), tuple(faces)


def _cut_point(
    low_node: int,
    axis: int,
    low_value: int,
    high_value: int,
    level: float,
    node_x: int,
    node_y: int,
    origin: tuple[float, float, float],
    voxel: float,
) -> Point:
    """Where the level crosses one grid edge, linearly between its two nodes.

    The fraction is clamped away from both ends. With the default half-integer
    level it can never reach them, but a caller who picks an integer level can
    put a node exactly on it, and a vertex sitting exactly on a node is a vertex
    two different cut edges can produce - which is how a zero-area triangle gets
    made.
    """
    span = high_value - low_value
    fraction = 0.5 if span == 0 else (level - low_value) / span
    fraction = min(max(fraction, MIN_EDGE_FRACTION), 1.0 - MIN_EDGE_FRACTION)
    k, remainder = divmod(low_node, node_x * node_y)
    j, i = divmod(remainder, node_x)
    position = [origin[0] + i * voxel, origin[1] + j * voxel, origin[2] + k * voxel]
    position[axis] += fraction * voxel
    return (position[0], position[1], position[2])


# --- provenance ---------------------------------------------------------------


class MeshFileFormat(enum.StrEnum):
    """The encodings this module writes, named as they appear in the record."""

    OBJ = "obj"
    PLY_ASCII = "ply-ascii"
    PLY_BINARY_LITTLE_ENDIAN = "ply-binary_little_endian"


@dataclass(frozen=True, slots=True)
class MeshProvenance:
    """Everything a mesh file cannot say about itself.

    Wraps :class:`scanner3d.pointcloud.CloudProvenance` rather than restating
    it: the units, the axis convention and the calibration are properties of the
    cloud this surface was fitted to, and a mesh that described them differently
    would be describing a different cloud. What is added here is what the
    reconstruction did to that cloud, which the cloud's own record cannot know.
    """

    cloud: CloudProvenance
    method: str
    voxel_size: float | None = None
    #: How many of a node's eight touching cells had to be solid - a parameter
    #: of :func:`reconstruct` and of nothing else. A TSDF surface has no such
    #: quantity, so :attr:`summary` and :meth:`as_dict` report the fixed zero
    #: level for that family and do not print this one at all.
    iso_level: float | None = None
    limitations: tuple[str, ...] = ()
    #: Which construction this was, and so which fixed caveats the file
    #: inherits. Defaults to :attr:`MeshMethod.OCCUPANCY` deliberately: every
    #: call site that predates the field built an occupancy mesh, and a default
    #: that dropped occupancy's caveats would silently weaken files nobody
    #: touched. A caller SELECTS a family; it cannot supply one.
    method_family: MeshMethod = MeshMethod.OCCUPANCY

    @property
    def summary(self) -> str:
        """One line, for the file's comment block.

        The occupancy wording is byte for byte what it always was, including
        ``iso level: not stated`` when a caller left it out: those bytes are in
        every occupancy file this repository has ever written and a reader
        diffing an old export against a new one should see no difference.

        A TSDF mesh says the level instead of the iso level, because the two are
        different quantities and only one of them happened. An iso level is how
        many of a grid node's eight touching cells must be solid; the level of a
        fused field is zero, not by choice but because that is where a signed
        distance changes sign. Printing ``iso level:`` here - with a number or
        with ``not stated`` - would name a parameter of a reconstruction that
        did not run, which is the whole reason :class:`MeshMethod` exists.
        """
        voxel = "not stated" if self.voxel_size is None else repr(self.voxel_size)
        if MeshMethod(self.method_family) is MeshMethod.TSDF:
            return f"{self.method}; voxel size: {voxel}; level set: zero (the sign change)"
        level = "not stated" if self.iso_level is None else repr(self.iso_level)
        return f"{self.method}; voxel size: {voxel}; iso level: {level}"

    @property
    def inherent_limitations(self) -> tuple[str, ...]:
        """The fixed caveats of :attr:`method_family`, whatever the caller said."""
        return METHOD_LIMITATIONS[MeshMethod(self.method_family)]

    @property
    def all_limitations(self) -> tuple[str, ...]:
        """:attr:`inherent_limitations`, then the cloud's, then the caller's.

        The inherent ones are not the caller's to omit, and de-duplication keeps
        a caller who restates one from making the file say it twice.
        """
        seen: list[str] = []
        for item in (
            *self.inherent_limitations,
            *self.cloud.all_limitations,
            *self.limitations,
        ):
            if item not in seen:
                seen.append(item)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        """The record, carrying the level the family actually has.

        ``iso_level`` is an OCCUPANCY key and it keeps its place - present, and
        ``null`` when unstated - for that family, so records written before
        there was a second one still read identically. A TSDF record carries
        ``level_set`` instead: a consumer that finds ``iso_level`` in a mesh
        record is looking at a mesh whose surface really was decided by counting
        a node's solid cells, and one that finds it absent is not.
        """
        level: dict[str, object] = (
            {"level_set": 0.0}
            if MeshMethod(self.method_family) is MeshMethod.TSDF
            else {"iso_level": self.iso_level}
        )
        return {
            "method": self.method,
            "method_family": str(MeshMethod(self.method_family)),
            "voxel_size": self.voxel_size,
            **level,
            "cloud": self.cloud.as_dict(),
            "limitations": list(self.all_limitations),
        }


@dataclass(frozen=True, slots=True)
class MeshExport:
    """A complete mesh file held in memory, plus the metadata that belongs beside it.

    ``path`` is ``None`` until the bytes have actually been written, so a record
    naming a path is a record of a file that exists in full. :meth:`as_dict`
    excludes :attr:`payload`: the metadata sits beside the geometry, never in
    place of it.
    """

    file_format: MeshFileFormat
    vertex_count: int
    face_count: int
    payload: bytes
    header: str
    sha256: str
    provenance: MeshProvenance
    limitations: tuple[str, ...]
    watertight: bool
    census: EdgeCensus
    max_output_bytes: int
    path: str | None = None

    @property
    def byte_count(self) -> int:
        return len(self.payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": {
                "path": self.path,
                "written": self.path is not None,
                "format": str(self.file_format),
                "byte_count": self.byte_count,
                "sha256": self.sha256,
            },
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edges": self.census.as_dict(),
            "watertight": {
                "measured": self.watertight,
                "how": (
                    "every undirected edge of every face was counted; a closed surface has "
                    "each of them used by exactly two faces"
                ),
            },
            "dimensional_accuracy_claimed": False,
            "provenance": self.provenance.as_dict(),
            "limitations": list(self.limitations),
            # The exact comment block, so a reader of the metadata alone can see
            # what the file itself declares without parsing it.
            "header": self.header,
            "bounds": {
                "max_output_bytes": self.max_output_bytes,
                "module_max_output_bytes": MAX_OUTPUT_BYTES,
                "module_max_vertices": MAX_VERTICES,
                "module_max_faces": MAX_FACES,
            },
        }


# --- encoding -----------------------------------------------------------------


def binary32(value: float) -> float:
    """Round ``value`` to the binary32 both encodings carry, or refuse it.

    NaN and the infinities have already been refused by
    :func:`scanner3d.geometry.ensure_finite`, which owns that question for the
    geometry stack; what is left is the range check, which belongs to the file
    format. Rounding here and rendering ``repr`` of the rounded value in the
    ASCII bodies is what keeps an OBJ, an ASCII PLY and a binary PLY of one mesh
    from decoding to three slightly different surfaces.
    """
    ensure_finite((value,), "vertex coordinate")
    try:
        packed = _BINARY32.pack(value)
    except OverflowError as exc:
        raise MeshCoordinateNotEncodable(
            f"coordinate {value!r} is finite as a float64 but overflows the binary32 range a "
            "PLY 'float' property carries, so it would be written as an infinity"
        ) from exc
    rounded: float = _BINARY32.unpack(packed)[0]
    return rounded


def _validated(mesh: Mesh, *, max_output_bytes: int) -> tuple[tuple[float, float, float], ...]:
    """Refuse everything refusable about ``mesh``, and return its rounded vertices.

    Called before a path is touched by both writers, so every one of these
    refusals leaves no file behind.
    """
    if not 0 < max_output_bytes <= MAX_OUTPUT_BYTES:
        raise MeshNotAllowed(
            f"max_output_bytes {max_output_bytes} is outside (0, {MAX_OUTPUT_BYTES}]; the "
            "module ceiling is never widened at run time"
        )
    if mesh.face_count == 0:
        raise EmptyMesh(
            "this mesh has no faces, so it is not a surface and no file is written; the "
            "reconstruction record says what was found instead"
        )
    if mesh.vertex_count > MAX_VERTICES:
        raise MeshCountExceeded(f"{mesh.vertex_count} vertices is past the {MAX_VERTICES} cap")
    if mesh.face_count > MAX_FACES:
        raise MeshCountExceeded(f"{mesh.face_count} faces is past the {MAX_FACES} cap")
    for position, face in enumerate(mesh.faces):
        if len(face) != 3:
            raise MeshNotAllowed(
                f"face {position} names {len(face)} vertices, not the 3 a triangle has"
            )
        for index in face:
            if not 0 <= index < mesh.vertex_count:
                raise MeshIndexOutOfRange(
                    f"face {position} names vertex {index}, and this mesh has "
                    f"{mesh.vertex_count}; the index would be written as it stands and would "
                    "resolve, for a consumer, to some other vertex or to nothing"
                )
        if len(set(face)) != 3:
            raise DegenerateFace(
                f"face {position} is {face}, which names the same vertex twice: it has no "
                "area and no normal, and nothing in a file can repair that"
            )
    return tuple(
        (binary32(vertex[0]), binary32(vertex[1]), binary32(vertex[2])) for vertex in mesh.vertices
    )


def _one_line(text: str) -> str:
    """Flatten ``text`` to a single line of printable ASCII.

    This is the injection guard as much as a tidying step. Both formats are line
    structured and both take provenance strings from a caller: a units label
    containing a newline and ``end_header`` would end a PLY header early, and one
    containing a newline and ``f 1 2 3`` would add a face to an OBJ. Everything
    outside printable ASCII - newlines included - becomes a space, so no
    caller-supplied string can produce a line of its own.
    """
    printable = "".join(ch if " " <= ch <= "~" else " " for ch in text)
    return " ".join(printable.split())


def _comment(label: str, text: str) -> list[str]:
    """Comment bodies carrying ``label: text``, wrapped rather than truncated."""
    body = _one_line(text) or "(not stated)"
    return textwrap.wrap(f"{label}: {body}", width=COMMENT_WRAP_WIDTH, subsequent_indent="  ")


def provenance_comments(mesh: Mesh, provenance: MeshProvenance) -> list[str]:
    """The comment bodies both formats carry, without either one's prefix.

    These are the whole reason the exporters take a provenance record at all.
    Neither OBJ nor PLY has a field for a unit, an axis convention or a caveat,
    so a consumer handed the geometry alone guesses - and for this repository
    every guess on offer is wrong. The caveats travel in the file because there
    is nowhere else in the file to put them.
    """
    lines = _comment("produced by", provenance.cloud.producer)
    lines += _comment(
        "why these comments exist",
        "neither OBJ nor PLY records a unit, an axis convention or a provenance, so they can "
        "only be stated here; they are not decoration and this file makes no sense without "
        "them",
    )
    lines += _comment("reconstruction", provenance.summary)
    lines += _comment("units", provenance.cloud.units)
    lines += _comment("dimensional accuracy", "none is claimed; see the limitations below")
    lines += _comment("coordinate convention", provenance.cloud.convention.summary)
    lines += _comment("calibration", provenance.cloud.calibration.summary)
    lines += _comment(
        "source frame sha256", provenance.cloud.source_frame_sha256 or "none recorded"
    )
    census = mesh.census
    lines += _comment(
        "closed surface",
        f"{'yes' if mesh.is_watertight else 'no'}, measured by counting edges: "
        f"{census.edge_count} edges, {census.shared_edge_count} used by exactly two faces, "
        f"{census.boundary_edge_count} by one, {census.nonmanifold_edge_count} by three or more",
    )
    for limitation in combined_limitations(mesh, provenance):
        lines += _comment("limitation", limitation)
    return lines


def combined_limitations(mesh: Mesh, provenance: MeshProvenance) -> tuple[str, ...]:
    """The provenance's caveats and the mesh's own, in that order, de-duplicated.

    A mesh carries what its own reconstruction measured - the grid it used, what
    it enclosed, whether it closed - and the provenance carries what the cloud
    and the caller know. Both belong in the file, and neither is a subset of the
    other.
    """
    seen: list[str] = []
    for item in (*provenance.all_limitations, *mesh.limitations):
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def _header_block(lines: Sequence[str], prefix: str) -> str:
    """The comment lines with their format's prefix, refused if too long."""
    text = "".join(f"{prefix}{line}\n" for line in lines)
    encoded = text.encode("ascii")
    if len(encoded) > MAX_HEADER_BYTES:
        raise MeshNotAllowed(
            f"the comment block is {len(encoded)} bytes, past the {MAX_HEADER_BYTES}-byte "
            "ceiling; it is refused rather than truncated, because half a stated caveat "
            "reads like a whole one"
        )
    return text


def _finish(
    mesh: Mesh,
    provenance: MeshProvenance,
    file_format: MeshFileFormat,
    header: str,
    payload: bytes,
    max_output_bytes: int,
) -> MeshExport:
    """The record for an encoded file, size-checked, with the edges counted once.

    :attr:`Mesh.census` is recomputed on every access, so the count is taken
    here and both the census and the closure verdict are derived from that one
    pass rather than from three.
    """
    if len(payload) > max_output_bytes:
        raise OutputSizeExceeded(
            f"the encoded {file_format} mesh is {len(payload)} bytes for {mesh.face_count} "
            f"faces, past the {max_output_bytes}-byte ceiling; nothing was written"
        )
    census = mesh.census
    return MeshExport(
        file_format=file_format,
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
        payload=payload,
        header=header,
        sha256=sha256_of(payload),
        provenance=provenance,
        limitations=combined_limitations(mesh, provenance),
        watertight=mesh.face_count > 0 and census.closed,
        census=census,
        max_output_bytes=max_output_bytes,
    )


def encode_obj(
    mesh: Mesh,
    *,
    provenance: MeshProvenance,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> MeshExport:
    """Encode a whole OBJ in memory, or refuse. Opens nothing and writes nothing.

    OBJ carries no counts, so a truncated one is a valid OBJ of a smaller mesh -
    which is why :func:`write_obj` writes whole or not at all, and why this
    function exists separately: everything that can refuse the mesh refuses it
    here, with no file in the picture.
    """
    rounded = _validated(mesh, max_output_bytes=max_output_bytes)
    header = _header_block(provenance_comments(mesh, provenance), "# ")
    body = "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in rounded)
    # OBJ indexes from one. Off-by-one here is silent in every viewer.
    body += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in mesh.faces)
    payload = (header + body).encode("ascii")
    return _finish(mesh, provenance, MeshFileFormat.OBJ, header, payload, max_output_bytes)


def _ensure_vertex_colours(mesh: Mesh) -> tuple[tuple[int, int, int], ...] | None:
    """``mesh.vertex_colours``, checked against the mesh it is attached to, or refused.

    Returns ``None`` unchanged when the mesh carries no colour at all, which
    is what keeps every caller of this function - just :func:`encode_ply_mesh`
    - byte-identical to a mesh built before this field existed: nothing else
    about the header or body construction below is reached for that case.
    """
    colours = mesh.vertex_colours
    if colours is None:
        return None
    if len(colours) != mesh.vertex_count:
        raise VertexColourCountMismatch(
            f"this mesh has {mesh.vertex_count} vertices but vertex_colours carries "
            f"{len(colours)} entries; colour is all-or-nothing, one triple per vertex"
        )
    for index, (red, green, blue) in enumerate(colours):
        for channel, value in (("red", red), ("green", green), ("blue", blue)):
            if not 0 <= value <= 255:
                raise VertexColourChannelOutOfRange(
                    f"vertex {index}'s {channel} channel is {value!r}, outside [0, 255]"
                )
    return colours


def encode_ply_mesh(
    mesh: Mesh,
    *,
    provenance: MeshProvenance,
    ply_format: PlyFormat = PlyFormat.ASCII,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> MeshExport:
    """Encode a whole PLY mesh in memory, or refuse. Opens nothing, writes nothing.

    ASCII by default, unlike :func:`scanner3d.pointcloud.encode_ply`: a mesh is
    a few hundred thousand triangles at worst where a raw cloud is a quarter of
    a million points per frame, and a surface a reader can open in an editor is
    worth more here than the bytes it costs. Binary is one argument away.

    When :attr:`Mesh.vertex_colours` is ``None`` - what every mesh built
    before this field existed has, and what :func:`reconstruct` still
    produces on its own - this function's header and body are constructed
    exactly as before this field existed, byte for byte: the three
    ``property uchar`` lines below and the packed colour bytes are only ever
    added, never substituted for anything that was already there. When it is
    present, PLY's own standard per-vertex ``red``/``green``/``blue`` uchar
    properties carry it; that is what the format already has room for, so
    nothing here is invented.
    """
    colours = _ensure_vertex_colours(mesh)
    rounded = _validated(mesh, max_output_bytes=max_output_bytes)
    ascii_body = ply_format is PlyFormat.ASCII
    lines = [PLY_MAGIC, f"format {ply_format} {PLY_VERSION}"]
    lines += [f"comment {line}" for line in provenance_comments(mesh, provenance)]
    if colours is not None:
        # INHERENT_LIMITATIONS above already states, unconditionally, that no
        # mesh from this module carries colour - true of what reconstruct()
        # produces on its own, and left untouched so a colourless export's
        # comment block stays byte-identical to what it was before this
        # field existed. For THIS export that blanket statement is wrong, so
        # it is corrected here rather than left to stand contradicted by the
        # property lines a few lines below it.
        lines += [
            f"comment {line}"
            for line in _comment(
                "vertex colour",
                "this file's vertices carry a red/green/blue colour transferred from a "
                "separately coloured point cloud (see scanner3d.texture); the 'no normals, "
                "colours, texture coordinates or materials are exported' limitation above "
                "does not apply to this particular export",
            )
        ]
    lines += [
        f"element vertex {mesh.vertex_count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colours is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines += [
        f"element face {mesh.face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    # "\n" and never "\r\n": a trailing carriage return ends up inside the last
    # token of readers that split on "\n" alone, which a binary body makes hard
    # to diagnose.
    header = "".join(f"{line}\n" for line in lines)
    encoded_header = header.encode("ascii")
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise MeshNotAllowed(
            f"the header is {len(encoded_header)} bytes, past the {MAX_HEADER_BYTES}-byte "
            "ceiling; it is refused rather than truncated"
        )
    if colours is None:
        if ascii_body:
            body = "".join(f"{x!r} {y!r} {z!r}\n" for x, y, z in rounded).encode("ascii")
        else:
            body = b"".join(VERTEX_STRUCT.pack(x, y, z) for x, y, z in rounded)
    elif ascii_body:
        body = "".join(
            f"{x!r} {y!r} {z!r} {r} {g} {b}\n"
            for (x, y, z), (r, g, b) in zip(rounded, colours, strict=True)
        ).encode("ascii")
    else:
        body = b"".join(
            VERTEX_COLOUR_STRUCT.pack(x, y, z, r, g, b)
            for (x, y, z), (r, g, b) in zip(rounded, colours, strict=True)
        )
    if ascii_body:
        body += "".join(f"3 {a} {b} {c}\n" for a, b, c in mesh.faces).encode("ascii")
    else:
        body += b"".join(FACE_STRUCT.pack(3, a, b, c) for a, b, c in mesh.faces)
    file_format = (
        MeshFileFormat.PLY_ASCII if ascii_body else MeshFileFormat.PLY_BINARY_LITTLE_ENDIAN
    )
    return _finish(mesh, provenance, file_format, header, encoded_header + body, max_output_bytes)


# --- writing ------------------------------------------------------------------


def write_obj(
    path: Path,
    mesh: Mesh,
    *,
    provenance: MeshProvenance,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> MeshExport:
    """Encode the mesh as OBJ, then write it whole or leave nothing behind.

    Everything that can refuse the mesh has already refused it by the time
    ``path`` is touched, so a rejected mesh never creates a file.
    """
    export = encode_obj(mesh, provenance=provenance, max_output_bytes=max_output_bytes)
    _write_new_file(path, export.payload, ceiling=max_output_bytes)
    return replace(export, path=str(path))


def write_ply_mesh(
    path: Path,
    mesh: Mesh,
    *,
    provenance: MeshProvenance,
    ply_format: PlyFormat = PlyFormat.ASCII,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> MeshExport:
    """Encode the mesh as PLY, then write it whole or leave nothing behind."""
    export = encode_ply_mesh(
        mesh,
        provenance=provenance,
        ply_format=ply_format,
        max_output_bytes=max_output_bytes,
    )
    _write_new_file(path, export.payload, ceiling=max_output_bytes)
    return replace(export, path=str(path))


def _write_new_file(path: Path, payload: bytes, *, ceiling: int = MAX_OUTPUT_BYTES) -> None:
    """Create ``path`` exclusively and write all of ``payload``, or leave nothing.

    The same discipline as ``scanner3d.pointcloud._write_new_file``,
    reimplemented rather than imported, for the reason that module states about
    its own copy: each layer raises its own error type, which is what makes each
    one's failure reportable where it happens. The guarantees are identical and
    are the point of the duplication.

    ``O_EXCL | O_NOFOLLOW`` because an output directory is often a
    world-writable ``/tmp``: an existing name is an error rather than a licence
    to clobber, and a symlink is never followed out of the directory.

    ``os.write`` may write fewer bytes than it was given, so one call is not a
    file. This resumes from the offset the kernel actually reached, treats a
    zero-progress write as an error rather than spinning, and unlinks whatever
    it made if any of that fails. A truncated OBJ is the worst case in this
    module: it carries no counts, so it is a perfectly valid file of a mesh that
    was never reconstructed.
    """
    if len(payload) > ceiling:
        raise MeshWriteIncomplete(
            errno_mod.EFBIG,
            f"refusing to write {len(payload)} bytes to {path}, past the {ceiling} B ceiling",
        )
    try:
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
    except FileExistsError as exc:
        raise MeshWriteIncomplete(
            errno_mod.EEXIST, f"refusing to overwrite existing mesh {path}"
        ) from exc
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise MeshWriteIncomplete(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
        # The mesh is derived evidence a later run may be compared against, so
        # it is on the platter before this function claims it exists.
        os.fsync(fd)
    except BaseException:
        _discard_partial(path)
        raise
    finally:
        os.close(fd)


def _discard_partial(path: Path) -> None:
    """Remove a file that was created but not completely written.

    Best effort by design: the write already failed, and a failure to unlink
    must not replace the error that explains why there is no mesh.
    """
    with contextlib.suppress(OSError):
        path.unlink()
