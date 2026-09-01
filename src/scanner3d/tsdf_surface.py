"""One connected triangle surface, marched out of an already integrated TSDF field.

What this adds, and to what
---------------------------
:mod:`scanner3d.tsdf` fuses depth frames into a scalar field and can hand back
the points where that field crosses zero. Those points are a *bag*: no
topology, no normals, no orientation, nothing an exporter or a renderer can
shade. This module is the missing step - it walks the cubes formed by
neighbouring voxel CENTRES, interpolates the level set along the cube edges it
crosses, and stitches the results into a :class:`scanner3d.mesh.Mesh` whose
vertices are SHARED between neighbouring cubes rather than duplicated per
triangle.

It consumes a field that has already been integrated. It does not register
frames, infer poses, deproject points, read a device or touch a capture, and it
never builds a surface out of the unconnected zero-level samples: those are a
measurement of the same field, taken a different way, and gluing them together
afterwards would be guessing at connectivity that the grid already knows.

Which side is inside
--------------------
:mod:`scanner3d.tsdf` fixes the sign: a voxel IN FRONT of the observed surface -
nearer the camera - is positive, and one BEHIND it is negative. So the negative
region is the material, and this module treats it as the inside of the solid.
Faces are wound so that their normal points towards INCREASING TSDF: out of the
negative region, back towards the camera that observed the surface. That is the
outward normal for a scanned object and it is a property of the field's own
sign, not a convention chosen here.

Why the tables are borrowed and not rewritten
---------------------------------------------
:mod:`scanner3d.mesh` already computes its marching cubes tables from the corner
signs alone, by pairing contour crossings around each of the six cube faces.
That construction is what makes the tables usable here, and the reason is worth
stating: the segments a cube contributes on one of its faces are a function of
that face's FOUR CORNER SIGNS and nothing else, so the cube on the other side of
that face - which sees the same four values - computes the same segment, in the
opposite direction. Two neighbouring cubes therefore cannot disagree about
where the surface crosses the wall between them, in the ambiguous four-crossing
cases as much as in the easy ones. Occupancy counts and signed floats produce
different NUMBERS, but the tables consume only the eight sign bits, so the
topology carries over unchanged and is exercised here exhaustively over every
shared-face pattern rather than assumed.

What does NOT carry over is :func:`scanner3d.mesh.case_index`'s reading of which
side is solid. There, "above the level" is solid. Here, negative is solid, so
:func:`cube_case` NEGATES the corner values before asking that function for the
case. The accelerated walk cannot call it - it decides eight corners of tens of
thousands of cubes at a time and a per-cube Python call would undo the point of
vectorising - so it tests ``value < 0.0`` inline instead. That is a second
statement of one rule, which is exactly the kind of pair that drifts, so
``tests/test_tsdf_surface.py`` enumerates all 256 corner sign patterns and both
exact-zero variants and asserts the two walks return the same cube; the comment
beside the inline test names this as its only justification.

The level set on the grid, exactly
----------------------------------
A stored value of exactly zero is the level set passing through a voxel centre
rather than between two of them. It is handled explicitly, and the policy is
one rule applied by every cube:

* A corner holding exactly zero counts as being IN FRONT of the surface - the
  case bit is set only for a strictly negative value. The test is per corner,
  so every cube sharing that corner reads it the same way and no crack can
  open between them.
* An edge from an exact zero to a negative neighbour is therefore a crossing,
  and its interpolated position lands exactly on the zero corner's centre. A
  vertex sitting exactly on a voxel centre is a vertex that a SECOND cut edge
  through the same centre could also produce, which is how coincident vertices
  and zero-area triangles are made. So the fraction along the edge is clamped
  away from both ends by :data:`scanner3d.mesh.MIN_EDGE_FRACTION`, exactly as
  the occupancy path clamps it, and the surface of a grid-aligned plane comes
  out one part in a million of a voxel off the centres instead of on them.

Evidence, not assumption
------------------------
A cube is eligible only when ALL EIGHT of its corners reach the caller's weight
threshold. Nothing is interpolated across a voxel no frame observed, or one
observed too thinly to trust, so the mesh stops where the evidence stops. That
means an OPEN mesh is the normal result of a scan taken from one side, and this
module never pads the grid, never closes it against the volume boundary and
never invents a back wall - which is precisely what the occupancy path in
:mod:`scanner3d.mesh` does do, and why that module's caveats are not attached to
a result from this one. Whether a particular surface came out closed is
MEASURED, by counting its edges, and reported either way.

Bounded, and refused before it is built
---------------------------------------
Vertices, faces, the working tile and the pure-Python path all have fixed
module ceilings that a caller may lower and can never widen. The count is taken
in a first pass that materialises no geometry at all - a set of grid-edge keys
and an integer - and the whole extraction is refused before a single vertex
tuple exists if it would exceed a ceiling. Nothing partial is ever returned.

numpy is optional
-----------------
As in :mod:`scanner3d.tsdf`, no import of numpy happens at module scope. The
accelerated path vectorises the pass over the CELLS - the part that grows with
the volume - and both paths then run the very same interpreted code to place
vertices and emit faces, so the two cannot produce different geometry from the
same field. There is no GPU backend and no new dependency.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point
from scanner3d.mesh import (
    CORNER_OFFSETS,
    EDGE_AXIS,
    EDGE_CORNERS,
    EDGE_TABLE,
    MIN_EDGE_FRACTION,
    TRIANGLE_TABLE,
    EdgeCensus,
    Mesh,
    case_index,
    edge_census,
)
from scanner3d.tsdf import (
    FIELD_LIMITATIONS,
    TsdfConfig,
    TsdfVolume,
    resolve_backend,
)

#: Every report this module emits stamps this.
SCHEMA_VERSION: Final = 1

#: The algorithm, named once so the report and the code cannot disagree.
EXTRACTION_MODE: Final = (
    "marching cubes over the cubes formed by neighbouring TSDF voxel centres, with vertices "
    "shared by global grid edge and linear interpolation of the stored signed values"
)

#: Which side of the field is the material. Written once, here, because every
#: statement about winding downstream depends on it.
INSIDE_CONVENTION: Final = (
    "a voxel whose stored value is strictly NEGATIVE is behind the observed surface and is "
    "treated as INSIDE the solid; zero and positive values are in front of it and outside. "
    "Faces are wound so that a triangle's normal points towards INCREASING TSDF - out of the "
    "negative region, back towards the camera that observed the surface"
)

#: What happens when the level set lands exactly on the grid. One rule, applied
#: by every cube that shares the corner.
EXACT_LEVEL_POLICY: Final = (
    "a corner holding exactly zero counts as IN FRONT of the surface, because the case bit is "
    f"set only for a strictly negative value; the crossing it implies is then clamped "
    f"{MIN_EDGE_FRACTION!r} of a voxel away from either end of its edge, so a level set lying "
    "on the voxel centres yields a connected patch beside them instead of coincident vertices "
    "and zero-area triangles at them"
)

#: The iso-level the case index is taken at. Zero, and not a parameter: the
#: zero crossing is what a signed-distance field means, and a caller who wanted
#: a different level would be asking for an offset surface, which is a
#: different operation with different evidence.
LEVEL: Final = 0.0

#: The most vertices one extraction may materialise. Each is a three-float
#: Python tuple costing on the order of 150 bytes once its floats are counted,
#: matching :data:`scanner3d.tsdf.MAX_ZERO_SAMPLES` for the same reason it has
#: the value it has: about 150 MiB of result is the point past which a caller
#: wants a streaming writer rather than a bigger ceiling. It is below
#: :data:`scanner3d.mesh.MAX_VERTICES`, so a surface this module returns can
#: always be handed to the exporters. Never widened at run time.
MAX_SURFACE_VERTICES: Final = 1_000_000

#: The most faces one extraction may materialise, on the same reasoning: a face
#: is a three-int tuple, and a triangle mesh has roughly two of them per vertex.
#: Also below :data:`scanner3d.mesh.MAX_FACES`.
MAX_SURFACE_FACES: Final = 2_000_000

#: Cubes per working tile on the accelerated path. Each tile builds about
#: twenty temporaries of this many elements, which fixes the transient cost at
#: roughly 10 MiB whatever the volume holds - a bigger volume costs more tiles,
#: not bigger arrays. Tiles are cut so that every cube appears in exactly one of
#: them, and the cut is taken across BOTH of the plane's axes: cutting only
#: ``j`` would leave every temporary spanning the whole ``k`` span, so a legal
#: long, thin field - which the shape ceiling permits up to millions of cubes
#: along one axis - would blow straight through this number while appearing to
#: respect it. :func:`_tile_plan` is what makes it a real bound, and is tested
#: directly at maximum-volume spans.
SURFACE_CHUNK_CUBES: Final = 65_536

#: The ceiling that binds the pure-Python reference path, matching
#: :data:`scanner3d.tsdf.PYTHON_BACKEND_MAX_VOXELS` because it is the same kind
#: of bound for the same reason: the interpreted path reads all eight corners of
#: every cube in Python, and letting it loose on the 16.7-million-voxel maximum
#: field would look from outside like a hang rather than a slow answer. A
#: wall-clock ceiling, fixed, and never widened at run time.
PYTHON_SURFACE_MAX_CELLS: Final = 64 * 64 * 64


# --- refusals -----------------------------------------------------------------


class TsdfSurfaceNotAllowed(ScannerError):
    """An extraction was requested that this module refuses.

    Raised before any vertex tuple exists, so a refused extraction leaves no
    partial mesh behind it and costs the bounded counting pass rather than the
    memory it is refusing.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class WeightThresholdUnusable(TsdfSurfaceNotAllowed):
    """The minimum weight is not a finite positive number.

    Zero is refused along with the non-finite values: a threshold of zero would
    make every voxel eligible, including the ones no frame ever observed, whose
    stored zero is the absence of evidence rather than the level set. That is
    the one input that turns an unobserved volume into a confident closed
    object.
    """


class SurfaceCeilingExceeded(TsdfSurfaceNotAllowed):
    """The surface would need more vertices or faces than the ceiling allows."""


class SurfaceBackendTooSmall(TsdfSurfaceNotAllowed):
    """The pure-Python path was asked for a field past its own wall-clock ceiling."""


class NoSurfaceFound(TsdfSurfaceNotAllowed):
    """No eligible cube of the field contains a zero crossing.

    Refused rather than answered with an empty mesh, for the reason
    :meth:`scanner3d.mesh.SurfaceAccumulator.surface` gives: downstream, a mesh
    with no faces is indistinguishable from a scan that found no surface, and
    the two want different responses from whoever asked.
    """


# --- the tables, read the way a signed field needs them -----------------------

#: For each of the twelve edges, the corner at its LOW end - the one whose
#: offset along the edge's own axis is zero, and therefore the one with the
#: smaller flat index whatever the grid's shape is. Naming the low end is what
#: lets two cubes that meet on an edge compute the same key for it.
EDGE_LOW_CORNER: Final[tuple[int, ...]] = tuple(
    first if CORNER_OFFSETS[first][EDGE_AXIS[edge]] == 0 else second
    for edge, (first, second) in enumerate(EDGE_CORNERS)
)

#: The corner at the other end of each edge.
EDGE_HIGH_CORNER: Final[tuple[int, ...]] = tuple(
    second if low == first else first
    for low, (first, second) in zip(EDGE_LOW_CORNER, EDGE_CORNERS, strict=True)
)


def cube_case(values: Sequence[float]) -> int:
    """The marching cubes case for one cube's eight stored TSDF values.

    Bit ``c`` is set when corner ``c`` is strictly negative - behind the
    observed surface, inside the solid. The values are negated and handed to
    :func:`scanner3d.mesh.case_index` rather than compared here, so that the
    rule at the level lives in exactly one place: that function sets a bit for
    a value STRICTLY ABOVE the level, and negating turns "strictly above zero"
    into "strictly below zero" without a second implementation of the
    comparison that could drift to ``<=``.

    Negating is exact in binary floating point, and ``-0.0`` is not greater
    than ``0.0``, so a corner holding either zero falls on the outside - the
    exact-level policy this module documents, applied per corner.
    """
    return case_index([-value for value in values], LEVEL)


def _corner_flat_offsets(ny: int, nz: int) -> tuple[int, ...]:
    """Each cube corner's offset from the cube's own low voxel, as a flat index.

    In :class:`scanner3d.tsdf.TsdfVolume`'s layout, which is ``(i * ny + j) *
    nz + k``. Computed per volume rather than baked in, because it is the only
    thing about the corner numbering that depends on the shape.
    """
    return tuple((dx * ny + dy) * nz + dz for dx, dy, dz in CORNER_OFFSETS)


def _edge_key_offsets(corner_offsets: Sequence[int]) -> tuple[int, ...]:
    """The vertex key each of the twelve edges contributes, relative to the cube.

    A vertex is identified by the grid edge it was cut from, and a grid edge by
    its low voxel's flat index and the axis it runs along, as ``flat * 3 +
    axis``. The key is global, so the up-to-four cubes that meet on one edge
    look up one vertex and the mesh is indexed rather than a triangle soup.

    What is returned is the part of that key which does not depend on the cube,
    so a caller adds ``cube_base * 3`` and NOT ``cube_base``. The distinction is
    the whole correctness of the sharing: fold the factor of three into only one
    of the two terms and neighbouring cubes compute different keys for the same
    edge while different edges collide on one key, which gives a mesh with about
    twice the vertices it should have and triangles built from a neighbour's
    coordinates.
    """
    return tuple(corner_offsets[EDGE_LOW_CORNER[edge]] * 3 + EDGE_AXIS[edge] for edge in range(12))


# --- the evidence -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurfaceEvidence:
    """What one extraction examined, kept, refused and measured. Immutable.

    Every count here is taken from the walk that built the mesh, not inferred
    from the mesh afterwards, so "how much of the field was even eligible" and
    "how much of it crossed zero" stay separable - which is the difference
    between a thin scan and a field with nothing in it.
    """

    config: TsdfConfig
    backend: str
    min_weight: float
    cubes_examined: int
    cubes_eligible: int
    cubes_active: int
    vertex_count: int
    face_count: int
    census: EdgeCensus
    bounds: tuple[Point, Point] | None
    max_vertices: int
    max_faces: int
    limitations: tuple[str, ...]

    @property
    def cubes_excluded_insufficient_weight(self) -> int:
        """Cubes skipped because at least one of their eight corners was too light.

        The complement of :attr:`cubes_eligible` within
        :attr:`cubes_examined`, and the number that says how much of the volume
        this surface declined to guess across.
        """
        return self.cubes_examined - self.cubes_eligible

    @property
    def cubes_eligible_without_crossing(self) -> int:
        """Fully observed cubes that the level set simply does not pass through."""
        return self.cubes_eligible - self.cubes_active

    def as_dict(self) -> dict[str, object]:
        extent = self.bounds
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": EXTRACTION_MODE,
            "inside_convention": INSIDE_CONVENTION,
            "exact_level_policy": EXACT_LEVEL_POLICY,
            "level": LEVEL,
            "backend": self.backend,
            "min_weight": self.min_weight,
            "cubes": {
                "examined": self.cubes_examined,
                "eligible": self.cubes_eligible,
                "excluded_insufficient_weight": self.cubes_excluded_insufficient_weight,
                "eligible_without_crossing": self.cubes_eligible_without_crossing,
                "active": self.cubes_active,
            },
            "mesh": {
                "vertex_count": self.vertex_count,
                "face_count": self.face_count,
                "vertex_ceiling": self.max_vertices,
                "face_ceiling": self.max_faces,
                "module_vertex_ceiling": MAX_SURFACE_VERTICES,
                "module_face_ceiling": MAX_SURFACE_FACES,
                "edges": self.census.as_dict(),
                "bounds": (
                    None if extent is None else {"min": list(extent[0]), "max": list(extent[1])}
                ),
            },
            "grid": {
                "voxel_size": self.config.voxel_size,
                "truncation": self.config.truncation,
                "shape": list(self.config.shape),
                "cells": self.config.cell_count,
            },
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class TsdfSurface:
    """A connected surface and the record of what produced it.

    The geometry is a plain :class:`scanner3d.mesh.Mesh` because that type
    already carries exactly what this surface is - shared vertices, indexed
    faces, and the caveats travelling with them - and a second mesh type would
    be a second thing for the exporters to learn. The TSDF-specific counts that
    a generic mesh has no field for live in :attr:`evidence` beside it.
    """

    mesh: Mesh
    evidence: SurfaceEvidence

    def as_dict(self) -> dict[str, object]:
        return {"mesh": self.mesh.as_dict(), "extraction": self.evidence.as_dict()}


# --- walking the cubes --------------------------------------------------------


def _tile_plan(span_j: int, span_k: int) -> tuple[int, int]:
    """``(j cubes, k cubes)`` per accelerated tile, for one layer's cube plane.

    Two properties, and the second is why the ``k`` step exists at all:

    * ``j step * k step`` never exceeds :data:`SURFACE_CHUNK_CUBES`, whatever
      the shape. Cutting only ``j`` bottoms its step out at one and then stops
      bounding anything, because a single ``j`` row of a long field is still
      ``span_k`` cubes wide - and the volume ceiling permits a slab millions of
      cubes long on one axis, whose "one row" temporaries would be tens of
      megabytes each rather than the fixed cost this ceiling promises.
    * The tiles, walked ``j`` outer and ``k`` inner, still visit the cubes in
      the volume's own ``i, j, k`` order. That holds because the ``j`` step is
      allowed above one ONLY when ``k`` is a single tile: a multi-row tile that
      also split ``k`` would emit ``(j, k0..k1)`` for every row before reaching
      ``k1``, which is a different order from the reference walk's and would
      change the vertex numbering of every mesh this module returns.

    Normal fields are unaffected: while ``span_k`` fits in the ceiling the plan
    is the same full-width ``SURFACE_CHUNK_CUBES // span_k`` rows it has always
    been, so the measured cost of an ordinary volume does not become a stream
    of tiny tiles.
    """
    step_k = min(span_k, SURFACE_CHUNK_CUBES)
    step_j = max(1, SURFACE_CHUNK_CUBES // step_k)
    return step_j, step_k


@dataclass(slots=True)
class _CubeWalk:
    """Yields every ACTIVE cube of one field, in one fixed order, twice if asked.

    Active means: all eight corners reach the weight threshold, and the case
    their signs imply cuts at least one edge. The order is ``i`` ascending,
    then ``j``, then ``k`` - the volume's own flat order - and both backends
    produce it identically, which is what lets the counting pass and the
    building pass agree on the same cubes without one of them being trusted.

    The counters are reset at the start of each walk rather than accumulated
    across walks, so two passes over the same field leave the same numbers
    behind rather than double them.

    :attr:`peak_tile_cubes` is the largest number of cubes any one vectorised
    tile covered, recorded so that :data:`SURFACE_CHUNK_CUBES` is an audited
    bound rather than a claim in a comment. It is deliberately NOT part of
    :class:`SurfaceEvidence`: it describes the working set of the walk that ran,
    not the surface, and the evidence a caller reads has to be identical on both
    backends for the same field.
    """

    values: memoryview
    weights: memoryview
    shape: tuple[int, int, int]
    min_weight: float
    backend: str
    examined: int = 0
    eligible: int = 0
    peak_tile_cubes: int = 0

    def cubes(self) -> Iterator[tuple[int, int, list[float]]]:
        """``(low voxel flat index, case, eight corner values)`` per active cube."""
        self.examined = 0
        self.eligible = 0
        self.peak_tile_cubes = 0
        nx, ny, nz = self.shape
        if nx < 2 or ny < 2 or nz < 2:
            # No two voxel centres face each other along some axis, so there is
            # no cube to march at all. Not an error here: the refusal for "this
            # field yielded no surface" belongs to the caller, which can say
            # why in one message rather than two.
            return
        if self.backend == "numpy":
            yield from self._cubes_numpy()
        else:
            yield from self._cubes_python()

    def _cubes_python(self) -> Iterator[tuple[int, int, list[float]]]:
        """The reference walk: every cube, eight corners, in interpreted arithmetic."""
        nx, ny, nz = self.shape
        # One cube at a time, so its working set is one cube. Recorded rather
        # than left at zero so the same invariant can be asserted of either
        # walk without the assertion quietly passing on an unset counter.
        self.peak_tile_cubes = 1
        values = self.values
        weights = self.weights
        offsets = _corner_flat_offsets(ny, nz)
        threshold = self.min_weight
        for i in range(nx - 1):
            for j in range(ny - 1):
                row = (i * ny + j) * nz
                for k in range(nz - 1):
                    base = row + k
                    self.examined += 1
                    corners: list[float] = []
                    for offset in offsets:
                        flat = base + offset
                        if weights[flat] < threshold:
                            break
                        corners.append(values[flat])
                    else:
                        self.eligible += 1
                        case = cube_case(corners)
                        if EDGE_TABLE[case]:
                            yield base, case, corners

    def _cubes_numpy(self) -> Iterator[tuple[int, int, list[float]]]:
        """The same walk, with the pass over the CELLS vectorised.

        Only the selection is accelerated. What comes out is the same tuples in
        the same order, and the caller then runs the same interpreted placement
        code on them either way - so a difference between the backends could
        only be a difference in which cubes are active, which is exactly what
        the parametrised agreement tests measure.

        Tiles are cut across BOTH axes of one ``i`` layer's cube plane, by
        :func:`_tile_plan`, so that no temporary exceeds
        :data:`SURFACE_CHUNK_CUBES` cubes even when the volume is a thin slab
        with an enormous cross-section OR a long, narrow one whose ``k`` span
        alone is larger than the whole ceiling.
        """
        import numpy as np

        nx, ny, nz = self.shape
        values = np.frombuffer(self.values, dtype=np.float32).reshape(nx, ny, nz)
        weights = np.frombuffer(self.weights, dtype=np.float32).reshape(nx, ny, nz)
        edge_table = np.array(EDGE_TABLE, dtype=np.int64)
        span_j, span_k = ny - 1, nz - 1
        step_j, step_k = _tile_plan(span_j, span_k)
        for i in range(nx - 1):
            for start in range(0, span_j, step_j):
                stop = min(start + step_j, span_j)
                block = stop - start
                for first in range(0, span_k, step_k):
                    last = min(first + step_k, span_k)
                    width = last - first
                    self.peak_tile_cubes = max(self.peak_tile_cubes, block * width)
                    self.examined += block * width
                    corner_values = [
                        values[i + dx, start + dy : stop + dy, first + dz : last + dz]
                        for dx, dy, dz in CORNER_OFFSETS
                    ]
                    heavy = [
                        weights[i + dx, start + dy : stop + dy, first + dz : last + dz]
                        >= self.min_weight
                        for dx, dy, dz in CORNER_OFFSETS
                    ]
                    eligible = heavy[0]
                    for corner_heavy in heavy[1:]:
                        eligible = eligible & corner_heavy
                    self.eligible += int(np.count_nonzero(eligible))
                    case = np.zeros((block, width), dtype=np.int64)
                    for corner, corner_value in enumerate(corner_values):
                        # The same rule :func:`cube_case` states, written a
                        # second time because it has to be vectorised: a bit is
                        # set for a STRICTLY negative value, which puts both
                        # zeros on the front side. ``value < 0.0`` is false for
                        # ``-0.0`` exactly as ``-value > 0.0`` is, so the two
                        # agree on the exact zeros as well - and a test
                        # enumerates all 256 patterns rather than leaving that
                        # to this comment.
                        case += np.where(corner_value < 0.0, 1 << corner, 0)
                    active = eligible & (edge_table[case] != 0)
                    if not bool(active.any()):
                        continue
                    rows, columns = np.nonzero(active)
                    bases = ((i * ny + (start + rows)) * nz + (first + columns)).astype(np.int64)
                    stacked = np.stack([value[active] for value in corner_values], axis=1)
                    yield from zip(
                        bases.tolist(),
                        case[active].tolist(),
                        stacked.astype(np.float64).tolist(),
                        strict=True,
                    )


# --- placing the vertices -----------------------------------------------------


def _cut_point(
    base: int,
    edge: int,
    corners: Sequence[float],
    corner_offsets: Sequence[int],
    shape: tuple[int, int, int],
    origin: Point,
    voxel: float,
) -> Point:
    """Where the zero level crosses one cube edge, linearly between two centres.

    Between voxel CENTRES, not cube corners: a voxel's stored value is a
    statement about the middle of the cell, and interpolating as though it
    described a corner would put the whole surface half a voxel out.

    The fraction is measured from the edge's low end and cannot divide by zero:
    a cut edge has one strictly negative corner and one that is not, so the two
    values differ. It is clamped away from both ends, which only ever bites
    when a corner holds exactly zero - see :data:`EXACT_LEVEL_POLICY` for why
    landing exactly on a centre is the case worth spending a clamp on.
    """
    axis = EDGE_AXIS[edge]
    low_value = corners[EDGE_LOW_CORNER[edge]]
    high_value = corners[EDGE_HIGH_CORNER[edge]]
    fraction = low_value / (low_value - high_value)
    fraction = min(max(fraction, MIN_EDGE_FRACTION), 1.0 - MIN_EDGE_FRACTION)
    _, ny, nz = shape
    flat = base + corner_offsets[EDGE_LOW_CORNER[edge]]
    i, remainder = divmod(flat, ny * nz)
    j, k = divmod(remainder, nz)
    position = [origin[0] + i * voxel, origin[1] + j * voxel, origin[2] + k * voxel]
    position[axis] += fraction * voxel
    return (position[0], position[1], position[2])


def _count(
    walk: _CubeWalk, key_offsets: Sequence[int], *, max_vertices: int, max_faces: int
) -> tuple[int, int, int]:
    """``(active cubes, vertices, faces)`` the walk would produce. Builds no geometry.

    The first of the two passes. It materialises one integer per distinct grid
    edge and nothing else - no coordinates, no tuples, no mesh - so the cost of
    discovering that a field is too big is a set of keys bounded by the
    ceiling, rather than the several hundred megabytes of Python objects the
    ceiling exists to refuse.

    The refusal happens the moment a ceiling is passed rather than at the end,
    which is why the message says "more than" and not an exact figure: counting
    the rest would mean growing the very structure being refused.
    """
    seen: set[int] = set()
    faces = 0
    active = 0
    for base, case, _corners in walk.cubes():
        active += 1
        cut = EDGE_TABLE[case]
        key_base = base * 3
        for edge in range(12):
            if cut >> edge & 1:
                seen.add(key_base + key_offsets[edge])
        faces += len(TRIANGLE_TABLE[case])
        if len(seen) > max_vertices:
            raise SurfaceCeilingExceeded(
                f"this field crosses zero on more than {max_vertices} distinct grid edges at "
                f"weight threshold {walk.min_weight!r}, so the surface would need more than "
                "that many vertices; nothing was materialised. Coarsen the volume, narrow the "
                "bounds, or raise the threshold rather than the ceiling - the module ceiling "
                f"is {MAX_SURFACE_VERTICES} and a caller may only lower it"
            )
        if faces > max_faces:
            raise SurfaceCeilingExceeded(
                f"this field's surface needs more than {max_faces} faces at weight threshold "
                f"{walk.min_weight!r}; nothing was materialised. Coarsen the volume, narrow "
                "the bounds, or raise the threshold rather than the ceiling - the module "
                f"ceiling is {MAX_SURFACE_FACES} and a caller may only lower it"
            )
    return active, len(seen), faces


def _build(
    walk: _CubeWalk,
    key_offsets: Sequence[int],
    corner_offsets: Sequence[int],
    config: TsdfConfig,
) -> tuple[tuple[Point, ...], tuple[tuple[int, int, int], ...]]:
    """The second pass: the same cubes, now placing vertices and emitting faces.

    A vertex is created the first time any cube asks for its grid edge and
    looked up by every cube afterwards, so the four cubes that meet on an edge
    index one vertex. Vertex order is first-encounter order along the walk,
    which is fixed by the walk and therefore by the field alone.
    """
    voxel = config.voxel_size
    shape = config.shape
    origin: Point = (
        config.bounds.min_x + 0.5 * voxel,
        config.bounds.min_y + 0.5 * voxel,
        config.bounds.min_z + 0.5 * voxel,
    )
    vertices: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    by_edge: dict[int, int] = {}
    for base, case, corners in walk.cubes():
        cut = EDGE_TABLE[case]
        key_base = base * 3
        local: dict[int, int] = {}
        for edge in range(12):
            if not cut >> edge & 1:
                continue
            key = key_base + key_offsets[edge]
            index = by_edge.get(key)
            if index is None:
                index = len(vertices)
                by_edge[key] = index
                vertices.append(
                    _cut_point(base, edge, corners, corner_offsets, shape, origin, voxel)
                )
            local[edge] = index
        faces.extend((local[a], local[b], local[c]) for a, b, c in TRIANGLE_TABLE[case])
    return tuple(vertices), tuple(faces)


# --- limitations --------------------------------------------------------------


def _closure_note(census: EdgeCensus, config: TsdfConfig) -> str:
    """What counting this surface's edges actually found, said either way."""
    if census.nonmanifold_edge_count:
        return (
            f"measured NON-MANIFOLD: {census.nonmanifold_edge_count} of {census.edge_count} "
            "edges are used by three or more faces. This surface should not be treated as a "
            "solid and should not be exported as one"
        )
    if census.closed:
        return (
            f"measured CLOSED: all {census.edge_count} edges are used by exactly two faces and "
            f"{census.reversed_edge_count} of them are traversed once each way. Closure here is "
            "evidence that the observed field genuinely surrounded the surface, because "
            "nothing pads the grid or closes it against the volume boundary; it is still not "
            "evidence that the object's dimensions are right"
        )
    return (
        f"measured OPEN: of {census.edge_count} edges, {census.boundary_edge_count} are used by "
        "a single face. Those are the rim where the evidence ends - a cube next to them had a "
        f"corner below the weight threshold, or fell outside the {config.shape[0]}x"
        f"{config.shape[1]}x{config.shape[2]} volume. No back wall was invented to close them, "
        "so this is an open surface and not a solid"
    )


def _limitations(
    *,
    config: TsdfConfig,
    min_weight: float,
    evidence_counts: tuple[int, int, int],
    census: EdgeCensus,
) -> tuple[str, ...]:
    """The caveats that travel with the geometry, measured from this run.

    The occupancy caveats in :data:`scanner3d.mesh.INHERENT_LIMITATIONS` are
    deliberately absent: they describe a surface standing outside a union of
    solid cells, marched over a padded grid that always closes. None of that is
    true here, and attaching them would tell a reader the opposite of what this
    module did.
    """
    examined, eligible, active = evidence_counts
    excluded = examined - eligible
    return (
        INSIDE_CONVENTION,
        EXACT_LEVEL_POLICY,
        (
            f"extracted at weight threshold {min_weight!r}: of {examined} cubes formed by "
            f"neighbouring voxel centres, {eligible} had all eight corners at or above it and "
            f"{excluded} did not and were skipped entirely, leaving {active} that the level "
            "set actually passes through. Nothing was interpolated across a voxel no frame "
            "observed, so this surface STOPS where the evidence stops rather than guessing "
            "across the gap"
        ),
        _closure_note(census, config),
        (
            f"resolution: the cubes are {config.voxel_size!r} apart in the field's own "
            "unestablished units, so no feature narrower than about one of them survives and "
            "every vertex lies on a straight line between two voxel centres. The values it "
            f"interpolates are truncated at {config.truncation!r}, and a value sitting at that "
            "clamp interpolates as though the surface really were that far away, which biases "
            "a crossing towards the clamped side"
        ),
        (
            "the surface is a level set of a fused field, not a fit to the individual samples: "
            "it passes through the weighted mean of what the frames observed, and where the "
            "frames disagreed it reports the mean rather than any of them"
        ),
        *FIELD_LIMITATIONS,
    )


# --- the extraction -----------------------------------------------------------


def extract_surface(
    volume: TsdfVolume,
    *,
    min_weight: float = 1.0,
    max_vertices: int = MAX_SURFACE_VERTICES,
    max_faces: int = MAX_SURFACE_FACES,
    backend: str | None = None,
) -> TsdfSurface:
    """Extract the connected zero-level surface of an integrated TSDF field.

    ``min_weight`` is the evidence gate, and it applies to ALL EIGHT corners of
    a cube rather than to the two ends of an edge. A cube with one unobserved
    corner is skipped whole, because triangulating it would mean deciding what
    the level set does inside a cell where part of the field is not a
    measurement. It must be finite and positive; see
    :class:`WeightThresholdUnusable` for why zero is not allowed.

    ``max_vertices`` and ``max_faces`` may be *lowered* by a caller and never
    raised, like every other ceiling in this repository. Both are checked
    against a counting pass that materialises no geometry, so an over-ceiling
    field is refused before the first vertex exists and never leaves a partial
    mesh behind.

    ``backend`` defaults to the volume's own. Naming one explicitly runs the
    extraction on that backend regardless of how the field was integrated,
    which is how the two paths are proved to produce byte-identical geometry
    from one field; ``"auto"``, ``"numpy"`` and ``"python"`` mean exactly what
    :func:`scanner3d.tsdf.resolve_backend` says they mean.

    Raises :class:`NoSurfaceFound` when no eligible cube contains a crossing,
    rather than returning a mesh with no faces.
    """
    if not math.isfinite(min_weight) or min_weight <= 0.0:
        raise WeightThresholdUnusable(
            f"min_weight {min_weight!r} must be finite and positive; a threshold of zero would "
            "make voxels no frame ever observed eligible, and their stored zero is the absence "
            "of evidence rather than the level set"
        )
    if not 0 < max_vertices <= MAX_SURFACE_VERTICES:
        raise SurfaceCeilingExceeded(
            f"max_vertices {max_vertices} is outside (0, {MAX_SURFACE_VERTICES}]; the module "
            "ceiling is never widened at run time"
        )
    if not 0 < max_faces <= MAX_SURFACE_FACES:
        raise SurfaceCeilingExceeded(
            f"max_faces {max_faces} is outside (0, {MAX_SURFACE_FACES}]; the module ceiling is "
            "never widened at run time"
        )
    resolved = volume.backend if backend is None else resolve_backend(backend)
    config = volume.config
    cells = config.cell_count
    if resolved == "python" and cells > PYTHON_SURFACE_MAX_CELLS:
        raise SurfaceBackendTooSmall(
            f"a {cells}-voxel field is past the {PYTHON_SURFACE_MAX_CELLS} ceiling that binds "
            "the pure-Python extraction path, which reads all eight corners of every cube in "
            "interpreted arithmetic; nothing was walked. This ceiling is a wall-clock bound "
            "rather than a memory one and is never widened at run time - install numpy, or "
            "reduce the volume"
        )

    _, ny, nz = config.shape
    corner_offsets = _corner_flat_offsets(ny, nz)
    key_offsets = _edge_key_offsets(corner_offsets)
    walk = _CubeWalk(
        values=volume.values,
        weights=volume.weights,
        shape=config.shape,
        min_weight=min_weight,
        backend=resolved,
    )

    active, counted_vertices, counted_faces = _count(
        walk, key_offsets, max_vertices=max_vertices, max_faces=max_faces
    )
    examined, eligible = walk.examined, walk.eligible
    if active == 0:
        raise NoSurfaceFound(
            f"no eligible cube of this {config.shape[0]}x{config.shape[1]}x{config.shape[2]} "
            f"field contains a zero crossing at weight threshold {min_weight!r}: {examined} "
            f"cubes were examined and {eligible} had all eight corners observed that heavily. "
            "Refused rather than answered with a mesh of no faces, which downstream cannot be "
            "told apart from a scan that found nothing"
        )

    vertices, faces = _build(walk, key_offsets, corner_offsets, config)
    if (len(vertices), len(faces)) != (counted_vertices, counted_faces):  # pragma: no cover
        # An invariant, not an expected path: both passes walk the same cubes
        # through the same code, so they cannot disagree unless the walk has
        # stopped being a function of the field. Checked rather than trusted,
        # because the thing it would catch is a mesh that silently exceeded the
        # ceiling the first pass had already cleared.
        raise TsdfSurfaceNotAllowed(
            f"the counting pass found {counted_vertices} vertices and {counted_faces} faces "
            f"and the building pass produced {len(vertices)} and {len(faces)}; the walk is not "
            "reproducible and no surface is returned"
        )
    census = edge_census(faces)
    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        limitations=_limitations(
            config=config,
            min_weight=min_weight,
            evidence_counts=(examined, eligible, active),
            census=census,
        ),
    )
    return TsdfSurface(
        mesh=mesh,
        evidence=SurfaceEvidence(
            config=config,
            backend=resolved,
            min_weight=min_weight,
            cubes_examined=examined,
            cubes_eligible=eligible,
            cubes_active=active,
            vertex_count=len(vertices),
            face_count=len(faces),
            census=census,
            bounds=mesh.bounds,
            max_vertices=max_vertices,
            max_faces=max_faces,
            limitations=mesh.limitations,
        ),
    )
