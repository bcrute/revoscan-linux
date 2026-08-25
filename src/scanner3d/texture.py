"""Vertex colour: transferring a coloured point cloud's colour onto a mesh's vertices.

Why a transfer, not a carry-along
----------------------------------
:mod:`scanner3d.rgb_project` puts a colour on a depth point - a sample that was
actually observed. :mod:`scanner3d.mesh` reconstructs a surface by marching
cubes over a fused occupancy grid, and every vertex it produces sits on a grid
*edge*, at whatever fraction of a voxel the iso-level crossing landed on - see
that module's own docstring. A mesh vertex is therefore a NEW point that no
sample sits on exactly; nothing about it was ever projected into the RGB
image, and there is no vendor colour to simply look up or carry along. The
only thing this module can do is ask "what colour did nearby REAL samples
carry", and say so plainly rather than implying a vertex's colour was
observed.

Three decisions, made deliberately
-----------------------------------
**How many neighbours, and what an empty neighbourhood means.** A vertex
draws on the ``neighbours`` closest coloured points that fall within
``radius`` - see :func:`colour_vertices`. A vertex with none is left
UNCOLOURED (``None``), never painted black or any other default. A black
patch on a rendered model is visually indistinguishable from a genuinely dark
surface, and silently choosing one over the other is exactly the kind of
plausible-looking wrong answer this repository refuses elsewhere (see
:mod:`scanner3d.rgb_project`'s own rejection reasons). :class:`VertexColours`
reports the uncoloured vertices explicitly, by index and by count, so a
caller cannot mistake "nothing nearby" for "measured black".

**No distance weighting.** A closer sample is not necessarily a MORE
CORRECT one: a vertex sitting near a depth discontinuity can have a sample
from the wrong surface (behind it, or in front of it) land closer than every
sample from the surface it is actually on - the two surfaces are, by
definition, close together right at the edge that separates them. Weighting
by distance would hand that wrong-surface sample MORE influence for being
close, which is the opposite of what is wanted. Every accepted candidate
within the radius is weighted equally instead, and outlier rejection is left
entirely to the combining rule below.

**Per-channel median, not mean.** A mean lets one wildly wrong sample - the
wrong-surface case above - drag the result partway to itself no matter how
many correct samples surround it; a single outlier among ``neighbours``
candidates always moves a mean, by an amount that shrinks only as
``neighbours`` grows. A median does not move at all until wrong values are a
MAJORITY of the candidates, which is the outlier-rejection property this
module needs and a mean does not have. It is taken per channel (red, green
and blue independently), which is the cheap, standard robust-statistics
choice; the one thing worth stating plainly is that a per-channel median can
therefore land on a colour no single candidate point actually carried - each
channel is the median of its OWN column, not of the closest whole triple.
That is accepted here as the price of the outlier rejection it buys.

What this does not do
----------------------
It does not build or accelerate a spatial index of its own.
:func:`colour_vertices` calls exclusively into
:class:`scanner3d.compute.ComputeBackend`, the same backend seam
:mod:`scanner3d.registration` uses, so "python" and a KD-tree-backed "scipy"
give the identical answer through the identical code path - never a second,
merely-equivalent implementation that could quietly drift from it.

It does not decide what a caller should do with an uncoloured vertex. Filling
one in with a neighbour's colour, a fixed sentinel, or refusing to export the
mesh at all are three different, defensible policies, and this module is not
in a position to know which one the caller wants - it only measures and
reports which vertices have no colour.

It does not attach the result to a :class:`scanner3d.mesh.Mesh`. That is a
caller step: build a :class:`VertexColours`, decide what its uncoloured
entries become, and pass a fully-populated ``tuple[int, int, int]`` per
vertex to :class:`scanner3d.mesh.Mesh`'s own ``vertex_colours`` field.

Cost
----
One call to :meth:`~scanner3d.compute.ComputeBackend.k_nearest_between` for
every vertex at once - the backend builds its own index (a KD-tree, for
"scipy") over the coloured cloud exactly ONCE and queries it with every
vertex, rather than once per vertex. An earlier version of this module
expressed the same question as a same-set search (:meth:`k_nearest_neighbors`
on a list built from one vertex plus every coloured point, one call per
vertex) specifically to keep a mesh vertex's own near neighbours - OTHER mesh
vertices - from crowding out the coloured points actually wanted; that
correctness property is unchanged, but it no longer costs a rebuilt index per
vertex to get it - :meth:`~scanner3d.compute.ComputeBackend.k_nearest_between`
is a genuine cross-set search, never a same-set one with a vertex spliced in.
Measured at 3000 vertices against 30000 coloured points - smaller than the
roughly 16000x109000 scale one real capture produces, but already enough to
show the shape of the difference: with the "scipy" backend the OLD
per-vertex shape (one rebuilt KD-tree per vertex) extrapolates to about 250s
for all 3000 vertices, and this one-call shape measures 0.067s for the same
3000 - about 3700x. With "numpy" (no tree; a GEMM per vertex over the whole
target cloud) the old shape is worse still, and the new one measures 1.3s.
Absolute figures are noisy - taken on a machine shared with other concurrent
work, not an isolated benchmark - but the SHAPE of the difference (one index
build against thousands of them) holds regardless of what else was running;
see ``tests/test_texture.py`` for how it was produced.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.compute import PYTHON_BACKEND, ComputeBackend, select_backend
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, distance, ensure_finite

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: One colour channel's valid range - what a PLY ``uchar`` property, and every
#: 8-bit RGB image this repository decodes, can actually hold.
_CHANNEL_RANGE: Final = range(256)

#: How many nearby coloured points a vertex draws on, capped at this even
#: when more fall within the radius. Large enough that the per-channel
#: median's outlier rejection means something - with 8 candidates, up to 3
#: (37.5%) can be wrong without moving the median off the correct cluster -
#: and small enough that one vertex's colour is not built from samples so far
#: away ``radius`` alone should have excluded them; see the module docstring.
DEFAULT_NEIGHBOURS: Final = 8


# --- refusals -------------------------------------------------------------------


class TextureNotAllowed(ScannerError):
    """A vertex-colouring input was refused before it could produce a
    silently wrong answer downstream.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class RadiusUnusable(TextureNotAllowed):
    """The search radius is zero, negative, or not finite.

    A non-positive radius could never admit a candidate point, which is
    indistinguishable, from the caller's side, from every vertex being
    reported uncoloured - refused instead, so that report is never confused
    with a real measurement of "nothing was nearby".
    """


class NeighbourCountUnusable(TextureNotAllowed):
    """The requested neighbour count is less than one, so no candidate could
    ever be kept.
    """


class NoVerticesOffered(TextureNotAllowed):
    """No vertices were offered, so there is nothing to colour."""


class NoColouredPointsOffered(TextureNotAllowed):
    """No coloured points were offered, so there is nothing to colour FROM.

    Refused rather than answered with every vertex reported uncoloured: that
    result would look identical to a real search that genuinely found nothing
    nearby, and the two have different causes a caller needs told apart.
    """


class ColourChannelOutOfRange(TextureNotAllowed):
    """A coloured point's channel is outside ``[0, 255]``.

    Checked at the boundary rather than left to fail inside the combining
    step or, worse, inside a PLY ``uchar`` encoder several calls downstream,
    where the error message would name a struct-packing failure instead of
    the actual bad input.
    """


# --- the result -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VertexColours:
    """One RGB colour per vertex, in vertex order - or ``None`` where none
    could be assigned.

    ``None`` entries are not a failure of this module; they are its answer
    for a vertex with no coloured point within ``radius``, and are reported
    rather than defaulted - see the module docstring.
    """

    colours: tuple[tuple[int, int, int] | None, ...]
    radius: float
    neighbours: int
    backend: str

    @property
    def vertex_count(self) -> int:
        return len(self.colours)

    @property
    def uncoloured_indices(self) -> tuple[int, ...]:
        """Which vertices, by index, got no colour - the report this module exists to make."""
        return tuple(index for index, colour in enumerate(self.colours) if colour is None)

    @property
    def coloured_count(self) -> int:
        return self.vertex_count - len(self.uncoloured_indices)

    @property
    def uncoloured_count(self) -> int:
        return len(self.uncoloured_indices)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "vertex_count": self.vertex_count,
            "coloured_count": self.coloured_count,
            "uncoloured_count": self.uncoloured_count,
            "uncoloured_indices": list(self.uncoloured_indices),
            "radius": self.radius,
            "neighbours": self.neighbours,
            "backend": self.backend,
            "colours": [None if colour is None else list(colour) for colour in self.colours],
        }


# --- combining --------------------------------------------------------------------


def _combine(candidates: Sequence[tuple[int, int, int]]) -> tuple[int, int, int]:
    """The per-channel median of ``candidates`` - see the module docstring for why
    median, per channel, unweighted by distance.

    ``statistics.median`` on an even count averages the two middle values,
    which can produce a channel value :func:`_combine`'s own caller never
    offered it; that is the ordinary, accepted behaviour of a median filter
    and is not treated specially here.
    """
    channels = tuple(
        round(statistics.median(candidate[axis] for candidate in candidates)) for axis in range(3)
    )
    return (channels[0], channels[1], channels[2])


# --- backend resolution -------------------------------------------------------------


def _resolve_backend(backend: str | ComputeBackend) -> ComputeBackend:
    """A live backend, whether ``backend`` names one or already is one.

    Unlike :mod:`scanner3d.registration`'s own resolver, there is no
    "python means the original unaccelerated code path" special case to
    preserve here - this module is new, has no pre-backend implementation to
    stay byte-for-byte with, and its correctness claim (equivalence between
    "python" and "scipy") is only meaningful if BOTH names run through the
    identical :class:`~scanner3d.compute.ComputeBackend` call, never a
    "python" that quietly takes a different path.
    """
    if isinstance(backend, str):
        return select_backend(backend).backend
    return backend


# --- colouring ----------------------------------------------------------------------


def colour_vertices(
    vertices: Sequence[Point],
    coloured_points: Sequence[tuple[Point, tuple[int, int, int]]],
    *,
    radius: float,
    neighbours: int = DEFAULT_NEIGHBOURS,
    backend: str | ComputeBackend = PYTHON_BACKEND,
) -> VertexColours:
    """Assign each vertex the per-channel median colour of its nearby coloured points.

    The ``neighbours`` closest points in ``coloured_points`` to EVERY vertex
    are found in one call to
    :meth:`~scanner3d.compute.ComputeBackend.k_nearest_between` - a genuine
    cross-set search, so a vertex's candidates are always coloured points,
    never crowded out by another mesh vertex the way a same-set search over a
    combined list would be; see the module docstring's "Cost" section for the
    measured difference this makes at real scale. Of those candidates, only
    the ones strictly closer than ``radius`` are kept - the same strict
    inequality :mod:`scanner3d.compute`'s own gate uses - and combined by
    :func:`_combine`. A vertex with no candidate inside the radius is reported
    :attr:`VertexColours.colours`-entry ``None``, counted in
    :attr:`VertexColours.uncoloured_count` and named in
    :attr:`VertexColours.uncoloured_indices`.

    Refuses (:class:`RadiusUnusable`) a non-positive or non-finite radius,
    (:class:`NeighbourCountUnusable`) fewer than one neighbour,
    (:class:`NoVerticesOffered` / :class:`NoColouredPointsOffered`) an empty
    ``vertices`` or ``coloured_points``, and (:class:`ColourChannelOutOfRange`)
    a channel outside ``[0, 255]`` - all before a single backend call is
    made.

    ``backend`` is resolved through :func:`scanner3d.compute.select_backend`
    exactly as :func:`scanner3d.registration.icp` resolves its own - a bare
    string name, or a live
    :class:`~scanner3d.compute.ComputeBackend` instance. The name actually
    used is recorded in :attr:`VertexColours.backend`.
    """
    ensure_finite((radius,), "radius")
    if radius <= 0.0:
        raise RadiusUnusable(f"radius {radius!r} is not positive; no candidate could ever be found")
    if neighbours < 1:
        raise NeighbourCountUnusable(f"neighbours {neighbours} must be at least 1")
    if not vertices:
        raise NoVerticesOffered("no vertices were offered, so there is nothing to colour")
    if not coloured_points:
        raise NoColouredPointsOffered(
            "no coloured points were offered, so there is nothing to colour vertices FROM; "
            "refused rather than reporting every vertex uncoloured, which would look identical "
            "to a real search that found nothing nearby"
        )

    for index, vertex in enumerate(vertices):
        ensure_finite(vertex, f"vertex {index}")
    colour_points: list[Point] = []
    point_colours: list[tuple[int, int, int]] = []
    for index, (point, colour) in enumerate(coloured_points):
        ensure_finite(point, f"coloured point {index}")
        for channel, value in zip(("red", "green", "blue"), colour, strict=True):
            if value not in _CHANNEL_RANGE:
                raise ColourChannelOutOfRange(
                    f"coloured point {index}'s {channel} channel is {value!r}, outside [0, 255]"
                )
        colour_points.append(point)
        point_colours.append(colour)

    resolved = _resolve_backend(backend)

    # ONE cross-set search for every vertex at once. The earlier shape of this
    # expressed the same question as a same-set search with the vertex spliced
    # in at index 0, which forced a spatial index to be rebuilt per vertex -
    # measured at about 20 ms each, over six minutes for a real mesh, with the
    # tree over the same coloured points thrown away and rebuilt every time.
    neighbour_rows = resolved.k_nearest_between(list(vertices), colour_points, k=neighbours)

    results: list[tuple[int, int, int] | None] = []
    for vertex, row in zip(vertices, neighbour_rows, strict=True):
        # The radius filter stays HERE rather than moving into the search: the
        # backends return the k nearest whatever their distance, and "near
        # enough to describe this vertex's colour" is a judgement this module
        # makes, not one the index makes.
        candidates = [
            point_colours[index] for index in row if distance(vertex, colour_points[index]) < radius
        ]
        results.append(_combine(candidates) if candidates else None)

    return VertexColours(
        colours=tuple(results), radius=radius, neighbours=neighbours, backend=resolved.name
    )
