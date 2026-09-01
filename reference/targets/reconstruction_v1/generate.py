#!/usr/bin/env python3
"""Generate the scanner reconstruction target and its independent CAD evidence.

The generated STL is the printable object.  The PLY is a deterministic,
area-weighted sampling of every externally scannable triangle except the bed
face.  The JSON binds both artifacts to the dimensions, regions and landmarks
that give a later physical replay its truth.

No CAD package or third-party module is required.  The solid is a 2.5-D height
field: every point from z=0 through the named top surface is material.  That
construction gives the target a flat print-bed face, no bridges or floating
islands, and a boundary whose watertightness can be checked exactly.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import struct
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

Point = tuple[float, float, float]

TARGET_ID = "reconstruction-target-v1"
UNITS = "millimetres"

MIN_X = -46.0
MAX_X = 46.0
MIN_Y = -41.0
MAX_Y = 41.0
BOTTOM_Z = 0.0
BASE_Z = 6.0
MAX_Z = 56.0

# 96 puts the maximum sampled deviation of the faceted STL from the declared
# smooth formula below 0.007 mm, comfortably below both the MINI's measured
# physical residual and an ordinary FDM layer.  A coarser 48-segment draft was
# about 0.025 mm out at the same probes, large enough to become part of the
# quantity this target is meant to judge.
BUMP_SEGMENTS = 96
REFERENCE_POINTS = 12_000
SHIPPED_VOXEL_BUDGET = 128**3
# Recomputed by a test through scanner3d.reconstruction_export rather than
# trusted as an unexplained design constant.
EXPECTED_SHIPPED_VOXEL_MM = 0.6392666016301444

STL_NAME = f"{TARGET_ID}.stl"
PLY_NAME = f"{TARGET_ID}-reference.ply"
MANIFEST_NAME = f"{TARGET_ID}.json"


@dataclass(frozen=True, slots=True)
class Patch:
    """One named top-surface formula selected over a rectangular grid cell."""

    label: str
    height: Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class Triangle:
    """One outward-oriented boundary triangle and its semantic surface label."""

    vertices: tuple[Point, Point, Point]
    normal: Point
    label: str

    @property
    def area(self) -> float:
        a, b, c = self.vertices
        cross = _cross(_subtract(b, a), _subtract(c, a))
        return 0.5 * _length(cross)


def _constant(value: float) -> Callable[[float, float], float]:
    return lambda _x, _y: value


BASE = Patch("base_top", _constant(BASE_Z))
PLATEAU = Patch("plateau_top", _constant(22.0))
POCKET = Patch("pocket_floor", _constant(10.0))
RAMP = Patch("ramp_top", lambda _x, y: 10.0 + 46.0 * (y + 33.0) / 66.0)
STEP_2 = Patch("step_2mm_top", _constant(8.0))
STEP_5 = Patch("step_5mm_top", _constant(11.0))
STEP_10 = Patch("step_10mm_top", _constant(16.0))
STEP_20 = Patch("step_20mm_top", _constant(26.0))
RAIL_1_5 = Patch("rail_1_5mm_top", _constant(10.0))
RAIL_3 = Patch("rail_3mm_top", _constant(10.0))
RAIL_6 = Patch("rail_6mm_top", _constant(10.0))


def _bump_height(x: float, y: float) -> float:
    """A two-axis smooth bump that joins the base at z=6 with zero slope."""

    u = (x - 16.0) / 26.0
    v = (y + 34.0) / 26.0
    return BASE_Z + 13.0 * math.sin(math.pi * u) ** 2 * math.sin(math.pi * v) ** 2


BUMP = Patch("smooth_bump", _bump_height)


def _inside(value: float, low: float, high: float) -> bool:
    return low < value < high


def patch_at(x: float, y: float) -> Patch:
    """Return the unique top patch containing a cell midpoint."""

    if _inside(x, -42.0, -14.0) and _inside(y, -8.0, 36.0):
        if _inside(x, -38.0, -29.0) and _inside(y, 22.0, 32.0):
            return POCKET
        return PLATEAU
    if _inside(x, -9.0, 11.0) and _inside(y, -33.0, 33.0):
        return RAMP
    if _inside(x, 16.0, 42.0) and _inside(y, -34.0, -8.0):
        return BUMP
    if _inside(x, 16.0, 42.0) and _inside(y, 3.0, 10.0):
        return STEP_2
    if _inside(x, 16.0, 42.0) and _inside(y, 10.0, 17.0):
        return STEP_5
    if _inside(x, 16.0, 42.0) and _inside(y, 17.0, 25.0):
        return STEP_10
    if _inside(x, 16.0, 42.0) and _inside(y, 25.0, 35.0):
        return STEP_20
    if _inside(x, -42.0, -32.0) and _inside(y, -16.5, -15.0):
        return RAIL_1_5
    if _inside(x, -29.0, -16.0) and _inside(y, -17.0, -14.0):
        return RAIL_3
    if _inside(x, 16.0, 34.0) and _inside(y, -5.5, 0.5):
        return RAIL_6
    return BASE


def _axis_breaks() -> tuple[list[float], list[float]]:
    xs = {
        MIN_X,
        -42.0,
        -38.0,
        -32.0,
        -29.0,
        -16.0,
        -14.0,
        -9.0,
        11.0,
        16.0,
        34.0,
        42.0,
        MAX_X,
    }
    ys = {
        MIN_Y,
        -34.0,
        -33.0,
        -17.0,
        -16.5,
        -15.0,
        -14.0,
        -8.0,
        -5.5,
        0.5,
        3.0,
        10.0,
        17.0,
        22.0,
        25.0,
        32.0,
        33.0,
        35.0,
        36.0,
        MAX_Y,
    }
    for index in range(BUMP_SEGMENTS + 1):
        xs.add(round(16.0 + 26.0 * index / BUMP_SEGMENTS, 12))
        ys.add(round(-34.0 + 26.0 * index / BUMP_SEGMENTS, 12))
    return sorted(xs), sorted(ys)


def _subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(vector: Point) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _triangle(a: Point, b: Point, c: Point, label: str) -> Triangle | None:
    # STL stores float32.  Quantise before deriving normals, area and volume so
    # the manifest describes the bytes a slicer receives rather than a
    # slightly finer in-memory precursor that is discarded during encoding.
    a, b, c = _float_point(a), _float_point(b), _float_point(c)
    cross = _cross(_subtract(b, a), _subtract(c, a))
    magnitude = _length(cross)
    if magnitude <= 1e-10:
        return None
    normal = tuple(value / magnitude for value in cross)
    return Triangle((a, b, c), normal, label)  # type: ignore[arg-type]


def _append_triangle(out: list[Triangle], a: Point, b: Point, c: Point, label: str) -> None:
    triangle = _triangle(a, b, c, label)
    if triangle is not None:
        out.append(triangle)


def _append_quad(out: list[Triangle], a: Point, b: Point, c: Point, d: Point, label: str) -> None:
    _append_triangle(out, a, b, c, label)
    _append_triangle(out, a, c, d, label)


def _append_polygon_fan(out: list[Triangle], points: Sequence[Point], label: str) -> None:
    """Triangulate an oriented planar polygon without dropping collinear vertices.

    Step corners put three different top heights on one vertical line.  Those
    intermediate vertices must remain explicit or an STL edge ends halfway
    through another triangle (a T-junction).  A centre fan gives every boundary
    segment one non-degenerate triangle and keeps the winding of ``points``.
    """

    centre = tuple(math.fsum(point[axis] for point in points) / len(points) for axis in range(3))
    for index, point in enumerate(points):
        _append_triangle(out, centre, point, points[(index + 1) % len(points)], label)


def _wall_label(high: Patch, low: Patch) -> str:
    labels = {high.label, low.label}
    if "pocket_floor" in labels and "plateau_top" in labels:
        return "pocket_wall"
    if all(label.startswith("step_") for label in labels):
        return "step_riser"
    if high.label.startswith("rail_"):
        return high.label.replace("_top", "_side")
    return high.label.replace("_top", "_side")


def _x_wall(
    out: list[Triangle],
    x: float,
    y0: float,
    y1: float,
    left: Patch,
    right: Patch,
    splits0: Sequence[float],
    splits1: Sequence[float],
) -> None:
    left0, left1 = left.height(x, y0), left.height(x, y1)
    right0, right1 = right.height(x, y0), right.height(x, y1)
    difference = left.height(x, (y0 + y1) / 2.0) - right.height(x, (y0 + y1) / 2.0)
    if abs(difference) <= 1e-9 and max(abs(left0 - right0), abs(left1 - right1)) <= 1e-9:
        return
    if (left0 - right0) * (left1 - right1) < -1e-9:
        raise RuntimeError("two top patches cross inside an x-boundary interval")
    if difference > 0.0:  # high material is west; outward normal is +x
        middle1 = [value for value in splits1 if right1 < value < left1]
        middle0 = [value for value in splits0 if right0 < value < left0]
        _append_polygon_fan(
            out,
            [
                (x, y0, right0),
                (x, y1, right1),
                *((x, y1, value) for value in middle1),
                (x, y1, left1),
                (x, y0, left0),
                *((x, y0, value) for value in reversed(middle0)),
            ],
            _wall_label(left, right),
        )
    else:  # high material is east; outward normal is -x
        middle0 = [value for value in splits0 if left0 < value < right0]
        middle1 = [value for value in splits1 if left1 < value < right1]
        _append_polygon_fan(
            out,
            [
                (x, y0, left0),
                *((x, y0, value) for value in middle0),
                (x, y0, right0),
                (x, y1, right1),
                *((x, y1, value) for value in reversed(middle1)),
                (x, y1, left1),
            ],
            _wall_label(right, left),
        )


def _y_wall(
    out: list[Triangle],
    y: float,
    x0: float,
    x1: float,
    south: Patch,
    north: Patch,
    splits0: Sequence[float],
    splits1: Sequence[float],
) -> None:
    south0, south1 = south.height(x0, y), south.height(x1, y)
    north0, north1 = north.height(x0, y), north.height(x1, y)
    difference = south.height((x0 + x1) / 2.0, y) - north.height((x0 + x1) / 2.0, y)
    if abs(difference) <= 1e-9 and max(abs(south0 - north0), abs(south1 - north1)) <= 1e-9:
        return
    if (south0 - north0) * (south1 - north1) < -1e-9:
        raise RuntimeError("two top patches cross inside a y-boundary interval")
    if difference > 0.0:  # high material is south; outward normal is +y
        middle0 = [value for value in splits0 if north0 < value < south0]
        middle1 = [value for value in splits1 if north1 < value < south1]
        _append_polygon_fan(
            out,
            [
                (x0, y, north0),
                *((x0, y, value) for value in middle0),
                (x0, y, south0),
                (x1, y, south1),
                *((x1, y, value) for value in reversed(middle1)),
                (x1, y, north1),
            ],
            _wall_label(south, north),
        )
    else:  # high material is north; outward normal is -y
        middle1 = [value for value in splits1 if south1 < value < north1]
        middle0 = [value for value in splits0 if south0 < value < north0]
        _append_polygon_fan(
            out,
            [
                (x0, y, south0),
                (x1, y, south1),
                *((x1, y, value) for value in middle1),
                (x1, y, north1),
                (x0, y, north0),
                *((x0, y, value) for value in reversed(middle0)),
            ],
            _wall_label(north, south),
        )


def _vertex_heights(
    patches: Sequence[Sequence[Patch]],
    xs: Sequence[float],
    ys: Sequence[float],
    ix: int,
    iy: int,
) -> list[float]:
    """Every top height incident on one grid vertex, sorted and de-duplicated."""

    heights: set[float] = set()
    for cell_x in (ix - 1, ix):
        for cell_y in (iy - 1, iy):
            if 0 <= cell_x < len(xs) - 1 and 0 <= cell_y < len(ys) - 1:
                heights.add(patches[cell_x][cell_y].height(xs[ix], ys[iy]))
    return sorted(heights)


def build_mesh() -> list[Triangle]:
    """Build one connected, outward-oriented, support-free solid."""

    xs, ys = _axis_breaks()
    patches = [
        [
            patch_at((xs[ix] + xs[ix + 1]) / 2.0, (ys[iy] + ys[iy + 1]) / 2.0)
            for iy in range(len(ys) - 1)
        ]
        for ix in range(len(xs) - 1)
    ]
    out: list[Triangle] = []

    # Top surface.  Every cell has one formula, so its two triangles meet the
    # neighbouring cells exactly at a shared grid edge.
    for ix, (x0, x1) in enumerate(pairwise(xs)):
        for iy, (y0, y1) in enumerate(pairwise(ys)):
            patch = patches[ix][iy]
            p00 = (x0, y0, patch.height(x0, y0))
            p10 = (x1, y0, patch.height(x1, y0))
            p11 = (x1, y1, patch.height(x1, y1))
            p01 = (x0, y1, patch.height(x0, y1))
            _append_quad(out, p00, p10, p11, p01, patch.label)

    # Vertical faces at every discontinuity in the height field.
    for ix in range(1, len(xs) - 1):
        for iy, (y0, y1) in enumerate(pairwise(ys)):
            _x_wall(
                out,
                xs[ix],
                y0,
                y1,
                patches[ix - 1][iy],
                patches[ix][iy],
                _vertex_heights(patches, xs, ys, ix, iy),
                _vertex_heights(patches, xs, ys, ix, iy + 1),
            )
    for iy in range(1, len(ys) - 1):
        for ix, (x0, x1) in enumerate(pairwise(xs)):
            _y_wall(
                out,
                ys[iy],
                x0,
                x1,
                patches[ix][iy - 1],
                patches[ix][iy],
                _vertex_heights(patches, xs, ys, ix, iy),
                _vertex_heights(patches, xs, ys, ix + 1, iy),
            )

    # Four exterior sides, split at the same boundary vertices used by the top
    # and bottom so the STL contains no T-junctions.
    west_ix, east_ix = 0, len(xs) - 2
    for iy, (y0, y1) in enumerate(pairwise(ys)):
        west = patches[west_ix][iy]
        _append_quad(
            out,
            (MIN_X, y0, BOTTOM_Z),
            (MIN_X, y0, west.height(MIN_X, y0)),
            (MIN_X, y1, west.height(MIN_X, y1)),
            (MIN_X, y1, BOTTOM_Z),
            "outer_west",
        )
        east = patches[east_ix][iy]
        _append_quad(
            out,
            (MAX_X, y0, BOTTOM_Z),
            (MAX_X, y1, BOTTOM_Z),
            (MAX_X, y1, east.height(MAX_X, y1)),
            (MAX_X, y0, east.height(MAX_X, y0)),
            "outer_east",
        )
    south_iy, north_iy = 0, len(ys) - 2
    for ix, (x0, x1) in enumerate(pairwise(xs)):
        south = patches[ix][south_iy]
        _append_quad(
            out,
            (x0, MIN_Y, BOTTOM_Z),
            (x1, MIN_Y, BOTTOM_Z),
            (x1, MIN_Y, south.height(x1, MIN_Y)),
            (x0, MIN_Y, south.height(x0, MIN_Y)),
            "outer_south",
        )
        north = patches[ix][north_iy]
        _append_quad(
            out,
            (x0, MAX_Y, BOTTOM_Z),
            (x0, MAX_Y, north.height(x0, MAX_Y)),
            (x1, MAX_Y, north.height(x1, MAX_Y)),
            (x1, MAX_Y, BOTTOM_Z),
            "outer_north",
        )

    # One downward fan closes the print-bed face.  Its perimeter uses every
    # exterior side split, but it does not inherit the thousands of internal
    # top-grid edges that have no topological relevance to the bottom.
    perimeter: list[Point] = []
    perimeter.extend((x, MIN_Y, BOTTOM_Z) for x in xs)
    perimeter.extend((MAX_X, y, BOTTOM_Z) for y in ys[1:])
    perimeter.extend((x, MAX_Y, BOTTOM_Z) for x in reversed(xs[:-1]))
    perimeter.extend((MIN_X, y, BOTTOM_Z) for y in reversed(ys[1:-1]))
    centre = (0.0, 0.0, BOTTOM_Z)
    for index, point in enumerate(perimeter):
        following = perimeter[(index + 1) % len(perimeter)]
        _append_triangle(out, centre, following, point, "bottom")

    validate_mesh(out)
    return out


def _float_point(point: Point) -> Point:
    return struct.unpack("<3f", struct.pack("<3f", *point))


def _edge_key(a: Point, b: Point) -> tuple[Point, Point]:
    first, second = _float_point(a), _float_point(b)
    return (first, second) if first < second else (second, first)


def validate_mesh(triangles: Sequence[Triangle]) -> None:
    """Refuse a degenerate, open, mis-oriented or wrongly sized print mesh."""

    if not triangles:
        raise RuntimeError("target mesh has no triangles")
    edges: Counter[tuple[Point, Point]] = Counter()
    directions: Counter[tuple[Point, Point]] = Counter()
    values = [[], [], []]
    signed_volume = 0.0
    for triangle in triangles:
        if triangle.area <= 1e-8:
            raise RuntimeError("target mesh contains a degenerate triangle")
        a, b, c = (_float_point(point) for point in triangle.vertices)
        for point in (a, b, c):
            for axis in range(3):
                if not math.isfinite(point[axis]):
                    raise RuntimeError("target mesh contains a non-finite coordinate")
                values[axis].append(point[axis])
        for start, end in ((a, b), (b, c), (c, a)):
            key = _edge_key(start, end)
            edges[key] += 1
            directions[key] += 1 if (start, end) == key else -1
        signed_volume += sum(a[i] * _cross(b, c)[i] for i in range(3)) / 6.0
    bad_counts = [key for key, count in edges.items() if count != 2]
    bad_directions = [key for key, direction in directions.items() if direction != 0]
    if bad_counts or bad_directions:
        raise RuntimeError(
            "target boundary is not a consistently oriented 2-manifold: "
            f"{len(bad_counts)} incidence failures, {len(bad_directions)} direction failures"
        )
    bounds = tuple((min(axis), max(axis)) for axis in values)
    expected = ((MIN_X, MAX_X), (MIN_Y, MAX_Y), (BOTTOM_Z, MAX_Z))
    if bounds != expected:
        raise RuntimeError(f"target bounds are {bounds!r}, expected {expected!r}")
    if signed_volume <= 0.0:
        raise RuntimeError(f"target signed volume is {signed_volume!r}, not positive")


def _stl_bytes(triangles: Sequence[Triangle]) -> bytes:
    title = (
        f"{TARGET_ID}; units=mm; generated by reference/targets/reconstruction_v1/generate.py"
    ).encode("ascii")
    header = title[:80].ljust(80, b"\0")
    records = bytearray(header)
    records.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        normal = _float_point(triangle.normal)
        vertices = tuple(_float_point(point) for point in triangle.vertices)
        records.extend(
            struct.pack(
                "<12fH",
                *normal,
                *vertices[0],
                *vertices[1],
                *vertices[2],
                0,
            )
        )
    return bytes(records)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def _sample_reference(
    triangles: Sequence[Triangle], surface_ids: dict[str, int]
) -> list[tuple[Point, Point, int]]:
    scannable = [triangle for triangle in triangles if triangle.label != "bottom"]
    cumulative: list[float] = []
    area = 0.0
    for triangle in scannable:
        area += triangle.area
        cumulative.append(area)
    samples: list[tuple[Point, Point, int]] = []
    for index in range(REFERENCE_POINTS):
        target = (index + 0.5) * area / REFERENCE_POINTS
        triangle = scannable[bisect.bisect_left(cumulative, target)]
        u = _radical_inverse(index + 1, 2)
        v = _radical_inverse(index + 1, 3)
        root = math.sqrt(u)
        weights = (1.0 - root, root * (1.0 - v), root * v)
        point = tuple(
            sum(weights[vertex] * triangle.vertices[vertex][axis] for vertex in range(3))
            for axis in range(3)
        )
        samples.append(
            (_float_point(point), _float_point(triangle.normal), surface_ids[triangle.label])
        )
    return samples


def _ply_bytes(samples: Sequence[tuple[Point, Point, int]]) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment target {TARGET_ID}\n"
        "comment units millimetres\n"
        "comment excludes the unscannable print-bed face labelled bottom\n"
        f"element vertex {len(samples)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property ushort surface_id\n"
        "end_header\n"
    ).encode("ascii")
    payload = bytearray(header)
    for point, normal, surface_id in samples:
        payload.extend(struct.pack("<6fH", *point, *normal, surface_id))
    return bytes(payload)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialise(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _mesh_statistics(triangles: Sequence[Triangle]) -> dict[str, float | int | bool]:
    volume = 0.0
    for triangle in triangles:
        a, b, c = triangle.vertices
        cross = _cross(b, c)
        volume += sum(a[index] * cross[index] for index in range(3)) / 6.0
    return {
        "triangle_count": len(triangles),
        "surface_area_mm2": math.fsum(triangle.area for triangle in triangles),
        "volume_mm3": volume,
        "watertight_oriented_2_manifold": True,
    }


def _manifest(
    triangles: Sequence[Triangle], stl: bytes, ply: bytes, surface_ids: dict[str, int]
) -> dict[str, object]:
    label_counts = Counter(triangle.label for triangle in triangles)
    return {
        "schema_version": 1,
        "target_id": TARGET_ID,
        "purpose": (
            "physical truth target for rigid relative-pose registration, coverage, edge "
            "preservation, dimension error, planar lumpiness and unsupported-fill evaluation"
        ),
        "coordinate_frame": {
            "unit": UNITS,
            "origin": "centre of the print-bed face",
            "x": "right across the 92 mm width in the documented top view",
            "y": "toward the 82 mm rear in the documented top view",
            "z": "up from the print-bed face",
        },
        "bounds_mm": {
            "minimum": [MIN_X, MIN_Y, BOTTOM_Z],
            "maximum": [MAX_X, MAX_Y, MAX_Z],
            "size": [MAX_X - MIN_X, MAX_Y - MIN_Y, MAX_Z - BOTTOM_Z],
            "under_100mm_on_every_axis": True,
        },
        "resolution_design": {
            "shipped_tsdf_voxel_budget": SHIPPED_VOXEL_BUDGET,
            "expected_voxel_for_full_cad_bounds_mm": EXPECTED_SHIPPED_VOXEL_MM,
            "rail_widths_mm": [1.5, 3.0, 6.0],
            "rail_widths_in_expected_voxels": [
                width / EXPECTED_SHIPPED_VOXEL_MM for width in (1.5, 3.0, 6.0)
            ],
            "why": (
                "the three rails span approximately 2.3, 4.7 and 9.4 shipped voxels; the "
                "smallest is deliberately vulnerable while the largest should survive"
            ),
        },
        "nominal_dimensions_mm": {
            "overall_x": 92.0,
            "overall_y": 82.0,
            "overall_z": 56.0,
            "base_thickness": 6.0,
            "plateau_top_z": 22.0,
            "plateau_height_above_base": 16.0,
            "pocket_opening_x": 9.0,
            "pocket_opening_y": 10.0,
            "pocket_floor_z": 10.0,
            "pocket_depth": 12.0,
            "ramp_low_z": 10.0,
            "ramp_high_z": 56.0,
            "step_heights_above_base": [2.0, 5.0, 10.0, 20.0],
            "smooth_bump_height_above_base": 13.0,
        },
        "planar_regions": [
            {
                "name": "base_front_left",
                "x": [-42.0, -16.0],
                "y": [-36.0, -19.0],
                "plane": "z = 6",
            },
            {
                "name": "plateau_main",
                "x": [-40.0, -17.0],
                "y": [-5.0, 16.0],
                "plane": "z = 22",
            },
            {
                "name": "ramp_interior",
                "x": [-7.0, 9.0],
                "y": [-27.0, 27.0],
                "plane": "z = 33 + (23/33) y",
            },
            {
                "name": "highest_step",
                "x": [18.0, 40.0],
                "y": [26.0, 33.0],
                "plane": "z = 26",
            },
            {
                "name": "pocket_floor",
                "x": [-36.0, -31.0],
                "y": [24.0, 30.0],
                "plane": "z = 10",
            },
        ],
        "curved_region": {
            "name": "smooth_bump",
            "domain": {"x": [16.0, 42.0], "y": [-34.0, -8.0]},
            "surface": ("z = 6 + 13 sin(pi (x-16)/26)^2 sin(pi (y+34)/26)^2"),
            "peak": [29.0, -21.0, 19.0],
            "mesh_segments_per_axis": BUMP_SEGMENTS,
        },
        "landmarks_mm": {
            "ramp_low_front_left": [-9.0, -33.0, 10.0],
            "ramp_high_back_right": [11.0, 33.0, 56.0],
            "smooth_bump_peak": [29.0, -21.0, 19.0],
            "pocket_floor_front_left": [-38.0, 22.0, 10.0],
            "highest_step_back_right": [42.0, 35.0, 26.0],
        },
        "features": {
            "deep_open_pocket": (
                "tests missed concave surface and whether completion bridges unsupported space"
            ),
            "smooth_two_axis_bump": "tests curvature, local ripples and blobby surface bias",
            "four_metric_steps": "tests edge placement and dimension preservation at four scales",
            "three_thin_rails": "tests feature survival around the shipped mesh resolution",
            "ramp_and_offset_layout": (
                "break rotational and flip symmetries so relative-pose tracking has one answer"
            ),
        },
        "mesh": _mesh_statistics(triangles),
        "surface_ids": {
            label: {
                "id": surface_ids[label],
                "triangle_count": label_counts[label],
                "included_in_reference_ply": label != "bottom",
            }
            for label in sorted(surface_ids)
        },
        "artifacts": {
            STL_NAME: {
                "role": "binary STL to slice in millimetres without scaling",
                "bytes": len(stl),
                "sha256": _sha256(stl),
            },
            PLY_NAME: {
                "role": (
                    "area-weighted CAD reference surface with normals and surface IDs; excludes "
                    "only the print-bed face"
                ),
                "points": REFERENCE_POINTS,
                "bytes": len(ply),
                "sha256": _sha256(ply),
            },
        },
        "manufacturing_truth": {
            "status": "nominal CAD truth until the individual print is measured",
            "required_before_physical_gating": [
                "record printer, material, nozzle and layer height",
                "take at least three caliper readings of overall x, y and z",
                "take at least three readings of plateau height, pocket opening and pocket depth",
                "record caliper resolution and the full reading spread; do not report only a mean",
                "photograph the printed target before scanning so damage and cleanup are visible",
                (
                    "record whether the camera or target supplied the relative motion; never "
                    "move both in one capture"
                ),
            ],
            "limitation": (
                "the STL is exact digital truth; an FDM print is not. Dense CAD agreement can "
                "gate scanner-scale defects only with a tolerance no tighter than the measured "
                "print error and layer surface"
            ),
        },
        "capture_constraint": (
            "preferred first run: keep the camera fixed and rotate the target continuously on a "
            "turntable; moving the camera around a stationary target is also valid, but never "
            "move both in one capture and do not flip, scale, sand, fill, lift or recenter the "
            "target"
        ),
    }


def generate(output_dir: Path) -> tuple[Path, Path, Path]:
    """Generate all three bound artifacts into an existing or new directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    triangles = build_mesh()
    labels = sorted({triangle.label for triangle in triangles})
    surface_ids = {label: index for index, label in enumerate(labels)}
    stl = _stl_bytes(triangles)
    samples = _sample_reference(triangles, surface_ids)
    ply = _ply_bytes(samples)
    manifest = _serialise(_manifest(triangles, stl, ply, surface_ids))
    paths = (
        output_dir / STL_NAME,
        output_dir / PLY_NAME,
        output_dir / MANIFEST_NAME,
    )
    for path, payload in zip(paths, (stl, ply, manifest), strict=True):
        path.write_bytes(payload)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for the STL, reference PLY and manifest (default: beside this script)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    for path in generate(arguments.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
