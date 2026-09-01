"""The printable reconstruction target is bound evidence, not an unchecked STL.

These tests do not trust the generator's own checks or manifest.  They parse
the committed binary artifacts independently, prove their geometry, verify the
digests, and regenerate the whole target in a temporary directory.  A slicer
should therefore receive exactly the object whose dimensions and reference
surface the physical protocol describes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import pytest

from scanner3d.reconstruction_export import choose_tsdf_resolution

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "reference" / "targets" / "reconstruction_v1"
GENERATOR = TARGET / "generate.py"
STL = TARGET / "reconstruction-target-v1.stl"
PLY = TARGET / "reconstruction-target-v1-reference.ply"
MANIFEST = TARGET / "reconstruction-target-v1.json"
PRINT_RECORD = TARGET / "prints" / "physical-print-20260831.json"
STL_TRIANGLE = struct.Struct("<12fH")
PLY_POINT = struct.Struct("<6fH")
Point = tuple[float, float, float]
Edge = tuple[Point, Point]


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stl_triangles():
    payload = STL.read_bytes()
    assert len(payload) >= 84
    count = struct.unpack_from("<I", payload, 80)[0]
    assert len(payload) == 84 + count * STL_TRIANGLE.size
    triangles = []
    for index in range(count):
        values = STL_TRIANGLE.unpack_from(payload, 84 + index * STL_TRIANGLE.size)
        triangles.append(
            (
                values[0:3],
                (values[3:6], values[6:9], values[9:12]),
                values[12],
            )
        )
    return triangles


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _length(vector) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _ply_points():
    payload = PLY.read_bytes()
    header_end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:header_end].decode("ascii").splitlines()
    assert header[:2] == ["ply", "format binary_little_endian 1.0"]
    element = next(line for line in header if line.startswith("element vertex "))
    count = int(element.rsplit(" ", 1)[1])
    assert header[-8:] == [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property ushort surface_id",
        "end_header",
    ]
    assert len(payload) == header_end + count * PLY_POINT.size
    return [
        PLY_POINT.unpack_from(payload, header_end + index * PLY_POINT.size)
        for index in range(count)
    ]


def test_committed_target_regenerates_byte_for_byte(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [MANIFEST.name, PLY.name, STL.name]
    )
    for committed in (MANIFEST, PLY, STL):
        assert (tmp_path / committed.name).read_bytes() == committed.read_bytes()


def test_manifest_binds_every_print_and_reference_byte():
    document = _manifest()

    assert document["schema_version"] == 1
    assert document["target_id"] == "reconstruction-target-v1"
    artifacts = document["artifacts"]
    assert set(artifacts) == {STL.name, PLY.name}
    for path in (STL, PLY):
        record = artifacts[path.name]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == _sha256(path)


def test_stl_is_one_closed_consistently_oriented_solid_with_exact_bounds():
    triangles = _stl_triangles()
    document = _manifest()
    edges: Counter[Edge] = Counter()
    directions: Counter[Edge] = Counter()
    graph: defaultdict[Point, set[Point]] = defaultdict(set)
    coordinates: list[list[float]] = [[], [], []]
    areas: list[float] = []
    signed_volumes: list[float] = []

    for encoded_normal, vertices, attribute in triangles:
        assert attribute == 0
        assert all(math.isfinite(value) for value in encoded_normal)
        a, b, c = vertices
        assert all(math.isfinite(value) for point in vertices for value in point)
        cross = _cross(_subtract(b, a), _subtract(c, a))
        magnitude = _length(cross)
        assert magnitude > 1e-8
        assert sum(encoded_normal[axis] * cross[axis] for axis in range(3)) > 0.0
        assert _length(encoded_normal) == pytest.approx(1.0, abs=2e-6)
        areas.append(magnitude / 2.0)
        signed_volumes.append(math.fsum(a[axis] * _cross(b, c)[axis] for axis in range(3)) / 6.0)
        for point in vertices:
            for axis in range(3):
                coordinates[axis].append(point[axis])
        for start, end in ((a, b), (b, c), (c, a)):
            key = _edge(start, end)
            edges[key] += 1
            directions[key] += 1 if (start, end) == key else -1
            graph[start].add(end)
            graph[end].add(start)

    # Every edge belongs to exactly two oppositely wound faces.  This catches
    # holes, duplicate sheets, non-manifold seams and the T-junction a first
    # version of the step corners would otherwise have contained.
    assert set(edges.values()) == {2}
    assert set(directions.values()) == {0}

    reached = {next(iter(graph))}
    queue = deque(reached)
    while queue:
        queue.extend(graph[queue.popleft()] - reached)
        reached.update(queue)
    assert reached == set(graph)

    bounds = [[min(axis), max(axis)] for axis in coordinates]
    assert bounds == [[-46.0, 46.0], [-41.0, 41.0], [0.0, 56.0]]
    assert all(high - low < 100.0 for low, high in bounds)
    assert len(triangles) == document["mesh"]["triangle_count"]
    assert math.fsum(areas) == pytest.approx(document["mesh"]["surface_area_mm2"], rel=2e-7)
    assert math.fsum(signed_volumes) == pytest.approx(document["mesh"]["volume_mm3"], rel=2e-7)
    assert math.fsum(signed_volumes) > 0.0


def test_reference_ply_is_dense_finite_and_excludes_only_the_bed_face():
    points = _ply_points()
    document = _manifest()
    surface_ids = {record["id"]: label for label, record in document["surface_ids"].items()}
    labels: Counter[str] = Counter()
    bump_deviations: list[float] = []

    assert len(points) == document["artifacts"][PLY.name]["points"] == 12_000
    for record in points:
        x, y, z, nx, ny, nz, surface_id = record
        assert all(math.isfinite(value) for value in (x, y, z, nx, ny, nz))
        assert -46.0 <= x <= 46.0
        assert -41.0 <= y <= 41.0
        assert 0.0 <= z <= 56.0
        assert math.sqrt(nx * nx + ny * ny + nz * nz) == pytest.approx(1.0, abs=2e-6)
        assert surface_id in surface_ids
        label = surface_ids[surface_id]
        labels[label] += 1
        if label == "smooth_bump":
            analytic_z = (
                6.0
                + 13.0
                * math.sin(math.pi * (x - 16.0) / 26.0) ** 2
                * math.sin(math.pi * (y + 34.0) / 26.0) ** 2
            )
            bump_deviations.append(abs(z - analytic_z))

    assert "bottom" not in labels
    for required in (
        "base_top",
        "plateau_top",
        "pocket_floor",
        "pocket_wall",
        "rail_1_5mm_top",
        "rail_3mm_top",
        "rail_6mm_top",
        "ramp_top",
        "smooth_bump",
        "step_2mm_top",
        "step_5mm_top",
        "step_10mm_top",
        "step_20mm_top",
    ):
        assert labels[required] > 0, required
    assert bump_deviations
    assert max(bump_deviations) < 0.007


def test_target_features_are_sized_against_the_real_export_resolution():
    document = _manifest()
    bounds = document["bounds_mm"]
    minimum = bounds["minimum"]
    maximum = bounds["maximum"]
    corners = [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    design = document["resolution_design"]
    resolution = choose_tsdf_resolution(corners, voxel_budget=design["shipped_tsdf_voxel_budget"])

    assert resolution.voxel_size == pytest.approx(
        design["expected_voxel_for_full_cad_bounds_mm"], abs=1e-15
    )
    expected_ratios = [width / resolution.voxel_size for width in (1.5, 3.0, 6.0)]
    assert design["rail_widths_in_expected_voxels"] == pytest.approx(expected_ratios)
    assert 2.0 < expected_ratios[0] < 3.0
    assert 4.0 < expected_ratios[1] < 5.0
    assert 9.0 < expected_ratios[2] < 10.0


def test_physical_truth_stays_explicitly_nominal_until_this_print_is_measured():
    document = _manifest()
    truth = document["manufacturing_truth"]

    assert truth["status"] == "nominal CAD truth until the individual print is measured"
    assert len(truth["required_before_physical_gating"]) >= 5
    assert "FDM print is not" in truth["limitation"]
    assert document["capture_constraint"].startswith(
        "preferred first run: keep the camera fixed and rotate the target"
    )


def test_first_print_measurements_are_consistent_and_remain_provisional():
    record = json.loads(PRINT_RECORD.read_text(encoding="utf-8"))

    assert record["schema_version"] == 1
    assert record["target_id"] == "reconstruction-target-v1"
    measurements = record["measurements"]
    for measurement in measurements.values():
        assert len(measurement["readings_mm"]) == 1
        assert measurement["error_from_nominal_mm"] == pytest.approx(
            measurement["readings_mm"][0] - measurement["nominal_mm"]
        )
    adjustment = record["dense_reference_adjustment"]
    assert adjustment["scale_x"] == pytest.approx(91.9 / 92.0)
    assert adjustment["scale_y"] == pytest.approx(81.85 / 82.0)
    assert adjustment["scale_z"] == pytest.approx(55.85 / 56.0)
    assert record["summary"]["largest_observed_absolute_nominal_deviation_mm"] == 0.15
    readiness = record["capture_readiness"]
    assert readiness["ready_for_first_comparative_scan"] is True
    assert readiness["ready_for_formal_physical_gate"] is False
    assert record["instrument"]["resolution_mm"] is None
    assert record["planned_capture"] == {
        "camera": "fixed",
        "motion": "target rotates continuously on a turntable",
        "selected_by": "owner",
        "status": "predeclared before capture",
    }
