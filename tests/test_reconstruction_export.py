"""What is on disk when an export finishes, and what is on disk when it does not.

The subject here is not the encoders - :mod:`tests.test_pointcloud` and
:mod:`tests.test_mesh` own those, and nothing in this module re-checks what they
already prove. It is the FOLDER and the SURFACE in it: that one run produces
exactly one new directory holding files a reader who did not write them can
open, that the mesh in those files is the fused one and not the occupancy
shrink-wrap this module used to write, that the record inside describes those
files byte for byte, that a second run leaves the first alone, and that a run
which fails or is cancelled leaves nothing that could be mistaken for a result.

Everything here is synthetic and offline. The frames are REAL 512000-byte
``Y16 `` buffers - :mod:`scanner3d.simulate` ray-casts a sphere whose radius is
exact ground truth into them - and the clouds are those same frames deprojected
through the production path and decimated, so the two inputs an export now takes
describe one object rather than two unrelated bags of numbers. The poses were
never fitted to anything and say so. Nothing opens a device, a devnode or sysfs.

The geometric claims are the reason the fused path exists, so they are measured
rather than described: a surface that is topologically perfect and sits three
quarters of a voxel outside the object is exactly the defect being replaced, and
only a distance to known truth can tell the two apart.
"""

from __future__ import annotations

import json
import math
import stat
import statistics
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import DepthUnitAssumption, deproject_depth_frame
from scanner3d.geometry import Point, Rigid, cross, dot, rotation_about_axis, subtract
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.mesh import INHERENT_LIMITATIONS as OCCUPANCY_LIMITATIONS
from scanner3d.mesh import TSDF_INHERENT_LIMITATIONS, MeshMethod, reconstruct, write_obj
from scanner3d.pipeline import SweepReconstruction
from scanner3d.pointcloud import MAX_POINTS, sha256_of
from scanner3d.preparation import DepthBand
from scanner3d.reconstruction_export import (
    CLOUD_NAME,
    MARGIN_VOXELS,
    MESH_METHOD,
    MESH_OBJ_NAME,
    MESH_PLY_NAME,
    OUTPUT_PREFIX,
    RECORD_NAME,
    STAGING_PREFIX,
    TRUNCATION_VOXELS,
    TSDF_MAX_DEPTH,
    TSDF_MIN_DEPTH,
    TSDF_MIN_WEIGHT,
    UPRIGHT_FROM_OPTICAL,
    ArtifactVerificationFailed,
    CaptureInputUnusable,
    FrameCountMismatch,
    GeometryNotExportable,
    NoGeometryOffered,
    ReconstructionExport,
    RegistrationMismatch,
    TsdfSurfaceUnavailable,
    choose_tsdf_resolution,
    export_reconstruction,
    model_points,
    registered_points,
    upright_mesh,
)
from scanner3d.simulate import Sphere, intrinsics_of, render
from scanner3d.tsdf import MAX_VOXELS, PYTHON_BACKEND_MAX_VOXELS, TsdfVolume
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

# A centred synthetic camera, so a geometric miss below is a defect in the
# export rather than in the framing.
INTRINSICS: IntrinsicsRecord = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)
DEPTH_SCALE = 0.004999999888241291

#: The truth every geometric claim here is measured against. Placed so the whole
#: surface sits well inside the accepted depth interval the export fixes, which
#: is a property the suite asserts rather than assumes.
SPHERE_CENTRE: Point = (0.0, 0.0, 90.0)
SPHERE_RADIUS = 20.0
VIEWS = 8

#: One point in this many is kept for the registered cloud, standing in for the
#: decimation the window applies before registering. The whole point of fusing
#: the RAW frames is that the mesh does not inherit it.
DECIMATION = 8

#: A voxel budget small enough that a whole export is a fraction of a second and
#: large enough that the chosen voxel lands near 1.0, where the difference
#: between a fused surface and an occupancy one is a number rather than a rounding
#: artefact. The resolution is derived from the extent and the budget together,
#: so lowering it exercises exactly the same arithmetic the window's own ceiling
#: does.
BUDGET = 120_000


@dataclass(frozen=True, slots=True)
class Capture:
    """One synthetic sweep: the raw frames, the decimated clouds, and the poses.

    All three in the order an export takes them, because the single most
    consequential thing a caller can get wrong is offering them out of step.
    """

    raw_frames: tuple[bytes, ...]
    clouds: tuple[tuple[Point, ...], ...]
    poses: tuple[Rigid, ...]

    @property
    def frames(self) -> int:
        return len(self.raw_frames)


def unit() -> DepthUnitAssumption:
    return DepthUnitAssumption(
        label="mm",
        stated_by="tests/test_reconstruction_export.py",
        basis="an assumption made up for a synthetic sphere; it measures nothing",
    )


def orbit(count: int = VIEWS, *, axis: Point = (0.0, 1.0, 0.0)) -> list[Rigid]:
    """Camera-to-model poses for a camera orbiting :data:`SPHERE_CENTRE`.

    :func:`scanner3d.simulate.render` moves the SCENE, so the pose a frame was
    rendered at is the model-to-camera one and the export wants its inverse.
    Getting that backwards is the mistake the fusion documents at length, so the
    inversion happens once, here, where it can be read.
    """
    return [
        rotation_about_axis(axis, 2.0 * math.pi * index / count, through=SPHERE_CENTRE).inverse()
        for index in range(count)
    ]


def sphere_capture(count: int = VIEWS) -> Capture:
    """A sphere seen from ``count`` points of one orbit about its own centre.

    Rendered ONCE and reused, which is exact rather than a shortcut: a sphere
    turned about its own centre is the same sphere, so every view of this orbit
    produces the identical depth image and only the pose differs. The assertion
    below proves that rather than asserting it in prose, because a reader who
    took it on trust would have to wonder whether this fixture fuses eight views
    or one. In the MODEL frame the eight cameras really are eight places, and
    the fused field really does see eight different caps of the sphere.
    """
    scene = Sphere(centre=SPHERE_CENTRE, radius=SPHERE_RADIUS)
    frame = render([scene], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    turned = render(
        [scene],
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
        pose=rotation_about_axis((0.0, 1.0, 0.0), 1.0, through=SPHERE_CENTRE),
    )
    assert turned.raw == frame.raw, "a sphere turned about its own centre is not the same sphere"
    conversion = deproject_depth_frame(
        frame.raw, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE, unit=unit()
    )
    cloud = tuple(camera_points(conversion))[::DECIMATION]
    poses = orbit(count)
    return Capture(
        raw_frames=tuple(frame.raw for _ in poses),
        clouds=tuple(cloud for _ in poses),
        poses=tuple(poses),
    )


@pytest.fixture(scope="module")
def capture() -> Capture:
    return sphere_capture()


def registration(poses: Sequence[Rigid], limitations: tuple[str, ...] = ()) -> SweepReconstruction:
    """A registration result that was never fitted to anything, and says so.

    It carries the poses the export is ACTUALLY about to fuse, not a bag of
    identities of the right length. The record this helper builds is what the
    manifest publishes about the geometry, and the export now proves the two
    describe one answer - so a helper that recorded eight identities beside
    eight orbit poses would have made every test in this module exercise the
    refusal rather than the export it is about.
    """
    return SweepReconstruction(
        poses=tuple(poses),
        frames=len(poses),
        revisits_found=0,
        chained_score=None,
        optimised_score=None,
        optimised=False,
        converged=False,
        limitations=limitations or ("synthetic poses; nothing here was registered",),
    )


def run(frames_dir: Path, capture: Capture, **overrides: object) -> ReconstructionExport:
    """One export of a synthetic capture, with every argument defaulted."""
    clouds = overrides.pop("clouds", capture.clouds)
    # Cast rather than left as ``object`` because the default registration below
    # is built FROM these poses: a test that overrides them gets a record of the
    # poses it actually asked for, so the consistency gate refuses only where a
    # test set out to break it.
    poses = cast("Sequence[Rigid]", overrides.pop("poses", capture.poses))
    arguments: dict[str, object] = {
        "raw_frames": capture.raw_frames,
        "intrinsics": INTRINSICS,
        "registration": registration(poses),
        "unit": unit(),
        "depth_scale": DEPTH_SCALE,
        "calibration_source": "a synthetic calibration that was never performed",
        "points_per_frame_target": len(capture.clouds[0]),
        "voxel_budget": BUDGET,
    }
    arguments.update(overrides)
    return export_reconstruction(frames_dir, clouds, poses, **arguments)  # type: ignore[arg-type]


def leftovers(frames_dir: Path) -> list[str]:
    """Everything in the capture folder, so a test can say "nothing was left"."""
    return sorted(entry.name for entry in frames_dir.iterdir())


def optical(point: Point) -> Point:
    """``point`` carried back out of the presentation frame into the model frame.

    The export applies the half turn; a claim about where the object really is
    has to undo it, and undoing it through the module's own inverse rather than
    by negating two coordinates by hand keeps the test from agreeing with a bug
    in the rotation.
    """
    return UPRIGHT_FROM_OPTICAL.inverse().apply(point)


def radial_error(points: list[Point]) -> tuple[float, float]:
    """``(median, rms)`` distance from the true sphere surface, in model space."""
    assert points, "nothing was recovered at all, so there is nothing to measure"
    errors = [abs(math.dist(point, SPHERE_CENTRE) - SPHERE_RADIUS) for point in points]
    return statistics.median(errors), math.sqrt(statistics.fmean(e * e for e in errors))


# -- readers that know only their format -----------------------------------


def read_ply_points(data: bytes) -> list[tuple[float, float, float]]:
    """The vertices of a binary little-endian PLY, by the format's own rules."""
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    lines = data[:end].decode("ascii").splitlines()
    assert lines[0] == "ply"
    assert lines[1].split()[1] == "binary_little_endian"
    (count,) = [int(line.split()[2]) for line in lines if line.startswith("element vertex ")]
    assert [line for line in lines if line.startswith("property ")] == [
        "property float x",
        "property float y",
        "property float z",
    ]
    body = data[end:]
    assert len(body) == count * 12
    return [struct.unpack_from("<fff", body, index * 12) for index in range(count)]


def read_ply_mesh(data: bytes) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]]]:
    """The vertices and faces of an ASCII PLY mesh."""
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    lines = data[:end].decode("ascii").splitlines()
    assert lines[0] == "ply"
    assert lines[1].split()[1] == "ascii"
    counts = {
        line.split()[1]: int(line.split()[2]) for line in lines if line.startswith("element ")
    }
    rows = data[end:].decode("ascii").splitlines()
    vertices = [tuple(float(value) for value in row.split()) for row in rows[: counts["vertex"]]]
    faces = []
    for row in rows[counts["vertex"] : counts["vertex"] + counts["face"]]:
        tokens = row.split()
        assert int(tokens[0]) == 3
        faces.append(tuple(int(value) for value in tokens[1:]))
    return vertices, faces


def read_obj(data: bytes) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]]]:
    """The vertices and faces of an OBJ, remembering that it indexes from one."""
    vertices: list[tuple[float, ...]] = []
    faces: list[tuple[int, ...]] = []
    for line in data.decode("ascii").splitlines():
        if line.startswith("# "):
            continue
        tokens = line.split()
        if tokens[0] == "v":
            vertices.append(tuple(float(value) for value in tokens[1:]))
        elif tokens[0] == "f":
            faces.append(tuple(int(value) - 1 for value in tokens[1:]))
        else:
            raise AssertionError(f"unexpected OBJ statement {tokens[0]!r}")
    return vertices, faces


def comments(data: bytes) -> str:
    """Every comment line of a PLY or an OBJ, as one blob to search."""
    text = data[: data.find(b"end_header") if b"end_header" in data else 40000].decode(
        "ascii", "replace"
    )
    return " ".join(
        line.removeprefix("comment ").removeprefix("# ")
        for line in text.splitlines()
        if line.startswith(("comment ", "# "))
    )


def section(record: Mapping[str, object], *keys: str) -> Any:
    """A nested part of the record, typed loosely enough for a test to index.

    The manifest is a ``Mapping[str, object]`` on purpose - it is JSON, and the
    module that writes it will not pretend to a shape a consumer could rely on
    statically. Reading it in a test therefore needs one place that says so,
    rather than a cast at every assertion.
    """
    value: Any = record
    for key in keys:
        value = value[key]
    return value


def squeezed(text: str) -> str:
    """``text`` with every space removed.

    A caveat is WRAPPED into the comment block, and the wrap can fall inside a
    hyphenated word as easily as at a space. Comparing the two with their spaces
    removed is what makes "is this sentence in the file" a question about the
    sentence rather than about where the wrapper happened to break it.
    """
    return "".join(text.split())


# -- one run ---------------------------------------------------------------


def test_one_run_writes_one_folder_of_geometry_and_a_record_that_matches_it(
    tmp_path: Path, capture: Capture
) -> None:
    """The whole contract of a successful export, read back off the disk.

    One new directory, four files, and a manifest whose lengths and digests are
    the lengths and digests of the bytes that are actually there - checked by
    hashing them again here rather than by trusting the number the encoder
    returned, because a manifest a consumer cannot rely on without re-deriving
    it is not a manifest.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)

    (directory,) = [path for path in frames_dir.iterdir() if path.is_dir()]
    assert directory.name == f"{OUTPUT_PREFIX}0001"
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [CLOUD_NAME, MESH_PLY_NAME, MESH_OBJ_NAME, RECORD_NAME]
    )
    record = json.loads((directory / RECORD_NAME).read_bytes())
    for artifact in record["artifacts"]:
        payload = (directory / artifact["file"]).read_bytes()
        assert len(payload) == artifact["byte_count"]
        assert sha256_of(payload) == artifact["sha256"]
        assert artifact["path"] == str(directory / artifact["file"])
        assert artifact["verified_after_write"] is True
    assert export.directory == directory
    assert export.point_count == sum(len(cloud) for cloud in capture.clouds)
    assert export.face_count > 0


def test_the_record_states_the_axis_mapping_the_frame_and_what_is_not_claimed(
    tmp_path: Path, capture: Capture
) -> None:
    """A consumer can undo the presentation rotation from the record alone.

    The exported frame is a decision this repository made, not one the scanner
    or the file format declares, so the record has to carry the rotation, the
    axis-by-axis mapping back to the optical frame, and the statement that no
    dimension in the file is a measured length.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    record = export.record

    convention = section(record, "coordinate_convention")
    assert convention["axis_mapping"] == {
        "x": "+x of the depth camera optical frame",
        "y": "-y of the depth camera optical frame",
        "z": "-z of the depth camera optical frame",
    }
    assert convention["rotation_applied"] == [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    assert convention["registration_frame_unchanged"] is True
    assert record["dimensional_accuracy_claimed"] is False
    assert "no dimensional accuracy is claimed" in str(record["dimensional_accuracy_disclaimer"])


# -- the surface is the fused one ------------------------------------------


def test_the_exported_mesh_is_the_fused_surface_and_not_the_occupancy_shrink_wrap(
    tmp_path: Path, capture: Capture
) -> None:
    """The whole reason this task exists, measured against known truth.

    An occupancy reconstruction marks a cell solid when any sample lands in it,
    so its boundary CONTAINS the samples and stands off the object by most of a
    voxel in every direction. The fused surface interpolates the zero crossing
    between voxel centres instead. Both are run here at the SAME voxel size, on
    the same placed points, so the comparison cannot be won by resolution - and
    what is asserted is a margin far outside any rounding, because a
    reconstruction that were merely as good would not be worth changing method
    for.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)

    voxel = export.resolution.voxel_size
    fused = radial_error([optical(vertex) for vertex in export.surface.vertices])
    placed = tuple(model_points(capture.clouds, capture.poses))
    occupancy = radial_error(list(reconstruct(placed, voxel_size=voxel).vertices))
    assert fused[0] * 4.0 < occupancy[0], (
        f"median distance to the true sphere: fused {fused[0]!r} vs occupancy {occupancy[0]!r} "
        f"at voxel {voxel!r}"
    )
    assert fused[1] * 4.0 < occupancy[1], (
        f"rms distance to the true sphere: fused {fused[1]!r} vs occupancy {occupancy[1]!r} "
        f"at voxel {voxel!r}"
    )
    # And an absolute claim as well as a relative one: a surface that beat
    # occupancy by sitting somewhere else entirely would pass the two above.
    assert fused[0] < 0.2 * voxel


def test_the_mesh_is_turned_upright_exactly_once_and_the_winding_is_left_alone(
    tmp_path: Path, capture: Capture
) -> None:
    """The vertices move; the faces do not, and the surface stays outward.

    The half turn is a PROPER rotation, so it preserves orientation. Reversing
    the faces to "compensate" - right for a mirror, wrong for a rotation - would
    turn every normal in the file inward, and nothing downstream that draws a
    surface can tell that from a surface extracted inside out. So the test is
    both halves: the exported sphere is centred where the rotation puts it, and
    every face still faces away from that centre.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    surface = export.surface

    centre = UPRIGHT_FROM_OPTICAL.apply(SPHERE_CENTRE)
    assert centre == (0.0, -0.0, -90.0)
    for vertex in surface.vertices:
        assert abs(math.dist(vertex, centre) - SPHERE_RADIUS) < 1.0
    # Un-rotated, the same vertices would sit around (0, 0, +90) instead.
    assert all(math.dist(vertex, SPHERE_CENTRE) > SPHERE_RADIUS for vertex in surface.vertices)
    outward = 0
    for a, b, c in surface.faces:
        normal = cross(
            subtract(surface.vertices[b], surface.vertices[a]),
            subtract(surface.vertices[c], surface.vertices[a]),
        )
        middle = tuple(
            (surface.vertices[a][axis] + surface.vertices[b][axis] + surface.vertices[c][axis])
            / 3.0
            for axis in range(3)
        )
        if dot(normal, subtract(middle, centre)) > 0.0:  # type: ignore[arg-type]
            outward += 1
    assert outward == surface.face_count, (
        f"{surface.face_count - outward} of {surface.face_count} faces point inward, which is "
        "what reversing the winding after a proper rotation does"
    )


def test_upright_mesh_moves_the_vertices_and_returns_the_faces_untouched() -> None:
    """The rotation seam on its own, without an export around it.

    Stated as a unit claim because it is the one operation in this module that a
    future edit could "fix" by flipping the faces, and an export-level test
    would then fail somewhere that does not name the cause.
    """
    from scanner3d.mesh import Mesh

    mesh = Mesh(
        vertices=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0)),
        faces=((0, 1, 2),),
        limitations=("carried through",),
    )

    turned = upright_mesh(mesh)

    assert turned.vertices == ((1.0, -2.0, -3.0), (4.0, -5.0, -6.0), (7.0, -8.0, -10.0))
    assert turned.faces == mesh.faces
    assert turned.limitations == mesh.limitations
    assert mesh.vertices == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0))


def test_the_field_is_measured_and_fused_in_the_model_frame_not_the_upright_one(
    tmp_path: Path, capture: Capture
) -> None:
    """Where the bounds come from, because the wrong frame still looks like a scan.

    The poses map camera coordinates into the MODEL frame and the fusion inverts
    them. A field measured around the upright points would be half a turn away
    from every camera it was supposed to see, so the bounds this reports have to
    be the model-frame ones - which for this sphere means a positive z, the
    opposite sign from everything in the exported files.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    resolution = export.resolution

    assert resolution.minimum[2] > 0.0 and resolution.maximum[2] > 0.0
    assert resolution.minimum[2] == pytest.approx(SPHERE_CENTRE[2] - SPHERE_RADIUS, abs=1.0)
    assert resolution.maximum[2] == pytest.approx(SPHERE_CENTRE[2] + SPHERE_RADIUS, abs=1.0)
    bounds = resolution.bounds
    assert bounds.min_z == pytest.approx(resolution.minimum[2] - resolution.margin)
    assert bounds.max_z == pytest.approx(resolution.maximum[2] + resolution.margin)
    config = export.integration.config
    assert config.bounds == bounds
    extent = export.surface.bounds
    assert extent is not None
    assert extent[1][2] < 0.0


# -- resolution ------------------------------------------------------------


def test_the_resolution_is_derived_deterministically_and_fits_the_budget(
    capture: Capture,
) -> None:
    """Same points, same budget, same answer - and it allocates nothing.

    A resolution nobody can re-derive is a number in a file rather than a
    parameter, so this asserts both halves: that the walk is a function of its
    inputs, and that the shape it predicts is the shape the field will really
    have, including the margin and the truncation the export fixes.
    """
    placed = tuple(model_points(capture.clouds, capture.poses))

    first = choose_tsdf_resolution(placed, voxel_budget=BUDGET)
    again = choose_tsdf_resolution(placed, voxel_budget=BUDGET)

    assert first == again
    assert first.predicted_voxels <= BUDGET
    assert math.prod(first.grid_shape) == first.predicted_voxels
    assert first.truncation == TRUNCATION_VOXELS * first.voxel_size
    assert first.margin == MARGIN_VOXELS * first.voxel_size
    assert first.coarsening_steps > 0
    # The prediction is what the field really allocates, not an estimate of it.
    from scanner3d.tsdf import TsdfConfig

    config = TsdfConfig.of(
        bounds=first.bounds,
        voxel_size=first.voxel_size,
        truncation=first.truncation,
        min_depth=TSDF_MIN_DEPTH,
        max_depth=TSDF_MAX_DEPTH,
        max_voxels=BUDGET,
    )
    assert config.shape == first.grid_shape
    assert config.cell_count == first.predicted_voxels


def test_a_larger_budget_buys_a_finer_voxel_and_the_module_ceiling_is_never_widened(
    capture: Capture,
) -> None:
    """A budget is a ceiling a caller may lower and may never raise.

    Both directions matter. A smaller budget must genuinely coarsen the field,
    or the parameter is decoration; and a budget past the TSDF module's own must
    be refused, because a ceiling an argument can widen is not a ceiling.
    """
    placed = tuple(model_points(capture.clouds, capture.poses))

    coarse = choose_tsdf_resolution(placed, voxel_budget=BUDGET // 8)
    fine = choose_tsdf_resolution(placed, voxel_budget=BUDGET)

    assert coarse.voxel_size > fine.voxel_size
    assert coarse.predicted_voxels <= BUDGET // 8
    with pytest.raises(GeometryNotExportable, match="never widened"):
        choose_tsdf_resolution(placed, voxel_budget=MAX_VOXELS + 1)
    with pytest.raises(GeometryNotExportable, match="never widened"):
        choose_tsdf_resolution(placed, voxel_budget=0)


def test_the_record_states_the_resolution_the_field_and_what_was_fused(
    tmp_path: Path, capture: Capture
) -> None:
    """Everything the manifest has to carry about the fusion, in one place.

    A reader deciding whether to trust this mesh needs what it was built from -
    how many raw frames, through which intrinsics and scale, at which pose
    direction - and what the field and the extraction actually did. None of it
    can be re-derived from the geometry files, which is the whole reason the
    record exists.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    mesh = section(export.record, "mesh")
    tsdf = mesh["tsdf"]

    assert mesh["method"] == MESH_METHOD
    assert mesh["method_family"] == str(MeshMethod.TSDF)
    assert mesh["occupancy_fallback_used"] is False
    assert tsdf["raw_frames_fused"] == capture.frames
    assert tsdf["raw_frame_bytes"] == EXPECTED_FRAME_BYTES
    assert tsdf["depth_scale"] == DEPTH_SCALE
    assert tsdf["depth_intrinsics"]["values"]["fx"] == 400.0
    assert "CAMERA coordinates into the model frame" in tsdf["pose_direction"]
    assert tsdf["upright_rotation_applied_after_extraction"] is True
    assert tsdf["faces_reversed_by_the_rotation"] is False
    assert tsdf["min_weight"] == TSDF_MIN_WEIGHT

    field = tsdf["integration"]
    assert field["backend"] == export.integration.backend
    assert field["frames"]["offered"] == capture.frames
    assert field["frames"]["integrated"] == capture.frames
    assert field["config"]["bounds"]["min"] == list(export.resolution.bounds.minimum)
    assert field["config"]["depth"]["min_depth"] == TSDF_MIN_DEPTH
    assert field["config"]["depth"]["max_depth"] == TSDF_MAX_DEPTH
    assert field["config"]["grid"]["cells"] == export.resolution.predicted_voxels
    assert field["weights"]["observed_voxels"] > 0
    accounting = field["accounting"]
    assert (
        accounting["voxel_updates"] + sum(entry["count"] for entry in accounting["ignored"])
        == accounting["voxel_observations"]
    )

    extraction = tsdf["extraction"]
    assert extraction["min_weight"] == TSDF_MIN_WEIGHT
    assert extraction["backend"] == export.extraction.backend
    assert extraction["cubes"]["active"] > 0
    assert extraction["mesh"]["face_count"] == export.face_count
    assert extraction["mesh"]["edges"]["edge_count"] > 0

    resolution = mesh["resolution"]
    assert resolution["voxel_size"] == export.resolution.voxel_size
    assert resolution["truncation_in_voxels"] == TRUNCATION_VOXELS
    assert resolution["margin_in_voxels"] == MARGIN_VOXELS
    assert resolution["module_max_voxels"] == MAX_VOXELS
    assert "ORIGINAL poses" in resolution["measured_in"]
    caveats = " ".join(tsdf["limitations"])
    assert "no dimensional accuracy is claimed" in caveats
    assert "poses are consumed exactly as given" in caveats
    assert "unestablished" in caveats


def test_the_record_does_not_ship_the_sample_bag_caveat_of_a_mesh_it_did_not_write(
    tmp_path: Path, capture: Capture
) -> None:
    """The field's caveats travel; the caveat about UNCONNECTED points does not.

    :mod:`scanner3d.tsdf` splits them apart precisely so a consumer that turns
    the field into a connected mesh can carry everything true of the field
    without also telling its reader that what they hold has no topology, no
    orientation and no watertightness. Shipping that here would be false, and
    the substitution has to be visible in the record rather than silent.
    """
    from scanner3d.tsdf import FIELD_LIMITATIONS, UNCONNECTED_SAMPLES_LIMITATION

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    payload = (export.directory / RECORD_NAME).read_bytes().decode("ascii")
    field = section(export.record, "mesh", "tsdf", "integration")

    assert field["limitations"] == list(FIELD_LIMITATIONS)
    assert "deliberately absent" in field["limitations_note"]
    assert squeezed(UNCONNECTED_SAMPLES_LIMITATION) not in squeezed(payload)


# -- the two files and the object in memory are one surface ----------------


def test_both_mesh_files_and_the_returned_surface_are_the_same_geometry(
    tmp_path: Path, capture: Capture
) -> None:
    """One surface, three renderings of it, and no chance for them to differ.

    A caller that draws :attr:`ReconstructionExport.surface` and a viewer that
    opens the PLY have to be looking at the same object; a window showing
    something the file does not contain is worse than a window showing nothing.
    The PLY carries binary32 coordinates, so the comparison is made at that
    width rather than at Python's.
    """
    from scanner3d.mesh import binary32

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)

    ply_vertices, ply_faces = read_ply_mesh((export.directory / MESH_PLY_NAME).read_bytes())
    obj_vertices, obj_faces = read_obj((export.directory / MESH_OBJ_NAME).read_bytes())
    assert ply_faces == obj_faces
    assert [tuple(face) for face in ply_faces] == [tuple(face) for face in export.surface.faces]
    expected = [tuple(binary32(value) for value in vertex) for vertex in export.surface.vertices]
    assert ply_vertices == expected
    assert obj_vertices == pytest.approx(expected)
    assert export.vertex_count == len(expected)


def test_the_registered_cloud_is_still_the_union_of_the_clouds_it_was_given(
    tmp_path: Path, capture: Capture
) -> None:
    """The mesh changed method; the raw evidence artifact did not.

    The cloud is the union of the placed samples, point for point and in order,
    because it is the raw evidence and fusing it would make it an opinion. The
    two artifacts therefore describe the object in two different ways
    deliberately, and this is the test that stops the cloud from quietly
    becoming a by-product of the field.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)

    from scanner3d.mesh import binary32

    written = read_ply_points((export.directory / CLOUD_NAME).read_bytes())
    expected = [
        tuple(binary32(value) for value in point)
        for point in registered_points(capture.clouds, capture.poses)
    ]
    assert written == expected


# -- the caveats are the fused ones ----------------------------------------


def test_the_files_and_the_record_carry_the_fused_caveats_and_none_of_occupancys(
    tmp_path: Path, capture: Capture
) -> None:
    """A caveat about a reconstruction that did not happen is a false statement.

    Occupancy's inherent caveats describe a solid that contains its samples, an
    iso level counted over a node's cells, a six-connected fill deciding an
    interior and a padded grid that always closes. None of that happened here,
    and a file that claimed it would tell its reader this surface carries
    occupancy's deterministic outward offset - which is that construction's
    defect and not this one's. Both mesh files and the JSON are checked, because
    three artifacts that disagree about what produced them are worse than one.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)
    ply = comments((export.directory / MESH_PLY_NAME).read_bytes())
    obj = comments((export.directory / MESH_OBJ_NAME).read_bytes())
    payload = (export.directory / RECORD_NAME).read_bytes().decode("ascii")

    occupancy_only = set(OCCUPANCY_LIMITATIONS) - set(TSDF_INHERENT_LIMITATIONS)
    assert len(occupancy_only) == 6
    for caveat in occupancy_only:
        for where, text in (("ply", ply), ("obj", obj), ("record", payload)):
            assert squeezed(caveat) not in squeezed(text), f"{where} claims an occupancy caveat"
    for caveat in TSDF_INHERENT_LIMITATIONS:
        assert squeezed(caveat) in squeezed(ply)
        assert squeezed(caveat) in squeezed(obj)
        assert squeezed(caveat) in squeezed(payload)
    assert squeezed(MESH_METHOD) in squeezed(ply)
    assert squeezed(MESH_METHOD) in squeezed(obj)
    # An iso level is an occupancy parameter - how many of a node's eight cells
    # had to be solid - and there is no such quantity here, so no artifact
    # carries the LABEL. The words themselves still appear, inside the caveat
    # that says none was counted, which is why what is searched for is the
    # label's colon; and the level that does exist, the fixed zero of a signed
    # distance, is stated in its place.
    for where, text in (("ply", ply), ("obj", obj)):
        assert squeezed("iso level:") not in squeezed(text), f"{where} states an iso level"
        assert squeezed("level set: zero") in squeezed(text)
    assert '"iso_level"' not in payload, "the record states an occupancy iso level"
    provenance = section(export.record, "mesh", "ply", "provenance")
    assert "iso_level" not in provenance
    assert provenance["level_set"] == 0.0
    assert section(export.record, "mesh", "obj", "provenance")["level_set"] == 0.0
    assert provenance["method_family"] == "tsdf"
    assert list(export.mesh_ply.limitations) == section(export.record, "limitations")
    assert export.mesh_ply.limitations == export.mesh_obj.limitations


# -- second runs and collisions --------------------------------------------


def test_a_second_reconstruction_makes_a_second_folder_and_changes_nothing_in_the_first(
    tmp_path: Path, capture: Capture
) -> None:
    """A prior reconstruction is evidence, and evidence is never written into."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    first = run(frames_dir, capture)
    before = {path.name: path.read_bytes() for path in first.directory.iterdir()}
    second = run(frames_dir, capture)

    assert second.directory != first.directory
    assert second.directory.name == f"{OUTPUT_PREFIX}0002"
    assert {path.name: path.read_bytes() for path in first.directory.iterdir()} == before


def test_an_existing_name_is_stepped_over_and_a_symlink_is_never_followed(
    tmp_path: Path, capture: Capture
) -> None:
    """``mkdir`` is the only test of a free name that cannot be raced.

    It fails on an existing directory, on an existing file, and on a SYMLINK of
    that name rather than following it out of the capture folder into somewhere
    this was never asked to write.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (frames_dir / f"{OUTPUT_PREFIX}0001").mkdir()
    (frames_dir / f"{OUTPUT_PREFIX}0002").symlink_to(elsewhere, target_is_directory=True)

    export = run(frames_dir, capture)

    assert export.directory.name == f"{OUTPUT_PREFIX}0003"
    assert list(elsewhere.iterdir()) == []


# -- nothing half-written --------------------------------------------------


def test_cancellation_partway_through_publishes_nothing(tmp_path: Path, capture: Capture) -> None:
    """Stop during the writing leaves no staging directory and no reserved name."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    def stop_at_the_mesh(message: str) -> None:
        if MESH_PLY_NAME in message:
            raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run(frames_dir, capture, on_progress=stop_at_the_mesh)

    assert leftovers(frames_dir) == []


def test_cancellation_during_the_fusion_stops_between_frames_and_publishes_nothing(
    tmp_path: Path, capture: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop is reachable WHILE the field is being fused, not only before it.

    Fusing is the slowest thing this module does and the only stage whose cost
    grows with the capture, so it is the stage an operator is most likely to
    stop in. A callback that fired once before the first projection and not
    again until the whole sequence was finished would make Stop unreachable for
    exactly the minutes it is wanted in, so what is proved here is that the
    fusion had genuinely started - one frame changed the field - and genuinely
    stopped - the remaining frames were never offered.

    Nothing has been created by then, so the assertion about the folder is
    stronger than "cleaned up": it was never touched at all.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    seen: list[str] = []
    built: list[TsdfVolume] = []

    def spy(*args: object, **kwargs: object) -> TsdfVolume:
        """The real volume, kept where the test can measure what it fused."""
        volume = TsdfVolume(*args, **kwargs)  # type: ignore[arg-type]
        built.append(volume)
        return volume

    monkeypatch.setattr("scanner3d.reconstruction_export.TsdfVolume", spy)

    def stop_after_the_first_frame(message: str) -> None:
        seen.append(message)
        if "fused frame" in message:
            raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt):
        run(frames_dir, capture, on_progress=stop_after_the_first_frame)

    assert leftovers(frames_dir) == []
    assert any("fusing at" in message for message in seen)
    assert seen[-1].strip().startswith(f"fused frame 1/{capture.frames}")
    (volume,) = built
    assert capture.frames > 1, "a one-frame capture could not tell the two failures apart"
    assert volume.frames_offered == 1, "the fusion did not stop between frames"
    assert volume.frames_integrated == 1, "the frame that did run changed nothing"


def test_a_write_that_fails_publishes_nothing(
    tmp_path: Path, capture: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in the middle of the artifacts leaves the capture folder clean."""
    from scanner3d import reconstruction_export

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise OSError("no space left on device")

    monkeypatch.setattr(reconstruction_export, "write_obj", refuse)

    with pytest.raises(OSError, match="no space left"):
        run(frames_dir, capture)

    assert leftovers(frames_dir) == []


def test_a_file_that_does_not_match_its_record_is_never_published(
    tmp_path: Path, capture: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is checked against the BYTES, not against the encoder's word."""
    from dataclasses import replace as replace_field

    from scanner3d import reconstruction_export

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    honest = write_obj

    def lie(*args: object, **kwargs: object) -> object:
        return replace_field(honest(*args, **kwargs), sha256="0" * 64)  # type: ignore[arg-type]

    monkeypatch.setattr(reconstruction_export, "write_obj", lie)

    with pytest.raises(ArtifactVerificationFailed, match="nothing"):
        run(frames_dir, capture)

    assert leftovers(frames_dir) == []


def test_no_staging_directory_survives_a_failure(
    tmp_path: Path, capture: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not even a hidden one: a half-written folder is not a smaller result."""
    from scanner3d import reconstruction_export

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    monkeypatch.setattr(
        reconstruction_export,
        "_write_record",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("interrupted")),
    )

    with pytest.raises(OSError, match="interrupted"):
        run(frames_dir, capture)

    assert not [name for name in leftovers(frames_dir) if name.startswith(STAGING_PREFIX)]
    assert leftovers(frames_dir) == []


def test_a_staging_directory_that_cannot_be_made_leaves_no_reserved_name(
    tmp_path: Path, capture: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reserved name is cleaned up even when nothing was ever staged in it."""
    import tempfile as tempfile_module

    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    def refuse(*_a: object, **_k: object) -> str:
        raise OSError("read-only file system")

    monkeypatch.setattr(tempfile_module, "mkdtemp", refuse)

    with pytest.raises(OSError, match="read-only"):
        run(frames_dir, capture)

    assert leftovers(frames_dir) == []


def test_the_published_directory_is_as_reachable_as_any_directory_this_repository_makes(
    tmp_path: Path, capture: Capture
) -> None:
    """A rename carries the SOURCE's mode, and ``mkdtemp``'s 0o700 is wrong for a result.

    The mode a published export ends up with is the one a plain ``mkdir`` in
    that folder gives - this account's umask, the same as
    :func:`scanner3d.config.allocate_capture_dir`'s own directories - and not
    the private 0o700 the staging directory was created with. Compared against a
    sibling made the same way rather than against a literal, because a test that
    hard-coded 0o755 would be asserting the umask of whoever ran it.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()

    export = run(frames_dir, capture)

    published = stat.S_IMODE(export.directory.stat().st_mode)
    assert published == stat.S_IMODE(ordinary.stat().st_mode)
    assert published != 0o700


# -- refusals, all of them before anything is created ----------------------


def test_raw_frames_clouds_and_poses_that_do_not_pair_up_are_refused(
    tmp_path: Path, capture: Capture
) -> None:
    """Never truncated to the shortest, in any of the three directions.

    Zipping them would fuse frames at other frames' poses, which is not a
    partial scan but a confidently wrong one - and it would publish it with a
    manifest that counted the frames it silently dropped.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    for kwargs in (
        {"raw_frames": capture.raw_frames[:-1]},
        {"poses": capture.poses[:-1]},
        {"clouds": capture.clouds[:-1]},
    ):
        with pytest.raises(FrameCountMismatch, match="cannot be paired"):
            run(frames_dir, capture, **kwargs)
    assert leftovers(frames_dir) == []


def test_a_registration_that_does_not_describe_these_poses_is_refused(
    tmp_path: Path, capture: Capture
) -> None:
    """The geometry and the record it is published with must be one answer.

    The poses arrive twice - as the explicit sequence that places the clouds and
    the frames, and inside the registration whose frame count, score and
    limitations the manifest serialises. Nothing forces a caller to pass the
    same ones, and a mismatch is the one defect a reader cannot detect: both
    halves of the export look internally consistent while the surface was fused
    from one answer and vouched for by another.

    The third case is the one that matters most and the one a count check cannot
    catch: a DIFFERENT orbit of exactly the same length. It is refused, named by
    the index of the first pose that disagrees, before a byte of field exists.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    recorded = registration(capture.poses)
    # Same length, same kind of thing, a different answer: an orbit about x
    # rather than the y-axis orbit these frames were placed by.
    other = orbit(len(capture.poses), axis=(1.0, 0.0, 0.0))
    first_difference = next(
        index
        for index, (offered, wrong) in enumerate(zip(capture.poses, other, strict=True))
        if offered != wrong
    )

    with pytest.raises(RegistrationMismatch, match="says 3 frames"):
        run(frames_dir, capture, registration=replace(recorded, frames=3))
    with pytest.raises(RegistrationMismatch, match="carries 7 poses"):
        run(frames_dir, capture, registration=replace(recorded, poses=capture.poses[:-1]))
    with pytest.raises(RegistrationMismatch, match=f"pose {first_difference} offered"):
        run(frames_dir, capture, registration=registration(other))

    assert leftovers(frames_dir) == []


def test_a_raw_frame_of_the_wrong_length_is_refused_before_anything_is_created(
    tmp_path: Path, capture: Capture
) -> None:
    """A buffer of another length is not this mode's frame, and is never padded.

    Named by INDEX, because a capture of a hundred frames with one short file in
    it is a question about which file, and an export that answered "one of them"
    would send its operator through the whole folder by hand.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    frames = list(capture.raw_frames)
    frames[2] = frames[2][:-2]

    with pytest.raises(CaptureInputUnusable, match="raw frame 2"):
        run(frames_dir, capture, raw_frames=tuple(frames))

    assert leftovers(frames_dir) == []


def test_calibration_the_projection_cannot_use_is_refused_before_anything_is_created(
    tmp_path: Path, capture: Capture
) -> None:
    """The intrinsics and the scale go through the deprojection's own gates.

    A pinhole projection divides by fx and fy and multiplies a raw code by the
    scale, so a zero focal length or a non-positive scale is not a degraded
    scan, it is arithmetic that means nothing. Refused with the capture folder
    untouched rather than fused into a field of nonsense.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    broken = intrinsics_of(fx=0.0, fy=400.0, cx=320.0, cy=200.0)

    with pytest.raises(CaptureInputUnusable, match="cannot be fused"):
        run(frames_dir, capture, intrinsics=broken)
    with pytest.raises(CaptureInputUnusable, match="cannot be fused"):
        run(frames_dir, capture, depth_scale=0.0)
    with pytest.raises(CaptureInputUnusable, match="cannot be fused"):
        run(frames_dir, capture, depth_scale=math.inf)

    assert leftovers(frames_dir) == []


def test_no_geometry_at_all_is_a_refusal_and_not_an_empty_export(
    tmp_path: Path, capture: Capture
) -> None:
    """Frames that carried no point are an outcome, and it is reported as one."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    with pytest.raises(NoGeometryOffered):
        run(frames_dir, capture, clouds=tuple(() for _ in capture.clouds))

    assert leftovers(frames_dir) == []


def test_a_cloud_with_no_extent_is_a_refusal(tmp_path: Path, capture: Capture) -> None:
    """A single coordinate is not an extent, and no voxel size follows from it."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    # One point, and the SAME pose for every frame: an orbit would otherwise
    # spread eight copies of one point around a circle and give the cloud a
    # perfectly good extent, which is not the case being tested.
    flat = tuple(((1.0, 2.0, 90.0),) for _ in capture.clouds)
    still = tuple(Rigid.identity() for _ in capture.poses)

    with pytest.raises(GeometryNotExportable, match="no extent"):
        run(frames_dir, capture, clouds=flat, poses=still)

    assert leftovers(frames_dir) == []


def test_frames_that_observed_nothing_are_a_refusal_and_never_an_occupancy_mesh(
    tmp_path: Path, capture: Capture
) -> None:
    """No zero crossing means no surface, and it must not be answered another way.

    This is the case a silent fallback would hide. The clouds still describe a
    sphere, so the occupancy path would happily produce a mesh here; the fused
    field, offered frames in which every sample is "no return", correctly has
    nothing in it. Publishing the occupancy answer under the fused method's name
    would make two exports of one capture incomparable while looking identical.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    blank = tuple(bytes(EXPECTED_FRAME_BYTES) for _ in capture.raw_frames)

    with pytest.raises(TsdfSurfaceUnavailable, match=r"no surface|no eligible cube"):
        run(frames_dir, capture, raw_frames=blank)

    assert leftovers(frames_dir) == []


def test_a_field_past_the_reference_backends_ceiling_is_refused_rather_than_fallen_back_from(
    tmp_path: Path, capture: Capture
) -> None:
    """Without numpy, a field this size is refused - not quietly re-meshed.

    The pure-Python backend's ceiling is a wall-clock bound, and it is never
    widened at run time. What matters here is what happens INSTEAD: nothing. A
    host with no numpy gets a refusal naming the ceiling, not an occupancy mesh
    wearing the fused method's provenance.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    budget = PYTHON_BACKEND_MAX_VOXELS * 2

    with pytest.raises(TsdfSurfaceUnavailable, match="NOT attempted") as raised:
        run(frames_dir, capture, voxel_budget=budget, tsdf_backend="python")

    assert str(PYTHON_BACKEND_MAX_VOXELS) in str(raised.value)
    assert leftovers(frames_dir) == []


def test_a_budget_past_the_tsdf_modules_own_is_refused_before_anything_is_created(
    tmp_path: Path, capture: Capture
) -> None:
    """A ceiling an argument can widen is not a ceiling."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    with pytest.raises(GeometryNotExportable, match="never widened"):
        run(frames_dir, capture, voxel_budget=MAX_VOXELS + 1)

    assert leftovers(frames_dir) == []


def test_more_points_than_the_writer_will_take_is_a_refusal_before_anything_is_created(
    tmp_path: Path, capture: Capture
) -> None:
    """The cloud writer's ceiling binds the export, and is never widened here."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    with pytest.raises(GeometryNotExportable, match="point ceiling"):
        run(frames_dir, capture, max_points=10)
    with pytest.raises(GeometryNotExportable, match="never widened"):
        run(frames_dir, capture, max_points=MAX_POINTS + 1)

    assert leftovers(frames_dir) == []


# -- the registered points -------------------------------------------------


def test_the_points_are_each_frames_pose_applied_and_then_turned_upright() -> None:
    """The composition, on numbers small enough to check by hand."""
    cloud = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    pose = Rigid.of(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (10.0, 20.0, 30.0))

    placed = list(registered_points([cloud], [pose]))

    assert placed == [(11.0, -22.0, -33.0), (14.0, -25.0, -36.0)]
    assert list(model_points([cloud], [pose])) == [(11.0, 22.0, 33.0), (14.0, 25.0, 36.0)]


def test_the_presentation_rotation_is_right_handed_and_moves_no_distance() -> None:
    """Determinant +1: a rotation, not a mirror, so nothing about the shape changes."""
    from scanner3d.geometry import determinant

    assert determinant(UPRIGHT_FROM_OPTICAL.rotation) == pytest.approx(1.0)
    a, b = (1.0, 2.0, 3.0), (-4.0, 0.5, 7.0)
    assert math.dist(UPRIGHT_FROM_OPTICAL.apply(a), UPRIGHT_FROM_OPTICAL.apply(b)) == pytest.approx(
        math.dist(a, b)
    )


def test_the_export_carries_every_point_of_every_frame_it_was_given(
    tmp_path: Path, capture: Capture
) -> None:
    """Nothing is sampled, decimated or averaged on the way into the cloud file."""
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture)

    assert export.point_count == sum(len(cloud) for cloud in capture.clouds)
    assert len(read_ply_points((export.directory / CLOUD_NAME).read_bytes())) == (
        export.point_count
    )


def test_a_stated_depth_band_is_the_interval_the_field_accepts(
    tmp_path: Path, capture: Capture
) -> None:
    """What was registered is what gets fused, and the record says so.

    The registration clouds were clipped to the band; a field that still
    accepted the sensor's whole interval would fuse room the poses never
    described. The field's own configuration is the evidence. The sphere sits
    at 70-110, so a band of 60-150 holds all of it.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    export = run(frames_dir, capture, depth_band=DepthBand(near=60.0, far=150.0))
    field = section(export.record, "mesh")["tsdf"]["integration"]

    assert field["config"]["depth"]["min_depth"] == 60.0
    assert field["config"]["depth"]["max_depth"] == 150.0


def test_a_band_that_excludes_the_subject_fuses_nothing_and_says_so(
    tmp_path: Path, capture: Capture
) -> None:
    """A band beyond the sphere leaves the field no surface, and the export refuses.

    Refused rather than published as a mesh with no faces: the same rule the
    export applies to a field that never saw anything, because that is exactly
    what it is.
    """
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()

    with pytest.raises(TsdfSurfaceUnavailable):
        run(frames_dir, capture, depth_band=DepthBand(near=150.0, far=250.0))
    assert leftovers(frames_dir) == []
