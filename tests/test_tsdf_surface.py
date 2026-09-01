"""What the connected TSDF surface must be true of before anything can select it.

Two kinds of claim are checked here, and they fail in different ways.

The GEOMETRIC ones measure the recovered surface against shapes whose truth is
known exactly - a plane at a known depth, a sphere of a known radius - because
a mesh can have perfect topology and still sit half a voxel outside the object,
which is precisely the defect the occupancy path has and the whole reason this
one exists.

The TOPOLOGICAL ones are exhaustive rather than representative. Marching cubes
fails at the seam between two cells, in the ambiguous cases, and a suite that
samples a few shapes will miss it because the crack is one edge wide and the
picture still looks like a surface. So every sign pattern two neighbouring
cubes can present to their shared face is enumerated, on all three face
orientations, with and without the exact zeros the level-set policy has to
handle - and what is asserted is the property a crack would violate, not that
the output matches a recorded blob.
"""

from __future__ import annotations

import array
import math
import statistics

import pytest

from scanner3d import tsdf as tsdf_module
from scanner3d import tsdf_surface as surface_module
from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.mesh import (
    CORNER_OFFSETS,
    EDGE_TABLE,
    MAX_FACES,
    MAX_VERTICES,
    TRIANGLE_TABLE,
    Mesh,
    edge_census,
    reconstruct,
)
from scanner3d.mesh import INHERENT_LIMITATIONS as OCCUPANCY_LIMITATIONS
from scanner3d.simulate import Plane, Sphere, intrinsics_of, render
from scanner3d.tsdf import (
    AUTO_BACKEND,
    FIELD_LIMITATIONS,
    UNCONNECTED_SAMPLES_LIMITATION,
    TsdfBackendUnavailable,
    TsdfBackendUnknown,
    TsdfConfig,
    TsdfVolume,
    VolumeBounds,
)
from scanner3d.tsdf_surface import (
    EXACT_LEVEL_POLICY,
    INSIDE_CONVENTION,
    MAX_SURFACE_FACES,
    MAX_SURFACE_VERTICES,
    PYTHON_SURFACE_MAX_CELLS,
    SURFACE_CHUNK_CUBES,
    NoSurfaceFound,
    SurfaceBackendTooSmall,
    SurfaceCeilingExceeded,
    WeightThresholdUnusable,
    cube_case,
    extract_surface,
)
from tests import tsdf_benchmark

# The same centred synthetic camera the field's own suite uses, for the same
# reason: a geometric miss here has to be a defect in the surface rather than in
# the framing.
INTRINSICS = intrinsics_of(fx=400.0, fy=400.0, cx=320.0, cy=200.0)
DEPTH_SCALE = 0.004999999888241291

#: A power-of-two scale, used where the depth has to survive the uint16 round
#: trip exactly: 100.0 / 0.0078125 is 12800, and 12800 * 0.0078125 is 100.0.
EXACT_DEPTH_SCALE = 0.0078125

SPHERE_CENTRE: Point = (0.0, 0.0, 100.0)
SPHERE_RADIUS = 20.0
ORBIT_CENTRE: Point = (0.0, 0.0, 80.0)
VIEWS = 8

#: Both walks, run over the same fields wherever the field is small enough for
#: the reference path's own ceiling. The accelerated walk decides a cube's case
#: with its own inline sign test rather than through :func:`cube_case`, so a
#: geometric claim proved on one walk is not proved on the other - and the
#: defect that would hide there is a surface wound inside out, which nothing
#: downstream can detect.
BOTH_WALKS = ("numpy", "python")


def orbit_poses(count: int = VIEWS, *, axis: Point = (0.0, 1.0, 0.0)) -> list[Rigid]:
    return [
        rotation_about_axis(axis, 2.0 * math.pi * index / count, through=ORBIT_CENTRE)
        for index in range(count)
    ]


def sphere_views(
    *, jitter: float, axes: tuple[Point, ...] = ((0.0, 1.0, 0.0),), count: int = VIEWS
):
    """Frames and camera-to-model poses for a sphere seen from one or more orbits.

    More than one orbit axis is not decoration: a single equatorial orbit never
    sees the poles, so the field's evidence genuinely stops there and the
    surface is genuinely open. Whether a surface closes has to be a statement
    about the evidence, so both cases need building deliberately.
    """
    scene: list[Rigid] = []
    for axis in axes:
        scene.extend(orbit_poses(count, axis=axis))
    frames = [
        render(
            [Sphere(centre=SPHERE_CENTRE, radius=SPHERE_RADIUS)],
            intrinsics=INTRINSICS,
            depth_scale=DEPTH_SCALE,
            pose=pose,
            jitter=jitter,
            seed=index,
        ).raw
        for index, pose in enumerate(scene)
    ]
    return frames, [pose.inverse() for pose in scene]


def sphere_config(**overrides: object) -> TsdfConfig:
    settings: dict[str, object] = {
        "bounds": VolumeBounds.of((-25.0, -25.0, 75.0), (25.0, 25.0, 125.0)),
        "voxel_size": 1.0,
        "truncation": 3.0,
        "min_depth": 30.0,
        "max_depth": 200.0,
    }
    settings.update(overrides)
    return TsdfConfig.of(**settings)  # type: ignore[arg-type]


def fuse(frames, poses, *, config=None, backend: str = AUTO_BACKEND) -> TsdfVolume:
    volume = TsdfVolume(config if config is not None else sphere_config(), backend=backend)
    volume.integrate(frames, poses, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    return volume


def radial_error(points) -> tuple[float, float]:
    """``(median, rms)`` distance from the true sphere surface."""
    assert points, "nothing was recovered at all, so there is nothing to measure"
    errors = [abs(math.dist(point, SPHERE_CENTRE) - SPHERE_RADIUS) for point in points]
    return statistics.median(errors), math.sqrt(statistics.fmean(e * e for e in errors))


@pytest.fixture(scope="module")
def noisy_views():
    return sphere_views(jitter=0.5)


@pytest.fixture(scope="module")
def noisy_volume(noisy_views) -> TsdfVolume:
    frames, poses = noisy_views
    return fuse(frames, poses)


@pytest.fixture(scope="module")
def enclosed_volume() -> TsdfVolume:
    """A sphere the evidence really does surround: three orbits, twelve views each."""
    frames, poses = sphere_views(
        jitter=0.0,
        axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5773502691896258,) * 3),
        count=12,
    )
    return fuse(frames, poses)


@pytest.fixture(scope="module")
def plane_volume() -> TsdfVolume:
    """A plane at z = 100 whose crossing falls BETWEEN voxel centres."""
    frame = render(
        [Plane(normal=(0.0, 0.0, 1.0), offset=100.0)],
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
    ).raw
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-40.0, -30.0, 90.0), (40.0, 30.0, 110.0)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=50.0,
        max_depth=200.0,
    )
    return fuse([frame], [Rigid.identity()], config=config)


@pytest.fixture(scope="module")
def grid_plane_volume() -> TsdfVolume:
    """A plane at z = 100 lying EXACTLY on the k = 10 slab of voxel centres.

    89.5 + (k + 0.5) is 90.0 + k, so those voxels store an exact zero and the
    surface has to come out of the exact-level policy rather than out of a sign
    change - there is no strictly opposite-sign neighbour pair along it.
    """
    frame = render(
        [Plane(normal=(0.0, 0.0, 1.0), offset=100.0)],
        intrinsics=INTRINSICS,
        depth_scale=EXACT_DEPTH_SCALE,
    ).raw
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-20.5, -15.5, 89.5), (20.5, 15.5, 109.5)),
        voxel_size=1.0,
        truncation=3.0,
        min_depth=50.0,
        max_depth=200.0,
    )
    volume = TsdfVolume(config)
    volume.integrate(
        [frame], [Rigid.identity()], intrinsics=INTRINSICS, depth_scale=EXACT_DEPTH_SCALE
    )
    return volume


# --- building an arbitrary field ----------------------------------------------


def made_field(
    shape: tuple[int, int, int],
    values,
    weights,
    *,
    voxel: float = 1.0,
    backend: str = AUTO_BACKEND,
) -> TsdfVolume:
    """A real :class:`TsdfVolume` whose field is written directly, for the seam tests.

    The stored arrays are private on purpose - :meth:`TsdfVolume.values` hands
    out a read-only view precisely so a caller cannot do this. A test may,
    because the alternative is worse: the topology claims below are about what
    happens for EVERY sign pattern two neighbouring cubes can present, and
    there is no set of depth frames that reaches all 4096 of them. A real
    volume is still used, so the layout, the config arithmetic and the ceiling
    checks are the product's own and not a mock's.
    """
    nx, ny, nz = shape
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((0.0, 0.0, 0.0), (nx * voxel, ny * voxel, nz * voxel)),
        voxel_size=voxel,
        truncation=3.0 * voxel,
        min_depth=1.0,
        max_depth=1000.0,
    )
    assert config.shape == shape
    volume = TsdfVolume(config, backend=backend)
    volume._tsdf[:] = array.array("f", values)
    volume._weight[:] = array.array("f", weights)
    return volume


def duplicate_vertices(mesh: Mesh) -> int:
    return mesh.vertex_count - len(set(mesh.vertices))


def degenerate_faces(mesh: Mesh) -> int:
    """Faces that use one vertex twice - zero area by construction, index-wise."""
    return sum(1 for face in mesh.faces if len(set(face)) < 3)


def zero_area_faces(mesh: Mesh) -> int:
    """Faces whose three DISTINCT vertices are nonetheless collinear or coincident."""
    count = 0
    for a, b, c in mesh.faces:
        first = mesh.vertices[a]
        edge_one = tuple(mesh.vertices[b][axis] - first[axis] for axis in range(3))
        edge_two = tuple(mesh.vertices[c][axis] - first[axis] for axis in range(3))
        cross = (
            edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
            edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
            edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
        )
        if math.sqrt(sum(component * component for component in cross)) == 0.0:
            count += 1
    return count


def triangle_normal(mesh: Mesh, face: tuple[int, int, int]) -> Point:
    a, b, c = (mesh.vertices[index] for index in face)
    edge_one = tuple(b[axis] - a[axis] for axis in range(3))
    edge_two = tuple(c[axis] - a[axis] for axis in range(3))
    return (
        edge_one[1] * edge_two[2] - edge_one[2] * edge_two[1],
        edge_one[2] * edge_two[0] - edge_one[0] * edge_two[2],
        edge_one[0] * edge_two[1] - edge_one[1] * edge_two[0],
    )


# --- the read-only field access -----------------------------------------------


def test_the_field_is_handed_out_as_a_read_only_view_and_never_copied(plane_volume):
    values = plane_volume.values
    weights = plane_volume.weights
    cells = plane_volume.config.cell_count
    assert isinstance(values, memoryview) and isinstance(weights, memoryview)
    assert values.format == weights.format == "f"
    assert values.itemsize == weights.itemsize == 4
    assert len(values) == len(weights) == cells
    assert values.readonly and weights.readonly
    assert values[0] == plane_volume.sample_at(0, 0, 0).value
    assert weights[0] == plane_volume.sample_at(0, 0, 0).weight


def test_a_caller_cannot_corrupt_the_field_through_the_view(plane_volume):
    before = plane_volume.sample_at(20, 20, 10).value
    with pytest.raises(TypeError):
        plane_volume.values[0] = 1.0
    with pytest.raises(TypeError):
        plane_volume.weights[0] = 1.0
    assert plane_volume.sample_at(20, 20, 10).value == before


def test_the_view_reads_the_live_field_rather_than_a_stale_snapshot():
    frame = render(
        [Plane(normal=(0.0, 0.0, 1.0), offset=100.0)],
        intrinsics=INTRINSICS,
        depth_scale=DEPTH_SCALE,
    ).raw
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-24.0, -24.0, 88.0), (24.0, 24.0, 112.0)),
        voxel_size=3.0,
        truncation=6.0,
        min_depth=50.0,
        max_depth=200.0,
    )
    volume = TsdfVolume(config)
    view = volume.weights
    assert max(view) == 0.0
    volume.integrate([frame], [Rigid.identity()], intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE)
    assert max(view) > 0.0


# --- the case index and the side that is inside -------------------------------


def test_a_corner_is_inside_only_when_it_is_strictly_negative():
    assert cube_case([-1.0] * 8) == 255
    assert cube_case([1.0] * 8) == 0
    # Both zeros are in FRONT of the surface, so neither sets a bit. That is the
    # policy the exact-level tests below then depend on.
    assert cube_case([0.0] * 8) == 0
    assert cube_case([-0.0] * 8) == 0
    for corner in range(8):
        values = [1.0] * 8
        values[corner] = -1.0
        assert cube_case(values) == 1 << corner


# --- an exact plane, between the centres and on them --------------------------


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_plane_between_voxel_centres_is_recovered_at_the_plane(plane_volume, backend):
    surface = extract_surface(plane_volume, backend=backend)
    depths = [vertex[2] for vertex in surface.mesh.vertices]
    worst = max(abs(depth - 100.0) for depth in depths)
    # The bound is the quantisation step and float32, not an arbitrary
    # tolerance: a depth code is 0.005 units wide.
    assert worst < 0.01, f"a mesh vertex stood {worst!r} off an exactly known plane"
    assert surface.evidence.census.nonmanifold_edge_count == 0
    assert surface.evidence.census.consistently_oriented
    assert duplicate_vertices(surface.mesh) == 0
    assert degenerate_faces(surface.mesh) == 0
    assert zero_area_faces(surface.mesh) == 0


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_plane_lying_exactly_on_the_voxel_centres_still_makes_a_connected_patch(
    grid_plane_volume, backend
):
    """The case the exact-level policy exists for, end to end.

    Every voxel on the plane stores an exact zero, so there is no strictly
    opposite-sign neighbour pair anywhere along it. A rule that only followed
    sign changes would return an empty mesh here - not a worse plane, no plane.
    """
    surface = extract_surface(grid_plane_volume, backend=backend)
    mesh = surface.mesh
    assert mesh.face_count > 1000
    depths = [vertex[2] for vertex in mesh.vertices]
    voxel = grid_plane_volume.config.voxel_size
    # The clamp is the only thing that moves the patch off the centres, and it
    # moves it by exactly one part in a million of a voxel.
    assert min(depths) == pytest.approx(100.0, abs=2e-6 * voxel)
    assert max(depths) == pytest.approx(100.0, abs=2e-6 * voxel)
    assert duplicate_vertices(mesh) == 0
    assert degenerate_faces(mesh) == 0
    assert zero_area_faces(mesh) == 0
    assert surface.evidence.census.nonmanifold_edge_count == 0
    assert surface.evidence.census.consistently_oriented


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_plane_is_wound_towards_the_camera_that_observed_it(grid_plane_volume, backend):
    """Normals point towards increasing TSDF, which here is towards -z.

    The field is positive in front of the surface - nearer the camera, smaller
    z - so out of the solid is the direction the camera is in. A surface wound
    the other way exports as a solid turned inside out, and nothing downstream
    can tell.
    """
    surface = extract_surface(grid_plane_volume, backend=backend)
    normals = [triangle_normal(surface.mesh, face)[2] for face in surface.mesh.faces]
    assert normals and all(component < 0.0 for component in normals)


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_the_planar_patch_is_bounded_only_by_where_the_evidence_stops(grid_plane_volume, backend):
    surface = extract_surface(grid_plane_volume, backend=backend)
    census = surface.evidence.census
    assert census.boundary_edge_count > 0, "a plane seen through one frame cannot be closed"
    assert not surface.mesh.is_watertight
    assert census.shared_edge_count > 10 * census.boundary_edge_count, (
        "a planar patch is mostly interior; this many boundary edges means it is fragmented"
    )
    assert any("measured OPEN" in note for note in surface.mesh.limitations)
    assert any("No back wall was invented" in note for note in surface.mesh.limitations)


# --- a closed, curved surface -------------------------------------------------


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_surrounded_sphere_comes_out_watertight_and_manifold(enclosed_volume, backend):
    surface = extract_surface(enclosed_volume, backend=backend)
    mesh = surface.mesh
    census = surface.evidence.census
    assert mesh.is_watertight
    assert census.boundary_edge_count == 0
    assert census.nonmanifold_edge_count == 0
    assert census.consistently_oriented
    assert duplicate_vertices(mesh) == 0
    assert degenerate_faces(mesh) == 0
    assert zero_area_faces(mesh) == 0
    assert len(set(tuple(sorted(face)) for face in mesh.faces)) == mesh.face_count
    # Euler characteristic of a sphere, measured rather than assumed.
    assert mesh.vertex_count - census.edge_count + mesh.face_count == 2


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_closed_sphere_is_wound_outwards(enclosed_volume, backend):
    surface = extract_surface(enclosed_volume, backend=backend)
    mesh = surface.mesh
    outward = 0
    for face in mesh.faces:
        normal = triangle_normal(mesh, face)
        centre = tuple(sum(mesh.vertices[index][axis] for index in face) / 3.0 for axis in range(3))
        radial = tuple(centre[axis] - SPHERE_CENTRE[axis] for axis in range(3))
        if sum(normal[axis] * radial[axis] for axis in range(3)) > 0.0:
            outward += 1
    assert outward == mesh.face_count


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_a_sphere_seen_from_one_orbit_stays_open_and_says_so(noisy_volume, backend):
    surface = extract_surface(noisy_volume, backend=backend)
    assert not surface.mesh.is_watertight
    assert surface.evidence.census.boundary_edge_count > 0
    assert surface.evidence.census.nonmanifold_edge_count == 0
    assert any("measured OPEN" in note for note in surface.mesh.limitations)


# --- the seam between two cubes, exhaustively ---------------------------------

SEAM_LAYOUTS = (
    ("x", (3, 2, 2), 0),
    ("y", (2, 3, 2), 1),
    ("z", (2, 2, 3), 2),
)


def seam_edge_uses(surface, axis: int, seam_coordinate: float):
    """How many faces use each mesh edge that lies entirely in the shared grid face.

    A vertex sits in that plane exactly when it was cut from an edge running
    along one of the OTHER two axes at the seam index, so testing the one
    coordinate identifies them without needing to know which cube made them.
    :func:`scanner3d.mesh._fan` guarantees no triangle's diagonal joins two cut
    edges of one grid face, so an edge between two such vertices can only be
    the contour segment itself - which both cubes must draw.
    """
    mesh = surface.mesh
    on_seam = {
        index
        for index, vertex in enumerate(mesh.vertices)
        if vertex[axis] == pytest.approx(seam_coordinate, abs=1e-9)
    }
    uses: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if start in on_seam and end in on_seam:
                key = (start, end) if start < end else (end, start)
                uses[key] = uses.get(key, 0) + 1
    return uses


@pytest.mark.parametrize("label,shape,axis", SEAM_LAYOUTS)
@pytest.mark.parametrize("front", [1.0, 0.0])
def test_two_cubes_never_crack_on_the_face_they_share(label, shape, axis, front):
    """Every one of the 4096 sign patterns two adjacent cubes can present.

    ``front`` is the value given to the corners that are NOT inside: 1.0 is an
    ordinary positive sample, and 0.0 is the level set landing exactly on the
    voxel centre, which the policy says counts as being in front. Running the
    whole enumeration twice is what makes the exact-zero variant exhaustive
    rather than sampled.

    What is asserted is the property a crack would violate: every mesh edge
    lying in the shared grid face is used by exactly two triangles, one from
    each cube. If the two cubes resolved an ambiguous face differently, one of
    them would draw a segment the other did not and that edge would be used
    once.
    """
    cells = shape[0] * shape[1] * shape[2]
    volume = made_field(shape, [0.0] * cells, [1.0] * cells, backend="python")
    seam = volume.config.voxel_centre(*[1 if index == axis else 0 for index in range(3)])[axis]
    weights = array.array("f", [1.0] * cells)
    empty = 0
    for pattern in range(1 << cells):
        values = array.array("f", [-1.0 if pattern >> cell & 1 else front for cell in range(cells)])
        volume._tsdf[:] = values
        volume._weight[:] = weights
        try:
            surface = extract_surface(volume, backend="python")
        except NoSurfaceFound:
            empty += 1
            continue
        for key, count in seam_edge_uses(surface, axis, seam).items():
            assert count == 2, f"{label} seam cracked on pattern {pattern:012b} at edge {key}"
        assert surface.evidence.census.nonmanifold_edge_count == 0, f"pattern {pattern:012b}"
        assert surface.evidence.census.consistently_oriented, f"pattern {pattern:012b}"
        assert duplicate_vertices(surface.mesh) == 0, f"pattern {pattern:012b}"
        assert degenerate_faces(surface.mesh) == 0, f"pattern {pattern:012b}"
        assert zero_area_faces(surface.mesh) == 0, f"pattern {pattern:012b}"
    # The two all-one-sign patterns produce nothing, and with front == 0.0 the
    # all-front one is the all-zero field. Anything more than that would mean
    # the enumeration was not reaching the cases it claims to.
    assert empty == 2, f"{empty} of {1 << cells} patterns produced no surface at all"


def test_a_vertex_on_a_shared_grid_edge_is_one_vertex_for_every_cube_that_meets_on_it():
    """Four cubes meet on an interior grid edge; the mesh must have one vertex there.

    Without shared indexing the surface is a soup of unattached triangles that
    an edge count cannot measure, and :attr:`Mesh.is_watertight` becomes a coin
    toss. A 3x3x3 field with a single negative voxel in the middle puts twelve
    cut edges into eight cubes at once.
    """
    cells = 27
    values = [1.0] * cells
    values[(1 * 3 + 1) * 3 + 1] = -1.0
    volume = made_field((3, 3, 3), values, [1.0] * cells, backend="python")
    surface = extract_surface(volume, backend="python")
    mesh = surface.mesh
    assert mesh.vertex_count == 6, "one vertex per cut grid edge around the negative voxel"
    assert duplicate_vertices(mesh) == 0
    assert mesh.is_watertight, "a single interior negative voxel is enclosed by its neighbours"
    assert surface.evidence.census.consistently_oriented


@pytest.mark.parametrize("front", [1.0, 0.0])
def test_both_walks_read_the_same_sign_rule_for_every_one_of_the_256_cases(front):
    """One cube, every corner sign pattern, both walks, both zeros.

    The accelerated walk cannot call :func:`cube_case` - it decides tens of
    thousands of cubes at a time and a per-cube Python call would undo the
    vectorisation - so the rule "a bit is set for a strictly negative value"
    exists twice in the module. This is the test that stops the two copies
    drifting, and it is exhaustive because the space is only 256 wide: there is
    no excuse for sampling it.

    ``front`` of 0.0 runs the whole enumeration again with every non-inside
    corner holding an exact zero, which is the value the two implementations
    could most easily disagree about - ``-0.0 > 0.0`` and ``-0.0 < 0.0`` are
    both false, and they have to stay that way together.
    """
    for pattern in range(256):
        values = [-1.0 if pattern >> corner & 1 else front for corner in range(8)]
        # The cube's corners in the table's numbering, laid into the volume's
        # own (i, j, k) order, so the case the walk computes is comparable with
        # the one cube_case computes from the same eight numbers.
        field = [0.0] * 8
        for corner, (dx, dy, dz) in enumerate(CORNER_OFFSETS):
            field[(dx * 2 + dy) * 2 + dz] = values[corner]
        volume = made_field((2, 2, 2), field, [1.0] * 8, backend="numpy")
        expected = cube_case(values)
        assert (EDGE_TABLE[expected] != 0) == (pattern not in {0, 255}), pattern
        if not EDGE_TABLE[expected]:
            for walk in BOTH_WALKS:
                with pytest.raises(NoSurfaceFound):
                    extract_surface(volume, backend=walk)
            continue
        accelerated = extract_surface(volume, backend="numpy")
        reference = extract_surface(volume, backend="python")
        assert accelerated.mesh.vertices == reference.mesh.vertices, pattern
        assert accelerated.mesh.faces == reference.mesh.faces, pattern
        assert accelerated.mesh.face_count == len(TRIANGLE_TABLE[expected]), pattern
        assert accelerated.evidence.cubes_active == 1, pattern


def test_a_case_that_cuts_no_edge_is_a_refusal_rather_than_an_empty_mesh():
    """Fully observed, every corner in front of the surface, nothing crossing."""
    volume = made_field((2, 2, 2), [1.0] * 8, [1.0] * 8, backend="python")
    with pytest.raises(NoSurfaceFound) as raised:
        extract_surface(volume, backend="python")
    assert "1 cubes were examined and 1 had all eight corners observed" in str(raised.value)


# --- the evidence gate --------------------------------------------------------


def test_a_cube_with_one_corner_below_the_threshold_is_skipped_whole():
    """Nothing is interpolated across a voxel the scan did not observe well enough."""
    cells = 8
    values = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    volume = made_field((2, 2, 2), values, [1.0] * cells, backend="python")
    assert extract_surface(volume, backend="python").evidence.cubes_active == 1

    volume = made_field((2, 2, 2), values, [1.0] * 7 + [0.0], backend="python")
    with pytest.raises(NoSurfaceFound) as raised:
        extract_surface(volume, backend="python")
    assert "1 cubes were examined and 0 had all eight corners observed" in str(raised.value)


def test_raising_the_threshold_narrows_the_surface_to_the_better_observed_part(noisy_volume):
    generous = extract_surface(noisy_volume, min_weight=1.0).evidence
    strict = extract_surface(noisy_volume, min_weight=4.0).evidence
    assert 0 < strict.cubes_eligible < generous.cubes_eligible
    assert 0 < strict.vertex_count < generous.vertex_count
    assert strict.cubes_excluded_insufficient_weight > generous.cubes_excluded_insufficient_weight
    assert (
        generous.cubes_examined
        == generous.cubes_eligible + generous.cubes_excluded_insufficient_weight
    )
    assert (
        generous.cubes_eligible == generous.cubes_active + generous.cubes_eligible_without_crossing
    )


def test_an_unobserved_field_yields_no_surface_at_all():
    """The stored zero of a voxel no frame saw is the absence of evidence.

    A threshold of zero would turn a 128 MiB block of unwritten memory into a
    confident closed object, so it is refused outright rather than merely
    discouraged.
    """
    volume = TsdfVolume(sphere_config())
    with pytest.raises(NoSurfaceFound) as raised:
        extract_surface(volume)
    assert "no eligible cube" in str(raised.value)
    assert "0 had all eight corners observed" in str(raised.value)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(WeightThresholdUnusable):
            extract_surface(volume, min_weight=bad)


def test_a_field_too_thin_on_one_axis_has_no_cube_to_march():
    volume = made_field((4, 4, 1), [1.0] * 16, [1.0] * 16, backend="python")
    with pytest.raises(NoSurfaceFound) as raised:
        extract_surface(volume, backend="python")
    assert "0 cubes were examined" in str(raised.value)


def test_the_walk_costs_the_same_whether_one_frame_or_sixteen_were_fused(noisy_views):
    """Extraction is a function of the FIELD, not of the capture that made it.

    This is the property that makes fusion worth having and it has to survive
    into the surface: a volume's memory and the work of walking it are fixed by
    its shape, so a longer capture costs more integration and not a larger
    extraction. Measured as identical cube counts rather than as a timing,
    because a timing would be a property of this host.
    """
    frames, poses = noisy_views
    brief = extract_surface(fuse(frames[:1], poses[:1])).evidence
    whole = extract_surface(fuse(frames, poses)).evidence
    assert brief.cubes_examined == whole.cubes_examined
    assert brief.config.cell_count == whole.config.cell_count
    # The evidence gate does move - more frames observe more voxels - which is
    # what makes the equality above a statement about the walk and not about
    # the two fields being the same field.
    assert brief.cubes_eligible < whole.cubes_eligible


# --- geometry against the occupancy surface -----------------------------------


def test_the_connected_surface_keeps_the_advantage_the_field_measured(noisy_views, noisy_volume):
    """The comparison this whole line of work turns on, now on a real mesh.

    The unconnected zero-level samples were already shown to beat the occupancy
    surface. What has to be shown here is that TRIANGULATING them did not throw
    that away - that the vertices of the connected mesh are still where the
    field said the surface was, and have not been pushed outward or roughened
    by the marching.
    """
    frames, poses = noisy_views
    voxel = noisy_volume.config.voxel_size
    mesh_error = radial_error(extract_surface(noisy_volume).mesh.vertices)
    samples_error = radial_error(noisy_volume.zero_level_samples().points)

    model_points: list[Point] = []
    for raw, pose in zip(frames, poses, strict=True):
        conversion = deproject_depth_frame(
            raw, intrinsics=INTRINSICS, depth_scale=DEPTH_SCALE, unit=UNVALIDATED_UNIT
        )
        model_points.extend(pose.apply_all(camera_points(conversion)))
    occupancy_error = radial_error(reconstruct(model_points, voxel_size=voxel).vertices)

    assert mesh_error[0] < 0.5 * occupancy_error[0], (
        f"median distance to truth: tsdf mesh {mesh_error[0]!r} vs occupancy {occupancy_error[0]!r}"
    )
    assert mesh_error[1] < 0.5 * occupancy_error[1], (
        f"rms distance to truth: tsdf mesh {mesh_error[1]!r} vs occupancy {occupancy_error[1]!r}"
    )
    # The occupancy surface really does stand outside the samples; this one does
    # not, and both halves of that are the claim.
    assert occupancy_error[0] > 0.5 * voxel
    assert mesh_error[0] < 0.15 * voxel

    # And triangulation did not roughen what the field already had. The mesh
    # vertices are a different population from the sample points - they lie on
    # cube edges rather than on every crossing - so exact equality is not the
    # claim; not losing the accuracy is.
    assert mesh_error[0] < 1.25 * samples_error[0], (
        f"marching moved the median from {samples_error[0]!r} to {mesh_error[0]!r}"
    )
    assert mesh_error[1] < 1.25 * samples_error[1], (
        f"marching moved the rms from {samples_error[1]!r} to {mesh_error[1]!r}"
    )


@pytest.mark.parametrize("backend", BOTH_WALKS)
def test_no_vertex_wanders_off_the_sphere(noisy_volume, backend):
    surface = extract_surface(noisy_volume, backend=backend)
    voxel = noisy_volume.config.voxel_size
    worst = max(
        abs(math.dist(vertex, SPHERE_CENTRE) - SPHERE_RADIUS) for vertex in surface.mesh.vertices
    )
    assert worst < 1.5 * voxel, f"a mesh vertex stood {worst!r} off the sphere"


# --- determinism and the two backends -----------------------------------------


def coarse_config() -> TsdfConfig:
    config = TsdfConfig.of(
        bounds=VolumeBounds.of((-24.0, -24.0, 76.0), (24.0, 24.0, 124.0)),
        voxel_size=3.0,
        truncation=6.0,
        min_depth=30.0,
        max_depth=200.0,
    )
    assert config.cell_count <= PYTHON_SURFACE_MAX_CELLS
    return config


@pytest.fixture(scope="module")
def coarse_volume(noisy_views) -> TsdfVolume:
    frames, poses = noisy_views
    return fuse(frames, poses, config=coarse_config())


def test_the_two_backends_produce_the_same_mesh_from_the_same_field(coarse_volume):
    """One field, two walks over it: any divergence is a defect in one of them."""
    accelerated = extract_surface(coarse_volume, backend="numpy")
    reference = extract_surface(coarse_volume, backend="python")
    assert accelerated.mesh.vertices == reference.mesh.vertices
    assert accelerated.mesh.faces == reference.mesh.faces
    assert accelerated.mesh.limitations == reference.mesh.limitations
    assert accelerated.evidence.vertex_count > 0
    one, two = accelerated.evidence.as_dict(), reference.evidence.as_dict()
    # The backend is the one thing that legitimately differs: it is the record
    # of which path ran, not a property of the surface.
    assert one.pop("backend") == "numpy"
    assert two.pop("backend") == "python"
    assert one == two


def test_the_same_field_extracts_identically_every_time(noisy_volume):
    first = extract_surface(noisy_volume)
    second = extract_surface(noisy_volume)
    assert first.mesh.vertices == second.mesh.vertices
    assert first.mesh.faces == second.mesh.faces
    assert first.evidence.as_dict() == second.evidence.as_dict()


def test_the_backend_defaults_to_the_volumes_own_and_an_explicit_name_is_honoured(coarse_volume):
    assert extract_surface(coarse_volume).evidence.backend == coarse_volume.backend
    assert extract_surface(coarse_volume, backend=AUTO_BACKEND).evidence.backend in {
        "numpy",
        "python",
    }
    for name in ("rocm", "hip", "opencl", "vulkan", "cuda", "scipy"):
        with pytest.raises(TsdfBackendUnknown):
            extract_surface(coarse_volume, backend=name)


def test_an_unavailable_backend_raises_rather_than_silently_using_another(
    coarse_volume, monkeypatch
):
    monkeypatch.setattr(
        tsdf_module, "_probe_numpy", lambda: (False, "numpy is absent in this test")
    )
    with pytest.raises(TsdfBackendUnavailable):
        extract_surface(coarse_volume, backend="numpy")
    assert extract_surface(coarse_volume, backend=AUTO_BACKEND).evidence.backend == "python"


def test_numpy_is_never_imported_into_this_module_s_namespace():
    assert "numpy" not in vars(surface_module)
    assert "np" not in vars(surface_module)


# --- the ceilings -------------------------------------------------------------


def test_the_module_ceilings_stay_inside_the_exporters_own():
    """A surface this module returns can always be handed to the mesh writers."""
    assert MAX_SURFACE_VERTICES <= MAX_VERTICES
    assert MAX_SURFACE_FACES <= MAX_FACES


def test_a_ceiling_may_be_lowered_and_never_widened(noisy_volume):
    for bad in (0, -1, MAX_SURFACE_VERTICES + 1):
        with pytest.raises(SurfaceCeilingExceeded):
            extract_surface(noisy_volume, max_vertices=bad)
    for bad in (0, -1, MAX_SURFACE_FACES + 1):
        with pytest.raises(SurfaceCeilingExceeded):
            extract_surface(noisy_volume, max_faces=bad)


def test_an_over_ceiling_surface_is_refused_with_nothing_built(noisy_volume):
    honest = extract_surface(noisy_volume).evidence
    with pytest.raises(SurfaceCeilingExceeded) as raised:
        extract_surface(noisy_volume, max_vertices=honest.vertex_count - 1)
    assert "nothing was materialised" in str(raised.value)
    with pytest.raises(SurfaceCeilingExceeded) as raised:
        extract_surface(noisy_volume, max_faces=honest.face_count - 1)
    assert "nothing was materialised" in str(raised.value)
    # Exactly at the ceiling is allowed; the refusal is for passing it.
    at_ceiling = extract_surface(
        noisy_volume, max_vertices=honest.vertex_count, max_faces=honest.face_count
    )
    assert at_ceiling.evidence.vertex_count == honest.vertex_count
    assert at_ceiling.evidence.face_count == honest.face_count


def test_the_reference_path_refuses_a_field_past_its_own_wall_clock_ceiling():
    """Refused from the cell count alone, before a single cube is walked."""
    volume = TsdfVolume(sphere_config(voxel_size=0.5, truncation=1.5))
    assert volume.config.cell_count > PYTHON_SURFACE_MAX_CELLS
    with pytest.raises(SurfaceBackendTooSmall) as raised:
        extract_surface(volume, backend="python")
    assert "nothing was walked" in str(raised.value)
    assert "never widened at run time" in str(raised.value)


def test_a_field_at_the_reference_path_ceiling_is_extracted_rather_than_refused():
    """Exactly at the ceiling, not one voxel under it: the boundary is the test."""
    cells = PYTHON_SURFACE_MAX_CELLS
    side = round(cells ** (1 / 3))
    assert side**3 == cells
    # A single flat step across the fastest axis, so the answer is known
    # exactly: one cut z-edge under every (i, j) node, and no other crossing.
    values = [-1.0 if flat % side >= side // 2 else 1.0 for flat in range(cells)]
    volume = made_field((side, side, side), values, [1.0] * cells, backend="python")
    assert volume.config.cell_count == PYTHON_SURFACE_MAX_CELLS
    surface = extract_surface(volume, backend="python")
    assert surface.evidence.cubes_examined == (side - 1) ** 3
    assert surface.evidence.cubes_eligible == (side - 1) ** 3
    assert surface.evidence.cubes_active == (side - 1) ** 2
    # Every node of the (i, j) grid owns exactly one cut edge, including the
    # ones on the far border that only one cube reaches.
    assert surface.evidence.vertex_count == side**2
    assert surface.evidence.face_count == 2 * (side - 1) ** 2
    assert surface.evidence.backend == "python"
    assert surface.evidence.census.nonmanifold_edge_count == 0


# --- the evidence a reader gets -----------------------------------------------


def test_the_evidence_accounts_for_every_cube_and_claims_no_accuracy(noisy_volume):
    surface = extract_surface(noisy_volume, min_weight=2.0)
    evidence = surface.evidence
    rendered = evidence.as_dict()
    cubes = rendered["cubes"]
    assert isinstance(cubes, dict)
    nx, ny, nz = noisy_volume.config.shape
    assert cubes["examined"] == (nx - 1) * (ny - 1) * (nz - 1)
    assert cubes["examined"] == cubes["eligible"] + cubes["excluded_insufficient_weight"]
    assert cubes["eligible"] == cubes["active"] + cubes["eligible_without_crossing"]

    mesh_block = rendered["mesh"]
    assert isinstance(mesh_block, dict)
    assert mesh_block["vertex_count"] == surface.mesh.vertex_count
    assert mesh_block["face_count"] == surface.mesh.face_count
    assert mesh_block["edges"] == edge_census(surface.mesh.faces).as_dict()
    assert mesh_block["module_vertex_ceiling"] == MAX_SURFACE_VERTICES
    assert mesh_block["module_face_ceiling"] == MAX_SURFACE_FACES
    assert mesh_block["bounds"] is not None

    grid = rendered["grid"]
    assert isinstance(grid, dict)
    assert grid["voxel_size"] == noisy_volume.config.voxel_size
    assert grid["truncation"] == noisy_volume.config.truncation
    assert grid["shape"] == [nx, ny, nz]
    assert rendered["min_weight"] == 2.0
    assert rendered["backend"] == noisy_volume.backend
    assert rendered["level"] == 0.0
    assert rendered["dimensional_accuracy_claimed"] is False
    assert surface.as_dict()["mesh"] == surface.mesh.as_dict()


def test_the_surface_states_its_own_convention_and_not_the_occupancy_paths(noisy_volume):
    surface = extract_surface(noisy_volume)
    stated = surface.mesh.limitations
    assert INSIDE_CONVENTION in stated
    assert EXACT_LEVEL_POLICY in stated
    for note in FIELD_LIMITATIONS:
        assert note in stated
    # A mesh is not a bag of unconnected samples, so it must not carry the one
    # caveat that says it is.
    assert UNCONNECTED_SAMPLES_LIMITATION not in stated
    # Nor any of the occupancy caveats: no cell was marked solid, no grid was
    # padded, and no back wall was invented.
    for note in OCCUPANCY_LIMITATIONS:
        assert note not in stated
    assert surface.evidence.limitations == stated
    assert not any("iso" in note and "3.5" in note for note in stated)


def test_the_surface_reports_the_threshold_it_actually_applied(noisy_volume):
    surface = extract_surface(noisy_volume, min_weight=3.0)
    assert surface.evidence.min_weight == 3.0
    assert any("weight threshold 3.0" in note for note in surface.mesh.limitations)


# --- the working tile is a real ceiling ---------------------------------------


def walk_over(volume: TsdfVolume, *, backend: str, min_weight: float = 1.0):
    """The product's own cube walk over a volume's field, so its counters can be read.

    :func:`extract_surface` builds one of these internally and does not hand it
    back, because how the walk tiled its work is a property of the walk rather
    than of the surface and has no business in a caller's evidence - the
    evidence has to be identical on both backends. The bound still has to be
    audited by something, so it is audited here, directly.
    """
    return surface_module._CubeWalk(
        values=volume.values,
        weights=volume.weights,
        shape=volume.config.shape,
        min_weight=min_weight,
        backend=backend,
    )


#: Cube-plane spans a legal volume can present. The last few are the point: at
#: :data:`scanner3d.tsdf.MAX_VOXELS` a field may legally be 2 x 2 x 4,194,304,
#: whose ``k`` span alone is sixty-four times the whole tile ceiling.
PATHOLOGICAL_SPANS = (
    (1, 1),
    (1, 49),
    (49, 49),
    (199, 199),
    (1, SURFACE_CHUNK_CUBES - 1),
    (1, SURFACE_CHUNK_CUBES),
    (1, SURFACE_CHUNK_CUBES + 1),
    (3, SURFACE_CHUNK_CUBES + 1),
    (1, 4_194_303),
    (2047, 4095),
    (4_194_303, 1),
)


@pytest.mark.parametrize(("span_j", "span_k"), PATHOLOGICAL_SPANS)
def test_no_tile_of_any_legal_cube_plane_exceeds_the_chunk_ceiling(span_j, span_k):
    """The bound the module documents, checked at spans it would actually be given.

    Cutting only ``j`` satisfies this for a thin slab and silently fails for a
    long, narrow one: the ``j`` step bottoms out at one and each temporary
    still spans every ``k`` cube, so a legal 2 x 2 x 4,194,304 field allocates
    four-million-element arrays while claiming a 65,536-cube ceiling.
    """
    step_j, step_k = surface_module._tile_plan(span_j, span_k)
    assert step_j >= 1 and step_k >= 1
    assert step_j * step_k <= SURFACE_CHUNK_CUBES
    # Walked j-outer, k-inner, the tiles reproduce the volume's own i, j, k
    # order only while a multi-row tile spans k whole. Splitting both at once
    # would emit a block of rows at low k before any of them reached high k,
    # renumbering every vertex the module returns.
    assert step_j == 1 or step_k >= span_k


@pytest.mark.parametrize(("span_j", "span_k"), PATHOLOGICAL_SPANS)
def test_an_ordinary_cube_plane_is_still_tiled_as_widely_as_it_ever_was(span_j, span_k):
    """A real ceiling must not become a stream of one-row tiles on normal fields.

    While the ``k`` span fits inside the ceiling - which is every volume this
    appliance actually builds - the plan is exactly the full-width
    ``SURFACE_CHUNK_CUBES // span_k`` rows it has always been, so the measured
    cost of an ordinary extraction is unchanged by the fix above.
    """
    step_j, step_k = surface_module._tile_plan(span_j, span_k)
    if span_k <= SURFACE_CHUNK_CUBES:
        assert (step_j, step_k) == (max(1, SURFACE_CHUNK_CUBES // span_k), span_k)
    else:
        assert (step_j, step_k) == (1, SURFACE_CHUNK_CUBES)


def test_a_field_whose_whole_layer_fits_is_walked_as_one_tile(noisy_volume):
    """The other half of the same claim, measured on a real fused field."""
    _, ny, nz = noisy_volume.config.shape
    plane = (ny - 1) * (nz - 1)
    assert plane <= SURFACE_CHUNK_CUBES
    walk = walk_over(noisy_volume, backend="numpy")
    for _cube in walk.cubes():
        pass
    assert walk.peak_tile_cubes == plane


def long_thin_field() -> tuple[TsdfVolume, tuple[int, int, int], list[int]]:
    """A legal 2 x 2 x (chunk + 2) field: one cube row longer than the whole ceiling.

    The negative voxels are placed so that active cubes land on BOTH sides of
    the tile seam at cube ``k == SURFACE_CHUNK_CUBES`` and in the final tile,
    which a walk that dropped or duplicated a tile boundary would get wrong.
    """
    nz = SURFACE_CHUNK_CUBES + 2
    shape = (2, 2, nz)
    negatives = (1, SURFACE_CHUNK_CUBES - 1, nz - 1)
    values = [
        -1.0 if k in negatives else 1.0 for _ in range(2) for _ in range(2) for k in range(nz)
    ]
    weights = [1.0] * (4 * nz)
    expected = sorted({0, 1, SURFACE_CHUNK_CUBES - 2, SURFACE_CHUNK_CUBES - 1, SURFACE_CHUNK_CUBES})
    return made_field(shape, values, weights), shape, expected


def test_a_long_thin_field_is_tiled_along_k_and_stays_inside_the_ceiling():
    """The pathological case the ``j``-only cut declared bounded and was not.

    ``nz - 1`` here is one larger than :data:`SURFACE_CHUNK_CUBES`, so a walk
    that tiles only ``j`` builds temporaries of ``span_k`` elements while
    reporting a 65,536-cube ceiling. The bound is asserted from the walk's own
    recorded peak rather than from the plan, so an implementation that
    computed a correct plan and then ignored it fails too.
    """
    volume, shape, expected = long_thin_field()
    span_k = shape[2] - 1
    assert span_k > SURFACE_CHUNK_CUBES

    walk = walk_over(volume, backend="numpy")
    cubes = list(walk.cubes())
    assert walk.peak_tile_cubes <= SURFACE_CHUNK_CUBES
    assert walk.examined == span_k
    assert walk.eligible == span_k
    assert [base for base, _case, _corners in cubes] == expected
    assert len(cubes) == len(expected)


def test_the_two_walks_agree_across_a_tile_seam():
    """Ordering is the thing a two-axis cut can break, so it is compared cube by cube.

    The reference walk knows nothing about tiles: it is a triple loop in ``i``,
    ``j``, ``k``. If the accelerated walk's tiles ever emitted their cubes in a
    different order, every vertex index in every mesh this module returns would
    change, and this is what would say so.
    """
    volume, _shape, _expected = long_thin_field()
    accelerated = walk_over(volume, backend="numpy")
    reference = walk_over(volume, backend="python")
    assert list(accelerated.cubes()) == list(reference.cubes())
    assert (accelerated.examined, accelerated.eligible) == (reference.examined, reference.eligible)
    assert reference.peak_tile_cubes == 1


def test_the_peak_tile_is_reset_rather_than_carried_between_walks(noisy_volume):
    walk = walk_over(noisy_volume, backend="numpy")
    first = [len(list(walk.cubes())), walk.peak_tile_cubes]
    second = [len(list(walk.cubes())), walk.peak_tile_cubes]
    assert first == second


# --- what the saved-capture benchmark row may and may not claim ---------------


@pytest.fixture(scope="module")
def saved_capture_row(noisy_volume):
    """The benchmark's saved-capture record, built from a measured synthetic row."""
    surface = extract_surface(noisy_volume)
    return (
        tsdf_benchmark.saved_capture_estimate(noisy_volume.config, surface.evidence, 0.25),
        noisy_volume.config,
        surface.evidence,
    )


def test_the_saved_capture_row_reports_its_examined_cubes_exactly(saved_capture_row):
    """How many cubes a walk visits is decided by the shape alone, which is recorded.

    So it is arithmetic on a known quantity, not a prediction, and reporting it
    as a ratio-scaled estimate would understate what is actually known.
    """
    row, config, evidence = saved_capture_row
    shape = tsdf_benchmark.SAVED_CAPTURE_SHAPE
    assert shape == (174, 141, 158)
    exact = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1)
    assert exact == 3_802_540
    cubes = row["cubes"]
    assert cubes["examined"] == exact
    assert cubes["examined_exact"] is True
    ratio = row["cell_ratio"]
    assert ratio == (shape[0] * shape[1] * shape[2]) / config.cell_count
    assert cubes["examined"] != round(evidence.cubes_examined * ratio)


def test_the_saved_capture_row_leaves_every_coverage_dependent_count_unknown(saved_capture_row):
    """Eligible, excluded and active are decided by real observation coverage.

    A synthetic sphere fused from clean simulated views is not evidence about
    how much of a chassis scan cleared its weight gate, or how much of the
    field the level set passes through, so a cell ratio cannot produce those
    numbers - exactly as it cannot produce vertices, faces or the edge census.
    """
    row, _config, evidence = saved_capture_row
    cubes = row["cubes"]
    assert cubes["eligible"] is None
    assert cubes["excluded_insufficient_weight"] is None
    assert cubes["active"] is None
    assert row["vertices_estimate"] is None
    assert row["faces_estimate"] is None
    assert row["edge_census_estimate"] is None
    # The shape a ratio-scaled coverage estimate used to live in, gone rather
    # than emptied, so a reader cannot find one there again.
    assert "cubes_estimate" not in row
    ratio = row["cell_ratio"]
    scaled = {
        round(evidence.cubes_eligible * ratio),
        round(evidence.cubes_excluded_insufficient_weight * ratio),
        round(evidence.cubes_active * ratio),
    }
    assert not scaled & {value for value in cubes.values() if isinstance(value, int)}


def test_the_saved_capture_row_estimates_only_the_extraction_time(saved_capture_row):
    """The one number a cell ratio can honestly carry, and it is labelled as rough."""
    row, config, _evidence = saved_capture_row
    capture_cells = math.prod(tsdf_benchmark.SAVED_CAPTURE_SHAPE)
    assert row["extract_seconds_estimate"] == pytest.approx(
        0.25 * capture_cells / config.cell_count
    )
    assert row["estimate"] is True
    assert row["measured"] is False
    assert "not measured" in row["case"]
    assert "not authorised" in row["caveat"]
    assert "rough cell-scaled estimate" in row["caveat"]
    assert row["source_field"]["cells"] == capture_cells
