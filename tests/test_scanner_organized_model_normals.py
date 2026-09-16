"""Independent ownership, sampling, rotation and equal-coordinate role checks."""

import importlib
import math
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.geometry import Rigid
from scanner3d.icp_plane import NormalEstimate, NormalField
from scanner3d.simulate import FACTORY_INTRINSICS

PIXEL = 200 * 640 + 310


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_organized_model_normals")


def deproject(raw, pixels):
    codes = np.frombuffer(raw, "<u2")
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    result = []
    for pixel in pixels:
        v, u = divmod(pixel, 640)
        z = int(codes[pixel]) * DEPTH_SCALE_METRES
        result.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return tuple(result)


def native(module, pixels=(PIXEL,), *, radius=5, code=40000, raw=None):
    if raw is None:
        raw = np.full(256000, code, "<u2").tobytes()
    return module.NativeField(raw, deproject(raw, pixels), radius)


def pooled(model, normal=(0.0, 0.0, -1.0)):
    return NormalField(
        tuple(NormalEstimate(p, normal, 10) for p in model.centers), 10, ("original",)
    )


def geometry(module, field=None, *, pose=None, index=7):
    field = native(module) if field is None else field
    pose = Rigid.identity() if pose is None else pose
    return module.ModelGeometry(
        (module.ModelFrame(index, field, pose),), tuple(pose.apply_all(field.points))
    )


@pytest.mark.parametrize("radius", [5, 10])
def test_exact_raw_field_and_immutability(module, radius):
    field = native(module, (PIXEL, PIXEL + 7), radius=radius)
    assert tuple(p.pixel for p in field.planes) == (PIXEL, PIXEL + 7)
    assert all(p.status == "usable" and p.normal == (0.0, 0.0, -1.0) for p in field.planes)
    assert all(p.support_count == (2 * radius + 1) ** 2 for p in field.planes)
    with pytest.raises(FrozenInstanceError):
        field.radius = 10
    with pytest.raises(FrozenInstanceError):
        field.planes[0].normal = (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "count,budget,nframes", [(1800, 1800, 8), (900, 1800, 3), (11, 8, 3), (4, 2, 3), (7, 1800, 2)]
)
def test_independent_sampling_offsets_and_final_cap(module, count, budget, nframes):
    field = native(module, tuple(100 * 640 + i for i in range(count)))
    frames = tuple(
        module.ModelFrame(20 + i, field, Rigid.of(Rigid.identity().rotation, (i * 3.0, 0.0, 0.0)))
        for i in range(nframes)
    )
    points, owners = [], []
    quota = max(1, budget // nframes)
    for position, frame in enumerate(frames):
        stride = math.ceil(count / quota) if count > quota else 1
        selected = list(range(position % stride, count, stride))[:quota]
        for i in selected:
            p = field.points[i]
            points.append((p[0] + position * 3.0, p[1], p[2]))
            owners.append((frame.index, i, field.planes[i].pixel))
    model = module.ModelGeometry(frames, tuple(points[:budget]), budget)
    assert [(o.frame_index, o.native_index, o.pixel) for o in model.origins] == owners[:budget]
    assert all(o.raw_sha256 == field.raw_sha256 for o in model.origins)


def test_equal_coordinates_distinct_raw_owners_and_first_index_tie(module):
    first = native(module)
    changed = np.frombuffer(first.raw, "<u2").copy()
    changed[PIXEL + 1] += 50
    second = native(module, raw=changed.tobytes())
    assert first.points == second.points and first.raw_sha256 != second.raw_sha256
    frames = (
        module.ModelFrame(9, first, Rigid.identity()),
        module.ModelFrame(3, second, Rigid.identity()),
    )
    model = module.ModelGeometry(frames, first.points + second.points)
    assert model.centers == first.points and model.selected == (0,)
    assert [o.frame_index for o in model.origins] == [9, 3]
    assert [o.raw_sha256 for o in model.origins] == [first.raw_sha256, second.raw_sha256]
    _, records = module.blend_model(model, pooled(model), maximum_weight=1.0)
    assert records[0]["frame_index"] == 9  # not lowest frame ID or equal-coordinate lookup


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_rotation_without_translation_and_pca_metadata(module, weight):
    angle = math.radians(12)
    c, s = math.cos(angle), math.sin(angle)
    pose = Rigid.of(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), (100.0, -42.0, 7.0))
    field = native(module)
    model = geometry(module, field, pose=pose)
    before = field.planes
    original = pooled(model)
    result, records = module.blend_model(model, original, maximum_weight=weight)
    record = records[0]
    assert record["model_normal"] == pytest.approx((-s, 0.0, -c), abs=1e-12)
    assert record["native_normal"] == (0.0, 0.0, -1.0)
    assert record["pca_neighbours"] == 10 and record["image_support_count"] == 121
    alpha = weight * (1 - (12 / 30) ** 2) ** 2
    expected = np.array([-alpha * s, 0.0, -(1 - alpha) - alpha * c])
    expected /= np.linalg.norm(expected)
    assert result.estimates[0].normal == pytest.approx(expected, abs=1e-12)
    assert result.estimates[0].point == original.estimates[0].point
    assert result.estimates[0].neighbours == original.estimates[0].neighbours
    assert result.requested_neighbours == original.requested_neighbours
    assert result.limitations[:1] == original.limitations
    assert field.planes == before and record["native_perpendicular_rms"] == 0


@pytest.mark.parametrize("missing", [False, True])
def test_fallback_and_missing_preserve_exact_original_estimate(module, missing):
    model = geometry(module, native(module, (0,)))  # image border has no plane
    original = pooled(model, None if missing else (0.0, 0.0, 1.0))
    result, records = module.blend_model(model, original, maximum_weight=1)
    assert result.estimates[0] is original.estimates[0]
    assert records[0]["confidence"]["mode"] == ("missing" if missing else "fallback")
    assert len(result.usable) == len(original.usable)


def test_missing_pca_never_created_from_usable_image(module):
    model = geometry(module)
    original = pooled(model, None)
    result, records = module.blend_model(model, original, maximum_weight=1)
    assert records[0]["native_status"] == "usable"
    assert result.estimates[0] is original.estimates[0] and result.usable == ()


@pytest.mark.parametrize(
    "roles", [("forward", "reverse", "forward"), ("reverse", "forward", "reverse")]
)
def test_identical_coordinates_roles_are_checked_before_cached_result(module, roles):
    model = geometry(module)
    theta = math.radians(12)
    original = pooled(model, (math.sin(theta), 0.0, -math.cos(theta)))
    dispatches = []

    @lru_cache(maxsize=8)
    def base(points, neighbours):
        dispatches.append((points, neighbours))
        return original

    role = [roles[0]]
    route = module.RoleDispatcher(
        model, model.centers, base, lambda: role[0], maximum_weight=1, observe=False
    )
    for direction in roles:
        role[0] = direction
        result = route(model.centers, 10)
        if direction == "reverse":
            assert result is original
        else:
            assert (
                result is not original
                and result.estimates[0].normal != original.estimates[0].normal
            )
    assert len(dispatches) == 1
    assert [call["direction"] for call in route.calls] == list(roles)


def test_observation_returns_original_without_extra_provider_calls(module):
    model = geometry(module)
    calls = []
    original = pooled(model)

    def base(points, neighbours):
        calls.append((points, neighbours))
        return original

    route = module.RoleDispatcher(
        model, model.centers, base, lambda: "forward", maximum_weight=1, observe=True
    )
    assert route(model.centers, 10) is original and len(calls) == 1
    assert route.result is None


@pytest.mark.parametrize(
    "change",
    [
        {"pixel": -1},
        {"pixel": True},
        {"radius": 2},
        {"required_support": 90},
        {"status": "invented"},
        {"normal": None},
        {"normal": [0.0, 0.0, -1.0]},
        {"normal": (0.0, 0.0, 0.0)},
        {"normal": (math.nan, 0.0, 1.0)},
        {"support_count": 90},
        {"support_count": 122},
        {"support_count": True},
        {"perpendicular_rms": None},
        {"perpendicular_rms": math.nan},
        {"perpendicular_rms": -0.01},
        {"perpendicular_rms": 0.250001},
        {"status": "border", "normal": None},
        {"status": "insufficient-support", "normal": None, "perpendicular_rms": None},
        {"status": "degenerate", "normal": None},
        {"status": "nonplanar", "normal": None},
    ],
)
def test_malformed_native_metadata_refused_even_before_blend(module, change):
    with pytest.raises(ValueError):
        replace(native(module).planes[0], **change)


@pytest.mark.parametrize(
    "kind",
    [
        "mutable-raw",
        "short-raw",
        "duplicate",
        "perturbed",
        "mutable-points",
        "out-of-band",
        "too-many",
        "bad-radius",
    ],
)
def test_invalid_raw_point_binding(module, kind):
    raw = np.full(256000, 40000, "<u2").tobytes()
    points, radius = deproject(raw, [PIXEL]), 5
    if kind == "mutable-raw":
        raw = bytearray(raw)
    elif kind == "short-raw":
        raw = raw[:-2]
    elif kind == "duplicate":
        points *= 2
    elif kind == "perturbed":
        p = points[0]
        points = ((p[0] + 1e-12, p[1], p[2]),)
    elif kind == "mutable-points":
        points = list(points)
    elif kind == "out-of-band":
        raw = np.full(256000, 60001, "<u2").tobytes()
        points = deproject(raw, [PIXEL])
    elif kind == "too-many":
        points *= 1801
    else:
        radius = True
    with pytest.raises(ValueError):
        module.NativeField(raw, points, radius)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate",
        "nine",
        "mixed-radii",
        "wrong-reference",
        "reversed-reference",
        "bad-budget",
    ],
)
def test_model_mapping_refused(module, kind):
    field = native(module, (PIXEL, PIXEL + 10))
    frame = module.ModelFrame(0, field, Rigid.identity())
    frames, expected, budget = (frame,), field.points, 1800
    if kind == "empty":
        frames = ()
    elif kind == "duplicate":
        frames *= 2
    elif kind == "nine":
        frames = tuple(replace(frame, index=i) for i in range(9))
    elif kind == "mixed-radii":
        frames += (replace(frame, index=1, native=native(module, radius=10)),)
    elif kind == "wrong-reference":
        expected = expected[:1]
    elif kind == "reversed-reference":
        expected = expected[::-1]
    else:
        budget = 1801
    with pytest.raises(ValueError):
        module.ModelGeometry(frames, expected, budget)


@pytest.mark.parametrize("kind", ["wrong-order", "wrong-neighbours", "invalid-count", "nonunit"])
def test_pooled_metadata_refused(module, kind):
    model = geometry(module, native(module, (PIXEL, PIXEL + 20)))
    original = pooled(model)
    if kind == "wrong-order":
        original = replace(original, estimates=original.estimates[::-1])
    elif kind == "wrong-neighbours":
        original = replace(original, requested_neighbours=20)
    else:
        changes = {"neighbours": -1} if kind == "invalid-count" else {"normal": (0.0, 0.0, 2.0)}
        original = replace(
            original, estimates=(replace(original.estimates[0], **changes), original.estimates[1])
        )
    with pytest.raises(ValueError):
        module.blend_model(model, original, maximum_weight=1)


@pytest.mark.parametrize(
    "role,neighbours,wrong_points",
    [
        ("unknown", 10, False),
        ("", 10, False),
        ("forward", 20, False),
        ("reverse", True, False),
        ("forward", 10, True),
        ("reverse", 10, True),
    ],
)
def test_role_dispatch_refuses_ambiguous_or_wrong_requests(module, role, neighbours, wrong_points):
    model = geometry(module)

    def forbidden(*args):
        pytest.fail("Invalid role/request reached provider")

    route = module.RoleDispatcher(
        model, model.centers, forbidden, lambda: role, maximum_weight=1, observe=False
    )
    points = ((0.0, 0.0, 1.0),) if wrong_points else model.centers
    with pytest.raises(ValueError):
        route(points, neighbours)


@pytest.mark.parametrize("part", ["translation", "rotation-row"])
def test_direct_mutable_pose_constructor_refused(module, part):
    rotation = Rigid.identity().rotation
    translation = (0.0, 0.0, 0.0)
    if part == "translation":
        translation = list(translation)
    else:
        rotation = (list(rotation[0]), rotation[1], rotation[2])
    with pytest.raises(ValueError, match="immutable"):
        module.ModelFrame(0, native(module), Rigid(rotation, translation))


def test_mutable_reverse_points_refused(module):
    model = geometry(module)
    with pytest.raises(ValueError, match="immutable"):
        module.RoleDispatcher(
            model,
            (list(model.centers[0]),),
            lambda p, n: pooled(model),
            lambda: "reverse",
            maximum_weight=1,
            observe=True,
        )
