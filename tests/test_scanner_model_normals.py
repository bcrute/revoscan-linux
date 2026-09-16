"""Native model normals preserve geometry, provenance, bounds and missing data."""

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid, rotation_about_axis
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import IncrementalTracker


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_model_normals")


def field(points, neighbours, normal=(1.0, 0.0, 0.0)):
    return icp_plane.NormalField(
        tuple(icp_plane.NormalEstimate(p, normal, neighbours) for p in points), neighbours, ()
    )


@pytest.mark.parametrize("count", [2, 3, 8])
def test_point_construction_equals_original_model_exactly(model, monkeypatch, count):
    frames = [
        model.NativeFrame(
            i,
            tuple((float(j), float(j % 13), float(i + 1)) for j in range(401)),
            rotation_about_axis((0.0, 0.0, 1.0), i * 0.1, through=(10.0, 2.0, 0.0)),
        )
        for i in range(count)
    ]
    tracker = IncrementalTracker(backend="scipy")
    tracker._accepted = [
        SimpleNamespace(index=f.index, cloud=f.points, pose=f.pose) for f in frames
    ]
    monkeypatch.setattr(tracker, "_reference_frames", lambda *args, **kwargs: list(range(count)))
    expected, used = tracker._model_reference(tracker._accepted[-1], Rigid.identity())
    actual, origins = model.reference_points(frames, 1800)
    assert expected == actual and used == tuple(range(count))
    assert len(actual) == len(origins) <= 1800


def test_normals_rotate_but_do_not_translate(model):
    points = ((1.0, 2.0, 3.0), (2.0, 3.0, 4.0))
    turn = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2)
    transform = Rigid.of(turn.rotation, (1000.0, -800.0, 300.0))
    frames = [model.NativeFrame(0, points, transform)]
    reference, _ = model.reference_points(frames, 1800)
    with icp_plane.normal_acceleration(field):
        normals, owners = model.model_field(frames, reference, model.NativeCache(), cell=0.01)
    assert owners == (0, 0)
    assert tuple(e.point for e in normals.estimates) == voxel_downsample(reference, 0.01)
    for estimate in normals.estimates:
        np.testing.assert_allclose(estimate.normal, (0.0, 1.0, 0.0), atol=1e-14)


@pytest.mark.parametrize("sign", [-1, 1])
def test_voxel_centroids_order_and_sign_are_preserved(model, sign):
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-2.0, 1.0, 1.0), (4.0, 0.0, 0.0))
    estimates = field(points, 10, (0.0, float(sign), 0.0)).estimates
    actual, owners = model.voxel_field(points, estimates, (1, 2, 3, 4), cell=2.0, neighbours=10)
    assert tuple(e.point for e in actual.estimates) == voxel_downsample(points, 2.0)
    assert owners == (1, 3, 4)
    assert all(e.normal == (0.0, float(sign), 0.0) for e in actual.estimates)


def test_closest_missing_normal_is_not_replaced_to_make_a_fit(model):
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    estimates = (
        icp_plane.NormalEstimate(points[0], None, 2),
        icp_plane.NormalEstimate(points[1], (1.0, 0.0, 0.0), 10),
    )
    actual, owners = model.voxel_field(points, estimates, (7, 8), cell=2.0, neighbours=10)
    assert owners == (7,)  # Exact tie follows original order, even when normal is absent.
    assert actual.estimates[0].normal is None and actual.estimates[0].neighbours == 2


def test_native_cache_is_bounded_and_provider_scoped(model):
    cache = model.NativeCache()

    def positive(p, n):
        return field(p, n)

    def negative(p, n):
        return field(p, n, (-1.0, 0.0, 0.0))

    points = ((0.0, 1.0, 2.0),)
    a = cache.estimate(points, 10, positive)
    assert cache.estimate(points, 10, positive) is a
    assert cache.estimate(points, 10, negative).estimates[0].normal != a.estimates[0].normal
    for i in range(40):
        cache.estimate(((float(i), 0.0, 0.0),), 10, positive)
    assert cache.estimate.cache_info().currsize == icp_plane.NORMAL_CACHE_ENTRIES


@pytest.mark.parametrize(
    "normal", [(float("nan"), 0.0, 1.0), (0.0, 0.0, 0.0), (float("inf"), 1.0, 0.0)]
)
def test_nonfinite_and_zero_native_normals_are_refused(model, normal):
    cache = model.NativeCache()
    with pytest.raises(ValueError, match="Native normal"):
        cache.estimate(((0.0, 1.0, 2.0),), 10, lambda p, n: field(p, n, normal))


def test_changed_native_point_mapping_and_model_geometry_are_refused(model):
    points = ((1.0, 2.0, 3.0),)
    with pytest.raises(ValueError, match="coordinates/order/count"):
        model.NativeCache().estimate(points, 10, lambda p, n: field((), n))
    frames = [model.NativeFrame(0, points, Rigid.identity())]
    with pytest.raises(ValueError, match="Reconstructed model points"):
        model.model_field(frames, ((1.0, 2.0, 4.0),), model.NativeCache())


def test_duplicate_and_oversized_native_frames_are_refused(model):
    frame = model.NativeFrame(0, ((1.0, 2.0, 3.0),), Rigid.identity())
    for frames, budget in (([], 1800), ([frame, frame], 1800), ([frame], 1801)):
        with pytest.raises(ValueError):
            model.reference_points(frames, budget)
    large = model.NativeFrame(0, frame.points * 901, Rigid.identity())
    with pytest.raises(ValueError, match=r"1\.\.900"):
        model.reference_points([large], 1800)


def tracker_fixture(model):
    tracker = model.NativeModelTracker(backend="scipy")
    points = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    frames = [model.NativeFrame(i, points, Rigid.identity()) for i in range(2)]
    tracker._accepted = [
        SimpleNamespace(index=f.index, cloud=f.points, pose=f.pose) for f in frames
    ]
    reference, _ = model.reference_points(frames, 1800)
    return (
        tracker,
        points,
        dict(
            origin="model",
            reference=reference,
            reference_frames=(0, 1),
            into_world=Rigid.identity(),
            initial=Rigid.identity(),
            previous=tracker._accepted[-1],
            seed="predicted",
        ),
    )


@pytest.mark.parametrize("failure", [None, "exception", "wrong_target", "wrong_neighbours"])
def test_ordinary_model_context_is_scoped_and_restored(model, monkeypatch, failure):
    tracker, points, kwargs = tracker_fixture(model)
    previous_provider = None
    returned = object()

    def base_candidate(self, *args, **passed):
        provider = icp_plane._NORMAL_ACCELERATOR.get()
        assert provider is not None and provider is not previous_provider
        if failure == "exception":
            raise RuntimeError("injected candidate error")
        sampled = voxel_downsample(passed["reference"], 1.5)
        actual = provider(
            sampled[:-1] if failure == "wrong_target" else sampled,
            11 if failure == "wrong_neighbours" else 10,
        )
        assert tuple(e.point for e in actual.estimates) == sampled
        return returned

    monkeypatch.setattr(IncrementalTracker, "_candidate", base_candidate)
    before = icp_plane._NORMAL_ACCELERATOR.get()
    with icp_plane.normal_acceleration(field):
        previous_provider = icp_plane._NORMAL_ACCELERATOR.get()
        if failure:
            with pytest.raises(RuntimeError if failure == "exception" else ValueError):
                tracker._candidate(2, points, **kwargs)
        else:
            assert tracker._candidate(2, points, **kwargs) is returned
        assert icp_plane._NORMAL_ACCELERATOR.get() is previous_provider
    assert icp_plane._NORMAL_ACCELERATOR.get() is before
    assert len(tracker.model_normal_records) == 1
    assert tracker.model_normal_records[0]["status"] == ("failed" if failure else "completed")


@pytest.mark.parametrize("origin", ["keyframe", "recovery", "adjacent"])
def test_non_model_candidates_are_transparent(model, monkeypatch, origin):
    tracker, points, kwargs = tracker_fixture(model)
    kwargs["origin"] = origin
    expected = object()
    previous_provider = None

    def base_candidate(self, *args, **passed):
        assert icp_plane._NORMAL_ACCELERATOR.get() is previous_provider
        assert args == (2, points) and passed == kwargs
        return expected

    monkeypatch.setattr(IncrementalTracker, "_candidate", base_candidate)
    with icp_plane.normal_acceleration(field):
        previous_provider = icp_plane._NORMAL_ACCELERATOR.get()
        assert tracker._candidate(2, points, **kwargs) is expected
    assert tracker.model_normal_records == []
    assert tracker.native_cache.estimate.cache_info().currsize == 0


def test_model_experiment_rejects_parameter_changes_and_excess_calls(model):
    with pytest.raises(ValueError, match="unchanged"):
        model.NativeModelTracker(backend="scipy", keep_fraction=1.0)
    tracker, points, kwargs = tracker_fixture(model)
    with pytest.raises(ValueError, match="ordinary-model only"):
        tracker._candidate(2, points, **kwargs, coarse_to_fine=True)
    tracker.model_normal_records = [{}] * 3000
    with pytest.raises(ValueError, match="3000-call"):
        tracker._candidate(2, points, **kwargs)


def test_replay_profile_keeps_default_policy(model):
    replay = importlib.import_module("scanner_motion_replay")
    assert "native-model-normals" in replay.REFERENCE_PROFILES
    assert replay.reference_policy("native-model-normals") is replay.DEFAULT_POLICY
