"""Measured model representatives preserve all other tracker inputs and scopes."""

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid
from scanner3d.registration import voxel_downsample
from scanner3d.tracking import IncrementalTracker


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_model_representation")


def field(points, neighbours):
    return icp_plane.NormalField(
        tuple(
            icp_plane.NormalEstimate(p, None if i % 3 == 0 else (1.0, 0.0, 0.0), neighbours)
            for i, p in enumerate(points)
        ),
        neighbours,
        (),
    )


def test_representatives_are_measured_points_with_existing_voxel_order(model):
    points = ((0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (1.4, 0.0, 0.0), (-2.0, 0.0, 0.0))
    reps = model.representatives(points, (4, 5, 6, 7))
    assert reps.centroids == voxel_downsample(points, 1.5)
    assert reps.indices == (1, 3)
    assert reps.owners == (5, 7)
    assert reps.measured == (points[1], points[3])
    assert reps.distances == tuple(
        math.dist(a, b) for a, b in zip(reps.centroids, reps.measured, strict=True)
    )


def test_equidistant_representative_keeps_first_original_index(model):
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    reps = model.representatives(points, (8, 2))
    assert reps.indices == (0,) and reps.owners == (8,)
    reverse = model.representatives(tuple(reversed(points)), (2, 8))
    assert reverse.indices == (0,) and reverse.owners == (2,)


@pytest.mark.parametrize(
    "points,owners,cell",
    [
        ((), (), 1.5),
        (((0.0, 0.0, 0.0),) * 1801, (1,) * 1801, 1.5),
        (((0.0, 0.0, 0.0),), (), 1.5),
        (((0.0, 0.0, 0.0),), (1,), 0.0),
        (((0.0, 0.0, 0.0),), (1,), float("nan")),
        (((float("inf"), 0.0, 0.0),), (1,), 1.5),
        (((0.0, 0.0),), (1,), 1.5),
    ],
)
def test_invalid_representative_inputs_refuse(model, points, owners, cell):
    with pytest.raises(ValueError, match="Finite bounded"):
        model.representatives(points, owners, cell=cell)


def fixture_tracker(model, profile="observed-point", normal_mode="pooled"):
    tracker = model.ModelRepresentationTracker(
        backend="scipy", model_profile=profile, normal_mode=normal_mode
    )
    points = tuple((i * 0.2, (i % 7) * 0.1, (i % 3) * 0.2) for i in range(45))
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


@pytest.mark.parametrize("profile", ["observe-centroid", "observed-point"])
@pytest.mark.parametrize("normal_mode", ["pooled", "native"])
def test_target_only_substitution_and_normals_preserve_inputs(
    model, monkeypatch, profile, normal_mode
):
    tracker, points, kwargs = fixture_tracker(model, profile, normal_mode)
    frames = [model.NativeFrame(a.index, a.cloud, a.pose) for a in tracker._accepted]
    _, origins = model.reference_points(frames, 1800)
    reps = model.representatives(kwargs["reference"], [i for i, _ in origins])
    expected_target = reps.measured if profile == "observed-point" else reps.centroids
    expected = object()
    original = icp_plane.voxel_downsample
    native = None
    with icp_plane.normal_acceleration(field):
        previous_provider = icp_plane._NORMAL_ACCELERATOR.get()
        if normal_mode == "native":
            native, _ = model.model_field(frames, kwargs["reference"], model.NativeCache())

        def base(self, *args, **passed):
            assert args == (2, points) and passed == kwargs
            assert icp_plane.voxel_downsample(args[1], 1.5) == voxel_downsample(points, 1.5)
            target = icp_plane.voxel_downsample(passed["reference"], 1.5)
            assert target == expected_target
            provider = icp_plane._NORMAL_ACCELERATOR.get()
            if normal_mode == "pooled":
                assert provider is previous_provider
            else:
                assert provider is not previous_provider and native is not None
                actual = provider(target, 10)
                assert tuple(e.point for e in actual.estimates) == target
                assert [(e.normal, e.neighbours) for e in actual.estimates] == [
                    (e.normal, e.neighbours) for e in native.estimates
                ]
            return expected

        monkeypatch.setattr(IncrementalTracker, "_candidate", base)
        assert tracker._candidate(2, points, **kwargs) is expected
        assert icp_plane._NORMAL_ACCELERATOR.get() is previous_provider
    assert icp_plane.voxel_downsample is original
    assert model._ACTIVE.get() is False
    record = tracker.model_representation_records[0]
    assert record["status"] == "completed"
    assert record["source_downsampling_unchanged"] and record["target_substitution_verified"]
    assert (tracker.model_normal_records is None) == (normal_mode == "pooled")


@pytest.mark.parametrize(
    "failure", ["source", "target", "cell", "extra", "no_call", "exception", "normal"]
)
def test_unexpected_calls_and_exceptions_refuse_and_restore(model, monkeypatch, failure):
    tracker, points, kwargs = fixture_tracker(model, normal_mode="native")
    original = icp_plane.voxel_downsample

    def base(self, *args, **passed):
        if failure == "exception":
            raise RuntimeError("injected")
        if failure == "no_call":
            return None
        icp_plane.voxel_downsample(points[:-1] if failure == "source" else points, 1.5)
        target = icp_plane.voxel_downsample(
            kwargs["reference"][:-1] if failure == "target" else kwargs["reference"],
            1.0 if failure == "cell" else 1.5,
        )
        if failure == "extra":
            icp_plane.voxel_downsample(kwargs["reference"], 1.5)
        if failure == "normal":
            icp_plane._NORMAL_ACCELERATOR.get()(target, 11)
        return None

    monkeypatch.setattr(IncrementalTracker, "_candidate", base)
    with icp_plane.normal_acceleration(field):
        previous_provider = icp_plane._NORMAL_ACCELERATOR.get()
        with pytest.raises(RuntimeError):
            tracker._candidate(2, points, **kwargs)
        assert icp_plane._NORMAL_ACCELERATOR.get() is previous_provider
    assert icp_plane.voxel_downsample is original and not model._ACTIVE.get()
    assert tracker.model_representation_records[0]["status"] == "failed"


@pytest.mark.parametrize("origin", ["adjacent", "keyframe", "recovery"])
def test_other_candidate_types_are_exact_passthrough(model, monkeypatch, origin):
    tracker, points, kwargs = fixture_tracker(model)
    kwargs["origin"] = origin
    original = icp_plane.voxel_downsample
    expected = object()

    def base(self, *args, **passed):
        assert args == (2, points) and passed == kwargs
        assert icp_plane.voxel_downsample is original
        return expected

    monkeypatch.setattr(IncrementalTracker, "_candidate", base)
    assert tracker._candidate(2, points, **kwargs) is expected
    assert tracker.model_representation_records == []


@pytest.mark.parametrize(
    "kwargs", [{"keep_fraction": 1.0}, {"model_profile": "guess"}, {"normal_mode": "guess"}]
)
def test_changed_parameters_refuse(model, kwargs):
    with pytest.raises(ValueError):
        model.ModelRepresentationTracker(
            **dict(backend="scipy", model_profile="observed-point", normal_mode="pooled") | kwargs
        )


def test_provenance_bounds_and_nested_calls_refuse(model):
    tracker, points, kwargs = fixture_tracker(model)
    with pytest.raises(ValueError, match="ordinary-model only"):
        tracker._candidate(2, points, **kwargs, coarse_to_fine=True)
    token = model._ACTIVE.set(True)
    try:
        with pytest.raises(RuntimeError, match="Nested"):
            tracker._candidate(2, points, **kwargs)
    finally:
        model._ACTIVE.reset(token)
    with pytest.raises(ValueError, match="differ"):
        tracker._candidate(2, points, **(kwargs | {"reference": kwargs["reference"][:-1]}))
    tracker._accepted[0].cloud = None
    with pytest.raises(ValueError, match="provenance"):
        tracker._candidate(2, points, **kwargs)
    tracker.model_representation_records = [{}] * 3000
    with pytest.raises(ValueError, match="3000-call"):
        tracker._candidate(2, points, **kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"diagnostic_model_profile": "guess"},
        {"diagnostic_factors": None},
        {"normals_backend": "cpu"},
        {"diagnostic_solver_profile": "planarity"},
    ],
)
def test_motion_api_rejects_out_of_roster_model_mixtures_before_loading(model, tmp_path, overrides):
    replay = importlib.import_module("scanner_motion_replay")
    parameters = dict(
        diagnostic_model_profile="observed-point",
        diagnostic_factors=("stride", "pooled", "residual"),
    )
    with pytest.raises(ValueError, match="M4 requires"):
        replay.run(tmp_path / "absent", tmp_path / "output", 100, (1,), **(parameters | overrides))
    assert not (tmp_path / "output").exists()


def test_static_api_rejects_out_of_roster_solver_before_loading(model, tmp_path):
    replay = importlib.import_module("scanner_factorial_replay")
    config = importlib.import_module("scanner_factorial_config")
    with pytest.raises(ValueError, match="M4 requires"):
        replay.run_static(
            tmp_path / "absent",
            tmp_path / "output",
            config.Factors(),
            diagnostic_model_profile="observed-point",
            diagnostic_solver_profile="generalized-covariance",
        )
    assert not (tmp_path / "output").exists()
