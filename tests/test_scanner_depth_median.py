"""M26 independent image arithmetic and scope checks; no scanner or GPU workload."""

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d.preparation import DepthBand


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_depth_median")


def scalar_reference(array, scale):
    result = array.copy()
    for v, u in zip(*np.nonzero(array), strict=True):
        if not 0 < v < 399 or not 0 < u < 639:
            continue
        values = sorted(int(array[y, x]) for y in range(v - 1, v + 2) for x in range(u - 1, u + 2))
        if (
            values[0] * scale > 60
            and values[-1] * scale < 300
            and (values[-1] - values[0]) * scale <= 1.5
        ):
            result[v, u] = values[4]
    return result


@pytest.mark.parametrize(
    "kind", ["constant", "ramp", "outliers", "hole", "edge", "near", "far", "maximum", "random"]
)
def test_exact_scalar_neighborhood_oracle(module, kind):
    array = np.zeros((400, 640), dtype="<u2")
    patch = np.full((13, 17), 20000, dtype=np.uint16)
    if kind == "ramp":
        patch += np.arange(17, dtype=np.uint16) + np.arange(13, dtype=np.uint16)[:, None] * 2
    elif kind == "outliers":
        patch[5:7, 6:8] = [[20200, 20100], [19800, 19900]]
    elif kind == "hole":
        patch[5:7, 6:8] = 0
    elif kind == "edge":
        patch[:, 8:] = 24000
    elif kind == "near":
        patch[:] = 12001
        patch[5:7, 6:8] = 12000
    elif kind == "far":
        patch[:] = 60000
        patch[5:7, 6:8] = 60001
    elif kind == "maximum":
        patch[:] = 65535
    elif kind == "random":
        patch = np.random.default_rng(2601).integers(19800, 20201, size=(13, 17), dtype=np.uint16)
    array[150:163, 210:227] = patch
    raw = array.tobytes()
    before = hashlib.sha256(raw).hexdigest()
    actual, stats = module.filter_depth(raw, DepthBand(60, 300))
    expected = scalar_reference(array, module.DEPTH_SCALE_METRES)
    assert actual == expected.tobytes() and hashlib.sha256(raw).hexdigest() == before
    assert np.array_equal(np.frombuffer(actual, dtype="<u2") == 0, array.ravel() == 0)
    assert stats["changed_pixels"] == int(np.count_nonzero(expected != array))
    assert stats["positive_changes"] + stats["negative_changes"] == stats["changed_pixels"]
    assert stats["return_membership_exact"] and stats["band_membership_exact"]


def test_borders_and_nonrecursive_neighbors(module):
    array = np.zeros((400, 640), dtype="<u2")
    array[:5, :5] = 20000
    array[0, :5] = 20100
    array[:5, 0] = 19900
    array[1:4, 1:4] = [[20002, 19990, 20002], [19990, 20010, 20002], [20010, 20010, 20002]]
    actual, _ = module.filter_depth(array.tobytes(), DepthBand(60, 300))
    result = np.frombuffer(actual, dtype="<u2").reshape(400, 640)
    assert np.array_equal(result, scalar_reference(array, module.DEPTH_SCALE_METRES))
    assert np.array_equal(result[0], array[0]) and np.array_equal(result[:, 0], array[:, 0])


@pytest.mark.parametrize(
    "raw,band",
    [
        (b"", DepthBand(60, 300)),
        (bytes(512001), DepthBand(60, 300)),
        (bytearray(512000), DepthBand(60, 300)),
        (bytes(512000), DepthBand(61, 300)),
    ],
)
def test_refuses_foreign_shape_type_or_band(module, raw, band):
    with pytest.raises(ValueError):
        module.filter_depth(raw, band)


@pytest.mark.parametrize("profile", ["observe-raw", "valid-nine-median3"])
@pytest.mark.parametrize("capacity", [900, 1800])
def test_real_grid_preparation_and_restored_aliases(module, profile, capacity):
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    expected = module.ORIGINAL_GRID(raw, target_points=capacity, depth_band=DepthBand(60, 300))
    experiment = module.DepthMedianExperiment(profile)
    with experiment.active():
        first = module.grid.prepare_frame(
            raw, target_points=capacity, depth_band=DepthBand(60, 300)
        )
        second = module.static_replay.prepare_grid(
            raw, target_points=capacity, depth_band=DepthBand(60, 300)
        )
    assert first == second == expected
    assert module.grid.prepare_frame is module.static_replay.prepare_grid is module.ORIGINAL_GRID
    assert len(experiment.records) == 2 and all(
        r["status"] == "completed" for r in experiment.records
    )
    assert all(r["original_raw_sha256"] == r["derived_raw_sha256"] for r in experiment.records)
    assert (
        all(r["filtering"] is None for r in experiment.records)
        if profile == "observe-raw"
        else all(r["filtering"]["changed_pixels"] == 0 for r in experiment.records)
    )


@pytest.mark.parametrize("profile", ["observe-raw", "valid-nine-median3"])
@pytest.mark.parametrize(
    "failure",
    ["raise", "interrupt", "nested", "foreign-hook", "foreign-owner", "call-bound", "budget"],
)
def test_scope_failures_restore_and_record(module, monkeypatch, profile, failure):
    experiment = module.DepthMedianExperiment(profile)
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    if failure == "foreign-hook":
        with monkeypatch.context() as isolated:
            isolated.setattr(module.grid, "prepare_frame", lambda *a, **k: None)
            with pytest.raises(RuntimeError), experiment.active():
                pytest.fail("foreign hook accepted")
    elif failure in ("raise", "interrupt"):
        kind = RuntimeError if failure == "raise" else KeyboardInterrupt

        def failed(*args, **kwargs):
            raise kind("injected preparation failure")

        with monkeypatch.context() as isolated:
            isolated.setattr(module, "ORIGINAL_GRID", failed)
            isolated.setattr(module.grid, "prepare_frame", failed)
            isolated.setattr(module.static_replay, "prepare_grid", failed)
            with pytest.raises(kind, match="injected preparation failure"), experiment.active():
                experiment.prepare(raw, target_points=900)
        assert experiment.records[0]["status"] == "failed"
    else:
        with experiment.active():
            if failure == "nested":
                with pytest.raises(RuntimeError), module.DepthMedianExperiment(profile).active():
                    pytest.fail("nested scope accepted")
            elif failure == "foreign-owner":
                with pytest.raises(ValueError):
                    module.DepthMedianExperiment(profile).prepare(raw, target_points=900)
            else:
                if failure == "call-bound":
                    experiment.records.extend([{}] * 3000)
                with pytest.raises(ValueError):
                    experiment.prepare(raw, target_points=901 if failure == "budget" else 900)
    assert module.grid.prepare_frame is module.static_replay.prepare_grid is module.ORIGINAL_GRID
    assert module._ACTIVE.get() is None


def test_explicit_profile_and_active_scope_required(module):
    with pytest.raises(ValueError):
        module.DepthMedianExperiment("fill-holes")
    with pytest.raises(ValueError):
        module.DepthMedianExperiment("observe-raw").prepare(bytes(512000), target_points=900)
