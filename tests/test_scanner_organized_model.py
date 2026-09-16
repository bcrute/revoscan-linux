"""Scoped adapter tests: original controls, admitted raw identity and hook cleanup."""

import importlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from scanner3d import icp_plane, tracking
from scanner3d.preparation import DepthBand


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_organized_model")


def experiment(
    module, *, seed="observe-forward", profile="observe-pooled", capacity="ordinary-900"
):
    return module.OrganizedModelExperiment(profile, capacity, seed, radius=5, maximum_weight=1.0)


def curved_raw():
    y, x = np.indices((400, 640))
    return (
        np.rint(
            40000
            + 0.03 * (x - 320) ** 2
            + 0.07 * (y - 200) ** 2
            + 600 * np.sin(x / 80) * np.sin(y / 50)
        )
        .astype("<u2")
        .tobytes()
    )


def assert_restored(module, ex):
    assert ex.hooks_restored
    assert module.grid.prepare_frame is module.ORIGINAL_GRID
    assert module.static_replay.prepare_grid is module.ORIGINAL_GRID
    assert tracking.IncrementalTracker.offer is module.ORIGINAL_OFFER
    assert tracking.IncrementalTracker._candidate is module.ORIGINAL_CANDIDATE
    assert module.ReverseSeedExperiment.fit is module.ORIGINAL_SEED_FIT
    assert module._ACTIVE.get() is None and module._FIT.get() is None and not module._MODEL.get()


@pytest.mark.parametrize("failure", [False, True])
def test_scope_restores_every_hook_after_exit_or_exception(module, failure):
    ex = experiment(module)
    with pytest.raises(RuntimeError, match="injected") if failure else nullcontext(), ex.active():
        assert module.grid.prepare_frame is module.static_replay.prepare_grid
        assert tracking.IncrementalTracker.offer is not module.ORIGINAL_OFFER
        if failure:
            raise RuntimeError("injected")
    assert_restored(module, ex)
    with pytest.raises(RuntimeError, match="reused"), ex.active():
        pass


def test_real_tracker_refusal_then_anchor_keeps_raw_identity(module):
    ex = experiment(module)
    tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)
    sparse = np.zeros(256000, "<u2")
    sparse[200 * 640 + 310] = 40000
    with ex.active():
        first = ex.prepare(sparse.tobytes(), target_points=900, depth_band=DepthBand(60, 300))
        assert not tracker.offer(first.points).accepted
        second = ex.prepare(curved_raw(), target_points=900, depth_band=DepthBand(60, 300))
        assert tracker.offer(second.points).accepted
        assert ex.binding.retained[0].offer.ordinal == 1
        assert ex.binding.retained[0].offer.raw == curved_raw()
    assert [r["status"] for r in ex.preparations] == ["refused", "accepted"]
    assert [r["accepted_index"] for r in ex.preparations] == [None, 0]
    assert_restored(module, ex)


@pytest.mark.parametrize("seed", ["observe-forward", "reverse-forward"])
def test_actual_seed_solver_tracker_observation_matches_parent(module, seed):
    solver_module = importlib.import_module("scanner_solver_mechanisms")
    raw = curved_raw()

    def run(observe):
        ex = experiment(module, seed=seed)
        tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)
        reverse = module.ReverseSeedExperiment(seed)
        solver = solver_module.SolverExperiment("centered")
        dispatches, decisions = [], []

        def provider(points, neighbours):
            dispatches.append(module.digest(points))
            return icp_plane.estimate_normals(points, neighbours=neighbours, backend="scipy")

        with (
            ex.active() if observe else nullcontext(),
            reverse.active(),
            solver.active(),
            icp_plane.normal_acceleration(provider),
        ):
            for index in range(6):
                cloud = module.grid.prepare_frame(
                    raw, target_points=900, depth_band=DepthBand(60, 300)
                )
                solver.mark_frame(index)
                decisions.append(tracker.offer(cloud.points).as_dict())
        if observe:
            assert_restored(module, ex)
        return decisions, reverse.records, solver.records, dispatches, ex

    parent, observed = run(False), run(True)
    assert observed[:4] == parent[:4]
    assert all(d["accepted"] for d in observed[0])
    assert observed[4].models and all(r["status"] == "completed" for r in observed[4].models)
    directions = {q["direction"] for r in observed[4].models for q in r["queries"]}
    assert directions == ({"forward"} if seed == "observe-forward" else {"forward", "reverse"})


@pytest.mark.parametrize("failure", [ValueError("binding"), ArithmeticError("binding")])
def test_callback_provenance_failure_is_not_swallowed_as_fit_refusal(module, monkeypatch, failure):
    ex = experiment(module, seed="reverse-forward", profile="organized-confidence")
    seed = module.ReverseSeedExperiment("reverse-forward")
    solver = importlib.import_module("scanner_solver_mechanisms").SolverExperiment("centered")
    tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)

    def broken(*args):
        raise failure

    monkeypatch.setattr(module.RoleDispatcher, "__call__", broken)

    def provider(points, neighbours):
        return icp_plane.estimate_normals(points, neighbours=neighbours, backend="scipy")

    with icp_plane.normal_acceleration(provider):
        original_provider = icp_plane._NORMAL_ACCELERATOR.get()
        with (
            pytest.raises(RuntimeError, match="normal-provider provenance"),
            ex.active(),
            seed.active(),
            solver.active(),
        ):
            for index in range(8):
                cloud = ex.prepare(curved_raw(), target_points=900, depth_band=DepthBand(60, 300))
                solver.mark_frame(index)
                tracker.offer(cloud.points)
        assert icp_plane._NORMAL_ACCELERATOR.get() is original_provider
    assert_restored(module, ex)
    assert ex.models[-1]["status"] == "failed"
    assert seed.records[-1]["status"] == "failed"
    assert seed.records[-1]["stages"][-1]["status"] == "failed"


@pytest.mark.parametrize("hook", ["grid", "static", "offer", "candidate", "seed"])
def test_mutated_scoped_hook_is_refused_and_restored(module, hook):
    ex = experiment(module)
    owner, name = {
        "grid": (module.grid, "prepare_frame"),
        "static": (module.static_replay, "prepare_grid"),
        "offer": (tracking.IncrementalTracker, "offer"),
        "candidate": (tracking.IncrementalTracker, "_candidate"),
        "seed": (module.ReverseSeedExperiment, "fit"),
    }[hook]
    with pytest.raises(RuntimeError, match="hooks changed"), ex.active():
        setattr(owner, name, lambda *args: None)
    assert_restored(module, ex)


def test_unconsumed_preparation_and_nested_scope_refused(module):
    ex = experiment(module)
    with pytest.raises(ValueError, match="Unconsumed"), ex.active():
        ex.prepare(curved_raw(), target_points=900, depth_band=DepthBand(60, 300))
    assert_restored(module, ex)
    outer, inner = experiment(module), experiment(module)
    with outer.active(), pytest.raises(RuntimeError, match="nonnested"), inner.active():
        pass
    assert_restored(module, outer)


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "unknown"},
        {"radius": 2},
        {"radius": True},
        {"maximum_weight": 0},
        {"maximum_weight": True},
        {"maximum_weight": float("nan")},
        {"capacity": "unknown"},
        {"seed_profile": "unknown"},
    ],
)
def test_closed_configuration_guards(module, change):
    options = dict(
        profile="observe-pooled",
        capacity="ordinary-900",
        seed_profile="observe-forward",
        radius=5,
        maximum_weight=1.0,
    )
    with pytest.raises(ValueError):
        module.OrganizedModelExperiment(**(options | change))


@pytest.mark.parametrize("seed_profile", ["observe-forward", "reverse-forward"])
def test_changed_synthetic_model_keeps_reverse_fields_original(module, seed_profile):
    ex = experiment(module, seed=seed_profile, profile="organized-confidence")
    seed = module.ReverseSeedExperiment(seed_profile)
    solver = importlib.import_module("scanner_solver_mechanisms").SolverExperiment("centered")
    tracker = tracking.IncrementalTracker(backend="scipy", policy=ex.policy)

    def provider(points, neighbours):
        return icp_plane.estimate_normals(points, neighbours=neighbours, backend="scipy")

    with ex.active(), seed.active(), solver.active(), icp_plane.normal_acceleration(provider):
        for index in range(6):
            cloud = ex.prepare(curved_raw(), target_points=900, depth_band=DepthBand(60, 300))
            solver.mark_frame(index)
            assert tracker.offer(cloud.points).accepted
    assert_restored(module, ex)
    queries = [q for model in ex.models for q in model["queries"]]
    assert any(
        q["actual_sha256"] != q["baseline"]["sha256"]
        for q in queries
        if q["direction"] == "forward"
    )
    assert all(
        q["actual_sha256"] == q["baseline"]["sha256"]
        for q in queries
        if q["direction"] == "reverse"
    )
    assert all(q["usable"] == q["baseline"]["usable"] for q in queries)
