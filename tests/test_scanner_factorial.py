"""Independent F1 factors, no silent qualification, complete omission accounting."""

import hashlib
import importlib
import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane
from scanner3d.geometry import Rigid


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return SimpleNamespace(
        **{
            short: importlib.import_module(name)
            for short, name in (
                ("config", "scanner_factorial_config"),
                ("cell", "scanner_factorial_replay"),
                ("motion", "scanner_motion_replay"),
                ("native", "scanner_model_normals"),
                ("rank", "scanner_pixel_rank"),
                ("grid", "scanner_pixel_grid"),
                ("settling", "scanner_pose_settling"),
            )
        }
    )


def test_exact_closed_product_and_frozen_values(modules):
    config = modules.config
    assert len(config.CONFIGURATIONS) == len({c.identifier for c in config.CONFIGURATIONS}) == 12
    assert {c.values for c in config.CONFIGURATIONS} == set(
        product(("stride", "rank", "grid"), ("pooled", "native"), ("residual", "settling"))
    )
    with pytest.raises(FrozenInstanceError):
        config.CONFIGURATIONS[0].sampling = "rank"
    for values in (
        ("unknown", "pooled", "residual"),
        ("grid", "unknown", "settling"),
        ("stride", "native", "unknown"),
    ):
        with pytest.raises(ValueError, match="Unknown factorial"):
            config.Factors(*values)


def configure_empty_motion(modules, monkeypatch, *, fail=False):
    motion = modules.motion
    raw = bytes(512000)
    monkeypatch.setattr(
        motion,
        "load_inputs",
        lambda *_a, **_k: (
            [],
            [raw] * 100,
            motion.DepthBand(60, 300),
            [{"index": i} for i in range(100)],
        ),
    )
    monkeypatch.setattr(
        motion, "check_coherence", lambda *_a, **_k: SimpleNamespace(as_dict=lambda: {})
    )
    events = []
    original_iteration = icp_plane._iterate

    def preparation(label):
        def run(*_args, **_kwargs):
            events.append((label, icp_plane._iterate is modules.settling.iterate))
            assert icp_plane._NORMAL_ACCELERATOR.get() is not None
            if fail:
                raise RuntimeError("injected preparation failure")
            return SimpleNamespace(points=(), deprojected_points=0, clipped_points=0)

        return run

    for module, label in ((motion, "stride"), (modules.rank, "rank"), (modules.grid, "grid")):
        monkeypatch.setattr(module, "prepare_frame", preparation(label))
    pooled, native = motion.IncrementalTracker, modules.native.NativeModelTracker
    selected = []

    def tracker(factory, name):
        def make(**kwargs):
            selected.append(name)
            return factory(**kwargs)

        return make

    monkeypatch.setattr(motion, "IncrementalTracker", tracker(pooled, "pooled"))
    monkeypatch.setattr(modules.native, "NativeModelTracker", tracker(native, "native"))
    return events, selected, original_iteration


@pytest.mark.parametrize(
    "values",
    list(product(("stride", "rank", "grid"), ("pooled", "native"), ("residual", "settling"))),
)
def test_all_twelve_factors_activate_independently_and_never_qualify(
    modules, monkeypatch, tmp_path, values
):
    events, selected, iteration = configure_empty_motion(modules, monkeypatch)
    provider = icp_plane._NORMAL_ACCELERATOR.get()
    output = tmp_path / "cell"
    modules.motion.run(
        tmp_path, output, 100, (1,), normals_backend="cpu", diagnostic_factors=values
    )
    assert selected == [values[1]]
    assert events == [(values[0], values[2] == "settling")] * 3
    assert icp_plane._iterate is iteration and icp_plane._NORMAL_ACCELERATOR.get() is provider
    report = json.loads((output / "report.json").read_text())
    expected = modules.config.Factors(*values)
    for record in (report, report["runs"][0]):
        assert record["diagnostic_factors"] == expected.as_dict()
        assert record["quality_qualification_allowed"] is False
        assert record["sampling_rule"] == expected.sampling_rule
        assert record["model_normal_rule"] == expected.model_normal_rule
        assert record["convergence_rule"] == expected.convergence_rule
    result = report["runs"][0]
    assert len(report["input_frames"]) == 100  # Includes the unoffered suffix.
    assert result["offered"] == 3 and result["tracking_stopped"] and result["accepted"] == 0


def test_combined_exception_restores_context_and_retains_failure(modules, monkeypatch, tmp_path):
    events, selected, iteration = configure_empty_motion(modules, monkeypatch, fail=True)
    provider = icp_plane._NORMAL_ACCELERATOR.get()
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected preparation"):
        modules.motion.run(
            tmp_path,
            output,
            100,
            (1,),
            normals_backend="cpu",
            diagnostic_factors=("grid", "native", "settling"),
        )
    assert events == [("grid", True)] and selected == ["native"]
    assert icp_plane._iterate is iteration and icp_plane._NORMAL_ACCELERATOR.get() is provider
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["quality_qualification_allowed"] is False
    assert report["failed_run"]["offered_before_error"] == 0
    assert report["failed_run"]["diagnostic_factors"]["convergence"] == "settling"
    assert len(report["input_frames"]) == 100


def test_legacy_profile_cannot_silently_override_factors(modules, tmp_path):
    with pytest.raises(ValueError, match="cannot be mixed"):
        modules.motion.run(
            tmp_path,
            tmp_path / "unused",
            100,
            (1,),
            diagnostic_reference_profile="pixel-rank",
            diagnostic_factors=("grid", "native", "settling"),
        )
    assert not (tmp_path / "unused").exists()


def static_fixture(modules, monkeypatch, tmp_path, *, error=None):
    raw = bytes(512000)
    capture = tmp_path / "raw"
    capture.mkdir()
    records = [
        dict(
            path=f"frame_{i:06d}.raw",
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            monotonic_s=float(i),
            delivery_index=i,
        )
        for i in range(320)
    ]
    parent = dict(status="completed", restoration_verified=True)
    if error == "restoration":
        parent["restoration_verified"] = False
    if error == "duplicate":
        records[-1]["path"] = records[0]["path"]
    if error == "time":
        records[-1]["monotonic_s"] = float("nan")
    if error == "time_order":
        records[-1]["monotonic_s"] = records[0]["monotonic_s"]
    if error == "escape":
        records[-1]["path"] = "../frame.raw"
    if error == "hash":
        records[-1]["sha256"] = "0" * 64
    if error == "short":
        records[-1]["bytes"] = 1
    if error == "count":
        records.pop()
    (tmp_path / "report.json").write_text(json.dumps(parent))
    (capture / "capture-progress.json").write_text(
        json.dumps(dict(frames=records, capture_error=None))
    )
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: raw if path.suffix == ".raw" else original(path)
    )
    return capture


@pytest.mark.parametrize(
    "error", ["restoration", "duplicate", "time", "time_order", "escape", "hash", "short", "count"]
)
def test_static_loader_rejects_corrupt_or_incomplete_evidence(
    modules, monkeypatch, tmp_path, error
):
    capture = static_fixture(modules, monkeypatch, tmp_path, error=error)
    with pytest.raises(ValueError):
        modules.cell.load_static(capture)


def test_static_loader_preserves_complete_prefix_and_provenance(modules, monkeypatch, tmp_path):
    capture = static_fixture(modules, monkeypatch, tmp_path)
    raws, inputs, provenance = modules.cell.load_static(capture)
    assert len(raws) == len(inputs) == 320 and inputs[0]["delivery_index"] == 0
    assert inputs[-1]["delivery_index"] == 319
    assert (
        provenance["raw_progress_sha256"]
        == hashlib.sha256((capture / "capture-progress.json").read_bytes()).hexdigest()
    )


def test_static_checks_keep_every_unoffered_endpoint(modules, monkeypatch):
    monkeypatch.setattr(modules.cell, "projection", lambda *_a, **_k: {"agreement": 1.0})
    rows = modules.cell.static_checks(
        [b""] * 320, {0: Rigid.identity()}, modules.cell.DepthBand(60, 300)
    )
    assert [r["source_index"] for r in rows] == [*range(16, 320, 16), 319]
    assert all(not r["checked"] and r["raw_placed"] is None for r in rows)
    assert all(r["raw_identity"]["agreement"] == 1.0 for r in rows)
    with pytest.raises(ValueError, match="full 320"):
        modules.cell.static_checks([b""] * 319, {}, modules.cell.DepthBand(60, 300))


@pytest.mark.parametrize("failure", [None, "preparation", "gpu_inactive"])
def test_static_combination_restores_context_and_preserves_terminal_evidence(
    modules, monkeypatch, tmp_path, failure
):
    cell = modules.cell
    iteration = icp_plane._iterate
    provider = icp_plane._NORMAL_ACCELERATOR.get()
    monkeypatch.setattr(
        cell,
        "load_static",
        lambda *_a: ([bytes(512000)] * 320, [{"index": i} for i in range(320)], {}),
    )

    @contextmanager
    def gpu(_parent, emit):
        if failure != "gpu_inactive":
            emit("GPU-assisted tracking: test double, not real hardware")
        with icp_plane.normal_acceleration(lambda p, n: icp_plane.NormalField((), n, ())):
            yield

    def preparation(*_a, **_k):
        assert icp_plane._iterate is modules.settling.iterate
        assert icp_plane._NORMAL_ACCELERATOR.get() is not None
        if failure == "preparation":
            raise RuntimeError("injected static preparation")
        return SimpleNamespace(points=(), deprojected_points=0, clipped_points=0)

    monkeypatch.setattr(cell, "live_normals", gpu)
    monkeypatch.setattr(cell, "prepare_grid", preparation)
    monkeypatch.setattr(cell, "projection", lambda *_a, **_k: {"agreement": 1.0})
    output = tmp_path / "static"
    factors = modules.config.Factors("grid", "native", "settling")
    if failure:
        with pytest.raises(RuntimeError):
            cell.run_static(tmp_path, output, factors)
    else:
        cell.run_static(tmp_path, output, factors)
    report = json.loads((output / "report.json").read_text())
    assert report["quality_qualification_allowed"] is False
    assert len(report["input_frames"]) == 320 and report["diagnostic_factors"] == factors.as_dict()
    assert report["accepted"] == 0 and report["offered"] == (0 if failure else 3)
    assert report["status"] == ("failed" if failure else "completed")
    assert report["tracking_stopped"] is (failure is None)
    if not failure:
        assert len(report["stationary_checks"]) == 20
        assert all(not r["checked"] for r in report["stationary_checks"])
    assert icp_plane._iterate is iteration and icp_plane._NORMAL_ACCELERATOR.get() is provider
