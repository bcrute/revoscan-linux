"""Closed loop experiment cannot train on its held-out endpoints or lose poses."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scanner3d.geometry import Rigid
from scanner3d.posegraph_lm import se3_exp
from scanner3d.preparation import DepthBand


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_loop_probe")


def sample(early, late, credible=True):
    return dict(frame=early, best_frame=late, credible=credible)


def test_split_fixed_before_fit_and_protects_either_validation_endpoint(probe):
    samples = [
        sample(0, 500),
        sample(20, 520),
        sample(24, 600),
        sample(60, 560),
        sample(100, 563),
        sample(101, 601, False),
        sample(120, 700, False),
    ]
    rows = probe.partition_samples(samples)
    assert [r["role"] for r in rows] == [
        "fit",
        "validation",
        "fit",
        "validation",
        "fit",
        "validation",
        "fit",
    ]
    assert rows[0]["exclusion_reasons"] == []
    assert rows[2]["exclusion_reasons"] == ["within_four_frames_of_validation_endpoint"]
    assert rows[4]["exclusion_reasons"] == ["within_four_frames_of_validation_endpoint"]
    assert rows[5]["exclusion_reasons"] == ["raw_image_match_not_credible"]
    assert rows[6]["exclusion_reasons"] == ["raw_image_match_not_credible"]
    assert len(rows) == len(samples)  # No failed/omitted view disappears.


def test_interpolation_keeps_all_frames_exact_nodes_and_originals(probe):
    original = [Rigid.identity()] * 65
    nodes = [0, 32, 64]
    solved = [se3_exp([float(i), 0, 0, 0, 0, 0]) for i in range(3)]
    corrected = probe.interpolate_corrections(original, nodes, solved)
    assert len(corrected) == 65
    assert [corrected[i] for i in nodes] == solved
    assert corrected[16].translation == pytest.approx((0.5, 0, 0))
    assert corrected[48].translation == pytest.approx((1.5, 0, 0))
    assert original == [Rigid.identity()] * 65


def test_world_correction_composes_on_left_and_preserves_rigid_rotation(probe):
    original = [se3_exp([0, 0, 10, 0, i * 0.01, 0]) for i in range(65)]
    fixed = se3_exp([2, 3, 0, 0.1, 0, 0])
    nodes = [0, 32, 64]
    solved = [fixed.compose(original[i]) for i in nodes]
    corrected = probe.interpolate_corrections(original, nodes, solved)
    for old, new in zip(original, corrected, strict=True):
        assert new.apply((3, 4, 5)) == pytest.approx(fixed.apply(old.apply((3, 4, 5))))


@pytest.mark.parametrize("indices", [[1, 64], [0, 32], [0, 64, 32], [0, 0, 64]])
def test_interpolation_refuses_missing_or_ambiguous_endpoints(probe, indices):
    with pytest.raises(ValueError, match="Ordered"):
        probe.interpolate_corrections(
            [Rigid.identity()] * 65, indices, [Rigid.identity()] * len(indices)
        )


def test_no_returns_rejected_before_alignment(probe, monkeypatch):
    def must_not_fit(*args, **kwargs):
        raise AssertionError("No geometry must never reach the pose solver")

    monkeypatch.setattr(probe, "icp_point_to_plane", must_not_fit)
    transform, result = probe.fit_link([bytes(512000)] * 2, DepthBand(60, 300))
    assert transform is None and result["accepted"] is False
    assert result["reasons"] == ["raw_descriptor_not_comparable"]


def test_perfect_plane_is_not_a_proven_six_dof_closure(probe):
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    transform, result = probe.fit_link([raw, raw], DepthBand(60, 300))
    assert transform is None and result["accepted"] is False
    assert "Unusable" in result["error"]


def source_report(tmp_path):
    capture = tmp_path / "raw"
    capture.mkdir()
    raw = bytes(512000)
    (capture / "0.raw").write_bytes(raw)
    report = dict(
        status="completed",
        capture=str(capture),
        depth_band=dict(near=60, far=300),
        input_frames=[
            dict(path=str(capture / f"{i}.raw"), sha256=hashlib.sha256(raw).hexdigest())
            for i in range(100)
        ],
        runs=[
            dict(
                stride=1,
                accepted_source_indices=list(range(100)),
                poses=[Rigid.identity().as_dict()] * 100,
            )
        ],
    )
    path = tmp_path / "source.json"
    path.write_text(json.dumps(report))
    return path, report


def test_raw_hash_failure_leaves_failed_report_and_never_poses(probe, tmp_path):
    path, source = source_report(tmp_path)
    Path(source["input_frames"][0]["path"]).write_bytes(b"changed")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="hash changed"):
        probe.run(path, output)
    result = json.loads((output / "report.json").read_text())
    assert result["status"] == "failed"
    assert result["quality_qualification_allowed"] is False
    assert "corrected_poses" not in result and "selected_poses" not in result
    assert json.loads(path.read_text()) == source


def test_insufficient_loops_never_call_solver_or_invent_poses(probe, tmp_path, monkeypatch):
    path, _ = source_report(tmp_path)
    monkeypatch.setattr(probe, "_verified_raw", lambda *args: bytes(512000))
    monkeypatch.setattr(probe, "image_period", lambda *args, **kwargs: None)

    def must_not_fit(*args, **kwargs):
        raise AssertionError("Missing revisits cannot reach loop fitting")

    monkeypatch.setattr(probe, "fit_link", must_not_fit)
    result = probe.run(path, tmp_path / "insufficient")
    assert result["status"] == "insufficient"
    assert result["quality_qualification_allowed"] is False
    assert "corrected_poses" not in result and "selected_poses" not in result


def test_experiment_never_overwrites_existing_evidence(probe, tmp_path):
    output = tmp_path / "kept"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("original evidence")
    with pytest.raises(FileExistsError):
        probe.run(tmp_path / "absent-source.json", output)
    assert marker.read_text() == "original evidence"
