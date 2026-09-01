"""What tests.shell_discriminant_replay still promises now that it has no fusion arm.

Two promises, and both are the kind that only fail on the day something else
has already gone wrong, so they are exercised here rather than read:

* the replay is read-only and PROVES it on EVERY exit path once a baseline
  fingerprint exists - the normal return, a refusal, an unexpected error - and
  a capture it changed is reported as the finding, dominating whatever exit
  code the error would otherwise have produced;
* the module contains no fusion, no surface extraction and no shell-population
  code at all, so it cannot report a physical shell population, not even on
  the path where the readiness policy raises no reason.

Everything here is offline and fast. The captures are tiny temporary
directories of throwaway bytes, no device is opened, and the one test that
walks past registration drives stubs in place of the pipeline: it exercises
this module's control flow, and it manufactures no physical claim because the
path under test publishes no physical claim to manufacture.
"""

from __future__ import annotations

import json

import pytest

from scanner3d import pipeline
from scanner3d.errors import ScannerError
from tests import shell_discriminant_replay as module
from tests.surface_oracle_replay import ReplayRefused


def write_capture(root, frames):
    """A tiny manifest-less capture, bound by its own frame digests."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        (root / f"frame_{index:04d}.raw").write_bytes(bytes([index + 1]) * 64)
    return root


def digests(capture):
    """Every file in the capture and its bytes, for an independent before/after check."""
    return {path.name: path.read_bytes() for path in sorted(capture.rglob("*")) if path.is_file()}


def published(path):
    return json.loads(path.read_text(encoding="utf-8"))


class StubEvidence:
    """Stands in for the readiness evidence objects, which this module only serialises."""

    def as_dict(self):
        return {"stub": True}


class StubFailure:
    """A readiness failure with the two members this module reads: reason and as_dict."""

    reason = "the sweep does not go all the way round"

    def as_dict(self):
        return {"reason": self.reason}


class StubSweep:
    """The members of a reconstruct_sweep result this module actually touches."""

    frames = 3
    revisits_found = 0
    optimised = False
    poses = ((), (), ())


class StubPrepared:
    points = ()


def stub_pipeline(monkeypatch, *, failures):
    """Drive replay() past registration without a device, a pipeline or a real capture."""
    monkeypatch.setattr(module, "prepare_frame", lambda raw, target_points: StubPrepared())
    monkeypatch.setattr(pipeline, "reconstruct_sweep", lambda clouds, backend: StubSweep())
    monkeypatch.setattr(module, "pose_evidence", lambda clouds, poses: (StubEvidence(), failures))
    monkeypatch.setattr(module, "coverage_evidence", lambda clouds, poses: (StubEvidence(), []))


def test_a_capture_left_byte_identical_is_reported_unchanged_and_the_run_succeeds(tmp_path):
    capture = write_capture(tmp_path / "capture", 2)
    before = digests(capture)
    out = tmp_path / "report.json"

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 0
    assert digests(capture) == before
    document = published(out)
    assert document["read_only"]["unchanged"] is True
    assert document["read_only"]["changed"] == []
    assert document["read_only"]["appeared"] == []


def test_a_capture_with_fewer_than_three_frames_stops_at_frame_count_with_no_population(tmp_path):
    capture = write_capture(tmp_path / "capture", 2)

    report = module.replay(capture, limit=None)

    assert report["stopped_at"] == "frame_count"
    assert report["shell_population"] is None
    assert report["shell_population_available"] is False
    assert report["shell_population_unavailable_reason"]


def test_a_capture_that_cannot_be_bound_at_all_is_refused_and_still_proves_read_only(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    out = tmp_path / "report.json"

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 2
    document = published(out)
    assert document["stopped_at"] == "refused"
    assert document["read_only"]["unchanged"] is True
    assert document["shell_population"] is None


def test_an_unexpected_scanner_error_still_publishes_the_after_fingerprint_comparison(
    tmp_path, monkeypatch
):
    capture = write_capture(tmp_path / "capture", 2)
    before = digests(capture)
    out = tmp_path / "report.json"

    def explode(capture_path, *, limit):
        raise ScannerError("the replay fell over half way through")

    monkeypatch.setattr(module, "replay", explode)

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 1
    assert digests(capture) == before
    document = published(out)
    assert document["stopped_at"] == "error"
    assert "fell over half way through" in document["stopped_because"]
    assert document["read_only"]["unchanged"] is True
    assert document["shell_population"] is None
    assert document["shell_population_available"] is False


def test_a_failure_that_also_mutates_the_capture_exits_three_rather_than_the_error_code(
    tmp_path, monkeypatch
):
    capture = write_capture(tmp_path / "capture", 2)
    out = tmp_path / "report.json"

    def explode_and_write(capture_path, *, limit):
        (capture_path / "frame_0000.raw").write_bytes(b"rewritten by a run that promised not to")
        raise ScannerError("the replay fell over, after touching the capture")

    monkeypatch.setattr(module, "replay", explode_and_write)

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 3
    document = published(out)
    assert document["read_only"]["unchanged"] is False
    assert document["read_only"]["changed"] == ["frame_0000.raw"]
    assert document["stopped_at"] == "error"


def test_a_refusal_that_mutates_the_capture_also_exits_three(tmp_path, monkeypatch):
    capture = write_capture(tmp_path / "capture", 2)
    out = tmp_path / "report.json"

    def refuse_and_write(capture_path, *, limit):
        (capture_path / "frame_9999.raw").write_bytes(b"a file this run had no business creating")
        raise ReplayRefused("this capture cannot be bound")

    monkeypatch.setattr(module, "replay", refuse_and_write)

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 3
    document = published(out)
    assert document["read_only"]["unchanged"] is False
    assert document["read_only"]["appeared"] == ["frame_9999.raw"]


def test_an_error_that_is_not_a_scanner_error_is_reported_rather_than_propagated(
    tmp_path, monkeypatch
):
    capture = write_capture(tmp_path / "capture", 2)
    out = tmp_path / "report.json"

    def explode(capture_path, *, limit):
        raise RuntimeError("something nobody predicted")

    monkeypatch.setattr(module, "replay", explode)

    code = module.main(["--capture", str(capture), "--out", str(out), "--quiet"])

    assert code == 1
    document = published(out)
    assert document["stopped_at"] == "unexpected_error"
    assert document["read_only"]["unchanged"] is True


def test_the_module_exposes_no_fusion_extraction_or_shell_population_path_any_more():
    removed = (
        "_closed_shell_records",
        "BACKEND",
        "MIN_DEPTH",
        "MAX_DEPTH",
        "MIN_WEIGHT",
        "TRUNCATION_VOXELS",
        "WEIGHT_POLICY_NAME",
        "TsdfConfig",
        "TsdfVolume",
        "VolumeBounds",
        "WeightPolicy",
        "NoSurfaceFound",
        "extract_surface",
        "choose_tsdf_resolution",
        "model_points",
        "upright_pose",
        "edge_census",
        "contract",
        "math",
    )
    for name in removed:
        assert not hasattr(module, name), f"{name} survived the removal of the fusion branch"


def test_the_module_says_plainly_that_it_reproduces_registration_and_readiness_only():
    text = (module.__doc__ or "").lower()
    assert "registration and readiness only" in text
    assert "fusion arm" in text
    joined = " ".join(module.LIMITATIONS).lower()
    assert "registration and readiness only" in joined


def test_a_capture_that_passes_readiness_reports_no_shell_population(tmp_path, monkeypatch):
    capture = write_capture(tmp_path / "capture", 3)
    stub_pipeline(monkeypatch, failures=[])

    report = module.replay(capture, limit=None)

    assert report["readiness_reasons"] == []
    assert report["ready_before_fusion"] is True
    assert report["stopped_at"] == module.FUSION_NOT_IMPLEMENTED
    assert report["shell_population"] is None
    assert report["shell_population_available"] is False
    assert "not implemented" in str(report["shell_population_unavailable_reason"])
    timing = report["timing"]
    assert isinstance(timing, dict)
    assert "fuse_s" not in timing
    assert "resolution" not in report
    assert module.render(report).splitlines()[0].startswith("Observable-shell discriminator")


def test_a_capture_the_readiness_policy_refuses_stops_at_readiness_with_no_population(
    tmp_path, monkeypatch
):
    capture = write_capture(tmp_path / "capture", 3)
    stub_pipeline(monkeypatch, failures=[StubFailure()])

    report = module.replay(capture, limit=None)

    assert report["readiness_reasons"] == [StubFailure.reason]
    assert report["ready_before_fusion"] is False
    assert report["stopped_at"] == "readiness"
    assert report["shell_population"] is None
    assert report["shell_population_available"] is False


def test_a_capture_directory_that_does_not_exist_is_refused_without_an_exception(tmp_path):
    missing = tmp_path / "nowhere"

    code = module.main(["--capture", str(missing), "--quiet"])

    assert code == 2


def test_binding_a_missing_capture_raises_the_shared_refusal(tmp_path):
    with pytest.raises(ReplayRefused):
        module.replay(tmp_path / "nowhere", limit=None)
