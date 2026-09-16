"""Fixed revisit selection cannot hide no-period data, bad pairs or missing poses."""

import importlib
import json
from pathlib import Path

import pytest

from scanner3d import coherence
from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.geometry import Rigid
from scanner3d.preparation import DepthBand


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_frozen_revisits")


@pytest.mark.parametrize("moving", [False, True])
def test_matcher_instrumentation_keeps_exact_results_and_all_anchor_records(
    audit, monkeypatch, moving
):
    raws = [bytes([i % 30 if moving else 0]) for i in range(120)]
    monkeypatch.setattr(coherence, "_grid", lambda raw, stride: [raw[0] + 100] * 256)
    original_type = coherence.PeriodSample
    expected = coherence.image_period(raws, depth_scale=DEPTH_SCALE_METRES, samples=40)
    result = audit.raw_schedule(raws)
    assert coherence.PeriodSample is original_type
    assert result["period_estimate"] == (None if expected is None else expected.as_dict())
    assert len(result["anchors"]) == 40
    assert any(r["credible"] for r in result["anchors"]) is moving
    if not moving:
        assert result["endpoint_separated_ordinals"] == []


def test_endpoint_separation_is_greedy_in_fixed_order(audit):
    rows = [
        dict(frame=a, best_frame=b, credible=True) for a, b in ((0, 20), (4, 24), (5, 25), (10, 20))
    ]
    assert audit.separated_ordinals(rows) == [0, 2]


def test_matcher_exception_restores_original_class(audit, monkeypatch):
    original_type = coherence.PeriodSample

    def failed(*args, **kwargs):
        assert coherence.PeriodSample is not original_type
        raise RuntimeError("injected matcher failure")

    monkeypatch.setattr(coherence, "image_period", failed)
    with pytest.raises(RuntimeError, match="injected matcher failure"):
        audit.raw_schedule([b"raw"] * 120)
    assert coherence.PeriodSample is original_type


def test_no_comparable_raw_frames_retains_every_anchor(audit, monkeypatch):
    monkeypatch.setattr(coherence, "_grid", lambda raw, stride: [0.0] * 256)
    result = audit.raw_schedule([b"raw"] * 120)
    assert result["period_estimate"] is None
    assert len(result["anchors"]) == 40
    assert all(row["reason"] == "no_comparable_later_frame" for row in result["anchors"])
    assert result["endpoint_separated_ordinals"] == []


@pytest.mark.parametrize("frames", [99, 3001])
def test_frame_count_remains_bounded_and_failure_is_retained(audit, tmp_path, frames):
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="Bounded frame count"):
        audit.run(tmp_path, output, [tmp_path / "replay.json"], frames=frames)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert not report["quality_qualification_allowed"]


@pytest.mark.parametrize(
    "stopped,period,agreement,count,verdict,reason",
    [
        (True, {}, 0.99, 25, "rejected", "tracking_stopped"),
        (False, None, 0.99, 25, "inconclusive", "no_credible_image_period"),
        (False, {}, None, 25, "inconclusive", "credible_pairs_unplaced_or_uncheckable"),
        (False, {}, 0.99, 24, "inconclusive", "fewer_than_25_endpoint_separated_checkable_pairs"),
    ],
)
def test_failure_and_insufficient_evidence_cannot_pass(
    audit, monkeypatch, stopped, period, agreement, count, verdict, reason
):
    schedule = dict(
        anchors=[dict(frame=i, best_frame=i + 30, credible=True) for i in range(25)],
        period_estimate=period,
        endpoint_separated_ordinals=list(range(count)),
    )
    monkeypatch.setattr(audit, "projection", lambda *args: {"agreement": agreement})
    result = audit.evaluate(
        schedule,
        [b"raw"] * 55,
        dict.fromkeys(range(55), Rigid.identity()),
        DepthBand(60, 300),
        tracking_stopped=stopped,
    )
    assert result["diagnostic_verdict"] == verdict
    assert reason in result["reasons"]
    assert not result["quality_qualification_allowed"]
    assert len(result["checks"]) == 25


@pytest.mark.parametrize(
    "missing,bad,expected",
    [
        (False, False, "observed_checks_meet_margin"),
        (True, False, "inconclusive"),
        (False, True, "rejected"),
    ],
)
def test_unseparated_bad_pair_and_missing_pose_are_never_hidden(
    audit, monkeypatch, missing, bad, expected
):
    anchors = [dict(frame=i, best_frame=i + 30, credible=True) for i in range(26)]
    schedule = dict(
        anchors=anchors,
        period_estimate={"period_frames": 30},
        endpoint_separated_ordinals=list(range(25)),
    )
    identity = Rigid.identity()
    positions = {i: identity for i in range(56)}
    if missing:
        del positions[30]
    raw = [bytes([i]) for i in range(56)]
    calls = []

    def measured(raws, poses, band):
        calls.append(raws)
        return {"agreement": 0.8 if bad and raws[0] == bytes([25]) else 0.99}

    monkeypatch.setattr(audit, "projection", measured)
    result = audit.evaluate(schedule, raw, positions, DepthBand(60, 300), tracking_stopped=False)
    assert result["diagnostic_verdict"] == expected
    assert result["quality_qualification_allowed"] is False
    assert len(result["checks"]) == 26
    assert schedule["anchors"] == anchors
    if missing:
        assert result["uncheckable_or_unplaced_ordinals"] == [0]
    if bad:
        assert result["weakest_agreement"] == 0.8
        assert result["endpoint_separated_checked_pairs"] == 25
