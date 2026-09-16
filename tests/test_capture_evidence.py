"""Diagnostic evidence cannot publish frames or hide an incomplete journal."""

import hashlib
import json

from scanner3d import capture_evidence as module
from scanner3d.capture_sequence import read_capture_sequence
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.v4l2_stream import FrameRecord


def record(index=0):
    return FrameRecord(index, index + 10, 2, EXPECTED_FRAME_BYTES, 123.5 + index, 222.0, True)


def rows(root):
    path = root / "diagnostics" / module.JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_failed_capture_keeps_host_timing_and_bounded_raw_but_no_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_RAW_FRAMES", 2)
    notices = []
    raw = bytes(EXPECTED_FRAME_BYTES)
    evidence = module.CaptureEvidence(tmp_path, dict(device_settings_readback=None), notices.append)
    evidence.delivery(raw, record(0), "running")
    for index in range(1, 5):
        evidence.delivery(raw, record(index), "recovering")
    evidence.close(stream_returned=False, tracking_lost=True)
    journal = rows(tmp_path)
    assert len([r for r in journal if r["event"] == "delivery"]) == 5
    assert journal[1]["monotonic_s"] == 123.5
    assert journal[1]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert journal[1]["diagnostic_raw"] is None
    assert journal[-1]["diagnostic_raw_saved"] == 2
    assert journal[-1]["diagnostic_raw_omitted"] == 2
    assert len(notices) == 1
    assert len(list(tmp_path.rglob("delivery_*.raw"))) == 2
    assert not list(tmp_path.rglob("frame_*.raw"))
    assert not read_capture_sequence(tmp_path).paths


def test_event_limit_retains_a_footer_and_does_not_write_unindexed_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_EVENTS", 1)
    notices = []
    evidence = module.CaptureEvidence(tmp_path, {}, notices.append)
    evidence.event("extra", arbitrary=1)
    evidence.delivery(bytes(EXPECTED_FRAME_BYTES), record(), "lost")
    evidence.close(stream_returned=False)
    evidence.close()
    journal = rows(tmp_path)
    assert [r["event"] for r in journal] == ["start", "end"]
    assert journal[-1]["journal_limited"]
    assert journal[-1]["diagnostic_raw_omitted"] == 1
    assert not list(tmp_path.rglob("*.raw"))
    assert len(notices) == 1


def test_existing_journal_is_never_replaced(tmp_path):
    first = module.CaptureEvidence(tmp_path, {}, lambda _: None)
    first.close()
    path = tmp_path / "diagnostics" / module.JOURNAL_NAME
    before = path.read_bytes()
    notices = []
    second = module.CaptureEvidence(tmp_path, {}, notices.append)
    second.delivery(bytes(EXPECTED_FRAME_BYTES), record(), "lost")
    second.close()
    assert second.failed and len(notices) == 1
    assert path.read_bytes() == before


def test_raw_write_failure_is_visible_and_does_not_raise_from_callback(tmp_path, monkeypatch):
    notices = []
    evidence = module.CaptureEvidence(tmp_path, {}, notices.append)
    original = module.Path.open

    def failed(path, *args, **kwargs):
        if path.suffix == ".raw":
            raise OSError("synthetic disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.Path, "open", failed)
    evidence.delivery(bytes(EXPECTED_FRAME_BYTES), record(), "lost")
    evidence.close()
    assert evidence.failed and evidence.closed
    assert len(notices) == 1 and "disk full" in notices[0]
    assert not any(r["event"] == "end" for r in rows(tmp_path))


def test_nonfinite_evidence_is_refused_without_relaxing_json(tmp_path):
    notices = []
    evidence = module.CaptureEvidence(tmp_path, {}, notices.append)
    evidence.event("invalid", value=float("nan"))
    evidence.close()
    assert evidence.failed and len(notices) == 1
    assert [r["event"] for r in rows(tmp_path)] == ["start"]


def test_resume_transition_copy_is_idempotent_and_not_a_second_delivery(tmp_path):
    evidence = module.CaptureEvidence(tmp_path, {}, lambda _: None)
    raw = bytes(EXPECTED_FRAME_BYTES)
    evidence.delivery(raw, record(), "paused")
    first = evidence.retain_recovery(raw, record())
    assert evidence.retain_recovery(raw, record()) == first
    evidence.close()
    assert len([r for r in rows(tmp_path) if r["event"] == "delivery"]) == 1
    assert len([r for r in rows(tmp_path) if r["event"] == "recovery_raw"]) == 1
    assert (tmp_path / first).read_bytes() == raw
