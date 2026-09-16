"""Live evidence survives loss/Stop without altering the authoritative scan."""

import json

import pytest

pytest.importorskip("PySide6")

from scanner3d import capture_evidence as evidence_module
from scanner3d.capture_sequence import TrackingLossUnresolved, read_capture_sequence
from scanner3d.geometry import Rigid
from scanner3d.gui import TentativeSegment
from tests.fake_v4l2 import short_frame, stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    accepting,
    alignments,
    capture_dir,
    clouds,
    decisions,
    live,
    models,
    refusing,
    start_capture,
    tentative_clouds,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "decisions",
    "live",
    "models",
    "stream_kernel",
    "stream_node",
    "tentative_clouds",
]


def journal(window):
    path = window._capture_dir / "diagnostics" / evidence_module.JOURNAL_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recovery_raw_and_all_delivery_times_survive_unresolved_loss(
    stream_kernel, capture_dir, live, tentative_clouds, decisions
):
    live(failed_alignments={12, 13, 14})
    decisions([refusing("no overlap")])
    window = CaptureWindow(frames=30)
    stream_kernel([whole_frame(i) for i in range(30)])
    start_capture(window)
    rows = journal(window)
    deliveries = [r for r in rows if r["event"] == "delivery"]
    assert [r["record"]["index"] for r in deliveries] == list(range(30))
    assert all(isinstance(r["monotonic_s"], float) for r in deliveries)
    assert [r["delivery_index"] for r in rows if r["event"] == "saved"] == list(range(15))
    assert len([r for r in rows if r["event"] == "tracking"]) == 15
    assert rows[-1]["outcome"]["tracking_lost"]
    assert rows[-1]["diagnostic_raw_saved"] == 15
    assert len(list(window._capture_dir.rglob("frame_*.raw"))) == 15
    assert len(list(window._capture_dir.rglob("delivery_*.raw"))) == 15
    with pytest.raises(TrackingLossUnresolved):
        read_capture_sequence(window._capture_dir)


def test_short_deliveries_have_metadata_but_never_fabricated_raw(stream_kernel, capture_dir, live):
    live()
    window = CaptureWindow(frames=3)
    stream_kernel([whole_frame(10), short_frame(), whole_frame(11)])
    start_capture(window)
    rows = journal(window)
    assert len([r for r in rows if r["event"] == "delivery"]) == 2
    shorts = [r for r in rows if r["event"] == "short_delivery"]
    assert len(shorts) == 1 and shorts[0]["record"]["index"] == 1
    assert not shorts[0]["record"]["complete"]
    assert not list(window._capture_dir.rglob("delivery_*.raw"))


def test_diagnostic_open_failure_does_not_prevent_capture_or_cleanup(
    stream_kernel, capture_dir, live, monkeypatch
):
    live()
    original = evidence_module.Path.open

    def failed(path, *args, **kwargs):
        if path.name == evidence_module.JOURNAL_NAME:
            raise OSError("synthetic diagnostic disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(evidence_module.Path, "open", failed)
    window = CaptureWindow(frames=3)
    stream_kernel([whole_frame(i) for i in range(3)])
    start_capture(window)
    assert window.result.delivered == 3
    assert len(read_capture_sequence(window._capture_dir).paths) == 3
    assert any("diagnostics incomplete" in str(r) for r in window.worker.progress.emitted)


def test_operator_stop_keeps_delivery_mapping_and_does_not_relabel_validity(
    stream_kernel, capture_dir, live
):
    live()
    window = CaptureWindow(frames=5)
    window.worker.saved.connect(lambda count: window._stop() if count == 2 else None)
    stream_kernel([whole_frame(i) for i in range(5)])
    start_capture(window)
    rows = journal(window)
    assert rows[-1]["outcome"]["stream_returned"]
    assert not rows[-1]["outcome"]["tracking_lost"]
    assert rows[-1]["deliveries_seen"] == 2
    assert len(read_capture_sequence(window._capture_dir).paths) == 2


def test_published_recovery_maps_exact_delivery_indices(
    stream_kernel, capture_dir, live, tentative_clouds, decisions
):
    live(failed_alignments={12, 13, 14})
    decisions([accepting(Rigid.identity())])
    window = CaptureWindow(frames=24)
    stream_kernel([whole_frame(i) for i in range(24)])
    start_capture(window)
    published = [r for r in journal(window) if r["event"] == "segment_published"]
    assert len(published) == 1
    assert published[0]["first_frame_index"] == 15
    assert published[0]["delivery_indices"] == [15, 16, 17]
    assert published[0]["segment_kind"] == "recovery"
    sequence = read_capture_sequence(window._capture_dir)
    assert [frame.index for frame in sequence.excluded] == [12, 13, 14]


def test_tentative_provenance_slides_with_the_raw_frames_and_clears():
    segment = TentativeSegment.empty()
    for index in range(3):
        segment.add(bytes([index]), (), Rigid.identity(), None, delivery_index=index + 20)
    segment._drop_oldest()
    assert segment.delivery_indices == [21, 22]
    assert segment.raws == [bytes([1]), bytes([2])]
    segment.clear()
    assert segment.delivery_indices == []
