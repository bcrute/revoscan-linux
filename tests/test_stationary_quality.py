"""Offline matched-pixel quality checks; no scanner commands or raw capture."""

import ctypes
import importlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("analyze_stationary_quality")


@pytest.fixture
def synthetic_capture(tmp_path, analyzer, monkeypatch):
    metadata = {
        "status": "completed",
        "restoration_verified": True,
        "timings": [20000, 10000, 9000, 8000, 7000, 20000],
        "sample_times": [[i * 0.1 for i in range(40)]] * 6,
    }
    (tmp_path / "report.json").write_text(json.dumps(metadata))
    baseline = np.zeros((40, 14, 14), dtype=np.uint16)
    baseline[:, 2:12, 2:12] = 20000
    candidate = baseline.copy()
    candidate[:, 2:12, 2] = 0  # A real loss confined to the outer edge.
    monkeypatch.setattr(
        analyzer, "frames", lambda path: candidate if path.name == "3-8000" else baseline
    )
    return tmp_path


def test_edge_loss_cannot_hide_behind_an_unchanged_interior(analyzer, synthetic_capture):
    report = analyzer.analyze(synthetic_capture)
    baseline, candidate = report["runs"][0], report["runs"][3]
    assert report["stable_pixels"] == 100
    assert baseline["persistently_lost_stable_pixels"] == 0
    assert candidate["persistently_lost_stable_pixels"] == 10
    assert candidate["persistently_lost_stable_fraction"] == 0.1
    assert candidate["object_return_fraction"] == 0.9
    assert candidate["core_return_fraction"] == 1.0
    assert candidate["edge_return_fraction"] < 0.9
    assert candidate["settled_delivery_fps"] == pytest.approx(10.0)
    assert candidate["temporal_mad_mm_p50_p90"] == [0.0, 0.0]
    assert report["baseline_change_absolute_mm_p50_p90"] == [0.0, 0.0]


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("restoration_verified", False)])
def test_incomplete_or_unrestored_sweep_is_not_a_quality_result(
    analyzer, synthetic_capture, field, value
):
    path = synthetic_capture / "report.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="complete and verify restoration"):
        analyzer.analyze(synthetic_capture)


def test_raw_reader_requires_all_settled_frames(analyzer, tmp_path):
    with pytest.raises(ValueError, match="all 40 complete settled frames"):
        analyzer.frames(tmp_path)


@pytest.mark.parametrize("annotation", [{"quality_qualification_allowed": False}, {}])
def test_operator_validity_warning_prevents_quality_comparison(
    analyzer, synthetic_capture, annotation
):
    (synthetic_capture / "quality-validity.json").write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="Operator validity warning"):
        analyzer.analyze(synthetic_capture)


def test_baseline_version_read_uses_only_the_known_read_tuple(analyzer, monkeypatch):
    baseline = importlib.import_module("scanner_validation_baseline")
    payload = b"v2.8.7.20220610|v7.5.7.0426".ljust(60, b"\x00")

    def ioctl(fd, request, packed, mutate):
        assert (fd, request, mutate) == (17, 0xC0107521, True)
        unit, selector, query, length, address = struct.unpack("@BBBHP", packed)
        assert (unit, selector, query, length) == (3, 6, 0x81, 60)
        ctypes.memmove(address, payload, 60)

    monkeypatch.setattr(baseline.fcntl, "ioctl", ioctl)
    result = baseline.version(17)
    assert result["firmware_version"] == "v2.8.7.20220610"


def test_five_stage_ladder_uses_its_actual_final_baseline(analyzer, synthetic_capture):
    path = synthetic_capture / "report.json"
    metadata = json.loads(path.read_text())
    metadata["timings"] = [20000, 15000, 12000, 10000, 20000]
    metadata["exposures"] = [20000, 15000, 12000, 10000, 20000]
    metadata["sample_times"] = metadata["sample_times"][:5]
    path.write_text(json.dumps(metadata))
    report = analyzer.analyze(synthetic_capture)
    assert len(report["runs"]) == 5
    assert report["runs"][2]["exposure_us"] == 12000


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_invalid_timestamps_do_not_produce_a_rate(analyzer, synthetic_capture, bad):
    path = synthetic_capture / "report.json"
    metadata = json.loads(path.read_text())
    metadata["sample_times"][2][4] = bad
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="strictly increasing"):
        analyzer.analyze(synthetic_capture)


@pytest.mark.parametrize(
    "timings,exposures",
    [
        ((20000, 6000, 20000), None),
        ((20000, 12000, 20000), (20000, 15000, 20000)),
        ((20000, 12000), None),
        ((20000, 12000, 20000), (20000, 12000)),
    ],
)
def test_invalid_ladder_refused_before_hardware(
    analyzer, monkeypatch, tmp_path, timings, exposures
):
    capture = importlib.import_module("scanner_stationary_quality")
    monkeypatch.setattr(
        capture, "resolve_serial", lambda: pytest.fail("hardware preflight escaped")
    )
    with pytest.raises(ValueError, match="Bounded timing pairs"):
        capture.run(tmp_path / "capture", timings, exposures)
    assert not (tmp_path / "capture").exists()


def test_stationary_probe_closes_control_handle_on_initial_read_failure(
    analyzer, monkeypatch, tmp_path
):
    capture = importlib.import_module("scanner_stationary_quality")
    monkeypatch.setattr(capture, "resolve_serial", lambda: "test-only")
    monkeypatch.setattr(
        capture, "resolve_capture_node", lambda serial: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(capture.os, "open", lambda *args: 917)
    closed: list[int] = []
    monkeypatch.setattr(capture.os, "close", closed.append)

    def failed_read(fd):
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(capture, "state", failed_read)
    with pytest.raises(RuntimeError, match="synthetic telemetry failure"):
        capture.run(tmp_path / "capture")
    assert closed == [917]


@pytest.mark.parametrize("fresh", [False, True])
def test_firmware_gain_read_requires_a_fresh_named_reply(analyzer, monkeypatch, fresh):
    inventory = importlib.import_module("scanner_firmware_inventory")
    logs = iter(
        [
            "[ 10.100] old message",
            "[ %s] id_msg_cam_get_gain_reg [24]" % ("10.200" if fresh else "10.000"),
        ]
    )
    monkeypatch.setattr(inventory, "kernel_log", lambda fd: next(logs))
    calls = []
    monkeypatch.setattr(inventory, "xu", lambda *args: calls.append(args))
    if fresh:
        assert inventory.gain(17)[0] == 24
    else:
        with pytest.raises(RuntimeError, match="No fresh"):
            inventory.gain(17)
    assert calls == [(17, 7, 1, b"echo s 0x9a3 0 >/dev/rk_preisp".ljust(60, b"\x00"))]


def test_spatial_summary_preserves_missing_depth_and_nominal_scale(analyzer):
    spatial = importlib.import_module("stationary_spatial_report")
    raw = np.zeros((3, 2, 2), dtype=np.uint16)
    raw[:, 0, 0] = [20000, 20010, 20020]
    median, frequency, mad = spatial.summary(raw)
    assert median[0, 0] == pytest.approx(20010 * analyzer.DEPTH_SCALE_METRES)
    assert mad[0, 0] == pytest.approx(10 * analyzer.DEPTH_SCALE_METRES)
    assert frequency[0, 0] == 1
    assert frequency[1, 1] == 0
    assert np.isnan(median[1, 1])


@pytest.mark.parametrize("value", [0, 15, 16, 24, 32, 33, 255])
def test_closed_gain_values_and_command_shape(analyzer, monkeypatch, value):
    probe = importlib.import_module("scanner_gain_probe")
    calls = []
    monkeypatch.setattr(probe, "xu", lambda *args: calls.append(args))
    if value in (16, 24, 32):
        probe.set_gain(19, value)
        payload = f"echo s 0x903 {value} >/dev/rk_preisp".encode().ljust(60, b"\x00")
        assert calls == [(19, 7, 1, payload)]
    else:
        with pytest.raises(ValueError, match="bounded gain"):
            probe.set_gain(19, value)
        assert not calls


def test_invalid_gain_ladder_cannot_reach_hardware(analyzer, monkeypatch, tmp_path):
    capture = importlib.import_module("scanner_stationary_quality")
    monkeypatch.setattr(capture, "resolve_serial", lambda: pytest.fail("hardware escaped"))
    for gains in [(24, 255, 24), (24, 16), (16, 16, 24), ()]:
        with pytest.raises(ValueError, match="Bounded timing pairs"):
            capture.run(tmp_path / "raw", (20000, 15000, 20000), gains=gains)
    assert not (tmp_path / "raw").exists()


def test_endurance_stages_require_completed_cool_restored_predecessor(analyzer):
    soak = importlib.import_module("scanner_thermal_soak")
    soak.validate_stage(300, None)
    for duration in (1, 600, 3600, 900, 1800):
        with pytest.raises(ValueError):
            soak.validate_stage(duration, None)
    previous = dict(
        status="completed",
        restoration_verified=True,
        serial=soak.SERIAL,
        requested_timing_us=7000,
        duration_limit_s=300,
        thermal_samples=[dict(elapsed_s=295, cpu_c=67.3, npu_c=66.6)],
    )
    soak.validate_stage(900, previous)
    for change in [
        dict(status="failed"),
        dict(restoration_verified=False),
        dict(serial="other"),
        dict(duration_limit_s=900),
        dict(thermal_samples=[dict(elapsed_s=295, cpu_c=75, npu_c=67)]),
    ]:
        with pytest.raises(ValueError, match="successful"):
            soak.validate_stage(900, {**previous, **change})


def test_selected_clean_reference_must_match_its_bracketing_settings(analyzer, synthetic_capture):
    path = synthetic_capture / "report.json"
    metadata = json.loads(path.read_text())
    metadata["timings"] = [20000, 20000, 12000, 20000, 12000, 20000]
    metadata["gains"] = [24, 16, 32, 16, 32, 24]
    path.write_text(json.dumps(metadata))
    report = analyzer.analyze(synthetic_capture, reference_stages=(1, 3))
    assert report["reference_stages"] == [1, 3]
    for indices in [(1, 5), (3, 1), (1, 6)]:
        with pytest.raises(ValueError):
            analyzer.analyze(synthetic_capture, reference_stages=indices)


def test_endurance_presets_cannot_borrow_another_timing_predecessor(analyzer):
    soak = importlib.import_module("scanner_thermal_soak")
    soak.validate_stage(300, None, 14000)
    with pytest.raises(ValueError, match="presets"):
        soak.validate_stage(300, None, 6000)
    previous = dict(
        status="completed",
        restoration_verified=True,
        serial=soak.SERIAL,
        requested_timing_us=7000,
        duration_limit_s=300,
        thermal_samples=[dict(elapsed_s=295, cpu_c=65, npu_c=64)],
    )
    with pytest.raises(ValueError, match="predecessor"):
        soak.validate_stage(900, previous, 14000)
    soak.validate_stage(900, {**previous, "requested_timing_us": 14000}, 14000)


@pytest.mark.parametrize(
    "failure,quiet",
    [
        ("none", False),
        ("stream", False),
        ("restoration_gain", False),
        ("none", True),
        ("stream", True),
        ("restoration_gain", True),
        ("quiet_boundary", True),
        ("stale_telemetry", True),
    ],
)
@pytest.mark.parametrize("timing", [14000, 20000])
def test_soak_uses_requested_timing_and_restores_on_stream_failure(
    analyzer, monkeypatch, tmp_path, failure, quiet, timing
):
    import sys

    soak = importlib.import_module("scanner_thermal_soak")
    quiet = quiet or timing == 20000
    output = tmp_path / "soak.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["soak", "--execute", "--output", str(output), "--timing-us", str(timing)]
        + (["--quiet-after-stop"] if quiet else []),
    )
    events = []

    class Witness:
        def __init__(self, *, hold):
            assert not hold
            self.report = dict(observations=[dict(same_device=True)])

        def observe(self, phase, seconds):
            events.append((phase, seconds))

    def quiet_boundary(witness):
        assert "stream_closed" in events
        events.append("quiet_wait")
        if failure == "quiet_boundary":
            raise RuntimeError("synthetic quiet boundary failure")
        witness.report["settled_for_readback"] = True

    monkeypatch.setattr(soak, "UsbIdleWitness", Witness)
    monkeypatch.setattr(soak, "quiet_stop_observation", quiet_boundary)
    monkeypatch.setattr(soak, "resolve_serial", lambda: soak.SERIAL)
    monkeypatch.setattr(soak, "wait_for_capture_access", lambda **kw: SimpleNamespace(ready=True))
    monkeypatch.setattr(
        soak, "resolve_capture_node", lambda serial: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(soak.os, "open", lambda *args: 919)
    closed = []
    monkeypatch.setattr(soak.os, "close", closed.append)
    gain_reads = []

    def gain_read(fd):
        gain_reads.append(fd)
        if len(gain_reads) > 2 and quiet:
            assert "quiet_wait" in events
        if failure == "restoration_gain" and len(gain_reads) == 4:
            raise OSError("synthetic final gain read failure")
        return 24, ""

    monkeypatch.setattr(soak, "gain", gain_read)
    monkeypatch.setattr(soak, "temperatures", lambda fd: dict(cpu_c=60, npu_c=59))
    settings = dict(soak.BASELINE)
    writes = []

    def set_timing(fd, frame, exposure):
        if writes and frame == 20000 and quiet:
            assert "quiet_wait" in events
        writes.append((frame, exposure))
        settings.update(frame_time=frame, exposure_time=exposure)

    monkeypatch.setattr(soak, "set_timing", set_timing)
    monkeypatch.setattr(soak, "state", lambda fd: (dict(settings), ""))
    clock = [0.0]
    monkeypatch.setattr(soak, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def stream(*args, **kwargs):
        assert settings["frame_time"] == settings["exposure_time"] == timing
        assert kwargs["duration_s"] == 300
        assert not kwargs["should_stop"]()
        if failure == "stale_telemetry":
            clock[0] = 16.0
            assert kwargs["should_stop"]()
        clock[0] = 300.0
        events.append("stream_closed")
        if failure == "stream":
            raise RuntimeError("synthetic stream failure")
        return SimpleNamespace(
            records=[], as_dict=lambda: dict(stop_reason="duration budget reached")
        )

    monkeypatch.setattr(soak, "stream_frames", stream)
    if failure in ("stream", "quiet_boundary"):
        with pytest.raises(RuntimeError, match="synthetic"):
            soak.main()
    elif failure == "stale_telemetry":
        with pytest.raises(RuntimeError, match="telemetry stale"):
            soak.main()
    elif failure == "restoration_gain":
        with pytest.raises(OSError, match="synthetic final gain"):
            soak.main()
    else:
        soak.main()
    result = json.loads(output.read_text())
    assert writes == (
        [(timing, timing)] if failure == "quiet_boundary" else [(timing, timing), (20000, 20000)]
    )
    assert result["restoration_verified"] is (failure not in ("restoration_gain", "quiet_boundary"))
    if failure == "quiet_boundary":
        assert events.count("quiet_wait") == 1
        assert len(gain_reads) == 2
    if failure == "restoration_gain":
        assert result["status_before_restoration_error"] == "completed"
        assert "restored_gain_register" not in result
    elif failure != "quiet_boundary":
        assert result["restored_gain_register"] == 24
    assert result["requested_timing_us"] == timing
    assert result["status"] == ("completed" if failure == "none" else "failed")
    assert closed == ([919] if failure == "quiet_boundary" else [919, 919])
