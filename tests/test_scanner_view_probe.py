"""View-check wrapper uses only mocked hardware; never runs acquisition here."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("scanner_view_probe")
    monkeypatch.setattr(module, "resolve_serial", lambda: module.SERIAL)
    monkeypatch.setattr(module, "wait_for_capture_access", lambda **kw: SimpleNamespace(ready=True))
    monkeypatch.setattr(
        module, "resolve_capture_node", lambda serial: (SimpleNamespace(node_path="fake"), None)
    )
    monkeypatch.setattr(module.os, "open", lambda *args: 918)
    monkeypatch.setattr(module, "state", lambda fd: (dict(module.BASELINE), ""))
    monkeypatch.setattr(module, "gain", lambda fd: (24, ""))
    monkeypatch.setattr(module, "temperatures", lambda fd: dict(cpu_c=60, npu_c=59))
    return module


@pytest.mark.parametrize("failed", [False, True])
def test_view_probe_checks_baseline_and_closes_on_capture_failure(
    probe, monkeypatch, tmp_path, failed
):
    closed = []
    monkeypatch.setattr(probe.os, "close", closed.append)

    def capture(serial, frames, buffers, poll, **kwargs):
        assert (serial, frames, buffers, poll) == (probe.SERIAL, 32, 8, 0.001)
        kwargs["monitor"](918)
        if failed:
            raise RuntimeError("synthetic acquisition failure")
        return dict(summary=dict(hashed_complete_frames=32))

    monkeypatch.setattr(probe, "capture", capture)
    output = tmp_path / "view"
    if failed:
        with pytest.raises(RuntimeError, match="synthetic"):
            probe.run(output)
    else:
        probe.run(output)
    report = json.loads((output / "report.json").read_text())
    assert report["baseline_verified"] and not report["settings_written"]
    assert report["status"] == ("failed" if failed else "completed")
    assert closed == [918, 918]


def test_view_probe_refuses_another_serial_before_files_or_hardware(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "resolve_serial", lambda: "other")
    monkeypatch.setattr(probe.os, "open", lambda *args: pytest.fail("hardware escaped"))
    with pytest.raises(RuntimeError, match="bound"):
        probe.run(tmp_path / "view")
    assert not (tmp_path / "view").exists()
