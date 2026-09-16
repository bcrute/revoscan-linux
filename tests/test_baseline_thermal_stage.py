"""Baseline endurance may not advance on incomplete lifecycle or telemetry."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def soak(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_thermal_soak")


def predecessor(soak):
    return dict(
        status="completed",
        restoration_verified=True,
        serial=soak.SERIAL,
        requested_timing_us=20000,
        duration_limit_s=300,
        quiet_after_stop=True,
        first_transport_fault=None,
        usb_stop_boundary=dict(settled_for_readback=True),
        post_restore_usb=dict(topology_stable=True),
        original=soak.BASELINE,
        restored=soak.BASELINE,
        original_gain_register=24,
        restored_gain_register=24,
        stream=dict(stop_reason="duration budget reached", short_frames=0, dropped_by_sequence=0),
        distinct_payloads=1350,
        hashed_complete_frames=1350,
        thermal_samples=[dict(elapsed_s=i, cpu_c=60, npu_c=59) for i in range(1, 300, 5)],
    )


def test_baseline_requires_its_own_clean_quiet_predecessor(soak):
    soak.validate_stage(300, None, 20000)
    previous = predecessor(soak)
    soak.validate_stage(900, previous, 20000)
    changes = [
        dict(quiet_after_stop=False),
        dict(first_transport_fault={"errno": 71}),
        dict(post_restore_usb={}),
        dict(usb_stop_boundary={}),
        dict(restored_gain_register=23),
        dict(distinct_payloads=1349),
        dict(stream={**previous["stream"], "short_frames": 1}),
        dict(stream={**previous["stream"], "dropped_by_sequence": 1}),
        dict(thermal_samples=previous["thermal_samples"][::4]),
        dict(thermal_samples=[dict(elapsed_s=296, cpu_c=60, npu_c=59)]),
        dict(
            thermal_samples=[
                *previous["thermal_samples"],
                dict(elapsed_s=299, cpu_c=float("nan"), npu_c=60),
            ]
        ),
    ]
    for change in changes:
        with pytest.raises(ValueError):
            soak.validate_stage(900, {**previous, **change}, 20000)


def test_thirty_minutes_requires_fifteen_not_five(soak):
    previous = predecessor(soak)
    with pytest.raises(ValueError):
        soak.validate_stage(1800, previous, 20000)
    previous.update(
        duration_limit_s=900,
        thermal_samples=[dict(elapsed_s=i, cpu_c=60, npu_c=59) for i in range(1, 900, 5)],
    )
    soak.validate_stage(1800, previous, 20000)
