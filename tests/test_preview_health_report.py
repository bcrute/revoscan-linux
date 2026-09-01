"""The physical capture, recomputed from its own bytes and checked against the report.

The capture
------------
``/home/ben/scans/scan/20260828-110641``, 166 raw ``Y16`` frames of 640x400.
This is the capture the owner reported on 2026-08-28 as "the preview stopped
showing the object", and the whole point of this file is that it did not: the
frames themselves stopped returning depth, and by the last frame what was left
had collapsed toward the LEFT of the sensor rather than fading evenly.

How this avoids asserting an assumption
----------------------------------------
Every number below is counted here, from the raw bytes, by code that does not
call :mod:`scanner3d.return_health` at all - see :func:`counted_by_hand`. The
harness's report is then required to agree with that independent count exactly,
because both are integer counts of the same bytes and there is no honest reason
for them to differ by one. The published figures Codex measured first (75.98%,
87.64%, 96.40% no-return; frame 165's quadrants at 89.66%, 99.94%, 96.02% and
100.00%) are checked separately and only to the precision they were quoted at:
a percentage printed to two decimals pins the underlying integer to within half
a unit in the last place, and :data:`QUOTED_TOLERANCE` is exactly that.

Skipping
---------
The capture lives outside the repository, on the owner's machine. Where it is
absent this file skips rather than fails: a machine without the capture cannot
answer a question about the capture, and a suite that failed there would be
asserting something about the machine instead.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from scanner3d.return_health import (
    COLLAPSING,
    HISTORY_FRAMES,
    NO_RETURNS,
    SCHEMA_VERSION,
    SEVERELY_SPARSE,
    USABLE,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH
from tests import preview_health_replay

#: Half a unit in the last decimal place of a figure quoted as ``NN.NN%``.
QUOTED_TOLERANCE = 0.00005

#: What Codex measured first, as no-return shares of the whole frame.
QUOTED_NO_RETURN = {0: 0.7598, 80: 0.8764, 165: 0.9640}

#: ...and of frame 165's four quadrants, in the module's own name order.
QUOTED_QUADRANTS_165 = {
    "top_left": 0.8966,
    "top_right": 0.9994,
    "bottom_left": 0.9602,
    "bottom_right": 1.0000,
}


@pytest.fixture(scope="module")
def capture() -> Path:
    directory = preview_health_replay.DEFAULT_CAPTURE
    if not directory.is_dir():
        pytest.skip(f"the physical capture {directory} is not on this machine")
    return directory


@pytest.fixture(scope="module")
def report(capture: Path) -> dict[str, Any]:
    return preview_health_replay.run(capture)


def counted_by_hand(path: Path) -> dict[str, Any]:
    """Count one frame's returns from scratch, without the module under test.

    Written out longhand on purpose: index arithmetic done a second, different
    way is the only thing that can catch a row-stride or transposition mistake
    made consistently in both places.
    """
    raw = path.read_bytes()
    assert len(raw) == EXPECTED_FRAME_BYTES
    samples = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", raw)
    quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    returned = 0
    left, right, top, bottom = FRAME_WIDTH, -1, FRAME_HEIGHT, -1
    for row in range(FRAME_HEIGHT):
        for column in range(FRAME_WIDTH):
            if samples[row * FRAME_WIDTH + column] == 0:
                continue
            returned += 1
            vertical = "top" if row < FRAME_HEIGHT // 2 else "bottom"
            horizontal = "left" if column < FRAME_WIDTH // 2 else "right"
            quadrants[f"{vertical}_{horizontal}"] += 1
            left, right = min(left, column), max(right, column)
            top, bottom = min(top, row), max(bottom, row)
    return {
        "returned": returned,
        "quadrants": quadrants,
        "footprint": None
        if right < 0
        else {"left": left, "right": right, "top": top, "bottom": bottom},
    }


def selected(report: dict[str, Any], index: int) -> dict[str, Any]:
    frames = report["selected_frames"]
    for frame in frames:
        if frame["index"] == index:
            return dict(frame)
    raise AssertionError(f"frame {index} is not in the report")


# --- the report agrees with an independent count of the same bytes -----------


@pytest.mark.parametrize("index", [0, 80, 165])
def test_the_reported_counts_are_the_counts_this_test_makes_itself(
    report: dict[str, Any], capture: Path, index: int
):
    expected = counted_by_hand(capture / f"frame_{index:04d}.raw")
    spatial = selected(report, index)["spatial"]

    assert spatial["returned"] == expected["returned"]
    assert spatial["samples"] == FRAME_WIDTH * FRAME_HEIGHT
    assert spatial["no_return"] == FRAME_WIDTH * FRAME_HEIGHT - expected["returned"]
    by_name = {quadrant["name"]: quadrant["returned"] for quadrant in spatial["quadrants"]}
    assert by_name == expected["quadrants"]
    footprint = spatial["footprint"]
    assert footprint is not None
    assert {key: footprint[key] for key in ("left", "right", "top", "bottom")} == expected[
        "footprint"
    ]


@pytest.mark.parametrize("index", [0, 80, 165])
def test_the_yields_match_the_figures_the_capture_was_first_reported_with(
    report: dict[str, Any], index: int
):
    """Frame 0 at 75.98%, frame 80 at 87.64%, frame 165 at 96.40% no-return."""
    spatial = selected(report, index)["spatial"]
    assert abs(spatial["no_return_fraction"] - QUOTED_NO_RETURN[index]) < QUOTED_TOLERANCE
    assert abs(spatial["yield_fraction"] - (1.0 - QUOTED_NO_RETURN[index])) < QUOTED_TOLERANCE


def test_the_capture_loses_returns_monotonically_across_the_three_named_frames():
    """The shape of the report: less came back at 80 than at 0, and less again at 165."""
    assert QUOTED_NO_RETURN[0] < QUOTED_NO_RETURN[80] < QUOTED_NO_RETURN[165]


# --- the collapse is one-sided, and the report says which side ---------------


def test_frame_165_collapses_to_the_left_rather_than_fading_evenly(report: dict[str, Any]):
    spatial = selected(report, 165)["spatial"]
    shares = {quadrant["name"]: quadrant["no_return_fraction"] for quadrant in spatial["quadrants"]}
    for name, quoted in QUOTED_QUADRANTS_165.items():
        assert abs(shares[name] - quoted) < QUOTED_TOLERANCE
    # The distinction a single percentage cannot make: the right-hand half is
    # essentially empty while the left still returns.
    assert shares["top_right"] > shares["top_left"]
    assert shares["bottom_right"] > shares["bottom_left"]
    # The bottom-right quadrant is the emptiest of the four, and empty enough
    # that it was first quoted as 100.00%: 2 returns in 64000 samples.
    returns = {q["name"]: q["returned"] for q in spatial["quadrants"]}
    assert returns["bottom_right"] == 2
    assert min(shares, key=lambda name: 1.0 - shares[name]) == "bottom_right"


def test_frame_165s_footprint_stops_well_short_of_the_right_edge(report: dict[str, Any]):
    footprint = selected(report, 165)["spatial"]["footprint"]
    assert footprint is not None
    assert footprint["right"] < FRAME_WIDTH - 1
    # ...and the earlier frames' returns reach much further across.
    assert selected(report, 0)["spatial"]["footprint"]["right"] > footprint["right"]


def test_the_summary_line_names_the_emptiest_quadrant_and_the_span(report: dict[str, Any]):
    summary = selected(report, 165)["summary"]
    assert "bottom right" in summary
    assert "columns 4-" in summary


# --- the verdicts a live operator would have been shown ----------------------


def test_the_replay_reports_verdict_transitions_over_the_bounded_window(
    report: dict[str, Any],
):
    transitions = report["verdict_transitions"]
    assert report["history_frames"] == HISTORY_FRAMES
    verdicts = {transition["verdict"] for transition in transitions}
    # All three reached from the raw evidence rather than constructed. NO
    # RETURNS is deliberately not asserted: this capture's emptiest frame still
    # returns 31 samples, and a test that demanded a completely empty frame
    # would be asserting something about the capture that is not true.
    assert {USABLE, SEVERELY_SPARSE, COLLAPSING} <= verdicts
    assert NO_RETURNS not in verdicts
    assert transitions[0]["index"] == 0
    assert transitions[0]["verdict"] == USABLE
    # A transition record only exists where the verdict CHANGED.
    assert all(
        transitions[i]["verdict"] != transitions[i + 1]["verdict"]
        for i in range(len(transitions) - 1)
    )


def test_every_frame_has_a_yield_and_the_last_one_is_the_reported_one(
    report: dict[str, Any],
):
    yields = report["yield_fractions"]
    assert len(yields) == report["frames_on_disk"] == 166
    assert abs(yields[165] - (1.0 - QUOTED_NO_RETURN[165])) < QUOTED_TOLERANCE
    # The emptiest frame in the capture, recorded rather than rounded away: 31
    # returned samples is not none, and this harness does not call it none.
    assert round(min(yields) * FRAME_WIDTH * FRAME_HEIGHT) == 31


# --- nothing was written, and the report says so -----------------------------


def test_the_replay_confirms_no_capture_file_changed(report: dict[str, Any]):
    read_only = report["read_only"]
    assert read_only["unchanged"] is True
    assert read_only["changed"] == []
    assert read_only["missing"] == []
    assert read_only["appeared"] == []
    assert read_only["files"] == 166


def test_the_capture_directory_holds_only_the_frames_it_started_with(capture: Path):
    """No report file, no sidecar, no temporary directory left behind."""
    assert sorted(path.name for path in capture.iterdir()) == [
        f"frame_{index:04d}.raw" for index in range(166)
    ]


# --- the report is honest about what it is ------------------------------------


def test_the_report_is_schema_versioned_strict_json(report: dict[str, Any]):
    document = json.dumps(report, allow_nan=False)
    assert json.loads(document)["schema_version"] == SCHEMA_VERSION


def test_the_report_calls_the_black_region_evidence_and_not_a_diagnosis(
    report: dict[str, Any],
):
    limitations = " ".join(report["limitations"])
    assert "not restored geometry" in limitations
    assert "does not prove a projector" in limitations
    assert "never fills, interpolates or reconstructs" in limitations
    assert "advisory" in limitations


def test_the_harness_refuses_a_directory_that_is_not_a_capture(tmp_path: Path):
    with pytest.raises(preview_health_replay.ReplayRefused):
        preview_health_replay.run(tmp_path / "nowhere")
    with pytest.raises(preview_health_replay.ReplayRefused):
        preview_health_replay.run(tmp_path)


def test_the_harness_refuses_a_frame_index_the_capture_does_not_have(capture: Path):
    with pytest.raises(preview_health_replay.ReplayRefused):
        preview_health_replay.run(capture, frames=(9999,))
