"""Where the returns are, how fast they are going, and what is said about it.

What these ask, and why they are worth asking
----------------------------------------------
On 2026-08-28 the owner reported that the false-coloured depth preview had
"stopped showing the object" partway through a capture. It had not: the frames
really had stopped returning depth, from 24.02% of the frame at frame 0 to
3.60% at frame 165, and by the end the returns had collapsed to the LEFT of the
sensor rather than fading evenly - 99.94% and 100.00% no-return in the two
right-hand quadrants.

Two failures follow from that, and this file is about both.

* A yield percentage on its own cannot tell a uniform fade from an object that
  has slid off one side. Those call for opposite corrections from whoever is
  holding the scanner, so :func:`~scanner3d.return_health.spatial_returns` has
  to put the returns SOMEWHERE, exactly, and a transposed index or an
  off-by-one row stride would still produce plausible-looking counts. Every
  spatial case below is hand-counted on a grid small enough to read, and the
  one-sided cases are flipped and rotated: a renderer that had confused rows
  with columns would move the wrong region and fail.
* A verdict about a collapse in progress is only useful while the scan can
  still be saved, and only trustworthy if it cannot be poisoned by the scan
  before it. So the temporal cases drive a whole deterministic sequence -
  working, collapsing, severe, empty, recovered - and check the window's bound,
  its hysteresis at the boundary, and that a reset really forgets.

Nothing here asks whether a verdict is CORRECT about the physical world; there
is no ground truth for "the operator should have slowed down". What is asked is
that the verdict is a deterministic, bounded function of the raw evidence, and
that it never claims a distance, a material or a cause the bytes do not carry.
"""

from __future__ import annotations

import json

import pytest

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.return_health import (
    COLLAPSE_MIN_DROP,
    COLLAPSE_RATIO,
    COLLAPSING,
    HISTORY_FRAMES,
    NO_RETURNS,
    QUADRANT_NAMES,
    REASON_BELOW_SEVERE,
    REASON_FELL_BELOW_BASELINE,
    REASON_HOLDING_BELOW_BASELINE,
    REASON_NO_BASELINE,
    REASON_NO_SAMPLES,
    REASON_WITHIN_BASELINE,
    RECOVERY_RATIO,
    SCHEMA_VERSION,
    SEVERE_YIELD_FRACTION,
    SEVERELY_SPARSE,
    USABLE,
    FrameShapeMismatch,
    InvalidYield,
    ReturnHealthError,
    ReturnHealthTracker,
    health_summary,
    return_health,
    spatial_returns,
    spatial_summary,
)

# --- little frames, written out so they can be counted by eye ----------------

#: A code that is a return. Its VALUE is irrelevant to everything in this
#: module - only "is it zero" is ever asked - and a deliberately odd number is
#: used so a test that started depending on the value would look wrong.
R = 4321


def grid(rows: list[str]) -> tuple[list[int], int, int]:
    """Turn ``["..#", "#.."]`` into (samples, width, height).

    ``#`` is a return, ``.`` is a no-return. Row-major, first string is row 0,
    which is the same layout a raw frame arrives in.
    """
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    samples = [R if char == "#" else 0 for row in rows for char in row]
    return samples, width, len(rows)


def counts(rows: list[str]) -> dict[str, int]:
    """Returned count per quadrant, by name."""
    samples, width, height = grid(rows)
    spatial = spatial_returns(samples, width, height)
    return {quadrant.name: quadrant.returned for quadrant in spatial.quadrants}


# --- spatial: exact counts, hand-checked -------------------------------------


def test_quadrant_counts_are_hand_computed_on_a_readable_grid():
    rows = [
        "##..",
        "#...",
        "...#",
        "..##",
    ]
    assert counts(rows) == {
        "top_left": 3,
        "top_right": 0,
        "bottom_left": 0,
        "bottom_right": 3,
    }


def test_an_empty_frame_has_no_returns_no_quadrants_and_no_footprint():
    samples, width, height = grid(["....", "....", "....", "...."])
    spatial = spatial_returns(samples, width, height)
    assert spatial.returned == 0
    assert spatial.no_return == 16
    assert spatial.yield_fraction == 0.0
    assert spatial.no_return_fraction == 1.0
    assert spatial.footprint is None
    assert [quadrant.returned for quadrant in spatial.quadrants] == [0, 0, 0, 0]
    # Every quadrant is equally empty, so the tie is broken by the fixed name
    # order rather than by whatever happened to be compared first.
    assert spatial.emptiest_quadrant.name == "top_left"


def test_one_returning_pixel_has_a_one_by_one_footprint_at_its_own_position():
    samples, width, height = grid(["....", "..#.", "....", "...."])
    spatial = spatial_returns(samples, width, height)
    footprint = spatial.footprint
    assert footprint is not None
    assert (footprint.left, footprint.right) == (2, 2)
    assert (footprint.top, footprint.bottom) == (1, 1)
    assert (footprint.width, footprint.height) == (1, 1)
    assert spatial.returned == 1
    assert counts(["....", "..#.", "....", "...."])["top_right"] == 1


def test_a_centre_island_is_bounded_by_itself_and_touches_every_quadrant():
    rows = [
        "......",
        "......",
        "..##..",
        "..##..",
        "......",
        "......",
    ]
    samples, width, height = grid(rows)
    spatial = spatial_returns(samples, width, height)
    footprint = spatial.footprint
    assert footprint is not None
    assert (footprint.left, footprint.right, footprint.top, footprint.bottom) == (2, 3, 2, 3)
    assert counts(rows) == {
        "top_left": 1,
        "top_right": 1,
        "bottom_left": 1,
        "bottom_right": 1,
    }


def test_returns_touching_all_four_edges_are_counted_at_the_boundary():
    rows = [
        "#..#",
        "....",
        "....",
        "#..#",
    ]
    samples, width, height = grid(rows)
    spatial = spatial_returns(samples, width, height)
    footprint = spatial.footprint
    assert footprint is not None
    assert (footprint.left, footprint.right, footprint.top, footprint.bottom) == (0, 3, 0, 3)
    assert counts(rows) == {
        "top_left": 1,
        "top_right": 1,
        "bottom_left": 1,
        "bottom_right": 1,
    }


# --- spatial: a transposed or mis-strided frame must fail ---------------------

#: The physical shape of the reported failure, in miniature: everything that
#: still returns sits on the LEFT, and the right-hand quadrants are empty.
ONE_SIDED = [
    "##..",
    "##..",
    "##..",
    "##..",
]


def test_a_one_sided_frame_names_the_side_the_returns_are_actually_on():
    assert counts(ONE_SIDED) == {
        "top_left": 4,
        "top_right": 0,
        "bottom_left": 4,
        "bottom_right": 0,
    }
    samples, width, height = grid(ONE_SIDED)
    footprint = spatial_returns(samples, width, height).footprint
    assert footprint is not None
    assert (footprint.left, footprint.right) == (0, 1)
    assert (footprint.top, footprint.bottom) == (0, 3)


def test_mirroring_the_one_sided_frame_moves_the_named_region_to_the_other_side():
    mirrored = [row[::-1] for row in ONE_SIDED]
    assert counts(mirrored) == {
        "top_left": 0,
        "top_right": 4,
        "bottom_left": 0,
        "bottom_right": 4,
    }
    samples, width, height = grid(mirrored)
    footprint = spatial_returns(samples, width, height).footprint
    assert footprint is not None
    assert (footprint.left, footprint.right) == (2, 3)


def test_transposing_a_one_sided_frame_moves_the_returns_to_the_top():
    """The case a row/column mix-up passes and this must not.

    Transposed, the same returns lie along the TOP rather than the left. An
    implementation that indexed ``column * height + row`` would report the
    original left-hand answer for this frame, and the previous two tests would
    not notice.
    """
    transposed = [
        "####",
        "####",
        "....",
        "....",
    ]
    assert counts(transposed) == {
        "top_left": 4,
        "top_right": 4,
        "bottom_left": 0,
        "bottom_right": 0,
    }
    samples, width, height = grid(transposed)
    footprint = spatial_returns(samples, width, height).footprint
    assert footprint is not None
    assert (footprint.top, footprint.bottom) == (0, 1)
    assert (footprint.left, footprint.right) == (0, 3)


def test_rotating_a_corner_frame_by_half_a_turn_moves_it_to_the_opposite_corner():
    corner = ["#...", "....", "....", "...."]
    rotated = [row[::-1] for row in reversed(corner)]
    assert counts(corner)["top_left"] == 1
    assert counts(rotated) == {
        "top_left": 0,
        "top_right": 0,
        "bottom_left": 0,
        "bottom_right": 1,
    }


def test_a_non_square_frame_is_not_secretly_read_as_its_own_transpose():
    """Six wide and two tall: reading it 2x6 would put the returns elsewhere."""
    rows = ["###...", "......"]
    samples, width, height = grid(rows)
    spatial = spatial_returns(samples, width, height)
    assert (spatial.width, spatial.height) == (6, 2)
    assert counts(rows) == {
        "top_left": 3,
        "top_right": 0,
        "bottom_left": 0,
        "bottom_right": 0,
    }


# --- spatial: the exact boundary rules ---------------------------------------


def test_an_odd_middle_column_and_row_belong_to_the_right_and_bottom_halves():
    """The stated rule, asked directly: 5x3 splits 2|3 across and 1|2 down."""
    samples, width, height = grid(
        [
            "..#..",
            ".....",
            ".....",
        ]
    )
    spatial = spatial_returns(samples, width, height)
    assert spatial.top_left.samples == 2 * 1
    assert spatial.top_right.samples == 3 * 1
    assert spatial.bottom_left.samples == 2 * 2
    assert spatial.bottom_right.samples == 3 * 2
    assert sum(q.samples for q in spatial.quadrants) == 15
    # The middle column (index 2) is in the RIGHT half, not the left.
    assert spatial.top_right.returned == 1
    assert spatial.top_left.returned == 0


def test_the_column_just_left_of_an_odd_middle_is_still_the_left_half():
    samples, width, height = grid([".#...", ".....", "....."])
    spatial = spatial_returns(samples, width, height)
    assert spatial.top_left.returned == 1
    assert spatial.top_right.returned == 0


def test_the_emptiest_quadrant_is_the_one_with_the_largest_no_return_share():
    rows = [
        "##..",
        "##..",
        "#...",
        "....",
    ]
    samples, width, height = grid(rows)
    spatial = spatial_returns(samples, width, height)
    assert spatial.emptiest_quadrant.name in {"top_right", "bottom_right"}
    assert spatial.emptiest_quadrant.no_return_fraction == 1.0
    assert spatial.top_left.no_return_fraction == 0.0


def test_quadrant_shares_are_counts_over_that_quadrants_own_samples():
    samples, width, height = grid(["#...", "....", "....", "...."])
    spatial = spatial_returns(samples, width, height)
    assert spatial.top_left.samples == 4
    assert spatial.top_left.returned == 1
    assert spatial.top_left.yield_fraction == 0.25
    assert spatial.top_left.no_return == 3
    assert spatial.top_left.no_return_fraction == 0.75


# --- spatial: refusals -------------------------------------------------------


def test_a_length_that_is_not_the_stated_shape_is_refused_not_reshaped():
    with pytest.raises(FrameShapeMismatch):
        spatial_returns([0] * 11, 4, 3)


def test_a_non_positive_shape_is_refused():
    with pytest.raises(FrameShapeMismatch):
        spatial_returns([], 0, 3)
    with pytest.raises(FrameShapeMismatch):
        spatial_returns([], 4, -1)


def test_every_refusal_is_a_scanner_error_with_the_unsupported_mode_status():
    for error in (FrameShapeMismatch, InvalidYield, ReturnHealthError):
        assert issubclass(error, ScannerError)
        assert error.status is DoctorStatus.UNSUPPORTED_MODE


# --- spatial: the record is strict JSON --------------------------------------


def test_the_spatial_record_is_schema_versioned_strict_json():
    samples, width, height = grid(["#...", "....", "..#.", "...."])
    record = spatial_returns(samples, width, height).as_dict()
    document = json.dumps(record, allow_nan=False)
    parsed = json.loads(document)
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert [quadrant["name"] for quadrant in parsed["quadrants"]] == list(QUADRANT_NAMES)
    assert parsed["footprint"] == {
        "left": 0,
        "right": 2,
        "top": 0,
        "bottom": 2,
        "width": 3,
        "height": 3,
    }
    assert parsed["returned"] == 2
    assert parsed["no_return"] == 14
    assert "not a distance" in parsed["note"]


def test_an_empty_frames_footprint_serialises_as_null_not_a_box_at_the_origin():
    samples, width, height = grid(["....", "...."])
    parsed = json.loads(json.dumps(spatial_returns(samples, width, height).as_dict()))
    assert parsed["footprint"] is None


# --- temporal: the verdict vocabulary ----------------------------------------


def test_a_frame_with_nothing_in_it_is_called_no_returns():
    health = return_health(0.0)
    assert health.verdict == NO_RETURNS
    assert health.reason == REASON_NO_SAMPLES
    assert health.is_warning


def test_a_frame_under_the_severe_share_is_called_severely_sparse():
    health = return_health(SEVERE_YIELD_FRACTION / 2.0, baseline_fraction=0.4)
    assert health.verdict == SEVERELY_SPARSE
    assert health.reason == REASON_BELOW_SEVERE
    assert health.is_warning


def test_the_severe_threshold_is_a_floor_not_a_ceiling():
    assert return_health(SEVERE_YIELD_FRACTION).verdict != SEVERELY_SPARSE
    assert return_health(SEVERE_YIELD_FRACTION - 1e-9).verdict == SEVERELY_SPARSE


def test_an_ordinary_frame_with_no_history_is_usable_and_says_it_has_no_baseline():
    health = return_health(0.24)
    assert health.verdict == USABLE
    assert health.reason == REASON_NO_BASELINE
    assert health.baseline_fraction is None
    assert not health.is_warning


def test_a_material_fall_from_the_baseline_is_called_collapsing():
    health = return_health(0.10, baseline_fraction=0.40)
    assert health.verdict == COLLAPSING
    assert health.reason == REASON_FELL_BELOW_BASELINE
    assert health.baseline_fraction == 0.40


def test_a_proportional_fall_that_is_a_tiny_share_of_the_frame_is_not_a_collapse():
    """Half of 8% is 4%, which is a smaller drop than the noise near the floor.

    Without the absolute floor this would be called a collapse every time a
    nearly-empty frame got slightly emptier, which is both useless and loud.
    """
    health = return_health(0.04, baseline_fraction=0.08)
    assert health.verdict == SEVERELY_SPARSE  # the absolute rule catches it instead
    health = return_health(0.055, baseline_fraction=0.09)
    assert health.verdict == USABLE
    assert COLLAPSE_MIN_DROP > (0.09 - 0.055)


def test_a_no_returns_frame_is_named_that_even_with_an_empty_baseline():
    assert return_health(0.0, baseline_fraction=0.0).verdict == NO_RETURNS


def test_no_verdict_names_a_distance_a_material_or_a_cause():
    """The advice may only say what the bytes support and what the operator can do."""
    forbidden = ("mm", "millimet", "metre", "meter", "distance from", "exposure", "projector")
    for health in (
        return_health(0.0),
        return_health(0.01),
        return_health(0.10, baseline_fraction=0.40),
        return_health(0.30, baseline_fraction=0.30),
    ):
        lowered = health.advice.lower()
        for word in forbidden:
            assert word not in lowered


def test_every_warning_says_the_blank_areas_are_no_returns_and_what_to_do():
    for health in (
        return_health(0.0),
        return_health(0.01),
        return_health(0.10, baseline_fraction=0.40),
    ):
        assert "returned no depth" in health.advice
        assert "Slow down" in health.advice
        assert "faces the scanner squarely" in health.advice
        assert "still showing" in health.advice


def test_the_verdict_record_is_strict_json_and_says_it_decides_nothing():
    parsed = json.loads(json.dumps(return_health(0.1, baseline_fraction=0.4).as_dict()))
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["verdict"] == COLLAPSING
    assert parsed["warning"] is True
    assert parsed["yield_fraction"] == 0.1
    assert parsed["baseline_fraction"] == 0.4
    assert "advisory only" in parsed["note"]


def test_a_yield_outside_zero_to_one_is_refused():
    for bad in (-0.001, 1.5):
        with pytest.raises(InvalidYield):
            return_health(bad)
    with pytest.raises(InvalidYield):
        return_health(0.5, baseline_fraction=2.0)


def test_a_boolean_is_not_a_yield_fraction():
    with pytest.raises(InvalidYield):
        return_health(True)


# --- temporal: the bounded window --------------------------------------------


def test_the_tracker_holds_no_more_than_its_fixed_window_however_long_a_run_is():
    tracker = ReturnHealthTracker()
    for _ in range(HISTORY_FRAMES * 20):
        tracker.update(0.3)
    assert tracker.frames == HISTORY_FRAMES
    assert tracker.capacity == HISTORY_FRAMES


def test_the_first_frame_of_a_run_is_judged_against_no_baseline_at_all():
    tracker = ReturnHealthTracker()
    assert tracker.baseline is None
    first = tracker.update(0.3)
    assert first.baseline_fraction is None
    assert first.verdict == USABLE


def test_a_frame_is_compared_with_the_frames_before_it_not_with_itself():
    tracker = ReturnHealthTracker()
    tracker.update(0.40)
    health = tracker.update(0.10)
    assert health.baseline_fraction == 0.40
    assert health.verdict == COLLAPSING


def test_the_baseline_is_the_median_of_the_window_not_its_peak():
    tracker = ReturnHealthTracker(history=5)
    for value in (0.9, 0.2, 0.2, 0.2, 0.2):
        tracker.update(value)
    assert tracker.baseline == 0.2


def test_a_deterministic_run_moves_through_every_verdict_and_back():
    """Working, collapsing, severe, empty, recovered - in that order, from yields alone."""
    tracker = ReturnHealthTracker(history=5)
    verdicts = [
        tracker.update(value).verdict for value in (0.40, 0.40, 0.40, 0.40, 0.10, 0.02, 0.00, 0.40)
    ]
    assert verdicts == [
        USABLE,
        USABLE,
        USABLE,
        USABLE,
        COLLAPSING,
        SEVERELY_SPARSE,
        NO_RETURNS,
        USABLE,
    ]


def test_a_recovered_frame_clears_the_warning_without_waiting_for_the_window_to_refill():
    tracker = ReturnHealthTracker(history=5)
    for _ in range(4):
        tracker.update(0.40)
    assert tracker.update(0.05).verdict == COLLAPSING
    recovered = tracker.update(0.40)
    assert recovered.verdict == USABLE
    assert not recovered.is_warning


def test_a_yield_hovering_at_the_collapse_boundary_does_not_flicker():
    """The failure hysteresis exists for: on, off, on, off, once per frame.

    Each of these frames is alternately just under and just over the entry
    ratio. Without the wider recovery ratio the verdict would change on every
    one of them, which on screen is a warning that cannot be read.
    """
    tracker = ReturnHealthTracker(history=3)
    for _ in range(3):
        tracker.update(0.40)
    baseline = tracker.baseline
    assert baseline is not None
    just_under = baseline * COLLAPSE_RATIO * 0.99
    just_over = baseline * COLLAPSE_RATIO * 1.01
    assert just_over < baseline * RECOVERY_RATIO
    # Feeding the window would move the baseline out from under the test, so
    # the boundary itself is asked directly with the state the tracker carries.
    assert return_health(just_under, baseline_fraction=baseline).verdict == COLLAPSING
    held = return_health(just_over, baseline_fraction=baseline, was_collapsing=True)
    assert held.verdict == COLLAPSING
    assert held.reason == REASON_HOLDING_BELOW_BASELINE
    assert return_health(just_over, baseline_fraction=baseline).verdict == USABLE


def test_a_collapse_is_released_once_the_yield_is_properly_back():
    tracker = ReturnHealthTracker(history=3)
    for _ in range(3):
        tracker.update(0.40)
    assert tracker.update(0.10).verdict == COLLAPSING
    back = tracker.update(0.40 * RECOVERY_RATIO + 0.01)
    assert back.verdict == USABLE
    assert back.reason == REASON_WITHIN_BASELINE


def test_reset_forgets_the_previous_run_completely():
    """The whole point of the reset: one scan's collapse is not the next one's evidence."""
    tracker = ReturnHealthTracker(history=5)
    for _ in range(4):
        tracker.update(0.40)
    assert tracker.update(0.05).verdict == COLLAPSING

    tracker.reset()
    assert tracker.frames == 0
    assert tracker.baseline is None
    fresh = tracker.update(0.05)
    # 5% is not under the severe floor and there is nothing to have fallen
    # from, so a new run's first frame is not called a collapse.
    assert fresh.verdict == USABLE
    assert fresh.reason == REASON_NO_BASELINE


def test_two_runs_in_one_process_do_not_see_each_others_frames():
    tracker = ReturnHealthTracker(history=5)
    for _ in range(5):
        tracker.update(0.60)
    tracker.reset()
    for _ in range(5):
        tracker.update(0.08)
    assert tracker.baseline == 0.08
    assert tracker.update(0.07).verdict == USABLE


def test_a_history_window_must_be_a_positive_number_of_frames():
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ReturnHealthError):
            ReturnHealthTracker(history=bad)  # type: ignore[arg-type]


# --- what the front ends are handed ------------------------------------------


def test_the_spatial_summary_names_the_emptiest_quadrant_and_the_footprint():
    samples, width, height = grid(ONE_SIDED)
    line = spatial_summary(spatial_returns(samples, width, height))
    assert "top right" in line or "bottom right" in line
    assert "columns 0-1 of 4" in line
    assert "rows 0-3 of 4" in line


def test_the_spatial_summary_of_an_empty_frame_says_there_is_nothing_anywhere():
    samples, width, height = grid(["....", "...."])
    assert "no returns anywhere" in spatial_summary(spatial_returns(samples, width, height))


def test_the_shared_summary_puts_the_verdict_first_and_the_geometry_second():
    samples, width, height = grid(ONE_SIDED)
    spatial = spatial_returns(samples, width, height)
    text = health_summary(return_health(0.0), spatial)
    verdict_line, spatial_line = text.split("\n")
    assert verdict_line.startswith(NO_RETURNS.upper() + ":")
    assert spatial_line == spatial_summary(spatial)
