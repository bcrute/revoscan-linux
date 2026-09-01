"""What the frame spacing does with a measurement, and what it refuses to do.

Every test here is arithmetic over stated measurements. That is deliberate and
it is the whole reason :mod:`scanner3d.cadence` is a module rather than a few
lines inside the capture: a controller entangled with a device, a clock or a
window can only be tested by driving all three, and the one property that
actually matters - that it converges into the band and stays there - is then
untestable without a scanner and an operator's arm.

The one test that is not arithmetic is the closed loop at the end. It is the
only one that can fail for a reason the others cannot see: a correction that is
individually sensible on every frame and still oscillates forever.
"""

from __future__ import annotations

import json
import math

import pytest

from scanner3d.cadence import (
    CEILING_REASON,
    FASTER_REASON,
    FLOOR_REASON,
    HELD_REASON,
    MAX_DECIMATE,
    MIN_DECIMATE,
    REFUSAL_REASON,
    SLOW_CONFIRMATIONS,
    SLOWER_REASON,
    UNMEASURED_REASON,
    CadenceController,
    CadenceNotAllowed,
    CadencePolicy,
)
from scanner3d.pipeline import NOISE_FLOOR, TRAVEL_MAX_MULTIPLE, TRAVEL_MIN_MULTIPLE

#: The band's two edges in depth units, and one travel comfortably inside it.
_LOW = TRAVEL_MIN_MULTIPLE * NOISE_FLOOR
_HIGH = TRAVEL_MAX_MULTIPLE * NOISE_FLOOR
_IN_BAND = (_LOW + _HIGH) / 2.0


# --- 1. the band is a deadband, and that is what makes this stable ------------------------


def test_a_travel_inside_the_band_changes_nothing() -> None:
    """The ordinary frame of an ordinary capture, and the commonest case by far.

    A controller chasing a single target value would correct on every frame,
    and every correction changes the very quantity it is measuring. The band is
    what stops that: inside it there is nothing to fix.
    """
    cadence = CadenceController(3)

    change = cadence.placed(_IN_BAND)

    assert change.reason == HELD_REASON
    assert not change.changed
    assert cadence.decimate == 3


def test_the_band_is_the_one_the_operator_is_already_shown() -> None:
    """A controller correcting towards a band other than the readout's would be
    answering a question nobody asked while the screen said something else."""
    low, high = CadencePolicy().band

    assert (low, high) == (_LOW, _HIGH)
    assert _LOW < CadencePolicy().target_travel < _HIGH


# --- 2. too fast: one frame is enough, and the correction is proportional ------------------


@pytest.mark.parametrize(
    ("nominal", "travel", "expected"),
    [
        # Measured against the arithmetic the docstring states: the cadence that
        # would have produced the target travel is current * target / travel.
        (3, 12.0, 1),
        (10, 7.0, 5),
        (10, 20.0, 2),
        (2, 6.0, 1),
    ],
)
def test_a_travel_over_the_band_corrects_towards_the_target_in_one_step(
    nominal: int, travel: float, expected: int
) -> None:
    """One step rather than one increment, and the reason is the clock.

    At this scanner's rate an increment a frame is seconds of scanning at a
    speed already known to be too fast, and those seconds are exactly the ones
    a capture is lost in.
    """
    cadence = CadenceController(nominal)

    change = cadence.placed(travel)

    assert change.reason == FASTER_REASON
    assert change.current == expected
    assert cadence.decimate == expected
    assert change.travel == travel


def test_a_correction_always_actually_goes_finer() -> None:
    """A change that rounded back to where it started would report a correction
    it did not make and leave the capture at a speed it has already judged too
    fast."""
    for nominal in range(2, MAX_DECIMATE + 1):
        cadence = CadenceController(nominal)
        change = cadence.placed(_HIGH * 1.001)
        assert change.current < nominal, nominal


def test_a_capture_already_keeping_every_frame_says_so_rather_than_holding() -> None:
    """The one state this module cannot answer, reported rather than hidden.

    There is no finer cadence than every complete delivery. An operator whose
    scan is still too fast here needs to be told that software has run out of
    things to do about it, not shown a controller quietly holding.
    """
    cadence = CadenceController(MIN_DECIMATE)

    change = cadence.placed(_HIGH * 4.0)

    assert change.reason == FLOOR_REASON
    assert not change.changed
    assert cadence.at_minimum


# --- 3. too slow: earned rather than computed ---------------------------------------------


def test_coarsening_takes_a_run_of_slow_frames_and_then_one_step() -> None:
    """The asymmetry with the fast direction is the safety property here.

    Keeping more frames than necessary costs disk and CPU. Keeping fewer loses
    data that cannot be recovered afterwards. So one direction acts on one
    frame and the other has to be earned.
    """
    cadence = CadenceController(3)

    for _ in range(SLOW_CONFIRMATIONS - 1):
        change = cadence.placed(_LOW / 2.0)
        assert change.reason == HELD_REASON
        assert cadence.decimate == 3

    change = cadence.placed(_LOW / 2.0)

    assert change.reason == SLOWER_REASON
    assert cadence.decimate == 4


def test_one_frame_back_inside_the_band_ends_the_run_of_slow_ones() -> None:
    """A run is a run. Two slow frames either side of a good one are not three."""
    cadence = CadenceController(3)

    for _ in range(SLOW_CONFIRMATIONS - 1):
        cadence.placed(_LOW / 2.0)
    assert cadence.placed(_IN_BAND).reason == HELD_REASON
    for _ in range(SLOW_CONFIRMATIONS - 1):
        assert cadence.placed(_LOW / 2.0).reason == HELD_REASON

    assert cadence.decimate == 3


def test_a_capture_already_at_the_widest_spacing_says_so() -> None:
    cadence = CadenceController(MAX_DECIMATE)

    for _ in range(SLOW_CONFIRMATIONS - 1):
        assert cadence.placed(_LOW / 2.0).reason == HELD_REASON
    change = cadence.placed(_LOW / 2.0)

    assert change.reason == CEILING_REASON
    assert cadence.decimate == MAX_DECIMATE


# --- 4. a refusal, which is the signal that arrives before its own explanation ------------


@pytest.mark.parametrize(("nominal", "expected"), [(10, 5), (5, 2), (3, 1), (2, 1)])
def test_a_refused_frame_halves_the_spacing_at_once(nominal: int, expected: int) -> None:
    """A frame that was not placed carries no travel, so there is no measurement
    to be proportional to - and waiting for the next one that does is waiting
    through the streak that ends the capture."""
    cadence = CadenceController(nominal)

    change = cadence.refused()

    assert change.reason == REFUSAL_REASON
    assert change.travel is None
    assert cadence.decimate == expected


def test_three_refusals_reach_the_finest_spacing_from_anywhere() -> None:
    """Three, because that is the consecutive-failure count that ends a capture.

    A controller that could not reach its floor within the streak would still
    be correcting when the capture it was correcting for had already stopped.
    """
    cadence = CadenceController(MAX_DECIMATE)

    for _ in range(3):
        cadence.refused()

    assert cadence.at_minimum


def test_a_refusal_at_the_finest_spacing_reports_the_floor() -> None:
    cadence = CadenceController(MIN_DECIMATE)

    change = cadence.refused()

    assert change.reason == FLOOR_REASON
    assert not change.changed


def test_a_refusal_ends_a_run_of_slow_frames() -> None:
    """The two signals disagree, and the refusal is the one to believe.

    A capture that was crawling and has just refused a frame is not a capture
    that should coarsen on the strength of the frames before it.
    """
    cadence = CadenceController(4)

    for _ in range(SLOW_CONFIRMATIONS - 1):
        cadence.placed(_LOW / 2.0)
    cadence.refused()
    for _ in range(SLOW_CONFIRMATIONS - 1):
        cadence.placed(_LOW / 2.0)

    assert cadence.decimate == 2


# --- 5. what is not a measurement --------------------------------------------------------


@pytest.mark.parametrize("travel", [0.0, -1.0, math.inf, math.nan])
def test_a_travel_that_is_not_a_positive_distance_decides_nothing(travel: float) -> None:
    """Deliberately not read as zero speed.

    A stationary scanner reports a small POSITIVE travel, not an unmeasurable
    one, so these values carry no information about speed at all - and a
    refused frame's travel is infinite by construction, which must not be read
    as the slowest frame of the capture and answered by coarsening.
    """
    cadence = CadenceController(3)

    change = cadence.placed(travel)

    assert change.reason == UNMEASURED_REASON
    assert not change.changed
    assert cadence.decimate == 3


def test_an_unmeasurable_travel_does_not_advance_a_run_of_slow_frames() -> None:
    cadence = CadenceController(3)

    for _ in range(SLOW_CONFIRMATIONS * 2):
        cadence.placed(math.inf)

    assert cadence.decimate == 3


# --- 6. the bounds hold, whatever it is fed ------------------------------------------------


def test_the_spacing_never_leaves_the_policy_range() -> None:
    """Asserted over a deliberately hostile sequence rather than argued.

    Every input this module accepts, alternating, for long enough that any
    single rule that could walk out of range would have done so.
    """
    cadence = CadenceController(5)
    feed = [_HIGH * 10.0, _LOW / 10.0, _IN_BAND, math.nan, _HIGH * 1.5, _LOW / 3.0]

    for round_trip in range(40):
        if round_trip % 5 == 4:
            cadence.refused()
        else:
            cadence.placed(feed[round_trip % len(feed)])
        assert MIN_DECIMATE <= cadence.decimate <= MAX_DECIMATE


def test_the_same_measurements_produce_the_same_decisions() -> None:
    """Nothing here reads a clock or a random number, so a capture's cadence can
    be replayed from its own record."""
    feed = [_HIGH * 3.0, _LOW / 4.0, _IN_BAND, _LOW / 4.0, _LOW / 4.0, _LOW / 4.0]

    def run() -> list[tuple[int, str]]:
        cadence = CadenceController(6)
        out = [(c.current, c.reason) for c in (cadence.placed(t) for t in feed)]
        out.append((cadence.refused().current, cadence.refused().reason))
        return out

    assert run() == run()


# --- 7. the policy is a value, and the run is a record --------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum": 0},
        {"maximum": MAX_DECIMATE + 1},
        {"minimum": 4, "maximum": 4},
        {"minimum": 6, "maximum": 3},
        {"target_travel_multiple": 0.0},
        {"target_travel_multiple": math.inf},
        # A target outside the band it corrects towards: every correction would
        # land somewhere this controller then moves away from.
        {"target_travel_multiple": TRAVEL_MAX_MULTIPLE + 1.0},
        {"target_travel_multiple": TRAVEL_MIN_MULTIPLE - 1.0},
        # Coarsening - the direction that loses frames - with no evidence.
        {"slow_confirmations": 0},
        # A divisor that does not divide, so a refusal would do nothing.
        {"refusal_divisor": 1},
    ],
)
def test_a_policy_whose_own_bounds_do_not_hold_is_refused_at_construction(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(CadenceNotAllowed):
        CadencePolicy(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("nominal", [0, MAX_DECIMATE + 1, -3])
def test_a_nominal_outside_the_range_is_refused(nominal: int) -> None:
    """It could never return to where the operator set it."""
    with pytest.raises(CadenceNotAllowed):
        CadenceController(nominal)


def test_a_nominal_that_is_not_a_whole_number_of_frames_is_refused() -> None:
    with pytest.raises(CadenceNotAllowed):
        CadenceController(2.5)  # type: ignore[arg-type]


def test_the_policy_survives_a_json_round_trip() -> None:
    """A record written with ``allow_nan=False`` refuses anything JSON has no
    word for, which is what makes the policy safe to embed."""
    policy = CadencePolicy().as_dict()

    assert json.loads(json.dumps(policy, allow_nan=False)) == policy


def test_the_record_says_what_the_cadence_did_and_not_only_where_it_ended() -> None:
    """A reader asking "was this capture's spacing constant" needs an answer that
    a final value alone cannot give: a cadence that went to 1 and back to 3 ends
    where it started."""
    cadence = CadenceController(3)
    cadence.placed(_HIGH * 5.0)
    for _ in range(SLOW_CONFIRMATIONS * 2):
        cadence.placed(_LOW / 2.0)

    record = cadence.record()

    assert record["nominal"] == 3
    assert record["lowest"] == 1
    assert record["final"] == 3
    assert record["changes"] == 3
    assert record["limitations"]
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_a_capture_that_never_left_the_band_records_no_changes() -> None:
    cadence = CadenceController(3)

    for _ in range(20):
        cadence.placed(_IN_BAND)

    record = cadence.record()
    assert record["changes"] == 0
    assert record["lowest"] == record["highest"] == 3


# --- 8. the closed loop, which is the only test the others cannot stand in for -------------


def _settled(speed: float, nominal: int, frames: int = 60) -> list[int]:
    """Drive the controller against a scanner moving at a constant speed.

    ``speed`` is surface travel per DELIVERED frame, so the travel a kept frame
    measures is ``speed * decimate`` - the proportionality the correction is
    computed from, stated here as the fixture rather than assumed by it.
    """
    cadence = CadenceController(nominal)
    seen = []
    for _ in range(frames):
        cadence.placed(speed * cadence.decimate)
        seen.append(cadence.decimate)
    return seen


@pytest.mark.parametrize("speed", [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("nominal", [1, 3, 10])
def test_a_steady_speed_settles_and_stays_settled(speed: float, nominal: int) -> None:
    """The property no amount of per-decision arithmetic can establish.

    A correction that is individually sensible on every frame can still
    oscillate forever - the classic failure of a controller whose own output
    changes what it measures. Both halves are asserted: the cadence stops
    moving, and where it stops is either inside the band or hard against a bound
    the operator's speed put it against.
    """
    seen = _settled(speed, nominal)
    tail = seen[-20:]

    assert len(set(tail)) == 1, f"still moving at {speed} from {nominal}: {seen[-30:]}"

    settled = tail[0]
    travel = speed * settled
    at_bound = settled in (MIN_DECIMATE, MAX_DECIMATE)
    assert _LOW <= travel <= _HIGH or at_bound, (
        f"settled at {settled} with travel {travel:.3f}, outside the band and not at a bound"
    )


def test_a_scanner_that_speeds_up_mid_capture_is_followed_down() -> None:
    """The case the whole module is for, and the lag it cannot avoid.

    The correction arrives one kept frame after the speed changed, because the
    measurement it acts on describes a frame that was already placed. That frame
    is exactly the one that may already have been refused, which is why a
    refusal is an input in its own right.
    """
    cadence = CadenceController(8)
    for _ in range(10):
        cadence.placed(0.4 * cadence.decimate)
    crawling = cadence.decimate

    cadence.placed(6.0 * crawling)
    sprinting = cadence.decimate

    assert sprinting < crawling
    assert _LOW <= 6.0 * sprinting <= _HIGH or sprinting == MIN_DECIMATE
