"""How many delivered frames one kept frame is worth, decided while scanning.

Decimation is the one capture setting whose right value is not a property of the
scanner, the object or the operator's intent, but of how fast the scanner
happens to be moving *at that moment*. It is currently a number typed in before
a capture starts and held for its whole length, and the two ends of that are
both wrong:

- Set for a brisk sweep, it wastes a slow one. Frames a fraction of the noise
  floor apart measure noise rather than motion, and every one of them is another
  step for an error to accumulate over - see the drift table on
  :data:`~scanner3d.pipeline.TRAVEL_MIN_MULTIPLE`, where the SLOWEST row is
  among the worst.
- Set for a slow sweep, it loses a brisk one. Past
  :data:`~scanner3d.pipeline.TRAVEL_MAX_MULTIPLE` the frames no longer overlap
  where alignment starts looking, and a capture that was working stops.

Neither is something the person carrying the scanner can hold by hand. The
travel readout already tells them which way they are wrong, four times a second,
and asks them to fix it with their arms.

What this module is NOT
-----------------------
It is not a fix for tracking loss, and it must not be described as one. The
whole range it can move within is the operator's own setting down to 1, so
everything it can do about a capture that is moving too fast can also be had by
setting the spin to 1 and leaving it there. What it adds is the other half:
holding the measured travel inside the band the reconstruction is measured to
work in, on a capture whose speed varies - which is every hand-carried one.

It also cannot see the future. Every measurement it acts on is the travel of a
frame that has ALREADY been placed, so it learns that the scanner sped up one
kept frame after it did. That is why a refusal is a signal in its own right
here: a frame that could not be placed carries no travel number at all, and it
is the one event that says "too fast" before the streak that ends the capture.

What it decides on
------------------
Two inputs, both of which the capture already produces per frame:

- :meth:`CadenceController.placed`, the travel of an accepted frame, judged
  against the same band :func:`~scanner3d.pipeline.travel_verdict` shows the
  operator. Not a second opinion about capture speed - the same one.
- :meth:`CadenceController.refused`, one alignment refusal.

Deterministic and self-contained. Nothing here reads a clock, a device or a
random number, so a controller fed the same measurements in the same order makes
the same decisions - which is what lets a capture's cadence be replayed from its
own record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.pipeline import (
    NOISE_FLOOR,
    TRAVEL_MAX_MULTIPLE,
    TRAVEL_MIN_MULTIPLE,
)

SCHEMA_VERSION: Final = 1

#: Fewest delivered frames one kept frame may be worth. One: every complete
#: delivery is kept, which is as fine as this scanner can be asked to go. There
#: is no finer setting to fall back on, so a capture already here that is still
#: moving too fast has run out of things software can do about it - which the
#: controller says rather than hides.
MIN_DECIMATE: Final = 1

#: Most. Ten, which is the range the window's own control has always offered;
#: this module does not widen a setting, it only moves within one.
MAX_DECIMATE: Final = 10

#: The travel one kept frame should ideally be worth, as a multiple of the noise
#: floor.
#:
#: The GEOMETRIC mean of the band's two edges, not the arithmetic one, and that
#: is not a flourish. The correction below is multiplicative - travel is
#: proportional to how many deliveries a kept frame is worth, so halving the
#: decimation halves the travel - and the geometric mean is the point equidistant
#: from both edges in the ratio the controller actually works in. The arithmetic
#: midpoint sits closer to the fast edge in ratio terms, which is the edge where
#: being wrong loses the capture rather than merely blurs it.
#:
#: Deliberately the middle of the band rather than the best row of the drift
#: table on :data:`~scanner3d.pipeline.TRAVEL_MIN_MULTIPLE`. That table's own
#: text says it is "guidance for turning a knob, not a calibration", four points
#: from two captures with one of them not a clean comparison. Aiming at its best
#: row would be treating it as a calibration.
TARGET_TRAVEL_MULTIPLE: Final = math.sqrt(TRAVEL_MIN_MULTIPLE * TRAVEL_MAX_MULTIPLE)

#: Consecutive too-slow frames before the cadence is allowed to COARSEN.
#:
#: The asymmetry with the fast direction, which acts on one frame, is the whole
#: safety property of this module. Keeping more frames than necessary costs disk
#: and CPU; keeping fewer than necessary loses data that cannot be recovered
#: afterwards and can end the capture. So going finer is immediate and going
#: coarser has to be earned.
#:
#: Three, matching :data:`~scanner3d.capture_sequence.TRACKING_LOSS_STREAK` for
#: the same reason that threshold is three: one frame is a frame, two is a
#: moment, and three in a row is the capture actually being like that.
SLOW_CONFIRMATIONS: Final = 3

#: What one alignment refusal divides the cadence by.
#:
#: Two, so that from anywhere in the range the controller reaches
#: :data:`MIN_DECIMATE` within the three consecutive failures that end a capture
#: - 10, 5, 2, 1 - and from the window's default of 3 it reaches it on the first
#: refusal. A refusal is the only evidence that arrives BEFORE the travel
#: measurement that would have explained it, because a frame that was not placed
#: has no travel, so it is answered decisively rather than proportionally.
#:
#: A refusal is not proof of speed. A hand across the view refuses a frame too,
#: and this will respond to that by keeping more frames for a while, which costs
#: disk and nothing else. That asymmetry is deliberate; see
#: :data:`SLOW_CONFIRMATIONS`.
REFUSAL_DIVISOR: Final = 2

#: Why the cadence last changed, or did not.
FASTER_REASON: Final = "travel_over_band"
SLOWER_REASON: Final = "travel_under_band"
REFUSAL_REASON: Final = "alignment_refused"
HELD_REASON: Final = "travel_in_band"
FLOOR_REASON: Final = "already_at_minimum"
CEILING_REASON: Final = "already_at_maximum"
UNMEASURED_REASON: Final = "travel_not_measurable"

LIMITATIONS: Final = (
    "Every measurement this acts on describes a frame that was already placed, so the cadence "
    "follows the operator's speed rather than anticipating it. A scanner that speeds up sharply "
    "is corrected one kept frame after it did, and the frame in between is exactly the one that "
    "may already have been refused.",
    "Travel is assumed proportional to how many deliveries a kept frame is worth. That holds on "
    "average and is noisier per frame than the correction implies, because travel is the "
    "product of THREE things and only one of them is the operator: the angular rate, the "
    "distance of the visible surface from the axis of motion, and the gap between kept frames. "
    "Measured on 20260901-191525, a 60-frame turntable sweep at a constant motor rate: surface "
    "travel varied 6.0x across the sweep (CV 37%) while the recovered ROTATION ANGLE varied "
    "3.0x (CV 18%), and dividing out the camera's own alternating delivery left 13.9%. So a "
    "single frame's travel overstates how much the operator changed speed, and the deadband is "
    "what keeps that from being corrected on every frame.",
    "The scanner delivers frames in pairs - about 69 ms then 373 ms - so kept frames are not "
    "evenly spaced in TIME whatever the decimation is. On the same sweep the alternating phases "
    "differed by 1.24x in rotation angle. Nothing here corrects for that; it is part of the "
    "noise the deadband absorbs.",
    "A refusal is treated as evidence of speed. It is not: a hand across the view, a burst of "
    "sensor noise or a genuinely singular flat panel all refuse a frame without the scanner "
    "moving at all, and this answers each of them by keeping more frames for a while.",
    "This holds a capture inside the band the reconstruction was MEASURED to work in; it does "
    "not establish that any capture inside that band reconstructs well. The band itself is four "
    "points from two captures - see pipeline.TRAVEL_MIN_MULTIPLE - and is guidance rather than "
    "a calibration.",
    "The span a capture used reaches its StreamResult, the log and the operator, and NOT the "
    "capture folder: a capture holds raw frames and the records its segments need, and nothing "
    "that describes the whole of it. A folder whose frames were unevenly spaced therefore does "
    "not say so on disk. Nothing downstream is wrong without it - a reconstruction fits "
    "adjacent pairs assuming a small step, which is what this maintains - but a later reader "
    "cannot tell.",
    "Nothing here makes tracking survive a speed it otherwise would not. The whole range is the "
    "operator's own setting down to one delivered frame per kept frame, so a capture this "
    "cannot hold is one that setting the control to 1 could not have held either.",
)


class CadenceNotAllowed(ScannerError):
    """A cadence policy or transition this module refuses to make."""

    status = DoctorStatus.UNSUPPORTED_MODE


@dataclass(frozen=True, slots=True)
class CadencePolicy:
    """Every bound this controller moves within, as one recordable value.

    Schema-versioned and carried into the durable record for the same reason
    :class:`~scanner3d.tracking.TrackingPolicy` is: a reader looking at a
    capture whose frame spacing varies has to be able to see what was allowed to
    vary it rather than infer it from the spacing.
    """

    minimum: int = MIN_DECIMATE
    maximum: int = MAX_DECIMATE
    target_travel_multiple: float = TARGET_TRAVEL_MULTIPLE
    slow_confirmations: int = SLOW_CONFIRMATIONS
    refusal_divisor: int = REFUSAL_DIVISOR

    def __post_init__(self) -> None:
        """Refuse a policy whose own bounds do not hold, before one is ever used."""
        if self.minimum < MIN_DECIMATE:
            raise CadenceNotAllowed(
                f"minimum {self.minimum} is under the {MIN_DECIMATE} delivered frame a kept "
                "frame is at least worth; there is no finer cadence to fall back on"
            )
        if self.maximum > MAX_DECIMATE:
            raise CadenceNotAllowed(
                f"maximum {self.maximum} is over the {MAX_DECIMATE} the capture control offers; "
                "this module moves within a setting rather than widening one"
            )
        if self.minimum >= self.maximum:
            # Equal bounds are not "pinned", they are a controller that cannot
            # answer either signal and would report holding for reasons that
            # have nothing to do with the measurement.
            raise CadenceNotAllowed(
                f"minimum {self.minimum} is not below maximum {self.maximum}, so the cadence "
                "could never move in either direction and every decision would report a hold "
                "it did not measure"
            )
        if not math.isfinite(self.target_travel_multiple) or self.target_travel_multiple <= 0.0:
            raise CadenceNotAllowed(
                f"target_travel_multiple {self.target_travel_multiple!r} must be a finite "
                "positive multiple of the noise floor"
            )
        if not TRAVEL_MIN_MULTIPLE <= self.target_travel_multiple <= TRAVEL_MAX_MULTIPLE:
            # A target outside the band is a controller aiming at a speed it
            # would then correct itself away from on the very next frame.
            raise CadenceNotAllowed(
                f"target_travel_multiple {self.target_travel_multiple} is outside the "
                f"[{TRAVEL_MIN_MULTIPLE}, {TRAVEL_MAX_MULTIPLE}] band it is correcting towards, "
                "so every correction would land somewhere this controller then moves away from"
            )
        if self.slow_confirmations < 1:
            raise CadenceNotAllowed(
                f"slow_confirmations {self.slow_confirmations} is under one, so coarsening would "
                "need no evidence at all - and coarsening is the direction that loses frames"
            )
        if self.refusal_divisor < 2:
            raise CadenceNotAllowed(
                f"refusal_divisor {self.refusal_divisor} does not divide, so a refusal would "
                "never move the cadence and the one signal that arrives before its own "
                "explanation would do nothing"
            )

    @property
    def target_travel(self) -> float:
        """The travel one kept frame should be worth, in depth units."""
        return self.target_travel_multiple * NOISE_FLOOR

    @property
    def band(self) -> tuple[float, float]:
        """The travel range that is left alone, in depth units.

        The same two numbers :func:`~scanner3d.pipeline.travel_verdict` shows the
        operator, deliberately. A controller correcting towards a different band
        from the one on screen would be answering a question nobody asked while
        the readout said something else.
        """
        return (TRAVEL_MIN_MULTIPLE * NOISE_FLOOR, TRAVEL_MAX_MULTIPLE * NOISE_FLOOR)

    def as_dict(self) -> dict[str, object]:
        low, high = self.band
        return {
            "schema_version": SCHEMA_VERSION,
            "noise_floor": NOISE_FLOOR,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "target_travel_multiple": self.target_travel_multiple,
            "target_travel": self.target_travel,
            "band_travel_low": low,
            "band_travel_high": high,
            "travel_min_multiple": TRAVEL_MIN_MULTIPLE,
            "travel_max_multiple": TRAVEL_MAX_MULTIPLE,
            "slow_confirmations": self.slow_confirmations,
            "refusal_divisor": self.refusal_divisor,
        }


#: What an ordinary capture runs under.
DEFAULT_POLICY: Final = CadencePolicy()


@dataclass(frozen=True, slots=True)
class CadenceChange:
    """One decision, with the measurement it was made on beside it.

    Returned from every input so a caller never has to compare the cadence
    before and after to find out what happened, and so a hold is as explicit as
    a change: "the travel was in the band" and "the travel could not be
    measured" both leave the number alone and are not the same event.
    """

    previous: int
    current: int
    reason: str
    #: The travel this was decided on, or ``None`` for a refusal, which has no
    #: travel to be decided on - that being the reason refusals are an input at
    #: all.
    travel: float | None
    #: Consecutive too-slow frames behind this decision, after it was applied.
    slow_streak: int

    @property
    def changed(self) -> bool:
        return self.current != self.previous

    def as_dict(self) -> dict[str, object]:
        return {
            "previous": self.previous,
            "current": self.current,
            "reason": self.reason,
            "travel": self.travel,
            "slow_streak": self.slow_streak,
            "changed": self.changed,
        }


class CadenceController:
    """The decimation in force, and the two measurements that move it.

    Usage is three calls: :meth:`placed` for every accepted frame's travel,
    :meth:`refused` for every alignment refusal, and :attr:`decimate` read by
    whatever is doing the selecting. The controller never selects anything
    itself and never touches a device.
    """

    def __init__(self, nominal: int, *, policy: CadencePolicy = DEFAULT_POLICY) -> None:
        if not isinstance(nominal, int) or isinstance(nominal, bool):
            raise CadenceNotAllowed(
                f"nominal cadence {nominal!r} is not an integer number of delivered frames"
            )
        if not policy.minimum <= nominal <= policy.maximum:
            raise CadenceNotAllowed(
                f"nominal cadence {nominal} is outside the [{policy.minimum}, {policy.maximum}] "
                "range this controller may move within, so it could never return to where the "
                "operator set it"
            )
        self._policy = policy
        #: What the operator asked for. Kept because the controller may be
        #: switched off mid-capture and because a record that shows only where
        #: the cadence ended up cannot say whether it moved.
        self._nominal = nominal
        self._current = nominal
        self._slow_streak = 0
        self._changes = 0
        self._lowest = nominal
        self._highest = nominal

    @property
    def policy(self) -> CadencePolicy:
        return self._policy

    @property
    def nominal(self) -> int:
        """What the operator set before the capture started."""
        return self._nominal

    @property
    def decimate(self) -> int:
        """Delivered frames one kept frame is currently worth."""
        return self._current

    @property
    def at_minimum(self) -> bool:
        """Whether every complete delivery is already being kept.

        Worth asking rather than inferring, because it is the one state in which
        "still too fast" is not something this module can answer - and telling
        the operator that is more use than silently holding.
        """
        return self._current <= self._policy.minimum

    def placed(self, travel: float) -> CadenceChange:
        """One accepted frame's measured surface travel.

        Inside the band nothing happens, and that deadband is what makes this
        stable: a controller chasing a single target value would correct on
        every frame, and the correction changes the very quantity it measures.
        Outside it, the direction decides how hard:

        - **over the band**, one frame is enough, and the correction is
          proportional. Travel is proportional to how many deliveries a kept
          frame is worth, so the cadence that would have produced the target
          travel is ``current * target / travel``, and going there in one step
          rather than one increment at a time is the point - an increment a
          frame, at the rate this scanner delivers kept frames, is seconds of
          scanning at a speed already known to be too fast.
        - **under the band**, :attr:`CadencePolicy.slow_confirmations` in a row,
          and then a single step. Coarsening loses frames that cannot be
          recovered afterwards, so it is earned rather than computed.

        A travel that is not a positive finite number decides nothing. It is
        not zero-speed evidence - a stationary scanner reports a small positive
        travel, not an unmeasurable one - so it is recorded and ignored.
        """
        previous = self._current
        if not math.isfinite(travel) or travel <= 0.0:
            return self._change(previous, UNMEASURED_REASON, None)
        low, high = self._policy.band
        if travel > high:
            self._slow_streak = 0
            if self.at_minimum:
                return self._change(previous, FLOOR_REASON, travel)
            # Rounded to the nearest whole delivery and then held strictly
            # below where it started: a proportional correction that rounded
            # back to the current value would report a change it did not make
            # and leave the capture at a speed it has already judged too fast.
            wanted = round(previous * self._policy.target_travel / travel)
            self._current = _clamped(min(wanted, previous - 1), self._policy)
            return self._change(previous, FASTER_REASON, travel)
        if travel < low:
            self._slow_streak += 1
            if self._slow_streak < self._policy.slow_confirmations:
                return self._change(previous, HELD_REASON, travel)
            self._slow_streak = 0
            if self._current >= self._policy.maximum:
                return self._change(previous, CEILING_REASON, travel)
            self._current = _clamped(previous + 1, self._policy)
            return self._change(previous, SLOWER_REASON, travel)
        self._slow_streak = 0
        return self._change(previous, HELD_REASON, travel)

    def refused(self) -> CadenceChange:
        """One frame the tracker could not place.

        Divided rather than decremented, and acted on immediately, because this
        is the only input that arrives before the measurement that would have
        explained it: a frame that was not placed has no travel. See
        :data:`REFUSAL_DIVISOR` for why the answer is decisive and for what it
        costs when the refusal had nothing to do with speed.
        """
        previous = self._current
        self._slow_streak = 0
        if self.at_minimum:
            return self._change(previous, FLOOR_REASON, None)
        self._current = _clamped(previous // self._policy.refusal_divisor, self._policy)
        return self._change(previous, REFUSAL_REASON, None)

    def _change(self, previous: int, reason: str, travel: float | None) -> CadenceChange:
        """Finish one decision, keeping the span this capture actually used."""
        if self._current != previous:
            self._changes += 1
        self._lowest = min(self._lowest, self._current)
        self._highest = max(self._highest, self._current)
        return CadenceChange(
            previous=previous,
            current=self._current,
            reason=reason,
            travel=travel,
            slow_streak=self._slow_streak,
        )

    def record(self) -> dict[str, object]:
        """What the cadence did over a capture, for a durable record.

        The span and the change count rather than the whole history: a reader
        asking "was this capture's frame spacing constant" is answered by
        ``changes``, and one asking "how far did it range" by the two bounds,
        while a per-frame list would grow with the capture and say the same
        thing.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": self._policy.as_dict(),
            "nominal": self._nominal,
            "final": self._current,
            "lowest": self._lowest,
            "highest": self._highest,
            "changes": self._changes,
            "limitations": list(LIMITATIONS),
        }


def _clamped(value: int, policy: CadencePolicy) -> int:
    """One cadence held inside the policy's own range."""
    return max(policy.minimum, min(policy.maximum, value))
