"""Where the returns are, how fast they are disappearing, and what to do about it.

The report this module answers
-------------------------------
On 2026-08-28 the owner watched the false-coloured depth preview of capture
``20260828-110641`` go almost entirely black and read it as "the preview has
stopped showing the object". The raw bytes say something else, and the
difference matters more than the picture did:

===========  =========  ===========
frame        returned   no return
===========  =========  ===========
0            24.02%     75.98%
80           12.36%     87.64%
165           3.60%     96.40%
===========  =========  ===========

Those samples are raw code 0, which this repository reads as *the sensor
returned no depth for this pixel* (:data:`scanner3d.depth_metric.NO_RETURN_SAMPLE`)
and nothing else. The panel was not broken and the renderer did not drop
anything: by frame 165 there genuinely was almost nothing to draw. Frame 165's
returns are also not spread thinly over the sensor - they collapse to the LEFT,
with the right-hand quadrants at 99.94% and 100.00% no-return. A single yield
percentage cannot tell that apart from a uniform fade, and "uniform fade" and
"the object has slid off one side of the sensor" call for opposite corrections
from whoever is holding the scanner.

So this module computes two kinds of evidence and one verdict over them, all of
it pure:

* :func:`spatial_returns` - WHERE the returns are. Counts per fixed quadrant
  and the bounding box of every non-zero sample, over one already-unpacked
  frame. Enough to distinguish uniform loss from one-sided collapse, which is
  the distinction the physical capture actually shows.
* :class:`ReturnHealthTracker` - HOW FAST they are going. A bounded, resettable
  comparison of this frame's yield against a small fixed window of recent
  frames, so a collapse in progress can be named while a scan is still
  salvageable.
* :func:`return_health` - what to SAY. One deterministic verdict/advice pair,
  shared by the Qt appliance and the standalone Tk and HTTP depth viewers, so
  all three front ends say the same sentence about the same frame.

What this module refuses to say
--------------------------------
* A no-return is never a distance. Nothing here converts, interpolates, fills,
  smears or otherwise invents a sample the sensor did not return, and no
  physical unit is implied by any number below.
* It never names a cause. Code 0 is what the firmware reported; whether that
  pixel was too dark, too bright, too far, too shiny, out of the projector's
  cone or occluded is not in the bytes, and the advice below therefore names
  only actions the operator can take and observe.
* It never decides anything. A verdict here is advisory: it accepts and refuses
  nothing, stops no capture, moves no tracking gate and changes no
  reconstruction threshold. The only thing it is allowed to do is appear on
  screen.

Everything here is deterministic and allocation-bounded: the same samples give
the same answer on every machine, and :class:`ReturnHealthTracker` holds at most
:data:`HISTORY_FRAMES` floats however long a scan runs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.errors import DoctorStatus, ScannerError

#: Every dict this module emits stamps this.
SCHEMA_VERSION: Final = 1


# --- refusals ---------------------------------------------------------------


class ReturnHealthError(ScannerError):
    """Base for every refusal here. Raised before a sample is looked at."""

    status = DoctorStatus.UNSUPPORTED_MODE


class FrameShapeMismatch(ReturnHealthError):
    """``samples`` was not exactly ``width * height`` long, or the shape was not positive.

    Refused rather than padded, cropped or reshaped: a sequence of another
    length is a frame of another shape, and guessing which one would put every
    quadrant boundary in the wrong place while still producing plausible
    numbers.
    """


class InvalidYield(ReturnHealthError):
    """A yield fraction outside ``[0, 1]``, or not a real number, was offered."""


# --- where the returns are --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Quadrant:
    """One of a frame's four fixed quadrants, counted.

    ``name`` is one of :data:`QUADRANT_NAMES`. ``samples`` is that quadrant's
    own pixel count, which is NOT ``frame samples / 4`` when a dimension is
    odd - see :func:`spatial_returns` for who owns the middle row/column.
    """

    name: str
    samples: int
    returned: int

    @property
    def no_return(self) -> int:
        """Samples here that carried :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`."""
        return self.samples - self.returned

    @property
    def yield_fraction(self) -> float:
        """Returned share of this quadrant, or ``0.0`` for an empty quadrant."""
        return 0.0 if self.samples == 0 else self.returned / self.samples

    @property
    def no_return_fraction(self) -> float:
        """No-return share of this quadrant, or ``0.0`` for an empty quadrant."""
        return 0.0 if self.samples == 0 else self.no_return / self.samples

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "samples": self.samples,
            "returned": self.returned,
            "no_return": self.no_return,
            "yield_fraction": self.yield_fraction,
            "no_return_fraction": self.no_return_fraction,
        }


#: The four quadrant names, in the fixed order :attr:`SpatialReturns.quadrants`
#: reports them.
QUADRANT_NAMES: Final = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True, slots=True)
class Footprint:
    """The smallest axis-aligned box containing every non-zero return.

    All four bounds are INCLUSIVE pixel indices, so a single returning pixel at
    column 7 has ``left == right == 7`` and :attr:`width` 1. There is no
    footprint at all for a frame with no returns; :attr:`SpatialReturns.footprint`
    is ``None`` there rather than a zero-sized box at the origin, which would
    be indistinguishable from a real return in the corner.
    """

    left: int
    right: int
    top: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class SpatialReturns:
    """Where one frame's returns are, in counts and pixel indices only.

    No colour, no unit and no cause: this is the sensor's own no-return
    bookkeeping, arranged so that a uniform fade and a one-sided collapse do
    not produce the same record.
    """

    width: int
    height: int
    samples: int
    returned: int
    top_left: Quadrant
    top_right: Quadrant
    bottom_left: Quadrant
    bottom_right: Quadrant
    footprint: Footprint | None

    @property
    def quadrants(self) -> tuple[Quadrant, Quadrant, Quadrant, Quadrant]:
        """The four quadrants in :data:`QUADRANT_NAMES` order."""
        return (self.top_left, self.top_right, self.bottom_left, self.bottom_right)

    @property
    def no_return(self) -> int:
        return self.samples - self.returned

    @property
    def yield_fraction(self) -> float:
        return 0.0 if self.samples == 0 else self.returned / self.samples

    @property
    def no_return_fraction(self) -> float:
        return 0.0 if self.samples == 0 else self.no_return / self.samples

    @property
    def emptiest_quadrant(self) -> Quadrant:
        """The quadrant with the largest no-return share.

        Ties are broken by :data:`QUADRANT_NAMES` order, so an all-zero frame
        names ``top_left`` deterministically rather than whichever quadrant a
        dict happened to iterate first.
        """
        return max(
            self.quadrants,
            key=lambda q: (q.no_return_fraction, -QUADRANT_NAMES.index(q.name)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "width": self.width,
            "height": self.height,
            "samples": self.samples,
            "returned": self.returned,
            "no_return": self.no_return,
            "yield_fraction": self.yield_fraction,
            "no_return_fraction": self.no_return_fraction,
            "quadrants": [quadrant.as_dict() for quadrant in self.quadrants],
            "footprint": None if self.footprint is None else self.footprint.as_dict(),
            "note": (
                "counts of raw samples only. A no-return is code "
                f"{NO_RETURN_SAMPLE}: the sensor reported no depth for that pixel. It is not a "
                "distance, no physical unit is implied, and no cause is claimed"
            ),
        }


def _split(size: int) -> int:
    """The first index of the second half of ``size`` pixels.

    ``size // 2``, which means the two halves are equal for an even dimension
    and the ODD MIDDLE ROW OR COLUMN BELONGS TO THE SECOND HALF - the right
    half, or the bottom half. The frames this repository actually renders are
    640x400 and never exercise that branch, but a fixture five pixels wide
    does, and one arbitrary rule stated here is worth more than four
    implementations each quietly choosing their own.
    """
    return size // 2


def spatial_returns(samples: Sequence[int], width: int, height: int) -> SpatialReturns:
    """Count where ``samples`` returned, in row-major ``width`` x ``height`` order.

    ``samples[row * width + column]`` is one raw uint16 code, exactly the
    layout :func:`scanner3d.viewer.frame_stats` already unpacks. Any code other
    than :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE` counts as a return;
    nothing here looks at what the code is worth, because this module has no
    opinion about distance.

    Refuses (:class:`FrameShapeMismatch`) a non-positive shape or a length that
    is not exactly ``width * height``, before reading a single sample.

    One pass. Quadrant counts and the footprint of the returns are accumulated
    together, so this costs the same walk over the frame that counting the
    returns alone would.
    """
    if width <= 0 or height <= 0:
        raise FrameShapeMismatch(
            f"a frame shape must be positive in both dimensions; got {width}x{height}. "
            "Nothing was read"
        )
    if len(samples) != width * height:
        raise FrameShapeMismatch(
            f"a {width}x{height} frame is exactly {width * height} samples; got {len(samples)}. "
            "Nothing was read: a sequence of another length is a frame of another shape, and "
            "reshaping it here would move every quadrant boundary"
        )

    mid_column = _split(width)
    mid_row = _split(height)
    counts = [0, 0, 0, 0]  # top_left, top_right, bottom_left, bottom_right
    returned = 0
    left = width
    right = -1
    top = height
    bottom = -1

    for row in range(height):
        base = row * width
        row_half = 2 if row >= mid_row else 0
        for column in range(width):
            if samples[base + column] == NO_RETURN_SAMPLE:
                continue
            returned += 1
            counts[row_half + (1 if column >= mid_column else 0)] += 1
            if column < left:
                left = column
            if column > right:
                right = column
            if row < top:
                top = row
            if row > bottom:
                bottom = row

    left_width = mid_column
    right_width = width - mid_column
    top_height = mid_row
    bottom_height = height - mid_row
    quadrant_samples = (
        left_width * top_height,
        right_width * top_height,
        left_width * bottom_height,
        right_width * bottom_height,
    )
    quadrants = tuple(
        Quadrant(name=name, samples=size, returned=count)
        for name, size, count in zip(QUADRANT_NAMES, quadrant_samples, counts, strict=True)
    )
    return SpatialReturns(
        width=width,
        height=height,
        samples=width * height,
        returned=returned,
        top_left=quadrants[0],
        top_right=quadrants[1],
        bottom_left=quadrants[2],
        bottom_right=quadrants[3],
        footprint=None if right < 0 else Footprint(left=left, right=right, top=top, bottom=bottom),
    )


# --- how fast they are going ------------------------------------------------

#: Verdicts, worst first. Exactly four, and every front end shows one of them.
NO_RETURNS: Final = "no returns"
SEVERELY_SPARSE: Final = "severely sparse"
COLLAPSING: Final = "collapsing"
USABLE: Final = "usable"

#: The verdicts that are a warning: the ones a front end must make loud.
WARNING_VERDICTS: Final = (NO_RETURNS, SEVERELY_SPARSE, COLLAPSING)

#: Below this share of the frame there is too little geometry left for the
#: scan to be going anywhere, whatever the recent frames looked like. Chosen
#: against the physical capture: its frame 165 returns 3.60% and is the frame
#: the owner reported as a black panel, while its ordinary working frames sit
#: between roughly 10% and 60%.
SEVERE_YIELD_FRACTION: Final = 0.05

#: A frame is CALLED collapsing when it has kept at most this share of the
#: recent baseline - half of what the last few frames were returning.
COLLAPSE_RATIO: Final = 0.5

#: ...and stops being called collapsing only once it is back to this share of
#: that baseline. The gap between the two is deliberate: a yield hovering
#: either side of one threshold would otherwise flip the warning on and off
#: every frame, which reads as a flickering panel rather than as a warning.
RECOVERY_RATIO: Final = 0.75

#: A relative drop is only worth naming if it is also this many percentage
#: points of the whole frame. Without it, 0.8% falling to 0.3% would be called
#: a collapse; it is noise near the floor, and the severe verdict already
#: covers everything down there.
COLLAPSE_MIN_DROP: Final = 0.05

#: How many recent frames the baseline is computed over. Fixed and small: the
#: tracker holds at most this many floats no matter how long a scan runs, and a
#: window this size spans a couple of seconds at the observed frame rate, which
#: is the horizon over which an operator can still act.
HISTORY_FRAMES: Final = 15

#: Named reasons. A front end shows the sentence; a report keeps the name, so a
#: reader can tell WHY a verdict was reached without parsing English.
REASON_NO_SAMPLES: Final = "no_samples_returned"
REASON_BELOW_SEVERE: Final = "below_severe_yield"
REASON_FELL_BELOW_BASELINE: Final = "fell_below_recent_baseline"
REASON_HOLDING_BELOW_BASELINE: Final = "holding_below_recent_baseline"
REASON_WITHIN_BASELINE: Final = "within_recent_baseline"
REASON_NO_BASELINE: Final = "no_baseline_yet"

#: Said by every warning verdict, once, in the words the physical evidence
#: supports: the missing area is the sensor reporting nothing, and the actions
#: offered are the ones the person holding the scanner can actually take.
NO_RETURN_SENTENCE: Final = (
    "The blank areas are pixels where the sensor returned no depth at all, not surface at some "
    "distance."
)
RECOVERY_ADVICE: Final = (
    "Slow down, and move or re-aim so more of the surface faces the scanner squarely - or go "
    "back to where the surface was still showing."
)


@dataclass(frozen=True, slots=True)
class ReturnHealth:
    """One frame's return-health verdict, with the evidence it was reached from.

    ``baseline_fraction`` is ``None`` when there was no recent history to
    compare against - the first frame of a run, and every frame after a
    :meth:`ReturnHealthTracker.reset`. It is never silently filled in with the
    current frame's own yield, which would make every first frame look stable.

    Advisory only. Nothing in this repository branches on this object; it is
    shown and it is written into reports, and that is all.
    """

    verdict: str
    advice: str
    reason: str
    yield_fraction: float
    baseline_fraction: float | None

    @property
    def is_warning(self) -> bool:
        """Whether a front end must show this loudly rather than quietly."""
        return self.verdict in WARNING_VERDICTS

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict,
            "reason": self.reason,
            "advice": self.advice,
            "yield_fraction": self.yield_fraction,
            "baseline_fraction": self.baseline_fraction,
            "warning": self.is_warning,
            "note": (
                "advisory only: this accepts or refuses no frame, stops no capture and changes "
                "no reconstruction threshold. No cause for the missing returns is claimed"
            ),
        }


def _ensure_yield(value: float) -> float:
    """Return ``value`` if it is a usable yield fraction, else raise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidYield(f"a yield fraction must be a real number; got {value!r}")
    number = float(value)
    if not (0.0 <= number <= 1.0):
        raise InvalidYield(
            "a yield fraction is the returned share of one frame and lies in [0, 1]; got "
            f"{number!r}"
        )
    return number


def return_health(
    yield_fraction: float,
    *,
    baseline_fraction: float | None = None,
    was_collapsing: bool = False,
) -> ReturnHealth:
    """The verdict and advice for one frame. Pure: same inputs, same sentence.

    ``baseline_fraction`` is what the recent frames were returning (see
    :class:`ReturnHealthTracker`, which is what normally supplies it); ``None``
    means there is no recent evidence and no collapse can honestly be claimed.
    ``was_collapsing`` carries the previous frame's collapse state so the
    verdict can be held until the yield is properly back - see
    :data:`RECOVERY_RATIO`.

    Precedence is worst-first and absolute beats relative: a frame with nothing
    in it is :data:`NO_RETURNS` even if the last few frames were empty too, and
    a frame under :data:`SEVERE_YIELD_FRACTION` is :data:`SEVERELY_SPARSE`
    whether or not it got there suddenly.
    """
    current = _ensure_yield(yield_fraction)
    baseline = None if baseline_fraction is None else _ensure_yield(baseline_fraction)

    if current <= 0.0:
        return ReturnHealth(
            verdict=NO_RETURNS,
            reason=REASON_NO_SAMPLES,
            advice=(
                f"Nothing in this frame returned depth. {NO_RETURN_SENTENCE} {RECOVERY_ADVICE}"
            ),
            yield_fraction=current,
            baseline_fraction=baseline,
        )
    if current < SEVERE_YIELD_FRACTION:
        return ReturnHealth(
            verdict=SEVERELY_SPARSE,
            reason=REASON_BELOW_SEVERE,
            advice=(
                f"Only {current:.1%} of the frame returned depth, under the "
                f"{SEVERE_YIELD_FRACTION:.0%} there needs to be for this to be worth recording. "
                f"{NO_RETURN_SENTENCE} {RECOVERY_ADVICE}"
            ),
            yield_fraction=current,
            baseline_fraction=baseline,
        )
    if baseline is not None and baseline > 0.0:
        entering = current <= baseline * COLLAPSE_RATIO
        holding = was_collapsing and current < baseline * RECOVERY_RATIO
        if (entering or holding) and (baseline - current) >= COLLAPSE_MIN_DROP:
            return ReturnHealth(
                verdict=COLLAPSING,
                reason=REASON_FELL_BELOW_BASELINE if entering else REASON_HOLDING_BELOW_BASELINE,
                advice=(
                    f"Returns have fallen to {current:.1%} of the frame from {baseline:.1%} over "
                    f"the last few frames. {NO_RETURN_SENTENCE} {RECOVERY_ADVICE}"
                ),
                yield_fraction=current,
                baseline_fraction=baseline,
            )
    return ReturnHealth(
        verdict=USABLE,
        reason=REASON_WITHIN_BASELINE if baseline is not None else REASON_NO_BASELINE,
        advice=f"{current:.1%} of the frame is returning depth.",
        yield_fraction=current,
        baseline_fraction=baseline,
    )


class ReturnHealthTracker:
    """The bounded recent history one preview or capture run is judged against.

    One instance belongs to one RUN. :meth:`reset` puts it back to its starting
    state, and every front end calls it when a preview or capture begins and
    when one is torn down, so a scan cannot be judged against the collapse of
    the scan before it - which is exactly how an old capture's evidence would
    otherwise poison a fresh one's first seconds.

    Memory is fixed: a :class:`collections.deque` of at most ``history`` floats
    plus one boolean, whatever the length of the run. No frame, image or
    per-pixel object is retained here at all.

    The baseline is the MEDIAN of the frames in that window rather than their
    peak or their mean. On the physical capture the per-frame yield swings
    widely between neighbouring frames while the scan is going normally, and a
    peak baseline would call half of those ordinary frames a collapse.
    """

    __slots__ = ("_capacity", "_history", "_was_collapsing")

    def __init__(self, *, history: int = HISTORY_FRAMES) -> None:
        if not isinstance(history, int) or isinstance(history, bool) or history < 1:
            raise ReturnHealthError(
                f"the history window must be a positive integer number of frames; got {history!r}"
            )
        self._capacity = history
        self._history: deque[float] = deque(maxlen=history)
        self._was_collapsing = False

    @property
    def frames(self) -> int:
        """How many frames are currently in the window. Never exceeds its bound."""
        return len(self._history)

    @property
    def capacity(self) -> int:
        """The fixed number of frames this tracker will ever hold."""
        return self._capacity

    @property
    def baseline(self) -> float | None:
        """The current comparison baseline, or ``None`` before any frame."""
        if not self._history:
            return None
        ordered = sorted(self._history)
        return ordered[len(ordered) // 2]

    def reset(self) -> None:
        """Forget every frame. The next one is judged with no baseline at all."""
        self._history.clear()
        self._was_collapsing = False

    def update(self, yield_fraction: float) -> ReturnHealth:
        """Judge one frame against the window, then add it to the window.

        In that order: a frame is compared with the frames BEFORE it, never
        with a window it is already part of, which would drag the baseline
        toward the very drop being measured.
        """
        current = _ensure_yield(yield_fraction)
        health = return_health(
            current, baseline_fraction=self.baseline, was_collapsing=self._was_collapsing
        )
        self._was_collapsing = health.verdict == COLLAPSING
        self._history.append(current)
        return health


# --- what the front ends say ------------------------------------------------


def spatial_summary(spatial: SpatialReturns) -> str:
    """One line naming where this frame's returns are, in pixels and shares.

    Shared by the Qt appliance, the Tk window and the HTTP page so that all
    three say the same thing about the same frame. Names the emptiest quadrant
    and the footprint of the returns, because "12% of the frame returned" reads
    identically for a thin uniform fade and for an object that has slid off one
    side of the sensor, and those two call for different corrections.
    """
    footprint = spatial.footprint
    if footprint is None:
        return (
            f"no returns anywhere in the frame ({spatial.samples} samples, "
            f"{spatial.width}x{spatial.height})"
        )
    emptiest = spatial.emptiest_quadrant
    return (
        f"returns {spatial.returned} of {spatial.samples} ({spatial.yield_fraction:.1%})   "
        f"emptiest quadrant {emptiest.name.replace('_', ' ')} at "
        f"{emptiest.no_return_fraction:.1%} no-return   "
        f"returns span columns {footprint.left}-{footprint.right} of {spatial.width}, "
        f"rows {footprint.top}-{footprint.bottom} of {spatial.height}"
    )


def health_summary(health: ReturnHealth, spatial: SpatialReturns) -> str:
    """The two lines every depth front end shows: the verdict, then where.

    The verdict is upper-cased because it is the part that has to be readable
    from across a room while the operator's hands are on the scanner.
    """
    return f"{health.verdict.upper()}: {health.advice}\n{spatial_summary(spatial)}"
