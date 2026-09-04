# Frame spacing, decided while the capture runs

Status: implemented, off by default, and tested against the fake driver and the
real capture path. The controller's own arithmetic is measured; what a held
spacing does to a physical scan is not, and the last section says so.

## What problem it solves

Decimation — "use 1 frame in every N" — is the one capture setting whose right
value is not a property of the scanner, the object, or what the operator wants.
It is a property of how fast the scanner happens to be moving *at that moment*.
It is currently a number typed in before a capture starts and held for its whole
length, and both ends of that are wrong:

| Set for | What it costs on the other kind of sweep |
| --- | --- |
| a brisk sweep | frames a fraction of the noise floor apart, each measuring noise rather than motion, and each one another step for an error to accumulate over |
| a slow sweep | frames that no longer overlap where alignment starts looking, and a capture that was working stops |

Neither is hypothetical. The drift table on `pipeline.TRAVEL_MIN_MULTIPLE` has
the *slowest* row among its worst: 31.2° of drift per lap at 1.7 travel, against
0.5° at 2.5. And `docs/tracking-loss.md` is written around a capture that
stopped tracking because the view moved too far between frames.

The travel readout already tells the operator which way they are wrong, several
times a second. It then asks them to fix it with their arms, while carrying a
scanner around an object.

## What it is not

**It is not a fix for tracking loss, and it must not be described as one.** The
whole range it can move within is the operator's own setting down to 1, so
everything it can do about a capture that is moving too fast can also be had by
setting the control to 1 and leaving it there. On this scanner that is a factor
of three from the default — against a hand-carried speed that overshoots the
band by considerably more.

What it adds is the other end, which nothing else addresses: holding the
measured travel inside the band the reconstruction is measured to work in, on a
capture whose speed varies. Which is every hand-carried one.

## What it decides on

Two inputs, both of which the capture already produced per frame before this
existed.

**An accepted frame's travel** (`CadenceController.placed`) — the median
distance that frame's own surface moved, judged against the same band
`pipeline.travel_verdict` shows on screen. Not a second opinion about capture
speed; the same one. A controller correcting towards a different band from the
one the operator is watching would be answering a question nobody asked.

**One alignment refusal** (`CadenceController.refused`) — because a frame that
could not be placed carries *no travel at all*. This is the only signal that
arrives before the measurement that would have explained it, and waiting for the
next frame that does have one is waiting through the three-failure streak that
ends the capture.

## What it does with them

The band is a **deadband**, and that is the whole of why this is stable: a
controller chasing a single target value corrects on every frame, and every
correction changes the very quantity it is measuring. Inside the band there is
nothing to fix.

Outside it, the two directions are deliberately not symmetric:

| Signal | Response | Why that response |
| --- | --- | --- |
| travel over the band | one frame is enough, and the correction is proportional — the spacing that would have produced the target travel is `current × target / travel` | an increment a frame, at this scanner's rate, is seconds of scanning at a speed already known to be too fast, and those are the seconds a capture is lost in |
| travel under the band | three in a row, then a single step | coarsening loses frames that cannot be recovered afterwards, so it is earned rather than computed |
| a refused frame | halve the spacing, at once | it reaches the finest setting within the three consecutive failures that end a capture, from anywhere in the range |

**The asymmetry is the safety property.** Keeping more frames than necessary
costs disk and CPU. Keeping fewer loses data. So going finer is immediate and
going coarser has to be earned — and the same asymmetry is why treating a
refusal as evidence of speed is acceptable even though it is not: a hand across
the view refuses a frame too, and this answers that by keeping more frames for a
while, which costs nothing that matters.

The target is the **geometric** mean of the band's edges, not the arithmetic
one. The correction is multiplicative, so the geometric mean is the point
equidistant from both edges in the ratio the controller actually works in; the
arithmetic midpoint sits closer to the fast edge in ratio terms, and that is the
edge where being wrong loses the capture rather than merely blurring it.

It is the middle of the band rather than the best row of that drift table on
purpose. The table's own text says it is "guidance for turning a knob, not a
calibration" — four points from two captures, one of them not a clean
comparison. Aiming at its best row would be treating it as a calibration.

## How the stream expresses a spacing that changes

`stream_frames` selected with `index % decimate == 0`, and that cannot express a
varying cadence. The modulo is a **phase** function of the delivered index:
change the modulus half way through and the phase jumps — two keeps back to
back, or a skip run longer than either the old value or the new one. Neither is
a spacing anybody asked for.

It is a **countdown** now: this many deliveries between keeps. That has one
meaning under change, and it is the meaning the name always had. For a fixed
`decimate` it selects the identical deliveries the modulo did — 0, 3, 6 at three
— which is what leaves every existing capture untouched.

Two properties are worth stating because they are what a varying cadence could
plausibly get wrong:

- **A change takes effect at the next kept frame, not retroactively.** The
  countdown is refilled from the value in force *at the keep*, so a spacing that
  widened describes the frames after the next one rather than reaching back into
  a gap already being measured under the old value.
- **Except a change to a finer spacing, which shortens the gap in progress.** At
  the widest setting, waiting that gap out is nine more deliveries — about two
  seconds at the measured rate — and a controller only ever asks for a finer
  spacing because the current one was already too coarse. Clamping the countdown
  can only bring a keep *forward*.

A short delivery still consumes its slot rather than promoting the next complete
one into it, exactly as before: selection stays a property of arrival order and
not of which frames happened to be usable.

**The controller is never allowed to end the run.** It is asked only after the
buffer has been re-queued, and if it raises or answers with something that is
not a usable spacing, the previous value stands and
`StreamResult.decimate_refusals` counts it. Raising out of the streaming loop is
the one failure this module must not have — a callback that raises between
`DQBUF` and `QBUF` leaves userspace owning a buffer, and this hardware answers
that by re-enumerating, which to a caller looks exactly like the scanner being
unplugged.

## What the record says

A single `decimate` was the whole record before a spacing could vary, and left
alone it would describe a spacing that was never true for the run. So
`StreamResult` now carries the span as well: `decimate` is where the run
*started*, `decimate_lowest` and `decimate_highest` are what it actually used,
and `decimate_varied` answers "was this capture's frame spacing constant" — which
a final value alone cannot, because a cadence that went to 1 and back to 3 ends
where it started.

The capture also says it once, at the end, in the operator's own words: how many
times the spacing moved, the range it ran between, and how many of those moves
were frames the tracker could not place. That last number is the scan telling
them they were going too fast.

Captures that were not asked to hold the spacing are completely unaffected: the
control is off by default, the stream is handed the same plain integer it always
was, and the summary line does not appear.

## Why it is on by default

It was off at first: a capture whose frame spacing is a number the operator
typed is a capture they can reason about afterwards, and one that varied has a
record that must be read to know what it did. Two hand-held captures on
2026-09-04 settled the trade the other way (`docs/tracking.md`, "Two hand-held
captures"). Both ran at the fixed default of one frame in three, 0.67 s apart at
the scanner's 4.5 depth frames a second, and a hand moving at 2-5 cm/s put
6-36 mm of camera motion between the frames the tracker saw. The second capture
lost tracking at a step where the two frames either side shared under a third
of their content and could not be aligned by any means, while the same motion
seen at every frame is 2-12 mm, inside the tracker's reach. Held spacing goes
to every frame the moment the measured travel leaves the band, and the record
still says what it did. An operator who wants a fixed number unticks the box.

## What this deliberately does not do

No change to any tracking gate. Nothing here makes a placement more likely to be
accepted; it changes only which deliveries are offered for placement at all.

The one place it touches the motion prior in `docs/tracking.md` is worth naming.
That prior is the step between the last two accepted frames, so immediately
after a spacing change one step spans a different amount of real time from the
next — the prediction is scaled wrong for exactly one frame, by the ratio of the
change, and is right again after one acceptance at the new spacing. It costs at
most one wasted solve: the identity-seeded attempt runs first regardless, the
prior is only ever a fallback behind it, and no gate is different for a frame
seeded from a stale prediction than for any other. It is not corrected for,
because correcting it would mean the tracker knowing about the cadence, and the
two are otherwise completely independent.

No prediction. Every measurement it acts on describes a frame that has *already*
been placed, so the cadence follows the operator's speed rather than
anticipating it. A scanner that speeds up sharply is corrected one kept frame
after it did — and that frame is exactly the one that may already have been
refused, which is why a refusal is an input in its own right rather than a thing
to be inferred from the travel that never arrived.

No per-capture measurement of its own core assumption. Travel is taken to be
proportional to how many deliveries a kept frame is worth, which is only true
while the operator's speed is roughly steady across a correction.

**No record on disk, and this is the one gap worth naming.** The span reaches
the run's `StreamResult`, the log and the operator; it does not reach the
capture folder, because there is nowhere in a capture folder for it to go. A
capture holds raw frames plus the `bridge.json` and `tracking-loss.json` its
segments need, and nothing that describes the capture as a whole. Adding one
means a new record format and a reader in `capture_sequence` strict enough to
refuse a malformed one, which is a larger piece of work than this. Nothing
downstream is wrong without it — a reconstruction fits adjacent pairs on the
assumption that they are a small step apart, which is precisely what the
controller maintains — but a folder whose frames were unevenly spaced does not
say so, and a reader a year from now cannot tell.

And above all: **this holds a capture inside the band the reconstruction was
measured to work in. It does not establish that a capture inside that band
reconstructs well.** The band is four points from two captures. It is guidance,
and this automates following it — not evidence that following it is enough.
