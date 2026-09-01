# Capture checklist: what to put in front of the scanner, and what to measure

Status: partly a result now. The M4 scale question has been answered by caliper
against real captures — the unit is millimetres — and the registration work in
M6 has been measured on a real two-lap sweep. What remains here is still the
physical half that no amount of software can do: the accuracy of that scale
across the working range, and the captures that would confirm on real scans what
has so far only been shown on real geometry with a simulated flip.

Budget roughly 30 minutes. You need: the scanner, your calipers, a turntable,
one **sphere**, and one **asymmetric object**. Those two must be different
objects, for a reason given below that is easy to get wrong.

## Why this is needed at all

The depth scale is a float32 that decodes to `1/200`, and for most of this
project's life what it measured was unknown. Across the nine earliest committed
frames the median non-zero depth code is 23398, which at that scale implies a
distance of 117 of *something*.

**That question is now answered: the unit is millimetres.** A can measuring
58.0 mm across the flange-free middle by caliper was scanned at roughly 122.7 mm
standoff and came back 57.705 mm — an error of −0.51%. See
[metric-validation.md](metric-validation.md) for the full sweep.

Two things this does NOT mean, both worth keeping in mind while capturing:

**The scale is identified, not trusted.** Across distances the measured error
ranges from −0.49% to −2.26%, and no correction model explains that spread. A
number in millimetres is not yet a number you should quote to three decimal
places.

**The code may still say `UNVALIDATED_UNIT`.** That flag describes what has been
proven *in software*, and the reader should check the source rather than assume
this document and the code agree. The measurement above is what a caliper
showed; the flag is what the pipeline asserts, and those are different claims.

The reason this distinction gets its own paragraph is that a plausible-looking
number has fooled this project before — see "The trap" in
[depth-to-pointcloud.md](depth-to-pointcloud.md). 117 was *consistent with*
millimetres long before it was *shown* to be millimetres, and consistency was
never proof.

## The working envelope, measured

From the nine committed frames and the factory intrinsics, with the unit now
known to be millimetres (see above):

| Quantity | Value |
| --- | --- |
| Depth codes actually returned | 1st percentile 16817, median 23398, 99th 58002 |
| Implied distance | ~84 to ~290 mm, median ~117 mm |
| Horizontal field of view | −2.70° to +33.58° (span 36.3°) |
| Vertical field of view | −12.11° to +12.92° (span 25.0°) |
| Visible width at median distance | ~83 mm |
| Visible height at median distance | ~52 mm |

**Target size:** something **40–70 mm across** fills a useful fraction of the
frame at the median working distance — a golf ball, a small bearing, a
3D-printed test cube. Much smaller and you get too few returns; much larger
and it will not fit.

**Note the field of view is wildly asymmetric.** The principal point `cx` sits
6.6% across a 640-wide frame — near the left edge, not the centre. That is
either a real property of this sensor or a wrong calibration value, and
distinguishing those is capture 2 below.

## Working distance and standoff, measured

Two real findings on the truck bed itself bracket how far from the scanner
the object should sit — one about moving closer, one a hard limit on going
further. Neither is specific to a turntable; both are about the distance
between the scanner and the object, however that distance is achieved.

### Closer is better, within the envelope

Two real capture sets, differing only in standoff:

| Standoff (median depth) | Per-pair alignment rmse | Frame overlap | Self-agreement at matched capture-order gaps |
| --- | --- | --- | --- |
| ~140 mm | 0.65 | 100% | 1.84 |
| ~250 mm | 0.723 | 94.8% | 3.07 |

Self-agreement improved **1.7x** at the closer standoff, and every column
moved the same direction — this is not a trade-off with a cost hidden in it.
Work as close as the object and its motion allow.

### A hard ceiling on standoff, and it is closer than it looks

Depth arrives as a 16-bit code at 0.005 mm per count, so the largest distance
this sensor can represent at all is `65535 x 0.005 = 327.675 mm`. Past that
the code has nowhere to go. The deepest returns seen across committed frames
already sit near **290 mm**.

This matters most if the scanner ends up mounted rather than handheld,
looking down at an angle. At 45 degrees the slant range to the object is
`height x sqrt(2)`:

| Mount height | Slant range at 45° | Verdict |
| --- | --- | --- |
| 12 inches | 431 mm | **unrepresentable** — past the sensor's ceiling |
| 8 inches | 290 mm | the practical maximum, with no margin |
| 6 inches | 212 mm | comfortably inside proven territory |

Prefer 6 inches of mount height (or the equivalent handheld standoff) over 8,
and do not mount higher than 8 inches at 45 degrees at all.

## Capture 1 — scale, with a sphere or an upright cylinder

**Object:** a sphere *or an upright cylinder*. A cylinder is the practical
choice for most people, and it works — but only in one orientation. See
[Using a cylinder instead](#using-a-cylinder-instead) below, which is measured
rather than assumed.

Matte is better than shiny; a mirror-finish bearing or a bare aluminium can may
return almost nothing.

**Why a sphere is the best single target:** its diameter is one number you can
measure with calipers from any orientation, so there is no alignment error in
the ground truth. And a sphere is the shape that *also* catches a wrong `cx`:
a principal-point error acts as a shear (`x += 0.3089·z` at the current
numbers), and shear turns a sphere into a tilted ellipsoid. One object tests
both scale and distortion.

**A flat target cannot do this.** Shear maps planes to planes, so a flat plate
would pass a plane-fit check cleanly while `cx` was badly wrong. Do not use a
flat target for this.

### The diameter alone will lie to you

Simulated directly, with `k = 0.3089` (the shear implied if the true `cx` were
the frame centre), on synthetic spheres of known radius:

| Sphere | Fitted radius | RMS residual |
| --- | --- | --- |
| clean, r = 20 | 20.0000 | 9.0e-15 |
| sheared, r = 20 | 20.2577 | **1.5938** (8.0% of radius) |
| sheared, r = 10 | 10.1340 | 0.7925 (7.9% of radius) |
| sheared, r = 20 at z = 250 | 20.2385 | 1.5810 (7.9% of radius) |

Two things fall out of that table, and both matter:

1. **The distortion is independent of distance.** 7.9–8.0% of radius whether the
   sphere sits at 117 mm or 250 mm. Shear deforms shape by a constant proportion, so
   you cannot escape it by moving the target, and you do not need to hunt for a
   distance where it shows up.
2. **The fitted radius is only 1.3% too large.** This is the trap. Measure the
   diameter alone and you see a small discrepancy that looks exactly like a
   modest scale error — and the natural response is to "correct" the depth scale
   to close it. That would bake a principal-point fault into the calibration as
   a scale fudge, and every subsequent measurement would carry it.

**So the residual is the tell, not the radius.** A sheared sphere fits as a
slightly-too-large sphere *with a large residual*. Report both, always. A
diameter that agrees to 1% with a residual of 8% of radius is not a pass; it is
evidence that `cx` is wrong.

**Do:**
1. Measure the sphere's diameter with the calipers. Take **five** readings at
   different orientations. Record every reading, not just the mean — the spread
   is your ground-truth uncertainty, and a verdict without it is not a verdict.
2. Place it at roughly the median working distance (~117 mm, so about 12 cm)
   against a plain, non-reflective background.
3. Capture. Then capture twice more without touching anything.
4. Move it nearer (~85 mm) and repeat. Move it further (~250 mm) and
   repeat.
5. **Then put the sphere on the turntable and sweep it, exactly as in capture 3.**
   This step is not optional and the reason is measured, not theoretical — see
   below.

### Sweep the sphere too — it makes the shear check quantitative

A single camera view sees less than a hemisphere of a sphere, and coverage is
what the shear indicator is limited by. This was briefly a hard failure: the
indicator could not fire at all on single-view data, so even a 278 px
principal-point error read as "not detected". **That is fixed** — it now detects
an error as small as 1 pixel from a single view, and stays quiet on a clean
sphere.

What remains is accuracy rather than detection. From one view the reported
`shear_coefficient` reads about 65–70% of the true value, so treat it as a
**lower bound**. Fused multi-view coverage removes that bias as well.

So sweeping the sphere is no longer required to *see* a `cx` error — it is what
makes the number trustworthy. It costs nothing once the turntable is already set
up for capture 3. The full history and measurements are in
[shear-detection-limits.md](shear-detection-limits.md).

The static captures in steps 2–4 still matter regardless: they are what test
**scale** at known distances, which does not need that coverage.

## Using a cylinder instead

Most people do not own a sphere. A cylinder works, and its diameter is just as
good a caliper measurement — but **orientation decides whether it can detect a
wrong principal point at all**, and the difference is total rather than
marginal.

Rendering a cylinder and deprojecting it with `cx` wrong by 278 px, then
measuring how far the points move off the true surface:

| Cylinder orientation | Points move by |
| --- | --- |
| **standing upright** (axis vertical) | **161.7% of its radius** |
| lying down, across the view | 0.0025 — *identical to unsheared* |
| end-on, circular face toward the camera | 0.0000 — *identical to unsheared* |

**So stand it up.** Lying down, a cylinder is exactly as blind to this fault as
a flat plate, because the shear slides it along its own length and the shape
does not change at all. That is not a weak signal; it is no signal.

Good candidates, in rough order of preference:

- **A pill bottle, cardboard tube, candle, or plastic jar** — matte, right size,
  stands up on its own.
- **A soup or drink can** — good size (~66 mm across) and a very clean
  diameter to measure, but bare aluminium is shiny; a strip of masking tape or
  a dusting of chalk fixes that and is standard practice.
- **A dowel, socket, or large bolt** — fine if wide enough to fill a useful part
  of the frame. Below about 40 mm across you get too few returns.
- **Avoid anything squat** (a coin, a puck, a jar lid): a cylinder shorter than
  it is wide makes the axis hard to estimate from the points alone.

Measure the **diameter** with calipers, five readings at different heights and
rotations, and record all five — the spread is your uncertainty.

## Capture 2 — the principal point, by translation

This is the experiment that decides whether `cx = 42.44` is real.

A wrong `cx` shears `x` in proportion to `z`. So an object moved **straight away
from the camera**, with no sideways motion at all, will appear to *slide
sideways* by `k·Δz` if `cx` is wrong — and will not move at all if `cx` is
right. It is a null test, which is the most trustworthy kind.

**Do:**
1. Put the sphere at the near distance. Capture.
2. Move it directly away from the camera — along the viewing direction, as
   straight as you can manage — by a distance you measure with the calipers
   (a caliper-measured spacer block behind it is ideal).
3. Capture. Repeat for two or three spacer thicknesses.

**Record:** each spacer thickness.

Sideways precision does not need to be perfect: the analysis compares the
*change* in fitted sphere centre against the *change* in depth, and a wrong `cx`
produces a systematic drift that random hand placement will not imitate across
three or four steps.

## Capture 3 — the turntable sweep, for registration

**Object: NOT the symmetric sphere/cylinder scale target above.** This is the
easiest mistake to make here. A sphere looks identical from every angle. An
upright cylinder standing on the turntable's own axis *also* looks identical
from every angle, for the same reason — it is
rotationally symmetric about the very axis you are turning it around. Either one
carries no information about how far it rotated, and registration would report a
perfect, meaningless alignment. (Re-confirmed synthetically: a cylinder
centred on the rotation axis returned exactly 0.0° of recovered rotation.)

Use something clearly **asymmetric**: a small figurine, a bracket, a crumpled
shape, a 3D print with distinct features. It must look different from every side.
The purpose-built
[`reconstruction-target-v1`](../reference/targets/reconstruction_v1/) is now the
preferred object: its ramp, pocket, bump, unequal steps and offset rails were
laid out to remove the rotational ambiguity, and its CAD supplies the dense
surface truth the improvised objects do not.

Being asymmetric to the eye is not quite enough — it has to be asymmetric *about
the rotation axis in use*. A rectangular block whose width and depth are equal
is symmetric under a quarter turn and would defeat a 4-frame sweep, which is a
mistake made and caught while writing the tests for this.

**If the only suitable object you have is the cylinder**, place it clearly
**off-centre** on the turntable. It then orbits the axis instead of spinning on
it, and its position changes from frame to frame even though its shape does
not.

**Do:**
1. Centre the object on the turntable, roughly at the median working distance.
2. Capture one frame, note the turntable angle.
3. Rotate by a **fixed step** and capture again. Repeat for a **full 360°**.
   - **24 frames at 15°** is the target. 36 at 10° is better if the turntable
     indexes that finely.
   - Consecutive frames must **overlap substantially** — that overlap is what
     registration aligns. Steps larger than ~20° tend to fail.
4. Completing the full circle is essential and not optional: the last frame
   landing back on the first is the entire accuracy metric.

**Record:** the angular step, the frame order, and whether the sweep closed the
full circle.

**What this gives you:** loop-closure error — compose all 24 transforms and
compare against the identity. Residual rotation and residual translation are
reported separately. It is a self-contained accuracy measurement that needs no
external ground truth at all, which is why the turntable is worth more here than
its cost.

**Optional but valuable:** a second sweep with the object tilted, so the top
surfaces are seen. A single level sweep never observes the top of an object, and
the fused result will have a hole there.

### Capture density is a quality lever, measured

Loop-closure error on a real two-lap capture, varying only the step between
consecutive frames:

| Step between frames | Drift per lap |
| --- | --- |
| 7.8° | 65° |
| 3.9° | 23° |

Halving the step cut the error **2.8x** — better than proportional. Error
accumulates roughly linearly with laps: two laps measured about 2.2x the
error of one lap. This is not turntable-specific: it is the size of the jump
between consecutive captures that matters, whether that jump comes from a
turntable's index or from how far the scanner (or the object) moved by hand.
Smaller, more frequent captures beat fewer, larger ones by more than the
frame count alone would predict.

### Move continuously rather than stopping at each step, measured

A continuous capture stream sustained **4.65 fps with zero dropped frames**,
which against a turntable taking about 2 minutes per revolution works out to
roughly **2°/frame** — a finer step than the "24 frames at 15°" target above,
for the same revolution. Large jumps between consecutive frames are where
alignment silently fails; moving steadily — turning the object at a constant
rate, or walking around it at an even pace, rather than stopping and
restarting — avoids ever taking one. This is separate from the 5-second gap
in "Leave at least 5 seconds between captures" below: that gap is about
discrete stop-and-shoot captures. Continuous motion has no stop to gap.

### Capture two full laps, not one, measured

One closed lap proves the trajectory returned to where it started; it gives
the software nothing to check accumulated drift against along the way. A
second lap — or, handheld, deliberately passing back over ground already
covered — revisits the same viewpoints, which the software can detect
geometrically and correct drift against. On a real two-lap capture this took
the held-out closure error from a median of **14.4° to 0.9°**. Without a
revisit there is nothing to correct against, so plan every capture to come
back around to somewhere it has already been, not just to close a single loop.

## Capture 4 — fusion and mesh, free

No extra work. Capture 3's frames feed fusion and meshing directly. When capture
3 uses `reconstruction-target-v1`, the same sweep already carries known
dimensions, predeclared planar regions and a dense CAD reference. Measure the
individual print as its README requires; the nominal STL alone is not a tighter
physical reference than the printer that made it. For an improvised asymmetric
object, still add a sweep of an object with a known simple dimension (a cube
with caliper-measured edges), so the final mesh can be checked against a real
number end to end rather than only at the point-cloud stage.

## If the object must be captured in two orientations

Some objects cannot be captured in one orientation at all — reaching the
underside means physically turning the object over and capturing it again.
Matching that second capture back to the first is harder than it looks, for
one specific structural reason, and it changes how the object should be
oriented for capture.

**A shape missing one face is symmetric under its own half-turn, regardless of
proportions.** An object that rests on one face throughout a capture — a
tray, a bed, anything scanned with its underside against whatever it sits
on — has four side faces that are each a plain rectangle, and a rectangle is
always identical to itself after a 180-degree turn about its own centre.
Stretching the object longer or narrower does not break that symmetry; it is
a property of the shape family, not the aspect ratio. A truck bed is close
enough to this shape to inherit the problem.

**Verified on the user's own truck bed geometry:** matching an
upright scan against a flipped one, the best-scoring candidate after
full-resolution alignment was the WRONG answer — an upside-down twin of the
object, 179 degrees from the truth — beating the correct flip by about 1%.
More alignment compute does not fix this: the two answers are genuinely
equally good fits to the shape alone.

**The practical consequence:** the object is distinguishable from its own
upside-down twin only because of its asymmetric features — wheel wells, the
recess under the cab, and anything else that differs between its two sides.
**Those features have to be captured well in both orientations.** If a wheel
well is only captured facing up in one scan and never seen facing down in
the other, nothing asymmetric connects the two captures, and they cannot be
matched with confidence. When flipping the object over, deliberately
re-cover its distinguishing features from the new orientation — do not just
recapture the featureless flat panels again.

**This is not yet proven end to end.** A way to tell the true flip apart from
its twin — using where the object was resting, not its shape — exists and
recovered the correct answer on this object's real geometry with a
*simulated* flip. It has not yet been run on two real physical scans of a
flipped object. Whatever eventually confirms or replaces it, capturing the
asymmetric features in both orientations is the one thing that makes any such
matching possible at all — that is the capture advice, independent of which
method ends up doing the matching.

## Two hardware faults found while capturing, and how to avoid them

Both were diagnosed during the first real metric session. Neither is obvious
from a single capture, and each wastes a lot of time if you do not know about it.

### Leave at least 5 seconds between captures

Back-to-back captures fail. Measured:

| gap between captures | complete frames |
| --- | --- |
| none | 1 of 3 |
| 1 second | 2 of 5 |
| **2 seconds** | **5 of 5** |
| 3 seconds | 5 of 5 |
| 4 seconds | 5 of 5 |
| 5 seconds | 3 of 3 |

**Two seconds is the threshold.** That matters for a continuously rotating
turntable, where the capture cadence sets the angular resolution: a cycle of
~1.1 s capture plus a 2 s gap is about **3.1 s per frame**, so the turntable
must take roughly `3.1 x frames` seconds per revolution to hit a given step.
For the 24 frames at 15 degrees that registration wants, that is about **75
seconds per revolution**.

Every failure is *exactly* 358402 bytes — 280 of 400 rows — never a random
partial. The kernel flags the buffer `V4L2_BUF_FLAG_ERROR` and the capture path
refuses to call it a frame, so a bad capture is never silently used. But it
looks like flakiness until you notice the truncation is always at the same row.

### Do not put the scanner behind a USB hub

On a Realtek USB3.0 hub the link dropped **28 times**, with the device number
climbing 73 to 79. Moving to a root port took that to **zero disconnects** in
the next run.

The two faults are independent and were initially mistaken for one: after the
hub was removed the short frames continued at the same rate until the 5-second
gap was introduced.

**A control worth knowing:** the same can, unmoved, measured 56.641 through the
failing hub (n=2) and 56.689 direct (n=7, spread 0.155). Link quality did not
bias the measurement — it only cost frames. So data captured through a flaky
link is still trustworthy if the frame came back complete.

## Practical notes

- **Surfaces matter more than you would expect.** Structured light returns
  nothing from dark, shiny, or transparent surfaces. Across the nine committed
  frames, 63–98% of samples per frame are empty. Matte and light-coloured is
  best; a dusting of chalk or matte spray on a shiny target is legitimate and
  standard practice.
- **Do not move the scanner between captures within a set.** Everything assumes
  a fixed camera and a moving object.
- **The emitter stays lit.** Still unsolved, and irrelevant to these captures.
- **Watch the ambient light.** Consistent, indirect lighting; avoid sunlight and
  avoid pointing it at a screen.
- **Zero is not a distance.** A sample of 0 means no return. Empty regions in a
  result are absence of measurement, never observed empty space.

## What to hand back

For each capture set: the raw frames, the caliper readings with the instrument
named, the distances or spacer thicknesses, and for the sweep the angular step
and frame order. The tolerance must be **declared before** the comparison is
run — a tolerance chosen after seeing the answer proves nothing, and the
validation harness records who declared it and when for exactly that reason.

## What will still be open afterwards

- **M5's stated exit criterion is unreachable** and should be rewritten. It asks
  for comparison against SDK-derived output, but the vendor SDK cannot connect
  to this unit at all (its licence authorizer cannot read an `sdk-license` blob
  from the camera). Comparison against caliper ground truth is a strictly better
  criterion and is what these captures deliver.
- Lens distortion is not modelled anywhere. The deprojection is an ideal
  pinhole. These captures will not establish a distortion model, only whether
  one is needed.
- **Resolved: a single validated distance is not a validated
  *range*.** Three standoffs were captured and measured (98, 123 and 244 mm),
  and the depth scale is confirmed as millimetres — see
  [metric-validation.md](metric-validation.md). What that work opened instead:
  a **calibration correction is still not established**. Errors ranged from
  −0.49% to −2.26% across those distances and surfaces, and no model for that
  spread has been found — a constant scale error and a constant
  principal-point shear were both tested and both ruled out by their distance
  signatures.
- **Matching a flipped scan back to its upright pair is not yet proven on
  real captures.** A way to disambiguate the true flip from its
  shape-alone twin exists and worked on this object's real geometry with a
  simulated flip; it has not been tested on two real physical scans of a
  flipped object yet. See "If the object must be captured in two
  orientations" above for what to capture in the meantime.
