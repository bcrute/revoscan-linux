# Capture checklist: what to put in front of the scanner, and what to measure

Status: a work order, not a result. Everything below is the physical half of M4
(metric validation) and M6 (registration) — the part no amount of software can
do. The software half is built and tested; these captures are what close the
milestones.

Budget roughly 30 minutes. You need: the scanner, your calipers, a turntable,
one **sphere**, and one **asymmetric object**. Those two must be different
objects, for a reason given below that is easy to get wrong.

## Why this is needed at all

No distance this scanner has ever produced has been metrically validated. The
depth scale is a float32 that decodes to `1/200`, and what it measures is
unknown. Across the nine committed frames the median non-zero depth code is
23398, which at that scale implies a distance of **117 unknown units**.

The Revopoint MINI is documented as working at roughly 100–200 mm. So 117 is
*consistent with* millimetres — and consistency is not proof. It is exactly the
kind of coincidence that has fooled this project before (see "The trap" in
[depth-to-pointcloud.md](depth-to-pointcloud.md)). One caliper measurement
settles it. Until then the code keeps saying `UNVALIDATED_UNIT`, and it is right
to.

## The working envelope, measured

From the nine committed frames and the factory intrinsics:

| Quantity | Value |
| --- | --- |
| Depth codes actually returned | 1st percentile 16817, median 23398, 99th 58002 |
| Implied distance (unit unknown) | ~84 to ~290, median ~117 |
| Horizontal field of view | −2.70° to +33.58° (span 36.3°) |
| Vertical field of view | −12.11° to +12.92° (span 25.0°) |
| Visible width at median distance | ~83 units |
| Visible height at median distance | ~52 units |

**Target size:** something **40–70 units across** fills a useful fraction of the
frame at the median working distance. If the unit turns out to be millimetres,
that is a 40–70 mm object — a golf ball, a small bearing, a 3D-printed test
cube. Much smaller and you get too few returns; much larger and it will not fit.

**Note the field of view is wildly asymmetric.** The principal point `cx` sits
6.6% across a 640-wide frame — near the left edge, not the centre. That is
either a real property of this sensor or a wrong calibration value, and
distinguishing those is capture 2 below.

## Capture 1 — scale, with a sphere

**Object:** a sphere. A ball bearing, a marble, a smooth ball. Matte is better
than shiny; a mirror-finish bearing may return almost nothing.

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
   sphere sits at 117 or 250. Shear deforms shape by a constant proportion, so
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
2. Place it at roughly the median working distance (~117 units; if that is mm,
   about 12 cm) against a plain, non-reflective background.
3. Capture. Then capture twice more without touching anything.
4. Move it nearer (~85 units) and repeat. Move it further (~250 units) and
   repeat.
5. **Then put the sphere on the turntable and sweep it, exactly as in capture 3.**
   This step is not optional and the reason is measured, not theoretical — see
   below.

### Sweep the sphere too, or the shear check cannot work

A single camera view sees less than a hemisphere of a sphere. Measured against
both the validation harness's own generator and independently rendered frames,
that is not enough coverage for the shear indicator to fire **at all**: its
correlation saturates near 0.25 against a 0.35 detection threshold, so even a
principal-point error of 278 px — the largest this unit could have — reads as
*not detected*. Full detail in
[shear-detection-limits.md](shear-detection-limits.md).

Coverage, not fault magnitude, is what the indicator is limited by:

| Coverage of the sphere | Shear correctly detected? |
| --- | --- |
| whole sphere | yes (correlation 0.997) |
| perfect hemisphere | yes (0.544) |
| what one camera view actually sees | **no** (0.25, below threshold) |

A turntable sweep of the sphere, fused, gives near-complete coverage — which is
the regime where the indicator works well. So the sweep is what makes the
`cx` half of this test possible. The static captures in steps 2–4 still matter:
they are what test **scale** at known distances, and scale does not need the
coverage that shear detection does.

Until that fused sweep exists, treat a quiet shear indicator as **no
information**. It must not be read as evidence that `cx` is correct.

**Record:** the five caliper readings, the instrument, and roughly where the
sphere sat for each set.

**What this establishes:** whether the depth scale is metrically correct at each
tested distance, and whether the sphere comes back spherical. **What it does
not:** anything about distances outside the tested range, or about lens
distortion.

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

**Object: NOT the sphere.** This is the easiest mistake to make here. A sphere
looks identical from every angle, so it carries no information about how far it
rotated — registration would report a perfect, meaningless alignment. Use
something clearly **asymmetric**: a small figurine, a bracket, a crumpled shape,
a 3D print with distinct features. It must look different from every side.

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

## Capture 4 — fusion and mesh, free

No extra work. Capture 3's frames feed fusion and meshing directly. The only
thing worth adding is one sweep of an object with a **known simple dimension**
(a cube with caliper-measured edges), so the final mesh can be checked against a
real number end to end rather than only at the point-cloud stage.

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
- A single validated distance is not a validated *range*. That is why capture 1
  asks for three distances rather than one.
