# The principal point, settled

Status: **measured on this unit, from captures already on disk.** The factory
`cx = 42.44` is real. It was the single largest unvalidated assumption in the
metric chain and it is no longer one.

## The question

`/data/camparam/Pl.bin` returns `fx = fy = 898.6829833984375`,
`cy = 192.8326873779297`, `cx = 42.43747329711914` for a 640 x 400 frame. A
principal point 277 pixels from the frame's 320 px centre is extraordinary, and
`intrinsics.py` has always flagged it — `principal_point_near_frame_centre`
fails against a 10% tolerance — while correctly declining to decide:

> "This is an observation, not a defect: a factory record for a cropped or
> offset sensor window legitimately fails it, and nothing here adjusts the
> values either way."

`docs/capture-checklist.md` called Capture 2 *"the experiment that decides
whether `cx = 42.44` is real"* and worked out the consequence if it were not: a
shear of `x += 0.3089·z`, which turns a sphere into a tilted ellipsoid and
deforms shape by about 8% of radius regardless of distance. That experiment was
never run: it wanted a sphere, and every capture on this machine is the
reconstruction target or a chassis.

Everything downstream rested on the answer. A shear is invisible to loop
closure — it is a fixed linear map of camera space, so a trajectory closes on
itself perfectly while every reconstructed shape is wrong.

## The test that did not need a sphere

A shear maps planes to planes, so a flat target cannot catch it — the checklist
says so, and it is why a sphere was specified. But a shear **rotates plane
normals**, and the reconstruction target carries known right angles.

Under `x += k·z` the linear map is `M = [[1,0,k],[0,1,0],[0,0,1]]`, and normals
transform by `M⁻ᵀ`, so `n = (nx, ny, nz)` becomes `(nx, ny, nz − k·nx)`. A face
whose normal lies along camera x tilts by `atan(k)` against the top face; a face
whose normal lies along y does not move at all.

Verified on the CAD reference before touching any scan:

| | outer_east | outer_west | outer_north | outer_south |
| --- | --- | --- | --- | --- |
| CAD, as designed | 0.00° | 0.00° | 0.00° | 0.00° |
| CAD, sheared at k = 0.3089 | **17.17°** | **17.17°** | 0.00° | 0.00° |

(deviation of each face-to-`base_top` angle from 90°)

Two properties make this usable where the sphere test was not:

- **It needs no registration.** The shear is fixed in camera space, so it is
  visible on a single frame. No poses, no sweep, no loop closure.
- **It needs no correspondence.** The signature is that near-perpendicular pairs
  *split into two groups*. Registration error moves every pair together; only a
  shear splits them.

## The measurement

Deproject the same 16 frames of `20260901-201341` at a range of candidate `cx`,
extract the dominant planes of each frame independently by RANSAC, and take the
median deviation from 90° of the closest near-perpendicular pair.

| cx | median \|angle − 90°\| | |
| --- | --- | --- |
| 10.0 | 0.88° | |
| **42.4** | **0.64°** | **factory** |
| 80.0 | 0.59° | minimum |
| 120.0 | 0.81° | |
| 160.0 | 0.97° | |
| 200.0 | 1.47° | |
| 240.0 | 2.63° | |
| 280.0 | 2.49° | |
| **320.0** | **3.50°** | **frame centre** |
| 400.0 | 2.97° | |
| 480.0 | 4.03° | |
| 560.0 | 4.06° | |
| 620.0 | 6.43° | |

The curve has one clear basin. **The factory value sits in it and the frame
centre is 5.5× worse.** The 0.05° between the factory value and the sampled
minimum is far inside this measurement's own precision.

## The fine sweep, and a positive control

Repeated on 26 frames with a denser grid, and with `cy` swept the same way while
`cx` was held at the factory value:

| cx | median \|angle − 90°\| | | cy | median \|angle − 90°\| |
| --- | --- | --- | --- | --- |
| 0.0 | 0.67° | | 100.0 | 1.64° |
| 15.0 | 0.42° | | 150.0 | 1.47° |
| 30.0 | 0.47° | | **192.8** | **0.29°** ← factory |
| **42.4** | **0.37°** ← factory | | 230.0 | 1.15° |
| 70.0 | 0.36° ← best | | 270.0 | 2.46° |
| 100.0 | 0.91° | | 320.0 | 3.40° |
| 175.0 | 1.94° | | | |

**`cx`: the best sampled value and the factory value differ by 0.01°.**
Indistinguishable at this precision, on better statistics than the coarse pass —
the residual floor itself fell from 0.64° to 0.37° with more frames.

**`cy` is the control, and it is the more valuable half of this table.** `cy =
192.83` sits 7 px from the frame's 200 px centre and was never in doubt. The
sweep finds it with a **sharp, deep minimum** — 0.29° at the factory value
against 1.15–1.47° just 40 px away, a 4–5× penalty.

That matters because it establishes the method has real discriminating power on
this data. A test that could not locate a parameter known to be right would say
nothing about one suspected of being wrong. It locates `cy` decisively, and it
places `cx` in a basin whose floor is the factory value.

## What this establishes, and what it does not

**Establishes.** The sensor really does present an offset window: `cx = 42.44`
is a property of the device, not a corrupt read. The `x += 0.3089·z` shear
hypothesis is dead — had it been real, deprojecting at `cx = 320` would have
*removed* it and produced the minimum there, and instead that is the worst
sampled value outside the frame edges.

**Does not establish.** That the intrinsics are *correct*, only that `cx` and
`cy` are close to right. `fx` and `fy` were held fixed throughout.

> **Correction, 2026-09-03.** This section claimed the construction is *blind*
> to focal length - "a focal-length error scales the reconstruction without
> rotating any plane normal". **That is false in general, and it cost this
> project the assumption that `fx` was untestable.** Deprojection is
> `X = (u - cx)·Z/fx`, so a wrong `fx` scales x and y but *not* z: an
> anisotropic `diag(s, s, 1)` in **camera** space, under which normals transform
> by `diag(1/s, 1/s, 1)` and angles between them do change.
>
> The claim is true only for an object whose faces are axis-aligned with the
> camera, which is why it survived: checked on the CAD in the CAD's own frame,
> where the target's faces *are* axis-aligned, a 5% `fx` error moves every
> face-to-`base_top` angle by exactly 0.000°. Rotate the same CAD to a general
> orientation - which is the only way it ever sits in front of a camera - and a
> 1% error moves them by 0.377°.
>
> So `fx` is identifiable by exactly this method, and is now measured below. The depth scale is
untouched by this — rigid geometry tests are blind to a global scale error, and
the −0.49% to −2.26% spread across distances is still open. And lens distortion
remains unmodelled and unfetched: `/data/camparam/Distort.bin` has never been
requested, so part of the 0.59° residual may be distortion this test cannot
separate from print tolerance and plane-fit noise.

**The residual is a floor, not an error budget.** 0.59° at the best `cx` is the
combined precision of plane fitting on real returns, the printed target's own
flatness, and any unmodelled distortion. It bounds how sharply this method can
locate a principal point; it is not a claim about the scanner's accuracy.

## `fx`, measured the same way

Swept on 36 frames of `20260902-224804`, `fy` tied to `fx`, `cx` and `cy` held
at the factory values. The surfaces are extracted **once** at the factory
intrinsics and their inlier indices reused for every candidate: re-running
RANSAC per candidate lets different candidates lock onto different surfaces,
which mixes the parameter's effect with the extraction's instability.

| fx | % of factory | median \|angle − 90°\| | |
| --- | --- | --- | --- |
| 808.8 | 90% | 3.560° | |
| 844.8 | 94% | 1.993° | |
| 871.7 | 97% | 0.942° | |
| 889.7 | 99% | **0.273°** | sampled minimum |
| **898.7** | **100%** | **0.279°** | **factory** |
| 907.7 | 101% | 0.539° | |
| 925.6 | 103% | 1.213° | |
| 988.6 | 110% | 2.450° | |

One basin, and the factory value sits at its floor. The sampled minimum is one
grid step away at 99% and differs by 0.006°, far inside this measurement's own
precision. **The factory `fx` is right to within about 1%**, which at the
target's 92 mm extent is under 1 mm.

With this, every parameter of the pinhole model on this unit has been measured
on this unit: `cx` and `cy` above, `fx` and `fy` here.

## Radial distortion is real, and small

The same sweep applied to a single radial coefficient, correcting in normalised
image coordinates as `x' = x·(1 + k1·r²)`:

| k1 | median \|angle − 90°\| | plane RMS |
| --- | --- | --- |
| −0.30 | 1.320° | 0.371 |
| −0.10 | 0.605° | 0.326 |
| −0.05 | 0.432° | 0.317 |
| **0.00** | **0.279°** | 0.313 |
| **0.05** | **0.235°** | **0.311** |
| 0.10 | 0.339° | 0.311 |
| 0.20 | 0.494° | 0.321 |
| 0.30 | 0.737° | 0.340 |

One basin, minimum at **k1 ≈ 0.05**, monotone either side, and the plane-fit
residual agrees rather than fighting it.

**And applying it end to end makes the reconstruction worse, so the basin is not
a calibration.** Squaring up right angles and describing the geometry correctly
are different claims, and only the second one matters. Re-fusing
`20260902-224804` through the pipeline's own preparation with k1 = 0.05 applied,
against an otherwise identical k1 = 0 control:

| | result |
| --- | --- |
| k1 = 0 (control) | all 190 frames placed, gate RMS 1.15 mm |
| k1 = 0.05 | **frame 162 unplaceable**, sweep refused |

Tracking refused the frame because two independent references - adjacent and
keyframe - each placed it acceptably and then disagreed about where it went by
**6.258 mm against a 5.200 mm gate**, in two equally large groups that agree
inside themselves and not with each other. The correction makes frames disagree
with *each other* more, which is a direct statement about geometry and does not
depend on the CAD, the placement search, or the gate's metric.

The likely reason is that the model is wrong for this device rather than the
coefficient being mistuned. On a structured-light depth camera the depth value
is computed by the device through its own calibration; scaling the deprojected
x and y by a radial term while leaving z untouched is not what an uncorrected
lens does, and it introduces an inconsistency that varies with where a surface
sits in the frame - which is exactly what the two references disagreed about.

So: **the plane-angle sweep is not sufficient to calibrate distortion**, even
though it is sufficient to locate `cx`, `cy` and `fx`. Those three change the
geometry coherently across the frame; a radial term does not. What the basin
establishes is that *something* varies radially in this data. What it does not
establish is that this correction is the description of it.

The factory `/data/camparam/Distort.bin` has still never been read - it needs an
authorisation that streaming does not, because the read detaches interface 0
from `uvcvideo`. It would settle the model rather than leaving it fitted.

## Why it matters

This was the ceiling. Every accuracy number this project produces — loop
closure, per-surface agreement, any future CAD comparison — is computed on
deprojected geometry, and a 277-pixel principal-point error would have made all
of them describe a sheared world while agreeing perfectly with themselves.

It also retires a standing worry about the failure pattern seen on 2026-09-01: a
shear's effect varies with where an object sits in the frame, which produces
exactly the good-median-bad-tail signature that a 1.20° held-out closure with a
10.06° worst case shows. **That tail is not a principal-point fault.** Whatever
causes it is still to be found.

## Method notes for reuse

The same construction settles any pinhole parameter that a known right angle is
sensitive to, on any target carrying one, from captures already taken:

1. Predict the signature analytically and **verify it on the reference geometry
   first** — the CAD control must read the nominal angle, or the test is broken
   before any scan is involved.
2. Sweep the parameter, re-deprojecting the same frames.
3. Require the curve to have one basin. A flat curve means the parameter is not
   identifiable from that geometry; several basins mean the plane extraction is
   finding different surfaces at different parameter values.

The plane extraction must reject candidates within ~25° of one already found.
Without that guard, RANSAC on a fused turntable cloud returns the same dominant
surface up to seven times as parallel slabs — measured, and the reason the first
attempt at this produced seven mutually-parallel "planes".
