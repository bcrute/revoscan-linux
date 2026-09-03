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

## What this establishes, and what it does not

**Establishes.** The sensor really does present an offset window: `cx = 42.44`
is a property of the device, not a corrupt read. The `x += 0.3089·z` shear
hypothesis is dead — had it been real, deprojecting at `cx = 320` would have
*removed* it and produced the minimum there, and instead that is the worst
sampled value outside the frame edges.

**Does not establish.** That the intrinsics are *correct*, only that `cx` is
close to right. `fx`, `fy` and `cy` were held fixed. The depth scale is
untouched by this — rigid geometry tests are blind to a global scale error, and
the −0.49% to −2.26% spread across distances is still open. And lens distortion
remains unmodelled and unfetched: `/data/camparam/Distort.bin` has never been
requested, so part of the 0.59° residual may be distortion this test cannot
separate from print tolerance and plane-fit noise.

**The residual is a floor, not an error budget.** 0.59° at the best `cx` is the
combined precision of plane fitting on real returns, the printed target's own
flatness, and any unmodelled distortion. It bounds how sharply this method can
locate a principal point; it is not a claim about the scanner's accuracy.

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
